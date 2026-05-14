"""
prism_daemon.py — PRISM proactive agent background service.

Three threads run in parallel:

  QueueRunner    — pops due tasks from task_queue.db, runs them serially via
                   subprocess (full PRISM pipeline preserved, no shared state).

  CronScheduler  — checks recurring cron job definitions every 60 s; seeds the
                   task queue when a job is due and advances its next_run_at.

  DeviceObserver — polls the Android sidecar every 30 s for new notifications
                   and clipboard content; PRISM-scans each new item; logs an
                   alert on BLOCK verdict. Skips silently when device or sidecar
                   is unreachable — daemon stays up even without an emulator.

Usage
-----
  python scripts/prism_daemon.py run      # foreground — logs to stdout (debug)
  python scripts/prism_daemon.py start    # background — logs to data/daemon.log
  python scripts/prism_daemon.py stop     # stop the background daemon
  python scripts/prism_daemon.py status   # queue stats + running state

Environment overrides
---------------------
  PRISM_SIDECAR_URL        default http://localhost:8765
  ANDROID_SERIAL           default emulator-5554
  DAEMON_QUEUE_INTERVAL    seconds between queue ticks  (default 15)
  DAEMON_CRON_INTERVAL     seconds between cron ticks   (default 60)
  DAEMON_OBS_INTERVAL      seconds between observer polls (default 30)
"""

import argparse
import hashlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import URLError
import json

# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE   = os.path.dirname(os.path.abspath(__file__))
_REPO   = os.path.dirname(_HERE)
_AGENT  = os.path.join(_HERE, "agent_prism.py")
_PID    = os.path.join(_REPO, "data", "prism_daemon.pid")
_LOG    = os.path.join(_REPO, "data", "daemon.log")

# Android sidecar URL (same constant as context_assembler.py)
_ANDROID_URL  = "http://127.0.0.1:8766"
_PRISM_URL    = os.getenv("PRISM_SIDECAR_URL", "http://localhost:8765")
_SERIAL       = os.getenv("ANDROID_SERIAL", "emulator-5554")

_QUEUE_INTERVAL = int(os.getenv("DAEMON_QUEUE_INTERVAL", "15"))
_CRON_INTERVAL  = int(os.getenv("DAEMON_CRON_INTERVAL",  "60"))
_OBS_INTERVAL   = int(os.getenv("DAEMON_OBS_INTERVAL",   "30"))

# Colour codes (stripped when logging to file)
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_RESET  = "\033[0m"

# ── Logging setup ─────────────────────────────────────────────────────────────

log = logging.getLogger("prism.daemon")


def _setup_logging(foreground: bool) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    level = logging.INFO

    handlers: list[logging.Handler] = []
    if foreground:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(h)
    else:
        os.makedirs(os.path.dirname(_LOG), exist_ok=True)
        h = logging.FileHandler(_LOG)
        h.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(h)

    logging.basicConfig(level=level, handlers=handlers, force=True)


# ── PID helpers ───────────────────────────────────────────────────────────────

def _save_pid(pid: int) -> None:
    os.makedirs(os.path.dirname(_PID), exist_ok=True)
    with open(_PID, "w") as f:
        f.write(str(pid))


def _load_pid() -> int | None:
    try:
        return int(open(_PID).read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _clear_pid() -> None:
    try:
        os.unlink(_PID)
    except FileNotFoundError:
        pass


# ── Stale-task recovery ───────────────────────────────────────────────────────

def _reset_stale_running(q) -> int:
    """On daemon start, tasks left in 'running' state from a prior crashed run
    are reset to 'pending' so they are retried."""
    import sqlite3
    count = 0
    with sqlite3.connect(q.db_path, timeout=10) as conn:
        count = conn.execute(
            "UPDATE tasks SET status='pending', started_at=NULL "
            "WHERE status='running'"
        ).rowcount
    if count:
        log.warning("Reset %d stale 'running' task(s) to 'pending'", count)
    return count


# ── Thread: QueueRunner ───────────────────────────────────────────────────────

class QueueRunner(threading.Thread):
    """Serially executes tasks from the queue as subprocesses.

    One task at a time — the Android emulator is a shared resource; parallel
    tasks would fight over the screen.
    """

    def __init__(self, q, serial: str, stop_event: threading.Event):
        super().__init__(name="QueueRunner", daemon=True)
        self.q           = q
        self.serial      = serial
        self.stop_event  = stop_event

    def run(self) -> None:
        log.info("QueueRunner started (interval=%ds)", _QUEUE_INTERVAL)
        while not self.stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("QueueRunner tick error")
            self.stop_event.wait(_QUEUE_INTERVAL)

    def _tick(self) -> None:
        due = self.q.get_due_tasks()
        if not due:
            return

        task = due[0]  # one at a time
        tid  = task["id"]
        txt  = task["task_text"]
        log.info("Running task [%s]: %s", tid[:8], txt[:60])

        self.q.mark_running(tid)
        ok, note = self._run_subprocess(task)
        self.q.mark_done(tid, ok=ok, note=note)

        level = logging.INFO if ok else logging.WARNING
        log.log(level, "Task [%s] %s — %s", tid[:8], "done" if ok else "FAILED", note)

    def _run_subprocess(self, task) -> tuple[bool, str]:
        cmd = [sys.executable, _AGENT]

        cmd += ["--task",   task["task_text"]]
        cmd += ["--serial", self.serial]
        cmd += ["--llm",    task["llm"]]

        if task["no_prism"]:
            cmd.append("--no-prism")
        if task["learn"]:
            cmd.append("--learn")

        log.debug("Subprocess: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                cwd=_HERE,
                timeout=600,          # 10-minute hard cap per task
                capture_output=False, # let output flow to daemon log
            )
            ok   = result.returncode == 0
            note = "returncode=0" if ok else f"returncode={result.returncode}"
            return ok, note
        except subprocess.TimeoutExpired:
            return False, "timed out after 600s"
        except Exception as exc:
            return False, str(exc)


# ── Thread: CronScheduler ─────────────────────────────────────────────────────

class CronScheduler(threading.Thread):
    """Seeds the task queue from cron job definitions."""

    def __init__(self, q, stop_event: threading.Event):
        super().__init__(name="CronScheduler", daemon=True)
        self.q          = q
        self.stop_event = stop_event

    def run(self) -> None:
        log.info("CronScheduler started (interval=%ds)", _CRON_INTERVAL)
        while not self.stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("CronScheduler tick error")
            self.stop_event.wait(_CRON_INTERVAL)

    def _tick(self) -> None:
        due = self.q.get_due_cron_jobs()
        for job in due:
            jid  = job["id"]
            name = job["name"] or job["task_text"][:40]
            log.info("Cron job [%s] due: %s", jid[:8], name)

            tid = self.q.add_task(
                job["task_text"],
                llm=job["llm"],
                no_prism=bool(job["no_prism"]),
                learn=bool(job["learn"]),
                source="cron",
                cron_job_id=jid,
            )
            next_run = self.q.advance_cron_job(jid)
            next_str = datetime.fromtimestamp(next_run).strftime("%Y-%m-%d %H:%M")
            log.info("  → queued task [%s]; next run: %s", tid[:8], next_str)


# ── Thread: DeviceObserver ────────────────────────────────────────────────────

class DeviceObserver(threading.Thread):
    """Polls the Android sidecar for new notifications/clipboard and PRISM-scans them.

    On a BLOCK verdict the observer auto-queues a defensive task so the agent
    can dismiss or investigate the threat without waiting for a user prompt.
    Set auto_queue=False to disable action and keep it monitor-only.

    Skips silently when the device or PRISM sidecar is unreachable — the daemon
    continues running even without an emulator attached.
    """

    def __init__(self, q, serial: str, stop_event: threading.Event,
                 auto_queue: bool = True):
        super().__init__(name="DeviceObserver", daemon=True)
        self.q             = q
        self.serial        = serial
        self.stop_event    = stop_event
        self.auto_queue    = auto_queue
        self._seen_hashes: set[str] = set()   # dedup across ticks
        self._warned_down  = False             # suppress repeated "sidecar down" logs

    def run(self) -> None:
        log.info("DeviceObserver started (interval=%ds)", _OBS_INTERVAL)
        while not self.stop_event.is_set():
            try:
                self._tick()
            except Exception:
                log.exception("DeviceObserver tick error")
            self.stop_event.wait(_OBS_INTERVAL)

    # ── device context fetch ───────────────────────────────────────────────────

    def _fetch_context(self) -> dict | None:
        try:
            req = Request(f"{_ANDROID_URL}/v1/context", method="GET")
            with urlopen(req, timeout=5) as resp:
                self._warned_down = False
                return json.loads(resp.read().decode())
        except (URLError, OSError):
            if not self._warned_down:
                log.debug("Android sidecar unreachable — observer waiting")
                self._warned_down = True
            return None

    # ── PRISM scan via sidecar ─────────────────────────────────────────────────

    def _prism_inspect(self, text: str, ingestion_path: str, source_name: str) -> dict | None:
        payload = json.dumps({
            "entry_id":      str(uuid.uuid4()),
            "text":          text,
            "ingestion_path": ingestion_path,
            "source_type":   ingestion_path,
            "source_name":   source_name,
            "session_id":    "daemon-observer",
        }).encode()
        try:
            req = Request(
                f"{_PRISM_URL}/v1/inspect",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except (URLError, OSError):
            return None

    # ── per-item handling ──────────────────────────────────────────────────────

    def _handle_item(self, text: str, path: str, label: str) -> None:
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self._seen_hashes:
            return
        self._seen_hashes.add(h)

        result = self._prism_inspect(text, path, label)
        if result is None:
            return  # PRISM sidecar unreachable — skip, don't alert

        verdict    = result.get("verdict", "ALLOW")
        confidence = result.get("confidence", 0.0)
        snippet    = text[:80].replace("\n", " ")

        if verdict == "BLOCK":
            log.warning(
                "%sBLOCKED%s [%s] %.0f%% — %s",
                _RED, _RESET, label, confidence * 100, snippet,
            )
            if self.auto_queue:
                self._queue_defensive_task(path, label, snippet)
        elif verdict == "QUARANTINE":
            log.warning(
                "%sQUARANTINE%s [%s] %.0f%% — %s",
                _YELLOW, _RESET, label, confidence * 100, snippet,
            )
        else:
            log.debug("ALLOW [%s] — %s", label, snippet)

    def _queue_defensive_task(self, path: str, label: str, snippet: str) -> None:
        if path == "clipboard":
            task_text = (
                "The clipboard contains content that was flagged as a prompt-injection "
                "attack by PRISM. Clear the clipboard immediately."
            )
        else:
            task_text = (
                f"A notification from '{label}' was flagged as a prompt-injection "
                f"attack by PRISM (snippet: \"{snippet[:60]}\"). "
                "Open the notification shade and dismiss this notification."
            )
        tid = self.q.add_task(task_text, llm="claude", source="observer")
        log.info("Auto-queued defensive task [%s] for blocked %s from %s",
                 tid[:8], path, label)

    def _tick(self) -> None:
        ctx = self._fetch_context()
        if ctx is None:
            return

        # notifications
        for notif in ctx.get("notifications", []):
            text = notif.get("text") or notif.get("title") or ""
            pkg  = notif.get("package", "unknown")
            if text.strip():
                self._handle_item(text, "notifications", pkg)

        # clipboard
        clip = ctx.get("clipboard", "")
        if clip and clip.strip():
            self._handle_item(clip, "clipboard", "clipboard")


# ── PrismDaemon orchestrator ──────────────────────────────────────────────────

class PrismDaemon:
    def __init__(self, serial: str, enable_observer: bool = True,
                 auto_queue: bool = True):
        # Import here so the module is importable without task_queue on sys.path
        sys.path.insert(0, _HERE)
        from task_queue import TaskQueue
        self.q                = TaskQueue()
        self.serial           = serial
        self.enable_observer  = enable_observer
        self.auto_queue       = auto_queue
        self._stop            = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        _reset_stale_running(self.q)

        self._threads = [
            QueueRunner(self.q, self.serial, self._stop),
            CronScheduler(self.q, self._stop),
        ]
        if self.enable_observer:
            self._threads.append(
                DeviceObserver(self.q, self.serial, self._stop,
                               auto_queue=self.auto_queue)
            )

        for t in self._threads:
            t.start()

        log.info(
            "PRISM daemon running — threads: %s",
            ", ".join(t.name for t in self._threads),
        )

    def wait(self) -> None:
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self) -> None:
        log.info("Shutting down daemon…")
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
        log.info("Daemon stopped.")


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    """Run in the foreground (blocks until Ctrl-C)."""
    _setup_logging(foreground=True)
    log.info("Starting PRISM daemon (foreground, PID %d)", os.getpid())
    daemon = PrismDaemon(
        args.serial,
        enable_observer=not args.no_observer,
        auto_queue=not args.no_auto_queue,
    )
    daemon.start()
    daemon.wait()


def cmd_start(args) -> None:
    """Fork the daemon to the background."""
    pid = _load_pid()
    if pid and _is_running(pid):
        print(f"Daemon already running (PID {pid})")
        return

    # Build the 'run' command and pass all flags through
    cmd = [sys.executable, __file__, "run", "--serial", args.serial]
    if args.no_observer:
        cmd.append("--no-observer")
    if args.no_auto_queue:
        cmd.append("--no-auto-queue")

    os.makedirs(os.path.dirname(_LOG), exist_ok=True)
    log_file = open(_LOG, "a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from terminal
        close_fds=True,
    )
    _save_pid(proc.pid)
    print(f"PRISM daemon started (PID {proc.pid})")
    print(f"Log:  {_LOG}")
    print(f"Stop: python scripts/prism_daemon.py stop")


def cmd_stop(_args) -> None:
    """Stop the background daemon."""
    pid = _load_pid()
    if not pid:
        print("No PID file found — daemon may not be running.")
        return
    if not _is_running(pid):
        print(f"Daemon (PID {pid}) is not running. Cleaning up PID file.")
        _clear_pid()
        return
    os.kill(pid, signal.SIGTERM)
    # wait up to 5 s for clean exit
    for _ in range(50):
        time.sleep(0.1)
        if not _is_running(pid):
            break
    if _is_running(pid):
        os.kill(pid, signal.SIGKILL)
        print(f"Daemon (PID {pid}) force-killed.")
    else:
        print(f"Daemon (PID {pid}) stopped.")
    _clear_pid()


def cmd_status(_args) -> None:
    """Show daemon state and queue stats."""
    sys.path.insert(0, _HERE)
    from task_queue import TaskQueue

    pid = _load_pid()
    if pid and _is_running(pid):
        print(f"{_GREEN}● running{_RESET}  PID {pid}  log: {_LOG}")
    elif pid:
        print(f"{_RED}● dead{_RESET}     PID {pid} no longer exists (stale PID file)")
    else:
        print(f"{_YELLOW}○ stopped{_RESET}")

    # queue stats
    q = TaskQueue()
    print()
    print("Task queue")
    for status in ("pending", "running", "done", "failed"):
        rows = q.list_tasks(status=status, limit=1000)
        print(f"  {status:<10} {len(rows)}")

    cron_jobs = q.list_cron_jobs(include_disabled=True)
    enabled   = sum(1 for j in cron_jobs if j["enabled"])
    print(f"\nCron jobs: {len(cron_jobs)} total, {enabled} enabled")
    for j in cron_jobs:
        nxt = datetime.fromtimestamp(j["next_run_at"]).strftime("%Y-%m-%d %H:%M")
        en  = f"{_GREEN}✓{_RESET}" if j["enabled"] else f"{_YELLOW}✗{_RESET}"
        print(f"  {en} [{j['id'][:8]}] next {nxt}  {(j['name'] or j['task_text'])[:50]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="PRISM daemon — proactive agent background service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  run     Run in the foreground (Ctrl-C to stop)
  start   Start in the background (logs → data/daemon.log)
  stop    Stop the background daemon
  status  Show running state and queue stats
""",
    )
    p.add_argument("command", choices=["run", "start", "stop", "status"])
    p.add_argument("--serial",      default=_SERIAL,
                   help="Android emulator serial (default: %(default)s)")
    p.add_argument("--no-observer",   action="store_true",
                   help="Disable the DeviceObserver thread (no emulator available)")
    p.add_argument("--no-auto-queue", action="store_true",
                   help="Observer monitors only — do not auto-queue defensive tasks on BLOCK")

    a = p.parse_args()

    dispatch = {
        "run":    cmd_run,
        "start":  cmd_start,
        "stop":   cmd_stop,
        "status": cmd_status,
    }
    dispatch[a.command](a)
