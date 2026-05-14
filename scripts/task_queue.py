"""
task_queue.py — Persistent task queue and cron job store for the PRISM daemon.

Storage: data/task_queue.db  (SQLite, WAL mode, stdlib only)

Tables
------
tasks     — one-shot tasks (pending → running → done | failed)
cron_jobs — recurring schedules that seed the task queue on each tick

Schedule syntax (cron_jobs.schedule column)
-------------------------------------------
  every:30s   every:5m   every:2h        — fixed interval
  daily:08:00 daily:22:30                — once per day at local time
  weekly:mon:09:00                       — once per week (mon/tue/…/sun)

Usage
-----
  from task_queue import TaskQueue
  q = TaskQueue()
  tid = q.add_task("Set alarm for 9 AM", llm="claude")
  q.add_cron_job("daily:08:00", "Scan notification stack for threats", name="morning-scan")
  for t in q.get_due_tasks(): ...
"""

import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

# ── Path ─────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DEFAULT_DB = os.path.join(_REPO, "data", "task_queue.db")

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS tasks (
    id            TEXT PRIMARY KEY,
    task_text     TEXT NOT NULL,
    llm           TEXT NOT NULL DEFAULT 'claude',
    no_prism      INTEGER NOT NULL DEFAULT 0,
    learn         INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'pending',
    source        TEXT NOT NULL DEFAULT 'manual',
    cron_job_id   TEXT REFERENCES cron_jobs(id) ON DELETE SET NULL,
    created_at    REAL NOT NULL,
    execute_after REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    result_ok     INTEGER,
    result_note   TEXT
);

CREATE TABLE IF NOT EXISTS cron_jobs (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    task_text   TEXT NOT NULL,
    llm         TEXT NOT NULL DEFAULT 'claude',
    no_prism    INTEGER NOT NULL DEFAULT 0,
    learn       INTEGER NOT NULL DEFAULT 0,
    schedule    TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL,
    last_run_at REAL,
    next_run_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status_due
    ON tasks(status, execute_after);
CREATE INDEX IF NOT EXISTS idx_cron_enabled_next
    ON cron_jobs(enabled, next_run_at);
"""

# ── Schedule parsing ──────────────────────────────────────────────────────────

_EVERY = re.compile(r"^every:(\d+)(s|m|h)$")
_DAILY = re.compile(r"^daily:(\d{1,2}):(\d{2})$")
_WEEKLY = re.compile(r"^weekly:(mon|tue|wed|thu|fri|sat|sun):(\d{1,2}):(\d{2})$", re.I)
_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def parse_schedule(schedule: str, from_ts: Optional[float] = None) -> float:
    """Return the next Unix timestamp after *from_ts* (default: now) for *schedule*."""
    now = from_ts if from_ts is not None else time.time()
    dt_now = datetime.fromtimestamp(now)

    m = _EVERY.match(schedule)
    if m:
        amount, unit = int(m.group(1)), m.group(2)
        seconds = amount * {"s": 1, "m": 60, "h": 3600}[unit]
        return now + seconds

    m = _DAILY.match(schedule)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        candidate = dt_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate.timestamp() <= now:
            candidate += timedelta(days=1)
        return candidate.timestamp()

    m = _WEEKLY.match(schedule)
    if m:
        target_wd = _WEEKDAYS[m.group(1).lower()]
        hour, minute = int(m.group(2)), int(m.group(3))
        days_ahead = (target_wd - dt_now.weekday()) % 7
        candidate = (dt_now + timedelta(days=days_ahead)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if candidate.timestamp() <= now:
            candidate += timedelta(weeks=1)
        return candidate.timestamp()

    raise ValueError(
        f"Unknown schedule format: {schedule!r}. "
        "Use every:Ns/Nm/Nh, daily:HH:MM, or weekly:mon:HH:MM"
    )


def describe_schedule(schedule: str) -> str:
    """Human-readable summary of a schedule string."""
    m = _EVERY.match(schedule)
    if m:
        return f"every {m.group(1)}{m.group(2)}"
    m = _DAILY.match(schedule)
    if m:
        return f"daily at {m.group(1).zfill(2)}:{m.group(2)}"
    m = _WEEKLY.match(schedule)
    if m:
        return f"weekly on {m.group(1).capitalize()} at {m.group(2).zfill(2)}:{m.group(3)}"
    return schedule


# ── TaskQueue ─────────────────────────────────────────────────────────────────

class TaskQueue:
    """Thread-safe SQLite-backed task queue and cron scheduler."""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    # ── Tasks ─────────────────────────────────────────────────────────────────

    def add_task(
        self,
        task_text: str,
        *,
        llm: str = "claude",
        no_prism: bool = False,
        learn: bool = False,
        execute_after: Optional[float] = None,
        source: str = "manual",
        cron_job_id: Optional[str] = None,
    ) -> str:
        """Enqueue a task. Returns the new task id."""
        tid = str(uuid.uuid4())
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id, task_text, llm, no_prism, learn, status, source,
                    cron_job_id, created_at, execute_after)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    tid,
                    task_text,
                    llm,
                    int(no_prism),
                    int(learn),
                    "pending",
                    source,
                    cron_job_id,
                    now,
                    execute_after if execute_after is not None else now,
                ),
            )
        return tid

    def get_due_tasks(self) -> list[sqlite3.Row]:
        """Return pending tasks whose execute_after <= now, oldest first."""
        now = time.time()
        with self._conn() as conn:
            return conn.execute(
                """SELECT * FROM tasks
                   WHERE status = 'pending' AND execute_after <= ?
                   ORDER BY execute_after ASC""",
                (now,),
            ).fetchall()

    def mark_running(self, task_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE tasks SET status='running', started_at=? WHERE id=?",
                (time.time(), task_id),
            )

    def mark_done(self, task_id: str, *, ok: bool, note: str = "") -> None:
        status = "done" if ok else "failed"
        with self._conn() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status=?, finished_at=?, result_ok=?, result_note=?
                   WHERE id=?""",
                (status, time.time(), int(ok), note, task_id),
            )

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task. Returns True if cancelled, False if not pending."""
        with self._conn() as conn:
            rows = conn.execute(
                "UPDATE tasks SET status='cancelled' WHERE id=? AND status='pending'",
                (task_id,),
            ).rowcount
        return rows > 0

    def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM tasks"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return conn.execute(query, params).fetchall()

    def get_task(self, task_id: str) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)
            ).fetchone()

    # ── Cron jobs ─────────────────────────────────────────────────────────────

    def add_cron_job(
        self,
        schedule: str,
        task_text: str,
        *,
        name: Optional[str] = None,
        llm: str = "claude",
        no_prism: bool = False,
        learn: bool = False,
    ) -> str:
        """Register a recurring job. Returns the new cron job id."""
        parse_schedule(schedule)  # validate early
        jid = str(uuid.uuid4())
        now = time.time()
        next_run = parse_schedule(schedule, from_ts=now)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO cron_jobs
                   (id, name, task_text, llm, no_prism, learn,
                    schedule, enabled, created_at, next_run_at)
                   VALUES (?,?,?,?,?,?,?,1,?,?)""",
                (
                    jid,
                    name or task_text[:60],
                    task_text,
                    llm,
                    int(no_prism),
                    int(learn),
                    schedule,
                    now,
                    next_run,
                ),
            )
        return jid

    def get_due_cron_jobs(self) -> list[sqlite3.Row]:
        """Return enabled cron jobs whose next_run_at <= now."""
        now = time.time()
        with self._conn() as conn:
            return conn.execute(
                """SELECT * FROM cron_jobs
                   WHERE enabled = 1 AND next_run_at <= ?
                   ORDER BY next_run_at ASC""",
                (now,),
            ).fetchall()

    def advance_cron_job(self, job_id: str) -> float:
        """Update last_run_at and compute + store the next_run_at. Returns next_run_at."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT schedule FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if not row:
                raise KeyError(job_id)
            now = time.time()
            next_run = parse_schedule(row["schedule"], from_ts=now)
            conn.execute(
                "UPDATE cron_jobs SET last_run_at=?, next_run_at=? WHERE id=?",
                (now, next_run, job_id),
            )
        return next_run

    def enable_cron_job(self, job_id: str, enabled: bool = True) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE cron_jobs SET enabled=? WHERE id=?",
                (int(enabled), job_id),
            )

    def remove_cron_job(self, job_id: str) -> bool:
        with self._conn() as conn:
            rows = conn.execute(
                "DELETE FROM cron_jobs WHERE id=?", (job_id,)
            ).rowcount
        return rows > 0

    def list_cron_jobs(self, include_disabled: bool = False) -> list[sqlite3.Row]:
        query = "SELECT * FROM cron_jobs"
        if not include_disabled:
            query += " WHERE enabled = 1"
        query += " ORDER BY next_run_at ASC"
        with self._conn() as conn:
            return conn.execute(query).fetchall()

    def get_cron_job(self, job_id: str) -> Optional[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM cron_jobs WHERE id=?", (job_id,)
            ).fetchone()


# ── CLI (python scripts/task_queue.py <sub-command>) ─────────────────────────

def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _status_color(status: str) -> str:
    colors = {
        "pending": "\033[33m",
        "running": "\033[34m",
        "done":    "\033[32m",
        "failed":  "\033[31m",
        "cancelled": "\033[90m",
    }
    reset = "\033[0m"
    return colors.get(status, "") + status + reset


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect the PRISM task queue and cron jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── tasks list ────────────────────────────────────────────────────────────
    tl = sub.add_parser("tasks", help="List tasks")
    tl.add_argument("--status", help="Filter by status (pending/running/done/failed)")
    tl.add_argument("--limit", type=int, default=20)

    # ── tasks cancel ──────────────────────────────────────────────────────────
    tc = sub.add_parser("cancel", help="Cancel a pending task")
    tc.add_argument("task_id")

    # ── cron list ─────────────────────────────────────────────────────────────
    sub.add_parser("cron", help="List cron jobs")

    # ── cron enable/disable ───────────────────────────────────────────────────
    ce = sub.add_parser("enable", help="Enable a cron job")
    ce.add_argument("job_id")
    cd = sub.add_parser("disable", help="Disable a cron job")
    cd.add_argument("job_id")

    # ── cron remove ───────────────────────────────────────────────────────────
    cr = sub.add_parser("remove", help="Remove a cron job")
    cr.add_argument("job_id")

    # ── db path ───────────────────────────────────────────────────────────────
    p.add_argument("--db", default=_DEFAULT_DB, help="Path to task_queue.db")

    a = p.parse_args()
    q = TaskQueue(a.db)

    if a.cmd == "tasks":
        rows = q.list_tasks(status=a.status, limit=a.limit)
        if not rows:
            print("No tasks.")
            return
        print(f"{'ID[:8]':<10} {'STATUS':<12} {'DUE / CREATED':<22} {'LLM':<7} TASK")
        print("─" * 90)
        for r in rows:
            due = _fmt_ts(r["execute_after"])
            tid = r["id"][:8]
            txt = r["task_text"][:55]
            print(f"{tid:<10} {_status_color(r['status']):<12} {due:<22} {r['llm']:<7} {txt}")

    elif a.cmd == "cancel":
        ok = q.cancel_task(a.task_id)
        print("Cancelled." if ok else "Task not found or not pending.")

    elif a.cmd == "cron":
        rows = q.list_cron_jobs(include_disabled=True)
        if not rows:
            print("No cron jobs.")
            return
        print(f"{'ID[:8]':<10} {'EN':<4} {'SCHEDULE':<18} {'NEXT RUN':<22} NAME")
        print("─" * 90)
        for r in rows:
            jid = r["id"][:8]
            en = "✓" if r["enabled"] else "✗"
            sched = describe_schedule(r["schedule"])
            nxt = _fmt_ts(r["next_run_at"])
            name = (r["name"] or r["task_text"])[:40]
            print(f"{jid:<10} {en:<4} {sched:<18} {nxt:<22} {name}")

    elif a.cmd == "enable":
        q.enable_cron_job(a.job_id, enabled=True)
        print("Enabled.")

    elif a.cmd == "disable":
        q.enable_cron_job(a.job_id, enabled=False)
        print("Disabled.")

    elif a.cmd == "remove":
        ok = q.remove_cron_job(a.job_id)
        print("Removed." if ok else "Cron job not found.")


if __name__ == "__main__":
    _cli()
