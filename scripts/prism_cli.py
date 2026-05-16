#!/usr/bin/env python3
"""PRISM Agent interactive CLI.  Type a task or /command."""
from __future__ import annotations

import os
import pathlib
import queue as _queue_mod
import re
import readline
import shlex
import sys
import time
from typing import Any

# ── Path ───────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# agent_prism takes ~5 s to import (uiautomator2, transformers, anthropic).
# Defer it until the first actual task execution.
_agent: Any = None

# Alert watcher — background thread + queue, both set up in repl().
_alert_queue:   _queue_mod.Queue = _queue_mod.Queue()
_alert_watcher: Any = None   # AlertWatcher instance

def _get_agent():
    global _agent
    if _agent is None:
        print(f"{DIM}  loading agent…{R}", flush=True)
        import agent_prism
        _agent = agent_prism
    return _agent

from task_queue import TaskQueue, describe_schedule  # fast — stdlib only

# ── ANSI ───────────────────────────────────────────────────────────────────
R   = "\033[0m";  B   = "\033[1m";  DIM = "\033[2m"
CYN = "\033[36m"; GRN = "\033[32m"; YLW = "\033[33m"
RED = "\033[31m"; MAG = "\033[35m"

PROMPT  = f"{B}{CYN}>{R} "
HISTORY = os.path.expanduser("~/.prism_history")
DEFAULT_SERIAL = os.getenv("ANDROID_SERIAL", "emulator-5554")

BANNER = f"""{B}{CYN}
  █████╗  ██████╗ ███████╗███╗   ██╗████████╗
 ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
 ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
 ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
 ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
 ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝  ╚═╝{R}
  Agent CLI  —  type a task or {CYN}/{R}command  ({CYN}/help{R})
"""

HELP = f"""
{B}Just type to chat{R} — Claude answers using agent memory, skills, routines.

{B}To run something on the phone:{R}
  {CYN}/run{R} <task>                  Execute now
  {CYN}/queue{R} <task>                Add to queue
  {CYN}/queue{R} list | run | clear    Manage queue

{B}Routines:{R}
  {CYN}/routine add{R} <name> <sched> <task>
    sched: daily:HH:MM | hourly | every:Ns | every:Nm | every:Nh
  {CYN}/routine{R} list | run <name> | del <name>

{B}Skills  (RAG store, retrieved semantically):{R}
  {CYN}/skill add{R} <name> <instructions>
  {CYN}/skill{R} list | show <name> | del <name>

{B}Memory  (auto-saved after every task):{R}
  {CYN}/memory list{R}               Show recent task memories
  {CYN}/memory clear{R}              Wipe all memories

{B}Daemon:{R}
  {CYN}/watch{R}                     Proactive loop (Ctrl-C to stop)
  {CYN}/status{R}                    Summary
  {CYN}/exit{R}                      Quit

  Tab after {CYN}/{R} shows all commands with descriptions.
"""

# ── RAG / Skill store ──────────────────────────────────────────────────────
# Skills live in the same MemShield-wrapped ChromaDB as the agent's KB.
# They are retrieved semantically (not keyword-matched) and flow through
# the full PRISM pipeline — making poisoned skills a demonstrable attack.

_DB_PATH    = os.path.join(_HERE, "..", "data", "chromadb")
_COLLECTION = "agent_kb"
_SKILL_PREFIX = "skill:"   # doc IDs for skills are "skill:<name>"

def _rag_collection():
    """Return the ChromaDB collection, or None if unavailable."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_DB_PATH)
        return client.get_or_create_collection(_COLLECTION)
    except Exception as e:
        print(f"{RED}  chromadb unavailable: {e}{R}")
        return None

def _rag_shield(collection):
    """Wrap collection in MemShield for scan-on-ingest."""
    try:
        sys.path.insert(0, os.path.join(_HERE, "..", "memshield", "src"))
        from memshield import MemShield, ShieldConfig
        return MemShield(collection=collection,
                         config=ShieldConfig(enable_normalization=True,
                                             enable_provenance=True))
    except Exception:
        return None

# ── Core execution ──────────────────────────────────────────────────────────

def _run(task: str, llm: str, serial: str) -> None:
    # Skills are retrieved by the agent's RAG pipeline automatically.
    print(f"{DIM}  running…{R}")
    ok = _get_agent().run(task, serial=serial, llm=llm)
    print(f"{GRN}✓ done{R}" if ok else f"{RED}✗ failed{R}")

# ── Status colour helper ────────────────────────────────────────────────────

def _sc(status: str) -> str:
    return GRN if status == "done" else (RED if status == "failed" else YLW)

# ── Command handlers ────────────────────────────────────────────────────────

def _queue(args: list[str], llm: str, serial: str) -> None:
    q = TaskQueue()
    sub = args[0] if args else "list"

    if sub == "list":
        rows = q.list_tasks(limit=20)
        print(f"{DIM}  queue empty{R}") if not rows else None
        for r in rows:
            print(f"  {DIM}{r['id'][:8]}{R}  {_sc(r['status'])}{r['status']:<9}{R}  {r['task_text'][:60]}")

    elif sub == "run":
        row = q.claim_next_due_task()
        if row is None:
            print(f"{DIM}  no tasks due{R}"); return
        print(f"{YLW}  [{row['id'][:8]}]{R} {row['task_text']}")
        ok = _get_agent().run(row["task_text"], llm=row["llm"] or llm, serial=serial)
        q.mark_done(row["id"], ok=ok, note="done" if ok else "failed")
        print(f"{GRN}✓ done{R}" if ok else f"{RED}✗ failed{R}")

    elif sub == "clear":
        with q._conn() as c:
            c.execute("UPDATE tasks SET status='cancelled' WHERE status='pending'")
        print(f"{YLW}  queue cleared{R}")

    else:
        task = " ".join(args)
        tid = q.add_task(task, llm=llm, source="cli")
        print(f"{GRN}  queued [{tid[:8]}]{R}  {task}")


def _routine(args: list[str], llm: str, serial: str) -> None:
    if not args:
        print(f"{RED}  /routine add|list|run|del{R}"); return
    q   = TaskQueue()
    sub = args[0]

    if sub == "list":
        rows = q.list_cron_jobs()
        print(f"{DIM}  no routines{R}") if not rows else None
        for r in rows:
            name = r["name"] or r["id"][:8]
            nxt  = time.strftime("%m-%d %H:%M", time.localtime(r["next_run_at"])) if r["next_run_at"] else "--"
            print(f"  {CYN}{name:<16}{R}  {DIM}{describe_schedule(r['schedule']):<20}{R}  {nxt}  {r['task_text'][:40]}")

    elif sub == "add":
        if len(args) < 4:
            print(f"{RED}  /routine add <name> <sched> <task>{R}"); return
        name, sched, *rest = args[1:]
        q.add_cron_job(sched, " ".join(rest), name=name, llm=llm)
        print(f"{GRN}  routine '{name}' added ({describe_schedule(sched)}){R}")

    elif sub == "run":
        if len(args) < 2:
            print(f"{RED}  /routine run <name>{R}"); return
        name = args[1]
        row  = next((r for r in q.list_cron_jobs() if r["name"] == name), None)
        if not row:
            print(f"{RED}  no routine '{name}'{R}"); return
        _run(row["task_text"], llm, serial)

    elif sub == "del":
        if len(args) < 2:
            print(f"{RED}  /routine del <name>{R}"); return
        with q._conn() as c:
            c.execute("DELETE FROM cron_jobs WHERE name=?", (args[1],))
        print(f"{YLW}  routine '{args[1]}' deleted{R}")

    else:
        print(f"{RED}  unknown: {sub}{R}")


def _skill(args: list[str]) -> None:
    if not args:
        print(f"{RED}  /skill add|list|show|del{R}"); return
    sub = args[0]
    col = _rag_collection()
    if col is None:
        return

    if sub == "list":
        results = col.get(where={"source": "skill"})
        ids   = results.get("ids", [])
        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not ids:
            print(f"{DIM}  no skills in RAG store{R}"); return
        for doc_id, doc, meta in zip(ids, docs, metas or [{}]*len(ids)):
            name = doc_id.removeprefix(_SKILL_PREFIX)
            trigger = doc[:60]
            print(f"  {CYN}{name:<16}{R}  {DIM}{trigger}{R}")

    elif sub == "show":
        if len(args) < 2:
            print(f"{RED}  /skill show <name>{R}"); return
        results = col.get(ids=[f"{_SKILL_PREFIX}{args[1]}"])
        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            print(f"{RED}  no skill '{args[1]}'{R}"); return
        meta = (metas or [{}])[0]
        print(f"{B}{args[1]}{R}")
        print(f"  {DIM}trigger:{R}   {docs[0]}")
        print(f"  {DIM}procedure:{R} {meta.get('body', docs[0])}")

    elif sub == "add":
        if len(args) < 3:
            print(f"{RED}  /skill add <name> <instructions>{R}"); return
        name, *rest = args[1:]
        body = " ".join(rest)
        # Auto-derive trigger: skill name (spaces → words) + first sentence
        name_words = name.replace("-", " ").replace("_", " ")
        first_sentence = body.split(".")[0].strip()
        description = f"{name_words}: {first_sentence}"
        sid = f"{_SKILL_PREFIX}{name}"
        shield = _rag_shield(col)
        if shield:
            stats = shield.ingest_with_scan(
                documents=[description], ids=[sid],
                metadatas=[{"source": "skill", "name": name, "body": body}],
                source="skill", session_id="cli",
            )
            if stats["blocked"]:
                print(f"{RED}  PRISM blocked skill '{name}' — content flagged as malicious{R}")
                return
            if stats["quarantined"]:
                print(f"{YLW}  skill '{name}' quarantined — content suspicious{R}")
                return
        else:
            col.upsert(documents=[description], ids=[sid],
                       metadatas=[{"source": "skill", "name": name, "body": body}])
        print(f"{GRN}  skill '{name}' added to RAG store{R}")
        print(f"  {DIM}matches on: {description[:80]}{R}")

    elif sub == "del":
        if len(args) < 2:
            print(f"{RED}  /skill del <name>{R}"); return
        col.delete(ids=[f"{_SKILL_PREFIX}{args[1]}"])
        print(f"{YLW}  skill '{args[1]}' removed from RAG store{R}")

    else:
        print(f"{RED}  unknown: {sub}{R}")


def _watch(llm: str, serial: str) -> None:
    print(f"{MAG}  watch mode — Ctrl-C to stop{R}")
    q = TaskQueue()
    try:
        while True:
            for job in q.get_due_cron_jobs():
                tid = q.add_task(job["task_text"], llm=job["llm"] or llm,
                                 source="cron", cron_job_id=job["id"])
                q.advance_cron_job(job["id"])
                print(f"\n{MAG}[watch]{R} '{job['name'] or job['id'][:8]}' queued [{tid[:8]}]")
            row = q.claim_next_due_task()
            if row:
                print(f"\n{MAG}[watch]{R} {row['task_text']}")
                ok = _get_agent().run(row["task_text"], serial=serial, llm=row["llm"] or llm)
                q.mark_done(row["id"], ok=ok, note="done" if ok else "failed")
                print(f"{GRN if ok else RED}[watch] {'done' if ok else 'failed'}{R}")
            else:
                print(f"{DIM}[watch] idle — 60s{R}", end="\r", flush=True)
            time.sleep(60)
    except KeyboardInterrupt:
        print(f"\n{YLW}  watch stopped{R}")


def _status() -> None:
    q       = TaskQueue()
    pending = q.list_tasks(status="pending", limit=100)
    crons   = q.list_cron_jobs()
    col     = _rag_collection()
    n_skills = len(col.get(where={"source": "skill"})["ids"]) if col else 0
    print(f"  {B}queue{R}    {len(pending)} pending")
    print(f"  {B}routines{R} {len(crons)}")
    print(f"  {B}skills{R}   {n_skills} in RAG store")


# ── / dropdown completion ───────────────────────────────────────────────────

_CMD_HELP: dict[str, str] = {
    "/run":     "Execute a task on the phone",
    "/queue":   "Add to queue or manage it  (list | run | clear)",
    "/routine": "Recurring jobs             (add | list | run | del)",
    "/skill":   "RAG skill store            (add | list | show | del)",
    "/memory":  "Agent memory               (list | save <text> | clear)",
    "/watch":   "Start proactive daemon loop",
    "/status":  "Queue, routines, skills summary",
    "/help":    "Show all commands",
    "/clear":   "Clear terminal screen",
    "/exit":    "Quit",
}

_SLASH_COMPLETIONS = list(_CMD_HELP.keys())

def _completer(text: str, state: int) -> str | None:
    prefix = text if text.startswith("/") else "/"
    matches = [c for c in _SLASH_COMPLETIONS if c.startswith(prefix)]
    return (matches[state] + " ") if state < len(matches) else None

def _display_matches(substitution: str, matches: list[str], _longest: int) -> None:
    print()
    for m in matches:
        cmd = m.strip()
        desc = _CMD_HELP.get(cmd, "")
        print(f"  {CYN}{cmd:<12}{R}  {desc}")
    sys.stdout.write(PROMPT + readline.get_line_buffer())
    sys.stdout.flush()

readline.set_completer(_completer)
readline.set_completion_display_matches_hook(_display_matches)
readline.set_completer_delims("")
readline.parse_and_bind("tab: complete")


# ── Agent memory list / clear ───────────────────────────────────────────────

def _memory(args: list[str]) -> None:
    col = _rag_collection()
    if col is None:
        return
    sub = args[0] if args else "list"

    if sub == "list":
        results = col.get(where={"source": "memory"}, include=["documents", "metadatas"])
        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            print(f"{DIM}  no memories yet{R}"); return
        for i, (doc, meta) in enumerate(zip(docs[-10:], (metas or [])[-10:]), 1):
            sealed = bool(meta and meta.get("content_hash"))
            tag = f"{GRN}✓{R}" if sealed else f"{RED}⚠ unsealed{R}"
            print(f"  {DIM}{i:2}.{R} {tag} {DIM}{doc[:100]}{R}")

    elif sub in ("save", "add"):
        if len(args) < 2:
            print(f"{RED}  /memory save <text>{R}"); return
        text = " ".join(args[1:])
        import hashlib
        ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
        doc = f"[MEMORY {ts}] {text}"
        doc_id = f"mem_{hashlib.sha256(doc.encode()).hexdigest()[:16]}"
        # Go through MemShield so the memory gets a valid HMAC provenance seal.
        # Poisoned docs injected directly into ChromaDB won't have this seal
        # and will be flagged by the retrieval-time defense.
        shield = _rag_shield(col)
        if shield:
            stats = shield.ingest_with_scan(
                documents=[doc], ids=[doc_id],
                metadatas=[{"source": "memory", "name": doc_id, "ts": ts}],
                source="memory", authority=0.5,
            )
            if stats["blocked"]:
                print(f"{RED}  PRISM blocked memory — content flagged as malicious{R}")
                return
            if stats["quarantined"]:
                print(f"{YLW}  PRISM quarantined memory — content suspicious, not stored{R}")
                return
        else:
            col.upsert(documents=[doc], ids=[doc_id],
                       metadatas=[{"source": "memory", "name": doc_id, "ts": ts}])
        print(f"{GRN}  memory saved{R}  {DIM}{doc[:80]}{R}")

    elif sub in ("del", "delete"):
        if len(args) < 2:
            print(f"{RED}  /memory del <index>  (use /memory list to see indices){R}"); return
        try:
            idx = int(args[1]) - 1
        except ValueError:
            print(f"{RED}  index must be a number{R}"); return
        results = col.get(where={"source": "memory"})
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        if idx < 0 or idx >= len(ids):
            print(f"{RED}  no memory at index {idx+1}{R}"); return
        col.delete(ids=[ids[idx]])
        print(f"{YLW}  deleted:{R} {DIM}{docs[idx][:80]}{R}")

    elif sub == "clear":
        results = col.get(where={"source": "memory"})
        ids = results.get("ids", [])
        if ids:
            col.delete(ids=ids)
        print(f"{YLW}  {len(ids)} memories cleared{R}")

    else:
        print(f"{RED}  /memory list | save <text> | del <index> | clear{R}")


# ── Chat with Claude ────────────────────────────────────────────────────────

_CHAT_SYSTEM = """\
You are the PRISM agent's assistant. PRISM is a research system that controls an \
Android phone autonomously while defending against prompt injection and RAG poisoning attacks.

You have access to the agent's memory (past tasks it ran), its skills, and its routines. \
Answer questions naturally and concisely. If the user wants the agent to actually DO \
something on the phone, tell them to state it as a command (no question mark needed).
"""

_session_history: list[dict] = []


def _chat(text: str) -> None:
    """Answer a question using Claude with agent memory + session history as context."""
    global _session_history

    # Build context: recent memories + skills from ChromaDB
    context_parts: list[str] = []
    col = _rag_collection()
    if col:
        try:
            mem = col.get(where={"source": "memory"})
            mem_docs = mem.get("documents", [])[-5:]
            if mem_docs:
                context_parts.append("Agent memories (recent tasks):\n" +
                                     "\n".join(f"  - {d[:120]}" for d in mem_docs))
        except Exception:
            pass
        try:
            skills = col.get(where={"source": "skill"})
            skill_ids = skills.get("ids", [])
            skill_docs = skills.get("documents", [])
            if skill_ids:
                context_parts.append("Available skills:\n" +
                                     "\n".join(f"  - {i.removeprefix(_SKILL_PREFIX)}: {d[:80]}"
                                               for i, d in zip(skill_ids, skill_docs)))
        except Exception:
            pass
    try:
        q = TaskQueue()
        routines = q.list_cron_jobs()
        if routines:
            context_parts.append("Scheduled routines:\n" +
                                 "\n".join(f"  - {r['name']}: {r['task_text'][:60]}"
                                           for r in routines))
    except Exception:
        pass

    system = _CHAT_SYSTEM
    if context_parts:
        system += "\n\nContext:\n" + "\n\n".join(context_parts)

    _session_history.append({"role": "user", "content": text})

    try:
        key_file = os.path.join(_HERE, "..", "anthropic", "api_key.txt")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key and os.path.isfile(key_file):
            key = open(key_file).read().strip()

        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=512,
            system=system,
            messages=_session_history,
        )
        reply = msg.content[0].text
        _session_history.append({"role": "assistant", "content": reply})
        # Keep session history bounded
        if len(_session_history) > 20:
            _session_history = _session_history[-20:]
        print()
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            Console().print(Markdown(reply))
        except ImportError:
            print(reply)
        print()
    except Exception as e:
        print(f"{RED}  chat error: {e}{R}")
        _session_history.pop()  # remove the user message on failure


# ── Alert watcher helpers ────────────────────────────────────────────────────

def _drain_alerts(llm: str, serial: str) -> None:
    """Drain the alert queue and display/execute pending items.

    Called at the top of every REPL iteration — BEFORE input() blocks —
    so alerts never corrupt the user's active readline input line.

    Each item can be:
      "alert"  — print a highlighted message to the terminal.
      "auto"   — print then synchronously run the agent task.
      "burst"  — coalesced summary when > _BURST_THRESHOLD items fired in one tick.
    """
    items: list[dict] = []
    while True:
        try:
            items.append(_alert_queue.get_nowait())
        except _queue_mod.Empty:
            break

    if not items:
        return

    _ts = lambda: time.strftime("%H:%M:%S")
    print()   # blank line separates alerts from previous output

    for item in items:
        kind = item.get("kind")

        if kind == "alert":
            notif = item.get("notif", {})
            print(f"{YLW}{B}  ⚡ [ALERT]{R} {item['msg']}")
            print(f"  {DIM}rule:{item['rule_id']} | {notif.get('package','?')} | {_ts()}{R}")

        elif kind == "auto":
            notif = item.get("notif", {})
            task  = item["task"]
            print(f"{CYN}{B}  ⚡ [AUTO-RUN]{R} {task}")
            print(
                f"  {DIM}rule:{item['rule_id']} | "
                f"[{notif.get('package','')}] {notif.get('title','')}{R}"
            )
            print()
            try:
                _run(task, llm, serial)
            except Exception as exc:
                print(f"{RED}  auto-run failed: {exc}{R}")

        elif kind == "burst":
            print(f"{YLW}{B}  ⚡ [ALERT BURST]{R} {item['msg']}")
            for sub in item.get("items", [])[:_BURST_PREVIEW]:
                preview = sub.get("msg") or sub.get("task", "")
                print(f"  {DIM}  • [{sub['kind']}] {preview[:70]}{R}")
            overflow = len(item.get("items", [])) - _BURST_PREVIEW
            if overflow > 0:
                print(f"  {DIM}  … and {overflow} more{R}")

    print()   # blank line after alert block


_BURST_PREVIEW = 3   # lines shown in a burst summary


def _alert(args: list[str], llm: str, serial: str) -> None:
    """Handler for /alert — manage background notification trigger rules.

    Subcommands:
      /alert add "condition" → "message {title}"
      /alert add "condition" → auto: "task for agent {text}"
      /alert list
      /alert del  <rule-id>
      /alert pause <rule-id>
      /alert resume <rule-id>
      /alert test "condition" [package] [title] [text]
      /alert status
      /alert reload

    Condition DSL (case-insensitive substring, AND/OR/NOT):
      package:com.google.android.gm
      title:Alice AND text:meeting
      text:OTP OR text:verification
      NOT title:advertisement
      app:gmail                          (friendly name → package lookup)
      any:urgent                         (searches all fields)

    Template variables in action string: {package}, {title}, {text}, {app}
    """
    global _alert_watcher

    if _alert_watcher is None:
        print(f"{RED}  alert watcher not initialised — start the CLI normally{R}")
        return

    sub = args[1] if len(args) > 1 else "list"

    # ── add ───────────────────────────────────────────────────────────────────
    if sub == "add":
        if len(args) < 3:
            print(f"{RED}  /alert add \"condition\" → \"action\"{R}")
            print(f"  {DIM}alert:    → \"Email: {{title}}\"{R}")
            print(f"  {DIM}auto-run: → auto: \"Read email from {{title}} and reply\"{R}")
            return

        raw = " ".join(args[2:])

        # Locate → or -> separator
        sep_char = "→" if "→" in raw else ("->" if "->" in raw else None)
        if sep_char is None:
            print(f"{RED}  Missing → separator.  Example: /alert add \"text:OTP\" → \"OTP: {{text}}\"{R}")
            return

        condition_str, _, action_str = raw.partition(sep_char)
        condition_str = condition_str.strip().strip("\"'")
        action_str    = action_str.strip().strip("\"'")

        if not condition_str:
            print(f"{RED}  Condition cannot be empty{R}"); return
        if not action_str:
            print(f"{RED}  Action cannot be empty{R}"); return

        # Parse action type
        if re.match(r'^auto\s*:', action_str, re.IGNORECASE):
            action_type = "auto"
            action      = re.sub(r'^auto\s*:\s*', '', action_str, flags=re.IGNORECASE).strip("\"'")
        else:
            action_type = "alert"
            action      = action_str

        # Optional --rate N flag
        rate = 3
        if "--rate" in args:
            idx = args.index("--rate")
            try:
                rate = max(1, int(args[idx + 1]))
            except (ValueError, IndexError):
                print(f"{RED}  --rate requires an integer (e.g. --rate 5){R}"); return

        try:
            rule = _alert_watcher.add_rule(condition_str, action_type, action, rate)
        except ValueError as exc:
            print(f"{RED}  {exc}{R}"); return

        print(f"{GRN}  rule added{R}  {DIM}id:{rule.id}{R}")
        print(f"  {DIM}condition : {condition_str}{R}")
        print(f"  {DIM}action    : [{action_type}] {action}{R}")
        print(f"  {DIM}rate limit: {rate}/min{R}")

    # ── list ──────────────────────────────────────────────────────────────────
    elif sub == "list":
        rules = _alert_watcher.get_rules()
        if not rules:
            print(f"  {DIM}no alert rules  —  add one with /alert add{R}")
            return
        print(f"\n  {'ID':<13} {'ON':<4} {'#':<6} {'TYPE':<7}  {'CONDITION  →  ACTION'}")
        print(f"  {'─' * 72}")
        for r in rules:
            on  = f"{GRN}✓{R}" if r.enabled else f"{RED}✗{R}"
            tag = f"{CYN}auto{R}" if r.action_type == "auto" else "alrt"
            cond   = r.condition[:32]
            action = r.action[:28]
            print(f"  {r.id:<13} {on}   {r.fire_count:<6} {tag:<7}  "
                  f"{DIM}{cond}  →  {action}{R}")
        print()

    # ── del ───────────────────────────────────────────────────────────────────
    elif sub in ("del", "delete", "rm"):
        if len(args) < 3:
            print(f"{RED}  /alert del <rule-id>{R}"); return
        rid = args[2]
        if _alert_watcher.del_rule(rid):
            print(f"{GRN}  rule {rid} removed{R}")
        else:
            print(f"{RED}  rule not found: {rid!r}{R}")

    # ── pause / resume ────────────────────────────────────────────────────────
    elif sub == "pause":
        if len(args) < 3:
            print(f"{RED}  /alert pause <rule-id>{R}"); return
        if _alert_watcher.set_enabled(args[2], False):
            print(f"{YLW}  rule {args[2]} paused{R}")
        else:
            print(f"{RED}  rule not found{R}")

    elif sub == "resume":
        if len(args) < 3:
            print(f"{RED}  /alert resume <rule-id>{R}"); return
        if _alert_watcher.set_enabled(args[2], True):
            print(f"{GRN}  rule {args[2]} resumed{R}")
        else:
            print(f"{RED}  rule not found{R}")

    # ── test ──────────────────────────────────────────────────────────────────
    elif sub == "test":
        # /alert test "condition" [package] [title] [text]
        if len(args) < 3:
            print(f"{RED}  /alert test \"condition\" [package] [title] [text]{R}")
            return
        cond    = args[2].strip("\"'")
        package = args[3].strip("\"'") if len(args) > 3 else ""
        title   = args[4].strip("\"'") if len(args) > 4 else ""
        text    = args[5].strip("\"'") if len(args) > 5 else ""
        try:
            matches = _alert_watcher.test_condition(cond, package, title, text)
        except Exception as exc:
            print(f"{RED}  error: {exc}{R}"); return
        mark = f"{GRN}MATCH ✓{R}" if matches else f"{RED}no match ✗{R}"
        print(f"  {mark}")
        print(f"  {DIM}package={package!r}  title={title!r}  text={text!r}{R}")

    # ── status ────────────────────────────────────────────────────────────────
    elif sub == "status":
        rules  = _alert_watcher.get_rules()
        active = sum(1 for r in rules if r.enabled)
        paused = len(rules) - active
        total_fires = sum(r.fire_count for r in rules)
        print(f"  Watcher : {GRN}RUNNING{R}  poll every {_POLL_INTERVAL}s")
        print(f"  Rules   : {active} active, {paused} paused  ({len(rules)} total)")
        print(f"  Fires   : {total_fires} total across all rules")
        print(f"  Rules   : {_alert_watcher._rules_path}")

    # ── reload ────────────────────────────────────────────────────────────────
    elif sub == "reload":
        n = _alert_watcher.reload_rules()
        print(f"{GRN}  {n} rule(s) loaded from disk{R}")

    else:
        print(f"  /alert  add | list | del | pause | resume | test | status | reload")


_POLL_INTERVAL = 5   # exported so _drain_alerts can reference it


# ── Dispatch table ──────────────────────────────────────────────────────────

_DISPATCH: dict[str, Any] = {
    "run":     lambda args, llm, serial: (
        print(f"{RED}  /run <task>{R}") if not args else _run(" ".join(args), llm, serial)
    ),
    "queue":   _queue,
    "routine": _routine,
    "skill":   lambda args, llm, serial: _skill(args),
    "memory":  lambda args, llm, serial: _memory(args),
    "alert":   _alert,
    "watch":   lambda args, llm, serial: _watch(llm, serial),
    "status":  lambda args, llm, serial: _status(),
    "help":    lambda args, llm, serial: print(HELP),
    "clear":   lambda args, llm, serial: os.system("clear"),
}

# ── REPL ───────────────────────────────────────────────────────────────────

def _launch_openclaw(serial: str) -> None:
    import subprocess
    try:
        subprocess.run(
            ["adb", "-s", serial, "shell", "am", "start",
             "-n", "com.openclaw.android.debug/com.openclaw.android.MainActivity"],
            capture_output=True, timeout=5,
        )
        print(f"{DIM}  OpenClaw launched on {serial}{R}")
    except Exception as e:
        print(f"{YLW}  could not launch OpenClaw: {e}{R}")


def repl(llm: str = "groq", serial: str = DEFAULT_SERIAL) -> None:
    global _alert_watcher

    print(BANNER)
    _launch_openclaw(serial)

    # Ensure ADB port-forward is in place before the alert watcher starts polling.
    # ContextAssembler sets this up per-run; we need it earlier for the watcher.
    import subprocess as _sp
    try:
        _sp.run(
            ["adb", "-s", serial, "forward", "tcp:8766", "tcp:8766"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass   # if ADB is unavailable the watcher backs off gracefully

    # Start background alert watcher.
    from alert_watcher import AlertWatcher
    _rules_path = pathlib.Path(_HERE).parent / "data" / "alert_rules.json"
    _alert_watcher = AlertWatcher(
        serial=serial,
        alert_queue=_alert_queue,
        rules_path=_rules_path,
    )
    _alert_watcher.start()
    n_rules = len(_alert_watcher.get_rules())
    print(
        f"{DIM}  alert watcher started — "
        f"{n_rules} rule(s) | poll every 5s | "
        f"/alert add to create triggers{R}"
    )

    try:
        readline.read_history_file(HISTORY)
    except FileNotFoundError:
        pass
    readline.set_history_length(500)

    try:
        while True:
            # Drain alert queue BEFORE showing prompt — never mid-keypress.
            _drain_alerts(llm, serial)

            try:
                line = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print(); break

            if not line:
                continue

            if line.startswith("/"):
                try:
                    parts = shlex.split(line[1:])
                except ValueError as e:
                    print(f"{RED}  parse error: {e}{R}"); continue
                if not parts:
                    continue
                cmd, *args = parts
                if cmd in ("exit", "quit"):
                    break
                handler = _DISPATCH.get(cmd)
                if handler:
                    try:
                        handler(args, llm, serial)
                    except KeyboardInterrupt:
                        print(f"\n{YLW}  interrupted{R}")
                else:
                    print(f"{RED}  unknown /{cmd} — try /help{R}")
            else:
                try:
                    _chat(line)
                except KeyboardInterrupt:
                    print(f"\n{YLW}  interrupted{R}")
    finally:
        # Graceful shutdown: flush fire counts and seen fingerprints to disk.
        if _alert_watcher is not None:
            _alert_watcher.stop()
            _alert_watcher.join(timeout=3)

    readline.write_history_file(HISTORY)
    print("bye.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PRISM Agent CLI")
    p.add_argument("--llm", choices=["groq", "claude", "deepseek", "local"], default="groq")
    p.add_argument("--serial", default=DEFAULT_SERIAL)
    a = p.parse_args()
    repl(llm=a.llm, serial=a.serial)
