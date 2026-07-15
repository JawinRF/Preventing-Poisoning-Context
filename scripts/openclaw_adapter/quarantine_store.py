"""Quarantine ticket store for PRISM sidecar. In-memory index, JSONL-persisted."""
from __future__ import annotations

import dataclasses
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prism_shield.base import FinalizedTicket

_STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "quarantine_store.jsonl"

_lock = threading.Lock()
_tickets: dict[str, FinalizedTicket] = {}
_loaded = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_locked() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not _STORE_PATH.exists():
        return
    with _STORE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                ticket = FinalizedTicket(**row)
            except (json.JSONDecodeError, TypeError):
                continue
            _tickets[ticket.ticket_id] = ticket


def save_ticket(ticket: FinalizedTicket) -> None:
    with _lock:
        _load_locked()
        _tickets[ticket.ticket_id] = ticket
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _STORE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(ticket)) + "\n")


def get_ticket(ticket_id: str) -> Optional[FinalizedTicket]:
    with _lock:
        _load_locked()
        return _tickets.get(ticket_id)
