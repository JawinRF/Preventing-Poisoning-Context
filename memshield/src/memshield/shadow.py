"""
shadow.py -- Shadow synthetic memory for RAG poisoning defense.

Model-generated content (summaries, synthesized answers, chain-of-thought
artifacts) must be stored separately from human-authored source documents
to prevent self-reinforcing hallucination loops and injection amplification.

Design:
  - Shadow entries have a TTL (time-to-live) and expire automatically.
  - Shadow entries require corroboration: they are only promoted to the
    primary store when N independent source documents support the same claim.
  - Uncorroborated shadow entries are served with a low authority prior.
  - Shadow entries that contradict source documents are flagged for review.

Storage is a simple JSON-lines file (no external DB dependency), with an
in-memory index for fast lookup. For production scale, swap the backend
for a proper DB (the interface is the same).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ShadowEntry:
    """A single shadow memory entry."""
    entry_id: str
    text: str
    source_query: str              # query that generated this content
    generator: str                 # which model/pipeline produced it
    created_ts: float              # epoch seconds
    ttl_seconds: float             # time-to-live
    corroboration_count: int = 0   # how many source docs support this
    corroboration_required: int = 2  # minimum to promote
    promoted: bool = False         # True once corroborated and moved to primary
    authority: float = 0.20        # low authority by default (synthetic)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > (self.created_ts + self.ttl_seconds)

    @property
    def is_corroborated(self) -> bool:
        return self.corroboration_count >= self.corroboration_required

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ShadowEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ShadowMemory:
    """Shadow store for model-generated content with TTL and corroboration.

    Usage:
        shadow = ShadowMemory("data/shadow.jsonl")
        eid = shadow.add("Paris is the capital of France", query="capital of France?",
                         generator="gpt-4", ttl_hours=24)
        shadow.corroborate(eid)  # called when a source doc confirms the claim
        shadow.corroborate(eid)  # second corroboration → promotable

        # At retrieval time:
        entries = shadow.query("capital of France")
        for e in entries:
            if e.is_expired:
                continue
            # use e.authority as the authority prior (low for uncorroborated)
    """

    def __init__(
        self,
        store_path: str | Path = "data/shadow_memory.jsonl",
        default_ttl_hours: float = 48.0,
        corroboration_required: int = 2,
    ):
        self._path = Path(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl_hours * 3600.0
        self._corr_required = corroboration_required
        self._entries: dict[str, ShadowEntry] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────

    def add(
        self,
        text: str,
        query: str,
        generator: str = "unknown",
        ttl_hours: float | None = None,
        authority: float = 0.20,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a synthetic entry to shadow memory. Returns entry_id."""
        entry_id = str(uuid.uuid4())[:12]
        entry = ShadowEntry(
            entry_id=entry_id,
            text=text,
            source_query=query,
            generator=generator,
            created_ts=time.time(),
            ttl_seconds=ttl_hours * 3600.0 if ttl_hours else self._default_ttl,
            corroboration_required=self._corr_required,
            authority=authority,
            metadata=metadata or {},
        )
        self._entries[entry_id] = entry
        self._append(entry)
        logger.debug(f"Shadow entry added: {entry_id} (TTL={entry.ttl_seconds/3600:.1f}h)")
        return entry_id

    def corroborate(self, entry_id: str) -> ShadowEntry | None:
        """Increment corroboration count. Returns updated entry or None."""
        entry = self._entries.get(entry_id)
        if not entry:
            return None
        if entry.is_expired:
            logger.debug(f"Cannot corroborate expired entry {entry_id}")
            return entry
        entry.corroboration_count += 1
        if entry.is_corroborated and not entry.promoted:
            entry.authority = min(0.70, entry.authority + 0.30)
            logger.info(f"Shadow entry {entry_id} corroborated ({entry.corroboration_count}x) — authority raised to {entry.authority:.2f}")
        self._persist()
        return entry

    def promote(self, entry_id: str) -> ShadowEntry | None:
        """Mark entry as promoted (moved to primary store by caller)."""
        entry = self._entries.get(entry_id)
        if not entry:
            return None
        if not entry.is_corroborated:
            logger.warning(f"Cannot promote uncorroborated entry {entry_id}")
            return entry
        entry.promoted = True
        entry.authority = 0.75
        self._persist()
        return entry

    def get(self, entry_id: str) -> ShadowEntry | None:
        return self._entries.get(entry_id)

    def query_active(self) -> list[ShadowEntry]:
        """Return all non-expired, non-promoted entries."""
        return [e for e in self._entries.values() if not e.is_expired and not e.promoted]

    def query_promotable(self) -> list[ShadowEntry]:
        """Return entries that are corroborated but not yet promoted."""
        return [
            e for e in self._entries.values()
            if e.is_corroborated and not e.promoted and not e.is_expired
        ]

    def query_expired(self) -> list[ShadowEntry]:
        """Return expired entries (for cleanup)."""
        return [e for e in self._entries.values() if e.is_expired]

    def gc(self) -> int:
        """Garbage-collect expired entries. Returns count removed."""
        expired = [eid for eid, e in self._entries.items() if e.is_expired]
        for eid in expired:
            del self._entries[eid]
        if expired:
            self._persist()
            logger.info(f"Shadow GC: removed {len(expired)} expired entries")
        return len(expired)

    def stats(self) -> dict[str, int]:
        """Return summary statistics."""
        active = [e for e in self._entries.values() if not e.is_expired]
        return {
            "total": len(self._entries),
            "active": len(active),
            "expired": len(self._entries) - len(active),
            "corroborated": sum(1 for e in active if e.is_corroborated),
            "promoted": sum(1 for e in self._entries.values() if e.promoted),
        }

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        """Load entries from JSONL store."""
        if not self._path.exists():
            return
        try:
            with self._path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    entry = ShadowEntry.from_dict(d)
                    self._entries[entry.entry_id] = entry
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Shadow memory load error: {e}")

    def _append(self, entry: ShadowEntry):
        """Append a single entry to the store file."""
        with self._path.open("a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def _persist(self):
        """Rewrite the full store (after mutations like corroborate/gc)."""
        with self._path.open("w") as f:
            for entry in self._entries.values():
                f.write(json.dumps(entry.to_dict()) + "\n")
