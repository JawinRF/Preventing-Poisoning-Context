#!/usr/bin/env python3
"""
memory_lineage.py — SQLite-backed memory lineage graph.

Tracks which stored memories were in context when a new memory was saved,
including BOTH previously-retrieved memories (ChromaDB IDs) AND T3 device
sources (notifications, SMS, clipboard, contacts) that were live in context
at save time.

When any parent is later flagged — either a ChromaDB doc purged at L1/L2,
or a T3 source blocked by PRISM in a subsequent session — suspicion propagates
to every descended memory.

Architecture
────────────
  SQLite WAL  — graph edges (parent_id → child_id), session/retrieval log,
                T3 source metadata
  ChromaDB    — trust_score stored in document metadata (float 0.0–1.0)

Node types in the graph
───────────────────────
  ChromaDB ID   e.g. "mem_a1b2c3d4e5f6"  — stored memory
  T3 fingerprint e.g. "t3:notif:abc123"  — ephemeral device source

T3 fingerprints are stable across sessions (sha256, not hash()) so the
same Gmail notification in session 1 and session 2 has the same fingerprint.
This enables retroactive auto-flagging: content that passed PRISM in session 1
but is blocked in session 2 automatically propagates suspicion back to any
memories born from it in session 1.

Session model
─────────────
  One REPL invocation = one session.
  T3 sources allowed by PRISM → record_t3_source()
  Memories retrieved from ChromaDB → record_retrieval()
  User /memory save → record_save() creates edges: all above → new_doc_id

Suspicion propagation (SENTRY §10.4)
─────────────────────────────────────
  L_r(C) = Σ influence_weight(i→C) × suspicion(i)
  new_trust(C) = max(0, trust(C) − L_r(C))

  BFS with max_depth=3. Cumulative weight decays with path length.

Thread safety
─────────────
  All calls from main REPL thread. No locks needed.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

TRUST_THRESHOLD  = 0.30   # docs below this are filtered at retrieval (legacy binary gate)
L1_SUSPICION     = 1.00   # full suspicion — direct injection bypass
L2_SUSPICION     = 0.50   # partial suspicion — passed seal but flagged by MemShield
T3_SUSPICION     = 0.70   # T3 source blocked in a later session → retroactive flag
MAX_DEPTH        = 3      # BFS depth cap for propagation

T3_PREFIX = "t3:"        # distinguishes T3 fingerprints from ChromaDB doc IDs

# ── Autonomous memory birth priors ────────────────────────────────────────────
# Applied ONLY to origin="auto" memories (_record_experience path).
# Manual /memory save always gets trust=1.0 with origin="user".
PRIOR_CLEAN    = 0.60   # auto-memory: clean context, no T3 in session
PRIOR_T3       = 0.35   # auto-memory: T3 source was in context during run
PRIOR_FLAGGED  = 0.15   # auto-memory: Stage-1 causal-overlap tripped → audit-only
AUDIT_FLOOR    = 0.10   # tombstoned memories: row kept, trust floored, not retrievable
CORROB_GAMMA   = 0.50   # graduation factor: trust ← 1-(1-trust)·γ per corroboration
EDGE_ATTEN     = 0.90   # per-edge attenuation for BFS — replaces 1/N dilution
RETRIEVAL_BETA = 1.00   # effective_score = cosine_sim × trust^β at retrieval


# ── T3 fingerprint helper ─────────────────────────────────────────────────────

def t3_fp(source_type: str, **fields: str) -> str:
    """Stable sha256-based fingerprint for an ephemeral T3 device source.

    Uses sha256 (not hash()) so the same content gets the same fingerprint
    across Python restarts and sessions — enabling retroactive auto-flagging.

    Examples:
        t3_fp("notification", pkg="com.google.android.gm", title="Hi", text="...")
        t3_fp("clipboard",    content="some text")
        t3_fp("sms",          sender="+1234", body="your code is 9999")
        t3_fp("contact",      name="Alice",   note="Call me")
    """
    payload = source_type + "\x00" + "\x00".join(
        f"{k}={v}" for k, v in sorted(fields.items())
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:14]
    return f"{T3_PREFIX}{source_type[:5]}:{digest}"


# ── Module-level active graph ─────────────────────────────────────────────────
# Allows prism_cli.py and context_assembler.py to share state without
# passing objects through every call frame.

_active_graph:      "LineageGraph | None" = None
_active_session_id: str = ""


def set_active(graph: "LineageGraph", session_id: str) -> None:
    global _active_graph, _active_session_id
    _active_graph      = graph
    _active_session_id = session_id


def get_active() -> tuple["LineageGraph | None", str]:
    return _active_graph, _active_session_id


# ── LineageGraph ──────────────────────────────────────────────────────────────

class LineageGraph:
    """SQLite-backed directed graph: parent memory → child memory.

    Edges are created when the user saves a memory while other memories
    were retrieved in the same session. Suspicion propagates from flagged
    parents to children via BFS.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS retrievals (
        session_id   TEXT NOT NULL,
        doc_id       TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        PRIMARY KEY (session_id, doc_id)
    );

    CREATE TABLE IF NOT EXISTS edges (
        parent_id  TEXT NOT NULL,
        child_id   TEXT NOT NULL,
        weight     REAL NOT NULL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        PRIMARY KEY (parent_id, child_id)
    );

    -- T3 source metadata: notifications, SMS, clipboard, contacts.
    -- fingerprint is the t3_fp() value — stable across sessions.
    -- flagged=1 means PRISM blocked this source (in any session) after it
    -- had previously been allowed; propagation has run.
    CREATE TABLE IF NOT EXISTS t3_meta (
        fingerprint  TEXT PRIMARY KEY,
        source_type  TEXT NOT NULL,
        description  TEXT,
        package      TEXT,
        first_seen   TEXT NOT NULL,
        last_seen    TEXT NOT NULL,
        flagged      INTEGER NOT NULL DEFAULT 0
    );

    CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges (parent_id);
    CREATE INDEX IF NOT EXISTS idx_edges_child  ON edges (child_id);
    CREATE INDEX IF NOT EXISTS idx_ret_session  ON retrievals (session_id);
    CREATE INDEX IF NOT EXISTS idx_t3_flagged   ON t3_meta (flagged);
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        logger.info(f"[Lineage] graph at {self._path}")

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self) -> str:
        """Create a new session and return its ID."""
        sid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO sessions (id, created_at) VALUES (?, ?)",
            (sid, _now()),
        )
        self._conn.commit()
        logger.info(f"[Lineage] session started: {sid[:8]}")
        return sid

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_retrieval(self, session_id: str, doc_ids: list[str]) -> None:
        """Log which ChromaDB doc IDs were retrieved in this session."""
        if not doc_ids:
            return
        rows = [(session_id, did, _now()) for did in doc_ids]
        self._conn.executemany(
            "INSERT OR IGNORE INTO retrievals (session_id, doc_id, retrieved_at) VALUES (?,?,?)",
            rows,
        )
        self._conn.commit()
        logger.debug(f"[Lineage] recorded {len(doc_ids)} retrieval(s) for session {session_id[:8]}")

    def record_t3_source(
        self,
        session_id:  str,
        fingerprint: str,
        source_type: str,
        description: str = "",
        package:     str = "",
    ) -> None:
        """Record a T3 device source that was allowed into context this session.

        This does two things:
        1. Inserts/updates t3_meta so we know this fp was seen (for retroactive
           auto-flagging if it gets blocked in a future session).
        2. Inserts into retrievals so record_save() will link it as a parent of
           any memory saved in this session — same mechanism as ChromaDB IDs.
        """
        now = _now()
        self._conn.execute(
            """INSERT INTO t3_meta (fingerprint, source_type, description, package,
                                   first_seen, last_seen, flagged)
               VALUES (?,?,?,?,?,?,0)
               ON CONFLICT(fingerprint) DO UPDATE SET last_seen=excluded.last_seen""",
            (fingerprint, source_type, description, package, now, now),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO retrievals (session_id, doc_id, retrieved_at) VALUES (?,?,?)",
            (session_id, fingerprint, now),
        )
        self._conn.commit()
        logger.debug(f"[Lineage] T3 source recorded: {fingerprint} ({description[:40]})")

    def was_t3_seen_before(self, fingerprint: str, current_session_id: str = "") -> bool:
        """True if this T3 fingerprint appeared in a session OTHER than the current one.

        Used for auto-flagging: if content that passed PRISM in a past session
        is now blocked, it likely turned out to be malicious → retroactive flag.
        """
        row = self._conn.execute(
            "SELECT 1 FROM t3_meta WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        if not row:
            return False
        # Exists in meta — check it was seen in a different session
        if not current_session_id:
            return True
        other = self._conn.execute(
            "SELECT 1 FROM retrievals WHERE doc_id=? AND session_id != ?",
            (fingerprint, current_session_id),
        ).fetchone()
        return other is not None

    def flag_t3_source(
        self,
        fingerprint:     str,
        suspicion_delta: float,
        collection,
    ) -> int:
        """Mark a T3 source as malicious and propagate suspicion to child memories.

        Called when:
          - Same content passed PRISM in session N but is blocked in session N+1
            (retroactive auto-flag)
          - User manually flags via /memory lineage flag-source <fp>

        Returns number of child memories updated.
        """
        self._conn.execute(
            "UPDATE t3_meta SET flagged=1 WHERE fingerprint=?", (fingerprint,)
        )
        self._conn.commit()

        row = self._conn.execute(
            "SELECT description, source_type FROM t3_meta WHERE fingerprint=?",
            (fingerprint,),
        ).fetchone()
        desc = f"{row[1]}:{row[0][:40]}" if row else fingerprint

        logger.warning(
            f"[Lineage] T3 source FLAGGED: {fingerprint}  ({desc})  "
            f"suspicion={suspicion_delta:.2f}"
        )

        if collection is None:
            return 0
        return self.propagate_suspicion(fingerprint, suspicion_delta, collection)

    def record_save(self, session_id: str, new_doc_id: str) -> int:
        """Create edges from all docs retrieved in this session to new_doc_id.

        Returns number of edges created.
        """
        rows = self._conn.execute(
            "SELECT doc_id FROM retrievals WHERE session_id = ?", (session_id,)
        ).fetchall()
        parents = [r[0] for r in rows if r[0] != new_doc_id]
        if not parents:
            return 0

        # t-norm: fixed per-edge attenuation, not 1/N dilution.
        # 1/N was the trust-laundering loophole: N clean co-parents would wash
        # a single tainted parent's influence to near-zero. EDGE_ATTEN is constant
        # so a tainted parent always propagates at least EDGE_ATTEN × its suspicion.
        weight = EDGE_ATTEN
        now    = _now()
        self._conn.executemany(
            "INSERT OR IGNORE INTO edges (parent_id, child_id, weight, created_at) VALUES (?,?,?,?)",
            [(pid, new_doc_id, weight, now) for pid in parents],
        )
        self._conn.commit()
        logger.info(
            f"[Lineage] {new_doc_id[:12]} ← {len(parents)} parent(s) "
            f"(weight={weight:.3f} each)"
        )
        return len(parents)

    # ── Graph traversal ───────────────────────────────────────────────────────

    def get_children(
        self, source_id: str, max_depth: int = MAX_DEPTH
    ) -> list[tuple[str, float]]:
        """BFS from source_id. Returns [(child_id, cumulative_weight), ...].

        Cumulative weight decays with path length: A→B (0.5) → B→C (0.33)
        gives A's influence on C = 0.5 × 0.33 = 0.165.
        """
        visited: dict[str, float] = {}
        queue: list[tuple[str, float, int]] = [(source_id, 1.0, 0)]

        while queue:
            curr, weight, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            rows = self._conn.execute(
                "SELECT child_id, weight FROM edges WHERE parent_id = ?", (curr,)
            ).fetchall()

            for child_id, edge_weight in rows:
                if child_id == source_id:
                    continue   # skip cycles
                cumulative = weight * edge_weight
                if child_id not in visited or visited[child_id] < cumulative:
                    visited[child_id] = cumulative
                    queue.append((child_id, cumulative, depth + 1))

        return list(visited.items())

    def get_parents(self, doc_id: str) -> list[tuple[str, float]]:
        """Direct parents only (depth=1)."""
        rows = self._conn.execute(
            "SELECT parent_id, weight FROM edges WHERE child_id = ?", (doc_id,)
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ── Session source queries ────────────────────────────────────────────────

    def get_session_t3_count(self, session_id: str) -> int:
        """Count T3 fingerprints recorded in this session."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM retrievals WHERE session_id=? AND doc_id LIKE 't3:%'",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_t3_texts(self, session_id: str) -> list[str]:
        """Return T3 source descriptions for this session (for Stage-1 causal overlap)."""
        rows = self._conn.execute(
            """SELECT t.description FROM retrievals r
               JOIN t3_meta t ON t.fingerprint = r.doc_id
               WHERE r.session_id = ? AND r.doc_id LIKE 't3:%'""",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_session_parent_trusts(self, session_id: str, collection) -> list[float]:
        """Trust scores of all ChromaDB docs (not T3) retrieved in this session."""
        rows = self._conn.execute(
            "SELECT doc_id FROM retrievals WHERE session_id=? AND doc_id NOT LIKE 't3:%'",
            (session_id,),
        ).fetchall()
        return [self.get_trust(did, collection) for (did,) in rows]

    # ── Trust management ─────────────────────────────────────────────────────

    def get_trust(self, doc_id: str, collection) -> float:
        """Read trust_score from ChromaDB metadata. Defaults to 1.0 (clean)."""
        try:
            result = collection.get(ids=[doc_id], include=["metadatas"])
            if not result["ids"]:
                return 1.0
            meta = result["metadatas"][0] or {}
            return float(meta.get("trust_score", 1.0))
        except Exception:
            return 1.0

    def corroborate(self, doc_id: str, collection) -> float:
        """Graduate an autonomous memory's trust after independent corroboration.

        trust ← 1 - (1 - trust) × CORROB_GAMMA  (one graduation step per call)

        Only applies to origin="auto" memories. User memories are untouched.
        Returns the new trust score (or current score if skipped).
        """
        try:
            result = collection.get(ids=[doc_id], include=["metadatas"])
            if not result["ids"]:
                return 0.0
            meta = dict(result["metadatas"][0] or {})
            if meta.get("origin", "user") != "auto":
                return float(meta.get("trust_score", 1.0))
            current   = float(meta.get("trust_score", PRIOR_T3))
            new_trust = 1.0 - (1.0 - current) * CORROB_GAMMA
            meta["trust_score"] = new_trust
            collection.update(ids=[doc_id], metadatas=[meta])
            logger.info(
                f"[Lineage] {doc_id[:12]} corroborated: "
                f"trust {current:.3f} → {new_trust:.3f}"
            )
            return new_trust
        except Exception as exc:
            logger.warning(f"[Lineage] corroborate({doc_id[:12]}) failed: {exc}")
            return 0.0

    def tombstone(self, doc_id: str, collection) -> None:
        """Drive trust to AUDIT_FLOOR — memory kept for audit, not retrievable.

        Non-destructive alternative to collection.delete(): the row survives,
        can be inspected, and can be recovered by corroborate() if it was a FP.
        """
        try:
            result = collection.get(ids=[doc_id], include=["metadatas"])
            if not result["ids"]:
                return
            meta = dict(result["metadatas"][0] or {})
            meta["trust_score"] = AUDIT_FLOOR
            meta["tombstoned"]  = "1"
            collection.update(ids=[doc_id], metadatas=[meta])
            logger.warning(
                f"[Lineage] {doc_id[:12]} TOMBSTONED "
                f"(trust → {AUDIT_FLOOR}, row retained for audit)"
            )
        except Exception as exc:
            logger.warning(f"[Lineage] tombstone({doc_id[:12]}) failed: {exc}")

    def propagate_suspicion(
        self,
        source_id:      str,
        suspicion_delta: float,
        collection,
    ) -> int:
        """Propagate suspicion from source_id to all descendant memories.

        Formula: new_trust(C) = max(0, trust(C) - weight × suspicion_delta)

        Returns number of documents updated.
        """
        children = self.get_children(source_id)
        if not children:
            logger.debug(f"[Lineage] {source_id[:12]} has no descendants — nothing to propagate")
            return 0

        updated = 0
        for child_id, weight in children:
            try:
                result = collection.get(ids=[child_id], include=["metadatas"])
                if not result["ids"]:
                    continue
                meta          = dict(result["metadatas"][0] or {})
                current_trust = float(meta.get("trust_score", 1.0))
                penalty       = weight * suspicion_delta
                new_trust     = max(0.0, current_trust - penalty)

                # Merge trust_score into existing metadata (full replace required by ChromaDB)
                meta["trust_score"] = new_trust
                collection.update(ids=[child_id], metadatas=[meta])
                updated += 1

                logger.warning(
                    f"[Lineage] {child_id[:12]} trust: "
                    f"{current_trust:.3f} → {new_trust:.3f} "
                    f"(penalty={penalty:.3f}, parent={source_id[:12]})"
                )
                if new_trust < TRUST_THRESHOLD:
                    logger.warning(
                        f"[Lineage] {child_id[:12]} BELOW THRESHOLD "
                        f"({new_trust:.3f} < {TRUST_THRESHOLD}) — "
                        f"will be filtered at next retrieval"
                    )
            except Exception as exc:
                logger.warning(f"[Lineage] failed to update {child_id[:12]}: {exc}")

        logger.info(
            f"[Lineage] propagated suspicion={suspicion_delta:.2f} "
            f"from {source_id[:12]} to {updated}/{len(children)} descendant(s)"
        )
        return updated

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        n_sessions   = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        n_edges      = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_retrievals = self._conn.execute("SELECT COUNT(*) FROM retrievals").fetchone()[0]
        n_t3         = self._conn.execute("SELECT COUNT(*) FROM t3_meta").fetchone()[0]
        n_t3_flagged = self._conn.execute("SELECT COUNT(*) FROM t3_meta WHERE flagged=1").fetchone()[0]
        return {
            "sessions":   n_sessions,
            "edges":      n_edges,
            "retrievals": n_retrievals,
            "t3_sources": n_t3,
            "t3_flagged": n_t3_flagged,
            "db_path":    str(self._path),
        }

    def close(self) -> None:
        self._conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
