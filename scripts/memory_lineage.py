#!/usr/bin/env python3
"""
memory_lineage.py — SQLite-backed memory provenance graph.

Tracks which stored memories and live device sources (T3: notifications, SMS,
clipboard, contacts) were in context when a new memory was born, and keeps a
persistent, queryable record of everything that happened to each node.

Schema (v2)
───────────
  sessions    one REPL/agent invocation; carries the task that was running
  nodes       registry of every graph participant: stored memories, skills,
              and T3 device sources — with label, type, origin, flag state.
              Labels survive ChromaDB deletion, so history stays renderable.
  edges       parent → child provenance, typed (retrieval | t3_context),
              stamped with the session and task that created them
  retrievals  which nodes were live in context per session
  events      append-only ledger: save, retrieval, flag, trust_penalty,
              corroborate, tombstone — the "why" behind every trust change

Trust itself lives in ChromaDB document metadata (single source of truth);
this graph records structure and history around it.

Suspicion propagation (SENTRY §10.4)
─────────────────────────────────────
  new_trust(C) = max(0, trust(C) − cumulative_weight × suspicion)
  BFS with max_depth=3; per-edge attenuation EDGE_ATTEN (no 1/N dilution).

Legacy v1 databases (bare edges + t3_meta) are migrated in place on open.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

TRUST_THRESHOLD  = 0.30
L1_SUSPICION     = 1.00
L2_SUSPICION     = 0.50
T3_SUSPICION     = 0.70
MAX_DEPTH        = 3

T3_PREFIX = "t3:"

PRIOR_CLEAN    = 0.60
PRIOR_T3       = 0.35
PRIOR_FLAGGED  = 0.15
AUDIT_FLOOR    = 0.10
CORROB_GAMMA       = 0.50
CORROB_SIM_THRESH  = 0.70
EDGE_ATTEN         = 0.90
RETRIEVAL_BETA     = 1.00


def t3_fp(source_type: str, **fields: str) -> str:
    """Stable sha256-based fingerprint for an ephemeral T3 device source."""
    payload = source_type + "\x00" + "\x00".join(
        f"{k}={v}" for k, v in sorted(fields.items())
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:14]
    return f"{T3_PREFIX}{source_type[:5]}:{digest}"


_active_graph:      "LineageGraph | None" = None
_active_session_id: str = ""


def set_active(graph: "LineageGraph", session_id: str) -> None:
    global _active_graph, _active_session_id
    _active_graph      = graph
    _active_session_id = session_id


def get_active() -> tuple["LineageGraph | None", str]:
    return _active_graph, _active_session_id


class LineageGraph:
    """Provenance graph over memories, skills, and T3 device sources."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        task       TEXT NOT NULL DEFAULT ''
    );

    CREATE TABLE IF NOT EXISTS nodes (
        id          TEXT PRIMARY KEY,
        node_type   TEXT NOT NULL DEFAULT 'memory',
        source_type TEXT NOT NULL DEFAULT '',
        label       TEXT NOT NULL DEFAULT '',
        package     TEXT NOT NULL DEFAULT '',
        origin      TEXT NOT NULL DEFAULT '',
        first_seen  TEXT NOT NULL,
        last_seen   TEXT NOT NULL,
        flagged     INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS edges (
        parent_id  TEXT NOT NULL,
        child_id   TEXT NOT NULL,
        weight     REAL NOT NULL DEFAULT 1.0,
        edge_type  TEXT NOT NULL DEFAULT 'retrieval',
        session_id TEXT NOT NULL DEFAULT '',
        task       TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        PRIMARY KEY (parent_id, child_id)
    );

    CREATE TABLE IF NOT EXISTS retrievals (
        session_id   TEXT NOT NULL,
        doc_id       TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        PRIMARY KEY (session_id, doc_id)
    );

    CREATE TABLE IF NOT EXISTS events (
        seq        INTEGER PRIMARY KEY AUTOINCREMENT,
        ts         TEXT NOT NULL,
        kind       TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        object_id  TEXT NOT NULL DEFAULT '',
        value      REAL,
        detail     TEXT NOT NULL DEFAULT '',
        session_id TEXT NOT NULL DEFAULT ''
    );

    CREATE INDEX IF NOT EXISTS idx_edges_parent  ON edges (parent_id);
    CREATE INDEX IF NOT EXISTS idx_edges_child   ON edges (child_id);
    CREATE INDEX IF NOT EXISTS idx_ret_session   ON retrievals (session_id);
    CREATE INDEX IF NOT EXISTS idx_nodes_type    ON nodes (node_type);
    CREATE INDEX IF NOT EXISTS idx_nodes_flagged ON nodes (flagged);
    CREATE INDEX IF NOT EXISTS idx_events_subj   ON events (subject_id);
    CREATE INDEX IF NOT EXISTS idx_events_kind   ON events (kind);
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._backfill_nodes()
        logger.info(f"[Lineage] graph at {self._path}")

    # ── Migration ─────────────────────────────────────────────────────────────

    def _table_columns(self, table: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def _table_exists(self, table: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None

    def _migrate(self) -> None:
        if self._table_exists("sessions") and "task" not in self._table_columns("sessions"):
            self._conn.execute("ALTER TABLE sessions ADD COLUMN task TEXT NOT NULL DEFAULT ''")
        if self._table_exists("edges"):
            cols = self._table_columns("edges")
            if "edge_type" not in cols:
                self._conn.execute(
                    "ALTER TABLE edges ADD COLUMN edge_type TEXT NOT NULL DEFAULT 'retrieval'"
                )
                self._conn.execute(
                    f"UPDATE edges SET edge_type='t3_context' WHERE parent_id LIKE '{T3_PREFIX}%'"
                )
            if "session_id" not in cols:
                self._conn.execute("ALTER TABLE edges ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
            if "task" not in cols:
                self._conn.execute("ALTER TABLE edges ADD COLUMN task TEXT NOT NULL DEFAULT ''")
        if self._table_exists("t3_meta"):
            self._conn.executescript(self._SCHEMA)
            self._conn.execute(
                """INSERT OR IGNORE INTO nodes
                       (id, node_type, source_type, label, package,
                        origin, first_seen, last_seen, flagged)
                   SELECT fingerprint, 't3', source_type,
                          COALESCE(description, ''), COALESCE(package, ''),
                          'device', first_seen, last_seen, flagged
                   FROM t3_meta"""
            )
            self._conn.execute("DROP TABLE t3_meta")
        self._conn.commit()

    def _backfill_nodes(self) -> None:
        now = _now()
        ids: set[str] = set()
        for (pid, cid) in self._conn.execute("SELECT parent_id, child_id FROM edges"):
            ids.add(pid)
            ids.add(cid)
        for (did,) in self._conn.execute("SELECT DISTINCT doc_id FROM retrievals"):
            ids.add(did)
        rows = [
            (i, "t3" if i.startswith(T3_PREFIX) else "memory", now, now)
            for i in ids
        ]
        if rows:
            self._conn.executemany(
                """INSERT OR IGNORE INTO nodes (id, node_type, first_seen, last_seen)
                   VALUES (?,?,?,?)""",
                rows,
            )
            self._conn.commit()

    # ── Node registry ─────────────────────────────────────────────────────────

    def touch_node(
        self,
        node_id:     str,
        node_type:   str = "memory",
        label:       str = "",
        source_type: str = "",
        package:     str = "",
        origin:      str = "",
    ) -> None:
        """Insert or refresh a node. A non-empty label always wins over an empty one."""
        now = _now()
        self._conn.execute(
            """INSERT INTO nodes (id, node_type, source_type, label, package,
                                  origin, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   last_seen = excluded.last_seen,
                   label     = CASE WHEN excluded.label != ''
                                    THEN excluded.label ELSE nodes.label END,
                   origin    = CASE WHEN excluded.origin != ''
                                    THEN excluded.origin ELSE nodes.origin END""",
            (node_id, node_type, source_type, _clip(label), package, origin, now, now),
        )
        self._conn.commit()

    def get_node(self, node_id: str) -> dict | None:
        row = self._conn.execute(
            """SELECT id, node_type, source_type, label, package, origin,
                      first_seen, last_seen, flagged
               FROM nodes WHERE id=?""",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        keys = ("id", "node_type", "source_type", "label", "package",
                "origin", "first_seen", "last_seen", "flagged")
        return dict(zip(keys, row))

    def get_label(self, node_id: str) -> str:
        node = self.get_node(node_id)
        return node["label"] if node else ""

    # ── Event ledger ──────────────────────────────────────────────────────────

    def _event(
        self,
        kind:       str,
        subject_id: str,
        object_id:  str = "",
        value:      float | None = None,
        detail:     str = "",
        session_id: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO events (ts, kind, subject_id, object_id, value, detail, session_id)
               VALUES (?,?,?,?,?,?,?)""",
            (_now(), kind, subject_id, object_id, value, _clip(detail), session_id),
        )
        self._conn.commit()

    def recent_events(self, limit: int = 20, subject_id: str = "") -> list[dict]:
        if subject_id:
            rows = self._conn.execute(
                """SELECT ts, kind, subject_id, object_id, value, detail, session_id
                   FROM events WHERE subject_id=? OR object_id=?
                   ORDER BY seq DESC LIMIT ?""",
                (subject_id, subject_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT ts, kind, subject_id, object_id, value, detail, session_id
                   FROM events ORDER BY seq DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        keys = ("ts", "kind", "subject_id", "object_id", "value", "detail", "session_id")
        return [dict(zip(keys, r)) for r in rows]

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def start_session(self, task: str = "") -> str:
        sid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO sessions (id, created_at, task) VALUES (?, ?, ?)",
            (sid, _now(), _clip(task)),
        )
        self._conn.commit()
        logger.info(f"[Lineage] session started: {sid[:8]}")
        return sid

    def set_session_task(self, session_id: str, task: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET task=? WHERE id=?", (_clip(task), session_id)
        )
        self._conn.commit()

    def get_session_task(self, session_id: str) -> str:
        row = self._conn.execute(
            "SELECT task FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return row[0] if row else ""

    # ── Recording ─────────────────────────────────────────────────────────────

    def record_retrieval(
        self, session_id: str, doc_ids: list[str], labels: dict[str, str] | None = None
    ) -> None:
        """Log ChromaDB doc IDs retrieved this session; register/refresh their nodes."""
        if not doc_ids:
            return
        now = _now()
        labels = labels or {}
        for did in doc_ids:
            self.touch_node(did, node_type="memory", label=labels.get(did, ""))
        self._conn.executemany(
            "INSERT OR IGNORE INTO retrievals (session_id, doc_id, retrieved_at) VALUES (?,?,?)",
            [(session_id, did, now) for did in doc_ids],
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
        """Record a T3 device source allowed into context this session."""
        is_new = self.get_node(fingerprint) is None
        self.touch_node(
            fingerprint,
            node_type="t3",
            label=description,
            source_type=source_type,
            package=package,
            origin="device",
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO retrievals (session_id, doc_id, retrieved_at) VALUES (?,?,?)",
            (session_id, fingerprint, _now()),
        )
        self._conn.commit()
        if is_new:
            self._event(
                "t3_first_seen", fingerprint,
                detail=f"{source_type}: {description}", session_id=session_id,
            )
        logger.debug(f"[Lineage] T3 source recorded: {fingerprint} ({description[:40]})")

    def was_t3_seen_before(self, fingerprint: str, current_session_id: str = "") -> bool:
        """True if this T3 fingerprint appeared in a session other than the current one."""
        if self.get_node(fingerprint) is None:
            return False
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
        """Mark a T3 source malicious and propagate suspicion to descendants."""
        self._conn.execute(
            "UPDATE nodes SET flagged=1 WHERE id=?", (fingerprint,)
        )
        self._conn.commit()

        node = self.get_node(fingerprint) or {}
        desc = f"{node.get('source_type', '')}:{node.get('label', '')[:40]}" or fingerprint
        self._event("flag", fingerprint, value=suspicion_delta, detail=desc)

        logger.warning(
            f"[Lineage] T3 source FLAGGED: {fingerprint}  ({desc})  "
            f"suspicion={suspicion_delta:.2f}"
        )

        if collection is None:
            return 0
        return self.propagate_suspicion(fingerprint, suspicion_delta, collection)

    def record_save(
        self, session_id: str, new_doc_id: str, label: str = "", task: str = ""
    ) -> int:
        """Create provenance edges from everything in context this session to new_doc_id."""
        if not task:
            task = self.get_session_task(session_id)
        self.touch_node(new_doc_id, node_type="memory", label=label)

        rows = self._conn.execute(
            "SELECT doc_id FROM retrievals WHERE session_id = ?", (session_id,)
        ).fetchall()
        parents = [r[0] for r in rows if r[0] != new_doc_id]
        if not parents:
            self._event(
                "save", new_doc_id, detail=f"no parents | {label}", session_id=session_id
            )
            return 0

        weight = EDGE_ATTEN
        now    = _now()
        self._conn.executemany(
            """INSERT OR IGNORE INTO edges
                   (parent_id, child_id, weight, edge_type, session_id, task, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    pid, new_doc_id, weight,
                    "t3_context" if pid.startswith(T3_PREFIX) else "retrieval",
                    session_id, _clip(task), now,
                )
                for pid in parents
            ],
        )
        self._conn.commit()
        self._event(
            "save", new_doc_id, value=float(len(parents)),
            detail=f"{len(parents)} parent(s) | {label}", session_id=session_id,
        )
        logger.info(
            f"[Lineage] {new_doc_id[:12]} ← {len(parents)} parent(s) "
            f"(weight={weight:.3f} each)"
        )
        return len(parents)

    # ── Graph traversal ───────────────────────────────────────────────────────

    def get_children(
        self, source_id: str, max_depth: int = MAX_DEPTH
    ) -> list[tuple[str, float]]:
        """BFS from source_id. Returns [(child_id, cumulative_weight), ...]."""
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
                    continue
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

    def get_parent_edges(self, doc_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT parent_id, weight, edge_type, task, created_at
               FROM edges WHERE child_id = ?""",
            (doc_id,),
        ).fetchall()
        keys = ("parent_id", "weight", "edge_type", "task", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def get_child_edges(self, doc_id: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT child_id, weight, edge_type, task, created_at
               FROM edges WHERE parent_id = ?""",
            (doc_id,),
        ).fetchall()
        keys = ("child_id", "weight", "edge_type", "task", "created_at")
        return [dict(zip(keys, r)) for r in rows]

    def describe(self, doc_id: str, collection=None) -> dict:
        """Everything known about one node: identity, trust, ancestry, history."""
        node = self.get_node(doc_id) or {
            "id": doc_id, "node_type": "unknown", "source_type": "",
            "label": "", "package": "", "origin": "",
            "first_seen": "", "last_seen": "", "flagged": 0,
        }
        trust = self.get_trust(doc_id, collection) if collection is not None else None

        parents = []
        for e in self.get_parent_edges(doc_id):
            p = self.get_node(e["parent_id"]) or {}
            parents.append({**e, "label": p.get("label", ""),
                            "node_type": p.get("node_type", ""),
                            "flagged": p.get("flagged", 0)})

        children = []
        for cid, cum in sorted(self.get_children(doc_id), key=lambda x: -x[1]):
            c = self.get_node(cid) or {}
            children.append({
                "child_id": cid, "cumulative_weight": cum,
                "label": c.get("label", ""), "node_type": c.get("node_type", ""),
                "flagged": c.get("flagged", 0),
            })

        return {
            "node":     node,
            "trust":    trust,
            "parents":  parents,
            "children": children,
            "events":   self.recent_events(limit=10, subject_id=doc_id),
        }

    def export_graph(self) -> dict:
        """Whole graph as a JSON-friendly dict (nodes + typed edges)."""
        nodes = [
            dict(zip(
                ("id", "node_type", "source_type", "label", "origin", "flagged"), r
            ))
            for r in self._conn.execute(
                "SELECT id, node_type, source_type, label, origin, flagged FROM nodes"
            )
        ]
        edges = [
            dict(zip(
                ("parent_id", "child_id", "weight", "edge_type", "task", "created_at"), r
            ))
            for r in self._conn.execute(
                "SELECT parent_id, child_id, weight, edge_type, task, created_at FROM edges"
            )
        ]
        return {"nodes": nodes, "edges": edges}

    # ── Session source queries ────────────────────────────────────────────────

    def get_session_t3_count(self, session_id: str) -> int:
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM retrievals WHERE session_id=? AND doc_id LIKE '{T3_PREFIX}%'",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_session_t3_texts(self, session_id: str) -> list[str]:
        rows = self._conn.execute(
            f"""SELECT n.label FROM retrievals r
                JOIN nodes n ON n.id = r.doc_id
                WHERE r.session_id = ? AND r.doc_id LIKE '{T3_PREFIX}%'""",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_session_t3_sources(self, session_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            f"""SELECT r.doc_id, COALESCE(n.label, '') FROM retrievals r
                LEFT JOIN nodes n ON n.id = r.doc_id
                WHERE r.session_id = ? AND r.doc_id LIKE '{T3_PREFIX}%'""",
            (session_id,),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_session_retrieved_ids(self, session_id: str) -> set[str]:
        rows = self._conn.execute(
            f"SELECT doc_id FROM retrievals WHERE session_id=? AND doc_id NOT LIKE '{T3_PREFIX}%'",
            (session_id,),
        ).fetchall()
        return {r[0] for r in rows}

    def get_session_parent_trusts(self, session_id: str, collection) -> list[float]:
        rows = self._conn.execute(
            f"SELECT doc_id FROM retrievals WHERE session_id=? AND doc_id NOT LIKE '{T3_PREFIX}%'",
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
        """Graduate an autonomous memory's trust after independent corroboration."""
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
            self._event(
                "corroborate", doc_id, value=new_trust,
                detail=f"trust {current:.3f} → {new_trust:.3f}",
            )
            logger.info(
                f"[Lineage] {doc_id[:12]} corroborated: "
                f"trust {current:.3f} → {new_trust:.3f}"
            )
            return new_trust
        except Exception as exc:
            logger.warning(f"[Lineage] corroborate({doc_id[:12]}) failed: {exc}")
            return 0.0

    def tombstone(self, doc_id: str, collection) -> None:
        """Drive trust to AUDIT_FLOOR — memory kept for audit, not retrievable."""
        try:
            result = collection.get(ids=[doc_id], include=["metadatas"])
            if not result["ids"]:
                return
            meta = dict(result["metadatas"][0] or {})
            prior = float(meta.get("trust_score", 1.0))
            meta["trust_score"] = AUDIT_FLOOR
            meta["tombstoned"]  = "1"
            collection.update(ids=[doc_id], metadatas=[meta])
            self._event(
                "tombstone", doc_id, value=AUDIT_FLOOR,
                detail=f"trust {prior:.3f} → {AUDIT_FLOOR} (row retained for audit)",
            )
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
        """Propagate suspicion from source_id to all descendant memories."""
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

                meta["trust_score"] = new_trust
                collection.update(ids=[child_id], metadatas=[meta])
                updated += 1
                self._event(
                    "trust_penalty", child_id, object_id=source_id, value=penalty,
                    detail=f"trust {current_trust:.3f} → {new_trust:.3f}",
                )

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

        self._event(
            "propagate", source_id, value=suspicion_delta,
            detail=f"{updated}/{len(children)} descendant(s) penalised",
        )
        logger.info(
            f"[Lineage] propagated suspicion={suspicion_delta:.2f} "
            f"from {source_id[:12]} to {updated}/{len(children)} descendant(s)"
        )
        return updated

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        q = lambda sql: self._conn.execute(sql).fetchone()[0]
        return {
            "sessions":   q("SELECT COUNT(*) FROM sessions"),
            "edges":      q("SELECT COUNT(*) FROM edges"),
            "retrievals": q("SELECT COUNT(*) FROM retrievals"),
            "nodes":      q("SELECT COUNT(*) FROM nodes"),
            "events":     q("SELECT COUNT(*) FROM events"),
            "t3_sources": q("SELECT COUNT(*) FROM nodes WHERE node_type='t3'"),
            "t3_flagged": q("SELECT COUNT(*) FROM nodes WHERE node_type='t3' AND flagged=1"),
            "db_path":    str(self._path),
        }

    def close(self) -> None:
        self._conn.close()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clip(text: str, limit: int = 160) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit]
