#!/usr/bin/env python3
"""
reembed_store.py — one-time migration: re-embed agent_kb under the new model.

ChromaDB binds the embedding function at collection creation; you cannot
change it in place. This reads every doc out, backs it up to JSON, deletes
the collection, recreates it with get_embedding_fn() (bge-small), and re-adds
everything so it is re-embedded under the new model. All metadata is
preserved verbatim — HMAC provenance seals, trust_score, origin, lineage,
skill bodies — only the vectors change.

Safety:
  * A timestamped JSON backup is written BEFORE anything is deleted.
  * If re-add fails, the backup is the recovery source (restore = re-run with
    --restore <backup.json>).

Usage:
  python scripts/reembed_store.py            # migrate
  python scripts/reembed_store.py --restore data/kb_backup_XXXX.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from embedding_fn import get_embedding_fn, embedding_tag

DB_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "chromadb")
COLLECTION = "agent_kb"
BATCH      = 100


def _dump(collection) -> dict:
    got = collection.get(include=["documents", "metadatas"])
    return {
        "ids":       got.get("ids", []),
        "documents": got.get("documents", []),
        "metadatas": got.get("metadatas", []),
    }


def _readd(client, data: dict) -> int:
    client.delete_collection(COLLECTION)
    col = client.create_collection(
        COLLECTION,
        embedding_function=get_embedding_fn(),
        metadata={"embed_tag": embedding_tag()},
    )
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    n = len(ids)
    for i in range(0, n, BATCH):
        sl = slice(i, i + BATCH)
        col.add(
            ids=ids[sl],
            documents=docs[sl],
            metadatas=[m or {} for m in metas[sl]],
        )
        print(f"  re-added {min(i + BATCH, n)}/{n}")
    return col.count()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", metavar="BACKUP_JSON",
                    help="recreate the collection from a prior backup file")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=DB_PATH)

    if args.restore:
        with open(args.restore) as f:
            data = json.load(f)
        print(f"Restoring {len(data['ids'])} docs from {args.restore} "
              f"under {embedding_tag()} …")
        total = _readd(client, data)
        print(f"Restore complete — collection now has {total} docs.")
        return 0

    try:
        col = client.get_collection(COLLECTION)
    except Exception as exc:
        print(f"No existing '{COLLECTION}' collection ({exc}); "
              f"nothing to migrate. New docs will use the new model.")
        return 0

    data = _dump(col)
    n = len(data["ids"])
    backup = os.path.join(
        os.path.dirname(DB_PATH), f"kb_backup_{int(time.time())}.json"
    )
    with open(backup, "w") as f:
        json.dump(data, f)
    print(f"Backed up {n} docs → {backup}")

    if n == 0:
        print("Collection is empty — recreating under new model, nothing to re-embed.")
    print(f"Re-embedding {n} docs under {embedding_tag()} …")
    total = _readd(client, data)

    if total != n:
        print(f"WARNING: count mismatch (had {n}, now {total}). "
              f"Backup retained at {backup} — restore with "
              f"--restore {backup} if needed.")
        return 1

    # Spot-check: source breakdown survived
    after = client.get_collection(COLLECTION).get(include=["metadatas"])
    srcs: dict[str, int] = {}
    for m in after.get("metadatas", []):
        s = (m or {}).get("source", "?")
        srcs[s] = srcs.get(s, 0) + 1
    print(f"Done. {total} docs re-embedded. By source: {srcs}")
    print(f"Backup kept at {backup} (safe to delete once verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
