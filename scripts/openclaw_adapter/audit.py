import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = PROJECT_ROOT / "data" / "audit_log.jsonl"


def log_audit(
    entry_id: str,
    verdict: str,
    ingestion_path: str,
    source_type: str,
    session_id: str,
    run_id: str,
    **extras: Any,
) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.touch(exist_ok=True)

    record = {
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "entry_id": entry_id,
        "verdict": verdict,
        "ingestion_path": ingestion_path,
        "source_type": source_type,
        "session_id": session_id,
        "run_id": run_id,
    }
    record.update(extras)

    with AUDIT_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")

