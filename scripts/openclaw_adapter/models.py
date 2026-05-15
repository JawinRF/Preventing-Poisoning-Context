from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InspectRequest(BaseModel):
    entry_id: str
    text: str
    ingestion_path: str
    # Correlation / audit fields — optional when ingestion_path is provided directly.
    # Required when ingestion_path is absent (used as routing fallback via map_ingestion_path).
    source_type: str = "unknown"
    source_name: str = "unknown"
    session_id: str = ""
    run_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class InspectResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    layer_triggered: str = ""
    normalized_text: str = ""
    ticket_id: str | None = None
    placeholder: str | None = None
    audit: dict[str, Any] = Field(default_factory=dict)

