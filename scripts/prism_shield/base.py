# scripts/prism_shield/base.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryEntry:
    id: str
    text: str
    ingestion_path: str
    metadata: dict = None

@dataclass
class ValidationResult:
    verdict: str  # "ALLOW", "BLOCK", "QUARANTINE"
    confidence: float
    reason: str
    layer_triggered: str  # "Layer1-Heuristics", "Layer2-LocalLLM", "Layer3-CloudLLM"
    normalized_text: Optional[str] = None
    ticket_id: Optional[str] = None


@dataclass
class FinalizedTicket:
    ticket_id: str
    status: str  # "PENDING", "APPROVED", "REJECTED"
    confidence: float
    reason: str
    layer_triggered: str
    created_at: str
