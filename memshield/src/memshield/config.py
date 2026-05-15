"""
config.py — FailurePolicy, ShieldConfig
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class FailurePolicy(Enum):
    BLOCK = "BLOCK"
    ALLOW = "ALLOW"

@dataclass
class ShieldConfig:
    enabled: bool = True
    failure_policy: FailurePolicy = FailurePolicy.BLOCK
    confidence_threshold: float = 0.5
    enable_provenance: bool = False
    enable_normalization: bool = True
    enable_ml_layers: bool = False
    enable_retrieval_defense: bool = False  # Full pipeline: influence + ragmask + authority + scorer
    enable_progrank: bool = False            # ProGRank instability (expensive: N re-retrievals)
    progrank_perturbations: int = 10
    retrieval_block_threshold: float = 0.75
    retrieval_quarantine_threshold: float = 0.50
    influence_gamma: float = 0.5
    ml_model_path: str = "models/tinybert_poison_classifier_v3"
    extra: dict[str, Any] = field(default_factory=dict)
