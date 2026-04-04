"""
config.py — FailurePolicy, ShieldConfig, KeywordHeuristicStrategy
"""
from __future__ import annotations
import os
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

# Import shared patterns (single source of truth)
_scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from shared_patterns import INJECTION_PATTERNS as _INJECTION_PATTERNS
from shared_patterns import SUSPICIOUS_PATTERNS as _SUSPICIOUS_PATTERNS

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
    ml_model_path: str = "models/tinybert_poison_classifier_v2"
    extra: dict[str, Any] = field(default_factory=dict)

class KeywordHeuristicStrategy:
    """
    Heuristic strategy used by MemShield and the sidecar.
    Mirrors PRISM Layer 1 patterns for consistency.
    """
    def validate(self, text: str) -> dict:
        for pat in _INJECTION_PATTERNS:
            if pat.search(text):
                return {
                    "verdict": "BLOCK",
                    "confidence": 0.97,
                    "reason": f"Injection pattern: {pat.pattern[:60]}",
                    "pattern": pat.pattern,
                }
        for pat in _SUSPICIOUS_PATTERNS:
            if pat.search(text):
                return {
                    "verdict": "QUARANTINE",
                    "confidence": 0.72,
                    "reason": f"Suspicious pattern: {pat.pattern[:60]}",
                    "pattern": pat.pattern,
                }
        return {
            "verdict": "ALLOW",
            "confidence": 0.95,
            "reason": "No injection patterns detected",
            "pattern": None,
        }
