# scripts/prism_shield/pipeline.py
"""
PrismShield pipeline — text-only defense layers.

Evaluate flow:
  UIExtractor (ui_accessibility only) → Normalizer → Layer 1 Heuristics →
  Layer 2 TinyBERT → Layer 3 DeBERTa → verdict (ALLOW / BLOCK)

QUARANTINE resolution is path-dependent:
  - Incoming untrusted text (clipboard, notifications, etc.): QUARANTINE → BLOCK
  - Agent output (agent_output): QUARANTINE → ALLOW (only high-confidence BLOCK stops it)

VLM-based quarantine resolution was removed — the current architecture uses
deterministic OS-level UI integrity checks (Android sidecar :8766) for tap
safety, not synchronous VLM inference in the request path.
"""
from __future__ import annotations

from .base import MemoryEntry, ValidationResult
from .normalizer import Normalizer
from .layer1_heuristics import HeuristicsEngine
from .layer2_local_llm import LocalLLMValidator
from .layer3_deberta import DeBERTaValidator
from .ui_extractor import UIExtractor
from .window_context_reader import start_reader


class PrismShield:
    """
    Main entrypoint for evaluating incoming agent context.
    UIExtractor (ui_accessibility only) -> Normalization -> Layer 1 -> Layer 2 -> Layer 3
    """
    def __init__(self):
        start_reader()
        self.ui_extractor = UIExtractor()
        self.normalizer = Normalizer()
        self.heuristics = HeuristicsEngine()
        self.local_llm = LocalLLMValidator()
        self.deberta = DeBERTaValidator()

    # Paths where QUARANTINE resolves to ALLOW (agent's own output — benefit of the doubt)
    _LENIENT_PATHS = frozenset({"agent_output"})

    def _resolve_quarantine(self, result: ValidationResult, layer: str,
                            ingestion_path: str) -> None:
        """Resolve QUARANTINE in-place based on ingestion path."""
        if result.verdict != "QUARANTINE":
            return
        if ingestion_path in self._LENIENT_PATHS:
            result.verdict = "ALLOW"
            result.reason = f"[{layer} quarantine→allow agent_output] {result.reason}"
        else:
            result.verdict = "BLOCK"
            result.reason = f"[{layer} quarantine→block] {result.reason}"

    def evaluate_sync(self, raw_entry: MemoryEntry) -> ValidationResult:
        path = raw_entry.ingestion_path

        # Step 0: UIExtractor pre-processing (ui_accessibility path only)
        if path == "ui_accessibility":
            raw_entry.text = self.ui_extractor.extract(raw_entry.text)

        # Step 1: Normalize
        normalized_text = self.normalizer.normalize(raw_entry)

        # Step 2: Layer 1 Fast Path
        l1_result = self.heuristics.evaluate(normalized_text)
        if l1_result is not None:
            l1_result.normalized_text = normalized_text
            return l1_result

        # Step 3: Layer 2 Local Model Path
        l2_result = self.local_llm.evaluate(normalized_text, path)
        l2_result.normalized_text = normalized_text

        # Resolve QUARANTINE from L2
        self._resolve_quarantine(l2_result, "L2", path)
        if l2_result.verdict == "BLOCK":
            return l2_result

        # Step 4: Layer 3 DeBERTa — only when Layer 2 returned ALLOW
        if l2_result.verdict == "ALLOW":
            l3_result = self.deberta.evaluate(normalized_text, path)
            l3_result.normalized_text = normalized_text
            self._resolve_quarantine(l3_result, "L3", path)
            if l3_result.verdict == "BLOCK":
                return l3_result

        return l2_result

    def evaluate_entry(self, raw_entry: MemoryEntry) -> ValidationResult:
        return self.evaluate_sync(raw_entry)
