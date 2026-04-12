# scripts/prism_shield/pipeline.py
"""
PrismShield pipeline — text-only defense layers.

Evaluate flow:
  UIExtractor (ui_accessibility only) → Normalizer →
  Layer 1 TinyBERT → Layer 2 DeBERTa → verdict (ALLOW / BLOCK)

QUARANTINE resolution is path-dependent:
  - Incoming untrusted text (clipboard, notifications, etc.): QUARANTINE → BLOCK
  - Agent output (agent_output): QUARANTINE → ALLOW (only high-confidence BLOCK stops it)

VLM-based quarantine resolution was removed — the current architecture uses
deterministic OS-level UI integrity checks (Android sidecar :8766) for tap
safety, not synchronous VLM inference in the request path.

Regex heuristic layer was removed — TinyBERT v3 (trained on 44K real+synthetic
samples) handles the same patterns with context awareness and fewer false positives.
"""
from .base import MemoryEntry, ValidationResult
from .normalizer import Normalizer
from .layer2_local_llm import LocalLLMValidator
from .layer3_deberta import DeBERTaValidator
from .ui_extractor import UIExtractor
from .window_context_reader import start_reader


class PrismShield:
    """
    Main entrypoint for evaluating incoming agent context.
    UIExtractor (ui_accessibility only) -> Normalization -> TinyBERT -> DeBERTa
    """
    def __init__(self):
        start_reader()
        self.ui_extractor = UIExtractor()
        self.normalizer = Normalizer()
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

        # Step 2: Layer 1 TinyBERT
        l1_result = self.local_llm.evaluate(normalized_text, path)
        l1_result.normalized_text = normalized_text

        self._resolve_quarantine(l1_result, "L1", path)
        if l1_result.verdict == "BLOCK":
            return l1_result

        # Step 3: Layer 2 DeBERTa — only when TinyBERT returned ALLOW
        if l1_result.verdict == "ALLOW":
            l2_result = self.deberta.evaluate(normalized_text, path)
            l2_result.normalized_text = normalized_text
            self._resolve_quarantine(l2_result, "L2", path)
            if l2_result.verdict == "BLOCK":
                return l2_result

        return l1_result

    def evaluate_entry(self, raw_entry: MemoryEntry) -> ValidationResult:
        return self.evaluate_sync(raw_entry)
