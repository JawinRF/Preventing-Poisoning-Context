# scripts/prism_shield/pipeline.py
"""
PrismShield pipeline — text-only defense layers.

Evaluate flow:
  ContentExtractor → Normalizer → [L2 TinyBERT ‖ L3 DeBERTa] → Ensemble → verdict

ContentExtractor (Step 0):
  Detects the container format of the incoming text and extracts the semantic
  payload before any ML model sees it. Prevents distribution-shift false positives
  where classifiers trained on natural language receive Android XML hierarchies,
  file wrappers, intent JSON, or HTML that look syntactically suspicious but are
  semantically benign.

Ensemble (Step 3):
  L2 and L3 run in parallel. Their post-resolved verdicts are combined:
    Both BLOCK  → BLOCK   (high confidence, both models agree)
    Disagree    → QUARANTINE (one model uncertain — route for review)
    Both ALLOW  → ALLOW

L3-advisory paths:
  DeBERTa is a frozen generalist trained on natural-language prompts. On paths
  whose extracted payload is structurally non-prompt-like (UI node label soup,
  RAG store records), its BLOCK verdicts are dominated by false positives while
  TinyBERT is trained in-distribution on exactly those payloads. On paths in
  PRISM_L3_ADVISORY_PATHS (default: ui_accessibility, rag_store), an L3 BLOCK
  opposed by an L2 ALLOW at >= PRISM_L2_OVERRIDE_CONFIDENCE (default 0.99)
  benign probability resolves to ALLOW. L3 keeps full veto power on every
  other path.

QUARANTINE resolution:
  Single-model QUARANTINE (medium-confidence injection from one model) is resolved
  before ensemble: it becomes BLOCK for untrusted paths, ALLOW for agent output.
  Ensemble QUARANTINE (disagreement signal) is NOT resolved to BLOCK — it is the
  correct output when the two models genuinely disagree.
"""
from __future__ import annotations

import concurrent.futures
import os

from .base import MemoryEntry, ValidationResult
from .normalizer import Normalizer
from .layer2_local_llm import LocalLLMValidator
from .layer3_deberta import DeBERTaValidator
from .content_extractor import ContentExtractor
from .window_context_reader import start_reader


class PrismShield:
    """
    Main entrypoint for evaluating incoming agent context.
    ContentExtractor → Normalizer → [TinyBERT ‖ DeBERTa] → Ensemble
    """

    def __init__(self) -> None:
        start_reader()
        self.content_extractor = ContentExtractor()
        self.normalizer        = Normalizer()
        self.local_llm         = LocalLLMValidator()
        self.deberta           = DeBERTaValidator()

    # Paths where a single-model QUARANTINE resolves to ALLOW (agent's own output)
    _LENIENT_PATHS = frozenset({"agent_output"})

    _L3_ADVISORY_PATHS = frozenset(
        p.strip()
        for p in os.getenv(
            "PRISM_L3_ADVISORY_PATHS", "ui_accessibility,rag_store"
        ).split(",")
        if p.strip()
    )
    _L2_OVERRIDE_CONFIDENCE = float(os.getenv("PRISM_L2_OVERRIDE_CONFIDENCE", "0.99"))

    # ── Quarantine resolution ────────────────────────────────────────────────

    def _resolve_quarantine(
        self, result: ValidationResult, layer: str, ingestion_path: str
    ) -> None:
        """Resolve QUARANTINE in-place based on ingestion path.

        Only called on single-model results before ensemble. Ensemble QUARANTINE
        (disagreement) is intentionally left unresolved.
        """
        if result.verdict != "QUARANTINE":
            return
        if ingestion_path in self._LENIENT_PATHS:
            result.verdict = "ALLOW"
            result.reason  = f"[{layer} quarantine→allow agent_output] {result.reason}"
        else:
            result.verdict = "BLOCK"
            result.reason  = f"[{layer} quarantine→block] {result.reason}"

    # ── Core pipeline ────────────────────────────────────────────────────────

    def evaluate_sync(self, raw_entry: MemoryEntry) -> ValidationResult:
        path = raw_entry.ingestion_path

        # ── Step 0: Content extraction ────────────────────────────────────
        # Detect container format; extract semantic payload before ML scoring.
        # Falls back to original text when format is unrecognised or extraction
        # yields empty (no silent data loss).
        extracted = self.content_extractor.extract(raw_entry.text, path)

        # ── Step 1: Normalize ─────────────────────────────────────────────
        norm_entry    = MemoryEntry(id=raw_entry.id, text=extracted,
                                    ingestion_path=path)
        normalized    = self.normalizer.normalize(norm_entry)

        # ── Step 2: Run L2 (TinyBERT) and L3 (DeBERTa) in parallel ───────
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_l2 = pool.submit(self.local_llm.evaluate, normalized, path)
            f_l3 = pool.submit(self.deberta.evaluate,   normalized, path)
            l2 = f_l2.result()
            l3 = f_l3.result()

        l2.normalized_text = normalized
        l3.normalized_text = normalized

        # 3-state shortcut: one model uncertain (QUARANTINE), other highly confident safe.
        # L3 DeBERTa ALLOW confidence = injection probability; near-zero = very confident safe.
        # L2 TinyBERT ALLOW confidence = benign probability; near-one = very confident safe.
        # Threshold 0.10: if the ALLOWING model assigns < 10% injection probability AND the
        # other model is merely uncertain (not strongly blocking), trust the confident ALLOW.
        # This resolves the case where TinyBERT is uncertain on natural-language text (e.g.
        # calendar summaries) while DeBERTa is highly confident the text is safe.
        _CONFIDENT_SAFE = 0.10
        if l2.verdict == "QUARANTINE" and l3.verdict == "ALLOW" and l3.confidence < _CONFIDENT_SAFE:
            return ValidationResult(
                verdict="ALLOW",
                confidence=1.0 - l3.confidence,
                reason=f"[Ensemble: L2 uncertain, L3 confident safe ({l3.confidence:.3f})] {l3.reason}",
                layer_triggered="Ensemble",
                normalized_text=normalized,
            )
        if l3.verdict == "QUARANTINE" and l2.verdict == "ALLOW" and (1.0 - l2.confidence) < _CONFIDENT_SAFE:
            return ValidationResult(
                verdict="ALLOW",
                confidence=l2.confidence,
                reason=f"[Ensemble: L3 uncertain, L2 confident safe] {l2.reason}",
                layer_triggered="Ensemble",
                normalized_text=normalized,
            )

        if (
            l3.verdict == "BLOCK"
            and l2.verdict == "ALLOW"
            and path in self._L3_ADVISORY_PATHS
            and l2.confidence >= self._L2_OVERRIDE_CONFIDENCE
        ):
            return ValidationResult(
                verdict="ALLOW",
                confidence=l2.confidence,
                reason=(
                    f"[Ensemble: L3 advisory on {path} — L2 confident benign "
                    f"({l2.confidence:.3f}) overrides L3 BLOCK ({l3.confidence:.3f})] {l2.reason}"
                ),
                layer_triggered="Ensemble",
                normalized_text=normalized,
            )

        # Resolve single-model QUARENTINEs (BLOCK or ALLOW depending on path)
        self._resolve_quarantine(l2, "L2", path)
        self._resolve_quarantine(l3, "L3", path)

        # ── Step 3: Ensemble agreement ────────────────────────────────────
        l2_blocks = l2.verdict == "BLOCK"
        l3_blocks = l3.verdict == "BLOCK"

        if l2_blocks and l3_blocks:
            # Both agree: return the higher-confidence verdict
            return l2 if l2.confidence >= l3.confidence else l3

        if not l2_blocks and not l3_blocks:
            # Both allow: return L2 result (TinyBERT — primary model)
            return l2

        # Models disagree — route to QUARANTINE for human review.
        # Do NOT call _resolve_quarantine here: disagreement QUARANTINE means
        # "uncertain", not "medium-confidence injection" — blocking would defeat
        # the purpose of running two models.
        if l2_blocks:
            high, low, high_lbl, low_lbl = l2, l3, "L2", "L3"
        else:
            high, low, high_lbl, low_lbl = l3, l2, "L3", "L2"

        return ValidationResult(
            verdict="QUARANTINE",
            confidence=high.confidence,
            reason=(
                f"[Ensemble disagreement] {high_lbl} BLOCK "
                f"({high.confidence:.3f}) / {low_lbl} ALLOW "
                f"({low.confidence:.3f}) — routing to review"
            ),
            layer_triggered="Ensemble",
            normalized_text=normalized,
        )

    def evaluate_entry(self, raw_entry: MemoryEntry) -> ValidationResult:
        return self.evaluate_sync(raw_entry)
