# scripts/prism_shield/layer3_deberta.py
"""
Layer 3: DeBERTa-based prompt injection classifier (ProtectAI/deberta-v3-base-prompt-injection-v2).
Invoked only when Layer 2 returns ALLOW. Apache 2.0, no gating.
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .base import ValidationResult


def _letter_word_count(text: str) -> int:
    """Count space-split tokens that contain at least 2 alphabetic characters.

    DeBERTa needs sentence-like context to score reliably. Pure codes, IDs, emails,
    and phone numbers have very few letter-words and are systematically mis-scored.
    Skipping DeBERTa for those avoids a class of calibration false positives while
    letting TinyBERT (which was fine-tuned on short direct injections) cover the gap.

    Uses space-split (not alpha-run counting) because dot-separated package names like
    'com.example.app' are one semantic unit; counting their sub-tokens inflates the
    count and incorrectly triggers DeBERTa on structured Android data.
    """
    return sum(
        1 for token in text.split()
        if sum(1 for c in token if c.isalpha()) >= 2
    )


class DeBERTaValidator:
    """ProtectAI DeBERTa prompt-injection classifier. BLOCK/QUARANTINE on INJECTION by confidence."""

    MODEL_ID = "ProtectAI/deberta-v3-base-prompt-injection-v2"
    # Local pinned copy — avoids HuggingFace Hub dependency at runtime
    LOCAL_MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "models", "deberta_prompt_injection_v2"
    )
    BLOCK_THRESHOLD = 0.90  # INJECTION with confidence >= this -> BLOCK
    # INJECTION with confidence < BLOCK_THRESHOLD -> QUARANTINE

    # Minimum number of letter-words required before DeBERTa scores the text.
    # Codes, IDs, emails, and phone numbers have < 5 letter-words and cause
    # systematic calibration false positives — TinyBERT handles those instead.
    _MIN_LETTER_WORDS = 5

    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer local pinned model; fall back to HuggingFace Hub
        model_path = self.LOCAL_MODEL_PATH if os.path.isdir(self.LOCAL_MODEL_PATH) else self.MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            device=device,
        )

    def evaluate(self, normalized_text: str, ingestion_path: str = "") -> ValidationResult:
        if _letter_word_count(normalized_text) < self._MIN_LETTER_WORDS:
            return ValidationResult(
                verdict="ALLOW",
                confidence=0.01,
                reason="Layer 3 DeBERTa skipped: insufficient natural-language context",
                layer_triggered="Layer3-DeBERTa",
            )

        out = self._classifier(normalized_text)[0]
        verdict_label = out["label"]
        confidence = out["score"]

        is_injection = verdict_label.upper() == "INJECTION"

        if is_injection and confidence >= self.BLOCK_THRESHOLD:
            return ValidationResult(
                verdict="BLOCK",
                confidence=confidence,
                reason="Layer 3 DeBERTa identified prompt injection",
                layer_triggered="Layer3-DeBERTa",
            )
        if is_injection and confidence < self.BLOCK_THRESHOLD:
            return ValidationResult(
                verdict="QUARANTINE",
                confidence=confidence,
                reason="Layer 3 DeBERTa detected possible injection; confidence below block threshold",
                layer_triggered="Layer3-DeBERTa",
            )
        return ValidationResult(
            verdict="ALLOW",
            confidence=1.0 - confidence,
            reason="Layer 3 DeBERTa deemed safe",
            layer_triggered="Layer3-DeBERTa",
        )
