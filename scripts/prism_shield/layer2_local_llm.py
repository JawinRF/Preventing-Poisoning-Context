# scripts/prism_shield/layer2_local_llm.py

import os
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from .base import ValidationResult

try:
    from unicode_defense import normalize_unicode
except ModuleNotFoundError:  # pragma: no cover
    from memshield_unicode_defense import normalize_unicode  # type: ignore[import]  # noqa: F401

DEFAULT_BLOCK_THRESH = float(os.getenv("PRISM_L2_BLOCK_THRESHOLD", "0.85"))
DEFAULT_ALLOW_THRESH = float(os.getenv("PRISM_L2_ALLOW_THRESHOLD", "0.35"))

# Minimum number of letter-words (tokens with >= 2 alphabetic chars) before TinyBERT scores.
# Single-token items like "password123!", URLs, and two-word placeholders like "Item A"
# lack enough word context for reliable classification — TinyBERT over-generalizes on them.
# Threshold is 3 (lower than DeBERTa's 5) since TinyBERT was specifically fine-tuned on
# short injection phrases; only skip for extremely short content.
_MIN_LETTER_WORDS_L2 = 3


def _l2_letter_word_count(text: str) -> int:
    """Count alpha-runs for TinyBERT minimum context guard.

    Uses consecutive alphabetic-run counting (same approach as DeBERTa's guard)
    so that code-injection tokens like 'agent.execute()' are counted correctly
    as multiple alpha-word units rather than a single space-split token.
    Single-token URLs are always counted as 0 (skip TinyBERT for bare URLs).
    """
    stripped = text.strip()
    if stripped.startswith("http") and " " not in stripped:
        return 0
    count = 0
    run_len = 0
    for c in stripped:
        if c.isalpha():
            run_len += 1
        else:
            if run_len >= 2:
                count += 1
            run_len = 0
    if run_len >= 2:
        count += 1
    return count

class LocalLLMValidator:
    def __init__(self, model_path: str = "models/tinybert_poison_classifier_v3"):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        tokenizer_path = os.path.join(base_dir, model_path)
        onnx_path = os.path.join(
            base_dir,
            "android/openclaw-prism/app/src/main/assets",
            "tinybert_prism.onnx",
        )

        if not os.path.exists(tokenizer_path):
            raise RuntimeError(f"TinyBERT model directory missing: {tokenizer_path}")
        if not os.path.exists(onnx_path):
            raise RuntimeError(
                f"Shared ONNX artifact missing: {onnx_path}. Run scripts/export_onnx.py"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.device = "cpu"
        print("[Layer2] Loaded shared ONNX Runtime model")

    def evaluate(self, normalized_text: str, ingestion_path: str | None = None) -> ValidationResult:
        cleaned_text = normalize_unicode(normalized_text)

        # Very short content (< 3 letter-words) causes systematic false positives:
        # single tokens like "password123!" and URLs have no sentence context and
        # TinyBERT over-generalizes on them. Skip scoring for those — both models
        # need enough word context to be reliable.
        if _l2_letter_word_count(cleaned_text) < _MIN_LETTER_WORDS_L2:
            return ValidationResult(
                verdict="ALLOW",
                confidence=0.99,
                reason="Layer 2 TinyBERT skipped: insufficient word context",
                layer_triggered="Layer2-LocalLLM",
            )

        logits = self._onnx_logits(cleaned_text)
        probs = torch.softmax(logits, dim=-1)

        # Make sure to handle batch size dimension if present
        if probs.dim() == 2:
             benign_prob = probs[0][0].item()
             malicious_prob = probs[0][1].item()
        else:
            benign_prob = probs[0].item()
            malicious_prob = probs[1].item()

        block_thresh = DEFAULT_BLOCK_THRESH
        allow_thresh = DEFAULT_ALLOW_THRESH

        if malicious_prob >= block_thresh:
            return ValidationResult(
                verdict="BLOCK",
                confidence=malicious_prob,
                reason="Layer 2 Local Model identified prompt injection",
                layer_triggered="Layer2-LocalLLM",
            )
        elif malicious_prob <= allow_thresh:
            return ValidationResult(
                verdict="ALLOW",
                confidence=benign_prob,
                reason="Entry deemed benign",
                layer_triggered="Layer2-LocalLLM",
            )
        else:
            return ValidationResult(
                verdict="QUARANTINE",
                confidence=malicious_prob,
                reason="Layer 2 Local Model detected anomalous context but confidence is borderline.",
                layer_triggered="Layer2-LocalLLM",
            )

    def _onnx_logits(self, cleaned_text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="np",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded.get(
                "token_type_ids",
                np.zeros_like(encoded["input_ids"], dtype=np.int64),
            ).astype(np.int64),
        }
        logits = self.model.run(None, inputs)[0]
        return torch.from_numpy(logits)
