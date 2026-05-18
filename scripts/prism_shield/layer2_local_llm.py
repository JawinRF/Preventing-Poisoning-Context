# scripts/prism_shield/layer2_local_llm.py

import os
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base import ValidationResult

try:
    from unicode_defense import normalize_unicode
except ModuleNotFoundError:  # pragma: no cover
    from memshield_unicode_defense import normalize_unicode  # type: ignore[import]  # noqa: F401

DEFAULT_BLOCK_THRESH = float(os.getenv("PRISM_L2_BLOCK_THRESHOLD", "0.85"))
UI_BLOCK_THRESH = float(os.getenv("PRISM_L2_UI_BLOCK_THRESHOLD", "0.70"))
DEFAULT_ALLOW_THRESH = float(os.getenv("PRISM_L2_ALLOW_THRESHOLD", "0.35"))
UI_ALLOW_THRESH = float(os.getenv("PRISM_L2_UI_ALLOW_THRESHOLD", "0.40"))

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

         _FP32_PATH = os.path.join(base_dir, model_path)
         _ONNX_PATH = os.path.join(
             base_dir,
             "android/openclaw-prism/app/src/main/assets",
             "tinybert_prism.onnx",
         )
         _INT8_PATH = os.path.join(base_dir, "models/tinybert_poison_classifier_v3_int8", "model_int8_scripted.pt")
         _INT8_TOKENIZER_PATH = os.path.join(base_dir, "models/tinybert_poison_classifier_v3_int8")

         # Always need a tokenizer. Try to load from INT8 dir if it exists, otherwise FP32 dir.
         if os.path.exists(_INT8_TOKENIZER_PATH):
              self.tokenizer = AutoTokenizer.from_pretrained(_INT8_TOKENIZER_PATH)
         elif os.path.exists(_FP32_PATH):
              self.tokenizer = AutoTokenizer.from_pretrained(_FP32_PATH)
         else:
              raise ValueError(f"Model path does not exist: {_FP32_PATH}")

         if os.path.exists(_ONNX_PATH):
              # Shared ONNX Runtime path keeps host and Android aligned on the
              # same quantized v3 artifact while still using the HF tokenizer.
              self.model = ort.InferenceSession(_ONNX_PATH, providers=["CPUExecutionProvider"])
              self._backend = "onnx"
              self._is_scripted = False
              self.device = "cpu"
              print("[Layer2] Loaded shared ONNX Runtime model")
         elif os.path.exists(_INT8_PATH):
              # TorchScript INT8 — fastest path, no Python class overhead
              self.model = torch.jit.load(_INT8_PATH, map_location="cpu")
              self._backend = "torchscript"
              self._is_scripted = True
              self.device = "cpu"
              print("[Layer2] Loaded INT8 TorchScript model")
         elif os.path.exists(_FP32_PATH):
              # Fallback to FP32 HuggingFace model if INT8 not yet built
              self.device = "cuda" if torch.cuda.is_available() else "cpu"
              self.model = AutoModelForSequenceClassification.from_pretrained(_FP32_PATH)
              self.model.to(self.device)
              self._backend = "hf"
              self._is_scripted = False
              print("[Layer2] WARNING: INT8 model not found, falling back to FP32")
         else:
              raise ValueError(f"Model path does not exist: {_FP32_PATH}")
              
         if self._backend != "onnx":
              self.model.eval()

    def evaluate(self, normalized_text: str, ingestion_path: str | None = None) -> ValidationResult:
        cleaned_text = normalize_unicode(normalized_text)

        # Very short content (< 3 letter-words) causes systematic false positives:
        # single tokens like "password123!" and URLs have no sentence context and
        # TinyBERT over-generalizes on them. Skip scoring for those — both models
        # need enough word context to be reliable. Exclude ui_accessibility because
        # that path re-wraps content into XML before scoring.
        if (
            ingestion_path != "ui_accessibility"
            and _l2_letter_word_count(cleaned_text) < _MIN_LETTER_WORDS_L2
        ):
            return ValidationResult(
                verdict="ALLOW",
                confidence=0.99,
                reason="Layer 2 TinyBERT skipped: insufficient word context",
                layer_triggered="Layer2-LocalLLM",
            )

        # Training wraps every ui_accessibility sample in <hierarchy><node .../></hierarchy>.
        # Bare button labels ("Send", "+") are OOD → garbage scores. Rewrap only for L2.
        if ingestion_path == "ui_accessibility":
            stripped = cleaned_text.strip()
            if stripped and not stripped.startswith("<"):
                safe = stripped.replace('"', "&quot;")
                cleaned_text = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n<hierarchy rotation="0">\n'
                    f'  <node index="0" class="android.widget.Button" text="{safe}" '
                    'content-desc="" resource-id="com.example:id/button1" '
                    'clickable="true" visible-to-user="true" bounds="[0,0][100,100]" />\n'
                    '</hierarchy>'
                )

        with torch.no_grad():
            if self._backend == "onnx":
                logits = self._onnx_logits(cleaned_text)
            else:
                inputs = self.tokenizer(
                    cleaned_text, return_tensors="pt",
                    truncation=True, max_length=128,
                    padding="max_length"     # fixed length = consistent latency, no dynamic padding cost
                ).to(self.device)
                if self._is_scripted:
                    # TorchScript forward — pass tensors directly
                    logits = self.model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs.get("token_type_ids")
                    )
                else:
                    logits = self.model(**inputs).logits

        # Extract logits tensor from whatever object the HF model / Traced model returned
        if hasattr(logits, "logits"):
            logits = getattr(logits, "logits")
        elif isinstance(logits, dict):
            logits = logits.get("logits", list(logits.values())[0])
        elif isinstance(logits, tuple) and len(logits) > 0:
            logits = logits[0]
            
        probs = torch.softmax(logits, dim=-1)

        # Make sure to handle batch size dimension if present
        if probs.dim() == 2:
             benign_prob = probs[0][0].item()
             malicious_prob = probs[0][1].item()
        else:
            benign_prob = probs[0].item()
            malicious_prob = probs[1].item()

        # TinyBERT v3 still places some benign short UI/action strings in the
        # 0.3-0.4 band. Widen the allow band so low-confidence scores like
        # "Hello" do not get escalated into a final BLOCK upstream.
        block_thresh = UI_BLOCK_THRESH if ingestion_path == "ui_accessibility" else DEFAULT_BLOCK_THRESH
        allow_thresh = UI_ALLOW_THRESH if ingestion_path == "ui_accessibility" else DEFAULT_ALLOW_THRESH

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
