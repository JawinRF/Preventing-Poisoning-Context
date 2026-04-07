# scripts/prism_shield/layer2_local_llm.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import ValidationResult
import os

try:
    from unicode_defense import normalize_unicode
except ModuleNotFoundError:  # pragma: no cover
    from memshield_unicode_defense import normalize_unicode  # type: ignore[import]  # noqa: F401

CLASSIFY_THRESHOLD = 0.45  # bias toward higher recall on poisoning

class LocalLLMValidator:
    def __init__(self, model_path: str = "models/tinybert_poison_classifier_v2"):
         base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
         
         _FP32_PATH = os.path.join(base_dir, model_path)
         _INT8_PATH = os.path.join(base_dir, "models/tinybert_poison_classifier_v2_int8", "model_int8_scripted.pt")
         _INT8_TOKENIZER_PATH = os.path.join(base_dir, "models/tinybert_poison_classifier_v2_int8")

         # Always need a tokenizer. Try to load from INT8 dir if it exists, otherwise FP32 dir.
         if os.path.exists(_INT8_TOKENIZER_PATH):
              self.tokenizer = AutoTokenizer.from_pretrained(_INT8_TOKENIZER_PATH)
         elif os.path.exists(_FP32_PATH):
              self.tokenizer = AutoTokenizer.from_pretrained(_FP32_PATH)
         else:
              raise ValueError(f"Model path does not exist: {_FP32_PATH}")

         if os.path.exists(_INT8_PATH):
              # TorchScript INT8 — fastest path, no Python class overhead
              self.model = torch.jit.load(_INT8_PATH, map_location="cpu")
              self._is_scripted = True
              self.device = "cpu"
              print("[Layer3] Loaded INT8 TorchScript model")
         elif os.path.exists(_FP32_PATH):
              # Fallback to FP32 HuggingFace model if INT8 not yet built
              self.device = "cuda" if torch.cuda.is_available() else "cpu"
              self.model = AutoModelForSequenceClassification.from_pretrained(_FP32_PATH)
              self.model.to(self.device)
              self._is_scripted = False
              print("[Layer3] WARNING: INT8 model not found, falling back to FP32")
         else:
              raise ValueError(f"Model path does not exist: {_FP32_PATH}")
              
         self.model.eval()

    def evaluate(self, normalized_text: str, ingestion_path: str | None = None) -> ValidationResult:
        cleaned_text = normalize_unicode(normalized_text)
        
        inputs = self.tokenizer(
            cleaned_text, return_tensors="pt",
            truncation=True, max_length=128,
            padding="max_length"     # fixed length = consistent latency, no dynamic padding cost
        ).to(self.device)

        with torch.no_grad():
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

        # UI text needs a higher threshold — benign Android labels like
        # "Export Contacts" or "Forward SMS" push TinyBERT into the 0.3-0.6
        # range. 0.70 blocks genuine injection while letting normal UI through.
        BLOCK_THRESH     = 0.70 if ingestion_path == "ui_accessibility" else 0.85
        ALLOW_THRESH     = 0.15

        if malicious_prob >= BLOCK_THRESH:
            return ValidationResult(
                verdict="BLOCK",
                confidence=malicious_prob,
                reason="Layer 2 Local Model identified prompt injection",
                layer_triggered="Layer2-LocalLLM",
            )
        elif malicious_prob <= ALLOW_THRESH:
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
