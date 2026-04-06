"""
Run the 8 prism_plus_memshield false-negative samples through Llama Prompt Guard 2 22M.
Same logic as predict_prompt.py: tokenize, forward pass, get verdict and confidence.
Findings only; no integration.

Requires: pip install transformers torch
Gated model: log in and accept license first:
  huggingface-cli login
  Then open https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M and accept the terms.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "fixtures" / "benchmark_samples.jsonl"
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-22M"

FN_ENTRY_IDS = {
    "obvious_notifications_02",
    "borderline_network_02",
    "borderline_ui_01",
    "borderline_ui_02",
    "borderline_notifications_01",
    "borderline_notifications_02",
    "borderline_network_03",
    "borderline_ui_03",
}


def load_fn_samples() -> list[dict]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    fn = [r for r in rows if r["entry_id"] in FN_ENTRY_IDS]
    fn.sort(key=lambda r: r["entry_id"])
    return fn


def main() -> None:
    samples = load_fn_samples()
    if len(samples) != 8:
        raise RuntimeError(f"Expected 8 FN samples, got {len(samples)}")

    print(f"Loading {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    # id2label: 0 = SAFE, 1 = MALICIOUS (per Meta docs)
    id2label = getattr(model.config, "id2label", {0: "SAFE", 1: "MALICIOUS"})

    print("\n--- Llama Prompt Guard 2 22M on 8 prism_plus_memshield false negatives ---\n")
    print("entry_id                     | ingestion_path    | label     | verdict   | confidence | malicious_score")
    print("-" * 95)

    for sample in samples:
        text = sample["text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=1)[0]
        pred_id = logits.argmax(dim=1).item()
        verdict = id2label.get(pred_id, f"class_{pred_id}")
        confidence = probs[pred_id].item()
        malicious_score = probs[1].item()  # P(MALICIOUS)

        print(
            f"{sample['entry_id']:28} | {sample['ingestion_path']:17} | {sample['label']:9} | {verdict:9} | {confidence:.4f}     | {malicious_score:.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "GatedRepoError" in type(e).__name__:
            print(
                "\nLlama Prompt Guard 2 22M is a gated model. Log in and accept the license:\n"
                "  huggingface-cli login\n"
                "  Then visit https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-22M and accept.\n",
                file=sys.stderr,
            )
        raise
