"""
Run the 8 prism_plus_memshield false-negative samples through
ProtectAI/deberta-v3-base-prompt-injection-v2. Findings only; no pipeline integration.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "fixtures" / "benchmark_samples.jsonl"

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

    tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
    model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=torch.device("cpu"),
    )

    print("\n--- ProtectAI/deberta-v3-base-prompt-injection-v2 on 8 prism_plus_memshield false negatives ---\n")
    print("entry_id                     | ingestion_path    | label     | verdict    | confidence")
    print("-" * 88)

    caught = 0
    for sample in samples:
        out = classifier(sample["text"])[0]
        verdict = out["label"]
        confidence = out["score"]
        if verdict.upper() == "INJECTION":
            caught += 1
        print(
            f"{sample['entry_id']:28} | {sample['ingestion_path']:17} | {sample['label']:9} | {verdict:10} | {confidence:.4f}"
        )

    print("-" * 88)
    print(f"\nCaught (verdict == INJECTION): {caught} of 8\n")


if __name__ == "__main__":
    main()
