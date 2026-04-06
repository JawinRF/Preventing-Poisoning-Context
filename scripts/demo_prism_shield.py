"""
Simple demo for the PRISM Shield ingestion pipeline.

This script:
  - Loads a small subset of the PRISM synthetic dataset
  - Runs the PrismShield pipeline on examples from each ingestion path
  - Prints the final verdicts (ALLOW/BLOCK) and reasons

Intended to illustrate how the pipeline protects a mobile agent from
poisoned context across Android-specific channels.
"""

import json
import os
from collections import defaultdict

from prism_shield import MemoryEntry, PrismShield


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "prism_synthetic_dataset.json",
)


def load_examples(max_per_path: int = 2):
    """Load up to `max_per_path` benign + poisoned examples per ingestion path."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Track up to N benign and N poisoned per path
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"benign": [], "poisoned": []}
    )

    for item in data:
        path = item["ingestion_path"]
        label = item["label"]
        bucket = buckets[path][label]
        if len(bucket) < max_per_path:
            bucket.append(item)

    return buckets


def run_demo():
    print("Initializing PrismShield pipeline...")
    try:
        pipeline = PrismShield()
    except Exception as e:
        print("\nERROR: Failed to initialize PrismShield.")
        print(f"Reason: {e}")
        print(
            "\nIf this mentions a missing TinyBERT model directory, "
            "train the local classifier first:\n"
            "  python scripts/train_tinybert.py\n"
            "and then re-run:\n"
            "  python scripts/demo_prism_shield.py\n"
        )
        return

    print(f"Loading examples from dataset at: {DATA_PATH}")
    buckets = load_examples()

    print("\n=== PRISM Shield Demo: Per-Path Examples ===\n")
    for path, labels in sorted(buckets.items()):
        print(f"--- Ingestion path: {path} ---")
        for label, items in labels.items():
            for item in items:
                entry = MemoryEntry(
                    id=item["id"],
                    text=item["text"],
                    ingestion_path=path,
                    metadata={
                        "attack_type": item.get("attack_type"),
                        "severity": item.get("severity"),
                        "target_action": item.get("target_action"),
                        "notes": item.get("notes"),
                    },
                )
                result = pipeline.evaluate_entry(entry)
                snippet = item["text"].replace("\n", " ")[:140]
                print(f"[{label.upper():8}] id={item['id']}")
                print(f"  text: {snippet}...")
                print(
                    f"  -> verdict={result.verdict} "
                    f"(conf={result.confidence:.2f}, layer={result.layer_triggered})"
                )
                if result.normalized_text is not None:
                    norm_snippet = result.normalized_text.replace("\n", " ")[:140]
                    print(f"  normalized: {norm_snippet}...")
                if item.get("attack_type"):
                    print(f"  attack_type={item['attack_type']}, target={item.get('target_action')}")
                print()
        print()


if __name__ == "__main__":
    run_demo()
