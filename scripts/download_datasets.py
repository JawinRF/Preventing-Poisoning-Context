#!/usr/bin/env python3
"""
Download external prompt-injection and benign datasets from HuggingFace.
Saves each to data/external/<name>/ as parquet for fast loading.

Usage:
    python scripts/download_datasets.py            # download all
    python scripts/download_datasets.py --only deepset neuralchemy   # selective
    python scripts/download_datasets.py --list      # show available datasets
"""

import argparse
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run:")
    print("  pip install datasets")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external"

# ── Registry ────────────────────────────────────────────────────────────────
# Each entry: (short_name, hf_id, split_or_None, description)
# split=None means download all splits

DATASETS = [
    # Attack datasets
    (
        "deepset",
        "deepset/prompt-injections",
        "train",
        "~600 samples, clean labels, widely-used baseline",
    ),
    (
        "neuralchemy",
        "neuralchemy/Prompt-injection-dataset",
        "train",
        "~16.9K samples, leakage-verified binary classification",
    ),
    (
        "jackhhao",
        "jackhhao/jailbreak-classification",
        "train",
        "Jailbreak-specific attack/benign classification",
    ),
    (
        "llmail_inject",
        "microsoft/llmail-inject-challenge",
        "Phase1",
        "462K real adversarial email injections from 224 human teams (MIT)",
    ),
    (
        "safeguard",
        "xTRam1/safe-guard-prompt-injection",
        "train",
        "Curated multi-source merge",
    ),
    # Benign calibration
    (
        "chatbot_instructions",
        "alespalla/chatbot_instruction_prompts",
        "train",
        "Normal instruction prompts (benign calibration)",
    ),
    (
        "xstest",
        "natolambert/xstest-v2-copy",
        None,
        "Over-refusal edge cases (hard negatives) — multi-split",
    ),
]


def download_one(name: str, hf_id: str, split: str | None, desc: str) -> bool:
    dest = OUTPUT_DIR / name
    if dest.exists() and any(dest.iterdir()):
        print(f"  SKIP  {name:25s} — already exists at {dest}")
        return True

    print(f"  GET   {name:25s} ← {hf_id} ({desc})")
    try:
        if split:
            ds = load_dataset(hf_id, split=split)
        else:
            ds = load_dataset(hf_id)

        dest.mkdir(parents=True, exist_ok=True)

        # Save as parquet (fast, compact, pandas-friendly)
        if hasattr(ds, "to_parquet"):
            # Single split → single file
            out_path = dest / f"{name}.parquet"
            ds.to_parquet(str(out_path))
            print(f"  OK    {name:25s} → {out_path}  ({len(ds)} rows)")
        else:
            # DatasetDict (multiple splits)
            for split_name, split_ds in ds.items():
                out_path = dest / f"{name}_{split_name}.parquet"
                split_ds.to_parquet(str(out_path))
                print(f"  OK    {name}/{split_name:15s} → {out_path}  ({len(split_ds)} rows)")

        return True

    except Exception as exc:
        print(f"  FAIL  {name:25s} — {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download external datasets for PRISM training")
    parser.add_argument("--only", nargs="+", help="Download only these datasets (by short name)")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':25s} {'HuggingFace ID':50s} Description")
        print("-" * 110)
        for name, hf_id, _, desc in DATASETS:
            print(f"{name:25s} {hf_id:50s} {desc}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    targets = DATASETS
    if args.only:
        valid = {d[0] for d in DATASETS}
        bad = [n for n in args.only if n not in valid]
        if bad:
            print(f"Unknown dataset names: {bad}")
            print(f"Available: {sorted(valid)}")
            sys.exit(1)
        targets = [d for d in DATASETS if d[0] in args.only]

    ok, fail = 0, 0
    for name, hf_id, split, desc in targets:
        if download_one(name, hf_id, split, desc):
            ok += 1
        else:
            fail += 1

    print(f"\nDone. {ok} succeeded, {fail} failed.")
    if fail:
        print("Re-run to retry failed downloads (existing ones will be skipped).")


if __name__ == "__main__":
    main()
