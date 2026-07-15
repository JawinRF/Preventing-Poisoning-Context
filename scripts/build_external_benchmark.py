#!/usr/bin/env python3
"""
build_external_benchmark.py — held-out benchmark from datasets NOT used in training.

The in-repo synthetic benchmark (data/prism_synthetic_dataset.json) shares its
benign generators with the training set, so a model can score well on it by
memorising those templates. This builds an independent evaluation set:

  Attacks  : Lakera gandalf (real 'ignore instructions' attempts), deepset test
             split, safe-guard test split — injection-labelled.
  Benign   : AG News (news), SMS ham (real messages), Banking77 (user
             utterances), deepset/safe-guard test-split benign.

Every raw text is decontaminated against ALL training source corpora by
normalized exact match, then wrapped in the same Android container formats the
live pipeline sees. Output: data/prism_external_benchmark.json.

Usage:
    python scripts/build_external_benchmark.py [--target 4000] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_training_set import (
    load_deepset, load_neuralchemy, load_jackhhao, load_safeguard,
    load_chatbot_instructions, load_xstest, load_llmail_inject,
    load_gandalf_train, load_sms_ham_train, load_ag_news_train,
    load_banking77_train, eval_holdout,
    wrap_notification, wrap_clipboard, wrap_ui_xml, wrap_rag_document,
    wrap_file_content, wrap_network_response, wrap_intent,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXT_DIR = DATA_DIR / "external"
OUTPUT = DATA_DIR / "prism_external_benchmark.json"

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", str(text).strip().lower())


def _training_text_index() -> set[str]:
    """Normalized raw texts from every training source — the contamination set."""
    loaders = [
        load_deepset, load_neuralchemy, load_jackhhao, load_safeguard,
        load_chatbot_instructions, load_xstest, load_llmail_inject,
        load_gandalf_train, load_sms_ham_train, load_ag_news_train,
        load_banking77_train,
    ]
    seen: set[str] = set()
    for loader in loaders:
        try:
            for text, _label in loader():
                seen.add(_norm(text))
        except Exception as exc:
            print(f"  warn: {loader.__name__} unavailable ({exc})")
    return seen


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _collect_raw() -> tuple[list[str], list[str]]:
    """Return (attack_texts, benign_texts) from held-out sources."""
    attacks: list[str] = []
    benign: list[str] = []

    g = EXT_DIR / "eval_gandalf"
    for f in ["eval_gandalf_validation.parquet", "eval_gandalf_test.parquet"]:
        p = g / f
        if p.exists():
            attacks += _read(p)["text"].astype(str).tolist()

    dp = EXT_DIR / "eval_deepset_test" / "eval_deepset_test.parquet"
    if dp.exists():
        df = _read(dp)
        attacks += df[df["label"] == 1]["text"].astype(str).tolist()
        benign += df[df["label"] == 0]["text"].astype(str).tolist()

    sg = EXT_DIR / "eval_safeguard_test" / "eval_safeguard_test.parquet"
    if sg.exists():
        df = _read(sg)
        attacks += df[df["label"] == 1]["text"].astype(str).tolist()
        benign += df[df["label"] == 0]["text"].astype(str).tolist()

    ag = EXT_DIR / "eval_ag_news" / "eval_ag_news.parquet"
    if ag.exists():
        benign += _read(ag)["text"].astype(str).tolist()

    sms = EXT_DIR / "eval_sms_ham" / "eval_sms_ham.parquet"
    if sms.exists():
        df = _read(sms)
        ham = df[df["label"] == 0]["sms"].astype(str)
        benign += [t for t in ham if eval_holdout(t)]

    bank = EXT_DIR / "eval_banking77" / "eval_banking77.parquet"
    if bank.exists():
        benign += _read(bank)["text"].astype(str).tolist()

    return attacks, benign


# Benign short/imperative text belongs on the same paths a real device would
# surface it. Attacks are spread across every untrusted path.
_ATTACK_WRAPPERS = [
    wrap_notification, wrap_clipboard, wrap_ui_xml, wrap_rag_document,
    wrap_file_content, wrap_network_response, wrap_intent,
]
_BENIGN_WRAPPERS = [
    wrap_notification, wrap_clipboard, wrap_ui_xml, wrap_rag_document,
    wrap_file_content, wrap_network_response, wrap_intent,
]


def _clean(texts: list[str], contamination: set[str]) -> list[str]:
    out, seen = [], set()
    dropped_contam = dropped_dup = 0
    for t in texts:
        t = t.strip()
        if len(t) < 8:
            continue
        n = _norm(t)
        if n in contamination:
            dropped_contam += 1
            continue
        if n in seen:
            dropped_dup += 1
            continue
        seen.add(n)
        out.append(t)
    print(f"    kept {len(out)}  (dropped {dropped_contam} contaminated, {dropped_dup} dup)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)

    print("Indexing training corpora for decontamination ...")
    contamination = _training_text_index()
    print(f"  {len(contamination)} unique training texts indexed")

    print("Collecting held-out eval sources ...")
    raw_attacks, raw_benign = _collect_raw()
    print(f"  raw: {len(raw_attacks)} attack, {len(raw_benign)} benign")

    print("Decontaminating attacks ...")
    attacks = _clean(raw_attacks, contamination)
    print("Decontaminating benign ...")
    benign = _clean(raw_benign, contamination)

    half = args.target // 2
    random.shuffle(attacks)
    random.shuffle(benign)
    attacks = attacks[:half]
    benign = benign[:half]

    samples = []
    for text in attacks:
        wrapped, path = random.choice(_ATTACK_WRAPPERS)(text)
        samples.append({
            "id": f"ext_{uuid.uuid4().hex[:10]}",
            "ingestion_path": path,
            "label": "poisoned",
            "text": wrapped,
        })
    for text in benign:
        wrapped, path = random.choice(_BENIGN_WRAPPERS)(text)
        samples.append({
            "id": f"ext_{uuid.uuid4().hex[:10]}",
            "ingestion_path": path,
            "label": "benign",
            "text": wrapped,
        })

    random.shuffle(samples)
    OUTPUT.write_text(json.dumps(samples, indent=2))

    from collections import Counter
    print(f"\nWrote {len(samples)} samples → {OUTPUT}")
    print("By label:", dict(Counter(s["label"] for s in samples)))
    print("By path :", dict(Counter(s["ingestion_path"] for s in samples)))


if __name__ == "__main__":
    main()
