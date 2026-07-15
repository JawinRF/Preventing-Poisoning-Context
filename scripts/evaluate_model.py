import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism_shield.base import MemoryEntry
from prism_shield.content_extractor import ContentExtractor
from prism_shield.normalizer import Normalizer

MODEL_PATH = "models/tinybert_poison_classifier_v3"
DATA_PATH = "data/prism_training_dataset.json"

import json
with open(DATA_PATH, encoding="utf-8") as _f:
    _data = json.load(_f)
import pandas as pd
df = pd.DataFrame(_data)
df["label"] = df["label"].map({"benign": 0, "poisoned": 1})

_extractor  = ContentExtractor()
_normalizer = Normalizer()


def _preprocess(row):
    path = row.get("ingestion_path", "")
    text = _extractor.extract(row["text"], path) or row["text"]
    return _normalizer.normalize(MemoryEntry(id="", text=text, ingestion_path=path))


df["text"] = df.apply(_preprocess, axis=1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

texts = df["text"].tolist()
labels = df["label"].tolist()

batch_size = 64
preds = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]

    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
    preds.extend(batch_preds)

accuracy = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

print("\nEvaluation Metrics")
print("------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

# Attack Success Rate
poison_indices = [i for i,l in enumerate(labels) if l == 1]

poison_total = len(poison_indices)
poison_missed = sum(1 for i in poison_indices if preds[i] == 0)

asr = poison_missed / poison_total

print("\nAttack Success Rate (ASR)")
print("-------------------------")
print("Poison samples:", poison_total)
print("Missed poisons:", poison_missed)
print("ASR:", asr)

print("\nPer-ingestion-path breakdown")
print("----------------------------")
df["pred"] = preds
for path, g in df.groupby("ingestion_path"):
    benign = g[g["label"] == 0]
    poison = g[g["label"] == 1]
    fpr = (benign["pred"] == 1).mean() if len(benign) else float("nan")
    fnr = (poison["pred"] == 0).mean() if len(poison) else float("nan")
    print(f"{path:20s}  n={len(g):5d}  FPR={fpr:6.1%}  FNR={fnr:6.1%}")
