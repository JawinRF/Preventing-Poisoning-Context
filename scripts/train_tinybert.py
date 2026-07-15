# scripts/train_tinybert.py
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from unicode_defense import normalize_unicode, confusable_augment
except ModuleNotFoundError:  # pragma: no cover
    from memshield_unicode_defense import normalize_unicode, confusable_augment  # type: ignore[import]  # noqa: F401

from prism_shield.base import MemoryEntry
from prism_shield.content_extractor import ContentExtractor
from prism_shield.normalizer import Normalizer

MODEL_NAME = "models/tinybert_poison_classifier"
DATA_PATH = "data/prism_training_dataset.json"
OUTPUT_DIR = "models/tinybert_poison_classifier_v3"

# load json
import json
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Match the inference pipeline exactly: ContentExtractor, then the full
# Normalizer (URL decode, base64 expansion, invisible-char/ANSI strip,
# whitespace compression, Unicode hardening).
_extractor  = ContentExtractor()
_normalizer = Normalizer()


def _preprocess(row):
    path = row.get("ingestion_path", "")
    text = _extractor.extract(row["text"], path) or row["text"]
    return _normalizer.normalize(MemoryEntry(id="", text=text, ingestion_path=path))


df["text"] = df.apply(_preprocess, axis=1)

# simple sanity: map label strings to integers 0/1
df["label"] = df["label"].map({"benign": 0, "poisoned": 1})

# Optional: adversarial augmentation for poisoned samples using confusables.
# Oversample weak ingestion paths identified via red-team results.
WEAK_CATEGORIES = {
    "ui_accessibility",
    "notifications",
    "network_responses",
    "android_intents",
}
DEFAULT_AUG_COPIES = 1
WEAK_AUG_COPIES = 2

augmented_rows = []
for _, row in df[df["label"] == 1].iterrows():
    text = row["text"]
    category = row.get("ingestion_path", "")
    is_weak = str(category) in WEAK_CATEGORIES
    n_copies = WEAK_AUG_COPIES if is_weak else DEFAULT_AUG_COPIES

    for _ in range(n_copies):
        adv_text = confusable_augment(text, substitution_rate=0.4)
        new_row = row.copy()
        new_row["text"] = adv_text
        augmented_rows.append(new_row)

if augmented_rows:
    df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128
    )

# tokenization (keep label column)
train_dataset = train_dataset.map(tokenize_batch, batched=True)
val_dataset = val_dataset.map(tokenize_batch, batched=True)

# remove raw text if you like (keeps label)
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in ["input_ids","attention_mask","label"]])
val_dataset = val_dataset.remove_columns([c for c in val_dataset.column_names if c not in ["input_ids","attention_mask","label"]])

# set format for Trainer
train_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label"])
val_dataset.set_format(type="torch", columns=["input_ids","attention_mask","label"])

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(p):
    preds = p.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.argmax(axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saved to", OUTPUT_DIR)
