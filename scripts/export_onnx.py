"""
export_onnx.py - PyTorch -> ONNX for Android (onnxruntime-android)
"""
import argparse, os
import torch
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/tinybert_poison_classifier_v2")
parser.add_argument("--output",    default="android/openclaw-prism/app/src/main/assets/tinybert_prism.onnx")
parser.add_argument("--seq_len",   type=int, default=128)
args = parser.parse_args()

print("[1/2] Loading model from", args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()

print("[2/2] Exporting to ONNX ...")
os.makedirs(os.path.dirname(args.output), exist_ok=True)

dummy_ids  = torch.zeros(1, args.seq_len, dtype=torch.long)
dummy_mask = torch.ones(1,  args.seq_len, dtype=torch.long)
dummy_type = torch.zeros(1, args.seq_len, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_ids, dummy_mask, dummy_type),
    args.output,
    input_names  = ["input_ids", "attention_mask", "token_type_ids"],
    output_names = ["logits"],
    opset_version = 18,
)

import onnx
onnx.checker.check_model(onnx.load(args.output))
size_kb = os.path.getsize(args.output) / 1024
print("\nDone. Model size: %.1f KB" % size_kb)
print("Output ->", args.output)
