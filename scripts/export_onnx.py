"""
export_onnx.py - PyTorch -> ONNX for Android (onnxruntime-android)
"""
import argparse, os
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/tinybert_poison_classifier_v3")
parser.add_argument("--output",    default="android/openclaw-prism/app/src/main/assets/tinybert_prism.onnx")
parser.add_argument("--seq_len",   type=int, default=128)
parser.add_argument(
    "--quantize-dynamic",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply dynamic INT8 quantization to the exported ONNX model (default: enabled).",
)
args = parser.parse_args()

print("[1/2] Loading model from", args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model.eval()

output_path = Path(args.output)
fp32_output = output_path.with_suffix(".fp32.onnx") if args.quantize_dynamic else output_path

print("[2/2] Exporting to ONNX ...")
os.makedirs(os.path.dirname(args.output), exist_ok=True)

dummy_ids  = torch.zeros(1, args.seq_len, dtype=torch.long)
dummy_mask = torch.ones(1,  args.seq_len, dtype=torch.long)
dummy_type = torch.zeros(1, args.seq_len, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_ids, dummy_mask, dummy_type),
    str(fp32_output),
    input_names  = ["input_ids", "attention_mask", "token_type_ids"],
    output_names = ["logits"],
    opset_version = 18,
)

onnx.checker.check_model(onnx.load(str(fp32_output)))

if args.quantize_dynamic:
    print("[3/3] Applying dynamic INT8 quantization ...")
    quantize_dynamic(
        model_input=str(fp32_output),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    onnx.checker.check_model(onnx.load(str(output_path)))
    fp32_output.unlink()

size_kb = os.path.getsize(args.output) / 1024
print("\nDone. Model size: %.1f KB" % size_kb)
print("Output ->", args.output)
