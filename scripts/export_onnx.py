"""
export_onnx.py - PyTorch -> ONNX for Android (onnxruntime-android)
"""
import argparse, os, shutil, tempfile
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic
from optimum.onnxruntime import ORTModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/tinybert_poison_classifier_v3")
parser.add_argument("--output",    default="android/openclaw-prism/app/src/main/assets/tinybert_prism.onnx")
parser.add_argument(
    "--quantize-dynamic",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply dynamic INT8 quantization to the exported ONNX model (default: enabled).",
)
args = parser.parse_args()

output_path = Path(args.output)
fp32_output = output_path.with_suffix(".fp32.onnx") if args.quantize_dynamic else output_path
os.makedirs(os.path.dirname(args.output), exist_ok=True)

print("[1/3] Exporting", args.model_dir, "to ONNX via optimum ...")
with tempfile.TemporaryDirectory() as tmp:
    ort_model = ORTModelForSequenceClassification.from_pretrained(args.model_dir, export=True)
    ort_model.save_pretrained(tmp)
    shutil.move(os.path.join(tmp, "model.onnx"), str(fp32_output))

print("[2/3] Checking model ...")
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
