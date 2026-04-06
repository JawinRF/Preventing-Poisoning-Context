import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "tests" / "fixtures" / "benchmark_samples.jsonl"
RESULTS_PATH = PROJECT_ROOT / "data" / "benchmark_results.jsonl"
PYTHON_BIN = PROJECT_ROOT / "env" / "bin" / "python"
SIDECAR_SCRIPT = PROJECT_ROOT / "scripts" / "openclaw_adapter" / "server.py"
PYTHONPATH_VALUE = str(PROJECT_ROOT / "scripts")


@dataclass
class Metrics:
    config: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


def load_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def is_positive(sample: dict[str, Any]) -> bool:
    return sample["label"] in {"attack", "borderline"}


def predicted_positive(verdict: str) -> bool:
    return verdict in {"BLOCK", "QUARANTINE"}


def score(config: str, rows: list[dict[str, Any]]) -> Metrics:
    tp = fp = fn = 0
    for row in rows:
        truth = is_positive(row)
        pred = predicted_positive(row["verdict"])
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif truth and not pred:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return Metrics(config, tp, fp, fn, precision, recall, f1)


def format_table(metrics: list[Metrics]) -> str:
    headers = ["config", "true_positives", "false_positives", "false_negatives", "precision", "recall", "f1"]
    rows = [
        [
            item.config,
            str(item.true_positives),
            str(item.false_positives),
            str(item.false_negatives),
            f"{item.precision:.3f}",
            f"{item.recall:.3f}",
            f"{item.f1:.3f}",
        ]
        for item in metrics
    ]
    widths = [max(len(headers[i]), max(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "-+-".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in rows:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def run_sidecar(port: int, enable_memshield_rag: bool) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH_VALUE
    env["PRISM_SIDECAR_HOST"] = "127.0.0.1"
    env["PRISM_SIDECAR_PORT"] = str(port)
    env["PRISM_ENABLE_MEMSHIELD_RAG"] = "1" if enable_memshield_rag else "0"

    proc = subprocess.Popen(
        [str(PYTHON_BIN), str(SIDECAR_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=1)
            raise RuntimeError(f"Sidecar exited early on port {port}.\nstdout:\n{stdout}\nstderr:\n{stderr}")
        try:
            response = requests.get(url, timeout=0.5)
            if response.status_code == 200:
                return proc
        except requests.RequestException:
            time.sleep(0.2)

    proc.kill()
    stdout, stderr = proc.communicate(timeout=1)
    raise RuntimeError(f"Sidecar health check timed out on port {port}.\nstdout:\n{stdout}\nstderr:\n{stderr}")


def stop_sidecar(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def inspect_sample(base_url: str, sample: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "entry_id": sample["entry_id"],
        "text": sample["text"],
        "ingestion_path": sample["ingestion_path"],
        "source_type": sample["source_type"],
        "source_name": sample["source_name"],
        "session_id": f"bench-{sample['entry_id']}",
        "run_id": f"bench-{sample['entry_id']}",
        "metadata": sample.get("metadata", {}),
    }
    response = requests.post(f"{base_url}/v1/inspect", json=payload, timeout=10)
    response.raise_for_status()
    data = response.json()
    data["label"] = sample["label"]
    data["ingestion_path"] = sample["ingestion_path"]
    data["entry_id"] = sample["entry_id"]
    return data


def evaluate_baseline(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "entry_id": sample["entry_id"],
            "label": sample["label"],
            "ingestion_path": sample["ingestion_path"],
            "verdict": "ALLOW",
            "reason": "baseline_no_filtering",
        }
        for sample in dataset
    ]


def evaluate_with_sidecar(dataset: list[dict[str, Any]], enable_memshield_rag: bool, port: int) -> list[dict[str, Any]]:
    proc = run_sidecar(port, enable_memshield_rag)
    try:
        base_url = f"http://127.0.0.1:{port}"
        return [inspect_sample(base_url, sample) for sample in dataset]
    finally:
        stop_sidecar(proc)


def append_results(metrics: list[Metrics]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        for item in metrics:
            fh.write(json.dumps(item.__dict__, ensure_ascii=True) + "\n")


def main() -> int:
    dataset = load_dataset()
    baseline_rows = evaluate_baseline(dataset)
    prism_rows = evaluate_with_sidecar(dataset, enable_memshield_rag=False, port=8890)
    prism_memshield_rows = evaluate_with_sidecar(dataset, enable_memshield_rag=True, port=8891)

    metrics = [
        score("baseline", baseline_rows),
        score("prism_only", prism_rows),
        score("prism_plus_memshield", prism_memshield_rows),
    ]
    append_results(metrics)
    print(format_table(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
