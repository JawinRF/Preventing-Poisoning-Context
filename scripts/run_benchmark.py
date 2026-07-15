# scripts/run_benchmark.py

import argparse
import json
import time
from collections import defaultdict
from prism_shield.base import MemoryEntry
from prism_shield.pipeline import PrismShield

def main():
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))

    parser = argparse.ArgumentParser(description="Benchmark the PRISM pipeline on a labelled dataset")
    parser.add_argument(
        "--dataset",
        default=os.path.join(base_dir, "data/prism_synthetic_dataset.json"),
        help="Path to a JSON list of {text, ingestion_path, label} entries",
    )
    args = parser.parse_args()

    print("Loading PRISM Shield Target...")
    pipeline = PrismShield()

    print(f"Loading dataset from {args.dataset} ...")
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} entries. Running evaluation...\n")
    
    # Metrics tracking: [TP, TN, FP, FN, TotalLatency, Count, Latencies Array]
    stats = defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "latency": 0.0, "count": 0, "latencies": []})
    
    # Constants
    BENIGN = "benign"
    POISONED = "poisoned"
    
    start_time_all = time.time()
    
    for item in dataset:
        entry = MemoryEntry(
            id=item["id"],
            text=item["text"],
            ingestion_path=item["ingestion_path"]
        )
        
        path = item["ingestion_path"]
        true_label = item["label"]
        
        t0 = time.time()
        result = pipeline.evaluate_entry(entry)
        latency = (time.time() - t0) * 1000
        
        # In our architecture, BLOCK and QUARANTINE both prevent direct execution
        pred_poisoned = result.verdict in ["BLOCK", "QUARANTINE"]
        
        if true_label == POISONED and pred_poisoned:
            stats[path]["TP"] += 1
            stats["OVERALL"]["TP"] += 1
        elif true_label == BENIGN and not pred_poisoned:
            stats[path]["TN"] += 1
            stats["OVERALL"]["TN"] += 1
        elif true_label == BENIGN and pred_poisoned:
            stats[path]["FP"] += 1
            stats["OVERALL"]["FP"] += 1
        elif true_label == POISONED and not pred_poisoned:
            stats[path]["FN"] += 1
            stats["OVERALL"]["FN"] += 1
            
        stats[path]["latency"] += latency
        stats[path]["count"] += 1
        stats[path]["latencies"].append(latency)
        stats["OVERALL"]["latency"] += latency
        stats["OVERALL"]["count"] += 1
        stats["OVERALL"]["latencies"].append(latency)
        
    total_time = time.time() - start_time_all
    import numpy as np
    
    # Calculate and Print Markdown Tables
    print("\n### Per-Path Metrics")
    print("| Ingestion Path | Precision | Recall | F1 Score | TP | FP | FN | TN |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for path in sorted(stats.keys()):
        if path == "OVERALL":
            continue
            
        d = stats[path]
        tp, tn, fp, fn = d["TP"], d["TN"], d["FP"], d["FN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"| `{path}` | {precision:.2%} | {recall:.2%} | {f1:.2%} | {tp} | {fp} | {fn} | {tn} |")
        
    print("\n### Latency Percentiles (End-to-End)")
    print("| Ingestion Path | Mean Latency | P50 | P95 | P99 |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for path in sorted(stats.keys()):
        if path == "OVERALL":
            continue
            
        d = stats[path]
        lats = d["latencies"]
        mean_lat = d["latency"] / len(lats) if lats else 0
        p50 = np.percentile(lats, 50) if lats else 0
        p95 = np.percentile(lats, 95) if lats else 0
        p99 = np.percentile(lats, 99) if lats else 0
        
        print(f"| `{path}` | {mean_lat:.2f}ms | {p50:.2f}ms | {p95:.2f}ms | {p99:.2f}ms |")

    # Overall line just to show total time as output
    d = stats["OVERALL"]
    count = d["count"]
    print(f"\nTotal Evaluation Time: {total_time:.2f}s for {count} entries.")

if __name__ == "__main__":
    main()
