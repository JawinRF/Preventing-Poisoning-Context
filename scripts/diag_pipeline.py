# scripts/diag_pipeline.py
import json
import time
from prism_shield.base import MemoryEntry
from prism_shield.pipeline import PrismShield

def main():
    print("Loading Prism Shield Pipeline (Layer 1 + Layer 2)...")
    pipeline = PrismShield()
    
    print("\nLoading synthetic dataset...")
    # Using absolute path resolution for reading from data/
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data/prism_synthetic_dataset.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} entries.\n")
    print("-" * 60)
    
    # Let's test a mix of 5 benign and 5 poisoned samples
    benign_tested = 0
    poisoned_tested = 0
    
    for item in dataset:
        if item["label"] == "benign" and benign_tested < 5:
            benign_tested += 1
        elif item["label"] == "poisoned" and poisoned_tested < 5:
            poisoned_tested += 1
        else:
            if benign_tested == 5 and poisoned_tested == 5:
                break
            continue
            
        entry = MemoryEntry(
            id=item["id"], 
            text=item["text"], 
            ingestion_path=item["ingestion_path"]
        )
        
        start_time = time.time()
        result = pipeline.evaluate_entry(entry)
        latency = (time.time() - start_time) * 1000  # ms
        
        print(f"ID: {item['id']} | True Label: {item['label'].upper()} \nPath: {item['ingestion_path']}")
        print(f"Verdict: {result.verdict} (Confidence: {result.confidence:.2f})")
        print(f"Triggered by: {result.layer_triggered}")
        print(f"Reason: {result.reason}")
        print(f"Latency: {latency:.2f}ms")
        print("-" * 60)

if __name__ == "__main__":
    main()
