import json
from prism_shield.base import MemoryEntry
from prism_shield.pipeline import PrismShield

def main():
    print("Loading Prism Shield Pipeline...")
    pipeline = PrismShield()
    
    import os
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data/prism_synthetic_dataset.json")
    
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} entries. Testing ui_accessibility...\n")
    
    ui_data = [d for d in dataset if d["ingestion_path"] == "ui_accessibility"]
    
    fp_count = 0
    fn_count = 0
    
    for item in ui_data:
        entry = MemoryEntry(
            id=item["id"], 
            text=item["text"], 
            ingestion_path=item["ingestion_path"]
        )
        
        result = pipeline.evaluate_entry(entry)
        
        is_poisoned = item["label"] == "poisoned"
        pred_poisoned = result.verdict in ["BLOCK", "QUARANTINE"]
        
        if not is_poisoned and pred_poisoned:
            if fp_count < 3:
                print(f"--- FALSE POSITIVE ---")
                print(f"Original: {item['text'][:300]}")
                print(f"Normalized: {result.normalized_text[:100]}")
                print(f"Verdict: {result.verdict} by {result.layer_triggered}")
                print(f"Reason: {result.reason}\n")
            fp_count += 1
            
        elif is_poisoned and not pred_poisoned:
            if fn_count < 3:
                print(f"--- FALSE NEGATIVE (Missed) ---")
                print(f"Original: {item['text'][:300]}")
                print(f"Notes: {item.get('notes', '')}")
                print(f"Normalized: {result.normalized_text[:100]}")
                print(f"Verdict: {result.verdict} by {result.layer_triggered}")
            fn_count += 1
            
    print(f"Total FP: {fp_count}")
    print(f"Total FN: {fn_count}")

if __name__ == "__main__":
    main()
