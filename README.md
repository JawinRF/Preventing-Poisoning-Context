## Preventing Poisoned Context for Mobile Agents

This repo implements a complete defense system against context poisoning attacks on LLM-driven mobile agents, covering all 7 Android ingestion paths.

### Architecture

```
Android Emulator
  ├─ dump_hierarchy()           → ui_accessibility
  ├─ adb dumpsys notification   → notifications
  ├─ adb clipboard              → clipboard
  ├─ adb logcat (intents)       → android_intents
  ├─ adb shell cat              → shared_storage
  └─ (network proxy)            → network_responses
         │
         ▼
ContextAssembler (scripts/context_assembler.py)
  │  For each source: POST /v1/inspect → Python sidecar (port 8765)
  │  For RAG: MemShield.query() wrapping ChromaDB → rag_store
  │
  ▼
Clean AssembledContext → only ALLOW verdicts pass through
         │
         ▼
Agent LLM (Groq or Claude) — sees ONLY sanitized context
         │
         ▼
Action Decision → PRISM checks sensitive outgoing actions
         │
         ▼
uiautomator2 executes on emulator
```

**Two complementary layers:**

- **PRISM Shield** (ingestion layer): Normalizer → Layer 1 Heuristics → Layer 2 TinyBERT → Layer 3 DeBERTa
- **MemShield** (retrieval layer): Wraps ChromaDB to filter poisoned RAG chunks with audit logging

### Project structure

```
scripts/
  agent_prism.py              # Defended agent (Groq or Claude + full PRISM filtering)
  context_assembler.py        # Gathers all 7 ingestion paths, filters through PRISM
  prism_client.py             # HTTP client for the PRISM sidecar
  agent.py                    # Original undefended agent (for A/B comparison)
  prism_shield/               # Defense pipeline
    pipeline.py               # PrismShield orchestrator
    normalizer.py             # De-obfuscation (URL, Base64, Unicode, ANSI)
    layer1_heuristics.py      # Fast regex-based injection detection
    layer2_local_llm.py       # TinyBERT binary classifier
    layer3_deberta.py         # DeBERTa deep semantic check
    ui_extractor.py           # Accessibility tree preprocessor
  openclaw_adapter/
    server.py                 # HTTP sidecar (/v1/inspect, /v1/inspect/batch)
    models.py                 # Request/response schemas
  demo/
    run_full_demo.py          # End-to-end demo (all 7 paths)
    run_demo.py               # Scenario-based sidecar test
    run_android_demo.sh       # Full emulator demo orchestration

android/
  prism-shield-service/       # On-device service (Compose dashboard, ONNX Layer 1+2)
  poison-app/                 # Attack simulator (sends poisoned notifications)

memshield/                    # Standalone MemShield Python package
data/                         # Synthetic dataset, audit logs, benchmarks
models/                       # TinyBERT (FP32 + INT8), VLM
```

### Quick start

```bash
# 1. Setup
python -m venv env && source env/bin/activate
pip install torch transformers datasets pandas scikit-learn numpy requests chromadb
pip install -e ./memshield[all]

# 2. Train the TinyBERT classifier
python scripts/train_tinybert.py

# 3. Run the demo (no emulator needed)
python scripts/demo/run_full_demo.py

# 4. Run the defended agent (requires emulator + GROQ_API_KEY)
python scripts/agent_prism.py --task "Set alarm for 9 AM"

# 5. Compare: undefended agent (PRISM bypassed)
python scripts/agent_prism.py --task "Set alarm for 9 AM" --no-prism
```

### Port assignment

| Service | Port | Purpose |
|---------|------|---------|
| Python PRISM sidecar | 8765 | Agent's primary filter (Layer 1+2+3) |
| Android PrismShieldService | 8766 | On-device dashboard, clipboard/notification hooks |

### Benchmark

```bash
python scripts/run_benchmark.py         # Per-path precision/recall/F1 + latency
python scripts/run_redteam_mutations.py  # Robustness against obfuscation attacks
```

### 7 defended ingestion paths

| Path | Source | Capture Method |
|------|--------|----------------|
| `ui_accessibility` | Screen content | `uiautomator2.dump_hierarchy()` |
| `notifications` | System notifications | `adb dumpsys notification` |
| `clipboard` | Clipboard content | `adb service call clipboard` |
| `android_intents` | Deep links, intents | `adb logcat ActivityManager` |
| `network_responses` | API responses | Network proxy / seeded data |
| `shared_storage` | Files on device | `adb shell cat` watched paths |
| `rag_store` | Vector DB retrieval | MemShield-wrapped ChromaDB |
