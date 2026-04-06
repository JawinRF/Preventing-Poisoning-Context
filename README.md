## Preventing Poisoned Context for Mobile Agents

This repo implements a defense system against context poisoning attacks on LLM-driven mobile agents, with 7 active ingestion paths plus RAG retrieval defense. A network-response interceptor is still planned and is not part of the active runtime.

### Current status

- The merged Android app at `android/openclaw-prism` is validated at a meaningful level:
  build/install/launch work, `:8766/health` responds, `/v1/inspect` responds, and
  the Security/Dashboard/Terminal tabs render on-device.
- The active Python PRISM sidecar path on `:8765` is now text-only:
  no synchronous Moondream/VLM runs in the request path.
- `agent_prism.py` currently depends on both sidecars during a defended run:
  `:8765` for ingestion filtering and `:8766` for UI integrity checks before taps.
- Known runtime caveat:
  some legitimate typed strings are still being falsely blocked by the text-defense
  pipeline, and the defended agent demo should be treated as in-progress until that
  policy tuning is fixed.

### Architecture

```
Android Emulator
  ├─ dump_hierarchy()           → ui_accessibility
  ├─ PrismNotificationListener  → notifications (via TCP socket)
  ├─ ContentProviderReader      → sms, contacts, calendar (via TCP socket)
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

**Three complementary defense layers:**

- **PRISM Shield** (ingestion layer): Normalizer → Layer 1 Heuristics → Layer 2 TinyBERT → Layer 3 DeBERTa
- **MemShield** (RAG defense): Two-phase pipeline wrapping ChromaDB:
  - *Ingest-time*: Normalization → regex → statistical anomaly → ML classifiers → SHA-256 provenance hashing
  - *Retrieval-time*: Provenance verification → leave-one-out influence → RAGMask token fragility → authority prior → copy ratio → composite poison scorer σ(w·x) → reranking
- **UI Integrity** (tap defense): OS-level checks via Android sidecar — foreground package verification, overlay detection, node validation, dual-snapshot stability

### Project structure

```
scripts/
  agent_prism.py              # Defended agent (Groq or Claude + full PRISM filtering)
  context_assembler.py        # Gathers active device context, filters through PRISM
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
    run_full_demo.py          # End-to-end Python sidecar demo (7 active paths)
    run_demo.py               # Scenario-based sidecar test
    run_android_demo.sh       # Full emulator demo orchestration

android/
  openclaw-prism/             # Merged Android app (PRISM + OpenClaw runtime)
    security/
      PrismAccessibilityService.kt  # Accessibility service (Layer 1+2 on-device)
      UiIntegrityChecker.kt         # OS-level tap safety (overlay, node, stability)
      OnnxClassifier.kt             # On-device ONNX Layer 2
    OpenClawService.kt        # NanoHTTPD sidecar (:8766, /v1/inspect, /v1/ui-integrity)
  prism-shield-service/       # Legacy on-device service (Compose dashboard)
  poison-app/                 # Attack simulator (sends poisoned notifications)

memshield/                    # MemShield RAG defense package (two-phase pipeline)
  src/memshield/
    shield.py                 # Core: ingest scan + retrieval defense orchestration
    influence.py              # Leave-one-out influence scoring (semantic + citation drift)
    ragmask.py                # RAGMask token-masking fragility
    authority.py              # Authority prior (source trust, domain rep, entity corroboration)
    progrank.py               # ProGRank perturbation instability
    shadow.py                 # Shadow synthetic memory (TTL + corroboration)
    scorer.py                 # Composite poison scorer + reranking + weight tuning
    provenance.py             # SHA-256 content hashing + tamper detection
    audit.py                  # JSONL audit logging
data/                         # Synthetic dataset, audit logs, benchmarks
models/                       # TinyBERT / DeBERTa assets + offline audit VLM weights
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

# 4. Start the Python PRISM sidecar
python scripts/openclaw_adapter/server.py

# 5. Start the merged Android app / on-device sidecar (:8766)
cd android/openclaw-prism
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.openclaw.android.debug/com.openclaw.android.MainActivity
adb forward tcp:8766 tcp:8766
cd ../..

# 6. Run the defended agent (requires emulator + ANTHROPIC_API_KEY or GROQ_API_KEY)
export PRISM_SIDECAR_URL=http://127.0.0.1:8765
python scripts/agent_prism.py --task "Set alarm for 9 AM"

# 7. Compare: undefended agent (PRISM bypassed)
python scripts/agent_prism.py --task "Set alarm for 9 AM" --no-prism

# 8. Run the MemShield RAG defense demo
cd memshield && PYTHONPATH=src:../scripts python demo_memshield.py

# 9. Run MemShield tests (37 tests covering full pipeline)
cd memshield && PYTHONPATH=src:../scripts python -m pytest tests/ -v
```

### Defended agent caveats

- `agent_prism.py` uses the Python sidecar on `:8765` for context filtering and the
  Android sidecar on `:8766` for `/v1/ui-integrity`.
- If `:8766` is not reachable, tap integrity checks fail open for availability and a
  warning is logged.
- Notification/SMS/contacts/calendar ingestion may be degraded in some emulator
  sessions if the relevant Android socket/listener path is unavailable.
- Some defended-agent false positives are still being tuned; treat the Python
  defended demo as in-progress rather than fully polished.

### Port assignment

| Service | Port | Purpose |
|---------|------|---------|
| Python PRISM sidecar | 8765 | Agent's primary filter (Layer 1+2+3) |
| Android sidecar (OpenClawService) | 8766 | On-device PRISM, UI integrity (`/v1/inspect`, `/v1/ui-integrity`) |
| PrismNotificationListener | 8767 | Notifications, SMS, contacts, calendar (TCP via ADB forward) |

### Benchmark

```bash
python scripts/run_benchmark.py         # Per-path precision/recall/F1 + latency
python scripts/run_redteam_mutations.py  # Robustness against obfuscation attacks
```

### Active ingestion paths

| Path | Source | Capture Method |
|------|--------|----------------|
| `ui_accessibility` | Screen content | `uiautomator2.dump_hierarchy()` |
| `notifications` | System notifications | `PrismNotificationListener` TCP socket |
| `sms` | SMS inbox | `ContentProviderReader` TCP socket |
| `contacts` | Contact notes | `ContentProviderReader` TCP socket |
| `calendar` | Calendar events | `ContentProviderReader` TCP socket |
| `clipboard` | Clipboard content | `adb service call clipboard` |
| `android_intents` | Deep links, intents | `adb logcat ActivityManager` |
| `shared_storage` | Files on device | `adb shell cat` watched paths |
| `rag_store` | Vector DB retrieval | MemShield-wrapped ChromaDB |

### Planned path

| Path | Source | Capture Method |
|------|--------|----------------|
| `network_responses` | API responses via proxy/VPN interceptor | Not yet implemented |
