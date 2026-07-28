# PRISM — Preventing Poisoned Context for Mobile Agents

PRISM is a runtime defense stack that protects Android AI agents against prompt injection. An agent operating on a phone receives data from many untrusted sources — notifications, clipboard, SMS, shared files, web content, accessibility trees — all of which can be weaponized by an attacker to hijack the agent's actions. PRISM intercepts every data path before it reaches the language model and makes a block/allow decision before any token is assembled.

## Architecture overview

```
Android emulator / device
  ├─ uiautomator2 dump_hierarchy()      → screen elements
  ├─ Chrome CDP                         → web page body text
  ├─ Android sidecar :8766/v1/context   → notifications, clipboard, SMS, contacts
  ├─ adb shell cat                      → watched shared-storage files
  └─ ChromaDB + MemShield               → RAG context
                 │
                 ▼
        ContextAssembler
  ├─ UI elements: annotated (not blocked)
  ├─ Device data: filtered via :8765
  └─ RAG: MemShield-wrapped ChromaDB
                 │
                 ▼
        AssembledContext
  ├─ TASK / SCREEN
  ├─ DEVICE DATA boundary markers
  ├─ security_note   (blocked device data)
  └─ security_warning (annotated UI elements)
                 │
                 ▼
        LLM agent (Claude / Groq / local)
                 │
                 ▼
        DefendedDevice
  ├─ outgoing text checks via :8765
  ├─ tap integrity via :8766/v1/ui-integrity
  ├─ web_tap / web_type via Chrome CDP
  └─ loop detection + stuck recovery
```

Two sidecars run in parallel:

| Sidecar | Port | Role |
|---------|------|------|
| Python PRISM | `8765` | Text-path filtering for all device-data ingestion paths |
| Android PRISM | `8766` | On-device UI integrity, context collection, `/v1/inspect` |
| Chrome CDP | `9222` | Web content extraction and `web_tap` / `web_type` |

The agent observation path (screenshots + UI hierarchy) is **annotate, not block** — the LLM sees the full screen with numbered Set-of-Mark bubbles so it can navigate reliably. Suspicious text elements are tagged with `prism_warning`. All device-sourced *data* (notifications, clipboard, SMS, RAG) runs through the full ML pipeline before reaching the LLM.

---

## Defense layers

### Layer 1 — PRISM Shield (`:8765`)

A multi-stage ML pipeline with format-aware pre-processing.

```
ContentExtractor → Normalizer → [TinyBERT v3 ‖ DeBERTa] → Ensemble → verdict
```

**ContentExtractor** detects the container format of incoming text and extracts only the semantic payload before any model sees it. This prevents distribution-shift false positives where classifiers trained on natural language receive Android XML hierarchies, file block wrappers, intent JSON, or HTML.

Handled formats:
- Android accessibility XML (`<?xml …<hierarchy>`)
- File block wrappers (`--- START FILE: … --- END FILE ---`)
- HTML / script tags
- Android intent JSON (`{"action":…, "data":…}`)
- Plain text (pass-through)

**Normalizer** applies:
- URL decoding
- Base64 detection and expansion
- Zero-width and invisible Unicode stripping
- ANSI escape code removal
- Whitespace flood compression
- Unicode confusable normalization

**TinyBERT v3** (Layer 2) — 44K-sample fine-tuned binary classifier, runs as a shared ONNX artifact on both host and Android (`tinybert_prism.onnx`). Training applies the identical preprocessing the inference pipeline applies (ContentExtractor, then the full Normalizer), so the model is in-distribution on every ingestion path. Uniform thresholds:

| Env var | Default | Scope |
|---------|---------|-------|
| `PRISM_L2_BLOCK_THRESHOLD` | `0.85` | All paths |
| `PRISM_L2_ALLOW_THRESHOLD` | `0.35` | All paths |

Both TinyBERT and DeBERTa apply a minimum context guard before scoring: texts that are too short for reliable ML assessment (sparse alphabetic content) are skipped by that layer. TinyBERT uses an alpha-run count threshold of 3; DeBERTa uses a space-word count threshold of 5. This avoids systematic calibration failures on short structured tokens (confirmation codes, phone numbers, bare URLs) while preserving detection of short natural-language injections.

**DeBERTa** (Layer 3) — `ProtectAI/deberta-v3-base-prompt-injection-v2` runs in parallel with TinyBERT. A local pinned copy is used when available (`models/deberta_prompt_injection_v2/`); the HuggingFace Hub is the fallback.

**Ensemble** combines both model outputs:

| L2 | L3 | Result |
|----|----|----|
| BLOCK | BLOCK | BLOCK |
| BLOCK | ALLOW | QUARANTINE |
| ALLOW | BLOCK | QUARANTINE |
| ALLOW | ALLOW | ALLOW |
| QUARANTINE | ALLOW (confident) | ALLOW |
| Any | Any other | QUARANTINE |

Single-model QUARANTINE (medium-confidence signal) is resolved before the ensemble:
- Untrusted paths (`notifications`, `clipboard`, etc.): `QUARANTINE → BLOCK`
- Agent output (`agent_output`): `QUARANTINE → ALLOW`

When one model returns QUARANTINE and the other returns ALLOW with very high safety confidence (injection probability < 10%), the pipeline trusts the confident-safe verdict and returns ALLOW. This handles cases where TinyBERT is uncertain on legitimate natural-language text that DeBERTa has high confidence is benign.

**L3-advisory paths.** DeBERTa is a frozen generalist trained on natural-language prompts. On paths whose extracted payload is structurally non-prompt-like (UI node label soup, RAG store records) its BLOCK verdicts are dominated by false positives, while TinyBERT is trained in-distribution on exactly those payloads. On paths in `PRISM_L3_ADVISORY_PATHS` (default `ui_accessibility,rag_store`), an L3 BLOCK opposed by an L2 ALLOW at ≥ `PRISM_L2_OVERRIDE_CONFIDENCE` (default `0.99`) benign probability resolves to ALLOW. L3 keeps full veto power on every other path.

**Quarantine tickets.** Every final QUARANTINE verdict from `/v1/inspect` mints a persistent review ticket (JSONL-backed, survives sidecar restarts), retrievable at `GET /v1/ticket/<id>`. The response carries the `ticket_id` and a quarantine placeholder instead of the original text.

**Active ingestion paths:**

| Path | Description |
|------|-------------|
| `notifications` | System and app notifications |
| `clipboard` | Clipboard read events |
| `sms` | SMS messages |
| `contacts` | Contact records |
| `shared_storage` | Watched files from external storage |
| `rag_store` | RAG retrieval results |
| `android_intents` | Incoming Android intent data |
| `network_responses` | HTTP / API responses |
| `ui_accessibility` | Accessibility tree text nodes |

**Benchmark performance** on the 1,498-entry synthetic evaluation set (full pipeline, QUARANTINE counted as a positive prediction):

| Metric | Value |
|--------|-------|
| Detection Rate (TPR) | 99.8% (594/595) |
| False Positive Rate | 0.0% (0/903) |
| Accuracy | 99.9% |
| Precision | 100.0% |
| F1 | 0.999 |

Per-path: 0.0% FPR on all seven paths; 100% recall on six paths, 98.8% on `android_intents` (one missed sample). Mean end-to-end latency 100-220 ms per entry on CPU.

**External held-out benchmark** (`data/prism_external_benchmark.json`, built by `scripts/build_external_benchmark.py`): 2,810 entries from datasets never used in training — Lakera Gandalf real injection attempts, deepset and safe-guard test splits, real SMS ham, AG News, Banking77 — decontaminated against all training corpora by normalized text match and wrapped in the same Android containers. This measures generalization rather than template fit:

| Metric | Full pipeline | TinyBERT alone |
|--------|--------------|----------------|
| False positive rate | 2.3% | 0.1% |
| Detection rate | 96.8% | 97.8% |

Run it with `python scripts/run_benchmark.py --dataset data/prism_external_benchmark.json`. The pipeline FPR above TinyBERT's own is DeBERTa disagreement routing benign external prose to QUARANTINE review on non-advisory paths, the intended defense-in-depth posture.

---

### Layer 2 — UI Integrity (`:8766`)

Deterministic checks run on the Android side before every tap:

- Foreground package verification (are we tapping what we think we're tapping?)
- Overlay and obscuration detection
- Target node validation
- Bounds and interactability checks
- Dual-snapshot stability (element didn't move between check and tap)

If `:8766` is unreachable, tap integrity fails open with a warning logged — availability is preserved at the cost of the integrity check.

---

### Layer 3 — MemShield (RAG defense)

MemShield wraps ChromaDB and defends the retrieval-augmented memory store against poisoning at both ingest and query time.

**Default lightweight mode** (always active):
- Ingest-time normalization
- String-pattern and statistical anomaly scanning
- SHA-256 provenance tracking per chunk
- Trust scoring based on ingestion context (notification/SMS source lowers initial trust)

**Optional full retrieval defense** (`PRISM_ENABLE_RETRIEVAL_DEFENSE=1`):
- Leave-one-out influence scoring
- RAGMask token fragility analysis
- Authority prior weighting
- Copy-ratio anomaly detection
- Composite poison scorer with reranking
- Optional ProGRank perturbation instability (`PRISM_ENABLE_PROGRANK=1`)

---

## Agent observation and action

Each agent step:

1. `context_assembler.py` dumps the UI hierarchy via uiautomator2 and parses every node into `{idx, xy, rid, class, text?, desc?, input_field?}`. Clickable icon buttons without labels are retained.
2. A screenshot is captured and overlaid with numbered circles at each element's `xy` (Set-of-Mark prompting). Red = clickable, blue = text input.
3. The LLM reads the element list and annotated screenshot and replies with a structured action like `{"action":"tap","params":{"idx":3}}`. `agent_prism.py` resolves `idx → xy` from the element list before calling `DefendedDevice.execute` — the LLM cannot hallucinate coordinates.
4. `defended_device.py` runs PRISM + UI-integrity checks, then executes via `adb shell input tap` (for `xy`) or uiautomator2 selectors (for `rid`, `text`, `desc`).

Loop and stuck detection escalates to `press back` then `press home` after several consecutive no-progress steps.

---

## Quick start

### 1. Python environment

```bash
python -m venv env
source env/bin/activate
pip install torch transformers datasets pandas scikit-learn numpy requests chromadb
pip install -e ./memshield[all]
```

### 2. Start the Python sidecar

```bash
python scripts/openclaw_adapter/server.py
```

### 3. Build and launch the Android app

```bash
cd android/openclaw-prism
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.openclaw.android.debug/com.openclaw.android.MainActivity
adb forward tcp:8766 tcp:8766
```

### 4. Run the defended agent

```bash
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)

# Default (lightweight MemShield)
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Buy groceries" \
  --llm claude

# Full retrieval defense
PRISM_ENABLE_RETRIEVAL_DEFENSE=1 \
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Buy groceries" \
  --llm claude

# Full retrieval defense + ProGRank
PRISM_ENABLE_RETRIEVAL_DEFENSE=1 \
PRISM_ENABLE_PROGRANK=1 \
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Buy groceries" \
  --llm claude
```

### 5. Demo: send a poison notification

```bash
bash scripts/send_poison_notification.sh
```

Run the defended agent with a benign task immediately after. The poisoned notification will be intercepted and blocked before reaching the LLM.

### 6. Demo: MemShield

```bash
cd memshield
PYTHONPATH=src:../scripts python demo_memshield.py
```

---

## Android app

The merged app at `android/openclaw-prism` is the on-device PRISM surface.

Tabs:

| Tab | Purpose |
|-----|---------|
| Terminal | OpenClaw host terminal / runtime UI |
| Dashboard | App overview and status |
| Security | PRISM counters, threat feed, sidecar health |
| Settings | Configuration and permissions |

The defended Python agent does not require interaction with the in-app Terminal — it communicates with the Android sidecar over ADB-forwarded TCP ports.

---

## Project structure

```
scripts/
  agent_prism.py              # Defended agent (main entry point)
  agent_claude.py             # Alternate agent entry
  defended_device.py          # Action-path enforcement, UI integrity, CDP actions
  context_assembler.py        # Assembles TASK / SCREEN / DEVICE DATA prompt context
  prism_client.py             # HTTP client for :8765
  openclaw_adapter/
    server.py                 # Python PRISM sidecar (:8765)
  prism_shield/
    pipeline.py               # Main pipeline: ContentExtractor → Normalizer → [L2 ‖ L3] → Ensemble
    content_extractor.py      # Format-aware semantic payload extractor
    normalizer.py             # Unicode + obfuscation normalization
    layer2_local_llm.py       # TinyBERT v3, shared ONNX runtime, path-aware thresholds
    layer3_deberta.py         # ProtectAI DeBERTa safety net
    base.py                   # Shared types (MemoryEntry, ValidationResult)

android/
  openclaw-prism/
    app/src/main/java/com/openclaw/android/
      OpenClawService.kt
      security/
        PrismAccessibilityService.kt
        PrismNotificationListener.kt
        UiIntegrityChecker.kt
        ContentProviderReader.kt
        OnnxClassifier.kt            # Runs shared tinybert_prism.onnx on-device
        BertWordPieceTokenizer.kt    # HF-compatible tokenizer

memshield/
  src/memshield/
    shield.py       # Main scan_chunk / query interface
    influence.py    # Leave-one-out influence scoring
    ragmask.py      # Token fragility analysis
    authority.py    # Authority prior weighting
    progrank.py     # Perturbation instability (ProGRank)
    shadow.py       # Shadow copy management
    scorer.py       # Composite poison scorer
    provenance.py   # SHA-256 chunk provenance

data/                             # tracked datasets; everything else is runtime state
  prism_synthetic_dataset.json    # 1,498-entry evaluation set (7 ingestion paths)
  prism_external_benchmark.json   # 2,810-entry held-out generalization set
  prism_training_dataset.json     # 55,019-entry TinyBERT v3 training set
  external/                       # raw corpora, NOT tracked (432MB)
                                  # restore with scripts/download_datasets.py --eval

models/
  tinybert_poison_classifier_v3/  # TinyBERT v3 FP32 weights
  deberta_prompt_injection_v2/    # Pinned DeBERTa weights (optional)

android/openclaw-prism/app/src/main/assets/
  tinybert_prism.onnx             # Shared ONNX artifact (host + Android)
```

---

## Tests

```bash
# Unit tests (no sidecar required)
python -m pytest tests/ --ignore=tests/test_sidecar.py -q

# Integration test (requires running sidecar on :8765)
python -m pytest tests/test_sidecar.py -q

# MemShield unit tests
python -m pytest memshield/tests/ -q

# Benchmark (1,498 samples, ~5 minutes)
python scripts/run_benchmark.py
```

---

## Notes

- Notification and accessibility services are auto-enabled by the agent via ADB when available.
- Calendar polling is disabled by default — it added attack surface without improving navigation.
- UI elements are never blocked from the agent's view; only device-sourced data (notifications, clipboard, SMS, storage, RAG) is filtered.
- Full MemShield retrieval defense is opt-in because of runtime cost. The lightweight default covers the vast majority of poisoning attempts at lower latency.
- The TinyBERT ONNX artifact (`tinybert_prism.onnx`) is shared between the Python host and the Android app so both surfaces run the same model weights.
