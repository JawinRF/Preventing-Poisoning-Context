# Preventing Poisoned Context for Mobile Agents

This repo implements a defense stack for Android mobile agents. The project now has two active runtime surfaces:

- `:8765` — Python PRISM sidecar for text-path filtering
- `:8766` — merged Android sidecar for on-device security and UI-integrity checks

The current defended-agent path is intentionally split:

- **Observation path**: the agent sees the full Android screen so it can navigate reliably. Screenshots are overlaid with numbered Set-of-Mark bubbles (red = clickable, blue = text input) so the LLM picks targets by `idx` instead of guessing coordinates.
- **Action path**: taps and typed text are verified before execution. `tap` accepts `idx` (preferred — host resolves to element xy), `xy`, `rid`, `text`, `desc`, or `class`.
- **Data paths**: notifications, clipboard, SMS, contacts, storage, and RAG are filtered before reaching the LLM.

## Current status

- The merged Android app at `android/openclaw-prism` builds, installs, launches, and serves `:8766/health`, `/v1/inspect`, `/v1/ui-integrity`, and `/v1/context`.
- The active Python sidecar on `:8765` is text-only in the request path. The old Moondream/VLM runtime dependency has been removed.
- UI observation is now **annotate, not filter**:
  suspicious screen text is marked with `prism_warning`, but screen elements are no longer hidden behind `[PRISM_FILTERED]`.
- Calendar is no longer polled by default for the live agent because it added attack surface and noise without helping navigation.
- MemShield remains fully implemented, but the live agent defaults to a **lightweight** mode. Full retrieval defense is optional via environment flags.

## Runtime architecture

```text
Android emulator / device
  ├─ uiautomator2 dump_hierarchy()      -> screen elements
  ├─ Chrome CDP (when WebContent seen)  -> web page body text
  ├─ Android sidecar :8766/v1/context   -> notifications, clipboard, SMS, contacts
  ├─ adb shell cat                      -> watched shared-storage files
  └─ ChromaDB + MemShield               -> RAG context
                 |
                 v
ContextAssembler (scripts/context_assembler.py)
  ├─ UI elements: unfiltered, regex-annotated only
  ├─ Notifications / clipboard / SMS / contacts / storage: filtered via :8765
  └─ RAG: MemShield query() wrapping ChromaDB
                 |
                 v
AssembledContext
  ├─ TASK / SCREEN
  ├─ DEVICE DATA boundary markers
  ├─ security_note (blocked device data)
  └─ security_warning (annotated UI elements)
                 |
                 v
LLM agent (Claude / Groq / local)
                 |
                 v
DefendedDevice
  ├─ outgoing text checks via :8765
  ├─ tap integrity via :8766/v1/ui-integrity
  ├─ web_tap / web_type via Chrome CDP
  └─ obvious-action fast path + loop detection
```

## Defense layers

### 1. PRISM Shield (`:8765`)

Text-path filtering for untrusted data before it reaches the LLM:

- **Normalizer** — Unicode NFC, confusable character folding, zero-width stripping
- **Layer 2 TinyBERT v3** — 44K-sample fine-tuned classifier, shared ONNX artifact with Android (`tinybert_prism.onnx`). For `ui_accessibility`, short inputs are wrapped in `<hierarchy><node .../></hierarchy>` to match training distribution
- **Layer 3 DeBERTa** — `ProtectAI/deberta-v3-base-prompt-injection-v2` safety net

The old Layer 1 regex heuristic stage was removed; TinyBERT v3 handles the same patterns with fewer false positives.

**Configurable thresholds** (env vars):

| Variable | Default | Scope |
|---|---|---|
| `PRISM_L2_BLOCK_THRESHOLD` | `0.85` | Default block threshold |
| `PRISM_L2_UI_BLOCK_THRESHOLD` | `0.70` | `ui_accessibility` path |
| `PRISM_L2_ALLOW_THRESHOLD` | `0.35` | Default allow band |
| `PRISM_L2_UI_ALLOW_THRESHOLD` | `0.40` | `ui_accessibility` path |

**QUARANTINE resolution** is path-dependent:

- Incoming text (`notifications`, `clipboard`, ...): `QUARANTINE → BLOCK`
- Agent's own output (`agent_output`): `QUARANTINE → ALLOW`

Active filtered paths:

- `notifications`
- `clipboard`
- `sms`
- `contacts`
- `shared_storage`
- `rag_store`

### 2. UI Integrity (`:8766`)

Deterministic pre-action checks on the Android side:

- foreground package verification
- overlay / obscuration detection
- target node validation
- bounds + interactability checks
- dual-snapshot stability checks

### 3. MemShield (RAG defense)

MemShield wraps ChromaDB and supports two modes.

Default live mode:

- ingest-time normalization
- regex/statistical checks
- SHA-256 provenance

Optional full retrieval defense:

- leave-one-out influence
- RAGMask token fragility
- authority prior
- copy ratio
- composite poison scorer + reranking
- optional ProGRank perturbation instability

## Agent observation + action

Each step:

1. `context_assembler.py` dumps the UI hierarchy via uiautomator2 and parses every element into `{idx, xy, rid, class, text?, desc?, input_field?}`. Clickable icon buttons with no label are kept.
2. A screenshot is captured and overlaid with numbered circles at each element's `xy` (Set-of-Mark prompting). Red = clickable, blue = text input.
3. The LLM reads the list + annotated screenshot and replies with `{"action":"tap","params":{"idx":N}}`. `agent_prism.py` resolves `idx → xy` from the element list before calling `DefendedDevice.execute`, so the LLM cannot hallucinate coordinates.
4. `defended_device.py` runs PRISM + UI-integrity checks, then executes via `adb shell input tap` (for `xy`) or uiautomator2 selectors (for `rid`, `text`, `desc`).

Loop / stuck detection escalates to `press back` then `press home` only after several consecutive no-progress steps.

## Runtime modes

### Defended agent, default lightweight mode

```bash
python scripts/agent_prism.py --task "Open the todo app and add a task: Buy groceries" --llm claude
```

### Defended agent, full retrieval defense

```bash
PRISM_ENABLE_RETRIEVAL_DEFENSE=1 \
python scripts/agent_prism.py --task "Open the todo app and add a task: Buy groceries" --llm claude
```

### Defended agent, full retrieval defense + ProGRank

```bash
PRISM_ENABLE_RETRIEVAL_DEFENSE=1 \
PRISM_ENABLE_PROGRANK=1 \
python scripts/agent_prism.py --task "Open the todo app and add a task: Buy groceries" --llm claude
```

## Quick start

### 1. Set up Python env

```bash
python -m venv env
source env/bin/activate
pip install torch transformers datasets pandas scikit-learn numpy requests chromadb
pip install -e ./memshield[all]
```

### 2. Start the Python sidecar

```bash
cd ~/Desktop/samsung_prism_project
python scripts/openclaw_adapter/server.py
```

### 3. Build and launch the merged Android app

```bash
cd ~/Desktop/samsung_prism_project/android/openclaw-prism
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.openclaw.android.debug/com.openclaw.android.MainActivity
adb forward tcp:8766 tcp:8766
```

### 4. Run the defended agent

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Meeting with Prof tomorrow at 3pm" \
  --llm claude
```

### 5. Run the poison demo

```bash
cd ~/Desktop/samsung_prism_project
bash scripts/send_poison_notification.sh
```

Then rerun the defended agent with a benign task.

### 6. Run the MemShield demo

```bash
cd ~/Desktop/samsung_prism_project/memshield
PYTHONPATH=src:../scripts python demo_memshield.py
```

## Merged Android app

The merged app in `android/openclaw-prism` is the on-device PRISM surface.

Tabs:

- `Terminal` — OpenClaw host terminal/runtime UI
- `Dashboard` — app overview/status
- `Security` — PRISM counters, threat feed, sidecar status
- `Settings` — configuration and permissions

For the poisoning-defense demos, the important merged-app surface is usually `Security`. The defended Python agent does not depend on using the in-app `Terminal` tab.

## Chrome / WebView behavior

Chrome browsing now has two extra pieces of support:

- accessibility service auto-enable for better Android UI visibility
- Chrome DevTools Protocol integration for web content and `web_tap` / `web_type`

If Chrome page text still seems unavailable, restart Chrome once after the command-line flag is written.

## Project structure

```text
scripts/
  agent_prism.py              # Defended agent
  agent_claude.py             # Alternate defended agent entry
  defended_device.py          # Action-path enforcement, UI integrity, CDP actions
  context_assembler.py        # Builds TASK / SCREEN / DEVICE DATA prompt context
  prism_client.py             # HTTP client for :8765
  shared_patterns.py          # Injection regexes used for annotation and filtering
  openclaw_adapter/
    server.py                 # Python PRISM sidecar
  prism_shield/
    pipeline.py               # PRISM text pipeline (Normalizer -> L2 TinyBERT -> L3 DeBERTa)
    normalizer.py             # Unicode NFC + confusable + zero-width stripping
    ui_extractor.py           # Flatten accessibility node dumps
    layer2_local_llm.py       # TinyBERT v3, shared ONNX runtime, path-aware thresholds
    layer3_deberta.py         # ProtectAI DeBERTa fallback

android/
  openclaw-prism/
    app/src/main/java/com/openclaw/android/
      OpenClawService.kt
      security/
        PrismAccessibilityService.kt
        PrismNotificationListener.kt
        UiIntegrityChecker.kt
        ContentProviderReader.kt
        OnnxClassifier.kt            # Uses shared tinybert_prism.onnx
        BertWordPieceTokenizer.kt    # HF-compatible tokenizer (fixes hash-id drift)

memshield/
  src/memshield/
    shield.py
    influence.py
    ragmask.py
    authority.py
    progrank.py
    shadow.py
    scorer.py
    provenance.py
```

## Port assignment

| Service | Port | Purpose |
|---------|------|---------|
| Python PRISM sidecar | `8765` | Text filtering for device-data paths |
| Android sidecar | `8766` | On-device `/v1/inspect`, `/v1/ui-integrity`, `/v1/context`, `/v1/status` |
| Chrome CDP forward | `9222` | Web page content + `web_tap` / `web_type` |

## Notes and caveats

- Notification and accessibility services are auto-enabled by the defended agent via ADB when possible.
- If `:8766` is unavailable, tap integrity checks fail open for availability and a warning is logged.
- UI elements are no longer blocked by ML scanning. They are visible to the agent, with `prism_warning` annotations on obviously suspicious text.
- The old Python-side defended demo and the merged Android app are related but separate:
  - `:8765` powers the Python text-filtering sidecar
  - `:8766` powers the merged Android sidecar
- Full MemShield retrieval defense is available, but it is intentionally not the default live mode because of runtime cost.

## Tests

```bash
env/bin/python -m pytest -q memshield/tests/test_memshield.py
env/bin/python -m pytest -q tests/test_agent_integration.py -k "not Reflection"
```
