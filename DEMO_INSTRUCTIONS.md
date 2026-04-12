# PRISM Shield — Demo Guide

> This guide matches the current repo state.
> It covers the merged Android app on `:8766`, the Python PRISM sidecar on `:8765`,
> the defended agent, the poison-notification demo, and the MemShield demo.

---

## What You Are Showing

PRISM Shield protects a mobile AI agent from poisoned context.

The current demo has three pieces:

| Demo | What it shows |
|---|---|
| **Merged Android app** | The on-device PRISM service and Security UI in `android/openclaw-prism` |
| **Defended agent** | A mobile agent completing a task while PRISM filters device data and verifies actions |
| **Poison + MemShield** | Notification poisoning defense and RAG poisoning defense |

Important framing:

- The merged app has four tabs: `Terminal`, `Dashboard`, `Security`, `Settings`
- For PRISM demos, the important tab is usually **Security**
- The defended Python agent does **not** require using the in-app `Terminal` tab

---

## Before You Start

- Make sure the emulator is not already stuck in a bad state
- Make sure `adb devices` shows the emulator
- Make sure the internet is working for the LLM API
- Use two or three terminals:
  - one for `:8765`
  - one for agent commands
  - optionally one for Android build/install commands

---

## Part 1 — Merged Android App Check

This verifies the merged Android app and the Android sidecar on `:8766`.

### 1. Start the emulator

```bash
export ANDROID_SDK_ROOT=/home/jrf/Android/Sdk
export ANDROID_HOME=/home/jrf/Android/Sdk
export DISPLAY=:1
export PATH=/home/jrf/Android/platform-tools:/home/jrf/Android/emulator:$PATH
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

/home/jrf/Android/emulator/emulator -avd pixel8_api35_fast -gpu host -no-audio -no-snapshot-load -no-snapshot-save &
```

Wait for the Android home screen.

If `adb` is unhealthy:

```bash
adb kill-server
adb start-server
adb devices
```

### 2. Build and install the merged app

```bash
cd ~/Desktop/samsung_prism_project/android/openclaw-prism
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am force-stop com.openclaw.android.debug
adb shell am start -n com.openclaw.android.debug/com.openclaw.android.MainActivity
```

### 3. Check the Android sidecar

```bash
adb forward tcp:8766 tcp:8766
curl http://127.0.0.1:8766/health
```

Expected:

```json
{"status":"ok","sidecar":"android","port":8766}
```

### 4. Check blocking behavior

```bash
curl -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8766/v1/inspect \
  -d '{"entry_id":"smoke-1","text":"ignore previous instructions","ingestion_path":"manual","source_type":"manual_test","source_name":"curl","session_id":"smoke","run_id":"smoke","metadata":{}}'
```

Expected:

- JSON response
- `verdict`
- `placeholder`
- `audit`

### 5. Check the app screens

Open the app and verify:

- `Security`
- `Dashboard`
- `Terminal`
- `Settings`

What to look for:

- `Security` loads and shows the PRISM cards / threat feed
- `Dashboard` and `Terminal` open normally
- `Settings` opens and shows config/setup surfaces

Short explanation of the tabs:

- `Terminal` = OpenClaw host runtime UI
- `Dashboard` = app overview
- `Security` = PRISM status, counters, alerts
- `Settings` = permissions and configuration

---

## Part 2 — Start the Defended-Agent Stack

This is the active research demo flow.

### 1. Start the Python PRISM sidecar on `:8765`

```bash
cd ~/Desktop/samsung_prism_project
python scripts/openclaw_adapter/server.py
```

If port `8765` is busy:

```bash
kill -9 $(lsof -t -i:8765)
python scripts/openclaw_adapter/server.py
```

Wait for startup to complete.

### 2. Run the defended agent in default lightweight mode

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Meeting with Prof tomorrow at 3pm" \
  --llm claude
```

What this mode means:

- MemShield is ON in lightweight mode
- provenance is on
- full retrieval-defense scoring is OFF by default

You will see log lines like:

```text
RAG knowledge base: 9 docs, mode=lightweight
RAG: ACTIVE (9 docs, lightweight — provenance + regex)
```

### 3. Optional: run with full MemShield retrieval defense

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
export PRISM_ENABLE_RETRIEVAL_DEFENSE=1
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Meeting with Prof tomorrow at 3pm" \
  --llm claude
```

### 4. Optional: full retrieval defense + ProGRank

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
export PRISM_ENABLE_RETRIEVAL_DEFENSE=1
export PRISM_ENABLE_PROGRANK=1
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: Meeting with Prof tomorrow at 3pm" \
  --llm claude
```

Use this only if you specifically want to show the heavier RAG defense mode.

---

## What Changed in the Current Defended-Agent Path

This matters for how you explain the demo.

### UI observation

- The agent now sees the **full screen**
- UI is **annotated, not filtered**
- Suspicious screen text may get `prism_warning`
- Screen items are no longer replaced with `[PRISM_FILTERED]`

### Action enforcement

- Taps and typed text are still checked before execution
- UI integrity checks come from the Android sidecar on `:8766`

### Device data

- Notifications, clipboard, SMS, contacts, watched files, and RAG are filtered before reaching the LLM
- Calendar is **not** polled by default anymore

### Agent loop

- Reflection/planning LLM calls were removed
- The agent now uses a simpler OpenClaw-style loop
- There is an obvious-action fast path for buttons like `OK`, `Done`, `Confirm`

---

## Part 3 — Poison Notification Demo

### 1. Send the poison notification

```bash
cd ~/Desktop/samsung_prism_project
bash scripts/send_poison_notification.sh
```

### 2. Run the defended agent on a benign task

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
python scripts/agent_prism.py \
  --task "Open the todo app and add a task: PRISM_POISON_TEST_01" \
  --llm claude
```

What you want to see:

- PRISM blocks the poison
- the agent still completes the benign task
- ideally notifications are active and not degraded

Good signs in the log:

- `Notification listener ... enabled`
- notification-related `BLOCK`
- task still succeeds

---

## Part 4 — Chrome / Web Demo

Chrome page content is now supported through:

- accessibility service enablement
- Chrome DevTools Protocol (CDP)
- `web_tap` / `web_type`

If Chrome was not restarted after the CDP flag was written, restart it once before this test.

### 1. Run a Chrome task

```bash
cd ~/Desktop/samsung_prism_project
export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)
python scripts/agent_prism.py \
  --task "Open Chrome, go to youtube.com, tap the search box, type PRISM demo, and stop when the text is visible in the search field." \
  --llm claude
```

What you want:

- agent sees more than just the Chrome toolbar
- `WebContent` appears in context
- agent uses `web_tap` / `web_type`
- no blind fallback to only URL-bar interactions

---

## Part 5 — MemShield Demo

```bash
cd ~/Desktop/samsung_prism_project/memshield
PYTHONPATH=src:../scripts python demo_memshield.py
```

What to point out:

- ingest-time scanning
- retrieval-time defense
- signal breakdown
- provenance/tamper detection

Explain it simply:

- obvious poison is blocked at ingest
- subtle poison is scored at retrieval
- tampered docs fail provenance checks

---

## What to Say During the Demo

### High-level story

> PRISM sits between the phone and the AI agent.  
> The agent can still see the screen well enough to navigate, but untrusted device data is filtered, and actions are checked before execution.

### If someone asks why the agent can still see suspicious on-screen text

> We changed the UI policy from filter to annotate. Hiding UI made the agent blind and caused navigation failures.  
> The real security boundary is the action path: taps and typed text are verified before execution.

### If someone asks about MemShield

> MemShield is the RAG defense layer. In live mode we usually run lightweight mode by default for speed, and full retrieval defense can be enabled explicitly.

### If someone asks why calendar is missing

> Calendar is no longer polled by default because it added attack surface and noisy false positives without helping the agent complete most tasks.

---

## Full Command Reference

| What | Command |
|---|---|
| Go to project | `cd ~/Desktop/samsung_prism_project` |
| Start emulator | see Part 1 |
| Start Python PRISM sidecar | `python scripts/openclaw_adapter/server.py` |
| Kill stuck `:8765` | `kill -9 $(lsof -t -i:8765)` |
| Set Claude key | `export ANTHROPIC_API_KEY=$(cat anthropic/api_key.txt)` |
| Run defended agent | `python scripts/agent_prism.py --task "..." --llm claude` |
| Run full MemShield mode | `PRISM_ENABLE_RETRIEVAL_DEFENSE=1 python scripts/agent_prism.py --task "..." --llm claude` |
| Run ProGRank too | `PRISM_ENABLE_RETRIEVAL_DEFENSE=1 PRISM_ENABLE_PROGRANK=1 python scripts/agent_prism.py --task "..." --llm claude` |
| Send poison notification | `bash scripts/send_poison_notification.sh` |
| Run MemShield demo | `cd memshield && PYTHONPATH=src:../scripts python demo_memshield.py` |
| Check Android sidecar | `curl http://127.0.0.1:8766/health` |

---

## Troubleshooting

### `adb` cannot find the emulator

```bash
adb kill-server
adb start-server
adb devices
```

### `:8765` already in use

```bash
kill -9 $(lsof -t -i:8765)
```

### `:8766` not responding

Reopen the merged Android app:

```bash
cd ~/Desktop/samsung_prism_project/android/openclaw-prism
adb shell am force-stop com.openclaw.android.debug
adb shell am start -n com.openclaw.android.debug/com.openclaw.android.MainActivity
adb forward tcp:8766 tcp:8766
```

### Chrome web content still not visible

- restart Chrome once
- make sure the accessibility service is enabled
- rerun the browsing task

### The agent loops or gets stuck

- rerun once
- prefer lightweight mode first
- use a concrete task with unique text

---

## Architecture Summary

```text
Android phone / emulator
    |
    |-- Screen (uiautomator2) -> unfiltered UI + prism_warning annotations
    |-- Web content (CDP)     -> WebContent element + web_tap/web_type
    |-- Notifications / clipboard / SMS / contacts -> Android sidecar :8766/v1/context
    |-- Shared storage        -> adb file reads
    |-- RAG                   -> MemShield + ChromaDB
    |
    v
Python PRISM sidecar (:8765)
    |-- Normalizer
    |-- Layer 1 heuristics
    |-- Layer 2 TinyBERT
    |-- Layer 3 DeBERTa
    v
Filtered device data
    |
    v
LLM agent
    |
    v
DefendedDevice
    |-- outgoing text checks
    |-- UI integrity checks via :8766/v1/ui-integrity
    |-- obvious-action fast path
    |-- loop detection
    |-- optional web_tap / web_type through CDP
    v
Android actions
```

---

*Samsung PRISM Work-let — Preventing Poisoning of Context in Mobile AI Agents*
