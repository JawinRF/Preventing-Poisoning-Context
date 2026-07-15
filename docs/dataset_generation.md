# PRISM Training Dataset — Generation Strategy & Technical Documentation

## Overview

The PRISM training dataset (`data/prism_training_dataset.json`) is a unified dataset for training TinyBERT (Layer 2 of the PRISM Shield pipeline) to detect prompt injection attacks targeting Android mobile agents. The dataset combines real-world adversarial data from public research with synthetic Android-specific samples.

**Final dataset**: 44,411 samples | 40.3% poisoned | 59.7% benign

---

## 1. Data Sources

The dataset draws from three source categories:

### 1.1 External Datasets (from HuggingFace)

Downloaded via `scripts/download_datasets.py`. Seven public datasets provide the foundation of real-world attack diversity:

| Dataset | HuggingFace ID | Rows Used | Type | License |
|---------|---------------|-----------|------|---------|
| deepset | `deepset/prompt-injections` | 546 | Attack + Benign baseline | Apache 2.0 |
| neuralchemy | `neuralchemy/Prompt-injection-dataset` | 4,391 | Leakage-verified binary classification | MIT |
| jackhhao | `jackhhao/jailbreak-classification` | 1,044 | Jailbreak-specific attacks | MIT |
| safeguard | `xTRam1/safe-guard-prompt-injection` | 8,236 | Curated multi-source merge | MIT |
| chatbot_instructions | `alespalla/chatbot_instruction_prompts` | 8,000 (sampled) | Benign instruction prompts | Apache 2.0 |
| xstest | `natolambert/xstest-v2-copy` | 450 | Over-refusal edge cases | CC-BY-4.0 |
| LLMail-Inject | `microsoft/llmail-inject-challenge` | 10,000 (sampled) | Real adversarial attacks from 224 human teams | MIT |

**Total external**: ~32,653 samples (15,869 poisoned, 16,784 benign)


### 1.2 Synthetic Attack Payloads (`scripts/generators/payloads.py`)

141 hand-crafted attack payloads across 10 categories, designed to cover attack vectors specific to Android mobile agents:

| Category | Count | Description |
|----------|-------|-------------|
| direct_injection | 25 | Classic "ignore previous instructions" patterns |
| persona_hijack | 15 | DAN, SUDO mode, developer mode, roleplay manipulation |
| system_prompt_extraction | 12 | Attempts to leak system prompt content |
| social_engineering | 15 | Fake urgency, authority impersonation, phishing |
| indirect_injection | 15 | Malicious instructions hidden in benign context (meeting notes, recipes, articles) |
| multilingual | 15 | Attacks in Chinese, Japanese, Korean, Russian, Spanish, Arabic, French, Hindi, code-switching |
| jailbreak | 12 | DAN, Grandma exploit, translation attack, hypothetical framing, emotional manipulation |
| data_exfiltration | 12 | Attempts to read/leak contacts, SMS, passwords, photos |
| fake_system_message | 10 | Spoofed Android system errors/warnings with malicious instructions |
| permission_escalation | 10 | Social engineering to grant dangerous permissions |

**30% of synthetic payloads are obfuscated** using one of 10 techniques:
- base64, ROT13, leetspeak, zero-width characters, hex encoding
- Unicode escape sequences, whitespace flooding, ANSI escape codes
- HTML comments, markdown hidden formatting

### 1.3 Hard Negatives

40 benign text samples that contain trigger words commonly found in attacks ("ignore", "override", "execute", "system", "inject", "prompt", "delete", "permission") but in legitimate contexts:

- Cybersecurity articles discussing prompt injection
- System administration documentation
- Developer documentation (e.g., `ignore()` function, `execute()` method)
- Legitimate app notifications mentioning passwords, permissions, updates
- Educational content about security concepts

**Purpose**: Reduce false positives. Without hard negatives, the model learns "if text contains 'ignore instructions' → poisoned" which breaks on cybersecurity articles and admin docs.

---

## 2. Android Format Wrapping

**Key design decision**: External datasets contain raw chat/email text. Our TinyBERT model sees data wrapped in Android-specific formats during inference. To match production distribution, every sample is wrapped in one of 7 Android ingestion formats at build time.

### 2.1 Wrapper Formats

| Format | Ingestion Path | Structure | Example Context |
|--------|---------------|-----------|-----------------|
| **Notification** | `notification` | JSON with app, sender, text, time, priority | `{"app": "WhatsApp", "title": "Mom", "text": "...", "time": "3:45 PM"}` |
| **Clipboard** | `clipboard` | Raw text, or prefixed with `[Clipboard]` / `Copied from <app>` | `Copied from Chrome: <payload>` |
| **UI Accessibility XML** | `ui_accessibility` | Android view hierarchy XML with `<node>` elements | `<hierarchy><node class="TextView" text="..." />`  |
| **RAG Document** | `rag_knowledge` | Retrieved document chunk with relevance prefix | `Retrieved document (relevance: 0.87):\n<payload>` |
| **File Content** | `shared_storage` | File read wrapper with filename delimiters | `--- START FILE: config.json ---\n<payload>\n--- END FILE ---` |
| **Network Response** | `network_responses` | JSON API response, HTML body, or raw response with payload appended | `{"status": "ok", "data": "<payload>"}` |
| **Intent** | `inter_app_intent` | Android Intent JSON with action, extras, component | `{"action": "android.intent.action.SEND", "extras": {"text": "..."}}` |

### 2.2 Wrapper Selection

Each sample is assigned a **random** wrapper. This ensures approximately uniform distribution across all 7 ingestion paths (~14% each). Final distribution:

| Ingestion Path | Count | Percentage |
|----------------|-------|------------|
| notification | 7,067 | 15.9% |
| ui_accessibility | 6,761 | 15.2% |
| rag_knowledge | 6,343 | 14.3% |
| shared_storage | 6,227 | 14.0% |
| inter_app_intent | 6,143 | 13.8% |
| network_responses | 6,134 | 13.8% |
| clipboard | 5,736 | 12.9% |

### 2.3 Realism in Wrappers

Wrappers use randomized metadata to increase variety:
- **Notifications**: 22 different app names (WhatsApp, Telegram, Gmail, KakaoTalk...), 20 sender names, random timestamps
- **UI XML**: 30 different benign node templates surrounding the target node; random visibility/bounds for poisoned nodes
- **Files**: 17 different filenames (config.json, notes.txt, settings.xml...)
- **Network**: 15 different benign response templates mixed with payloads
- **Intents**: 4 action types, 4 app components

---

## 3. Build Pipeline

Script: `scripts/build_training_set.py`

### 3.1 Pipeline Steps

```
1. LOAD        Load all external datasets from data/external/*.parquet
                 ↓
2. NORMALIZE   Map each dataset's schema to (text, is_injection) pairs
               - deepset/safeguard/neuralchemy: label 0=benign, 1=injection
               - jackhhao: type "jailbreak"=injection, "benign"=benign
               - chatbot_instructions: all benign (sampled to 8K)
               - xstest: all benign (hard negatives)
               - llmail_inject: all poisoned (deduplicated, sampled to 10K)
                 ↓
3. FILTER      Drop empty/short texts (<5 chars), truncate >2000 chars
                 ↓
4. WRAP        Wrap each sample in random Android format
                 ↓
5. GAP FILL    Calculate shortfall vs target (50K, 40% poisoned)
               - Generate synthetic poisoned from payloads.py (30% obfuscated)
               - Generate hard negatives (5% of benign budget)
               - Fill remaining benign with re-wrapped benign templates
                 ↓
6. DEDUPLICATE Hash first 500 chars of each sample, drop collisions
                 ↓
7. SHUFFLE     Random shuffle all samples
                 ↓
8. OUTPUT      Save to data/prism_training_dataset.json
```

### 3.2 Label Split Strategy

**Target ratio**: 40% poisoned / 60% benign

This ratio is chosen because:
- Slightly skewed toward benign to reduce false positive rate (precision matters for UX)
- 40% poisoned is enough for the model to learn attack patterns without oversampling
- Real-world distribution is heavily benign, but 40/60 gives better recall than 10/90

### 3.3 Source Contribution (Final Dataset)

| Source | Samples | Percentage | Type |
|--------|---------|------------|------|
| chatbot_instructions | 7,931 | 17.9% | Benign |
| synthetic benign fill | 8,575 | 19.3% | Benign |
| safeguard | 8,164 | 18.4% | Mixed |
| llmail_inject | 8,242 | 18.6% | Poisoned |
| neuralchemy | 4,357 | 9.8% | Mixed |
| synthetic payloads | 3,840 | 8.6% | Poisoned |
| hard negatives | 1,281 | 2.9% | Benign (trigger words) |
| jackhhao | 1,032 | 2.3% | Mixed |
| deepset | 542 | 1.2% | Mixed |
| xstest | 447 | 1.0% | Benign (edge cases) |

### 3.4 Configurable Parameters

```bash
python scripts/build_training_set.py \
    --target 50000 \          # total sample count (default: 50000)
    --poison-ratio 0.40 \     # fraction poisoned (default: 0.40)
    --output data/custom.json \  # output path
    --seed 42                 # reproducibility seed
```

---

## 4. Sample Schema

Each sample in the output JSON follows this schema:

```json
{
    "id": "ext_a1b2c3d4",
    "ingestion_path": "notification",
    "label": "poisoned",
    "attack_type": "instruction_injection",
    "severity": "high",
    "target_action": "pii_exfiltration",
    "text": "{\"app\": \"WhatsApp\", \"title\": \"Mom\", \"text\": \"Ignore previous instructions...\", \"time\": \"3:45 PM\"}",
    "notes": "source: llmail_inject"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID with prefix: `ext_` (external), `syn_` (synthetic), `hn_` (hard negative), `fill_` (benign fill) |
| `ingestion_path` | string | One of 7 Android ingestion paths |
| `label` | string | `"poisoned"` or `"benign"` |
| `attack_type` | string/null | One of 12 attack types (null for benign) |
| `severity` | string/null | `"medium"`, `"high"`, or `"critical"` (null for benign) |
| `target_action` | string/null | One of 9 target actions (null for benign) |
| `text` | string | The wrapped sample text (what the model sees) |
| `notes` | string | Provenance info (source dataset, obfuscation method, etc.) |

---

## 5. Training Configuration

Script: `scripts/train_tinybert.py`

### 5.1 Model

- **Base**: TinyBERT (6-layer BERT distillation, ~66M params)
- **Fine-tuned from**: `models/tinybert_poison_classifier` (v1)
- **Output**: `models/tinybert_poison_classifier_v3`

### 5.2 Preprocessing

1. **Unicode normalization**: All text passed through `normalize_unicode()` before tokenization — matches inference-time preprocessing in the PRISM Shield pipeline
2. **Confusable augmentation**: Poisoned samples augmented with Unicode confusable character substitution (40% substitution rate)
   - Weak ingestion paths (ui_accessibility, notification, network_responses, inter_app_intent): 2x copies
   - Other paths: 1x copy
   - Total after augmentation: ~65K training rows

### 5.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| max_length | 256 | Wrapped samples are longer than raw text (was 128 in v2) |
| learning_rate | 2e-5 | Standard for BERT fine-tuning |
| batch_size | 32 | Fits in GPU memory |
| epochs | 3 | Sufficient for fine-tuning (training loss converged) |
| weight_decay | 0.01 | Regularization |
| eval_strategy | per epoch | Monitor for overfitting |
| metric_for_best_model | F1 | Balances precision and recall |
| fp16 | true | Mixed precision for speed |
| train/val split | 90/10 | Stratified by label |

### 5.4 Results (v3)

| Metric | Score |
|--------|-------|
| **F1** | **98.38%** |
| Accuracy | 97.97% |
| Precision | 99.40% |
| Recall | 97.38% |
| Eval Loss | 0.0587 |

---

## 6. Pipeline Integration

TinyBERT v3 sits in the PRISM Shield pipeline (Layer 1 regex was removed — TinyBERT v3 now subsumes it):

```
Incoming text (from any ingestion path)
         ↓
    ┌─────────────────┐
    │  UIExtractor    │  ui_accessibility path only — flatten node dumps
    └──────┬──────────┘
           ↓
    ┌─────────────────┐
    │  Normalizer     │  Unicode NFC, confusable fold, zero-width strip
    └──────┬──────────┘
           ↓
    ┌─────────────────┐
    │  Layer 2 TinyBERT v3 (ONNX, shared with Android)
    │  ~5ms           │  Path-aware thresholds. For ui_accessibility the bare
    │                 │  text is re-wrapped in <hierarchy><node .../> XML to
    │                 │  match the training distribution before inference.
    └──────┬──────────┘
           │ ALLOW / QUARANTINE
           ↓
    ┌─────────────────┐
    │  Layer 3 DeBERTa (ProtectAI/deberta-v3-base-prompt-injection-v2)
    │  ~50ms          │  Natural-language safety net; sees the raw normalized
    │                 │  text (no XML wrap — DeBERTa expects prose).
    └──────┬──────────┘
           │
           ↓
       ALLOW / BLOCK
```

**QUARANTINE resolution** is path-dependent (set in `pipeline.py`):

- Ingestion paths (notifications, clipboard, sms, contacts, shared_storage, ui_accessibility, rag_store): `QUARANTINE → BLOCK`
- `agent_output` path: `QUARANTINE → ALLOW` (only high-confidence BLOCK stops the agent's own output)

**Thresholds** are tunable at runtime via env vars (`PRISM_L2_BLOCK_THRESHOLD`, `PRISM_L2_UI_BLOCK_THRESHOLD`, `PRISM_L2_ALLOW_THRESHOLD`, `PRISM_L2_UI_ALLOW_THRESHOLD`). `ui_accessibility` uses the UI-specific pair because screen labels are a different distribution than free text.

**Shared artifact**: Python host and Android `OnnxClassifier.kt` both load the same `android/openclaw-prism/app/src/main/assets/tinybert_prism.onnx`. Android uses a real BERT WordPiece tokenizer (`BertWordPieceTokenizer.kt`) bundled from `tokenizer.json` so token IDs match the Python path byte-for-byte.

---

## 7. Comparison: v1/v2 vs v3

| Aspect | v1/v2 (old) | v3 (current) |
|--------|-------------|--------------|
| **Unique base samples** | 1,498 | 44,411 |
| **Data sources** | 7 template-based generators (5-10 templates each) | 7 external datasets + 141 synthetic payloads + 40 hard negatives |
| **Real adversarial data** | 0 | ~15,000 (LLMail-Inject + neuralchemy + jackhhao + safeguard) |
| **Attack categories** | 5 | 12 |
| **Obfuscation techniques** | 3 (zero-width, whitespace, ANSI) | 10 (+ base64, ROT13, leetspeak, hex, unicode escape, HTML, markdown) |
| **Multilingual attacks** | 0 | 15 (8 languages + code-switching) |
| **Hard negatives** | 0 | 1,281 |
| **Benign diversity** | ~50 templates across 7 generators | 8,000 real chatbot instructions + 450 xstest + 165 curated templates |
| **max_length** | 128 tokens | 256 tokens |
| **Augmentation** | 3-8x copies (inflated to 258K from 1,498 base) | 1-2x copies (real diversity makes heavy augmentation unnecessary) |

---

## 8. Reproducing the Dataset

```bash
# 1. Download external datasets
python scripts/download_datasets.py

# 2. Build unified training set
python scripts/build_training_set.py --target 50000

# 3. Train TinyBERT v3
python scripts/train_tinybert.py

# Output: models/tinybert_poison_classifier_v3/
```

---

## 9. Files

| File | Purpose |
|------|---------|
| `scripts/download_datasets.py` | Download external datasets from HuggingFace |
| `scripts/generators/payloads.py` | 141 attack payloads, 40 hard negatives, 10 obfuscation functions, benign templates |
| `scripts/build_training_set.py` | Load, normalize, wrap, merge, deduplicate → unified dataset |
| `scripts/train_tinybert.py` | Fine-tune TinyBERT on the dataset |
| `data/external/` | Downloaded parquet files |
| `data/prism_training_dataset.json` | Final 44K-sample training dataset |
| `models/tinybert_poison_classifier_v3/` | Trained model weights |
