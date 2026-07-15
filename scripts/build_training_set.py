#!/usr/bin/env python3
"""
Build unified PRISM training dataset from external HuggingFace datasets +
our synthetic payloads. Each sample is wrapped in a random Android ingestion
format (notification, clipboard, UI XML, RAG doc, file, network, intent).

Usage:
    python scripts/build_training_set.py                          # default 50K
    python scripts/build_training_set.py --target 20000           # smaller
    python scripts/build_training_set.py --output data/custom.json
"""

import argparse
import hashlib
import json
import random
import uuid
import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pandas as pd

# ── Add generators to path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generators.rag_generator import RAGGenerator
from generators.intent_generator import IntentGenerator
from generators.clipboard_generator import ClipboardGenerator
from generators.ui_generator import UIGenerator
from generators.notification_generator import NotificationGenerator
from generators.storage_generator import StorageGenerator
from generators.network_generator import NetworkGenerator
from generators.payloads import (
    ALL_ATTACK_PAYLOADS,
    PAYLOAD_CATEGORIES,
    HARD_NEGATIVES,
    BENIGN_NOTIFICATION_TEXTS,
    BENIGN_CLIPBOARD_TEXTS,
    BENIGN_RAG_DOCUMENTS,
    BENIGN_FILE_CONTENTS,
    BENIGN_NETWORK_RESPONSES,
    BENIGN_UI_NODES,
    BENIGN_INTENT_DATA,
    random_obfuscation,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXT_DIR = DATA_DIR / "external"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANDROID FORMAT WRAPPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

APPS = [
    "WhatsApp", "Telegram", "Instagram", "Gmail", "Slack", "Teams",
    "Signal", "Messenger", "SMS", "Chrome", "Samsung Notes", "KakaoTalk",
    "Line", "WeChat", "Viber", "Discord", "Twitter/X", "LinkedIn",
    "Snapchat", "TikTok", "Reddit", "YouTube",
]

SENDERS = [
    "Mom", "Dad", "Boss", "Sarah", "Alex", "John", "HR Department",
    "IT Support", "Mike", "Dr. Smith", "Amazon", "Bank Alert",
    "Package Delivery", "Calendar", "System", "Security Team",
    "Jane", "Work Group", "Family Chat", "Neighbour",
]

FILE_NAMES = [
    "config.json", "notes.txt", "settings.xml", "data.csv", "readme.md",
    "backup.json", "preferences.txt", "cache.dat", "log.txt", "todo.txt",
    "contacts.vcf", "schedule.ics", "response.json", "page.html",
    "report.pdf.txt", "shared_doc.txt", "meeting_notes.md",
]


def wrap_notification(text: str) -> tuple[str, str]:
    """Wrap text as an Android notification."""
    app = random.choice(APPS)
    sender = random.choice(SENDERS)
    hour = random.randint(1, 12)
    minute = random.randint(0, 59)
    ampm = random.choice(["AM", "PM"])
    notif = json.dumps({
        "app": app,
        "title": sender,
        "text": text,
        "time": f"{hour}:{minute:02d} {ampm}",
        "priority": random.choice(["default", "high", "low"]),
    })
    return notif, "notifications"


def wrap_clipboard(text: str) -> tuple[str, str]:
    """Wrap text as clipboard content."""
    style = random.choice(["raw", "prefixed", "with_source"])
    if style == "raw":
        return text, "clipboard"
    elif style == "prefixed":
        return f"[Clipboard] {text}", "clipboard"
    else:
        app = random.choice(APPS + ["Browser", "PDF Viewer", "Notes"])
        return f"Copied from {app}: {text}", "clipboard"


def wrap_ui_xml(text: str) -> tuple[str, str]:
    """Wrap text as a UI accessibility node."""
    classes = [
        "android.widget.TextView", "android.widget.Button",
        "android.widget.EditText", "android.widget.ImageView",
    ]
    resource_ids = [
        "com.example:id/text_view", "com.example:id/button1",
        "com.example:id/message_text", "com.example:id/content",
        "com.example:id/notification_text", "com.example:id/title",
    ]
    cls = random.choice(classes)
    rid = random.choice(resource_ids)

    # Some benign nodes around it
    benign_count = random.randint(2, 4)
    benign_nodes_xml = ""
    for bn in random.choices(BENIGN_UI_NODES, k=benign_count):
        b_cls = bn["class"]
        b_text = quoteattr(bn["text"])
        b_desc = quoteattr(bn["content-desc"])
        benign_nodes_xml += f'  <node index="0" class="{b_cls}" text={b_text} content-desc={b_desc} clickable="true" visible-to-user="true" bounds="[0,0][100,100]" />\n'

    # The target node (could be hidden for poisoned)
    visible = random.choice(["true", "false"])
    bounds = "[0,0][1,1]" if visible == "false" else "[0,0][100,100]"

    xml = f'<?xml version="1.0" encoding="UTF-8"?>\n<hierarchy rotation="0">\n'
    xml += benign_nodes_xml
    xml += f'  <node index="0" class="{cls}" text={quoteattr(text)} content-desc="" resource-id="{rid}" clickable="false" visible-to-user="{visible}" bounds="{bounds}" />\n'
    xml += "</hierarchy>"
    return xml, "ui_accessibility"


def wrap_rag_document(text: str) -> tuple[str, str]:
    """Wrap text as a RAG-retrieved document chunk."""
    prefixes = [
        "Retrieved document (relevance: 0.87):\n",
        "Knowledge base result:\n",
        "FAQ match:\n",
        "Support article excerpt:\n",
        "User manual section:\n",
        "Wiki page fragment:\n",
    ]
    suffixes = [
        "\n\n[End of retrieved content]",
        "\n\n---",
        "",
        "\n\n[Source: internal knowledge base]",
    ]
    return f"{random.choice(prefixes)}{text}{random.choice(suffixes)}", "rag_store"


def wrap_file_content(text: str) -> tuple[str, str]:
    """Wrap text as file content read from storage."""
    fname = random.choice(FILE_NAMES)
    return f"--- START FILE: {fname} ---\n{text}\n--- END FILE ---", "shared_storage"


def wrap_network_response(text: str) -> tuple[str, str]:
    """Wrap text as a network API/web response."""
    style = random.choice(["json_field", "html_embed", "raw_append", "prose_then_json"])
    if style == "prose_then_json":
        base = random.choice([
            "Welcome to our site Here is some content.",
            "Request completed. See response payload below.",
            "Page loaded successfully. Embedded data follows.",
        ])
        return (
            f'{base} {{"status": "success", "data": {json.dumps(text)}}}',
            "network_responses",
        )
    if style == "json_field":
        base = random.choice(BENIGN_NETWORK_RESPONSES)
        if base.startswith("{"):
            try:
                data = json.loads(base)
                data["_content"] = text
                return json.dumps(data), "network_responses"
            except json.JSONDecodeError:
                pass
        return f'{{"status": "ok", "data": {json.dumps(text)}}}', "network_responses"
    elif style == "html_embed":
        return f"<html><body><p>{text}</p></body></html>", "network_responses"
    else:
        base = random.choice(BENIGN_NETWORK_RESPONSES)
        return f"{base}\n\n{text}", "network_responses"


def wrap_intent(text: str) -> tuple[str, str]:
    """Wrap text as Android intent extras."""
    actions = [
        "android.intent.action.SEND", "android.intent.action.VIEW",
        "android.intent.action.SENDTO", "android.intent.action.PROCESS_TEXT",
    ]
    intent = json.dumps({
        "action": random.choice(actions),
        "data": None,
        "extras": {"text": text},
        "component": f"com.{random.choice(['whatsapp', 'chrome', 'gmail', 'sms'])}.app/.MainActivity",
    })
    return intent, "android_intents"


WRAPPERS = [
    wrap_notification,
    wrap_clipboard,
    wrap_ui_xml,
    wrap_rag_document,
    wrap_file_content,
    wrap_network_response,
    wrap_intent,
]


def wrap_random(text: str) -> tuple[str, str]:
    """Apply a random Android format wrapper. Returns (wrapped_text, ingestion_path)."""
    return random.choice(WRAPPERS)(text)


_SYNTH_GENERATORS = [
    RAGGenerator(),
    IntentGenerator(),
    ClipboardGenerator(),
    UIGenerator(),
    NotificationGenerator(),
    StorageGenerator(),
    NetworkGenerator(),
]


def make_id(prefix: str = "ext") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATASET LOADERS — normalize each to (text, is_injection) pairs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATTACK_TYPES = [
    "instruction_injection", "role_override", "context_flooding",
    "obfuscated_payload", "encoded_payload", "persona_hijack",
    "indirect_injection", "social_engineering", "multi_lingual",
    "jailbreak", "exfiltration", "fake_system_message",
]

TARGET_ACTIONS = [
    "pii_exfiltration", "permission_escalation", "unauthorized_transaction",
    "delete_data", "modify_settings", "phishing_redirect",
    "system_prompt_leak", "credential_theft", "app_installation",
]


def load_deepset() -> list[tuple[str, bool]]:
    """deepset/prompt-injections: text + label (0=benign, 1=injection)"""
    path = EXT_DIR / "deepset" / "deepset.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["text"], bool(row["label"])) for _, row in df.iterrows()]


def load_neuralchemy() -> list[tuple[str, bool]]:
    """neuralchemy: text + label (0=benign, 1=injection)"""
    path = EXT_DIR / "neuralchemy" / "neuralchemy.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["text"], bool(row["label"])) for _, row in df.iterrows()]


def load_jackhhao() -> list[tuple[str, bool]]:
    """jackhhao: prompt + type (jailbreak/benign)"""
    path = EXT_DIR / "jackhhao" / "jackhhao.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["prompt"], row["type"] == "jailbreak") for _, row in df.iterrows()]


def load_safeguard() -> list[tuple[str, bool]]:
    """safeguard: text + label (0=benign, 1=injection)"""
    path = EXT_DIR / "safeguard" / "safeguard.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["text"], bool(row["label"])) for _, row in df.iterrows()]


def load_chatbot_instructions(max_samples: int = 8000) -> list[tuple[str, bool]]:
    """chatbot_instructions: all benign prompts. Sample subset."""
    path = EXT_DIR / "chatbot_instructions" / "chatbot_instructions.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return [(row["prompt"], False) for _, row in df.iterrows()]


def load_xstest() -> list[tuple[str, bool]]:
    """xstest: prompts that look dangerous but are benign (hard negatives)."""
    path = EXT_DIR / "xstest" / "xstest_prompts.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["prompt"], False) for _, row in df.iterrows()]


def eval_holdout(text: str) -> bool:
    """Deterministic 20% eval partition for single-split corpora (SMS ham).

    Shared with build_external_benchmark.py: rows where this returns True are
    reserved for the external benchmark and MUST NOT enter training.
    """
    digest = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:8]
    return int(digest, 16) / 0xFFFFFFFF >= 0.80


def load_gandalf_train() -> list[tuple[str, bool]]:
    """Lakera gandalf train split — real injection attempts. Val/test are eval-only."""
    path = EXT_DIR / "eval_gandalf" / "eval_gandalf_train.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return [(row["text"], True) for _, row in df.iterrows()]


def load_sms_ham_train() -> list[tuple[str, bool]]:
    """UCI SMS ham (benign) — 80% train partition; 20% reserved for eval."""
    path = EXT_DIR / "eval_sms_ham" / "eval_sms_ham.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    ham = df[df["label"] == 0]["sms"].astype(str)
    return [(t, False) for t in ham if not eval_holdout(t)]


def load_ag_news_train(max_samples: int = 4000) -> list[tuple[str, bool]]:
    """AG News train split (benign news texture). Test split is eval-only."""
    path = EXT_DIR / "ag_news_train" / "ag_news_train.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sample(frac=1.0, random_state=13).head(max_samples)
    return [(row["text"], False) for _, row in df.iterrows()]


def load_banking77_train(max_samples: int = 3000) -> list[tuple[str, bool]]:
    """Banking77 train split (benign user utterances). Test split is eval-only."""
    path = EXT_DIR / "banking77_train" / "banking77_train.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path).sample(frac=1.0, random_state=13).head(max_samples)
    return [(row["text"], False) for _, row in df.iterrows()]


def load_llmail_inject(max_samples: int = 10000) -> list[tuple[str, bool]]:
    """LLMail-Inject: 370K real adversarial email injection attacks. All poisoned.
    Sample a subset since the full set is huge. Use 'body' column as the payload."""
    path = EXT_DIR / "llmail_inject" / "llmail_inject.parquet"
    if not path.exists():
        return []
    df = pd.read_parquet(path, columns=["body"])
    # Deduplicate by body text (many teams submit similar attacks)
    df = df.drop_duplicates(subset=["body"])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return [(row["body"], True) for _, row in df.iterrows() if row["body"] and len(str(row["body"]).strip()) > 10]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYNTHETIC PAYLOAD GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_synthetic_poisoned(count: int) -> list[dict]:
    """Generate poisoned samples from our curated payloads in payloads.py."""
    samples = []
    for _ in range(count):
        payload = random.choice(ALL_ATTACK_PAYLOADS)

        # 30% chance of obfuscation
        if random.random() < 0.3:
            payload, obf_method = random_obfuscation(payload)
            notes_extra = f", obfuscation: {obf_method}"
        else:
            notes_extra = ""

        wrapped, ingestion_path = wrap_random(payload)

        cat = "unknown"
        for cat_name, cat_payloads in PAYLOAD_CATEGORIES.items():
            if payload in cat_payloads:
                cat = cat_name
                break

        samples.append({
            "id": make_id("syn"),
            "ingestion_path": ingestion_path,
            "label": "poisoned",
            "attack_type": random.choice(ATTACK_TYPES),
            "severity": random.choice(["medium", "high", "critical"]),
            "target_action": random.choice(TARGET_ACTIONS),
            "text": wrapped,
            "notes": f"synthetic payload (category: {cat}{notes_extra})",
        })
    return samples


def generate_synthetic_hard_negatives(count: int) -> list[dict]:
    """Generate benign samples that contain trigger words."""
    samples = []
    for _ in range(count):
        text = random.choice(HARD_NEGATIVES)
        wrapped, ingestion_path = wrap_random(text)
        samples.append({
            "id": make_id("hn"),
            "ingestion_path": ingestion_path,
            "label": "benign",
            "attack_type": None,
            "severity": None,
            "target_action": None,
            "text": wrapped,
            "notes": "hard negative (benign with trigger words)",
        })
    return samples


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN BUILD PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def convert_external(pairs: list[tuple[str, bool]], source_name: str) -> list[dict]:
    """Convert (text, is_injection) pairs to our schema with Android wrapping."""
    samples = []
    for text, is_injection in pairs:
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            continue

        text = text.strip()
        # Truncate very long texts (some external samples are huge)
        if len(text) > 2000:
            text = text[:2000]

        wrapped, ingestion_path = wrap_random(text)

        if is_injection:
            samples.append({
                "id": make_id("ext"),
                "ingestion_path": ingestion_path,
                "label": "poisoned",
                "attack_type": random.choice(ATTACK_TYPES),
                "severity": random.choice(["medium", "high", "critical"]),
                "target_action": random.choice(TARGET_ACTIONS),
                "text": wrapped,
                "notes": f"source: {source_name}",
            })
        else:
            samples.append({
                "id": make_id("ext"),
                "ingestion_path": ingestion_path,
                "label": "benign",
                "attack_type": None,
                "severity": None,
                "target_action": None,
                "text": wrapped,
                "notes": f"source: {source_name}",
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Build unified PRISM training dataset")
    parser.add_argument("--target", type=int, default=50000, help="Target total samples")
    parser.add_argument("--output", type=str, default="data/prism_training_dataset.json", help="Output path")
    parser.add_argument("--poison-ratio", type=float, default=0.40, help="Target poisoned ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading external datasets...")

    loaders = [
        ("deepset", load_deepset),
        ("neuralchemy", load_neuralchemy),
        ("jackhhao", load_jackhhao),
        ("safeguard", load_safeguard),
        ("chatbot_instructions", load_chatbot_instructions),
        ("xstest", load_xstest),
        ("llmail_inject", load_llmail_inject),
        ("gandalf_train", load_gandalf_train),
        ("sms_ham_train", load_sms_ham_train),
        ("ag_news_train", load_ag_news_train),
        ("banking77_train", load_banking77_train),
    ]

    all_external: list[dict] = []
    for name, loader in loaders:
        pairs = loader()
        if not pairs:
            print(f"  SKIP  {name:25s} — not found")
            continue
        converted = convert_external(pairs, name)
        n_inj = sum(1 for s in converted if s["label"] == "poisoned")
        n_ben = sum(1 for s in converted if s["label"] == "benign")
        print(f"  OK    {name:25s} → {len(converted):>6} samples ({n_inj} poisoned, {n_ben} benign)")
        all_external.extend(converted)

    ext_poisoned = [s for s in all_external if s["label"] == "poisoned"]
    ext_benign = [s for s in all_external if s["label"] == "benign"]
    print(f"\nExternal total: {len(all_external)} ({len(ext_poisoned)} poisoned, {len(ext_benign)} benign)")

    # ── Calculate how much synthetic data we need ───────────────────────────
    target_poisoned = int(args.target * args.poison_ratio)
    target_benign = args.target - target_poisoned

    need_more_poisoned = max(0, target_poisoned - len(ext_poisoned))
    need_more_benign = max(0, target_benign - len(ext_benign))

    print(f"\nTarget: {args.target} total ({target_poisoned} poisoned, {target_benign} benign)")
    print(f"Need synthetic: {need_more_poisoned} poisoned, {need_more_benign} benign")

    # ── Generate synthetic to fill the gap ──────────────────────────────────
    print("\nGenerating synthetic poisoned samples...")
    syn_poisoned = generate_synthetic_poisoned(need_more_poisoned)
    print(f"  Generated {len(syn_poisoned)} synthetic poisoned")

    # Hard negatives: ~5% of benign budget
    hard_neg_count = min(int(target_benign * 0.05), need_more_benign)
    remaining_benign = need_more_benign - hard_neg_count

    print("Generating hard negatives...")
    hard_negatives = generate_synthetic_hard_negatives(hard_neg_count)
    print(f"  Generated {len(hard_negatives)} hard negatives")

    # Fill remaining benign with re-wrapped external benign (different formats)
    print("Filling remaining benign with re-wrapped external...")
    extra_benign: list[dict] = []
    if remaining_benign > 0 and ext_benign:
        for _ in range(remaining_benign):
            src = random.choice(ext_benign)
            # Re-wrap the original text in a different format
            # Extract original text from the wrapped version — use a benign template instead
            roll = random.random()
            if roll < 0.30:
                gen = random.choice(_SYNTH_GENERATORS)
                sample = gen.generate_benign(1)[0]
                wrapped, ingestion_path = sample["text"], sample["ingestion_path"]
            elif roll < 0.40:
                intent = dict(random.choice(BENIGN_INTENT_DATA))
                pkg = random.choice([
                    "com.whatsapp.app", "com.chrome.app", "com.gmail.app",
                    "com.google.android.browser", "com.android.dialer",
                    "com.google.android.apps.maps", "com.android.contacts",
                ])
                intent["component"] = f"{pkg}/.MainActivity"
                wrapped, ingestion_path = json.dumps(intent), "android_intents"
            elif roll < 0.55:
                nodes_xml = ""
                for bn in random.choices(BENIGN_UI_NODES, k=random.randint(3, 8)):
                    nodes_xml += (
                        f'  <node index="0" class="{bn["class"]}" '
                        f'text={quoteattr(bn["text"])} '
                        f'content-desc={quoteattr(bn["content-desc"])} '
                        'clickable="true" visible-to-user="true" bounds="[0,0][100,100]" />\n'
                    )
                wrapped = (
                    '<?xml version="1.0" encoding="UTF-8"?>\n<hierarchy rotation="0">\n'
                    f'{nodes_xml}</hierarchy>'
                )
                ingestion_path = "ui_accessibility"
            else:
                benign_text = random.choice(
                    BENIGN_NOTIFICATION_TEXTS + BENIGN_CLIPBOARD_TEXTS +
                    BENIGN_RAG_DOCUMENTS +
                    [c for _, c in BENIGN_FILE_CONTENTS] +
                    BENIGN_NETWORK_RESPONSES
                )
                wrapped, ingestion_path = wrap_random(benign_text)
            extra_benign.append({
                "id": make_id("fill"),
                "ingestion_path": ingestion_path,
                "label": "benign",
                "attack_type": None,
                "severity": None,
                "target_action": None,
                "text": wrapped,
                "notes": "synthetic benign fill",
            })
    print(f"  Generated {len(extra_benign)} benign fill samples")

    # ── Combine everything ──────────────────────────────────────────────────
    all_samples = ext_poisoned + ext_benign + syn_poisoned + hard_negatives + extra_benign

    # Trim to target if we overshot
    if len(all_samples) > args.target:
        random.shuffle(all_samples)
        all_samples = all_samples[:args.target]

    random.shuffle(all_samples)

    # ── Deduplicate by text ─────────────────────────────────────────────────
    seen_texts: set[str] = set()
    deduped: list[dict] = []
    for s in all_samples:
        text_hash = hash(s["text"][:500])  # hash first 500 chars
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            deduped.append(s)
    dropped = len(all_samples) - len(deduped)
    if dropped:
        print(f"\nDeduplication dropped {dropped} samples")
    all_samples = deduped

    # ── Stats ───────────────────────────────────────────────────────────────
    n_poisoned = sum(1 for s in all_samples if s["label"] == "poisoned")
    n_benign = sum(1 for s in all_samples if s["label"] == "benign")

    paths = {}
    for s in all_samples:
        p = s["ingestion_path"]
        paths[p] = paths.get(p, 0) + 1

    sources = {}
    for s in all_samples:
        src = s["notes"].split("source: ")[-1] if "source:" in s["notes"] else s["notes"].split("(")[0].strip()
        sources[src] = sources.get(src, 0) + 1

    # ── Save ────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Saved to {output_path}")
    print(f"Total: {len(all_samples)}")
    print(f"Poisoned: {n_poisoned} ({n_poisoned/len(all_samples)*100:.1f}%)")
    print(f"Benign:   {n_benign} ({n_benign/len(all_samples)*100:.1f}%)")
    print(f"\nBy ingestion path:")
    for p, c in sorted(paths.items(), key=lambda x: -x[1]):
        print(f"  {p:25s} {c:>6}")
    print(f"\nBy source:")
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s:40s} {c:>6}")


if __name__ == "__main__":
    main()
