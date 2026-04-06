"""
unicode_defense.py (scripts/)
-----------------------------
Preprocessing hardening against Unicode confusable (homoglyph) attacks.
This copy lives under `scripts/` so it is importable when running:

    python scripts/your_script.py
"""

import re
import unicodedata
import random
from typing import List, Dict

# --- Confusables map (expanded coverage for high-impact attacks) ---
CONFUSABLES: Dict[str, str] = {
    # === Cyrillic lookalikes ===
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043E": "o",  # о
    "\u0440": "r",  # р
    "\u0441": "c",  # с
    "\u0445": "x",  # х
    "\u0443": "y",  # у
    "\u0456": "i",  # і
    "\u0458": "j",  # ј
    "\u04CF": "l",  # ӏ

    # === Greek lookalikes ===
    "\u03BF": "o",  # ο
    "\u03B1": "a",  # α
    "\u03B5": "e",  # ε
    "\u03C1": "p",  # ρ
    "\u03BD": "v",  # ν
    "\u03BA": "k",  # κ
    "\u03C5": "u",  # υ
    "\u0399": "I",  # Ι

    # === Latin Extended / IPA lookalikes ===
    "\u0261": "g",  # ɡ
    "\u0269": "i",  # ɩ
    "\u026A": "I",  # ɪ
    "\u0274": "n",  # ɴ
    "\u0280": "r",  # ʀ
    "\u0299": "b",  # ʙ
    "\u1D04": "c",  # ᴄ
    "\u1D07": "e",  # ᴇ
    "\u1D0A": "j",  # ᴊ
    "\u1D0B": "k",  # ᴋ
    "\u1D0D": "m",  # ᴍ
    "\u1D0F": "o",  # ᴏ
    "\u1D18": "p",  # ᴘ
    "\u1D1B": "t",  # ᴛ
    "\u1D1C": "u",  # ᴜ
    "\u1D20": "v",  # ᴠ
    "\u1D21": "w",  # ᴡ
    "\u1D22": "z",  # ᴢ

    # === Enclosed / circled letters ===
    **{chr(0x24B6 + i): chr(0x41 + i) for i in range(26)},  # Ⓐ-Ⓩ -> A-Z
    **{chr(0x24D0 + i): chr(0x61 + i) for i in range(26)},  # ⓐ-ⓩ -> a-z

    # === Fullwidth ASCII (U+FF01–U+FF5E) ===
    **{chr(0xFF01 + i): chr(0x21 + i) for i in range(94)},

    # === Superscript / subscript digits ===
    "\u2070": "0",
    "\u00B9": "1",
    "\u00B2": "2",
    "\u00B3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",

    # === Smart punctuation -> ASCII ===
    "\u2018": "'",
    "\u2019": "'",
    "\u201C": '"',
    "\u201D": '"',
    "\u2014": "-",
    "\u2013": "-",
    "\u2015": "-",
    "\u00AB": '"',
    "\u00BB": '"',
    "\u2039": "'",
    "\u203A": "'",

    # === Invisible / zero-width characters (strip) ===
    "\u00AD": "",
    "\u200B": "",
    "\u200C": "",
    "\u200D": "",
    "\u200E": "",
    "\u200F": "",
    "\uFEFF": "",
    "\u2060": "",
    "\u2061": "",
    "\u2062": "",
    "\u2063": "",
    "\u2064": "",
    "\u206A": "",
    "\u206B": "",
    "\u206C": "",
    "\u206D": "",
    "\u206E": "",
    "\u206F": "",
}


def _build_math_alphanum_map() -> Dict[str, str]:
    """Map Unicode Mathematical Alphanumeric Symbols to ASCII."""
    mapping: Dict[str, str] = {}

    style_offsets = [
        0x1D400,
        0x1D434,
        0x1D468,
        0x1D49C,
        0x1D4D0,
        0x1D504,
        0x1D538,
        0x1D56C,
        0x1D5A0,
        0x1D5D4,
        0x1D608,
        0x1D63C,
        0x1D670,
    ]
    for base in style_offsets:
        for i in range(26):
            upper_cp = base + i
            lower_cp = base + 26 + i
            mapping[chr(upper_cp)] = chr(0x41 + i)
            mapping[chr(lower_cp)] = chr(0x61 + i)

    digit_offsets = [0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6]
    for base in digit_offsets:
        for i in range(10):
            mapping[chr(base + i)] = str(i)

    return mapping


CONFUSABLES.update(_build_math_alphanum_map())


def normalize_unicode(text: str) -> str:
    """Idempotent Unicode hardening suitable for pre-tokenization."""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = "".join(CONFUSABLES.get(ch, ch) for ch in text)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7E]", "", text)
    text = unicodedata.normalize("NFC", text)
    return text


def normalize_batch(texts: List[str]) -> List[str]:
    return [normalize_unicode(t) for t in texts]


REVERSE_CONFUSABLES: Dict[str, List[str]] = {}
for uni_ch, ascii_ch in CONFUSABLES.items():
    if ascii_ch and len(ascii_ch) == 1:
        REVERSE_CONFUSABLES.setdefault(ascii_ch, []).append(uni_ch)


def confusable_augment(text: str, substitution_rate: float = 0.3) -> str:
    """
    Generate an adversarial variant by randomly replacing ASCII characters
    with Unicode confusables at the given rate.
    """
    if not isinstance(text, str):
        text = str(text)

    out_chars: List[str] = []
    for ch in text:
        if ch in REVERSE_CONFUSABLES and random.random() < substitution_rate:
            out_chars.append(random.choice(REVERSE_CONFUSABLES[ch]))
        else:
            out_chars.append(ch)
    return "".join(out_chars)

