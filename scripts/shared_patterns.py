"""
shared_patterns.py — Single source of truth for injection/suspicious regex patterns.
Used by both MemShield (memshield/config.py) and PRISM Layer 1 (prism_shield/layer1_heuristics.py).

When adding new attack patterns, add them HERE. Both systems pick them up automatically.
"""
import re

# High-confidence injection patterns → BLOCK
INJECTION_PATTERNS = [
    # Standard prompt injection keywords
    re.compile(r"(?i)\bignore\s+(all\s+)?previous\s+(instructions|context|prompts)\b"),
    re.compile(r"(?i)\bforget\s+(everything|all|prior)\b"),
    re.compile(r"(?i)\bnew\s+(instruction|directive|objective)\s*[:;]"),
    re.compile(r"(?i)\bsystem\s*:\s*override\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(a|an)\b"),
    re.compile(r"(?i)\bdo\s+not\s+(follow|obey|respect)\b"),
    re.compile(r"(?i)\bdisregard\s+(the\s+)?(previous|prior|original)\b"),
    re.compile(r"(?i)\byou\s+must\s+now\s+act\s+as\b"),
    re.compile(r"(?i)\bnew\s+task\s*:"),
    # Exfiltration
    re.compile(r"(?i)(exfil|send|upload|transfer).{0,40}(attacker|evil|malicious)"),
    # Jailbreak/DAN keywords
    re.compile(r"(?i)\b(jailbreak|dan\b|prompt\s*injection)\b"),
    # Structural injection markers
    re.compile(r"(?i)\[\s*(inject|override|system|admin)\s*\]"),
    re.compile(r"(?i)\[AGENT\s*INSTRUCTION:.*?\]"),
    re.compile(r"(?i)\[AUTO-AGENT.*?\]"),
    re.compile(r"(?i)<system_override>"),
    re.compile(r"(?i)<hidden_instruction>"),
    # UIExtractor structural flags
    re.compile(r"(?i)\[HIDDEN_UI_ELEMENT\]"),
    re.compile(r"(?i)\[CONTEXT_MISMATCH.*?\]"),
]

# Medium-confidence suspicious patterns → QUARANTINE
#
# These must NOT match normal Android UI labels. A button labelled
# "Export Contacts" is not an attack — an *instruction* telling the
# agent to silently export contacts IS. Keep patterns specific to
# injected directives, not feature names.
SUSPICIOUS_PATTERNS = [
    # Role-play / restriction bypass
    re.compile(r"(?i)\bpretend\s+(you\s+are|to\s+be)\b"),
    re.compile(r"(?i)\bwithout\s+(restrictions|limits|filters)\b"),
    re.compile(r"(?i)\bin\s+developer\s+mode\b"),
    # Privilege escalation commands (shell-level, not UI labels)
    re.compile(r"(?i)\badb\s+shell\s+pm\s+grant\b"),
    re.compile(r"(?i)chmod\s+777"),
]
