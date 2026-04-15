from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from prism_shield.layer2_local_llm import DEFAULT_ALLOW_THRESH, UI_ALLOW_THRESH


def test_default_allow_threshold_relaxes_borderline_scores() -> None:
    assert DEFAULT_ALLOW_THRESH >= 0.31


def test_ui_allow_threshold_relaxes_borderline_scores() -> None:
    assert UI_ALLOW_THRESH >= 0.31
