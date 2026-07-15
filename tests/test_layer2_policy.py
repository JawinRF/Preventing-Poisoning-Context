from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


from prism_shield.layer2_local_llm import DEFAULT_ALLOW_THRESH, DEFAULT_BLOCK_THRESH


def test_default_allow_threshold_relaxes_borderline_scores() -> None:
    assert DEFAULT_ALLOW_THRESH >= 0.31


def test_block_threshold_above_allow_threshold() -> None:
    assert DEFAULT_BLOCK_THRESH > DEFAULT_ALLOW_THRESH


def test_no_path_specific_thresholds_remain() -> None:
    import prism_shield.layer2_local_llm as l2
    assert not hasattr(l2, "UI_ALLOW_THRESH")
    assert not hasattr(l2, "UI_BLOCK_THRESH")
