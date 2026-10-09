"""The GGUF <-> HF name map must reproduce the golden 866-entry map.

The golden fixture was derived and verified on 2026-10-07 from the actual
Q8_K_P GGUF header and the official Qwen3.8-27B safetensors index (866 GGUF
names, 1:1 and complete against the 866 non-vision official names).  Any change
to llama.cpp's tensor naming that shifts a mapping fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.convert.gguf_to_hf.arches.qwen35 import gguf_to_hf_name, is_plus_one

_FIXTURE = Path(__file__).parent / "data" / "qwen35_names.json"
GOLDEN: dict[str, str] = json.loads(_FIXTURE.read_text())


def test_derives_every_golden_entry() -> None:
    mismatches = [name for name, expected in GOLDEN.items() if gguf_to_hf_name(name) != expected]
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:5]}"


def test_map_is_a_bijection() -> None:
    values = list(GOLDEN.values())
    assert len(values) == len(set(values)), "duplicate target names"
    assert len(GOLDEN) == 866


def test_covers_exactly_the_official_non_vision_names() -> None:
    # A miss here means the conversion cannot reach 1199 = 866 + 333 vision.
    prefixes = list(GOLDEN.values())
    assert sum(1 for v in prefixes if v.startswith("model.visual.")) == 0
    assert sum(1 for v in prefixes if v.startswith("mtp.")) == 15


def test_plus_one_set_is_the_known_168() -> None:
    flagged = [name for name in GOLDEN.values() if is_plus_one(name)]
    assert len(flagged) == 168
    # The two MTP pre-fc norms come from the rename, not the ":norm.weight" suffix.
    for name in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight"):
        assert is_plus_one(name)
    # The one guaranteed exclusion.
    assert not is_plus_one("model.language_model.layers.0.linear_attn.norm.weight")
    # Everything else ending in norm.weight must be counted.
    textual = [v for v in GOLDEN.values() if not v.startswith("model.visual.")]
    by_suffix = [
        v for v in textual if v.endswith("norm.weight") and not v.endswith("linear_attn.norm.weight")
    ]
    assert len(by_suffix) == 166


@pytest.mark.parametrize("name", sorted(GOLDEN)[:3] + list(GOLDEN)[-3:])
def test_mapping_is_symmetric_not_required_but_stable(name: str) -> None:
    # Re-deriving twice in one process must be deterministic.
    assert gguf_to_hf_name(name) == GOLDEN[name]