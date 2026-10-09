"""ShardWriter: HF-style sharding, index metadata, flush and rename behaviour."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from tools.convert.gguf_to_hf.common import ShardWriter


def test_single_shard_roundtrip(tmp_path) -> None:
    writer = ShardWriter(tmp_path)
    a = torch.full((4, 5), 1.0, dtype=torch.bfloat16)
    b = torch.full((6,), 2.0, dtype=torch.bfloat16)
    writer.add("a", a)
    writer.add("b", b)
    weight_map = writer.close()

    assert weight_map == {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"}
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == (4 * 5 + 6) * 2
    assert not (tmp_path / "model-00001.safetensors").exists()  # provisional renamed

    with safe_open(str(tmp_path / "model-00001-of-00001.safetensors"), framework="pt") as handle:
        assert torch.equal(handle.get_tensor("a"), a)
        assert torch.equal(handle.get_tensor("b"), b)


def test_flush_spills_to_a_second_shard(tmp_path) -> None:
    writer = ShardWriter(tmp_path, max_bytes=45)
    a = torch.full((4, 5), 1.0, dtype=torch.bfloat16)  # 40 bytes
    b = torch.full((6,), 2.0, dtype=torch.bfloat16)  # 12 bytes
    writer.add("a", a)
    writer.add("b", b)  # 40 + 12 > 45 -> a spills to shard 1, b goes to shard 2
    weight_map = writer.close()

    assert weight_map["a"] == "model-00001-of-00002.safetensors"
    assert weight_map["b"] == "model-00002-of-00002.safetensors"
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == weight_map
    assert writer.shard_count == 2
    assert (tmp_path / "model-00001-of-00002.safetensors").exists()
    assert (tmp_path / "model-00002-of-00002.safetensors").exists()


def test_duplicate_name_raises(tmp_path) -> None:
    writer = ShardWriter(tmp_path)
    writer.add("a", torch.zeros(2, dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        writer.add("a", torch.zeros(2, dtype=torch.bfloat16))


def test_wrong_dtype_raises(tmp_path) -> None:
    writer = ShardWriter(tmp_path)
    with pytest.raises(TypeError):
        writer.add("a", torch.zeros(2))  # float32


def test_tensor_larger_than_shard_target_raises(tmp_path) -> None:
    writer = ShardWriter(tmp_path, max_bytes=8)
    with pytest.raises(ValueError):
        writer.add("a", torch.zeros(8, dtype=torch.bfloat16))  # 16 bytes > 8


def test_reserve_flushes_before_a_large_tensor(tmp_path) -> None:
    writer = ShardWriter(tmp_path, max_bytes=64)
    small = torch.zeros(10, dtype=torch.bfloat16)  # 20 bytes
    big = torch.zeros(30, dtype=torch.bfloat16)  # 60 bytes
    writer.add("small", small)
    writer.reserve(60)  # 20 + 60 > 64 -> flushes now
    writer.add("big", big)
    weight_map = writer.close()

    assert weight_map["small"] == "model-00001-of-00002.safetensors"
    assert weight_map["big"] == "model-00002-of-00002.safetensors"