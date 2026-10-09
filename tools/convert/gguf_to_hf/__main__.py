"""Convert a text-model GGUF into a Hugging Face BF16 safetensors directory.

Canonical invocation::

    python3 -m tools.convert.gguf_to_hf \
      --gguf /path/to/model.gguf \
      --reference-dir /path/to/Qwen3.8-27B \
      --out /path/to/hf-bf16 \
      --vision-from-reference

The output directory is a complete Hugging Face model directory: the six
frontend files the NInfer converter hashes plus ``config.json``/``vocab.json``/
``merges.txt`` copied from ``--reference-dir``, BF16 shards, and a fresh
``model.safetensors.index.json`` covering exactly the official tensor names.

Tensor by tensor, never holding the model in RAM: ``token_embd`` alone is 5.1 GB
in float32.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open

from .arches import get_arch
from .common import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_SHARD_BYTES,
    INDEX_FILENAME,
    ShardWriter,
    copy_reference_files,
    dequantize_f32,
    dequantize_to_bf16,
    read_architecture,
    read_index,
    torch_shape,
)


def _nbytes(shape: tuple[int, ...], itemsize: int) -> int:
    return int(np.prod(shape, dtype=np.int64)) * itemsize


def _vision_shard(reference_dir: Path) -> Path:
    shards = sorted(reference_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {reference_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as handle:
            if any(name.startswith("model.visual.") for name in handle.keys()):
                return shard
    raise FileNotFoundError(f"no shard in {reference_dir} carries model.visual.* tensors")


def _write_vision(writer: ShardWriter, reference_dir: Path, expected: set[str]) -> None:
    shard = _vision_shard(reference_dir)
    with safe_open(str(shard), framework="pt") as handle:
        names = [name for name in handle.keys() if name.startswith("model.visual.")]
        missing = expected - set(names)
        if missing:
            raise ValueError(f"{shard.name} is missing {len(missing)} vision tensors")
        for name in names:
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)
            writer.reserve(_nbytes(tuple(tensor.shape), tensor.element_size()))
            writer.add(name, tensor.contiguous())
    print(f"vision: {len(names)} tensors copied from {shard.name}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gguf_to_hf",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gguf", required=True, type=Path, help="source GGUF")
    parser.add_argument(
        "--reference-dir",
        required=True,
        type=Path,
        help="pinned Hugging Face checkpoint: frontend files, index, vision weights",
    )
    parser.add_argument("--out", required=True, type=Path, help="output model directory")
    parser.add_argument("--arch", default=None, help="override general.architecture")
    parser.add_argument(
        "--vision-from-reference",
        action="store_true",
        help="copy model.visual.* from the reference shards instead of the mmproj GGUF",
    )
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--force", action="store_true", help="overwrite an existing output index")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="dev only: convert the first N GGUF tensors and skip the name check",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    reference_dir: Path = args.reference_dir
    index_path = out_dir / INDEX_FILENAME
    if index_path.exists() and not args.force:
        print(f"refusing to overwrite {index_path} (pass --force)", flush=True)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = read_index(reference_dir)
    if args.vision_from_reference:
        vision_expected = {n for n in expected if n.startswith("model.visual.")}

    reader = GGUFReader(str(args.gguf))
    arch_name = args.arch or read_architecture(reader)
    arch = get_arch(arch_name)
    print(
        f"gguf: {args.gguf}  arch={arch_name}  tensors={len(reader.tensors)}  "
        f"official names={len(expected)}",
        flush=True,
    )

    writer = ShardWriter(out_dir, max_bytes=args.shard_bytes)
    copied = copy_reference_files(reference_dir, out_dir)
    print(f"copied {len(copied)} reference files", flush=True)

    total = len(reader.tensors)
    if args.limit is not None:
        total = min(total, args.limit)
    started = time.monotonic()

    for position, tensor in enumerate(reader.tensors[:total], 1):
        hf_name = arch.gguf_to_hf_name(tensor.name)
        shape = torch_shape(tensor)
        if arch.is_identity(hf_name):
            # Streamed: the float32 form would be twice the BF16 we keep.
            writer.reserve(_nbytes(shape, 2))
            out = dequantize_to_bf16(tensor, args.chunk_bytes)
        else:
            # Transform needs the whole float32 tensor (shape may change too).
            writer.reserve(_nbytes(shape, 4))
            data = arch.invert(tensor.name, dequantize_f32(tensor))
            out = torch.from_numpy(np.ascontiguousarray(data)).to(torch.bfloat16)
            if tuple(data.shape) != tuple(out.shape):  # pragma: no cover - defensive
                raise ValueError(f"{hf_name}: shape changed at cast time")
            del data
        if tuple(out.shape) != tuple(shape) and not hf_name.endswith(".conv1d.weight"):
            # Every transform preserves the shape except conv1d, which grows back
            # its squeezed channel dimension.
            raise ValueError(f"{hf_name}: expected shape {shape}, got {tuple(out.shape)}")
        writer.add(hf_name, out)
        elapsed = time.monotonic() - started
        print(
            f"[{position}/{total}] {tensor.name} -> {hf_name} {tuple(out.shape)} "
            f"{elapsed:.1f}s",
            flush=True,
        )

    if args.vision_from_reference:
        _write_vision(writer, reference_dir, vision_expected)

    weight_map = writer.close()
    print(
        f"wrote {len(weight_map)} tensors in {writer.shard_count} shards, "
        f"{writer.total_bytes / 1e9:.2f} GB bf16 -> {out_dir / INDEX_FILENAME}",
        flush=True,
    )

    if args.limit is not None:
        print(f"PARTIAL run (--limit {args.limit}): name check skipped", flush=True)
        return 0

    produced, wanted = set(weight_map), set(expected)
    missing, extra = sorted(wanted - produced), sorted(produced - wanted)
    if missing or extra:
        print(f"FAIL name check: {len(missing)} missing, {len(extra)} extra", flush=True)
        for name in missing[:20]:
            print(f"  missing {name}", flush=True)
        for name in extra[:20]:
            print(f"  extra   {name}", flush=True)
        return 1
    print(f"name check: all {len(wanted)} official tensors present, none extra", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
