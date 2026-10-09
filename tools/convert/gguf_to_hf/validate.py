"""Gate 3 and gate 4 checks for a GGUF -> HF conversion.

* Gate 3 (structure): the produced index carries exactly the official tensor
  names; every tensor matches the official shape and is BF16.
* Gate 4 (values): cosine similarity and max-abs-diff against the official
  reference tensors.  Cosine alone cannot catch a missed ``+1`` on the norm
  tensors (near-constant vectors stay near 1.0), so gate 4 also compares the
  shifted ``ours + 1`` against the reference and fails any norm where the
  shifted version clearly fits better -- that is the missed ``-1`` bug.

Canonical invocation::

    python3 -m tools.convert.gguf_to_hf.validate \
      --out /path/to/hf-bf16 --reference-dir /path/to/Qwen3.8-27B
"""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open

from .arches.qwen35 import is_plus_one
from .common import INDEX_FILENAME, read_index, reference_shapes

CHUNK = 4_000_000  # elements; keeps the f32 working set ~32 MB


def _load(name: str, model_dir: Path, index: dict[str, str]) -> torch.Tensor:
    shard = model_dir / index[name]
    with safe_open(str(shard), framework="pt") as handle:
        return handle.get_tensor(name)


def _min_max_cosine(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """(min, max, cosine) over chunked float32 reductions of two BF16 tensors."""
    n = a.numel()
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    min_abs = float("inf")
    max_abs = 0.0
    for start in range(0, n, CHUNK):
        x = flat_a[start : start + CHUNK].to(torch.float32)
        y = flat_b[start : start + CHUNK].to(torch.float32)
        d = (x - y).abs()
        min_abs = min(min_abs, d.min().item())
        max_abs = max(max_abs, d.max().item())
        dot += (x * y).sum().item()
        norm_a += (x * x).sum().item()
        norm_b += (y * y).sum().item()
    cosine = dot / sqrt(norm_a * norm_b)
    return min_abs, max_abs, cosine


def check_gate3(out: Path, reference_dir: Path) -> int:
    official = read_index(reference_dir)
    produced = read_index(out)
    failures: list[str] = []

    wanted, got = set(official), set(produced)
    if got != wanted:
        failures.append(
            f"names: produced {len(got)}, official {len(wanted)} "
            f"({len(wanted - got)} missing, {len(got - wanted)} extra)"
        )

    ref_shapes = reference_shapes(reference_dir)
    out_shapes = reference_shapes(out)
    checked = matched = bf16 = 0
    for name in sorted(produced):
        if name not in out_shapes:
            failures.append(f"{name}: produced tensor missing from its own shards")
            continue
        shape, dtype = out_shapes[name]
        if dtype != "BF16":
            failures.append(f"{name}: dtype {dtype}, expected BF16")
        else:
            bf16 += 1
        if name in ref_shapes:
            checked += 1
            ref_shape, ref_dtype = ref_shapes[name]
            if shape != ref_shape:
                failures.append(f"{name}: shape {shape} != official {ref_shape}")
            else:
                matched += 1

    unchecked = set(wanted) - set(ref_shapes)
    if unchecked:
        print(f"warn: {len(unchecked)} official tensors not shape-checked "
              f"(reference shards incomplete): {sorted(unchecked)[:5]} ...", flush=True)
    print(
        f"gate3 shapes: {matched}/{checked} checked tensors match, "
        f"{bf16}/{len(produced)} are BF16",
        flush=True,
    )
    if failures:
        for line in failures[:20]:
            print(f"FAIL gate3: {line}", flush=True)
        return 1
    print("gate3: PASS", flush=True)
    return 0


def check_gate4(out: Path, reference_dir: Path, min_cosine: float, limit: int | None) -> int:
    official = read_index(reference_dir)
    produced = read_index(out)
    failures: list[str] = []
    compared = low_cosine = 0
    norm_checked = norm_flagged = 0
    worst_cosine = 1.0
    worst_name = ""

    names = [n for n in produced if not n.startswith("model.visual.")]
    if limit is not None:
        names = names[:limit]
    vision_names = [n for n in produced if n.startswith("model.visual.")]

    for name in names:
        if name not in official:
            failures.append(f"{name}: no official counterpart")
            continue
        ours = _load(name, out, produced)
        ref = _load(name, reference_dir, official)
        if ours.shape != ref.shape:
            failures.append(f"{name}: shape {tuple(ours.shape)} != official {tuple(ref.shape)}")
            continue
        min_abs, max_abs, cosine = _min_max_cosine(ours, ref)
        compared += 1
        if cosine < worst_cosine:
            worst_cosine, worst_name = cosine, name
        flag = ""
        if cosine < min_cosine:
            low_cosine += 1
        if is_plus_one(name):
            norm_checked += 1
            shifted = (ours.to(torch.float32) + 1.0).to(torch.bfloat16)
            _, shifted_max, _ = _min_max_cosine(shifted, ref)
            if shifted_max < max_abs:
                flag = "  <-- ours+1 fits better: missed -1?"
                norm_flagged += 1
                failures.append(f"{name}: missed +1? max_abs {max_abs:.4g} vs shifted {shifted_max:.4g}")
        print(
            f"4a {name} cos={cosine:.6f} maxdiff={max_abs:.4g} mindiff={min_abs:.4g}{flag}",
            flush=True,
        )
        del ours, ref

    if vision_names:
        identical = 0
        for name in vision_names:
            ours = _load(name, out, produced)
            ref = _load(name, reference_dir, official)
            if torch.equal(ours, ref):
                identical += 1
            else:
                failures.append(f"4b {name}: not bit-identical to reference")
        print(f"4b vision: {identical}/{len(vision_names)} bit-identical (copied) tensors", flush=True)

    print(
        f"gate4a: compared {compared} text tensors, {low_cosine} below cosine {min_cosine}, "
        f"worst {worst_name} cos={worst_cosine:.6f}; "
        f"norms {norm_checked} checked, {norm_flagged} missed-+1 flags",
        flush=True,
    )
    if failures or low_cosine or norm_flagged:
        for line in failures[:20]:
            print(f"FAIL gate4: {line}", flush=True)
        return 1
    print("gate4: PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gguf_to_hf.validate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--gate", choices=("3", "4", "all"), default="all")
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--limit", type=int, default=None, help="gate 4 dev shortcut")
    args = parser.parse_args(argv)

    failures = 0
    if args.gate in ("3", "all"):
        failures += check_gate3(args.out, args.reference_dir)
    if args.gate in ("4", "all"):
        failures += check_gate4(args.out, args.reference_dir, args.min_cosine, args.limit)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())