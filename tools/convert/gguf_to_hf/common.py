"""GGUF reading, dequantization and BF16 safetensors sharding.

These helpers are arch agnostic. Everything arch specific (the name map and the
inverse of llama.cpp's transforms) lives under ``arches/``.

Conventions used throughout:

* GGUF stores tensor dimensions reversed with respect to torch.  ``gguf.GGUFReader``
  already gives us ``ReaderTensor.data`` in torch order, but ``ReaderTensor.shape``
  is still GGUF order -- always go through :func:`torch_shape`.
* Everything is done in float32 and cast to BF16 exactly once, at the point the
  tensor is handed to the shard writer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from gguf import GGUFReader, ReaderTensor
from gguf.quants import dequantize
from safetensors.torch import save_file


#: Files the stock NInfer converter hashes; must be copied from the reference dir.
FRONTEND_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

#: Copied verbatim alongside the frontend files so the directory is a complete
#: Hugging Face model dir.
EXTRA_REFERENCE_FILES = (
    "config.json",
    "vocab.json",
    "merges.txt",
)

#: Default target shard size.  Kept below the 4 GB gate so a shard plus the tensor
#: being materialized next stays comfortably inside this box's available RAM.
DEFAULT_SHARD_BYTES = 3_000_000_000

#: Largest float32 working chunk.  ``token_embd`` is 5.1 GB in float32, so tensors
#: are dequantized in slices instead of all at once.
DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024

INDEX_FILENAME = "model.safetensors.index.json"


def torch_shape(tensor: ReaderTensor) -> tuple[int, ...]:
    """Logical tensor shape in torch/HF order."""
    return tuple(int(d) for d in reversed(tensor.shape.tolist()))


def read_architecture(reader: GGUFReader) -> str:
    field = reader.get_field("general.architecture")
    if field is None:
        raise ValueError("GGUF has no general.architecture")
    return str(field.contents())


def _fresh_f32(array: np.ndarray) -> np.ndarray:
    """Return a writable, C-contiguous float32 copy when one is needed."""
    if array.dtype == np.float32 and array.flags.c_contiguous and array.flags.writeable:
        return array
    return np.array(array, dtype=np.float32, copy=True, order="C")


def dequantize_f32(tensor: ReaderTensor) -> np.ndarray:
    """Dequantize a whole tensor to float32 in torch order.

    Only use this for tensors whose float32 form fits comfortably in RAM; the big
    ones go through :func:`iter_f32_chunks`.
    """
    return _fresh_f32(dequantize(tensor.data, tensor.tensor_type))


def iter_f32_chunks(
    tensor: ReaderTensor,
    max_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Iterator[tuple[tuple[slice, ...], np.ndarray]]:
    """Yield ``(index, float32_chunk)`` slices covering ``tensor`` along axis 0.

    Slicing only axis 0 keeps the last axis intact, which is what gguf-py's
    blockwise dequantization needs.
    """
    data = tensor.data
    rows = int(data.shape[0])
    if rows <= 0:
        raise ValueError(f"{tensor.name}: empty tensor")
    row_bytes = int(data.nbytes) // rows
    per_row = max(1, int(max_chunk_bytes) // max(row_bytes, 1))
    trailing = (slice(None),) * (data.ndim - 1)
    for start in range(0, rows, per_row):
        stop = min(start + per_row, rows)
        index = (slice(start, stop),) + trailing
        yield index, _fresh_f32(dequantize(data[index], tensor.tensor_type))


def dequantize_to_bf16(
    tensor: ReaderTensor,
    max_chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> torch.Tensor:
    """Dequantize a tensor straight to a BF16 torch tensor, in bounded chunks."""
    shape = torch_shape(tensor)
    out = torch.empty(shape, dtype=torch.bfloat16)
    for index, chunk in iter_f32_chunks(tensor, max_chunk_bytes):
        out[index] = torch.from_numpy(chunk).to(torch.bfloat16)
    return out


def read_index(model_dir: Path | str) -> dict[str, str]:
    """Return ``name -> shard filename`` from a Hugging Face safetensors index."""
    path = Path(model_dir) / INDEX_FILENAME
    return dict(json.loads(path.read_text())["weight_map"])


def copy_reference_files(reference_dir: Path | str, out_dir: Path | str) -> list[str]:
    """Copy the frontend + config files the converter expects to find flat in the dir."""
    reference_dir = Path(reference_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in FRONTEND_FILES + EXTRA_REFERENCE_FILES:
        src = reference_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"reference dir is missing {name}: {src}")
        (out_dir / name).write_bytes(src.read_bytes())
        copied.append(name)
    return copied


class ShardWriter:
    """Accumulate BF16 tensors and spill them to HF-style safetensors shards."""

    def __init__(
        self,
        out_dir: Path | str,
        max_bytes: int = DEFAULT_SHARD_BYTES,
        prefix: str = "model",
    ) -> None:
        self.out_dir = Path(out_dir)
        self.max_bytes = int(max_bytes)
        self.prefix = prefix
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._sizes: dict[str, int] = {}
        self._shards: list[dict[str, object]] = []
        self._total_bytes = 0

    @property
    def names(self) -> list[str]:
        return [name for shard in self._shards for name in shard["names"]]  # type: ignore[union-attr]

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def shard_count(self) -> int:
        return len(self._shards) + (1 if self._buffer else 0)

    def reserve(self, nbytes: int) -> None:
        """Flush first when ``nbytes`` would not fit, so peak RAM stays bounded."""
        if self._buffer and self._buffer_bytes + nbytes > self.max_bytes:
            self.flush()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name}: expected BF16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise TypeError(f"{name}: tensor must be contiguous")
        if name in self._sizes or name in self._buffer:
            raise ValueError(f"{name}: duplicate tensor")
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self.max_bytes:
            raise ValueError(f"{name}: {nbytes} bytes exceeds the {self.max_bytes} shard target")
        self.reserve(nbytes)
        self._buffer[name] = tensor
        self._buffer_bytes += nbytes
        self._sizes[name] = nbytes
        self._total_bytes += nbytes

    def flush(self) -> None:
        if not self._buffer:
            return
        self._shards.append(
            {
                "index": len(self._shards) + 1,
                "provisional": f"{self.prefix}-{len(self._shards) + 1:05d}.safetensors",
                "names": list(self._buffer),
            }
        )
        path = self.out_dir / str(self._shards[-1]["provisional"])
        save_file(self._buffer, str(path), metadata={"format": "pt"})
        self._buffer = {}
        self._buffer_bytes = 0

    def close(self) -> dict[str, str]:
        """Flush, rename shards to the HF convention and return ``name -> shard``."""
        self.flush()
        total = len(self._shards)
        final = f"{{prefix}}-{{index:05d}}-of-{total:05d}.safetensors"
        weight_map: dict[str, str] = {}
        for shard in self._shards:
            fname = final.format(prefix=self.prefix, index=shard["index"])
            if fname != shard["provisional"]:
                (self.out_dir / str(shard["provisional"])).rename(self.out_dir / fname)
            for name in shard["names"]:  # type: ignore[union-attr]
                weight_map[name] = fname
        index = {
            "metadata": {"total_size": self._total_bytes},
            "weight_map": weight_map,
        }
        (self.out_dir / INDEX_FILENAME).write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        return weight_map


def reference_shapes(reference_dir: Path | str) -> dict[str, tuple[tuple[int, ...], str]]:
    """Read ``name -> (shape, dtype)`` from every safetensors shard in ``reference_dir``.

    Headers only: no tensor data is touched.
    """
    from safetensors import safe_open

    reference_dir = Path(reference_dir)
    shapes: dict[str, tuple[tuple[int, ...], str]] = {}
    for shard in sorted(reference_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for name in handle.keys():
                slice_ = handle.get_slice(name)
                shapes[name] = (tuple(slice_.get_shape()), slice_.get_dtype())
    return shapes
