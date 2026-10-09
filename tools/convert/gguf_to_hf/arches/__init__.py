"""Architecture registry: ``general.architecture`` -> inverse module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from . import qwen35


@runtime_checkable
class ArchModule(Protocol):
    """What :mod:`tools.convert.gguf_to_hf.__main__` needs from an architecture."""

    ARCH: str

    def gguf_to_hf_name(self, gguf_name: str) -> str:
        """Official Hugging Face name for a GGUF tensor name."""
        ...

    def invert(self, gguf_name: str, data: np.ndarray) -> np.ndarray:
        """Inverse of llama.cpp's transform, on a float32 tensor in torch order."""
        ...

    def is_identity(self, hf_name: str) -> bool:
        """True when llama.cpp only renamed the tensor (safe to stream in chunks)."""
        ...


#: Keys are the ``general.architecture`` value written into the GGUF.
REGISTRY: dict[str, ArchModule] = {
    qwen35.ARCH: qwen35,
}


def get_arch(name: str) -> ArchModule:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<none>"
        raise ValueError(f"unsupported GGUF architecture {name!r} (known: {known})") from None
