"""GGUF -> Hugging Face BF16 safetensors conversion.

This closes the one gap between how uncensored Qwen checkpoints are published
(GGUF only) and what NInfer's converter reads (HF-layout BF16 safetensors).  No
engine code is involved; see ``README.md`` in this directory for how to run it
and how to add another architecture.
"""

from __future__ import annotations

__all__ = ["common", "arches"]
