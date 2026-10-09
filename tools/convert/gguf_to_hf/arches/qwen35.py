"""GGUF arch ``qwen35`` (Qwen3.5 / Qwen3.8 dense) -> Hugging Face layout.

llama.cpp converts HF -> GGUF with two passes; this module inverts them.

``_LinearAttentionVReorderBase.modify_tensors`` runs first (see
``conversion/qwen.py``): it transposes V heads from grouped order -- the order HF
stores them in -- into tiled order so ggml's broadcast matches ``ggml_repeat``.
With ``num_k_heads=16`` and ``num_v_heads=48`` that permutation is **not** its own
inverse, so :func:`_v_inverse_perm` computes it explicitly.

``Qwen3NextModel.modify_tensors`` runs second and applies four more changes:

``A_log``      stored as ``-exp(A_log)``                 -> inverse ``log(-x)``
``dt_bias``    renamed ``dt_proj.bias`` -> ``ssm_dt.bias``  -> name only
``conv1d``     squeezed ``[10240, 1, 4]`` -> ``[10240, 4]`` -> restore with unsqueeze
``*norm.weight`` stored as ``weight + 1``                 -> inverse ``-1``

Because the inner pass runs *after* the reorder, the inverse undoes the value
changes first, then the reorder, then the shape squeeze.

The ``+1`` rule is subtler than it looks. llama.cpp's ``filter_tensors`` renames
tensors *before* ``modify_tensors``, so ``model.language_model.norm.weight``
becomes ``model.norm.weight`` and the MTP ``mtp.pre_fc_norm_embedding/hidden``
become ``model.layers.64.enorm/hnorm`` -- all of which end in ``norm.weight`` and
therefore all get the ``+1``. ``linear_attn.norm.weight`` is the one exclusion.
That is 168 tensors; :data:`PLUS_ONE_COUNT` guards against the count drifting.
"""

from __future__ import annotations

import re

import numpy as np


ARCH = "qwen35"

#: Verified from the Q8_K_P GGUF header (2026-10-07) and Qwen's config.json.
NUM_K_HEADS = 16
NUM_V_HEADS = 48
NUM_V_PER_K = NUM_V_HEADS // NUM_K_HEADS  # 3
HEAD_K_DIM = 128
HEAD_V_DIM = 128

#: ``q`` + ``k`` rows of ``in_proj_qkv`` / leading channels of ``conv1d``.
QKV_CHANNELS = HEAD_K_DIM * NUM_K_HEADS * 2  # 4096
#: ``v`` rows of ``in_proj_qkv`` and the length of ``in_proj_z`` / ``out_proj`` cols.
V_ROWS = NUM_V_HEADS * HEAD_V_DIM  # 6144

#: The two MTP norms that get ``+1`` through a rename rather than a suffix.
_MTP_PRE_FC_NORMS = frozenset(
    {
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
    }
)

PLUS_ONE_COUNT = 168

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_TOP_LEVEL = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}

_MTP_NEXTPN = {
    "nextn.eh_proj.weight": "mtp.fc.weight",
    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
}

_MLP = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_LINEAR_ATTN = {
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

_SELF_ATTN = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}

_LAYER_SUFFIXES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
}


def gguf_to_hf_name(gguf_name: str) -> str:
    """Map a GGUF tensor name to the official Hugging Face name."""
    if gguf_name in _TOP_LEVEL:
        return _TOP_LEVEL[gguf_name]

    match = _BLOCK_RE.match(gguf_name)
    if match is None:
        raise ValueError(f"unmapped GGUF tensor {gguf_name!r}")
    block, suffix = int(match.group(1)), match.group(2)
    if block < 0 or block > 64:
        raise ValueError(f"unexpected block index in {gguf_name!r}")

    if block == 64:
        # The native NextN MTP head llama.cpp stores as blk.64.
        if suffix in _MTP_NEXTPN:
            return _MTP_NEXTPN[suffix]
        prefix = "mtp.layers.0."
    else:
        prefix = f"model.language_model.layers.{block}."

    if suffix in _LAYER_SUFFIXES:
        return prefix + _LAYER_SUFFIXES[suffix]
    if suffix in _MLP:
        return prefix + _MLP[suffix]
    if suffix in _LINEAR_ATTN:
        if block == 64:
            raise ValueError(f"blk.64 has no linear attention: {gguf_name!r}")
        return prefix + _LINEAR_ATTN[suffix]
    if suffix in _SELF_ATTN:
        return prefix + _SELF_ATTN[suffix]
    raise ValueError(f"unmapped GGUF tensor {gguf_name!r}")


def is_plus_one(hf_name: str) -> bool:
    """True when llama.cpp stored this tensor as ``weight + 1``."""
    if hf_name in _MTP_PRE_FC_NORMS:
        return True
    return hf_name.endswith("norm.weight") and not hf_name.endswith("linear_attn.norm.weight")


def needs_v_unreorder(hf_name: str) -> bool:
    """True when llama.cpp applied the grouped -> tiled V head reorder."""
    return ".linear_attn." in hf_name and not hf_name.endswith("linear_attn.norm.weight")


def is_identity(hf_name: str) -> bool:
    """True when llama.cpp only renamed the tensor (no value or shape change)."""
    if is_plus_one(hf_name):
        return False
    if hf_name.endswith(".A_log") or hf_name.endswith(".conv1d.weight"):
        return False
    return not needs_v_unreorder(hf_name)


def _v_inverse_perm(length: int, head_dim: int) -> np.ndarray:
    """Inverse of ``_reorder_v_heads`` over one axis of ``length`` elements."""
    expected = NUM_K_HEADS * NUM_V_PER_K * head_dim
    if length != expected:
        raise ValueError(f"V axis is {length} elements, expected {expected} (head_dim={head_dim})")
    index = np.arange(length, dtype=np.int64).reshape(NUM_K_HEADS, NUM_V_PER_K, head_dim)
    # llama.cpp: reshape (k, vpk, d) -> transpose the first two axes -> reshape back,
    # i.e. out[i] = in[forward[i]].
    forward = np.transpose(index, (1, 0, 2)).reshape(length)
    inverse = np.empty(length, dtype=np.int64)
    inverse[forward] = np.arange(length, dtype=np.int64)
    return inverse


def _undo_v_reorder(hf_name: str, data: np.ndarray) -> np.ndarray:
    if hf_name.endswith(".in_proj_qkv.weight"):
        # Only the V rows are reordered; Q and K pass through untouched.
        qk = QKV_CHANNELS
        if data.shape[0] != qk + V_ROWS:
            raise ValueError(f"{hf_name}: unexpected shape {data.shape}")
        out = np.array(data, copy=True)
        out[qk:] = np.take(data[qk:], _v_inverse_perm(V_ROWS, HEAD_V_DIM), axis=0)
        return out

    if hf_name.endswith(".conv1d.weight"):
        # Same split, still in the squeezed [channels, kernel] layout.
        qk = QKV_CHANNELS
        if data.shape[0] != qk + V_ROWS:
            raise ValueError(f"{hf_name}: unexpected shape {data.shape}")
        out = np.array(data, copy=True)
        out[qk:] = np.take(data[qk:], _v_inverse_perm(V_ROWS, HEAD_V_DIM), axis=0)
        return out

    if hf_name.endswith((".in_proj_z.weight",)):
        return np.take(data, _v_inverse_perm(V_ROWS, HEAD_V_DIM), axis=0)

    if hf_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
        return np.take(data, _v_inverse_perm(NUM_V_HEADS, 1), axis=0)

    if hf_name.endswith(".A_log") or hf_name.endswith(".dt_bias"):
        vector = np.take(data.reshape(-1), _v_inverse_perm(NUM_V_HEADS, 1), axis=0)
        return vector.reshape(data.shape)

    if hf_name.endswith(".out_proj.weight"):
        # Columns, not rows: out_proj consumes the V activation dimension.
        if data.shape[1] != V_ROWS:
            raise ValueError(f"{hf_name}: unexpected shape {data.shape}")
        return np.take(data, _v_inverse_perm(V_ROWS, HEAD_V_DIM), axis=1)

    raise ValueError(f"{hf_name}: nothing to un-reorder")


def invert(gguf_name: str, data: np.ndarray) -> np.ndarray:
    """Turn a dequantized GGUF tensor into the Hugging Face tensor.

    ``data`` must already be float32 in torch order (see
    ``common.dequantize_f32``).  The array is never modified in place.
    """
    hf_name = gguf_to_hf_name(gguf_name)
    if is_identity(hf_name):
        return data

    out = data
    if hf_name.endswith(".A_log"):
        # forward stored -exp(x); log(-y) is exact for any finite negative y.
        out = np.log(-np.asarray(out, dtype=np.float32))
    elif is_plus_one(hf_name):
        out = np.asarray(out, dtype=np.float32) - 1.0

    if needs_v_unreorder(hf_name):
        out = _undo_v_reorder(hf_name, out)

    if hf_name.endswith(".conv1d.weight"):
        # Restore the singleton in-channel llama.cpp squeezed away.
        if out.ndim != 2 or out.shape[1] != 4:
            raise ValueError(f"{hf_name}: unexpected shape {out.shape}")
        out = out[:, np.newaxis, :]
    return out
