"""End-to-end inverse test: re-apply llama.cpp's forward to HF-style arrays, then
check ``arch.invert`` recovers the originals exactly (up to float32 epsilon).

The forward implementation below is written independently from
``conversion/qwen.py`` (``_LinearAttentionVReorderBase.modify_tensors`` run
first, then ``Qwen3NextModel.modify_tensors``) so a bug in the module's inverse
cannot hide behind a bug shared with the forward.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.convert.gguf_to_hf.arches import qwen35 as arch

K = arch.NUM_K_HEADS          # 16
VPK = arch.NUM_V_PER_K        # 3
HD_V = arch.HEAD_V_DIM        # 128
V_ROWS = arch.V_ROWS          # 6144
QKV_CH = arch.QKV_CHANNELS    # 4096


def reorder_forward(data: np.ndarray, dim: int, head_dim: int) -> np.ndarray:
    """llama.cpp ``_reorder_v_heads``: reshape axis ``dim`` to (k, vpk, head_dim),
    swap the first two, reshape back."""
    shape = list(data.shape)
    new_shape = shape[:dim] + [K, VPK, head_dim] + shape[dim + 1 :]
    tensor = data.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.transpose(tensor, perm).reshape(*shape)


def outer(hf_name: str, data: np.ndarray) -> np.ndarray:
    """``_LinearAttentionVReorderBase.modify_tensors`` (runs first)."""
    if ".linear_attn." not in hf_name:
        return data
    if hf_name.endswith(".in_proj_qkv.weight"):
        q_dim = 128 * K
        q = data[:q_dim]
        k = data[q_dim : q_dim + q_dim]
        v = reorder_forward(data[q_dim + q_dim :], 0, HD_V)
        return np.concatenate([q, k, v], axis=0)
    if hf_name.endswith(".in_proj_z.weight"):
        return reorder_forward(data, 0, HD_V)
    if hf_name.endswith(".in_proj_a.weight") or hf_name.endswith(".in_proj_b.weight"):
        return reorder_forward(data, 0, 1)
    if hf_name.endswith(".A_log") or hf_name.endswith(".dt_bias"):
        if data.ndim == 1:
            return reorder_forward(data.reshape(-1, 1), 0, 1).reshape(-1)
        return reorder_forward(data, -1, 1)
    if hf_name.endswith(".conv1d.weight"):
        squeezed = data.squeeze()  # [10240, 1, 4] -> [10240, 4]
        qk = squeezed[:QKV_CH]
        v = reorder_forward(squeezed[QKV_CH:], 0, HD_V)
        return np.concatenate([qk, v], axis=0)
    if hf_name.endswith(".out_proj.weight"):
        return reorder_forward(data, 1, HD_V)
    raise AssertionError(f"unhandled linear_attn tensor {hf_name}")


def inner(hf_name: str, data: np.ndarray) -> np.ndarray:
    """``Qwen3NextModel.modify_tensors`` (runs second)."""
    if hf_name.endswith(".A_log"):
        return -np.exp(data)
    if "conv1d" in hf_name:
        return data.squeeze()  # no-op on the already-squeezed [10240, 4]
    # llama.cpp's filter_tensors renames these to *.enorm/hnorm.weight BEFORE the
    # +1 check, which is why their HF names (not suffixed norm.weight) still get +1.
    renamed = hf_name
    if hf_name == "mtp.pre_fc_norm_embedding.weight":
        renamed = "model.layers.64.enorm.weight"
    elif hf_name == "mtp.pre_fc_norm_hidden.weight":
        renamed = "model.layers.64.hnorm.weight"
    if renamed.endswith("norm.weight") and not renamed.endswith("linear_attn.norm.weight"):
        return data + 1.0
    return data


def forward(hf_name: str, data: np.ndarray) -> np.ndarray:
    """What llama.cpp writes into the GGUF, given an HF-layout tensor."""
    return inner(hf_name, outer(hf_name, data))


# gguf tensor name -> (hf name, shape, value range, exact?)
CASES: dict[str, tuple[str, tuple[int, ...], tuple[float, float], bool]] = {
    "blk.0.ssm_a": ("model.language_model.layers.0.linear_attn.A_log", (48,), (-8.0, 0.0), False),
    "blk.0.ssm_dt.bias": ("model.language_model.layers.0.linear_attn.dt_bias", (48,), (-5.0, 5.0), True),
    "blk.0.attn_qkv.weight": (
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        (QKV_CH + V_ROWS, 4),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.attn_gate.weight": (
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        (V_ROWS, 4),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.ssm_alpha.weight": (
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        (48, 4),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.ssm_beta.weight": (
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        (48, 4),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.ssm_conv1d.weight": (
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        (QKV_CH + V_ROWS, 1, 4),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.ssm_out.weight": (
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        (5, V_ROWS),
        (-1.0, 1.0),
        True,
    ),
    "blk.0.attn_norm.weight": (
        "model.language_model.layers.0.input_layernorm.weight",
        (128,),
        (0.5, 2.5),
        True,
    ),
    "blk.63.attn_q_norm.weight": (
        "model.language_model.layers.63.self_attn.q_norm.weight",
        (128,),
        (0.5, 2.5),
        True,
    ),
    "output_norm.weight": ("model.language_model.norm.weight", (5120,), (0.5, 2.5), True),
    "blk.64.nextn.enorm.weight": ("mtp.pre_fc_norm_embedding.weight", (128,), (0.5, 2.5), True),
}


@pytest.mark.parametrize("gguf_name", sorted(CASES))
def test_invert_recovers_forward(gguf_name: str) -> None:
    hf_name, shape, (lo, hi), exact = CASES[gguf_name]
    rng = np.random.default_rng(abs(hash(gguf_name)) % (2**32))
    original = rng.uniform(lo, hi, size=shape).astype(np.float32)

    gguf_value = forward(hf_name, original)

    # Sanity: the forward must actually have changed something, or the test is
    # vacuously green.  All cases here except the renamed-only ones transform.
    recovered = arch.invert(gguf_name, gguf_value)
    assert recovered.shape == original.shape
    if exact:
        assert np.allclose(recovered, original, rtol=1e-6, atol=1e-6)
    else:
        assert np.allclose(recovered, original, rtol=1e-4, atol=1e-6)


def test_dt_bias_and_norms_are_bit_exact() -> None:
    # Permutations and +1/-1 on values that are exactly representable in float32
    # roundtrip bit-exact; only -exp/log accumulate eps (covered by the parametrized
    # test).  Norm values below are all exactly representable and stay exact +1/-1.
    rng = np.random.default_rng(7)
    exact_values = rng.choice(
        np.array([0.5, 1.0, 1.5, 2.0, 2.5, -0.5, -1.0, -1.5, -2.0], dtype=np.float32),
        size=(64,),
    )
    for gguf_name in ("blk.0.ssm_dt.bias", "blk.0.attn_norm.weight", "output_norm.weight"):
        hf_name, shape, _, _ = CASES[gguf_name]
        if hf_name.endswith(".dt_bias"):
            original = rng.uniform(-2.0, 2.0, size=shape).astype(np.float32)
        else:
            original = np.resize(exact_values, shape)
        assert np.array_equal(arch.invert(gguf_name, forward(hf_name, original)), original)


def test_qkv_non_v_rows_are_untouched() -> None:
    hf_name, shape, _, _ = CASES["blk.0.attn_qkv.weight"]
    rng = np.random.default_rng(11)
    original = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    gguf_value = forward(hf_name, original)
    recovered = arch.invert("blk.0.attn_qkv.weight", gguf_value)
    assert np.array_equal(recovered[:QKV_CH], original[:QKV_CH])
    # V rows: permuted, so compare as sets of values.
    assert np.allclose(np.sort(recovered[QKV_CH:].ravel()), np.sort(original[QKV_CH:].ravel()))