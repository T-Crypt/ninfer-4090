"""The inverse of llama.cpp's ``_reorder_v_heads`` must be a true inverse.

``_reorder_v_heads`` is a transpose of the (num_k_heads, num_v_per_k) block, and
a transpose is only an involution when the two axes have equal length -- with
16 K heads and 3 V heads per K it is **not**, which is why the inverse
permutation is computed explicitly instead of reused.  These tests pin that.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.convert.gguf_to_hf.arches.qwen35 import (
    HEAD_V_DIM,
    NUM_K_HEADS,
    NUM_V_PER_K,
    _v_inverse_perm,
)


def forward_perm(length: int, head_dim: int) -> np.ndarray:
    """llama.cpp's ``_reorder_v_heads`` as a flat index permutation.

    out_flat[i] = in_flat[forward[i]]  (forward is applied with ``np.take``).
    """
    index = np.arange(length).reshape(NUM_K_HEADS, NUM_V_PER_K, head_dim)
    return np.transpose(index, (1, 0, 2)).reshape(length)


@pytest.mark.parametrize("n,head_dim", [(6144, 128), (48, 1)])
def test_forward_then_inverse_is_identity(n: int, head_dim: int) -> None:
    arr = np.arange(n * 3, dtype=np.float64).reshape(n, 3) % 11
    fwd = forward_perm(n, head_dim)
    inv = _v_inverse_perm(n, head_dim)
    assert np.array_equal(np.take(np.take(arr, fwd, axis=0), inv, axis=0), arr)
    assert np.array_equal(np.take(np.take(arr, inv, axis=0), fwd, axis=0), arr)


@pytest.mark.parametrize("n,head_dim", [(6144, 128), (48, 1)])
def test_permutation_is_not_an_involution(n: int, head_dim: int) -> None:
    fwd = forward_perm(n, head_dim)
    inv = _v_inverse_perm(n, head_dim)
    assert not np.array_equal(fwd, inv)


def test_columns_axis_uses_the_same_index() -> None:
    # out_proj reorders the *input* (column) axis; verify the index applies the
    # same way on axis 1.
    n = 6144
    arr = np.arange(5 * n, dtype=np.float64).reshape(5, n) % 7
    fwd = forward_perm(n, HEAD_V_DIM)
    inv = _v_inverse_perm(n, HEAD_V_DIM)
    assert np.array_equal(np.take(np.take(arr, fwd, axis=1), inv, axis=1), arr)


def test_wrong_length_raises() -> None:
    with pytest.raises(ValueError):
        _v_inverse_perm(6143, HEAD_V_DIM)