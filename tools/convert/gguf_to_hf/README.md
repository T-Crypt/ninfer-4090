# GGUF to Hugging Face BF16 conversion

Inverts llama.cpp's HF → GGUF conversion for dense Qwen checkpoints so a GGUF-only
release (e.g. the uncensored `HauhauCS` Qwen3.8-27B) can be fed to NInfer's stock
converter, which only accepts HF-layout BF16 safetensors.

No engine code is involved. The output directory is a complete Hugging Face model
directory: six frontend files + config files copied from a pinned reference
checkpoint, BF16 shards, and a fresh `model.safetensors.index.json`.

## Run

```bash
python3 -m tools.convert.gguf_to_hf \
  --gguf /path/to/model.gguf \
  --reference-dir /path/to/Qwen3.8-27B \
  --out /path/to/hf-bf16 \
  --vision-from-reference
```

* `--reference-dir` is the pinned official checkpoint (frontend files, index, the
  shard(s) carrying `model.visual.*`).
* `--vision-from-reference` copies the vision tensors straight from the reference
  shards instead of reading an mmproj GGUF.
* The tool refuses to overwrite an existing output index unless `--force` is passed.

The converter runs tensor by tensor, streaming the two 2.5 GB embeddings through
bounded float32 chunks; `--shard-bytes` (default 3 GB) sizes the output shards.
Progress is one flushed line per tensor.

## Validate

```bash
python3 -m tools.convert.gguf_to_hf.validate \
  --out /path/to/hf-bf16 --reference-dir /path/to/Qwen3.8-27B
```

Gate 3 (structure): produced names == official index names, every shape matches,
every dtype is BF16. Gate 4 (values): cosine vs the reference tensors, with a
dedicated missed-`+1` check on the 168 norm tensors (cosine alone cannot see a
near-constant `v` vs `v-1`).

## Adding an architecture

1. Create `arches/<arch>.py` with, at minimum:

   * `ARCH` — the `general.architecture` string written in the GGUF;
   * `gguf_to_hf_name(gguf_name) -> str`;
   * `is_identity(hf_name) -> bool` — False for anything llama.cpp changed
     (values, shapes), so the converter streams renamed-only tensors in chunks;
   * `invert(gguf_name, data) -> np.ndarray` — the inverse of llama.cpp's
     transforms, on a float32 array in torch order, never mutating in place.

2. Register it in `arches/__init__.py`'s `REGISTRY`.

3. Add a golden name-map fixture under `tests/convert/gguf_to_hf/data/` and an
   independent forward implementation in `test_invert_roundtrip.py`.

## Why each transform looks the way it does

llama.cpp applies `_LinearAttentionVReorderBase.modify_tensors` **first**, then
`Qwen3NextModel.modify_tensors`. The inverse therefore undoes the value changes
first, then the V-head reorder, then the squeezed shape:

| GGUF tensor | HF tensor | Inverse |
|---|---|---|
| `blk.N.ssm_a` | `...linear_attn.A_log` | `log(-x)` then un-reorder |
| `blk.N.ssm_dt.bias` | `...linear_attn.dt_bias` | un-reorder only |
| `blk.N.ssm_conv1d.weight` | `...linear_attn.conv1d.weight` | un-reorder V rows, `unsqueeze(1)` |
| `blk.N.attn_qkv.weight` | `...linear_attn.in_proj_qkv.weight` | un-reorder V rows only |
| `blk.N.attn_gate.weight` | `...linear_attn.in_proj_z.weight` | un-reorder rows |
| `blk.N.ssm_alpha/beta.weight` | `...linear_attn.in_proj_a/b.weight` | un-reorder rows, head dim 1 |
| `blk.N.ssm_out.weight` | `...linear_attn.out_proj.weight` | un-reorder columns |
| `*.norm.weight` | *(168 tensors)* | subtract 1 |

Details that matter:

* The V-head reorder (grouped → tiled, `num_k_heads=16`, `num_v_per_k=3`) is a
  transpose of a 16×3 block, so it is **not** its own inverse — the inverse
  permutation is computed explicitly (`_v_inverse_perm`) and unit-tested against
  the forward.
* Only the V rows of `in_proj_qkv` / `conv1d` are reordered; Q and K rows pass
  through untouched.
* The `+1` set is 168: every `*norm.weight` except `linear_attn.norm.weight`,
  **plus** `mtp.pre_fc_norm_embedding/hidden` — llama.cpp renames those to
  `enorm`/`hnorm` before the suffix check, so they get the `+1` even though their
  HF names do not end in `norm.weight`. `test_names.py` pins the count.