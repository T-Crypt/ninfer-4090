# Uncensored Qwen3.8-27B from GGUF (status, draft)

This doc tracks `ninfer-uncensored`: a second production line that serves HauhauCS's uncensored Qwen3.8-27B on the
RTX 4090. The PR stays a draft until the artifact converts, passes its checks and serves through llama-swap.
Board ticket `0N7219T` on the homelab board; the full operator play-by-play lives in the homelab repo at
`projects/active/NINFER-UNCENSORED.md`, which is the authority for stage numbering and gates — the list below is a
mirror and the homelab doc wins when they drift.

## Why

`HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` is the dense Qwen3.8-27B with changed weight values.
It keeps the architecture this fork already serves: 64 layers (48 GDN, 16 full attention), vocab 248320, the native
NextN MTP head, GGUF arch `qwen35`. On llama.cpp we run it at a 131K `-c` setting and 91.6 t/s; on NInfer the same
checkpoint gets 262K context, ~130 t/s decode and the INT8 prefill — a runtime/setting difference, not a format
limit (the GGUF itself advertises `qwen35.context_length = 262144`).

The gap: `tools/convert/qwen3_8_27b/convert.py` reads BF16 safetensors in the Hugging Face layout, and HauhauCS
publishes GGUF only. This branch adds a GGUF-to-HF tool and leaves the engine alone.

## Branches

| Branch | Base | Carries |
|---|---|---|
| `ninfer-qwen27b` | `deploy/aphotic` @ `46645ada` | the engine as deployed (INT8 prefill, tolerant tool calls). Serves the official artifact |
| `ninfer-uncensored` | `ninfer-qwen27b` | + `tools/convert/gguf_to_hf/` and this doc. Serves the HauhauCS artifact |

Rebase `ninfer-uncensored` onto `ninfer-qwen27b` whenever production moves. Both llama-swap entries run the same
`ninfer-serve` binary until this branch touches `src/`.

## Facts checked 2026-10-07

| Fact | Value |
|---|---|
| Source | `...-Aggressive-Q8_K_P.gguf`, 31.46 GB: 453 `Q8_0`, 53 `BF16` (output head, MTP head), 360 `F32`. Stock ggml types; gguf-py dequantizes all of them |
| Vision | separate `mmproj-...-BF16.gguf`. Official `model.visual.*` sits entirely in `model-00001-of-00018.safetensors` of `Qwen/Qwen3.8-27B` @ `1d4bf0f2`, next to 59 early text tensors |
| Converter checks | config, every recipe tensor's name/shape/BF16 dtype, SHA-256 of six frontend files. Weight hashes go unchecked |
| DFlash2 | the converter requires `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c`; we serve with `--spec mtp`, so it rides along unused |
| Transforms to invert | llama.cpp `conversion/qwen.py`: `_LinearAttentionVReorderBase.modify_tensors` (V-head grouped-to-tiled reorder, 16 K / 48 V heads) then `Qwen3NextModel.modify_tensors` (`A_log` to `-exp`, `dt_bias` rename, conv1d squeeze, `+1` on every `*norm.weight` except `linear_attn.norm`) |

## Tool layout

```
tools/convert/gguf_to_hf/
  __main__.py      CLI: --gguf, --reference-dir, --out, --vision-from-reference
  common.py        GGUF read, dequantize, BF16 shard writer, index.json
  arches/qwen35.py name map + inverse transforms (one module per arch)
  validate.py      cosine / max-abs-diff against reference shards
  README.md
```

Undo the inner transforms first, then the V reorder. Work in fp32, cast to BF16 once. Keep the artifact identity
(`qwen3.8-27b` / `groupwise-int` / `qwen3_8_27b-v2`) so the binder resolves `Qwen38GroupwiseInt`; record the real
source in a `PROVENANCE.json` sidecar.

## Stages (each passes before the next)

- [x] 1. `ninfer-qwen27b` from `46645ada`, `ninfer-uncensored` on top, worktree `~/wt/ninfer-0N7219T`
- [x] 2. CPU-only venv (`CUDA_VISIBLE_DEVICES=`), pinned downloads, GGUF SHA256SUMS pass (2026-10-08)
- [x] 3. `gguf_to_hf` writes 1199 tensors matching the official index in name, shape and BF16 dtype — 866 derived from the GGUF (that is where the mapping risk lives) plus 333 vision copied from the reference (2026-10-08: 20 shards, 55.56 GB, name check exact)
- [x] 4. Mapping proof against official shard 1: cosine > 0.99 on all 59 text tensors **and** max-abs-diff ~ 0 (cosine alone cannot catch a missed `+1` on near-constant norm vectors — `validate.py` compares `ours + 1` and flags any norm where the shifted version fits better); mmproj vs official vision equal or cosine > 0.9999. The MTP norms (`mtp.pre_fc_norm_*`, `mtp.norm`) are not in shard 1, but all 18 official shards are local, so gate 4 compares them like every other tensor. **PASS 2026-10-08:** 866/866 compared, worst cos 0.994197, 168 norms checked, 0 missed-`+1` flags, MTP norms cos 1.000000
- [x] 5. Stock converter; `tools.artifact.inspect` matches the current converter's expected counts — **PASS 2026-10-08 (cuda, 181.8 s):** 1190 objects (1184 tensors, 6 resources); BF16 627, FP32 96, I32 1, Q4G64_F16S 183, Q5G64_F16S 246, Q6G64_F16S 1, W8G32_F16S 30. Note: the official artifact on disk (`qwen3_8_27b.ninfer`, built 2026-10-02) is an older build — 1124 objects, BF16 582, W8G32_F16S 9; the current converter emits 66 more tensors and the 46645ada engine consumes them. Provenance sidecar written next to the artifact
- [x] 6. Perplexity `--quick` within ~5% of the official artifact; refusal prompts answered; tool call and image request pass; MTP acceptance and decode t/s recorded. **PASS 2026-10-08:** PPL 4.4005 vs 4.3446 official (+1.29%, int8 KV, quick corpus); 2 refusal probes answered; `/v1/messages` tool_use structured + vision description correct; decode 131.2 tok/s, MTP acceptance 70.8% (51/72), TTFT 294 ms
- [x] 7. llama-swap entry `NInfer-HauHauCS-27B` serves through `:9090`, no role aliases (2026-10-08: homelab commit `d266540`, unit env var `NINFER_MODEL_HAUHAUCS`, lsw validate 25 models / 0 problems, cold load 41.5 s, chat completion through `:9090` returns, only `ninfer-serve` holds the card)
- [ ] 8. Docs and close-out: worktree removed, scratch data deleted

## Next dense Qwen (Qwen4)

`common.py`, `validate.py`, the stages and the branch pattern carry over. Each new model needs official support on
`ninfer-qwen27b` first (converter plus engine), a new `arches/<arch>.py` that inverts that model's llama.cpp
`modify_tensors`, and the official shard holding vision and the early text layers for stage 4.
