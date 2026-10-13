---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - uncensored
  - multimodal
  - conversational
  - cuda
  - rtx-4090
---

# Qwen3.8-27B-Uncensored-HauhauCS for NInfer (RTX 4090)

The [HauhauCS Aggressive uncensored tune](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted to the native
[NInfer](https://github.com/sergiuszm/ninfer-4090) `.ninfer` artifact format for the RTX 4090 (sm_89)
line. The HauhauCS release ships GGUF only; this repository carries the artifact built from its
`Q8_K_P` quant, plus the conversion report and the provenance sidecar.

`.ninfer` is NInfer's own container. This is not a Transformers checkpoint, not safetensors and not a
GGUF file, and it does not load in `transformers`, vLLM, llama.cpp or LM Studio. It is read by the
NInfer engine only.

> [!WARNING]
> This model is uncensored by design: HauhauCS reports 0/465 refusals on its probe set, and the
> Aggressive profile answers hard prompts directly with minimal preamble. It will produce content that
> other models refuse to. You are responsible for what you generate with it and for complying with
> the source licenses and applicable law.

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_hauhaucs.ninfer` |
| Size | 20,437,336,576 bytes (19.03 GiB) |
| SHA-256 | `c58617dd7dc5c171eb90f48dd8e0a1db6d1669bed3dccdf4e431aff014c992b6` |
| Container | NInfer v2 (`NINFER\0\2` header, JSON object directory) |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `groupwise-int` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,190 (1,184 tensors and 6 resources) |

The text body uses the registered Q4/Q5/Q6 groupwise allocation; the token embedding and full output
head use `W8G32_F16S`. Vision, MTP, DFlash2, proposal-head, tokenizer, chat-template, generation and
media-processor objects are included, so one file serves text, image and video input.

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'c58617dd7dc5c171eb90f48dd8e0a1db6d1669bed3dccdf4e431aff014c992b6' \
  'qwen3_8_27b_hauhaucs.ninfer' | sha256sum --check
```

## Requirements

- NInfer built from source for `sm_89` from the [4090 port line](https://github.com/sergiuszm/ninfer-4090).
  Built and served here from the tree at revision `46645ada`, the same binary that serves the official
  Qwen3.8-27B artifact. This line is built with CUDA 13.4 and GCC 15.
- 64-bit Linux, an NVIDIA driver new enough for that toolkit.
- One NVIDIA GeForce RTX 4090 (24 GB). The engine runs on a single device.

## How it was built

`HauhauCS` publishes GGUF, and NInfer's converter reads BF16 safetensors in the Hugging Face layout,
so the pipeline ran in two steps:

1. `tools/convert/gguf_to_hf` dequantized the `Q8_K_P` GGUF and inverted the transforms llama.cpp
   applies on export: the grouped-to-tiled V-head reorder, `A_log` to `-exp`, the `dt_bias` rename,
   the conv1d squeeze, and the `+1` on every `*norm.weight` except `linear_attn.norm`. The result was
   1,199 BF16 tensors in 20 shards: 866 derived from the GGUF, 333 vision tensors and the frontend
   files taken from the reference checkpoint.
2. The stock `qwen3_8_27b` converter then quantized that tree to this artifact on CUDA (181.8 s).

| Field | Value |
|---|---|
| Source file | `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf` |
| Source file SHA-256 | `4e7735df4d1e2ec721f2551f531b815702a2f89123238c564797eda4b0304bc` |
| Source repository revision | `993a5971fda8f30dd1b7eb2654792ba4415c7460` |
| Reference checkpoint | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (frontend + all 333 `model.visual.*` tensors) |
| Conversion recipe | `qwen3_8_27b-v2` |
| Conversion tool | `ninfer-4090` branch `ninfer-uncensored` @ `b9b9e19d` (`gguf_to_hf` @ `dbc56a6d`) |
| DFlash2 companion | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` (carried, unused with `--spec mtp`) |
| HauhauCS mmproj GGUF | not used; the 333 `model.visual.*` tensors are the reference checkpoint's, copied verbatim |

The identity fields (`qwen3.8-27b` / `groupwise-int` / `qwen3_8_27b`) are kept identical to the
official artifact so the runtime binder resolves `Qwen38GroupwiseInt`; the weights are the HauhauCS
tune.

## Checks run before publishing

Mapping, against the official checkpoint shards: all 866 derived text tensors compared, worst cosine
`0.994197`, 168 norm tensors checked with zero missed-`+1` flags, MTP norms cosine `1.000000`, and the
vision tower equal to the reference it was copied from. The norm check compares `ours + 1` as well,
because cosine alone cannot catch a missed offset on a near-constant norm vector.

Quality and performance, measured on one RTX 4090 through NInfer's OpenAI-compatible route at a
262,144-token budget, INT8 KV, `--spec mtp --draft-tokens 3`, thinking enabled, greedy:

| Metric | This artifact | Official Qwen3.8-27B artifact |
|---|---|---|
| Perplexity, quick corpus, INT8 KV | 4.4005 | 4.3446 (+1.29% for this one) |
| Decode, MTP acceptance | 131.2 tok/s, 70.8% (51/72) | not recorded in this run |
| Time to first token | 294 ms | not recorded in this run |
| Tool call via `/v1/messages` | structured `tool_use` returned | not recorded in this run |
| Image request | correct description | not recorded in this run |
| Refusal probes | both answered (the point of this tune) | both refused |

For scale: the same GGUF under llama.cpp at a 131K context setting decoded at 91.6 tok/s here. The
context, KV type and draft settings differ between the two engines, so that pair is a sanity check,
not a controlled comparison. These are single-run spot measurements, not a benchmark suite.

## Serving

```bash
./build/apps/ninfer-serve qwen3_8_27b_hauhaucs.ninfer \
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b-uncensored \
  --max-context 262144 --kv-capacity 262144 \
  --kv-dtype rk4v4-e8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --preserve-thinking
```

`--kv-dtype rk4v4-e8` is the 4-bit KV codec of the 4090 port. In this lab the artifact is served
behind llama-swap as `NInfer-HauHauCS-27B`, alongside the official artifact on the same binary.

## Limits

- The artifact is engine- and line-specific: it needs the sm_89 NInfer port, and the v2 container.
  The ext line's `upgrade_ninfer_v2_to_v3.py` is what upgrades a file like this to a v3 container.
- Perplexity is about 1.3% above the official artifact's quantized build on the quick corpus. That is
  the price of re-quantizing a Q8_K_P GGUF that was itself a dequantize of the tune.
- The uncensoring is a weight-level change made by HauhauCS. It is reported here as published, not
  re-measured beyond the probes above.
- Single-sample spot measurements only; no pass@k, no multi-run medians.
- NInfer executes on one GPU with a startup-fixed request capacity; there is no multi-GPU or
  continuous-batching path.

## Files

| File | What |
|---|---|
| `qwen3_8_27b_hauhaucs.ninfer` | the artifact |
| `qwen3_8_27b_hauhaucs.ninfer.conversion.json` | full conversion report: source paths, config summary, per-object inventory, environment |
| `qwen3_8_27b_hauhaucs.ninfer.PROVENANCE.json` | the short provenance record: source file hashes, pipeline and gate results |
| `artifact-manifest.json` | artifact identity, source and inventory in the NInfer manifest schema |
| `SHA256SUMS` | checksums for the artifact and both reports |

## License

The weights follow the source model's license, Apache-2.0, for both
[HauhauCS's release](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF)
and [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). The `.ninfer` format, the converter
and the engine are part of the [NInfer](https://github.com/sergiuszm/ninfer-4090) codebase.
