# Dossier: TensorRT-LLM (+ ModelOpt)

Researched 2026-10-07 (public sources only: repo, docs, release notes, PRs/issues, NVIDIA blogs).

## What it is / backing / license

- **What it is:** NVIDIA's LLM and Visual-Gen inference stack: Python API (`LLM`, `trtllm-serve`,
  `trtllm-bench`), custom CUDA kernels (attention/GEMM/MoE), runtime (in-flight batching, chunked
  prefill, paged KV cache, CUDA graphs, overlap scheduler, PD disaggregation, speculative
  decoding). Since the 1.0 line it is "Architected on PyTorch" and PyTorch is the **sole**
  execution backend (the TensorRT engine backend was removed in release 1.2).
  ([README](https://github.com/NVIDIA/TensorRT-LLM),
  [release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html))
- **Backing:** NVIDIA (in-house, ~14.8k GitHub stars; community PRs merged).
- **License:** **Apache 2.0** — LICENSE on main says "This project is licensed under the Apache 2.0
  license"; README badge "Apache 2"; README news: "TensorRT LLM is now fully open-source, with
  developments moved to GitHub" (2026-03-22). Third-party portions are listed in LICENSE: CUTLASS
  (BSD-3), causal-conv1d (BSD-3), flash-linear-attention (MIT), FlashInfer (Apache-2.0), SGLang
  (Apache-2.0), Mamba (Apache-2.0), XGrammar (Apache-2.0), LTX-2 (community license, confined to
  `visual_gen/models/ltx2/`). **Porting code is license-OK.**
- **ModelOpt:** companion optimization library (post-training quantization FP8/NVFP4/INT8
  SmoothQuant/INT4 AWQ/SVDQuant, QAT/distillation, pruning/NAS, sparsity, draft-module training
  for speculative decoding). Apache 2.0. Open source since 2025-01-28, rebranded from "TensorRT
  Model Optimizer" to "NVIDIA Model Optimizer" on 2025-12-08.
  ([repo](https://github.com/NVIDIA/Model-Optimizer))

## Repo, version, last release date

- Repo: <https://github.com/NVIDIA/TensorRT-LLM>; ModelOpt: <https://github.com/NVIDIA/Model-Optimizer>
- Latest tag: **v1.3.0rc29** (commit 2026-09-24); `main` carries version **1.4.0rc0** (README
  badge). Docs default version is 1.3.0rc29; some pages (support matrix, precision) still render
  the older **1.1.0rc5** build, which lags `main`.
- Lineage: 0.5 (2023, first Windows/TRT-engine era) → 0.12 (2024-08, FP8 FMHA on Ada) → 0.16
  (2024, W4A8 on Ada; cited as `0.16.0.dev` in NVIDIA's Nov 2024 Llama 3.2 blog) → **1.0**
  (PyTorch backend + LLM API stable) → **1.1** (GPT-OSS/Hunyuan/Seed-OSS, B300/GB300) → **1.2**
  (DGX Spark beta; TRT backend removed; two-model spec decode removed) → **1.3** (2026-09,
  TRITON MoE backend deprecated).
  ([tags](https://github.com/NVIDIA/TensorRT-LLM/tags),
  [release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html))

## Runs on an RTX 4090 (sm_89)?

**Partly — yes for the core stack and FP8, no for the newest fast paths.**

- The published support matrix lists **Ada Lovelace** among supported GPU architectures, and
  precision per arch: "Ada Lovelace (SM89) — FP32, FP16, BF16, FP8, INT8, INT4"
  ([support-matrix, docs 1.1.0rc5](https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html)).
- Ada-specific items from the release notes
  ([release notes](https://nvidia.github.io/TensorRT-LLM/release-notes.html)):
  - 0.9.0: "NVIDIA Ampere (SM80, SM86), NVIDIA Ada Lovelace (SM89), NVIDIA Hopper (SM90) all
    support head sizes [32, 40, 64, 80, 96, 104, 128, 160, 256]"
  - 0.10.0: "Added W4A(fp)8 CUTLASS kernels for the NVIDIA Ada Lovelace architecture."
  - 0.12.0: "Supported FP8 FMHA for NVIDIA Ada Lovelace Architecture."
  - 0.15.0: "Added support for per-token per-channel FP8 (namely row-wise FP8) on Ada."
  - 0.16.0: "Added W4A8 quantization support to BF16 models on Ada (SM89)."
  - 1.0: "Fallback to cubins for fp8 fmha kernels on Ada (#5779)"; "Fix tileN cannot % 16==0 &
    support sm89 deepgemm bmm (#5531)"
- But the **GDN hybrid fast path is not for sm_89**. In
  [`_torch/modules/mamba/gdn_mixer.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/mamba/gdn_mixer.py):
  "FlashInfer ships the GDN prefill kernel for Hopper (SM90), datacenter Blackwell (SM100/SM103)
  and consumer Blackwell (SM120/SM121); on other archs it aborts at launch, so we fall back to
  Triton there." The gates in
  [`_utils.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_utils.py):
  prefill archs `(90, 100, 103, 120, 121)`, decode/verify archs `(90, 100, 103)`. **On a 4090
  TRT-LLM runs GDN via the vendored Triton (flash-linear-attention) kernels only.**
- Other Blackwell/Hopper-only paths (so: no on 4090): NVFP4 ("NVFP4 inference requires Blackwell
  GPUs and TensorRT-LLM v1.2 or later" — ModelOpt
  [hf_ptq support matrix](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md));
  DeepSeek-V4 "only supported on Blackwell GPUs (SM100+)"; Kimi-K3 "only supported on NVIDIA
  Blackwell GPUs (SM100 family)"; MLA chunked prefill "only … on SM90/SM100/SM103/SM107/SM120";
  DFlash `TRTLLM` attention backend "supports SM100/SM103 only"
  ([supported-models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md),
  [spec-decode](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)).
- The published perf tables only cover datacenter GPUs + "RTX 6000 Pro Blackwell Server Edition"
  ([perf overview](https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html)) —
  **no RTX 4090 in any published TRT-LLM benchmark table.**

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, day-0-grade support.** `Qwen3NextForCausalLM` is in the model zoo
  ([`_arch_index.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/_arch_index.py));
  support added by PR [#7892](https://github.com/NVIDIA/TensorRT-LLM/pull/7892) "[None][feat]
  Support Qwen3 next", opened 2025-09-22, **merged 2025-09-29** (issue
  [#7694](https://github.com/NVIDIA/TensorRT-LLM/issues/7694) opened 2025-09-13, closed
  2025-10-02). The implementation is "Adapted from" SGLang's `hybrid_linear_attn_backend.py` and
  `qwen3_next.py` config; it reuses the Mamba2 state machinery
  (`Qwen3NextGatedDeltaNet`, `Mamba2Metadata`) and the vendored FLA Triton kernels (see above).
- The Qwen3.5/3.8/4 family is already in the zoo (main-branch
  [supported-models](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md)):
  `Qwen3_5MoeForCausalLM` = "Qwen3.8-MoE, Qwen3.5-MoE" (HF `Qwen/Qwen3.8-2.4T-A95B`);
  `Qwen4ExpForCausalLM` = "Qwen3.8-Flash-Next (text)"; dense `Qwen3_5ForCausalLM` in the arch
  index; VLM `Qwen3_5ForConditionalGeneration` / `Qwen3_5MoeForConditionalGeneration` (L+I+V).
  So NVIDIA's stack is tracking the same Qwen3.8 family line as our `qwen3.8-27b` identity.
- **MTP:** "MTP is supported by DeepSeek models and other architectures that ship native MTP
  modules, including **Qwen3.8 MoE, Qwen3.5 MoE**, and Step-3.x" with `num_nextn_predict_layers`,
  and `use_relaxed_acceptance_for_thinking` / `relaxed_topk` / `relaxed_delta` for reasoning models
  ([spec-decode](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)).
  Feature matrix: MTP = Yes for `Qwen3_5MoeForCausalLM` and `Qwen4ExpForCausalLM`; **Qwen3Next =
  "No"** (it has no MTP head — if the next dense Qwen has both GDN and an MTP head, TRT-LLM has no
  public example of that combination yet).
- **Qwen3Next feature row** (same page): Overlap Scheduler Yes, CUDA Graph Yes, Attention DP Yes,
  Disaggregated Serving Untested, Chunked Prefill Yes, Speculative Decoding No, KV Cache Reuse Yes
  ("requires an explicit recurrent-state snapshot policy … model default disables reuse when no
  snapshot policy is configured"), Sliding Window No, Guided Decoding Yes. Footnote: "Qwen3-Next-
  80B-A3B exhibits relatively low accuracy on the SciCode-AA-v2 benchmark."
- **Vision:** Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5 VLM all listed (support matrix
  + multimodal feature matrix).
- **256K:** supported in principle — `max_seq_len` defaults to `max_position_embeddings`, paged KV
  cache + chunked context remove the input-length constraint
  ([scheduler doc](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md));
  **no public validation of a 256K hybrid-Qwen run on a 4090 found** (not found).

## Published numbers on a 4090 or similar

- **TRT-LLM tok/s on an RTX 4090: not found.** NVIDIA publishes no 4090 TRT-LLM tables (perf
  overview covers Hopper/Blackwell/GB200/RTX 6000 Pro Blackwell only).
- Closest published 4090 figures (different stack, same GPU), NVIDIA dev blog
  [Llama 3.2 full-stack](https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/)
  (2024-11-19):
  - On RTX 4090, **ONNX Runtime** (not TRT-LLM) with Llama 3.2 3B AWQ INT4 (Windows/DirectML):
    output tok/s, BS=1: 253 (100→100), 203 (2000→100), 165 (4000→100); BS=4: 615, 374, 251.
  - Same blog, TRT-LLM numbers are on **8×H200**: Llama 3.2 90B, BF16 encoder + FP8 decoder
    (TRT-LLM `0.16.0.dev`, ModelOpt 0.21 pre-release, TRT 10.4.0): max throughput 2,646 / 1,417 /
    480 output tok/s for 8K/20K/60K input → 2K output (batch tuned for max node throughput); min
    latency 64 / 63 / 55 tok/s (BS=1, TP8).
- Puget Systems, [Benchmarking with TensorRT-LLM](https://www.pugetsystems.com/labs/hpc/benchmarking-with-tensorrt-llm/)
  (2024-02-16): TRT-LLM v0.5.0 + TRT 9.1.0.4, Llama-2-7B AWQ INT4, Windows 11, RTX 4090 FE vs
  4080S/4070Ti/4060Ti; per-test tok/s only in chart images (not extractable from text); in text:
  for 2048-in/512-out BS=8 "The RTX 4090 completed this test in about 35 seconds, and the 16GB
  cards each took a little over 50 seconds"; 12 GB cards ~260 s and the 4060 Ti 960 s (sysmem
  fallback when VRAM is exceeded).
- NVIDIA blog [TRT-LLM for Windows](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)
  (2023-10-17): Llama 2 / Code Llama on RTX "up to 4x faster" — qualitative, no tok/s in text.

## Ideas we could port into ninfer-4090

(All Apache-2.0/MIT/BSD-3 in the TRT-LLM tree. "Gain as they measured it" is not published for
sm_89 — no 4090 numbers exist, so treat these as reference designs to validate ourselves.)

1. **GDN hybrid execution reference** — `tensorrt_llm/_torch/modules/mamba/gdn_mixer.py`
   (`Qwen3NextGatedDeltaNet`), `tensorrt_llm/_torch/modules/fla/` (vendored
   flash-linear-attention Triton: `chunk_gated_delta_rule` prefill,
   `fused_recurrent_gated_delta_rule_update` decode, cached-replay update for spec-verify,
   `fused_sigmoid_gating_recurrent`), `causal_conv1d_triton.py`, `fuse_elementwise_ops.py`
   (`fused_gdn_post_conv`, `pack_gdn_decode_qkv`, `extract_transpose_prefill_slice`),
   `layernorm_gated.py`. This is a complete, working sm_89-OK Triton GDN stack to compare
   numerically against our replayssm path and to mine kernels from (post-conv fusion, decode qkv
   packing, gated RMSNorm).
2. **Low-M GEMM for decode** — `tensorrt_llm/_torch/modules/low_m_gemm.py`: row-specialized
   GEMM for M≤32 with AutoTuner M-buckets (1,2,…,8,16,32); the fast backends (cute-dsl,
   flashinfer) are Blackwell, but the *tuning-by-M-bucket* method and M≤32 specialization is the
   portable idea; NInfer decode GEMMs could be autotuned per batch size the same way.
3. **MTP one-model verify + relaxed acceptance for thinking** — `modeling_speculative.py` +
   `MTPDecodingConfig` (`relaxed_topk`/`relaxed_delta`/`use_relaxed_acceptance_for_thinking`):
   accept a draft token if it lands in a relaxed top-k candidate set during the thinking phase.
   Directly applicable to our MTP3 head on a reasoning model.
4. **IFB/chunked-prefill scheduler semantics** —
   [paged-attention-ifb-scheduler.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md):
   `max_num_tokens` token-budget scheduling, context-before-generation packing order, chunk size
   an integer multiple of the KV block size. Good checklist against NInfer's ingress/compact-batch
   rules.
5. **Breakable CUDA graphs** — `pyexecutor/breakable_cuda_graph.py` (`eager_on_graph`): run
   custom ops (e.g. the GDN `torch.library` custom op) eagerly inside a CUDA graph. Relevant to
   keeping NInfer's graph strategy on a hybrid model.
6. **KV-cache salting + host offload + recurrent-state snapshot policy** (1.1 release notes;
   `mamba_state_config.periodic_snapshot_interval`): hybrid-model cache-reuse policy — snapshot
   the recurrent GDN state periodically so prefix reuse is safe; NInfer's context-cache would
   need the same for GDN layers.
7. **Spec-decode extras:** NGram (prompt lookup), PARD (parallel drafting, arXiv:2504.18583),
   DFlash (arXiv:2602.06036), Suffix-Automaton enhancement (from baseten `sa_spec`, Apache-2.0),
   EAGLE-3 dynamic tree.
8. **ModelOpt PTQ on the 4090:** row-wise (per-token/per-channel) FP8 PTQ works for
   "QWen3, 3.5 MOE, Next" (✅ FP8 in the
   [ModelOpt hf_ptq matrix](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md));
   NVFP4 is Blackwell-only. Useful to generate FP8 checkpoints for a future TRT-LLM/vLLM side-by-side.

## Worth running beside NInfer/llama.cpp?

**Partly — yes as a validation/reference, weak as a speed competitor on this box.** Its fast GDN
kernels (FlashInfer) don't exist for sm_89, so a 4090 run exercises the Triton fallback; NVIDIA
publishes no 4090 tuning at all; and it's a heavyweight Python/CUDA stack to install and drive.
If we do run it, a fair test: same weights/quant (ModelOpt FP8 or our groupwise-int), single RTX
4090, batch 1–8, 32K–256K context (GDN layers make long context cheap — that's the point),
compare prefill tok/s, decode tok/s, TTFT/TPOT against NInfer and llama.cpp on the same hybrid
Qwen checkpoint. Expect its value to be: (a) a second independent GDN implementation to check
numerics against, (b) kernel/scheduling ideas (above), not a faster 4090 engine out of the box.

## Risk / unknowns

- No public sm_89 performance data; NVIDIA's current investment is visibly datacenter (Blackwell
  -only features, DGX Spark beta, GB200/GB300 work). Ada is maintained, not a frontier.
- On a 4090, GDN prefill/decode runs the **Triton fallback**; relative speed vs NInfer's replayssm
  GDN path is unknown until measured (that measurement is out of scope this week).
- The exact combination the next dense Qwen likely has — **GDN + MTP head** — is not publicly
  demonstrated in TRT-LLM (Qwen3Next has no MTP; MTP is shown for Qwen3.5/3.8 MoE stacks). Their
  FlashInfer GDN *verify* kernels (for spec-decoding) are SM90/SM100/SM103-only.
- Fast churn / breaking changes: TRT backend and two-model spec decode removed in 1.2; TRITON MoE
  backend deprecated in 1.3 (2026-09); published docs (1.1.0rc5 pages) lag main (1.4.0rc0), so
  "supported" claims from older doc versions can be stale in both directions.
- Telemetry is **on by default** (README): disable with `TRTLLM_NO_USAGE_STATS=1` before any
  benchmarking.
- Qwen3-Next-80B-A3B "exhibits relatively low accuracy on the SciCode-AA-v2 benchmark" (their
  own note) — accuracy tracking on the hybrid family is still maturing.

## Sources

- https://github.com/NVIDIA/TensorRT-LLM (README, LICENSE, model zoo, examples)
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/_arch_index.py
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/models/modeling_qwen3_next.py
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/mamba/gdn_mixer.py
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_utils.py (GDN arch gates)
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/low_m_gemm.py
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/paged-attention-ifb-scheduler.md
- https://nvidia.github.io/TensorRT-LLM/release-notes.html (docs 1.3.0rc29)
- https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html (docs 1.1.0rc5)
- https://nvidia.github.io/TensorRT-LLM/reference/precision.html
- https://nvidia.github.io/TensorRT-LLM/developer-guide/perf-overview.html
- https://github.com/NVIDIA/TensorRT-LLM/pull/7892 (Qwen3-Next support, merged 2025-09-29)
- https://github.com/NVIDIA/TensorRT-LLM/issues/7694 (Qwen3-Next request, closed 2025-10-02)
- https://github.com/NVIDIA/TensorRT-LLM/tags (v1.3.0rc29 = 2026-09-24)
- https://github.com/NVIDIA/Model-Optimizer (README: license, news, techniques)
- https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md (PTQ support matrix)
- https://developer.nvidia.com/blog/llama-3-2-full-stack-optimizations-unlock-high-performance-on-nvidia-gpus/ (4090 + H200 tables)
- https://www.pugetsystems.com/labs/hpc/benchmarking-with-tensorrt-llm/ (RTX 40-series TRT-LLM v0.5.0)
- https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/
