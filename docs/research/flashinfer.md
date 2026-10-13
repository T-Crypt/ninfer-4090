# Dossier: FlashInfer

Researched 2026-10-06 (public sources only: repo, docs.flashinfer.ai, release notes, arXiv paper,
FlashInfer blog, and the SGLang/vLLM consumer source that wires in the GDN kernels). No installs, no
weight downloads, no GPU jobs.

## What it is / backing / license

- **What it is:** a **kernel library + JIT kernel generator for LLM serving** (not an end-to-end
  engine). "High-Performance GPU Kernels for Inference … unified APIs for attention, GEMM, and MoE
  operations with multiple backend implementations including FlashAttention-2/3, cuDNN, CUTLASS, and
  TensorRT-LLM." It supplies the low-level kernels that the engines in our other dossiers call: the
  README "Adoption" list is **SGLang, vLLM, TensorRT-LLM, TGI, MLC-LLM, LightLLM, lorax, ScaleLLM**
  ([README](https://github.com/flashinfer-ai/flashinfer)). Attention (paged/ragged KV, prefill/decode/
  append, MLA, Cascade, sparse, POD), GEMM (BF16/FP8/FP4, grouped), MoE (fused, FP8/FP4), sampling
  (sorting-free top-k/top-p/min-p, chain speculative sampling), AllReduce/MNNVL/NVSHMEM, RoPE, norm,
  activations.
- **Backing:** **University of Washington + CMU + NVIDIA** (paper authors span UW, NVIDIA, CMU,
  Perplexity; first author Zihao Ye interning at NVIDIA). The SAMPL (UW) group lists FlashInfer as its
  flagship project; the 2025 sampling blog is "Shanli Xing (UW), Zihao Ye (UW, NVIDIA), Bohan Hou
  (CMU), Luis Ceze (UW, NVIDIA), Tianqi Chen (CMU, NVIDIA)". Heavily maintained by NVIDIA engineers
  in 2026 (the `cake_*`/`primTS`/`cutedsl` Blackwell backends in v0.7.0 are all NVIDIA-`-nv`/`-hub`
  authored). ([paper](https://arxiv.org/abs/2501.01005),
  [blog](https://flashinfer.ai/2025/03/10/sampling.html),
  [SAMPL project](https://sampl.cs.washington.edu/projects/flashinfer.html))
- **License:** **Apache 2.0** (repo license field + README). **Porting code is license-OK** (no GPL,
  no non-commercial).

## Repo, version, last release date

- Repo: <https://github.com/flashinfer-ai/flashinfer> (6.5k stars, 1.5k forks, ~1.1k issues);
  docs <https://docs.flashinfer.ai>; blog <https://flashinfer.ai>; bench/contest
  <https://bench.flashinfer.ai>.
- **Latest stable: v0.7.0, released 2026-09-22** (615 commits to `main` since); docs serve
  **v0.7.0.post1**. Nightly/RC stream: `v0.7.1rc1..rc5`, `v0.7.2rc1`. Prior stable **v0.6.18**
  ([releases](https://github.com/flashinfer-ai/flashinfer/releases)).
- Cadence: fast (v0.6.18 → v0.7.0 in ~a month with ~450 PRs); breaking API changes each release
  (v0.7.0 removes `comm.trtllm_custom_all_reduce`, `BatchDecodeMlaWithPagedKVCacheWrapper`, no-op
  `end_forward()`, `mamba.checkpointing_ssu` argument signature change).
- Toolchain: **CUDA 12.9 / 13.0 / 13.4 (PyTorch nightly)**; devcontainers for cu129/cu130/cu134;
  Blackwell CuTe-DSL kernels need the `[cu13]` extra. Install: `pip install flashinfer-python`
  (+ `install-cubin-wheel` / `install-jit-cache-wheel` for prebuilt arch-specific kernels).
- **GPU support matrix (README):** Turing **SM7.5** (RTX 20), Ampere **SM8.0/8.6** (RTX 30), **Ada
  SM8.9 (L4, L40, RTX 40 series)**, Hopper SM9.0, Blackwell SM10.0/10.3/11.0/**12.0/12.1 (RTX 50,
  DGX Spark)**. Note: "Not all features are supported across all compute capabilities."

## Runs on an RTX 4090 (sm_89)?

**Partly — the attention path runs on sm_89 via FlashAttention-2 (fa2), but every GDN/Gated-DeltaNet
kernel, the fast FMHA/FP4/FP8-MoE backends, and XQA decode are Hopper/Blackwell-only, so a GDN-hybrid
model on a 4090 gets FlashInfer only for the full-attention layers and the GEMM/sampling/KV plumbing.**

- The README support matrix lists **Ada SM8.9 / RTX 40 series** and "Support for SM75 (Turing) and
  later" — sm_89 is a supported arch ([README](https://github.com/flashinfer-ai/flashinfer)). The SGLang
  install doc restates the floor: "**FlashInfer is the default attention kernel backend. It only
  supports sm75 and above**."
- **But the attention backend on sm_89 is `fa2` only.** The paper is explicit: "Our implementations
  utilize the **FlashAttention2 (FA2) algorithm for architectures up to Ada(sm89)**, and the
  FlashAttention3 (FA3) algorithm for Hopper" ([paper §3.2](https://arxiv.org/abs/2501.01005)). The
  maintainer's backend/CC audit (issue [#3170](https://github.com/flashinfer-ai/flashinfer/issues/3170))
  confirms the gates: **fa2 = "Generic"** (works on sm_89); **fa3 = "SM90 only"**; **XQA =
  `major in {9, 10, 12}`**; **trtllm-gen = `major == 10`**; **CUTLASS FMHA = `is_sm100a || is_sm110a`**;
  **FMHAv2 = `is_sm12x`**. So on a 4090 the only in-tree attention decode/prefill backends are **fa2**
  (and cuDNN, runtime-dependent). The XQA gate is directly in source:
  `get_nvcc_flags_list(supported_major_versions=[9, 10, 12])`
  ([jit/xqa.py](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/xqa.py)) —
  **XQA (the Ampere/Ada decode kernel vLLM exposes) is NOT available on sm_89**.
- **GDN / Gated DeltaNet: NOT supported on sm_89.** The GDN decode/prefill/MTP kernels require SM90+.
  The SGLang consumer that wires them in gates on
  `torch.cuda.get_device_capability()[0] >= 9`
  ([gdn_flashinfer.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py):
  "SM90 (Hopper): full support — decode, prefill, MTP … SM100 (Blackwell): full support … Requires
  flashinfer >= 0.6.14"; target-verify only `sm_major in (9, 10)`). v0.7.0's GDN work is all
  Blackwell/Hopper: "Add Qwen fused GDN decode step for sm120 (#4481)", "fused_GDN_step … Qwen 3.6
  35B A3B on sm120 (#4708)", "gdn prefill decode mtp Cake backend for SM100 and SM103 (#4581)",
  "Kimi K3 … CuTe-DSL recurrent prefill on B200/B300 and SM120". **On a 4090 a GDN hybrid falls back
  to the engine's Triton/FLA GDN path — FlashInfer provides none.** (Same conclusion as the vLLM and
  SGLang dossiers.)
- FP4/NVFP4 GEMM and MoE, b12x, cute-dsl, DeepGEMM and the whole `cake_*`/`primTS` Blackwell stack are
  SM100/SM12x-only. What sm_89 keeps: fa2 attention, cuDNN BF16 GEMM (`[80,86,89,90,100,103]` includes
  89), FP8 GEMM (cutlass/cublas, "may fall back to SM89 tactics when native occupancy is zero"), FP8
  norm (`sm_version >= 89`), FP8/INT8 attention, sorting-free sampling. **No FP4, no XQA, no trtllm-gen,
  no GDN.**

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: NO sm_89 path; partial (full-attention layers only).** FlashInfer's GDN/Gated-DeltaNet
  kernels are SM90/SM100/SM120-only (see above). It has model-specific GDN entries for **Qwen 3.6
  (fused GDN decode, sm120)**, Kimi K3 (KDA), Nemotron-H (Mamba SSD), and a generic
  `gated_delta_rule_decode / _prefill / _mtp` API
  ([gdn_decode docs](https://docs.flashinfer.ai/api/gdn_decode.html)) — but none run on sm_89. So for
  the next dense Qwen (48/64 GDN layers per the Qwen3.8-27B card in the SGLang dossier), FlashInfer on
  a 4090 contributes **only the 16 full-attention (GQA) layers**, GEMM, sampling and KV-cache
  management; the GDN recurrence — the dominant compute — is the engine's own (NInfer's replayssm or
  an FLA/Triton fallback). **FlashInfer does not accelerate the hybrid's signature layers on sm_89.**
- **MTP: partial / no sm_89 GDN-verify.** The full-attention verify step (speculative `q_seq_len>1`
  tree attention) can run on sm_89 via fa2 (FlashInfer's prefill/append kernels "are used to serve
  models in speculative decoding"). But the **GDN MTP-verify kernel
  (`gated_delta_rule_mtp`) is SM90/SM100/SM120-only** ("`gated_delta_rule_mtp` still resolves an
  omitted `disable_state_update` to True in 0.7.0", #4581 sm100/103, SGLang `supports_target_verify =
  sm_major in (9,10)`). On a 4090 the MTP draft/verify for the GDN layers is the engine's own path;
  FlashInfer covers only the GQA layers' verify. Same split as vLLM/SGLang on sm_89.
- **Vision: N/A.** FlashInfer is a kernel library (attention/GEMM/MoE/sampling/norm/RoPE/comm); it has
  no vision-tower or multimodal support. The GDN VLM tower in the next Qwen is not a FlashInfer
  concern. (No source claims vision support; the lib simply has no vision operators.)
- **256K: handled by the paged/fa2/cascade stack, no 4090-specific number.** Paged KV with page size
  {16,32,64,128} (XQA) or arbitrary BSR block sizes (fa2), plus Cascade for shared prefixes, scales to
  long context; the 2024 blog exercises seq lengths up to 65536 on the 4090 with decode bandwidth
  utilization "close to 100% for long sequences." No published sm_89 256K end-to-end figure (FlashInfer
  is a library; a 256K run is a property of the host engine, covered in the vLLM/SGLang dossiers).

## Published numbers on a 4090 or similar

**FlashInfer is a kernel library, so there is no standalone "FlashInfer-on-4090 tok/s" — its published
results are kernel-level (TFLops/s for prefill, bandwidth-utilization% for decode) inside a host
engine.** Two FlashInfer sources do benchmark **directly on an RTX 4090 (sm_89)**; the rest are
Hopper/datacenter. All copied as-is.

**A. "Accelerating Self-Attentions for LLM Serving with FlashInfer" (blog, 2024-02-02, Zihao Ye et al.)
— benchmarked on H100 SXM, A100 PCIe, RTX 6000 Ada AND RTX 4090.** FlashInfer **0.0.1**, CUDA 12.3.1,
nvbench cold; baselines FlashAttention 2.4.2 and vLLM v0.2.6; model **Llama2-7B** (32 heads,
head_dim 128), GQA on **llama2-70b tp2** (num_kv_heads 4, num_qo_heads 32).
([blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html))
- **`allow_fp16_qk_reduction` → "could bring 50% speedup on RTX 4090"** for prefill: 4090 tensor cores
  are 2× faster with fp16-accum than fp32-accum, and the `q·kᵀ/√d` term has a small value range, so
  FlashInfer accumulates it in fp16 (score·v stays fp32). (Fig 4: single prefill, Llama2-7B, seq
  32–65535.) **This is the single most sm_89-specific finding in the whole effort.**
- **4090 roofline ridge (tensor-core, fp32-accum) = 163** (165 TFLops/s ÷ 1008 GB/s); prefill becomes
  compute-bound when the query length reaches **256**, which `allow_fp16_qk_reduction` alleviates.
- **Single-request decode: "GPU bandwidth utilization is close to 100% for long sequences"** on the
  4090 (Fig 5, Llama2-7B, seq 32–65536).
- **"split-KV do[es] not improve performance … for RTX Ada 6000 and RTX 4090"** (smaller bandwidth,
  stronger CUDA cores; 32 of 108 SMs still saturates decode bandwidth) — the opposite of A100. So
  single-request decode on a 4090 should **not** split-KV.
- **FP8 decode "up to 2× compared with fp16"** kernels (Fig 11, Llama2-7B, on all 4 GPUs incl. 4090).
- **Fused-RoPE "negligible overhead … especially for RTX 6000 Ada and RTX 4090"** (strong CUDA cores
  handle the sin/cos).
- **Page size "has little effect"** on FlashInfer PageAttention (page indices prefetched into SMEM) —
  so `page_size=1` (SGLang-style) is fine on the 4090.
- (GQA decode via tensor-core/prefill kernel: "3× faster than vLLM PageAttention when batch_size=64",
  llama2-70b tp2 — measured on A100/H100, **not** 4090, but the mechanism applies to the 16 GQA
  layers.)

**B. "Cascade Inference: Memory Bandwidth Efficient Shared Prefix Batch Decoding" (blog, 2024-02-02).**
Shared-prefix batch decode (multi-query prefill over the common prefix in SMEM + single-query decode
over the unique suffix, merged by the LSE operator). **Up to 31× vs vLLM PageAttention and 26× vs
FlashInfer non-cascading on an H100 SXM 80GB**, for **Llama2-7B** (32 heads, 128 dim), shared prefix
32768, batch ≥128, unique suffix ≤256, page size 16; also on A100 PCIe 80GB.
([blog](https://flashinfer.ai/2024/02/02/cascade-inference.html))

**C. Paper — "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving"
(arXiv 2501.01005, MLSys 2025).** Eval on **H100 SXM + Llama 3.1 / Llama 3.1 70B** (not sm_89):
"**29–69% inter-token-latency reduction** compared to compiler backends … **28–30% latency
reduction** for long-context inference, and **13–17% speedup** for LLM serving with parallel
generation." ([arXiv](https://arxiv.org/abs/2501.01005))

**D. FlashInfer-Bench** (https://bench.flashinfer.ai) is an AI-kernel-generation contest leaderboard
"evaluating submitted kernels … on **NVIDIA B200 GPUs**" (arXiv 2601.00227) — Blackwell, not 4090.

**Not found:** any FlashInfer-on-RTX-4090 **end-to-end tok/s** (it is always embedded in vLLM/SGLang/
TRT-LLM/TGI/MLC/LightLLM; those engines' 4090 runs — covered in the vLLM and SGLang dossiers — use
fa2 for the full-attention layers and a Triton/FLA fallback for GDN).

## Ideas we could port into ninfer-4090

All Apache-2.0. The sm_89 value here is **real, 4090-measured attention-kernel technique**, not
Blackwell kernel speed. None of it touches the GDN layers (which we keep as NInfer's own path).

1. **`allow_fp16_qk_reduction` — 50% prefill speedup on the 4090** (blog A, the headline sm_89 win).
   Accumulate the `q·kᵀ/√d` term in fp16 (keep `score·v` in fp32). NInfer's 16 full-attention GQA
   layers' prefill should do exactly this: the 4090's tensor cores are 2× faster with fp16
   accumulation, and the QK product is small-range. Direct, cheap, sm_89-specific, and it lowers the
   compute-bound threshold (query len 256 → ridge 163). **Highest-value, most-verified port in this
   dossier.**
2. **Cascade inference for shared-prefix batch decode** (blog B, up to 31×/26×). Split a batched
   decode into (a) a multi-query prefill over the *shared* prefix with the prefix KV resident in SMEM
   and (b) a single-query decode over the *unique* suffix, then merge the two attention states by the
   associative LSE operator. This is the attention-level twin of NInfer's context-cache/prefix reuse
   and directly serves agentic multi-turn (tool-call loops, shared system prompts, compaction).
   NInfer's paged-KV + one-compact-decode-batch model composes with it: the shared-prefix KV is
   already in the cache; the win is computing it through the high-bandwidth multi-query kernel in
   SMEM instead of re-reading it from global memory once per request.
3. **No split-KV for single-request decode on the 4090** (blog A). FlashInfer's measurement: the 4090
   has strong CUDA cores, so even 32/108-SM decode saturates memory bandwidth — split-KV adds a merge
   step for no gain (unlike A100). NInfer's bs1 decode for the GQA layers should skip split-KV. Cheap
   policy change with a clean measured basis.
4. **Tensor-core (prefill-style) decode for GQA layers** (blog A: "3× faster than vLLM PageAttention at
   batch_size=64", A100/H100, llama2-70b tp2). Use the compute-bound multi-query kernel to run
   grouped-query decode so the 16 GQA layers use tensor cores instead of CUDA cores. Portable idea
   (FlashInfer's fa2 GQA decode); verify on the 4090 since the published 3× is A100/H100.
5. **SMEM page-index prefetch for PageAttention** (blog A: "page size has little effect … pre-fetching
   page indices in GPU shared memory"). Keeps paged-decode performance independent of page size, which
   is what lets a `page_size=1` hybrid-KV scheme (Jenga/Unified-Radix style, see the vLLM/SGLang
   dossiers) run without a page-size penalty on the 4090.
6. **Sorting-free sampling (top-k/top-p/min-p without sorting; chain speculative sampling)**
   (blog 2025-03-10, flashinfer-ai sampling). If NInfer's sampler sorts logits, the sorting-free
   kernels are a drop-in decode-side win; they're arch-generic (no Blackwell dependency) so they apply
   on sm_89.

## Worth running beside NInfer/llama.cpp?

**No, not as an engine — FlashInfer is a kernel library, so there is no standalone "FlashInfer engine"
to run head-to-head.** The correct use is as a **second, independent attention kernel for the 16
full-attention (GQA) layers** to validate NInfer's own full-attention prefill/decode against, and as a
**source of the sm_89 techniques above** (esp. `allow_fp16_qk_reduction` and cascade). Concretely:
- It will **not** touch the GDN layers on a 4090 (SM90+ only), so it can't serve as a whole-model
  reference for the hybrid the way vLLM/SGLang can; it is a partial reference (GQA attention + GEMM +
  sampling only).
- A fair comparison, if we do it: **same groupwise-int or FP8 checkpoint, single RTX 4090, batch 1–8**,
  and (a) numerically cross-check NInfer's full-attention prefill/decode tokens against FlashInfer's
  fa2 on the 16 GQA layers (like-for-like at the arch), and (b) measure the 4090-specific knobs —
  fp16-QK-accum on/off, cascade on/off for a shared-prefix agentic trace, split-KV on/off at bs1 —
  on NInfer's kernels, using FlashInfer's 4090 numbers (A above) as the target to hit. That turns
  FlashInfer from "another engine" into a concrete checklist of sm_89 attention wins to replicate.

## Risk / unknowns

- **The GDN gap is the whole story on a 4090.** Every Gated-DeltaNet kernel in FlashInfer is SM90/
  SM100/SM120-only (`get_device_capability()[0] >= 9`; target-verify `sm_major in (9,10)`). The next
  dense Qwen is 48/64 GDN layers; FlashInfer accelerates only the 16 GQA layers on sm_89. Anything we
  port must not assume FlashInfer covers the recurrence.
- **The 4090 blog numbers are FlashInfer 0.0.1 / CUDA 12.3.1 / Llama2, kernel-level (2024).** They are
  the only sm_89-4090 published data, but they predate the current release by ~30 minor versions and
  use Llama2 (32-head MHA) not Qwen GQA head counts; treat them as direction + the fp16-QK/cascade/
  no-split-KV mechanisms, not as a precise throughput to match. The `allow_fp16_qk_reduction` 50%
  figure is the robust one (it's a hardware property of the 4090, not a version artifact).
- **XQA / trtllm-gen / FMHA / FP4 / b12x / GDN-MTP are all newer-arch.** Do not plan an sm_89 build
  around them; the only sm_89 attention backend is fa2 (+cuDNN), and FP4/NVFP4 does not exist on Ada.
- **Fast churn + breaking API changes** (v0.6.18 → v0.7.0 removed several wrappers and changed
  `mamba.checkpointing_ssu`'s signature; ~450 PRs/release). A pinned-version reference test needs a
  fixed tag; the Python API surface moves under you.
- **NVIDIA has taken the wheel** (2026 backends are all NVIDIA-authored Blackwell `cake_*`/`primTS`
  work). The project's center of gravity is datacenter Blackwell; sm_89 is maintained (fa2 still
  works) but is not a frontier target — expect no new sm_89-specific attention kernels to appear.
- The **Cascade** and **fp16-QK** wins are real on the 4090 but are *attention-kernel* gains; on a
  GDN hybrid they only apply to 16/64 layers, so the end-to-end model-level gain is a fraction of the
  headline kernel numbers.

## Sources

- https://github.com/flashinfer-ai/flashinfer (README: kernel-library scope, GPU matrix w/ Ada SM8.9,
  adoption SGLang/vLLM/TRT-LLM/TGI/MLC/LightLLM/lorax/ScaleLLM, Apache-2.0, CUDA 12.9/13.0/13.4,
  pip install, news v0.4.0 Blackwell + sampling blog, citation)
- https://github.com/flashinfer-ai/flashinfer/releases (v0.7.0 2026-09-22; v0.7.0.post1; v0.7.1rc1..5,
  v0.7.2rc1; v0.6.18)
- https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0 (full notes: Qwen 3.6 fused GDN decode
  sm120 #4481, Qwen 3.6 35B-A3B fused_GDN_step sm120 #4708, gdn prefill/decode/mtp cake sm100/103
  #4581, Kimi K3 KDA sm120 #4605/#4633/#4709, gdn pooled-state/checkpointing #4436, mamba
  checkpointing_ssu, gating_delta_rule_mtp, autotune_v2, MoE EP, PCIe-IPC allreduce, FP8 small-batch
  GEMM notice, API removals)
- https://docs.flashinfer.ai/index.html (v0.7.0.post1 docs; FlashInfer = "kernel library and generator
  for LLM serving")
- https://docs.flashinfer.ai/api/gdn_decode.html (gated_delta_rule_decode / _pretranspose / _mtp —
  "Mamba-2 / GDN-style sequence models")
- https://docs.flashinfer.ai/generated/flashinfer.xqa.xqa.html (XQA paged-KV decode API; sm90-fp8 spec
  note; page_size {16,32,64,128})
- https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/jit/xqa.py (`supported_major_versions
  = [9, 10, 12]`; MLA xqa `= [12]`)
- https://github.com/flashinfer-ai/flashinfer/issues/3170 (maintainer SM121 support audit: per-backend
  CC gates — fa2 generic, fa3 sm90-only, xqa {9,10,12}, trtllm-gen sm100-only, cutlass-FMHA sm100/110,
  FMHAv2 sm12x; BF16 GEMM cudnn [80,86,89,90,100,103]; FP8 MoE "may fall back to SM89 tactics")
- https://github.com/flashinfer-ai/flashinfer/issues/1147 (SM120 support: "SM120 kernels will follow
  sm80 kernel templates … lowering to fa2")
- https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py
  (FlashInfer GDN gate: `get_device_capability()[0] >= 9`; "SM90 … SM100 … full support — decode,
  prefill, MTP"; `supports_target_verify = sm_major in (9, 10)`; Triton fallback)
- https://arxiv.org/abs/2501.01005 (FlashInfer paper, MLSys 2025; §3.2 "FA2 for architectures up to
  Ada(sm89), FA3 for Hopper"; "Ada(sm89) has limited shared memory"; eval on H100 SXM + Llama 3.1 /
  3.1-70B: 29–69% ITL, 28–30% long-context, 13–17% parallel-gen)
- https://flashinfer.ai/2024/02/02/introduce-flashinfer.html (blog: 4-GPU bench incl. RTX 4090 sm_89
  + RTX 6000 Ada + A100 + H100; Llama2-7B/70b; FlashInfer 0.0.1; `allow_fp16_qk_reduction` ~50% on
  4090; 4090 roofline ridge 163; no split-KV on 4090; FP8 2×; fused-RoPE ~0; page-size insensitivity)
- https://flashinfer.ai/2024/02/02/cascade-inference.html (Cascade: up to 31× vs vLLM PageAttention,
  26× vs FlashInfer non-cascade, H100 SXM 80GB, Llama2-7B, prefix 32768, batch ≥128; also A100)
- https://flashinfer.ai/2025/03/10/sampling.html (Sorting-Free GPU Kernels for LLM Sampling; Xing UWe,
  Ye UW/NVIDIA, Hou CMU, Ceze UW/NVIDIA, Chen CMU/NVIDIA)
- https://sampl.cs.washington.edu/projects/flashinfer.html (UW SAMPL group project page)
- https://bench.flashinfer.ai (FlashInfer-Bench kernel contest on B200; arXiv 2601.00227)
