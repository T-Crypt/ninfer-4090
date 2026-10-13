# Dossier: KTransformers (MADSys @ Tsinghua + Approaching.AI)

Researched 2026-10-07 (public sources only: GitHub repo `kvcache-ai/ktransformers`, docs, release notes,
issues, the SOSP'25 paper, the LMSYS integration blog, the LMSYS-adjacent docs site ktransformers.net, and
PyPI). No installs, no weight downloads, no GPU jobs.

## What it is / backing / license

- **What it is:** a **CPU–GPU heterogeneous inference (and fine-tuning) framework for large MoE models**. The
  core idea (Fiddler-style, per the SOSP'25 paper): attention + shared experts + dense layers run on the GPU;
  **routed MoE experts persist in CPU DRAM and are computed on the CPU** (no weight transfer over PCIe).
  Today the repo ships two products from the `kt-kernel` source tree: **inference** (kt-kernel CPU expert
  kernels + the kvcache-ai fork of SGLang, `sglang-kt`) and **SFT** (KTransformers × LLaMA-Factory for MoE
  fine-tuning on consumer hardware). The original integrated framework (HuggingFace Transformers injection,
  `balance_serve`, the 2024–2025 DeepSeek-R1/671B-on-4090 work) is **archived** under `archive/`.
- **Backing:** **MADSys Lab @ Tsinghua University** + **Approaching.AI** + 9#AISoft + community (README
  "Developed and maintained by"). SOSP'25 authors span Tsinghua, Approaching.AI, UESTC, Hangzhou Dianzi,
  BUPT. 19,569 stars / 1,591 forks / 518 open issues ([repo](https://github.com/kvcache-ai/ktransformers)).
  Since Oct 2025 the CPU kernels are **upstreamed into SGLang** (roadmap issue
  [sglang#11425](https://github.com/sgl-project/sglang/issues/11425);
  [LMSYS blog 2025-10-22](https://www.lmsys.org/blog/2025-10-22-KTransformers/)).
- **License:** **Apache-2.0** (repo license field; same for the `kvcache-ai/sglang` fork and the archived
  original). **Porting code is license-OK** (no GPL, no non-commercial).

## Repo, version, last release date

- Repo: <https://github.com/kvcache-ai/ktransformers>; docs site <https://ktransformers.net>; SGLang fork
  <https://github.com/kvcache-ai/sglang> (14 stars, Apache-2.0); PyPI packages `kt-kernel`, `sglang-kt`,
  `ktransformers`.
- **Latest release: v0.7.1, 2026-09-15** ("Qwen VLM and Kimi LoRA Fine-Tuning": BF16 image–text LoRA for
  Qwen3-VL-30B-A3B-Instruct and Qwen3.5-35B-A3B; native RAWINT4 Kimi K2.5/K2.6) ([release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.1);
  PyPI `kt-kernel` latest = 0.7.0.post4).
- Prior: **v0.7.0, 2026-08-17** (full AVX512 LoRA SFT without AMX, native block-FP8 LoRA — host memory for
  DeepSeek-V3.1 LoRA ~1.4 TB BF16 → ~800 GB FP8, Qwen3-VL MoE SFT, DeepSeek-V4 Docker guide; release
  validation incl. "Qwen3.5-397B-A17B BF16 LoRA on two GPUs") ([release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.0));
  v0.6.4 (2026-07-23, PyPI `ktransformers`); v0.6.1 (2026-04-30, doc refresh: separate Inference and SFT
  entry points).
- Cadence: very fast (2026: v0.6.1 → v0.7.1 in ~5 months; monthly model day-0s: Kimi-K2.5 Jan, GLM-5 Feb,
  DeepSeek-V4-Flash May, GLM-5.2 Jun, MiniMax-M3 Jun, DeepSeek-V4-Flash-on-Ascend Aug, GLM-5.3-flash Aug).
- Toolchain: `pip install kt-kernel` (prebuilt manylinux wheels, Python 3.10–3.12, static CUDA runtime,
  no CUDA toolkit needed); source build via `./install.sh` (auto-detects AMX/AVX512, `-march=native`).
- **Qwen coverage today** (docs status page, [ktransformers.net Qwen models](https://ktransformers.net/en/docs/supported-models/qwen)):
  **Qwen3-Coder-Next** (SGLang-KT, FP8/BF16 — "Needs smoke"), **Qwen3.5** (SGLang-KT, BF16 — "Needs mainline
  cleanup and smoke"), **Qwen3-30B-A3B** (BF16/AMXINT8/LLAMAFILE — "Needs method-specific smoke"); SFT
  candidates Qwen3-235B-A22B and Qwen3.5-397B-A17B. Earlier: Qwen3-Next-80B-A3B (Sept 2025),
  Qwen3-235B/30B-A3B (AMX day-0, Apr 2025). **All of KTransformers' Qwen support is MoE or MoE-hybrid —
  there is no dense-model offload path at all** (see below).

## Runs on an RTX 4090 (sm_89)?

**Yes — sm_89 is a first-class target, and most of the published numbers are 4090-based. But the engine only
offloads MoE routed experts, so a dense model on a 4090 gets nothing from KTransformers beyond "a fork of
SGLang".**

- The `kt-kernel` wheel explicitly supports **"SM 80/86/89/90 (Ampere, Ada, Hopper)"**; GPU compatibility
  matrix: "**Ada Lovelace | 8.9 | ✅ | RTX 4090, 4080, 4070**" (single wheel, static CUDA runtime,
  CUDA 11.8+/12.x drivers) ([kt-kernel README](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)).
- The GLM-5.3-Flash tutorial states the current implementation supports "**NVIDIA SM89 and SM120 GPUs
  (RTX 40 and 50 series)**" ([GLM-5.3-Flash tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)).
- The legacy 671B-on-desktop work was benchmarked on **RTX 4090 / 4090D + Xeon Gold 6454S** (see numbers
  below); the Qwen3-Coder-Next tutorial's *recommended* config is "**1 x NVIDIA RTX 4090 24 GB** + AVX512
  CPU + ≥100 GB RAM" ([Qwen3-Coder-Next tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)).
- Caveats on sm_89: attention runs on the SGLang fork's **triton/flashinfer** backends — DeepGEMM
  (sm90/sm100) and flashmla are not usable (user-confirmed in [sglang#11425 comments](https://github.com/sgl-project/sglang/issues/11425));
  the GDN/linear-attention layers of a Qwen3-Next-class model run through SGLang's Triton path, and the
  legacy framework notes "Due to Qwen3-Next's use of linear attention, CUDA Graph optimization is not yet
  supported" ([Qwen3-Next tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3-Next.md)).
  A user's DeepSeek-V4-Flash-0731 **MTP** launch on **avx2 + sm89** (8 GPUs) fails to start
  ([issue #2127](https://github.com/kvcache-ai/ktransformers/issues/2127)) — sm_89 is supported but the
  spec-decode path is rough on it.

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: YES for MoE-hybrid Qwens, via SGLang-KT; NO for dense models.**
  - **Qwen3-Next-80B-A3B** (GDN + full-attention hybrid MoE) has been supported since **2025-09-11** (legacy
    `balance_serve`; ~320 GB RAM + 6 GB GPU for the 512-expert variant, `cache_lens 32768`)
    ([Qwen3-Next tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3-Next.md)).
  - **Qwen3-Next-80B-A3B-Instruct-FP8** runs on the new stack: the CPU–GPU expert-scheduling tutorial's
    benchmark model, 4×RTX 4090 TP4 (numbers below) ([expert-sched tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)).
  - **Qwen3-Coder-Next (80B-A3B)** has a current single-4090 recipe (FP8 or BF16, triton attention, 256K KV)
    ([Qwen3-Coder-Next tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)).
  - **Qwen3.5 (MoE-400B)** tutorial on 4×RTX 4090 + 800 GB RAM, BF16, on the `qwen3.5` branch
    ([Qwen3.5 tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3.5.md)).
  - Non-Qwen GDN/linear hybrids also land: GLM-5.3-flash (34 linear-attention + 11 DSA layers), MiniMax-M2.x,
    DeepSeek-V4-Flash.
  - **Dense Qwen (27–32B): not supported as an offload target.** kt-kernel is a `KTMoEWrapper` — routed
    expert kernels only; everything else must fit on GPU. A dense model on a 4090 through this stack is just
    SGLang (the fork), with zero KTransformers contribution. (The SOSP paper is likewise explicitly
    "for MoE models": dense offload is out of scope.)
- **MTP: partial, shaky on sm_89, no published gains.** `sglang-kt` exposes SGLang's speculative stack;
  **as of v0.6.4 it supports `EAGLE`/`NEXTN` only** — DSpark (DeepSeek-V4-Flash-0731's new draft head) is
  not ported, "forcing users to disable speculative decoding entirely (~2x decode speed loss)"
  ([issue #2118](https://github.com/kvcache-ai/ktransformers/issues/2118)). Open crash reports: MTP launch
  failure on avx2+sm89 ([#2127](https://github.com/kvcache-ai/ktransformers/issues/2127)), MTP crash on KT
  0.6.2 ([#2009](https://github.com/kvcache-ai/ktransformers/issues/2009)), EAGLE3-on-Kimi crash
  ([#2007](https://github.com/kvcache-ai/ktransformers/issues/2007)), standalone-draft segfault
  ([#1815](https://github.com/kvcache-ai/ktransformers/issues/1815)). Qwen3-Next/Coder-Next MTP heads would
  go through the NEXTN/EAGLE path; **no published MTP numbers on a 4090** (not found).
- **Vision: partial (non-Qwen) inference, Qwen VLM only via fine-tuning.** GLM-5.3-flash (Zhipu) runs
  "text, multiple images, video" + tool calling on SM89/SM120 consumer GPUs (Aug 26, 2026; ≤8 images or 1
  video per request; FP8 weights ~306 GiB in ≥350 GB RAM) ([GLM-5.3-Flash tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)).
  Qwen VLM (Qwen3-VL-30B-A3B-Instruct, Qwen3.5-35B-A3B) appears only as **SFT/LoRA** targets in v0.7.0/v0.7.1
  ([v0.7.1 release](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.1)); no published Qwen-VL
  *inference* numbers via kt-kernel (not found).
- **256K: yes, demonstrated.** The Qwen3-Coder-Next single-4090 launch sets **`--max-total-tokens 256000`**
  (256K KV) with `--chunked-prefill-size 16384` ([tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md));
  GLM-5.3-flash is validated at `--context-length 501025` (≈501K, up to 1M native); a user runs
  DeepSeek-V4-Flash-0731 with `--context-length 1048576` on sm89 ([#2127](https://github.com/kvcache-ai/ktransformers/issues/2127)).
  Legacy 24 GB recipes reached 139K context for DeepSeek-V3/R1 (Mar 2025 README update). The 3-layer
  **GPU→CPU→disk prefix cache** (PhotonLibOS `kvc2`) is for long-context KV reuse
  ([prefix_cache doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/prefix_cache.md)).

## Published numbers on a 4090 or similar

All copied verbatim from the linked sources. "4090-class" = the published RTX 4080/5090/L20 cells are noted
as such; I have not converted or averaged anything.

**A. Legacy framework — DeepSeek-V3/R1 671B on 1×(or up to 4×) RTX 4090/4090D + dual Xeon Gold 6454S (32c×2)
+ 382 GB/1 TB DRAM** ([DeepseekR1_V3_tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md)):
- V0.2 (q4km, 500-token prompt): dual-socket KTrans 6-experts **prefill 97.32 / decode 13.69 tok/s**; 8
  experts 82.94 / 12.208; single-socket 6 experts 65.14 / 10.303; 8 experts 54.21 / 8.73; llama.cpp 8 experts
  **10.31 / 4.51**. "The highest speedup reaches up to 3.03x in decoding and 9.44x in prefill."
- V0.2.1 (4090, q4km, 6 experts): prefill 13 / 111 / 112.5 / 102 / 101 and decode **16.8 / 15.9 / 15.4 /
  14.9 / 13.9** tok/s for 2 / 1K / 2K / 4K / 8K-token prompts (300-token outputs); 8 experts: decode 13.4 /
  13.5 / 13.4 / 13.2 / 12.4. Memory: single socket 382 GB DRAM, **≥14 GB VRAM**; dual socket 1 TB, ≥14 GB VRAM.
- V0.3-preview (1–4× 4090D, 644 GB DRAM): prefill 185.96 / 255.26 / 252.58 / 195.62 (8 experts) and
  203.70 / 286.55 / 271.08 / 207.20 (6 experts) at 1K / 2K / 4K / 8K. "up to 3.45x faster than KTrans V0.2,
  and up to 27.79x faster than llama.cpp."
- README: "Support Deepseek-R1 and V3 on single (24GB VRAM)/multi gpu and 382G DRAM, **up to 3~28x
  speedup**" (Feb 10, 2025); later "Longer Context (from 4K to 8K for 24GB VRAM) & Slightly Faster Speed
  (+15%, up to 16 Tokens/s)" (Feb 15, 2025).

**B. Qwen3MoE (235B-A22B / 30B-A3B) + RTX 4090, KTransformers 0.3 + AMX** ([AMX.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md), Apr 29, 2025):
- Two test rigs: "Server CPU (Xeon 4) + RTX 4090" and "Consumer-grade CPU (Core i9-14900KF +
  dual-channel DDR5-4000 MT/s) + RTX 4090."
- "we achieve **up to 347 tokens/s prefill** performance in the workstation scenario"; "On consumer-grade
  CPUs, we're able to run the large model (235B-A22) and deliver smooth performance on the smaller 30B-A3B."
- "our kernel can achieve **21 TFLOPS** of BF16 throughput and **35 TOPS** of Int8 throughput on Xeon4 CPUs
  — about 4× faster than PyTorch's general AMX kernel. For DeepSeek-V3, pairing a Xeon4 CPU with a single
  RTX 4090 GPU achieves **418 tokens/s end-to-end throughput**."
- Pre-AMX (v0.2, Xeon 4 + 4090): DeepSeek-V3 prefill "only 91 tokens/s."

**C. New framework (kt-kernel + SGLang-KT) — Qwen3-Next-80B-A3B-Instruct-FP8 on 4× RTX 4090** (Jan 22, 2026
feature; [expert-sched tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)):
"measured … with 4 x RTX 4090, Intel Xeon Gold 6454S, tensor parallel size 4, using ShareGPT dataset" —
throughput (tokens/s) by GPU-expert ratio × placement strategy:

| GPU expert ratio | random | uniform | front-loading | frequency | dynamic-expert-update |
|---|---|---|---|---|---|
| 0% | 53.01 | 52.96 | 54.18 | 52.72 | 53.37 |
| 10% | 56.63 | 56.57 | 57.18 | 58.60 | 70.22 |
| 20% | 58.75 | 60.28 | 58.82 | 61.92 | 74.73 |
| 30% | 62.86 | 62.08 | 63.87 | 66.50 | 75.55 |
| 40% | 66.81 | 66.82 | 67.45 | 72.78 | 80.98 |
| 50% | 70.38 | 65.25 | 73.65 | 76.19 | 81.17 |
| 60% | 71.33 | 72.80 | 77.95 | 82.33 | 82.30 |
| 70% | 74.40 | 76.17 | 81.59 | 89.37 | 88.70 |
| 80% | 79.71 | 79.20 | 89.20 | 100.67 | 92.31 |
| 90% | 88.82 | 81.06 | 98.14 | 107.15 | 95.04 |
| 100% | 112.61 | 112.32 | 111.82 | 114.26 | 112.99 |

  Minimum config for the feature: "1 x NVIDIA RTX 4090 24 GB + AVX512 CPU + ≥256 GB RAM" — i.e. a single
  4090 is the documented floor, but the published table is 4×4090.

**D. LMSYS integration blog (2025-10-22)** ([blog](https://www.lmsys.org/blog/2025-10-22-KTransformers/)):
- Single-GPU + CPU: dual-socket Xeon Platinum 8452Y (36c×2, 1 TB DDR5) + **A100 40 GB (full-precision) /
  RTX 4080 16 GB (quantized)**; models DeepSeek-V3-0324, DeepSeek-V2.5-1210, Qwen2-57B-A14B. Prefill:
  "speedups of up to **20×**" vs llama.cpp/Fiddler across prompt lengths; decode: "up to **4× speedup**";
  CPU MoE kernel "achieves **21.3 TFLOPS** on DS-3, a 3.98× improvement over the PyTorch baseline"; decode
  "2.42×–4.09× speedups over Fiddler and 1.25×–1.76× over Llama.cpp on full-precision models. With
  quantized models … 1.77×–1.93× vs. Llama.cpp"; "reduces GPU launch overhead from over 20% to nearly zero."
- Multi-GPU + CPU: int4 DeepSeek-V3, **8×L20 + Xeon Gold 6454S** (128-in / 512-out workloads): 8-way
  concurrency "achieves a **264% throughput** gain compared to 1 GPU … each request achieves nearly 20
  tokens per second on average."
- ShareGPT, **8×L20 + Xeon Gold 6454S, DeepSeek-R1-0528 FP8** (1000 requests, 301K in / 188K out):
  **total 227.85 tok/s, output 87.58 tok/s, 0.46 req/s, mean ITL 431.61 ms** (same table in the README).
- NUMA-aware tensor parallelism: "up to **63%** decoding throughput improvement on dual-socket servers."
  Expert Deferral: "up to **1.45×** higher decoding throughput, with accuracy variation below 0.5%."

**E. SOSP'25 paper** ("KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference for MoE
Models", ACM SOSP '25, DOI [10.1145/3731569.3764843](https://doi.org/10.1145/3731569.3764843),
[pdf](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)):
- Abstract: "**4.62–19.74× prefilling speedups and 1.25–4.09× decoding speedups** compared to existing
  systems"; Expert Deferral "increasing CPU utilization from typically below 75% to almost 100% … up to
  **1.45×** additional throughput … average model accuracy drop of no more than 0.5%".
- Setup: dual Xeon Platinum 8452Y (36c), 1 TB DDR5 (220 GB/s intra-socket, 125 GB/s cross-socket), **A100
  40 GB or RTX 4080 16 GB, PCIe 4.0 (32 GB/s)**. Fiddler baseline on this box for DeepSeek-V3: "70.02
  tokens per second during prefill phase and 4.68 tokens per second during decode phase," GPU utilization
  <30%.
- AMX kernel: "up to **21.3 TFLOPS** on a single-socket CPU, delivering a **3.98× speedup** over … oneDNN
  based PyTorch"; MoE-layer microbenchmarks "1.69–4.30× speedups"; adaptive AVX-512/AMX switching "up to
  1.20× … in decode … up to 10.81× … in prefill"; fused-MoE + dynamic scheduling "up to 1.83× … in prefill".
- Decode of 671B DS-3 "remains limited to **5.87 tokens per second**" before Expert Deferral; with deferral
  "CPU and GPU utilization from 74%/28% to 100%/37% … a **33%** throughput increase"; NUMA-aware TP "up to
  **1.63×** … decode" (1.22× prefill); single-CUDA-Graph decode "up to **1.23×**".
- Accuracy (Table 2): DS-3 (8+0) 83.0 / 71.2 / 94.8 / 83.0 → (2+6) 83.0 / 70.2 / 95.2 / 82.9 (HumanEval /
  MBPP / GSM8K / StrategyQA); QW-2 (8+0) 65.7 / 52.4 / 84.7 / 83.6 → (4+4) 67.4 / 53.4 / 83.4 / 82.5; DS-2
  (6+0) 80.5 / 67.6 / 93.3 / 79.7 → (2+4) 82.5 / 66.8 / 92.8 / 80.4. LiveBench DS-3, 6 experts affected:
  deferral −0.5% vs expert-skipping −13.3%.
- Implementation: "11,000 lines of C++ extensions … 2,000 lines of Python"; "trillion-parameter-scale MoE
  models on a single server with only one consumer-grade GPU."

**F. Qwen3-Coder-Next (80B-A3B, GDN-hybrid MoE)** ([tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md)):
- Recommended: 1× RTX 4090 24 GB + AVX512 CPU + ≥100 GB RAM (FP8 weights 80.4 GB); `--kt-num-gpu-experts
  100` (FP8) / 60 (BF16); `--max-total-tokens 256000`; triton attention; CUDA 12.8+ "for best FP8 support."
- Published benchmark (single concurrency, prefill / decode tps): "**1 x RTX 5090 (32 GB), 2 x AMD EPYC
  9355, PCIe 5.0, FP8**: 64 tokens **362 / 75.9**; 2048 tokens **1746 / 75.6**; 8192 tokens **2407 / 69.1**;
  32768 tokens **6233 / 51.7**." **No 4090 row in the table** (not found for a 4090).

**G. SFT on 4090 (inference-adjacent, different workload)** (README SFT table): "Qwen3-30B-A3B | ~24 GB
total | **8+ it/s** | 1x RTX 4090"; "DeepSeek-V3 / DeepSeek-R1 | ~80 GB | 3.7 it/s | 4x RTX 4090."

**Not found:** a 4090 end-to-end number for a *dense* Qwen (by construction — KT has no dense offload
path); 4090 MTP/spec-decode decode tok/s; a 4090 ShareGPT number for Qwen3-Next (only 4×4090).

## Ideas we could port into ninfer-4090

License is Apache-2.0, so all of it is portable in principle. Split by relevance to our **dense** target:

**Directly useful even for dense NInfer:**
1. **3-layer (GPU → CPU DRAM → NVMe) prefix cache** (PhotonLibOS `kvc2`, page_size 16 KV pages, `cpu_memory_size_GB`,
   `disk_path`) — for 256K agentic reuse when KV no longer fits in 24 GB. Complements NInfer's existing
   context-cache with host/disk tiers instead of eviction. ([prefix_cache doc](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/prefix_cache.md))
2. **Whole-decode single CUDA Graph across host callbacks** — KT wraps the CPU submit/sync barriers in
   `cudaLaunchHostFunc` so "the entire decode path for one token now fits in a single CUDA Graph, …
   improving decoding speed by up to 1.23×" and cutting GPU launch overhead from >20% to ~0 (SOSP §3.3).
   If NInfer ever moves any per-token work off the device (or keeps it), this is the generic recipe for
   keeping the decode graph unbroken.
3. **ARIs-based kernel selection** (heavy matrix-matrix path for prefill, light vector-matrix path for
   decode, same memory layout) — the *principle* maps to our GDN recurrence: pick the chunked/kernel
   variant by tokens-per-iteration rather than one fused kernel for everything (SOSP §3.2, Fig. 7: AMX
   worse than AVX-512 at ≤4 tokens/expert).

**Only useful if the next Qwen is MoE (see the qwen4_exp signal below) or if we add expert offload:**
4. **Expert Deferral** (SOSP'25): compute a subset of a layer's routed experts on the CPU concurrently
   with the *next* layer's GPU attention; "up to 1.45× decoding … accuracy drop no more than 0.5%."
   In the SGLang-KT product form: `--kt-max-deferred-experts-per-token` (1–4 recommended; 5–7 "may
   introduce noticeable accuracy loss"). For a dense model there are no routed experts — N/A.
5. **NUMA-aware tensor parallelism for CPU-resident weights** (column/row-slice every expert across
   sockets, local compute + reduce-scatter; up to 1.63× decode on dual-socket) — NInfer is single-GPU and
   currently single-socket-agnostic; relevant only if we add DRAM-resident weights on a multi-socket box.
6. **Hotness-aware GPU expert placement**: `uniform` / `frequency` (from recorded routing stats) /
   `front-loading` / `random` + **dynamic re-placement from prefill routing statistics** — the table in C
   shows 100% on-GPU ≈ 112–114 tok/s vs ~53 tok/s at 0% on 4×4090; dynamic update wins at low GPU ratios
   (10%: 53.37 → 70.22). A dense model has no experts to place.
7. **AMX/AVX-512 MoE kernels with tiling-aware memory layout** (21.3 TFLOPS CPU-side, 3.98× over oneDNN)
   and **layerwise GPU prefill + CPU expert streaming** (`--kt-gpu-prefill-token-threshold`) — the
   offload engine proper; NInfer has no CPU path today and the dense target doesn't need one.

## Worth running beside NInfer/llama.cpp?

**Conditional yes — its entire relevance hinges on what the next Qwen is.**

- **If Qwen4 is a MoE hybrid (increasingly likely, see risks):** **yes, strongly.** It is the only engine
  with an official single-4090 recipe for exactly our architecture class (Qwen3-Next / Coder-Next /
  Qwen3.8-Flash-Next), and — uniquely among our targets — **Qwen's own model card for Qwen3.8-Flash-Next
  recommends KTransformers** ([issue #2179](https://github.com/kvcache-ai/ktransformers/issues/2179)). A
  fair test: same FP8 checkpoint, **1× RTX 4090 + our CPU/RAM** (the recipe asks for AVX512 CPU + 100–300 GB
  DRAM), batch 1–8, ShareGPT/MTBench; compare KTransformers (sglang-kt) vs llama.cpp MoE-offload vs
  NInfer, reporting prefill tok/s, decode tok/s, ITL, and the 4090 decode floor. The 5090 table in F is
  the shape of the comparison (6233/51.7 at 32K prompt on 5090).
- **If Qwen4 is dense 27–32B and fits in 24 GB:** **no.** KTransformers offloads only routed experts; a
  dense model through it is bare SGLang-fork with zero added value, and its SFT product is irrelevant to us.
  llama.cpp remains the correct dense offload cross-check.
- Either way it is the **reference implementation of the CPU-expert-offload scheduling ideas** (deferral,
  NUMA, single-graph decode) that we would port into NInfer only if Qwen4 needs DRAM offload.

## Risk / unknowns

- **The target may not be dense at all.** [Issue #2179](https://github.com/kvcache-ai/ktransformers/issues/2179)
  (2026-08-28) documents **Qwen3.8-Flash-Next** (HF `model_type: qwen4_exp`, released 2026-08-26) as a
  "Qwen4 architecture preview": **125B main model, 6B active/token, 48 layers, 512 routed experts + 1
  shared, top-10 routing, 51B n-gram PLE embedding table, 4B MTP module; 36 GDN + 12 QSA sparse-attention
  layers**; "Qwen's own model card recommends KTransformers for production serving of this model." If the
  next dense Qwen ships *alongside* a Qwen4 MoE line, KTransformers' offload path (and the whole
  >24 GB problem this target was filed for) becomes central, not tangential. Cross-check against the
  Qwen-official dossier before betting on it.
- **Dense blind spot:** no KT feature touches a dense model's weights; if Qwen4-dense > 24 GB, KT is a
  dead end and llama.cpp-style dense offload is the only route.
- **CPU is the real hardware:** the strong numbers assume Xeon AMX/AVX512 + 256 GB+ DRAM (4090 alone
  doesn't help; the GPU holds only attention + shared experts + a few hot experts). A 4090 with a
  consumer CPU + 64 GB RAM will be far below the published tables.
- **MoE-only by construction** (SOSP paper: "for MoE models"; kt-kernel = `KTMoEWrapper`); do not read the
  671B-on-4090 press as dense-model capability.
- **sm_89 spec-decode is rough**: MTP launch/crash reports open on avx2+sm89 (#2127, #2009), DSpark not
  ported to sglang-kt (#2118), EAGLE3-on-Kimi crash (#2007); legacy Qwen3-Next path had no CUDA graphs.
- **Docs status is "needs smoke"** for all current Qwen inference entries (Qwen3-Coder-Next, Qwen3.5,
  Qwen3-30B-A3B) on the official docs page; Qwen3.5 lives on a `qwen3.5` branch, not main.
- **Rapid architecture churn:** DeepSeek-V4-Flash-0731 changed its draft-head layout (NextN → DSpark
  `mtp.0/1/2` + confidence/markov heads) and broke sglang-kt; expect the same for Qwen4 (QSA, Gated
  Residual, PLE are all new vs Qwen3-Next) — the day-0 lag risk is on the SGLang model graph
  (sglang#36497/#36585), not on KT itself.
- **The legacy framework is archived** (Qwen3-Next via `balance_serve` requires the `archive/` code and
  ~320 GB RAM + 6 GB GPU); the current stack is SGLang-fork based, which is also a big moving target.
- **4090-specific data is thinner than the 4090D-era data**: the new-framework 4090 table is 4×4090 TP4;
  the single-4090 published benchmark is on a 5090; legacy 4090 numbers are 2025-era DeepSeek, not Qwen
  hybrids.

## Sources

- https://github.com/kvcache-ai/ktransformers (README: two products — kt-kernel Inference + SFT; updates
  feed; Apache-2.0; 19.6k stars; MADSys Tsinghua + Approaching.AI + 9#AISoft; SOSP'25 bibtex; DeepSeek-R1
  8×L20 perf table; Qwen3-30B-A3B 1×4090 SFT table; archive split)
- https://github.com/kvcache-ai/ktransformers/releases (v0.7.1 2026-09-15; v0.7.0 2026-08-17; v0.6.4
  2026-07-23; v0.6.1 2026-04-30)
- https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.1 (Qwen VLM MoE SFT: Qwen3-VL-30B-A3B-
  Instruct, Qwen3.5-35B-A3B; Kimi K2.5/K2.6 RAWINT4 LoRA; PR #2156)
- https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.0 (AVX512-without-AMX LoRA; native block-
  FP8 LoRA, DeepSeek-V3.1 host memory ~1.4 TB → ~800 GB; Qwen3-VL SFT; release validation incl.
  Qwen3.5-397B-A17B BF16 LoRA 2-GPU; ktransformers[sft]==0.7.0 / transformers-kt 5.6.0.post2)
- https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md (wheel = SM 80/86/89/90;
  GPU matrix "Ada Lovelace 8.9 ✅ RTX 4090, 4080, 4070"; 6 CPU variants AMX→AVX2; SGLang-KT integration;
  Qwen3-30B-A3B complete example on 1× RTX 4090 + 2× Xeon 6454S; KT-Kernel parameter table: deferred
  experts, GPU expert placement strategies, dynamic expert update, prefill thresholds)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md
  (2026-01-22 feature; 4×4090 + Xeon 6454S + 512 GB, TP4, ShareGPT throughput table for
  Qwen3-Next-80B-A3B-Instruct-FP8; min config 1×4090 + AVX512 + 256 GB)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/Qwen3-Coder-Next-Tutorial.md
  (1×4090 recommended; FP8 80.4 GB weights; --max-total-tokens 256000; 5090/EPYC-9355 benchmark
  362/75.9 … 6233/51.7 prefill/decode)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md
  (2026-08-26; SM89+SM120; 321B, 34 linear-attn + 11 DSA layers; FP8 ~306 GiB + ≥350 GB RAM; context
  501025 validated, 1M native; multimodal ≤8 images or 1 video; single-GPU launch with
  --kt-num-gpu-experts 0)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3-Next.md (2025-09-11;
  Qwen3-Next-80B-A3B-Thinking/Instruct; ~320 GB + 6 GB GPU; balance_serve; no CUDA graphs for linear
  attention yet)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Qwen3.5.md (Qwen3.5 MoE-400B; 4×4090 +
  800 GB RAM + ~800 GB storage; qwen3.5 branch; BF16; --kt-num-gpu-experts 1, TP4)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md (2025-02; 4090/
  4090D + Xeon 6454S + 382 GB/1 TB DRAM; V0.2/V0.2.1/V0.3 bench tables vs llama.cpp; 14 GB min VRAM;
  139K-context recipe; FP8 kernel + hybrid quantization)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/AMX.md (2025-04-29; Qwen3-235B-A22B /
  30B-A3B day-0; Xeon 4 + 4090 up to 347 tok/s prefill; i9-14900KF + 4090; 21 TFLOPS BF16 / 35 TOPS
  Int8 on Xeon4; DS-V3 Xeon4 + 1×4090 = 418 tok/s; AMX tiling-aware layout, AVX-512 low-ARI switch,
  fused MoE, dynamic scheduling)
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/prefix_cache.md (3-layer GPU-CPU-disk KV
  via PhotonLibOS kvc2; page_size 16; recompile to enable)
- https://github.com/kvcache-ai/ktransformers/blob/main/archive/README.md (legacy framework split:
  kt-kernel + kt-sft; same SOSP'25 citation; SFT table DeepSeek-V3 LoRA ~40 tok/s 70 GB multi-GPU)
- https://github.com/kvcache-ai/ktransformers/issues/1921 (Roadmap 2026Q2: consumer x86+NVIDIA focus,
  "Improve support quality and stability for RTX 30/40/50 series", AVX2 first-class, VNNI, AI SSD,
  nvfp4/mxfp4 carried over; maintainer Q&A)
- https://github.com/kvcache-ai/ktransformers/issues/2179 (Qwen3.8-Flash-Next qwen4_exp spec: 125B /
  6B active / 48L / 512+1 experts top-10 / PLE 51B / MTP 4B / 36 GDN + 12 QSA; "Qwen's own model card
  recommends KTransformers"; SGLang graph in sgl-project/sglang#36497, #36585)
- https://github.com/kvcache-ai/ktransformers/issues/2127 (MTP launch failure on avx2 + sm89,
  DeepSeek-V4-Flash-0731, --context-length 1048576, flashmla, TP8)
- https://github.com/kvcache-ai/ktransformers/issues/2118 (sglang-kt==0.6.4 supports only EAGLE/NEXTN;
  DSpark port request; "~2x decode speed loss" with spec disabled)
- https://github.com/kvcache-ai/ktransformers/issues/2009 (MTP crash, KT 0.6.2, DeepSeek-V4-Flash)
- https://github.com/kvcache-ai/ktransformers/issues/2007 (EAGLE3 on Kimi K2.5/K2.6 crashes in
  sglang-kt; vendored-sglang reporting gap)
- https://github.com/kvcache-ai/ktransformers/issues/1815 (speculative decoding segfault, 2× RTX PRO 6000
  + MiniMax-M2.1 + standalone draft)
- https://github.com/kvcache-ai/ktransformers/issues/2038 (user rig 1×4090 + 2× EPYC 7532 + 256 GB DDR4
  running GLM-4.7 via LLAMAFILE backend)
- https://github.com/sgl-project/sglang/issues/11425 (SGLang KT-integration roadmap: GPU TP + CPU/GPU
  hybrid EP; single-GPU benchmark preview on 8452Y×2 + A100/RTX 4080: prefill up to 20×, decode 2.42×–
  4.09× vs Fiddler, 1.25×–1.76×/1.77×–1.93× vs llama.cpp; 21.3 TFLOPS CPU MoE kernel; user reports on
  sm_120/sm_89 attention-backend limits)
- https://www.lmsys.org/blog/2025-10-22-KTransformers/ (KVCache.AI + Approaching AI; AMX kernels, NUMA-
  aware TP 63% decode, CUDA-Graph launch overhead >20%→~0, Expert Deferral 1.45×/<0.5% accuracy; 8×L20
  264% at 8-way concurrency; L20×8 + 6454S ShareGPT 227.85/87.58 tok/s, ITL 431.61 ms)
- https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
  (SOSP'25 paper: 4.62–19.74× prefill / 1.25–4.09× decode; setup 8452Y×2 + A100/4080 + PCIe 4.0;
  Fiddler baseline 70.02 / 4.68 tok/s; 21.3 TFLOPS AMX kernel 3.98×; DS-3 decode 5.87 tok/s pre-deferral;
  deferral 74%/28% → 100%/37%, +33%; NUMA TP 1.63×; CUDA graph 1.23×; Table 2 accuracy; LiveBench
  −0.5% vs −13.3%; 11k lines C++ + 2k Python; DOI 10.1145/3731569.3764843)
- https://ktransformers.net/en/docs/supported-models/qwen (Qwen inference status: Qwen3-Coder-Next FP8/
  BF16 "Needs smoke"; Qwen3.5 BF16 "Needs mainline cleanup and smoke"; Qwen3-30B-A3B "Needs method-
  specific smoke"; SFT: Qwen3-235B-A22B, Qwen3.5-397B-A17B)
- https://github.com/kvcache-ai/sglang (SGLang fork: Apache-2.0, 14 stars; stock SGLang README, KT
  backend in-tree)
- https://pypi.org/project/kt-kernel/#history (PyPI latest kt-kernel 0.7.0.post4; wheel claims SM 80/
  86/89/90)
