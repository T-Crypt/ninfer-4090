# Dossier: SGLang

Researched 2026-10-06 (public sources only: repo, docs.sglang.io, cookbooks, LMSYS/SGLang blogs,
arXiv, release notes, and community single-card recipes). No installs, no weight downloads, no
GPU jobs.

## What it is / backing / license

- **What it is:** a high-performance serving framework for LLMs, multimodal models and diffusion
  models, "optimized for agentic workloads, RL rollouts, and large-scale serving"
  ([README](https://github.com/sgl-project/sglang)). Signature idea is **RadixAttention** — a radix
  tree over token prefixes for cross-request KV reuse (SGLang paper, arXiv 2312.07104, NeurIPS 2024:
  "up to 6.4x higher throughput compared to state-of-the-art inference systems" on agent control,
  few-shot, JSON decoding, RAG and multi-turn chat). On top: zero-overhead scheduler, chunked
  prefill, EAGLE/EAGLE3/MTP/UNO/DFlash/DSpark/NGRAM spec decode, **Unified Radix Cache** that
  extends the radix tree to hybrid models (FULL KV + SWA windows + MAMBA/GDN state checkpoints on
  one tree), HiCache (GPU→host→external-storage KV tiers, Mooncake/TensorCast backends),
  PD disaggregation with runtime prefill/decode role switching, breakable/piecewise CUDA graphs,
  torch.compile, and an in-progress Rust server/router. ~36.8k GitHub stars, 9.3k forks,
  5.5k issues (2026-10-06).
- **Backing:** originated at UC Berkeley (SGLang paper authors incl. Lianmin Zheng, Ying Sheng,
  Ion Stoica); "hosted by LMSYS, a non-profit open-source organization" (README). The 2026
  day-0 Qwen work is done by "the SGLang & Miles team at **RadixArk**" (blog footers), which also
  publishes the NVFP4/DSpark checkpoints used in the consumer recipes below.
- **License:** **Apache 2.0** (repo license field + README badge). **Porting code is license-OK.**

## Repo, version, last release date

- Repo: <https://github.com/sgl-project/sglang>; docs <https://docs.sglang.io>; cookbooks
  <https://cookbook.sglang.io>; blog <https://www.sglang.io/blog>; LMSYS blog
  <https://www.lmsys.org/blog>.
- **Latest: v0.5.21, released 2026-10-02** (779 PRs / 227 contributors; 451 commits to main
  since). Prior: **v0.5.20 (2026-09-18)**, **v0.5.19 (2026-09-05)**, v0.5.18 (2026-08-22),
  v0.5.17 (2026-08-08), v0.5.16 (2026-07-25) — **~biweekly cadence**
  ([releases](https://github.com/sgl-project/sglang/releases)).
- **Toolchain note:** v0.5.20+ **requires CUDA 13** (PyTorch 2.14); "CUDA 12 (`cu129`) wheels and
  images were retired … SGLang 0.5.19 is the last release with a CUDA 12 lane (PyTorch 2.13)"
  ([install](https://docs.sglang.io/docs/get-started/install)).
- v0.5.21 highlights relevant to us: **prefix cache on a Rust tree core by default (#39627)**;
  spec decode: XQA verify backend (#32269), windowed draft-decode attention for EAGLE/MTP drafts
  (#32673), "avoid materializing GDN QKV tensors during target verification" (#33778), ReplaySSM
  for KDA (#40517), "plan NextN/MTP draft layers as one-layer models" (#41194); **Agentic-Aware
  Tail-Optimized LRU eviction** (p99 ITL −43.9% at c32, DSV4-Pro FP4 4×B300, #34012); "enable
  optimistic prefill for Mamba radix-cache models" (#40184); SWA/Mamba radix caches and HiRadixCache
  removed, Unified Radix Tree is now the only path (#40313/#40775/#40780/#40787).

## Runs on an RTX 4090 (sm_89)?

**Partly — the base stack nominally supports sm_89, but every verified Qwen hybrid recipe in
SGLang's own cookbooks is for datacenter GPUs or Blackwell consumer cards, and the newest hybrid
model paths need sm_89 patches that only exist in a community fork.**

- Hardware table: NVIDIA row lists "A100; H100/H200/…; B200/B300/GB200/GB300; **select RTX 30/40/50
  series, RTX 6000 Ada / PRO 6000**; DGX Spark, Jetson Orin" ([README](https://github.com/sgl-project/sglang)).
- Install docs: "**FlashInfer is the default attention kernel backend. It only supports sm75 and
  above** … switch to other kernels by adding `--attention-backend triton --sampling-backend
  pytorch`" ([install](https://docs.sglang.io/docs/get-started/install)).
- **No 4090 cells for our model family.** The Qwen3.8-27B cookbook (our exact architecture, see
  below) has `supportedHardware: ["h200", "rtx6000", "rtx5090", "dgx-spark", "gb300"]` — every cell
  is `verified: true` on those five, **none on an RTX 4090**; the Qwen3.8 (2.4T) and Qwen3-Next
  cookbooks are datacenter-only (B200/B300/H200/H100/MI3xx/Xeon)
  ([Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B),
  [Qwen3-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next)).
- The fast hybrid kernels are not sm_89: the cookbook states "on SM120 both precisions run the
  **Triton linear-attn prefill path** — the FlashInfer GDN prefill fast path gates on **SM100**",
  and "`trtllm_mha` is SM100-only" ([config tips](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)).
  Release notes ship SM100/SM12x-only items (e.g. "SM100 NVFP4 GenMHA and speculative decoding …
  #36340"; "TMA-staged host<->device KV transfer kernel (**sm_90+**): #40278"). On a 4090,
  GDN prefill/decode and spec verify all fall to Triton/PTX paths.
- Community evidence that stock SGLang needs sm_89 patching for current hybrids:
  **xltzsoft/deepseek-v4-sm89** — "面向 8×RTX 4090（SM89）的 … SGLang 实验分支", baseline SGLang
  `131bd51b` + 2026-08-11 patches, "当前版本是性能研究快照，不是生产版本" (a performance
  research snapshot, not a production version). The patch set itself shows the gap: enable MXFP4
  Marlin MoE on SM89, route sparse MLA decode/prefill to SM89-executable Triton kernels, a custom
  BF16 sparse-prefill kernel tuned for Ada (BLOCK_K 16→64, ~1.33×), fixed 32-token sparse-decode
  tiles, native 8 query heads instead of FlashMLA's 64-head padding, SM89 fallbacks for FP8
  indexer and MHC prenorm, DSpark draft weight-loading fix ([repo](https://github.com/xltzsoft/deepseek-v4-sm89)).

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, first-class, day-0 for the whole family.**
  - **Qwen3.8-27B** (the model our `qwen3.8-27b` identity targets): "a dense hybrid Gated Delta
    Networks (GDN) **vision-language** model … 64 layers, laid out as 16 repeats of 3 × (Gated
    DeltaNet → FFN) followed by 1 × (Gated Attention → FFN) — 48 linear-attention layers to 16
    full-attention ones. Gated DeltaNet runs 48 value heads and 16 QK heads at head_dim 128; Gated
    Attention is GQA 24/4 at head_dim 256 … Hidden size is 5120 over a 17,408-dim FFN, and the
    checkpoint ships an **MTP head trained with multiple steps**. Context is **262,144 tokens
    natively, extensible to 1,000,000**. … **The serving-relevant architecture is identical to
    Qwen3.6-27B.**" Served through the Qwen3-VL path with the vision tower live
    ([cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)). Checkpoints:
    `Qwen/Qwen3.8-27B` (BF16), `Qwen/Qwen3.8-27B-FP8` (blockwise), `RadixArk/Qwen3.8-27B-NVFP4`
    (+ `-BF16-LMHead`), `nvidia/Qwen3.8-27B-NVFP4` (NVIDIA ModelOpt export, 21.9 GB on disk).
  - **Qwen3-Next / Qwen3.5 / Qwen3.6:** cookbooks with Mamba Radix Cache (V1/V2 strategies:
    `--mamba-radix-cache-strategy extra_buffer --page-size 64`) and EAGLE spec flags;
    [Qwen3-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next),
    [Qwen3.6](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6).
  - **Qwen3.8-2.4T-A95B** day-0 (2026-08-12): 92 layers = 69 GDN + 23 GQA in 3:1, 512 experts
    top-10; ReplaySSM for GDN state under MTP; Unified Radix Cache "for both full-attention KV
    and GDN state"; PD disagg transfers all three state types (KV, GDN recurrent state, GDN conv
    windows) ([blog](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support)).
  - **Qwen3.8-Flash-Next** (Qwen4 preview, day-0 2026-08-26): 125B MoE + 51B N-gram embedding,
    6B active; 48 layers = 36 GDN + 12 QSA; "GDN+QSA: KV cache memory management for the GDN+QSA
    hybrid architecture, compatible with Radix Cache"; IndexShare MTP reuses the QSA indexer
    selection across draft steps ([blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next)).
- **MTP: yes, in-checkpoint, via the EAGLE/NEXTN path.** Cookbook: "`--speculative-algorithm EAGLE
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` uses the
  in-checkpoint MTP head. (This recipe was originally documented with `NEXTN`, an alias of
  `EAGLE` — same algorithm.)" With `--enable-linear-replayssm-spec` the MTP draft intermediates
  move onto a fixed ring and the memory `D` term is 0 ([config tips](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)).
  Spec-decode menu ([docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)):
  EAGLE-2/EAGLE-3 (EAGLE3 paper arXiv 2503.01840), MTP, UNO (LoRA-based, validated on Qwen3-8B),
  DFLASH (trained block-diffusion drafter; DFlash2 upstreamed 2026-08-19, PR #35371/#35496),
  STANDALONE, NGRAM; plus DSpark (trained GDN drafter, e.g. `RadixArk/Qwen3.8-27B-DSpark`).
- **Vision: yes** — Qwen3.8-27B is a VLM and "the vision tower is live on the recipes below"
  (cookbook); SGLang serves Qwen3-VL / Qwen2.5-VL (cookbook index).
- **256K: yes, natively, on consumer single cards.** Qwen3.8-27B's 262,144-token window is served
  with "`--context-length 262144` … no YaRN needed"; the community 96 GB recipe gets a ~1.7M-token
  fp8 KV pool covering it with room to spare ([MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark));
  the 32 GB RTX 5090 recipe is validated at 8192-in/1024-out (context-length pin not published).
  Extensible to 1M per the model card (cookbook). **No public sm_89 256K run found.**
- **Prefix caching for the hybrid (the RadixAttention core):** **Unified Radix Cache** — one radix
  tree, a slot per component: FULL (KV pages), SWA (window slots), MAMBA (GDN checkpoint at the
  prefix endpoint, restored via copy-on-write); "computation resumes from the deepest boundary
  accepted by all components"; checkpoints at prefill chunk boundaries + regular decode intervals;
  Rust tree core lowers SWA TTFT by up to 42% ([blog](https://www.sglang.io/blog/unified-radix-cache),
  [LMSYS](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/)). HiCache layers
  GPU/host/Mooncake tiers on top; session-aware eviction (`session_id`, `/close_session`) lowers
  SWE-bench TTFT by up to 11.0% (DeepSeek-V4-Pro) / 16.6% (Qwen3.5-397B-A17B).

## Published numbers on a 4090 or similar

**No SGLang-published RTX 4090 (sm_89) tok/s figure found.** The only sm_89 SGLang numbers are from
the community fork below; SGLang's own published numbers for this model family are on Blackwell
consumer/datacenter cards.

- **Community 4090 run (not SGLang-official):** xltzsoft/deepseek-v4-sm89, 8×RTX 4090 TP8,
  DeepSeek-V4-Flash-0731 MXFP4, ctx 8192, max-8 requests, full decode CUDA graphs, 256-in/128-out
  greedy: **C1 decode 54.75 tok/s**, C1 e2e 50.36 tok/s, TTFT 0.222 s; **C8 scheduler output
  382.8–383.8 tok/s**, decode-window aggregate 374.58, e2e 318.59, GPU util ~92.7%; DSpark
  single-stream **96–126 tok/s** (target-only 48–55) after its C128 garble fix
  ([repo](https://github.com/xltzsoft/deepseek-v4-sm89)). Different model (not Qwen) — useful as a
  "SGLang-on-4090 is patched community territory" datapoint.

Closest SGLang-published analogs (all **not** sm_89 — the Qwen3.8-27B family on consumer cards):

- **RTX 5090 (32 GB, sm_120), single GPU, Qwen3.8-27B** (SGLang cookbook
  [Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), all on v0.5.19,
  1319-question GSM8K scores 93.18–95.15% across the 202-cell sweep; ISL 8192 / OSL 1024,
  concurrency 1):
  - Day-0 (NVFP4 + DSpark): **206.1 tok/s** decode on a single RTX 5090 ([Alibaba Qwen X post](https://x.com/Alibaba_Qwen/status/2088293486995087461),
    [reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1voearc/sglang_support_for_qwen3827b_200_toks_on_5090_38/),
    echoed by [yage.ai](https://yage.ai/share/2x5090-27b-sweet-spot-en-20260824.html) as the cookbook's single-5090 DFlash2 reference).
  - NVFP4 + EAGLE (MTP 3/1/4): **152.9 tok/s/user at fp32 GDN state vs 144.5 at bf16**; FP8 + EAGLE:
    **106.3 vs 116.1** (state dtype flips the winner by quant) (cookbook config tips).
  - **DFlash2 (block-8 drafter): 4.92 ms median TPOT at accept length 4.29 — "the best result on
    this card"** (cookbook); KV pool 97,280 tokens (bf16 state) vs 68,588 (fp32) with no speculation.
- **RTX PRO 6000 (96 GB, sm_120):** community recipe NVFP4-BF16-head + DFlash2, full 256K,
  ~1.7M-token fp8 KV pool: **single stream 240+ tok/s**; DFlash2 measured head-to-head beat MTP
  (GSM8K acceptance 3.43× vs 2.59×) and DSpark (2.69×) at c1, 2.84× vs 2.19×/2.23× at c8
  ([MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark),
  [drafter card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)). NVIDIA-forum runs:
  "Qwen3.8 Flash-Next on 1x RTX PRO 6000: **171 tok/s, 524K**, HiCache/NIXL persistence"
  ([thread](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722)).
- **DGX Spark (128 GB unified, sm_121 aarch64):** **34–38 tok/s** Qwen3.8-27B NVFP4 + DSpark
  ([MiaAI-Lab DGX Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark),
  [NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257)).
- **2×RTX 5090 TP=2, SGLang, NVFP4** ([yage.ai, 2026-08-24](https://yage.ai/share/2x5090-27b-sweet-spot-en-20260824.html)):
  total throughput 268 / 461 / 692 / 962 tok/s and per-stream 286 / 253 / 196 / 141 tok/s at
  concurrency 1/2/4/8 (math prompt, thinking off); DFlash2 215–256 tok/s (math, accept steps 3.3–5.6)
  and 222–232 (code); 5×16K-doc long-context 156.1 (thinking on) / 195.4 (off), JSON-extraction
  peak 334 tok/s; production log: ~224M prefill tokens/day of which **182M were prefix-cache hits
  at near-zero compute**, 2.25M decode tokens.
- **Datacenter (context for the technique gains, not 4090):** Qwen3.8-2.4T-A95B on B300: **346
  tok/s** b1 with MTP (accept 3.3) / **378 tok/s** DSpark (accept 4), TP8; 8K/1K serving 5,126
  tok/s/GPU (NVFP4, PD) ([Qwen3.8 blog](https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support));
  Qwen3.8-Flash-Next on B200 TP4: **540 tok/s** b1 MTP, accept 3.3 ([blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next));
  Kimi K3 (2.8T KDA hybrid): ~113 tok/s b1 pre-spec → **~423 tok/s** with DSpark; 2,808 tok/s/GPU
  PD-disagg ([blog](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)); EAGLE3 on 1×H100
  with Llama-3.1-8B: 158.34 → 244.10 (EAGLE-2) → 373.25 tok/s (MT bench) ([spec docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding));
  HiCache multi-turn (DSV4-Flash, 4×H200): effective input throughput **9.4K (L1) → 14.3K (L1+L2)
  → 145.5K (L1+L2+L3 Mooncake)** tok/s, hit rate ~98%, avg TTFT <9 s ([blog](https://www.sglang.io/blog/unified-radix-cache)).

## Ideas we could port into ninfer-4090

All Apache-2.0. Kernel-speed claims below are as published on Hopper/Blackwell; the portable value
on sm_89 is the **policies and state-management design**, plus second-implementation references.

1. **Unified Radix Cache: one tree, per-component reuse rules** (SGLang `mem_cache` unified tree +
   Rust TreeCore, default since v0.5.21 #39627; old separate SWA/Mamba caches and HiRadixCache
   deleted #40313/#40780/#40787). MAMBA component = a **GDN checkpoint (recurrent state + conv
   windows) stored at the prefix endpoint**; shared checkpoints are restored to a private request
   slot by **copy-on-write** before the forward mutates them; each component votes on candidate
   boundaries and "computation resumes from the deepest boundary accepted by all components".
   This is the cleanest public design for NInfer's context cache on a GDN hybrid: today either a
   prefix hit is safe for KV and wrong for GDN state, or prefix reuse is disabled on hybrid
   models. Their measured payoff (not 4090-specific): ~98% multi-turn hit rate with an L3 tier,
   16.6% lower SWE-bench TTFT with session-aware eviction.
2. **GDN state checkpoint placement + sparse retention** (Kimi K3 blog, also shipped in
   `--mamba-radix-cache-strategy extra_buffer/extra_buffer_lazy`): checkpoints only at aligned
   radix nodes, at prefill chunk boundaries and a fixed decode interval, kept sparse under a
   per-path cap + LRU, prioritizing **branching points** (Marconi-inspired, arXiv 2411.19379).
   Directly answers "how many GDN snapshots does a 24 GB card afford on agentic multi-turn
   traffic" — the S slot accounting below shows it's the binding budget.
3. **State-pool sizing formula (`--mamba-full-memory-ratio`)**: `ratio = (S + D) × state_bytes /
   (L × kv_bytes_per_token)`, with the exact per-geometry constants for our model published:
   **GDN state slot 153.9 MB (fp32) / 78.4 MB (bf16)** per request (48 GDN layers × 48 heads ×
   128 × 128 + bf16 conv state); **KV 32.8 KB/token (fp8) / 65.5 KB (bf16)** (16 attn layers ×
   GQA 4 × 256 × K+V); S = 5/4/3/1 for extra_buffer/lazy/no_buffer/disabled-cache. On a 24 GB
   4090 this is the exact computation NInfer's memory planner needs, and it shows why bf16 GDN
   state (halving 153.9→78.4 MB) doubles the concurrency headroom on small cards.
4. **ReplaySSM for MTP verify over GDN state** (tridao.me/blog/2026/replayssm; integrated into
   SGLang's verify kernels; `--enable-linear-replayssm-spec`): record each draft step's raw
   inputs (v, k, gk, β ≈ 1 KB) instead of snapshotting the full state per step; one fold kernel
   replays the accepted prefix from the committed checkpoint — **bit-identical**, draft-window
   memory 512 KB → 16 KB (32×), and it makes MTP compose with prefix caching + spec verify with
   **zero extra state-pool cost (D = 0)**. This is the same problem class NInfer hits with MTP3 on
   hybrid layers, and SGLang now ships *both* the ring-based (ReplaySSM) and slot-based (D > 0)
   variants.
5. **MTP via the EAGLE/NEXTN machinery**: `--speculative-algorithm EAGLE --speculative-num-steps
   3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` runs the *in-checkpoint* MTP
   head with no separate draft model — the same shape as our MTP-3. Supporting pieces worth
   reading: windowed draft-decode attention for EAGLE/MTP drafts (#32673), XQA verify backend
   (#32269 — XQA is an Ampere/Ada-friendly kernel), "avoid materializing GDN QKV tensors during
   target verification" (#33778), "plan NextN/MTP draft layers as one-layer models" (#41194).
6. **IndexShare MTP** (Qwen3.8-Flash-Next blog): the QSA top-k selection computed in the
   draft-extend pass is held for the whole MTP iteration, so draft decode steps skip the indexer
   entirely (N → 1 indexer invocations per iteration, "accept length is unchanged"). If the next
   dense Qwen carries QSA, this is a free decode-step speedup to replicate; it's model-agnostic
   logic, not a kernel.
7. **Confidence-scheduled verify trimming** (DSpark, Kimi K3 blog): the draft's trained confidence
   head + a one-time cost profile let a per-step planner keep only verify tokens whose expected
   value covers marginal cost; decode throughput **+68% (chat) / +24% (math) at bs 256**, break-
   even under bs 8. NInfer's fixed "one compact decode batch per round" policy could adopt the
   same value/cost trim at high batch without a draft model.
8. **Agentic-aware tail-optimized LRU eviction** (#34012, opt-in): p99 ITL −43.9% at c32 on an
   agentic trace (DSV4-Pro FP4, 4×B300). The eviction policy (protect tail entries of active
   agentic sessions) is portable as a scheduler policy to NInfer's context cache.
9. **Session-aware eviction** (`session_id`, `/close_session`, [docs](https://docs.sglang.io/docs/advanced_features/session_radix_cache)):
   track per-session references so eviction reclaims unreferenced data first; SWE-bench TTFT
   −11.0% (DSV4-Pro) / −16.6% (Qwen3.5-397B-A17B) vs HiRadixCache+LRU. Cheap policy addition.
10. **Fused GDN decode ops** (SGLang PR #32919: "fused the SplitKV reshape and Conv1D operations
    for low-latency tensor-parallel configurations, improving end-to-end decode performance by
    2% to 3%") and **context-parallel GDN prefill** (FlashInfer #3491, 2–3% prefill): the sm_89-
    relevant references are the Triton/FLA GDN kernels SGLang runs on SM120 (which is also the
    path a 4090 takes), useful as a numerical cross-check of NInfer's GDN prefill/decode.

## Worth running beside NInfer/llama.cpp?

**Yes — as the prefix-cache/hybrid-memory reference, not as the fastest 4090 engine.** On a 4090
SGLang is unverified for our model: no sm_89 cookbook cell, GDN runs the Triton fallback, the
NVFP4 fast paths that make the 5090 numbers work are Blackwell-only (no FP4 tensor cores on Ada;
the yage.ai analysis: 4090 = 1008 GB/s vs 1792 GB/s, ~40% slower single-stream decode, and
NVFP4 weights infeasible — 4-bit on a 4090 means INT4/FP8 with the VRAM penalty), and the only
sm_89 SGLang runs are patched community snapshots. A fair test would be: **same groupwise-int or
FP8 checkpoint** (not NVFP4), single 4090, batch 1–8, 32K–256K, and an **agentic multi-turn
workload with high prefix reuse** (tool-call loops, shared system prompts, compaction patterns) —
the one axis where SGLang's design (Unified Radix Cache + MAMBA checkpoints + session-aware
eviction) is meant to win, and where NInfer's context cache is directly comparable. Compare
prefix-hit rate, TTFT/TPOT and decode tok/s against NInfer and llama.cpp; expect SGLang's edge,
if any, to show up in TTFT and cache hit rate rather than raw single-stream decode.

## Risk / unknowns

- **No sm_89 validation anywhere for the Qwen3.8-27B hybrid** — every verified cell is
  H200/PRO 6000/5090/DGX Spark/GB300; the 4090 is "select RTX 30/40/50 series" at best, and the
  only 8×4090 SGLang run (DSV4-Flash) is a community snapshot whose own README says the
  compressed-state lifecycle "still needs GPU stress validation".
- **CUDA 13 floor since v0.5.20** (2026-09-18): a v0.5.19-and-older pin is needed for CUDA 12
  hosts; our toolchain is CUDA 13.1, so this is manageable but another version pin to track.
- **Fast churn, ~biweekly, with breaking refactors** (SWA/Mamba cache classes deleted and
  unified in v0.5.21; CUDA 12 lane retired; model-specific flags like `--mamba-full-memory-ratio`
  are new in the v0.5.19/0.5.20 era). Any SGLang-vs-NInfer test needs a pinned tag.
- **The 27B-dense numbers all ride on NVFP4 (Blackwell)**; the 4090-relevant FP8/INT4 path has
  far fewer published numbers (FP8+EAGLE 106–116 tok/s/user on the 5090 is the only 4090-
  transferable single-GPU datapoint, and even that is sm_120).
- **GDN state pool is the binding budget on 24–32 GB cards** ("on this 32GB card the GDN state
  pool, not KV, is what runs out first" — cookbook), so 4090 concurrency will be capped by the
  S-slot accounting long before KV; any SGLang-4090 test must set `--mamba-full-memory-ratio` /
  `--max-mamba-cache-size` deliberately.
- Spec decode on sm_89 for hybrid models is unproven in SGLang (DSpark/DSpark-C128 garble needed
  a 4090 fork fix; ReplaySSM-verify for GDN is documented for KDA on B300; the XQA verify backend
  #32269 is the Ada-friendly piece but unpublished numbers).

## Sources

- https://github.com/sgl-project/sglang (README: features, Apache-2.0, LMSYS hosting, hardware table incl. "select RTX 30/40/50 series", SGL Ecosystem: SpecForge/HiCache/Mooncake/SMG)
- https://github.com/sgl-project/sglang/releases (v0.5.21 2026-10-02 779 PRs/227 contributors; v0.5.20 2026-09-18; v0.5.19 2026-09-05; dates via GitHub API)
- https://github.com/sgl-project/sglang/releases/tag/v0.5.21 (full notes: Rust TreeCore default #39627, spec decode incl. XQA #32269 / windowed draft attention #32673 / GDN-verify #33778 / ReplaySSM #40517 / NextN planning #41194; unified cache removals #40313/#40775/#40780/#40787; agentic LRU #34012; Qwen3.8-Next MTP #40041/#40501)
- https://docs.sglang.io/docs/get-started/install (CUDA 13 requirement, v0.5.19 last cu12; FlashInfer sm75+ default backend, triton fallback)
- https://docs.sglang.io/docs/advanced_features/speculative_decoding (EAGLE-2/3, MTP/NEXTN, UNO, DFLASH, STANDALONE, NGRAM; EAGLE3 158.34→244.10→373.25 tok/s on 1×H100, Llama-3.1-8B MT bench)
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B (architecture, MTP 3/1/4 via EAGLE/NEXTN, ReplaySSM-spec, mamba ratio formula + state/KV bytes, SM100/SM120 backend notes, RTX 5090 pins and 152.9/144.5, 106.3/116.1, 4.92 ms/4.29 accept, GSM8K 93.18–95.15%, checkpoints)
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next (hardware B200..Xeon only, EAGLE 3/1/4, Mamba Radix Cache V1/V2)
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8 (2.4T-A95B, datacenter-only hardware)
- https://docs.sglang.io/docs/advanced_features/session_radix_cache
- https://www.lmsys.org/blog/2026-08-12-qwen3-8-day0-support (Qwen3.8-2.4T: ReplaySSM-GDN, Unified Radix Cache+HiCache, chunked PP prefill 5231/8363 tok/s/GPU, 5126 tok/s/GPU @8k/1k, MTP 346 / DSpark 378 tok/s B300 b1, kernel gains, 2.33× per-user with MTP)
- https://www.lmsys.org/blog/2026-08-26-qwen-flash-next (Qwen3.8-Flash-Next: GDN+QSA, IndexShare MTP, 540 tok/s B200 b1 MTP accept 3.3, PLE pinned-host offload +78.54% KV, Mix/Combine kernel gains)
- https://www.lmsys.org/blog/2026-08-11-unified-radix-cache + https://www.sglang.io/blog/unified-radix-cache (Unified Radix Cache design; HiCache 9.4K→14.3K→145.5K, ~98% hit, TTFT <9 s; SWE-bench TTFT −11.0%/−16.6%; Rust core −42% SWA TTFT)
- https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support (KDA state reuse: COW/snapshot/donate; branch-point checkpoints; unified memory pool; DSpark 113→423 tok/s b1, 2808 tok/s/GPU; verify trimming +68%/+24% at bs256; ReplaySSM 32× memory cut; DCP 1.5M→12.2M tokens)
- https://arxiv.org/abs/2312.07104 (SGLang, NeurIPS 2024; RadixAttention; "up to 6.4x higher throughput")
- https://tridao.me/blog/2026/replayssm/ (ReplaySSM raw-input replay; referenced by SGLang blogs)
- https://github.com/xltzsoft/deepseek-v4-sm89 (8×RTX 4090 SM89 SGLang experimental branch; C1 54.75, C8 318.59 e2e tok/s; DSpark 96–126 tok/s; sm89 kernel patch set)
- https://github.com/MiaAI-Lab/Qwen3.8-27B-RTX-6000-PRO-SGLang-DSpark (PRO 6000 96 GB: 240+ tok/s DFlash2, full 256K, ~1.7M-token fp8 KV pool, DFlash2 vs MTP/DSpark head-to-head)
- https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark (DGX Spark recipe)
- https://yage.ai/share/2x5090-27b-sweet-spot-en-20260824.html (2×5090 TP=2 SGLang: 268→962 total, 286→141 per-stream tok/s; DFlash2 215–256; 182M/224M prefix-hit tokens/day; 4090 bandwidth/NVFP4 analysis)
- https://x.com/Alibaba_Qwen/status/2088293486995087461 (day-0: 206.1 tok/s decode on a single RTX 5090)
- https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722 (PRO 6000: 171 tok/s, 524K, HiCache/NIXL)
- https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257
- https://huggingface.co/Qwen/Qwen3.8-27B (model card: apache-2.0 weights, 256K context, sampling defaults)
- https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4 ; https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 ; https://huggingface.co/incoai/Qwen3.8-27B-DFlash2 ; https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark
