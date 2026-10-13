# Dossier: Qwen3.8-Flash-Next (qwen4_exp) — Qwen's official Qwen4 architecture preview

Researched 2026-10-07 (public sources only: Qwen HF model card + `config.json` + `LICENSE`, the qwen.ai
blog, the tech report PDF in `QwenLM/Qwen3.8-Flash-Next`, SGLang PRs #36497/#36585/#37500 + cookbook,
llama.cpp PR #27742, KTransformers issue #2179, the vLLM recipe, and community consumer-GPU runs
(tonyd2wild 4×3090, antirez ds4). No installs, no weight downloads, no GPU jobs. Cross-referenced:
`qwen-official.md`, `sglang.md`, `vllm.md`, `ktransformers.md`, `tokenspeed.md` in this directory.

## What it is / backing / license

- **What it is:** not an engine — the model that is Qwen's declared **"early preview of the architecture
  used in Qwen4"** (blog): "It plays the same role that Qwen3-Next played for Qwen3.5 … We are again
  releasing the architectural changes early, so that the community can examine them before the full Qwen4
  model family is built on top of them." A multimodal MoE: **125B main model, 6B activated/token, plus 51B
  n-gram embedding (PLE) and 4B MTP** ([blog](https://qwen.ai/blog?id=qwen3.8-flash-next),
  [HF card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
- **Backing:** Qwen team, Alibaba (Tongyi Lab / Alibaba Cloud). Weights on HF `Qwen` and ModelScope; tech
  report in [`QwenLM/Qwen3.8-Flash-Next`](https://github.com/QwenLM/Qwen3.8-Flash-Next); hosted production
  version **Qwen3.8-Flash** on QwenCloud ("the official version based on Qwen3.8-Flash-Next with more
  production features, e.g., 1M context length by default", 0.15 USD / 0.47 USD per million input/output
  tokens).
- **License:** weights are **`qwen-community-1.0`** (custom, not Apache). The license is permissive for
  use/copy/modify/commercial deployment, with two conditions: (1) products with >100M MAU or >$20M monthly
  revenue must display the model name in the UI; (2) a **separate license from Qwen is required to use the
  model in a commercial "Model as a Service" or "AI Work Assistant" business** (internal use exempt)
  ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE)). For our internal 4090
  work there is no restriction; shipping it as a service would need Qwen's separate license. Engine code
  (llama.cpp/transformers/vLLM/SGLang PRs) is Apache-2.0; unsloth GGUFs carry the same qwen-community
  license. So: **code is portable, weights are conditionally usable** — read before any product use.

## Repo, version, release date

- HF: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (+`-FP8`, 1.6M downloads,
  `transformers` library, `qwen4_exp` tag, `license: other`); `unsloth/Qwen3.8-Flash-Next-GGUF` (dynamic
  quants incl. `UD-IQ4_XS`/`UD-Q4_K_XL` + `MTP/` heads); `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`
  (129 GB, W4A16 experts + FP8 PLE + INT4 MTP draft, used by the 4×3090 vLLM run);
  `nvidia/Qwen3.8-Flash-Next-NVFP4` (ModelOpt MIXED_PRECISION, SGLang #38121).
- Released **2026-08-26** (weights + transformers `qwen4_exp` support same day, per
  [qwen-official.md](qwen-official.md)); tech report dated 2026-08-26, "On the Design of Qwen3.8-Next
  Architecture: Evaluation, Efficiency, and Training Stability".
- `config.json`: `architectures: ["Qwen4ExpForConditionalGeneration"]`, `model_type: "qwen4_exp"`,
  `transformers_version: "5.8.0.dev0"`, `language_model_only: false`, dtype bfloat16
  ([config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json)).

## Runs on an RTX 4090 (sm_89)?

**No public single-4090 run exists, and none can exist as-is: the model does not fit in 24 GB at any
published precision** (172.78 GiB FP8 per the [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
via [vllm.md](vllm.md); 93.7 GB unsloth `UD-IQ4_XS` GGUF, ~60 GB resident in the 4×3090 run; 137.10 GiB /
41.73 GiB-resident antirez Q2 GGUF). Every consumer-hardware run found uses **multiple** cards or 128 GB
unified memory (4×3090, 2×DGX Spark, 2×A6000, RTX PRO 6000). What sm_89 support actually looks like:

- **llama.cpp: yes, the only engine with an Ampere/Ada consumer proof.** PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
  "model: add Qwen3.8-Flash-Next (qwen4exp)" (Unsloth, danielhanchen) **merged 2026-08-27** — one day after
  release. "no new ggml op, and no change to any existing one"; the tonyd2wild 4×RTX 3090 (sm_86) box runs
  it in production (numbers below), which is the closest public evidence that an sm_89 24 GB card behaves
  the same per-card (the run uses 20.5–22.6 GB VRAM per card).
- **SGLang: graph is merged; sm_89 is still rough.** `qwen4_exp` support **landed in main via
  [#37500](https://github.com/sgl-project/sglang/pull/37500) merged 2026-09-08** (Qiaolin-Yu, 47 commits,
  "initial pr: #36497" — the original [#36497](https://github.com/sgl-project/sglang/pull/36497) from the
  Qwen/SGLang team was closed on Sep 8 superseded by it). sm_89-relevant open work:
  [#36968](https://github.com/sgl-project/sglang/pull/36968) "fix(qsa): **fail early on SM89 without classic
  FA2**" (open, Sep 14) and [#36993](https://github.com/sgl-project/sglang/pull/36993) "[MoE][Ada] Add
  Qwen4-Exp **FP8 Triton configs for NVIDIA L20**" (open — L20 is the sm_89 datacenter SKU, i.e. Ada MoE
  kernels are still being added). NVFP4 load via [#38121](https://github.com/sgl-project/sglang/pull/38121)
  (merged); PLE file-backed table backend for unified-memory devices (GB10/DGX Spark) via
  [#37068](https://github.com/sgl-project/sglang/pull/37068).
- **vLLM: model support merged (2026-08-31, #53896) but all fast paths are Hopper/Blackwell** — SM121
  QSA prefill (#55430), SM90/SM12x GDN kernels; sm_89 gets baseline Triton/FLA paths (see
  [vllm.md](vllm.md)). The recipe runs on H200/H100/GB200/GB300/MI355X/4×RTX Pro 6000 — no 4090 recipe.
- **KTransformers: not supported** — [issue #2179](https://github.com/kvcache-ai/ktransformers/issues/2179)
  (2026-08-28) is an open feature request for `qwen4_exp` kt-kernel expert offload ("no qwen4_exp support
  exists yet"), no PR, no assignee; the author offers dual-Xeon-Gold-6526Y (Emerald Rapids, AMX) + 512 GB +
  2×RTX 6000 Ada 48 GB (SM89) and a Ryzen 9950X + RTX 5080 box for validation.
- **antirez `ds4`** (community C engine) has a dedicated qwen4exp Metal/CUDA graph; its measured runs are on
  DGX Spark (sm_121), not sm_89 ([doc](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)).
- **Qwen's own GDN kernel (FlashQLA) is SM90/SM100/SM103/SM120/SM121 only** — no sm_89, no 4090
  benchmarks (see [qwen-official.md](qwen-official.md)).

## Supports Qwen3.5-3.8 hybrid (Gated DeltaNet), MTP, vision, 256K?

This model *is* the next generation of the hybrid architecture. Per the [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json)
and [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf):

- **GDN hybrid: yes, upgraded to GDN + QSA.** 48 layers = **36 `linear_attention` (GDN) + 12
  `full_attention`** (config `layer_types`, `full_attention_interval: 4`; the 12 attention layers are QSA
  in the post-trained weights — "for Qwen3.8-Flash-Next, all full-attention layers in the backbone and MTP
  module are replaced with QSA"). GDN is the same family as our Qwen3.8-27B dense: `linear_num_key_heads:
  16`, `linear_num_value_heads: 48`, key/value head dim 128, `linear_conv_kernel_dim: 4`,
  `mamba_ssm_dtype: "float32"`; **`output_gate_type: "sigmoid"`** (vs "swish" in the dense 27B; llama.cpp
  #27742: "the only delta from Qwen3.5 is a sigmoid output gate instead of silu"). Hidden 2560, 512 experts
  `moe_intermediate_size: 640` + 1 shared expert, `num_experts_per_tok: 10`, `router_aux_loss_coef: 0.001`.
- **QSA (new, Qwen4-only):** micro-block sparse attention. Indexer = MQA (4 query heads, 1 shared key head,
  128-dim, partial RoPE on 64 dims); keys compressed to non-overlapping **r=4 token blocks by AvgPool
  before RoPE**; ReLU block-causal scores; **top-512 blocks = 2048-token budget** (`indexer_budget: 2048`,
  `indexer_compress_ratio: 4`), tail partial block always included; per-layer (no cross-layer index
  sharing). Trained by two-stage continued pretraining (dense distillation, then sparse training; loss
  gap to full attention "on the order of 10⁻⁴").
- **Gated Residual (new, Qwen4-only):** `hc_count: 4` branches, `hc_lowrank: 320` (= d/8) elementwise
  sigmoid read gate (low-rank), per-branch scalar `2σ(·)` write gate, **no branch-mixing operator**
  (drops Hyper-Connection's `H_res`), group RMSNorm per branch; **residual state can be stored FP8**
  ("halves the bytes moved for the residual state relative to BF16, with almost no loss in quality").
- **N-gram embedding / PLE (new, Qwen4-only):** one layer at **layer 2** (`ple_layer_ids: [2]`),
  `ple_conv_kernel_size: 4`, `ngram_size: 3` (bigrams+trigrams), `ngram_vocab_size_base: 20,000,000`,
  `ple_embed_dim: 2560`, `heads_per_ngram: 8`, `split_ngram_parts: 128` → **51B params**; "can be stored
  in Host Memory and asynchronously prefetched in parallel with model computation, without permanently
  occupying GPU memory" (model card). Placed at layer 2 specifically "allowing host-memory prefetching to
  overlap with the computation of the first layer" (tech report §2.3.1).
- **MTP: yes.** `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false` (shared embeddings),
  multi-step trained; `mtp: {hybrid: true, layer_types: ["full_attention"]}` — **the MTP module's
  attention layers are QSA too**, and it reuses top-k QSA indices across speculative steps (accepted length
  4.06 → 4.07, neutral — tech report Table 4). Draft heads ship in-checkpoint (vLLM recipe) and as a 2.6 GB
  unsloth GGUF `mtp-*-shared-Q8_0.gguf`.
- **Vision: yes.** Same ViT as the dense line: `vision_config` depth 27, hidden 1152, 16 heads, patch 16,
  `spatial_merge_size: 2`, `temporal_patch_size: 2`, `num_position_embeddings: 2304`, `out_hidden_size:
  2560`, `deepstack_visual_indexes: []`; MRoPE interleaved `[11,11,10]`. The HF card demos image and
  hour-scale video; llama.cpp wires "stock Qwen3-VL ViT through the existing clip path".
- **256K: yes.** `max_position_embeddings: 262144` natively; **1M via static YaRN** (`factor: 4.0`,
  recommended block in the model card); production Qwen3.8-Flash serves 1M by default.

**Consequence for prep4qwen:** the "next dense Qwen" hypothesis (27–32B dense GDN hybrid on one 4090) is
not contradicted by this model — Qwen still ships dense lines (Qwen3.8-27B shipped 9 days before
Flash-Next) — but this is Qwen's explicit statement of where the **architecture** is going: the four deltas
(QSA, Gated Residual, n-gram PLE, Muon-shaped checkpoints) plus the sigmoid GDN gate are the most likely
additions on the next dense model, and Flash-Next itself is the direct proof that Qwen's flagship-class
models now exceed any 24 GB card by 5–10×.

## Published numbers on a 4090 or similar

Copied verbatim from the linked sources. There is **no RTX 4090 (sm_89) number for this model anywhere
public** (not found). The closest published hardware is Ampere/Ada multi-card and 128 GB unified memory:

**A. 4× RTX 3090 24 GB + 31 GB RAM (sm_86, 96 GB VRAM)** — [tonyd2wild/Qwen38-Flash-Next-4x3090](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090)
("every number from the box", measured 2026-09-06 / lane updated 2026-10-01):

| metric | vLLM lane (W4A16 experts Marlin, BF16 attn, FP8 PLE on NVMe, INT4 MTP draft 3-token expert-parallel) | llama.cpp lane (unsloth `UD-IQ4_XS` GGUF 93.7 GB, ~60 GB in VRAM) |
|---|---|---|
| Count to 100, single stream, temp 0 | **193.3 tok/s median** (55.8 no draft) | 96–102 tok/s copy/edit; 40–49 freeform |
| Real prompts, single stream (prose/chat/code) | **109.5 / 108.4 / 145.8 tok/s** | 40–49 freeform |
| 6 parallel coding agents (12,189 tokens) | **317 tok/s sustained, 570 peak**, 53 tok/s per agent, TTFT 0.47 s | not run |
| 6 concurrent, gmu 0.95 | 0 OOM, TTFT p99 7.5 s; 252.6 tok/s aggregate, 48 per stream | not run |
| Context / KV | 262,144 native; pool **303,079 tokens** (tuned lane, `--kv-cache-memory-bytes 2400000000`) | 2 slots × 262,144 (`-c 524288 --parallel 2`), f16 KV ~12 KB/token/card |
| Long prefill | 155K-token prompt: **2,249 tok/s prefill, 69 s TTFT** | ~311 tok/s; TTFT 150–170 ms; load ~85 s |
| VRAM | gmu 0.95–0.97 | 20.5–22.6 GB used per card at full config |
| MTP | albucino INT4 draft (in checkpoint) | unsloth MTP head: count100 58.8 → **102.3** (1.74×), code 58.8 → 96.1 (1.63×), prose 58.7 → 81.6 (1.39×), JSON 58.4 → 95.5 (1.64×); acceptance 0.66 prose → 0.92 structured; "unsloth measured 1.67x on a B200" |
| Other | tool suite (69 scenarios): quality 91.3, median turn 681 ms, 88.1 tok/s | `ngram-mod` (no draft model): 43 → 62 tok/s on copy/edit, neutral on prose |

- Box: "4× NVIDIA RTX 3090 24 GB (NVLink pairs 0↔1 and 2↔3), 31 GB system RAM, NVMe, driver 580.173.02".
- Ampere findings: "FP8 e4m3 KV does not compile on 3090s; e5m2 does through our overlay variant and
  passed the needle test at 7K, 28K and 53K"; the 47.7 GB FP8 PLE table "lives on the NVMe, 16 rows per
  token read per step … inside CUDA graphs" (vLLM lane patch); llama.cpp "leaves it memory-mapped on NVMe
  and the OS page cache serves the hot rows … first requests after a restart run ~25% slower until the
  page cache warms"; quantized KV cache (`-ctk/-ctv q8_0`) "aborts on this arch in the mainline-derived
  builds. Keep KV at f16"; "MTP is for low concurrency — unsloth measure a net loss (0.81–0.87×) at
  concurrency 8".
- Reference: "this beats a 2× DGX Spark TP2 deployment of the same model in NVFP4 with MTP4 speculation
  (~33 tok/s)".

**B. DGX Spark (128 GB unified, sm_121)** — antirez [`ds4`](https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md)
("resident weights and disk-only n-grams"; 95.37 GiB n-gram table stays on disk, rows read directly from
the GGUF; two runs of `speed-bench/promessi_sposi.txt`, 8192-token prefill chunks, 128 teacher-forced
decode tokens, no MTP):

| model | file size / resident weights | first 1024 tokens | next 7168 tokens | decode at 8192 |
|---|---|---|---|---|
| Q2 (IQ2_XXS gate/up, Q2_K down) | 137.10 GiB / 41.73 GiB | 516 t/s | 745 t/s | 22.6 t/s |
| Q4 (Q4_K gate/up, MXFP4 down) | 165.11 GiB / 69.74 GiB | 513 t/s | 755 t/s | 21.2 t/s |

- Q2 also "reached about 771 t/s on a fresh 32K-token prefix with `--prefill-chunk 32768`".

**C. 2× RTX A6000 48 GB (sm_86)** — SGLang PR [#36585](https://github.com/sgl-project/sglang/pull/36585)
(open, "native Qwen4-Exp support completed by Argus", waltstephen, 2026-08-27): official checkpoint with
PLE + MoE expert tensors **CPU-resident**, LM head on GPU — peak CUDA memory **18,876.996 MB (TP1)** and
**14,899.813 MB per rank (TP2)**; real TP1/TP2 API/streaming smokes passed (load 829.30 s / 997.30 s);
synthetic (random small model, not real weights): GDN Triton chunk path 127.753 → 1.144 ms/iter
(111.71×, seq=512 batch=1), fused MoE top-10/64-expert 21.190 → 4.087 ms (5.184×), PLE-enabled synthetic
TP1 30.020 → 10.040 ms (2.990×). "The real API smoke timings include that CPU-resident route and are not
an optimized real-weight throughput claim."

**D. Qwen's own figures (no GPU stated for any of them; datacenter-class assumed):**
- QSA kernel speedups at 1M context: **7.6× prefill / 4.9× decode** vs dense (baseline = FlashInfer paged
  GQA; prefill = 16K-token chunk, batch 1; decode = batch 4, next_n=4 = three MTP steps) ([tech report
  Fig. 6](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf), [blog](https://qwen.ai/blog?id=qwen3.8-flash-next)).
- "achieves **8.6× the Prefill throughput of Qwen3.7-Plus at a 1M-token context length**" at 90% prefix
  cache hit rate ([blog](https://qwen.ai/blog?id=qwen3.8-flash-next)).
- QSA quality (tech report Tables 2–4): short-context avg **75.9 → 76.8**; RULER 512K–1M **90.08 → 93.00**;
  8-needle MRCR 512K **30.66 → 40.53**, 1M **20.71 → 26.44**; MTP accepted length (4-step) **4.06 → 4.07**.
- "training takes only about 1/9 as much [as Qwen3.7-Plus]"; base model "leads the 397B-A17B predecessor
  on eight [of 14 pre-training benchmarks] … at 1/3 the activated parameters, 1/3 the training tokens, and
  roughly 1/9 the training FLOPs" ([blog](https://qwen.ai/blog?id=qwen3.8-flash-next), [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)).

**E. Datacenter runs (for scale, not 4090-relevant):** SGLang B200 TP4 **540 tok/s** b1 MTP, accept 3.3
([LMSYS blog 2026-08-26](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next), via [sglang.md](sglang.md));
NVIDIA GB300 NVL72 FP8 "**over 16K tokens per second per GPU and over 200 tokens per second per user**"
([NVIDIA blog](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/),
via [tokenspeed.md](tokenspeed.md)); RTX PRO 6000 (96 GB) "171 tok/s, 524K" with HiCache/NIXL
([NVIDIA forum](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722),
via [sglang.md](sglang.md)); vLLM recipe: 172.78 GiB FP8, vLLM 0.29.0+, no 4090 recipe
([recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next), via [vllm.md](vllm.md)).

**F. Accuracy of the llama.cpp port** (#27742, against the reference implementation): wikitext-2 perplexity
**4.0068 ± 0.02271** (145 chunks, ctx 2048) vs 4.0126 reference; top-1 agreement 98.0% on 512 tokens of
prose; **QSA vs dense bit-identical below the budget** (`indexer_top_k + compress_ratio - 1` cached tokens,
max logit delta 0.0 over 2051 rows); above budget at 8192 tokens diverges on 3% of positions; indexer
selection 0.975 mean Jaccard vs a 0.991 precision floor; `test-llama-archs -a qwen4exp` OK on CPU/CUDA/Metal.

## Ideas we could port into ninfer-4090

Ranked for the dense-next-Qwen-on-4090 scenario; "gain as they measured it" is the source's own number
(nearly all non-sm_89 — direction only).

1. **QSA block-level indexer + sparse core attention** (tech report §2.1.2; SGLang `QwenSparseAttnBackend`
   via #37500; vLLM indexer PRs #54513/#54873/#54890; llama.cpp graph in #27742). If the next dense Qwen
   ships QSA (likely — it is the Qwen4 preview), NInfer must add: MQA indexer (4 q-heads, 1 k-head, 128-dim,
   partial RoPE on 64 dims) with r=4 AvgPool key compression *before* RoPE, ReLU block-causal scores,
   top-512-block/2048-token selection with mandatory tail block, feeding paged KV gathers in the 12 (or 16)
   sparse layers. On a 4090 the win is **KV memory and KV-read traffic, not FLOPs**: it is what makes
   262K–1M resident on 24 GB; Qwen measured 7.6×/4.9× kernel speedups at 1M (datacenter GPUs). The indexer
   is tiny (128-dim MQA) — cheap even on Ada. SGLang's production shape to copy: separate prefill (compact
   logits workspace) vs decode (FP8 indexer cache) indexer paths; and the sm_89 caveat: classic FA2 is the
   fallback attention on Ada (#36968).
2. **MTP reusing top-k/QSA indices across speculative steps** (tech report §2.1.2, "following GLM"; SGLang
   IndexShare per the [LMSYS blog](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next): "N → 1 indexer
   invocations per [MTP] iteration, accept length is unchanged"). Free win for NInfer's MTP3 verify-window
   batching: verify steps skip the indexer entirely. Measured: neutral (4.06 → 4.07 accepted).
3. **Host-memory n-gram (PLE) prefetch** (vLLM #53899 host offload + #54129 disk mmap; SGLang #37068
   file-backed PLE backend; llama.cpp lookup-only mmap path; antirez ds4 reads rows straight from the GGUF
   on disk; tonyd2wild's vLLM patch reads "16 rows per token per step inside CUDA graphs"). The 51B table
   (97.7 GiB in llama.cpp's GGUF; 95.37 GiB in ds4's) **cannot live in 24 GB** — the only way a
   Qwen4-class model with PLE runs on one 4090 is deterministic row prefetch from DRAM/NVMe overlapped with
   layer-1 compute (Qwen deliberately placed it at layer 2 for this). If the next dense model keeps
   `ple_*` fields, NInfer's loader + scheduler must model a per-token PCIe/NVMe gather overlapping compute;
   note the measured cold-start cost ("~25% slower until the page cache warms").
4. **Gated Residual as a new state stream** (tech report §2.2): 4-branch widened residual (4×hidden per
   token-slot) with per-branch group RMSNorm + low-rank sigmoid read gate + scalar `2σ` write gate, no
   branch mixing; **residual state FP8-storable** because the gates bound the stream. Budget 4×hidden×dtype
   per KV slot (halved in FP8) in the paged-KV math; the same bounded-range argument is what makes
   NInfer's `rk4v4-e8`-class KV compression safe. If the next dense model ships `hc_count: 4`, this is a
   second state stream next to the GDN recurrent state.
5. **GDN numerics convention update** (config `output_gate_type: "sigmoid"`; llama.cpp #27742; tech report
   §2.1.1): bounded sigmoid output gate replaces SiLU, FP32 recurrent state, L2-norm q/k,
   `α = exp(−exp(A)·softplus(·))`, state layout `S ∈ R^{d_k×d_v}`. NInfer's GDN op must be checked against
   this exact convention for the next model — a silent mismatch is a long-context numerics bug (cross-ref
   [qwen-official.md](qwen-official.md) item 1).
6. **llama.cpp's QSA graph design as a C++ reference** (#27742, Apache-2.0): `build_attn_mask_top_k`
   lifted out of the DSA `build_attn` overload so both sparse families share it; block-sparse attention as
   "an optional third cache in `llama_memory_hybrid`" (i.e. block-sparse state coexists with the GDN
   recurrent state and paged KV in one hybrid memory manager); `filter_idx` defaulting to `nullptr` with
   guarded use; no new ggml ops. This is the closest public implementation of "how a from-scratch C++
   engine carries QSA state" — directly comparable to NInfer's Op/transaction structure.
7. **`ngram-mod`-style context drafting as MTP complement** (tonyd2wild 4×3090 lane: 43 → 62 tok/s on
   copy/edit/tool output, neutral on prose, no draft model needed). NInfer-4090 already has n-gram
   prompt-lookup drafting ([ninfer-4090-udpsendtofailed.md](ninfer-4090-udpsendtofailed.md)); this is
   confirmation the technique scales to the Qwen4 architecture and remains the fallback drafter when no
   MTP head is available.
8. **Expert-offload scheduling (only if Qwen4-dense > 24 GB or we adopt the MoE line):** KTransformers
   deferral/NUMA/hotness placement from [ktransformers.md](ktransformers.md); the 10/512 routing skew makes
   a hot-expert GPU set especially effective per the #2179 author. Not applicable to a dense model.

## Worth running beside NInfer/llama.cpp?

**No — not the model itself: it cannot run on a single 24 GB card in any published precision** (smallest
found: 41.73 GiB resident weights, antirez Q2; llama.cpp's smallest GGUF lane still needs ~60 GB VRAM +
95 GB table). Any "side-by-side" on our 4090 box is impossible; the published consumer runs all use
4×24 GB cards or 128 GB unified memory. So:

- **As a 4090 benchmark target: no.** The fair consumer comparison (vLLM W4A16+FP8-PLE vs llama.cpp
  UD-IQ4_XS vs KTransformers-once-supported) requires a 4–8×24 GB box, which is a different machine than
  ours. Track the tonyd2wild repo's results instead of re-running them.
- **As an architecture oracle: yes, strongly.** It is Qwen's own statement of what the next model looks
  like. The right test *for us* is: when the next dense Qwen (qwen4 family) ships, compare NInfer vs
  llama.cpp (master, which carries #27742) vs vLLM on the same 4090 with the same 4-bit quant, on
  256K-context agentic prompts — the same shape as our Qwen3.8-27B baselines. The llama.cpp lane is the
  meaningful cross-check (it is our llama-swap baseline's engine, and the qwen4exp support is already
  merged upstream, so no fork needed).
- **One concrete "worth testing" item** (a Friday decision, not this week): whether NInfer's paged-KV
  math survives a QSA sparse layer — i.e. can our KV pool index serve "only 2048 of N tokens attended per
  query" without changing the pool layout (the 3090 llama.cpp lane shows f16 KV at ~12 KB/token/card is
  still the cost driver even when GDN layers dominate).

## Risk / unknowns

- **The target may be MoE, not dense.** Flash-Next is the *architecture* preview for Qwen4; if Qwen4's
  flagship/open-weight line follows it (125B/6B-MoE + PLE), "next dense Qwen" either never ships or ships
  as a smaller member, and the whole prep4qwen premise (one 4090, dense, fits in 24 GB) shifts to
  multi-card or expert-offload territory (KTransformers #2179 is the tracking issue). Mitigation: the dense
  27B line (Qwen3.5 → 3.6 → 3.8) has been maintained in parallel for three generations.
- **No single-24 GB data point exists for this model** (not found); the 3090 numbers are per-card
  20.5–22.6 GB on a 4-way split of a 125B MoE, not a 27B dense on one card. Extrapolating them to a
  4090 dense Qwen4 would be guesswork.
- **QSA kernel support on sm_89 is unfinished across the board**: SGLang needs classic FA2 on SM89
  (#36968 open) and Ada MoE configs are still landing (#36993 open); vLLM's QSA fast paths are SM121/SM90;
  FlashQLA (GDN) has no sm_89 at all. Expect the first Qwen4 dense models to run on baseline Triton/FLA
  attention on a 4090 — which is exactly where a tuned NInfer GDN/QSA kernel could lead.
- **The 51B PLE table dominates host memory** (95–98 GB in published GGUFs; 47.7 GB FP8 in the vLLM lane)
  and its performance depends on NVMe/DRAM bandwidth and page-cache warmth; a 4090 box with <64 GB RAM
  cannot host an FP8/BF16 table and must quantize it (no public quant-quant PLE table found) or drop the
  layer (no public "PLE-removed" quality numbers found).
- **License is conditional** (qwen-community-1.0): fine internally, but a service or AI-work-assistant
  product on these weights needs a separate Qwen license; our Apache-2.0 dense Qwen weights are not affected.
- **Day-0 churn already visible:** mixed-chunk + compressed QSA device-asserts (SGLang #38180), fast_topk
  bin-overflow (fixed in the #38121/#38144 line), PLE state pool unsized for PD disaggregation (#39624),
  ROCm MFMA padding (#38910), spec-decode OOMs mid-traffic (#38341). The qwen4_exp paths in all engines
  were ~6 weeks old at research time and are still stabilizing — expect NInfer's converter to need
  iteration on GR/PLE/QSA field names (the config already diverges from qwen3_5: `hc_*`, `indexer_*`,
  `ple_*/ngram_*`, `heads_per_ngram`, `output_gate_type`).
- **Numbers above are mostly non-sm_89** (A6000/3090/Spark/B200/GB300); the 3090 data is sm_86 with
  NVLink between card pairs and 31 GB host RAM — directionally the closest thing, but not a 4090.

## Sources

- https://huggingface.co/Qwen/Qwen3.8-Flash-Next — model card (architecture overview, benchmarks,
  "SGLang, KTransformers or vLLM are strongly recommended", Qwen3.8-Flash = official 1M-context production
  version, YaRN block, sampling params, video `longest_edge: 469762048`)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json — full `qwen4_exp` config
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE — Qwen Community License 1.0 text
- https://qwen.ai/blog?id=qwen3.8-flash-next — blog: GDN+QSA, Gated Residual, N-gram Embedding, Muon;
  7.6×/4.9× at 1M; 8.6× prefill vs Qwen3.7-Plus @1M (90% prefix hit); 1/9 training cost; API pricing
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf — "On the Design of Qwen3.8-Next
  Architecture" (2026-08-26): GDN recurrence/equations, QSA indexer equations + Fig. 6 kernels, GR
  ablations (Tab. 5/6/7), PLE placement/vocab tables (Tab. 7–9), Muon/Canzona, scaling-law refit (Fig. 8–9),
  stress tests (Fig. 10–13), base-model eval (Tab. 11)
- https://github.com/QwenLM/Qwen3.8-Flash-Next — repo README (llama.cpp "supports … (text & vision)",
  MLX, framework links)
- https://github.com/kvcache-ai/ktransformers/issues/2179 — open feature request for `qwen4_exp` expert
  offload (2026-08-28): architecture summary, "Qwen's own model card recommends KTransformers", test
  hardware (2×Xeon Gold 6526Y AMX + 512 GB + 2×RTX 6000 Ada 48 GB SM89; Ryzen 9950X + RTX 5080)
- https://github.com/sgl-project/sglang/pull/36497 — original "Introduce Qwen 3.8 Flash Next" (Qwen team,
  2026-08-26), closed 2026-09-08; carries #38121 (NVFP4 load), #37068 (file-backed PLE), #38209/#38144/
  #38180/#38341 QSA/spec-decode fixes
- https://github.com/sgl-project/sglang/pull/37500 — "support qwen 3.8 flash next" **merged into main
  2026-09-08** (47 commits, "initial pr: #36497")
- https://github.com/sgl-project/sglang/pull/36585 — open native Qwen4-Exp text path (2×A6000 evidence,
  PLE/MoE CPU-resident, synthetic benchmarks)
- https://github.com/sgl-project/sglang/pull/36968 / /pull/36993 — open sm_89/Ada work (fail early on
  SM89 without classic FA2; Qwen4-Exp FP8 Triton MoE configs for L20)
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next — day-0 recipe page (176B-param
  (6B active) hybrid MoE on NVIDIA/AMD)
- https://github.com/ggml-org/llama.cpp/pull/27742 — **merged 2026-08-27**: qwen4exp converter/graph
  (GDN sigmoid gate, 512-expert top-10 MoE, hyper-connections, PLE host rows, QSA budget 2048/ratio 4,
  vision), accuracy table, no new ggml ops
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next — vLLM recipe: 172.78 GiB FP8, vLLM 0.29.0+,
  H200/H100/GB200/GB300/MI355X/4×RTX Pro 6000, no 4090
- https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090 — 4×RTX 3090 + 31 GB RAM: vLLM W4A16+FP8-PLE
  lane (193.3 tok/s median; 317 tok/s 6-agent; 303,079-token KV pool @262K; 2,249 tok/s 155K prefill)
  and llama.cpp UD-IQ4_XS lane (2×262K slots; 96–102 tok/s MTP; ~311 tok/s prefill); MTP head A/B table;
  traps (K-quant repack 41.3 GiB, silent CPU fallback, `-md` for MTP heads)
- https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md — ds4 qwen4exp support: Q2/Q4 GGUF
  sizes, 95.37 GiB disk-only n-gram table, DGX Spark numbers (516/745/22.6 t/s Q2), vision/MTP/steering
- https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF — dynamic quants + `MTP/` heads (2.6 GB shared
  Q8_0); https://unsloth.ai/docs/models/qwen3.8-next (local-run guide)
- Cross-referenced local dossiers: [qwen-official.md](qwen-official.md), [sglang.md](sglang.md),
  [vllm.md](vllm.md), [ktransformers.md](ktransformers.md), [tokenspeed.md](tokenspeed.md),
  [flashinfer.md](flashinfer.md)
