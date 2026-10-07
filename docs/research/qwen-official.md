# Dossier: Qwen official (blog, papers, HF configs, transformers/vllm PRs)

Researched 2026-10-07 (public sources only: qwen.ai blog posts, the Qwen3.8-Flash-Next tech report
PDF on GitHub, Hugging Face model cards and `config.json` drops from the Qwen org, arXiv papers Qwen
cites, and GitHub PRs in huggingface/transformers and vllm-project/vllm). No installs, no weight
downloads, no GPU jobs. No leaked material — the "next model" signals below are all from Qwen's own
early-releases practice: Qwen released the Qwen3-Next architecture two days before its first weights
and Qwen3.8-Flash-Next as an explicit "early preview of the architecture used in Qwen4".

## What it is / backing / license

- **What it is:** not an engine — the model vendor. This dossier collects Qwen's public architecture
  statements (blogs, tech report, HF configs) plus the day-0 framework PRs, to predict the next dense
  Qwen (~27-32B, GDN hybrid, MTP, 256K+, vision) and find portables.
- **Backing:** Qwen team, Alibaba (Alibaba Cloud / Tongyi Lab). Public surfaces: qwen.ai blog,
  Hugging Face org `Qwen`, ModelScope, QwenCloud API, GitHub orgs `QwenLM` and `Qwen` (FlashQLA,
  Qwen3.8-Flash-Next tech report, Qwen-Agent, Qwen-Live-Harness, Qwen-MM-Plugins).
- **License (weights):** **apache-2.0** for Qwen3.8-27B, Qwen3.6-27B, Qwen3.5/3.6/3.8 dense and MoE
  open weights (HF `config`/`README` license field); **`qwen-community-1.0`** (custom, non-standard)
  for Qwen3.8-Flash-Next and its FP8 variant. **License (code):** transformers and vLLM are Apache-2.0;
  QwenLM/FlashQLA is **MIT**; the GDN reference kernels live in `fla-org/flash-linear-attention`
  (MIT). So: Qwen3.8-27B class weights + all framework code are port-license-clean; Flash-Next weights
  are under a custom license — read before touching.
- **Generation timeline (public):** Qwen3-Next (2025-09, first GDN-hybrid release, MoE text-only) →
  Qwen3.5 (2026-02-15, native multimodal, 250K vocab, dense+MoE sizes 0.8B-397B) → Qwen3.6
  (2026-04, 27B dense + 35B-A3B) → Qwen3.8 (2026-08: 27B dense VL on 2026-08-17, 2.4T-A95B open Max,
  Flash-Next "Qwen4 preview" 2026-08-26) → Qwen3.8-Omni-Flash (2026-09-18, hosted omni, 1M context).
  ([HF org listing](https://huggingface.co/Qwen),
  [Qwen3.5 blog](https://qwen.ai/blog?id=qwen3.5),
  [Qwen3.6-27B blog](https://qwen.ai/blog?id=qwen3.6-27b),
  [Flash-Next blog](https://qwen.ai/blog?id=qwen3.8-flash-next),
  [Omni-Flash blog](https://qwen.ai/blog?id=qwen3.8-omni-flash))

## Repo, version, last release date

There is no single "Qwen repo" for inference. The public artifacts:

- **Hugging Face `Qwen` org** — weight/config drops. Current relevant repos (as of 2026-10-07):
  `Qwen3.8-27B` (+`-FP8`), `Qwen3.8-2.4T-A95B` (+`-FP8`), `Qwen3.8-Flash-Next` (+`-FP8`),
  `Qwen3.6-27B`/`-FP8`, `Qwen3.6-35B-A3B`, `Qwen3.5-{0.8B,2B,4B,9B,27B,35B-A3B,122B-A10B,397B-A17B}`,
  `Qwen3-Next-80B-A3B-Instruct` (+`-FP8`), plus VL/Omni/TTS/ASR/Embedding/Reranker families.
  ([HF](https://huggingface.co/Qwen/Qwen3.8-27B))
- **Config `transformers_version` pin** is a real compatibility signal: Qwen3.5 configs say
  `4.57.0.dev0`; Qwen3.8-27B and Qwen3.8-Flash-Next configs say **`5.8.0.dev0`** (transformers 5.x
  era); Qwen3.8-2.4T-A95B says `4.57.3`.
  ([Qwen3.8-27B config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json),
  [Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json))
- **Tech report:** [`QwenLM/Qwen3.8-Flash-Next` GitHub repo, `tech_report.pdf`](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
  "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"
  (Qwen Team, Aug 2026) — the Qwen4 architecture authority.
- **Framework day-0 PRs** (earliest legitimate next-model signal, per our ground rules):
  - transformers **#40771 "Adding Support for Qwen3-Next"** merged **2025-09-09** (two days before the
    Qwen3-Next day-0 vLLM blog of 2025-09-11; the HF drop followed).
    ([PR](https://github.com/huggingface/transformers/pull/40771))
  - transformers **#43830 "Adding Support for Qwen3.5"** merged **2026-02-09**, six days before the
    2026-02-15 Qwen3.5 release blog. ([PR](https://github.com/huggingface/transformers/pull/43830))
  - transformers **#48337 "Add Qwen4Exp model"** merged **2026-08-26** — same day as the
    Qwen3.8-Flash-Next weights (Qwen4 = `qwen4_exp` model type).
    ([commit](https://github.com/huggingface/transformers/commit/fc5c5bde))
  - vLLM **#53896 "[Model] Support Qwen3.8-Flash-Next"** merged 2026-08-31; **#53899** PLE (n-gram)
    offload to host; **#54129** disk-backed mmap PLE table; **#54513/#54517/#54873/#54890/#54915**
    QSA indexer prefill/decode separation, fused PLE kernels, QSA sparse-GQA, FP8 indexer cache;
    **#55430** tile-union QSA prefill **on SM121**; **#55272** removed torch.compile for the NVIDIA
    Flash-Next path; **#55617 (open WIP)** "Hybrid GDN (Qwen3.5/Qwen3.8 27B-class) + MTP ~3
    concurrent sequences at batch >= 4".
    ([search](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+Qwen3.8))
  - Open/closed transformers follow-ups showing active numeric debugging of the GDN path: **#45513**
    (merged 2026-04-27) "Fix GDN linear attention multi-token cached forward"; **#46831/#46859**
    (2026-06) garbage generation under device_map CPU offload / variable-length linear attention;
    **#47887** (2026-08, closed unmerged) "sample GatedDeltaNet A in float32 to avoid …".
    ([search](https://github.com/huggingface/transformers/pulls?q=is%3Apr+qwen3.5+in%3Atitle))

## Runs on an RTX 4090 (sm_89)?

**The weights/configs are architecture-only (no GPU requirement). Every fast kernel in the Qwen
ecosystem is SM90+; a 4090 must run the Triton/FLA fallback paths, and Qwen's own kernel library
does not support sm_89 at all:**

- **FlashQLA (Qwen's own GDN kernel, MIT): "Requirements: SM90, SM100, SM103, SM120 or SM121; CUDA
  12.8 or above."** No sm_89. Its published benchmarks are `benchmark_results_H200.txt` and
  `benchmark_results_GB200.txt` — no 4090 data exists from Qwen. v0.1.2+ also "serves as a backend
  for flash-linear-attention's GDN". ([FlashQLA README](https://github.com/QwenLM/FlashQLA))
- vLLM's fast GDN paths are Blackwell-only (FlashInfer GDN #3001 SM12x/SM110; fused GDN MTP decode
  SM110; CuteDSL GDN prefill) and vLLM's QSA work for Flash-Next includes an **SM121**-targeted
  prefill kernel (#55430) — on a 4090 both the GDN hybrid and QSA models run only the baseline
  Triton/FLA implementations (per the [vLLM dossier](vllm.md) and [SGLang dossier](sglang.md);
  SGLang's GDN gate is `get_device_capability()[0]>=9`).
- None of Qwen's published benchmarks uses an RTX 4090 or any Ada card. The one single-consumer-card
  datapoint for this exact model family is the community 5090 run already documented in
  [vllm.md](vllm.md) (~160 tok/s, Qwen3.8-27B NVFP4, sm_120 — **not transferable to sm_89**).

## Qwen3.5-3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

All from Qwen's own configs/model cards; the numbers below are copied from those sources.

### The dense hybrid that is our target (Qwen3.8-27B config)

[config.json](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json) —
`Qwen3_5ForConditionalGeneration` (note: the 3.8 dense reuses the **qwen3_5** model type),
`language_model_only: false`:

- Text: `qwen3_5_text` — **64 layers, hidden 5120, vocab 248,320 (padded), dtype bfloat16**;
  `full_attention_interval: 4` → `layer_types` is exactly 48× `linear_attention` + 16×
  `full_attention` (3:1, every 4th layer full); `max_position_embeddings: 262144`.
- GDN ("linear_attention") shape: **`linear_num_key_heads: 16`, `linear_num_value_heads: 48` (3 V-heads
  per QK-head), `linear_key_head_dim: 128`, `linear_value_head_dim: 128`, `linear_conv_kernel_dim: 4`
  (causal short conv), `mamba_ssm_dtype: "float32"`** (recurrent state kept FP32).
- Full attention: **24 Q heads / 4 KV heads (GQA 6:1), `head_dim: 256`**, `attn_output_gate: true`,
  **`output_gate_type: "swish"`**, `partial_rotary_factor: 0.25` (RoPE on 64 of 256 dims — the model
  card says "Rotary Position Embedding Dimension: 64"), `rope_theta: 10000000`, `rope_type: default`,
  **`mrope_interleaved: true`, `mrope_section: [11, 11, 10]`** (interleaved 3-D MRoPE → t/h/w for
  video; 11+11+10 = 32 pairs = 64 dims).
- **MTP: `mtp_num_hidden_layers: 1`, `mtp_use_dedicated_embeddings: false`** — one MTP layer sharing
  the base embeddings; model card: "MTP (Multi-Token Prediction): trained with multiple steps".
- FFN: `intermediate_size: 17408`; `rms_norm_eps: 1e-06`; no QK-norm fields (zero-centered RMSNorm is
  the Qwen3-Next+ convention per the Qwen3-Next blog).
- Vision: **`vision_config` — ViT depth 27, hidden 1152, 16 heads, patch 16, `spatial_merge_size: 2`,
  `temporal_patch_size: 2` (video), `num_position_embeddings: 2304` (= 48×48 patch grid),
  `out_hidden_size: 5120`, `deepstack_visual_indexes: []`** (deepstack injection disabled, unlike
  Qwen3-VL); image/video tokens 248056/248057, vision start/end 248053/248054.
- Model card ([README](https://huggingface.co/Qwen/Qwen3.8-27B)): "Hidden Layout: 16 × (3 × (Gated
  DeltaNet → FFN) → 1 × (Gated Attention → FFN))", "Context Length: 262,144 natively and extensible
  up to 1,000,000 tokens", native image+video understanding, thinking on by default with
  `reasoning_effort` (xhigh/medium/low) and `preserve_thinking`. Recommended YaRN block for 1M:
  `rope_type: "yarn"`, `factor: 4.0`, `original_max_position_embeddings: 262144` (static YaRN —
  "potentially impacting performance on shorter texts"). Best practices: thinking mode
  `temperature=1.0, top_p=0.95, top_k=20`; video `longest_edge: 469,762,048` ≈ 224K video tokens for
  hour-scale video.

**The Qwen3.5-27B dense is the same architecture minus `output_gate_type`** (i.e. the swish/sigmoid
gate typing is new in 3.8) — same 64 layers/5120 hidden/48:16 GDN/24:4 attention/262144/MRope/
identical ViT ([Qwen3.5-27B config](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json)).
So the dense 27B layout has been **frozen across two generations** (3.5 → 3.8); the only text-side
diffs found: `output_gate_type: "swish"` added, `transformers_version` 4.57→5.8.

### Scaling the GDN head count (config drops)

| model (config) | layers | hidden | GDN V:QK heads | full attn Q:KV | head dim | MoE | MTP |
|---|---|---|---|---|---|---|---|
| [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) | 48 | 2048 | 32:16 | 16:2 | 256 | 512/10+1 | not in config (weights carry MTP) |
| [Qwen3.5-27B (dense)](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) | 64 | 5120 | 48:16 | 24:4 | 256 | — | 1 layer |
| [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) | 60 | 4096 | 64:16 | 32:2 | 256 | 512/10+1 | 1 layer |
| [Qwen3.8-2.4T-A95B (open Max, text-only)](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json) | 92 | 8192 | 128:16 | 64:4 | 256 | 512/10+1 | 1 layer |
| [Qwen3.8-Flash-Next (Qwen4Exp)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) | 48 | 2560 | 48:16 | 24:2 + QSA indexer | 256 | 512/10+1 | 1 layer, QSA, `hybrid: true` |

Reads: **QK head count is pinned at 16 with 128-dim keys at every scale; V-head count scales
32→48→64→128; conv kernel is 4 everywhere; GDN state is FP32 everywhere; full-attention head dim is
256 with partial-RoPE 0.25 everywhere.** The "next dense 27-32B" (if it keeps the Qwen3.5/3.8 dense
line) is therefore extremely likely to be: 64-96 layers, hidden ~5120, 3:1 GDN:attn, 16×128 QK GDN
heads, 24-ish Q / 4 KV @ 256, MTP-1 shared embeddings, 262144 native + YaRN 1M, MRoPE-interleaved
ViT (27×1152, patch 16).

### The Qwen4 preview (what the *next* model adds)

[Flash-Next blog](https://qwen.ai/blog?id=qwen3.8-flash-next) + [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)
(125B/6B-active MoE + 51B n-gram + 4B MTP; "an early preview of the architecture used in Qwen4 …
We are again releasing the architectural changes early"):

1. **GDN + QSA instead of GDN + full attention** — "those full-attention layers are replaced by QSA"
   at continued-pretraining time, including the MTP module's attention layers. QSA = micro-block
   sparse attention: a lightweight **MQA indexer (4 query heads, 1 shared key head, 128-dim, partial
   RoPE on 64 dims)** scores non-overlapping key blocks of **r=4 tokens (AvgPool compression)** with
   ReLU block-causal similarity, selects top-`K=2048` tokens = **512 blocks** per query (tail partial
   block always included); per-layer (no cross-layer index sharing, which the report shows loses to
   IndexCache-style sharing in hybrids: "QSA matches the full-attention baseline at a relative indexer
   latency of 0.25, whereas IndexShare remains below the baseline at 0.5"). Config fields:
   `indexer_budget: 2048`, `indexer_compress_ratio: 4`, `indexer_head_dim: 128`, `indexer_kv_heads: 1`,
   `indexer_n_heads: 4`, `heads_per_ngram: 8`.
2. **Gated Residual (GR)** — widened 4-branch residual stream (`hc_count: 4`), per-branch group
   RMSNorm, element-wise **sigmoid** read gate through a low-rank bottleneck (`hc_lowrank: 320` =
   d/8 for d=2560), per-branch scalar write gate `2σ(·)`, **no branch-mixing operator** (drops
   Hyper-Connection's `H_res`), read and write each fused into one kernel; **residual state can be
   stored FP8** ("halves the bytes moved for the residual state relative to BF16, with almost no loss
   in quality"). One branch self-organizes as a long-range path (layer 0 → attention layers). The
   dense 27B configs do **not** have `hc_*` fields — GR is a Qwen4-preview addition.
3. **N-gram embedding ("PLE")** — one layer **at layer 2** (`ple_layer_ids: [2]`,
   `ple_conv_kernel_size: 4`, `ngram_size: 3` = bigrams+trigrams, `ngram_vocab_size_base: 20000000`,
   51B params, 2560-dim) that deterministically looks up local n-grams; "can be stored in Host Memory
   and asynchronously prefetched in parallel with model computation, without permanently occupying
   GPU memory" (vLLM shipped this as PLE-Offload #53899 and disk-mmap #54129).
4. **Muon optimizer + refit scaling law** (training-side, but it shapes the shipped checkpoints):
   Newton-Schulz 8 steps (Polar-Express coefficients), `γ = 0.2·max(A,B)` scaling, Muon on attention/
   GDN/expert projections, AdamW on embeddings/router/GR low-rank, fused matrices split before
   orthogonalization; batch-size warmup dropped ("costs 18.8% more optimizer steps"); at 4× optimal
   LR the old AdamW recipe spikes 183 times/10k steps while Muon+GR never crosses the clip threshold.
5. **MTP** — still 1 layer, shared embeddings, multi-step trained; in Qwen4 the MTP module reuses
   top-k QSA indices across speculative steps (accepted-length table: 4.06 full-attn vs 4.07 QSA at
   4-step speculative decoding, i.e. reuse is neutral).
6. **GDN numerics (tech report §2.1.1):** state convention `S_t ∈ R^{d_k×d_v}` (transpose of the
   original GDN paper); `q,k = L2Norm(SiLU(ShortConv(Wx)))`, `v = SiLU(ShortConv(Wx))`;
   `β_t = σ(W_β x_t)`, `α_t = exp(−exp(A)·softplus(W_α x_t + b_α))`; output
   `o_t = W_o[σ(W_z x_t) ⊙ RMSNorm(y_t)]` — **"Unlike the original GDN, which uses a SiLU output
   gate, we use the bounded sigmoid gate … and observe consistent improvements"** (Flash-Next config
   says `output_gate_type: "sigmoid"`; the 27B dense says `"swish"`). Zero-centered RMSNorm
   throughout. RoPE is **mandatory** in the full-attention/QSA layers: "the NoPE variant exhibits a
   substantially higher rate of endless generation after post-training and is therefore more likely
   to fail to terminate."

### MTP, vision, 256K — per-capability verdicts

- **MTP: yes, shipped since Qwen3-Next** (weights) and config-level (`mtp_num_hidden_layers: 1`,
  `mtp_use_dedicated_embeddings: false`) for every Qwen3.5+ drop; transformers MTP support for the
  qwen3_5 family is still **unmerged** (#45637/#45638 closed without merge, both 2026-04) — vLLM/SGLang
  carry MTP, transformers does not (Qwen3-Next blog: "Multi-Token Prediction (MTP) is not generally
  available in Hugging Face Transformers").
- **Vision: yes, native** in the dense line since Qwen3.5 ("trained from scratch on interleaved text,
  image, and video tokens" — [transformers model doc](https://huggingface.co/docs/transformers/model_doc/qwen3_5));
  Qwen3.8-27B understands images *and* hour-scale video (ViT `temporal_patch_size: 2`; MRoPE-interleaved);
  the omni line (audio in/out) exists only hosted (Qwen3.8-Omni-Flash, 1M context) — no audio fields in
  any open config found.
- **256K: yes, native 262,144** in every Qwen3-Next/3.5/3.6/3.8 open config; 1M via static YaRN
  (`factor: 4.0`) or by the hosted services.

## Published numbers on a 4090 or similar

**No Qwen-published number uses an RTX 4090 or any sm_89 GPU.** Qwen's own figures (all datacenter
hardware or unspecified; GPU/batch given when stated in the source, "not found" when not):

- **QSA kernel speedups** (Flash-Next tech report, Fig 6; baseline = FlashInfer paged GQA; prefill =
  16K-token chunk, **batch size 1**; decode = **batch size 4, next_n=4**, i.e. three MTP steps; GPU
  not stated in text): at **1M context**, QSA attention module is **7.6× (prefill) and 4.9× (decode)**
  faster than dense attention; indexer cost goes from O(n²) to O(n²/r) with r=4.
  ([tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf),
  [blog](https://qwen.ai/blog?id=qwen3.8-flash-next))
- **Prefill throughput** (blog): "In an experimental setup representative of online serving scenarios
  with a 90% Prefix Cache hit rate, Qwen3.8-Flash-Next achieves **8.6× the Prefill throughput of
  Qwen3.7-Plus at a 1M-token context length**" (GPU/batch not stated).
- **QSA quality** (tech report Tables 2-4, Flash-Next): short-context benchmark average **75.9 →
  76.8** (full attn → QSA); RULER 512K-1M **90.08 → 93.00**; 8-needle MRCR 512K **30.66 → 40.53**,
  1M **20.71 → 26.44**; MTP accepted length (4-step spec decode) **4.06 → 4.07** avg across
  MT-Bench/GSM8K/MATH/HumanEval/MBPP.
- **FlashQLA** (Qwen's GDN kernel, MIT): "**2-3× forward speedup and 2× backward speedup over the FLA
  Triton kernel … on NVIDIA Hopper and Blackwell**" ([README](https://github.com/QwenLM/FlashQLA));
  benchmark files are H200 and GB200 only — **no sm_89 numbers exist**.
- **Training-efficiency claims** (Qwen3.5 blog, GPU/batch not stated): "Under the 32k/256k context
  length, the decoding throughput of Qwen3.5-397B-A17B is **8.6x/19.0x** that of Qwen3-Max …
  **3.5x/7.2 times** that of Qwen3-235B-A22B." (Qwen3-Next blog likewise: prefill "nearly 7x" at 4K,
  "over 10x" beyond 32K vs Qwen3-32B; decode "nearly 4x" at 4K, "over 10x" beyond 32K; all GPU
  not stated.)
- **Architecture ablations** (tech report Table 1, 28-layer 25B-A3B MoE, 400B tokens @4K + 80B @32K,
  same eval pipeline): GDN hybrid improves on 8 of 9 benchmarks vs full-attention Transformer and on
  7 of 9 vs SWA hybrid (window 128). GR ablation (Table 5, 25B-A3B, 560B tokens): widening static
  +1.58 and data-dependent +1.98 points average accuracy over pre-norm; GR loss 1.590 vs pre-norm
  1.617. N-gram placement (Table 7): single layer at layer 2 best (avg 47.94) vs no n-gram 45.44.
- **Stability stress tests** (tech report §3.3, 28L 25B-A3B, constant LR): at 4× optimal LR, AdamW
  + Qwen3.5 structure spikes **183/10k steps** and crosses clip on 213/19,932 steps; Muon+GR: **0**
  spikes.
- Single-card consumer datapoints for this model family come from community, not Qwen: the vLLM-
  5090 run in [vllm.md](vllm.md) (~160 tok/s, MTP-3, 262K, sm_120) and the xltzsoft 8×4090
  DeepSeek-V4-Flash fork in [sglang.md](sglang.md). For Qwen on 4090 specifically: **not found** in
  any Qwen-published source.

## Ideas we could port into ninfer-4090

Everything below is public; licenses: configs/papers are ideas, FlashQLA is MIT, transformers/vLLM
code is Apache-2.0, GDN reference (FLA) is MIT. "Gain as they measured it" is Qwen's own number,
almost all on H200/GB200-class GPUs — **none are sm_89 numbers**, so treat them as direction only.

1. **GDN output-gate and state-dtype conventions** (tech report §2.1.1): bounded sigmoid gate on the
   GDN output (SiLU gate is the old FLA convention; Qwen moved to sigmoid for Flash-Next and the 27B
   dense ships `output_gate_type`), FP32 recurrent state (`mamba_ssm_dtype: float32`), L2-normalized
   q/k, `α = exp(−exp(A)·softplus(·))` parameterization. NInfer's GDN op should be checked against
   exactly this convention (transpose state layout `S ∈ R^{d_k×d_v}` too) — a mismatch is a silent
   numerics bug on long context, and transformers #47887 (GDN `A` sampled in FP32) shows the ecosystem
   is still fixing GDN dtype corners.
2. **QSA for the 256K+ agentic turns on a 24 GB card.** If the next dense Qwen ships QSA (likely,
   since Flash-Next *is* the Qwen4 preview), NInfer needs: a block-level indexer pass (MQA, r=4 block
   compression via AvgPool before partial-RoPE, ReLU block-causal scores, top-512-block/2048-token
   budget with tail block always included) feeding the 16 full-attention layers' paged KV gathers.
   On a 4090 the payoff is KV memory, not FLOPs: QSA bounds the full-attention KV *reads* per query,
   which is what keeps 262K resident on 24 GB; Qwen measured 7.6×/4.9× kernel speedups at 1M on
   datacenter GPUs. The vLLM PRs (#54513/#54873/#54890) show the production shape: separate prefill
   (compact logits workspace) and decode (FP8 indexer cache) indexer paths.
3. **MTP reusing top-k/sparse indices across speculative steps** (tech report §2.1.2, "following GLM"):
   with 4-step MTP the verify steps skip re-running the indexer — a free win for NInfer's MTP3 verify
   window batching; Qwen reports accepted length unchanged (4.06→4.07).
4. **Host-memory n-gram embedding prefetch** (51B table, layer 2, "deterministically addressed"): on a
   PCIe-4.0 4090 this is the *only* way a 27-32B Qwen4-class model with 51B extra embedding params
   stays on one card — prefetch the layer-2 n-gram rows during layer-1 compute. vLLM's #53899/#54129
   (host offload + disk mmap) is the reference design. If the next dense model keeps n-gram layers
   (config signal: `ple_*` fields in qwen4_exp), NInfer's loader + scheduler must model a per-token
   PCIe fetch overlapping compute; if it drops them, this item is moot.
5. **FP8 (or our 4-bit) residual state** — GR "supports FP8 storage" because all gates bound the
   stream; the same bound argument applies to NInfer's KV compression: Qwen's gates (GDN α/β,
   attention output gate, GR read/write) exist partly to keep streams in a narrow range, which is
   exactly what makes `rk4v4-e8`-class compression safe. If the next model ships GR, the 4-branch
   residual is a new state stream to cache/compress — budget for it in the paged-KV math
   (4×hidden×dtype per token-slot for the widened stream, or FP8 to halve it).
6. **FlashQLA as a GDN prefill oracle, not a 4090 kernel** — it is SM90+/SM120-only, but it is Qwen's
   *reference* GDN chunked-prefill implementation (TileLang, gate-driven intra-card CP, fused
   warp-specialized kernels) and it now backs FLA's GDN API. On a 4090 we cannot run it, so NInfer's
   sm_89 GDN prefill kernel should be validated against (a) FLA Triton and (b) FlashQLA on a Hopper
   machine if one is ever available; the 2-3×/2× H200 numbers set the bar for what a tuned sm_89
   chunked-GDN prefill should approach relative to FLA.
7. **Day-0 checklist inputs (feeds docs/prep4qwen.md item 5):** from the config drops above, the
   converter's target for a next dense Qwen is: model_type likely `qwen4_exp` (or a new
   `qwen4`), 3:1 `full_attention_interval` (or QSA interval), 16 QK GDN heads @128 with V-heads
   ~48, conv kernel 4, FP32 ssm state, MTP-1 shared embeddings, 262144 + YaRN, interleaved MRoPE
   [11,11,10], 27-layer ViT (patch 16, merge 2, temporal 2) — and the diff to watch: `hc_count/
   hc_lowrank` (GR), `indexer_*` (QSA), `ple_*/ngram_*` (n-gram), `output_gate_type`.

## Worth running beside NInfer/llama.cpp?

**Yes — but the "engine" to run beside us is the Qwen-recommended serving stack on the *next*
checkpoint, not a Qwen product.** Qwen publishes no engine of its own; it blesses SGLang, vLLM and
TokenSpeed (all three are named in the Qwen3.8-27B model card with official cookbooks/recipes;
llama.cpp and MLX/Unsloth are named for Flash-Next). For a fair test once the next dense Qwen drops:
same quant as NInfer (groupwise-int class — **not** the NVFP4/Blackwell-only checkpoints), single
4090, batch 1-8, 32K/128K/262K contexts, prefill tok/s, decode tok/s with MTP on/off, TTFT/TPOT,
prefix-cache reuse — NInfer vs vLLM (Triton/FLA GDN path on sm_89) vs SGLang (triton GDN fallback)
vs llama.cpp. The value is (a) an independent GDN/QSA/MTP implementation to validate numerics
against, and (b) detecting day-0 support gaps: the transformers MTP gap (#45637 unmerged), the vLLM
open WIP on "Hybrid GDN + MTP ~3 concurrent sequences at batch ≥ 4" (#55617), and the
transformers GDN dtype follow-ups (#47887) all indicate the ecosystem's hybrid+MTP path on consumer
GPUs is still maturing — a gap NInfer can own first, the way it already does for 3.8-27B.

## Risk / unknowns

- **The next dense model is not announced.** All "next" signals are from Flash-Next, which Qwen
  explicitly frames as a MoE 125B/6B preview of Qwen4; whether a 27-32B *dense* Qwen4 exists (vs the
  dense line ending at 3.8) is unknown. If Qwen4 ships MoE-only for the mid-size, our dense target
  shifts to "Qwen4-27B dense" only if Qwen keeps the dense line (they kept 27B dense for three
  straight generations: 3.5, 3.6, 3.8 — that is the strongest continuity signal found).
- **QSA in a dense model is unproven by Qwen at dense scale.** The tech report ablations (indexer
  heads, block size) were run at 35B-A3B MoE scale; the dense 27B never had QSA (it uses plain
  GQA-24:4 full attention). If the next dense keeps full attention instead of QSA, items 2-3 above
  shrink to "keep paged KV + sparse-read at 262K" (already in our ranked plan).
- **Config drops lie by omission:** the open configs omit anything proprietary. `output_gate_type`
  appearing only in 3.8, `hc_*`/`indexer_*`/`ple_*` only in qwen4_exp, and the Qwen3-Next config
  lacking `mtp_*` fields (MTP lives in weights only) mean the *first* config of the next family will
  under-specify; treat the Flash-Next config as the ceiling of what will appear, not a promise.
- **No sm_89 performance data from Qwen for anything** — every Qwen efficiency number is
  datacenter-shaped (H200/GB200 benches, unspecified-GPU throughput ratios). 4090 extrapolation of
  QSA/Qwen4 kernels is exactly what the prep4qwen branch exists to de-risk.
- **Licensing fork:** Qwen3.8-Flash-Next (the Qwen4-preview weights) is under `qwen-community-1.0`,
  not Apache-2.0; anything we build against it (converter paths, kernel shapes) may not be freely
  redistributable — keep the day-0 pipeline license-neutral (config-schema driven) like the v3 port.
- **Ecosystem churn around the new model types:** transformers 5.x is a moving target for the
  qwen3_5/qwen4_exp model definitions (dev-version pins in the configs; unmerged MTP PRs; closed
  unmerged GDN dtype fix #47887), and vLLM's Flash-Next NVIDIA path is still landing
  (torch.compile removal #55272, SM121 prefill #55430). A day-0 NInfer converter must not track
  upstream model code 1:1.

## New targets found

- **FlashQLA** (QwenLM, MIT, TileLang-based GDN chunked-prefill/forward-backward kernel library,
  v0.1.3 2026-09; SM90/SM100/SM103/SM120/SM121 only; 2-3× fwd / 2× bwd vs FLA Triton on H200/GB200;
  now the FLA GDN backend) — worth its own dossier: it is the authoritative Qwen GDN kernel and the
  sm_89 gap in it is a direct NInfer opportunity.
- **TokenSpeed** (LightSeek, lightseek.org) — third engine Qwen officially recommends in the
  Qwen3.8 model card with an official Qwen3.8 recipe; check sm_89 + GDN support when its turn comes.

## Sources

- https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd (Qwen3-Next: 3:1 GDN:attention, 80B-A3B, MTP, 256K/1M YaRN, prefill/decode 7x/10x vs Qwen3-32B, references incl. GDN paper)
- https://qwen.ai/blog?id=qwen3.5 (Qwen3.5-397B-A17B: GDN+MoE, 8.6x/19.0x decode vs Qwen3-Max, 250K vocab, 201 languages, FP8 training pipeline)
- https://qwen.ai/blog?id=qwen3.6-27b (dense 27B generation; family: 27B, 35B-A3B, Plus, Max-Preview)
- https://qwen.ai/blog?id=qwen3.8 (Qwen3.8-Max 2.4T/95B-active, open weights)
- https://qwen.ai/blog?id=qwen3.8-flash-next (Qwen4 preview: GDN+QSA, Gated Residual, N-gram embedding, Muon; 7.6x/4.9x QSA kernel at 1M; 8.6x prefill @1M/90% cache)
- https://qwen.ai/blog?id=qwen3.8-omni-flash (omni, 1M context, Qwen-Live-Harness, Qwen-MM-Plugins)
- https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf (full architecture + ablations: GDN equations/sigmoid gate/FlashQLA, QSA indexer math + RULER/MRCR/MTP tables, GR + FP8 residual, n-gram placement/vocab, Muon/Canzona, scaling law, stress tests)
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json (+ README: layout, MTP, 262K→1M, sampling, video longest_edge, YaRN block, apache-2.0)
- https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json (dense qwen3_5, identical core to 3.8-27B)
- https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json (qwen3_5_moe, 60L, GDN 64:16, transformers 4.57.0.dev0)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json (qwen4_exp: indexer_*, ple_*, ngram_*, hc_*, mtp hybrid)
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next (README: qwen-community-1.0 license, 125B+51B+4B/6B, tech report link, framework support incl. llama.cpp)
- https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/main/config.json (open Max: 92L, hidden 8192, GDN 128 V heads, text-only)
- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json (original hybrid: 48L, GDN 32:16, 150K vocab, no mtp_*/vision fields)
- https://huggingface.co/Qwen (org model listing — full family map incl. Qwen3.5 dense sizes 0.8B-27B)
- https://github.com/QwenLM/FlashQLA (README: SM90/SM100/SM103/SM120/SM121 requirement, 2-3x/2x vs FLA Triton on H200/GB200, MIT, news v0.1.1-v0.1.3, FLA backend)
- https://arxiv.org/abs/2412.06464 (Gated DeltaNet paper, ICLR 2025, Yang/Kautz/Hatamizadeh — the GDN Qwen adopts; note: NVIDIA/MIT authors, not Qwen)
- https://arxiv.org/abs/2309.00071 (YaRN — Qwen's stated 1M-context method)
- transformers PRs: #40771 (Qwen3-Next, merged 2025-09-09), #43830 (Qwen3.5, merged 2026-02-09), #48337 (Qwen4Exp, merged 2026-08-26), #45513 (GDN multi-token cached forward fix), #45637/#45638 (MTP, unmerged), #46831/#46859 (CPU-offload garbage / varlen GDN), #47887 (GDN A in fp32, closed unmerged) — https://github.com/huggingface/transformers
- vLLM PRs: #53896, #53899, #54129, #54513, #54517, #54873, #54890, #54915, #55272, #55309, #55430, #55617 — https://github.com/vllm-project/vllm
- Cross-referenced local dossiers: docs/research/vllm.md, docs/research/sglang.md, docs/research/tensorrt-llm.md, docs/research/flashinfer.md
