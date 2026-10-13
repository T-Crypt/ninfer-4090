# Dossier: DFlash / PARD (parallel-draft speculative decoding)

Researched 2026-10-07 (public sources only: arXiv 2602.06036 (DFlash) & 2504.18583 (PARD, ICLR 2026);
GitHub repos `z-lab/dflash` and `AMD-AGI/PARD`; the Inco AI DFlash2 blog; NVIDIA TensorRT-LLM
spec-decoding docs; Hugging Face model cards). No installs, no weight downloads, no GPU jobs.

This dossier covers the **parallel-draft** line of speculative decoding: the drafter predicts a whole
*block* of tokens in a **single forward pass** (T_draft ≈ one pass, independent of block length), unlike
the autoregressive drafters in `eagle3-medusa-specforge.md` (EAGLE drafts token-by-token; Medusa heads
are one-step) and unlike our in-checkpoint **MTP** (K sequential heads). Both methods here are shipped in
TensorRT-LLM (`decoding_type: PARD` / `DFlash`). They are the closest "MTP-family alternative" for the
next dense Qwen. Cross-referenced: `tensorrt-llm.md` (TRT-LLM integration + sm_89),
`eagle3-medusa-specforge.md` (DFlash2/Domino/DSpark checkpoint head-to-heads already tabulated there),
`vllm.md` / `sglang.md` (engine serving paths).

## What it is / backing / license

Two distinct methods, one shared idea (parallel drafting), different backers:

- **DFlash** — a **block-diffusion** drafter: a small (1/3/5-layer, 8 for Qwen3-Coder) transformer that
  denoises a masked block of tokens in one pass. Its key move is **KV-injection conditioning**: hidden
  states sampled uniformly from the target's layers are fused and injected into the *K/V of every draft
  layer* (stored in the drafter's own KV cache, reused across drafting iterations). Unlike EAGLE-3 (which
  fuses target features only as the draft *input* and thus dilutes with depth), DFlash's per-layer
  injection lets acceptance length scale with draft depth, so a 5-layer / 16-token draft is *cheaper* and
  *better* than EAGLE-3's 1-layer / 8-token draft. **Target-dependent**: needs a per-target drafter.
  (arXiv [2602.06036](https://arxiv.org/abs/2602.06036), CC BY 4.0)
- **DFlash 2** (Inco AI, 2026-08-18) — same one-pass block drafter plus two cheap additions that recover
  the "coherence headroom" plain DFlash leaves on the table: (a) a **path selector** (keep top-16
  candidates/position, score adjacent pairs with 256-dim token embeddings + a context gate = low-rank
  bilinear term; +2.0M params, +0.6% cycle latency), and (b) a **two-tap dynamic convolution** before/after
  each attn+FF sublayer to fix *suffix decay* (the block's within-block attention mass shrinks 30%→8% from
  layer 1→5; +16.5M params / +3%, +0.7% latency; 5L+conv ≈ 15L). Net: "over 20% more output per verify
  pass, ~1% latency, output provably unchanged; gain runs 16–25%."
- **PARD** (PARallel Draft, [2504.18583](https://arxiv.org/abs/2504.18583)) — **target-independent**:
  adapt a small *off-the-shelf autoregressive* drafter (e.g. LLaMA-3.2-1B, Qwen2.5-0.5B) into a parallel
  drafter via mask-token (Mask-Predict) training, so one drafter serves a whole family (LLaMA3/3.1/3.3,
  Qwen2/2.5). Training cost is cut by **COD** (COnditional Drop-token, retention `r=0.7`, `r_min=0.2`):
  geometric decay of retained tokens keeps prefix K/V states complete, dropping K·N training tokens → 2N
  (r=0.5) — "3×" vs masked-prediction training, and "7× higher than EAGLE / 10× higher than EAGLE-3"
  training efficiency. **Target-independent** = one drafter per family (no per-model retrain), at the cost
  of lower acceptance than target-dependent methods (DFlash's paper puts PARD's ceiling at ~3× because
  "the resulting small models lack the modeling capacity of the target LLMs").

**Backing:** DFlash — **Z Lab / UC San Diego** (Jian Chen, Yesheng Liang, Zhijian Liu; z-lab.ai), with
Inference productization + DFlash2 by **Inco AI** (inco.ai — "building the inference stack scaled to the
token economics of tomorrow"; Modal is the infra partner). PARD — **AMD** (AIG/AGI, Zihao An, Huajun Bai
(AMD/Tsinghua), Ziqiong Liu, Dong Li, Emad Barsoum); ICLR 2026. *The target row's "inco.ai" label is the
DFlash/DFlash2 side; PARD is AMD, not inco.*

**License (portable in all cases):** DFlash code **MIT** (`z-lab/dflash/LICENSE`, "Copyright (c) 2026 Z
Lab"); DFlash paper CC BY 4.0. DFlash2 draft checkpoints **Apache-2.0** (HF tag `license:apache-2.0` on
`incoai/Qwen3.8-27B-DFlash2`; z-lab mirrors are Apache-2.0 per `eagle3-medusa-specforge.md`). PARD **MIT**
(`AMD-AGI/PARD` root `LICENSE`; GitHub API `spdx_id: MIT`). No GPL, no non-commercial.

## Repo, version, last release date

- [z-lab/dflash](https://github.com/z-lab/dflash): created 2026-01-04; last push **2026-08-18** (DFlash2);
  6,140★ / 434 forks / 107 open issues; no tags — versioned by the DFlash (Jan) → DFlash2 (Aug) release
  line. `pip install dflash` (server-only) / `dflash[local]` (Transformers on Linux, MLX on Apple
  Silicon). DFlash2 checkpoints: `z-lab/Qwen3.8-27B-DFlash2`, `z-lab/Muse-Glimmer-30B-DFlash2`
  ([collection](https://huggingface.co/collections/z-lab/dflash-2)); DFlash v1 collection covers
  Qwen3.6 (27B, 35B-A3B), Qwen3.5 (4B/9B/27B/35B-A3B/122B-A10B/397B-A17B), Qwen3 (4B/8B non-thinking,
  Coder-Next, Coder-30B-A3B), Gemma 4, MiniMax M2.5/M2.7, Kimi K2.5/K2.6/K2.7-Code, GPT-OSS, Llama-3.1-8B,
  GLM 5.1.
- [AMD-AGI/PARD](https://github.com/AMD-AGI/PARD) (redirected from `AMD-AIG-AIMA/PARD`): created 2025-02-06;
  last push **2026-06-10**; 35★ / 4 forks / 4 open issues. `vllm` integration is the reference runtime.
- **Engine integrations** (the "shipped" surface): SGLang [PR 35371](https://github.com/sgl-project/sglang/pull/35371)
  (`--speculative-algorithm DFLASH`), vLLM [PR 52816](https://github.com/vllm-project/vllm/pull/52816)
  (`"method":"dflash"`), llama.cpp [PR 27342](https://github.com/ggml-org/llama.cpp/pull/27342)
  (`--spec-type draft-dflash`), ollama [PR 17865](https://github.com/ollama/ollama/pull/17865), oMLX fork;
  **TensorRT-LLM** ships both natively (`PARDDecodingConfig` / `DFlashDecodingConfig`;
  [spec-decoding doc](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md)).
  Ecosystem drafters: NVIDIA, Red Hat, Modal (DFlash); Meta Muse-Glimmer, Poolside Laguna, Xiaomi MiMo,
  NVIDIA Nemotron-3.5-Lightning ship official drafters; "3.5M HF downloads (as of Aug 2026)" (inco blog).

## Runs on an RTX 4090 (sm_89)?

**Partly — the DFlash drafter runs on sm_89 (no published 4090 numbers); PARD has none.**

- **DFlash drafter is a tiny (5-layer) block-diffusion transformer** → runs on any CUDA GPU, so it is
  sm_89-compatible. The 4090 routes, per the inco blog + z-lab README:
  - **llama.cpp [PR 27342](https://github.com/ggml-org/llama.cpp/pull/27342)** is the clean 4090 path:
    `llama-server -hf ggml-org/Qwen3.8-27B-GGUF:Q4_K_M -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M
    --spec-type draft-dflash --spec-draft-n-max 7`. (GGUF draft exists for the 4090 route.)
  - **SGLang / vLLM / llama.cpp OpenAI server** (`dflash generate openai --base-url ...`) or the
    Transformers backend (Linux). The inco blog also shows a **prebuilt oMLX** build — that is Apple
    Silicon (M5 Max), not a 4090; the M5 Max DFlash2 video is the only *consumer* hardware evidence, and
    it is not sm_89.
  - **TensorRT-LLM on 4090**: `DFlashDecodingConfig(attention_backend="VANILLA")` = "FlashAttention …
    **runs anywhere**" (works on sm_89); the fast `"TRTLLM"` backend (TRTLLM-Gen FMHA over a private
    paged context K/V) is **SM100/SM103 only** and `"FA4"` is SM90 only. So on a 4090 you are on the
    VANILLA path; the target model's GDN verify would use the Triton fallback (see `tensorrt-llm.md`).
- **PARD**: inference evaluated on **vLLM + A100-40GB** only; trained on **8×MI250X** (AMD). **No sm_89 /
  4090 data anywhere** (not found).
- **No 4090 tok/s is published for either method** — every number below is H200/B200 (DFlash paper),
  A100-40GB (PARD), or M5 Max (inco DFlash2 consumer demo). The z-lab H200 Qwen3.8-27B DFlash2 figures
  (in `eagle3-medusa-specforge.md` §C) are the reference to check *ratios* against, not absolutes.

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes for DFlash, no evidence for PARD.** DFlash conditions on the target's *hidden
  states* (all layers), **not** on attention-only features — so it is *not* subject to EAGLE-3's
  "aux layers must be full-attention layers" constraint (the GDN catch in `eagle3-medusa-specforge.md`:
  thoughtworks Coder-Next needed a fork + only 3-of-48 aux layers). DFlash/DFlash2 checkpoints exist for
  GDN-hybrid Qwens (Qwen3-Coder-Next, Qwen3.6-27B/35B-A3B) and DFlash2 for **Qwen3.8-27B** (our current
  dense hybrid). **PARD** was trained/evaluated only on dense **Qwen2/2.5** and LLaMA3 — no GDN hybrid,
  no Qwen3.x, no MTP head; nothing public shows it on a GDN target.
- **MTP: both are the parallel alternative to MTP, and DFlash2 directly beats it.** DFlash2 vs native
  MTP on Qwen3.8-27B (inco Table 4, block size 8): DFlash2 mean accept **4.80** > MTP **4.28** (GSM8K 5.46
  vs 5.02; MT-Bench 4.10 vs 3.74). On Qwen3.5-4B (Table 3, thinking, T=1.0): DFlash2 mean **5.97** vs MTP
  **4.54** (gains "1.05 tokens over DFlash (21%)"). MTP remains in the comparison set throughout — these
  are drop-in alternatives, not a replacement for the MTP head itself.
- **Vision: not covered.** Both drafters are text-LM drafters; no vision/VLM drafter found (not found).
- **256K: not run; the drafter degrades past 4K without fine-tuning but recovers.** DFlash §5.4
  ([Table 4](https://arxiv.org/html/2602.06036v2)): a base Qwen3.5-27B drafter (trained on 4K) fine-tuned
  on **1.6K LongAlign-10K** samples (3 epochs) tested on LongBench — acceptance length by context
  (hotpotqa / qasper / gov_report): 1K 4.91/5.27/4.53 → 4K 4.91/5.17/3.93 → **16K 3.61/3.57/2.67 (base)
  vs 6.05/6.00/3.81 (long-finetuned)** → 32K (gov_report) base 2.09 vs long 3.56. "The extracted target
  features remain representative at long contexts"; no 256K run exists. Spec decoding is context-agnostic
  by construction (the drafter sees the same context as the target).

## Published numbers on a 4090 or similar

All copied verbatim with source + model + batch. **None are on a 4090** (DFlash = H200/B200 batch 1
unless noted; PARD = A100-40GB batch 1; serving tables are the only concurrency sweep).

**A. DFlash paper (arXiv [2602.06036](https://arxiv.org/abs/2602.06036), v2).** Claim: "over 6× lossless
acceleration … up to 2.5× higher speedup than EAGLE-3." Table 1 (Transformers backend, Qwen3, thinking
off, max 2048 tokens; speedup over AR / τ) — DFlash(16) vs EAGLE-3(16)/(60):

| Model (T=0) | GSM8K | MATH-500 | AIME25 | HumanEval | MBPP | MT-Bench | Avg τ |
|---|---|---|---|---|---|---|---|
| Q3-4B DFlash(16) | 5.15× / 6.53 | 6.09× / 7.84 | 5.68× / 7.27 | 5.21× / 6.64 | 4.78× / 6.09 | 2.85× / 4.35 | 4.91 |
| Q3-4B EAGLE-3(60) | 2.27× / 3.77 | 2.10× / 3.52 | 2.13× / 3.51 | 2.12× / 3.47 | 2.02× / 3.38 | 3.49× / 2.08 | 3.48 |
| Q3-8B DFlash(16) | 5.15× / 6.54 | 6.08× / 7.87 | 5.62× / 7.08 | 5.14× / 6.50 | 4.65× / 5.95 | 2.75× / 4.24 | 4.86 |
| Q3-8B EAGLE-3(60) | 2.23× / 3.71 | 2.05× / 3.49 | 2.05× / 3.44 | 2.17× / 3.65 | 1.93× / 3.25 | 3.26× / 2.02 | 3.40 |

- §5.1: "average speedup of **4.9×** over the AR baseline, a **2.4×** improvement over EAGLE-3(16); at
  temperature=1, **4.1×** over baseline and **2.2×** over EAGLE-3."
- §5.2 (thinking on, Transformers): Q3-4B T=0 GPQA 4.23×/5.23, MATH-500 4.59×/5.74, AIME25 4.39×/5.54;
  Q3-8B T=0 GPQA 4.17×/5.17, MATH-500 4.64×/5.82, AIME25 4.51×/5.74.
- §5.3 serving, **single B200**, SGLang FA4 backend, tok/s @ C1/4/8/16/32: Qwen3-8B MATH-500 DFlash
  1175/3884/7485/12268/16076 (5.1×/4.5×/4.5×/3.9×/2.8×; AR 230/861/1666/3133/5694); Qwen3-Coder-30B-A3B
  HumanEval DFlash 802/2078/3442/5429/8314 (3.5×/3.0×/3.2×/3.2×/3.1×). "up to **5.1×** on Qwen3-8B."
- §5.5.1 LLaMA-3.1-8B-Instruct on **single B200** (FlashInfer): GSM8K baseline 249/923/1739/3245/5349
  tok/s; EAGLE-3(10) 1.6/1.5/1.4/1.2/1.0×; EAGLE-3(60) 1.9/1.6/1.3/0.9/0.6×; **DFlash(10) 2.4/2.2/2.1/1.8/1.6×**
  (avg τ 4.32) — DFlash stays >1× to C32 where EAGLE-3(60) falls to 0.6×.

**B. DFlash2 (inco blog [2026-08-18](https://inco.ai/blog/dflash2/)).** "close to 3× the speed of
autoregressive decoding, about a third of the compute per token, with the same output."
- Table 2 (5L Qwen3-4B GSM8K, accept length): DFlash T=0 4.27 / T=1 3.78; +DSpark correction (+77.8M,
  +9.6%) 4.49 / 4.08; **+path selection (+2.0M, +0.6%) 4.61 / 4.25**. Recall@1 85.4%@pos0 → 72.9%@pos6;
  Recall@16 99.5% → 87.8% (oracle selection lifts accept 4.27→6.79).
- Table 3 (Qwen3.5-4B, thinking, T=1.0, top-p 0.95, top-k 20, presence 1.5, lossless rejection sampling),
  per-request mean accept — MTP / DFlash / DSpark / DFlash2: GSM8K 4.78/4.99/5.69/**6.20**, MATH-500
  5.04/5.42/6.20/**6.76**, HumanEval 4.84/5.43/5.80/**6.28**, MBPP 4.16/4.49/4.96/**5.41**, MT-Bench
  3.90/4.26/4.77/**5.20**, mean 4.54/4.92/5.49/**5.97** (combined selector+conv adds **1.3%** cycle latency).
- Table 4 (Qwen3.8-27B, default sampling, block 8) MTP / DSpark / DFlash2: GSM8K 5.02/4.36/**5.46**,
  MATH-500 4.72/3.92/**5.28**, HumanEval 3.91/3.30/**4.39**, MBPP 3.99/3.51/**4.79**, MT-Bench 3.74/3.01/
  **4.10**, mean 4.28/3.62/**4.80**. → "SGLang serves at **2.7–3.4×** AR at batch 1" (Qwen3.8-27B),
  3.1–4.6× (Muse Glimmer). (Full tok/s C1/C8/C32 breakdown is in `eagle3-medusa-specforge.md` §C: H200,
  DFlash2 C1 3.43×, C32 1.45× on GSM8K.)
- Adoption: "NVIDIA measured **up to 15×** throughput … on Blackwell"; Google "3× more tokens per second
  on TPUs"; CoreWeave's Kimi K2.7 Code production endpoint "runs DFlash by default."

**C. PARD (arXiv [2504.18583](https://arxiv.org/abs/2504.18583), v4).** vLLM, **A100-40GB**, batch 1,
greedy. Table 1 (target / AR / VSD / PARD, TPS — HumanEval · GSM8K · SpecBench · Avg):

| Target | AR avg | VSD avg | EAGLE-3 avg | PARD avg (HumanEval / GSM8K / SpecBench) |
|---|---|---|---|---|
| Qwen2 7B | 76.14 | 164.78 (2.16×) | — | 208.49 (2.74×) (268.98/221.43/135.05) |
| Qwen2.5 3B | 130.28 | 202.21 (1.55×) | — | 345.06 (2.65×) (425.96/406.17/203.05) |
| Qwen2.5 7B | 75.99 | 149.73 (1.97×) | — | 241.34 (3.18×) (285.82/292.56/145.64) |
| Qwen2.5 14B | 40.83 | 121.05 (2.97×) | — | 151.59 (3.71×) (181.19/182.25/91.33) |
| LLaMA3 8B | 73.35 | 123.70 (1.69×) | — | 194.70 (2.65×) (235.84/200.45/147.80) |
| **LLaMA3.1 8B** | 73.05 | 134.28 (1.84×) | 193.86 (2.65×) | **219.15 (3.00×)** (264.88/235.09/157.49) |
| LLaMA3.3 70B | 24.18 | 70.31 (2.91×) | 77.88 (3.22×) | 84.29 (3.49×) (99.88/95.58/57.41) |

- Headline (abstract/§6): "PARD achieves up to **3.67×** speedup on LLaMA3.1-8B, reaching **264.88 tokens
  per second** … **1.15× faster than EAGLE-3**" (the table's LLaMA3.1-8B HumanEval cell is 264.88 / 3.63×;
  3.67× is the abstract's rounded peak).
- Table 2 (LLaMA3.1-8B, k-α acceptance): EAGLE 1-α 0.83/0.79, 4-α 0.72/0.66; EAGLE-3 0.87/0.82, 0.85/0.79;
  **PARD 0.93/0.88, 0.90/0.85** (HumanEval / GSM8K). Table 3 (draft-phase memory BW, bf16, k=4/6/8):
  EAGLE/EAGLE-3 5.94/8.90/11.88 GB (grows with k); **PARD 2.48 GB constant**. Training: 8×MI250X, TRL,
  4 epochs, k=8, r=0.7, r_min=0.2. Large-batch (Appendix D): PARD **1.33×–3.63×** at batch 1–16.
  Extrapolation: shared mask-token ID lets `K_infer > K_train` (best K_infer=12, stable K_train≥8).

## Ideas we could port into ninfer-4090

Ranked by relevance to our dense-hybrid + MTP3 + 24 GB box. (DFlash/DFlash2 checkpoint + llama.cpp
PR-27342 GGUF route is the one that runs on a 4090 *this week* — see below.)

1. **KV-injection conditioning (DFlash's core, and the thing that makes deep drafters worth it).** Inject
   the fused target hidden feature into the **K/V of every draft layer** (kept in the drafter's KV cache,
   reused across draft iterations) instead of EAGLE-3's input-only fusion. Consequence: acceptance length
   *scales with draft depth* (the EAGLE-3 dilution problem disappears), so a small multi-layer drafter
   beats a 1-layer EAGLE head at equal or lower latency. In NInfer terms: our verify pass already computes
   the target distribution over all drafted tokens; a drafter that reads the final hidden state (like MTP)
   and conditions every layer on it is architecturally compatible with a GDN target (unlike EAGLE-3's
   attention-only aux layers). This is the strongest reason the *parallel-draft* line is a better fit for
   our hybrid than EAGLE-3.
2. **One-pass block drafting as a verify-budget knob.** T_draft = one pass, independent of block length
   γ (EAGLE/MTP draft cost grows linearly with γ). On a bandwidth-bound 4090 decode, this is exactly the
   right shape: spend the draft budget on *more* parallel positions (bigger block) rather than more
   sequential steps. If we ever add a non-MTP drafter, draft K tokens in one forward pass and keep the
   block at the point where acceptance saturates (DFlash: block 16 for the 4B/8B; block 8 for DFlash2
   Qwen3.8; the inco selector shows the marginal value is picking the *right path*, not more positions).
3. **DFlash2's two cheap upgrades (nearly free if we train our own drafter).** (a) **Path selector** over
   the drafter's own top-16 lists — a low-rank bilinear adjacent-pair scorer (256-dim token embeddings +
   context gate), +2.0M params / +0.6% latency, +0.34 accept tokens at T=0; our verify pass already sees
   the target distribution, so a pre-verify path filter is a small addition and also addresses MTP-S15's
   "verify a path the target rejects" failure (0.83×@C32 in `eagle3-medusa-specforge.md` §6). (b) **Two-tap
   dynamic conv** (Canon/Dynamic-Short-Conv family) to fix suffix decay — 5L+conv ≈ 15L for +3% params /
   +0.7% latency; the block's within-block attention mass falls 30%→8% across layers, and the conv absorbs
   the local work so attention can read context again.
4. **PARD's COD mechanism — cheaply parallelize an existing small drafter.** COD (retention `r=0.7`,
   `r_min=0.2`, geometric decay with complete prefix K/V) adapts an *autoregressive* small model into a
   parallel drafter at K·N→2N training cost (3× vs masked prediction, "7×/10×" vs EAGLE/EAGLE-3 training
   efficiency). Its **target-independence** (one drafter per family) is attractive if the next Qwen is
   bigger than our 24 GB and we want a drafter that doesn't need a per-checkpoint retrain — but note its
   acceptance ceiling (~3×) is below DFlash2/MTP on our models, so this is a *training-method* import, not
   a *checkpoint* import.
5. **TRT-LLM's `relaxed_topk`/`relaxed_delta` acceptance for thinking** (MTP config, `tensorrt-llm.md`)
   still applies to our MTP3 head on a reasoning next-Qwen, independent of which drafter we pick.

## Worth running beside NInfer/llama.cpp?

**Yes for DFlash2-on-llama.cpp (the 4090-runnable artifact); PARD is a no on this box.**

- **The clean 4090 test (low effort, high information):** llama.cpp PR 27342 on one 4090, same
  `Qwen3.8-27B` GGUF (groupwise-int or FP8 — not NVFP4, per `sglang.md`), three arms: (1) AR baseline,
  (2) `--spec-type draft-mtp` (our MTP3's public analog), (3) `--spec-type draft-dflash` with
  `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` (`--spec-draft-n-max 7`) — plus NInfer itself (MTP3). Workloads:
  the pedroalonso-style 5-turn regenerate-the-file agentic session **and** HumanEval/MT-Bench single-turn
  (the code-vs-chat split is where accept rates diverge most: DFlash2 Qwen3.8-27B HumanEval 4.39 vs
  MT-Bench 4.10; DSpark loses to MTP on MT-Bench, 3.01 vs 3.74). Report output tok/s + acceptance length
  at C1 and C8. Check the *ratios* against the z-lab H200 card (DFlash2/MTP ≈ 1.33× at C1, ≈1.30× at C8),
  not the absolutes. This directly answers the target's "MTP-family alternative, check acceptance on
  agentic text" question on our exact model.
- **Why not PARD on a 4090:** it is vLLM + A100-40GB only, has no sm_89 numbers, no GDN/Qwen3.x
  checkpoint, and is target-independent (its acceptance ceiling is below the target-dependent DFlash2/MTP
  on our models). Its value to us is the *COD training method* (idea 4), not a side-by-side run.
- **Why DFlash2 beats EAGLE-3 as the 4090 drafter to test:** no per-target EAGLE3 checkpoint exists for
  Qwen3.8-27B (only Qwen3-32B / Coder-Next community heads in `eagle3-medusa-specforge.md`), and EAGLE-3's
  attention-layer aux restriction is architectural on our GDN hybrid; DFlash2 reads the final hidden state
  and has a GGUF for the 4090 route out of the box.

## Risk / unknowns

- **No 4090 numbers exist for either method.** Every table is H200/B200 (DFlash), A100-40GB (PARD), or
  M5 Max (DFlash2 consumer demo). On a 4090 the *verify* pass is bandwidth-bound the same way, so C1
  ratios should roughly transfer (consistent with pedroalonso's 4090 MTP 2.4–2.9× vs the A100/H200 MTP
  2.59× in `eagle3-medusa-specforge.md`), but the 24 GB KV budget at 256K forces KV quantization that
  changes accept dynamics — untested anywhere.
- **The parallel-vs-sequential question is model-specific and the data splits.** DFlash2 > MTP on
  Qwen3.5-4B and Qwen3.8-27B (mean accept 5.97/4.80 vs 4.54/4.28); MTP ≥ DFlash/DSpark on Qwen3.6-35B-A3B
  (B200, `eagle3-medusa-specforge.md` §D). Whether the *next* dense Qwen's MTP head is strong or weak
  decides whether MTP3 already suffices — not answerable from public data. Parallel drafting's edge is
  biggest when the drafter is *cheap enough to run many positions* (small target, C1); at high
  concurrency the verify pass dominates and the gap closes (DFlash B200 C32: 2.8–2.9× vs 5.1× at C1;
  DFlash2 C32 GSM8K 1.45×) — the same curve our 1–8 concurrency box sits on.
- **DFlash's drafter is target-dependent (per-model retrain); PARD's is target-independent but weaker.**
  For the next Qwen we either wait for a published DFlash2 drafter (NVIDIA/RedHat/Modal/inco have all
  landed drafters for other models — a next-Qwen drafter may be downloadable rather than trained), or
  train one (SpecForge, MIT, per `eagle3-medusa-specforge.md` — it already has DFlash/DSpark recipe
  YAMLs). PARD's one-drafter-per-family property is the cheaper hedge but buys a lower ceiling.
- **DFlash2 GGUF is INT4-quantized draft** (`:Q4_K_M`); the public cards are BF16/FP8. Accept-length loss
  from INT4 draft quantization is not published; the MLX note that "quantized matmul is less efficient at
  larger verify widths, use `block_size ≤ 5`" hints the 4090 run may need a smaller block than the H200
  block-8.
- **The GDN verify pass is the unsolved engine piece** (cross-ref `sglang.md` / `tensorrt-llm.md`):
  verifying 7–16 drafted tokens over a hybrid KV needs the ReplaySSM-class mamba-state-replay mechanism or
  equivalent. Whatever drafter we run, the 4090 work is in the verify, not the draft.
- **PARD is thin/low-momentum** (35★, 4 forks, AMD-AGI, last push 2026-06) and only ever wired into
  vLLM (+ TRT-LLM); its practical future is subsumed by DFlash/DFlash2 (inco, 6.1k★, wired into
  SGLang/vLLM/llama.cpp/ollama/TRT-LLM + NVIDIA/RedHat/Modal/Meta drafters). The big-diffusion-drafter
  line DFlash positions against (DiffuSpec, SpecDiff-2 — 7B drafters, 3–4× ceiling, per the DFlash paper)
  does not fit a 24 GB box and is deprioritized; likewise TiDAR (non-lossless).

## Sources

- DFlash paper: <https://arxiv.org/abs/2602.06036> (v2 HTML <https://arxiv.org/html/2602.06036v2>, CC BY 4.0).
- PARD paper: <https://arxiv.org/abs/2504.18583> (v4 HTML <https://arxiv.org/html/2504.18583v4>; ICLR 2026
  proceedings <https://proceedings.iclr.cc/paper_files/paper/2026/file/ea17f1a1685d65e66af19da44471b9ae-Paper-Conference.pdf>).
- DFlash repo: <https://github.com/z-lab/dflash> (README, MIT `LICENSE`; 6,140★; DFlash/DFlash2 model
  collections). PARD repo: <https://github.com/AMD-AGI/PARD> (redirected from `AMD-AIG-AIMA/PARD`; MIT;
  35★; vLLM integration).
- Inco AI DFlash2 blog (2026-08-18): <https://inco.ai/blog/dflash2/> (Tables 1–5, path selector, suffix
  decay/conv, adoption, M5 Max oMLX).
- TRT-LLM integration: <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/speculative-decoding.md>
  (`PARDDecodingConfig`, `DFlashDecodingConfig` + `attention_backend` VANILLA/TRTLLM(SM100/SM103)/FA4(SM90),
  DFlash2 support, `decoding_type` list).
- Model cards: <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2> (Apache-2.0, 289,632 downloads),
  <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> + <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>,
  <https://huggingface.co/collections/z-lab/dflash> (DFlash v1), <https://huggingface.co/collections/z-lab/dflash-2>.
- Engine PRs: SGLang <https://github.com/sgl-project/sglang/pull/35371>, vLLM <https://github.com/vllm-project/vllm/pull/52816>,
  llama.cpp <https://github.com/ggml-org/llama.cpp/pull/27342>, ollama <https://github.com/ollama/ollama/pull/17865>.
- Cross-referenced internal dossiers: `eagle3-medusa-specforge.md` (DFlash2/Domino/DSpark Qwen3.6/3.8-27B
  head-to-heads + H200 C1/C8/C32 tok/s), `tensorrt-llm.md` (sm_89 + GDN Triton fallback), `vllm.md`
  (`dflash`/`eagle3` methods), `sglang.md` (ReplaySSM verify, sm_89 status).
