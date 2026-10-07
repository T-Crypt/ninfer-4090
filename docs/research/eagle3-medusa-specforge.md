# Dossier: EAGLE-3 / Medusa / SpecForge (draft-model speculative decoding vs our MTP3)

Researched 2026-10-07 (public sources only: GitHub repos `SafeAILab/EAGLE`, `sgl-project/SpecForge`,
`FasterDecoding/Medusa`; arXiv papers; LMSYS blogs; Hugging Face model cards; two community blogs and
one community benchmark repo). No installs, no weight downloads, no GPU jobs.

This dossier covers the **draft-model side** of speculative decoding: EAGLE-3 (feature-fusion draft head,
the technique line behind EAGLE-2/EAGLE-3.1/P-EAGLE), Medusa (parallel decoding heads, 2024-era, now
dormant), and SpecForge (the LMSYS/SGLang training framework that produces the EAGLE3/DFlash/Domino/DSpark
draft checkpoints the ecosystem actually ships). The engine-side sm_89 story (SGLang/vLLM/llama.cpp EAGLE/MTP
support) is in `sglang.md`, `vllm.md`, `tensorrt-llm.md` — cross-referenced below.

## What it is / backing / license

- **EAGLE / EAGLE-2 / EAGLE-3** — speculative *sampling* that drafts at the feature level: the draft model
  is a tiny single-decoder-layer transformer conditioned on the target model's own hidden states. EAGLE-3
  (arXiv [2503.01840](https://arxiv.org/abs/2503.01840), NeurIPS'25) removes EAGLE's feature-prediction
  constraint (direct token prediction via "training-time test") and fuses low/mid/high-level target
  features instead of top-layer features only. Backing: **Peking University** (Yuhui Li, Chao Zhang),
  **Microsoft Research** (Fangyun Wei), **U. Waterloo / Vector Institute** (Hongyang Zhang); maintained as
  SafeAI Lab (SAIL), repo [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) (formerly SJTU-IPADS/EAGLE;
  the `SJTU-IPADS/EAGLE-3` and `SJTU-IPADS/SpecForge` paths 404 — the repos were transferred).
- **Medusa** — K extra lightweight decoding heads on the target's last hidden states + tree attention +
  "typical acceptance"; Medusa-1 (frozen backbone) / Medusa-2 (joint training). arXiv
  [2401.10774](https://arxiv.org/abs/2401.10774) (Cai, Li, Geng, Peng, Lee, Chen, Dao — Princeton / Together
  AI / UIUC / CMU / UConn). **Dormant since 2024-06** (last push 2024-06-25); the idea lineage it started is
  what EAGLE-3 and the DFlash family now supersede.
- **SpecForge** — "Train speculative decoding models effortlessly and port them smoothly to SGLang serving"
  ([sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)). Backing: **LMSYS / SGLang team**
  (Jiaping Wang, Shenggui Li et al.) with RadixArk (Cheng Mao, Yi Sun, Kan Wu), the Domino team (Jianuo
  Huang), Ant Group AQ, China Merchants Bank, and community; Voltage Park is the official infra partner.
  One typed entry point (`specforge train --config ...`) over drafting families: **EAGLE3** (+ optional LK
  loss), **P-EAGLE**, **EAGLE3.1**, **DFlash** (+ optional D-PACE), **Domino**, **DSpark**; online (SGLang
  `--enable-spec-capture` → Mooncake feature store → trainer pools) or offline; FSDP/USP topologies.
  Trained checkpoints publish to the **SpecBundle** HF collection
  ([lmsys/specbundle](https://huggingface.co/collections/lmsys/specbundle)); "Nine of the eleven checkpoints
  listed below came in this way" (community/partner-trained, open data only) (v0.3 blog).
- **License:** EAGLE repo **Apache-2.0** (`LICENSE`: "Copyright 2025 SafeAI Lab (SAIL)", Apache 2.0; the
  GitHub API reports "Other"/NOASSERTION, the checked-in file is Apache-2.0). Medusa **Apache-2.0**.
  SpecForge **MIT**. Draft checkpoints: Apache-2.0 or MIT (z-lab DFlash2 Apache-2.0, Huang2020-domino MIT,
  CMB Apache-2.0, lmsys SpecBundle models MIT). **Porting is license-OK across the board** (no GPL, no
  non-commercial).

## Repo, version, last release date

- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE): version **v3.0.0** (README badge); last push
  **2026-02-20**; 2,548 stars / 297 forks / 101 open issues; **no GitHub releases or tags**. News: 2025-09-18
  "EAGLE-3 is accepted to NeurIPS'25"; 2025-07-23 "We strongly recommend using SpecForge for out-of-the-box
  training of EAGLE-3 with SGLang"; 2025-03-19 EAGLE-3 released. Todo list still has
  **"[ ] Support official EAGLE-3 for Qwen-3"** unchecked — i.e., no first-party Qwen3 draft; all Qwen3
  checkpoints in its table are community-trained (AngelSlim, Tengyunw, Zjcxy-SmartAI, nvidia, lmsys).
- [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge): **v0.3.0, 2026-08-04**
  ([LMSYS blog](https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/)); v0.2.0 + SpecBundle phase 1
  2025-12-23; v0.1 2025-07-25 with Llama-4 EAGLE3 checkpoints ([blog](https://www.lmsys.org/blog/2025-07-25-spec-forge/)).
  1,200 stars / 358 forks / 202 open issues; last push **2026-10-07** (daily cadence); no releases/tags via
  API. Paper: arXiv [2603.18567](https://arxiv.org/abs/2603.18567) "SpecForge: A flexible and efficient
  open-source training framework for speculative decoding" (2026).
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa): Apache-2.0; 2,770 stars / 205 forks /
  57 open issues; **last push 2024-06-25** (≈16 months stale); no releases. `pip install medusa-llm`;
  weights only for Vicuna-7B/13B/33B (v1.3/v1.5) and Zephyr-7B — **no Llama-3.1, no Qwen at all**.
  Adopted by TensorRT-LLM, TGI, RTP-LLM (its README "Community Adoption" list).

## Runs on an RTX 4090 (sm_89)?

**Partly.** The three projects publish essentially no 4090 results themselves; the sm_89 evidence is
community- and issue-sourced, and the production path for a 4090 is via the engines (llama.cpp / SGLang /
vLLM), not via the EAGLE/SpecForge repos.

- **EAGLE standalone code works on a 4090**: a user ran EAGLE-1 llama-2-7b on a 4090 ("The speed ratio is
  normal, but the alpha values do not match" — [issue #89](https://github.com/SafeAILab/EAGLE/issues/89));
  the EAGLE-1 README claims the draft is "trainable (within 1-2 days) and testable on 8x RTX 3090 GPUs".
  Inference code "automatically allocates model weights (loading a model across multiple GPUs)".
- **SpecForge trains on 4090-class hardware** (at least in the 8-GPU form): a user trained Qwen3-32B EAGLE3
  with `train_eagle3.py --target-model-backend sglang --tp-size 8` on **8×4090** (open issue
  [#380](https://github.com/sgl-project/SpecForge/issues/380) — GPU mem spikes ~15→23 GB/card at train
  start); the spec-capture SGLang server path is validated on "**one RTX 4090**" (PR
  [#886](https://github.com/sgl-project/SpecForge/pull/886), SGLang v0.5.20: `test_server_capture_gate.py`
  "2 passed"). The v0.3 training examples themselves target H20/B200 pools.
- **Serving on a 4090 goes through the engine** (cross-ref dossiers): SGLang has **no verified sm_89
  recipe for our Qwen hybrid** (sglang.md); vLLM ships EAGLE/EAGLE-3 (`"method": "eagle3"`) and DFlash
  (vllm.md); llama.cpp is the clean 4090 route — **MTP landed in mainline** (`--spec-type draft-mtp`,
  master b1544 `70adb1b4c` per [pedroalonso.net](https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090))
  and **DFlash2 runs via PR [27342](https://github.com/ggml-org/llama.cpp/pull/27342)
  (`--spec-type draft-dflash`) with `incoai/Qwen3.8-27B-DFlash2-GGUF`** ([inco blog](https://inco.ai/blog/dflash2/)).
- **No published 4090 decode tok/s for EAGLE-3 or Medusa from any of the three projects** (not found).
  Medusa: single-GPU inference with optional `--load-in-8bit`/`--load-in-4bit`; batch-1 only, by design;
  the paper's own numbers are A100-class.

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, but EAGLE-3's information diet is cut.** EAGLE3 draft heads exist for GDN-hybrid Qwens:
  [lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan](https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan)
  (trained on 1.4M open-perfectblend samples with SpecForge, SpecBundle phase 1) and
  [thoughtworks/Qwen3-Coder-Next-Eagle3](https://huggingface.co/thoughtworks/Qwen3-Coder-Next-Eagle3) (Qwen3-Coder-Next,
  36 GDN + 12 attention layers — our exact layer split). The GDN catch (Thoughtworks [blog](https://huggingface.co/blog/lujangusface/tw-eagle3-qwen3-coder-next),
  2026-04-15): **"EAGLE3 auxiliary layers must be selected from attention layers only, since GDN recurrent
  states are incompatible with speculative decoding"** — aux layers [3,23,47], i.e. 3 of 48 layers for
  Qwen3-Coder-Next vs the full stack in a plain transformer. Their measured consequence: Qwen3-Coder-Next
  mean EAGLE3 speedup **1.37x at B=1** (portfolio: Llama-3.1-8B 1.70x, GLM-4.7-FP8 1.69x, GLM-4.7-Flash
  1.66x, MiniMax-M2.5 1.39x, Gemma-4-31B 1.30x — all H200). Upstream SGLang had no EAGLE3 hooks for
  `qwen3_next.py` — they needed a fork (tails-mpt/sglang) with six patches including automatic GDN-layer
  filtering. For dense Qwen3.6/3.8-27B the same constraint applies to the 12 (or so) full-attention layers;
  the lmsys Qwen3-Next EAGLE3 recipe uses `--speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4`.
- **DFlash-family drafters exist for our exact dense-27B targets** (trained via SpecForge, served by
  SGLang/vLLM/TRT-LLM/llama.cpp): [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
  (mirror of incoai's; + [GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) for llama.cpp),
  [Huang2020/qwen3.6-27B-domino](https://huggingface.co/Huang2020/qwen3.6-27B-domino) (Domino b16 for
  Qwen3.6-27B, MIT), z-lab/Qwen3.6-27B-DFlash (official DFlash ckpt, rev `0919688`, per the domino card),
  and community [RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) (the
  DSpark/DSpark2 checkpoint question is tracked separately under the RadixArk target row). No EAGLE3
  checkpoint for a dense 27B was found (closest: Qwen3-32B community heads — CMB, nex-agi, thoughtworks,
  Zjcxy-SmartAI).
- **MTP: first-class in-checkpoint, and the head-to-head data now exists on our exact models.** Numbers
  below (domino card, z-lab DFlash2 card, inco blog, shreyansh26 benchmark). SGLang runs the in-checkpoint
  MTP head through its EAGLE/NEXTN machinery (`--speculative-algorithm EAGLE --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; sglang.md); vLLM uses
  `{"method":"mtp"}` (vllm.md); llama.cpp `--spec-type draft-mtp` (Qwen3.8 ships the heads as a separate
  sidecar GGUF, per pedroalonso.net).
- **Vision: not covered by any of these.** All draft heads are text-LM heads; the only VLM draft in
  SpecBundle is `AQ-MedAI/Qwen2.5-VL-72B-Instruct-eagle3` (old VLM, not the next Qwen). SpecForge v0.3's
  "What's Next" says VLM testing is upcoming. Vision support for the next Qwen is an engine-side question
  (see qwen-official.md).
- **256K: no public spec-decode run at 256K on a hybrid target found** (not found). Spec decoding is
  context-agnostic by construction (the draft head sees the same context as the target), and 256K non-spec
  runs on 4090-class boxes are documented in the KTransformers dossier (Qwen3-Coder-Next `--max-total-tokens
  256000`).

## Published numbers on a 4090 or similar

All copied verbatim. "4090-class" cells are noted; I have not converted or averaged anything.

**A. EAGLE-3 paper (arXiv [2503.01840](https://arxiv.org/abs/2503.01840v3))** — batch 1, temperature 0,
speedup / acceptance length τ; SGLang tables below are 1×H100 (SGLang v0.4.4), vLLM table per the paper
text "on RTX3090" (table caption says "on A100" — the paper is inconsistent):

| Model (batch 1) | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | Mean |
|---|---|---|---|---|---|---|
| Vicuna-13B EAGLE-3 | 5.58x (τ 6.65) | 6.47x (7.54) | 5.32x (6.29) | 5.16x (6.17) | 5.63x (6.38) | 6.32x |
| LLaMA-3.1-8B EAGLE-3 | 4.40x (6.13) | 4.85x (6.74) | 4.48x (6.23) | 4.82x (6.70) | — | — |
| LLaMA-3.3-70B EAGLE-3 | 4.11x (5.63) | 4.79x (6.52) | 4.34x (6.15) | 4.30x (6.09) | — | — |
| DS-R1-Distill-Llama-8B EAGLE-3 | 4.05x (5.58) | 4.59x (6.38) | 5.01x (6.93) | 3.65x (5.37) | — | — |
| Vicuna-13B EAGLE-2 | 4.26x (4.83) | 4.96x (5.41) | 4.22x (4.79) | 4.25x (4.89) | 4.09x | 4.38x |
| Vicuna-13B Medusa | 2.07x (2.59) | 2.50x (2.78) | 2.23x (2.64) | 2.08x (2.45) | 1.71x (2.09) | 2.12x |

- SGLang, 1×H100, Llama-3.1-8B, MT-bench, chain length 3, no tree: baseline **158.34 tok/s**; EAGLE-2
  **244.10**; EAGLE-3 **373.25**. Batch sweep (throughput vs no-spec): EAGLE-3 1.81x@b2, 1.82x@b4,
  1.62x@b8, 1.48x@b16, 1.39x@b24, 1.32x@b32, 1.38x@b48, 1.34x@b56, **1.38x@b64**; EAGLE falls below 1.0x
  from b24 (0.93x@24 … 0.88x@48). (Abstract: "EAGLE-3 achieves a 1.38x throughput improvement at a batch
  size of 64.")
- vLLM, Llama-3.1-8B, MT-bench, chain length 2: EAGLE-3 1.75x@b2, 1.68x@b4, 1.58x@b8, 1.49x@b16, 1.42x@b24,
  1.36x@b32, 1.21x@b48, 1.01x@b56.
- Medusa paper ([2401.10774](https://arxiv.org/abs/2401.10774), v3 abstract): "Medusa-1 can achieve over
  2.2x speedup … Medusa-2 further improves the speedup to 2.3–2.8x" (v4 abstract says 2.3–3.6x); Vicuna-7B:
  Medusa-1 **2.18x**, Medusa-2 (two-stage) **2.83x**, "without compromising generation quality"; training
  "5 hours for Medusa-1 on Vicuna 7B model with a single NVIDIA A100 PCIE GPU … on 60k ShareGPT samples";
  roofline analysis on A100-80GB-PCIe/A40/A6000; simulated speedup "when the batch size exceeds 32, the
  speedup decreases and may even have a negative effect."

**B. Qwen3.6-27B (dense, hybrid) — domino model card** ([Huang2020/qwen3.6-27B-domino](https://huggingface.co/Huang2020/qwen3.6-27B-domino)):
`Qwen/Qwen3.6-27B`, **2×A100 80GB TP2 BF16**, FlashInfer, O4096, thinking enabled, greedy, C1/C8/C32.
MTP-S3/S7/S15 = the built-in Qwen3.6 MTP heads at 3/7/15 steps (4/8/16 draft tokens, top-k 1) — **MTP-S3 is
the direct analog of our MTP3**. Output tok/s (speedup vs AR):

| Workload (C1) | AR | MTP-S3 | MTP-S7 | MTP-S15 | DFlash b8 | DFlash b16 | Domino b8 | Domino b16 |
|---|---|---|---|---|---|---|---|---|
| GSM8K | 47.2 | 126.8 (2.68x) | 151.9 (3.22x) | 133.9 (2.84x) | 179.2 (3.79x) | 200.9 (4.25x) | 206.0 (4.36x) | **248.0 (5.25x)** |
| MATH500 | 47.3 | 132.3 (2.80x) | 168.1 (3.55x) | 151.6 (3.20x) | 203.2 (4.29x) | 240.1 (5.07x) | 217.7 (4.60x) | **270.6 (5.72x)** |
| HumanEval | 47.2 | 125.3 (2.65x) | 149.9 (3.18x) | 129.9 (2.75x) | 188.0 (3.98x) | 211.1 (4.47x) | 198.5 (4.20x) | **235.1 (4.98x)** |
| MBPP | 47.6 | 122.6 (2.57x) | 141.8 (2.98x) | 117.0 (2.46x) | 177.5 (3.73x) | 186.2 (3.91x) | 189.0 (3.97x) | **214.0 (4.49x)** |
| MT-Bench | 47.1 | 115.1 (2.44x) | 125.0 (2.65x) | 100.7 (2.14x) | 141.5 (3.00x) | 143.8 (3.05x) | 155.7 (3.31x) | **161.9 (3.44x)** |
| Alpaca | 47.2 | 112.1 (2.38x) | 119.8 (2.54x) | 96.0 (2.03x) | 135.7 (2.87x) | 133.9 (2.84x) | 150.1 (3.18x) | **157.7 (3.34x)** |

C8 GSM8K: AR 324.4, MTP-S3 778.7 (2.40x), DFlash b8 1016.1 (3.13x), **Domino b8 1153.2 (3.55x)**;
C32 GSM8K: AR 862.9, MTP-S3 1555.9 (1.80x), MTP-S15 1034.2 (1.20x), **Domino b8 1817.5 (2.11x)**;
C32 MT-Bench: **MTP-S3 1352.0 (1.56x)** beats everything, MTP-S15 721.8 (**0.83x — slower than AR**).
Macro (mean over 6 workloads): C1 MTP-S3 2.59x / DFlash b8 3.61x / Domino b16 **4.54x**; C8 Domino b8
**3.22x**; C32 Domino b8 **1.84x**; overall MTP-S3 2.22x vs Domino 3.00x. Accept length (C1, GSM8K):
MTP-S3 3.556 (max 4), MTP-S7 5.503, MTP-S15 6.855, DFlash b8 5.449, DFlash b16 6.878, Domino b8 6.335,
Domino b16 **9.163**. Takeaways: (i) trained DFlash/Domino heads beat the native MTP head on every
workload at C1/C8; (ii) at high concurrency the gap mostly closes and MTP-S3 stays the best *MTP* config —
more MTP steps (S15) actively *hurts*; (iii) the draft backbone "is not shortened" for b8 — b8 only
verifies the first 8 of 16 positions.

**C. Qwen3.8-27B (our current target, dense hybrid) — DFlash2 card**
([z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), mirror of
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2); [inco blog](https://inco.ai/blog/dflash2/), 2026-08-18):
runtime **one NVIDIA H200**, SGLang, FlashAttention 3, block size 8 (7 draft tokens), Qwen3.8 default
sampling (T=1.0, top-p 0.95, top-k 20) with `xhigh` thinking, max 4096 tokens. MTP = "Qwen3.8's built-in
seven-token MTP"; DSpark = community RadixArk drafter; all speculative methods propose 7 draft tokens:

| Task (accept length) | MTP | DSpark | DFlash 2 |
|---|---|---|---|
| GSM8K | 5.02 | 4.36 | **5.46** |
| MATH-500 | 4.72 | 3.92 | **5.28** |
| HumanEval | 3.91 | 3.30 | **4.39** |
| MBPP | 3.99 | 3.51 | **4.79** |
| MT-Bench | 3.74 | 3.01 | **4.10** |
| Mean (inco Table 4) | 4.28 | 3.62 | **4.80** |

Output tok/s (speedup vs AR), C1: GSM8K AR 68.9 / MTP 178.5 (2.59x) / DSpark 185.3 (2.69x) /
**DFlash2 236.1 (3.43x)**; MT-Bench AR 68.9 / MTP 134.9 (1.96x) / **DFlash2 184.0 (2.67x)**.
C8: GSM8K AR 467.2, MTP 1022.1 (2.19x), **DFlash2 1328.7 (2.84x)**; MT-Bench AR 480.5, MTP 835.2 (1.74x),
**DFlash2 1090.2 (2.27x)**. C32: GSM8K AR 1329.8, MTP 1381.1 (1.04x), DSpark 1506.5 (1.13x), **DFlash2
1922.5 (1.45x)**; MT-Bench AR 1507.4, MTP 1159.7 (**0.77x**), DSpark 1115.5 (0.74x), DFlash2 1525.3
(1.01x). Inco's claim: "SGLang serves at 2.7–3.4× the throughput of autoregressive decoding at batch size
1" for Qwen3.8-27B; "close to 3× … about a third of the compute per token." DFlash2 = DFlash + lightweight
candidate-path selector (+2.0M params, +0.6% cycle latency: accept length 4.27→4.61 at T=0 on
Qwen3-4B/GSM8K) + two-tap dynamic convolution (+16.5M params / +3%: 5L+conv ≈ 15L, +0.7% latency);
"across benchmarks the gain runs 16–25%"; DFlash (v1) "now runs in SGLang, vLLM, TensorRT-LLM, and
llama.cpp"; "NVIDIA measured up to 15× throughput … on Blackwell"; "DFlash models have been downloaded
more than 3.5 million times (as of August 2026)"; a Qwen3.8-27B DFlash2 draft also runs on an Apple M5 Max
via oMLX (video in blog) — consumer-hardware evidence, but not a 4090.

**D. Qwen3.5-4B / Qwen3.6-35B-A3B — other head-to-heads.** Inco Table 3 (Qwen3.5-4B, per-request mean
accept length, thinking enabled, T=1.0, lossless rejection sampling): GSM8K MTP 4.78 / DFlash 4.99 /
DSpark 5.69 / **DFlash2 6.20**; means MTP 4.54 / DFlash 4.92 / DSpark 5.49 / **DFlash2 5.97**
("1.05 tokens over DFlash (21%) and 0.48 over DSpark"). [shreyansh26/qwen-spec-decode-benchmarking](https://github.com/shreyansh26/qwen-spec-decode-benchmarking)
(single GPU **NVIDIA B200**, C8, 8192-in/128-out, LongBench-v2, MIT): on `Qwen3.6-35B-A3B-FP8` (SGLang
0.5.16, which needs the repo's `sglang-0.5.16-qwen35-eagle3-capture.patch` for EAGLE3): **native MTP
(steps5_k1_d6) 752.046 tok/s, accept 3.715** — beats EAGLE3 (589.129 tok/s, accept **1.251**) and
DFlash V2 (328.426, 1.442); on `RedHatAI/Qwen3.6-35B-A3B-NVFP4` via vLLM: native MTP (tokens5)
1081.635 tok/s (accept 3.698) vs DSpark 1073.840 (3.318) vs target-only 1044.943 — "DSpark improves
output throughput by 2.8% … cuts median TTFT by 49.4%; native MTP is 0.7% faster than DSpark". (B200, not
sm_89; but it is the only public C8 head-to-head where **native MTP wins or ties** a hybrid model.)

**E. 4090 datapoints (community, our model family).** [pedroalonso.net](https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090)
(2026-06-09, updated 2026-08-24): **single RTX 4090 24GB**, llama.cpp `mtp-pr` branch `267f8afe8` (now
mainline `b1544`, `--spec-type draft-mtp`), `Qwen3.6-27B-MTP-Q4_K_M.gguf` (16 GB), `-c 16384`,
`--cache-type-k/v q8_0`, T=0.6/top-k 20/top-p 0.95, 5-turn agentic coding session (regenerate-the-whole-file
loop), ~18 GB VRAM used:

| Turn | MTP OFF | MTP ON | Speedup |
|---|---|---|---|
| 1 cold build | 47.7 | 115.7 | 2.4x |
| 2 add slider | 47.5 | 132.2 | 2.8x |
| 3 add button | 47.0 | 133.5 | 2.8x |
| 4 fix bug | 46.7 | 134.6 | 2.9x |
| 5 add trails | 46.3 | 133.3 | 2.9x |

  i.e. **~47 → ~133 tok/s, 2.4–2.9×** with `--spec-draft-n-max 4`; "MTP was still 2.4× faster writing the
  very first file from scratch"; ngram-mod crashed on that branch; "when acceptance drops — long context,
  novel content, thinking tokens — the overhead of guessing can make you slower than baseline".
  CMB's EAGLE3 head for **Qwen3-32B** (nearest dense size; [CMBTech/CMB-Qwen3-32B-Eagle3](https://huggingface.co/CMBTech/CMB-Qwen3-32B-Eagle3),
  2×H800, draft_tokens=3, T=0.6): accept length Coding 2.11, Math 2.60, EN-sum 1.95, Chinese-mixed 2.08,
  Chinese-finance 2.15, LCSTS 2.05.

**F. Agentic acceptance.** [PayPal/NVIDIA](https://arxiv.org/abs/2604.19767) (2026-03-27): EAGLE3 via vLLM
on PayPal's Commerce Agent (fine-tuned llama3.1-nemotron-nano-8B), 2×H100, 40 configs (γ=3/5,
concurrency 1–32, T=0/0.5): "γ=3 achieves 22–49% throughput improvement and 18–33% latency reduction";
"acceptance rates remain stable at approximately **35.5%** for γ=3 across all conditions"; "γ=5 yields
diminishing returns (approximately 25% acceptance rate)"; "speculative decoding on a single H100 matches or
exceeds NIM on two H100s, enabling 50% GPU cost reduction". [AgentSpec](https://arxiv.org/abs/2608.24004)
(2026-08-25): "two dominant factors of speedup degradation: high rejection rate of speculative tokens, and
under-utilization of dynamic token budgets" → structure-isolated drafting constrained to "semantically
coherent segments of the agent workflow"; evaluated in vLLM on five agent workloads / four model families.
[Performance or Illusion?](https://arxiv.org/abs/2601.11580) (MLSys'26, LMSYS/Berkeley): first systematic
SD study in production vLLM across n-gram/EAGLE/EAGLE-3/Draft-Model/**MTP**: "verification by the target
model dominates the execution, while acceptance length varies markedly across output token positions,
requests, and datasets" — the gap between measured and theoretical-bound speedups is the paper's main
finding. Also, the domino/dflash2 tables above show the agentic-relevant split: **code/math accept length
stays 1.3–1.7× open-chat's** (e.g. Qwen3.8-27B: GSM8K 5.46 vs MT-Bench 4.10 for DFlash2), and MT-Bench is
consistently the *lowest*-speedup row (z-lab C1: 2.67x vs 3.43x).

## Ideas we could port into ninfer-4090

All Apache/MIT-licensed; ranked by relevance to our dense-hybrid + MTP3 + 24 GB situation.

1. **Re-baseline MTP3 vs a trained draft head on OUR model, not on paper numbers.** The public head-to-heads
   split by model: on Qwen3.6/3.8-27B DFlash2 and Domino beat the in-checkpoint MTP (C1: 3.43x/4.54x vs
   2.59x/2.59x; accept 4.80 vs 4.28), while on Qwen3.6-35B-A3B (B200, C8) native MTP wins or ties
   (752 vs 589 EAGLE3 vs 328 DFlash tok/s; vLLM NVFP4: MTP 1081.6 vs DSpark 1073.8 tok/s). MTP's
   structural edge on hybrids: its head sits *after all* layers (sees GDN outputs); an EAGLE-3 head can
   only tap full-attention layers (thoughtworks: 3-of-48 aux layers on Coder-Next). The z-lab
   `Qwen3.8-27B-DFlash2-GGUF` + llama.cpp PR 27342 route lets us run a DFlash2 drafter beside NInfer on a
   4090 this week (see below). If the next Qwen's MTP head is as strong as Qwen3.8's, MTP may already be
   near-optimal for us; if it's weaker, a DFlash2-style drafter is a ready-made upgrade.
2. **DFlash2's design, in two cheap pieces** (inco blog / z-lab card): (a) **candidate-path selector** —
   keep top-16 candidates per position, score adjacent pairs with 256-dim token embeddings + context gate
   (a low-rank bilinear term), greedy/sampled path walk; +2.0M params, +0.6% cycle latency, +0.34 tokens
   accept at T=0. In NInfer terms: our verify pass already computes the target distribution over all
   drafted tokens, so a selector over the draft's own top-k lists is a tiny pre-verify filter that changes
   which draft path the target verifies — worth prototyping on our MTP drafts too (MTP-S15's 0.83x@C32 on
   the domino card is exactly "verifying a path the target rejects"). (b) **two-tap dynamic convolution**
   in the drafter (+3% params, +0.7% latency, fixes "suffix decay" — recall@16 falls 99.5%→87.8% across
   the 8-token block; 5L+conv ≈ 15L). If we ever train our own drafter, both are nearly free.
3. **EAGLE-3 training-time test (TTT) + multi-layer feature fusion** — the actual recipe (arXiv
   2503.01840 §3.2; SpecForge implements it): train the draft head on simulated multi-step drafting with
   diagonal attention masks; fuse low/mid/high target features through an FC layer; direct token loss (no
   feature regression loss). The TTT idea is what makes acceptance survive multi-step drafting (EAGLE-3's
   n-α "remains almost unchanged" vs EAGLE's collapse, Fig. 7) — the same failure mode our MTP chain would
   show if we extended it beyond its trained depth. SpecForge (MIT) is the turnkey trainer: it already has
   `qwen3.6-27b-dspark/dflash2-disaggregated` recipe YAMLs and an `examples/configs` tree, and its
   v0.3 runtime separates target-capture (SGLang servers → Mooncake) from draft training — we could reuse
   the trainer to produce a drafter for the next Qwen from target-generated agentic data ("regenerating
   dataset responses with the target model … has been the single largest lever on final acceptance", v0.3
   blog), which is also the right training distribution for our agentic workload.
4. **Dynamic draft tree (EAGLE-2)** — approximate acceptance from draft-model confidence, prune the tree
   per step (EAGLE-2: "4x faster than vanilla decoding (13B), 1.4x faster than EAGLE-1"). Maps to
   variable `num_draft_tokens`/top-k per verify round in NInfer instead of a static MTP3 tree; the
   SGLang dossier already tracks SGLang's XQA verify backend and windowed draft-decode attention for the
   engine side.
5. **Tree shape is a concurrency knob, not a constant** — thoughtworks B=32 results: wide tree
   (topk=4/steps=3/tokens=8) gives 1.31x on MT-Bench but **regresses Terminal-Bench to 0.89x** (extra tree
   tokens trigger redundant compute under load); narrow tree (topk=1/steps=5/tokens=6) removes the
   regression. For our 1–8 concurrency decode box the right default is the narrow-tree end; encode
   (topk, steps, draft-tokens) as a per-batch-size config.
6. **MTP step count is not free** — domino card: MTP-S15 beats MTP-S3 on GSM8K at C1 (2.84x vs 2.68x) but
   is *slower than AR* at C32 on MT-Bench (0.83x); C8: S15 2.08–2.43x vs S3 2.32–2.55x. Our MTP3 (3
   steps) is on the robust side of the curve; extending to S7/S15 would cost us at moderate concurrency —
   keep 3, spend the verify budget on better drafts instead.
7. **Draft head footprint** — EAGLE3 heads are ~0.2–0.9B params / 278 MB–2 GB (thoughtworks: 278 MB for
   an 80B target, "well under 1% of model memory"); the 27B DFlash2/Domino heads are ~1–2B. On a 24 GB
   box where our 27B target + 256K-class KV already crowds VRAM, any drafter must stay ≤~2B — all of
   these do; Medusa-style K-heads (≈K×d² params each) would too, but Medusa is dormant and its published
   acceptance is strictly worse than EAGLE-3/DFlash2, so there is nothing to port from Medusa itself
   beyond "heads on last hidden state + tree attention", which EAGLE/DFlash2 subsume.

## Worth running beside NInfer/llama.cpp?

**Yes — but the interesting comparison is drafter-vs-drafter on our exact checkpoint, and the cleanest
4090 test is via llama.cpp, not via EAGLE-3 itself.**

- **Test to run (low effort, high information):** llama.cpp on one 4090, same `Qwen3.8-27B` GGUF
  (groupwise-int or FP8 — not NVFP4, per the SGLang dossier's 4090 quant analysis), four arms:
  (1) AR baseline, (2) `--spec-type draft-mtp` with the shipped MTP sidecar GGUF (our MTP3's public
  analog), (3) `--spec-type draft-dflash` (llama.cpp PR 27342) with `incoai/Qwen3.8-27B-DFlash2-GGUF`
  (`-hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`), (4) NInfer itself (MTP3). Workloads: the
  pedroalonso-style 5-turn regenerate-the-file agentic session **plus** HumanEval/MT-Bench single-turn
  (code vs open chat split is where accept rates diverge most), C1 and C8. Report output tok/s and
  acceptance length; the z-lab card is the H200 reference to check the *ratios* against (DFlash2/MTP
  ≈ 1.33× at C1, ≈1.30× at C8), not the absolute values.
- **EAGLE-3 on SGLang/vLLM for Qwen3.8-27B: skip as a 4090 test** — no EAGLE3 checkpoint exists for our
  exact model (only community Qwen3-32B/Qwen3-Coder-Next ones), and SGLang-on-sm_89 is patched-community
  territory for hybrids (sglang.md). Its role is as the *training recipe* (idea 3) and as the
  engine-side mechanism MTP already runs through (SGLang's EAGLE/NEXTN path).
- **Medusa: no.** Dormant, A100-era numbers, strictly dominated by EAGLE-3 (paper Table 1: Medusa 2.07x
  vs EAGLE-3 5.58x on Vicuna-13B/MT-bench) and DFlash2. Its TRT-LLM/TGI integrations are the only reason
  it still appears in engine menus.
- What this tells us about NInfer: whether MTP3 is the right default drafter for the next dense Qwen, or
  whether a DFlash2-class drafter (or a drafter we train with SpecForge on agentic data) is worth adding
  a draft-model kernel path to the engine.

## Risk / unknowns

- **No 4090 numbers from any of the three projects.** Every published table here is A100/H100/H200/B200;
  the only 4090 data is one community llama.cpp MTP post (Qwen3.6-27B, 16K context, June 2026) plus issue
  reports (EAGLE on 4090 works, SpecForge 8×4090 training is rough). On 4090 the verify pass is
  compute-bandwidth-bound the same way, so C1 ratios should roughly transfer (pedroalonso's 2.4–2.9× MTP
  4090 result vs the domino card's 2.59–3.22× MTP A100 is consistent), but the 24 GB KV budget at 256K
  may force KV quantization that changes accept dynamics — untested anywhere.
- **The head-to-head data is model-specific and splits.** DFlash2/Domino > MTP on Qwen3.6-27B (A100) and
  Qwen3.8-27B (H200); MTP ≥ DFlash/DSpark/EAGLE3 on Qwen3.6-35B-A3B (B200, C8); EAGLE3 on GDN-hybrids
  (Qwen3-Coder-Next) delivers the weakest speedups of a six-model portfolio (1.37x vs 1.66–1.70x). Whether
  the next dense Qwen's MTP head is strong or weak decides whether MTP3 suffices — the answer is not in
  the public data.
- **Agentic acceptance is measured on proxies.** No public study reports EAGLE-3/Medusa/DFlash accept
  length specifically on *tool-calling agentic trajectories* (PayPal's is a commerce agent but
  EAGLE3-on-nemotron-8B; AgentSpec is vLLM/batch-oriented; the domino/dflash2 suites use MT-Bench as the
  chat proxy). The CMB numbers (accept 1.95–2.60 on mixed/finance/summarization, T=0.6) and thoughtworks'
  "code beats chat" inversion are the nearest agentic-flavored data. A drafter trained on
  ShareGPT/PerfectBlend (what every SpecBundle model was) is likely *miscalibrated for tool-call JSON and
  thinking tokens* — the v0.3 blog's "target-generated data is the single largest lever" claim points the
  fix: train/refresh the drafter on the target's own agentic rollouts.
- **Draft-tree verification on GDN state is the unsolved engine piece** (cross-ref sglang.md): verifying
  4–16 drafted tokens over a hybrid KV needs the ReplaySSM-class mechanism (mamba state replay, 32× state
  memory reduction) or an equivalent; the SGLang Qwen3.8-27B cookbook uses
  `--mamba-scheduler-strategy extra_buffer` / `--enable-linear-replayssm-spec`. Whatever drafter we pick,
  the verify pass is where the 4090 work happens.
- **EAGLE-3's aux-layer constraint is architectural, not incidental** (GDN recurrent states can't
  decompose to per-token features). If the next Qwen keeps the ~1:3 GDN:attention ratio, any EAGLE-3 head
  we train gets its features from a minority of layers — expect the Coder-Next-class 1.3–1.4x, not the
  paper's 4–6x. DFlash/MTP-style draffers (which read the final hidden state) do not have this problem.
- **Churn:** EAGLE repo last pushed 2026-02 (v3.0.0; "official EAGLE-3 for Qwen-3" still an open todo);
  Medusa dead since 2024-06; SpecForge is the only actively developed piece (daily pushes, v0.3.0
  2026-08-04, capture patch re-pinned every SGLang release — e.g. PR #886 for v0.5.20). Anything we port
  should come from SpecForge + the checkpoint authors, not from the EAGLE research repo.
- **Quantized-drafter effects unmeasured:** the 4090 test will use Q4_K_M GGUF drafts; the public cards
  are BF16/FP8. Accept-length loss from INT4 draft quantization is not published for any of these
  (llama.cpp's own docs note draft quantization as an open knob; pedroalonso notes IQ4_KS MTP failed to
  load on the MTP branch).
- **NVIDIA/RedHat/Modal also ship DFlash drafters** (nvidia/Kimi-K2.6-DFlash, RedHatAI gemma-4-31B-it
  speculator.dflash, modal-labs/Kimi-K3-DFlash per inco blog) and vLLM's speculators pipeline
  (RedHatAI/Qwen3-32B-speculator.eagle3, "1.5 to 2.5x speedup", [RedHat](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding)) —
  more drafters are landing for the Qwen3-32B class than for Qwen3.8-27B; a next-Qwen drafter may simply
  be downloadable rather than trained.

## Sources

- EAGLE repo: <https://github.com/SafeAILab/EAGLE> (README, LICENSE, weights tables); issues
  [#89](https://github.com/SafeAILab/EAGLE/issues/89), [#196](https://github.com/SafeAILab/EAGLE/issues/196),
  [#234](https://github.com/SafeAILab/EAGLE/issues/234).
- EAGLE-3 paper: <https://arxiv.org/abs/2503.01840> (v3 HTML); EAGLE-2 <https://arxiv.org/abs/2406.16858>;
  EAGLE-1 <https://arxiv.org/abs/2401.15077>; NeurIPS'25 proceedings PDF
  <https://proceedings.neurips.cc/paper_files/paper/2025/file/c7b5a35ea98b62512a869c19ea7b03cb-Paper-Conference.pdf>.
- SpecForge repo: <https://github.com/sgl-project/SpecForge> (README); docs <https://docs.sglang.io/SpecForge/>;
  PR [#886](https://github.com/sgl-project/SpecForge/pull/886); issues [#194](https://github.com/sgl-project/SpecForge/issues/194),
  [#380](https://github.com/sgl-project/SpecForge/issues/380); blog
  <https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/>,
  <https://www.lmsys.org/blog/2025-07-25-spec-forge/>,
  <https://www.lmsys.org/blog/2025-12-23-spec-bundle-phase-1/>; paper arXiv
  <https://arxiv.org/abs/2603.18567>.
- Medusa: <https://github.com/FasterDecoding/Medusa>; <https://arxiv.org/abs/2401.10774>;
  <https://sites.google.com/view/medusa-llm>.
- SpecBundle collection: <https://huggingface.co/collections/lmsys/specbundle>.
- Model cards: <https://huggingface.co/Huang2020/qwen3.6-27B-domino> (domino paper arXiv
  <https://arxiv.org/abs/2605.29707>, repo <https://github.com/jianuo-huang/Domino>);
  <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2> and <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>
  (mirror of <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2>); <https://huggingface.co/CMBTech/CMB-Qwen3-32B-Eagle3>;
  <https://huggingface.co/lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan>;
  <https://huggingface.co/thoughtworks/Qwen3-Coder-Next-Eagle3>; <https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>;
  DFlash repo <https://github.com/z-lab/dflash> (arXiv <https://arxiv.org/abs/2602.06036>), DSpark
  <https://arxiv.org/abs/2607.05147>, P-EAGLE <https://arxiv.org/abs/2602.01469>, LK loss
  <https://arxiv.org/abs/2602.23881>, D-PACE <https://arxiv.org/abs/2605.18810>.
- Blogs: <https://inco.ai/blog/dflash2/> (2026-08-18);
  <https://huggingface.co/blog/lujangusface/tw-eagle3-qwen3-coder-next> (2026-04-15);
  <https://www.pedroalonso.net/blog/qwen-mtp-speculative-decoding-4090/> (2026-06-09, upd. 2026-08-24);
  <https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding>.
- Papers: <https://arxiv.org/abs/2604.19767> (PayPal EAGLE3), <https://arxiv.org/abs/2608.24004>
  (AgentSpec), <https://arxiv.org/abs/2601.11580> (SD: Performance or Illusion?, MLSys'26).
- Community benchmark repo: <https://github.com/shreyansh26/qwen-spec-decode-benchmarking> (B200, C8).
- Cross-referenced internal dossiers: `sglang.md` (MTP-via-EAGLE/NEXTN, ReplaySSM, sm_89 status),
  `vllm.md` (eagle3/dflash methods, qwen3_next_mtp), `ktransformers.md` (RadixArk DSpark checkpoints,
  256K-on-4090 runs).
