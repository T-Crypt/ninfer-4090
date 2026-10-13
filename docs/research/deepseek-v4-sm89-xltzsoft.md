# Dossier: xltzsoft/deepseek-v4-sm89 (SGLang on 8×RTX 4090 for DeepSeek-V4-Flash MXFP4)

Researched 2026-10-13 (public sources only: the GitHub repo — README, commit history, kernel
source, benchmark JSON, branches; SGLang repo/issue #29747; SGLang DSV4 cookbook; Hugging Face
model card; arXiv 2606.19348; wtdcode/vllm-backport README; theainews.cc 2026-08-07 article).
No installs, no weight downloads, no GPU jobs.

This is the only public 8×RTX 4090 (sm_89) SGLang run found: DeepSeek-V4-Flash-0731 MXFP4,
TP8, with a hand-rolled sm_89 kernel patch set. It is **not** a Qwen engine — DSV4-Flash is a
284B MoE with DeepSeek's own compressed-sparse-attention (CSA/HCA + SWA) stack, not Gated
DeltaNet. Its value to prep4qwen is (a) the concrete sm_89 gap list it documents for SGLang's
fast paths, and (b) transferable engineering patterns (pinned launch geometry, spec-decode
state lifecycle, AB/BA paired kernel tuning). Cross-referenced: `sglang.md` (upstream SGLang
sm_89/Qwen status), `vllm.md` (vLLM sm_89 fallbacks), `dflash-pard.md` (DSpark is the
parallel-draft family member here), `ninfer-4090-udpsendtofailed.md` (single-4090 analog).

## What it is / backing / license

- **What it is:** a **SGLang experimental branch** whose sole purpose is running
  `DeepSeek-V4-Flash-0731` **MXFP4** weights on a **single node of 8×NVIDIA RTX 4090 24 GB
  (Ada SM89)** with TP=8. GitHub description (verbatim): "面向 8×RTX 4090（SM89）的
  DeepSeek-V4-Flash-0731 MXFP4 SGLang 实验分支" — a SGLang branch for 8×RTX 4090 (SM89).
  The repo ships the inference framework source only — no model weights, JIT caches, benchmark
  request files, credentials, or server-specific config. The README is explicit that "当前版本
  是性能研究快照，不是生产版本" (the current version is a **performance research snapshot, not
  a production version**): the target-only path starts and completed perf tests, but
  8-card/multi-request forward+reverse end-to-end stress re-validation is still outstanding,
  and long-context composite configs are still limited by draft-model VRAM.
- **Backing:** individual/community — GitHub user **xltzsoft** ("charles xu", profile:
  "a student in THU working for AGI", Beijing; 56 public repos), with two earlier commits by
  account **X_Bee**. Not DeepSeek, not SGLang/LMSYS, not RadixArk. The author has submitted
  **zero** PRs to sgl-project/sglang (GitHub issue search: 0 results). One mirror fork
  exists (arm7star/deepseek-v4-sm89, created 2026-10-08, 0 stars).
- **License:** **Apache-2.0** — inherited from SGLang; "本分支修改同样按照该许可证发布"
  (branch modifications released under the same license). Code is therefore readable and,
  conceptually, portable; model weights are MIT (DeepSeek-V4) but **not distributed** by this
  repo and are irrelevant to us (we won't run DSV4).

## Repo, version, last release date

- Repo: <https://github.com/xltzsoft/deepseek-v4-sm89> — 8 stars, 1 fork, 0 open issues,
  topics `deepseek-v4 llm-inference mxfp4 rtx-4090 sglang sm89`; created **2026-08-02**.
- **No GitHub releases or tags.** The README overview table labels the tree
  `v0.2.0-experimental`; framework baseline `d8f0ff966` + 2026-08-11 tuning patches; upstream
  base **SGLang `131bd51b`** (an upstream commit of 2026-08-01).
- Commit history: 19 branch commits, 2026-06-18 → 2026-08-17 (two X_Bee commits predate the
  chosen SGLang base and sit at the bottom of the stack), last human commit
  `ecd6f7f25c` 2026-08-17 ("docs: DSpark C128 fix verified; update known limitations").
  Since then only weekly `bot/update-est-time-*` branches (last `bot/update-est-time-20261012`,
  matching the repo's 2026-10-12 push timestamp); `main` itself has been unchanged since
  2026-08-17.
- Patch stack (all commit subjects from `main`): `fc9fe15d` SM89 FP8 indexer fallback (X_Bee);
  `9ccbec06` disable SM89 DeepGEMM MHC prenorm fast paths (X_Bee); `c08f81fa` enable MXFP4
  Marlin on SM89; `2a5641dd` select Marlin for SM89 DSV4 MXFP4; `75869705` avoid SM89 indexer
  KV pool copies; `25bb129a` route SM89 sparse MLA through Triton; `ed0d7839` route SM89
  sparse prefill through Triton; `3eb2041c` add SM89 sparse prefill kernel; `552b6f81` +
  `3f9814fd` keep native TP heads (DSV4 / DSpark); `89f9c07f` load fused DSpark shared
  experts; `e20d2945` pin SM89 sparse decode launch config; `d8f0ff96` initialize DSV4
  compression state pools; `94a89e65` publish SM89 paired tuning and lifecycle fixes;
  `45f67255` clear unaccepted C128 draft states after DSpark verify; docs commits
  `31f90d7f`, `6e34c255`, `ecd6f7f2`.

## Runs on an RTX 4090 (sm_89)?

**Yes — that is the entire point of the branch.** Sources:

- README overview table: "GPU 架构: NVIDIA Ada SM89"; "已验证拓扑: 8×RTX 4090 24 GB，TP=8"
  (verified topology: 8×RTX 4090 24 GB, TP=8).
- Launch section sets `TORCH_CUDA_ARCH_LIST=8.9`, `CUDA_VISIBLE_DEVICES=0..7`, Python 3.10,
  CUDA 13 user-mode libs, ABI-matched PyTorch + SGLang kernel wheel, Triton + TVM-FFI JIT.
- The committed tuning artifact self-identifies: `"hardware": "NVIDIA GeForce RTX 4090",
  "compute_capability": "sm89"` ([kernel-tuning-20260811.json](https://github.com/xltzsoft/deepseek-v4-sm89/blob/main/docs/benchmarks/sm89/kernel-tuning-20260811.json)).
- The DSpark fix commit is validated "on 8xRTX4090 (SM89, TP8)"
  ([commit `45f67255`](https://github.com/xltzsoft/deepseek-v4-sm89/commit/45f6725506)).
- Context: upstream SGLang's own DSV4 cookbook supports `h100/h200/b200/b300/gb200/gb300/rtx6000(96GB)/rtx5090(32GB)/dgx-spark/mi300x/mi355x` — **no RTX 4090 cell exists upstream**
  ([SGLang DSV4 cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4)),
  and SGLang issue [#29747](https://github.com/sgl-project/sglang/issues/29747) (2026-06-30,
  another user, 8×4090 + `lmsysorg/sglang:dev-cu12`, `--attention-backend flashmla`,
  `--context-length 500000`) confirms stock SGLang failed for 4090 DSV4; the author closed it
  after getting DSV4-Flash-FP8 running, noting "there are not any support for run
  Deepseek-v4-flash on RTX 4090".

## Supports Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K?

- **Qwen GDN hybrid: no.** Every sm_89 patch in this branch targets DSV4-specific paths: the
  `dsv4` attention backend (sparse MLA / CSA "C4" topk-512 sparse + HCA "C128" compressed KV
  tiers + 128-token SWA window, page size hardcoded 256, MQA head_dim 512 = 448 nope + 64
  rope — see `python/sglang/srt/layers/attention/deepseek_v4_backend.py`), the DSpark
  speculator, and the MXFP4 MoE quant gate. Nothing in the patch stack touches SGLang's Qwen
  model files or GDN kernels, and no Qwen config is verified. The base SGLang `131bd51b` does
  carry the Qwen3.8-27B hybrid path (Triton GDN fallback on sm_89 — see `sglang.md`), so a
  Qwen build *could* be started from this tree, but this branch neither claims nor tests it.
- **MTP:** for **DSV4** yes — the backend asserts `topk in [0, 1]` ("MTP Topk > 1 not
  supported for DeepSeek V4"), and the 0731 checkpoint ships its **DSpark** drafter in-checkpoint
  (target and draft weights from the same checkpoint; `--speculative-algorithm DSPARK`, block
  size 6 per the SGLang cookbook). For **Qwen**, MTP would come from upstream SGLang
  (cross-ref `sglang.md`); unverified on this branch.
- **Vision: no.** DSV4-Flash is text-only; DSV4-Flash-Vision-Exp (305B) is a separate variant
  this branch never serves.
- **256K: no evidence.** The verified launch pins `--context-length 8192`,
  `--max-total-tokens 8192`, `--max-prefill-tokens 8192`; the README states long-context
  composite configs "仍受 draft 显存占用限制" (still limited by draft VRAM occupation — the
  DSpark draft adds ~2.3 GB per card). No 256K run is published anywhere in the repo.

## Model background (public)

- **DeepSeek-V4-Flash-0731** — official release of DSV4-Flash, "the same model structure as
  DeepSeek-V4-Flash-DSpark, i.e. it comes with a speculative decoding module attached"
  ([HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)); MIT license
  for repo and weights. Served via `--speculative-algorithm DSPARK` (SGLang) or
  `--speculative-config {"method":"dspark",...}` (vLLM).
- [arXiv 2606.19348](https://arxiv.org/abs/2606.19348) (DeepSeek-AI, submitted 2026-04-26,
  "preview"): V4-Pro 1.6T params / 49B activated; **V4-Flash 284B params / 13B activated**;
  both 1M-token context; hybrid attention = **Compressed Sparse Attention (CSA) + Heavily
  Compressed Attention (HCA)** (the code's C4/C128 tiers); Manifold-Constrained
  Hyper-Connections (mHC); Muon optimizer; at 1M context V4-Pro needs 27% of V3.2's
  single-token FLOPs and 10% of its KV cache.
- Quant formats: this branch runs **MXFP4 routed experts** + **FP8 E4M3 KV**
  (`--kv-cache-dtype fp8_e4m3`, `--moe-runner-backend marlin`); upstream cookbooks also use
  FP8/NVFP4/FP4 variants on datacenter GPUs.

## Published numbers on a 4090 (copied, no arithmetic of our own)

All from the branch README's "性能实测" table — model **DeepSeek-V4-Flash-0731 MXFP4**,
**8×RTX 4090 24 GB, TP=8, context 8192, max concurrent 8, decode full CUDA Graph,
custom all-reduce disabled**; benchmark = 256 input / 128 output tokens, greedy sampling,
`ignore_eos=true`, C8 requests fired simultaneously via barrier. The README warns these
"只代表这一组固定硬件、软件和请求配置，不是通用 SGLang 性能结论" (describe only this fixed
hardware/software/request configuration; not a general SGLang performance conclusion).

| scenario | metric | result (README) |
|---|---|---|
| C1 | decode throughput | **54.75 token/s** |
| C1 | end-to-end throughput | 50.36 token/s |
| C1 | TTFT | 0.222 s |
| C8 | scheduler output throughput | 382.8–383.8 token/s |
| C8 | decode-window aggregate throughput | 374.58 token/s |
| C8 | end-to-end aggregate throughput | **318.59 token/s** |
| C8 | average GPU utilization | ~92.7% |

The README's own scaling reference: ideal linear `54.75×8 = 438.00 token/s`; C8 scheduler ≈
87.5% of ideal, decode-window ≈ 85.5%, end-to-end ≈ 72.7%; decode window retains ≈ 97.7% of
scheduler output, end-to-end ≈ 83.1% of scheduler output (their ratios, copied from the
README/chart notes).

**DSpark (spec decode) on the same 8×4090 box:** single-stream decode **96–126 token/s**
vs target-only **~48–55 token/s** (commit `45f67255`, validated on 8×RTX4090 SM89 TP8 after
the C128 garbage-output fix; also README known-limitation #2). The draft model costs ~2.3 GB
extra VRAM per card, shrinking the long-context KV pool.

**Kernel-level paired tuning** (RTX 4090, sm89; direct kernel/launch-meta measurements,
explicitly *not* end-to-end model TPS; JSON completed 2026-08-10):

| component | comparison | result | decision |
|---|---|---|---|
| sparse-prefill | `BLOCK_K=16` → `64` (BLOCK_H=16, 8 warps, 2 stages) | ~1.33× | applied |
| DSpark schedule, seed 0 | `bc64-bcp256-w2` vs production launch meta | 0.991327×, 95% CI [0.989033, 0.993626], worst case 0.985673 | not applied |
| DSpark schedule, seed 100000 | `bc64-bcp256-w4` vs production | 0.988452×, 95% CI [0.983479, 0.993449], worst case 0.980533 | not applied |

Promotion policy (from the JSON): geomean ≥ 1.01 **and** 95% CI lower bound ≥ 1.0 **and**
worst case ≥ 0.98; candidates must pass JIT + exact parity first; AB/BA alternating pair
order; 4 warmup / 20 paired samples; timing window 10–30 ms. Both DSpark runs returned
`no_improvement` and emitted no winner artifact — "没有把'候选中最快'误当成'快于生产'"
(never mistook "fastest candidate" for "faster than production").

**Regression tests shipped:** 9 registered test files (sm89 backend selection, native head
geometry, sparse prefill, DSpark weight loading, schedule, compress-state init/lifecycle).
This release passed DSpark tuner CPU unit tests **18 passed**, remote lifecycle regression
**6 passed**, remote DSpark scheduler CPU regression **26 passed** — three separate commands,
not summed into one total.

**No other 4090 numbers in the repo.** (The sibling vLLM backport's published throughput is
on A6000, not 4090 — see Risks/new-target note below.)

## Ideas we could port into ninfer-4090

Ranked by relevance to a single-4090 dense hybrid Qwen:

1. **Spec-decode state lifecycle discipline** (highest value, bug-class warning). The branch's
   most substantive fix: DSpark verify wrote C128 compressor states for *all* gamma+1 draft
   positions but never reset the rejected ones, so polluted ring states corrupted later
   rounds and greedy output degenerated into repeating garbage — fixed by clearing
   `[seq_len+commit_len, seq_len+num_draft)` after accept/finalize, mirroring the EAGLE path
   (`eagle_worker_common.py`). Plus: sentinel-initialize offline compressed-state pools
   (`d8f0ff96`), clear C128 on request-slot re-allocation, and clear C4 attention/indexer
   state *before* physical SWA pages are returned, covering both the plain allocator and the
   HiCache rollback entry. In NInfer terms: audit that every rejected MTP step clears KV +
   GDN recurrent state for exactly the unaccepted positions (our MTP K≤15 path + paged KV +
   GDN state pool is the same hazard class), and that state cleanup happens before page
   return on every eviction path. Their state-lifecycle test suite is a decent model for our
   regression tests.
2. **Pinned launch geometry / no runtime autotune on 24 GB cards.** Sparse decode is pinned
   to a 32-token tile, 8 warps, 2 stages precisely to avoid runtime autotune doing transient
   VRAM allocations that OOM the box; sparse prefill geometry (`BLOCK_K=64`) was found by a
   paired sweep and then *frozen* as the SM89 config. Two concrete sm_89 data points for our
   kernel choices: wider K-tiles win on Ada (~1.33× for a Triton sparse-prefill kernel) and
   decode kernels should ship precomputed launch configs on a memory-constrained card.
3. **AB/BA paired kernel-tuning methodology with an explicit promotion policy** (parity + JIT
   first; geomean + CI lower bound + worst-case gates; winner artifacts; honest
   `no_improvement` outcomes). Worth adopting as a convention in our `bench/` harness: it
   prevents both overfitting one seed and shipping "fastest of candidates" that isn't
   actually better than production.
4. **Don't pad to kernel-favored shapes.** Under TP8 they keep each rank's native 8 query
   heads instead of padding to FlashMLA's 64-head layout — padded heads burn SM time on
   garbage. NInfer is single-GPU so head padding isn't our issue, but the "never pad for a
   kernel's convenience" principle applies to any GQA/sparse kernel we write.
5. **MXFP4 Marlin MoE on SM89** — a two-line arch-gate fix (`mxfp4.py` /
   `mxfp4_marlin_moe.py`: "requires SM89, SM90, or SM120"): Marlin's MXFP4 kernels already
   run on Ada, upstream just refused. Relevant to us **only** if the next Qwen ships MXFP4
   MoE weights (the 2.4T/Qwen4-preview family) — a dense 27–32B has no MoE path.
6. **sm_89 fast-path fallback pattern** (FP8 indexer, MHC prenorm DeepGEMM fast paths
   disabled on Ada; FlashMLA AOT → custom Triton kernel). DSV4-specific, but it's a clean
   map of which SGLang fast paths are SM90/SM120-only and what a Qwen-hybrid 4090 SGLang
   build would additionally need (GDN prefill/decode already fall to Triton upstream — see
   `sglang.md`).
7. **PCIe-box serving defaults:** decode full CUDA graph + prefill graph disabled +
   `--disable-custom-all-reduce` (consumer boards lack working P2P for custom all-reduce);
   `CUDA_MODULE_LOADING=LAZY`, single-arch build. NInfer is single-GPU (no collectives), so
   this is context, not a port.

## Worth running beside NInfer/llama.cpp?

**No, as a head-to-head:** it needs 8×4090, DSV4-Flash-0731 MXFP4 weights (not distributed
here; MIT-licensed elsewhere), and is a text-only 284B MoE — none of which match our
single-4090 dense hybrid Qwen workload. **Yes, as a reference artifact:** (a) its patch set
is the best available map of "what SGLang on 4090 is missing" for the *next* frontier model,
which predicts the same gap list for a Qwen-hybrid SGLang 4090 build; (b) the state-lifecycle
and paired-tuning patterns above are directly adoptable. A fair comparison, if we ever had
the hardware, would be same weights/quant, C1/C8, 256-in/128-out greedy (their shape),
scheduler/decode-window/end-to-end metrics reported with identical accounting — their README
explicitly demands this when comparing engines.

## Risk / unknowns

- **Self-reported numbers, single configuration, no independent replication.** One
  individual's snapshot; the README itself downgrades the C8 multi-request path as not fully
  stress-verified (forward/reverse end-to-end pending), and DSpark's residual per-token
  drift vs target is only "on par with target's own forward/reverse rerun noise floor"
  (the stack has no run-level determinism).
- **Staleness:** `main` frozen since 2026-08-17 on SGLang `131bd51b` (~2 months behind the
  current SGLang v0.5.21 line, which moved Unified Radix Cache to the only path and retired
  the CUDA-12 lane); only bot branches have landed since. The patch stack would need a
  re-merge to mean anything for a current-Qwen SGLang 4090 build.
- **Model mismatch:** every kernel here serves DSV4's CSA/HCA/SWA/MLA-512 design; none
  transfer as code to GDN hybrid attention. The MoE patches are irrelevant to a dense
  model. Nothing here validates GDN prefill/decode on sm_89 (that question lives in
  `sglang.md`/`lmdeploy-turbomind.md`/`tokenspeed.md`).
- **No 4090-runnable Qwen numbers anywhere in this repo** — and none upstream: the SGLang
  DSV4 and Qwen3.8-27B cookbooks both lack an RTX 4090 cell.
- **Ecosystem context suggests the 4090 frontier-model niche is active but fragmented** —
  which surfaced a **new target** during this research:
  **wtdcode/vllm-backport** ([repo](https://github.com/wtdcode/vllm-backport), Apache-2.0,
  302★/49 forks, created 2026-08-04, last push 2026-10-06, "Dogfooding vLLM backport for
  older GPUs") — a vLLM fork with single-arch sm80/sm86/sm89 images (sm89 = **RTX 4090,
  L40S**) covering DSV4-Flash-0731 (native FP4, "Fully Supported 0.6.0+"), **Qwen3.8-27B
  (BF16, AWQ W4A16, v0.8.0+)**, **Qwen3.8-Flash-Next** (FP8 + AWQ W4A16, v0.9.0+; PLE
  51B n-gram table offloaded to pinned host RAM; `--mamba-cache-mode=align` +
  `--prefix-match-unit=16` prefix-caching fix; AWQ build hits "up to 936 tps" on 4×A100),
  GLM-5.3-Flash, DSV4.1-Flash + Engram; DSpark on Ampere pinned to `num_speculative_tokens=5`
  (7 needs ~200 KB smem vs the 163 KB Ada limit); default determinism switches (stable MoE
  token sort, fixed decode split-k, token-bucket padding, fused DSpark draft-sampling
  chain, chunked indexer-logits prefill fixing the >134K-token crash); shipped as Docker
  images with a paired LMCache fork for KV serving. The theainews.cc 2026-08-07 writeup
  reports 8×A6000 (TP4/PP2) **prefill 3435 tok/s, decode 948 tok/s** on DSV4-Flash
  (A6000, not 4090; the current README no longer carries that table). This is the one
  public vLLM lane that already serves *our exact model family* on a 4090 — added to
  `targets.md` as a new row.

## Sources

- <https://github.com/xltzsoft/deepseek-v4-sm89> — README (overview, patch list, benchmark
  table, launch commands, known limitations, license), [kernel-tuning JSON](https://github.com/xltzsoft/deepseek-v4-sm89/blob/main/docs/benchmarks/sm89/kernel-tuning-20260811.json),
  [commit history](https://github.com/xltzsoft/deepseek-v4-sm89/commits/main),
  [`45f67255` DSpark C128 fix](https://github.com/xltzsoft/deepseek-v4-sm89/commit/45f6725506),
  [`c08f81fa` MXFP4 Marlin SM89 gate](https://github.com/xltzsoft/deepseek-v4-sm89/commit/c08f81fae8),
  [`3eb2041c` SM89 sparse prefill kernel](https://github.com/xltzsoft/deepseek-v4-sm89/commit/3eb2041c80),
  [sparse_prefill_bf16_sm89.py](https://github.com/xltzsoft/deepseek-v4-sm89/blob/main/python/sglang/kernels/ops/attention/dsv4/sparse_prefill_bf16_sm89.py),
  [deepseek_v4_backend.py](https://github.com/xltzsoft/deepseek-v4-sm89/blob/main/python/sglang/srt/layers/attention/deepseek_v4_backend.py),
  [branches](https://github.com/xltzsoft/deepseek-v4-sm89/branches),
  [user profile](https://github.com/xltzsoft), [fork](https://github.com/arm7star/deepseek-v4-sm89)
- <https://github.com/sgl-project/sglang/issues/29747> — 8×4090 DSV4-Flash on stock SGLang
  (community, 2026-06-30, closed)
- <https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4> — upstream DSV4
  cookbook (hardware list: no 4090 cell; DSpark block size 6; variant table)
- <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731> — model card (MIT; DSpark
  in-checkpoint; vLLM/SGLang serve flags)
- <https://arxiv.org/abs/2606.19348> — DeepSeek-V4 technical report (284B/13B-activated
  Flash; CSA/HCA; mHC; 1M context efficiency)
- <https://github.com/wtdcode/vllm-backport> — README (model table incl. Qwen3.8-27B /
  Qwen3.8-Flash-Next on sm89, DSpark-on-Ampere notes, determinism switches), repo metadata
- <https://theainews.cc/articles/2026-08-07-vllm-backport-ampere/> — secondary writeup with
  the 8×A6000 3435/948 tok/s figures (secondary; the pinned table is no longer in the
  upstream README)
