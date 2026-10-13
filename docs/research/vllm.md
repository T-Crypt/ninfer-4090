# Dossier: vLLM

Researched 2026-10-06 (public sources only: repo, docs, release notes, PRs/issues, vLLM blog, model
recipes, arXiv papers, and one community single-card recipe). No installs, no weight downloads, no
GPU jobs.

## What it is / backing / license

- **What it is:** a high-throughput, memory-efficient LLM inference and serving engine. Core
  contribution is **PagedAttention** — attention KV cache managed like OS virtual memory/paging to
  get near-zero KV waste and flexible KV sharing within/across requests. On top: continuous
  batching, chunked prefill, prefix caching, piecewise and full CUDA/HIP graphs, and a large kernel
  zoo. README lists attention backends **FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, Triton**;
  GEMM/MoE via **CUTLASS, TRTLLM-GEN, CuTeDSL**; spec decode **n-gram, suffix, EAGLE, DFlash**
  (+ MTP); torch.compile auto-kernel-gen; disaggregated prefill/decode/encode.
  ([README](https://github.com/vllm-project/vllm))
- **Backing:** originally developed in the **Sky Computing Lab at UC Berkeley** (the PagedAttention
  paper is a SOSP 2023 Berkeley submission); now "built and maintained by a diverse community of
  dozens of academic institutions and companies from over 2000 contributors" (README). ~93.3k GitHub
  stars, 23.1k forks, 8.5k issues.
- **License:** **Apache 2.0** (repo license field + README badge). **Porting code is license-OK.**

## Repo, version, last release date

- Repo: <https://github.com/vllm-project/vllm>; docs <https://docs.vllm.ai>; blog <https://blog.vllm.ai>;
  recipes <https://recipes.vllm.ai>.
- **Latest stable: v0.31.0, released 05 Oct 2026** (717 commits / 307 contributors); `main` is at
  `v0.31.1rc0`. Prior stable **v0.30.0, released 22 Sep 2026** (762 commits / 315 contributors).
  ([releases](https://github.com/vllm-project/vllm/releases))
- Cadence: ~biweekly (README/docs; community single-card recipes pin e.g. `vllm==0.27.1`,
  vLLM 0.26.0 on PyPI, v0.24.0 referenced in mid-2026 articles) — fast churn, breaking changes every
  release (see Risks).
- Install: `uv pip install vllm`; prebuilt CUDA **12.9** default (also 12.8/13.0), nightly wheels at
  `wheels.vllm.ai`. GPU floor: **compute capability 7.5+** (docs list T4, RTX20xx, A100, L4, H100,
  B200) ([GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)).

## Runs on an RTX 4090 (sm_89)?

**Partly — the core stack and the GDN Triton/FLA fallback run on sm_89; the fast GDN path and the
newest fast kernels are Blackwell-only; and there are open 4090 correctness/OOM issues.**

- GPU requirement "compute capability 7.5 or higher" means **sm_89 is nominally supported**
  ([GPU install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)).
- **But the fast GDN path is Blackwell-only.** vLLM's Qwen3.5 perf work is explicitly Blackwell:
  "On supported Blackwell configurations, vLLM automatically selects the FlashInfer path when the
  GDN backend is set to `auto`" (FlashInfer Blackwell GDN prefill, flashinfer PR #3001, vLLM PR
  #40717) ([Qwen3.5 blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)). v0.30.0/v0.31.0 note
  **"FlashInfer GDN prefill on SM12x (#55715)"** and **"fused GDN MTP decode for SM110 (#53835)"**
  and **"CuteDSL BF16 GDN prefill … on Qwen3.5 (#53864)"** — all Blackwell. **On a 4090 vLLM runs
  GDN through the baseline Triton/Flash-Linear-Attention path only** (see GDN section).
- Open 4090 issues (real, in-tree): issue **[#42049](https://github.com/vllm-project/vllm/issues/42049)**
  "vLLM 0.20.1 hard-pins torch 2.11.0, which OOMs during CUDA initialization on RTX 4090 / cu130"
  (open, 2026-05-08). A community fork **yhfgyyf/vllm-deepseek-v4-sm89** exists to run DeepSeek-V4 on
  "SM89 (Ada / RTX 4090) … Validated on 4x RTX 4090" (patch over PR #41834) — i.e. stock vLLM needed
  a patch for a current model on 4×4090.
- No RTX 4090 appears in any vLLM-published benchmark table (perf work is H100/H200/B200/GB200;
  closest consumer-card recipe is RTX 5090/Pro 6000, see numbers below).

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, day-0-grade.** vLLM blog **"vLLM Now Supports Qwen3-Next: Hybrid Architecture
  with Extreme Efficiency" (2025-09-11)**: "Qwen3-Next introduces a hybrid architecture … vLLM offers
  full support of its functionalities." It **integrates Triton kernels from Flash Linear Attention
  (fla-org/flash-linear-attention)** and adopts the **hybrid KV cache manager** (arXiv 2503.18292,
  "Jenga"); "vLLM automatically tunes the 'logical' block size of the full attention layers to ensure
  that the state for the full attention layers and linear attention layers occupy the same amount of
  'physical' GPU memory" for simple paged memory on hybrid models; **full CUDA graph mode by default**
  to hide Triton CPU overhead on decode. 65K context at launch; MoE 1:50 activation (80B-A3B = 3B
  active/token). ([blog](https://vllm-project.github.io/2025/09/11/qwen3-next.html))
- The GDN kernel module `vllm.model_executor.layers.mamba.gdn` ships
  `qwen_gdn_linear_attn` — documented as **"Inference-only Qwen3-Next/Qwen3.5 model"** (plus
  `kimi_gdn_linear_attn`, `olmo_gdn_linear_attn`). So the Qwen GDN linear-attention path is the
  registered one for the Qwen3-Next/Qwen3.5 family. ([API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/gdn))
- vLLM tracks the same Qwen3.8 family as our `qwen3.8-27b` identity: **Qwen3.8-Flash-Next (Qwen4Exp)**
  is in-tree — v0.30.0 adds "separate prefill and decode QSA indexer kernels (#54513), fused PLE
  kernels (#54517), FP8 indexer cache (#54890)"; v0.31.0 adds "FP8 main KV cache on the QSA path
  (#55557), SM90 QSA tuning (#57273)". That model is the **125B/6B-active MoE** (GDN + Qwen Sparse
  Attention, 51B N-gram embedding, MTP, 262,144-token native context), recipe requires **vLLM 0.29.0+**
  and runs on H200/H100/GB200/GB300/MI355X/RTX Pro 6000 4× — **no RTX 4090 recipe** (it is 172.78 GiB
  FP8, won't fit 24 GB). ([recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next),
  [releases](https://github.com/vllm-project/vllm/releases))
- **MTP: yes, first-class and Qwen-specific.** `--speculative-config '{"method":"qwen3_next_mtp",
  "num_speculative_tokens":2}'` (dense Qwen3-Next recipe) and `{"method":"mtp","num_speculative_tokens":3}`
  (Qwen3.8-Flash-Next + the 5090 single-card run). MTP is a named spec-decode method; "fused GDN MTP
  decode for SM110 (#53835)" is the fast verify kernel (Blackwell).
- **Vision: yes** (README: "Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)"; Qwen3.5 is served with
  `--language-model-only` to drop the multimodal tower; Qwen VL processors in-tree). Not separately
  validated for the next dense Qwen.
- **256K: yes, on paper and in one recipe.** Qwen3.8-Flash-Next is native 262,144-token (recipe:
  "a single 262K-token request was not tested"); the 5090 single-card Qwen3.8-27B recipe pins
  `--max-model-len 262144` with a 5.5 GiB KV pool (see numbers). **No public sm_89 256K run found.**

## Published numbers on a 4090 or similar

**No verified published vLLM-on-RTX-4090 (sm_89) tok/s figure found.** The closest vLLM-on-4090
article (markaicode, "vLLM + Qwen3-8B on RTX 4090", 2026-06) **retracted** its latency/throughput
numbers as unverifiable ("We could not verify those numbers … Rather than publish another set of
invented figures, this revision … gives you the exact … commands to run the benchmark yourself"); its
sibling "vLLM vs Ollama on RTX 4090" URL resolves to an **Ollama** Qwen3.6 article, not vLLM. So for
vLLM specifically on a 4090: **not found.**

Closest published numbers (all **not** sm_89 — treat as reference for the target model family, not as
4090 vLLM results):

- **Qwen3.8-27B on a single consumer card (sm_120) via vLLM** — the one run of *our exact target
  model family* on a single consumer GPU. MiaAI-Lab recipe, **RTX 5090 (32 GB, sm_120)**, `vllm==0.27.1`
  + `flashinfer-python>=0.6.13` + `nvidia-cutlass-dsl>=4.5.2`, model `RadixArk/Qwen3.8-27B-NVFP4`,
  `--max-model-len 262144`, `--kv-cache-dtype turboquant_4bit_nc` (5.5 GiB = 286,466 tokens),
  `--max-num-seqs 1`, `--max-num-batched-tokens 512`,
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`:
  - **~160 tok/s** single-stream generation (MTP-3), batch size 1.
  - **Full 262,144-token context** resident (KV: 4-bit TurboQuant, 5.5 GiB).
  - **0/15** garble-battery fails **with a backport of vLLM PR #40914** (stock 0.27.1 garbles **13/15**).
  Note: NVFP4 and this exact recipe are **Blackwell (sm_120) only — they do not transfer to sm_89**;
  the MTP path and the #40914 MTP×low-bit-KV bug are the transferable lessons.
  ([repo](https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090),
  [issue #40880](https://github.com/vllm-project/vllm/issues/40880), [PR #40914](https://github.com/vllm-project/vllm/pull/40914))
- **vLLM Qwen3.5-397B-A17B-NVFP4, 8×B200 / GB200 NVL72, disaggregated** (datacenter, not consumer):
  FlashInfer Blackwell GDN prefill kernel is **1.02×–5.78×** over the prior FLA/Triton GDN across
  Qwen3.5 sizes/TP/seq-len/batch; on the 8×B200 integration **up to 5.92×** GDN kernel perf,
  **1.13×** end-to-end prefill throughput (ISL/OSL 8192/1), **12% lower mean TTFT** (prefill-only
  8K/1); final system **25,000 total TPS/GPU** (concurrency swept 64→5120, single 8×GB200 decode
  endpoint, `--async-scheduling`). ([blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps))
- **PagedAttention paper** (arXiv 2309.06180, SOSP 2023): "vLLM improves the throughput of popular
  LLMs by **2–4×** with the same level of latency compared to … FasterTransformer and Orca" (A100-era
  eval, no 4090). ([arXiv](https://arxiv.org/abs/2309.06180))
- **Jenga** hybrid-KV-cache allocator (arXiv 2503.18292, "hybrid KV cache manager" in the Qwen3-Next
  blog): "improves GPU memory utilization by **up to 79.6%**, and increases serving throughput by
  **up to 4.92× (1.80× on average)**" (diverse LLMs/GPUs, no 4090-specific table). ([arXiv](https://arxiv.org/abs/2503.18292))
- **HMA+NixlConnector** (vLLM PR #35758) for hybrid P/D: maps logical blocks to the right physical
  regions, "reducing transferred descriptors from 4,284 to 1,650 and improving throughput by up to
  approximately 7% in a small-scale intra-node H100 setup." ([blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps))

## Ideas we could port into ninfer-4090

All Apache-2.0. "Gain as they measured it" is as published; the Blackwell-only kernel gains do **not**
apply to sm_89, so the portable value here is mostly design/policy, not kernel speed.

1. **Jenga hybrid KV-cache manager / LCM block sizing** (arXiv 2503.18292, vLLM `hybrid_kv_cache`
   manager, `--no-disable-hybrid-kv-cache-manager` now default). Two-level allocator keyed on the
   **LCM of the per-layer state sizes** so linear (GDN) and full-attention states share one paged
   region. This is exactly NInfer's paged-KV design question for the hybrid model: size the full-
   attention logical block so it equals the GDN recurrent-state block, and page them as one pool.
   They measured **up to 4.92× / 1.80× avg throughput, +79.6% mem util** (not 4090-specific, but the
   mechanism is what makes long-context hybrid serving fit a 24 GB card at all).
2. **Mamba/GDN state checkpointing for prefix caching** (`--enable-mamba-shared-prefix-checkpoint`,
   renamed from `--enable-mamba-fine-grained-prefix-cache`, #57382; **stateless GDN first chunks
   #51565**; **batched Mamba2 prefill state saves without GPU-CPU syncs #49371**; "prompt-end
   checkpoints kept under sparse retention #59146"). The policy: periodically snapshot the recurrent
   GDN state so a prefix-cache hit on a hybrid model is safe. NInfer's context cache needs the same
   snapshot discipline for GDN layers, or prefix reuse must be disabled on GDN state.
3. **The MTP × low-bit-KV verify fix (PR #40914 / issue #40880).** Stock vLLM captures the MTP
   verify step as a context-free first-chunk full-attention CUDA graph that never reads the KV cache,
   so MTP over 4-bit KV degenerates into repetition/broken tool calls (13/15 garble). The fix routes
   **uniform K+1 spec-verify batches through the decode kernel with all-GPU `synth_seq_lens`** (no
   CPU-side per-request branching, so graph capture/replay stay valid) → 0/15, ~160 tok/s. This is the
   exact bug class NInfer will hit with **MTP3 + low-bit KV on a hybrid**; the synthetic-args decode
   routing is a concrete pattern to steal.
4. **FLA/Triton GDN reference implementation** (the sm_89-capable baseline): `vllm/model_executor/
   layers/mamba/gdn/qwen_gdn_linear_attn.py`; vLLM warms "Qwen3.5/Qwen3-Next Triton kernels (#54797)
   and GDN gated RMSNorm (#54251)" at startup, and keeps "the recurrent state in FP32 for long
   prefills" (FlashKDA #58846). A second independent GDN to check NInfer's replayssm path against
   numerically on sm_89 (both are Triton/FLA-derived, so the comparison is like-for-like at the arch).
5. **Full CUDA graph by default for Triton-heavy hybrids** (Qwen3-Next blog: "vLLM enables full CUDA
   graph mode by default" to hide Triton launch CPU overhead on decode). NInfer already uses graphs;
   the specific hazard vLLM hit is the **breakable/piecewise graph** interaction with the GDN custom
   op — `--compilation_config.cudagraph_mode=PIECEWISE` is the documented escape when DP-mode IMA
   appears. Worth keeping in the NInfer graph policy for a hybrid model.
6. **Async scheduling for hybrid P/D** (PRs #48481, #45357 fix the KV-block-transfer races that made
   async scheduling collapse accuracy to zero on hybrid attn models). Relevant only if NInfer ever
   moves off its "one compact decode batch per round" model; noted as the known failure mode.

## Worth running beside NInfer/llama.cpp?

**Yes, but as a reference/second-implementation and idea source, not as the fastest 4090 engine.**
On a 4090 vLLM's GDN runs the **Triton/FLA fallback** (the FlashInfer fast path is Blackwell-only),
vLLM publishes no 4090 tuning, it's a heavy Python/CUDA stack, and there are open 4090 OOM/correctness
issues (#42049) plus a documented MTP×4-bit-KV garble bug that needs a backport (#40914). Its value to
us is: (a) a second independent GDN + hybrid-KV-cache implementation to validate NInfer numerics and
the LCM paging policy against, and (b) the concrete MTP-verify and prefix-checkpoint fixes above.
If we do run it, a fair test: **same weights/quant as NInfer** (e.g. our groupwise-int or an FP8
checkpoint — **not** the NVFP4 5090 recipe, which is sm_120-only), single RTX 4090, batch 1–8, 32K–
256K context, compare prefill tok/s, decode tok/s, TTFT/TPOT, and prefix-cache reuse against NInfer
and llama.cpp on the same hybrid Qwen checkpoint. Expect vLLM to show its strength at **long-context
hybrid memory efficiency** (the whole point of GDN + the LCM paged cache), not necessarily raw 4090
decode speed.

## Risk / unknowns

- **No sm_89 performance data** — vLLM's investment is visibly datacenter (Blackwell-only GDN/MoE/
  MLA fast paths, GB200/B200/H200 benches); Ada is maintained (compute-7.5 floor) but not a frontier
  target. On a 4090 everything GDN-related is the Triton/FLA fallback.
- **Open 4090 correctness/OOM issues** (#42049 torch-2.11/cu130 CUDA-init OOM on 4090; the sm89
  community fork exists to patch stock vLLM for current models on 4×4090). A vLLM-on-4090 test would
  need version pinning and possibly the #40914 MTP fix.
- **The MTP + hybrid + low-bit-KV combination is fragile in vLLM too** — #40880/#40914 (13/15 garble
  on stock 0.27.1) is exactly the MTP3-on-hybrid path the next dense Qwen will use; NInfer must not
  assume a naive full-graph verify step is correct over low-bit KV.
- **Fast churn / breaking changes:** breaking changes in both v0.30.0 and v0.31.0 (e.g.
  `--enable-mamba-fine-grained-prefix-cache` renamed to `--enable-mamba-shared-prefix-checkpoint`;
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` removed; `quantization="fp8"` → `fp8_per_tensor`); biweekly
  releases mean a pinned-version test can be stale quickly.
- The 27–32B **dense** hybrid is not a headline vLLM recipe (the flagship recipes are the 80B-A3B
  MoE and the 125B/6B Qwen3.8-Flash-Next); the 27B dense path is proven by one community 5090
  recipe, so a 4090 dense-hybrid vLLM run is plausible but unvalidated in any published source.
- The GDN fast kernels (FlashInfer #3001, CuteDSL #53864) are Blackwell-only, so any kernel we borrow
  must be re-derived for sm_89 — we can port the *policy* (LCM paging, state checkpointing, MTP-verify
  routing) but not the *kernel speedup*.

## Sources

- https://github.com/vllm-project/vllm (README: features, license, Sky Computing Lab origin, model list)
- https://github.com/vllm-project/vllm/releases (v0.31.0 2026-10-05, v0.30.0 2026-09-22, v0.31.1rc0)
- https://github.com/vllm-project/vllm/releases/tag/v0.31.0 (GDN/SM12x/SM110 kernels, QSA, mamba prefix cache, kernels)
- https://github.com/vllm-project/vllm/releases/tag/v0.30.0 (Qwen3.8-Flash-Next perf, FlashInfer GDN prefill, Jenga-era kernels)
- https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html (compute-7.5 GPU floor, CUDA 12.9)
- https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/gdn (qwen_gdn_linear_attn = Qwen3-Next/Qwen3.5)
- https://vllm-project.github.io/2025/09/11/qwen3-next.html (day-0 Qwen3-Next: FLA Triton kernels, hybrid KV cache manager, full CUDA graph, MTP)
- https://vllm.ai/blog/2026-08-06-qwen35-25k-tps (Qwen3.5 GDN Blackwell kernels, 25K TPS/GPU on GB200, HMA/NIXL, async scheduling)
- https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next (Qwen4Exp 125B/6B MoE, GDN+QSA, 262K, MTP, vLLM 0.29.0+, no 4090)
- https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html (dense recipe: 4×H200/A100, qwen3_next_mtp, prefix caching, PIECEWISE)
- https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090 (single RTX 5090 run: ~160 tok/s MTP-3, 262K, #40914 fix)
- https://github.com/vllm-project/vllm/issues/40880 (MTP×TurboQuant garble) ; https://github.com/vllm-project/vllm/pull/40914 (fix)
- https://github.com/vllm-project/vllm/issues/42049 (4090/cu130 torch-2.11 OOM)
- https://github.com/yhfgyyf/vllm-deepseek-v4-sm89 (community sm_89 fork, 4×4090)
- https://arxiv.org/abs/2309.06180 (PagedAttention, SOSP 2023, 2–4× vs FasterTransformer/Orca)
- https://arxiv.org/abs/2503.18292 (Jenga hybrid KV cache, +79.6% mem util, up to 4.92×/1.80× throughput)
- https://markaicode.com/benchmarks/vllm-qwen-3-rtx-4090-latency-benchmark/ (vLLM-4090 article; author retracted its numbers)
- https://markaicode.com/benchmarks/cuda-qwen-3-rtx-4090-latency-benchmark/ (resolves to Ollama Qwen3.6, not vLLM)
