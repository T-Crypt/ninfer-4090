# Dossier: TokenSpeed (LightSeek)

Researched 2026-10-08 (public sources only: the repo, lightseek.org blog + docs, PyTorch
ecosystem blog, NVIDIA developer blog, HuggingFace Qwen model cards/configs, and the project's
issues/PRs). No installs, no weight downloads, no GPU jobs.

## What it is / backing / license

- **What it is:** a **Python-based LLM inference engine** aimed at **agentic workloads**: "speed-of-light
  LLM inference engine designed for agentic workloads, with TensorRT-LLM-level performance and vLLM-level
  usability" ([README](https://github.com/lightseekorg/tokenspeed)). Three pieces: a **C++ scheduler
  control plane** (request lifecycle, KV-cache ownership and overlap timing as a finite-state machine,
  safe KV reuse "enforced by the type system at compile time") with a **Python execution plane**; a
  **pluggable kernel registry** (public op API, `select_kernel` on tensor-format signature + arch
  capability + traits; solutions are in-tree Triton/Gluon/CuteDSL JIT or vendor wrappers for
  FlashInfer/FlashAttention/TRT-LLM — "no in-tree C++ build" for the vendor layer, per
  [tokenspeed-kernel README](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/README.md));
  and an **SMG-integrated AsyncLLM** entrypoint (OpenAI-compatible + SGLang-compatible HTTP surface).
  Day-0s it has shipped: TML Inkling (2026-07), Kimi K3 (2026-07), **Qwen3.8 2.4T (2026-08)**,
  **Qwen3.8 Flash Next + GLM 5.3 Flash (2026-08)** ([README news](https://github.com/lightseekorg/tokenspeed)).
- **Backing:** **LightSeek Foundation** (lightseek.org), a **501(c)(3) nonprofit** — the project calls
  itself "the Switzerland of open-source LLM inference engines: neutral because it's backed by the
  non-profit LightSeek Foundation rather than a commercial company, with serious hardware and engineering
  support from vendors like NVIDIA and AMD" ([README](https://github.com/lightseekorg/tokenspeed)). Launch
  blog co-creator list: NVIDIA DevTech, AMD (Triton), **Qwen Inference**, Together AI, Mooncake, LongCat,
  FluentLLM ("parts of the TokenSpeed runtime reflect upstream lineage from FluentLLM"), EvalScope; compute
  from OpenAI, NVIDIA, AMD, Verda, Nebius ([launch blog](https://lightseek.org/blog/lightseek-tokenspeed.html)).
  Joined the **PyTorch Ecosystem** (Q3 landscape update, 2026-08); the PyTorch ecosystem post calls it
  "the first [engine] to separate the control plane from the execution plane".
- **License:** **MIT** (repo license field; every file SPDX-licensed MIT). **Porting/reading code is
  license-free** — this is the one engine in the queue with both an official Qwen3.8 recipe and an
  unencumbered portable GDN kernel stack.

## Repo, version, last release date

- Repo: <https://github.com/lightseekorg/tokenspeed> (≈2.2k stars, 310 forks, 65 issues);
  docs <https://lightseek.org/tokenspeed/>; blog <https://lightseek.org/blog>.
- **v0.1.0, 2026-07-24; v0.1.1, 2026-10-06** (latest; [releases](https://github.com/lightseekorg/tokenspeed/releases)).
  Daily **nightlies** (`.postYYYYMMDD`) on the `lightseek.org/whl/nightly` index for `tokenspeed` and
  `tokenspeed-kernel`; ROCm 7.2 nightly stream in parallel ([getting started](https://lightseek.org/tokenspeed/guides/getting-started)).
- History: development began **mid-March 2026**, announced **2026-05** with a B200 performance preview
  ("the engine and kernels remain under active development"). Cadence is very fast (≈2,000 PRs in ~5
  months; 65 open issues; issue creation restricted to collaborators — external PRs go through a
  selective, "Triton-like lean core team" process per the
  [ethos issue #149](https://github.com/lightseekorg/tokenspeed/issues/149)).
- CI hardware: workflows for **nvidia-b200 / nvidia-gb200 / nvidia-gb300 (+nightly) / AMD**; a
  **Hopper / CUDA 12.9 source-install recipe** for H100/H200 and a Kimi-K3 Hopper PD guide are the only
  sub-Blackwell paths. **There is no CI on Ada / RTX consumer GPUs** (`.github/workflows/` listing).

## Runs on an RTX 4090 (sm_89)?

**Partly, on paper — and nowhere validated.** The *portable* halves of the stack are arch-open; the
fast halves are Blackwell/Hopper-only; Ada is an untested platform with no recipe, no CI, and no public
issue activity.

- **GDN (the 48 linear layers) — yes via the Triton/FLA fallback.** All four GDN ops register a portable
  Triton solution with **vendor-only, arch-unrestricted** capability
  (`CapabilityRequirement(vendors={"nvidia","amd"})`): `triton_gdn_chunk_prefill`, `triton_gdn_decode_step`,
  `triton_gdn_decode_mtp`, `triton_gdn_replay_commit`
  ([gdn/triton.py](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/triton.py)).
  The FlashInfer GDN fast paths are gated above that: prefill
  `min_arch_version=ArchVersion(10,0), max_arch_version=ArchVersion(10,3)` + "CUDA>=13 + bf16 + head_dim==128 …
  **this module has no Triton fallback for it**", decode/MTP `min_arch_version=ArchVersion(9,0)`
  ([gdn/flashinfer.py](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/flashinfer.py)).
  On a 4090 the prefill selection simply lands on the Triton/FLA solution.
- **Full attention (the 16 GQA layers) — the published 27B recipe won't run there.** The recipe pins
  `--attention-backend trtllm`, and that leaf is documented "**TRT-LLM fused kernels for SM100
  (Blackwell)**" ([paged/trtllm.py](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/backends/paged/trtllm.py)).
  A 4090 run would have to use the default `mha` backend (auto kernel-solution selection) or explicitly
  `triton`/`flashinfer` (fa2); `fa3`=Hopper, `fa4`=Blackwell, `trtllm_mla`/`tokenspeed_mla` are
  MLA/Blackwell paths ([server parameters](https://lightseek.org/tokenspeed/configuration/server),
  [paged/mha.py](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/backends/paged/mha.py)).
  The KDA tiering shows the intended pre-Hopper pattern: `cutedsl_kda` (Blackwell) / `flashkda` (Hopper+)
  / `fla` ("else") ([registry.py](https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/registry.py)).
- **Not found:** any sm_89/Ada/RTX-4090 support statement, CI, recipe, issue or user report. Repo issue
  search for "4090" and "sm_89" returns **zero** issues. The closest hardware discussion is
  [issue #502](https://github.com/lightseekorg/tokenspeed/issues/502) ("What models are supported under
  SM90/Hopper/H20?"), where Qwen3.5-35B-A3B failed on an H20 with "GDN prefill requires the flashinfer
  Blackwell fast-path (sm100/sm103 + CUDA 13 + …)" — i.e. in June 2026 even Hopper GDN prefill was
  Blackwell-fast-path-only, and the portable path has since been the Triton/FLA one.
- **Bottom line:** the engine *can* be pointed at a 4090 (JIT kernels, no in-tree C++ arch wall), and a
  Qwen3.8-27B hybrid run would work through the Triton/FLA + triton/fa2-attention fallback — but **no one
  has done it publicly**, and every published configuration, number and CI job in the project targets
  B200/GB200/GB300/Hopper/AMD. Treat "runs on a 4090" as unverified until someone boots it.

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes — native, day-0 for Qwen3.8.** Qwen3.8-27B's `config.json` declares
  `architectures: ["Qwen3_5ForConditionalGeneration"]` / `model_type: qwen3_5`
  ([config](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json)) — **exactly the arch string in
  TokenSpeed's `_HYBRID_GDN_ARCHITECTURES` set** (Qwen3_5/Qwen3_5Moe variants + NextN), so the 27B is
  served by the same `HybridLinearAttnBackend` as Qwen3.5/3.6: 64 layers, `full_attention_interval: 4`
  → 48 GDN (`linear_num_value_heads: 48`, `linear_num_key_heads: 16`, head_dim 128, conv kernel 4, SSM
  state fp32) + 16 gated-attention (24 Q / 4 KV heads, head_dim 256) layers. Qwen3.8-**2.4T-A95B** (92
  layers, 23×(3 GDN + 1 GatedAttn), 512-expert MoE, 4 KV heads) got an official day-0: FP8 2-node
  TP8×DP2×EP16 or 4-node TP4×DP4×EP16 on (G)B200/(G)B300, NVFP4 on one Blackwell node, FlatKV + DSpark
  ([day-0 blog](https://lightseek.org/blog/tokenspeed-qwen3-8.html)).
- **MTP: yes — first-class, and the 27B recipe is self-speculative MTP.** The published Qwen3.8-27B recipe
  runs `--speculative-algorithm MTP --speculative-draft-model-path Qwen/Qwen3.8-27B-FP8
  --speculative-num-steps 3` on `--world-size 1` ([recipes](https://lightseek.org/tokenspeed/recipes/models#qwen3-8-27b));
  the 27B config has `mtp_num_hidden_layers: 1`. GDN MTP-verify runs through `gdn_decode_mtp` (portable
  Triton or FlashInfer SM90+); for the 2.4T MoE they also ship DSpark with **one CUDA graph per decode
  step**, draft-KV materialization folded into the verify stream, and **vocab-shard DSpark sampling**
  (Markov-bigram bias computed inside the sharded argmax, "bit-exact with full-vocabulary argmax",
  two 2·bs-scalar all-gathers per step) ([day-0 blog](https://lightseek.org/blog/tokenspeed-qwen3-8.html)).
  EAGLE3/DFlash/DFlash2 drafters are supported too (recipes for Kimi K2.5/K3, MiniMax M3, GLM).
- **Vision: supported for the qwen3_5 family in-tree; nothing published for the 27B specifically.** The
  27B is a VLM (qwen3_5 vision tower, images and video). TokenSpeed's VLM path is exercised on
  Qwen3.5: PR [#2035](https://github.com/lightseekorg/tokenspeed/pull/2035) fixes vision-encoder OOMs
  for "Qwen3.5-35B-A3B's vision MLP", PR [#715](https://github.com/lightseekorg/tokenspeed/pull/715)
  restores "Qwen3.5 OCRBench accuracy", PR [#1940](https://github.com/lightseekorg/tokenspeed/pull/1940)
  runs multimodal stress on "Qwen3.5-35B-A3B-FP8 TP1 … 128/128" on an 80 GB GPU; the K3 recipe has
  `--mm-encoder-tp-mode` for a MoonViT tower. **Not found:** a Qwen3.8-27B multimodal recipe or any VLM
  benchmark — the 27B recipe is a text launch.
- **256K: yes natively for the 27B; the recipe requests it.** Config: `max_position_embeddings: 262144`;
  model card: "262,144 natively and extensible up to 1,000,000 tokens"; the recipe passes
  `--max-model-len 262144`. TokenSpeed's cache design handles the hybrid at long context: **FlatKV**
  represents each GDN layer's fixed state (one SSM + one conv tensor) as a single logical cache block in
  the same arena as KV pages, with per-group packing factors derived from byte-size ratios ("under
  Qwen3.5's validated TP4 FP8 geometry, one GDN state page occupies roughly the same space as 32
  full-attention KV pages") and demand-driven slab binding, so capacity shifts between KV and state as the
  workload changes ([day-0 blog](https://lightseek.org/blog/tokenspeed-qwen3-8.html)). 256K on a 24 GB
  4090 is a memory question the project never addresses (all its long-context numbers are Blackwell,
  below).

## Published numbers on a 4090 or similar

**Not found on any consumer GPU — every number below is datacenter Blackwell/Hopper.** No RTX 4090 /
sm_89 figure exists anywhere in the project (blog, docs, issues).

**A. PyTorch ecosystem blog, 2026-05-28 — Qwen3.5-397B-A17B(-NVFP4) on NVIDIA B200, EvalScope
benchmarks, TokenSpeed latest Docker** ([blog](https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/)):
- Agentic suite (50K first-turn context, +800 tokens/turn, 10–15 turns): **all four configs
  (TP4 / TP4EP4 / TP8 / TP8EP8) sustain 500+ tok/s at bs=1; TP8 peaks at ~580 tok/s/user**;
  at concurrent=16 the TP4 family reaches ~2K tok/min/GPU and the TP8 family ~1K; multi-turn
  **KV cache hit rate > 90%**.
- **MTP (self-spec) vs off: +100%~+159% throughput at bs=1; +38%~+90% at bs=32/64 for
  output > 4096; "near-zero or slightly negative" at bs=64 with 1024-token outputs** (speculation
  overhead outweighs acceptance once decoding is throughput-bound).
- NIAH long context, TP8: **~530 tok/s/user within 128K, ~495 at 256K, ~445 at 1M — "~16%"
  end-to-end degradation from 128K to 1M**.

**B. Launch blog, 2026-05 — Kimi K2.5 on NVIDIA B200 vs TensorRT-LLM (SWE-smith agentic traces,
TPS/User vs TPM/GPU Pareto, no PD disaggregation)** ([blog](https://lightseek.org/blog/lightseek-tokenspeed.html)):
  attention TP4 + MoE TP4 dominates the TRT-LLM frontier: "**roughly 9% faster in the min-latency case
  (batch size 1), and roughly 11% higher throughput around 100 TPS/User**"; TokenSpeed MLA prefill beats
  TRT-LLM MLA "across all five typical prefill workloads for coding agents"; decode "nearly halves
  latency relative to TensorRT-LLM on typical decode workloads with speculative decoding (batch sizes 4,
  8, and 16 with long prefix KV cache)".

**C. Qwen3.8 day-0 blog, 2026-08 — Qwen3.8-2.4T-A95B on (G)B200/(G)B300 nodes**
([blog](https://lightseek.org/blog/tokenspeed-qwen3-8.html)): at fixed 1024/8192 io, batch 32–256,
**TP8×DP2×EP16 sustains 1.28–1.36× the throughput of TP16 (1.31× average)**; KV-head replication
(4 KV heads, `tp >= num_kv_heads` replicates pages) makes **TP8 give "roughly 2× the effective
KV-cache capacity and concurrent batch size"** of TP16.

**D. NVIDIA blog, 2026-08-26 — Qwen3.8-Flash-Next FP8 on GB300 NVL72**
([blog](https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/)):
"**over 16K tokens per second per GPU and over 200 tokens per second per user**" (TensorRT-LLM curve;
TokenSpeed publishes its own recipe for the same model). Also, from Alibaba's benchmarks cited there:
QSA attention "up to 7.6x prefill and 4.9x decoding speedups over full attention … 8.6x the prefill
throughput of Qwen3.7-Plus at 1M-token context with a 90% prefix-cache hit rate".

**E. Kernel-level, in-repo (GB200/GB300/H20 — not 4090):** PR [#2066](https://github.com/lightseekorg/tokenspeed/pull/2066)
  (GDN gated-norm, 4 rows/program, GB200, "Qwen3.8 rows of 128"): 528 rows 1.87→1.74 µs, 8448 rows
  5.36→2.83 µs, 67200 rows 36.4→13.0 µs, 393216 rows 208→68 µs, bit-identical outputs. PR
  [#2068](https://github.com/lightseekorg/tokenspeed/pull/2068) (Lamport A2A, 2× GB300 NVLink, BF16):
  e.g. 128 tokens/rank: 11.28 µs packet vs 20.21 µs FlashInfer + quant. V4.1 recipe H20 MoE-layer
  numbers: CUTLASS W4A16 vs Marlin 502 vs 656 µs @96 tokens, 4517 vs 7999 µs @8192-token chunk
  ([recipes](https://lightseek.org/tokenspeed/recipes/models)).

**Not found:** any TokenSpeed number on an RTX 4090 / RTX 5090 / RTX PRO or any other sm_89/sm_120
consumer card (web + repo search; the only 4090-on-a-hybrid-Qwen data point in the whole queue is the
community SGLang run tracked under `xltzsoft/deepseek-v4-sm89`).

## Ideas we could port into ninfer-4090

All MIT. Ranked for a single-GPU sm_89 dense GDN hybrid; NInfer already owns its GDN math, so these are
mostly *design and state-management* ports, not kernel drops.

1. **FlatKV hybrid cache geometry** (day-0 blog §"Unified FlatKV for Mamba States"): one arena;
   byte-blind logical blocks (allocation/hashing/prefix-match at granularity P, independent of tensor
   shape/dtype/offsets); physical "LCM slabs" where each slab's packing factor = byte-size ratio of the
   group's block to the slab; slabs bind to a cache group on first use and release when empty, so KV and
   GDN-state capacity shift with the workload instead of a static partition. GDN prefix = **longest
   P-aligned boundary shared by every cache group**, with copy-on-write of the state page on a hit (no
   copy of the cached state). This is the cleanest published answer to exactly NInfer's problem — paging
   *both* a growing KV cache and a fixed-size GDN state in one memory pool with one prefix-cache key —
   and it maps 1:1 onto our context-resource design (the `32 KV pages ≈ 1 GDN state page` packing
   insight is directly reusable for sizing state-page vs KV-page ratios in our 27B geometry).
2. **MambaSlot prefix-cache lifecycle** (580tps blog §2.1): every Mamba request holds a *working* slot
   plus an optional *checkpoint* slot; checkpoints are published only at aligned boundaries; a fresh
   working slot is safe by exactly two rules (CoW copy from a known-clean checkpoint, or explicit
   zero); under overlap scheduling the snapshot of a chunk's state happens on the execution stream and
   the next chunk's reuse is ordered after it; under decode, block-aligned states are snapshotted before
   dispatching the next decode. The invariant — "every slot reachable from the prefix tree contains a
   clean, aligned state for that prefix" — is a useful correctness contract for our own checkpoint/
   replica logic and a checklist of the overlap-scheduling hazards we have to hit.
3. **State indirection instead of state copies for MTP verify** (580tps blog §3.1): append a draft
   region of physical state rows after the scheduler's base slots; each speculative step writes its SSM
   state directly to its own row (`output_state_indices`), the kernel reads from the row recorded in
   `current_input_indices`; after acceptance, a single **O(1) integer write** re-points the canonical
   row — replacing an intermediate state cache plus a per-step `num_layers × state_dim` scatter kernel.
   Direct replacement for any per-draft-token GDN state handling in NInfer's MTP path; their
   `gdn_decode_mtp` Triton kernel shows the exact kernel-side contract (per-token output rows,
   intermediate buffer only at tree branch points, `-1` padding rows guarded).
4. **The portable GDN kernel set as an MIT reference** ([gdn/triton.py](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/triton.py)
   + [_triton/](https://github.com/lightseekorg/tokenspeed/tree/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/_triton)):
   chunked prefill at **chunk size 64**, causal-conv1d prefill/decode, fused QKV-split, gated RMSNorm,
   decode/MTP single-kernel with prefetch of the next step's operands, replay-commit. This is the
   strongest open GDN stack written for *non-Blackwell* silicon (it's what runs on Hopper/Ada); worth a
   line-by-line diff against NInfer's replayssm/chunked-prefill for the 27B's exact geometry
   (48 V / 16 QK heads, head_dim 128, conv 4), plus its numerics/reference harness
   (`python -m tokenspeed_kernel.numerics` compares any registered kernel against a PyTorch reference)
   as a qualification oracle.
5. **Gated-norm multi-row-per-program** (PR #2066): fold 4 128-wide rows into one program (one warp
   each, identical lane layout/reduction order → bit-identical). Their measured gain on GB200 is
   208→68 µs at 393,216 rows (a full 5M-token prefill's worth of GDN gating); the same shape applies
   to any per-token-per-head gating/norm kernel on the 4090.
6. **PDL (programmatic dependent launch) inside the GDN kernel chain** ([gdn README](https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/README.md)
   + `ENABLE_PDL` in the Triton kernels, `pdl_enabled()` in the runtime): `gdc_wait()`/
   `gdc_launch_dependents()` overlap consecutive GDN kernels' launch latency; PDL is an Ampere+ feature,
   so it works on sm_89. Cheap port for any fused kernel chain we have.
7. **One CUDA graph per decode step that contains verify + draft** (day-0 blog §DSpark): target verify,
   accept/bonus, draft-KV materialization (fc projection split column-wise per capture layer so the
   hidden-state slices are folded in *while the remaining target layers run*), draft forward and
   sampling all in one replay; scratch preallocated at `max_bs` so every captured graph has fixed
   addresses. The "split fc per capture layer" overlap is directly applicable to our MTP3 verify even
   without DSpark.
8. **Vocab-shard draft sampling** (day-0 blog): compute the Markov-bigram correction
   (`hₖ·W_shardᵀ + w1[prev]·w2_shardᵀ`) inside each rank's sharded argmax instead of materializing
   `[bs, V]` logits; the TP reduction is two `(max, id)` scalar all-gathers and is bit-exact with
   full-vocab argmax. Single-GPU NInfer doesn't need the sharding, but the "bias inside the argmax
   primitive" trick is a free decode-side win for any draft head we add.
9. **Single-kernel `fused_qk_rmsnorm_rope_gate`** (580tps blog §3.2.2): QK RMSNorm + partial RoPE +
   gate split in one Triton kernel, five HBM round-trips eliminated; the 27B's gated attention (attn
   output gate, partial_rotary 0.25) matches the shape.
10. **Kernel-registry architecture** (kernel README): `register_kernel(family, mode, signatures,
    capability, priority)` + `select_kernel` with format signatures and traits. NInfer is C++ with
    closed per-target implementations, so we wouldn't copy the machinery, but the *selection contract*
    (capability + dtype + traits + priority bands, fail-fast when no kernel matches) is a tidy model for
    how our GDN backends should declare support as we add 4090-specific kernels.

## Worth running beside NInfer/llama.cpp?

**As a 4090 head-to-head: no. As a reference: yes — it is the reference.**

- It is the **third engine Qwen officially points at** in the Qwen3.8 model cards ("dedicated serving
  engines such as SGLang, vLLM, or TokenSpeed are recommended", with a dedicated
  [Qwen3.8 recipe](https://lightseek.org/tokenspeed/recipes/models#qwen3-8)) and it ships the only
  *Qwen-authored-collaborator* engine (Qwen Inference is a named co-creator) — so its 27B recipe and
  2.4T day-0 are the best signal of what Qwen expects an engine to do for the next dense Qwen.
- But a 4090 run would sit entirely on the **fallback tier** (Triton/FLA GDN + triton/fa2 attention),
  the tier nobody publishes or CI-tests, with **zero public 4090 numbers** — a head-to-head against
  NInfer would measure *their untested fallback*, not their design. llama.cpp is the better 4090
  comparator; TokenSpeed's value to us is (a) **design** (FlatKV, MambaSlot, state indirection — the
  three items above are the closest published analogs of what NInfer has to build), and (b) **reference
  numerics** (their PyTorch reference impls + numerics harness give an independent GDN oracle for our
  kernels at the exact 27B geometry). If Friday's decision allows one extra install, the useful test is
  a *small-context correctness cross-check* of NInfer's GDN state math against TokenSpeed's
  (Triton-GDN) tokens on the same Qwen3.8-27B weights, not a throughput race.

## Risk / unknowns

- **Ada is an afterthought, not a target.** No CI, no recipe, no issue activity on sm_89; the whole
  project (kernels, recipes, blogs, sponsor hardware) centers on Blackwell. Expect the portable paths to
  *work* (Triton is portable) but to be *slow and un-tuned* — the Triton/FLA GDN fallback is the Hopper
  compromise, and the sm100 GDN prefill they call the "fast path" is the thing we'd have to beat or
  replace on the 4090 (see the FlashQLA dossier for the Qwen-side reference for what a tuned GDN
  prefill should reach).
- **The published 27B recipe is a Blackwell recipe**: `Qwen/Qwen3.8-27B-FP8` + `--attention-backend
  trtllm` (SM100 kernels) + `--moe-backend flashinfer_trtllm`. Reproducing it on a 4090 means rewriting
  the backend choices (triton/flashinfer attention, FP8-or-BF16 weights) and nothing in their docs
  guarantees that combination boots; the FP8-KV-on-Ada path of their triton/fa2 MHA solutions is not
  documented.
- **Young project, fast churn, gated contributions**: 5 months old, ~2k PRs, breaking changes between
  releases (v0.1.0→v0.1.1 in 2.5 months), external PRs selectively merged (ethos issue #149); nightly
  wheels move daily. Any reference test must pin a tag.
- **FluentLLM lineage**: "parts of the TokenSpeed runtime reflect upstream lineage from FluentLLM" —
  so "original" is partly a re-architecture of an existing engine; the genuinely new bits for us are the
  hybrid-cache designs (FlatKV/MambaSlot) and the GDN kernel set, not the scheduler.
- **The MTP gains they publish are Blackwell/TP8 numbers** (+100–159% at bs=1 on B200 for a 397B MoE):
  our 27B dense 4090 MTP economics (draft weight traffic, acceptance on agentic text) will differ; the
  bs=64 short-output *negative* result is a warning that self-speculative MTP does not uniformly pay.
- **Vision on the 27B is unproven in TokenSpeed** (qwen3_5 VLM path exists; no 27B VLM recipe found) —
  same as with SGLang/vLLM, nobody has published a single-GPU 27B-VL run anywhere.
- Not verified: whether the current `tokenspeed` wheel's vendored JIT stack (Triton fork, FlashInfer
  pin) builds cleanly on a stock 4090 box with CUDA 12.x/13.x — no Ada install instructions are
  published (only the Hopper cu129 source recipe and the CUDA-13 nightly default).

## Sources

- https://github.com/lightseekorg/tokenspeed — README (positioning, C++-FSM scheduler, kernel registry,
  "Switzerland" nonprofit framing, PyTorch ecosystem quote, news list, B200 Kimi K2.5 perf chart, MIT)
- https://github.com/lightseekorg/tokenspeed/releases — v0.1.0 (2026-07-24), v0.1.1 (2026-10-06)
- https://lightseek.org/blog/lightseek-tokenspeed.html — launch blog (2026-05): co-creators (incl. Qwen
  Inference, FluentLLM lineage), Kimi K2.5 B200 Pareto vs TensorRT-LLM (9% min-latency / 11% @100 TPS),
  TokenSpeed MLA prefill/decode wins, SWE-smith agentic methodology
- https://lightseek.org/blog/tokenspeed-qwen3-8.html — Qwen3.8-2.4T-A95B day-0 (2026-08): 92-layer
  hybrid + MoE arch, FP8 cross-node TP8×DP2×EP16 (1.28–1.36× vs TP16; ~2× KV capacity), FlatKV
  (logical blocks, LCM slabs, packing from byte ratios, "1 GDN state page ≈ 32 KV pages" at Qwen3.5 TP4
  FP8), hybrid prefix cache (longest P-aligned boundary, CoW), DSpark (one CUDA graph/step, KV
  materialization overlap, vocab-shard sampling)
- https://lightseek.org/tokenspeed/recipes/models — model recipes: **Qwen3.8-27B** (single-GPU FP8,
  `--attention-backend trtllm`, `--max-model-len 262144`, `--kv-cache-dtype fp8_e4m3`, self-speculative
  MTP 3 steps), Qwen3.8-2.4T-A95B (2-node TP8×DP2×EP16 / TP4×DP4×EP16), Qwen3.8 Flash Next
  (GDN+QSA hybrid, PLE, 262144 native/1M YaRN, B200/B300 QSA CuTe-DSL), Kimi K3 Hopper/Blackwell,
  DeepSeek V4.x (H20 MoE kernel µs table), Nemotron-3 (Mamba2 + MTP)
- https://lightseek.org/tokenspeed/configuration/server — server parameters: attention backend list
  (`mha, fa3, fa4, triton, flashinfer, trtllm_mla, tokenspeed_mla`), MoE/sampling backends, MTP
  chain/tree semantics, DSpark/DFlash conventions, FlatKV L3 (Mooncake), `--numerics` contracts
- https://lightseek.org/tokenspeed/guides/getting-started — install: CUDA-13 nightly wheels, Hopper
  cu129 source recipe, RDNA4 notes; no Ada/RTX path
- https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/README.md — kernel layer design
  (register/select, in-tree JIT solutions, vendor wrappers "no in-tree C++ build", numerics harness,
  JIT-compile discipline)
- https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/flashinfer.py
  — GDN FlashInfer gates: prefill sm100–sm103 + CUDA≥13 + bf16 + head_dim 128 + "no Triton fallback
  for it"; decode/MTP SM90+; K-last state layout convention
- https://github.com/lightseekorg/tokenspeed/blob/main/tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/triton.py
  — portable Triton GDN stack: `triton_gdn_chunk_prefill` / `triton_gdn_decode_step` /
  `triton_gdn_decode_mtp` / `triton_gdn_replay_commit`, all
  `CapabilityRequirement(vendors={"nvidia","amd"})` (no arch floor → sm_89-eligible), chunk size 64,
  per-step operand prefetch, tree-verify state mechanism
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/registry.py
  — `_HYBRID_GDN_ARCHITECTURES` incl. `Qwen3_5ForConditionalGeneration(+NextN)`; hybrid backend
  selection; KDA tier `cutedsl_kda` / `flashkda` (Hopper+) / `fla`
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/backends/hybrid/linear.py
  — `HybridLinearAttnBackend`: per-layer-id routing between full-attention and GDN children, shared
  CUDA-graph/plumbing
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/backends/paged/trtllm.py
  — `trtllm` MHA leaf: "TRT-LLM fused kernels for SM100 (Blackwell)"
- https://github.com/lightseekorg/tokenspeed/blob/main/python/tokenspeed/runtime/layers/attention/backends/paged/mha.py
  — `mha` backend solution mapping (`mha/fa3/fa4/triton/flashinfer`)
- https://github.com/lightseekorg/tokenspeed/issues/502 — H20 (SM90) Qwen3.5-35B-A3B GDN prefill
  failure: "requires the flashinfer Blackwell fast-path (sm100/sm103 + CUDA 13 + …)"; Hopper sm90 GDN
  support volunteered; closed 2026-06-24
- https://github.com/lightseekorg/tokenspeed/issues/149 — project ethos: lean core team, selective
  external contributions
- https://github.com/lightseekorg/tokenspeed/issues/965 — "Model Support: Top Open-Source Models & Day 0
  Collaboration" (top-5 partners, early access = Day 0)
- https://github.com/lightseekorg/tokenspeed/issues/589 — RFC "route GDN prefill through
  tokenspeed-kernel registry" (closed not-planned 2026-07; documents the Triton-FLA-portable vs
  FlashInfer-Blackwell split and the Qwen3.5 head-dim geometry)
- https://github.com/lightseekorg/tokenspeed/pull/2066 — GDN gated-norm multi-row-per-program (GB200
  µs numbers, Qwen3.8 row geometry, bit-identical)
- https://github.com/lightseekorg/tokenspeed/pull/2035, /pull/1940, /pull/715 — qwen3_5 VLM support
  evidence (vision encoder OOM cap, multimodal memory stress on Qwen3.5-35B-A3B-FP8 TP1, OCRBench CI
  fixes)
- https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/
  — 580 tps B200 benchmark post: MambaSlot/CoW prefix cache design, dual-pool scheduler, PD state
  transfer, state-indirection MTP verify, fusion catalog, MTP +100–159% bs1 / +38–90% bs32-64,
  NIAH 128K→1M ~16% decay, >90% KV hit rate
- https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-176b-model-on-nvidia-gb300-nvl72-for-agentic-coding/
  — NVIDIA GB300 NVL72 blog (2026-08-26): Qwen3.8-Flash-Next = Qwen4 preview (125B+51B PLE MoE,
  GDN+QSA 3:1), >16K tok/s/GPU & >200 tok/s/user FP8; SGLang/vLLM/TokenSpeed all publish recipes
- https://huggingface.co/Qwen/Qwen3.8-27B — model card: 64 layers `16×(3×(GDN→FFN)→1×(GatedAttn→FFN))`,
  48 V / 16 QK heads head_dim 128, 24 Q / 4 KV heads head_dim 256, MTP, 262144 native / 1M extensible,
  VLM (image+video), Apache-2.0; "serving engines such as SGLang, vLLM, or TokenSpeed are recommended"
  with links to each recipe
- https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json — `Qwen3_5ForConditionalGeneration`,
  `full_attention_interval: 4`, `mtp_num_hidden_layers: 1`, `mamba_ssm_dtype: float32`,
  `max_position_embeddings: 262144`, qwen3_5 vision tower
- https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B — 2.4T card: "compatible with vLLM, SGLang,
  TokenSpeed, etc."; 92 layers, 23×(3 GDN + 1 GatedAttn), 512 experts 10+1, 262144 native / ~1,010,000
  extensible
