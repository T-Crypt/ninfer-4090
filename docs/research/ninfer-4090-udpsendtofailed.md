# Dossier: NInfer-4090 (UDPSendToFailed/ninfer-4090)

Researched 2026-10-07 (public sources only: the repo's `feat/rtx-4090-sm89-native` branch tree,
README, in-repo release notes and maintainer docs, GitHub issues/PRs, and the parent port
`Don-Chad/ninfer-3090`). No installs, no weight downloads, no GPU jobs.

## What it is / backing / license

- **What it is:** a from-scratch **C++20/CUDA inference engine specialized to one RTX 4090
  (`sm_89`, AD102, 128 SMs) running Qwen3.8-27B** — the exact target card + model of the
  prep4qwen effort. It is a **sibling port of our own engine lineage**: a fork of
  `Don-Chad/ninfer-3090` (itself a fork of `Neroued/ninfer`), re-targeted from `sm_86` to native
  `sm_89` ("NInfer port for Qwen 3.8 27B on RTX 4090" — repo description). Same public surface:
  groupwise `.ninfer` artifact, CLI + OpenAI/Anthropic-compatible server, `ninfer_bench`
  ([README](https://github.com/UDPSendToFailed/ninfer-4090)).
- **Backing:** community — single maintainer (GitHub user UDPSendToFailed) with a handful of
  community PRs (#6 eyedark, #7 keylimesoda, #9 Suffice, #12 diodiogod). README disclaimer: "a
  fork of NInfer I am developing for fun to push the limits of the speed and context window for
  Qwen 3.8 27B on the RTX 4090 … Co-developed with Gemini 3.7 Flash" (i.e. heavily AI-assisted).
  As of 2026-10-07: **176 stars, 24 forks, 6 open issues** (GitHub API).
- **License:** **Apache-2.0** (repo license field, in-tree LICENSE) — porting code is
  license-free.

## Repo, version, last release date

- Repo: <https://github.com/UDPSendToFailed/ninfer-4090>, default branch
  `feat/rtx-4090-sm89-native` (fork parent `Don-Chad/ninfer-3090`, network root `Neroued/ninfer`).
- Created **2026-08-15**; last push **2026-09-09**. Releases: v0.7.0 (2026-08-15) → v0.8.0 →
  v0.9.0 → v1.0.0 (2026-08-18) → v1.1.0 (2026-08-25) → **v1.2.0-rtx4090 (2026-09-09, latest)**
  ([releases](https://github.com/UDPSendToFailed/ninfer-4090/releases)). No release since —
  4.5 weeks idle at research time.
- Branches also keep the 3090 release line (`release/v0.4.0-rtx3090` … `v0.6.0-rtx3090`) and a
  `feature/qwen38-rk8v4-paged-sm86` branch — the repo doubles as a 3090/4090 fork of Don-Chad.
- **The parent `Don-Chad/ninfer-3090`** (the "popular port" this derives from): 436 stars,
  50 forks, Apache-2.0, sm_86/RTX 3090, Qwen3.8-27B first-class with ReplaySSM + MTP3 + C1–C8,
  measured 171K-token INT8 context, Windows prebuilt archives + Linux Docker/source builds
  ([README](https://github.com/Don-Chad/ninfer-3090)).

## Runs on an RTX 4090 (sm_89)?

**Yes — sm_89 is its only target.** "Specialized, high-performance C++20/CUDA inference engine for
Qwen3.8-27B on a single 24 GB NVIDIA GeForce RTX 4090 (`sm_89`)" ([README](https://github.com/UDPSendToFailed/ninfer-4090));
AGENTS.md: "Target Hardware: NVIDIA GeForce RTX 4090 (24 GB GDDR6X, AD102, 128 SMs)"; v1.2.0
release notes: "specializes the runtime and operator pipelines for the RTX 4090 (sm_89, AD102)",
and the release **deleted 45+ Blackwell SM120/NVFP4 kernel files** to keep the codebase strictly
Ada. One caveat: the headline capacity features (D3D12/WDDM residency eviction, DirectStorage 1.3
disk state cache) are **Windows-only** (`if(WIN32)` CMake blocks, `d3d12.lib`/`dstorage.lib`);
the early `sm_89` qualification ran on a **Vast.ai Linux** box (CUDA 12.8.93,
[docs/rtx-4090-early.md](https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/rtx-4090-early.md)),
but current build docs are Windows 11 + MSVC + CUDA 13.3 only, and a Linux/Docker build issue is
open (#14). On Linux the engine compiles without those Windows features; no Linux performance
qualification is published for v1.0+.

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, first-class** — same ReplaySSM design as upstream NInfer (raw-input replay of
  GDN state transitions instead of per-position state snapshots; [docs/maintainer/replayssm-gdn.md](https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/maintainer/replayssm-gdn.md)),
  plus fork-specific Ada tuning: XOR-butterfly (`__shfl_xor_sync`) all-reduce in the GDN
  transition, 64-bit packed value loads, vectorized causal-conv1d (bf16x2 over all 5,120
  channels), 2D batched state-pool transfers (96 driver calls → 2), and — the v1.2.0 centerpiece —
  **GDN/attention input projections (q/k 2048 rows, v/z 6,144 rows each) run on Ada Tensor
  Cores** (`mma.sync.aligned.m16n8k16` + `cp.async` double-buffered smem staging, `T in [2,16]`
  bands) with an in-CTA epilogue that stages the MMA column accumulators into smem and executes
  the 1D causal conv + state publish in the same kernel
  ([v1.2.0 notes](https://github.com/UDPSendToFailed/ninfer-4090/releases/tag/v1.2.0-rtx4090)).
- **MTP: yes, deep** — `--spec mtp --draft-tokens 1..15` (window expanded K=5→15, `W≤16` in
  v1.1.0, with dynamic shared memory for `TokenTile ≥ 7`), optional `--lm-head-draft` (dedicated
  proposal head, not the full base LM head), plus **hierarchical n-gram prompt-lookup drafting**
  (N=5→4→3→2, zero-allocation, v0.8.0) layered alongside MTP. 100% draft acceptance is achieved
  on their repetitive bench corpus at K=7 (8.00 tok/round). DFlash: v3 artifact `dflash2/*`
  tensors (feature projection, candidate selector with 248,320×256 codebooks) are consumed as a
  **validate-only stub** — no DFlash execution on this fork (fixes #13; v1.2.0 notes).
- **Vision: yes** — `--vision` for the Qwen3.8-27B VLM (image/video, configurable
  `--vision-max-tokens` scratchpad, default 8,192); context-ceilings matrix includes vision rows
  ([README](https://github.com/UDPSendToFailed/ninfer-4090)).
- **256K+: yes, via their KV-quant ladder** — INT8 group-64 KV tops out at 223K (MTP0) / 181K
  (MTP4) tokens on 24 GB; the E8 4-bit modes reach 433K and **2-bit E8-cylinder keys (`rk2v4-e8`,
  100 B/tok) reach 567K text-only** ([README](https://github.com/UDPSendToFailed/ninfer-4090));
  native context constant lifted to 1,048,576 with **YaRN 1M frequency scaling + attention
  temperature scaling** (v1.1.0). 100% recall on a 359,169-token needle-in-a-haystack with
  `rk2v4-e8`.

## Published numbers on a 4090 or similar

All single RTX 4090 (24 GB), Qwen3.8-27B groupwise `.ninfer` (16.96 GiB), CUDA 13.3 unless noted.
`ninfer_bench` convention: `ppP` = P-token prefill, 1 output token; `tgG` = G decode tokens;
prefix reuse disabled in the bench matrix.

**v1.2.0 (2026-09-09), [README](https://github.com/UDPSendToFailed/ninfer-4090), batch = 1 request:**

| Case | Config | Throughput | Notes |
|---|---|---:|---|
| Prefill | `pp512`, chunk 1024, INT8 KV | 1,971.5 ± 6.4 tok/s | |
| Prefill | `pp2048`, chunk 1024, INT8 KV | 2,146.3 ± 3.0 tok/s | |
| Prefill | `pp4096`, chunk 1024, INT8 KV | 2,637.6 ± 3.3 tok/s | |
| Decode | `pp32768+tg128`, greedy, MTP7, `rk4v4-e8` | 272.5 ± 0.7 tok/s | 100% draft acceptance (8.00 tok/round) |
| Decode | `pp2048+tg128`, greedy, MTP7, INT8 | 220.8 ± 23.5 tok/s | 88.0% acceptance (7.11 tok/round) |
| Decode | `pp2048+tg128`, greedy, MTP7, `rk4v4-e8` | 226.0 ± 23.2 tok/s | 88.0% acceptance |
| Decode | `tg128` cold corpus, greedy, MTP4, `rk4v4-e8` | 79.5 ± 8.7 tok/s | 28.4% acceptance on cold seed |
| Decode | `tg128`, no spec, INT8, CUDA Graph | 51.9 ± 2.2 tok/s | baseline |
| DMA | DirectStorage cold restore, 77,615 tok (1.51 GiB) | 150 ms (10.1 GB/s) | drops cold TTFT 52.6 s → 1.86 s (Windows) |
| NIAH | 359,169 prompt tokens, `rk2v4-e8` | 100% (5/5), 666.7 tok/s avg prefill | |

**v1.1.0 (2026-08-25)** ([release notes](https://github.com/UDPSendToFailed/ninfer-4090/releases/tag/v1.1.0-rtx4090)):
pp2048 int8 2,093.50 ± 0.68; pp4096 int8 2,079.51 ± 0.37; pp512 int8 1,738.07 ± 82.94;
pp2048 `rk4v4-e8` 2,085.58; MTP7@2k int8 218.33 ± 0.91 / e8 216.85 ± 1.11 (88%, 7.11 tok/round);
MTP7@32k e8 229.86 ± 0.09 (100%, 8.00); MTP4 cold 89.20 ± 3.45 (30.7%); MTP0 52.77 ± 0.03.

**v1.0.0 (2026-08-18)** ([release notes](https://github.com/UDPSendToFailed/ninfer-4090/releases/tag/v1.0.0-rtx4090)):
pp2048 1,863.8 ± 1.8; pp4096 1,849.3 ± 2.1; pp512 1,736.8 ± 36.0; MTP3 code/math
103.5–148.2 tok/s (55–91% acceptance); MTP4 code&schemas 96.8–129.9 (46–88%); MTP3 bench corpus
83.7 ± 2.9 (36.0%); MTP0 51.4 ± 0.5; NIAH 359,169 tok `rk2v4-e8` 100% (5/5).

**v0.8.0 (2026-08-15)** ([release notes](https://github.com/UDPSendToFailed/ninfer-4090/releases/tag/v0.8.0-rtx4090)):
pp2048 1,938.0 ± 2.7 (+12.4% vs v0.7.0's 1,724.2); MTP4 code/schemas 110.0–110.4 (peak 123.7)
with 67–75% acceptance (3.39–3.68 tok/round) from the n-gram prompt-lookup drafter; MTP3 corpus
77.0 ± 0.1; MTP0 48.8 ± 4.3; multi-turn checkpoint TTFT 1.79–2.05 s vs 3.80 s (reused 24.4k
prompt tokens).

**Early `sm_89` compatibility build (2026-08-15, Vast.ai RTX 4090, CUDA 12.8.93, INT8 KV,
MTP3, C1–C8 cohort sweep, 1,024 outputs/request)**
([rtx-4090-early.md](https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/rtx-4090-early.md)):
C1 102.13 end-to-end / 103.35 decode tok/s (MTP accept 45.31%, TTFT 112 ms, 18,250 MiB);
C2 162.46/165.74 (53.13%); C4 193.49/198.76 (56.91%); C8 299.82/315.09 tok/s (53.16%,
20,708 MiB). This is the only C8 number on a 4090; the v1.0+ matrices are all single-request.

**Context ceilings (24 GB, binary-searched, v1.0.0/v1.2.0)**: `rk2v4-e8` 567K (MTP0) / 462K
(MTP4) / 415K (vision 8k + MTP4); `rk4v4-e8` & `rk4v4` 433K / 352K; `rk8v4` 294K / 239K; INT8
223K / 181K / 163K (vision). Cosine vs FP32: 96.2% (rk2v4-e8) → 99.8% (int8). Recommended safe
operating point: 20–30K below the physical ceiling. L1-block-table GQA decode supports a 1M-token
envelope.

**KV-mode footprint (v0.9.0 notes)**: `rk2v4-e8` 100 B/tok (8-bit index over the 240 minimal E8
roots + 4-bit log-scale radius + 4-bit residual axis for keys; 4-bit values), `rk4v4-e8` 136 B/tok
(Conway-Sloane E8 lattice keys + 4-bit values), `rk4v4` 132, `rk8v4` 196, int8 260 B/tok.

**Parent 3090 port for scale** (Don-Chad, sm_86, [README](https://github.com/Don-Chad/ninfer-3090)):
C1 70.19/71.00 tok/s end-to-end/decode (MTP3, 61.13% acceptance, 19,641 MiB); C8 161.28/165.33;
aggregate prefill 861.51 tok/s (C1, 4,362 fresh tokens); 171K-token INT8 context. So the 4090
fork's prefill (2,146 at pp2048) is roughly 2.5× the 3090's, decode MTP3 ~1.7×.

`docs/performance.md` in the fork still carries the upstream **RTX 5090** Qwen3.6 campaign
(27B groupwise-int C1 185.8 → C8 535.0 tok/s aggregate, MTP3, MTP acceptance ~68%) and explicitly
excludes Qwen3.8-27B from that campaign — it is not a 4090 data source.

## Ideas we could port into ninfer-4090

Apache-2.0 throughout. Ranked for our own 4090 target (we already own the NInfer core, so these
are the sm_89-specific divergences that matter):

1. **E8 lattice/cylinder KV quantization** (`src/ops/kernel/e8_lattice.cuh`, `e8_root_codec.cuh`,
   + the `kv_cache_append` / fused-attention quantization paths): 8-D Conway-Sloane E8 nearest
   lattice projection for 4-bit keys, 240-root S^7 factorization for 2-bit keys (H8 Hadamard
   rotation in registers, multiplier-free dot decode, `bfi.b32` packing + `redux.sync.add` warp
   reduction). This is *the* technique that buys 223K → 567K context on 24 GB — i.e. the only
   public path to 256K+ INT8-free KV on a 24 GB card. Their measured quality: 98.7% / 96.2%
   cosine vs FP32, 100% NIAH recall at 1M (4-bit) and 360k (2-bit). Port cost: they fused
   quantize/decode into the GQA attention + append ops, so this is a KV-contract change, not a
   drop-in. The 2-bit cylinder (240-root codebook + log-scale + residual axis) is the interesting
   new primitive; v1.1 "harden E8 codec, document half-coset, gate verifier on needle retrieval"
   (b01692c6) shows the verification story.
2. **Direct L1-cached global block-table lookups in GQA decode** (v0.9.0: "replaced the static
   shared memory block table array … with direct L1-cached global lookups"; `kNativeContext` and
   visible-key ceilings lifted 262,144 → 1,048,576): this is what makes a 1M-token envelope
   cheap to address, and it composes with any KV dtype. Cheap to port, big context-headroom win.
3. **YaRN 1M scaling machinery** (`yarn.h` + dynamic device frequency tables + dynamic
   `attn_scale` threaded through prefill/decode/spec-verify): 3-band inverse-frequency
   interpolation + attention temperature scaling anchored at 1M, updated at runtime. Exactly what
   the next dense Qwen (256K+) will want on a 4090.
4. **Ada Tensor-Core small-T decode path** (v1.2.0): `mma.sync.aligned.m16n8k16` + `cp.async`
   double-buffered smem pipelines for Q4/Q5/W8 small-T projections (T ∈ [2,16]), **Split-K=2
   integer-wave scheduling on 128 SMs** (640/64.0/5.0/7.0 exact waves, BF16 `atomicAdd`
   accumulation), and **stream fork-join co-scheduling of Q/K and G/V projections** (896 blocks →
   exactly 7.0 waves). Their T=1 draft-head path maps to 64.0 integer waves (8,192 blocks). If our
   decode still runs these projections SIMT-style, this is the concrete Ada schedule to diff
   against; the GDN input-proj + causal-conv in-CTA fusion (one kernel: tensor-core projection →
   smem staging → 1D conv → state publish) is the standout.
5. **72 MB persisting-L2 pinning for MTP proposal weights** (v0.8.0: `cudaLimitPersistingL2CacheSize`
   at the Ada max + `cudaAccessPropertyPersisting` stream access-policy windows): keeps the whole
   drafter weight set out of DRAM on every speculation round. One-line class of change, Ada-specific
   (72 MB L2); worth A/B testing on our drafter.
6. **Hierarchical n-gram prompt-lookup drafting** (v0.8.0, `find_prompt_lookup_draft`):
   zero-allocation scan of the sequence ledger for N=5→4→3→2 grams, drafting continuations from
   prompt history/code syntax/repeated tool schemas; 67–75% acceptance (3.39–3.68 tok/round) on
   structured code/schemas, layered with MTP. For agentic workloads — where MTP acceptance is
   worst (their cold-seed MTP4: 28.4%) — this is a direct complement to our MTP3; vLLM/SGLang
   ship the same trick, but their ledger-integrated version is written for this engine's request
   model.
7. **GDN Ada micro-optimizations** (v1.0.0): XOR-butterfly all-reduce in `apply_gdn_transition`
   (5-step `__shfl_xor_sync`, all 4 rows simultaneously), 64-bit packed value loads at load time,
   vectorized bf16x2 causal-conv1d, 2D `cudaMemcpy2DAsync`/`cudaMemset2DAsync` batched state-pool
   & cyclic-KV transfers (96 driver calls → 2, ~5 µs each), `bfe.s32` dequant atoms, zero-bank-
   conflict smem staging. Incremental but all free wins for our GDN stack; the XOR butterfly
   pattern transfers to any per-head row reduction.
8. **MTP window K ≤ 15** (v1.1.0) incl. the `TokenTile ≥ 7` dynamic-smem sizing that keeps static
   smem under the 48 KB linker limit on sm_89, and the `W ≤ 16` small-T verification routing
   (`kSmallTChunkTokens = 8`, 27B `q_heads = 24`): we currently run MTP3; their 100%-acceptance
   K=7 result on repetitive corpora and the 88% (7.11 tok/round) MTP7@2k result argue for testing
   deeper draft windows on our MTP head.
9. **WDDM/DirectStorage layer: skip on Linux** — crash-safe D3D12 residency, evictable budgeting,
   and the CoW DirectStorage disk state cache (`pool_data.ninfer_pages` 64-token pages, mark-and-
   sweep compaction guards) are the right *design* for persisting turn checkpoints across
   process restarts (their 150 ms / 10.1 GB/s cold restore), but Windows-only. For our Linux box
   the portable analog is a CoW page-journaling disk cache over ordinary NVMe DMA — note the
   design, skip the code.
10. **dflash2 validate-only stub** (`bind_dflash2_stub`, 66 tensors incl. candidate-selector
    248,320×256 VQ codebooks): their answer to the updated v3 Qwen3.8-27B-NInfer artifact; a
    pattern for us to consume next-Qwen drafter payloads without executing them.

## Worth running beside NInfer/llama.cpp?

**Yes — it is the closest public analog to our project** (same card, same model, same artifact
family, same engine lineage), and it's the only public single-4090 Qwen3.8-27B engine with a full
kernel-tuning history. A fair head-to-head against our ninfer-4090 build:

- **Same artifact** (official `qwen3_8_27b.ninfer` groupwise, 16.96 GiB) or identical weights;
  **greedy** decoding on both (their matrices are all `--greedy`); same corpus + seeds
  (`bench/fixtures/bench_corpus.ids` lineage), same KV dtype per leg.
- Legs: (a) prefill `pp512/pp2048/pp4096` chunk-1024, INT8 KV; (b) decode `tg128` MTP0 / MTP3 /
  MTP7 (C1); (c) `pp2048+tg128` and `pp32768+tg128` MTP7; (d) NIAH 359k with `rk2v4-e8`-equivalent
  KV on both sides (if we port it) and INT8 on both (parity); (e) C4/C8 cohort decode to check
  the 315 tok/s C8 early-build number.
- **Practical obstacle: platform.** Their flagship builds are Windows 11 (MSVC + CUDA 13.3; WDDM
  eviction is what unlocks the 567K ceiling numbers). On our Linux box the build path is
  unproven (open issue #14; no Linux qualification published since the early 2026-08-15 Vast.ai
  run). A Linux build (or Windows VM with GPU passthrough) is a prerequisite for the race; until
  then its value to us is the port list above, not a benchmark.
- llama.cpp remains the better *portable* baseline; NInfer-4090 is the better *design* reference.

## Risk / unknowns

- **Single-maintainer, AI-assisted, short-lived.** "Developing for fun … co-developed with Gemini
  3.7 Flash"; 2.5 months of history, idle since 2026-09-09; 176 stars but small community. No
  published CI. Expect churn and possible abandonment; pin the exact release tag for any reuse.
- **Self-reported numbers, no independent validation.** All matrices come from the maintainer's
  own `ninfer_bench` runs; variances are sometimes large (MTP7@2k ± 23.5 on a 220 tok/s mean —
  ~10%), and the headline MTP7 "100% draft acceptance" is on their repetitive bench corpus, not
  agentic text (their own cold-seed MTP4 acceptance is 28.4–30.7%). Treat the 272.5 tok/s decode
  as a corpus artifact.
- **Windows-only headline features.** WDDM residency, DirectStorage, and the ceiling table were
  all measured under Windows WDDM (and the ceilings only reproduce with
  `--wddm-evictable-budget` — issue #10). A Linux 4090 with a full 24 GB budget should be at or
  above these ceilings, but nobody has published a Linux v1.x matrix.
- **Agentic prefix-reuse bug (open #15, v1.2.0):** agent-loop histories ending in `role:"tool"`
  get pinned to a turn-checkpoint frontier that never advances → 14.2–14.6 s re-prefill per turn
  at ~35K context vs 0.30–0.34 s with token-exact replay. Directly relevant to our agent
  workload: it means their "compatible-prefix reuse" is *not* a drop-in fix for tool-loop
  histories, and the workaround (client replays its own generation, `preserve_thinking`) is
  exactly the kind of protocol asymmetry we should design against.
- **E8 2-bit KV quality.** 96.2% cosine vs FP32 and a 360k NIAH pass are necessary but not
  sufficient evidence for a 256K agentic context; there is no long-reasoning or MTBench-class
  accuracy study of `rk2v4-e8` anywhere in the repo. Their own 3090 parent already retracted the
  old RotorQuant numbers once when upstream changed the KV-quantization Op contract ("the
  previously published 226K/248K RotorQuant context figures do not apply to this build") — a
  reminder that these context claims track a moving contract.
- **MTP KV quantization not planned** (issue #3 closed "not planned"): MTP-layer KV stays
  unquantized, which is what eats the headroom in their MTP4 ceilings (462K vs 567K).
- **v3 artifact / DFlash2 are stubs, not engines** — the next-Qwen drafter story (DFlash2
  candidate selector) is present in the artifact but not executed here; z-lab's drafter remains
  the executable option (see the z-lab row in `targets.md`).
- **No published C2–C8 numbers for v1.0+** on the 4090 (only the early C1–C8 compatibility
  sweep); their serving concurrency work (admission-shortfall fix, executor health reporting)
  post-dates the only cohort measurements.

## Sources

- <https://github.com/UDPSendToFailed/ninfer-4090> — repo (fork of `Don-Chad/ninfer-3090`; network
  root `Neroued/ninfer`), 176 stars / 24 forks (GitHub API, 2026-10-07), default branch
  `feat/rtx-4090-sm89-native`, Apache-2.0, created 2026-08-15, last push 2026-09-09
- <https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/README.md> —
  positioning, v1.2.0 benchmark matrix, context-ceilings matrix, serving commands, build
  prerequisites (Windows 11 + CUDA 13.3), license/credits, "co-developed with Gemini 3.7 Flash"
- <https://github.com/UDPSendToFailed/ninfer-4090/releases> + in-tree `RELEASE_NOTES_0.8.0.md`,
  `RELEASE_NOTES_0.9.0.md` (E8 lattice/cylinder KV, rk4v4, L1 block-table 1M), `RELEASE_NOTES_1.0.0.md`
  (WDDM residency, GDN/GQA/vectorization, 567K ceiling, NIAH 360k), `RELEASE_NOTES_1.1.0.md`
  (DirectStorage CoW cache, MTP K=15, SFU math, 128-SM wave alignment, YaRN 1M),
  `RELEASE_NOTES_1.2.0.md` (Ada MMA small-T, split-K waves, GDN in-CTA epilogue, NVFP4 purge,
  dflash2 stub)
- <https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/rtx-4090-early.md> —
  early sm_89 qualification, Vast.ai Linux C1–C8 sweep (2026-08-15)
- <https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/performance.md> —
  upstream RTX 5090 Qwen3.6 campaign (not a 4090 data source)
- <https://github.com/UDPSendToFailed/ninfer-4090/blob/feat/rtx-4090-sm89-native/docs/maintainer/replayssm-gdn.md>,
  `.../docs/maintainer/paged-kv-cache.md`, `.../docs/serving.md` (prefix-reuse semantics, KV pool
  admission), `.../AGENTS.md` (sm_89 invariants: 128 SM waves, 72 MB L2, 128-register budgets,
  16-byte vectorization, `-rdc=false`), `.../model-cards/Qwen3.8-27B-NInfer/README.md` (16.96 GiB
  artifact), `src/ops/kernel/e8_lattice.cuh`, `e8_root_codec.cuh`, `src/ops/gdn_input_proj/`,
  `src/ops/linear_attention/gated_delta_net/`, `src/targets/qwen3_6_27b/impl/config.h`
  (`kNativeContext = 1048576`), `.../yarn.h`
- Issues: <https://github.com/UDPSendToFailed/ninfer-4090/issues/2> (vision + MTP hot-swap),
  #3 (MTP KV quantization — closed not planned), #14 (Linux Docker build), #15 (agent-loop
  prefix-reuse pinning), #16 (DirectStorage crash), #17 (reasoning level), #22 (v3 artifact
  support), <https://github.com/UDPSendToFailed/ninfer-4090/issues/23> (sm120 + Q3 request;
  references sibling ports `natpate/ninfer-windows`, `toddballinger/ninfer-5080`)
- <https://github.com/Don-Chad/ninfer-3090> — parent port: README (436 stars, 50 forks,
  Apache-2.0, sm_86, 171K INT8 context, C1–C8 MTP3 numbers, RotorQuant retraction, Linux/Windows
  guides), `RELEASE_NOTES_0.6.0_RTX4090_EARLY1.md` (the 4090 early build this fork starts from)
- <https://github.com/natpate/ninfer-windows> — Windows 11 NInfer port for RTX 5090 (v3 artifacts,
  mixed-format weights incl. Q3)
- <https://github.com/toddballinger/ninfer-5080> — RTX 5080 16 GB Qwen3.8-27B port, true 131K
  context + vision, Q4 group64 KV, MTP-3, v1.5 CUDA 13.4 Blackwell runtime
