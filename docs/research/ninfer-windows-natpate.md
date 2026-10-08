# Dossier: natpate/ninfer-windows

Researched 2026-10-07 (public sources only: the repository tree, README, in-repo
maintainer docs and model cards, GitHub releases/issues/PRs, and the child fork
`cometkim/ninfer`). No installs, no weight downloads, no GPU jobs.

## What it is / backing / license

- **What it is:** a **Windows 11 x64 port of `Neroued/ninfer`** — our engine's own upstream
  lineage — re-targeted to run on a single **RTX 5090 (`sm_120a`, 32 GB)**. It tracks
  upstream `master` and layers a Windows compatibility layer on top (ported from
  `Don-Chad/ninfer-3090`, without that fork's sm_86 retargeting or kernel reschedules),
  MSVC/TMA compile fixes, an in-process stock llama.cpp WebUI, `meta.n_ctx` context
  advertising in the OpenAI dialect, and self-contained portable release zips
  ([README](https://github.com/natpate/ninfer-windows)). The engine, artifacts, API
  surface and published benchmarks are upstream's; the fork diff is small.
- **Backing:** community — single maintainer (GitHub user natpate); the repo's "Support"
  section points to Neroued's Ko-fi, i.e. it's a personal port of Neroued's personal
  project. As of 2026-10-07: **167 stars, 19 forks, 3 open issues** (GitHub). One
  community PR was merged (E8-KV PR #1); one direct child fork exists
  (`cometkim/ninfer`, see New targets below).
- **License:** **Apache-2.0** (repo license field, in-tree LICENSE, model cards) — porting
  is license-free.

## Repo, version, last release date

- Repo: <https://github.com/natpate/ninfer-windows> (fork of `Neroued/ninfer`; network
  root `Neroued/ninfer`), default branch `master`.
- First release **v0.4.0 (2026-08-24)**; then v0.5.0 (2026-08-29), v0.6.x, v0.7.0
  (2026-09-07), v0.8.0 (2026-09-19), v0.8.1 (2026-09-25), v0.9.0 (2026-09-25),
  **v0.9.1 (2026-09-29, latest)** ([releases](https://github.com/natpate/ninfer-windows/releases)).
  Nine weeks of history, active through research time.
- Release history (each a portable win64 zip, CUDA 13.1 runtime baseline, built on the
  13.3 toolchain):
  - **v0.4.0 (2026-08-24):** community PR #1 — live compressed-KV quantization, four
    `--kv-dtype` modes `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` (Conway-Sloane E8 lattice
    keys at 2–4 bits/dim; "200k+ context on a 32 GB card"); source was
    [Neroued/ninfer PR #35](https://github.com/Neroued/ninfer/pull/35), contributed by
    @alexandergwosdz. Tool-message array content (text + image_url) accepted.
  - **v0.5.0 (2026-08-29):** upstream sync (2026-08-28, ~50 commits: full `/v1/responses`
    surface, Anthropic thinking signatures, paged/cyclic KV stores, host KV arena,
    context-cost admission, restructured attention kernel tree, TTFT harness); MSVC/CMake
    build of the full engine + serve stack with statically linked cudart; launcher
    context baseline 190k.
  - **v0.7.0 (2026-09-07):** **DFlash2** speculative backend (`--spec dflash2
    --draft-tokens 1-15 --lm-head-draft`; Qwen3.8-27B artifacts ship companion weights)
    — "measured up to ~42% faster decode than MTP on cache-heavy workloads at 131k
    context (small fixed-budget comparison, not a quality evaluation)"; new `nvfp4full`
    profile (cometkim's artifact); upstream sync (sparse-MoE prefill kernel, DFlash2 Op).
  - **v0.8.0 (2026-09-19):** **v3 artifact format required** (model/weight decoupling;
    v2 files need the offline `tools/upgrade_ninfer_v2_to_v3.py`); nvfp4full excluded;
    custom jinja chat templates (llama.cpp base imported); perf tuning (q4/q8 linear-add,
    nvfp4 34816×5120 route, MoE gate/up pipeline depth, W4A4 TMA rasterization, flat
    open-addressed BPE merge table); Windows port of the v2→v3 upgrade + artifact tools.
  - **v0.8.1 (2026-09-25):** nvfp4 **W4A4 TMA route** (activation scales one tile per TMA
    request, partial final token tile, accurate-silu fused SwiGLU epilogue); Q4/Q5
    dispatch retunes; **`NINFER_CUDA_SYNC` env knob, engine now defaults to spin-wait**
    on CUDA completion (`spin`/`blocking`/`yield`/`auto`).
  - **v0.9.0 (2026-09-25):** **weight-only NVFP4 DFlash2 execution** (drafter on NVFP4
    artifacts; `--spec dflash2` for `qwen3_8_27b_nvfp4full`); nvfp4full launcher
    restored.
  - **v0.9.1 (2026-09-29):** **Kimi Delta Attention (KDA)** linear-attention Op (chunked +
    recurrent paths, CPU reference test) — "the kernel foundation for KDA-based models";
    **GDN chunked prefill replaced by a two-stage kernel**; unified MMA templates +
    tuned sliced-k schedules across q4/q5/q6/q8/FP8/NVFP4/BF16 linear, linear-add,
    SwiGLU Ops; MSVC compile fix for the new template-launcher kernel family (constexpr
    kernel-alias lambda capture).

## Runs on an RTX 4090 (sm_89)?

**No.** The fork inherits upstream's Blackwell-only target and hard-enforces it:
`CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 120a` and any other value fails
configure ("NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a"); README: "The build
rejects CUDA architectures other than `120a`" and "NVIDIA GeForce RTX 5090
(`sm_120a`)" under Requirements; `AGENTS.md`: "The implementation targets `sm_120a`
and is tuned on NVIDIA GeForce RTX 5090". Every published number is on a 5090.
Windows itself is well supported (that is the point of the fork), but the GPU support
is `sm_120a` only. Its Windows layer is a *plain CUDA-on-Windows* path — no D3D12
residency, no WDDM budget/eviction handling, no DirectStorage anywhere in the tree
(grep of README/docs/CMakeLists: zero hits); the platform work is vcpkg (FFmpeg/
libcurl/zlib), statically linked cudart, memory-mapped artifact reads with unbuffered
overlapped I/O (Windows counterpart of POSIX `O_DIRECT`/`pread`, same 4096-byte
alignment contract), and launcher defaults of a 150k context "to leave VRAM headroom
for the Windows desktop". That is a thinner WDDM story than the UDPSendToFailed 4090
fork's D3D12 residency + DirectStorage layer (see
[ninfer-4090 dossier](ninfer-4090-udpsendtofailed.md)).

## Qwen3.5–3.8 hybrid (Gated DeltaNet), MTP, vision, 256K

- **GDN hybrid: yes, first-class** — same upstream `Qwen3_5ForCausalLM` /
  `Qwen3_5MoeForCausalLM` native paths (fork `AGENTS.md`); the v0.9.1 GDN two-stage
  chunked-prefill kernel and KDA Op additions land here at the same time as upstream.
- **MTP: yes, window 1–5** (README capability list; methodology: MTP3 = `--spec mtp
  --draft-tokens 3 --lm-head-draft`), plus **DFlash2 window 1–15** on Qwen3.8-27B
  artifacts (companion weights from `z-lab/Qwen3.8-27B-DFlash2` rev
  `50307d4c` embedded in the artifacts, per the model card), incl. the v0.9.0
  weight-only NVFP4 drafter route.
- **Vision: yes** — image/multi-image/video/mixed messages via `--vision`; multimodal
  evaluation ran at an 81,920-token context limit.
- **256K: yes** — the Qwen3.8-27B campaigns run at a 262,144-token context ceiling
  (260,096-token prefill points); the Qwen3.8-27B NVFP4 eval used 252,928 "to fit the
  RTX 5090 after weights". **Beyond 256K: not in this fork** — YaRN/1M support is an
  open issue ([#24 "Future YaRN support?"](https://github.com/natpate/ninfer-windows/issues/24))
  and lives instead in the child fork `cometkim/ninfer` (`feat/1m-context`, up to
  1,048,576 logical tokens).

## Published numbers on a 4090 or similar

**Not found for an RTX 4090** — every published measurement in this repo is on one
**RTX 5090 (32 GB)** with the common serving profile (CUDA 13.1, loopback OpenAI
Chat Completions, prefill chunk 1,024, INT8 group-64 KV, CUDA Graphs, prefix reuse
disabled; [methodology](https://github.com/natpate/ninfer-windows/blob/master/docs/performance/methodology.md)).
The numbers below are the 5090 reference points (useful as the upper bound of what
our 4090 build should chase per-feature, not as 4090 expectations):

**Concurrent MTP3 decode saturation** (8,192-token generation per active request;
[README](https://github.com/natpate/ninfer-windows)):

| Model profile | C=1 tok/s / accept | C=2 | C=4 | C=8 | C8/C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% | 2.15× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

**Qwen3.8-27B single-request serving** (serial, INT8 KV; [qwen3.8-27b performance page](https://github.com/natpate/ninfer-windows/blob/master/docs/performance/qwen3.8-27b.md)):

| Profile | 7,680-tok prefill | 260,096-tok prefill | Structured MTP3 decode | MTP0 decode @ 260,096 ctx |
|---|---:|---:|---:|---:|
| `groupwise-int` | 3,274.7 ± 11.3 tok/s | 1,609.7 ± 5.3 tok/s | 224.4 ± 13.6 tok/s (89.5% accept, 3.68 tok/round) | 56.2 ± 0.6 tok/s |
| `nvfp4` | 8,340.4 ± 13.0 tok/s | 2,203.1 ± 13.4 tok/s | 219.8 ± 8.6 tok/s (90.8% accept, 3.72) | 52.9 ± 2.3 tok/s |

**Qwen3.8-27B corpus makespan** (75 requests/point, MTP3, 131,072 ctx; decode tok/s
over full makespan, acceptance over the wave):

| Profile | C=1 | C=2 | C=4 | C=8 | auto KV @ C=8 |
|---|---:|---:|---:|---:|---:|
| `groupwise-int` | 161.7 (58.5%) | 214.0 (59.5%) | 258.2 (58.3%) | 315.3 (58.9%, avg batch 4.76) | 313,984 tok |
| `nvfp4` | 161.1 (60.8%) | 294.7 (59.2%) | 432.9 (58.0%) — best | 334.2 (57.6%, avg batch 2.36) | 187,712 tok (memory-pressure-limited) |

Weight arenas: groupwise 16.672 GiB vs NVFP4 19.729 GiB resident — the smaller
profile's C=8 headroom is why groupwise wins C8 and nvfp4 wins C4.

**DFlash2 K=7** (2026-09-06, post terminal-settlement fix): corpus decode 192.5
tok/s (nvfp4, 37.0% accept) vs 147.0 (groupwise, 39.2%); single-request long
reasoning groupwise 224.2 / 133.5 / 175.2 tok/s (AIME26 01/15/30; 175.2 includes a
repetition-loop sample at 302.6 tok/s, other four average 143.3) vs MTP3's 193.4 /
150.1 / 171.1. Published per-workload deltas vs historical MTP3 at C=1: nvfp4
+64.5% (aime01) … +62.3% (structured), −3.8% (story); groupwise −35.6% (story) …
+19.1% (structured). Full-corpus: nvfp4 decode +19.5% / makespan −22.7%; groupwise
−9.1% / +12.1%. The page is explicit these compare different campaigns/continuations
and "do not isolate the backend's effect".

**Evaluation (5090, MTP3, EvalScope 1.9.0, 0-shot rule):** Qwen3.8-27B groupwise
AIME25 96.67% (29/30), AIME26 96.67% (29/30), GPQA-D 87.37% (173/198), ERQA 66.25%
(265/400), RealWorldQA 82.22% (629/765), IFBench 77.67% (233/300) vs official BF16
79.5/—/89.2/65.5/85.9 — "deltas stay within ±3.1 points on the four overlapping
benchmarks" ([model card](https://github.com/natpate/ninfer-windows/blob/master/model-cards/Qwen3.8-27B-NInfer/README.md)).
NVFP4: AIME25/26 96.67%, GPQA-D 90.40%, ERQA 66.25%, RealWorldQA 83.53%.

**Artifacts (v3 containers, SHA-256-verified):** qwen3_6_27b 16.29 GiB; qwen3_6_27b_nvfp4
17.07 GiB; qwen3_8_27b 19.03 GiB (1,190 objects: 1,184 tensors + 6 resources;
Q4/Q5 projections, `q8_g32_fp16` embedding/head); qwen3_8_27b_nvfp4 22.09 GiB;
qwen3_8_27b_nvfp4full 18.07 GiB (cometkim); qwen3_6_35b_a3b 21.23 GiB.

## Ideas we could port into ninfer-4090

Our engine is the upstream lineage itself, so most "features" here are upstream
commits we already own or sync; the fork-specific items and the ones that actually
inform a 4090/WDDM strategy:

1. **Windows/WDDM runtime design (the point of this dossier).** A working NInfer
   Windows path with *no* D3D12/WDDM-API tricks: statically linked cudart (no
   `cudart*.dll` from the toolkit), vcpkg manifest pinning FFmpeg/libcurl/zlib,
   mmap + unbuffered overlapped reads for artifact I/O (exact Windows twin of our
   `O_DIRECT`/`pread` path, same 4096-byte alignment contract), portable console
   logging, and plain driver-mode CUDA with desktop headroom managed by policy
   (launcher default `--max-context 150000`, "smaller models can be safely raised
   to 200,000 when VRAM is completely free at startup"; the two big profiles must
   stay at 150,000). If our Windows/WDDM path ever matters, this is the minimal
   reference: WDDM cost shows up as memory budgeting, not as driver-API changes.
2. **`NINFER_CUDA_SYNC` spin-wait default (v0.8.1).** The engine now defaults to
   spin-waiting on CUDA completion, overridable via env (`spin`/`blocking`/`yield`/
   `auto`). Cheap knob worth matching on our decode path; on WDDM in particular the
   wait mode interacts with TDR, which is probably why they made it configurable.
3. **`meta.n_ctx` in `/v1/models` (fork addition).** Served context ceiling
   advertised in the OpenAI dialect so stock WebUI/frontends auto-detect it —
   one-field serving change, useful for us on any OS.
4. **In-process stock llama.cpp WebUI** (`--webui` / `--webui-dir`, API dialect
   compat with upstream `tools/ui`) — a zero-cost UX bridge for non-technical
   users; trivially portable.
5. **v2→v3 offline artifact upgrade tool** (`tools/upgrade_ninfer_v2_to_v3.py`):
   preserves stored weight values/formats, installs the matching chat template, no
   re-download — the exact migration pattern for our next artifact-container bump.
6. **Mixed-format recipe system (upstream, documented here in detail).**
   Per-parameter `format`/`layout`/`method`/`source`/`activation_policy` assignment
   with pattern selectors, `import_encoded` for pre-quantized sources (NVFP4 code +
   scale + matrix weight divisor copied without requantization), `LogicalSource`
   hook for foreign quantizers, and auto/packed parent grouping for fused
   projections. The Qwen3.8-27B `nvfp4` profile is a concrete **mixed allocation**:
   NVFP4 MLP weights in Text layers 0–55, row-scaled FP8 for embeddings, attention
   in/out projections, GDN Q/K/V/Z + output projections, output head, remaining MLP
   weights (source: unsloth/Qwen3.8-27B-NVFP4). Note for the targets.md row: the
   registry has **no Q3** — nine formats: `bf16`/`fp32`/`int32`, `q4_g64_fp16`/
   `q5_g64_fp16`/`q6_g64_fp16`/`q8_g32_fp16`, `nvfp4`, `fp8_e4m3fn_row_bf16`
   ([tensor-formats.md](https://github.com/natpate/ninfer-windows/blob/master/docs/maintainer/tensor-formats.md)).
   The closest low-bit point is `q4_g64_fp16` (4.25 bits/weight); anyone planning a
   24 GB next-Qwen recipe should think Q4/Q5, not Q3.
7. **E8-KV modes (`rk8v4`/`rk4v4`/`rk4v4-e8`/`rk2v4-e8`) via PR #1** — the same
   Conway-Sloane E8 lattice family the UDPSendToFailed 4090 fork ships; here it
   arrived as a community PR sourced from Neroued/ninfer#35 (contribution:
   alexandergwosdz). Cross-fork E8 diffusion confirms it as the community's chosen
   KV-compression primitive; our E8 work (see
   [ninfer-4090 dossier](ninfer-4090-udpsendtofailed.md)) stays the 4090-tuned
   reference.
8. **KDA Op + GDN two-stage chunked prefill (v0.9.1).** Upstream kernel work, but
   KDA is "the kernel foundation for KDA-based models" — relevant if the next dense
   Qwen changes its linear-attention flavor; track rather than port now.
9. **MSVC/TMA compat fixes** (device-pointer NVFP4 TMA descriptors, pair-row SwiGLU
   TMA epilogue, v0.9.1 constexpr kernel-alias lambda capture): only relevant if we
   ever build on Windows/MSVC; they are the known-fail list for that path.

## Worth running beside NInfer/llama.cpp?

**No — it cannot run on our hardware** (build rejects everything but `sm_120a`), and
it is not a competing engine: it is our upstream with a Windows shell. Its value is
(1) the WDDM/runtime design reference above and (2) issue-trail tracking: #24 YaRN
(1M context demand), #25 "combine ninfer speed with the dynamic allocation of strata
for qwen 3.8 flash-next" (2026-10-06 — community asking for exactly the Strata-style
dynamic allocation on the next Qwen, cross-referenced in our Strata tracking), #23
Kaspersky false positive on v0.9.1. If a 4090/WDDM question does come up, the fair
test would be a head-to-head of *our* Windows build vs this fork's runtime layer on a
5090 (same artifact, same corpus), which is out of scope this week.

## Risk / unknowns

- **Blackwell-only, all numbers on a 5090.** Nothing here transfers numerically to
  sm_89: the MMA-template/sliced-k schedules, W4A4 TMA route, and saturation curves
  are Blackwell-tuned. Treat every figure as a same-family upper bound, not a 4090
  prediction.
- **Fork diff is thin; don't over-credit natpate.** Engine behavior (DFlash2, E8 KV,
  v3 format, KDA, GDN prefill, `NINFER_CUDA_SYNC`) is upstream's; the fork's own
  contribution is the Windows layer, webui/`n_ctx` glue, release packaging, and the
  MSVC fixes. Lineage confusion inflates the apparent independence of this project.
- **Validation gaps on the fork's own platform.** The 2026-09-06 Windows campaign
  (MSVC 14.44, CUDA 13.1, driver 616.56): 5 CPU + 15 GPU suites + Qwen3.8 NVFP4 real
  model passes (DFlash2 at 131,072 ctx/KV, 3.04 GiB free after startup) — but
  "groupwise-int model execution, eight concurrent requests and Linux execution were
  not tested in this campaign". No 4090 or C>1 Windows validation exists anywhere in
  the repo.
- **nvfp4full churn:** broken out of 0.8.0/0.8.1 (stay on 0.7.1), restored in v0.9.0
  only after weight-only NVFP4 DFlash2 landed — release-to-release capability is
  unstable on the minor releases.
- **AV flag on the prebuilt zip** (issue #23, Kaspersky false positive reported on
  v0.9.1): distribution-trust consideration if we ever ship a Windows zip the same
  way.
- **Q3 misconception in our targets row** — the registry has no Q3 format; the
  low-bit recipes are Q4–Q8 grouped, NVFP4, row-scaled FP8, BF16.

## New targets found

- **`cometkim/ninfer`** — child fork of natpate's Windows port (13 stars, 3 forks,
  Apache-2.0): `feat/mtp7` (MTP 1–7), `feat/hyperquant` (**HQ-E8-Rice-2B packed
  Main Text/MTP KV profile** — 2-bit E8 + Rice-coded residual KV), `feat/1m-context`
  (startup YaRN scaling, HQ execution envelope to **1,048,576 logical tokens** on one
  5090; "the whole context fits on my single 5090 (even 1M if I disabled vision or
  MTP), and up to 524k (2x scaling) is mostly fine to use" — PR #1 comment, 2026-08-24),
  NVFP4 DFlash2 integration, `nvfp4full`/`nvfp4qat` recipes (its artifacts are the
  `nvfp4full` profile in natpate's table), and `feat/kernel-perf` (runtime QK/RoPE and
  attention-gate fusion, PDL/GDN launch chains, Small-T cache routes, INT8 prompt
  split reduction). HyperQuant is
  [arXiv 2606.23406](https://arxiv.org/abs/2606.23406). The closest public
  "NInfer-family, small VRAM, 1M context" reference to the UDPSendToFailed 4090 work.

## Sources

- <https://github.com/natpate/ninfer-windows> — repo (fork of `Neroued/ninfer`;
  network root `Neroued/ninfer`), 167 stars / 19 forks / 3 open issues (2026-10-07),
  Apache-2.0
- <https://github.com/natpate/ninfer-windows/blob/master/README.md> — positioning,
  artifact table with SHA-256s, profile descriptions (nvfp4 mixed allocation), fork
  additions, 5090 performance/evaluation excerpts, requirements ("build rejects CUDA
  architectures other than `120a`"), launcher 150k/200k context policy, Docker/CLI/
  serve examples
- <https://github.com/natpate/ninfer-windows/blob/master/docs/windows.md> — Windows
  11 build/run guide, vcpkg manifest, static cudart, mmap + unbuffered overlapped
  I/O vs `O_DIRECT`, 2026-09-06 Windows validation campaign (MSVC 14.44, CUDA 13.1,
  driver 616.56, suite lists, explicit untested list), DFlash2 launch/verification
- <https://github.com/natpate/ninfer-windows/blob/master/docs/weight-conversion.md> —
  v2→v3 upgrade tool, official recipes table, format/method/activation-policy
  assignment, `import_encoded`, `LogicalSource`, fused-parent grouping
- <https://github.com/natpate/ninfer-windows/blob/master/docs/maintainer/tensor-formats.md> —
  nine-format registry (no Q3), NVFP4 E2M1/E4M3FN/divisor semantics, FP8 row-scale
  semantics, layout registry
- <https://github.com/natpate/ninfer-windows/blob/master/AGENTS.md> — `sm_120a` /
  RTX 5090 target, product boundary, ownership model (mirrors upstream)
- <https://github.com/natpate/ninfer-windows/blob/master/CMakeLists.txt> —
  `CMAKE_CUDA_ARCHITECTURES 120a` hard gate
- <https://github.com/natpate/ninfer-windows/releases> + release notes
  [v0.4.0](https://github.com/natpate/ninfer-windows/releases/tag/v0.4.0) (2026-08-24,
  E8 KV PR #1), [v0.5.0](https://github.com/natpate/ninfer-windows/releases/tag/v0.5.0)
  (2026-08-29), [v0.7.0](https://github.com/natpate/ninfer-windows/releases/tag/v0.7.0)
  (2026-09-07, DFlash2), [v0.8.0](https://github.com/natpate/ninfer-windows/releases/tag/v0.8.0)
  (2026-09-19, v3 format), [v0.8.1](https://github.com/natpate/ninfer-windows/releases/tag/v0.8.1)
  (2026-09-25, W4A4 TMA, `NINFER_CUDA_SYNC`), [v0.9.0](https://github.com/natpate/ninfer-windows/releases/tag/v0.9.0)
  (2026-09-25, NVFP4 DFlash2), [v0.9.1](https://github.com/natpate/ninfer-windows/releases/tag/v0.9.1)
  (2026-09-29, KDA, GDN two-stage prefill, unified MMA templates)
- <https://github.com/natpate/ninfer-windows/blob/master/docs/performance/methodology.md> —
  common 5090 serving profile, MTP/DFlash2 run labels, makespan/saturation methods
- <https://github.com/natpate/ninfer-windows/blob/master/docs/performance/qwen3.8-27b.md> —
  run records, context profiles, per-request scenario tables, corpus makespan, DFlash2
  completion outcomes and comparisons, reproduction commands
- <https://github.com/natpate/ninfer-windows/blob/master/model-cards/Qwen3.8-27B-NInfer/README.md> —
  v3 artifact manifest (1,190 objects, SHA-256), DFlash2 provenance (z-lab rev
  `50307d4c`), eval table vs official BF16, capabilities/limits
- <https://github.com/natpate/ninfer-windows/pull/1> — E8-KV community PR (source
  Neroued/ninfer#35, @alexandergwosdz; @cometkim 1M/HyperQuant comment)
- Issues: [#23](https://github.com/natpate/ninfer-windows/issues/23) (Kaspersky FP on
  v0.9.1), [#24](https://github.com/natpate/ninfer-windows/issues/24) (YaRN),
  [#25](https://github.com/natpate/ninfer-windows/issues/25) (Strata dynamic
  allocation for qwen 3.8 flash-next)
- <https://github.com/cometkim/ninfer> — child fork: feature-branch table (mtp7,
  hyperquant, 1m-context, dflash2, kernel-perf), roadmap docs, recipe lineage
