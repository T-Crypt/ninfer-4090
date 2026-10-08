# v3 catch-up for the RTX 4090 port (status, multi-night)

This doc tracks the port of upstream's v3 artifact format and runtime to `rtx4090-port` (sm_89, RTX 4090).
The PR stays a draft until the port builds and passes on v3. Plan on several nights.

## Why

`rtx4090-port` reads only the **v2** `.ninfer` artifact (`8febce7e` pins the Qwen3.8 download to a v2 revision).
Upstream moved the converter, loader and engine to v3. Until this branch catches up, we can't load new official
artifacts or take upstream fixes.

## Divergence (2026-10-03)

- merge-base `d4929686`; `origin/rtx4090-port` has 152 commits not in `upstream/master`, upstream has 83 not here
  (upstream head `d44ab584`, 2026-09-29). `git diff --stat` across the split: 1265 files, +83.7K / -77.0K.
- Upstream targets RTX 5090 / `sm_120a` (native NVFP4, dflash), so we port commit by commit instead of merging.

## The v3 chain upstream (what must come over)

| Upstream commit | What |
|---|---|
| `168fdd81` | converter: artifact production switched to v3 |
| `4cde7ad0` | loader: C++ weight loading switched to v3 |
| `04350ba9` | engine: v3 models run from bound instance parameters; runtime lives in `src/models/qwen3_5/` |
| `9b884034`, `cde57e48` | docs: v3 architecture references, artifact upgrade instructions |
| `b9219f3f`, `98dada0e`, `8eaed538` | frontend: llama.cpp jinja source base, custom jinja chat templates, literal content fix |

`rtx4090-port` has no `src/models/` tree. The 4090 runtime paths (Ada kernels, `rk4v4-e8` KV, MTP, vision) sit in
the old layout, and we have to move them into `src/models/qwen3_5/`.

## Attempts and findings so far

1. **Assessment of upstream (2026-10-02).** We counted the divergence, confirmed the 5090 targeting and ruled out a
   wholesale merge. No v3 code exists on this branch yet.
2. **Fork changes stay on v2 for now.** We rebased #1 (`--tolerant-tool-calls`, merged) and #2 (INT8 group-64
   activation prefill, +60-70% prefill on the dense body, perplexity +0.01%) onto `rtx4090-port` and tested both on
   the v2 artifact, the only format this branch reads. Both move to the new routes after v3.
3. **Reference port elsewhere.** `gzenz/ninfer` ported its monitor to V3 (`601215d3`, "ninfer-yarn") and keeps
   engine work on v3 branches (`engine/host-pool-and-planner-instruments`, `engine/cache-reuse-and-host-budget`).
   Those branches show which interfaces moved. They don't target the 4090.
4. **Serving bug we hit on v2 in production:** upstream issue #9. After a 500 on the Anthropic path the engine
   latches "unavailable". We run `--auto-long-anchors 0` as a workaround; PR #13 is the fix candidate. The context
   cache moved in v3, so we retest this after the port.

## Plan (one stage per night, each must build and pass before the next)

1. Converter + loader (`168fdd81`, `4cde7ad0`): produce/read a v3 artifact for Qwen3.8-27B; v2 path kept behind
   the artifact version until parity.
2. Engine re-home (`04350ba9`): move the 4090 runtime into `src/models/qwen3_5/`; Ada kernels, `rk4v4-e8`, MTP3,
   vision. Gate: same tokens as v2 on a greedy fixed-prompt set.
3. Frontend jinja (`b9219f3f`, `98dada0e`, `8eaed538`): Qwen tool-call template parity; re-apply tolerant tool calls.
4. Re-home INT8 dense prefill (#2) onto the new routes; re-measure prefill and perplexity.
5. Bench against the v2 build (prefill/decode t/s, 262K at `rk4v4-e8`, tool-call stress) before switching deploy.

## Done when

v3 Qwen3.8-27B artifact loads and serves on the 4090 at 262K with MTP and vision, the INT8 and tolerant-tool-call
changes are carried, and the bench shows no regression against `deploy/ninfer-serve-46645ada` (v2).

## 2026-10-05: upstream moved, and the plan flips direction

**What changed upstream since 2026-10-03.** Six commits landed on `upstream/master` (head `68c54356`). The big one is
`b9114396` "replace context cache and add preemptive scheduling": 176 files, +21.4K / -32.6K, almost all of it in
`src/models/qwen3_5` and `src/runtime/engine`. That is the tree stage 2 planned to move our runtime into, so the
target moved under the plan. The other five add Prometheus metrics (`abb7f14f`), TTFT benches (`f854788b`), prep
overhead cuts (`a8e212ac`), doc/bench alignment (`c772812b`) and a template-trimming cache fix (`68c54356`).
Upstream now leads us by 94 commits.

**Why the direction flips.** The three v3 commits alone touch 884 files (`168fdd81` 289, `4cde7ad0` 362, `04350ba9`
233). Our whole 4090 delta since the merge-base is 208 files, +14.9K / -1.1K, and 90 commits once docs and merges
are set aside (`docs/v3-port-inventory.md`). Porting v3 back into the old layout means rewriting the larger side.
So we branch from upstream and carry our delta forward instead: new branch `port/v3-forward` from an upstream
commit, then the inventory, commit by commit.

**Two targets, in order.**

- **Target A: `d44ab584`** (2026-09-29), the last upstream commit before the cache rewrite. It has the v3 converter,
  loader, engine and jinja templates, and the old context cache our serving fixes were written against. Carry the
  inventory here first and get v3 serving on the 4090.
- **Target B: `68c54356`** (current head). Rebase A onto it. The cache rewrite replaces the planners that upstream
  issue #9 lives in (`fix/issue9-entitlement` on `4090-base` patches the old planner), so retest #9 here and drop
  `--auto-long-anchors 0` if the latch is gone. Expect our slot-spill and prefix-reuse changes to need a rewrite at
  this step, not a replay.

**Revised stages** (each builds and passes before the next):

1. `port/v3-forward` from `d44ab584`. Build for `sm_89` with no fork commits. Expect it to compile and refuse to
   start or run slowly: upstream targets `sm_120a` (native NVFP4, dflash). Record what breaks; that list is the
   real scope of the Ada work.
2. Carry the Ada kernel and target commits from the inventory (`src/ops`, `src/targets`, `src/core`) into the v3
   layout. Gate: the op tests in `tests/ops` pass on sm_89.
3. Produce a v3 Qwen3.8-27B artifact with upstream's converter, or pull an official v3 one, and serve it at 262K
   with `rk4v4-e8`, MTP3 and vision. Gate: greedy fixed-prompt tokens match the v2 deploy build (`46645ada`).
4. Carry serve-side commits (`src/serve`, tolerant tool calls from upstream PR #14, INT8 prefill from #15).
5. Target B rebase and the issue #9 retest.
6. Bench against `deploy/ninfer-serve-46645ada` before any deploy switch: prefill/decode t/s, 262K, tool-call stress.

## Stage 1 brief (for a local agent)

You are working in `~/ninfer-v3` (a worktree of `~/ninfer-4090`). Do not touch `~/ninfer-4090` itself: its detached
HEAD is the production build, and llama-swap runs `ninfer-serve` from it.

1. `git switch -c port/v3-forward d44ab584` in a **new** worktree (`git worktree add ~/wt/ninfer-KKCF9MR d44ab584
   -b port/v3-forward`), so this branch and the docs branch stay separate.
2. Configure and build Release for `CMAKE_CUDA_ARCHITECTURES=89` with apps, tests and benchmarks, `-j 4`. Never
   higher: a `-j16` CUDA build beside a resident model froze this machine on 2026-10-03.
3. Record every configure error, compile error and `#error`/arch guard in a new section of this doc, with file and
   line. Do not fix anything yet.
4. Then, for each inventory row that touches `src/ops` or `src/targets`, find the v3 path that replaced its file
   (`git log --follow` or a grep for the kernel name on `d44ab584`) and write it in the row's status as
   `maps to <path>` or `no v3 home`.
5. Stop there. Commit the doc changes on `port/v3-catchup` with a `docs(v3-port):` subject. Do not push, do not
   run the model, do not stop or restart llama-swap.

Done when: the build log summary and the `maps to` column are filled in, and nothing outside the two worktrees
changed.

## Stage 1 results (2026-10-08, aphotic)

Worktree `~/wt/ninfer-KKCF9MR` on `port/v3-forward` (branched from `d44ab584`). Release, `CMAKE_CUDA_ARCHITECTURES=89`,
apps + tests + benchmarks, `-j 4` (configure pass plus two build passes, the second with ninja `-k 0` to enumerate every
failure). Logs: `/tmp/opencode/kkcf9mr-{configure-89,configure-89b,build-89,build-89k}.log`. Break 1 blocks configure
outright, so the survey ran with the arch guard bypassed locally — a 4-line uncommitted edit (FATAL_ERROR → WARNING),
restored right after the run. Nothing committed on `port/v3-forward`; no push; no model run.

### Break 1 — configure rejects sm_89 outright

`CMakeLists.txt:9-12`: `message(FATAL_ERROR "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got '89'")`. Configure
exits 1 before CUDA compiler detection. Stage 2's first decision is how to relax this (option flag vs per-target arch
properties).

### Break 2 — 53 files in `src/ops` fail ptxas for sm_89 (the real Ada scope)

The C++/host side is **clean**: zero compile errors; the non-CUDA libraries build and link (artifact, media_decode,
product *, runtime_support, text, jinja, spdlog). All breakage is at PTX assembly: **53 source files, 200,146 ptxas
error lines, all on `.target sm_89`** (172,026 name sm_89 explicitly; the rest are "requires .target sm_90 or higher").

Feature census (per error-line counts):

| Feature | Lines | Note |
|---|---|---|
| `mma with block scale` | 33,664 | fp8 `.kind::mxf8f6f4` + `.scale_vec::1X` (29,568); nvfp4 `.kind::mxf4nvf4` + `.scale_vec::4X` (4,096) |
| `cvt.bf16x2.e4m3x2` / `cvt.bf16x2.e2m1x2` | 21,692 / 13,656 | Blackwell pair-widening converts |
| TMA: `cp.async.bulk.tensor` + `.tile` + `.mbarrier::complete_tx::bytes` | 5,112 each | sm_90+ |
| `mul.bf16x2` (as emitted here) | 6,828 | reported "requires .target sm_90 or higher" |
| `cvt.e2m1x2.f32` | 1,992 | |
| `.cluster scope` + `.op_restrict` | 166 each | sm_90+ |
| `griddepcontrol` | 159 | sm_90+ |
| `setmaxnreg.inc` / `.dec` | 15 each | sm_90+ |

The 53 = 47 files in the `ninfer_ops` target + 6 in `ninfer_nvfp4_non_rdc`. By area: linear + shape-specialized 17,
attn_input_proj 7, linear_add 6, gdn_input_proj 6, linear_swiglu 5, dense causal-cache (fp8/k8v4/nvfp4) 6,
kv_cache/append 2, linear_topk 2, sparse_moe 2. Full list: appendix at the bottom of `docs/v3-port-inventory.md`.

Implications for stage 2:

1. Every fp8/nvfp4 A4/A8/A16 route needs an sm_89 fallback or gating (fork precedent: `25c782aa` gated NVFP4 A4 tests
   behind `NINFER_SM86`); the TMA kernels need Ada schedules or exclusion from the sm_89 build.
2. **The fork's own Ada kernels are not in the failure list.** `gqa_attention_*`, `e8_root_codec` / `e8_lattice`,
   `q4_q5_attn_input_int8.cu`, the rk4v4/e8 ops — none fail. Written for sm_89, they compile against the v3 tree
   as-is. The failing surface is upstream's Blackwell-first fp8/nvfp4/k8v4/TMA routes plus `sparse_moe`.

### Structure finding — `src/targets/` is gone on d44ab584

Upstream replaced the fork's `src/targets/` (`qwen3_6`, `qwen3_6_27b`, `qwen3_6_35b_a3b`, `registry.*`) with
`src/models/` (`registry.{cpp,h}`, `load_options.h`) and `src/models/qwen3_5/{load,execution,frontend,program,state}`.
Every fork commit touching `src/targets/**` is a rewrite into `src/models/qwen3_5/**`, not a replay — inventory
statuses mark these `targets/** → src/models/qwen3_5 (rewrite)`. `src/ops` and `src/core` keep their layout. Across
the 44 inventory rows that touch ops/targets/core (192 files): 39 same-path, 49 fork-added carries at the same path,
19 moved (mostly into `src/core/{device,host_worker_pool,host_kv_arena}.cpp`, `src/ops/kernel/sampling.cuh`,
`src/ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu`, `src/ops/candidate_selector/bf16/candidate_selector_path.cu`),
78 in the targets rewrite bucket, 6 with no v3 home (`ops/linear/w8/w8_config.h`,
`ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu`, `ops/softmax_attention/dense/causal_cache/small_t.cu`, plus 3
targets-tree files). The symbol-grep mapping is provisional; verify per file during the stage-2 replay.

### Also noted

- Upstream moved again during stage 1: head is now `81c8ce09` (fetched 2026-10-08), +33 commits past the `68c54356`
  Target B named above; lead over the fork is 127. Pin Target B to whatever upstream head is live when stage 4
  completes rather than chasing.
- Inventory header corrected: 155 = 90 non-docs/non-merge + 53 docs-only + **12 merges** (2 upstream, 10 internal);
  the old "2 are upstream merges" made 90+53+2=145.
- Untouched by design: the 46 inventory rows that only touch `src/serve`/`src/runtime`/apps (stage-4 mapping scope),
  the `~/ninfer-4090` production checkout, llama-swap. Nothing pushed.

## Stage 2 direction (2026-10-08, operator decision + failure taxonomy)

**Operator decision: NVFP4 does not port to sm_89 at all — fp8 is the floor on Ada.** No nvfp4 work; the sm_89 build
compiles it out. (It stays live in the upstream 120a build.)

Per-file attribution of the 53 ptxas failures (features counted per file from the `-k 0` log):

| Bucket | Files | What breaks | Stage-2 action |
|---|---|---|---|
| NVFP4 family | 22 | `.kind::mxf4nvf4`, `.scale_vec::4X`, `cvt.*e2m1*` | Compile out of the sm_89 build |
| FP8 block-scale MMA | 13 | `mma with block scale .kind::mxf8f6f4` + `.scale_vec::1X` | Rework to sm_89 non-block-scale fp8 mma, scales in software. Includes the 2 k8v4 causal-cache files (9,728 + 8,192 error lines) |
| FP8 A16 + topk (convert-only) | 8 | `cvt.bf16x2.e4m3x2` only | Mechanical: sm_89 pair-widen outputs f16x2, not bf16x2 — swap the convert or widen scalar |
| TMA bf16 gemm | 5 | `cp.async.bulk.tensor` + `.tile` + `mbarrier::complete_tx` | Ada schedule rework (cp.async staging) |
| griddepcontrol (PDL) | 4 | `griddepcontrol` in `q4_q5_gdn_input_*` + `sparse_moe` | Strip the dependent-launch hints (perf-only loss) |
| k8v4 append | 1 | `cvt.e2m1x2.f32` x128 | Small convert rework |

NVFP4 family = the 14 `.kind::mxf4nvf4` files plus the 8 nvfp4 files that fail on `e2m1` converts / `.op_restrict` /
cluster scope only (`*_a16`, `nvfp4_launch`, `nvfp4_a4`, `nvfp4_linear_add_a16`, `nvfp4_linear_swiglu_small_t`,
causal-cache `nvfp4/launch` + `tiled_launch`).

Open questions for the stage-2 review:

1. **Gate strategy**: compile the nvfp4 sources out of the sm_89 build (CMake target split, like `NINFER_SM86`
   gating in `25c782aa`) vs keeping them but never dispatching. Compile-out is the honest option — they cannot work.
2. **fp8 rework scope**: which of the 13 block-scale files does the Ada runtime actually dispatch? If the 4090 keeps
   the fork's own i8 dense prefill + rk4v4-e8 KV, upstream's fp8/k8v4 routes may stay compiled-out at first and the
   rework only covers what the v3 artifact/loader forces (the 2 k8v4 causal-cache files are the likely
   load-bearing ones).
3. **Converter (feeds stage 3)**: the v3 Qwen3.8 artifact for the 4090 must be produced in Ada-compatible formats
   (q4/q5/i8, rk4v4-e8 KV) — never nvfp4. Confirm the v3 converter still supports those targets.
