# v3 port inventory: fork commits to carry onto upstream

Every non-docs, non-merge commit on `rtx4090-port` since the merge-base `d4929686`, oldest first (90 of 155;
the other 53 are docs-only and 12 are merges — recount 2026-10-08: 2 upstream, 10 internal). Generated 2026-10-05 from `origin/rtx4090-port`.

Fill the `status` column as you go: `carried <new sha>`, `upstream has it`, `dropped: <reason>`, or `blocked: <what>`.
`files` counts every path the commit touches, tests included.

| fork commit | type | subject | files | status |
|---|---|---|---|---|
| `09a93fd4` | feat | feat(release): add concurrent RTX 3090 serving v0.4 | 29 | same x1; carry x1; moved x4 → `ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu`; NO V3 HOME x4: `ops/linear/w8/w8_config.h` … |
| `785d856b` | feat | feat(qwen3.8): add 27b SM86 runtime support | 5 | moved x3 → `core/device.cu`; targets/** → src/models/qwen3_5 (rewrite) x2 |
| `1351c196` | bench | bench(qwen3.8): add RTX 3090 concurrency sweeps | 3 |  |
| `6fed5d27` | test | test(artifact): cover legacy v1 compatibility | 1 |  |
| `e271cc83` | feat | feat(release): add Qwen3.8 RTX 3090 v0.5 | 2 |  |
| `127bd74c` | fix | fix(frontend): accept contiguous system instructions | 2 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `e733d039` | bench | bench(qwen3.8): qualify MTP K4 and K5 on SM86 | 3 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `8bdf1837` | feat | feat(vision): publish guarded 35B RTX 3090 profile | 4 |  |
| `2ae51915` | feat | feat(vision): publish Qwen3.8 RTX 3090 launcher | 3 |  |
| `0d2d4a5a` | build | build(linux): add RTX 3090 source build | 4 |  |
| `9b802a89` | feat | feat(scripts): add Linux release helpers | 13 |  |
| `a3cb1abc` | fix | fix(gdn): correct sm_86 cooperative-launch residency limits | 2 | same x1 (replay) |
| `1eef8d83` | build | build: retarget runtime to sm_89 (RTX 4090) | 3 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `25c782aa` | test | test: gate NVFP4 A4 cases behind NINFER_SM86 | 7 |  |
| `0f308358` | feat | feat(serve): report context_window in models payloads | 4 |  |
| `4277f14b` | feat | feat(serve): add Prometheus /metrics endpoint | 7 |  |
| `74abe217` | feat | feat(serve): add /slots and requests_processing gauges | 6 |  |
| `8f3e943c` | fix | fix(gdn): raise cooperative residency limits to sm_89 | 2 | same x1 (replay) |
| `7888feed` | bench | bench(swiglu): add sm_89 schedule variants sweep | 2 |  |
| `fd3f5d80` | perf | perf(swiglu): prefer WN32 for the large-T route on sm_89 | 1 | same x1 (replay) |
| `ce50e995` | perf | perf(gqa): retune the i8 prefill schedule for sm_89 | 2 | same x1; carry x1 |
| `b0893e79` | feat | feat(serve): retain session depth on idle /slots | 4 |  |
| `85f685a3` | fix | fix(serve): keep route-written bodies on 413 responses | 1 |  |
| `bde2765c` | fix | fix(qwen3_6): lift processor prompt cap for media requests | 1 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `5a08683d` | fix | fix(media): pad sws_scale destination stride | 1 |  |
| `d78df936` | fix | fix(serve): accept content-part arrays on OpenAI tool messages | 2 |  |
| `7c2ce91e` | feat | feat(kv): add experimental paged rk8v4 mode | 24 | same x1; carry x6 (e.g. `ops/kernel/gqa_attention_decode_i8.cuh`); moved x1 → `core/host_kv_arena.cpp`; targets/** → src/models/qwen3_5 (rewrite) x5 |
| `5db12fac` | feat | feat(ops): add rk4v4 rotated 4-bit key and value kv-cache quantization | 18 | same x1; carry x5 (e.g. `ops/kernel/gqa_attention_decode_i8.cuh`); moved x1 → `core/host_kv_arena.cpp`; targets/** → src/models/qwen3_5 (rewrite) x5 |
| `5df406b0` | feat | feat(ops): add E8 Conway-Sloane lattice KV cache quantization | 24 | same x1; carry x6 (e.g. `ops/kernel/e8_lattice.cuh`); moved x1 → `core/host_kv_arena.cpp`; targets/** → src/models/qwen3_5 (rewrite) x5 |
| `816fe3fe` | feat | feat(ops): add E8 cylinder-factorized 240-root codebook KV cache quantization | 21 | same x1; carry x6 (e.g. `ops/kernel/e8_root_codec.cuh`); moved x1 → `core/host_kv_arena.cpp`; targets/** → src/models/qwen3_5 (rewrite) x5 |
| `c3a6e5c4` | perf | perf(ops): accelerate rk2v4-e8 prefill and decode with warp-cooperative quantization and L1 lookup | 3 | carry x3 |
| `ec56f922` | feat | feat(ops): lift context limits and switch GQA decode to direct block table lookup | 5 | carry x3; targets/** → src/models/qwen3_5 (rewrite) x2 |
| `2619adf7` | fix | fix(serve): restore int8-group64 request-log kv name | 1 |  |
| `68e2d0be` | fix | fix(ops): prevent out-of-bounds column store on single-token W8 linear add | 1 | moved x1 → `ops/gdn_input_proj/q8/q8_gdn_input_gemm_splitk.cu` |
| `b6172f24` | feat | feat(serve): report vision modality in models payloads | 4 |  |
| `0c3d2bee` | feat | feat(vision): make vision scratchpad token capacity configurable via --vision-max-tokens | 9 | targets/** → src/models/qwen3_5 (rewrite) x2 |
| `73b42127` | fix | fix(frontend): wire processor vision budget to --vision-max-tokens | 5 | moved x2 → `core/host_worker_pool.cpp`; targets/** → src/models/qwen3_5 (rewrite) x3 |
| `694e01f0` | perf | perf(attention): split interior key blocks from causal boundary path | 3 | carry x3 |
| `0f95b32e` | feat | feat(serve): emit llama.cpp-compatible timings for proxy stats | 4 |  |
| `265011d9` | fix | fix(serve): default the final-chunk usage param for schema test callers | 1 |  |
| `beaeb70a` | feat | feat(serve): save and restore slot sessions to disk | 26 | same x2; moved x1 → `ops/kernel/sampling.cuh`; targets/** → src/models/qwen3_5 (rewrite) x6 |
| `8e478945` | feat | feat(serve): attribute slot sessions with digests | 15 | targets/** → src/models/qwen3_5 (rewrite) x4 |
| `1614ef54` | fix | fix(engine): prefer cheapest lane on prefix-reuse ties | 1 |  |
| `4a4fa92b` | fix | fix(serve): publish slot states instead of locking per read | 1 |  |
| `656b0df7` | feat | feat(serve): source llamacpp metrics from live engine totals | 6 |  |
| `3a2e7f07` | feat | feat(engine): add host turn-checkpoint ring | 10 | targets/** → src/models/qwen3_5 (rewrite) x8 |
| `cba2c1f8` | feat | feat(serve): wire --turn-checkpoints and /slots list | 5 |  |
| `2cbe488d` | test | test: cover fork goldens and checkpoint ring | 1 |  |
| `8093c640` | feat | feat(engine): auto-save evicted sessions to slot files | 11 |  |
| `bc569eb8` | fix | fix(kv): harden E8 codec, document half-coset, gate verifier on needle retrieval | 4 | carry x3 |
| `a0e03d37` | chore | chore(kv): drop unused kInvSqrt2 from the E8 root codec | 1 | carry x1 |
| `94830b3f` | test | test(kv): cover the production E8 root codec | 2 |  |
| `6e239351` | perf | perf(gdn): reduce the QK norm with an XOR butterfly | 1 | same x1 (replay) |
| `c4d09b61` | perf | perf(gdn): hold the value pack in every lane | 1 | same x1 (replay) |
| `77bca849` | feat | feat(telemetry): expose detailed state pool breakdown in CLI and server logs | 6 | same x1; targets/** → src/models/qwen3_5 (rewrite) x2 |
| `981b685e` | test | test(kv): skip the E8 oracle when the card is full | 2 |  |
| `6affed2e` | feat | feat(serve): accept enable_thinking and reasoning_effort under chat_template_kwargs | 3 |  |
| `ff925039` | fix | fix(docker): remove CUDA forward-compat libs so GeForce cards can run | 1 |  |
| `bf8c8d65` | feat | feat(runtime): port session slots to catalog | 17 | same x2; moved x1 → `ops/kernel/sampling.cuh`; targets/** → src/models/qwen3_5 (rewrite) x6 |
| `f7be3177` | fix | fix(ops): accept packed and E8 KV modes in append | 1 | same x1 (replay) |
| `edac7921` | feat | feat(runtime): persist checkpoints in slot snapshots | 2 | targets/** → src/models/qwen3_5 (rewrite) x2 |
| `481e0608` | test | test: run session-persistence E2E on Qwen3.8 artifact | 1 |  |
| `065f9bb1` | test | test(serve): guard the effort alias and models payload | 1 |  |
| `f84d7bb1` | refactor | refactor(serve): retire --turn-checkpoints | 4 |  |
| `df08f261` | test | test(ops): verify i8 keys in the full-cache append | 1 |  |
| `60764d66` | fix | fix(serve): report the engine's real state on /health | 6 |  |
| `7b134d2a` | feat | feat(serve): restate speculative and host timings on the request line | 10 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `59419263` | feat | feat(serve): report thinking accounting on the request line | 2 |  |
| `ba56a232` | fix | fix(runtime): unbind slot file with its session | 3 |  |
| `b98904d0` | feat | feat(runtime): report the reuse the planner was offered | 3 |  |
| `9f63c77b` | feat | feat(serve): place private long anchors automatically | 14 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `03a99aa1` | fix | fix(runtime): stop a stale session copy from clobbering its slot file | 8 |  |
| `5d57fed9` | fix | fix(gdn): sum the unsplit gating projection's K reduction pairwise | 1 | same x1 (replay) |
| `6f327f49` | fix | fix(attention): reduce fork int8 modes as bf16 partials after the merge | 2 | NO V3 HOME x1: `ops/softmax_attention/dense/causal_cache/small_t.cu` |
| `e565fe50` | kv | kv: thread compute stream through activate() so membership publish is stream-ordered | 2 | targets/** → src/models/qwen3_5 (rewrite) x2 |
| `02d0976d` | fix | fix(materialization): drop stale-plan requests instead of wedging the engine | 3 | targets/** → src/models/qwen3_5 (rewrite) x2 |
| `51bf3597` | feat | feat(serve): map neighbouring reasoning-effort tiers instead of rejecting | 1 |  |
| `87230fd9` | test | test(prefix-real): run shared-release-source on the groupwise artifact | 1 |  |
| `87189c68` | fix | fix(ops): fit the DFlash2 W8 routes into the sm_89 static shared-memory cap | 5 | moved x3 → `ops/candidate_selector/bf16/candidate_selector_path.cu`; NO V3 HOME x1: `ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu` |
| `55dd995e` | test | test(attention): keep the fork's unrotated int8 oracle after the harness rewrite | 1 |  |
| `25297d06` | fix | fix(serve): account requests before engine submission | 7 |  |
| `a2834f43` | fix | fix(serve): report client disconnect instead of response-render 500 on cancelled Anthropic streams | 1 |  |
| `8febce7e` | fix | fix(scripts): pin the Qwen3.8 artifact download to a v2 revision | 4 |  |
| `539ccdcd` | fix | fix(planner): count only exclusive checkpoints in the active entitlement | 1 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `2c046095` | fix | fix(runtime): keep the original invariant when a materialization abort fails | 1 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `81b68a20` | diag | diag(runtime): describe both sides of the active-entitlement mismatch | 1 | targets/** → src/models/qwen3_5 (rewrite) x1 |
| `328d9aa8` | fix | fix(frontend): apply vision cap per item | 7 | targets/** → src/models/qwen3_5 (rewrite) x4 |
| `d46e1e3c` | feat | feat(serve): opt-in tolerant recovery for Qwen tool calls (--tolerant-tool-calls) | 11 | targets/** → src/models/qwen3_5 (rewrite) x3 |
| `6e454dad` | perf | perf(ada): INT8 group-64 activation prefill for the Q4/Q5 dense body | 48 | same x21; carry x11 (e.g. `ops/attn_input_proj/q4_q5/q4_q5_attn_input_int8.cu`); targets/** → src/models/qwen3_5 (rewrite) x1 |
| `8e616981` | test | test(linear_add): graph replay at T=128 uses the case's policy | 1 |  |

## Appendix: sm_89 ptxas failures on d44ab584 (stage 1, 2026-10-08)

53 files fail PTX assembly for sm_89 (200,146 ptxas error lines; all Blackwell PTX). 47 in the `ninfer_ops` target,
6 in `ninfer_nvfp4_non_rdc`. Full feature census in `docs/v3-port.md` § Stage 1 results. Paths relative to `src/ops/`.

| cmake target | file |
|---|---|
| `ninfer_nvfp4_non_rdc` | `attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma.cu` |
| `ninfer_nvfp4_non_rdc` | `gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma.cu` |
| `ninfer_nvfp4_non_rdc` | `linear_add/nvfp4/nvfp4_linear_add_a4_tma.cu` |
| `ninfer_nvfp4_non_rdc` | `linear/nvfp4/nvfp4_a4_tma.cu` |
| `ninfer_nvfp4_non_rdc` | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma.cu` |
| `ninfer_nvfp4_non_rdc` | `softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cu` |
| `ninfer_ops` | `attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu` |
| `ninfer_ops` | `attn_input_proj/fp8/fp8_attn_input_a16_gemm.cu` |
| `ninfer_ops` | `attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu` |
| `ninfer_ops` | `attn_input_proj/fp8/fp8_attn_input_a8.cu` |
| `ninfer_ops` | `attn_input_proj/nvfp4/nvfp4_attn_input_a16.cu` |
| `ninfer_ops` | `attn_input_proj/nvfp4/nvfp4_attn_input_a4.cu` |
| `ninfer_ops` | `gdn_input_proj/fp8/fp8_gdn_input_a8.cu` |
| `ninfer_ops` | `gdn_input_proj/fp8/fp8_gdn_input_matrix.cu` |
| `ninfer_ops` | `gdn_input_proj/nvfp4/nvfp4_gdn_input_a16.cu` |
| `ninfer_ops` | `gdn_input_proj/nvfp4/nvfp4_gdn_input_a4.cu` |
| `ninfer_ops` | `gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu` |
| `ninfer_ops` | `gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu` |
| `ninfer_ops` | `kv_cache/append/k8v4_launch.cu` |
| `ninfer_ops` | `kv_cache/append/nvfp4_launch.cu` |
| `ninfer_ops` | `linear_add/bf16/bf16_linear_add_gemm_mma.cu` |
| `ninfer_ops` | `linear_add/fp8/fp8_linear_add_a16.cu` |
| `ninfer_ops` | `linear_add/fp8/fp8_linear_add_a8.cu` |
| `ninfer_ops` | `linear_add/nvfp4/nvfp4_linear_add_a16.cu` |
| `ninfer_ops` | `linear_add/nvfp4/nvfp4_linear_add_a4.cu` |
| `ninfer_ops` | `linear/bf16/shapes/n14336_k5120.cu` |
| `ninfer_ops` | `linear/bf16/shapes/n256_k5120.cu` |
| `ninfer_ops` | `linear/bf16/shapes/n5120_k6144.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n14336_k5120.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n16384_k5120.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n248320_k5120.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n34816_k5120.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n5120_k17408.cu` |
| `ninfer_ops` | `linear/fp8/shapes/n5120_k6144.cu` |
| `ninfer_ops` | `linear/nvfp4/nvfp4_a4.cu` |
| `ninfer_ops` | `linear/nvfp4/shapes/n14336_k5120.cu` |
| `ninfer_ops` | `linear/nvfp4/shapes/n16384_k5120.cu` |
| `ninfer_ops` | `linear/nvfp4/shapes/n34816_k5120.cu` |
| `ninfer_ops` | `linear/nvfp4/shapes/n5120_k17408.cu` |
| `ninfer_ops` | `linear/nvfp4/shapes/n5120_k6144.cu` |
| `ninfer_ops` | `linear_swiglu/fp8/fp8_linear_swiglu_a16.cu` |
| `ninfer_ops` | `linear_swiglu/fp8/fp8_linear_swiglu_a8.cu` |
| `ninfer_ops` | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4.cu` |
| `ninfer_ops` | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu` |
| `ninfer_ops` | `linear_topk/fp8.cu` |
| `ninfer_ops` | `linear_topk/fp8_m64.cu` |
| `ninfer_ops` | `softmax_attention/dense/causal_cache/fp8/launch.cu` |
| `ninfer_ops` | `softmax_attention/dense/causal_cache/fp8/tiled_launch.cu` |
| `ninfer_ops` | `softmax_attention/dense/causal_cache/k8v4/launch.cu` |
| `ninfer_ops` | `softmax_attention/dense/causal_cache/k8v4/tiled_launch.cu` |
| `ninfer_ops` | `softmax_attention/dense/causal_cache/nvfp4/launch.cu` |
| `ninfer_ops` | `sparse_moe/decode/sparse_moe_decode_kernels.cu` |
| `ninfer_ops` | `sparse_moe/small_t/sparse_moe_small_t_kernels.cu` |
