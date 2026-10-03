# v3 catch-up for the RTX 4090 port (status, multi-night)

Tracking document for bringing upstream's v3 artifact format and runtime to `rtx4090-port` (sm_89, RTX 4090).
Draft until the port builds and passes on v3; expect several nights.

## Why

`rtx4090-port` reads only the **v2** `.ninfer` artifact (`8febce7e` pins the Qwen3.8 download to a v2 revision).
Upstream moved the converter, loader and engine to v3, so new official artifacts and upstream fixes no longer apply
here without this catch-up.

## Divergence (2026-10-03)

- merge-base `d4929686`; `origin/rtx4090-port` has 152 commits not in `upstream/master`, upstream has 83 not here
  (upstream head `d44ab584`, 2026-09-29). `git diff --stat` across the split: 1265 files, +83.7K / -77.0K.
- Upstream still targets RTX 5090 / `sm_120a` (native NVFP4, dflash). A wholesale merge is not appropriate; port
  selectively.

## The v3 chain upstream (what must come over)

| Upstream commit | What |
|---|---|
| `168fdd81` | converter: artifact production switched to v3 |
| `4cde7ad0` | loader: C++ weight loading switched to v3 |
| `04350ba9` | engine: v3 models run from bound instance parameters; runtime lives in `src/models/qwen3_5/` |
| `9b884034`, `cde57e48` | docs: v3 architecture references, artifact upgrade instructions |
| `b9219f3f`, `98dada0e`, `8eaed538` | frontend: llama.cpp jinja source base, custom jinja chat templates, literal content fix |

`rtx4090-port` has no `src/models/` tree; the 4090 runtime paths (Ada kernels, `rk4v4-e8` KV, MTP, vision) live
in the old layout and must be re-homed into `src/models/qwen3_5/`.

## Attempts and findings so far

1. **Assessment of upstream (2026-10-02).** Counted divergence, confirmed the 5090 targeting, decided against a
   wholesale merge. No code ported yet.
2. **Fork-side changes kept on v2 deliberately.** #1 (`--tolerant-tool-calls`, merged) and #2 (INT8 group-64
   activation prefill, +60-70% prefill on the dense body, perplexity +0.01%) were rebased onto `rtx4090-port` and
   tested on the v2 artifact only, because that is the only format this branch reads. Both need re-homing after v3.
3. **Reference port elsewhere.** `gzenz/ninfer` ported its monitor to V3 (`601215d3`, "ninfer-yarn") and keeps
   engine work on v3 branches (`engine/host-pool-and-planner-instruments`, `engine/cache-reuse-and-host-budget`).
   Useful as a map of which interfaces moved; not a 4090 port.
4. **Serving-side issue found while running v2 in production:** upstream issue #9 (engine latches "unavailable"
   after a 500 on the Anthropic path; workaround `--auto-long-anchors 0`, fix candidate PR #13). Re-check after v3,
   since the context cache moved.

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
