# Research: giveen/ninfer-ext (fork review, in progress)

Date: 2026-10-13. Branch: `ninfer-4090-ext` (cut from `rtx4090-port`).
Status: **scaffold landed** — research carried from `prep4qwen`, v3
carry-over verified from branch logs. The live-fork diff is still pending
(`git remote add` / `fetch` for giveen/ninfer-ext needs network from a
non-plan session); the review steps below are ready to run.

## Verified (fetched 2026-10-13, remote `ext`, 21 branches)

- `ext/master` @ `0d5788d7`. **No common ancestor with `rtx4090-port`**
  (`merge-base` fails) — different tree layout (`src/models/`, upstream-v3
  style, vs our `src/targets/` v2 line). Direct diff is 1905 files of noise;
  review family-by-family, not by diff.
- Registered architectures (`src/models/registry.h`): **Qwen3_5, Qwen3_5Moe,
  Qwen4Exp, Gemma4**. Families: Qwen3.8-Flash-Next (~180B MoE, stock NInfer
  cannot load it), Gemma4 26B-A4B QAT + 31B IT (text+vision), Qwen3.6-35B-A3B
  (groupwise-int AND nvfp4), Qwen3.8-27B groupwise-int.
- **NVFP4 confirmed Blackwell-only**: tensor-core A16 kernel for NVFP4 MLP
  projections. 4090 port = load-time repack to groupwise-int, no FP4 exec.
- **No sm_89 anywhere in `ext/master` `src/`** — 5090-only, as expected.
- Overlap with our research: `prep4qwen` already studied qwen4_exp
  (Qwen3.8-Flash-Next, `docs/research/qwen38-flash-next.md`) — read that
  first before touching ext's Qwen4Exp.
- Useful extras beyond families: EXL3 quantization docs, constrained
  decoding + tool calls, `tools/upgrade_ninfer_v2_to_v3.py` (compare with
  our v3-forward port), honest benchmark-vs-stock methodology.

## Keep / skip (first verdicts)

| ext change | Verdict | 4090 port note |
|---|---|---|
| Qwen4Exp / Flash-Next family | Keep, after v3-forward | Needs v3 loader; MoE 180B won't fit 24GB without offload — measure first |
| Gemma4 family (26B-A4B, 31B) | Keep, after v3-forward | New Variant per AGENTS.md; QAT weights repack to groupwise-int |
| Qwen3.6-35B-A3B nvfp4 | Repack only | No FP4 exec on sm_89; groupwise-int line already exists upstream |
| EXL3 / constrained decoding / tool calls | Evaluate | Small, possibly cherry-pickable independent of families |
| v2→v3 upgrade script | Compare | Our port/v3-forward may already cover; diff before adopting |

## v3 carry-over (verified 2026-10-13)

- `port/v3-forward` stage 4 green on sm_89: INT8 group-64 prefill (322a7f51),
  #9 entitlement fix (49deb207), auto-long-anchors (9f63c77b),
  tolerant tool calls (d46e1e3c), stage-3 gate (72a24f46).
- `port/v3-catchup` docs stages 3–4 recorded; draft PR sergiuszm/ninfer-4090#16.
- `ninfer-4090-ext` starts at `rtx4090-port` (v2-validated); the v3 forward port
  merges in as the next step, not in this cut.

## Research carried from prep4qwen (done 2026-10-13)

`docs/prep4qwen.md` + all 13 `docs/research/*` files checked out from
`prep4qwen` onto this branch, unmodified. This file is the only new doc.
