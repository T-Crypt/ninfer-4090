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

## Artifact source (operator-provided 2026-10-13)

- https://huggingface.co/jabbatheduck/ninfer-ext-models — believed to hold
  the ext `.ninfer` model artifacts. Not yet inventoried (needs HF CLI file
  listing, no downloads). Next: list files/revisions per family, pin
  revision + SHA per artifact in the port spec before converting anything.
- The uncensored HauhauCS GGUF→HF→ninfer path (branch `ninfer-uncensored`,
  tool `gguf_to_hf`) is the template for any family whose ext artifact is
  NVFP4-only: repack at convert/load time to groupwise-int.

## v3 merge (done 2026-10-13, commit 42837d7c)

- Merged `port/v3-forward` (stage 4, 322a7f51) with `-X theirs`; 11
  modify/delete conflicts resolved to the v3 deletions after verifying each
  HEAD-side change was carried: slot save/restore, vision_max_tokens,
  auto-long-anchors, E8 KV, sm_89 gates, #6/#8 serve fixes, 1M visible-keys
  envelope, DFlash2 sm_89 shared-mem shape of the work.
- Two known follow-ups, not blockers: v3 reads native context from the
  artifact (no kNativeContext constant) — verify 262K+ loads at serve-test
  time; v3 redesigned the DFlash loader — verify the compact-artifact
  (no-DFlash-weights) case fails gracefully.
- Gotcha: repo `.gitignore` line 37 (`models/`, unanchored) ignores new
  files under `src/models/` — new family files need `git add -f`. Root
  `models/` (weights) is already covered by line 35 (`/models/`), so that
  line 37 is a candidate for removal on this line.

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

## Carried and merged (2026-10-13, branch `gemma4-26b`)

The Gemma 4 family is carried and merged into `ninfer-4090-ext` (fast-forward to `efdc6d5a`). It
builds on sm_89 and it runs: the 26B-A4B IT QAT artifact loads in 34 s at 131072 context, takes
17,454 MiB, answers coherently across turns, and returns a real tool call. Live through llama-swap on
aphotic as `Gemma4-26B-A4B-QAT` (alias `gemma4-26b`).

What came over: `src/models/gemma4/**`, the Gemma engine instance, the `q4_g32_fp16_rows` GEMM family
(`src/ops/linear/q4rows/`) with the `row_group_v1` layout, Gemma q5/q6 shapes, the Gemma MoE
router/experts wrappers, twelve new op headers, proportional RoPE, weightless and sigmoid-gated
RMSNorm, and the 5-argument embedding. NVFP4 routes sit behind `NINFER_ENABLE_NVFP4_FAMILY` (0 on
Ada) with a runtime throw, so no FP4 symbols exist in this build.

Most of the work was not the copy. The base already had the v3 merge applied (`42837d7c`) and mixed
pre-v3 fork code with v3 headers, so the bulk was repairing that fallout. Three things to know:

- The unanchored `models/` line in `.gitignore` silently ignored every new file under `src/models/`.
  It is gone; `/models/` on line 35 already covers the root weights directory.
- Session slot persistence is **stubbed**: `save_slot`/`restore_slot`/`erase_slot` throw, because the
  fork's snapshot code needs the deleted `src/targets/` runtime. Production passes no
  `--slot-save-path`, so nothing live touches it.
- Gemma needs `--kv-dtype bf16` and must NOT be given `--lm-head-draft` (it runs MTP on the full
  proposal head only). No prefix reuse, `top_k` capped at 20.

Not yet through: perplexity, long-context recall, vision, and an MTP acceptance number. The homelab
record is `reference/records/ninfer-gemma4.md`; the ticket is `08EA0Q9` (done) with the gates in
`EJYZAVT`.
