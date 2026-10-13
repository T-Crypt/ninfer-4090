# Research: giveen/ninfer-ext (fork review, in progress)

Date: 2026-10-13. Branch: `ninfer-4090-ext` (cut from `rtx4090-port`).
Status: **scaffold landed** — research carried from `prep4qwen`, v3
carry-over verified from branch logs. The live-fork diff is still pending
(`git remote add` / `fetch` for giveen/ninfer-ext needs network from a
non-plan session); the review steps below are ready to run.

## What we believe (unverified, from the operator's description)

- Extends ninfer to other model families and larger support.
- Currently NVFP4-only — needs a port to the 4090 groupwise-int path.

## Review steps (once network is available)

1. `git remote add ext https://github.com/giveen/ninfer-ext.git && git fetch ext`
2. `git diff --stat rtx4090-port..ext/<default-branch>` — new `src/targets/*`,
   new identities, NVFP4-only ops, converter changes, container version.
3. `git diff --stat gzenz/master..ext/<default-branch>` — what is genuinely new
   vs upstream vs inherited from the 3090 base.
4. Record commit hashes + keep/skip verdicts in the table below.

## Keep / skip (to fill in)

| ext change | Verdict | 4090 port note |
|---|---|---|
| (pending fetch) | — | NVFP4 exec → load-time repack to groupwise-int; no FP4 tensor cores on sm_89 |

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
