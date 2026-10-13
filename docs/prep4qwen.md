# prep4qwen: getting the 4090 engine ready for the next Qwen dense model

Started 2026-10-06. Nothing here is confirmed about Qwen's next model. The percentages are our guesses, written down
so we can score them on release day.

## Ground rules

- Sources are public only: Qwen's papers and blog, Hugging Face `config.json` drops, and architecture PRs to
  `transformers`, `llama.cpp` and `vllm`. Support for a new model usually lands in those projects a few days before the
  weights, and that is the earliest legitimate signal. No leaked weights or confidential material.
- Research jobs run on the local roles (`opencode run --agent local`) only when nobody is using NInfer, because the
  card serves one request at a time (`--max-concurrency 1`).
- Every engine change still goes through the v3 forward port first (`docs/v3-port.md`). This branch collects
  research and small prototypes. It does not replace that plan.

## Guesses about the next dense Qwen, with what each one would mean for us

| Guess | Our odds | If true, what pays off on the 4090 |
|---|---|---|
| Keeps the hybrid layout: Gated DeltaNet linear attention with a few full-attention layers | 80% | GDN prefill/decode kernels on sm_89 and chunked GDN prefill. Our current work carries straight over |
| Ships an MTP head | 75% | MTP3 spec decode and verify-window batching (Strata #646/#109, +6-10% decode) |
| Native context 256K or more | 60% | KV compression (`rk4v4-e8`), top-k sparse attention at 262K+ (Strata #603, +26% on a 243K prompt), prefix-cache reuse |
| Vision built in | 70% | Keep `--vision` in the v3 port's stage 3 gate |
| Size still fits 24 GB at about 4 bits (25-32B) | 55% | The dense path we have. INT8 prefill (#15) is the prefill lever |
| Bigger (35-45B dense) | 30% | Layer split into host RAM and streaming over PCIe. The 4090 is on PCIe 4.0 x16, so measure host-to-device first (Strata #44 probes it) |
| FP4/FP8 QAT weights as the main release | 40% | The 4090 has no FP4 tensor cores. A fast INT4/INT8 repack at load time matters more than a new kernel |
| Tool-call or template format changes | 50% | Keep the tolerant parser driven by the template, not hard-coded to today's tags |

## Ranked work (expected value = odds x gain, our judgement)

1. **Finish the v3 forward port.** Every later item lands on v3. Without it we chase a target that keeps moving.
2. **Long-context attention: top-k/QSA at 128K-262K.** It helps today's long agentic turns and any model with 256K
   context.
3. **Prefix-cache survival across compaction.** Re-enable long anchors once issue #9 is fixed (PR #13, or the v3
   cache rewrite). Look at Strata's shared-prefix LRU (#62) and conversation parking in host RAM (#752).
4. **Verify-window batching for MTP3.** Strata measured +6-10% decode with byte-identical output.
5. **A day-0 checklist:** converter path for a new `config.json`, the GDN/attention layer ratio, the RoPE/YaRN
   settings, the MTP head layout, the chat template.
6. **RAM offload/layer split** only if the size guess comes out bigger.

## Strata watch (upstream Niko1221/Strata; 15.6K stars on 2026-10-06; v0.1.40.1)

| Strata change | Applies to dense 27B on the 4090? |
|---|---|
| #646 / #109 verify pass with fewer launches, batched MTP steps | Yes, worth porting |
| #603 per-warp histogram top-k past 262K; #783 QSA early exit (+2.7% decode at 120K) | Yes, for long agentic turns |
| #62 / #8 conversation cache keeps the shared prefix, LRU for the rest; #752 parking | Yes, compaction and multi-session |
| #21 Q4_0 KV with FWHT-256 Hadamard rotation | Probably covered by `rk4v4-e8`; compare quality and speed |
| #583 byte-sized expert ring, `--prefill auto` | MoE expert streaming only. Applies only if Qwen4 needs offload |
| v0.1.40.1 rules for tool calls quoted inside thinking | Compare with our tolerant rules (PR #14) |
