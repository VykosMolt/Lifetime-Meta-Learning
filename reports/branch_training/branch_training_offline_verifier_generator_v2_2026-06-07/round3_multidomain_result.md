# Round 3 (C_r3 on v4 multi-domain worked data) — result note (2026-06-12)

ROUND3_VERDICT = MATH_REPAIR_CONFIRMED_CODING_NEEDS_REASONED_CODE
K_CANARY_SELECTION_VERDICT = NO_ADAPTER_IMPROVES_ON_CANARY (all 9 arms, 3 rounds)

## C-lineage paired deltas vs base

| arm | macro | math | coding | logic | chars_vs_base |
| --- | ---: | ---: | ---: | ---: | ---: |
| C (r1, meta-text) | -0.244 | -0.533 | -0.500 | -0.233 | 0.264 |
| C_r2 (executed logic) | -0.052 | -0.200 | -0.067 | 0.000 | 0.401 |
| C_r3 (+rationale math, +canonical code) | -0.112 | -0.133 | -0.467 | 0.000 | 0.337 |

## Findings

1. **Math: monotone recovery across rounds (-0.53 -> -0.20 -> -0.13)** with genuinely worked
   generations (step-by-step arithmetic). GSM8K/Hendrycks rationale data works; more data /
   more steps plausibly closes the rest.
2. **Canonical-solution coding data backfired (-0.067 -> -0.467).** Code-only completions are
   the coding analog of bare answers: the model learned terse canonical style ("just write the
   function"), which yields confident wrong code on unseen problems. Base wins by deliberating
   (~2.3k chars) before coding. The correct coding substrate is REASONED code
   (deliberation + implementation), which exists only in the 60 gen-pool groups; producing it
   at scale requires GPU generation + unit-test labeling.
3. Logic repair is durable (0.000 again); reasoning/alignment unaffected.

## Generalized lesson (3 rounds)

The generator learns the *style* of its data with high fidelity; verifier-DPO transmits style
efficiently. "Worked" must mean **reasoning-bearing, per domain**: executed derivations
(logic), rationales (math), deliberation+code (coding). Any answer-shaped shortcut in the
data — strategy narration, bare finals, canonical code — reappears as degeneration.

## Policy

External DualAnchor + CoreContent_v2 remains champion (unchanged across 9 arms / 3 rounds).
The offline-bounded series ends here. Next investment: the converged M+N run, whose prep
begins with generating reasoned-coding data at scale (model generations, unit-test labeled) —
plus uncapping GSM8K/Hendrycks for math. v4 minus the canonical-coding rows is the SFT/DPO
substrate to carry forward.
