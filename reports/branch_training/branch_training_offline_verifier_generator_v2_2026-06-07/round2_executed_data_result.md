# Round 2 (r2 arms on v3 executed data) — result note (2026-06-12)

ROUND2_VERDICT = EXECUTED_DATA_REPAIRS_WHERE_COVERED
K_CANARY_SELECTION_VERDICT = NO_ADAPTER_IMPROVES_ON_CANARY (combined, unchanged)

## Result (paired canary deltas vs base)

| arm | macro | math | coding | logic | reasoning | chars_vs_base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C_offline_verifier_dpo (r1) | -0.244 | -0.533 | -0.500 | -0.233 | +0.067 | 0.264 |
| **C_offline_verifier_dpo_r2** | **-0.052** | -0.200 | -0.067 | **0.000** | +0.033 | 0.401 |
| G_budget_plus_verifier_r2 | -0.254 | -0.600 | -0.200 | -0.300 | -0.133 | 0.159 |
| H_from_base_best_recipe_r2 | -0.169 | -0.433 | -0.133 | -0.133 | -0.100 | 0.187 |

## Reading

1. **The executed-data hypothesis is confirmed where data existed.** v3 views are 99.7% logic
   (rejection_sft: 13,888 logic / 7 math; the executed renderer covers only the 4 symbolic
   logic families, and cheap math pool "branches" are ~265-char problem restatements with no
   work). Logic: fully repaired (-0.233 -> 0.000). Math/coding had no worked data; the
   residual harm there is cross-domain format bleed (C_r2 still bare-answers math).
2. **Budget-as-text-style is net harmful regardless of data quality** (G_r2 worse than G).
   The budget decision should be a separate prediction head / external policy, not a
   generation-style training target.
3. Combined selection still `NO_ADAPTER_IMPROVES_ON_CANARY` -> the external
   DualAnchor + CoreContent_v2 baseline remains champion, consistent with v1's
   `KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE`.

## Implied round 3 (not started)

- Build worked MATH views from dataset rationales (GSM8K/hendrycks step-by-step solutions,
  verifier-checked) + the model-generated math pools (real 1400-token solutions from the
  reachability run); coding analog from canonical solutions with tests.
- Re-run the C-style arm (verifier-DPO, no budget loss) on logic+math+coding worked data.
- Keep budget out of the generation channel.
- Per v1's policy: converged M (teacher distillation) + N (verifier-reward RL) at scale
  remains the endgame; the offline rounds have now validated the data substrate for it.
