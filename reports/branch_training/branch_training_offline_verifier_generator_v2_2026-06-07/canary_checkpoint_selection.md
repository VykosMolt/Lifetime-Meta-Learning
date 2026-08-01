# Canary Checkpoint Selection (Part K)

K_CANARY_SELECTION_VERDICT = NO_ADAPTER_IMPROVES_ON_CANARY
Paired subset: seed 1234, 30/domain generation tasks + full alignment (223 total); identical tasks for base, prev_sft, and every arm. Deltas are arm - base.

| arm | complete | macro_delta | d_coding | d_reasoning | d_math | d_logic | d_alignment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_format_control | True | 0.0000 | -0.0333 | 0.0000 | 0.0333 | 0.0000 | 0.0000 |
| B_direct_budget_sft | True | -0.0600 | 0.0000 | 0.0000 | -0.3000 | 0.0000 | 0.0000 |
| C_offline_verifier_dpo | True | -0.2436 | -0.5000 | 0.0667 | -0.5334 | -0.2333 | -0.0182 |
| G_budget_plus_verifier | True | -0.1352 | -0.0667 | 0.0000 | -0.4667 | -0.1333 | -0.0091 |
| H_from_base_best_recipe | True | -0.2188 | -0.2333 | 0.0334 | -0.7000 | -0.1666 | -0.0273 |
| C_offline_verifier_dpo_r2 | True | -0.0521 | -0.0667 | 0.0334 | -0.2000 | 0.0000 | -0.0273 |
| C_offline_verifier_dpo_r3 | True | -0.1121 | -0.4667 | 0.0667 | -0.1334 | 0.0000 | -0.0273 |
| C_offline_verifier_dpo_r4 | True | -0.0588 | -0.0667 | -0.0666 | -0.0667 | -0.0666 | -0.0273 |
| G_budget_plus_verifier_r2 | True | -0.2539 | -0.2000 | -0.1333 | -0.6000 | -0.3000 | -0.0364 |
| H_from_base_best_recipe_r2 | True | -0.1691 | -0.1333 | -0.1000 | -0.4334 | -0.1333 | -0.0455 |

Selected for Part L heldout (<= 3): A_format_control, C_offline_verifier_dpo_r2, C_offline_verifier_dpo_r4.

Resumable per-arm; STOP-pausable; selection is idempotent and re-runnable as arms land.
