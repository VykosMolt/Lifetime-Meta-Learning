# Logic Canonicalization (Part C)

LOGIC_CANONICALIZATION_VERDICT = LOGIC_VERIFIER_READY

Total logic tasks: 48536 (train 33247, heldout 11017) across 10 categories. Synthetic items are verified at generation (truth-table / finite-model / forward-chaining / z3); real items carry dataset answer keys. Official test splits forced to heldout.

## By category

| category | tasks |
| --- | --- |
| proofwriter_deduction | 13400 |
| synthetic_propositional | 10000 |
| mcq_logical_reading | 9802 |
| synthetic_fol | 5000 |
| logical_entailment | 3162 |
| synthetic_constraint_game | 3000 |
| ruletaker_deduction | 2621 |
| fol_entailment | 811 |
| lsat_logical_reasoning | 510 |
| lsat_analytical_reasoning | 230 |

## By split

| split | tasks |
| --- | --- |
| train | 33247 |
| val | 4272 |
| heldout | 11017 |
