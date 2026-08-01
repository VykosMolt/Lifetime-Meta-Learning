# Canary Suite (Part C)

CANARY_SUITE_VERDICT = CANARY_READY
Total 610 groups; dropped 1458 leak/train-overlap tasks.

## Per domain (n, Wilson ±halfwidth @ p≈0.7)
- coding: 100 (±0.089)
- reasoning: 110 (±0.084)
- math: 110 (±0.084)
- logic: 180 (±0.067)
- alignment: 110 (±0.084)

## Logic families (9)
- fol_entailment: 20
- lsat_analytical_reasoning: 20
- lsat_logical_reasoning: 20
- mcq_logical_reading: 20
- proofwriter_deduction: 20
- ruletaker_deduction: 20
- synthetic_constraint_game: 20
- synthetic_fol: 20
- synthetic_propositional: 20

## K=8 hard subset: 108 groups (hard logic families + math sample).

## Slices
- logic: hard_logic vs logic_std; reasoning: direct_answer_candidate vs reasoning_std; math: math_exact; coding: coding_parse; alignment: preference-pair sentinel (no generation).

Disjointness: every canary prompt-hash excluded if present in any train/val split (logic + corecontent_v2). Stable canary_id + seed per task; K=4 default, K=8 hard subset.
