# Offline Verifier-Backed Branch-Generator v2 — Run State

OFFLINE_BRANCH_GENERATOR_INIT_VERDICT = READY

Goal: train a useful **branch generator** (raise positive_oracle@K) **offline-first** from the v1 verifier-backed substrate, before paying for online RL. Near-term target: `USE_BRANCH_TRAINED_MODEL_AS_GENERATOR` (better pools; DualAnchor still prunes; CoreContent_v2 still ranks). Self-selection beating CoreContent not required.

## Locked external baseline (unchanged unless heldout evidence demotes it)
- Branch survival: DualAnchor (MIX_CODE_REASONING + MIX_OBJECTIVE_ALL).
- Content/final selection: CoreContent_v2_blockwise.
- Terminal: top5/full survivor-set handoff; no unconditional terminal top1. Science: diagnostic only.
- Steering: claimed only if a trained write-path adapter wins on heldout. Tap/readout selection is NOT steering.

## Ground truth
- Correctness only from external verifiers (unit tests / exact / MCQ keys / parser / z3 / symbolic / rubrics / preference labels). DualAnchor = keep/prune/rescue/expand/defer policy teacher; CoreContent_v2 = terminal soft ranker. Neither gives correctness/reward/answer labels.

## Inputs (12/12 present)
- all required v1 inputs present.
- base model: True; prev SFT adapter: True.

## Stack
- torch: 2.12.0.dev20260407+cu128
- transformers: 4.54.1
- peft: 0.19.1
- trl: 0.20.0
- bitsandbytes: 0.49.2
- z3: 4.16.0
- datasets: 4.8.5

## GPU
- free 11453 MiB / util 5% / orphans 0.

## Operational
- STOP-file pausable (`artifacts/models/branch_training_offline_verifier_generator_v2/_control/STOP` or `STOP_<job>`).
- Resumable checkpoints + shard manifests; frequent logging; no cron unless asked; single active-pid source of truth.
- Online RL is GATED (Part M); not run here.
