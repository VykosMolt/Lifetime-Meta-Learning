# Reviewer confirmation pass (commit e27edd3) — new findings N1–N4

Recorded verbatim-in-substance from the independent adversarial reviewer's
confirmation report, 2026-08-09/10. Original C1–C5 and M1–M10 were all
verified FIXED or RECORDED_APPROPRIATELY by the reviewer's own probes (not
the package's fixtures): re-ran its own e2e entry and supervisor-resume
probes, re-measured hint leakage across all 12 families (3600 items/family,
zero deterministic branches), ran an 8-case adversarial attack on the
two-phase sealed gate, and a full independent dress rehearsal.

## N1 — CRITICAL. Trained-arm evaluations decode in train() mode with
gradient checkpointing active; generations are corrupted; and the package
records the opposite as fact.

Mechanism: `training/trainer.py:497` sets `model.train()` and never restores
eval mode. `campaign/stage_definitions.py::_run_dev_eval` (and
`mechanisms/stage_support.dev_records`, and FL5's
`mechanisms/fl5_training.py` ~line 636) then evaluate that same bundle. The
checkpoint's `OuroDecoderLayer` inherits transformers 4.54's
`GradientCheckpointingLayer`, whose `__call__` does:
`if self.gradient_checkpointing and self.training: use_cache=False;
past_key_values=None` — silently discarding the manual greedy decoder's KV
cache.

Executable proof (tiny model): decode(train+gc) != decode(eval) token
sequences; identical again after `.eval()`. Mode spy at the real call site:
after `run_training_arm`, `model.training == True` during `_run_dev_eval`
decode calls. Independent corroboration (verifier, pristine-commit
rehearsal): 1600 occurrences of `Caching is incompatible with gradient
checkpointing in OuroDecoderLayer. Setting use_cache=False,
past_key_value=None.` — training forwards pass use_cache=False, so these
necessarily come from the decode loop.

Additional honesty defect (verifier Defect B): `model_loading.py`'s
`layer_checkpointing_is_consulted` inspects only
`inspect.getsource(type(module))` and misses the inherited base-class
consultation via the MRO, so `layer_level_checkpointing_active: False /
NOT_CONSULTED_BY_FORWARD` is written into every arm result and the manifest
— a false record. Amendment 12 item 16 asserts the wrong conclusion and must
be superseded (not edited) by Amendment 16.

Contract impact: FL1/FL2/FL3 and DEV_GRID development curves — LR selection,
DEV_DECISIONS_FROZEN.json, best-DEV checkpoint rule, the FL3→extensions
gate, ΔAULC vs FL1/FL2 — would all be computed from corrupted generations.
FL5 shares the pattern; FL7 handles modes correctly; SEALED_EVAL/diagnostics
use a fresh bundle and are probably clean — which is worse: the single
sealed shot would be spent on an arm chosen by corrupted development
evidence.

Required repair (reviewer-specified, small):
1. restore `.eval()` + `gradient_checkpointing_disable()` when
   `run_training_arm` returns;
2. enforce eval mode idempotently in `_run_dev_eval`,
   `stage_support.dev_records`, `promoted_arm_bundle`, and the FL5 eval path;
3. hard assert at the decode entry (`evaluation/generation.py`) that the
   model is not in training mode;
4. fix the detector to walk the MRO / account for the transformers base
   class, and rewrite the identity record honestly;
5. supersede Amendment 12.16 in Amendment 16;
6. add hostile fixture pinning: evaluation refuses a train-mode model, and
   post-training arms decode identically to a fresh eval-mode load of the
   same weights.

## N2 — MINOR. `stage_definitions.py:1721` places
`ctx.checkpoint("sealed:walk")` inside the `with sealed_gate.sealed_opening`
block, so a watchdog/timing abort consumes one of the two sealed attempts
for a purely temporal reason. Move it above the `with` (next to
`sealed:promoted_arm`).

## N3 — NOTE. The dress rehearsal's 1200 s wall-clock check fails spuriously
on loaded hosts (reviewer measured 1887.7 s under contention with all 15
substantive checks passing; verifier reproduced the same pattern). Make the
wall-clock bound advisory or machine-relative; keep the substantive checks
hard.

## N4 — NOTE (residual, record only). A crash in the ~1 s window between
`unlock.commit()` and the two immutable result writes consumes the seal with
records only in memory. Inherent to evidence-commits-the-seal; vastly
smaller than the pre-repair exposure. Record in Amendment 16 as residual
risk.

## Reviewer's closing position

"Once N1 is repaired (and N2's one-line move made), I would expect ACCEPT
WITH RESIDUAL RISK, the residuals being M6's few-cluster statistics, M10's
FL2 confound, the off-policy scripted-history threat (prereg §13.7), and N4
— all now recorded rather than hidden."

## Verifier confirmation Defect C (same round, minor, portability)

In a fresh clone without `artifacts_fl/pregen_tiny` staged,
`tests/test_campaign_entry.py::test_build_stage_context_binds_the_session_configuration`
hard-fails with EntryError while its ~26 siblings skip on the same
missing-data condition. Make it skip like its siblings. (After staging tiny
pregen the clone suite is 1646 passed / 8 skipped / 0 failed.)
