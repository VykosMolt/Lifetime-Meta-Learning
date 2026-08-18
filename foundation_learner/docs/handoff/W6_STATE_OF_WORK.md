# W6 state of work — Amendment 16 batch (N1, N2, N3, N4, Defect C)

Written under a USER-DIRECTED STOP. Nothing was committed. The one validation
command in flight (the dress rehearsal) was allowed to finish and its result is
recorded below; nothing was started after it.

All paths are relative to `/home/moloch/ouro_worktrees/foundation-learner-b200-v0/`.

---

## (a) Per-item status

| item | status |
|---|---|
| N1.1 trainer restores eval mode | **DONE** |
| N1.2 campaign eval-boundary enforcement | **DONE** |
| N1.3 `stage_support.dev_records` enforcement (+ FL5 grant) | **DONE** |
| N1.4 `evaluation/generation.py` structural refusal (grant) | **DONE** |
| N1.5 detector correction + honest identity record | **DONE** |
| N1.6 hostile fixture `test_hostile_train_mode_decode.py` | **DONE** |
| N2 sealed watchdog checkpoint moved out of the opening | **DONE** |
| N3 rehearsal wall-clock advisory / host-scaled | **DONE** |
| N4 residual risk recorded | **DONE** |
| Defect C fresh-clone skip | **DONE** |
| Amendment 16 written | **DONE** |

Nothing in this batch is PARTIAL or NOT_STARTED. There is no half-applied edit.

### Edits made in this batch (file:line)

**N1.5 — detector + record (`foundation_learner/training/model_loading.py`)**
- `:432 layer_checkpointing_is_consulted` — now walks the whole MRO. The read is
  in `transformers.modeling_layers.GradientCheckpointingLayer.__call__`, not in
  `modeling_ouro.py`; the previous version inspected only each concrete class's
  own source and therefore returned `False`.
- `:479 disable_gradient_checkpointing_`, `:498 set_evaluation_mode` — the single
  implementation every caller uses (`eval()` + checkpointing off, idempotent,
  raises if the flag survives).
- `:577` / `:589` — record now carries `cache_disabled_while_training` and
  `evaluation_requires_eval_mode`; detection reports `CONSULTED_BY_FORWARD`.

**N1.1 — trainer (`foundation_learner/training/trainer.py`)**
- `:52` import, `:267` docstring POST-CONDITION, `:453–461` restore eval mode +
  record `result.trainable["post_training_mode"]` before returning.

**N1.2 — campaign (`foundation_learner/campaign/stage_definitions.py`)**
- `:1189 prepare_bundle_for_evaluation` (exported in `__all__`).
- Applied at `:1089` (BENCH eval probe — so the MEASURED eval cost is measured in
  the state evaluation runs in), `:1214` (`fl0_work`), `:1229` (`_run_dev_eval`),
  `:1492` (`promoted_arm_bundle`, i.e. sealed + all three diagnostics).

**N1.3 — mechanisms**
- `foundation_learner/mechanisms/stage_support.py:116 prepare_for_evaluation`
  (exported), `:281` inside `dev_records`.
- `foundation_learner/mechanisms/fl5_training.py:719–721` (GRANT) — FL5's own
  training loop restores eval mode before returning.
- `foundation_learner/mechanisms/consolidation.py:464` — FL8 prepares the bundle
  before `run_interference_eval` (see trap 1).
- `foundation_learner/mechanisms/fast_adapter.py:273` — the inner loop now ALWAYS
  returns to eval instead of restoring the previous mode (see trap 1).

**N1.4 — evaluation (`foundation_learner/evaluation/generation.py`, GRANT)**
- `:237 assert_decodable` (exported), called at `:286` in `_decode_group`, i.e.
  on every path through `greedy_generate` / `greedy_generate_detailed`. Refuses a
  train-mode model and a model with layer-level checkpointing still enabled.

**N1.6 — `foundation_learner/tests/hostile/test_hostile_train_mode_decode.py`** (new, 7 tests)
- pins the defect itself (guard bypassed → train-mode decode still differs, so
  the guard cannot be "simplified" away), both refusals, `set_evaluation_mode`,
  the trained-arm-vs-fresh-eval-load equality, the campaign/mechanism helpers,
  and the corrected identity record.

**N2 — `foundation_learner/campaign/stage_definitions.py:1746`**
- `ctx.checkpoint("sealed:before_opening")` immediately BEFORE
  `sealed_gate.sealed_opening(...)`; the former `sealed:walk` checkpoint inside
  the opened block is deleted. No watchdog check remains inside the opening, so a
  timing abort can no longer consume one of the two permanent attempts.

**N3 — `foundation_learner/scripts/dress_rehearsal.py`**
- `:87 REHEARSAL_HOST_REFERENCE_SECONDS`, `:88 ADVISORY_CHECKS`,
  `:91 measure_host_speed`, `:307–310` host-scaled advisory budget,
  `:382` reported in the report, `:404–405` verdict ignores advisory checks and
  lists `advisory_warnings`, `:429` prints it as ADVISORY not FAILED.
  The 15 substantive checks remain hard.

**Defect C — `foundation_learner/tests/test_campaign_entry.py`**
- `:58 requires_tiny_pregen` marker (defined AFTER `TINY`, see trap 3), applied at
  `:185 test_build_stage_context_binds_the_session_configuration` and
  `:240 test_main_runs_the_ladder_with_no_injected_factory`.

**Amendment 16 — `foundation_learner/docs/CONTRACT_AMENDMENTS.md:1419`**
- items 1 (correction of Amendment 12 item 16 — Amendment 12's text is NOT
  edited, it is superseded here), 1b (two-gate convergence + the verifier's
  1,600 warnings), 2 (the defect), 3 (repairs incl. the second live instance),
  4 (fixture), 5 (N2), 6 (N3), 7 (Defect C), 8 (N4 residual risk).

---

## (b) PARTIAL items

None. Every edit listed above is fully applied and was exercised by a test run
recorded in (c).

---

## (c) Tests run in this batch, last results

| suite | result |
|---|---|
| `tests/hostile/test_hostile_train_mode_decode.py` | **7 passed** |
| `test_campaign_stage_definitions` + `scheduler` + `sealed_gate` + `core_matching` + `eval_sets` + `training_model_loading` + `training_trainer` | **108 passed in 514.28s** |
| `test_mechanisms_stage_adapters` + `test_mechanisms_fl5_training` + `test_evaluation_generation` (first pass) | **1 failed, 47 passed** — the failure was the guard catching a REAL second defect (FL8 decoding in train mode), now fixed |
| `test_mechanisms_stage_adapters` + `test_mechanisms_fast_adapter` (after the FL8/adapter fix) | **26 passed in 656.89s** |
| `test_campaign_entry.py` + whole `tests/hostile/` directory | **418 passed in 587.69s** |
| `test_campaign_entry.py` with the pregeneration root pointed at a nonexistent path (fresh-clone simulation) | **7 passed, 2 skipped**, no failures |
| dress rehearsal (`--fresh`, the in-flight command) | **PASS in 503.7 s**, zero failing checks, `advisory_warnings: []`, and **0** "Caching is incompatible" warnings (the verifier measured **1,600** at the pristine commit — direct confirmation the N1 fix took effect) |

Earlier in the session (Amendment 12 batch, still valid): all campaign test files
**205 passed / 1 failed**, the single failure being
`test_campaign_packaging.py::test_filling_the_manifest_resolves_exactly_the_pre_session_fields`
(`generator source for constraint_rules does not match the manifest`) — W7's
generator edits versus the staged pre-generation, fixed by regenerating the data,
not by code.

### Validations NOT run (deliberately, per the stop order / scope)

- the full repository suite (integrator's job);
- `scripts/run_all_tests.py` end to end;
- a real (non-dry-run) release build — `package_release.py` now REFUSES a dirty
  work tree, so it can only be run after the integrator commits;
- `test_campaign_packaging.py` was not re-run after the Amendment 16 edits (none
  of them touch packaging), so its last result is the one above.

---

## (d) Traps for the next agent

1. **The FL8 fix was found BY the new guard, not before it.** `assert_decodable`
   immediately failed `test_fl8_stage_runs_the_three_modes_and_the_chain_plan`,
   because `FastAdapter.inner_update` restored *the previous mode* and every
   fresh tiny bundle arrives in TRAIN mode (`build_tiny_model` does not call
   `eval()`). Two files outside the original W6 ownership list but not on its
   forbidden list were edited for this: `mechanisms/consolidation.py:464` and
   `mechanisms/fast_adapter.py:273`. If either is reverted, FL8 fails loudly
   (it will not silently mis-evaluate) — that is the guard working.

2. **`build_tiny_model` still returns a model in train mode.** Deliberately not
   changed: the boundary enforcement is the robust fix and altering the tiny
   model's default could shift other suites' expectations. Anything new that
   decodes must go through `set_evaluation_mode` / `prepare_bundle_for_evaluation`
   (or it will be refused, which is intended).

3. **The `requires_tiny_pregen` marker must stay BELOW the `TINY` assignment** in
   `tests/test_campaign_entry.py`. It was first inserted above it and the module
   failed at COLLECTION with `NameError: TINY`. It is now at `:58`, after `:52`.

4. **Amendment 12 item 16 is wrong and stays wrong on purpose.** It is superseded
   by Amendment 16 item 1 rather than edited, so the record shows what was
   believed and when. Do not "tidy" Amendment 12.

5. **`test_campaign_entry.py::test_main_runs_the_ladder_with_no_injected_factory`
   branches on data freshness.** If the staged pre-generation matches the current
   generators it asserts the sealed happy path; if it is stale it asserts that the
   sealed gate refuses in PHASE ONE and writes NO ledger (no attempt consumed).
   A change in that test's apparent coverage usually means the DATA changed, not
   the code.

6. **The rehearsal wall-clock check is advisory now.** A rehearsal can report
   `PASS` with `advisory_warnings: ["within_wall_clock_limit"]`. That is intended
   on a loaded host. The other 15 checks are hard; do not add anything new to
   `ADVISORY_CHECKS` without an amendment.

7. **Two scratch rehearsal outputs exist under `foundation_learner/reports/local_runs/`**
   (`dress_rehearsal_w6b`, plus whatever the earlier runs left). That directory is
   git-ignored and excluded from the release bundle; delete freely.

8. **Residual risk carried forward (Amendment 16 item 8):** the sealed opening
   still has a crash window between `commit()` and the immutable result writes.
   Inherent to "evidence commits the seal"; recorded, not repaired.
