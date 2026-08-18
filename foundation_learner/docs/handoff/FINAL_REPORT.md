# Foundation Learner B200 V0 final build report

## Scope and claim boundary

This report closes the build, local validation, packaging, reproducibility, and
confirmation-review handoff. It does not report a B200 run, Foundation Learner
training result, development-selection result, promotion result, or sealed-test
result. The real-checkpoint smoke test and the tiny dress rehearsal are
nonscientific mechanics checks.

- Branch: `foundation-learner-b200-v0`
- Worktree: `/home/moloch/ouro_worktrees/foundation-learner-b200-v0`
- N-batch source commit: `f475ee8e7a743e38669e1161b3d92d59bcf65cd3`
- Tiny-smoke boundary repair and final packaged-source commit:
  `90fad00f77fae76cde22e43e47757b305754108e`
- Release-metadata commit: `37d50732af5504a04f8729eb11e9b65ae01d648f`
- Validation-report and pushed release-validation snapshot commit:
  `8025a5dd4a1cc1e293d3d7404766a9c7c9163fb4`
- This report and `FINAL_CONFIRMATION_REVIEW.md` are post-release handoff
  records intentionally left untracked. The frozen packaging policy includes
  every tracked `foundation_learner/**` file outside its explicit exclusions;
  committing these records after validation would change the release input set
  and invalidate the exact-coverage/reproduction claim. The pushed ref remains
  the validated snapshot above.
- Frozen backbone path: `/home/moloch/ouro_project/models/ouro_rltt_local`
- Frozen backbone tree SHA-256:
  `a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889`
- Backbone implementation: `OuroForCausalLM`, 48 layers,
  `total_ut_steps=4`, bf16; scientific paths pin `transformers==4.54.1`.

The software state is `PRE_RENTAL_BUILD`. No cloud resources or B200 hardware
were used, no rental was initiated, and spend was USD 0.00.

## Exact-verifier task ecology and frozen split

The twelve implemented families are:

1. `boolean_rule`: hidden bounded-DNF evaluation.
2. `propositional_transform`: hidden formula transformation.
3. `modular_arithmetic`: hidden polynomial evaluation modulo `m`.
4. `sequence_transform`: hidden sequence-operation composition.
5. `string_rewrite`: hidden rewrite productions.
6. `finite_state_transducer`: hidden Mealy-machine transduction.
7. `permutation_composition`: hidden permutation application/composition.
8. `set_operations`: hidden set-expression evaluation.
9. `graph_edge_semantics`: reachability under hidden edge semantics.
10. `dsl_execution`: hidden register-program execution.
11. `constraint_rules`: hidden constraint satisfaction.
12. `grammar_classification`: hidden regular-language membership.

The assignment is frozen and hash-bound:

- TRAIN: `grammar_classification`, `set_operations`, `dsl_execution`,
  `string_rewrite`, `finite_state_transducer`, `modular_arithmetic`.
- DEVELOPMENT: `sequence_transform`, `graph_edge_semantics`, `boolean_rule`.
- SEALED_TEST: `constraint_rules`, `permutation_composition`,
  `propositional_transform`.
- Split digest:
  `6cf330b47af9406edf8022564ea330f3f164fbaef22618710a11fa3b2f40d9b5`.

No family, task, or rule overlaps splits. SEALED_TEST is excluded from model
selection and promotion. SEALED_TEST plaintext and outcomes remain unopened.

## Episode and rung implementation

`EPISODE_STRUCTURE_V0` freezes `K=6` as the maximum interaction index
(indices 0-6), giving seven scored positions: attempt P1 at index 0, P1
revision at 1, P2 at 2, P2 revision at 3,
related P3 at 4, mean Q1-Q3 at 5, and mean T1-T2 at 6. Training histories are
scripted/off-policy with frozen error rates 0.70, 0.25, and 0.30. Evaluation is
online greedy generation with at most 64 new tokens and exact-verifier
feedback.

The implemented FL0-FL8 ladder is:

- FL0: untrained base DEVELOPMENT learning curves under structured feedback,
  correctness-only feedback, and context reset.
- FL1: isolated TASK-to-ANSWER examples with answer-token loss only; it has no
  history or feedback.
- FL2: complete successful-history imitation with NLL over attempt, revision,
  related, query, and transfer answer spans.
- FL3: ordered-feedback meta-training on imperfect histories with weighted
  answer-span NLL. The fixed index weights are `0:0.0`, `1:0.25`, `2:0.0`,
  `3:0.25`, `4:0.0`, `5:1.0`, `6:1.0`. This optimizes future competence; it is
  not a novelty or reward claim.
- FL4: a stop-gradient hidden-state MLP value head. At the final FL3
  checkpoint, target `y_j` is the mean across Q1-Q3 query items of each correct
  answer's mean per-token teacher-forced log-likelihood with feedback item `j`
  present, minus the same quantity with `j` ablated. Training targets are
  TRAIN-only; the objective combines ranking hinge loss and z-scored MSE.
  DEVELOPMENT target computation is permitted only for head evaluation;
  SEALED_TEST target computation is refused.
- FL5: a 1,024-dimensional GRU persistent state updated from feedback, hint,
  and reveal hidden states, then injected as eight norm-clamped prefix vectors.
  State resets between episodes and persists across textual context reset. Its
  required comparison is `ON` / `OFF` / `ON_S0`; `ON_S0` uses the trained ON
  module with state pinned to zero. The triad decomposes the state-update
  contribution (`ON - ON_S0`) from the learned static-prefix/extra-parameter
  contribution (`ON_S0 - OFF`). FL5 is not directly FL3-comparable because its
  segmented forward makes OFF an impossible-task control, ON trains about 25
  million unmatched parameters, and the zero-state prefix is itself learned.
- FL6: applies fast state/update only when the frozen FL4 prediction satisfies
  `v_hat_j > 0`; unconditional and value-gated variants are compared under
  poisoned feedback.
- FL7: resettable custom LoRA fast parameters with rank 8, alpha 16, dropout 0,
  on `q_proj` and `v_proj` in all 48 attention layers. Each episode uses four
  SGD steps at `1e-3` on revealed support answers, with Frobenius clip 1.0,
  episode reset, context-reset persistence, and ungated/gated variants.
- FL8: adapts on A then B, merges selected fast deltas into a separate slow bank
  at scale 0.5 under none/all/value-gated policies, clears fast state, and tests
  A, B, unrelated families, and A-to-B-to-A interference.

## Metrics and decision rules

The fourteen frozen metrics are:

1. macro-AULC;
2. delta-AULC versus FL1;
3. delta-AULC versus FL2;
4. `R_0`;
5. `R_K`;
6. improvement slope;
7. interactions to threshold 0.5;
8. related-task transfer `R_4`;
9. whole-family transfer;
10. context-reset persistence;
11. A-to-B-to-A retention/interference;
12. poisoned-feedback robustness;
13. surface-remap robustness;
14. FL4/FL6 ranking, calibration, and regret.

The four amendment-added quantities are:

15. `answer_line_rate`, finish-reason breakdown, and
    `FORMAT_NONCOMPLIANT` below 0.5;
16. fresh-item AULC restricted to indices 4, 5, and 6;
17. transcript-derived `flip_attributable_success_rate`;
18. per-family constant-answer floor and AULC above that floor.

Headline FL2 deltas retain their preregistered definition and additionally
report companions excluding index 0 and restricted to indices 4, 5, and 6.

Development selection uses exactly two FL3 learning rates at 25% of core
steps: PEFT `{1e-4, 3e-4}` or FULL `{1e-5, 3e-5}`. AdamW uses betas
`(0.9, 0.95)`, weight decay 0.01, cosine decay, and 3% warmup. Higher
DEVELOPMENT macro-AULC wins, with lower learning rate as the tie-breaker; the
winner is locked for FL1, FL2, and FL3.

FL1-FL3 use the same optimizer-update count `U`, the same per-update token
budget, and the same PEFT/FULL mode. Unequal loss-token totals are reported.
FULL is eligible only if a projected core using `U_min=600`, the grid, three
core arms, and evaluations, multiplied by 1.25 and with a 1,200-second reserve,
fits the remaining authorized Foundation Learner time. Otherwise all core arms
use PEFT. `U` is the largest affordable rung in `{600, 1200, 2400, 4800}`.

The current frozen stage order is BENCH, FL0, DEV_GRID, the FL1/FL2/FL3 core,
CORE_MATCHING, FL4, FL5, FL6, FL7, FL8, unconditional REMAP_DIAG /
INTERFERENCE_DIAG / POISON_DIAG, SECOND_SEED, then SEALED_EVAL last; O1 has
absolute priority before the FL allocation. A stage is admitted only when
`projected_seconds * 1.25 + remaining_reserve <= remaining_authorized`.
The stage watchdog aborts at
`max(projected * 1.25, 900 seconds)` or when remaining time reaches the final
1,200-second transfer reserve, records `STAGE_ABORTED_OVERRUN`, applies only
the predeclared fallback, and continues the ladder. It checks phase boundaries
and trainer steps, not an in-flight model call; the reserve has no floor and is
never consumed by a stage.

Promotions are DEVELOPMENT-only. FL3 requires stable training, macro-AULC at
least FL1 + 0.02, and a positive slope. FL4 requires FL3 completion and at
least 200 realized episodes that each carry at least two scoreable feedback
items. FL5 requires the completed core comparison and scheduler admission.
FL6 requires FL4 DEVELOPMENT ranking at least 0.55. FL7 requires scheduler
admission and a stable fast-update dress-rehearsal flag; it may run after a
null FL5, but is skipped when only its FL6-gated variant remains after FL4
failure. FL8 requires positive FL5 or FL7 DEVELOPMENT context-reset
persistence with a confidence interval excluding zero. SEALED_TEST never
promotes.

None of these hardware-derived selections, core runs, or promotions has been
executed.

## Leakage, hostile validation, and gate history

The exact verifier enforces the final `ANSWER:` grammar. Certified FEEDBACK is
truthful; HINT may be poisoned and is rendered without a poison indicator;
REVEAL is restricted to declared supervised conditions. O1 outputs are
isolated and cannot be Foundation Learner features, targets, task selection,
or training data. The sealed gate requires frozen DEVELOPMENT decisions and a
single-use ledger; procedural protection is not represented as cryptographic
unopenability.

All 32 hostile fixture classes were included in the full local gate. Grouped
exhaustively, they cover: split leakage, instance/family conflation,
task-ID collision, family-collapse, family domination, remap leakage, feedback
answer leakage, hint-selection leakage, and poison mislabeling; static-arm
history, imitation future-objective leakage, and FL3 mask reduction to static;
value-head future leakage and value-gate ground-truth use; fast-state episode
reset, cross-episode leakage, context-reset text leakage, and fast-adapter
reset/norm; consolidation, checkpoint selection, DEV selection, and promotion
attempts to consume sealed evidence; sealed plaintext exposure and double-open
lifecycle; unequal core compute and unequal base/checkpoint identity; O1 path
isolation, online-budget overflow, O1-before-FL supervisor ordering,
hard-reserve violation, and pod-termination behavior; plus mandatory refusal
of train-mode or checkpoint-enabled decoding. Synthetic two-phase sealed
lifecycle and immutable-result tests supplement those fixtures.

The public episode walker and all campaign/mechanism boundaries normalize
models before evaluation. Inherited gradient-checkpointing consultation is
detected through the actual MRO. The train-mode hostile fixture independently
demonstrated that bypassing the guard produces cache-incompatibility warnings
while the guarded/eval path produces none; the two independent defect
investigations had observed 1,600 such warnings before the N-batch repair.

The auditable gate chronology is retained in [GATE_HISTORY.md](GATE_HISTORY.md),
[VERIFIER_CONFIRMATION.md](VERIFIER_CONFIRMATION.md),
[REVIEWER_CONFIRMATION_N1.md](REVIEWER_CONFIRMATION_N1.md), and
[W6_STATE_OF_WORK.md](W6_STATE_OF_WORK.md):

- Round 1 rejected C1-C5 and M1-M10; the verifier separately failed release
  reproducibility/coverage and found a production compute-match invocation
  gap.
- Repairs were recorded through Amendments 12-15.
- Round 2 found those repairs present but rejected the N1 train-mode decoding
  defect; its verifier independently converged on N1 and Defects B/C.
- Amendment 16 was appended without rewriting prior amendments. It records the
  N1-N3 and Defect C repairs, the corroborating 1,600-warning evidence, and the
  still-open N4 crash window.
- The initial fresh confirmation review rejected a broken documented
  `local_smoke_test --tiny` boundary. Commit `90fad00...` normalized both tiny
  and real loaded bundles before generation and added a real, unmocked
  end-to-end tiny regression.
- The [final fresh Sol confirmation review](FINAL_CONFIRMATION_REVIEW.md)
  independently ran 109 high-risk
  tests, a 422.485-second rehearsal, tiny and real smoke probes, and structural
  N1-N4/package/provenance checks. Its explicit verdict was `ACCEPT`, with no
  substantive defect found.

The retained residuals are not hidden: M6 has few family clusters for broad
population inference; M10 retains an FL2 attempt-zero/data confound; scripted
training histories are off-policy relative to model-generated evaluation
histories; and N4 can consume a sealed opening if a crash occurs after the
`SEALED_OPENED` commit but before immutable result writes.

## Final local and release evidence

- Full post-package suite: 1,654 passed, 8 skipped, 0 failed, with 14 Python
  3.14/Torch deprecation warnings; aggregate duration 1,912.24 seconds.
- Dress rehearsal: PASS in 422.25 seconds; all 15 hard checks passed, the one
  wall-clock check remained advisory, and decode-path cache-incompatibility
  warning count was zero.
- Real frozen-checkpoint smoke: PASS, 10/10 checks, CUDA, 20.985 seconds,
  classified `NONSCIENTIFIC`; it performed one throwaway optimizer update and
  produced no scientific result. The final reviewer independently repeated it
  in 22.545 seconds with 10/10 checks.
- Release version: `0.1.0`.
- ZIP: `FOUNDATION_LEARNER_B200_V0.1.0.zip`, 506,897,443 bytes, 263 archive
  entries, SHA-256
  `99a66e20d81b43708edefd665f2f06a409e81aa11ca11907fe5b72cf97c22c7b`.
- `foundation_learner/SHA256SUMS`: 262/262 exact file coverage, SHA-256
  `4231b8a6464f239e57f052b7e765a17dd349a728411f2239a461144bdd7965a1`.
- At the release-validation and fresh-clone gate, and at final handoff, the
  pushed remote ref
  `refs/heads/foundation-learner-b200-v0` resolved exactly to
  `8025a5dd4a1cc1e293d3d7404766a9c7c9163fb4`.
- GitHub reported that `VykosMolt/ouro_project` moved to
  `VykosMolt/Hidden-State-Evaluator`; the configured old origin redirected and
  was deliberately left unchanged.
- Fresh shallow clone: `/tmp/foundation-learner-repro-3jGoDP`, exact
  `8025a5d...` release-validation snapshot. Full pregeneration produced 25,800
  episodes, 110,400 instances, 13,800 rules, and 60 generated files;
  verify-only passed, and a repeated generation rewrote zero files. Its ZIP,
  sidecar, and `SHA256SUMS` matched the release byte-for-byte, including the ZIP
  hash, size, and split digest above.

The campaign manifest contains exactly these seven unresolved B200/operator
fields, with no non-B200 unresolved field:

1. `available_foundation_learner_seconds`
2. `container_registry_digest_ref`
3. `eval_batch_size_post_equivalence_gate`
4. `measured_throughput_tokens_per_second`
5. `o1_entry_command`
6. `training_scope_selected`
7. `updates_per_core_arm_U`

FOUNDATION LEARNER V0 SOFTWARE: COMPLETE
EXACT-VERIFIER TASK ECOLOGY: COMPLETE
TRAIN/DEV/SEALED-TEST FAMILY SPLIT: SEALED
FL0–FL8 LADDER IMPLEMENTATION: COMPLETE
LOCAL HOSTILE VALIDATION: PASS
OURO-RLTT BACKBONE: HASH-BOUND
B200 HARDWARE EXECUTION: NOT STARTED
FOUNDATION LEARNER TRAINING: NOT STARTED
SEALED TEST OUTCOMES: UNOPENED
O1 CALIBRATION DATA USED BY FOUNDATION LEARNER: NO
CLOUD SPEND FOR THIS TASK: USD 0.00
