# FOUNDATION LEARNER V0 — PREREGISTRATION

Frozen: 2026-08-09, before any accelerator access, before any training, before
any sealed-test generation was readable. Companion authority for exact
engineering values: `docs/IMPLEMENTATION_CONTRACT.md` (+ amendments ledger).
Where this document and the contract state the same quantity, they agree; the
contract is the tie-breaker for engineering detail, this document for
scientific claims and analysis policy.

This is Foundation Learner **Pilot 0**. A null result is a useful result.

## 1. Research question

Can Ouro-RLTT 2.6B be trained over complete learning histories so that it
becomes better at **learning a previously unseen task family from attempts and
feedback**, rather than merely becoming better at the training tasks
themselves? The object of study is the within-episode learning trajectory
R_0..R_6, not static accuracy.

## 2. Backbone identity (frozen)

Single checkpoint for every arm: `models/ouro_rltt_local`, tree SHA-256
`a701f7a75300ddf57098572fef3894bef59d5179580ec7eae7cd561a36056889` (identical
byte-for-byte to the O1 v2.1 binding), OuroForCausalLM, 48 physical layers,
`total_ut_steps=4`, bf16, tokenizer.json SHA-256 `fcb808fe…c8fa8d`,
transformers exactly 4.54.1. No other model, no proxy model in any scientific
path. Every arm begins from a fresh load of this checkpoint. No O1 axis,
calibration result, or O1-mutated state enters training.

## 3. Task ecology and split (frozen)

Twelve computationally distinct exact-verifier generator families
(contract §4): boolean_rule, propositional_transform, modular_arithmetic,
sequence_transform, string_rewrite, finite_state_transducer,
permutation_composition, set_operations, graph_edge_semantics, dsl_execution,
constraint_rules, grammar_classification. Fresh latent rule per episode;
exact string-grammar verifiers; no LLM judge anywhere in principal outcomes.

Split rule (public, deterministic, frozen before computation):
`h(fid) = SHA-256("FOUNDATION_LEARNER_V0_FAMILY_SPLIT" ‖ 0x00 ‖ fid)`, sort
ascending by hex, positions 0–5 TRAIN, 6–8 DEVELOPMENT, 9–11 SEALED TEST.
Computed once (Amendment 2):

- TRAIN: grammar_classification, set_operations, dsl_execution,
  string_rewrite, finite_state_transducer, modular_arithmetic
- DEVELOPMENT: sequence_transform, graph_edge_semantics, boolean_rule
- SEALED TEST: constraint_rules, permutation_composition,
  propositional_transform

Zero family / exact-task / latent-rule overlap across splits (mechanically
enforced and hostile-tested). Sealed shards enciphered at generation; the only
decipher path is gated on a frozen development-decisions record and a
single-use opening ledger. Sealed outcomes are opened once, after all
development decisions freeze, and never justify further model modification.

## 4. Episode structure (frozen: EPISODE_STRUCTURE_V0)

One latent rule per episode. Interaction indices: 0 = attempt on P1,
1 = revision of P1 after feedback, 2 = attempt on P2, 3 = revision of P2,
4 = related problem P3, 5 = mean over queries Q1–Q3, 6 = mean over transfers
T1–T2 (K = 6). Feedback channels: certified `FEEDBACK` (always truthful;
correctness or certified structured), uncertified `HINT` (structured; may be
poisoned in poison conditions; rendering never distinguishes poisoned from
truthful hints), `REVEAL` (support answers only, only in declared supervised
conditions). Hidden query/transfer labels are never exposed. Training
histories are off-policy, from frozen scripted attempt policies (error rates
0.7 / 0.25 / 0.30; declared v0 design decision). Evaluation episodes are
online: real greedy generations, real verifier feedback. Plain-text protocol;
no chat template; strict final `ANSWER:` grammar.

## 5. Arms (frozen definitions; contract §7 for exact values)

- FL0 base evaluation (no training).
- FL1 static training on isolated task→answer pairs (no histories).
- FL2 imitation of complete successful histories (all model-span NLL).
- FL3 ordered-feedback meta-training: weighted NLL only on post-feedback
  competence spans — query 1.0, transfer 1.0, revisions 0.25, attempt-0 and
  all context 0. Stated plainly: this is direct query-loss optimization over
  learning histories; no "learning-progress reward" novelty is claimed.
- FL4 future-competence head: predicts realized Δ in query log-likelihood
  from including vs ablating a feedback item (targets from TRAIN families
  only, computed with the final FL3 checkpoint; ranking + MSE loss).
- FL5 persistent fast state: 1024-d GRU state over feedback events, injected
  as 8 norm-clamped prefix embeddings; FAST_STATE_ON vs FAST_STATE_OFF on the
  identical FL3 objective; survives textual context reset; reset at episode
  boundaries.
- FL6 value-gated adaptation: incorporate item iff FL4 prediction > 0
  (threshold frozen at 0), vs unconditional; tested under poisoned feedback.
- FL7 fast parameter adaptation: custom low-rank adapter (r=8, α=16, all
  attention q/v projections), 4 inner SGD steps (lr 1e-3) per revealed
  support item, global Frobenius clip 1.0, per-episode reset; ungated and
  value-gated variants. Runs only if predecessors justify it and time
  remains; never silently replaces FL5.
- FL8 consolidation: merge value-selected fast deltas into a slow adapter
  bank (scale 0.5); none vs indiscriminate vs value-gated; A→B→A retention,
  interference, unrelated-family degradation. Prepared, not required.

## 6. Primary comparison and headline rule

FL3 vs FL1 vs FL2 on whole-family-held-out learning curves. FL4–FL8 are
mechanistic extensions; an extension result can never replace a failed core
comparison in the headline. Unseen-instance and unseen-family generalization
are always reported separately; instance-level generalization is never
described as transferable learning. The principal Foundation Learner claim
requires improvement on the sealed whole-family holdout.

## 7. Metrics (frozen)

(1) macro-AULC over interaction indices 0–6, family-macro;
(2) ΔAULC vs FL1; (3) ΔAULC vs FL2; (4) R_0; (5) R_K; (6) improvement slope;
(7) interactions-to-threshold (0.5); (8) related-task transfer (R_4);
(9) whole-family transfer; (10) context-reset persistence; (11) A→B→A
retention/interference; (12) poisoned-feedback robustness; (13) surface-remap
robustness; (14) FL4/FL6 value ranking, calibration, top-choice regret vs
oracle/random/surface heuristics. Family-clustered bootstrap (families, then
episodes; 10,000 replicates; shared resample indices for paired differences).
No row-independent intervals; no single-family domination of aggregates.

## 8. Compute matching (frozen policy)

Core arms FL1/FL2/FL3: identical optimizer-update count U and identical
per-update token budget (FLOP-matched); loss-token totals differ by objective
construction and are reported explicitly (declared option B). Full per-arm
compute ledgers (updates, loss/forward/backward tokens, wall time, GPU time,
examples, episodes) are published. The same trainable-parameter scope
(PEFT_MODE or FULL_MODEL_MODE, contract §8) for all core arms — never mixed;
scope chosen by the mechanical affordability rule, never by outcomes.

## 9. Development grid and selection (frozen)

Grid: exactly 2 learning rates ({1e-4, 3e-4} PEFT / {1e-5, 3e-5} FULL) on
FL3 at 25% step count; one optimizer family (AdamW 0.9/0.95, wd 0.01), one
scheduler family (cosine, 3% warmup); one interaction horizon
(EPISODE_STRUCTURE_V0). Selection: higher DEV macro-AULC, tie → lower LR;
winner locked for FL1/FL2/FL3. TRAIN+DEV families only. No new candidates
after seeing dev performance. Root seed 20260809; second predeclared seed
20260810 only if affordable.

## 10. Promotion, fallback, scheduling (frozen)

Promotion rules, fallback work, and the time-aware scheduler are frozen in
contract §10–§11 (FL3→extensions gate: stability + DEV macro-AULC ≥ FL1 +
0.02 + positive slope evidence; FL6 needs FL4 dev pairwise ≥ 0.55; FL8 needs
nonzero persistence evidence; SEALED TEST never used for promotion; a failed
rung triggers predeclared fallback work only — no live objective invention;
safety factor 1.25; final transfer reserve 1200 s; checkpoint every 600 s or
200 steps). O1 calibration has first claim on the accelerator; FL runs only
after O1 records are verified, transferred, and the O1 process closed; FL
never reads O1 scientific outputs; USD 45 total budget authoritative; rental
confirmation NOT AUTHORIZED at freeze time.

## 11. Allowed conclusion vocabulary

NO_META_LEARNING_SIGNAL · STATIC_CAPABILITY_GAIN_ONLY ·
HISTORY_IMITATION_GAIN · WITHIN_EPISODE_LEARNING_GAIN ·
WHOLE_FAMILY_TRANSFER_GAIN · CONTEXT_ONLY_ADAPTATION ·
PERSISTENT_FAST_STATE_GAIN · LEARNING_VALUE_PREDICTIVE · VALUE_GATING_GAIN ·
FAST_PARAMETER_GAIN · CONSOLIDATION_GAIN · INTERFERENCE_FAILURE ·
POISON_ROBUSTNESS_FAILURE · INCONCLUSIVE_UNDER_COMPUTE_BUDGET.
Never "recursive self-improvement"; never "generally self-improving system".

## 12. Deliberately unresolved (B200-derived, mechanical only)

1. Measured B200 training/eval throughput (BENCH stage output).
2. U (updates per core arm) — largest of {600, 1200, 2400, 4800} passing the
   frozen affordability inequality.
3. FULL_MODEL_MODE vs PEFT_MODE — outcome of the frozen affordability rule.
4. `available_foundation_learner_seconds` — computed at session time after O1
   closes, from remaining authorized budget.
5. Evaluation batch size (post equivalence-gate) and resulting eval episode
   counts per stage (frozen per-stage maxima in stage_definitions).
6. `o1_entry_command` — operator-bound (the sealed O1 package's pod
   entrypoint is currently a refusing stub; recorded, out of FL scope).
7. Container registry digest reference — operator-bound until the GHCR
   push (see §13.11).

No conceptual decision is left open for the accelerator session.

## 13. Pre-run amendments after adversarial review (frozen 2026-08-09, before
any accelerator use, before any training run, sealed set still unopened)

An independent adversarial review and an independent verification of the
complete package produced findings that are repaired in code and/or recorded
here as claim-scope constraints. Everything in this section is frozen BEFORE
any experiment ran; no outcome data existed when it was written.

1. **Statistical power and claim scope (few clusters).** With 3 development
   and 3 sealed families, the frozen family-clustered bootstrap's nominal-95%
   intervals cover the *population* ("an unseen family in general") estimand
   at only ≈74–83% (simulated under realistic between-family spread), and a
   CI-excludes-0 rule has ≈17–26% type-I error. Therefore: all sealed
   results are reported with the number of family clusters, per-family
   effects, and a sign statement; intervals are explicitly conditional on
   the three specific sealed families; the allowed conclusion
   WHOLE_FAMILY_TRANSFER_GAIN is always qualified "on the three sealed
   holdout families" and never presented as a population-level unseen-family
   claim. The frozen +0.02 development promotion margin is acknowledged to
   sit near this noise floor; it gates extension *spending*, not scientific
   claims.
2. **Headline decomposition (binary answer-flip shortcut).** Binary-answer
   families admit a "repeat the other label after INCORRECT" heuristic worth
   far more than the promotion margin at indices 1 and 3. The headline
   ΔAULC is therefore always reported alongside (a) AULC restricted to
   interaction indices {4,5,6} (fresh items, flip-immune) and (b) a
   transcript-computed flip-attributable success rate. A positive headline
   not supported by the {4,5,6}-restricted contrast is reported as
   shortcut-suspect, not as meta-learning.
3. **FL0 format floor.** The base model is expected to frequently fail the
   strict `ANSWER:` grammar within 64 new tokens. `answer_line_rate` is
   reported per arm × family × interaction index; any macro cell with rate
   < 0.5 is flagged FORMAT_NONCOMPLIANT next to its value; persistence
   ratios with zero-valued denominators are reported UNDEFINED, never 0.
4. **Feedback information taxonomy.** Structured hints are legitimately
   informative — probabilistic evidence is the point of feedback. The
   defect class is *deterministic decodability* of the pending answer from
   hint features. THREE such defects were found and FIXED pre-run, before
   any training or sealed access: (a) graph_edge_semantics hint-NODE
   SELECTION encoded reachability (generator → 1.1.0, answer-independent
   selection); (b) constraint_rules hint CONTENT ("number of violated
   constraints": zero ⟺ SAT) decoded the label with P=1.0 — replaced by a
   candidate-constraint probe whose probed candidate need not be active in
   the hidden rule (generator → 1.1.0+); (c) grammar_classification hint
   content ("which conjunct fails": none ⟺ IN) likewise — replaced by a
   pool-predicate probe (generator → 1.1.0). A permanent power-asserted
   fixture (≈6,500 items/family) now forbids deterministic hint branches in
   ALL twelve families with an EMPTY allow-list, and proves its own
   non-vacuity by reconstructing each of the three defective 1.0.0 rules
   and requiring the violation to reappear. Measured probabilistic lifts
   are recorded in the fixture's report (e.g. boolean_rule pivot-ABSENT
   ≈0.89 vs 0.63 prior; modular_arithmetic residue-band hints reduce the
   candidate set to ≈3.5 after one hint and fully determine ≈48% of items
   after two). The repaired probes' status is computable from the displayed
   prompt; their value is that they point at and pre-evaluate a hypothesis
   from the family's own pool, so rule identification comes from
   accumulating (probe, status, verdict) evidence across rounds.
   Interpretation rule: R_1/R_3 gains may reflect hint exploitation; the
   leak-robust quantities are the {4,5,6}-restricted metrics and fresh-item
   transfer.
4b. **Label balance and constant-answer floors.** constraint_rules item
   sampling was measured 89.3% UNSAT — a constant-answer policy would score
   0.893 on that sealed family. Fixed pre-run: its sampler is conditioned
   to approximately balanced labels (target P(SAT) ∈ [0.4, 0.6]); no other
   family's distribution changed (boolean_rule's 0.63 majority rate is
   recorded and accepted). Every family-level result is reported against a
   per-family constant-answer baseline column so that no majority-class
   floor can be read as competence.
5. **FL2 comparison confounds.** FL2's successful-history data forces
   attempt-0 wrong ≈70% of the time and imitates it with weight 1.0, so FL2
   is trained to be wrong at R_0 by construction; FL2 and FL3 also train on
   different history variants (not merely different objectives). Metric 3
   (ΔAULC vs FL2) is therefore reported both including and excluding
   interaction index 0, and is described as measuring objective + data
   jointly.
6. **FL5 claim scope.** FL5 arms use a segmented pipeline and are NOT
   FL3-comparable; every FL5 result carries `comparable_to_fl3: false`. The
   evaluation triad is frozen as ON / OFF / ON_S0 (trained ON module with
   state pinned to zero). A persistent-fast-state claim requires ON >
   ON_S0 under context reset — separating the recurrent state's
   contribution from the learned static prefix; failing that, the result is
   reported as prefix-tuning-equivalent (CONTEXT_ONLY_ADAPTATION /
   PERSISTENT_FAST_STATE_GAIN not granted).
7. **Off-policy scripted histories — named threat to validity.** Training
   histories use hand-written plausible-error samplers; evaluation attempts
   are the model's own. The error-style distribution shift could mute or
   mimic treatment effects. Planned diagnostic (post-run, local): compare
   scripted vs realized attempt/error distributions on TRAIN families from
   FL0/FL3 transcripts; conclusions are restricted accordingly.
8. **Eval-time online context allowance.** Online evaluation contexts
   (scripted text + real generations) may exceed the 2048-token
   *data-generation* budget; the frozen eval-time allowance is 4096 tokens
   (model limit 65536), with per-episode overflow isolation (recorded and
   excluded, never silently dropped, never aborting the batch).
9. **Sealed-opening robustness.** The single sealed opening is two-phase:
   evaluation must produce records before the opening commits; an aborted
   attempt is permanently ledgered and permits exactly one retry. The
   sealed evaluation runs the development-selected promoted arm from its
   recorded checkpoint (never the untrained base), and requires the
   completed core comparison as an entry condition.
10. **Diagnostics.** Surface-remap, A→B→A interference, and
    poisoned-feedback diagnostics run unconditionally after the core
    comparison (metrics 9/11/12/13 are produced in V0), including under the
    FL3-null fallback.
11. **Seventh operator-bound unresolved field.** The B200 container
    registry digest reference remains unresolved until the operator's
    registry push (mirroring the O1 record); it joins the declared
    mechanical unresolved set.
