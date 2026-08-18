# Gate history — Foundation Learner B200 V0

Chronology of independent gates run on this package. Full texts of the two
confirmation reports are in this directory. All gate agents were independent
contexts with no access to the implementers' transcripts.

## Round 1 — adversarial review of commit 00adeba: REJECT

Five CRITICAL findings (C1 no production entry wiring; C2 no guaranteed
termination + broken crash-resume; C3 global eval cap collapsed every eval to
one family; C4 sealed eval ran the untrained base model; C5 sealed opening
ledgered before evidence existed — any exception burned the seal). Ten MAJOR
findings (M1 online token-budget overflow aborted whole batches; M2 FL0
format floor invisible; M3 graph_edge_semantics hint-node selection leaked
the answer deterministically; M4 binary answer-flip shortcut ≈5× the FL3
promotion margin; M5 FL5 OFF-arm impossible-task + static-prefix confounds;
M6 3-cluster bootstrap under-covers the population estimand; M7 metrics
9/11/12/13 had no producing stage; M8 stage projections wrong by 4–10×;
M9 eval-determinism call was dead code; M10 FL2 attempt-0 imitation handicap
+ data confound). Plus minors.

## Round 1 — independent verification of commit 00adeba: FAILED (packaging only)

12/14 acceptance items VERIFIED PASS with primary evidence (mutation-tested
hostile fixtures, bit-exact pregen/packaging/smoke reproductions,
independently reimplemented split + checkpoint tree hash, 10-step sealed-gate
adversarial probe, 14-probe isolation matrix). FAILED confined to release
packaging: SHA256SUMS coverage false on the committed tree; the zip omitted
the filled campaign manifest (build-order circularity); the zip contained
98 MB of non-reproducible smoke checkpoints, so the package hash was not
reproducible from a fresh clone. Also: cross-arm compute-matching asserts
existed but were never invoked in the production path (Gap 4).

## Repairs — commits 5eb48c5, 448d2fb, e27edd3 (Amendments 12–15)

All C/M findings and packaging defects repaired or honestly recorded;
preregistration §13 added (claim-scope constraints frozen pre-run); three
hint-leak repairs (graph selection channel; constraint_rules and
grammar_classification content channels → probe designs); constraint_rules
label balance (0.893 → ≈0.50) + constant-answer floor reporting; packaging
policy = git-tracked + regenerable pregen only.

## Round 2 — reviewer confirmation on e27edd3: all repairs FIXED; new CRITICAL N1 → REJECT

Every original finding verified FIXED or RECORDED_APPROPRIATELY by the
reviewer's own probes. New findings: N1 (CRITICAL) trained arms are
dev-evaluated in train() mode with gradient checkpointing active —
transformers' GradientCheckpointingLayer discards the KV cache in that state,
corrupting greedy decoding (proven by execution; see
REVIEWER_CONFIRMATION_N1.md); N2 (minor) a watchdog checkpoint inside the
sealed-opening block could consume a sealed attempt; N3 (note) rehearsal
wall-clock bound trips on loaded hosts; N4 (residual) crash window between
seal commit and immutable writes. Reviewer's stated expectation: ACCEPT WITH
RESIDUAL RISK once N1 (+N2 one-line move) is fixed.

## Round 2 — verifier confirmation on e27edd3: assigned repairs PASS; overall FAILED on the same defect

All packaging repairs verified with primary evidence, including independent
fresh-clone reproduction of zip 439fe915… byte-exactly. Independently
CONVERGED on the reviewer's N1 (its Defect A: 1600 'Caching is incompatible
with gradient checkpointing' warnings in one pristine-commit rehearsal) and
found Defect B (the layer-checkpointing detector misses the transformers base
class via MRO, writing a false record into every arm result) and Defect C
(minor: one test hard-fails instead of skipping in a fresh clone without
tiny pregen staged). See VERIFIER_CONFIRMATION.md.

## Round 3 — in progress at handoff

Fix batch for N1/N2/N3/N4-recording/Defect-C (Amendment 16) was being
implemented by a worker at handoff time; see W6_STATE_OF_WORK.md for exact
state. After the fix: full suite, repackage (zip hash changes), commit, push,
fresh-clone reproduction, and a focused independent confirmation review of
the N1 fix are required before ACCEPT.
