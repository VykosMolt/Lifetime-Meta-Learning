# O1 oracle-reachability preregistration package — v2.1.0

O1 v2.0 is a new preregistered axis-bank design. It is not a repair of
v1.5.3. The failed v1 reconstruction and `AXIS_DIAGNOSIS_V2` remain immutable
predecessors.

The v2 primary bank has four L3_24 axes and eight signed actions:

1. `A1_READABLE`
2. `A2_WRITABLE`
3. `A3_CAUSAL_MEAN`
4. `A4_SEQUENCE_MEAN`

`A3_CAUSAL_MEAN` and `A4_SEQUENCE_MEAN` are separate,
reference-conditioned mean actuator directions. Each is the unit-RMS
uncentered arithmetic mean of its exact frozen `[8,2048]` induced-update
matrix. No centered PC is used, and the two directions are not averaged.

The original inadequate d2, the rejected shared-PC d4, and the separate
`D2_OUTCOME_AXIS_V2` diagnosis are excluded from the primary bank.

## Preserved experiment contract

v2 retains:

- K=8, four axes × two antipodal signs;
- common-random-number pairing and cyclic Latin-square stream assignment;
- oracle any-correct-of-8 endpoint;
- zero-alpha exact-parity gate;
- transport measurement as diagnostic only;
- exact McNemar power and budget derivation;
- calibration precommit and pre-generation seal;
- external chronology commitments and append-only attempt ledger;
- no calibration choice based on correctness except the preregistered
  difficulty-band and power calculations;
- no confirmatory outcome used for selection or repair.

## Axis construction

```bash
python axis_reconstruction.py \
  --a1-readable a1_readable_source_l3_24.npy \
  --a2-writable a2_writable_source_l3_24.npy \
  --causal-updates causal_induced_updates_l3_24.npy \
  --sequence-updates sequence_induced_updates_l3_24.npy \
  --reference-tasks actuator_reference_tasks.jsonl \
  --provenance AXIS_SOURCE_PROVENANCE.json \
  --output AXIS_PACKAGE_V2

python verify_axis_artifact.py AXIS_PACKAGE_V2
```

The axis verifier recomputes A3 and A4 directly from the embedded source
matrices, checks A1/A2 identity to embedded frozen source-axis bytes, checks
the exact eight-task order, regenerates the random bank, and verifies complete
hash coverage. It distinguishes structurally recomputed facts from upstream
capture or reconstruction facts that are only hash-bound or attested.

## Verification

Run from the package directory:

```bash
python run_all_tests.py
```

The suite verifies package checksums before importing code and disables
bytecode generation. `__pycache__`, `.pyc`, and `.pyo` are prohibited.

## Required real chronology

1. Build and verify the real v2 axis package.
2. Resolve every non-calibration runtime field and create frozen calibration
   task JSONL.
3. Create `CALIBRATION_PRECOMMIT.json` before any calibration record exists.
4. Push the precommit commit to an authenticated external remote, or stop.
5. Run calibration through the sealed orchestrator and derive results
   mechanically.
6. Materialize the confirmatory cohort with `build_confirmatory_cohort.py`,
   then build the final manifest, seed matrix, provenance, and
   `PREGENERATION_SEAL.json`.
7. Rerun all tests and verify every digest.
8. Push the pre-generation seal and append the attempt ledger.
9. Only then may confirmatory generation begin.

A local commit alone does not prove chronology against its operator.

The shipped runtime is `run_o1_v2_generation.py` (single sealed branch) driven
ONLY by the sealed orchestrator `run_o1_v2_orchestrator.py`, which owns arm
dispatch, the Latin square, seeds, record assembly, and the frozen transport
computation in `o1_transport_v2.py`. It binds physical layer 24 to
`model.model.layers[23]`, applies the edit only at zero-based loop 2 during
prefill, preserves a true identity path at alpha zero, and measures the
normalized L4_47 pre-decode boundary for paired transport. Frozen tasks are
built by `build_o1_v2_cohorts.py`; the calibration seed matrix uses
`build_seed_matrix.py --calibration`; the confirmatory cohort is materialized
only by `build_confirmatory_cohort.py` from the sealed pool and the
calibration-derived allocation (v2.1: zero post-calibration operator
discretion, G_confirm capped by eligible pool capacity).

## Calibration and confirmation commands

The v1.5.3 execution pipeline remains, with v2 axis semantics bound through the
manifest and package hash:

```bash
python calibration_precommit.py \
  --output CALIBRATION_PRECOMMIT.json \
  --manifest-design FREEZE_MANIFEST.precalibration.json \
  --artifact-paths RUNTIME_ARTIFACT_PATHS.json \
  --calibration-task-manifest calibration_tasks.jsonl \
  --records calibration_records.jsonl

python run_o1_v2_orchestrator.py calibration \
  --manifest-design FREEZE_MANIFEST.precalibration.json \
  --artifact-paths RUNTIME_ARTIFACT_PATHS.json \
  --precommit CALIBRATION_PRECOMMIT.json \
  --calibration-task-manifest calibration_tasks.jsonl \
  --axis-package AXIS_PACKAGE_V2 \
  --checkpoint <checkpoint_dir> \
  --output calibration_records.jsonl \
  --metadata-output CALIBRATION_METADATA.json

python calibration_analysis.py \
  calibration_records.jsonl \
  CALIBRATION_METADATA.json \
  FREEZE_MANIFEST.precalibration.json \
  calibration_tasks.jsonl \
  CALIBRATION_PRECOMMIT.json \
  confirmatory_candidate_pool.jsonl \
  CALIBRATION_RESULTS.json \
  --apply-manifest FREEZE_MANIFEST.json

python build_confirmatory_cohort.py \
  --manifest FREEZE_MANIFEST.json \
  --calibration-results CALIBRATION_RESULTS.json \
  --candidate-pool confirmatory_candidate_pool.jsonl \
  --output confirmatory_tasks.jsonl \
  --report COHORT_DERIVATION_REPORT.json

python build_seed_matrix.py \
  --manifest FREEZE_MANIFEST.json \
  --confirmatory-task-manifest confirmatory_tasks.jsonl \
  --output seed_matrix.json

python build_run_provenance.py \
  --manifest FREEZE_MANIFEST.json \
  --artifact-paths RUNTIME_ARTIFACT_PATHS.json \
  --output RUN_PROVENANCE.json

python pregeneration_seal.py \
  --output PREGENERATION_SEAL.json \
  --manifest FREEZE_MANIFEST.json \
  --provenance RUN_PROVENANCE.json \
  --confirmatory-task-manifest confirmatory_tasks.jsonl \
  --calibration-task-manifest calibration_tasks.jsonl \
  --calibration-precommit CALIBRATION_PRECOMMIT.json \
  --seed-matrix seed_matrix.json \
  --calibration-records calibration_records.jsonl \
  --calibration-metadata CALIBRATION_METADATA.json \
  --calibration-results CALIBRATION_RESULTS.json \
  --axis-package AXIS_PACKAGE_V2 \
  --candidate-pool confirmatory_candidate_pool.jsonl \
  --records confirmatory_records.jsonl
```

`run_o1_primary.py` additionally requires `--candidate-pool`; gate G23 rejects
any confirmatory task that is not a byte-identical member of the sealed pool.

The package is code and design infrastructure. A verified package is not an
empirical O1 result. Calibration is not confirmatory evidence, and a launched
confirmatory run is not a completed result.
