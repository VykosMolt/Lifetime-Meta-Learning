# O1 oracle-reachability preregistration package — v1.5.3

This package implements the sealed O1 oracle gate for the Ouro-RLTT writable action space. It separates three commitments:

1. **Calibration precommit** — freezes the calibration design and runtime artifacts before calibration generations exist.
2. **Pre-generation seal** — binds the completed manifest, executable calibration result, task manifests, deterministic seed matrix, runtime provenance, and axis package before confirmatory generations exist.
3. **Result seal** — opens the zero-alpha gate and primary endpoint in append-only order.

The code and synthetic integrity tests are verified. The actual experiment is still **UNSEALED** because the real checkpoint, four source-derived L3_24 axes, task cohorts, runtime artifact hashes, and calibration records are not included and have not been invented.

## Package contents

```text
PREREGISTRATION.md                 scientific contract and stopping logic
FREEZE_MANIFEST.template.json      v1.5 scientific manifest; remaining slots are real hashes or calibrated outputs
CALIBRATION_METADATA.template.json elapsed calibration time + precommit binding
RUNTIME_ARTIFACT_PATHS.template.json paths used to hash the actual runtime artifacts

o1_analysis.py                    confirmatory gates, exact inference, seals, and primary analysis
calibration_analysis.py            deterministic calibration derivation and final-manifest binding
calibration_precommit.py           creates CALIBRATION_PRECOMMIT.json before calibration generation
build_run_provenance.py            hashes actual runtime artifacts and builds RUN_PROVENANCE.json
build_seed_matrix.py               derives every confirmatory RNG stream deterministically
pregeneration_seal.py              preflights and creates PREGENERATION_SEAL.json
run_o1_primary.py                  command-line entry point for the confirmatory analysis

axis_reconstruction.py             deterministic assembler; requires the real frozen source axes
verify_axis_artifact.py            A0–A17 structural verifier and adversarial self-tests

test_o1_fixtures.py                53 confirmatory/integrity fixtures
test_calibration_fixtures.py       17 calibration/precommit fixtures
test_v15_integrity.py              13 cross-artifact, chronology, and package-integrity fixtures
test_cli_workflow.py               7 end-to-end command-line workflow fixtures
run_all_tests.py                   compiles modules and runs every shipped suite
AXIS_SOURCE_BUNDLE.template.json    required real pre-O1 source-artifact inventory
AXIS_RECONSTRUCTION_STATUS.md       current reconstruction boundary and missing binaries
```

## Verified test result

```text
53/53 confirmatory fixtures passed
17/17 calibration fixtures passed
13/13 v1.5.3 integration fixtures passed
7/7 end-to-end CLI workflow fixtures passed
15/15 axis-artifact adversarial checks passed
Exact McNemar planning minima: 324 / 817 / 2238 / 337
```

The distributed package must contain **no** `__pycache__`, `.pyc`, or `.pyo` artifacts. `run_all_tests.py` rejects them before imports and again after all suites; child processes run with bytecode generation disabled. This prevents timestamp/size-valid bytecode from shadowing the reviewed source.

Run all checks from either the nested development directory or a flattened extracted archive:

```bash
python run_all_tests.py
```

Individual suites:

```bash
python test_o1_fixtures.py
python test_calibration_fixtures.py
python test_v15_integrity.py
python verify_axis_artifact.py --selftest
```

## Required execution order

### 1. Reconstruct and verify the L3_24 axis package

Supply the frozen pre-O1 source directions and the two independently reconstructed d4 components to `axis_reconstruction.py`. The verifier requires the reconstruction script, executing verifier, all tensors, manifest, Gram matrix, attestation, and complete `SHA256SUMS` coverage.

```bash
python axis_reconstruction.py \
  --d1 d1_l3_24.npy \
  --d2 d2_l3_24.npy \
  --d3 d3_l3_24.npy \
  --d4-component1 d4_adapter1_l3_24.npy \
  --d4-component2 d4_adapter2_l3_24.npy \
  --provenance AXIS_SOURCE_PROVENANCE.json \
  --output AXIS_PACKAGE

python verify_axis_artifact.py AXIS_PACKAGE
```

A `SEALABLE` result proves the package's structural and hash claims. Claims about remote source runs remain attested unless the source artifacts or a hash-verified reconstruction log are supplied.

### 2. Create the calibration precommit

Fill all policy and runtime fields needed to generate calibration records, create the calibration task manifest, and point `RUNTIME_ARTIFACT_PATHS.json` at the real artifacts. The command refuses to overwrite an existing precommit or to run after the calibration-record path exists.

```bash
python calibration_precommit.py \
  --output CALIBRATION_PRECOMMIT.json \
  --manifest-design FREEZE_MANIFEST.precalibration.json \
  --artifact-paths RUNTIME_ARTIFACT_PATHS.json \
  --calibration-task-manifest calibration_tasks.jsonl \
  --records calibration_records.jsonl
```

Every calibration row and `CALIBRATION_METADATA.json` must carry the resulting precommit hash.

### 3. Derive calibration results mechanically

```bash
python calibration_analysis.py \
  calibration_records.jsonl \
  CALIBRATION_METADATA.json \
  FREEZE_MANIFEST.precalibration.json \
  calibration_tasks.jsonl \
  CALIBRATION_PRECOMMIT.json \
  CALIBRATION_RESULTS.json \
  --apply-manifest FREEZE_MANIFEST.json
```

This derives the difficulty settings, complete alpha pass/fail partition, `alpha_star`, optional secondary magnitude, bank-level discordance bound, throughput, optional-arm drop plan, exact power requirement, budget limit, confirmatory task count, and budget MDE. The final manifest is rejected unless those values reproduce exactly.

### 4. Build the confirmatory artifacts

Create the sealed confirmatory task manifest with task ID, stratum, generator setting, rank within stratum, and content hash. Then build the deterministic seed matrix and runtime provenance:

```bash
python build_seed_matrix.py \
  --manifest FREEZE_MANIFEST.json \
  --confirmatory-task-manifest confirmatory_tasks.jsonl \
  --output seed_matrix.json

python build_run_provenance.py \
  --manifest FREEZE_MANIFEST.json \
  --artifact-paths RUNTIME_ARTIFACT_PATHS.json \
  --output RUN_PROVENANCE.json
```

### 5. Create the pre-generation seal

This preflights the manifest, runtime configuration, axis package, calibration recomputation, task disjointness, and deterministic seed matrix. It refuses to run after the confirmatory-record path exists.

```bash
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
  --axis-package AXIS_PACKAGE \
  --records confirmatory_records.jsonl
```

Every confirmatory row must carry the exact task-content hash, runtime-provenance hash, and pre-generation-seal hash.

### External chronology requirement

The local hash chain detects changes relative to the files presented at analysis time, but it cannot by itself prove chronology against an operator who controls both the records and the local filesystem: old records can be rewritten to carry a newly created seal hash. Before the first confirmatory branch, publish or commit the SHA-256 of `PREGENERATION_SEAL.json` to an external system you do not rewrite retrospectively (for example, a pushed Git commit in the public experiment repository or an independent timestamp service), and record every sealed attempt in the append-only experiment ledger. The primary report must cite that external commitment. Without it, the package verifies internal consistency, not that the seal genuinely predates generation or that unsuccessful sealed attempts were not discarded.

### 6. Run the sealed primary analysis

```bash
python run_o1_primary.py \
  --records confirmatory_records.jsonl \
  --manifest FREEZE_MANIFEST.json \
  --confirmatory-task-manifest confirmatory_tasks.jsonl \
  --calibration-task-manifest calibration_tasks.jsonl \
  --calibration-precommit CALIBRATION_PRECOMMIT.json \
  --seed-matrix seed_matrix.json \
  --result-seal RESULT_SEAL.json \
  --run-provenance RUN_PROVENANCE.json \
  --pregeneration-seal PREGENERATION_SEAL.json \
  --calibration-records calibration_records.jsonl \
  --calibration-metadata CALIBRATION_METADATA.json \
  --calibration-results CALIBRATION_RESULTS.json \
  --axis-package AXIS_PACKAGE
```

The analysis verifies the pre-generation commitment before opening records, reruns the axis verifier and calibration derivation, enforces every record-level gate, halts on any zero-alpha parity failure, and opens the primary once. It never computes secondaries on a halted or unopened run.

## Scope of the result

O1 tests one predeclared action family at one instrumented writable boundary: one-shot final-prompt-position injection at L3_24 along four source-derived geometries. The declared bank always contains eight signed actions, but the report must also state how many signed actions and axes produced measurable downstream transport. A bank-level null is scoped to the actions that transported; mechanically dead axes are evidence about readout–writability or locus compatibility, not evidence that the whole action space is barren.

Appendix J.2 supplies a prior quantitative warning: at the matched L4 loci, readable↔writable and outcome↔writable angles were near 90° and at the writable-span null. The preregistered expectation is therefore that d1 and d2 may be transport-dead at L3_24, d3 should transport by construction, and d4 is unknown. This is a diagnostic prediction, never a gate or a reason to remove d1/d2. Calibration still measures q_disc under the complete eight-action bank, so its power calculation remains valid; a smaller live bank will simply appear as lower discordance and potentially larger G.

A null is not a claim that Ouro has no useful intervention space. Only the exact practical-null verdict excludes headroom at the frozen threshold; an ordinary nonsignificant result is `INCONCLUSIVE`.
