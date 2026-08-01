# O1 v1.5.3 verification report

**Status:** code-verified; real experiment unsealed.

This report covers the package produced from the uploaded v1.4 archive after implementing the remaining precommit, runtime, randomization, calibration, and workflow changes directly.

## Changes completed

### Pre-outcome chronology

- Added `CALIBRATION_PRECOMMIT.json` creation before calibration records exist.
- Bound every calibration row and calibration metadata object to that precommit.
- Added `PREGENERATION_SEAL.json`, created only after full preflight and before confirmatory records exist.
- Bound every confirmatory row to the committed runtime-provenance and pre-generation-seal hashes.
- Kept the result seal separate, append-only, atomic, and single-primary.

### Executable calibration

- Added deterministic calibration analysis for difficulty settings, alpha coherence, the complete pass/fail partition, `alpha_star`, optional secondary magnitude, bank-level discordance, the one-sided discordance bound, measured throughput, optional-arm drop order, exact power, `G_max`, `G_confirm`, and budget MDE.
- Added an explicit `TARGET_HEADROOM_EXCLUDED_BY_DISCORDANCE_BOUND` halt when the target effect exceeds the calibrated discordance bound.
- Made final-manifest values reproducible from raw calibration records and metadata.
- Fixed `apply_results_to_manifest()` so it copies the exact calibration-precommit hash into the final manifest; a dedicated fixture now covers this path.

### Runtime and record binding

- Added deterministic seed derivation from a fixed domain, master seed, task ID, and stream index.
- Added executable seed-matrix generation and exact regeneration in G16.
- Added canonical recursive runtime configuration and real artifact hashing.
- Bound checkpoint, tokenizer, prompt, parser, verifier, structured/random tensors, generation module, repository-clean state, and repository diff.
- Bound task content, stratum, generator setting, and Latin-square rank to the task manifests.

### Axis artifact

- The random bank is regenerated from a fixed public domain plus the structured tensor hash; there is no discretionary package-local seed.
- Copying the structured bank as the random bank is rejected.
- Canonical row hashes are recomputed.
- d4 is reconstructed from both required components and checked against the submitted row.
- The verifier and reconstruction script are required and hash-covered.
- `SHA256SUMS` must cover every axis-package file.
- Structural verification and remote-source attestation are reported separately.

### Statistical contract

- Exact McNemar planning uses the finite-sum calculation, not Monte Carlo.
- Planning minima reproduce 324 / 817 / 2238 / 337.
- The positive verdict is powered for the exact directional McNemar criterion; the percentile bootstrap is descriptive.
- Practical equivalence uses a conservative exact one-sided Clopper–Pearson bound rather than the degenerate percentile bootstrap.
- `required_G()` now builds exact critical tables adaptively, preserving results while removing unnecessary construction to 20,000 for small answers.

### Distribution and usability

- `run_all_tests.py` now verifies every digest and complete file coverage in the shipped `SHA256SUMS` before executing code. A fixture mutates `o1_analysis.py` and confirms the package check rejects it.
- `required_G()` now returns the documented `-2` sentinel when `delta > q_disc`; the guarded confirmatory path remains unchanged, while direct library callers receive a machine-readable infeasibility verdict.
- Corrected the documented command-line syntax for every executable.
- Added a complete v1.5 README and chronology.
- Added `run_all_tests.py`.
- Added seven end-to-end CLI workflow fixtures, including a complete synthetic primary opening through the public command-line entry point.
- The package runs from a flattened archive as well as the development layout.


### Transport-aware null scope

- Added an axis-level transport profile on top of the existing per-signed-action summary.
- The primary output now reports the number of signed actions and axes that measurably transported.
- Predeclared the Appendix J.2 prediction that d1/d2 may be transport-dead, d3 should transport by construction, and d4 is unknown.
- A null is explicitly scoped to transported actions; dead d1/d2 are not folded into a claim that the action space is barren.
- Added the real source-bundle template and reconstruction-status document to the distributed archive.

## Verification result

The final run completed successfully:

```text
53/53 confirmatory and record-integrity fixtures passed
17/17 calibration and calibration-precommit fixtures passed
13/13 cross-artifact, chronology, package-integrity, and bytecode-exclusion fixtures passed
7/7 end-to-end command-line workflow fixtures passed
15/15 axis-verifier adversarial checks passed
Exact McNemar planning minima: 324 / 817 / 2238 / 337
Python compilation: passed
JSON templates: parsed and fixed policy fields checked
CLI parsers: passed
Flattened-layout execution: passed
```

The complete console transcript is in `TEST_OUTPUT.txt`.

## What is not verified by this package

The local pre-generation seal does not independently prove that it predates the records against an operator who controls and can rewrite the local filesystem, nor does it prove that every sealed attempt was disclosed. Before the first real confirmatory branch, its digest must be externally committed or timestamped and every attempt logged. This is an operational trust boundary, not a defect that another local hash can close.

The package does not contain the real Ouro-RLTT checkpoint, the frozen pre-O1 source runs, the four reconstructed L3_24 source axes, real Horizon Logic task cohorts, runtime prompt/parser/verifier artifacts, or calibration generations. Therefore:

- the actual experiment is not sealed;
- the 56 exact `FREEZE_SLOT` leaves remain intentionally unfilled;
- source-run claims for the axes remain attested until the source artifacts or a hash-verified reconstruction log are supplied;
- no empirical claim about O1 headroom has been made.

The code now refuses to substitute synthetic or guessed values for those missing artifacts. The next real step is axis reconstruction from the frozen pre-O1 material, followed by the calibration precommit and executable calibration workflow in `README.md`.
