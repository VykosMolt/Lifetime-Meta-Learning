# O1 L3_24 axis reconstruction status

Status: **BLOCKED ON SOURCE BINARIES — no axis values fabricated.**

The v1.5.3 package contains the deterministic assembler and adversarial verifier, but the active runtime does not contain any of the frozen pre-O1 `.pt`, `.npy`, `.npz`, `.safetensors`, or adapter artifacts needed to derive the four real 2048-dimensional axes.

## Required source bundle

1. **d1 — readable process-quality**
   - L3_24 raw feature tensors from `cross_loop_early_layer_taps_20260720/features/`
   - source-item-disjoint train/validation/task manifests
   - standardization/PCA/probe parameters or the exact fitting script and seed

2. **d2 — empirical verifier outcome**
   - L3_24 raw hidden states
   - deterministic correctness labels and task IDs
   - the task-bootstrap construction inputs needed to reproduce the first singular mean-difference direction

3. **d3 — writable transport**
   - exact-protocol S1/S3 injection-delta bundle containing L3_24 deltas
   - locus metadata and reconstruction log from `s1_s3_exact_injection_orthogonality_null_audit_2026-06-17.*`

4. **d4 — learned actuator**
   - both independently trained adapter checkpoints
   - the frozen pre-O1 reference prompt/task set
   - Ouro-RLTT checkpoint/tokenizer and exact capture code, or the two already captured L3_24 induced-update matrices

## Required provenance

Every supplied source file must arrive with its existing SHA-256 record. Cross-locus substitutes, summary statistics, paper tables, reconstructed values from prose, or synthetic vectors are not acceptable.

Once these binaries are present, run `axis_reconstruction.py` from the O1 package and then `verify_axis_artifact.py`. A structurally valid result must contain four unit-RMS rows in the fixed order d1–d4, a Gram-matched deterministic random bank, the two d4 components, complete package hashes, and the source-provenance records.
