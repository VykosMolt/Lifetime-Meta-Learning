# AXIS_DIAGNOSIS_V2 execution log

## Preservation

Timestamp: 2026-07-29T11:42:15+02:00

- Hash-locked 77 files from the complete failed `O1_REAL_001` run and immutable
  source package.
- Verified all 77 hashes.
- Committed only those roots and the detached snapshot manifests on `main`:
  `ea40a48` (`preserve failed O1 real axis reconstruction`).
- Unrelated dirty-tree files were not staged.

## Isolated worktree

Timestamp: 2026-07-29T11:43:01+02:00

- Created `/tmp/o1_axis_diagnosis_v2` from exact commit
  `e4776dd41a85cad699ac36f309b5986ab48bd171`.
- Imported the preservation commit byte-for-byte as `31ef40b`.
- Reverified the preservation manifest: 77/77.
- All diagnosis writes occurred in the isolated worktree.

## Capture parity and historical proxy

Timestamp: 2026-07-29T12:20:17+02:00

- Rehashed the full local checkpoint tree and both adapters; all matched.
- Audited application, scaling, locus, boundary, position, reference ordering,
  dtype, normalization, sign, phase, hooks, and zero control.
- `D4_CAPTURE_PARITY_REPORT = PASS`; no reconstruction defect found.
- Recomputed the historical random-probe proxy on the RTX 5070 Ti directly from
  both checkpoint binaries: `0.9510943054713733`, within
  `8.349041868971341e-08` of the historical diagnostics artifact.
- Did not reuse the value from prose.

## d4 and d2 diagnosis

Timestamp: 2026-07-29T12:20:17+02:00

- Ran 2,000 task-cluster bootstrap draws per adapter, eight leave-one-task-out
  fits per adapter, SVD/explained-variance diagnostics, principal angles,
  projection overlap, bidirectional held-out Procrustes, and held-out CCA.
- PC1 was task-bootstrap unstable for each adapter. Mean-update directions were
  individually stable but had cross-adapter cosine only `0.5720586047705143`.
- No supported shared one-dimensional or higher-dimensional actuator geometry
  was found.
- `D4_VERDICT = STABLE_BUT_DISTINCT_ADAPTER_GEOMETRIES`.
- Enumerated current d2 support and confirmed the sole positive task
  `math_0313a1abc4`.
- Found but did not substitute the 16-task `s3b2_features.pt` redesign
  candidate.
- `D2_ADEQUACY_VERDICT = INSUFFICIENT_POSITIVE_TASK_SUPPORT`.
- No replacement axis, bank change, calibration precommit, calibration, or
  confirmatory generation was created.
