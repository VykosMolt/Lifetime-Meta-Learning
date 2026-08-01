# AXIS_DIAGNOSIS_V2

Scientific boundary: this is a reconstruction diagnosis over frozen pre-O1
artifacts. No O1 calibration or confirmatory generation was started, no
replacement axis was emitted, and the frozen d4 gate was not changed.

## Verdicts

- `D4_VERDICT = STABLE_BUT_DISTINCT_ADAPTER_GEOMETRIES`
- `D2_ADEQUACY_VERDICT = INSUFFICIENT_POSITIVE_TASK_SUPPORT`
- d4 capture parity: `PASS`
- historical adapter proxy: `REPRODUCED_FROM_CHECKPOINT_BINARIES`
- frozen d4 gate: still failed, observed `|cosine| = 0.13812127400617838`
  versus required `0.90`

The d4 verdict means the uncentered mean-update direction of each adapter is
stable under the available task-cluster diagnostics, but those two directions
are distinct. Neither centered PC1 is task-bootstrap stable, and the adapters
do not share a supported higher-dimensional subspace. This verdict does not
authorize replacing the preregistered PC1 construction with a mean-update
construction.

## Preservation and isolation

The complete failed `O1_REAL_001` snapshot was committed unchanged on `main` as
`ea40a48` and imported byte-for-byte into this isolated worktree as `31ef40b`.
The preservation manifest covers 77 files and verifies 77/77. This worktree
descends from the required base
`e4776dd41a85cad699ac36f309b5986ab48bd171`.

## d4 capture parity

Every audited item passed:

- local Ouro-RLTT checkpoint tree:
  `7dbbe4ef91de369bd712bad587c1af4f7c5dc42deff5377d4d70086c4c08a802`;
- causal adapter:
  `f0831a9abf846cdf702d1211da346e2de3d32fe2ce11275247bad8400af215f4`;
- sequence adapter:
  `75140bf3af7dfba4ee6851a500693b019f884e25dad0a2112f493dc99630597c`;
- exact hook implementation hashes matched the capture report;
- layer-36 post-layer adapter hooks used `multi_loop_decayed` scaling:
  causal `[0.0025, 0.005, 0.0075, 0.01]`, sequence
  `[0.005, 0.01, 0.015, 0.02]`;
- the capture was zero-indexed loop 2 / paper L3, physical layer 24;
- the captured boundary was the post-block output of `model.model.layers[23]`;
- all eight examples and their ordering matched the frozen reference hash
  `d145955fdd15a7c413fae26276d4a61a0d4eab05f5241b73365982e4ed49c75b`;
- every position was the final non-padding prompt token;
- the model ran bfloat16 and stored induced updates were float32, with float64
  PC diagnostics and unit-RMS component normalization;
- the sign convention was positive dot with each adapter's uncentered mean;
- the phase was prompt-only;
- all hooks fired at all four loops, all matrices were finite, and the
  zero-alpha control was bit-exact.

There is no evidence of a capture or reconstruction defect.

## Historical 0.951 proxy

The historical proxy was recomputed from the two checkpoint binaries, not
copied from prose. The computation used the original adapter implementation and
the original CUDA construction: both adapters were evaluated directly on the
same 64 seeded standard-Gaussian 2048-dimensional vectors, each adapter
normalized its outputs per example, and those outputs were averaged.

- recomputed cosine: `0.9510943054713733`;
- historical diagnostics value: `0.951094388961792`;
- absolute difference: `8.349041868971341e-08`.

This reproduces the historical proxy within the declared `1e-6` tolerance. It
does not reproduce d4 because it omits the model, prompts, alpha, loop scaling,
recurrent propagation, downstream residual subtraction, centering, and PC
extraction. It is a native layer-36 adapter-output proxy over Gaussian vectors,
not transported induced L3_24 geometry.

The proxy is nearly unrelated to the captured quantities:

- causal proxy versus causal induced PC1: `0.00133`;
- sequence proxy versus sequence induced PC1: `0.01132`;
- causal proxy versus causal mean update: `0.07925`;
- sequence proxy versus sequence mean update: `0.10351`.

## Individual adapter geometry

Each induced-update matrix is `[8,2048]` with one independent task cluster per
row. Centering therefore limits effective rank to seven; PC8, PC16, and
centered top-8/top-16 subspaces are not estimable.

| Diagnostic | Causal | Sequence |
|---|---:|---:|
| Effective centered rank | 7 | 7 |
| PC1 explained variance | 0.21485 | 0.19077 |
| PC2 explained variance | 0.16843 | 0.17258 |
| PC4 explained variance | 0.13141 | 0.13134 |
| Cumulative through PC4 | 0.66142 | 0.64616 |
| PC1 bootstrap median absolute cosine | 0.51314 | 0.37583 |
| PC1 bootstrap q05 | 0.05000 | 0.05829 |
| PC1 leave-one-task-out minimum | 0.56504 | 0.40661 |
| Mean-update bootstrap median cosine | 0.80834 | 0.86419 |
| Mean-update bootstrap q05 | 0.66817 | 0.75238 |
| Mean-update leave-one-task-out minimum | 0.94146 | 0.96620 |

The singular spectra are flat rather than one-dimensional. PC1 is not stable
under task-cluster resampling for either adapter. Each mean-update direction is
substantially more stable, but changing d4 from centered PC1 to a mean direction
would require a new preregistration.

## Cross-adapter geometry

- PC1 signed cosine: `-0.13812127400617838`;
- mean-update cosine: `0.5720586047705143`;
- per-example update cosine: range `0.28614–0.53509`, median `0.50142`;
- top-2 canonical cosines: `[0.50666, 0.30005]`;
- top-4 canonical cosines: `[0.60690, 0.54998, 0.35761, 0.11207]`;
- top-4 principal angles: `[52.63°, 56.63°, 69.05°, 83.57°]`;
- top-4 symmetric projection overlap: `0.20281`;
- causal energy captured by the sequence top-4 subspace: `0.15982`;
- sequence energy captured by the causal top-4 subspace: `0.16598`;
- leave-one-task-out Procrustes median cosine:
  causal→sequence `0.29001`, sequence→causal `0.11718`.

Held-out CCA correlations were `[0.66966, -0.68278, 0.29119, -0.01535]`
across only eight tasks. Their mixed signs, fold-specific bases, and small
sample size make them exploratory; they do not establish a shared subspace.
Top-8 and top-16 comparisons are mathematically unavailable.

## d2 adequacy

The current source contains 344 L3_24 rows:

- positives: 8 rows, one task, `math_0313a1abc4`;
- negatives: 336 rows across four tasks:
  `coding_00715d44da` 86, `logic_00013d9d03` 86,
  `reasoning_00bf397683` 86, and `math_0313a1abc4` 78.

Task-clustered bootstrap variation on the positive side is unidentifiable:
every draw necessarily selects the same sole positive task. Conditional
negative-task resampling cannot certify task-general positive geometry. When
`math_0313a1abc4` is left out, the positive class disappears and the estimator
is undefined.

A broader frozen candidate was found at
`cross_loop_early_layer_taps_20260720/s3b2_features.pt`, SHA-256
`0bf994770ee6e14fa834680d5f621211a2758de92e55d0dc45e5db0926d3f34d`.
It contains 160 `[5,4,2048]` branch features, 29 positive rows from eight
tasks, and 131 negative rows from 16 tasks. It is explicitly marked
small-N/exploratory, and coding has zero positive tasks. It was not substituted
for d2. Its split, leakage, domain balance, and source eligibility must be
preregistered and audited in a new experiment.

The larger cross-loop candidate corpus was also found, but it mixes
deterministic correctness, synthetic mutant rewards, preferences, and alignment
rewards; it is not one uniform verifier-outcome corpus. The May empirical
direction artifact is excluded because it is at physical layer 36 rather than
L3_24.

## Evidence boundary

- Real measurements: the two real captured adapter-induced update matrices;
  checkpoint hashes and checkpoint-tree rehash; frozen task labels and IDs; the
  GPU recomputation of the historical proxy.
- Reconstruction diagnostics: SVDs, explained variance, task-clustered
  bootstrap, leave-one-task-out fits, principal angles, Procrustes maps, and
  held-out CCA.
- O1 calibration evidence: none.
- O1 confirmatory evidence: none.
