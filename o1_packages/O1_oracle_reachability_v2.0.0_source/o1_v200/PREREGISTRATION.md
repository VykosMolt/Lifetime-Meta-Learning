# O1_V2 oracle-reachability preregistration

**Design version:** 2.0.0  
**Status at design construction:** no O1 calibration or confirmatory
generation has occurred.

## 1. Design identity and non-retroactivity

O1 v2 is a new preregistered experiment design. It does not change, repair, or
reinterpret the failed v1.5.3 axis reconstruction.

The following conclusions are fixed:

- the original d2 is inadequate because its eight positive rows belong to one
  task;
- the original shared-PC d4 is rejected because the induced L3_24 PC1 absolute
  cosine is `0.1381212740`, below its frozen `0.90` gate;
- the historical proxy cosine `0.9510943` is real but is not induced L3_24
  geometry;
- both centered adapter PC1s are unstable;
- each adapter’s mean induced-update direction is individually stable, and the
  two means are moderately rather than strongly aligned.

No v2 statistic changes any v1 verdict.

## 2. Primary estimand

For task \(g\), let \(R\) be one only when a branch is well formed and the
deterministic truth-table verifier marks its committed answer correct.
Malformed branches are zero. The task-level oracle is the maximum across the
eight common-random-number streams.

The primary endpoint is:

\[
H_{\mathrm{reach}} =
\operatorname{mean}_g[
\max_i R_{\mathrm{structured},g,i}
-
\max_i R_{\mathrm{baseline},g,i}
].
\]

Both calibrated strata are pooled. The primary test is the two-sided exact
McNemar test with directional rejection. Paired task bootstrap is descriptive.
Practical equivalence uses the exact one-sided Clopper–Pearson bound.

## 3. Primary bank

All rows are finite raw-residual L3_24 vectors with shape `[2048]`, normalized
to unit RMS only after their fixed source construction.

### A1_READABLE

The accepted real L3_24 readable process-quality covector is carried forward
byte-for-byte with its source-item-disjoint fitting provenance and canonical
row hash.

### A2_WRITABLE

The accepted leading direction of the centered exact-protocol S1/S3 L3_24
injection-delta bundle is carried forward byte-for-byte. It remains explicitly
classified as a reconstruction from the hash-verified exact-protocol bundle;
the historical tensor itself was absent.

### A3_CAUSAL_MEAN

Let \(U_C\) be the exact frozen causal-adapter-induced update matrix with shape
`[8,2048]`, one prompt-only final-position row per frozen reference task.

\[
A3 = \operatorname{unitRMS}(\operatorname{mean}_{task}(U_C)).
\]

The stored float32 matrix is cast to float64 for the arithmetic mean. There is
no centering, PC extraction, task replacement, or cross-adapter averaging.

### A4_SEQUENCE_MEAN

With the sequence adapter matrix \(U_S\) in the identical eight-task order:

\[
A4 = \operatorname{unitRMS}(\operatorname{mean}_{task}(U_S)).
\]

The same no-centering/no-PC rule applies.

### Fixed exclusions

- A3 and A4 are not averaged into a shared axis.
- Neither rejected centered PC1 is used.
- The inadequate original d2 is not used.
- `D2_OUTCOME_AXIS_V2` cannot modify the primary bank regardless of its
  diagnosis.
- No direction can be selected, dropped, or rotated using O1 correctness.

## 4. Axis-package gates

The verifier must establish:

- row order exactly
  `[A1_READABLE, A2_WRITABLE, A3_CAUSAL_MEAN, A4_SEQUENCE_MEAN]`;
- all structured and random tensors have shape `[4,2048]`, are finite
  little-endian float64, and structured rows are unit RMS;
- embedded A1/A2 source axes match their structured rows exactly;
- embedded causal and sequence matrices are exact finite float32 `[8,2048]`
  sources;
- the ordered eight-task reference JSONL is identical to the capture
  reference;
- A3/A4 equal the recomputed unit-RMS uncentered means;
- manifest claims explicitly say `centered=false` and
  `pc_extraction=false`;
- canonical row hashes and stored Gram matrix reproduce;
- pairwise absolute cosine is below `0.98`;
- the random bank exactly matches the non-discretionary v2 regeneration and
  has the structured Gram matrix within tolerance;
- every package file is covered by `SHA256SUMS`;
- no packaged bytecode exists.

Upstream A1 fitting, A2 exact-protocol reconstruction, and live adapter capture
remain hash-bound/attested unless all upstream execution inputs are embedded.
This assurance boundary must be reported separately from structural checks.

## 5. Random bank and action geometry

The random bank seed is derived only from the fixed public domain
`O1-v2.0-fixed-random-domain` and the canonical structured tensor bytes. A
canonical QR basis is Gram-matched so \(RR^T = DD^T\). No discretionary seed is
accepted.

The primary bank is always four axes × two signs, K=8. The structured bank is
not redefined using transport. The random bank is an optional budget arm, not
an alternate primary.

## 6. Intervention and transport

Intervention occurs once during prefill at:

- zero-based loop 2 / paper L3;
- physical layer 24 post-layer residual boundary, exactly
  `model.model.layers[23]` in the zero-based Python module list;
- final non-padding prompt position only.

For unit-RMS direction \(d\), pre-intervention residual RMS \(r\), sign \(s\),
and magnitude \(\alpha\):

\[
\Delta = s\alpha r d.
\]

It is not repeated on generated tokens or later visits.

Transport is measured before stochastic decoding as downstream residual
change at the normalized paper L4, physical layer 47 loop boundary
(`OuroModel` `per_loop_hidden_states[3]`), final prompt position, divided by
injected RMS. The transport-dead threshold is `0.0001`. Transport is diagnostic
only: no axis is removed or replaced because it transports weakly.

## 7. Decoding and common random numbers

- temperature `0.7`;
- top-p `0.95`;
- maximum new tokens `448`;
- stop after the first parseable `FINAL ANSWER:` commitment or the token cap;
- deterministic truth-table verification;
- eight explicit per-task stream seeds;
- same seed at a stream index across paired arms;
- cyclic Latin-square action-to-stream assignment.

Exact runtime module, prompt, tokenizer, parser, verifier, checkpoint, and
generator hashes must be frozen in the calibration precommit.

## 8. Calibration

The alpha grid is `[0.005, 0.01, 0.02, 0.04, 0.08]`.

Alpha passes only the frozen coherence criteria:

- finite logits;
- non-empty continuation;
- no catastrophic repetition;
- reachable valid final-answer syntax;
- per-stratum well-formed rate at least `0.85`;
- lower one-sided 95% confidence bound satisfies the `-0.05`
  noninferiority margin versus matched baseline.

Correctness does not select alpha. `alpha_star` is the largest passing
magnitude. `alpha_secondary` is the next smaller passing magnitude or null.

Calibration measures baseline any-correct-of-8, bank-level CRN discordance,
its preregistered upper bound, per-axis/per-sign transport, coupling survival,
throughput, exact `G_power`, `G_max`, `G_confirm`, budget MDE, and any optional
arm drop. Dead actions remain in the bank.

If the discordance upper bound is below `delta_target=0.05`, calibration emits
the declared infeasibility verdict and confirmation does not start.

The exact power implementation must continue to reproduce planning minima:
`324 / 817 / 2238 / 337`.

## 9. Zero-alpha gate

The full zero-alpha bank must match baseline exactly in:

- generated token IDs and lengths;
- termination state;
- well-formedness;
- verifier correctness;
- derived binary reward;
- zero downstream transport.

Any mismatch halts before the primary endpoint. The zero-alpha effect is never
subtracted from the primary.

## 10. Budget and interruption

- maximum wall-clock: seven days;
- calibration excluded from that confirmatory budget;
- rerun reserve: `0.15`;
- completion requirement: `100%`;
- drop order if needed: random bank, then secondary magnitude;
- zero-alpha is never dropped;
- interruption resumes only missing sealed shards with identical tasks,
  hashes, and streams.

## 11. Chronology and attempts

Before calibration, a calibration precommit must bind:

- this design and code commit;
- the exact v2 structured/random axis hashes and axis-package tree hash;
- checkpoint, tokenizer, generator revision, prompt, parser, verifier, and
  generation module;
- decoding and boundary semantics;
- alpha grid;
- calibration task manifest and seed design;
- repository clean state.

Its digest must be externally committed before the first calibration branch.
After calibration, a distinct pre-generation seal must be externally committed
and appended to the attempt ledger before confirmation.

No sealed attempt may be repaired in place, task-replaced, or omitted from the
ledger. A local commit is an integrity aid, not external chronology proof.

## 12. D2_OUTCOME_AXIS_V2

The separate diagnosis uses only the frozen
`s3b2_features.pt` artifact with SHA-256
`0bf994770ee6e14fa834680d5f621211a2758de92e55d0dc45e5db0926d3f34d`.
It reports domain/task support, clustered stability, leave-positive-task-out,
domain-held-out, contributions, and cohort leakage.

Its possible verdict cannot alter this primary bank. A later secondary
outcome-axis experiment requires a new preregistration, frozen source/task
exclusions, and target-domain justification.

## 13. Reporting boundary

The report must distinguish:

- synthetic/package validation;
- real pre-O1 artifact reconstruction;
- real O1 calibration;
- real O1 confirmatory evidence.

A verified v2 package and constructed real axis bank are not calibration. A
calibration run is not confirmatory evidence. A launched run is not a completed
result.
