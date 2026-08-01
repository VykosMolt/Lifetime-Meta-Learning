# O1 v2 interpretation addendum — binding, pre-outcome

Status at creation: no real O1 calibration or confirmatory outcome exists.
This addendum deliberately binds the interpretation of O1 v2 results BEFORE
any outcome is observed. It narrows what may later be claimed; it never
widens. It is sealed alongside the v2.1 package and externally committed with
the replacement calibration precommit.

## A3 / A4: what they are and what they are not

- A3_CAUSAL_MEAN and A4_SEQUENCE_MEAN were introduced in a NEW preregistration
  (v2) after the original shared-PC d4 failed its frozen 0.90 cosine gate
  (observed |cosine| 0.1381).
- The accepted AXIS_DIAGNOSIS_V2 explicitly stated that its evidence did not
  permit silently swapping the mean for PC1 INSIDE the failed v1 attempt
  ("The present evidence does not permit changing from PC1 to the more stable
  mean after seeing the results"). v2 did not repair v1: it preserved the
  failure and defined a new fixed instrument. That instrument choice was
  informed by post-hoc stability diagnostics on frozen, outcome-free
  reconstruction data. This is acknowledged, not hidden, and it cannot bias
  H_reach because no O1 outcome existed; its cost is interpretive scope, which
  this addendum fixes.
- A3/A4 are stable only RELATIVE to the unstable centered PCs. Their absolute
  stability evidence is weak and sits near the small-N noise floor: for eight
  near-orthogonal rows, leave-one-task-out cosine is mechanically ~0.935 for
  pure noise, and the pure-noise bootstrap q05 is ~0.61 against A3's observed
  0.675 and A4's 0.755. The per-row cosine to the mean is only ~0.34-0.59:
  each mean captures a small shared component of a widely dispersed set.
- There are only 8 reference tasks, drawn from ARC-Challenge and MMLU
  high-school-biology multiple choice. None is a Horizon Logic or any
  propositional-logic task; the reference domain is disjoint from the O1
  target domain, and the ARC-half/MMLU-half mean cosines (0.22 causal, 0.40
  sequence) show the directions are strongly domain-dependent.
- The causal and sequence adapters share lineage: the sequence checkpoint
  records initialization from the causal checkpoint, and both were trained
  targeting layer 36 in multi-loop-decayed mode, not the L3_24 injection
  locus. Their 0.572 cosine partially reflects shared ancestry; the bank has
  closer to three effective directions than four.
- A3/A4 therefore represent FIXED REFERENCE-CONDITIONED MEAN UPDATE PROBES,
  not universal actuator geometry. Axis-specific results (per-direction
  secondaries, transport profiles) are descriptive/exploratory in every
  report.

## Null scope

A null pooled O1 primary result means ONLY:

> No oracle-reachability expansion was detected for these four fixed
> intervention axes at the L3_24 boundary, over the frozen coherent magnitude
> range, on the selected Horizon Logic difficulty slice, at K=8 under CRN.

It does NOT imply:

- that no writable Ouro action space exists;
- that no trainable controller could work;
- that readable directions are inherently unwritable;
- that all adapter-induced controls are ineffective.

## Positive-primary scope

The Gram-matched random arm is the only control separating
direction-specific reachability from generic matched-perturbation diversity.
If the random arm is dropped under the predeclared budget rule, a positive
structured primary supports:

> reachability headroom for the structured bank

and does NOT support:

> specificity of those directions relative to generic matched perturbations.

That weaker claim must be stated as the headline in any report of a positive
primary obtained without the random arm.

## Reporting rules

- Per-axis secondaries may never replace a null pooled primary as the
  headline.
- Transport remains diagnostic only; a dead axis narrows the scope of a null
  and licenses no other claim.
- A completed calibration is not an O1 result. A launched confirmatory run is
  not a completed result.
