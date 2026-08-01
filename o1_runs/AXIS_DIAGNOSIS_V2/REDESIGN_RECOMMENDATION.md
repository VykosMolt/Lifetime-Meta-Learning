# Recommended redesign

This diagnosis does not repair O1 in place. Any continuation must be a newly
versioned preregistration with a new chronology commitment.

## d4

1. Decide before capture whether the estimand is a centered PC1, an uncentered
   mean update, or a higher-dimensional actuator subspace. The present evidence
   does not permit changing from PC1 to the more stable mean after seeing the
   results.
2. Use genuinely independent adapter replicas. The current sequence adapter
   records initialization from the causal checkpoint, so the pair represents
   different objectives/optimization runs but not independent initialization.
   A new design should use independently seeded initialization and should
   preregister whether training data are shared or disjoint.
3. Expand the frozen reference cohort before any adapter capture. A proposed
   minimum is 32 independent task clusters, balanced across at least four
   domains. This is a design recommendation, not a power result: it exceeds the
   mathematical minimum for a centered top-16 space and leaves tasks for
   held-out evaluation.
4. Keep the parity-confirmed mechanism fixed unless the new preregistration
   deliberately changes it: zero-indexed loop 2 / paper L3, physical layer 24,
   post-layer boundary, final non-padding prompt position, prompt-only phase,
   bfloat16 model, float32 captured updates, and exact adapter scaling.
5. Predeclare task-cluster bootstrap and held-out stability thresholds.
   Separate within-adapter stability from cross-adapter sharing. Do not use
   in-sample CCA or Procrustes fit as evidence without held-out task evaluation.
6. If the target is a shared subspace rather than one axis, create a new O1
   version with new bank geometry and power analysis. The current four-axis
   package and its d4 gate cannot be repurposed.

## d2

1. Retire the current one-positive-task vector from task-general certification.
2. Audit the found `s3b2_features.pt` candidate under a new preregistration:
   bind its hash, task manifest, verifier implementation, splits, trajectory
   phase, and exact `[physical layer 24, paper L3]` tensor index.
3. Require positive and negative task support in every intended target domain.
   The found candidate currently has no positive coding task and is therefore
   not yet a balanced general source.
4. Freeze task-level weighting and task-cluster bootstrap. Candidate rows from
   one task must never be treated as independent task evidence.
5. If frozen artifacts remain insufficient, preregister a new pre-O1 source
   collection before looking at any O1 calibration or confirmatory outcomes.

## Gate

No calibration precommit should be created until the new preregistration has
produced a complete d1–d4 package that passes its own frozen reconstruction and
axis-verification gates. The current d4 failure and d2 inadequacy remain exact
blockers.
