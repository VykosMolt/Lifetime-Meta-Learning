# Data-Flaw Note — rendered branch sets are strategy narrations, not solutions (2026-06-11)

META_BRANCH_RENDER_VERDICT = RENDERED_BRANCHES_ARE_META_TEXT

Found during Part K canary inspection of `arm_C_offline_verifier_dpo` after its anomalously
fast generation (~32 s/task vs ~150 s/task base) prompted a manual output audit.

## Symptom

Arm C generates short branch-set *summaries* in the trained format and asserts a final answer
with no computation (mean 263 chars vs base 970 on the same tasks; a math "solution" can be
literally `Branch 1: Answer: FINAL ANSWER: 89`). Some emitted "branches" are verbatim
failure-mode descriptions from the training renderings ("stop too early", "keep a distractor").

Paired canary deltas (arm C vs base, same tasks): math −0.533, logic −0.233, coding 0/5,
reasoning +0.067. Arm B (budget/direct-answer SFT, no DPO) shows a milder form: math −0.30ish
on its slice. Arm A (v1 branch-format SFT over real generations) holds at base.

## Root cause

Part E/G rendered branch sets describe strategies at the meta level instead of containing
executed, full-text solution branches. Heuristic scan of `branch_set_dpo.jsonl` (first 3k
non-empty rows): ~71% of chosen texts have a first branch under 200 chars — strategy
sentences, not worked solutions. The solver knew the answers, so the rendering never had to
show the work; DPO then optimized the model toward narrating strategies and asserting answers.

## Implications

- The Part K canary is doing its job: this was caught in hours, pre-heldout, pre-deployment.
- Expect G and H (same DPO data) to share the degeneration; selection will likely return
  `NO_ADAPTER_IMPROVES_ON_CANARY` → external DualAnchor + CoreContent_v2 baseline stands.
- Fix is upstream, not in the trainer: rendered branch sets must contain *executed* branches
  (full worked solutions per branch, as in the v1 model-generated pools), with meta-labels
  kept out of the completion text. Re-render Part E, rebuild Part G prefs, re-run arms C/G/H.
- The speed signature (output length collapse) is a cheap degeneration alarm worth adding to
  Part K aggregates (mean branch chars per arm vs base).
