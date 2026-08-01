# Resume Plan

Run is sequential A→P; each stage writes a verdict to `run_state.json` `stages`. Re-running a stage is idempotent. Training/eval stages are STOP-file pausable and checkpoint-resumable.

## Order & gates
1. A init → B input audit → C canary suite → D canary baselines. **Gate: do not train until C+D saved.**
2. E rendered logic + F budget/direct-answer + G offline prefs + H reward spec (offline data; CPU-cheap).
3. I matrix → J offline training (resumable) → K canary checkpoint eval → L final heldout (once).
4. M online-RL gate (decision only) → N error audit → O policy → P docs.

## Parts
- A — init: run state + manifest
- B — input_audit: verify v1 substrate
- C — canary_suite: fixed >=100/domain canary
- D — canary_baselines: base / prev-SFT / oracle / random baselines (GPU)
- E — rendered_logic_branch_sets: solver-rendered verified logic branch sets (offline)
- F — branch_budget_data: direct-answer + branch-budget views (data-composition repair)
- G — offline_branch_set_preferences: rejection-SFT + branch-set DPO + reward groups
- H — offline_reward_spec: deterministic verifier-backed reward function
- I — training_matrix: controlled offline arms A..I
- J — offline_training: bf16 LoRA offline training (resumable)
- K — canary_checkpoint_eval: select <=3 adapters for heldout
- L — final_heldout_eval: task-disjoint heldout, once
- M — online_rl_gate: decide if online RL is justified (no GRPO here)
- N — error_audit: hand-audit failures vs artifacts
- O — policy_decision: next architecture state
- P — doc_update: docs (append-only)

## Cost note
- Ouro-RLTT generation is the wall (~150–170 s/task at v1 budgets). Offline data (E/F/G) needs no generation. Generation-based stages (D baselines, K/L evals) are resumable long runs; canary GEN size is a cost/precision knob.
