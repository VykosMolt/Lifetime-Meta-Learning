# Branch-Training Final Policy Decision (Part Q)

BRANCH_TRAINING_POLICY_DECISION = KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE
BRANCH_TRAINING_LOGIC_EXPANSION_STATUS = LOGIC_EXPANSION_READY_TRAINING_NOT_READY

## Primary deliverable

- branch-training DATA + EVALUATION HARNESS (logic-expanded, verifier-labeled, 5 training views) — READY.
- The L/O training is a **bounded, 300-step proof-of-capability** (bf16 LoRA on Ouro-RLTT), not a converged model.

## Trained vs Ouro-RLTT (no adapter)

- macro positive_oracle@K: {'base': 0.708, 'sft': 0.688}; lift (sft-base): -0.02; diversity lift: 0.45.

## Locked state

- **branch_survival**: DualAnchor (MIX_CODE_REASONING + MIX_OBJECTIVE_ALL) — unchanged (teacher + external baseline)
- **content_final_selection**: CoreContent_v2_blockwise_pruned_24_36 — unchanged (validated terminal ranker; H confirmed it beats DualAnchor forced-top1 within survivors)
- **terminal**: top5/full survivor-set handoff (selection, not survival, is the bottleneck per H)
- **branch_trained_model**: branching_sft LoRA adapter (bounded 300-step proof-of-capability); decision=KEEP_EXTERNAL_DUALANCHOR_CORECONTENT_BASELINE
- **logic**: expanded: 33247 train groups across 10 families, verifier-backed
- **science**: diagnostic only
- **steering**: not run, not claimed

## Stage verdicts

    INIT = READY
    DATA_PULL = LOGIC_EXPANDED_READY
    LOGIC_CANON = LOGIC_VERIFIER_READY
    BRANCH_POOLS = LOGIC_BRANCH_POOLS_READY
    VERIFIER_LABELING = LOGIC_LABELS_READY
    DEDUP = LEAKAGE_FOUND_FIXED
    INTEGRATED_TERMINAL = CORECONTENT_IMPROVES_TERMINAL
    REACHABILITY = LOGIC_REACHABILITY_READY
    TEACHER_TRACES = TEACHER_MISMATCH_ON_LOGIC
    TRAINING_DATASET = TRAINING_DATA_READY
    BRANCHING_SFT = SFT_TRAINED
    TEACHER_DISTILL = NOT_RUN
    VERIFIER_REWARD = NOT_RUN
    TRAINED_EVAL = MODEL_INTERNAL_BRANCHING_PARTIAL

## Headline findings

- math_reachability_fix: 0.31 -> 0.83 via brutal tool-free prompt + 1400 math budget + early-stop + LaTeX verifier
- terminal_bottleneck: selection not survival (H: CoreContent_v2 0.658 vs DualAnchor forced-top1 0.379, survivor oracle retention 1.0)
- teacher_useful_except_logic: J: DualAnchor pruning beats random +0.11 coding/reasoning, only +0.03 logic
- failures_audited_genuine: math/logic/reasoning/coding fails hand-verified after fixing math-LaTeX, coding-name, hendrycks-uid artifacts

## Next

scale + run M (teacher distillation) and N (verifier-reward RL) to convergence for a real internal-branching model
