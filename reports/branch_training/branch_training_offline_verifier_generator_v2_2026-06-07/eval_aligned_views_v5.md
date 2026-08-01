# Eval-Aligned Views v5 (Round 4 data)

OFFLINE_VIEWS_V5_VERDICT = V5_EVAL_ALIGNED_READY
Counts: {"branch_set_rejection_sft": 21817, "branch_set_dpo": 28172, "one_branch_sft": 13137} | rejection_sft domains: {"math": 8893, "logic": 12000, "coding": 924}
DPO pair types: {"show_work": 4000, "oracle": 12172, "wrong_final": 6000, "reasoning_error": 6000} | unique coding sft 70 (dup x12)
Format contract: {"prompts_chat_templated": true, "prompts_have_gen_system": true, "prompts_end_with_assistant": true, "no_branch_prefix_completions": true, "noncoding_end_with_final": true, "coding_fenced_no_final": true} | token-gate drops {"one_branch_sft": 229, "branch_set_rejection_sft": 229, "branch_set_dpo": 4} | skipped {"math_canary_overlap": 50}
Round-4 fix: every prompt is the exact eval-side rendering (B1.build_prompt: GEN_SYSTEM posture, chat template, MBPP fn-note via unit-test rejoin; gen-pool rows under their own scaffold), every completion is ONE single solution (executed logic derivation / <<>>-stripped rationale + FINAL ANSWER / verbatim verified reasoned code, fenced, no FINAL ANSWER line). DPO adds reasoning_error negatives (corrupted mid-work number + wrong final). Reasoning excluded (options not persisted); v3/v4 rows not carried (misaligned format). Trainer must include the pre-rendered _fmt patch.
