# Input Audit (Part B)

INPUT_AUDIT_VERDICT = INPUTS_READY_WITH_WARNINGS
logic_substrate_ready = False

## Logic tasks
- 48536 tasks; splits {'train': 33247, 'heldout': 11017, 'val': 4272}; 10 families; uid_unique=True; prompt-hash leakage {'heldout∩train': 81, 'heldout∩val': 12, 'train∩val': 54}.
- families: fol_entailment, logical_entailment, lsat_analytical_reasoning, lsat_logical_reasoning, mcq_logical_reading, proofwriter_deduction, ruletaker_deduction, synthetic_constraint_game, synthetic_fol, synthetic_propositional
## Branch pools
- labeled 33590 groups (pos-oracle 0.9981); external_label=True, label_source_external_only=True.
- deduped 31460 groups; multi-split group_ids 0.
- terminal survivor sets 3912 (oracle retention 1.0).
## Teacher traces (policy, not ground truth)
- 13667 rows; cols ['group_id', 'domain', 'split', 'n_keep', 'defer', 'n_rescued']; has_correctness_cols=False; schema gt-false=True.
## Training views
- branch_format_sft: 177 rows; keys ['completion', 'domain', 'prompt', 'text']
- branch_diversity_sft: 240 rows; keys ['completion', 'domain', 'n_strategies', 'prompt', 'text']
- branch_policy_distillation: 10209 rows; keys ['completion', 'defer', 'domain', 'prompt', 'teacher', 'teacher_is_ground_truth']
- final_self_selection: 177 rows; keys ['completion', 'domain', 'prompt', 'text']
- verifier_reward_rl: 240 rows; keys ['branch_rewards', 'domain', 'group_reward', 'n_strategies', 'parse_ok_rate', 'positive_oracle_present', 'prompt']
- branch_preference_dpo: 194 rows; keys ['chosen', 'domain', 'prompt', 'rejected']
## Warnings / errors
- WARN: logic prompt-hash overlap across splits: {'heldout∩train': 81, 'heldout∩val': 12, 'train∩val': 54}
