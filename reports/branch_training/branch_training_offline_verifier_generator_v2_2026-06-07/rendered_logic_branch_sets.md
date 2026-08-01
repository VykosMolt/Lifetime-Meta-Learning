# Rendered Logic Branch Sets (Part E)

RENDERED_LOGIC_BRANCH_SET_VERDICT = LOGIC_BRANCH_SETS_READY
Groups 23172 | attempts 104372 | positive-oracle groups 20801 (0.898).

**External labels = direct match to the generator-verified gold** (truth-table/z3/finite-model). Renderer wiring self-check (valid-strategy branch gold-matches): 20801/20801 = 1.0.
**Diagnostic:** v1 free-text verifier agreement with gold-match = 0.9084 — low because that parser mishandles synthetic_fol/MCQ option-matching (it scores 'Valid' inside 'validity'); rendered data bypasses it by gold-matching the constructed final answer, which is why we use direct gold-match here.

## By category
- fol_entailment: 187
- mcq_logical_reading: 2123
- proofwriter_deduction: 6111
- ruletaker_deduction: 524
- synthetic_constraint_game: 2804
- synthetic_fol: 5083
- synthetic_propositional: 6340

Each branch carries strategy_label, failure_modes, and an EXTERNAL verifier label (pass/fail by gold-match). Valid: forward_chaining / option_elimination / finite_model_check / constraint_table / counterexample. Invalid failure modes: quantifier_reversal / negation_flip / premise_hallucination / wrong_elimination / invalid_contradiction / failed_counterexample. ~12% all-negative groups for hard DPO negatives.
Output: `data/branch_training_logic_expansion_v1/train_v2/rendered_logic_branch_sets.jsonl`.
