# Offline Reward Spec (Part H)

OFFLINE_REWARD_SPEC_VERDICT = LOGIC_REWARD_READY
Deterministic reward over EXTERNAL verifier labels (not a learned reward model). It scores a branch SET; correctness dominates and **diversity counts only when tied to verifier-positive branches**, so Branch 1/2/3 formatting alone earns nothing.

**Audit:** mean reward positive-oracle sets 4.523 vs all-negative sets -0.399 → margin **4.922** (separates good from bad). Overall reward std 1.137; 5 degenerate components; logic domain-validity fired on 3435 logic groups.

## Components (weight · meaning)
- `positive_oracle_present` (+2.0) — + any branch reaches the verified answer
- `n_verifier_positive` (+0.25) — + capped count of verifier-positive branches
- `verifier_positive_diversity` (+0.4) — + distinct answers AMONG correct branches (useful diversity only)
- `appropriate_branch_budget` (+0.5) — + few branches when easy, several when hard
- `parse_ok` (+0.5) — + branches emit a parseable final answer
- `final_answer_correct` (+1.5) — + selected final matches a verifier-positive answer
- `domain_validity` (+1.0) — + valid proof/contradiction/counterexample/elimination; - quantifier-reversal/negation-flip/etc
- `duplicate_branch_penalty` (-0.6) — - duplicate branch finals
- `superficial_diversity_penalty` (-0.6) — - diverse-looking but all-wrong branches
- `overbranching_penalty` (-0.8) — - many branches on an all-correct (easy) task
- `underbranching_penalty` (-0.6) — - a single failing attempt on a hard task
- `unsupported_premise_penalty` (-0.5) — - premise hallucination
- `invalid_final_format_penalty` (-0.4) — - unparseable final
- `ambiguity_penalty` (-0.3) — - pool mostly unparseable

## Domain validity
- logic: + valid forward-chaining/elimination/finite-model/constraint-table/counterexample; - quantifier_reversal/negation_flip/premise_hallucination/invalid_contradiction/wrong_elimination/option_letter_mismatch.
- coding: external = executed unit tests; math: exact/sympy; reasoning: correct + direct-when-appropriate; alignment: preference-label match, concision (penalize needless branch scaffolding).
