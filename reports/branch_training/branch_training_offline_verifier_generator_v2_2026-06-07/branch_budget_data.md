# Branch-Budget & Direct-Answer Views (Part F)

BRANCH_BUDGET_DATA_VERDICT = BRANCH_BUDGET_READY
Data-composition repair for the v1 branch-heavy SFT — teaches *when to answer directly* vs *when to branch*.

## View counts
- direct_answer_sft: 21501
- one_branch_sft: 17878
- multi_branch_sft: 17946
- branch_budget_policy: 12065
- overbranch_negative_pairs: 3585
- underbranch_negative_pairs: 17932

direct/one-branch vs multi-branch fraction = **0.687** (≥0.4 target; v1 was branch-heavy).

## Branch-budget label distribution (balanced, capped per class)
- TWO_BRANCH: 3500
- FOUR_BRANCH: 3500
- DIRECT: 3500
- EIGHT_BRANCH: 2305
- DEFER_TO_EXTERNAL_SEARCH: 2077
- ONE_BRANCH: 14

Raw (pre-balance): {'TWO_BRANCH': 4114, 'FOUR_BRANCH': 28063, 'EIGHT_BRANCH': 2305, 'DEFER_TO_EXTERNAL_SEARCH': 2077, 'DIRECT': 6427, 'ONE_BRANCH': 14}.

**Caveat (honest):** budget labels come from *constructed-pool* pass-rate — a weak difficulty proxy that clusters at FOUR_BRANCH, so the view is class-capped to avoid a classifier that always says FOUR_BRANCH (which would re-encourage overbranching). The authoritative difficulty signal is *base-model success* (canary / online RL), folded in at K/training time.
Reasoning targets from E rendered logic + gen pools; budget signal from labeled pools; alignment from corecontent_v2.
