# Branch Pool Reachability (Part I / Experiment 3)

BRANCH_POOL_REACHABILITY_VERDICT = LOGIC_REACHABILITY_READY

Model-generated pools (240 groups): {'groups': 240, 'positive_oracle@1': 0.5969, 'positive_oracle@2': 0.6813, 'positive_oracle@4': 0.7375, 'reward_diverse_rate': 0.3042, 'all_wrong_rate': 0.2625, 'all_correct_rate': 0.4333, 'parse_ok_rate': 0.9552, 'avg_branches': 4.0, 'strategy_diversity': 2.0}. No selector can rescue a pool with no good branch; pools reach oracle often: selection/self-selection is the lever.

## Model-generated reachability by domain

| domain | groups | positive_oracle@1 | positive_oracle@4 | all_wrong_rate | parse_ok_rate | avg_branches |
| --- | --- | --- | --- | --- | --- | --- |
| coding | 60 | 0.3208 | 0.4333 | 0.5667 | 0.8500 | 4.0000 |
| logic | 60 | 0.5167 | 0.7333 | 0.2667 | 0.9917 | 4.0000 |
| math | 60 | 0.7375 | 0.8333 | 0.1667 | 0.9917 | 4.0000 |
| reasoning | 60 | 0.8125 | 0.9500 | 0.0500 | 0.9875 | 4.0000 |

## By logic family (generated)

| family | groups | positive_oracle@1 | positive_oracle@4 | all_wrong_rate |
| --- | --- | --- | --- | --- |
| fol_entailment | 7 | 0.6786 | 0.8571 | 0.1429 |
| logical_entailment | 7 | 0.0357 | 0.1429 | 0.8571 |
| mcq_logical_reading | 7 | 0.6071 | 1.0000 | 0.0000 |
| proofwriter_deduction | 8 | 0.4688 | 0.7500 | 0.2500 |
| ruletaker_deduction | 7 | 0.3571 | 0.4286 | 0.5714 |
| synthetic_constraint_game | 8 | 0.2812 | 0.6250 | 0.3750 |
| synthetic_fol | 8 | 0.9062 | 1.0000 | 0.0000 |
| synthetic_propositional | 8 | 0.7500 | 1.0000 | 0.0000 |

Cheap dataset/option pools (control, gold present): positive_oracle@4 0.9629.
