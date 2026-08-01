# External Verifier Labeling (Part F)

EXTERNAL_VERIFIER_LABELING_VERDICT = LOGIC_LABELS_READY

All labels are external (dataset keys / parsers / z3 / executed unit tests). DualAnchor/CoreContent are NOT used for correctness here.

Overall: {'groups': 33590, 'positive_oracle_rate': 0.9981, 'reward_diverse_rate': 0.995, 'all_wrong_rate': 0.0019, 'all_correct_rate': 0.0031, 'parse_ok_rate': 0.9996, 'ambiguous_rate': 0.0001}

## By domain

| domain | groups | positive_oracle_rate | reward_diverse_rate | all_wrong_rate | parse_ok_rate |
| --- | --- | --- | --- | --- | --- |
| alignment | 3000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |
| coding | 1793 | 0.9810 | 0.9743 | 0.0190 | 0.9957 |
| logic | 23077 | 0.9993 | 0.9985 | 0.0007 | 1.0000 |
| math | 3060 | 0.9967 | 0.9850 | 0.0033 | 0.9999 |
| reasoning | 2660 | 0.9989 | 0.9846 | 0.0011 | 0.9997 |

## By logic family

| family | groups | positive_oracle_rate | reward_diverse_rate | all_wrong_rate |
| --- | --- | --- | --- | --- |
| fol_entailment | 398 | 0.9975 | 0.9899 | 0.0025 |
| logical_entailment | 1954 | 0.9969 | 0.9969 | 0.0031 |
| lsat_analytical_reasoning | 230 | 1.0000 | 1.0000 | 0.0000 |
| lsat_logical_reasoning | 510 | 1.0000 | 1.0000 | 0.0000 |
| mcq_logical_reading | 4895 | 1.0000 | 0.9998 | 0.0000 |
| proofwriter_deduction | 6702 | 0.9997 | 0.9994 | 0.0003 |
| ruletaker_deduction | 1384 | 0.9971 | 0.9957 | 0.0029 |
| synthetic_constraint_game | 1159 | 0.9974 | 0.9974 | 0.0026 |
| synthetic_fol | 1984 | 1.0000 | 0.9970 | 0.0000 |
| synthetic_propositional | 3861 | 1.0000 | 0.9990 | 0.0000 |

Generated-pool coverage so far: 240 groups ({'groups': 240, 'positive_oracle_rate': 0.7375, 'reward_diverse_rate': 0.3042, 'all_wrong_rate': 0.2625, 'all_correct_rate': 0.4333, 'parse_ok_rate': 0.9552, 'ambiguous_rate': 0.0073}).
