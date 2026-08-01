# DualAnchor Teacher Traces (Part J)

DUALANCHOR_TEACHER_TRACE_VERDICT = TEACHER_MISMATCH_ON_LOGIC

DualAnchor is a branch-POLICY teacher (keep/prune/rescue/defer/rank). These are NOT correctness labels; external verifiers remain the only ground truth. Teacher quality = oracle_retention vs RANDOM pruning (lift). keep_vs_correctness_divergence is expected to be high (keep != correctness, by design).

Traces: 13667. Overall: oracle_retention 0.9064 vs random 0.861 (lift 0.0454); prune_rate 0.139, rescue_rate 0.1367, defer_rate 0.1755, false_prune_rate 0.0936.

## By domain (oracle_retention vs random)

| domain | prune | rescue | defer | oracle_retention | random_keep_retention | false_prune |
| --- | --- | --- | --- | --- | --- | --- |
| coding | 0.2290 | 0.2088 | 0.2516 | 0.8797 | 0.7710 | 0.1203 |
| reasoning | 0.2089 | 0.2092 | 0.2053 | 0.9012 | 0.7911 | 0.0988 |
| math | 0.2005 | 0.2005 | 0.2830 | 0.8487 | 0.7995 | 0.1513 |
| logic | 0.2500 | 0.2500 | 0.1903 | 0.7788 | 0.7500 | 0.2212 |
| alignment | 0.0000 | 0.0000 | 0.0716 | 1.0000 | 1.0000 | 0.0000 |
