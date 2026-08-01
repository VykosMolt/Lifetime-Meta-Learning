# Branch Pool Generation (Part E)

BRANCH_POOL_GENERATION_VERDICT = LOGIC_BRANCH_POOLS_READY

Cheap candidate pools: 33350 groups (23017 logic), 114824 branches, avg 3.44/group, 33350 reward-diverse. Model-generated branch pools are produced by the resumable/pausable `gen` entry point (logic-priority, bounded).

## By domain

| domain | groups |
| --- | --- |
| logic | 23017 |
| math | 3000 |
| alignment | 3000 |
| reasoning | 2600 |
| coding | 1733 |
## By split

| split | groups |
| --- | --- |
| heldout | 12082 |
| train | 18888 |
| val | 2380 |
