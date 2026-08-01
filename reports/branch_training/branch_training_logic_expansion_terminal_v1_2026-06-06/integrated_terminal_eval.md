# Integrated DualAnchor + CoreContent Terminal Eval (Part H / Experiment 1)

INTEGRATED_DUALANCHOR_CORECONTENT_TERMINAL_VERDICT = CORECONTENT_IMPROVES_TERMINAL

Survivor-set handoff = top-5 by DualAnchor; selectors rank within survivors; external labels only. Survivor oracle retention (macro): 1.0.

## Selector macro top1 (within DualAnchor survivors, heldout)

- oracle: 1.0000
- CoreContent_v2_blockwise: 0.6584
- mixedhead_MIX_HH_OBJECTIVE: 0.5525
- MIX_OBJECTIVE_ALL: 0.3833
- DualAnchor_forced_top1: 0.3787
- DualAnchor_terminal: 0.3787
- MIX_CODE_REASONING: 0.3640
- random_survivor: 0.2934

CoreContent_v2 0.6584 vs best alt selector 0.5525 vs DualAnchor forced top1 0.3787. Logic: CoreContent_v2 0.4464 vs HH 0.3527.

| selector | domain | top1_oracle | groups |
| --- | --- | --- | --- |
| CoreContent_v2_blockwise | alignment | 0.5944 | 2954 |
| CoreContent_v2_blockwise | coding | 0.9146 | 199 |
| CoreContent_v2_blockwise | logic | 0.4464 | 224 |
| CoreContent_v2_blockwise | math | 0.6505 | 309 |
| CoreContent_v2_blockwise | reasoning | 0.6858 | 226 |
| DualAnchor_forced_top1 | alignment | 0.5328 | 2954 |
| DualAnchor_forced_top1 | coding | 0.4271 | 199 |
| DualAnchor_forced_top1 | logic | 0.2366 | 224 |
| DualAnchor_forced_top1 | math | 0.3074 | 309 |
| DualAnchor_forced_top1 | reasoning | 0.3894 | 226 |
| DualAnchor_terminal | alignment | 0.5328 | 2954 |
| DualAnchor_terminal | coding | 0.4271 | 199 |
| DualAnchor_terminal | logic | 0.2366 | 224 |
| DualAnchor_terminal | math | 0.3074 | 309 |
| DualAnchor_terminal | reasoning | 0.3894 | 226 |
| MIX_CODE_REASONING | alignment | 0.5003 | 2954 |
| MIX_CODE_REASONING | coding | 0.4171 | 199 |
| MIX_CODE_REASONING | logic | 0.2366 | 224 |
| MIX_CODE_REASONING | math | 0.2945 | 309 |
| MIX_CODE_REASONING | reasoning | 0.3717 | 226 |
| MIX_OBJECTIVE_ALL | alignment | 0.4973 | 2954 |
| MIX_OBJECTIVE_ALL | coding | 0.4472 | 199 |
| MIX_OBJECTIVE_ALL | logic | 0.2455 | 224 |
| MIX_OBJECTIVE_ALL | math | 0.3107 | 309 |
| MIX_OBJECTIVE_ALL | reasoning | 0.4159 | 226 |
| mixedhead_MIX_HH_OBJECTIVE | alignment | 0.5342 | 2954 |
| mixedhead_MIX_HH_OBJECTIVE | coding | 0.6131 | 199 |
| mixedhead_MIX_HH_OBJECTIVE | logic | 0.3527 | 224 |
| mixedhead_MIX_HH_OBJECTIVE | math | 0.5858 | 309 |
| mixedhead_MIX_HH_OBJECTIVE | reasoning | 0.6770 | 226 |
| oracle | alignment | 1.0000 | 2954 |
| oracle | coding | 1.0000 | 199 |
| oracle | logic | 1.0000 | 224 |
| oracle | math | 1.0000 | 309 |
| oracle | reasoning | 1.0000 | 226 |
| random_survivor | alignment | 0.5115 | 2954 |
| random_survivor | coding | 0.2563 | 199 |
| random_survivor | logic | 0.2679 | 224 |
| random_survivor | math | 0.1748 | 309 |
| random_survivor | reasoning | 0.2566 | 226 |
