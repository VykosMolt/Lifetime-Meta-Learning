# Data-Hygiene Note — empty-prompt rows in Part G offline preference views (2026-06-10)

DPO_EMPTY_PROMPT_VERDICT = EMPTY_PROMPTS_SKIPPED_AT_LOAD

Found during arm `C_offline_verifier_dpo` (Part J): training crashed at DPO step ~3 with
`RuntimeError: embedding got FloatTensor indices` — an empty prompt tokenizes to an empty
tensor whose default float dtype poisons the concatenated `input_ids`.

Counts (prompt empty after strip):

| view | rows | empty-prompt |
| --- | ---: | ---: |
| branch_set_dpo | 66255 | 7734 |
| branch_set_rejection_sft | 37504 | 7743 |
| generator_reward_groups | 32496 | 9 |

Affected rows are coding-domain `pair=oracle` branch-set pairs whose task prompt was lost
during Part G pair construction; chosen/rejected render the branch sets but condition on
nothing, so they are semantically invalid for preference training regardless of the crash.

Decision: skip at load (strip-based filters in `train_offline_branch_generator_v2.py` for
both DPO and SFT record loaders). Effective DPO pool ≈ 58.5k pairs. Upstream Part G fix /
regeneration deferred; `generator_reward_groups` (9 rows) only matters for the gated online
RL and is noted, not fixed. Arms A/B were unaffected (SFT loader already rejected `""`).
