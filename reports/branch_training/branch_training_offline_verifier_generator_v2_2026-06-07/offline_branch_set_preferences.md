# Offline Branch-Set Preferences (Part G)

OFFLINE_BRANCH_SET_PREFERENCE_VERDICT = OFFLINE_PREFS_READY
- branch_set_rejection_sft: **37504** accepted (verifier-positive, correct-final) branch sets
- branch_set_dpo: **66255** pairs — types {'oracle': 39569, 'overbranch': 4405, 'underbranch_defer': 2379, 'overbranch_F': 3580, 'underbranch_F': 17932}
- generator_reward_groups: **32496** reward records (mean 4.402, range [-0.733, 6.088])

Built offline from E rendered logic + labeled train pools + F contrastive pairs; deterministic reward via `utilities/branch_training/offline_reward_v2.py`. No online generation. Diversity rewarded only when tied to verifier-positive branches; budget pairs penalize overbranching easy / underbranching hard.
