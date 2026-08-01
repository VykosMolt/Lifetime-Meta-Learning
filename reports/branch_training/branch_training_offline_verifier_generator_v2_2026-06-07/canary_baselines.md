# Canary Baselines (Part D)

CANARY_BASELINE_VERDICT = PREVIOUS_SFT_BEHAVIOR_CONFIRMED
Pre-training fixed baselines on the 610-group canary. positive_oracle@K (alignment = preference accuracy). Deltas are prev-SFT − base; labeled FLAG_* (not MEASURED_*) until paired CIs exclude zero.

| domain | n_base | base | prev_sft | delta | flag | base_parse | base_div | small_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coding | 100 | 0.5100 | 0.5300 | 0.0200 | FLAG_UP | 0.8525 | 2.3600 | False |
| reasoning | 110 | 0.8818 | 0.8727 | -0.0091 | FLAG_DOWN | 1.0000 | 2.3640 | False |
| math | 110 | 0.9273 | 0.8727 | -0.0546 | FLAG_DOWN | 0.9966 | 2.3360 | False |
| logic | 180 | 0.8167 | 0.8667 | 0.0500 | FLAG_UP | 0.9958 | 4.0000 | False |
| alignment | 110 | 0.5364 | 0.5273 | -0.0091 | FLAG_DOWN |  |  |  |

External stack reference (from v1 Experiment-1): CoreContent_v2 within DualAnchor survivors 0.658 vs DualAnchor forced-top1 0.379; feature extraction on the canary deferred to Part L.

Generation: batched (num_return_sequences=K) for logic/reasoning/coding, single-seq for math (KV guard); resumable per-variant; STOP-pausable.
