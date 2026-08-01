# Branch Training + Logic Expansion + Terminal v1 — Run State

BRANCH_TRAINING_LOGIC_EXPANSION_INIT_VERDICT = READY

Goal: move from external branch selection (DualAnchor + CoreContent_v2) toward **model-internal branching**, with **logic** as the priority expansion. DualAnchor is a *teacher*, not a permanent crutch; correctness comes ONLY from external verifiers.

## Locked selected state

- **branch_survival**: {"policy": "DualAnchor", "anchors": ["MIX_CODE_REASONING", "MIX_OBJECTIVE_ALL"], "status": "unchanged (teacher + external baseline)"}
- **content_final_selection**: {"policy": "CoreContent_v2_blockwise_pruned_24_36", "prev_baseline": "mixedhead_MIX_HH_OBJECTIVE (beaten on expanded heldout)", "role": "terminal-ranking baseline + optional soft teacher; NOT ground truth"}
- **terminal**: top5/full survivor-set handoff (locked); content selector ranks within survivors; no unconditional top1
- **science**: diagnostic only (never a promotion gate)
- **steering**: not run, not claimed
- **priority_domain**: logic (largest expansion; weakest absolute core domain in v2)
- **training**: allowed ONLY under artifacts/models/branch_training_logic_expansion_v1/ (new adapters/LoRA/SFT/RL)
- **label_policy**: external verifiers ONLY for correctness; DualAnchor/CoreContent are policy/soft teachers, never correctness labels

## Inputs present

- ouro_model: True
- corecontent_v2_policy_pt: True
- corecontent_v2_features: True
- pure_content_taps: True
- dualanchor_arch_loop_common: True
- core_tap_audit_root: True

Training libs (peft/trl): True; z3 solver: True; GPU: True.

## Do not overwrite

- `constructed_taps/pure_content_taps.pt`
- `constructed_taps/transplanted_taps.pt`
- `artifacts/reports/probes/bg_corecontent_dataset_expansion_refit_v2_2026-06-04/`
- `data/corecontent_v2/`
- `base Ouro checkpoint`
- `tokenizer`
- `any tap registry`
