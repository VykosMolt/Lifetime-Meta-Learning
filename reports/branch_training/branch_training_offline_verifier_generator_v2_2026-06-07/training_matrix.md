# Offline Training Matrix (Part I)

TRAINING_MATRIX_VERDICT = MATRIX_REDUCED
Configs written for all 9 arms under `artifacts/models/branch_training_offline_verifier_generator_v2/configs/`; **5 ON** (resource-reduced), rest OFF.

## Arms
- ✅ **A_format_control** — reproduce v1 branch-format SFT (control) (init base, losses ['sft'], 300 steps)
- ✅ **B_direct_budget_sft** — fix overbranching via direct-answer + budget (init base, losses ['sft', 'budget_cls'], 400 steps)
- ✅ **C_offline_verifier_dpo** — MAIN generator-improvement arm (verifier prefs) (init base, losses ['sft', 'dpo'], 500 steps)
- ⬜ **D_logic_rendered_verifier** — spend the logic substrate (init base, losses ['sft', 'dpo'], 500 steps)
- ⬜ **E_teacher_coding_reasoning** — test teacher where v1 lift was real (coding/reasoning) (init base, losses ['teacher_policy'], 300 steps)
- ⬜ **F_teacher_plus_verifier** — does teacher add after verifier signal? (init base, losses ['sft', 'dpo', 'teacher_policy'], 500 steps)
- ✅ **G_budget_plus_verifier** — improve reachability WITHOUT overbranching (preferred) (init base, losses ['sft', 'dpo', 'budget_cls'], 500 steps)
- ✅ **H_from_base_best_recipe** — best recipe from clean base init (preferred) (init base, losses ['sft', 'dpo', 'budget_cls'], 600 steps)
- ⬜ **I_continue_previous_sft** — can the v1 300-step adapter be rescued? (init prev_sft, losses ['sft', 'dpo'], 300 steps)

## Domain-gated teacher weights (when teacher on)
- coding: 0.7
- reasoning: 0.5
- math: 0.25
- logic: 0.0
- alignment: 0.3

Common: bf16 LoRA (r=16, all-linear), grad-checkpointing, base/tokenizer untouched, adapters separate, no heldout training, no online generation, canary eval every 200 steps, STOP-pausable, save every 100.

Honest scope: one 12GB GPU + slow generation precludes converging all 9; the ON set covers the spec's preferred candidates (C verifier-DPO, G budget+verifier, H best-recipe) plus the A/B controls.
