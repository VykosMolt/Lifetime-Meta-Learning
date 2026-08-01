# Branching SFT (Part L) — pausable/resumable LoRA

BRANCHING_SFT_VERDICT = SFT_TRAINED

bf16 base + LoRA(all-linear), gradient-checkpointed (4-bit only as OOM fallback via BNB_4BIT=1); base Ouro unmodified — same precision used everywhere else for a clean trained-vs-base eval. Examples: 594. global_step 300/300 (this run capped at 300); train_loss 0.3444545237223307. Reached target.

Adapter: `artifacts/models/branch_training_logic_expansion_v1/branching_sft`. STOP sentinel: `/home/moloch/ouro_project/artifacts/models/branch_training_logic_expansion_v1/branching_sft/STOP`.
