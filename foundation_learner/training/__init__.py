"""Foundation Learner B200 V0 — training core (worker W2).

Modules
-------
``model_loading``      frozen-backbone loading recipe (contract §1/§22/§23)
``tiny_model``         tiny NONSCIENTIFIC OuroForCausalLM for mechanics tests (§18)
``lora``               custom LoRA implementation (Amendment 3; no peft at runtime)
``peft_modes``         PEFT_MODE / FULL_MODEL_MODE trainable-parameter modes (§8)
``tokenization``       episode -> {input_ids, loss_mask, spans_meta} per arm (§7/§23)
``objectives``         weighted-NLL objectives, incl. FL3 per-episode token mean (§7)
``arms``               frozen ArmConfig registry + compute-matching assertions (§8)
``trainer``            the single training loop used by every arm (§8/§17/§15)
``compute_accounting`` per-arm compute ledger, ``flb200.compute_ledger.v1`` (§8)
``checkpointing``      atomic checkpoints + bitwise-resumable RNG state (§15)
``stability``          §17 detectors and the bounded response policy

Nothing in this package ever calls ``model.generate``; evaluation-side decoding
is a manual greedy loop owned by W4 (contract §22).
"""

TRAINING_CORE_VERSION = "0.1.0"

__all__ = ["TRAINING_CORE_VERSION"]
