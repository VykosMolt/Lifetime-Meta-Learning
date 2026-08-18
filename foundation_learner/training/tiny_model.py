"""Tiny NONSCIENTIFIC Ouro model for mechanics tests (contract §18, §23).

The tiny model is built from the FROZEN CHECKPOINT'S OWN
``configuration_ouro.py`` / ``modeling_ouro.py`` code (loaded through
``trust_remote_code`` from the checkpoint directory) with a tiny configuration
and random initialisation.  NO checkpoint weights are read: only ``config.json``
and the two ``.py`` files are touched, so unit tests never materialise the
2.6 B-parameter backbone.

Frozen tiny configuration (contract §18): hidden 64, 2 layers, 2 UT steps.
Additional choices made here and documented as implementation decisions:

* ``intermediate_size = 128`` and ``num_attention_heads = num_key_value_heads =
  2`` (head_dim 32) — the smallest shapes that keep the architecture's
  structure (GQA groups of 1, SwiGLU MLP) intact.
* ``vocab_size`` defaults to the REAL frozen tokenizer's vocabulary
  (49 152).  Rationale: tokenisation exactness tests must run against the real
  tokenizer, and the trainer/tokenizer/objective plumbing is only meaningful
  end-to-end if token ids from the real tokenizer are valid model inputs.  The
  cost is ~6.4 M parameters, which is still trivially CPU-trainable.  A smaller
  ``vocab_size`` may be passed explicitly for pure-mechanics tests; pass
  ``attach_tokenizer=False`` as well, since the real tokenizer would then emit
  out-of-range ids.
* dtype float32 on CPU (bf16 CPU matmuls are slow and not the object of study).

Every bundle produced here carries ``identity["scientific"] = False`` and
``identity["NONSCIENTIFIC"] = True``; ``ModelBundle.assert_scientific()`` fails
on it by construction (contract §20: "the tiny model is for mechanics tests
only and is clearly non-scientific").
"""

from __future__ import annotations

from typing import Any

from foundation_learner.training.model_loading import (
    FROZEN_CHECKPOINT_DIR,
    FROZEN_EARLY_EXIT_THRESHOLD,
    FROZEN_RLTT_LOGPROB_CHUNK_SIZE,
    ModelBundle,
    assert_transformers_version,
    build_identity,
    load_tokenizer,
)

TINY_HIDDEN_SIZE = 64
TINY_INTERMEDIATE_SIZE = 128
TINY_NUM_LAYERS = 2
TINY_NUM_HEADS = 2
TINY_TOTAL_UT_STEPS = 2
TINY_MAX_POSITION_EMBEDDINGS = 4096

_TOKENIZER_CACHE: dict[str, Any] = {}


def _cached_tokenizer(checkpoint_dir: str):
    if checkpoint_dir not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[checkpoint_dir] = load_tokenizer(checkpoint_dir)
    return _TOKENIZER_CACHE[checkpoint_dir]


def build_tiny_config(
    checkpoint_dir: str = FROZEN_CHECKPOINT_DIR,
    *,
    vocab_size: int,
    total_ut_steps: int = TINY_TOTAL_UT_STEPS,
):
    """Instantiate the checkpoint's own ``OuroConfig`` class with tiny shapes."""
    from transformers import AutoConfig

    assert_transformers_version()
    base = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True, local_files_only=True
    )
    cfg = type(base)(
        vocab_size=vocab_size,
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=TINY_INTERMEDIATE_SIZE,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=TINY_NUM_HEADS,
        num_key_value_heads=TINY_NUM_HEADS,
        hidden_act=base.hidden_act,
        max_position_embeddings=TINY_MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=base.rms_norm_eps,
        rope_theta=base.rope_theta,
        rope_scaling=base.rope_scaling,
        use_sliding_window=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        total_ut_steps=total_ut_steps,
        early_exit_threshold=FROZEN_EARLY_EXIT_THRESHOLD,
        rltt_logprob_chunk_size=FROZEN_RLTT_LOGPROB_CHUNK_SIZE,
        rltt_loop_level_checkpointing=True,
        bos_token_id=base.bos_token_id,
        eos_token_id=base.eos_token_id,
        pad_token_id=base.pad_token_id,
        attn_implementation="eager",
    )
    # Keep the remote-code binding so `from_config` resolves modeling_ouro.py
    # from the frozen checkpoint directory (not from any packaged copy).
    cfg.auto_map = dict(base.auto_map)
    cfg._name_or_path = checkpoint_dir
    cfg.architectures = ["OuroForCausalLM"]
    return cfg


def build_tiny_model(
    seed: int,
    device: str = "cpu",
    *,
    checkpoint_dir: str = FROZEN_CHECKPOINT_DIR,
    vocab_size: int | None = None,
    attach_tokenizer: bool = True,
    total_ut_steps: int = TINY_TOTAL_UT_STEPS,
) -> ModelBundle:
    """Contract §23: ``build_tiny_model(seed, device="cpu") -> ModelBundle``.

    Determinism: initialisation runs inside ``torch.random.fork_rng`` seeded
    with ``seed``, so the parameters are a pure function of ``seed`` (given the
    torch version) and the caller's global RNG state is left untouched.
    """
    import torch
    from transformers import AutoModelForCausalLM

    assert_transformers_version()
    tokenizer = None
    if vocab_size is None or attach_tokenizer:
        tokenizer = _cached_tokenizer(checkpoint_dir)
    if vocab_size is None:
        vocab_size = len(tokenizer)

    cfg = build_tiny_config(
        checkpoint_dir, vocab_size=vocab_size, total_ut_steps=total_ut_steps
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    model = model.to(device=device, dtype=torch.float32)

    identity = build_identity(
        kind="TINY_NONSCIENTIFIC",
        scientific=False,
        checkpoint_dir=checkpoint_dir,
        tree_sha256=None,
        tree_hash_verified=False,
        config=model.config,
        dtype="float32",
        device=device,
        core_extra={"tiny_init_seed": int(seed)},
        extra={
            "seed": int(seed),
            "code_source": "frozen checkpoint configuration_ouro.py / modeling_ouro.py",
            "weights_source": "RANDOM_INIT (no checkpoint weights read)",
            "purpose": "mechanics tests only (contract §18)",
        },
    )
    return ModelBundle(
        model=model, tokenizer=tokenizer, device=device, identity=identity
    )
