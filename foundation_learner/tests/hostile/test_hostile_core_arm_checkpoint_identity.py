"""HOSTILE: core arms start from different base checkpoints.

Attack: run one arm from the frozen checkpoint and another from a different
(or already-trained, or tampered) starting point, so the "comparison" measures
the starting weights rather than the objective.

Required behaviour: every arm records the base identity (checkpoint tree hash,
config digest, architecture, recurrence pinning) in every checkpoint it writes;
a set of core-arm checkpoints whose base identities disagree is REFUSED, and a
resume across a different base identity is refused too.
Contract §1, §13, §15, §18.
"""

from __future__ import annotations

import os

import pytest
import torch

from foundation_learner.training import checkpointing as cp
from foundation_learner.training import model_loading as ml
from foundation_learner.training.arms import CORE_ARMS


class _Cfg:
    """Minimal config stand-in for identity construction."""

    def __init__(self, **kw):
        self.model_type = "ouro"
        self.architectures = ["OuroForCausalLM"]
        self.hidden_size = 2048
        self.num_hidden_layers = 48
        self.num_attention_heads = 16
        self.total_ut_steps = 4
        self.early_exit_threshold = 1.0
        self.vocab_size = 49152
        self.__dict__.update(kw)

    def to_dict(self):
        return {
            k: v for k, v in self.__dict__.items() if not k.startswith("_")
        }


def _identity(tree_sha256=ml.FROZEN_CHECKPOINT_TREE_SHA256, **cfg_overrides):
    return ml.build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir="/artifacts/ouro_rltt_local",
        tree_sha256=tree_sha256,
        tree_hash_verified=True,
        config=_Cfg(**cfg_overrides),
        dtype="bfloat16",
        device="cuda",
    )


def assert_core_arms_share_a_base(manifests):
    """The check a campaign runner must apply before comparing arms."""
    hashes = {m["arm_config"]["arm_id"]: m["base_identity_hash"] for m in manifests}
    trees = {
        m["arm_config"]["arm_id"]: m["base_checkpoint_tree_sha256"] for m in manifests
    }
    if len(set(hashes.values())) != 1 or len(set(trees.values())) != 1:
        raise AssertionError(
            "REFUSED: core arms did not start from one identical frozen "
            f"checkpoint; identity hashes {hashes}, tree hashes {trees}"
        )


def _write(tmp_path, arm_id, identity):
    return cp.save_checkpoint(
        str(tmp_path / arm_id),
        "final",
        trained_state={"w": torch.zeros(2)},
        trained_scope="PEFT_MODE",
        step=600,
        arm_config={"arm_id": arm_id},
        arm_config_hash=f"hash-{arm_id}",
        base_identity=identity,
    )


def test_identical_base_identity_passes(tmp_path):
    identity = _identity()
    manifests = [_write(tmp_path, arm, identity) for arm in CORE_ARMS]
    assert_core_arms_share_a_base(manifests)
    assert {m["base_checkpoint_tree_sha256"] for m in manifests} == {
        ml.FROZEN_CHECKPOINT_TREE_SHA256
    }


def test_different_checkpoint_tree_hash_is_refused(tmp_path):
    good = _identity()
    tampered = _identity(tree_sha256="0" * 64)
    manifests = [
        _write(tmp_path, "FL1", good),
        _write(tmp_path, "FL2", good),
        _write(tmp_path, "FL3", tampered),
    ]
    with pytest.raises(AssertionError) as exc:
        assert_core_arms_share_a_base(manifests)
    assert "REFUSED" in str(exc.value)


def test_different_recurrence_pinning_is_a_different_base(tmp_path):
    good = _identity()
    unpinned = _identity(total_ut_steps=1)
    assert good["identity_hash"] != unpinned["identity_hash"]
    manifests = [
        _write(tmp_path, "FL1", good),
        _write(tmp_path, "FL2", unpinned),
        _write(tmp_path, "FL3", good),
    ]
    with pytest.raises(AssertionError):
        assert_core_arms_share_a_base(manifests)


def test_different_early_exit_threshold_is_a_different_base():
    assert _identity()["identity_hash"] != _identity(early_exit_threshold=0.5)[
        "identity_hash"
    ]


def test_nonscientific_model_can_never_masquerade_as_the_frozen_backbone():
    tiny = ml.build_identity(
        kind="TINY_NONSCIENTIFIC",
        scientific=False,
        checkpoint_dir="/artifacts/ouro_rltt_local",
        tree_sha256=None,
        tree_hash_verified=False,
        config=_Cfg(hidden_size=64, num_hidden_layers=2, total_ut_steps=2),
        dtype="float32",
        device="cpu",
    )
    assert tiny["scientific"] is False
    assert tiny["NONSCIENTIFIC"] is True
    assert tiny["identity_hash"] != _identity()["identity_hash"]
    bundle = ml.ModelBundle(model=None, tokenizer=None, device="cpu", identity=tiny)
    with pytest.raises(ml.FrozenBindingError):
        bundle.assert_scientific()


@pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory not present",
)
def test_the_recorded_base_identity_matches_the_real_frozen_binding():
    """The identity a real load would record must match contract §1 exactly."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        ml.FROZEN_CHECKPOINT_DIR, trust_remote_code=True, local_files_only=True
    )
    ml.assert_frozen_architecture(config)
    identity = ml.build_identity(
        kind="FROZEN_BACKBONE",
        scientific=True,
        checkpoint_dir=ml.FROZEN_CHECKPOINT_DIR,
        tree_sha256=ml.FROZEN_CHECKPOINT_TREE_SHA256,
        tree_hash_verified=False,
        config=config,
        dtype="bfloat16",
        device="cuda",
    )
    assert identity["num_hidden_layers"] == 48
    assert identity["total_ut_steps"] == 4
    assert identity["early_exit_threshold"] == 1.0
    assert identity["vocab_size"] == 49152
    assert identity["architecture"] == "OuroForCausalLM"
    assert identity["checkpoint_tree_sha256"] == ml.FROZEN_CHECKPOINT_TREE_SHA256
