"""HOSTILE: the FL7 fast adapter must reset and stay inside its norm bound.

Attacks (contract §18: "fast adapter not reset / exceeding norm bound"):

1. Drive the inner loop with a large learning rate over several items and
   check that the global trainable-delta Frobenius norm NEVER exceeds
   beta = 1.0 after an item, and that the clip really bit (pre-clip > beta).
2. Skip ``reset_episode`` between two episodes and prove the delta would have
   carried, then prove the real per-episode reset returns it to EXACT zero.
3. Verify that no base parameter changed, bit for bit, at any point.
4. Verify that the recorded checkpoint contains only adapter tensors.
"""
from __future__ import annotations

import pytest
import torch

from foundation_learner.mechanisms.fast_adapter import FastAdapter
from foundation_learner.tests._w3_support import fresh_tiny_bundle, real_episode
from foundation_learner.training.peft_modes import FL7_FAST_DELTA_MAX_FROBENIUS

BETA = FL7_FAST_DELTA_MAX_FROBENIUS

#: ``clip_lora_frobenius_`` rescales the float32 ``B`` factors in place, so the
#: RE-measured norm carries float32 rounding (relative ~1e-8 here; float32 eps
#: is 1.2e-7).  The tolerance is that rounding, not slack in the bound: the
#: pre-clip norms in these attacks exceed beta by three to four ORDERS of
#: magnitude, so a real violation could never hide inside it.
NORM_TOLERANCE = 1e-6 * BETA


def assert_within_bound(adapter, tolerance: float = NORM_TOLERANCE) -> None:
    norm = adapter.delta_norm()
    assert norm <= BETA + tolerance, f"fast delta norm {norm} exceeds beta {BETA}"


def test_large_inner_updates_stay_inside_the_frozen_bound():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=91, inner_lr=50.0,
                          inner_steps=4)
    records = adapter.adapt_episode(real_episode(reveal=True))
    assert len(records) >= 2
    assert any(r.pre_clip_norm > 100.0 * BETA for r in records), (
        "the attack did not even reach the bound; it proves nothing")
    for record in records:
        assert record.post_clip_norm <= BETA + NORM_TOLERANCE
    assert_within_bound(adapter)


def test_bound_holds_across_repeated_episodes_without_reset():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=92, inner_lr=50.0,
                          inner_steps=2)
    for index in range(3):
        adapter.adapt_episode(real_episode(index=index, reveal=True))
        assert_within_bound(adapter)


def test_skipping_the_reset_would_carry_and_the_reset_removes_it():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=93, inner_steps=1)
    adapter.adapt_episode(real_episode(reveal=True))
    carried = adapter.delta_norm()
    assert carried > 0.0
    # no reset -> the delta is still there (this is what the attack exploits)
    assert adapter.delta_norm() == carried
    adapter.reset_episode()
    assert adapter.delta_norm() == 0.0
    for name in adapter.handles.names():
        layer = adapter.handles.layers[name]
        assert torch.count_nonzero(layer.trainable_delta()) == 0


def test_the_bound_assertion_has_teeth():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=94, inner_steps=1)
    adapter.adapt_episode(real_episode(reveal=True))
    with torch.no_grad():                       # forge an over-long delta
        for name in adapter.handles.names():
            adapter.handles.layers[name].lora_B.mul_(1e6)
    with pytest.raises(AssertionError):
        assert_within_bound(adapter)


def test_base_weights_are_never_written():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=95, inner_lr=50.0,
                          inner_steps=4)
    before = adapter.base_fingerprint()
    for index in range(2):
        adapter.adapt_episode(real_episode(index=index, reveal=True))
        assert adapter.base_fingerprint() == before
        adapter.reset_episode()
    assert adapter.base_fingerprint() == before
    for name, param in adapter.model.named_parameters():
        if not name.endswith((".lora_A", ".lora_B")):
            assert param.requires_grad is False, name


def test_checkpoint_holds_adapter_tensors_only():
    adapter = FastAdapter(fresh_tiny_bundle(), seed=96, inner_steps=1)
    adapter.adapt_episode(real_episode(reveal=True))
    state = adapter.state_dict()
    assert set(state) == {"schema", "config", "config_hash", "lora"}
    assert state["lora"]
    for key in state["lora"]:
        assert key.endswith((".lora_A", ".lora_B")) or ".merged_" in key
