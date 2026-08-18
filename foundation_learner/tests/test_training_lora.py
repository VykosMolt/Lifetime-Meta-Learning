"""Custom LoRA (Amendment 3) — mechanics plus a peft numerical oracle.

``peft`` is NOT available on the B200 (contract §22), so the custom
implementation is authoritative.  Locally, ``peft==0.19.1`` is used as a
numerical oracle: the same A/B injected into both implementations must produce
the same logits.  The oracle test is lazily imported and skipped when peft is
absent.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from foundation_learner.training import model_loading as ml
from foundation_learner.training.lora import (
    LoraError,
    LoraSpec,
    attach_lora,
    clip_lora_frobenius_,
    detach_lora_,
    load_lora_state_,
    lora_enabled,
    lora_frobenius_norm,
    lora_state_dict,
    merge_scaled_,
    set_lora_enabled,
    zero_lora_,
)
from foundation_learner.training.tiny_model import build_tiny_model

pytestmark = pytest.mark.skipif(
    not os.path.isdir(ml.FROZEN_CHECKPOINT_DIR),
    reason="frozen checkpoint directory (code) not present",
)

SPEC = LoraSpec(
    target_suffixes=("q_proj", "v_proj", "down_proj"), rank=4, alpha=8.0, seed=5
)


def _tiny(seed: int = 21):
    return build_tiny_model(seed=seed, vocab_size=64, attach_tokenizer=False)


def _randomise(handles, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in handles.layers.values():
            layer.lora_A.copy_(torch.randn(layer.lora_A.shape, generator=gen) * 0.05)
            layer.lora_B.copy_(torch.randn(layer.lora_B.shape, generator=gen) * 0.05)


def test_spec_validation():
    with pytest.raises(LoraError):
        LoraSpec(target_suffixes=("q_proj",), rank=0, alpha=1.0)
    with pytest.raises(LoraError):
        LoraSpec(target_suffixes=(), rank=4, alpha=8.0)
    with pytest.raises(LoraError):
        LoraSpec(target_suffixes=("q_proj",), rank=4, alpha=8.0, dropout=1.5)
    assert SPEC.scaling == 2.0


def test_attach_targets_exactly_the_named_linears():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    assert handles.names() == [
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.v_proj",
    ]
    assert handles.num_trainable_parameters() == sum(
        p.numel() for p in handles.parameters()
    )
    trainable = [n for n, p in bundle.model.named_parameters() if p.requires_grad]
    assert all(("lora_A" in n or "lora_B" in n) for n in trainable)
    assert len(trainable) == 12


def test_attach_refuses_unmatched_suffixes():
    bundle = _tiny()
    with pytest.raises(LoraError):
        attach_lora(bundle.model, LoraSpec(("no_such_proj",), 4, 8.0))
    with pytest.raises(LoraError):
        attach_lora(bundle.model, LoraSpec(("q_proj", "no_such_proj"), 4, 8.0))


def test_zero_initialised_adapter_is_an_exact_no_op():
    bundle = _tiny()
    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        before = bundle.model(input_ids=ids, use_cache=False).logits.clone()
    handles = attach_lora(bundle.model, SPEC)
    with torch.no_grad():
        after = bundle.model(input_ids=ids, use_cache=False).logits
    assert torch.equal(before, after)
    assert lora_frobenius_norm(handles) == 0.0


def test_enable_disable_and_detach_restore_the_base_forward():
    bundle = _tiny()
    ids = torch.randint(0, 64, (2, 9))
    with torch.no_grad():
        base = bundle.model(input_ids=ids, use_cache=False).logits.clone()
    handles = attach_lora(bundle.model, SPEC)
    _randomise(handles, seed=1)
    with torch.no_grad():
        adapted = bundle.model(input_ids=ids, use_cache=False).logits.clone()
    assert not torch.equal(base, adapted)

    set_lora_enabled(handles, False)
    assert lora_enabled(handles) is False
    with torch.no_grad():
        disabled = bundle.model(input_ids=ids, use_cache=False).logits
    assert torch.equal(base, disabled)

    set_lora_enabled(handles, True)
    detach_lora_(handles)
    with torch.no_grad():
        detached = bundle.model(input_ids=ids, use_cache=False).logits
    assert torch.equal(base, detached)


def test_zero_lora_resets_the_delta_exactly():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    a_before = {n: handles.layers[n].lora_A.clone() for n in handles.names()}
    _randomise(handles, seed=2)
    assert lora_frobenius_norm(handles) > 0
    zero_lora_(handles)
    assert lora_frobenius_norm(handles) == 0.0
    for name in handles.names():
        assert torch.equal(handles.layers[name].lora_A, a_before[name])
        assert torch.count_nonzero(handles.layers[name].lora_B) == 0

    _randomise(handles, seed=3)
    kept = {n: handles.layers[n].lora_A.clone() for n in handles.names()}
    zero_lora_(handles, reinit_a=False)
    assert lora_frobenius_norm(handles) == 0.0
    for name in handles.names():
        assert torch.equal(handles.layers[name].lora_A, kept[name])


def test_frobenius_identity_matches_the_dense_delta():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    _randomise(handles, seed=4)
    dense = 0.0
    for layer in handles.layers.values():
        delta = layer.trainable_delta().detach()
        dense += float((delta * delta).sum())
    assert lora_frobenius_norm(handles) == pytest.approx(math.sqrt(dense), rel=1e-5)


def test_clip_returns_pre_clip_norm_and_enforces_the_bound():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    _randomise(handles, seed=5)
    before = lora_frobenius_norm(handles)
    assert before > 0.5
    returned = clip_lora_frobenius_(handles, 0.5)
    assert returned == pytest.approx(before, rel=1e-9)
    assert lora_frobenius_norm(handles) == pytest.approx(0.5, rel=1e-6)
    # already inside the bound -> unchanged
    state = {k: v.clone() for k, v in lora_state_dict(handles).items()}
    again = clip_lora_frobenius_(handles, 10.0)
    assert again == pytest.approx(0.5, rel=1e-6)
    for key, value in lora_state_dict(handles).items():
        assert torch.equal(value, state[key])
    with pytest.raises(LoraError):
        clip_lora_frobenius_(handles, 0.0)


def test_state_dict_round_trip_is_bitwise():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    _randomise(handles, seed=6)
    state = {k: v.clone() for k, v in lora_state_dict(handles).items()}
    zero_lora_(handles)
    assert lora_frobenius_norm(handles) == 0.0
    load_lora_state_(handles, state)
    for key, value in lora_state_dict(handles).items():
        assert torch.equal(value, state[key])
    with pytest.raises(LoraError):
        load_lora_state_(handles, {"bogus": torch.zeros(1)})


def test_merge_scaled_is_exact_and_leaves_base_weights_untouched():
    src_bundle = _tiny()
    dst_bundle = _tiny(seed=22)
    src = attach_lora(src_bundle.model, SPEC)
    dst = attach_lora(dst_bundle.model, SPEC)
    _randomise(src, seed=7)
    base_before = {
        name: dst.layers[name].base_layer.weight.clone() for name in dst.names()
    }
    trainable_before = {k: v.clone() for k, v in lora_state_dict(dst).items()}

    merged = merge_scaled_(dst, src, 0.5)
    assert merged == len(dst)
    for name in dst.names():
        expected = 0.5 * src.layers[name].trainable_delta().detach()
        assert torch.allclose(dst.layers[name].effective_delta(), expected, atol=1e-7)
        assert torch.equal(dst.layers[name].base_layer.weight, base_before[name])
    for key in trainable_before:
        assert torch.equal(lora_state_dict(dst)[key], trainable_before[key])

    # merged components survive a state-dict round trip
    state = {k: v.clone() for k, v in lora_state_dict(dst).items()}
    assert any(".merged_0." in k for k in state)
    load_lora_state_(dst, state)
    for name in dst.names():
        expected = 0.5 * src.layers[name].trainable_delta().detach()
        assert torch.allclose(dst.layers[name].effective_delta(), expected, atol=1e-7)

    # a second merge accumulates exactly
    merge_scaled_(dst, src, 0.25)
    for name in dst.names():
        expected = 0.75 * src.layers[name].trainable_delta().detach()
        assert torch.allclose(dst.layers[name].effective_delta(), expected, atol=1e-7)


def test_merge_scaled_refuses_mismatched_targets():
    src_bundle = _tiny()
    dst_bundle = _tiny(seed=22)
    src = attach_lora(src_bundle.model, LoraSpec(("q_proj",), 4, 8.0))
    dst = attach_lora(dst_bundle.model, LoraSpec(("q_proj", "v_proj"), 4, 8.0))
    with pytest.raises(LoraError):
        merge_scaled_(dst, src, 0.5)


def test_merged_component_changes_the_forward_pass():
    src_bundle = _tiny()
    dst_bundle = _tiny(seed=22)
    src = attach_lora(src_bundle.model, SPEC)
    dst = attach_lora(dst_bundle.model, SPEC)
    _randomise(src, seed=8)
    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        before = dst_bundle.model(input_ids=ids, use_cache=False).logits.clone()
    merge_scaled_(dst, src, 0.5)
    with torch.no_grad():
        after = dst_bundle.model(input_ids=ids, use_cache=False).logits
    assert not torch.equal(before, after)
    for layer in dst.layers.values():
        layer.clear_merged_components()
    with torch.no_grad():
        cleared = dst_bundle.model(input_ids=ids, use_cache=False).logits
    assert torch.equal(before, cleared)


def test_lora_parameters_receive_gradients():
    bundle = _tiny()
    handles = attach_lora(bundle.model, SPEC)
    _randomise(handles, seed=9)
    ids = torch.randint(0, 64, (2, 8))
    out = bundle.model(input_ids=ids, use_cache=False)
    out.logits.float().pow(2).mean().backward()
    for name in handles.names():
        layer = handles.layers[name]
        assert layer.lora_A.grad is not None and torch.isfinite(layer.lora_A.grad).all()
        assert layer.lora_B.grad is not None and torch.isfinite(layer.lora_B.grad).all()
        assert layer.base_layer.weight.grad is None


def test_numerical_oracle_against_peft_0_19_1():
    peft = pytest.importorskip("peft", reason="peft is a local-only oracle")
    from peft import LoraConfig, get_peft_model

    ours_bundle = _tiny()
    peft_bundle = _tiny()
    ours = attach_lora(ours_bundle.model, SPEC)
    _randomise(ours, seed=10)

    peft_model = get_peft_model(
        peft_bundle.model,
        LoraConfig(
            r=SPEC.rank,
            lora_alpha=SPEC.alpha,
            target_modules=list(SPEC.target_suffixes),
            lora_dropout=SPEC.dropout,
            bias="none",
        ),
    )
    matched = []
    for name, module in peft_model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module.lora_A, "keys"):
            key = name.replace("base_model.model.", "")
            matched.append(key)
            with torch.no_grad():
                module.lora_A["default"].weight.copy_(ours.layers[key].lora_A)
                module.lora_B["default"].weight.copy_(ours.layers[key].lora_B)
    assert sorted(matched) == ours.names()

    ids = torch.randint(0, 64, (2, 11))
    ours_bundle.model.eval()
    peft_model.eval()
    with torch.no_grad():
        mine = ours_bundle.model(input_ids=ids, use_cache=False).logits
        theirs = peft_model(input_ids=ids, use_cache=False).logits
    assert torch.allclose(mine, theirs, atol=1e-6, rtol=0), float(
        (mine - theirs).abs().max()
    )
    assert peft.__version__ == "0.19.1"
