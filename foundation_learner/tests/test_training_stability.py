"""Stability detectors and the bounded response policy (contract §17)."""

from __future__ import annotations

import dataclasses

import pytest

from foundation_learner.training import stability as st


def _monitor(**overrides):
    return st.StabilityMonitor(st.StabilityConfig(**overrides), arm_id="FL3")


def _kinds(events):
    return [e.kind for e in events]


def test_contract_frozen_thresholds():
    cfg = st.StabilityConfig()
    assert cfg.grad_norm_threshold == 50.0
    assert cfg.grad_norm_consecutive == 3
    assert cfg.loss_divergence_factor == 2.0
    assert cfg.loss_divergence_window == 200
    assert cfg.throughput_collapse_fraction == 0.40
    assert cfg.throughput_collapse_window_s == 300.0


def test_nan_loss_is_fatal():
    monitor = _monitor()
    events = monitor.observe_step(step=1, loss=float("nan"), grad_norm=1.0)
    assert _kinds(events) == [st.KIND_NAN_LOSS]
    assert monitor.fatal
    assert events[0].severity == st.SEVERITY_SCIENTIFIC_INSTABILITY


def test_inf_gradient_norm_is_fatal():
    monitor = _monitor()
    events = monitor.observe_step(step=1, loss=2.0, grad_norm=float("inf"))
    assert st.KIND_NAN_LOSS in _kinds(events)
    assert monitor.fatal


def test_non_finite_gradients_raise_optimizer_overflow():
    monitor = _monitor()
    events = monitor.observe_step(step=1, loss=2.0, grad_norm=1.0, grads_finite=False)
    assert _kinds(events) == [st.KIND_OPTIMIZER_OVERFLOW]
    assert monitor.fatal


def test_grad_norm_explosion_requires_three_consecutive_steps():
    monitor = _monitor()
    assert not monitor.observe_step(step=1, loss=1.0, grad_norm=60.0)
    assert not monitor.observe_step(step=2, loss=1.0, grad_norm=51.0)
    events = monitor.observe_step(step=3, loss=1.0, grad_norm=99.0)
    assert _kinds(events) == [st.KIND_GRAD_NORM_EXPLOSION]

    calm = _monitor()
    calm.observe_step(step=1, loss=1.0, grad_norm=60.0)
    calm.observe_step(step=2, loss=1.0, grad_norm=1.0)  # streak broken
    calm.observe_step(step=3, loss=1.0, grad_norm=60.0)
    assert not calm.observe_step(step=4, loss=1.0, grad_norm=60.0)
    assert not calm.fatal


def test_loss_divergence_uses_the_trailing_median_over_200_steps():
    monitor = _monitor()
    for step in range(200):
        assert not monitor.observe_step(step=step, loss=1.0, grad_norm=1.0)
    assert not monitor.observe_step(step=200, loss=1.9, grad_norm=1.0)
    events = monitor.observe_step(step=201, loss=2.5, grad_norm=1.0)
    assert _kinds(events) == [st.KIND_LOSS_DIVERGENCE]


def test_repeated_zero_gradient_is_detected():
    monitor = _monitor()
    for step in range(9):
        assert not monitor.observe_step(step=step, loss=1.0, grad_norm=0.0)
    events = monitor.observe_step(step=9, loss=1.0, grad_norm=0.0)
    assert _kinds(events) == [st.KIND_REPEATED_ZERO_GRAD]


def test_memory_growth_is_an_observation_not_a_scientific_failure():
    monitor = _monitor(memory_growth_window=5, memory_growth_factor=1.25)
    for step in range(6):
        monitor.observe_step(step=step, loss=1.0, grad_norm=1.0, memory_bytes=1000.0)
    events = monitor.observe_step(
        step=6, loss=1.0, grad_norm=1.0, memory_bytes=2000.0
    )
    assert _kinds(events) == [st.KIND_MEMORY_GROWTH]
    assert events[0].severity == st.SEVERITY_OBSERVATION
    assert not monitor.fatal


def test_throughput_collapse_against_bench():
    monitor = _monitor(bench_tokens_per_second=1000.0)
    for step in range(6):
        monitor.observe_step(
            step=step, loss=1.0, grad_norm=1.0, tokens=10_000, step_seconds=60.0
        )
    events = monitor.observe_step(
        step=6, loss=1.0, grad_norm=1.0, tokens=100, step_seconds=60.0
    )
    assert st.KIND_THROUGHPUT_COLLAPSE in _kinds(events)


def test_fast_state_norm_explosion():
    monitor = _monitor(fast_state_norm_max=10.0)
    assert not monitor.observe_fast_state_norm(1, 5.0)
    events = monitor.observe_fast_state_norm(2, 11.0)
    assert _kinds(events) == [st.KIND_FAST_STATE_NORM]
    assert monitor.fatal


def test_corrupted_checkpoint_event():
    monitor = _monitor()
    monitor.record_corrupted_checkpoint(5, "ckpt_final.pt", "hash mismatch")
    assert monitor.fatal
    assert monitor.events[-1].kind == st.KIND_CORRUPTED_CHECKPOINT


def test_exception_classification_is_narrow():
    classify = st.StabilityMonitor.classify_exception
    assert classify(RuntimeError("CUDA out of memory. Tried to allocate")) == (
        st.SEVERITY_TRANSIENT_INFRA
    )
    assert classify(OSError("Input/output error")) == st.SEVERITY_TRANSIENT_INFRA
    assert classify(ValueError("loss became nan")) == (
        st.SEVERITY_SCIENTIFIC_INSTABILITY
    )
    assert classify(KeyError("missing shard")) == st.SEVERITY_SCIENTIFIC_INSTABILITY


def test_failure_record_shape_and_no_hyperparameter_improvisation():
    monitor = _monitor()
    config_snapshot = {"learning_rate": 1e-4, "updates": 600, "grad_clip_norm": 1.0}
    frozen_copy = dict(config_snapshot)
    monitor.observe_step(step=7, loss=float("nan"), grad_norm=1.0)
    record = monitor.failure_record(
        arm_id="FL3", step=7, reason="NAN_LOSS", config_snapshot=config_snapshot
    )
    assert record["schema"] == "flb200.stability_failure.v1"
    assert record["arm_id"] == "FL3"
    assert record["step"] == 7
    assert "no hyperparameter improvisation" in record["response"]
    assert record["thresholds"]["grad_norm_threshold"] == 50.0
    assert record["events"][0]["kind"] == st.KIND_NAN_LOSS
    # the monitor must not have touched the arm configuration
    assert config_snapshot == frozen_copy
    assert record["arm_config"] == frozen_copy


def test_stability_config_is_frozen():
    cfg = st.StabilityConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.grad_norm_threshold = 100.0


def test_summarise_events_counts_kinds():
    monitor = _monitor()
    monitor.observe_step(step=1, loss=float("nan"), grad_norm=1.0)
    monitor.record_oom(2, "fragmentation")
    assert st.summarise_events(monitor.events) == {
        st.KIND_NAN_LOSS: 1,
        st.KIND_OOM: 1,
    }
