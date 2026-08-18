"""Development grid + selection + the freezing of DEV_DECISIONS_FROZEN.json."""
from __future__ import annotations

import json

import pytest

from foundation_learner.campaign import (dev_selector as ds, o1_isolation,
                                         sealed_gate)
from foundation_learner.campaign.promotion import DevMetrics
from foundation_learner.ecology.split import compute_split

DEV_FAMILIES = tuple(compute_split()["DEVELOPMENT"])


def dev(aulc):
    return DevMetrics(stage="FL3_GRID", macro_aulc=aulc, slope=0.05,
                      n_episodes=150, family_ids=DEV_FAMILIES)


def runs(scores, scope="PEFT_MODE", updates=150):
    grid = ds.frozen_grid(scope)
    return [ds.GridRun(learning_rate=lr, scope=scope, updates=updates,
                       dev=dev(score), arm_config_hash=f"hash_{lr}")
            for lr, score in zip(grid, scores)]


def test_the_grid_is_exactly_two_frozen_learning_rates_per_scope():
    assert ds.frozen_grid("PEFT_MODE") == (1e-4, 3e-4)
    assert ds.frozen_grid("FULL_MODEL_MODE") == (1e-5, 3e-5)


def test_grid_updates_are_25_percent_of_u():
    assert ds.expected_grid_updates(600) == 150
    assert ds.expected_grid_updates(4800) == 1200


def test_selection_picks_max_dev_macro_aulc():
    result = ds.select_learning_rate(runs([0.40, 0.55]), updates_U=600)
    assert result.chosen_learning_rate == 3e-4
    assert result.tie_broken is False


def test_a_tie_breaks_to_the_lower_learning_rate():
    result = ds.select_learning_rate(runs([0.55, 0.55]), updates_U=600)
    assert result.chosen_learning_rate == 1e-4
    assert result.tie_broken is True


def test_a_grid_outside_the_frozen_one_is_refused():
    bad = runs([0.4, 0.5])
    tampered = [bad[0], ds.GridRun(learning_rate=5e-4, scope="PEFT_MODE",
                                   updates=150, dev=dev(0.9))]
    with pytest.raises(ds.DevSelectorError) as exc:
        ds.select_learning_rate(tampered, updates_U=600)
    assert "hyperparameter invention" in str(exc.value)


def test_a_third_configuration_is_refused():
    extra = runs([0.4, 0.5]) + [ds.GridRun(learning_rate=1e-4, scope="PEFT_MODE",
                                           updates=150, dev=dev(0.9))]
    with pytest.raises(ds.DevSelectorError):
        ds.select_learning_rate(extra, updates_U=600)


def test_mixed_scopes_are_refused():
    mixed = [runs([0.4, 0.5])[0],
             ds.GridRun(learning_rate=3e-4, scope="FULL_MODEL_MODE",
                        updates=150, dev=dev(0.9))]
    with pytest.raises(ds.DevSelectorError):
        ds.select_learning_rate(mixed, scope="PEFT_MODE", updates_U=600)


def test_a_wrong_step_count_is_refused():
    with pytest.raises(ds.DevSelectorError) as exc:
        ds.select_learning_rate(runs([0.4, 0.5], updates=600), updates_U=600)
    assert "25 %" in str(exc.value)


def test_the_grid_runs_on_fl3_only():
    bad = [ds.GridRun(learning_rate=lr, scope="PEFT_MODE", updates=150,
                      dev=dev(0.4), arm_id="FL1")
           for lr in ds.frozen_grid("PEFT_MODE")]
    with pytest.raises(ds.DevSelectorError):
        ds.select_learning_rate(bad, updates_U=600)


def test_freeze_writes_a_valid_sealed_gate_prerequisite(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    selection = ds.select_learning_rate(runs([0.4, 0.6]), updates_U=600)
    path = tmp_path / sealed_gate.DEV_DECISIONS_NAME
    payload = ds.freeze_dev_decisions(
        str(path), selection=selection, updates_U=600,
        stage_states={"FL3": "COMPLETE"}, checkpoint_tags={"FL3": ["final"]},
        hashes={"split_manifest_sha256": "a" * 64}, runs=runs([0.4, 0.6]),
        guard=guard)
    assert payload["chosen_learning_rate"] == 3e-4
    assert payload["chosen_scope"] == "PEFT_MODE"
    assert payload["updates_U"] == 600
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    sealed_gate.validate_dev_decisions(on_disk)
    assert on_disk["dev_decisions_sha256"] == payload["dev_decisions_sha256"]
    assert "SEALED_TEST was not read" in " ".join(on_disk["notes"])


def test_freezing_twice_is_refused(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    selection = ds.select_learning_rate(runs([0.4, 0.6]), updates_U=600)
    path = tmp_path / sealed_gate.DEV_DECISIONS_NAME
    kwargs = dict(selection=selection, updates_U=600,
                  stage_states={"FL3": "COMPLETE"},
                  checkpoint_tags={"FL3": ["final"]},
                  hashes={"x": "y"}, guard=guard)
    ds.freeze_dev_decisions(str(path), **kwargs)
    with pytest.raises(ds.DevSelectorError):
        ds.freeze_dev_decisions(str(path), **kwargs)


def test_load_dev_episodes_reads_plain_development_shards(tmp_path):
    from foundation_learner.data.shards import write_shard

    guard = o1_isolation.IsolationGuard(label="TEST")
    family = DEV_FAMILIES[0]
    shard = tmp_path / "episodes" / "DEVELOPMENT" / f"{family}.scripted.jsonl"
    write_shard(str(shard), [{"episode_id": "e0", "family_id": family}])
    assert ds.load_dev_episodes([str(shard)], guard=guard)[0]["episode_id"] == "e0"
