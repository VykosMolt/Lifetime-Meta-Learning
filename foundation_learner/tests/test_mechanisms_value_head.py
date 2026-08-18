"""FL4 value head (W3, contract §7/§9): head, targets, training, DEV helpers."""
from __future__ import annotations

import dataclasses

import pytest
import torch

from foundation_learner.episodes.schema import Role
from foundation_learner.mechanisms.value_head import (
    MSE_WEIGHT,
    RANKING_MARGIN,
    SURFACE_HEURISTICS,
    VALUE_HEAD_INNER,
    AblationPlanError,
    SplitRefusedError,
    ValueHead,
    ValueHeadError,
    ValueHeadTrainConfig,
    ValueItem,
    ablation_plan_for_episode,
    build_value_dataset,
    episode_targets,
    evaluate_value_head,
    head_context_indices,
    head_input_hidden,
    pairwise_ranking_hinge,
    query_answer_score,
    surface_heuristic_predictions,
    train_value_head,
    validate_ablation_plan,
    value_head_predictions,
)
from foundation_learner.tests._w3_support import (
    DEV_FAMILY,
    SEALED_FAMILY,
    feedback_event_indices,
    real_episode,
    tiny_bundle,
)

HIDDEN = 16


# --------------------------------------------------------------------------
# the head itself
# --------------------------------------------------------------------------
def test_head_architecture_matches_the_contract():
    head = ValueHead(2048)
    assert head.fc1.in_features == 2048
    assert head.fc1.out_features == VALUE_HEAD_INNER == 256
    assert head.fc2.out_features == 1
    assert isinstance(head.act, torch.nn.GELU)
    assert head.fc1.weight.dtype is torch.float32
    assert (RANKING_MARGIN, MSE_WEIGHT) == (1.0, 0.5)


def test_head_shapes_and_fp32_output():
    head = ValueHead(HIDDEN, inner=8, seed=2)
    single = head(torch.zeros(HIDDEN))
    assert single.shape == torch.Size([])
    batch = head(torch.zeros(5, HIDDEN, dtype=torch.float64))
    assert batch.shape == (5,) and batch.dtype is torch.float32


def test_head_input_is_stop_gradient_by_construction():
    head = ValueHead(HIDDEN, inner=8, seed=2)
    hidden = torch.randn(3, HIDDEN, requires_grad=True)
    head(hidden).sum().backward()
    assert hidden.grad is None          # the backbone can never be trained here
    assert head.fc1.weight.grad is not None


def test_head_init_is_deterministic_and_seed_dependent():
    a, b, c = ValueHead(HIDDEN, seed=3), ValueHead(HIDDEN, seed=3), ValueHead(HIDDEN, seed=4)
    assert torch.equal(a.fc1.weight, b.fc1.weight)
    assert not torch.equal(a.fc1.weight, c.fc1.weight)
    assert a.config_hash() == b.config_hash() != c.config_hash()


def test_head_rejects_a_wrong_hidden_size():
    with pytest.raises(ValueHeadError):
        ValueHead(HIDDEN, inner=8)(torch.zeros(HIDDEN + 1))


# --------------------------------------------------------------------------
# losses / training on synthetic separable targets
# --------------------------------------------------------------------------
def _synthetic_items(n_groups: int = 12, per_group: int = 4,
                     seed: int = 20260809) -> list[ValueItem]:
    gen = torch.Generator().manual_seed(seed)
    direction = torch.randn(HIDDEN, generator=gen)
    items: list[ValueItem] = []
    for g in range(n_groups):
        for j in range(per_group):
            hidden = torch.randn(HIDDEN, generator=gen)
            items.append(ValueItem(
                episode_id=f"ep{g}", family_id="fam", split="TRAIN",
                item_id=f"it{g}-{j}", event_index=j, role="HINT",
                ablation_id=f"ABLATE_{j}",
                target=float(torch.dot(hidden, direction)) * 0.01,
                hidden=hidden,
                surface={"item_length": float(j), "item_type": 1.0,
                         "lexical_overlap": 0.0}))
    return items


def test_pairwise_hinge_is_zero_when_the_ranking_is_wide_enough():
    preds = torch.tensor([3.0, 1.0])
    targets = torch.tensor([1.0, 0.0])
    assert float(pairwise_ranking_hinge(preds, targets)) == pytest.approx(0.0)
    # inverted prediction -> margin + gap
    assert float(pairwise_ranking_hinge(-preds, targets)) == pytest.approx(3.0)


def test_pairwise_hinge_ignores_tied_targets():
    preds = torch.tensor([5.0, -5.0])
    targets = torch.tensor([1.0, 1.0])
    loss = pairwise_ranking_hinge(preds, targets)
    assert float(loss) == 0.0
    assert loss.requires_grad or preds.requires_grad is False


def test_training_decreases_the_loss_and_learns_the_ranking():
    items = _synthetic_items()
    head = ValueHead(HIDDEN, inner=32, seed=5)
    before = evaluate_value_head(head, items)["head"]["pairwise"]["accuracy"]
    report = train_value_head(head, items,
                              ValueHeadTrainConfig(epochs=60, groups_per_batch=4))
    history = report["history"]
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["ranking_hinge"] < history[0]["ranking_hinge"]
    after = evaluate_value_head(head, items)["head"]["pairwise"]["accuracy"]
    assert after > before
    assert after > 0.9
    assert report["target_std"] > 0.0


def test_training_is_deterministic():
    items = _synthetic_items()
    cfg = ValueHeadTrainConfig(epochs=5, groups_per_batch=4)
    a = ValueHead(HIDDEN, inner=8, seed=6)
    b = ValueHead(HIDDEN, inner=8, seed=6)
    ra = train_value_head(a, items, cfg)
    rb = train_value_head(b, items, cfg)
    assert ra["history"] == rb["history"]
    assert torch.equal(a.fc1.weight, b.fc1.weight)


def test_training_refuses_degenerate_targets():
    items = _synthetic_items(n_groups=2, per_group=2)
    for item in items:
        item.target = 1.0
    with pytest.raises(ValueHeadError):
        train_value_head(ValueHead(HIDDEN, inner=8, seed=7), items)


def test_predictions_and_baselines_feed_w4_metrics():
    items = _synthetic_items(n_groups=4)
    head = ValueHead(HIDDEN, inner=8, seed=8)
    preds = value_head_predictions(head, items)
    assert len(preds["predictions"]) == len(items) == len(preds["targets"])
    assert preds["groups"][0] == items[0].episode_id
    report = evaluate_value_head(head, items)
    assert set(report["baselines"]) == set(SURFACE_HEURISTICS)
    assert report["head"]["pairwise"]["n_groups"] == 4
    assert len(surface_heuristic_predictions(items, "item_length")) == len(items)
    with pytest.raises(ValueHeadError):
        surface_heuristic_predictions(items, "oracle")


# --------------------------------------------------------------------------
# head-input context (structural leak guard)
# --------------------------------------------------------------------------
def test_head_context_stops_at_the_item_and_holds_no_query():
    episode = real_episode()
    index = feedback_event_indices(episode)[-1]
    keep = head_context_indices(episode, index)
    assert keep == list(range(index + 1))
    roles = {episode.events[i].role for i in keep}
    assert Role.QUERY_TASK not in roles and Role.TRANSFER_TASK not in roles


def test_head_context_refuses_a_non_feedback_event():
    episode = real_episode()
    with pytest.raises(ValueHeadError):
        head_context_indices(episode, 0)          # TASK
    with pytest.raises(ValueHeadError):
        head_context_indices(episode, len(episode.events) + 5)


def test_head_input_hidden_shape_and_determinism():
    bundle = tiny_bundle()
    episode = real_episode()
    index = feedback_event_indices(episode)[0]
    a = head_input_hidden(bundle, episode, index)
    b = head_input_hidden(bundle, episode, index)
    assert a.shape == (bundle.model.config.hidden_size,)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# ablation plans and targets
# --------------------------------------------------------------------------
def test_plan_matches_the_pre_generated_shape():
    episode = real_episode()
    plan = ablation_plan_for_episode(episode)
    assert plan["plans"][0]["ablation_id"] == "FULL"
    assert plan["scoreable_feedback_events"] == feedback_event_indices(episode)
    assert validate_ablation_plan(episode, plan) is plan


def test_plan_that_ablates_a_task_event_is_refused():
    episode = real_episode()
    plan = dict(ablation_plan_for_episode(episode))
    plan["plans"] = list(plan["plans"]) + [{
        "ablation_id": "ABLATE_TASK", "ablated_event_indices": [0],
        "ablated_item_id": None, "ablated_role": "TASK"}]
    with pytest.raises(AblationPlanError):
        validate_ablation_plan(episode, plan)


def test_targets_are_finite_differences_of_present_minus_ablated():
    bundle = tiny_bundle()
    episode = real_episode()
    result = episode_targets(bundle, episode)
    assert result["targets"], "episode produced no scoreable items"
    for record in result["targets"]:
        assert record["target"] == pytest.approx(
            record["score_present"] - record["score_ablated"], abs=1e-9)
        assert episode.events[record["event_index"]].role in (
            Role.FEEDBACK, Role.HINT, Role.REVEAL)
    ids = [r["event_index"] for r in result["targets"]]
    assert ids == sorted(ids)


def test_query_score_is_deterministic():
    bundle = tiny_bundle()
    episode = real_episode()
    from foundation_learner.mechanisms.value_head import _scoring_episode

    scoring = _scoring_episode(episode)
    keep = list(range(len(scoring.events)))
    a = query_answer_score(bundle, scoring, keep)
    b = query_answer_score(bundle, scoring, keep)
    assert a["score"] == pytest.approx(b["score"], abs=1e-9)
    assert len(a["per_item"]) == 3


def test_targets_refuse_dev_and_sealed_families():
    bundle = tiny_bundle()
    for family in (DEV_FAMILY, SEALED_FAMILY):
        with pytest.raises(SplitRefusedError):
            episode_targets(bundle, real_episode(family))


def test_targets_refuse_a_mislabelled_split():
    bundle = tiny_bundle()
    episode = dataclasses.replace(real_episode(), split="DEVELOPMENT")
    with pytest.raises(SplitRefusedError):
        episode_targets(bundle, episode)


def test_build_value_dataset_produces_head_inputs_and_surfaces():
    bundle = tiny_bundle()
    episode = real_episode()
    items = build_value_dataset(bundle, [episode])
    assert items
    for item in items:
        assert item.hidden.shape == (bundle.model.config.hidden_size,)
        assert set(item.surface) == set(SURFACE_HEURISTICS)
        assert item.episode_id == episode.episode_id
        assert item.split == "TRAIN"
