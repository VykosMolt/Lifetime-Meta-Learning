"""FL4-FL8 campaign stage adapters (W3) driven through W5's real stage table.

Every test here runs the REAL ``campaign.stage_definitions`` work function
(``fl4_work`` .. ``fl8_work``), which resolves the mechanism entry point through
its lazy dotted path and wraps it in ``_mechanism_stage``.  The tiny model, tiny
caps and ``rehearsal=True`` keep the walk short; the assertions are about the
CONTRACT of the payload (schema, status, the fields ``campaign/promotion.py``
consumes), about the fresh-bundle rule, and about guarded file access — never
about the competence of a randomly initialised model.
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import o1_isolation, promotion
from foundation_learner.campaign.stage_definitions import (
    STAGES_BY_ID,
    StageContext,
    fl4_entry,
    fl4_work,
    fl5_work,
    fl6_entry,
    fl6_work,
    fl7_entry,
    fl7_work,
    fl8_entry,
    fl8_work,
    resolve_dotted,
)
from foundation_learner.mechanisms import stage_support as _s
from foundation_learner.tests._w3_support import fresh_tiny_bundle

PREGEN_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "artifacts_fl", "pregen_tiny"))

MECHANISM_ENTRY_POINTS = {
    "FL4": "foundation_learner.mechanisms.value_head:run_fl4_stage",
    "FL5": "foundation_learner.mechanisms.fast_state:run_fl5_stage",
    "FL6": "foundation_learner.mechanisms.value_gating:run_fl6_stage",
    "FL7": "foundation_learner.mechanisms.fast_adapter:run_fl7_stage",
    "FL8": "foundation_learner.mechanisms.consolidation:run_fl8_stage",
}

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PREGEN_ROOT),
    reason=f"tiny pre-generation missing at {PREGEN_ROOT}")


class CountingGuard(o1_isolation.IsolationGuard):
    """The real guard, counting the file operations the adapters perform."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.json_writes: list[str] = []

    def guard(self, path, mode=o1_isolation.MODE_READ):
        resolved = super().guard(path, mode)
        (self.writes if mode == o1_isolation.MODE_WRITE else self.reads).append(resolved)
        return resolved

    def write_json(self, path, obj):
        out = super().write_json(path, obj)
        self.json_writes.append(out)
        return out


class CountingBundleFactory:
    """``bundle_factory`` that records how many FRESH bundles were built."""

    def __init__(self, seed: int = 20260809):
        self.seed = int(seed)
        self.calls = 0
        self.bundles: list[object] = []

    def __call__(self):
        self.calls += 1
        bundle = fresh_tiny_bundle(self.seed)
        self.bundles.append(bundle)
        return bundle


def make_ctx(tmp_path, **extra) -> StageContext:
    factory = CountingBundleFactory()
    guard = CountingGuard(label="W3_STAGE_TEST")
    payload = {"max_new_tokens": 1, "run_equivalence_gate": False}
    payload.update(extra)
    ctx = StageContext(
        out_dir=str(tmp_path), pregen_root=PREGEN_ROOT,
        bundle_factory=factory, guard=guard,
        eval_episode_cap=1, train_example_cap=2, rehearsal=True,
        label="W3_STAGE_TEST", extra=payload)
    ctx.extra["_factory"] = factory
    return ctx


def assert_mechanism_payload(payload: dict, stage_id: str) -> None:
    assert payload["schema"] == _s.MECHANISM_STAGE_SCHEMA
    assert payload["stage"] == stage_id
    assert payload["status"] == _s.STATUS_COMPLETE
    assert payload["rehearsal"] is True
    assert "DRESS REHEARSAL" in payload["note"]


def assert_report_written(ctx: StageContext, stage_id: str) -> str:
    stage = STAGES_BY_ID[stage_id]
    path = os.path.join(ctx.out_dir, stage_id.lower(), stage.outputs[0])
    assert os.path.isfile(path), f"{stage_id} wrote no {stage.outputs[0]}"
    assert path in ctx.guard.json_writes, "the report bypassed the guard"
    assert ctx.guard.reads, "no guarded read happened"
    return path


# --------------------------------------------------------------------------
# the entry points exist and are resolvable exactly as the stage table says
# --------------------------------------------------------------------------
@pytest.mark.parametrize("stage_id,dotted", sorted(MECHANISM_ENTRY_POINTS.items()))
def test_entry_point_resolves_with_the_declared_signature(stage_id, dotted):
    import inspect

    fn = resolve_dotted(dotted)
    assert callable(fn)
    params = list(inspect.signature(fn).parameters)
    assert params == ["ctx", "stage"], (stage_id, params)


# --------------------------------------------------------------------------
# FL4
# --------------------------------------------------------------------------
def test_fl4_stage_runs_and_feeds_promotion(tmp_path):
    ctx = make_ctx(tmp_path)
    stage = STAGES_BY_ID["FL4"]
    payload = fl4_work(ctx, stage)

    assert_mechanism_payload(payload, "FL4")
    assert_report_written(ctx, "FL4")
    assert ctx.results["FL4"] is payload

    # --- the field campaign/promotion.py consumes through fl6_entry --------
    accuracy = payload["pairwise_ranking_accuracy"]
    assert accuracy is None or 0.0 <= float(accuracy) <= 1.0
    decision = fl6_entry(ctx)
    assert isinstance(decision, promotion.PromotionDecision)
    assert decision.stage == "FL6"
    assert "pairwise ranking accuracy" in " ".join(decision.reasons)
    assert decision.admitted is (float(accuracy) >= promotion.FL6_MIN_PAIRWISE_ACCURACY)

    # --- FL4's own results -------------------------------------------------
    assert payload["n_train_episodes"] >= 1 and payload["n_dev_episodes"] >= 1
    assert payload["n_train_items"] >= 1 and payload["n_dev_items"] >= 1
    assert payload["dev"]["head"]["n"] == payload["n_dev_items"]
    assert set(payload["dev"]["baselines"]) == {"item_length", "item_type",
                                                "lexical_overlap"}
    assert payload["training"]["history"], "the head was not trained"
    assert payload["target_model"]["loaded"] is False       # no FL3 arm here
    assert payload["target_model"]["model"] == "BASE_CHECKPOINT_NO_TRAINED_ARM"

    # --- data-sufficiency evidence for promote_fl4 -------------------------
    counts = ctx.extra["fl4_scoreable_items_per_episode"]
    assert counts and all(int(c) >= 2 for c in counts)
    sufficiency = promotion.fl4_data_sufficiency(counts)
    assert sufficiency["episodes_with_enough_items"] == len(counts)
    assert isinstance(fl4_entry(ctx), promotion.PromotionDecision)

    # --- the head is handed on to FL6 / FL7 --------------------------------
    head = _s.value_head_from_ctx(ctx)
    assert head is not None
    assert head.config_hash() == payload["head_config_hash"]
    assert not any(p.requires_grad for p in head.parameters())
    assert os.path.isfile(payload["head_state_path"])


def test_fl4_uses_one_fresh_bundle(tmp_path):
    ctx = make_ctx(tmp_path)
    fl4_work(ctx, STAGES_BY_ID["FL4"])
    factory = ctx.extra["_factory"]
    assert factory.calls == 1
    assert len({id(b) for b in factory.bundles}) == factory.calls


def test_fl4_targets_come_from_the_fl3_checkpoint_when_one_exists(tmp_path):
    """The §7 target definition: restore the FINAL FL3 checkpoint, not the base."""
    from foundation_learner.training.arms import STAGE_SMOKE, make_arm_config
    from foundation_learner.training.tokenization import iter_examples
    from foundation_learner.training.trainer import run_training_arm

    ctx = make_ctx(tmp_path)
    bundle = ctx.bundle_factory()
    episodes = _s.train_episodes(ctx, 1)
    examples = list(iter_examples(episodes, bundle.tokenizer, "FL3"))
    cfg = make_arm_config("FL3", learning_rate=1e-4, updates=1,
                          max_tokens_per_batch=2048, peft_mode=ctx.scope,
                          seed=int(ctx.root_seed), stage=STAGE_SMOKE)
    out_dir = os.path.join(ctx.out_dir, "fl3", "arm")
    run_training_arm(cfg, bundle, examples, out_dir, None)

    fresh = ctx.bundle_factory()
    state = _s.arm_checkpoint_state(ctx, fresh, "FL3")
    assert state["loaded"] is True
    assert state["arm_id"] == "FL3" and state["tag"] == "final"
    assert state["trained_scope"] == ctx.scope
    assert state["manifest_hash"]
    assert state["model"] == "FL3_final"


# --------------------------------------------------------------------------
# FL5
# --------------------------------------------------------------------------
def test_fl5_stage_runs_both_arms_and_publishes_persistence(tmp_path):
    ctx = make_ctx(tmp_path, fl5_updates=1)
    payload = fl5_work(ctx, STAGES_BY_ID["FL5"])

    assert_mechanism_payload(payload, "FL5")
    assert_report_written(ctx, "FL5")
    # Amendment 13 adds the EVAL-ONLY control arm; there is still no third
    # training run (its arm_result is None and it reuses ON's trained module).
    assert set(payload["arms"]) == {"FAST_STATE_ON", "FAST_STATE_OFF",
                                    "FAST_STATE_ON_S0"}

    on = payload["arms"]["FAST_STATE_ON"]
    off = payload["arms"]["FAST_STATE_OFF"]
    s0 = payload["arms"]["FAST_STATE_ON_S0"]
    assert s0["eval_only"] is True and s0["arm_result"] is None
    assert s0["control_of"] == "FAST_STATE_ON"
    assert s0["state_pinned_to_zero"] is True
    assert s0["fast_state_config_hash"] == on["fast_state_config_hash"]
    # the control still injects a prefix (the LEARNED STATIC one)
    for mode in ("history", "context_reset"):
        assert all(s0["prefix_used"][mode]), mode
    assert payload["control_arm"] == "FAST_STATE_ON_S0"
    for key in ("state_update_contribution", "static_prefix_contribution"):
        assert set(payload[key]) >= {"point", "ci"}

    # every FL5 payload and per-arm record disclaims FL3 comparability
    assert payload["comparable_to_fl3"] is False
    assert "impossible-task control" in payload["comparable_to_fl3_reason"]
    for arm in payload["arms"].values():
        assert arm["comparable_to_fl3"] is False
        assert arm["comparable_to_fl3_reason"]
    assert on["arm_result"]["comparable_to_fl3"] is False
    assert on["arm_result"]["steps"] == 1
    assert on["fast_state_config"]["state_dim"] == 1024
    # the arms differ in exactly one thing: the injected prefix
    for mode in ("history", "context_reset"):
        assert all(on["prefix_used"][mode]), mode
    assert not any(off["prefix_used"]["history"])
    assert not any(off["prefix_used"]["context_reset"])
    for arm in (on, off):
        for mode in ("history", "context_reset"):
            assert arm["dev"][mode]["split"] == "DEVELOPMENT"

    # --- the field campaign/promotion.py consumes through fl8_entry --------
    evidence = ctx.extra["persistence_evidence"]
    assert evidence is payload["persistence_evidence"]
    assert set(evidence) >= {"point", "ci", "split", "source"}
    assert evidence["split"] == "DEVELOPMENT" and evidence["source"] == "FL5"
    assert len(evidence["ci"]) == 2
    decision = fl8_entry(ctx)
    assert isinstance(decision, promotion.PromotionDecision)
    assert decision.stage == "FL8"
    assert "persistence" in " ".join(decision.reasons)


def test_fl5_uses_a_fresh_bundle_per_arm(tmp_path):
    ctx = make_ctx(tmp_path, fl5_updates=1)
    fl5_work(ctx, STAGES_BY_ID["FL5"])
    factory = ctx.extra["_factory"]
    assert factory.calls == 2                      # one per arm (contract §7)
    assert len({id(b) for b in factory.bundles}) == 2


# --------------------------------------------------------------------------
# FL6
# --------------------------------------------------------------------------
def test_fl6_stage_compares_gated_and_unconditional(tmp_path):
    ctx = make_ctx(tmp_path)
    fl4_work(ctx, STAGES_BY_ID["FL4"])             # produces the frozen head
    payload = fl6_work(ctx, STAGES_BY_ID["FL6"])

    assert_mechanism_payload(payload, "FL6")
    assert_report_written(ctx, "FL6")
    assert payload["gate_threshold"] == 0.0
    assert set(payload["variants"]) == {"unconditional", "value_gated"}
    assert payload["conditions"] == list(("clean", "corrupted"))
    for variant, per_condition in payload["variants"].items():
        for condition, cell in per_condition.items():
            assert cell["n_episodes"] >= 1, (variant, condition)
            assert cell["dev"]["split"] == "DEVELOPMENT"
            total = cell["updates_admitted"] + cell["updates_declined"]
            assert total >= 1
            assert 0.0 <= cell["admission_rate"] <= 1.0
    # the unconditional arm never declines; that is what makes it the control
    for cell in payload["variants"]["unconditional"].values():
        assert cell["updates_declined"] == 0
    assert payload["head_config_hash"] == ctx.results["FL4"]["head_config_hash"]


def test_fl6_refuses_to_invent_a_head(tmp_path):
    ctx = make_ctx(tmp_path)
    payload = fl6_work(ctx, STAGES_BY_ID["FL6"])   # no FL4 in this session
    assert payload["status"] == _s.STATUS_SKIPPED_INPUT
    assert "never fabricated" in payload["reason"]
    assert ctx.extra["_factory"].calls == 0


# --------------------------------------------------------------------------
# FL7
# --------------------------------------------------------------------------
def test_fl7_stage_runs_ungated_and_gated(tmp_path):
    ctx = make_ctx(tmp_path)
    fl4_work(ctx, STAGES_BY_ID["FL4"])
    before = ctx.extra["_factory"].calls
    payload = fl7_work(ctx, STAGES_BY_ID["FL7"])

    assert_mechanism_payload(payload, "FL7")
    assert_report_written(ctx, "FL7")
    assert payload["gated_variant_available"] is True
    assert set(payload["variants"]) == {"ungated", "value_gated"}
    assert payload["inner_loop"] == {"steps": 4, "lr": 1e-3, "max_frobenius": 1.0}
    for name, variant in payload["variants"].items():
        assert variant["base_weights_unchanged"] is True, name
        assert variant["final_delta_norm"] <= 1.0 + 1e-6, name
        assert variant["config"]["spec"]["rank"] == 8
        for mode in ("history", "context_reset"):
            assert variant["dev"][mode]["split"] == "DEVELOPMENT"
    assert payload["variants"]["ungated"]["n_updates_applied"] >= 1
    assert payload["fast_update_stable"] is True
    assert ctx.extra["fast_update_stable"] is True
    # one fresh bundle per variant plus the matched no-adapter baseline
    assert ctx.extra["_factory"].calls == before + 3
    assert isinstance(fl7_entry(ctx), promotion.PromotionDecision)
    assert ctx.extra["fl7_persistence_evidence"]["source"] == "FL7"


# --------------------------------------------------------------------------
# FL8
# --------------------------------------------------------------------------
def test_fl8_stage_runs_the_three_modes_and_the_chain_plan(tmp_path):
    ctx = make_ctx(tmp_path, fl8_chains=1, fl8_unrelated_episodes=1)
    payload = fl8_work(ctx, STAGES_BY_ID["FL8"])

    assert_mechanism_payload(payload, "FL8")
    assert_report_written(ctx, "FL8")
    assert payload["modes"] == ["none", "indiscriminate", "value_gated"]
    assert payload["scale"] == 0.5
    assert payload["family_a"] != payload["family_b"]
    assert payload["n_chains"] == 1
    assert payload["eval_plan"]["schema"] == "flb200.fl8_eval_plan.v1"
    assert payload["eval_plan"]["plan_hash"]

    results = payload["results"]
    assert results["none"]["consolidation"]["n_selected"] == 0
    assert results["none"]["slow_bank"]["n_components"] == 0
    assert results["indiscriminate"]["consolidation"]["n_selected"] == 2
    assert results["indiscriminate"]["slow_bank"]["n_components"] > 0
    for mode, result in results.items():
        assert result["fast_state_cleared"] is True
        assert result["consolidation"]["fast_delta_norm_after_clear"] == 0.0
        # A -> B -> A really ran through W4's interference evaluator
        assert len(result["chains"]) == 1
        chain = result["chains"][0]
        assert set(chain) >= {"retention_ratio", "interference_cost",
                              "recovery_interactions", "aulc_a1", "aulc_a2"}
        assert result["unrelated_dev"]["split"] == "DEVELOPMENT"
        assert all(d["split"] == "TRAIN" or d["split"] == "DEVELOPMENT"
                   for d in result["deltas"])
    # one fresh bundle per mode (contract §7)
    assert ctx.extra["_factory"].calls == len(payload["modes"])


def test_fl8_refuses_an_unknown_mode(tmp_path):
    from foundation_learner.campaign.stage_definitions import StageError

    ctx = make_ctx(tmp_path, fl8_modes=("none", "everything"))
    with pytest.raises(StageError):
        fl8_work(ctx, STAGES_BY_ID["FL8"])
