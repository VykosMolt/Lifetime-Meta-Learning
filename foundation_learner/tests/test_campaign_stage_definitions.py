"""The declarative stage table (contract §7, §10, §11).

Structure, priority order, lazy resolution, projections, and the
feature-detected mechanism rungs.  Nothing here imports ``mechanisms``: that
package is under construction by another worker and its API is deliberately not
pinned by this table.
"""
from __future__ import annotations

import os

import pytest

from foundation_learner.campaign import o1_isolation, promotion
from foundation_learner.campaign import stage_definitions as sd
from foundation_learner.campaign.affordability import BenchMeasurement, PEFT_MODE


def bench(spu=1.0, eval_per_episode=0.5, model_load=0.0, forward=0.25):
    return BenchMeasurement(scope=PEFT_MODE, seconds_per_update=spu,
                            tokens_per_second=10.0, updates_measured=4,
                            wall_seconds=4 * spu, forward_tokens=100,
                            max_tokens_per_batch=2048,
                            eval_seconds_per_episode=eval_per_episode,
                            model_load_seconds=model_load,
                            forward_seconds_per_episode=forward)


def test_the_table_covers_bench_fl0_to_fl8_and_the_sealed_opening():
    ids = [s.stage_id for s in sd.STAGE_TABLE]
    for expected in ("BENCH", "FL0", "DEV_GRID", "FL1", "FL2", "FL3", "FL4",
                     "FL5", "FL6", "FL7", "FL8", "SECOND_SEED", "SEALED_EVAL"):
        assert expected in ids
    assert len(ids) == len(set(ids))


def test_priority_order_is_the_frozen_session_order():
    order = [s.stage_id for s in sd.stages_in_priority_order()]
    assert order[0] == "BENCH"
    assert order[1] == "FL0"
    assert order.index("DEV_GRID") < order.index("FL1")
    for arm in ("FL1", "FL2", "FL3"):
        assert order.index(arm) < order.index("CORE_MATCHING")
        assert order.index(arm) < order.index("FL4")
    # the compute/base-identity audit runs BEFORE any extension consumes the
    # comparison it audits
    assert order.index("CORE_MATCHING") < order.index("FL4")
    for earlier, later in (("FL4", "FL5"), ("FL5", "FL6"), ("FL6", "FL7"),
                           ("FL7", "FL8"), ("FL8", "SECOND_SEED")):
        assert order.index(earlier) < order.index(later)
    # the frozen diagnostics come after the FL4-FL8 admission attempts and
    # before any additional predeclared seed; the sealed opening stays last
    for diag in sd.DIAGNOSTIC_STAGES:
        assert order.index("FL8") < order.index(diag)
        assert order.index(diag) < order.index("SECOND_SEED")
    assert order[-1] == "SEALED_EVAL"


def test_the_diagnostics_are_unconditional_and_produce_metrics_11_12_13():
    for stage_id in sd.DIAGNOSTIC_STAGES:
        stage = sd.STAGES_BY_ID[stage_id]
        assert stage.entry is None, (
            f"{stage_id} must have NO promotion entry condition: contract §10's "
            "FL3-null fallback names exactly these diagnostics")
        assert stage.requires_modules == ()
        assert callable(sd.resolve_dotted(stage.work))


def test_every_stage_declares_work_outputs_and_fallback_work():
    for stage in sd.STAGE_TABLE:
        assert ":" in stage.work
        assert stage.outputs, f"{stage.stage_id} declares no outputs"
        assert stage.fallback_work, f"{stage.stage_id} declares no fallback"


def test_every_dotted_path_in_the_table_resolves_or_is_a_mechanism():
    for stage in sd.STAGE_TABLE:
        assert callable(sd.resolve_dotted(stage.work))
        if stage.entry:
            assert callable(sd.resolve_dotted(stage.entry))


def test_resolve_optional_returns_none_for_a_missing_module():
    assert sd.resolve_optional("foundation_learner.not_a_module:thing") is None


def test_resolve_optional_raises_for_a_present_module_missing_the_entry_point():
    with pytest.raises(sd.StageError):
        sd.resolve_optional(
            "foundation_learner.campaign.stage_definitions:not_defined_here")


def test_bad_dotted_path_is_refused():
    with pytest.raises(sd.StageError):
        sd.resolve_dotted("foundation_learner.campaign.promotion")


def test_projection_formulas():
    b = bench(spu=2.0, eval_per_episode=0.5)
    train = sd.project_stage_seconds(sd.STAGES_BY_ID["FL3"], bench=b,
                                     updates=600, eval_episodes=10)
    assert train["train_seconds"] == 1200.0 and train["eval_seconds"] == 5.0
    grid = sd.project_stage_seconds(sd.STAGES_BY_ID["DEV_GRID"], bench=b,
                                    updates=600, eval_episodes=10)
    assert grid["updates_per_config"] == 150
    assert grid["projected_seconds"] == 2 * 150 * 2.0 + 2 * 10 * 0.5
    with pytest.raises(sd.StageError):
        sd.project_stage_seconds(sd.STAGES_BY_ID["FL3"], bench=b, updates=None)


def test_multi_cell_stages_are_projected_with_their_real_multipliers():
    """R-M8: FL0 runs four cells, FL6 ten walks, FL8 nine; not one each."""
    b = bench(spu=2.0, eval_per_episode=0.5, model_load=7.0)
    fl0 = sd.project_stage_seconds(sd.STAGES_BY_ID["FL0"], bench=b,
                                   updates=None, eval_episodes=4)
    # 4 episodes x 0.5 s x 4 cells + one fresh load
    assert fl0["eval_multiplier"] == 4
    assert fl0["eval_seconds"] == 8.0
    assert fl0["projected_seconds"] == 8.0 + 7.0
    fl6 = sd.project_stage_seconds(sd.STAGES_BY_ID["FL6"], bench=b, updates=100,
                                   eval_episodes=10)
    assert fl6["eval_multiplier"] == 10 and fl6["eval_seconds"] == 50.0
    fl8 = sd.project_stage_seconds(sd.STAGES_BY_ID["FL8"], bench=b, updates=100,
                                   eval_episodes=10)
    assert fl8["eval_multiplier"] == 9
    fl5 = sd.project_stage_seconds(sd.STAGES_BY_ID["FL5"], bench=b, updates=100,
                                   eval_episodes=10)
    assert fl5["train_multiplier"] == 2
    assert fl5["train_seconds"] == 2 * 100 * 2.0


def test_fl4_is_projected_from_measured_forward_passes_not_optimizer_updates():
    b = bench(spu=2.0, eval_per_episode=0.5, model_load=0.0, forward=0.25)
    fl4 = sd.project_stage_seconds(
        sd.STAGES_BY_ID["FL4"], bench=b, updates=600, eval_episodes=10,
        extra={"fl4_train_episodes": 20, "fl4_dev_episodes": 10})
    assert fl4["train_seconds"] == 0.0        # the head is not a backbone arm
    assert fl4["forwards_per_episode"] == sd.FL4_TARGET_FORWARDS_PER_EPISODE
    assert fl4["forward_seconds"] == 0.25 * 8 * 30


def test_a_missing_forward_measurement_refuses_the_fl4_projection():
    b = BenchMeasurement(scope=PEFT_MODE, seconds_per_update=1.0,
                         tokens_per_second=10.0, updates_measured=4,
                         wall_seconds=4.0, forward_tokens=100,
                         max_tokens_per_batch=2048,
                         eval_seconds_per_episode=0.5, model_load_seconds=1.0)
    with pytest.raises(sd.StageError) as exc:
        sd.project_stage_seconds(sd.STAGES_BY_ID["FL4"], bench=b, updates=600,
                                 eval_episodes=10)
    assert "may not be guessed" in str(exc.value)


def test_an_audit_stage_is_a_declared_bookkeeping_budget():
    b = bench()
    audit = sd.project_stage_seconds(sd.STAGES_BY_ID["CORE_MATCHING"], bench=b,
                                     updates=None)
    assert audit["projected_seconds"] == sd.AUDIT_STAGE_SECONDS


def test_eval_maxima_are_frozen_per_stage():
    for stage in sd.STAGE_TABLE:
        if stage.kind in ("EVAL", "TRAIN", "GRID", "MECHANISM", "SEALED"):
            if stage.stage_id in sd.EVAL_EPISODES_PER_STAGE:
                assert stage.eval_episodes == \
                    sd.EVAL_EPISODES_PER_STAGE[stage.stage_id]


def test_mechanism_stages_are_skipped_when_the_package_is_absent(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    stage = sd.StageDefinition(
        stage_id="FLX", priority=99, kind="MECHANISM",
        work="x:y", projection="MECHANISM", outputs=("report.json",),
        requires_modules=("foundation_learner.definitely_absent_module",))
    payload = sd._mechanism_stage(ctx, stage, "x:y")
    assert payload["status"] == "SKIPPED_MECHANISMS"
    assert payload["missing_modules"] == \
        ["foundation_learner.definitely_absent_module"]
    assert os.path.isfile(os.path.join(str(tmp_path), "flx", "report.json"))


def test_mechanism_stage_with_a_present_module_but_no_entry_point_is_an_error(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    stage = sd.StageDefinition(
        stage_id="FLX", priority=99, kind="MECHANISM", projection="MECHANISM",
        work="x:y", outputs=("r.json",),
        requires_modules=("foundation_learner.campaign.promotion",))
    with pytest.raises(sd.StageError):
        sd._mechanism_stage(ctx, stage,
                            "foundation_learner.campaign.promotion:nope")


def test_entry_conditions_come_from_the_promotion_module(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard)
    decision = sd.fl4_entry(ctx)
    assert isinstance(decision, promotion.PromotionDecision)
    assert decision.admitted is False           # no core comparison yet
    ctx.results["_dev_metrics"] = {
        "FL3": promotion.DevMetrics(stage="FL3", macro_aulc=0.6, slope=0.1),
        "FL1": promotion.DevMetrics(stage="FL1", macro_aulc=0.5, slope=0.0),
    }
    ctx.results["FL3"] = {"ok": True}
    ctx.extra["fl4_scoreable_items_per_episode"] = [2] * 200
    assert sd.fl4_entry(ctx).admitted is True


def test_fl1_pool_rendering_matches_the_episode_surface():
    from foundation_learner.episodes.render import INSTRUCTION_HEADER_V0

    text, (start, end) = sd._render_fl1_pair("what is 2+2?", "4")
    assert text.startswith(INSTRUCTION_HEADER_V0)
    assert "TASK: what is 2+2?" in text
    assert text[start:end] == "ANSWER: 4"
    assert text.endswith("\n")


# ---------------- evaluation sets cover every split family (R-C3) ---------

class _Ep:
    def __init__(self, family_id, episode_id="e"):
        self.family_id = family_id
        self.episode_id = episode_id


def _ctx(tmp_path, **kw):
    guard = o1_isolation.IsolationGuard(label="TEST")
    return sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                           bundle_factory=lambda: None, guard=guard, **kw)


def test_the_eval_plan_divides_the_cap_across_the_split_families(tmp_path):
    ctx = _ctx(tmp_path)
    plan = sd.eval_plan(ctx, sd.STAGES_BY_ID["FL0"])
    assert plan["split"] == "DEVELOPMENT"
    assert plan["n_families"] == 3
    assert plan["limit_per_family"] == 100          # 300 / 3, never 300 / 1
    assert plan["total"] == 300
    sealed_plan = sd.eval_plan(ctx, sd.STAGES_BY_ID["SEALED_EVAL"])
    assert sealed_plan["split"] == "SEALED_TEST"
    assert sealed_plan["limit_per_family"] == 100


def test_a_rehearsal_cap_still_keeps_every_family(tmp_path):
    ctx = _ctx(tmp_path, eval_episode_cap=1)
    plan = sd.eval_plan(ctx, sd.STAGES_BY_ID["FL0"])
    assert plan["limit_per_family"] == 1 and plan["total"] == 3


def test_an_eval_set_missing_a_split_family_is_refused():
    families = sd.split_families("DEVELOPMENT")
    full = [_Ep(f, f"e{i}") for i, f in enumerate(families)]
    report = sd.assert_eval_family_coverage(full, "DEVELOPMENT", context="ok")
    assert report["n_episodes"] == 3
    partial = [_Ep(families[0], "a"), _Ep(families[0], "b")]
    with pytest.raises(sd.StageError) as exc:
        sd.assert_eval_family_coverage(partial, "DEVELOPMENT", context="attack")
    assert "missing" in str(exc.value)
    assert families[1] in str(exc.value)


def test_a_sealed_family_smuggled_into_a_dev_eval_set_is_refused():
    dev = list(sd.split_families("DEVELOPMENT"))
    sealed = sd.split_families("SEALED_TEST")[0]
    episodes = [_Ep(f) for f in dev] + [_Ep(sealed)]
    with pytest.raises(sd.StageError) as exc:
        sd.assert_eval_family_coverage(episodes, "DEVELOPMENT", context="attack")
    assert "unexpected" in str(exc.value)


# ---------------- the promoted candidate + the sealed entry (R-C4) --------

def test_the_promoted_arm_defaults_to_the_frozen_core_treatment(tmp_path):
    ctx = _ctx(tmp_path)
    assert sd.promoted_arm_id(ctx) == "FL3"
    ctx.extra["promoted_arm"] = "FL1"
    assert sd.promoted_arm_id(ctx) == "FL1"
    ctx.extra["promoted_arm"] = "FL9"
    with pytest.raises(sd.StageError):
        sd.promoted_arm_id(ctx)


def test_the_sealed_entry_requires_the_complete_core_comparison(tmp_path):
    ctx = _ctx(tmp_path)
    decision = sd.sealed_eval_entry(ctx)
    assert decision.admitted is False
    assert decision.evidence["missing_core_arms"] == ["FL1", "FL2", "FL3"]
    assert decision.fallback[0] == "leave the sealed set unopened"

    for arm in ("FL1", "FL2", "FL3"):
        ctx.results[arm] = {"ok": True}
    assert sd.sealed_eval_entry(ctx).admitted is False   # no frozen decisions

    guard = ctx.guard
    guard.write_json(os.path.join(ctx.out_dir, "DEV_DECISIONS_FROZEN.json"),
                     {"schema": "flb200.dev_decisions.v1"})
    assert sd.sealed_eval_entry(ctx).admitted is False   # no arm checkpoint

    guard.write_json(os.path.join(ctx.out_dir, "fl3", "arm",
                                  "ckpt_final.manifest.json"), {"tag": "final"})
    decision = sd.sealed_eval_entry(ctx)
    assert decision.admitted is True, decision.reasons
    assert decision.evidence["promoted_arm"] == "FL3"


def test_generation_budget_is_frozen_outside_a_rehearsal(tmp_path):
    guard = o1_isolation.IsolationGuard(label="TEST")
    ctx = sd.StageContext(out_dir=str(tmp_path), pregen_root=str(tmp_path),
                          bundle_factory=lambda: None, guard=guard,
                          extra={"max_new_tokens": 4})
    with pytest.raises(sd.StageError):
        sd._generation_config(ctx)
    ctx.rehearsal = True
    assert sd._generation_config(ctx).max_new_tokens == 4
    ctx.extra.pop("max_new_tokens")
    assert sd._generation_config(ctx).max_new_tokens == sd.FROZEN_MAX_NEW_TOKENS
