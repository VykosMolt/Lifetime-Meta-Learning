"""Context-reset evaluation and its leak check (contract §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import context_reset as CR
from foundation_learner.evaluation import learning_curve as LC
from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.tests import _w4_support as S

pytestmark = [
    pytest.mark.skipif(not S.checkpoint_present(),
                       reason="frozen checkpoint directory not present"),
    pytest.mark.skipif(not S.integration_available(),
                       reason="ecology/episodes packages absent"),
]

FAST_GEN = GenerationConfig(max_new_tokens=2)


def _cfg(**kwargs) -> LC.LearningCurveConfig:
    kwargs.setdefault("generation", FAST_GEN)
    return LC.LearningCurveConfig(**kwargs)


# -- the leak checker itself ---------------------------------------------
def test_leak_checker_finds_planted_history_text():
    from foundation_learner.episodes.schema import Event, Role

    prior = [Event(role=Role.TASK, text="support statement with MARKER_A",
                   item_id="P1", interaction_index=None, certified=False,
                   poison=False, reveal=False, slot="P1"),
             Event(role=Role.FEEDBACK, text="INCORRECT", item_id="P1",
                   interaction_index=None, certified=True, poison=False,
                   reveal=False, slot="P1")]
    clean = CR.check_no_history_leak("QUERY: a fresh question\nATTEMPT: ", prior)
    assert clean.clean and clean.n_probes == 2 and clean.n_skipped_short == 0
    dirty = CR.check_no_history_leak(
        "QUERY: support statement with MARKER_A\nATTEMPT: ", prior)
    assert not dirty.clean
    assert dirty.leaks[0][0] == "TASK"
    with pytest.raises(CR.ContextLeakError):
        CR.assert_no_history_leak("support statement with MARKER_A", prior)


def test_a_query_with_the_same_surface_as_a_support_item_is_not_a_leak():
    """Two items of one family can carry byte-identical surfaces; the current
    query's own text must not be mistaken for retained history."""
    from foundation_learner.episodes.schema import Event, Role

    same = "Assignment: x0=0, x1=0\nWhat value does the hidden rule give?"
    prior = [Event(role=Role.TASK, text=same, item_id="P1",
                   interaction_index=None, certified=False, poison=False,
                   reveal=False, slot="P1")]
    prompt = f"HEADER\nQUERY: {same}\nATTEMPT: "
    assert CR.check_no_history_leak(prompt, prior, current_text=same).clean
    # ... but a genuine second copy is still caught
    leaky = f"HEADER\nTASK: {same}\nATTEMPT: x\nQUERY: {same}\nATTEMPT: "
    assert not CR.check_no_history_leak(leaky, prior, current_text=same).clean


def test_leak_checker_reports_skipped_short_probes():
    from foundation_learner.episodes.schema import Event, Role

    prior = [Event(role=Role.FEEDBACK, text="ok", item_id="P1",
                   interaction_index=None, certified=True, poison=False,
                   reveal=False, slot="P1")]
    report = CR.check_no_history_leak("QUERY: q", prior)
    assert report.n_skipped_short == 1 and report.n_probes == 0
    assert report.to_dict()["clean"] is True


# -- the real reset path -------------------------------------------------
def test_reset_prompts_contain_no_learning_history():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    env = S.env_for(ep, family, rule)
    record = LC.run_episode(bundle, ep, env,
                            cfg=_cfg(context_mode="context_reset",
                                     record_prompts=True),
                            walker_class=CR.LeakCheckingWalker)
    reset_attempts = [a for a in record["attempts"]
                      if a["context_mode"] == "context_reset"]
    assert len(reset_attempts) == 6  # indices 4, 5x3, 6x2
    assert record["reset_prompts_used"] == 6
    assert record["reset_render_source"] in ("render_reset_query", "render_subset")
    for attempt in reset_attempts:
        prompt = attempt["prompt_text"]
        assert S.SUPPORT_MARKER not in prompt
        assert "FEEDBACK:" not in prompt
        assert "HINT:" not in prompt
        assert prompt.count("ATTEMPT:") == 1
    assert len(record["context_reset_leak_reports"]) == 6
    assert all(r["clean"] for r in record["context_reset_leak_reports"])


def test_history_arm_does_contain_the_history():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                            cfg=_cfg(record_prompts=True))
    late = record["attempts"][-1]["prompt_text"]
    assert S.SUPPORT_MARKER in late
    assert "FEEDBACK:" in late


def test_a_leaky_renderer_is_caught_rather_than_passing_vacuously():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    with pytest.raises(CR.ContextLeakError):
        LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                       cfg=_cfg(context_mode="context_reset"),
                       render_module=S.leaky_render_module(),
                       walker_class=CR.LeakCheckingWalker)


def test_run_context_reset_eval_pairs_the_matched_history_arm():
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(2)
    out = CR.run_context_reset_eval(bundle, episodes, env_factory, cfg=_cfg())
    summary = out["summary"]
    assert summary["schema"] == CR.CONTEXT_RESET_REPORT_SCHEMA
    assert summary["n_episodes"] == 2
    assert len(out["history_records"]) == 2
    assert summary["n_leak_reports"] == 12
    assert summary["reset_from_index"] == 4
    persistence = summary["persistence"]
    assert set(persistence["per_family"]) == {"boolean_rule"}
    assert persistence["reset_from_index"] == 4
    assert all(r["context_mode"] == "context_reset" for r in out["reset_records"])
    assert all(r["context_mode"] == "history" for r in out["history_records"])


def test_reset_records_keep_a_valid_record_hash():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                            cfg=_cfg(context_mode="context_reset"),
                            walker_class=CR.LeakCheckingWalker)
    payload = dict(record)
    claimed = payload.pop("record_hash")
    assert claimed == LC.domain_sha256(LC.EPISODE_RECORD_DOMAIN, payload)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
