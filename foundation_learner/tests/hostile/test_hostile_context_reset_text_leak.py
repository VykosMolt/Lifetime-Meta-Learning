"""HOSTILE: textual learning history must not survive a context reset.

Attack: make the context-reset evaluation look like it removed the history
while the reset prompt still contains it.  Contract §9 calls context reset
LOAD BEARING, because FL5/FL7 persistence is only meaningful if the textual
history is genuinely gone; a leak would let history-only behaviour masquerade
as persistent adaptation.

The fixture plants a unique marker inside every SUPPORT statement and requires
(a) that a real reset run never shows it, and (b) that a renderer which DOES
keep the history is caught rather than silently accepted.
"""

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

FAST = LC.LearningCurveConfig(context_mode="context_reset",
                              generation=GenerationConfig(max_new_tokens=1),
                              record_prompts=True)


def test_planted_support_marker_never_reaches_a_reset_prompt():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    assert any(S.SUPPORT_MARKER in e.text for e in ep.events), "marker not planted"

    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=FAST,
                            walker_class=CR.LeakCheckingWalker)
    reset_prompts = [a["prompt_text"] for a in record["attempts"]
                     if a["context_mode"] == "context_reset"]
    assert reset_prompts
    for prompt in reset_prompts:
        assert S.SUPPORT_MARKER not in prompt
        assert "FEEDBACK:" not in prompt
        assert "HINT:" not in prompt
        assert "REVEALED ANSWER" not in prompt


def test_a_renderer_that_keeps_the_history_is_caught():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    with pytest.raises(CR.ContextLeakError):
        LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=FAST,
                       render_module=S.leaky_render_module(),
                       walker_class=CR.LeakCheckingWalker)


def test_the_leak_check_is_not_vacuous():
    """A checker that probed nothing would pass everything."""
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=FAST,
                            walker_class=CR.LeakCheckingWalker)
    reports = record["context_reset_leak_reports"]
    assert len(reports) == 6
    for report in reports:
        assert report["n_probes"] >= 4, report
        assert report["clean"] is True


def test_the_history_arm_is_the_control_and_does_contain_the_marker():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_support_marker(ep)
    from dataclasses import replace

    history = LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                             cfg=replace(FAST, context_mode="history"))
    assert history["reset_prompts_used"] == 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
