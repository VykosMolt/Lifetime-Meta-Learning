"""ONLINE learning-curve walker over EPISODE_STRUCTURE_V0 (contract §6, §9)."""

from __future__ import annotations

import pytest
import torch

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


class RecordingCallbacks(LC.NullCallbacks):
    def __init__(self, prefix: torch.Tensor | None = None):
        self.prefix = prefix
        self.starts: list[LC.EpisodeContext] = []
        self.ends: list[LC.EpisodeContext] = []
        self.attempts: list[LC.AttemptEvent] = []
        self.feedback: list[LC.FeedbackEvent] = []
        self.summaries: list[tuple] = []
        self.prefix_calls = 0

    def on_episode_start(self, ctx):
        self.starts.append(ctx)

    def on_episode_end(self, ctx):
        self.ends.append(ctx)

    def prefix_provider(self):
        self.prefix_calls += 1
        return self.prefix

    def on_attempt(self, attempt):
        self.attempts.append(attempt)

    def on_feedback(self, event, hidden_summary_fn):
        self.feedback.append(event)
        if len(self.feedback) == 1:  # only pay for one forward
            mean = hidden_summary_fn()
            tokens = hidden_summary_fn("tokens")
            last = hidden_summary_fn("last")
            self.summaries.append((mean, tokens, last))


class ForcedOutcomeEnv:
    """Real environment with the verdict of chosen items forced.

    Used ONLY to exercise the R aggregation arithmetic: the environment is an
    injected collaborator of the walker, so forcing a verifier outcome tests
    the walker's bookkeeping without faking the model, the renderer or the
    generation path.
    """

    def __init__(self, env, correct_items: set[str]):
        self.env = env
        self.correct_items = set(correct_items)
        self.instances = env.instances

    def verify(self, item_id, answer_text):
        outcome = LC.VerifierOutcome.coerce(self.env.verify(item_id, answer_text))
        if item_id in self.correct_items:
            return LC.VerifierOutcome(True, outcome.canonical_answer, outcome.parsed)
        return outcome

    def feedback_text(self, *args, **kwargs):
        return self.env.feedback_text(*args, **kwargs)


# -- the load-bearing regression -----------------------------------------
def test_walker_uses_the_models_real_generation_not_the_scripted_one():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_scripted_markers(ep)
    env = S.env_for(ep, family, rule)
    record = LC.run_episode(bundle, ep, env, cfg=_cfg(record_prompts=True))

    prompts = [a["prompt_text"] for a in record["attempts"]]
    assert prompts, "no attempt slots were filled"
    for prompt in prompts:
        assert S.SCRIPTED_ATTEMPT_MARKER not in prompt
        assert S.SCRIPTED_FEEDBACK_MARKER not in prompt
    # the model's own generation is what re-enters the context
    first = record["attempts"][0]
    assert first["attempt_text"] in prompts[1]
    assert first["generated_text"]
    assert first["generated_token_ids"]


def test_feedback_is_recomputed_from_the_real_attempt():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    ep = S.plant_scripted_markers(ep)
    env = S.env_for(ep, family, rule)
    record = LC.run_episode(bundle, ep, env, cfg=_cfg(record_prompts=True))

    first = record["attempts"][0]
    verdict = "CORRECT" if first["correct"] else "INCORRECT"
    context = record["attempts"][1]["prompt_text"]
    assert f"FEEDBACK: {verdict}" in context
    # the hint is the family's real hint for the model's real attempt
    expected_hint = env.feedback_text(
        first["item_id"], first["attempt_text"],
        LC.VerifierOutcome(first["correct"], first["canonical_answer"],
                           first["parsed"]),
        role="HINT", certified=False, poison=False, condition="structured",
        attempt_number=0,
        event=next(e for e in ep.events if LC.role_name(e) == "HINT"))
    assert f"HINT: {expected_hint}" in context


# -- structure -----------------------------------------------------------
def test_R_covers_the_frozen_interaction_indices():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=_cfg())
    assert record["interaction_indices"] == list(range(7))
    assert set(record["R"]) == {str(k) for k in range(7)}
    by_index: dict[int, int] = {}
    for a in record["attempts"]:
        by_index[a["interaction_index"]] = by_index.get(a["interaction_index"], 0) + 1
    assert by_index[5] == 3 and by_index[6] == 2
    assert record["n_attempts"] == 10
    assert record["schema"] == LC.EPISODE_RECORD_SCHEMA
    assert record["structure"] == "EPISODE_STRUCTURE_V0"


def test_R_5_is_the_mean_over_the_three_query_items():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    env = S.env_for(ep, family, rule)
    query_items = [LC.event_attr(e, "item_id") for e in ep.events
                   if LC.role_name(e) == "QUERY_TASK"]
    assert len(query_items) == 3
    forced = ForcedOutcomeEnv(env, {query_items[0]})
    record = LC.run_episode(bundle, ep, forced, cfg=_cfg())
    assert record["R"]["5"] == pytest.approx(1.0 / 3.0)
    assert record["R"]["6"] == 0.0
    assert record["aulc"] == pytest.approx(sum(
        float(v) for v in record["R"].values()) / 7)


def test_correctness_only_drops_hints_and_keeps_certified_feedback():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    env = S.env_for(ep, family, rule)
    structured = LC.run_episode(bundle, ep, env, cfg=_cfg(record_prompts=True))
    plain = LC.run_episode(bundle, ep, env,
                           cfg=_cfg(feedback_condition="correctness_only",
                                    record_prompts=True))
    assert structured["n_feedback_events"] > plain["n_feedback_events"]
    last_prompt = plain["attempts"][-1]["prompt_text"]
    assert "HINT:" not in last_prompt
    assert "FEEDBACK:" in last_prompt


def test_run_episodes_batches_across_episodes_in_lockstep():
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(2)
    records = LC.run_episodes(bundle, episodes, env_factory, cfg=_cfg())
    assert len(records) == 2
    assert all(r["batch_sizes"] and max(r["batch_sizes"]) == 2 for r in records)
    ids = {r["episode_id"] for r in records}
    assert len(ids) == 2


# -- callbacks -----------------------------------------------------------
def test_callback_protocol_is_driven_as_documented():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    cb = RecordingCallbacks()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=_cfg(),
                            callbacks=cb)
    assert len(cb.starts) == 1 and len(cb.ends) == 1
    assert cb.prefix_calls == record["n_attempts"]
    assert len(cb.attempts) == record["n_attempts"]
    assert len(cb.feedback) == record["n_feedback_events"]
    assert record["callbacks_class"] == "RecordingCallbacks"

    mean, tokens, last = cb.summaries[0]
    hidden = bundle.model.config.hidden_size
    assert mean.shape == (hidden,)
    assert tokens.ndim == 2 and tokens.shape[1] == hidden
    assert last.shape == (hidden,)
    assert torch.allclose(mean, tokens.mean(dim=0), atol=1e-5)
    assert torch.allclose(last, tokens[-1], atol=1e-6)
    assert torch.isfinite(mean).all()


def test_callback_payloads_carry_no_hidden_label_and_no_poison_flag():
    cb_fields = {f.name for f in LC.AttemptEvent.__dataclass_fields__.values()}
    assert cb_fields == {"role", "item_id", "interaction_index", "text", "parsed"}
    fb_fields = {f.name for f in LC.FeedbackEvent.__dataclass_fields__.values()}
    assert "poison" not in fb_fields
    assert "correct" not in fb_fields and "canonical_answer" not in fb_fields


def test_prefix_provider_is_applied_to_generation():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    hidden = bundle.model.config.hidden_size
    gen = torch.Generator().manual_seed(11)
    prefix = torch.empty(8, hidden).normal_(0.0, 1.0, generator=gen)
    env = S.env_for(ep, family, rule)
    plain = LC.run_episode(bundle, ep, env, cfg=_cfg(),
                           callbacks=RecordingCallbacks(None))
    prefixed = LC.run_episode(bundle, ep, env, cfg=_cfg(),
                              callbacks=RecordingCallbacks(prefix))
    assert prefixed["prefix_used"] is True and plain["prefix_used"] is False
    assert all(a["prefix_len"] == 8 for a in prefixed["attempts"])
    assert ([a["generated_text"] for a in prefixed["attempts"]]
            != [a["generated_text"] for a in plain["attempts"]])


def test_sharing_one_callbacks_object_across_episodes_is_refused():
    bundle = S.tiny_bundle()
    episodes, family, env_factory = S.episode_batch(2)
    shared = RecordingCallbacks()
    with pytest.raises(ValueError, match="lockstep"):
        LC.run_episodes(bundle, episodes, env_factory, cfg=_cfg(),
                        callbacks_factory=lambda ep: shared)


# -- invariants ----------------------------------------------------------
def test_sequence_budget_is_enforced_and_never_truncated():
    """Amendment 13: the budget is still enforced, but it now stops THIS
    episode instead of the whole batch, and the episode is recorded, marked and
    excluded rather than truncated."""
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                            cfg=_cfg(max_seq_len=40))
    assert record["status"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED
    assert record["R"] == {} and record["aulc"] is None
    assert record["R_missing_reason"] == LC.STATUS_ONLINE_BUDGET_EXCEEDED
    event = record["online_budget_event"]
    assert event["truncated"] is False
    assert event["allowance"] == 40
    assert event["projected_tokens"] > 40


def test_the_eval_allowance_is_frozen_and_separate_from_the_render_budget():
    assert LC.MAX_SEQ_LEN == 2048               # contract §7, training side
    assert LC.EVAL_ONLINE_MAX_SEQ_LEN == 4096   # Amendment 13, eval side
    assert LC.LearningCurveConfig().max_seq_len == LC.EVAL_ONLINE_MAX_SEQ_LEN


def test_records_carry_the_fields_the_format_metric_needs():
    """Amendment 13 minor (b): ``answer_line_rate`` reads ``parsed`` and the
    finish-reason diagnostic reads ``finish_reason``; both must exist on every
    attempt of every record."""
    from foundation_learner.evaluation import metrics as EM

    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=_cfg())
    assert record["attempts"]
    for attempt in record["attempts"]:
        assert "parsed" in attempt and "finish_reason" in attempt
        assert attempt["role"] == "MODEL_ATTEMPT"
        assert attempt["finish_reason"] in (
            "answer_line", "eos", "max_new_tokens", "nonfinite_logits")
    rate = EM.record_answer_line_rate(record)
    assert rate is not None and 0.0 <= rate <= 1.0
    assert EM.answer_line_rate([record]) == pytest.approx(rate)
    assert set(EM.finish_reason_counts([record])) <= {
        "answer_line", "eos", "max_new_tokens", "nonfinite_logits"}


def test_certified_event_flagged_poison_is_refused():
    from dataclasses import replace

    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    events = [replace(e, poison=True) if LC.role_name(e) == "FEEDBACK" else e
              for e in ep.events]
    broken = replace(ep, events=events)
    with pytest.raises(LC.CertifiedChannelViolation):
        LC.run_episode(bundle, broken, S.env_for(broken, family, rule), cfg=_cfg())


def test_upto_event_with_cue_prompt_mode_is_available():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule),
                            cfg=_cfg(prompt_mode="upto_event_with_cue",
                                     record_prompts=True))
    assert record["prompt_mode"] == "upto_event_with_cue"
    assert record["attempts"][0]["prompt_text"].endswith("ATTEMPT:")
    assert record["interaction_indices"] == list(range(7))


def test_invalid_configuration_is_refused():
    with pytest.raises(ValueError):
        LC.LearningCurveConfig(feedback_condition="whatever")
    with pytest.raises(ValueError):
        LC.LearningCurveConfig(context_mode="whatever")
    with pytest.raises(ValueError):
        LC.LearningCurveConfig(prompt_mode="whatever")


def test_record_hash_certifies_the_record_content():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    record = LC.run_episode(bundle, ep, S.env_for(ep, family, rule), cfg=_cfg())
    payload = dict(record)
    claimed = payload.pop("record_hash")
    assert claimed == LC.domain_sha256(LC.EPISODE_RECORD_DOMAIN, payload)

    annotated = LC.annotate_records([record], generalization_scope="unseen_family")[0]
    assert annotated["generalization_scope"] == "unseen_family"
    assert annotated["record_hash"] != claimed
    check = dict(annotated)
    check.pop("record_hash")
    assert annotated["record_hash"] == LC.domain_sha256(
        LC.EPISODE_RECORD_DOMAIN, check)


def test_environment_from_episode_reconstructs_items_rule_and_schedule():
    ep, family, rule = S.real_episode()
    env = LC.environment_from_episode(ep, family)
    assert set(env.instances) == set(ep.items)
    assert env.rule.params == rule.params
    assert env.hint_conditions  # per-slot poison schedule from the episode meta
    item_id = next(iter(env.instances))
    outcome = LC.VerifierOutcome.coerce(env.verify(item_id, "ANSWER: definitely-wrong"))
    assert outcome.correct is False
    assert outcome.canonical_answer == env.instances[item_id].answer_canonical


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
