"""Surface-remap evaluation (contract §4, §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import learning_curve as LC
from foundation_learner.evaluation import remap_eval as RE
from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.tests import _w4_support as S

pytestmark = [
    pytest.mark.skipif(not S.checkpoint_present(),
                       reason="frozen checkpoint directory not present"),
    pytest.mark.skipif(not S.integration_available(),
                       reason="ecology/episodes packages absent"),
]

FAST = LC.LearningCurveConfig(generation=GenerationConfig(max_new_tokens=1),
                              record_prompts=True)


def _remap_items(ep, family, rule, remap_id: str) -> dict:
    from foundation_learner.ecology.base import TaskInstance

    out = {}
    for item_id, record in ep.items.items():
        instance = TaskInstance.from_dict(record["instance"])
        out[item_id] = family.surface_remap(rule, instance, remap_id)
    return out


def _pair(index: int = 0, variants=("remap-a", "remap-b")):
    ep, family, rule = S.real_episode(index=index)
    table = {v: _remap_items(ep, family, rule, v) for v in variants}
    return RE.RemapPair(canonical=ep, variants=table), family, rule


# -- splicing ------------------------------------------------------------
def test_evaluation_items_are_exactly_the_query_and_transfer_items():
    ep, _family, _rule = S.real_episode()
    ids = RE.evaluation_item_ids(ep, 5)
    query_ids = {e.item_id for e in ep.events
                 if LC.role_name(e) in ("QUERY_TASK", "TRANSFER_TASK")}
    assert ids == query_ids and len(ids) == 5


def test_build_remap_episode_resurfaces_only_the_evaluation_phase():
    pair, family, rule = _pair()
    remapped = RE.build_remap_episode(pair.canonical, pair.variants["remap-a"], 5)
    targets = RE.evaluation_item_ids(pair.canonical, 5)
    for original, new in zip(pair.canonical.events, remapped.events):
        if original.item_id in targets and LC.role_name(original) in (
                "QUERY_TASK", "TRANSFER_TASK"):
            assert new.text != original.text  # resurfaced
        elif LC.role_name(original) in ("TASK", "RELATED_TASK"):
            assert new.text == original.text  # learning phase untouched
    # the item table now carries the remapped instances for those items
    for item_id in targets:
        assert (remapped.items[item_id]["instance"]["surface_map_id"] == "remap-a")
        assert (pair.canonical.items[item_id]["instance"]["surface_map_id"]
                == "canonical")


def test_build_remap_episode_refuses_an_incomplete_table():
    pair, _family, _rule = _pair()
    partial = dict(list(pair.variants["remap-a"].items())[:1])
    with pytest.raises(RE.RemapSpliceError):
        RE.build_remap_episode(pair.canonical, partial, 5)


def test_splicing_two_full_episodes_is_also_supported():
    ep_a, family, rule = S.real_episode(index=0)
    ep_b, _f, _r = S.real_episode(index=1)
    spliced = RE.splice_remapped_episode(ep_a, ep_b, 5)
    cut = RE._first_eval_event_index(ep_a, 5)
    assert spliced.events[:cut] == ep_a.events[:cut]
    assert spliced.events[cut:] == ep_b.events[RE._first_eval_event_index(ep_b, 5):]


# -- verification in the remapped surface --------------------------------
def test_the_verifier_operates_in_the_remapped_surface():
    pair, family, rule = _pair()
    remapped = RE.build_remap_episode(pair.canonical, pair.variants["remap-a"], 5)
    env = LC.environment_from_episode(remapped, family, rule=rule)
    item_id = sorted(RE.evaluation_item_ids(pair.canonical, 5))[0]
    remapped_answer = pair.variants["remap-a"][item_id].answer_canonical
    outcome = LC.VerifierOutcome.coerce(
        env.verify(item_id, f"ANSWER: {remapped_answer}"))
    assert outcome.correct is True
    assert outcome.canonical_answer == remapped_answer


# -- the eval ------------------------------------------------------------
def test_run_remap_eval_reports_per_variant_gaps():
    bundle = S.tiny_bundle()
    pair, family, rule = _pair()
    env_factory = (lambda ep: LC.environment_from_episode(ep, family, rule=rule))
    out = RE.run_remap_eval(bundle, [pair], env_factory, cfg=FAST)
    assert out["schema"] == RE.REMAP_REPORT_SCHEMA
    assert out["variants"] == ["remap-a", "remap-b"]
    assert len(out["canonical_records"]) == 1
    assert set(out["records_by_variant"]) == {"remap-a", "remap-b"}
    assert out["canonical_records"][0]["remap_id"] == "canonical"
    assert out["records_by_variant"]["remap-a"][0]["remap_id"] == "remap-a"
    assert set(out["robustness"]["remapped_macro_aulc"]) == {"remap-a", "remap-b"}
    # the learning phase text is identical between canonical and remapped arms
    canonical_first = out["canonical_records"][0]["attempts"][0]["prompt_text"]
    remapped_first = out["records_by_variant"]["remap-a"][0]["attempts"][0][
        "prompt_text"]
    assert canonical_first == remapped_first


def test_pairs_from_remap_records_matches_the_data_layer_rows():
    ep, family, rule = S.real_episode()
    rows = [{"episode_id": ep.episode_id, "remap_id": "remap-a",
             "items": {k: v.to_dict()
                       for k, v in _remap_items(ep, family, rule, "remap-a").items()}}]
    pairs = RE.pairs_from_remap_records([ep], rows)
    assert len(pairs) == 1 and set(pairs[0].variants) == {"remap-a"}
    built = RE.build_remap_episode(pairs[0].canonical, pairs[0].variants["remap-a"], 5)
    assert built.episode_id == ep.episode_id


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
