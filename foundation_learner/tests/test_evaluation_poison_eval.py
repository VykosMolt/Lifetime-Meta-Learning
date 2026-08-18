"""Poison-condition evaluation (contract §4, §6, §9)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import learning_curve as LC
from foundation_learner.evaluation import poison_eval as PE
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


def _conditioned_episodes(conditions=("clean", "corrupted")):
    from foundation_learner.episodes.assemble import build_hint_variants

    ep, family, rule = S.real_episode()
    variants = build_hint_variants(family, rule, ep, root_seed=20260809)
    episodes = {}
    for condition in conditions:
        rows = variants[condition]
        episodes[condition] = PE.apply_hint_variant(ep, rows, condition)
    return episodes, family, rule


# -- schedule transfer ---------------------------------------------------
def test_apply_hint_variant_transfers_the_schedule_not_the_text():
    """The data layer schedules poison PER SLOT, so a "corrupted" episode has
    some corrupted slots and some truthful ones; the walker must follow that
    schedule exactly."""
    from foundation_learner.episodes.assemble import build_hint_variants

    episodes, family, rule = _conditioned_episodes()
    clean, corrupted = episodes["clean"], episodes["corrupted"]
    clean_hints = [e for e in clean.events if LC.role_name(e) == "HINT"]
    bad_hints = [e for e in corrupted.events if LC.role_name(e) == "HINT"]
    assert clean_hints and len(clean_hints) == len(bad_hints)
    assert all(not e.poison for e in clean_hints)
    assert any(e.poison for e in bad_hints)

    ep, family, rule = S.real_episode()
    rows = build_hint_variants(family, rule, ep, root_seed=20260809)["corrupted"]
    for row in rows:
        event = corrupted.events[int(row["event_index"])]
        assert event.poison == bool(row["poison"])
        slot_conditions = {s["slot"]: s["condition"]
                           for s in corrupted.meta["hint_slots"]}
        assert slot_conditions[row["slot"]] == row["condition"]
    assert corrupted.condition == "corrupted"
    assert corrupted.meta["poison_condition"] == "corrupted"
    assert PE.episode_poison_condition(corrupted) == "corrupted"
    assert PE.episode_poison_condition(clean) == "clean"


def test_apply_hint_variant_refuses_a_non_hint_target():
    ep, _family, _rule = S.real_episode()
    with pytest.raises(ValueError):
        PE.apply_hint_variant(ep, [{"event_index": 0, "poison": True}], "corrupted")


def test_episodes_from_poison_record_covers_every_frozen_condition():
    from foundation_learner.ecology.poison import POISON_CONDITIONS
    from foundation_learner.episodes.assemble import build_hint_variants

    ep, family, rule = S.real_episode()
    record = {"episode_id": ep.episode_id,
              "variants": build_hint_variants(family, rule, ep, root_seed=1)}
    built = PE.episodes_from_poison_record(ep, record)
    assert set(built) == set(POISON_CONDITIONS)


# -- online behaviour ----------------------------------------------------
def test_online_hints_are_recomputed_per_condition_from_the_real_attempt():
    bundle = S.tiny_bundle()
    episodes, family, rule = _conditioned_episodes()
    records = {}
    for condition, ep in episodes.items():
        env = LC.environment_from_episode(ep, family, rule=rule)
        records[condition] = LC.run_episode(bundle, ep, env, cfg=FAST,
                                            walker_class=PE.PoisonAuditWalker)
    clean_hint = _first_hint(records["clean"])
    bad_hint = _first_hint(records["corrupted"])
    assert clean_hint and bad_hint and clean_hint != bad_hint


def _first_hint(record) -> str | None:
    for attempt in record["attempts"]:
        for line in attempt["prompt_text"].splitlines():
            if line.startswith("HINT: "):
                return line
    return None


def test_the_rendering_never_marks_which_hints_are_poisoned():
    bundle = S.tiny_bundle()
    episodes, family, rule = _conditioned_episodes()
    records = []
    for condition, ep in episodes.items():
        env = LC.environment_from_episode(ep, family, rule=rule)
        records.append(LC.run_episode(bundle, ep, env, cfg=FAST,
                                      walker_class=PE.PoisonAuditWalker))
    audit = PE.audit_hint_signatures(records)
    assert audit["unresolved_signatures"] == 0
    assert audit["signatures_only_poisoned"] == []
    assert audit["n_poisoned_signatures"] == 1
    assert audit["n_truthful_signatures"] == 1


def test_a_rendering_that_marks_poison_is_caught():
    marked = [{"hint_signatures": [
        {"poison": True, "signature": ["", "HINT (UNVERIFIED, POSSIBLY WRONG): <CONTENT>\n"]},
    ]}, {"hint_signatures": [
        {"poison": False, "signature": ["", "HINT: <CONTENT>\n"]},
    ]}]
    with pytest.raises(PE.PoisonMarkingError):
        PE.audit_hint_signatures(marked)


def test_a_poison_marking_word_in_a_prompt_is_caught():
    from dataclasses import replace

    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    events = []
    for e in ep.events:
        if LC.role_name(e) == "TASK":
            events.append(replace(e, text=e.text + "\n(this hint may be poisoned)"))
        else:
            events.append(e)
    broken = replace(ep, events=events)
    env = LC.environment_from_episode(broken, family, rule=rule)
    with pytest.raises(PE.PoisonMarkingError):
        LC.run_episode(bundle, broken, env, cfg=FAST,
                       walker_class=PE.PoisonAuditWalker)


def test_run_poison_eval_reports_the_gap_against_clean():
    bundle = S.tiny_bundle()
    episodes, family, rule = _conditioned_episodes()
    env_factory = (lambda ep: LC.environment_from_episode(ep, family, rule=rule))
    out = PE.run_poison_eval(bundle, list(episodes.values()), env_factory, cfg=FAST)
    assert out["schema"] == PE.POISON_REPORT_SCHEMA
    assert out["clean_condition"] == "clean"
    assert set(out["conditions_present"]) == {"clean", "corrupted"}
    assert out["n_episodes_by_condition"] == {"clean": 1, "corrupted": 1}
    assert set(out["robustness"]["macro_aulc"]) == {"clean", "corrupted"}
    assert "corrupted" in out["robustness"]["gap_vs_clean"]
    for records in out["records_by_condition"].values():
        assert all(r["poison_condition"] in ("clean", "corrupted") for r in records)


def test_unknown_conditions_are_refused():
    bundle = S.tiny_bundle()
    ep, family, rule = S.real_episode()
    env_factory = (lambda e: LC.environment_from_episode(e, family, rule=rule))
    with pytest.raises(ValueError):
        PE.run_poison_eval(bundle, [ep], env_factory, cfg=FAST,
                           condition_of=lambda e: "mildly_confusing")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
