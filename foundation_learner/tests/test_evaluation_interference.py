"""A -> B -> A interference chains (contract §9, §16)."""

from __future__ import annotations

import pytest

from foundation_learner.evaluation import interference as IF
from foundation_learner.evaluation import learning_curve as LC
from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.tests import _w4_support as S

pytestmark = [
    pytest.mark.skipif(not S.checkpoint_present(),
                       reason="frozen checkpoint directory not present"),
    pytest.mark.skipif(not S.integration_available(),
                       reason="ecology/episodes packages absent"),
]

FAST = LC.LearningCurveConfig(generation=GenerationConfig(max_new_tokens=1))


def _chain(index: int = 0):
    a1, fam_a, rule_a = S.real_episode(family_id="boolean_rule", index=index)
    b, fam_b, rule_b = S.real_episode(family_id="graph_edge_semantics", index=index)
    a2, _fa, rule_a2 = S.real_episode(family_id="boolean_rule", index=index + 50)
    envs = {id(a1): (fam_a, rule_a), id(b): (fam_b, rule_b), id(a2): (fam_a, rule_a2)}
    spec = IF.chain_spec_from_dict({
        "chain_id": f"chain-{index}", "episode_a1": a1, "episode_b": b,
        "episode_a2": a2})
    return spec, (lambda ep: S.env_for(ep, *envs[id(ep)]))


# -- spec validation -----------------------------------------------------
def test_chain_spec_validates_family_roles():
    spec, _env = _chain()
    assert spec.family_a == "boolean_rule"
    assert spec.family_b == "graph_edge_semantics"
    assert set(spec.episodes()) == {"a1", "b", "a2"}

    a1, _f, _r = S.real_episode(family_id="boolean_rule")
    with pytest.raises(IF.ChainSpecError):
        IF.chain_spec_from_dict({"episode_a1": a1, "episode_b": a1,
                                 "episode_a2": a1})
    with pytest.raises(IF.ChainSpecError):
        IF.chain_spec_from_dict({"episode_a1": a1})


def test_chain_spec_resolves_ids_through_a_resolver():
    a1, _fa, _ra = S.real_episode(family_id="boolean_rule")
    b, _fb, _rb = S.real_episode(family_id="sequence_transform")
    table = {"A": a1, "B": b}
    spec = IF.chain_spec_from_dict(
        {"chain_id": "c", "episode_a1_id": "A", "episode_b_id": "B",
         "episode_a2_id": "A"}, episode_resolver=table.__getitem__)
    assert spec.family_a == "boolean_rule" and spec.family_b == "sequence_transform"


def test_pair_balance_is_enforced():
    counts = IF.check_pair_balance([])
    assert counts == {}
    spec, _env = _chain()
    assert IF.check_pair_balance([spec, spec]) == {
        "boolean_rule->graph_edge_semantics": 2}
    other, _e = _chain(1)
    unbalanced = [spec, spec, IF.ChainSpec("x", "famX", "famY", None, None, None)]
    with pytest.raises(IF.ChainSpecError):
        IF.check_pair_balance(unbalanced)
    assert IF.check_pair_balance(unbalanced, require_balanced=False)


# -- recovery arithmetic -------------------------------------------------
def test_recovery_interactions_is_the_first_index_reaching_a1_terminal():
    a1 = S.make_record("a1", "famA", {0: 0.0, 1: 0.5, 2: 1.0})
    a2 = S.make_record("a2", "famA", {0: 0.0, 1: 0.5, 2: 1.0})
    assert IF.recovery_interactions(a1, a2) == 2
    slow = S.make_record("a2", "famA", {0: 0.0, 1: 0.0, 2: 0.5})
    assert IF.recovery_interactions(a1, slow) is None
    easy = S.make_record("a1", "famA", {0: 0.0, 1: 0.0, 2: 0.0})
    assert IF.recovery_interactions(easy, slow) == 0
    assert IF.recovery_interactions(S.make_record("x", "f", {}), a2) is None


# -- the real chain run --------------------------------------------------
def test_run_interference_eval_produces_family_balanced_summaries():
    bundle = S.tiny_bundle()
    spec, env_factory = _chain()
    out = IF.run_interference_eval(bundle, [spec], env_factory, cfg=FAST)
    assert out["schema"] == IF.INTERFERENCE_REPORT_SCHEMA
    assert out["n_chains"] == 1
    assert out["pair_counts"] == {"boolean_rule->graph_edge_semantics": 1}
    assert set(out["records"]) == {"a1", "b", "a2"}
    chain = out["chains"][0]
    assert chain["family_a"] == "boolean_rule"
    assert chain["family_b"] == "graph_edge_semantics"
    for key in ("aulc_a1", "aulc_b", "aulc_a2"):
        assert 0.0 <= chain[key] <= 1.0
    assert out["summary"]["n_chains"] == 1
    assert set(out["summary"]["per_family"]) == {"boolean_rule"}
    assert out["records"]["a1"][0]["chain_position"] == "a1"
    # the B episode really came from the other family
    assert out["records"]["b"][0]["family_id"] == "graph_edge_semantics"


def test_each_chain_owns_one_callbacks_object():
    bundle = S.tiny_bundle()
    spec, env_factory = _chain()
    shared = LC.NullCallbacks()

    class Stateful(LC.NullCallbacks):
        def __init__(self):
            self.episodes = 0

        def on_episode_start(self, ctx):
            self.episodes += 1

    made: list[Stateful] = []

    def factory(chain):
        cb = Stateful()
        made.append(cb)
        return cb

    IF.run_interference_eval(bundle, [spec], env_factory, cfg=FAST,
                             callbacks_factory=factory)
    # one object per chain, reused across that chain's three episodes
    assert len(made) == 1 and made[0].episodes == 3

    # a shared object across two DISTINCT chains is refused before any decode
    second, _env2 = _chain(1)
    with pytest.raises(IF.ChainSpecError, match="DISTINCT"):
        IF.run_interference_eval(bundle, [spec, second], env_factory, cfg=FAST,
                                 callbacks_factory=lambda chain: shared)
    # duplicate chain ids are refused too
    with pytest.raises(IF.ChainSpecError, match="unique"):
        IF.run_interference_eval(bundle, [spec, spec], env_factory, cfg=FAST)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
