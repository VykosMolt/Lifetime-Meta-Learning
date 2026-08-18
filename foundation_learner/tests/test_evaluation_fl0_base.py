"""FL0 base-model sweep over the four frozen cells (contract §7, §9)."""

from __future__ import annotations

import json
import os

import pytest

from foundation_learner.evaluation import fl0_base as FL0
from foundation_learner.evaluation.generation import GenerationConfig
from foundation_learner.tests import _w4_support as S

pytestmark = [
    pytest.mark.skipif(not S.checkpoint_present(),
                       reason="frozen checkpoint directory not present"),
    pytest.mark.skipif(not S.integration_available(),
                       reason="ecology/episodes packages absent"),
]


def test_fl0_runs_the_four_frozen_cells_and_emits_records(tmp_path):
    bundle = S.tiny_bundle()
    episodes, _family, env_factory = S.episode_batch(1)
    out = FL0.run_fl0_base(bundle, episodes, env_factory,
                           generation=GenerationConfig(max_new_tokens=1),
                           out_dir=str(tmp_path), run_gate=False,
                           trained_family_ids=["grammar_classification"])
    report = out["report"]
    assert report["schema"] == FL0.FL0_REPORT_SCHEMA
    assert report["arm_tag"] == "FL0_BASE"
    assert report["n_episodes"] == 1
    expected = {f"{f}__{c}" for f in FL0.FEEDBACK_CONDITIONS
                for c in FL0.CONTEXT_MODES}
    assert set(report["cells"]) == expected
    assert set(out["records_by_cell"]) == expected

    for cell, records in out["records_by_cell"].items():
        assert len(records) == 1
        assert records[0]["feedback_condition"] == cell.split("__")[0]
        assert records[0]["context_mode"] == cell.split("__")[1]
        assert os.path.isfile(tmp_path / f"fl0_{cell}_records.jsonl")

    for cell in expected:
        if cell.endswith("context_reset"):
            assert "persistence" in report["cells"][cell]
            assert report["cells"][cell]["persistence"]["reset_from_index"] == 4
        assert report["cells"][cell]["whole_family_transfer"] is not None

    written = json.loads((tmp_path / "fl0_base_report.json").read_text())
    assert written["schema"] == FL0.FL0_REPORT_SCHEMA
    assert (tmp_path / "fl0_base_report.json").read_text().endswith("\n")


def test_fl0_runs_the_equivalence_gate_and_uses_its_policy():
    bundle = S.tiny_bundle()
    episodes, _family, env_factory = S.episode_batch(1)
    out = FL0.run_fl0_base(bundle, episodes, env_factory,
                           generation=GenerationConfig(max_new_tokens=1),
                           run_gate=True)
    gate = out["report"]["equivalence_gate"]
    assert gate is not None and gate["passed"] is True
    assert out["report"]["generation"]["force_unbatched"] is False


def test_load_episodes_reads_the_data_layer_shards(tmp_path):
    from foundation_learner.data import shards

    ep, _family, _rule = S.real_episode()
    path = str(tmp_path / "dev.jsonl")
    shards.write_shard(path, [ep.to_dict()])
    loaded = FL0.load_episodes([path])
    assert len(loaded) == 1
    assert loaded[0].episode_id == ep.episode_id
    assert loaded[0].events[0].role == ep.events[0].role
    assert loaded[0].items == ep.items


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(S.run_module_tests(dict(globals()), __name__))
