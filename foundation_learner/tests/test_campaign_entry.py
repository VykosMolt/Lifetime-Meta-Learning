"""The PRODUCTION campaign entry (contract §13, §22; review R-C1 + R-M9).

``session_supervisor.main()`` is what ``deploy/fl_b200_entry.sh`` execs on the
pod.  These tests walk it end to end with NO injected context factory and NO
injected ladder runner, so the wiring under test is the production wiring: the
context is built by ``campaign.entry.build_stage_context`` from the session
config, and the ladder is the real scheduler over the real stage table.

The only thing the rehearsal changes is the bundle SOURCE (``fl_tiny_model``);
the code path is otherwise identical to a real session, which is exactly what
makes this test meaningful.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

from foundation_learner.campaign import entry as ce
from foundation_learner.campaign import o1_isolation
from foundation_learner.campaign import session_supervisor as ss
from foundation_learner.campaign import stage_definitions as sd
from foundation_learner.ecology.base import sha256_tree



def _pregeneration_matches_the_generators() -> bool:
    """Is the tiny pre-generation current with respect to the generators?

    The sealed gate verifies the split manifest's generator source digests
    before it will derive the key, so a stale pre-generation legitimately
    refuses the opening.  That is a DATA precondition of this walk, not the
    behaviour under test, and it is asserted separately below either way.
    """
    from foundation_learner.ecology.manifests import (read_split_manifest,
                                                      verify_split_manifest)

    path = os.path.join(TINY, "family_split_manifest.json")
    if not os.path.isfile(path):
        return False
    try:
        verify_split_manifest(read_split_manifest(path), check_sources=True)
    except Exception:                       # noqa: BLE001 - a precondition probe
        return False
    return True

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TINY = os.path.join(REPO, "artifacts_fl", "pregen_tiny")
#: Several tests build a real StageContext, which REFUSES a missing
#: pre-generation root (that refusal is itself tested below with an explicitly
#: nonexistent path).  In a fresh clone the tiny pre-generation is not staged
#: yet, so those tests skip exactly like their siblings instead of hard-failing
#: on absent DATA (Amendment 16, Defect C).
requires_tiny_pregen = pytest.mark.skipif(
    not os.path.isfile(os.path.join(TINY, "PREGEN_MANIFEST.json")),
    reason="no tiny pregeneration present (stage artifacts_fl/pregen_tiny)")


STUB = (
    "import hashlib,json,os,sys;"
    "root=sys.argv[1];"
    "os.makedirs(root, exist_ok=True);"
    "open(os.path.join(root,'o1_records.jsonl'),'w').write('{\"row\": 0}\\n');"
    "open(os.path.join(root,'O1_COMPLETE'),'w').write('done\\n');"
    "d=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest();"
    "json.dump({'schema':'o1b200.transfer_manifest.v1','artifacts':{"
    "'records':{'path':'o1_records.jsonl','kind':'file',"
    "'sha256':d(os.path.join(root,'o1_records.jsonl'))}}},"
    "open(os.path.join(root,'TRANSFER_MANIFEST.json'),'w'))"
)


def write_config(tmp_path, **overrides):
    o1_root = tmp_path / "o1_calibration"
    o1_root.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "ouro"}\n',
                                            encoding="utf-8")
    payload = {
        "schema": ss.SESSION_CONFIG_SCHEMA,
        "session_id": "ENTRY_TEST",
        "label": "ENTRY_TEST",
        "rehearsal": True,
        "o1_entry_command": [sys.executable, "-c", STUB, str(o1_root)],
        "o1_completion_markers": [str(o1_root / "O1_COMPLETE")],
        "o1_hash_manifests": [str(o1_root / "TRANSFER_MANIFEST.json")],
        "o1_transfer_command": [sys.executable, "-c", "print('t')"],
        "o1_close_command": [sys.executable, "-c", "print('c')"],
        "o1_roots": [str(o1_root)],
        "checkpoint_dir": str(checkpoint),
        "checkpoint_tree_sha256": sha256_tree(str(checkpoint)),
        "pregen_root": TINY,
        "fl_out_dir": str(tmp_path / "session"),
        "session_authorized_seconds": 3600.0,
        "fl_transfer_command": [sys.executable, "-c", "print('ft')"],
        "terminate_command": [sys.executable, "-c", "print('term')"],
        # the ONLY rehearsal-specific switch: the bundle source
        "fl_tiny_model": True,
        "fl_device": "cpu",
        "fl_eval_episode_cap": 1,
        "fl_train_example_cap": 2,
        "fl_extra": {
            "update_ladder": [2],
            "bench_probe_updates": 2,
            "bench_probe_episodes": 1,
            "bench_declared_budget_seconds": 900.0,
            "run_equivalence_gate": False,
            "max_new_tokens": 8,
        },
    }
    payload.update(overrides)
    path = tmp_path / "session_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")
    return str(path), payload


@pytest.fixture(autouse=True)
def _restore_global_torch_flags():
    """The entry configures torch GLOBALLY (as it must in production).

    The flags are snapshotted and restored around every test in this module so
    that exercising the production behaviour here cannot silently change the
    execution mode of unrelated test modules later in the session.
    """
    before = (torch.are_deterministic_algorithms_enabled(),
              torch.backends.cudnn.deterministic,
              torch.backends.cudnn.benchmark)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic = before[1]
        torch.backends.cudnn.benchmark = before[2]


def supervisor_for(tmp_path, config_path):
    guard = o1_isolation.IsolationGuard(label="ENTRY_TEST")
    return ss.SessionSupervisor(
        config=ss.SessionConfig.load(config_path, guard=guard),
        out_dir=str(tmp_path / "session"), guard=guard)


# ---------------- determinism (R-M9) ----------------

def test_the_entry_applies_and_records_the_determinism_configuration(tmp_path):
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    record = ce.configure_determinism(label="TEST")
    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert record["deterministic_algorithms"] is True
    assert record["called"].endswith("set_deterministic_eval")


def test_attaching_the_entry_journals_the_determinism_record(tmp_path):
    path, _ = write_config(tmp_path)
    sup = supervisor_for(tmp_path, path)
    ce.attach_to_supervisor(sup)
    events = [r for r in sup.read_journal()
              if r["event"] == "CAMPAIGN_ENTRY_WIRED"]
    assert len(events) == 1
    assert events[0]["determinism"]["deterministic_algorithms"] is True
    assert sup.context_factory is ce.build_stage_context
    assert sup.ladder_runner is ce.run_ladder


def test_an_injected_factory_is_left_alone(tmp_path):
    path, _ = write_config(tmp_path)
    sup = supervisor_for(tmp_path, path)
    sentinel = lambda s: None            # noqa: E731 - a test injection point
    sup.context_factory = sentinel
    ce.attach_to_supervisor(sup)
    assert sup.context_factory is sentinel


# ---------------- the context the production path builds ----------------

@requires_tiny_pregen
def test_build_stage_context_binds_the_session_configuration(tmp_path):
    path, payload = write_config(tmp_path)
    sup = supervisor_for(tmp_path, path)
    ctx = ce.build_stage_context(sup)
    assert ctx.pregen_root == os.path.realpath(TINY)
    assert ctx.out_dir == os.path.join(sup.out_dir, "ladder")
    assert ctx.checkpoint_dir == payload["checkpoint_dir"]
    assert ctx.rehearsal is True
    assert ctx.extra["bundle_source"]["source"] == "TINY_NONSCIENTIFIC_MODEL"
    bundle = ctx.bundle_factory()
    assert bundle.identity["scientific"] is False


def test_the_real_path_loads_the_frozen_backbone_with_tree_verification(tmp_path):
    """The REAL bundle factory is the frozen loader with verify_tree_hash."""
    path, payload = write_config(tmp_path)
    payload = dict(payload, fl_tiny_model=False)
    factory, description = ce.build_bundle_factory(payload, rehearsal=False)
    assert description["source"] == "FROZEN_BACKBONE"
    assert description["verify_tree_hash"] is True
    assert description["checkpoint_dir"] == payload["checkpoint_dir"]
    # calling it must go through the frozen loader (which refuses this stub
    # checkpoint directory) - never a silent fallback to something loadable
    from foundation_learner.training.model_loading import FrozenBindingError

    with pytest.raises(FrozenBindingError):
        factory()


def test_the_tiny_model_is_refused_outside_a_rehearsal(tmp_path):
    _, payload = write_config(tmp_path)
    with pytest.raises(ce.EntryError) as exc:
        ce.build_bundle_factory(dict(payload), rehearsal=False)
    assert "rehearsal" in str(exc.value)


def test_evaluation_caps_are_refused_outside_a_rehearsal(tmp_path):
    path, _ = write_config(tmp_path, rehearsal=False)
    sup = supervisor_for(tmp_path, path)
    with pytest.raises(ce.EntryError) as exc:
        ce.build_stage_context(sup)
    assert "fl_eval_episode_cap" in str(exc.value)


def test_a_missing_pregen_root_refuses(tmp_path):
    path, _ = write_config(tmp_path, pregen_root=str(tmp_path / "nope"))
    sup = supervisor_for(tmp_path, path)
    with pytest.raises(ce.EntryError) as exc:
        ce.build_stage_context(sup)
    assert "pregen_root" in str(exc.value)


# ---------------- main(): the pod entry, no injection at all --------------

@requires_tiny_pregen
def test_main_runs_the_ladder_with_no_injected_factory(tmp_path):
    """R-C1: the pod entry must be able to run the ladder by itself."""
    path, _ = write_config(tmp_path)
    out = str(tmp_path / "session")
    rc = ss.main(["--config", path, "--out", out, "--no-resume"])
    status = json.loads(open(os.path.join(out, "SESSION_FINAL_STATUS.json"),
                             encoding="utf-8").read())
    assert status["outcome"] == "COMPLETE", status["failure"]
    assert rc == 0
    assert status["determinism"]["deterministic_algorithms"] is True

    summary = json.loads(open(os.path.join(out, "ladder", "LADDER_SUMMARY.json"),
                              encoding="utf-8").read())
    states = summary["states"]
    stages = {s["stage_id"]: s for s in summary["stages"]}
    # the production context really produced a runnable ladder
    assert states["BENCH"] == "COMPLETE", states
    assert states["FL0"] == "COMPLETE", states
    assert states["DEV_GRID"] == "COMPLETE", states
    for arm in ("FL1", "FL2", "FL3"):
        assert states[arm] == "COMPLETE", states
    assert states["CORE_MATCHING"] == "COMPLETE", stages["CORE_MATCHING"]
    for diag in ("REMAP_DIAG", "INTERFERENCE_DIAG", "POISON_DIAG"):
        assert states[diag] == "COMPLETE", stages[diag]
        report = json.loads(open(os.path.join(
            out, "ladder", diag.lower(),
            f"{diag.lower()}_report.json"), encoding="utf-8").read())
        # the diagnostics run the PROMOTED arm, restored from its checkpoint
        assert report["model"]["promoted_arm"] == "FL3"
        assert report["model"]["loaded"] is True
        # and their evaluation set covers every DEVELOPMENT family
        coverage = report["eval_plan"]["coverage"]["episodes_per_family"]
        assert set(coverage) == set(sd.split_families("DEVELOPMENT")), coverage

    ledger = os.path.join(out, "ladder", "SEALED_OPENING_LEDGER.jsonl")
    if _pregeneration_matches_the_generators():
        assert states["SEALED_EVAL"] == "COMPLETE", stages["SEALED_EVAL"]
        report = json.loads(open(os.path.join(
            out, "ladder", "sealed_eval", "sealed_eval_report.json"),
            encoding="utf-8").read())
        assert report["promoted_arm"] == "FL3"
        assert report["promoted_arm_checkpoint"]["loaded"] is True
        events = [json.loads(l)["event"] for l in open(ledger, encoding="utf-8")
                  if l.strip()]
        assert events[0] == "SEALED_OPENED"
    else:
        # the tiny pre-generation is stale with respect to the current
        # generator sources.  The sealed gate must then refuse in PHASE ONE and
        # consume NO opening attempt: no ledger, no burnt seal.
        assert states["SEALED_EVAL"] == "FAILED", stages["SEALED_EVAL"]
        assert "does not match the manifest" in stages["SEALED_EVAL"]["error"]
        assert not os.path.exists(ledger), (
            "a refusal before the provisional opening must write NO ledger "
            "entry, so it cannot consume one of the two attempts")

    # the bench really measured the two new projection inputs
    bench = json.loads(open(os.path.join(out, "ladder", "bench",
                                         "bench_measurement.json"),
                            encoding="utf-8").read())
    assert bench["model_load_seconds"] > 0.0
    assert bench["forward_seconds_per_episode"] > 0.0
