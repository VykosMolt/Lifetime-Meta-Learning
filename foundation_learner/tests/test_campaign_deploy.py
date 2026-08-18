"""Deploy artefacts: the environment lock, the pod entry, and the docs (§22).

These are frozen FACTS, so the tests pin them against the contract and against
the code that must agree with them.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY = os.path.join(PACKAGE, "deploy")
LOCK_PATH = os.path.join(DEPLOY, "environment_lock.json")
ENTRY_PATH = os.path.join(DEPLOY, "fl_b200_entry.sh")
INTEGRATION_PATH = os.path.join(DEPLOY, "INTEGRATION.md")


def lock():
    with open(LOCK_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_lock_pins_transformers_exactly_and_matches_the_loader():
    from foundation_learner.training import model_loading as ml

    payload = lock()
    assert payload["schema"] == "flb200.environment_lock.v1"
    assert payload["assert_at_runtime"]["transformers"] == "4.54.1"
    assert ml.REQUIRED_TRANSFORMERS_VERSION == "4.54.1"


def test_the_lock_records_both_torch_builds_with_their_roles():
    drift = lock()["torch_pin_drift"]
    assert drift["local_venv"] == "2.12.1+cu130"
    assert drift["b300_image"] == "2.12.1+cu130"
    assert "LOCAL" in drift["local_role"]
    assert "accelerator" in drift["b300_role"]
    assert "2.12.0.dev20260407+cu128" in drift["sealed_reference"]
    assert "replacement manifest" in drift["sealed_reference"]


def test_the_lock_records_that_peft_is_absent_on_the_b200():
    payload = lock()["no_peft_on_b200"]
    assert "requirements.b300.lock" in payload["fact"]
    assert "training/lora.py" in payload["consequence"]
    assert payload["amendment"].startswith("CONTRACT_AMENDMENTS.md Amendment 3")
    assert lock()["packages"]["peft"].endswith("LOCAL ONLY")


def test_the_lock_binds_the_container_and_changes_nothing_in_it():
    container = lock()["container"]
    assert container["image_name"] == "o1-b300-runner:v0.3.1"
    assert container["base_image"] == "python:3.14-slim-bookworm"
    assert container["base_image_digest"].startswith("python@sha256:")
    assert container["venv"] == "/opt/venv"
    # v0.3.1 bakes the FL SOURCE (the image previously carried none, so a
    # combined session refused at the handover).  What must still be true is
    # that FL changes nothing about the runtime it borrows.
    assert container["fl_changes_to_the_image"].startswith("SOURCE ONLY")
    assert "no package installation" in container["fl_changes_to_the_image"].lower()
    assert container["foundation_learner_source_sha256"]
    assert container["pregen_corpus_baked"].startswith("NO")
    assert lock()["cuda"]["toolkit"] == "13.0"


def test_the_lock_binds_the_frozen_checkpoint_hash():
    from foundation_learner.training import model_loading as ml

    checkpoint = lock()["checkpoint"]
    assert checkpoint["tree_sha256"] == ml.FROZEN_CHECKPOINT_TREE_SHA256
    assert checkpoint["pod_mount"] == "/artifacts/ouro_rltt_local"
    assert checkpoint["baked_into_image"] is False


def test_the_entry_script_is_valid_bash_and_execs_the_supervisor():
    subprocess.run(["bash", "-n", ENTRY_PATH], check=True)
    text = open(ENTRY_PATH, encoding="utf-8").read()
    assert "foundation_learner.campaign.session_supervisor" in text
    assert "/opt/venv/bin/python" in text
    assert "set -euo pipefail" in text


def _local_env():
    """The documented local override: the pod venv does not exist here."""
    import sys

    return dict(os.environ, FL_B200_PYTHON=sys.executable)


def test_the_entry_script_refuses_an_unresolved_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema": "flb200.session_config.v1",
        "o1_entry_command": "UNRESOLVED_OPERATOR_BOUND"}), encoding="utf-8")
    proc = subprocess.run(["bash", ENTRY_PATH, "--config", str(config),
                           "--out", str(tmp_path / "out")],
                          capture_output=True, text=True, env=_local_env())
    assert proc.returncode != 0
    assert "REFUSED" in proc.stdout + proc.stderr
    assert "o1_entry_command" in proc.stdout + proc.stderr


def test_the_entry_script_refuses_a_foreign_schema(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": "o1b200.session_config.v1"}),
                      encoding="utf-8")
    proc = subprocess.run(["bash", ENTRY_PATH, "--config", str(config),
                           "--out", str(tmp_path / "out")],
                          capture_output=True, text=True, env=_local_env())
    assert proc.returncode != 0
    assert "REFUSED" in proc.stdout + proc.stderr


def test_the_entry_script_refuses_a_missing_config(tmp_path):
    proc = subprocess.run(["bash", ENTRY_PATH, "--config",
                           str(tmp_path / "nope.json")],
                          capture_output=True, text=True)
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr


def test_the_entry_script_announces_a_rehearsal_loudly(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "schema": "flb200.session_config.v1", "rehearsal": True,
        "o1_entry_command": "UNRESOLVED_OPERATOR_BOUND"}), encoding="utf-8")
    env = dict(os.environ, FL_B200_PYTHON="/nonexistent/python")
    proc = subprocess.run(["bash", ENTRY_PATH, "--config", str(config)],
                          capture_output=True, text=True, env=env)
    # it refuses on the missing interpreter, i.e. AFTER the config check
    assert "REFUSED: python interpreter not executable" in proc.stderr


def test_the_entry_script_never_touches_o1_files():
    text = open(ENTRY_PATH, encoding="utf-8").read()
    for forbidden in ("o1_b200/", "/outputs", "start_b200.sh"):
        assert not re.search(rf"^[^#]*{re.escape(forbidden)}", text,
                             re.MULTILINE), forbidden


def test_the_session_config_template_refuses_as_shipped():
    from foundation_learner.campaign import session_supervisor as ss

    path = os.path.join(DEPLOY, "FL_SESSION_CONFIG.template.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["schema"] == ss.SESSION_CONFIG_SCHEMA
    assert payload["rehearsal"] is False
    for key in ss._REQUIRED_CONFIG_FIELDS:
        assert key in payload, key
    with pytest.raises(ss.SessionConfigError) as exc:
        ss.SessionConfig(payload=dict(payload)).validate()
    assert "o1_entry_command" in str(exc.value)
    from foundation_learner.campaign.o1_isolation import FROZEN_FORBIDDEN_ROOTS

    assert tuple(payload["o1_roots"]) == FROZEN_FORBIDDEN_ROOTS


def test_integration_doc_states_the_unresolved_gap_honestly():
    text = open(INTEGRATION_PATH, encoding="utf-8").read()
    assert "o1_entry_command" in text
    # the historical refusing-stub gap is CLOSED by the B300 migration and
    # the doc must say so honestly, keeping the history visible
    assert "RESOLVED (B300 migration)" in text
    assert "refusing stub" in text
    assert "start_b300.sh" in text
    assert "NOT AUTHORIZED" in text
    # From v0.3.1 the image DOES carry the FL source — the doc must say that
    # plainly rather than keep repeating the old "no image change" claim, and
    # must still record what stayed unchanged (the borrowed runtime).
    # Prose wraps, so match against whitespace-normalised text.
    flat = " ".join(text.split())
    assert "no package installation" in flat
    assert "no dependency change" in flat
    assert "/opt/foundation_learner" in flat
    assert "exit 78" in flat, (
        "the doc must record what the pre-v0.3.1 image actually did when a "
        "combined session was requested")


def test_the_isolation_roots_in_the_doc_match_the_code():
    from foundation_learner.campaign.o1_isolation import (
        FROZEN_FORBIDDEN_ROOTS, SHARED_READONLY_INPUTS)

    text = open(INTEGRATION_PATH, encoding="utf-8").read()
    for root in FROZEN_FORBIDDEN_ROOTS + SHARED_READONLY_INPUTS:
        assert root in text, root


def test_the_budget_policy_is_referenced_and_separate_from_o1():
    text = open(INTEGRATION_PATH, encoding="utf-8").read()
    assert "FL_BUDGET_POLICY.json" in text
    assert "runner/budget.py" in text
    assert "1200" in text and "1.25" in text
