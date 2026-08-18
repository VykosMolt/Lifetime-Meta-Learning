"""One HF_TOKEN, two directions: prove it before the ladder starts."""
from __future__ import annotations

import json
import os

import pytest

from foundation_learner.campaign import check_hf_scope as cs


def runner_returning(write_ok: bool, calls: list | None = None):
    def run(repo, mode):
        if calls is not None:
            calls.append((repo, mode))
        return {"repo": repo, "mode": mode, "read": True,
                "write": write_ok and mode == "write", "identity": "tester"}
    return run


# ----------------------------------------------------------------------- parse


def test_repo_parsing():
    assert cs.parse_repo("hf://ns/repo") == "ns/repo"
    assert cs.parse_repo("hf://ns/repo/fl/session") == "ns/repo"
    assert cs.parse_repo("/mounted/path") is None
    assert cs.parse_repo("") is None


@pytest.mark.parametrize("uri", ["hf://onlyone", "hf://", "hf:///repo"])
def test_a_malformed_hf_uri_refuses(uri):
    with pytest.raises(cs.ScopeError):
        cs.parse_repo(uri)


# ----------------------------------------------------------------------- check


def test_a_read_only_token_refuses():
    with pytest.raises(cs.ScopeError, match="cannot write"):
        cs.check("hf://ns/pregen", "hf://ns/fl-results",
                 runner=runner_returning(False))


def test_a_read_write_token_passes():
    calls: list = []
    report = cs.check("hf://ns/pregen", "hf://ns/fl-results",
                      runner=runner_returning(True, calls))
    assert report["schema"] == "flb200.hf_scope_report.v1"
    assert ("ns/pregen", "read") in calls
    assert ("ns/fl-results", "write") in calls


def test_one_repo_in_both_directions_is_still_write_probed():
    calls: list = []
    cs.check("hf://ns/same", "hf://ns/same", runner=runner_returning(True, calls))
    assert ("ns/same", "read") in calls and ("ns/same", "write") in calls


def test_an_unset_durable_destination_refuses():
    with pytest.raises(cs.ScopeError, match="durable mirror"):
        cs.check("hf://ns/pregen", "", runner=runner_returning(True))


def test_an_unresolved_durable_destination_refuses():
    with pytest.raises(cs.ScopeError, match="durable mirror"):
        cs.check("hf://ns/pregen", "UNRESOLVED_OPERATOR_BOUND (hf://ns/x)",
                 runner=runner_returning(True))


# ------------------------------------------------------------- local mirror


def test_a_mounted_destination_is_proven_by_writing(tmp_path):
    dest = str(tmp_path / "durable")
    report = cs.check("", dest, runner=runner_returning(True))
    assert report["checks"][0]["write"] is True
    assert not os.path.exists(os.path.join(dest, ".preflight_write_probe"))


def test_an_unwritable_mounted_destination_refuses(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        with pytest.raises(cs.ScopeError, match="not writable"):
            cs.check("", str(blocked / "durable"), runner=runner_returning(True))
    finally:
        blocked.chmod(0o700)


# ------------------------------------------------------------------------ main


def _config(tmp_path, **over) -> str:
    cfg = {"schema": "flb200.session_config.v1", "rehearsal": False,
           "fl_durable_destination": "hf://ns/fl-results"}
    cfg.update(over)
    path = str(tmp_path / "session.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return path


def test_main_takes_the_destination_from_the_session_config(
        tmp_path, monkeypatch):
    dest = str(tmp_path / "durable")
    config = _config(tmp_path, fl_durable_destination=dest)
    out = str(tmp_path / "scope.json")
    monkeypatch.setenv("HF_TOKEN", "synthetic-not-a-real-token")
    monkeypatch.setattr("sys.argv", [
        "check_hf_scope", "--config", config, "--out", out])
    assert cs.main() == 0
    with open(out, encoding="utf-8") as fh:
        assert json.load(fh)["checks"][0]["write"] is True


def test_main_refuses_a_real_session_with_an_unresolved_destination(
        tmp_path, monkeypatch, capsys):
    config = _config(tmp_path,
                     fl_durable_destination="UNRESOLVED_OPERATOR_BOUND")
    monkeypatch.setenv("HF_TOKEN", "synthetic-not-a-real-token")
    monkeypatch.setattr("sys.argv", ["check_hf_scope", "--config", config])
    assert cs.main() == 2
    assert "durable mirror" in capsys.readouterr().err


def test_a_rehearsal_with_no_destination_is_announced_not_refused(
        tmp_path, monkeypatch, capsys):
    config = _config(tmp_path, rehearsal=True,
                     fl_durable_destination="UNRESOLVED_OPERATOR_BOUND")
    monkeypatch.setattr("sys.argv", ["check_hf_scope", "--config", config])
    assert cs.main() == 0
    assert "DRESS_REHEARSAL" in capsys.readouterr().err


def test_main_refuses_without_a_token_for_an_hf_destination(
        tmp_path, monkeypatch, capsys):
    config = _config(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["check_hf_scope", "--config", config])
    assert cs.main() == 2
    assert "HF_TOKEN is unset" in capsys.readouterr().err


# ------------------------------------------------------------ recursion guard
#
# The FL supervisor runs O1's pod entrypoint as a subprocess, and that
# entrypoint dispatches to the FL supervisor whenever it sees
# O1_FL_SESSION_CONFIG.  With the FL package now actually present in the
# image, an unstripped environment is an unbounded recursion on a paid
# accelerator.  Two independent guards, both asserted here.


def test_the_supervisor_strips_the_session_config_from_the_o1_child(monkeypatch):
    from foundation_learner.campaign import session_supervisor as ss
    monkeypatch.setenv("O1_FL_SESSION_CONFIG", "/session.json")
    env = ss.SessionSupervisor._child_env(object())
    assert "O1_FL_SESSION_CONFIG" not in env, (
        "the O1 phase would see the session config and start a second "
        "combined session")
    assert env["O1_B300_ENTRY_ACTIVE"] == "1"


def test_the_o1_entrypoint_refuses_to_redispatch_when_already_inside():
    """O1's own guard, independent of the supervisor doing the right thing."""
    import subprocess
    start = "/opt/o1_b200/o1_b200/deploy/start_b300.sh"
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "o1-v2-b200-runner", "o1_b200", "deploy", "start_b300.sh")
    path = start if os.path.exists(start) else local
    if not os.path.exists(path):
        pytest.skip("O1 entrypoint not present in this checkout")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "O1_B300_ENTRY_ACTIVE" in text, "the recursion guard is gone"
    guard = text.index("O1_B300_ENTRY_ACTIVE")
    dispatch = text.index("combined session: handing over")
    assert guard < dispatch, (
        "the guard must be evaluated before the FL dispatch")
    subprocess.run(["bash", "-n", path], check=True)


# -------------------------------------------------------------- entry wiring


def test_the_fl_entry_checks_scope_before_downloading_the_corpus():
    entry = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "deploy", "fl_b200_entry.sh")
    with open(entry, encoding="utf-8") as fh:
        text = fh.read()
    assert "check_hf_scope" in text
    assert text.index("check_hf_scope") < text.index("fetch_pregen"), (
        "the scope check must precede the ~480 MB corpus download")
    assert text.index("fetch_pregen") < text.index("session_supervisor"), (
        "the corpus must be on the pod before the supervisor starts")
