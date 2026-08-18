"""Hostile regression: live provider credentials can never contaminate mocks.

Empirically reproduced defect (pre-repair): with a real RUNPOD_API_KEY
exported, os.environ.setdefault() in the mock tests was a no-op, transports
built without an explicit key fell back to load_api_key(), and the local
mock suite sent the OPERATOR credential to the mock server (which correctly
401'd the unexpected key).  This suite proves the repair and pins the
invariant: local/mock tests are hermetic with respect to
RUNPOD_API_KEY / RUNPOD_API_KEY_FILE, in every direction.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from _h import SCRATCH, Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

# assembled at runtime so the tree's secret scan can never trip on this file
FAKE_LIVE_KEY = "rpa_" + "FAKELIVE" + "_operator_key_0123456789abcdef"

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUITE = ["test_runpod_adapter.py", "test_runpod_interlock.py",
          "test_runpod_lifecycle.py", "test_runpod_zero_touch.py"]


def _base_env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("RUNPOD_API_KEY", "RUNPOD_API_KEY_FILE",
                        "RUNPOD_MOCK_API_KEY")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_one(label: str, name: str, env: dict) -> tuple[str, str, dict]:
    env = dict(env)
    # Per-(condition, module) scratch so the 16 runs are concurrent-safe:
    # every module picks fixed fresh_dir() names, which would otherwise be
    # deleted out from under a sibling.  Still under SCRATCH, so the
    # credential-leak grep below covers everything they wrote.
    env["O1_B200_TEST_SCRATCH"] = os.path.join(SCRATCH, "cred_iso_runs",
                                               label, name)
    os.makedirs(env["O1_B200_TEST_SCRATCH"], exist_ok=True)
    proc = subprocess.run(
        [sys.executable, os.path.join(_TESTS_DIR, name)],
        capture_output=True, text=True, timeout=600,
        cwd=_TESTS_DIR, env=env)
    text = proc.stdout + proc.stderr
    m = re.search(r"\[(\w+)\] (\d+) passed, (\d+) failed", text)
    return label, name, {"rc": proc.returncode, "output": text,
                         "summary": m.group(0) if m else "NO_SUMMARY"}


def _run_conditions(specs: dict[str, dict]) -> dict[str, dict]:
    """Run every (credential condition x mock module) pair concurrently."""
    out: dict[str, dict] = {label: {} for label in specs}
    with ThreadPoolExecutor(
            max_workers=min(len(specs) * len(_SUITE),
                            (os.cpu_count() or 1))) as pool:
        futures = [pool.submit(_run_one, label, name, env)
                   for label, env in specs.items() for name in _SUITE]
        for fut in futures:
            label, name, res = fut.result()
            out[label][name] = res
    return out


def run() -> Runner:
    r = Runner("credential_isolation")

    conditions: dict[str, dict] = {}

    def suite_hermetic_across_credential_conditions():
        d = fresh_dir("cred_iso")
        # Two distinct key files: the owner-only and world-readable forms are
        # separate fixtures so the conditions can run concurrently instead of
        # chmod-ing one shared file out from under each other.
        def _keyfile(name: str, mode: int) -> str:
            path = os.path.join(d, name)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(FAKE_LIVE_KEY + "\n")
            os.chmod(path, mode)
            return path

        # condition 1: no live credential variables at all
        env1 = _base_env()
        # condition 2: a recognizable fake live RUNPOD_API_KEY exported
        env2 = _base_env()
        env2["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        # condition 3: fake RUNPOD_API_KEY and RUNPOD_API_KEY_FILE together
        env3 = _base_env()
        env3["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        env3["RUNPOD_API_KEY_FILE"] = _keyfile(
            "fake_key_file", stat.S_IRUSR | stat.S_IWUSR)
        # condition 4 (hostile): ONLY a key file, deliberately world-readable.
        # If any mock code path consulted load_api_key()'s file fallback it
        # would raise SecretHandlingError and the suite outcome would change.
        env4 = _base_env()
        env4["RUNPOD_API_KEY_FILE"] = _keyfile("fake_key_file_world", 0o644)

        conditions.update(_run_conditions({
            "none": env1, "fake_key": env2,
            "fake_key_and_file": env3, "world_readable_file_only": env4}))

        baseline = conditions["none"]
        for label, results in conditions.items():
            for name, res in results.items():
                assert res["rc"] == 0, (
                    f"[{label}] {name} failed (rc={res['rc']}):\n"
                    f"{res['output'][-2000:]}")
                assert res["summary"] == baseline[name]["summary"], (
                    f"[{label}] {name} summary {res['summary']!r} differs "
                    f"from baseline {baseline[name]['summary']!r}")
    r.check("complete local/mock RunPod suite passes IDENTICALLY with no "
            "credential vars, fake RUNPOD_API_KEY, fake key+file, and a "
            "world-readable key file", suite_hermetic_across_credential_conditions)

    def fake_key_never_in_outputs():
        assert conditions, "depends on the previous check having run"
        for label, results in conditions.items():
            for name, res in results.items():
                assert FAKE_LIVE_KEY not in res["output"], (
                    f"[{label}] {name} echoed the live-style credential")
        # nothing under the test scratch tree may carry the fake live key
        # (excluding this suite's own key-file fixture under cred_iso/)
        hits = subprocess.run(
            ["grep", "-rIl", "--exclude-dir=cred_iso", FAKE_LIVE_KEY, SCRATCH],
            capture_output=True, text=True)
        assert hits.stdout.strip() == "", (
            f"live-style credential persisted into: {hits.stdout}")
    r.check("the fake live credential appears in no test output and no "
            "scratch artifact", fake_key_never_in_outputs)

    def transport_refuses_ambient_credentials_for_mock_hosts():
        from o1_b200.provider.runpod.transport import (
            CredentialIsolationError, ReadOnlyTransport)
        os.environ["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        try:
            try:
                ReadOnlyTransport(base_url="http://127.0.0.1:9")
            except CredentialIsolationError:
                pass
            else:
                raise AssertionError(
                    "transport for a non-production host accepted the "
                    "ambient credential fallback")
            # explicit synthetic keys still work for non-production hosts
            t = ReadOnlyTransport(base_url="http://127.0.0.1:9",
                                  api_key=MOCK_KEY)
            assert t.has_credential()
        finally:
            os.environ["RUNPOD_API_KEY"] = MOCK_KEY
    r.check("transport structurally refuses ambient credentials for any "
            "non-production base_url", transport_refuses_ambient_credentials_for_mock_hosts)

    def mock_server_still_validates_auth():
        from o1_b200.provider.runpod.mock_server import MockRunpodServer
        from o1_b200.provider.runpod.adapter import RunpodV2Adapter
        from o1_b200.provider.runpod.transport import ApiHttpError
        with MockRunpodServer() as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url,
                                 api_key=FAKE_LIVE_KEY, sleep=lambda s: None)
            try:
                ad.list_gpu_types()
            except ApiHttpError as exc:
                assert exc.status == 401
                assert FAKE_LIVE_KEY not in str(exc)
                return
        raise AssertionError("mock server accepted a non-synthetic key — "
                             "authentication checking was weakened")
    r.check("mock server still 401s any key other than the synthetic mock "
            "credential (auth not weakened)", mock_server_still_validates_auth)

    def watchdog_refuses_operator_key_for_mock_hosts():
        from o1_b200.provider.runpod.watchdog_terminate import _load_key
        os.environ["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        os.environ.pop("RUNPOD_MOCK_API_KEY", None)
        try:
            assert _load_key("127.0.0.1") is None, (
                "watchdog offered the operator credential to a mock host")
            assert _load_key("api.runpod.io") == FAKE_LIVE_KEY
            os.environ["RUNPOD_MOCK_API_KEY"] = MOCK_KEY
            assert _load_key("127.0.0.1") == MOCK_KEY
        finally:
            os.environ.pop("RUNPOD_MOCK_API_KEY", None)
            os.environ["RUNPOD_API_KEY"] = MOCK_KEY
    r.check("independent watchdog path never sends the operator credential "
            "to a non-production host", watchdog_refuses_operator_key_for_mock_hosts)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
