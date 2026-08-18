"""Live-mutation interlock, read-only enforcement, authorization attacks."""
from __future__ import annotations

import json
import os
import time

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.provider.runpod.adapter import RunpodV2Adapter
from o1_b200.provider.runpod.authorization import (
    AUTH_SCHEMA, AuthorizationError, CLI_FLAG, ENV_FLAG, ENV_FLAG_VALUE,
    LiveMutationAuthorization,
)
from o1_b200.provider.runpod.mock_server import MockRunpodServer, Scenario
from o1_b200.provider.runpod.pod_request import (
    PodRequestError, build_pod_request, render_canonical_deployment,
)
from o1_b200.provider.runpod.policy import PROFILE_B200, PROFILE_B300
from o1_b200.provider.runpod.transport import ReadOnlyTransport, ReadOnlyViolation

NOSLEEP = lambda s: None  # noqa: E731
GOOD_IMAGE = "ghcr.io/x/o1@sha256:" + "a" * 64


def _requests(image=GOOD_IMAGE, dc="US-KS-2"):
    return {p.key: build_pod_request(profile=p, image_digest_ref=image,
                                     datacenter_id=dc)
            for p in (PROFILE_B300, PROFILE_B200)}


def _identity(rendered):
    return {"project": "O1_B200", "package_zip_sha256": "p" * 64,
            "provider": "runpod", "budget_policy_sha256": "b" * 64,
            "deployment_spec_sha256": rendered["request_sha256"]}


def _authdoc(rendered, **over):
    doc = {
        "schema": AUTH_SCHEMA, "project": "O1_B200",
        "package_zip_sha256": "p" * 64, "provider": "runpod",
        "budget_policy_sha256": "b" * 64,
        "expires_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(time.time() + 3600)),
        "launch_nonce": f"nonce-{time.time_ns()}",
        "deployment_spec_sha256": rendered["request_sha256"],
        "allow_create_pod": True, "allow_real_calibration": True,
        "allow_confirmation": False, "max_pod_creations": 4,
    }
    doc.update(over)
    return doc


def _write(d, doc):
    path = os.path.join(d, "auth.json")
    with open(path, "w") as fh:
        json.dump(doc, fh)
    return path


def _verify(path, rendered, d, cli=None, environ=None):
    return LiveMutationAuthorization.verify(
        path=path, expected_identity=_identity(rendered),
        cli_args=cli if cli is not None else [CLI_FLAG],
        nonce_ledger=os.path.join(d, "nonces.txt"),
        environ=environ)


def run() -> Runner:
    r = Runner("runpod_interlock")
    reqs = _requests()
    req = reqs["B300"]
    rendered = render_canonical_deployment(reqs, {"package": "test"})

    def readonly_rejects_before_network():
        t = ReadOnlyTransport(base_url="http://127.0.0.1:1",  # unroutable
                              api_key="rpa_x_123456789012")
        for method in ("POST", "PATCH", "PUT", "DELETE"):
            try:
                t.request(method, "/v2/pods", {})
            except ReadOnlyViolation:
                continue
            raise AssertionError(f"{method} passed the read-only transport")
        # if it had reached the network, the unroutable port would raise a
        # TransportError instead of ReadOnlyViolation — proving local refusal
    r.check("37/38. read-only transport rejects POST/PATCH/PUT/DELETE "
            "BEFORE any socket (mutating call refused during this task)",
            readonly_rejects_before_network)

    def unauthorized_adapter_cannot_mutate():
        with MockRunpodServer() as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP,
                                 api_key=MOCK_KEY)
            for fn in (lambda: ad.create_instance(req, rendered),
                       lambda: ad.stop_instance("x"),
                       lambda: ad.terminate_instance("x")):
                try:
                    fn()
                except AuthorizationError as exc:
                    assert "LIVE_MUTATION_NOT_AUTHORIZED" in str(exc)
                    continue
                raise AssertionError("mutation without authorization")
            assert not any(m != "GET" for m, _ in srv.scenario.requests), \
                "a mutating request reached the mock server"
    r.check("adapter without authorization cannot mutate (no request sent)",
            unauthorized_adapter_cannot_mutate)

    def full_interlock_requirements():
        d = fresh_dir("interlock_all")
        env_ok = {ENV_FLAG: ENV_FLAG_VALUE}
        # each missing condition must refuse
        cases = []
        cases.append(("missing env flag", _authdoc(rendered), [CLI_FLAG], {}))
        cases.append(("missing CLI flag", _authdoc(rendered), [], env_ok))
        cases.append(("33. missing authorization file", None, [CLI_FLAG], env_ok))
        cases.append(("bad schema", _authdoc(rendered, schema="x.v9"),
                      [CLI_FLAG], env_ok))
        cases.append(("identity mismatch",
                      _authdoc(rendered, package_zip_sha256="q" * 64),
                      [CLI_FLAG], env_ok))
        cases.append(("34. expired", _authdoc(
            rendered, expires_utc="2020-01-01T00:00:00Z"), [CLI_FLAG], env_ok))
        cases.append(("35. wrong deployment digest", _authdoc(
            rendered, deployment_spec_sha256="0" * 64), [CLI_FLAG], env_ok))
        cases.append(("allow_create_pod false", _authdoc(
            rendered, allow_create_pod=False), [CLI_FLAG], env_ok))
        cases.append(("allow_real_calibration false", _authdoc(
            rendered, allow_real_calibration=False), [CLI_FLAG], env_ok))
        cases.append(("32. allow_confirmation true refused", _authdoc(
            rendered, allow_confirmation=True), [CLI_FLAG], env_ok))
        cases.append(("max_pod_creations missing", {
            k: v for k, v in _authdoc(rendered).items()
            if k != "max_pod_creations"}, [CLI_FLAG], env_ok))
        cases.append(("max_pod_creations zero", _authdoc(
            rendered, max_pod_creations=0), [CLI_FLAG], env_ok))
        cases.append(("max_pod_creations above ceiling", _authdoc(
            rendered, max_pod_creations=999), [CLI_FLAG], env_ok))
        cases.append(("max_pod_creations non-integer", _authdoc(
            rendered, max_pod_creations="4"), [CLI_FLAG], env_ok))
        for label, doc, cli, environ in cases:
            path = _write(d, doc) if doc is not None else \
                os.path.join(d, "missing.json")
            try:
                _verify(path, rendered, d, cli=cli, environ=environ)
            except AuthorizationError:
                continue
            raise AssertionError(f"interlock passed with: {label}")
        # and the fully satisfied case verifies
        ok = _verify(_write(d, _authdoc(rendered)), rendered, d,
                     environ=env_ok)
        assert ok.launch_nonce
    r.check("32-35. every interlock condition is individually load-bearing",
            full_interlock_requirements)

    def template_never_authorizes():
        d = fresh_dir("interlock_template")
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "provider",
            "runpod", "B300_RENTAL_AUTHORIZATION.template.json")
        try:
            _verify(os.path.abspath(template_path), rendered, d,
                    environ={ENV_FLAG: ENV_FLAG_VALUE})
        except AuthorizationError as exc:
            assert "UNRESOLVED" in str(exc) or "unresolved" in str(exc)
            return
        raise AssertionError("the shipped template authorized a mutation")
    r.check("the shipped authorization template is unusable by construction",
            template_never_authorizes)

    def nonce_replay_refused():
        d = fresh_dir("interlock_nonce")
        doc = _authdoc(rendered, launch_nonce="replayable-nonce-123456",
                       max_pod_creations=1)
        path = _write(d, doc)
        env_ok = {ENV_FLAG: ENV_FLAG_VALUE}
        auth = _verify(path, rendered, d, environ=env_ok)
        auth.consume_nonce()
        try:
            _verify(path, rendered, d, environ=env_ok)
        except AuthorizationError as exc:
            assert "replay" in str(exc)
            return
        raise AssertionError("nonce replay accepted")
    r.check("36. authorization nonce replay refused", nonce_replay_refused)

    def creation_budget_bounded():
        d = fresh_dir("interlock_slots")
        doc = _authdoc(rendered, launch_nonce="bounded-nonce-abcdef",
                       max_pod_creations=3)
        path = _write(d, doc)
        env_ok = {ENV_FLAG: ENV_FLAG_VALUE}
        auth = _verify(path, rendered, d, environ=env_ok)
        assert auth.creations_remaining() == 3
        for _ in range(3):
            auth.consume_nonce()      # initial + 2 eviction reacquisitions
        assert auth.creations_remaining() == 0
        try:
            auth.consume_nonce()
        except AuthorizationError as exc:
            assert "exhausted" in str(exc)
        else:
            raise AssertionError("4th creation on a 3-slot authorization")
        try:
            _verify(path, rendered, d, environ=env_ok)
        except AuthorizationError as exc:
            assert "replay" in str(exc) or "exhausted" in str(exc)
            return
        raise AssertionError("exhausted authorization re-verified")
    r.check("eviction reacquisition is bounded: max_pod_creations slots, "
            "then LIVE_MUTATION_NOT_AUTHORIZED", creation_budget_bounded)

    def v1_ledger_counts():
        # a v1-format ledger line (bare nonce) counts as one consumed slot
        d = fresh_dir("interlock_v1ledger")
        nonce = "legacy-v1-nonce-xyz123"
        with open(os.path.join(d, "nonces.txt"), "w") as fh:
            fh.write(nonce + "\n")
        doc = _authdoc(rendered, launch_nonce=nonce, max_pod_creations=1)
        path = _write(d, doc)
        try:
            _verify(path, rendered, d, environ={ENV_FLAG: ENV_FLAG_VALUE})
        except AuthorizationError as exc:
            assert "replay" in str(exc) or "exhausted" in str(exc)
            return
        raise AssertionError("v1 ledger entry ignored")
    r.check("legacy v1 nonce-ledger entries still refuse replay",
            v1_ledger_counts)

    def mutable_tag_refused():
        try:
            build_pod_request(profile=PROFILE_B300,
                              image_digest_ref="ghcr.io/x/o1:latest",
                              datacenter_id="US-KS-2")
        except (PodRequestError, ValueError) as exc:
            assert "digest" in str(exc)
            return
        raise AssertionError("mutable :latest tag accepted")
    r.check("40. mutable image tag refused; digest required", mutable_tag_refused)

    def launcher_refuses_without_authorization():
        import subprocess
        root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        proc = subprocess.run(
            ["bash", os.path.join(root, "o1_b200",
                                  "o1_runpod_b300_zero_touch.sh")],
            capture_output=True, text=True, timeout=60)
        assert proc.returncode == 78, proc.returncode
        assert "LIVE_MUTATION_NOT_AUTHORIZED" in proc.stderr
    r.check("zero-touch launcher refuses without --authorization before any "
            "network call", launcher_refuses_without_authorization)

    def secret_scan_in_package_paths():
        # inject a recognizable fake secret and prove the redaction layer and
        # the packaged tree stay clean
        from o1_b200.provider.runpod import redaction
        # assemble the needle at runtime so this source file itself can never
        # trip the scan
        fake = "rpa_" + "INJ" + "ECTED_FAKE_SECRET_0123456789"
        needle = "rpa_" + "INJ" + "ECTED_FAKE"
        redaction.register_secret(fake)
        assert fake not in redaction.redact(f"key={fake}")
        root = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), ".."))
        import subprocess
        out = subprocess.run(
            ["grep", "-rIl", "--exclude-dir=local_runs", "-e", needle,
             "-e", "RUNPOD_API_KEY=rpa" + "_", root],
            capture_output=True, text=True)
        assert out.stdout.strip() == "", f"secret-like content: {out.stdout}"
    r.check("39. injected fake secret never lands in the packaged tree",
            secret_scan_in_package_paths)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
