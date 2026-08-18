"""Regression coverage for the defects found by the adversarial review.

Every check here corresponds to a finding that was empirically reproduced
BEFORE the repair, so each one fails against the pre-repair code.  Grouped
by the review's finding ids.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.provider.runpod import redaction
from o1_b200.provider.runpod.adapter import RunpodV2Adapter
from o1_b200.provider.runpod.authorization import (
    AUTH_SCHEMA, AuthorizationError, CLI_FLAG, ENV_FLAG, ENV_FLAG_VALUE,
    LiveMutationAuthorization,
)
from o1_b200.provider.runpod.billing import (
    BudgetViolation, hard_compute_seconds, remaining_compute_seconds,
)
from o1_b200.provider.runpod.mock_server import MockRunpodServer, Scenario
from o1_b200.provider.runpod.pod_request import (
    PodRequestError, build_pod_request, identity_body,
    render_canonical_deployment, verify_deployment_rendering,
)
from o1_b200.provider.runpod.policy import (
    MAX_COMPUTE_USD, PROFILE_B200, PROFILE_B300,
)
from o1_b200.provider.runpod.transport import (
    CredentialIsolationError, ReadOnlyTransport, resolve_credential,
)

NOSLEEP = lambda s: None  # noqa: E731
GOOD_IMAGE = "ghcr.io/x/o1@sha256:" + "a" * 64
FAKE_LIVE_KEY = "rpa_" + "REVIEWLIVE" + "_operator_0123456789abcdef"


def _reqs(env_values=None):
    return {p.key: build_pod_request(profile=p, image_digest_ref=GOOD_IMAGE,
                                     datacenter_id="US-KS-2",
                                     env_values=env_values)
            for p in (PROFILE_B300, PROFILE_B200)}


def run() -> Runner:
    r = Runner("review_repairs")

    # ---------------- A1: credential resolution is centralized ----------

    def a1_preflight_cannot_forward_ambient_credential():
        from o1_b200.provider.runpod.preflight import run_preflight
        os.environ["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        try:
            for base in ("http://127.0.0.1:9", "https://evil.example"):
                try:
                    run_preflight(base_url=base)
                except CredentialIsolationError:
                    continue
                raise AssertionError(
                    f"preflight forwarded the ambient operator credential "
                    f"to {base}")
            # resolve_credential is the single decision point
            try:
                resolve_credential("http://127.0.0.1:9", None)
            except CredentialIsolationError:
                pass
            else:
                raise AssertionError("resolve_credential leaked")
            assert resolve_credential("http://127.0.0.1:9", MOCK_KEY) == MOCK_KEY
            assert resolve_credential(
                "https://api.runpod.io", None) == FAKE_LIVE_KEY
        finally:
            os.environ["RUNPOD_API_KEY"] = MOCK_KEY
    r.check("A1. preflight/session cannot forward the ambient operator "
            "credential to a non-production base_url",
            a1_preflight_cannot_forward_ambient_credential)

    # ---------------- A2/A3: redaction reaches every rendering ----------

    def a2_a3_secret_shapes_all_redacted():
        token = "hf_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        shapes = {
            "bare": token,
            "graphql_env": json.dumps(
                {"env": [{"key": "HF_TOKEN", "value": token}]}),
            "graphql_env_opaque": json.dumps(
                {"env": [{"key": "HF_TOKEN", "value": "opaque-Zz9-abc123"}]}),
            "nested_json": json.dumps(
                {"event": "E", "data": json.dumps({"HF_TOKEN": token})}),
            "rest_env": json.dumps({"env": {"HF_TOKEN": token}}),
        }
        for label, text in shapes.items():
            out = redaction.redact(text)
            assert token not in out, f"{label} leaked the token"
            if label == "graphql_env_opaque":
                assert "opaque-Zz9-abc123" not in out, \
                    "an opaque secret in the key/value rendering leaked"
        # literal registration covers unrecognized shapes
        os.environ["HF_TOKEN"] = "unrecognizable-shape-9911"
        try:
            assert "HF_TOKEN" in redaction.register_env_secrets()
            assert "unrecognizable-shape-9911" not in redaction.redact(
                json.dumps({"a": {"b": "unrecognizable-shape-9911"}}))
        finally:
            os.environ.pop("HF_TOKEN", None)
    r.check("A2/A3. HF tokens and key/value-rendered secrets are redacted, "
            "including inside nested JSON strings",
            a2_a3_secret_shapes_all_redacted)

    def a5_watchdog_refuses_cleartext_and_foreign_hosts():
        from o1_b200.provider.runpod.watchdog_terminate import _load_key
        os.environ["RUNPOD_API_KEY"] = FAKE_LIVE_KEY
        os.environ.pop("RUNPOD_MOCK_API_KEY", None)
        try:
            assert _load_key("api.runpod.io", "https") == FAKE_LIVE_KEY
            assert _load_key("api.runpod.io", "http") is None, \
                "operator credential offered over cleartext HTTP"
            assert _load_key("127.0.0.1", "https") is None
        finally:
            os.environ["RUNPOD_API_KEY"] = MOCK_KEY
    r.check("A5. the watchdog offers the operator credential only to the "
            "production host over TLS",
            a5_watchdog_refuses_cleartext_and_foreign_hosts)

    # ---------------- B1: per-pod deadlines are budget-aware -------------

    def b1_deadlines_derive_from_remaining_allocation():
        rate = "7.89"
        full = hard_compute_seconds(rate)
        assert remaining_compute_seconds(rate, "0") == full
        # once spend accumulates, every later deadline must shrink
        half = remaining_compute_seconds(rate, str(MAX_COMPUTE_USD / 2))
        assert 0 < half < full and abs(half - full / 2) <= 2, (full, half)
        assert remaining_compute_seconds(rate, str(MAX_COMPUTE_USD)) == 0
        assert remaining_compute_seconds(rate, "999") == 0
        # the sum over an eviction sequence can never exceed the allocation
        spent, total_seconds = 0.0, 0
        for _ in range(4):
            secs = remaining_compute_seconds(rate, str(spent))
            total_seconds += secs
            spent += secs * float(rate) / 3600.0
        assert spent <= float(MAX_COMPUTE_USD) + 0.01, spent
    r.check("B1. per-pod termination horizons derive from the REMAINING "
            "allocation, so N acquisitions cannot authorize N x the budget",
            b1_deadlines_derive_from_remaining_allocation)

    def b1_policy_constants_must_stay_coherent():
        from o1_b200.provider.runpod import billing, policy
        billing.assert_policy_coherent()
        assert policy.MAX_COMPUTE_USD + policy.RESERVED_NONCOMPUTE_USD == \
            policy.TOTAL_AUTHORIZED_USD
    r.check("B1. budget constants are coherent (reserved non-compute is no "
            "longer dead)", b1_policy_constants_must_stay_coherent)

    # ---------------- B3: creation-slot ledger is race-safe -------------

    def b3_concurrent_consumption_cannot_share_a_slot():
        d = fresh_dir("repair_slots")
        ledger = os.path.join(d, "nonces.txt")
        nonce = "concurrent-nonce-abcdef"
        doc = {
            "schema": AUTH_SCHEMA, "project": "P",
            "package_zip_sha256": "p" * 64, "provider": "runpod",
            "budget_policy_sha256": "b" * 64,
            "expires_utc": "2099-01-01T00:00:00Z",
            "launch_nonce": nonce, "deployment_spec_sha256": "d" * 64,
            "allow_create_pod": True, "allow_real_calibration": True,
            "allow_confirmation": False, "max_pod_creations": 2,
        }
        path = os.path.join(d, "auth.json")
        with open(path, "w") as fh:
            json.dump(doc, fh)
        ident = {k: doc[k] for k in
                 ("project", "package_zip_sha256", "provider",
                  "budget_policy_sha256", "deployment_spec_sha256")}

        def verify():
            return LiveMutationAuthorization.verify(
                path=path, expected_identity=ident, cli_args=[CLI_FLAG],
                nonce_ledger=ledger, environ={ENV_FLAG: ENV_FLAG_VALUE})
        # two independently-verified handles (two session drivers) each
        # consume: the ledger must charge TWO slots, not collapse them
        a, b = verify(), verify()
        a.consume_nonce()
        b.consume_nonce()
        with open(ledger, encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        assert len(lines) == 2, lines
        assert len(set(lines)) == 2, f"ledger lines collapsed: {lines}"
        try:
            verify()
        except AuthorizationError as exc:
            assert "replay" in str(exc) or "exhausted" in str(exc)
            return
        raise AssertionError("a third creation was authorized on 2 slots")
    r.check("B3. concurrent slot consumption charges one slot each (no "
            "set-dedup collapse)", b3_concurrent_consumption_cannot_share_a_slot)

    # ---------------- C1: deployment hash is re-derived ------------------

    def c1_forged_rendering_cannot_deploy():
        reqs = _reqs()
        rendered = render_canonical_deployment(reqs, {"package": "test"})
        verify_deployment_rendering(rendered)          # honest doc verifies
        # a doc whose body was swapped but whose hash field was left intact
        forged = json.loads(json.dumps(rendered))
        forged["create_pod_bodies"]["B300"]["image"] = \
            "ghcr.io/attacker/backdoor@sha256:" + "b" * 64
        try:
            verify_deployment_rendering(forged)
        except PodRequestError as exc:
            assert "does not match its own contents" in str(exc)
        else:
            raise AssertionError("a forged rendering verified")
        # and a doc that recomputes consistently but was never rendered by us
        forged["request_sha256"] = None
        try:
            verify_deployment_rendering(forged)
        except PodRequestError:
            return
        raise AssertionError("a hashless rendering verified")
    r.check("C1. the deployment hash is re-derived from the rendering's "
            "CONTENTS, so a forged body cannot deploy",
            c1_forged_rendering_cannot_deploy)

    def c1_adapter_refuses_forged_rendering_end_to_end():
        d = fresh_dir("repair_forge")
        env_values = {"O1_B200_RESULT_DESTINATION": "hf://ns/repo/s",
                      "O1_B200_ARTIFACT_SOURCE": "hf://ns/stage",
                      "O1_B200_OUT": "/outputs",
                      "O1_IMAGE_DIGEST": GOOD_IMAGE}
        reqs = _reqs(env_values)
        rendered = render_canonical_deployment(reqs, {"package": "test"})
        doc = {
            "schema": AUTH_SCHEMA, "project": "P",
            "package_zip_sha256": "p" * 64, "provider": "runpod",
            "budget_policy_sha256": "b" * 64,
            "expires_utc": "2099-01-01T00:00:00Z",
            "launch_nonce": f"forge-nonce-{os.getpid()}-abcdef",
            "deployment_spec_sha256": rendered["request_sha256"],
            "allow_create_pod": True, "allow_real_calibration": True,
            "allow_confirmation": False, "max_pod_creations": 2,
        }
        path = os.path.join(d, "auth.json")
        with open(path, "w") as fh:
            json.dump(doc, fh)
        auth = LiveMutationAuthorization.verify(
            path=path, expected_identity={
                k: doc[k] for k in ("project", "package_zip_sha256",
                                    "provider", "budget_policy_sha256",
                                    "deployment_spec_sha256")},
            cli_args=[CLI_FLAG], nonce_ledger=os.path.join(d, "n.txt"),
            environ={ENV_FLAG: ENV_FLAG_VALUE})
        evil = json.loads(json.dumps(rendered))
        evil["create_pod_bodies"]["B300"]["image"] = \
            "ghcr.io/attacker/backdoor@sha256:" + "b" * 64
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            with MockRunpodServer() as srv:
                ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                                     authorization=auth, sleep=NOSLEEP)
                ad.quote_instance()
                req = build_pod_request(
                    profile=PROFILE_B300,
                    image_digest_ref="ghcr.io/attacker/backdoor@sha256:"
                                     + "b" * 64,
                    datacenter_id="US-KS-2", env_values=env_values)
                try:
                    ad.create_instance(req, evil)
                except (AuthorizationError, PodRequestError):
                    assert not srv.scenario.rent_calls, \
                        "a pod was created from a forged rendering"
                    return
                raise AssertionError("forged rendering created a pod")
        finally:
            os.environ.pop(ENV_FLAG, None)
    r.check("C1. end to end: a valid authorization plus a forged rendering "
            "creates NO pod", c1_adapter_refuses_forged_rendering_end_to_end)

    # ---------------- C2: artifact provenance is identity ----------------

    def c2_artifact_provenance_is_part_of_identity():
        good = {"O1_B200_RESULT_DESTINATION": "hf://ns/legit/s",
                "O1_B200_ARTIFACT_SOURCE": "hf://ns/legit-stage",
                "O1_B200_OUT": "/outputs", "O1_IMAGE_DIGEST": GOOD_IMAGE}
        evil = dict(good, O1_B200_RESULT_DESTINATION="hf://attacker/exfil/s")
        a = identity_body(build_pod_request(
            profile=PROFILE_B300, image_digest_ref=GOOD_IMAGE,
            datacenter_id="US-KS-2", env_values=good).to_json())
        b = identity_body(build_pod_request(
            profile=PROFILE_B300, image_digest_ref=GOOD_IMAGE,
            datacenter_id="US-KS-2", env_values=evil).to_json())
        assert a != b, ("redirecting the result destination must change the "
                        "deployment identity")
        # launch-variable values and the datacenter must NOT change identity
        v1 = dict(good, O1_ACQUIRED_PROFILE="B300",
                  O1_SESSION_AUTHORIZED_SECONDS="18000",
                  O1_HOURLY_RATE_USD="7.89", HF_TOKEN="hf_secret_value_x")
        v2 = dict(good, O1_ACQUIRED_PROFILE="B200",
                  O1_SESSION_AUTHORIZED_SECONDS="9000",
                  O1_HOURLY_RATE_USD="6.79", HF_TOKEN="hf_other_value_y")
        c = identity_body(build_pod_request(
            profile=PROFILE_B300, image_digest_ref=GOOD_IMAGE,
            datacenter_id="US-KS-2", env_values=v1).to_json())
        e = identity_body(build_pod_request(
            profile=PROFILE_B300, image_digest_ref=GOOD_IMAGE,
            datacenter_id="EU-RO-1", env_values=v2).to_json())
        assert c == e, "launch-time facts leaked into deployment identity"
        assert not any(c["env"][n] for n in
                       ("HF_TOKEN", "O1_ACQUIRED_PROFILE")), \
            "a secret or launch value survived into the identity body"
    r.check("C2. artifact source/result destination ARE deployment "
            "identity; launch-variable values and secrets are not",
            c2_artifact_provenance_is_part_of_identity)

    # ---------------- D1: mirror manifest under concurrent append -------

    def d1_records_mirror_survives_concurrent_append():
        from o1_b200.runner.durability import (
            CheckpointDurability, LocalDurableStore,
        )
        d = fresh_dir("repair_mirror")
        store = LocalDurableStore(os.path.join(d, "durable"))
        records = os.path.join(d, "o1_records.jsonl")
        with open(records, "w", encoding="utf-8") as fh:
            for i in range(10):
                fh.write(json.dumps({"row": i}) + "\n")

        class AppendingStore(LocalDurableStore):
            """Models the sealed orchestrator writing while we mirror."""

            def push_file(self, local_path, remote_rel):
                if local_path.endswith(".mirror_snapshot"):
                    with open(records, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"row": "appended"}) + "\n")
                return super().push_file(local_path, remote_rel)

        mirror = CheckpointDurability(
            AppendingStore(os.path.join(d, "durable")),
            remote_prefix="durable_o1_records")
        manifest = mirror.sync_checkpoint(records, {"progress": "partial"})
        assert manifest["rows"] == 10, manifest
        # the next pod must be able to restore from it
        got = CheckpointDurability(
            store, remote_prefix="durable_o1_records"
        ).restore_latest(os.path.join(d, "restore"))
        assert got is not None, "no durable checkpoint to restore"
        assert got["sha256"] == manifest["sha256"]
        with open(got["archive"], encoding="utf-8") as fh:
            assert sum(1 for ln in fh if ln.strip()) == 10
        # and the live file kept growing independently
        with open(records, encoding="utf-8") as fh:
            assert sum(1 for ln in fh if ln.strip()) == 11
    r.check("D1. the records mirror snapshots before hashing, so a row "
            "landing mid-push cannot poison the manifest",
            d1_records_mirror_survives_concurrent_append)

    def d1_row_count_marker_is_published():
        from o1_b200.runner.durability import (
            CheckpointDurability, LocalDurableStore,
        )
        d = fresh_dir("repair_marker")
        store = LocalDurableStore(os.path.join(d, "durable"))
        mirror = CheckpointDurability(store,
                                      remote_prefix="durable_o1_records")
        assert mirror.latest_row_count() is None, "unknown must not be zero"
        records = os.path.join(d, "o1_records.jsonl")
        with open(records, "w", encoding="utf-8") as fh:
            for i in range(7):
                fh.write(json.dumps({"row": i}) + "\n")
        mirror.sync_checkpoint(records, {"progress": "partial"})
        assert mirror.latest_row_count() == 7
    r.check("D1/D4. durable row count is published and readable off-pod "
            "(the zero-progress guard's real input)",
            d1_row_count_marker_is_published)

    # ---------------- D3: hub reachability is scoped, not global --------

    def d3_transfer_helper_is_the_only_online_surface():
        from o1_b200.runner.hf_transfer import OFFLINE_FLAGS, child_env
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            child = child_env("tok_dummy_123456")
            for flag in OFFLINE_FLAGS:
                assert flag not in child, f"{flag} survived into the helper"
            assert child.get("HF_HUB_DISABLE_TELEMETRY") == "1"
            # the parent stays offline-locked
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            # and the helper refuses loudly if spawned WITHOUT child_env
            proc = subprocess.run(
                [sys.executable, "-m", "o1_b200.runner.hf_transfer",
                 "list", "--repo", "ns/repo"],
                capture_output=True, text=True, timeout=120,
                env={**os.environ,
                     "PYTHONPATH": os.path.abspath(
                         os.path.join(os.path.dirname(__file__), "..", ".."))})
            assert proc.returncode != 0
            assert "HF_HUB_OFFLINE is still set" in (proc.stdout + proc.stderr)
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)
    r.check("D3. hub access is scoped to the transfer helper: the parent "
            "stays offline-locked and the helper refuses if it is not",
            d3_transfer_helper_is_the_only_online_surface)

    def d3_hf_store_routes_through_the_helper_with_digests():
        from o1_b200.runner.durability import DurabilityError, HfDurableStore
        d = fresh_dir("repair_hfstore")
        local = os.path.join(d, "payload.bin")
        with open(local, "wb") as fh:
            fh.write(b"durable-bytes")
        import hashlib
        digest = hashlib.sha256(b"durable-bytes").hexdigest()
        calls = []

        def runner(args):
            calls.append(args[0])
            if args[0] == "list":
                return {"files": ["results/o1_results.tar.gz"]}
            return {"sha256": digest}
        store = HfDurableStore("ns/repo", token="tok", runner=runner)
        assert store.push_file(local, "results/x")["sha256"] == digest
        assert store.list_prefix("results") == ["results/o1_results.tar.gz"]
        assert calls == ["push", "list"]

        # a helper that reports a different digest must fail closed
        def lying(args):
            return {"sha256": "0" * 64, "files": []}
        try:
            HfDurableStore("ns/repo", runner=lying).push_file(local, "r/x")
        except DurabilityError:
            return
        raise AssertionError("an unverified HF transfer was accepted")
    r.check("D3/F4. HF transfers are digest-verified end to end and fail "
            "closed on disagreement",
            d3_hf_store_routes_through_the_helper_with_digests)

    # ---------------- E5: version comparison is numeric -----------------

    def e5_cuda_floor_is_numeric():
        from o1_b200.deploy.hardware_gate import _version_at_least
        assert _version_at_least("13.0", "13.0")
        assert _version_at_least("13.2", "13.0")
        assert not _version_at_least("9.0", "13.0"), \
            "lexicographic compare accepted a single-digit-major runtime"
        assert not _version_at_least("8.6", "13.0")
        assert not _version_at_least("12.8", "13.0")
        assert not _version_at_least(None, "13.0")
        assert not _version_at_least("garbage", "13.0")
    r.check("E5. the CUDA floor is a numeric version comparison, not a "
            "string compare", e5_cuda_floor_is_numeric)

    # ---------------- E1: replacement manifest diff is bounded ----------

    def e1_manifest_diff_helper_is_exact():
        from o1_b200.runner.production_entry import _diff_paths
        base = {"code": {"torch": "a", "numpy": "n", "note": "x"},
                "model": {"flags": {"tf32": False}}}
        same = json.loads(json.dumps(base))
        assert _diff_paths(base, same) == []
        changed = json.loads(json.dumps(base))
        changed["code"]["torch"] = "b"
        assert _diff_paths(base, changed) == ["code.torch"]
        sneaky = json.loads(json.dumps(base))
        sneaky["model"]["flags"]["tf32"] = True
        assert _diff_paths(base, sneaky) == ["model.flags.tf32"], \
            "a nested scientific flag change would go unnoticed"
    r.check("E1. the replacement-manifest diff check detects ANY field "
            "change, including nested ones", e1_manifest_diff_helper_is_exact)

    # ---------------- E3: eligibility needs a per-config verdict --------

    def e3_config_without_its_own_equivalence_is_ineligible():
        from o1_b200.runner.selection import is_eligible
        gates = {g: True for g in (
            "structural_pass", "parser_verifier_pass",
            "action_seed_mapping_exact", "intervention_pass",
            "transport_pass", "resume_pass", "no_missing_or_duplicate_rows",
            "no_oom", "free_hbm_fraction_ok", "no_unvalidated_optimization",
            "throughput_stable", "scientific_config_unchanged")}
        measured = {"config_id": "B200_BATCHED_w1_b8",
                    "backend": "B200_BATCHED", "batch": 8,
                    "completed_rows_per_hour": 5000.0,
                    "steady_state_free_hbm_fraction": 0.5,
                    "throughput_stability_spread": 0.05, **gates}
        assert is_eligible(measured)[0]
        # the deep-batch config with NO verdict of its own: structural and
        # the core-identity gates are False, so it can never be selected
        unmeasured = dict(measured, config_id="B200_BATCHED_w1_b48",
                          batch=48, completed_rows_per_hour=9000.0,
                          structural_pass=False,
                          action_seed_mapping_exact=False,
                          intervention_pass=False, transport_pass=False)
        ok, failures = is_eligible(unmeasured)
        assert not ok and "structural_pass" in failures
        from o1_b200.runner.selection import select_backend
        chosen = select_backend([measured, unmeasured])["selected"]
        assert chosen["config_id"] == "B200_BATCHED_w1_b8", (
            "the faster but unmeasured deep-batch config was selected")
    r.check("E3. a configuration without its own equivalence verdict is "
            "ineligible, even when it is the fastest",
            e3_config_without_its_own_equivalence_is_ineligible)

    run_extra(r)
    return r



def run_extra(r: Runner) -> Runner:
    """G-group: test-honesty repairs."""

    def g2_every_graphql_document_on_the_refusal_path_is_a_query():
        from o1_b200.provider.runpod.zero_touch import (
            _render_all_profiles, run_session,
        )
        d = fresh_dir("repair_g2")
        config = {
            "project": "P", "package_zip_sha256": "p" * 64,
            "budget_policy_sha256": "b" * 64,
            "image_digest_ref": GOOD_IMAGE,
            "identities": {"package": "test"}, "adapter_commit": "test",
            "result_source": os.path.join(d, "r.tar.gz"),
        }
        with open(config["result_source"], "wb") as fh:
            fh.write(b"x")
        rendered = _render_all_profiles(config)
        doc = {
            "schema": AUTH_SCHEMA, "project": "P",
            "package_zip_sha256": "p" * 64, "provider": "runpod",
            "budget_policy_sha256": "b" * 64,
            "expires_utc": "2099-01-01T00:00:00Z",
            "launch_nonce": f"g2-nonce-{os.getpid()}-abcdef",
            "deployment_spec_sha256": rendered["request_sha256"],
            "allow_create_pod": True, "allow_real_calibration": True,
            "allow_confirmation": False, "max_pod_creations": 2,
        }
        auth_path = os.path.join(d, "auth.json")
        with open(auth_path, "w") as fh:
            json.dump(doc, fh)
        sc = Scenario()
        os.environ.pop(ENV_FLAG, None)          # NOT authorized
        with MockRunpodServer(sc) as srv:
            status = run_session(
                authorization_path=auth_path, out_dir=d,
                cli_args=[CLI_FLAG], base_url=srv.base_url,
                api_key=MOCK_KEY, sleep=NOSLEEP, config=config,
                spawn_watchdog_fn=lambda **kw: None)
        assert status["outcome"] == "LIVE_MUTATION_NOT_AUTHORIZED"
        assert not sc.rent_calls and not sc.pods
        assert sc.graphql_documents, "no GraphQL traffic was recorded"
        for document in sc.graphql_documents:
            head = document.strip().split("{")[0].strip()
            assert head.startswith("query"), (
                f"a non-query GraphQL document was sent on the unauthorized "
                f"path: {head!r}")
    r.check("G2. every GraphQL document sent without authorization is a "
            "read-only query operation (not merely 'no rent call')",
            g2_every_graphql_document_on_the_refusal_path_is_a_query)

    def g4_missing_pod_env_does_not_silently_degrade_ownership():
        """The live REST surface may omit env; nonce checking must not
        quietly become name-only matching."""
        d = fresh_dir("repair_g4")
        sc = Scenario()
        sc.pod_json_omit_env = True
        sc.drop_rent_response = True        # force the reconcile path
        env_values = {"O1_B200_RESULT_DESTINATION": "hf://ns/repo/s",
                      "O1_B200_ARTIFACT_SOURCE": "hf://ns/stage",
                      "O1_B200_OUT": "/outputs",
                      "O1_IMAGE_DIGEST": GOOD_IMAGE}
        reqs = _reqs(env_values)
        rendered = render_canonical_deployment(reqs, {"package": "test"})
        doc = {
            "schema": AUTH_SCHEMA, "project": "P",
            "package_zip_sha256": "p" * 64, "provider": "runpod",
            "budget_policy_sha256": "b" * 64,
            "expires_utc": "2099-01-01T00:00:00Z",
            "launch_nonce": f"g4-nonce-{os.getpid()}-abcdef",
            "deployment_spec_sha256": rendered["request_sha256"],
            "allow_create_pod": True, "allow_real_calibration": True,
            "allow_confirmation": False, "max_pod_creations": 2,
        }
        path = os.path.join(d, "auth.json")
        with open(path, "w") as fh:
            json.dump(doc, fh)
        auth = LiveMutationAuthorization.verify(
            path=path, expected_identity={
                k: doc[k] for k in ("project", "package_zip_sha256",
                                    "provider", "budget_policy_sha256",
                                    "deployment_spec_sha256")},
            cli_args=[CLI_FLAG], nonce_ledger=os.path.join(d, "n.txt"),
            environ={ENV_FLAG: ENV_FLAG_VALUE})
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            with MockRunpodServer(sc) as srv:
                ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                                     authorization=auth, sleep=NOSLEEP)
                ad.quote_instance()
                req = build_pod_request(
                    profile=PROFILE_B300, image_digest_ref=GOOD_IMAGE,
                    datacenter_id="US-KS-2", env_values=env_values)
                pod = ad.create_instance(req, rendered)
                assert pod.id in sc.pods
                assert len(sc.rent_calls) == 1, "a second create was sent"
                notes = [n["event"] for n in ad._reconciliation_notes]
                assert "NONCE_UNVERIFIED_ADOPTION" in notes, (
                    "ownership was inferred without a verifiable nonce and "
                    "the decision was NOT recorded")
        finally:
            os.environ.pop(ENV_FLAG, None)
    r.check("G4. when the pod listing omits env, unverifiable ownership is "
            "recorded loudly instead of silently accepted",
            g4_missing_pod_env_does_not_silently_degrade_ownership)

    def g4_foreign_nonce_is_never_adopted():
        d = fresh_dir("repair_g4b")
        sc = Scenario()
        sc.pods["foreign"] = {
            "id": "foreign", "name": "o1-b300-calibration",
            "image": GOOD_IMAGE, "env": {"O1_LAUNCH_NONCE": "someone-else"},
            "gpu": {"id": "NVIDIA B300 SXM6 AC", "count": 1},
            "cloud": "SECURE", "status": "RUNNING", "poll_count": 0,
            "plan": ["RUNNING"], "terminated": False,
            "terminate_polls_left": 0, "createdAt": "2026-01-01T00:00:00Z"}
        with MockRunpodServer(sc) as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                                 sleep=NOSLEEP)
            found = ad._reconcile_by_identity("o1-b300-calibration",
                                              "our-own-nonce")
            assert found is None, (
                "a pod carrying a DIFFERENT launch nonce was adopted as ours")
            # without a nonce (pre-create duplicate check) it must still block
            assert ad._reconcile_by_identity("o1-b300-calibration") is not None
    r.check("G4. a pod carrying a different launch nonce is never adopted, "
            "while the duplicate-name guard still blocks creation",
            g4_foreign_nonce_is_never_adopted)

    def operator_profile_preference_reorders_without_adding_capability():
        """An operator may prefer the B200 first (B300 demand), but the
        frozen profile SET is closed and both bodies stay authorized."""
        from o1_b200.provider.runpod.policy import PROFILE_PREFERENCE
        from o1_b200.provider.runpod.quote import resolve_preference
        from o1_b200.provider.runpod.policy import PolicyViolation
        default = resolve_preference(None)
        assert [p.key for p in default] == [p.key for p in PROFILE_PREFERENCE]
        flipped = resolve_preference(["B200", "B300"])
        assert [p.key for p in flipped] == ["B200", "B300"]
        # a partial preference still keeps every frozen profile reachable
        partial = resolve_preference(["B200"])
        assert [p.key for p in partial] == ["B200", "B300"], (
            "dropping a profile from the preference must not remove it as a "
            "fallback")
        for bad in (["H100"], ["B200", "GB300"], ["nonsense"]):
            try:
                resolve_preference(bad)
            except PolicyViolation:
                continue
            raise AssertionError(f"unknown profile accepted: {bad}")
        # and selection honours it end to end, without calling a
        # first-choice B200 a "fallback"
        with MockRunpodServer(Scenario()) as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                                 sleep=NOSLEEP,
                                 profile_preference=["B200", "B300"])
            q = ad.quote_instance()
            assert q["profile"] == "B200" and q["fallback_reason"] is None
            assert q["operator_preference_applied"] is True
            assert q["gpu_type_id"] == "NVIDIA B200"
    r.check("operator profile preference reorders the FROZEN profiles "
            "without adding capability or mislabelling the choice",
            operator_profile_preference_reorders_without_adding_capability)

    def preferred_profile_still_falls_back_when_refused():
        """Preferring B200 must still fall back to B300 if B200 is gone."""
        sc = Scenario()
        from o1_b200.provider.runpod.mock_server import _b200, _b300
        sc.gpus = [_b300(), _b200(availability="NONE",
                                  dcs=[{"id": "US-KS-2",
                                        "availability": "NONE"}])]
        with MockRunpodServer(sc) as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                                 sleep=NOSLEEP,
                                 profile_preference=["B200", "B300"])
            q = ad.quote_instance()
            assert q["profile"] == "B300"
            assert q["fallback_reason"] and "B200" in q["fallback_reason"]
    r.check("a preferred profile that is refused still falls back, with the "
            "reason recorded", preferred_profile_still_falls_back_when_refused)

    # ---------- second review round: verifier's NEW findings -------------

    def n1_secret_env_guard_is_live_not_dead():
        """A newly added secret that nobody classified must be REFUSED.

        The guard used to run after the values were blanked, so it could
        never fire and a forgotten secret was frozen in cleartext into the
        authorization-covered rendering.
        """
        from o1_b200.provider.runpod import pod_request as pr
        original = pr.ENV_NAMES
        pr.ENV_NAMES = original + ("NEW_REGISTRY_PASSWORD",)
        try:
            reqs = {p.key: pr.build_pod_request(
                profile=p, image_digest_ref=GOOD_IMAGE,
                datacenter_id="US-KS-2",
                env_values={"NEW_REGISTRY_PASSWORD": "p@ssw0rd-secret-value"})
                for p in (PROFILE_B300, PROFILE_B200)}
            try:
                rendered = pr.render_canonical_deployment(reqs, {"p": "t"})
            except PodRequestError as exc:
                assert "NEW_REGISTRY_PASSWORD" in str(exc)
                return
            raise AssertionError(
                f"an unclassified secret was frozen into the rendering: "
                f"{json.dumps(rendered['create_pod_bodies'])[:200]}")
        finally:
            pr.ENV_NAMES = original
    r.check("NEW#1. an unclassified env VALUE is refused by the canonical "
            "rendering (the guard is live, not dead)",
            n1_secret_env_guard_is_live_not_dead)

    def n2_every_acquisition_keeps_its_own_evidence():
        """A multi-pod session must not erase earlier pods' lifecycle
        evidence (a TERMINATION_UNCONFIRMED on pod 1 has to survive)."""
        from o1_b200.provider.runpod.mock_server import _b300
        d = fresh_dir("repair_n2")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        sc.evict_after_polls = 2
        real_evict = sc.evict

        def evict_once(pod_id):
            real_evict(pod_id)
            sc.evict_after_polls = None
        sc.evict = evict_once
        config, auth_path = _n2_setup(d)
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            with MockRunpodServer(sc) as srv:
                from o1_b200.provider.runpod.zero_touch import run_session
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    api_key=MOCK_KEY, sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=lambda **kw: _FakeWd())
        finally:
            os.environ.pop(ENV_FLAG, None)
        assert status["outcome"] == "COMPLETE", status["outcome"]
        assert status["acquisitions_used"] == 2
        history = sorted(os.listdir(os.path.join(d, "lifecycle")))
        assert len(history) == 2, history
        ledger = [json.loads(ln) for ln in
                  open(os.path.join(d, "POD_IDS.jsonl")) if ln.strip()]
        assert len(ledger) == 2, ledger
        assert len({e["pod_id"] for e in ledger}) == 2
        assert [e["attempt"] for e in ledger] == [1, 2]
        # the FIRST pod's evidence must still be readable
        first = json.load(open(os.path.join(d, "lifecycle", history[0])))
        assert first["attempt"] == 1
        kinds = [e["event"] for e in first["events"]]
        assert "POD_CREATED" in kinds, kinds
        assert any(k in kinds for k in
                   ("EVICTED", "EVICTED_BEFORE_RUNNING")), kinds
        assert "SPEND_FROZEN" in kinds, (
            "the evicted pod's spend evidence was lost")
        # and the SECOND report must be a different pod's story, not a
        # copy of the first
        second = json.load(open(os.path.join(d, "lifecycle", history[1])))
        assert second["attempt"] == 2
        assert "POD_RUNNING" in [e["event"] for e in second["events"]]
    r.check("NEW#2. each acquisition keeps its own lifecycle report and "
            "pod-id ledger entry", n2_every_acquisition_keeps_its_own_evidence)

    def n5_result_witness_states_are_distinguished():
        """A broken store must not read as a verified COMPLETE."""
        from o1_b200.provider.runpod.zero_touch import _expected_result_digest
        d = fresh_dir("repair_n5")
        # ABSENT: no destination configured at all
        assert _expected_result_digest({}) == ("ABSENT", None)
        # ABSENT: destination exists but carries no sidecar
        store_root = os.path.join(d, "store")
        os.makedirs(os.path.join(store_root, "results"))
        state, digest = _expected_result_digest(
            {"result_destination": store_root})
        assert (state, digest) == ("ABSENT", None), (state, digest)
        # VERIFIABLE: sidecar present
        with open(os.path.join(store_root, "results",
                               "o1_results.tar.gz.sha256"), "w") as fh:
            json.dump({"archive": "o1_results.tar.gz", "sha256": "d" * 64,
                       "rows": 4608}, fh)
        assert _expected_result_digest(
            {"result_destination": store_root}) == ("VERIFIABLE", "d" * 64)
        # ERROR: destination is a FILE, so no store can be opened
        broken = os.path.join(d, "not_a_store.bin")
        with open(broken, "wb") as fh:
            fh.write(b"x")
        state, detail = _expected_result_digest({"result_destination": broken})
        assert state == "ERROR", (state, detail)
        assert detail, "an error state must carry a reason"
    r.check("NEW#5. result-witness states distinguish ABSENT from ERROR so "
            "a broken store cannot read as verified",
            n5_result_witness_states_are_distinguished)

    def n5_broken_witness_downgrades_the_verdict():
        d = fresh_dir("repair_n5b")
        sc = Scenario()
        sc.log_text = "ZERO_TOUCH_COMPLETE\n"
        config, auth_path = _n2_setup(d)
        broken = os.path.join(d, "not_a_store.bin")
        with open(broken, "wb") as fh:
            fh.write(b"x")
        config["result_destination"] = broken
        # re-render: result_destination is deployment identity
        config, auth_path = _n2_setup(d, config=config)
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            with MockRunpodServer(sc) as srv:
                from o1_b200.provider.runpod.zero_touch import run_session
                status = run_session(
                    authorization_path=auth_path, out_dir=d,
                    cli_args=[CLI_FLAG], base_url=srv.base_url,
                    api_key=MOCK_KEY, sleep=NOSLEEP, config=config,
                    spawn_watchdog_fn=lambda **kw: _FakeWd())
        finally:
            os.environ.pop(ENV_FLAG, None)
        assert status["outcome"] == "COMPLETE_RESULT_DIGEST_UNVERIFIED", (
            f"a store that could not be consulted still reported "
            f"{status['outcome']}")
        assert status["result_digest_verified"] is False
        assert all(p["terminated"] for p in sc.pods.values())
    r.check("NEW#5. a result store that cannot be consulted downgrades the "
            "verdict instead of claiming a verified COMPLETE",
            n5_broken_witness_downgrades_the_verdict)

    def n6_arming_failure_never_orphans_a_billable_pod():
        """If anything fails between create and watchdog-arm, the pod must
        be terminated — not left billing with its id unreturned."""
        from o1_b200.provider.runpod.lifecycle import PodLifecycleController
        from o1_b200.provider.runpod.quote import QuoteError
        d = fresh_dir("repair_n6")
        sc = Scenario()
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized(srv, d)

            def boom():
                raise QuoteError("quote expired between create and arm")
            ad.validate_quote = boom
            ctl = PodLifecycleController(ad, d, sleep=NOSLEEP,
                                         spawn_watchdog_fn=lambda **kw: _FakeWd())
            try:
                ctl.provision(req, rendered)
            except QuoteError:
                assert sc.pods, "no pod was created; test is vacuous"
                assert all(p["terminated"] for p in sc.pods.values()), (
                    "a billable pod was orphaned when arming failed")
                return
            raise AssertionError("arming failure did not propagate")
    r.check("NEW#6. a failure between pod creation and watchdog arming "
            "terminates the pod instead of orphaning it",
            n6_arming_failure_never_orphans_a_billable_pod)

    def n7_monitor_is_bounded():
        """A pod that never emits its witness must not run to the budget."""
        from o1_b200.provider.runpod.lifecycle import (
            LifecycleError, PodLifecycleController,
        )
        d = fresh_dir("repair_n7")
        sc = Scenario()
        sc.lifecycle_plan = ["RUNNING"]
        sc.log_text = "no completion marker ever\n"
        with MockRunpodServer(sc) as srv:
            ad, req, rendered = _authorized(srv, d)
            ctl = PodLifecycleController(
                ad, d, sleep=NOSLEEP, spawn_watchdog_fn=lambda **kw: _FakeWd(),
                monitor_timeout_seconds=0.05)
            pod_id = ctl.provision(req, rendered)
            ctl.wait_until_running(pod_id)
            try:
                ctl.monitor(pod_id, until=lambda pod: False)
            except LifecycleError as exc:
                assert "completion witness" in str(exc)
                assert sc.pods[pod_id]["terminated"]
                return
            raise AssertionError("monitor ran unbounded")
    r.check("NEW#7. monitor is bounded: a pod that never completes is "
            "terminated rather than paid for to the budget", n7_monitor_is_bounded)

    def n11_preference_rejects_duplicates():
        from o1_b200.provider.runpod.policy import PolicyViolation
        from o1_b200.provider.runpod.quote import resolve_preference
        try:
            resolve_preference(["B200", "B200"])
        except PolicyViolation as exc:
            assert "duplicate" in str(exc).lower()
            return
        raise AssertionError("a duplicated profile preference was accepted")
    r.check("NEW#11. a duplicated profile preference is refused",
            n11_preference_rejects_duplicates)

    return r


class _FakeWd:
    def terminate_now(self):
        pass

    def alive(self):
        return True


def _n2_setup(d, config=None):
    """Session config + authorization bound to its all-profile rendering."""
    import time as _t

    from o1_b200.provider.runpod.zero_touch import _render_all_profiles
    if config is None:
        config = {
            "project": "P", "package_zip_sha256": "p" * 64,
            "budget_policy_sha256": "b" * 64,
            "image_digest_ref": GOOD_IMAGE,
            "identities": {"package": "test"}, "adapter_commit": "test",
            "result_source": os.path.join(d, "fake_results.tar.gz"),
        }
        with open(config["result_source"], "wb") as fh:
            fh.write(b"results")
    rendered = _render_all_profiles(config)
    doc = {
        "schema": AUTH_SCHEMA, "project": "P",
        "package_zip_sha256": "p" * 64, "provider": "runpod",
        "budget_policy_sha256": "b" * 64,
        "expires_utc": _t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   _t.gmtime(_t.time() + 3600)),
        "launch_nonce": f"n2-nonce-{_t.time_ns()}",
        "deployment_spec_sha256": rendered["request_sha256"],
        "allow_create_pod": True, "allow_real_calibration": True,
        "allow_confirmation": False, "max_pod_creations": 4,
    }
    auth_path = os.path.join(d, "auth.json")
    with open(auth_path, "w") as fh:
        json.dump(doc, fh)
    return config, auth_path


def _authorized(srv, d):
    """An authorized adapter plus its request/rendering (mock server)."""
    import time as _t
    reqs = _reqs()
    rendered = render_canonical_deployment(reqs, {"package": "test"})
    doc = {
        "schema": AUTH_SCHEMA, "project": "P",
        "package_zip_sha256": "p" * 64, "provider": "runpod",
        "budget_policy_sha256": "b" * 64,
        "expires_utc": _t.strftime("%Y-%m-%dT%H:%M:%SZ",
                                   _t.gmtime(_t.time() + 3600)),
        "launch_nonce": f"auth-nonce-{_t.time_ns()}",
        "deployment_spec_sha256": rendered["request_sha256"],
        "allow_create_pod": True, "allow_real_calibration": True,
        "allow_confirmation": False, "max_pod_creations": 4,
    }
    path = os.path.join(d, "auth.json")
    with open(path, "w") as fh:
        json.dump(doc, fh)
    os.environ[ENV_FLAG] = ENV_FLAG_VALUE
    auth = LiveMutationAuthorization.verify(
        path=path, expected_identity={
            k: doc[k] for k in ("project", "package_zip_sha256", "provider",
                                "budget_policy_sha256",
                                "deployment_spec_sha256")},
        cli_args=[CLI_FLAG], nonce_ledger=os.path.join(d, "nonces.txt"))
    ad = RunpodV2Adapter(base_url=srv.base_url, api_key=MOCK_KEY,
                         authorization=auth, sleep=NOSLEEP)
    ad.quote_instance()
    return ad, reqs["B300"], rendered



if __name__ == "__main__":
    raise SystemExit(run().report())
