"""The LAUNCH PATH: what actually happens when the container starts.

Round-3 review found the stack could not start at all — the manifest the
pod verified carried build-host absolute paths, nothing ever delivered the
checkpoint, and a deterministic startup failure was indistinguishable from
an eviction.  The suites passed anyway, because they exercise the runner
in-tree against the mock provider, which is the one configuration where
host paths resolve.  These checks are about the pod's own reality.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEPLOY = os.path.join(_ROOT, "o1_b200", "deploy")
POD_MANIFEST = os.path.join(DEPLOY, "POD_TRANSFER_MANIFEST.json")
HOST_MANIFEST = os.path.join(DEPLOY, "TRANSFER_MANIFEST.json")
CONTAINER_ROOTS = ("/opt/o1_b200", "/artifacts")


def run() -> Runner:
    r = Runner("launch_path")

    def pod_manifest_paths_are_container_paths():
        with open(POD_MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)
        assert manifest["artifacts"], "pod manifest is empty"
        offenders = {}
        for name, spec in manifest["artifacts"].items():
            path = spec["path"]
            if not any(path == root or path.startswith(root + "/")
                       for root in CONTAINER_ROOTS):
                offenders[name] = path
        assert not offenders, (
            f"the manifest the POD verifies names paths that cannot exist "
            f"inside the container: {offenders}")
    r.check("the pod manifest names CONTAINER paths only (a build-host path "
            "here fails every artifact at startup)",
            pod_manifest_paths_are_container_paths)

    def host_manifest_is_not_the_pod_manifest():
        with open(HOST_MANIFEST, encoding="utf-8") as fh:
            host = json.load(fh)
        host_paths = {s["path"] for s in host["artifacts"].values()}
        assert any(p.startswith("/home/") for p in host_paths), (
            "the host manifest should carry build-host paths; if it does "
            "not, the two manifests have been conflated")
        assert "path_domain" in host, (
            "the host manifest must say which domain its paths are in")
    r.check("the host manifest is build provenance and says so, so the two "
            "cannot be confused again", host_manifest_is_not_the_pod_manifest)

    def every_pod_artifact_is_reachable_in_the_image():
        """Either baked under /opt/o1_b200, or fetched to /artifacts."""
        from o1_b200.runner.fetch_artifacts import required_from_manifest
        with open(POD_MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)
        fetched = set(required_from_manifest(POD_MANIFEST))
        for name, spec in manifest["artifacts"].items():
            path = spec["path"]
            if path.startswith("/artifacts"):
                rel = os.path.relpath(path, "/artifacts")
                assert rel in fetched, (
                    f"{name} lives under /artifacts but the fetch step does "
                    f"not know to obtain it")
            else:
                # baked: the same file must exist in the source tree the
                # image COPYs from, at the mapped location
                host_rel = os.path.relpath(path, "/opt/o1_b200")
                assert os.path.exists(os.path.join(_ROOT, host_rel)), (
                    f"{name} is expected at {path} in the image but "
                    f"{host_rel} is absent from the build context")
    r.check("every pod artifact is either baked into the image or on the "
            "fetch list — nothing is expected to appear by magic",
            every_pod_artifact_is_reachable_in_the_image)

    def the_checkpoint_is_on_the_fetch_list():
        from o1_b200.runner.fetch_artifacts import required_from_manifest
        wanted = required_from_manifest(POD_MANIFEST)
        assert "ouro_rltt_local" in wanted, (
            f"the checkpoint is never baked into the image, so it MUST be "
            f"fetched; fetch list is {wanted}")
    r.check("the checkpoint — never in the image, never in Git — is on the "
            "pod's fetch list", the_checkpoint_is_on_the_fetch_list)

    def fetch_refuses_without_a_source_when_something_is_missing():
        from o1_b200.runner.fetch_artifacts import (
            ArtifactFetchError, fetch, parse_hf_source,
        )
        d = fresh_dir("launch_fetch")
        empty = os.path.join(d, "artifacts")
        os.makedirs(empty)
        try:
            fetch("", empty, manifest_path=POD_MANIFEST)
        except ArtifactFetchError as exc:
            assert "staging URI" in str(exc) or "hf://" in str(exc)
        else:
            raise AssertionError("fetch proceeded with no staging source")
        for bad in ("s3://x/y", "hf://onlyone", "hf://", "/local/path"):
            try:
                parse_hf_source(bad)
            except ArtifactFetchError:
                continue
            raise AssertionError(f"accepted malformed source {bad!r}")
    r.check("artifact fetch refuses a missing/!hf:// staging source rather "
            "than starting a session that cannot verify",
            fetch_refuses_without_a_source_when_something_is_missing)

    def fetch_is_a_no_op_when_artifacts_are_already_mounted():
        from o1_b200.runner.fetch_artifacts import fetch
        d = fresh_dir("launch_mounted")
        root = os.path.join(d, "artifacts")
        os.makedirs(os.path.join(root, "ouro_rltt_local"))

        def refuse(args):
            raise AssertionError(f"fetched despite a mounted artifact: {args}")
        out = fetch("", root, runner=refuse, manifest_path=POD_MANIFEST)
        assert out["fetched"] == []
        assert "ouro_rltt_local" in out["already_present"]
    r.check("a pod whose artifacts are already mounted needs no staging "
            "source and transfers nothing",
            fetch_is_a_no_op_when_artifacts_are_already_mounted)

    def fetch_asks_the_helper_for_exactly_what_is_missing():
        from o1_b200.runner.fetch_artifacts import fetch
        d = fresh_dir("launch_fetch_calls")
        root = os.path.join(d, "artifacts")
        os.makedirs(root)
        calls = []

        def fake(args):
            calls.append(args)
            # the helper materialises the tree; emulate that
            os.makedirs(os.path.join(root, "ouro_rltt_local"), exist_ok=True)
            return {"count": 3}
        out = fetch("hf://ns/staging", root, runner=fake,
                    manifest_path=POD_MANIFEST)
        assert out["fetched"] == ["ouro_rltt_local"], out
        assert len(calls) == 1, calls
        assert calls[0][0] == "snapshot"
        assert "--repo" in calls[0] and "ns/staging" in calls[0]
    r.check("the fetch asks the isolated helper for exactly the missing "
            "artifacts", fetch_asks_the_helper_for_exactly_what_is_missing)

    def entrypoint_order_is_fetch_verify_validate_then_handover():
        path = os.path.join(DEPLOY, "start_b300.sh")
        text = open(path, encoding="utf-8").read()
        subprocess.run(["bash", "-n", path], check=True)
        order = [text.index("fetch_artifacts"),
                 text.index("verify_artifacts.py"),
                 text.index("validate_environment.py"),
                 text.index("production_entry")]
        assert order == sorted(order), (
            f"the entrypoint must fetch BEFORE verifying (nothing to verify "
            f"otherwise) and verify before handing over: {order}")
        assert "POD_TRANSFER_MANIFEST.json" in text, (
            "the entrypoint must verify the container-path manifest")
        assert "HF_HUB_OFFLINE=1" in text, (
            "model loading must stay offline-locked")
    r.check("the entrypoint fetches, then verifies the pod manifest, then "
            "validates, then hands over",
            entrypoint_order_is_fetch_verify_validate_then_handover)

    # ---- the combined O1 -> Foundation Learner session can actually start ----
    #
    # Round-3 finding: start_b300.sh grew an FL dispatch branch while nothing
    # ever put the FL package into the image, so a combined rental would
    # acquire an accelerator and refuse at the handover.  These checks bind
    # the dispatch target, the image layout and the build context together so
    # the three cannot drift apart again.

    def _fl_facts() -> dict:
        import re
        start = open(os.path.join(DEPLOY, "start_b300.sh"),
                     encoding="utf-8").read()
        dockerfile = open(os.path.join(DEPLOY, "Dockerfile.b300"),
                          encoding="utf-8").read()
        build = open(os.path.join(_ROOT, "o1_b200", "scripts",
                                  "build_b300_image.sh"), encoding="utf-8").read()
        entry = re.search(r"O1_FL_ENTRY:-([^}]+)\}", start)
        assert entry, "start_b300.sh no longer declares an FL entry default"
        copy = re.search(r"^COPY\s+(build_ctx/foundation_learner)\s+(\S+)\s*$",
                         dockerfile, re.M)
        assert copy, ("Dockerfile.b300 does not COPY the FL source from the "
                      "build context; a combined session cannot start")
        pythonpath = re.search(r"^ENV PYTHONPATH=(\S+)", dockerfile, re.M)
        assert pythonpath, "Dockerfile.b300 declares no PYTHONPATH"
        return {"entry": entry.group(1), "copy_src": copy.group(1),
                "copy_dest": copy.group(2),
                "pythonpath": pythonpath.group(1).split(":"),
                "start": start, "build": build}

    def fl_dispatch_target_exists_in_the_image():
        f = _fl_facts()
        expected = os.path.join(f["copy_dest"], "deploy", "fl_b200_entry.sh")
        assert f["entry"] == expected, (
            f"start_b300.sh dispatches to {f['entry']} but the image places "
            f"the FL package at {f['copy_dest']}, so the entry is at "
            f"{expected}; a combined session would refuse at handover")
    r.check("the FL dispatch target in the entrypoint is where the image "
            "actually puts the FL entry script",
            fl_dispatch_target_exists_in_the_image)

    def fl_source_is_staged_into_the_build_context():
        f = _fl_facts()
        assert f["copy_src"] in f["build"], (
            f"the Dockerfile COPYs {f['copy_src']} but the build script never "
            f"stages it; the build would fail or ship an empty FL tree")
        assert "O1_B300_WITHOUT_FL" in f["build"], (
            "omitting FL must be a deliberate, named choice — not the "
            "silent default that produced this defect")
        # the staged tree must be the real package, not an empty directory
        src = os.path.join(os.path.dirname(_ROOT),
                           "foundation-learner-b200-v0", "foundation_learner")
        if os.path.isdir(src):
            assert os.access(os.path.join(src, "deploy", "fl_b200_entry.sh"),
                             os.X_OK), "the FL entry script is not executable"
    r.check("the FL source the image COPYs is actually staged by the build "
            "script", fl_source_is_staged_into_the_build_context)

    def fl_import_root_is_on_the_image_pythonpath():
        f = _fl_facts()
        import_root = os.path.dirname(f["copy_dest"])
        assert import_root in f["pythonpath"], (
            f"the FL package is at {f['copy_dest']}, so {import_root} must be "
            f"on PYTHONPATH for `python -m foundation_learner...` to resolve; "
            f"PYTHONPATH is {f['pythonpath']}")
    r.check("the FL import root is on the image PYTHONPATH",
            fl_import_root_is_on_the_image_pythonpath)

    def a_combined_session_refuses_rather_than_silently_running_o1_only():
        f = _fl_facts()
        text = f["start"]
        assert "O1_FL_SESSION_CONFIG" in text
        # the refusal must come BEFORE the O1-only handover, or a missing FL
        # entry degrades into a half session that still spends the budget
        refusal = text.index("REFUSED: O1_FL_SESSION_CONFIG")
        o1_only = text.index("O1-only session")
        assert refusal < o1_only, (
            "a combined session with an absent FL entry must refuse, not "
            "fall through to the O1-only path")
        assert "exit 78" in text
    r.check("a combined session whose FL entry is absent refuses instead of "
            "silently running O1 only",
            a_combined_session_refuses_rather_than_silently_running_o1_only)

    # ---- one HF_TOKEN, two directions ----
    #
    # A read-only token passes ingestion and every gate, then loses every
    # durability push.  Under interruptible capacity the run looks healthy
    # until the eviction that destroys it, so the scope is proven up front.

    def scope_preflight_refuses_a_read_only_token():
        from o1_b200.runner.check_hf_scope import ScopeError, check
        calls = []

        def fake(repo, mode):
            calls.append((repo, mode))
            return {"repo": repo, "mode": mode, "read": True,
                    "write": False, "identity": "tester"}
        try:
            check("hf://ns/staging", "hf://ns/results", runner=fake)
        except ScopeError as exc:
            assert "cannot write" in str(exc)
        else:
            raise AssertionError(
                "a token that cannot write to the durable destination was "
                "accepted; eviction would destroy the run")
        assert ("ns/results", "write") in calls, calls
    r.check("the scope preflight refuses a token that can read the staging "
            "repo but not write the durable destination",
            scope_preflight_refuses_a_read_only_token)

    def scope_preflight_probes_write_even_when_both_uris_are_one_repo():
        from o1_b200.runner.check_hf_scope import check
        calls = []

        def fake(repo, mode):
            calls.append((repo, mode))
            return {"repo": repo, "mode": mode, "read": True,
                    "write": mode == "write", "identity": "tester"}
        check("hf://ns/same", "hf://ns/same", runner=fake)
        assert ("ns/same", "read") in calls and ("ns/same", "write") in calls, (
            f"read access to a repo says nothing about write access to it; "
            f"both must be probed: {calls}")
    r.check("the same repo in both directions is still write-probed",
            scope_preflight_probes_write_even_when_both_uris_are_one_repo)

    def scope_preflight_refuses_when_nothing_is_verifiable():
        from o1_b200.runner.check_hf_scope import ScopeError, check, parse_repo
        for read, write, why in (
                ("", "", "no destination at all"),
                ("hf://ns/staging", "", "a verified read but no write"),
                ("hf://ns/staging", "UNRESOLVED_OPERATOR_BOUND",
                 "an unresolved write destination")):
            try:
                check(read, write,
                      runner=lambda r, m: {"repo": r, "mode": m, "read": True,
                                           "write": True, "identity": "t"})
            except ScopeError:
                continue
            raise AssertionError(
                f"a session with {why} was allowed to start; every committed "
                f"row would be lost on eviction")
        assert parse_repo("hf://ns/repo/with/prefix") == "ns/repo"
        assert parse_repo("/mounted/path") is None
        try:
            parse_repo("hf://onlyone")
        except ScopeError:
            pass
        else:
            raise AssertionError("accepted a malformed hf:// URI")
    r.check("the scope preflight refuses a session whose durability it "
            "cannot verify at all",
            scope_preflight_refuses_when_nothing_is_verifiable)

    def scope_preflight_checks_a_mounted_destination_by_writing():
        from o1_b200.runner.check_hf_scope import ScopeError, check
        d = fresh_dir("scope_local")
        good = os.path.join(d, "durable")
        out = check("", good, runner=lambda *a: {})
        assert out["checks"][0]["write"] is True
        assert not os.path.exists(
            os.path.join(good, ".preflight_write_probe")), \
            "the probe artefact was left behind"
        blocked = os.path.join(d, "blocked")
        os.makedirs(blocked)
        os.chmod(blocked, 0o500)
        try:
            check("", os.path.join(blocked, "durable"), runner=lambda *a: {})
        except ScopeError as exc:
            assert "not writable" in str(exc)
        else:
            raise AssertionError("an unwritable durable destination passed")
        finally:
            os.chmod(blocked, 0o700)
    r.check("a mounted durable destination is proven by actually writing to "
            "it", scope_preflight_checks_a_mounted_destination_by_writing)

    def scope_preflight_does_not_invent_a_token_requirement():
        """An all-mounted pod needs no hub credential at all."""
        import o1_b200.runner.check_hf_scope as mod
        d = fresh_dir("scope_no_token")
        dest = os.path.join(d, "durable")
        saved_argv, saved_token = sys.argv, os.environ.pop("HF_TOKEN", None)
        sys.argv = ["check_hf_scope", "--read-source", "",
                    "--write-destination", dest]
        try:
            assert mod.main() == 0, (
                "a pod with mounted artifacts and a mounted durable path was "
                "refused for lacking an unnecessary HF_TOKEN")
            sys.argv = ["check_hf_scope", "--read-source", "hf://ns/staging",
                        "--write-destination", dest]
            assert mod.main() == 2, (
                "an hf:// source with no token must still refuse")
        finally:
            sys.argv = saved_argv
            if saved_token is not None:
                os.environ["HF_TOKEN"] = saved_token
    r.check("the scope preflight requires a token only when a hub repo is "
            "actually named", scope_preflight_does_not_invent_a_token_requirement)

    def the_entrypoint_checks_scope_before_the_multi_gigabyte_fetch():
        text = open(os.path.join(DEPLOY, "start_b300.sh"),
                    encoding="utf-8").read()
        assert "check_hf_scope" in text, (
            "the entrypoint never verifies the credential scope")
        assert text.index("check_hf_scope") < text.index("fetch_artifacts"), (
            "the scope check must precede the checkpoint fetch, or a "
            "doomed session pays for a 5 GB download first")
    r.check("the entrypoint proves the credential scope before fetching "
            "anything", the_entrypoint_checks_scope_before_the_multi_gigabyte_fetch)

    def a_deterministic_pod_abort_is_not_an_eviction():
        """ZERO_TOUCH_ABORTED_AT_* must stop the session, not reacquire."""
        from o1_b200.provider.runpod.zero_touch import DeterministicPodFailure
        import o1_b200.provider.runpod.zero_touch as zt
        # the marker the pod actually prints on a deterministic failure
        assert "ZERO_TOUCH_ABORTED_AT_" in open(
            os.path.join(_ROOT, "o1_b200", "runner", "production_entry.py"),
            encoding="utf-8").read(), "the pod no longer emits the marker"
        source = open(zt.__file__, encoding="utf-8").read()
        assert "ZERO_TOUCH_ABORTED_AT_" in source, (
            "the session driver does not look for the pod's deterministic "
            "abort marker, so it would reacquire against a reproducible "
            "failure until the budget or slots ran out")
        assert issubclass(DeterministicPodFailure, RuntimeError)
        assert "ABORTED_DETERMINISTIC_POD_FAILURE" in source
    r.check("a deterministic pod abort is recognised and stops the session "
            "instead of driving reacquisition",
            a_deterministic_pod_abort_is_not_an_eviction)

    def session_prefix_is_honoured_not_discarded():
        from o1_b200.runner.durability import (
            HfDurableStore, _PrefixedStore, store_for_destination,
        )
        plain = store_for_destination("hf://ns/repo")
        assert isinstance(plain, HfDurableStore)
        scoped = store_for_destination("hf://ns/repo/SESSION_A")
        assert isinstance(scoped, _PrefixedStore)
        assert scoped.prefix == "SESSION_A"
        assert scoped.inner.repo_id == "ns/repo"
        seen = {}

        class Rec(HfDurableStore):
            def push_file(self, local_path, remote_rel):
                seen["push"] = remote_rel
                return {"remote": remote_rel, "sha256": "x"}

            def list_prefix(self, remote_prefix):
                seen["list"] = remote_prefix
                return [f"{remote_prefix}/LATEST.json"]
        s = _PrefixedStore(Rec("ns/repo", runner=lambda a: {}), "SESSION_A")
        s.push_file(__file__, "durable_o1_records/x.jsonl")
        assert seen["push"] == "SESSION_A/durable_o1_records/x.jsonl"
        out = s.list_prefix("durable_o1_records")
        assert seen["list"] == "SESSION_A/durable_o1_records"
        assert out == ["durable_o1_records/LATEST.json"], out
    r.check("a session prefix in the durable destination scopes every key, "
            "so a fresh run cannot silently resume an old one",
            session_prefix_is_honoured_not_discarded)

    def termination_survives_an_expired_authorization():
        from o1_b200.provider.runpod.authorization import (
            AuthorizationError, ENV_FLAG, ENV_FLAG_VALUE,
            LiveMutationAuthorization,
        )
        doc = {"expires_utc": "2020-01-01T00:00:00Z",
               "launch_nonce": "expired-nonce-abcdef",
               "max_pod_creations": 2}
        auth = LiveMutationAuthorization(doc, "p", "/tmp/ledger")
        os.environ[ENV_FLAG] = ENV_FLAG_VALUE
        try:
            try:
                auth.recheck()
            except AuthorizationError as exc:
                assert "expired" in str(exc)
            else:
                raise AssertionError("an expired authorization still allows "
                                     "acquisition")
            # releasing must still be permitted, or an over-running session
            # loses its primary way to shut a billing pod down
            auth.recheck(releasing=True)
        finally:
            os.environ.pop(ENV_FLAG, None)
    r.check("an expired authorization still permits TERMINATION while "
            "refusing acquisition", termination_survives_an_expired_authorization)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
