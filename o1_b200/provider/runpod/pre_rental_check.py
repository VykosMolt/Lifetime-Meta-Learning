#!/usr/bin/env python3
"""RunPod pre-rental master check -> RUNPOD_PRE_RENTAL_READINESS.{json,md}.

Verdicts: PASS | PASS_PENDING_READONLY_CREDENTIAL_CHECK | BLOCKED.
Never PASS while any code, image, artifact, termination, budget or
zero-touch test fails.  A missing live credential caps the verdict at
PASS_PENDING_READONLY_CREDENTIAL_CHECK.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PY = sys.executable


def _run(cmd, cwd=None, timeout=3600, live_credentials=False):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _ROOT}
    if not live_credentials:
        # local phases are hermetic: neither the operator credential nor the
        # live-mutation authorization flag enters a local/mock test process
        # (the independent watchdog gates only on that flag)
        env.pop("RUNPOD_API_KEY", None)
        env.pop("RUNPOD_API_KEY_FILE", None)
        env.pop("RUNPOD_ALLOW_BILLABLE_MUTATIONS", None)
        env.pop("HF_TOKEN", None)
    proc = subprocess.run(cmd, cwd=cwd or _ROOT, capture_output=True,
                          text=True, timeout=timeout, env=env)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(_ROOT, "o1_b200",
                                                     "reports"))
    p.add_argument("--skip-full-suite", action="store_true",
                   help="reuse the latest TEST_REPORT.json instead of "
                        "rerunning the multi-minute full suite")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    checks = {}

    def record(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)[-1500:]}
        print(("PASS " if ok else "FAIL ") + name)

    # 1. base-package integrity
    rc, out = _run([PY, "run_all_tests.py"],
                   cwd=os.path.join(_ROOT, "o1_packages",
                                    "O1_oracle_reachability_v2.1.0_source",
                                    "o1_v210"))
    record("base_package_suite", rc == 0, out.splitlines()[-1] if out else "")

    # 2. full B200-runner + adapter suites (or reuse the fresh report)
    if a.skip_full_suite:
        rep_path = os.path.join(_ROOT, "o1_b200", "reports", "TEST_REPORT.json")
        with open(rep_path, encoding="utf-8") as fh:
            rep = json.load(fh)
        ok = rep["total_failed"] == 0 and any(
            m["module"].startswith("test_runpod") for m in rep["modules"])
        record("b200_runner_and_adapter_suites", ok,
               f"{rep['total_passed']} checks, {rep['total_failed']} failed "
               f"(reused {rep['utc']})")
    else:
        rc, out = _run([PY, "run_all_b200_tests.py"],
                       cwd=os.path.join(_ROOT, "o1_b200", "tests"))
        record("b200_runner_and_adapter_suites", rc == 0,
               out.splitlines()[-1] if out else "")

    # 3. pinned OpenAPI schema
    try:
        from o1_b200.provider.runpod.schema_check import verify_pinned_schema
        record("openapi_schema_pin", True, verify_pinned_schema()["sha256"])
    except Exception as exc:  # noqa: BLE001
        record("openapi_schema_pin", False, exc)

    # 4. dry-run all-profile deployment rendering (B300 primary + B200
    #    explicit fallback under one authorization hash)
    try:
        from o1_b200.provider.runpod.pod_request import (
            build_pod_request, render_canonical_deployment)
        from o1_b200.provider.runpod.policy import PROFILE_PREFERENCE
        reqs = {p.key: build_pod_request(
            profile=p,
            image_digest_ref="local/o1-b300-runner@sha256:" + "0" * 64,
            datacenter_id="DRYRUN-DC") for p in PROFILE_PREFERENCE}
        rendered = render_canonical_deployment(reqs, {"dry_run": True})
        record("dry_run_deployment_render", True,
               f"profiles={rendered['profile_preference']} "
               f"{rendered['request_sha256'][:16]}")
    except Exception as exc:  # noqa: BLE001
        record("dry_run_deployment_render", False, exc)

    # 5. container image record (B300/cu130 stack)
    img_record_path = os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                   "CONTAINER_IMAGE_RECORD.json")
    try:
        with open(img_record_path, encoding="utf-8") as fh:
            img = json.load(fh)
        env = img["environment"]
        no_ptx = not any(a.startswith("compute_") for a in env["arch_list"])
        ok = (env["transformers"] == "4.54.1"
              and env["torch"] == "2.12.1+cu130"
              and env["torch_cuda_runtime"] == "13.0"
              and "sm_100" in env["arch_list"]
              and no_ptx
              and img["platform"] == "linux/amd64"
              and img["schema"] == "o1b300.container_image_record.v2")
        record("container_image_built_and_asserted", ok,
               f"torch={env['torch']} arch={env['arch_list']} no_ptx={no_ptx}")
    except Exception as exc:  # noqa: BLE001
        record("container_image_built_and_asserted", False, exc)

    # 6. artifact transfer manifest
    try:
        man_path = os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                "RUNPOD_ARTIFACT_TRANSFER_MANIFEST.json")
        with open(man_path, encoding="utf-8") as fh:
            man = json.load(fh)
        record("artifact_transfer_manifest", len(man["artifacts"]) >= 12,
               f"{len(man['artifacts'])} artifacts")
    except Exception as exc:  # noqa: BLE001
        record("artifact_transfer_manifest", False, exc)

    # 7. secret scan over the tree
    rc, out = _run(["grep", "-rIlE",
                    "(rpa_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY|AKIA[A-Z0-9]{16})",
                    "--exclude-dir=local_runs", "--exclude-dir=.git",
                    os.path.join(_ROOT, "o1_b200")])
    record("secret_scan_clean", rc != 0, out.strip() or "no secrets found")

    # 8. templates unusable as authorization
    try:
        from o1_b200.provider.runpod.authorization import (
            AuthorizationError, LiveMutationAuthorization, CLI_FLAG,
            ENV_FLAG, ENV_FLAG_VALUE)
        try:
            LiveMutationAuthorization.verify(
                path=os.path.join(_ROOT, "o1_b200", "provider", "runpod",
                                  "B300_RENTAL_AUTHORIZATION.template.json"),
                expected_identity={k: "x" for k in
                                   ("project", "package_zip_sha256",
                                    "provider", "budget_policy_sha256",
                                    "deployment_spec_sha256")},
                cli_args=[CLI_FLAG],
                nonce_ledger=os.path.join(a.out_dir, "nonce_probe.txt"),
                environ={ENV_FLAG: ENV_FLAG_VALUE})
            record("authorization_template_unusable", False,
                   "template validated!")
        except AuthorizationError:
            record("authorization_template_unusable", True)
    except Exception as exc:  # noqa: BLE001
        record("authorization_template_unusable", False, exc)

    # 9. read-only live check when a credential exists
    from o1_b200.provider.runpod.redaction import load_api_key
    live_verdict = "SKIPPED_NO_CREDENTIAL"
    if load_api_key():
        rc, out = _run([PY, "-m", "o1_b200.provider.runpod.preflight",
                        "--out", os.path.join(a.out_dir,
                                              "RUNPOD_READONLY_PREFLIGHT.json")],
                       live_credentials=True)
        live_verdict = "PASS" if rc == 0 else "FAIL"
        record("readonly_live_preflight", rc == 0, out.strip()[-200:])
    else:
        print("SKIP readonly_live_preflight (no credential)")

    all_local_ok = all(c["ok"] for c in checks.values())
    if not all_local_ok:
        verdict = "BLOCKED"
    elif live_verdict == "PASS":
        verdict = "PASS"
    elif live_verdict == "SKIPPED_NO_CREDENTIAL":
        verdict = "PASS_PENDING_READONLY_CREDENTIAL_CHECK"
    else:
        verdict = "BLOCKED"

    report = {
        "schema": "o1b200.runpod_pre_rental_readiness.v1",
        "verdict": verdict,
        "readonly_live_check": live_verdict,
        "checks": checks,
    }
    json_path = os.path.join(a.out_dir, "RUNPOD_PRE_RENTAL_READINESS.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    md = ["# RunPod Pre-Rental Readiness", "",
          f"**Verdict: {verdict}**",
          f"Read-only live check: {live_verdict}", "", "| check | ok |", "|---|---|"]
    for name, c in sorted(checks.items()):
        md.append(f"| {name} | {'PASS' if c['ok'] else 'FAIL'} |")
    with open(os.path.join(a.out_dir, "RUNPOD_PRE_RENTAL_READINESS.md"),
              "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"RUNPOD PRE-RENTAL READINESS: {verdict}")
    return 0 if verdict != "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
