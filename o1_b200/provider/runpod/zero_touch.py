#!/usr/bin/env python3
"""RunPod zero-touch session driver (B300 primary / B200 fallback, spot).

Layering: the SCIENTIFIC zero-touch state machine (hardware validation,
non-O1 equivalence, bounded benchmark, backend selection, replacement
precommit, external commit verification, affordability gate, calibration,
record verify) runs ON THE POD via the container start command.  This
driver owns the PROVIDER side with no human interaction:

  1. live-mutation interlock (or immediate refusal);
  2. read-only preflight + FRESH quote (stale quotes refused);
  3. canonical all-profile deployment render + authorization hash check;
  4. reconciled, duplicate-safe INTERRUPTIBLE Pod creation via the pinned
     GraphQL surface; watchdogs armed;
  5. monitor to RUNNING; poll logs/status; budget soft/hard stops;
  6. EVICTION of interruptible capacity is normal: terminate the remnant,
     carry the spend forward, re-quote fresh (profile re-selected), and
     reacquire — bounded by the hard dollar budget, max_pod_creations, and
     a zero-progress repeat-failure guard; the on-pod state machine
     revalidates hardware/environment from scratch on every pod and
     resumes from durable off-pod state (rows/checkpoints);
  7. wait for the pod's machine-readable FINAL_STATUS artifact;
  8. verified result download;
  9. TERMINATE (never merely stop) + independent confirmation;
 10. machine-readable local diagnostics package.

Every failure path terminates the pod and returns a complete local
diagnostic package; no path asks the user to SSH, read logs, choose a
datacenter/backend/bid, monitor spend, press Terminate, or repair anything.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .adapter import RunpodV2Adapter
from .authorization import (
    AuthorizationError, CLI_FLAG, LiveMutationAuthorization,
)
from .billing import BudgetViolation, remaining_compute_seconds
from .identityutil import utcnow_iso
from .lifecycle import LifecycleError, PodLifecycleController
from .pod_request import build_pod_request, render_canonical_deployment
from .policy import MAX_POD_ACQUISITIONS, PROFILE_PREFERENCE, PROFILES_BY_KEY
from .preflight import run_preflight
from .redaction import redact, register_env_secrets


class DeterministicPodFailure(RuntimeError):
    """The pod reported a reproducible failure; reacquiring cannot help."""


def load_session_config(root: str) -> dict:
    """Deployment facts resolved before launch (image digest, identities)."""
    path = os.path.join(root, "o1_b200", "provider", "runpod",
                        "RUNPOD_SESSION_CONFIG.json")
    if not os.path.exists(path):
        raise AuthorizationError(
            "RUNPOD_SESSION_CONFIG.json missing: the authorized session "
            "requires the resolved image digest and identity hashes")
    with open(path, encoding="utf-8") as fh:
        config = json.load(fh)
    unresolved = [k for k, v in config.items()
                  if isinstance(v, str) and v.startswith("UNRESOLVED")]
    if unresolved:
        raise AuthorizationError(
            f"session config carries unresolved template fields {unresolved}")
    return config


def identity_env_values(config: dict) -> dict:
    """Env values that ARE deployment identity (frozen into the rendering).

    These decide what the pod fetches and where it publishes, so they are
    bound by the authorization rather than free at launch.
    """
    return {
        "O1_B200_OUT": config.get("pod_out_dir", "/outputs"),
        "O1_B200_ARTIFACT_SOURCE": config.get("artifact_source", ""),
        "O1_B200_RESULT_DESTINATION": config.get("result_destination", ""),
        "O1_IMAGE_DIGEST": config["image_digest_ref"],
    }


def _render_all_profiles(config: dict) -> dict:
    env_values = identity_env_values(config)
    reqs = {p.key: build_pod_request(profile=p,
                                     image_digest_ref=config["image_digest_ref"],
                                     datacenter_id="AUTHORIZED-ANY",
                                     env_values=env_values)
            for p in PROFILE_PREFERENCE}
    return render_canonical_deployment(reqs, config["identities"])


def _expected_result_digest(config: dict) -> tuple[str, str | None]:
    """The result digest the POD recorded, read from the durable store.

    Returns (state, digest) where state is:
      "VERIFIABLE" — a sidecar exists and its digest was read;
      "ABSENT"     — no durable store is configured, or no sidecar exists;
      "ERROR"      — the store is configured but could not be consulted.

    ABSENT and ERROR are deliberately NOT the same: conflating them let a
    broken store yield the same COMPLETE verdict as a fully verified run.
    """
    destination = config.get("result_destination", "")
    if not destination:
        return ("ABSENT", None)
    try:
        import json as _json
        import tempfile

        from o1_b200.runner.durability import store_for_destination
        store = store_for_destination(destination)
        listing = store.list_prefix("results")
        target = next((r for r in listing
                       if r.endswith("o1_results.tar.gz.sha256")), None)
        if target is None:
            return ("ABSENT", None)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "digest.json")
            store.fetch_file(target, local)
            with open(local, encoding="utf-8") as fh:
                return ("VERIFIABLE", _json.load(fh)["sha256"])
    except Exception as exc:  # noqa: BLE001
        return ("ERROR", redact(str(exc))[:200])


def durable_progress_probe(destination: str):
    """Off-pod durable-progress reader for the zero-progress guard.

    The driver cannot see the pod's filesystem, but it CAN read the same
    durable store the pod syncs to: the committed-row count in the durable
    records manifest is real, externally-visible progress.  Returns None on
    any error (unknown != zero, so a store hiccup never trips the guard).
    """
    def probe():
        try:
            from o1_b200.runner.durability import (
                CheckpointDurability, store_for_destination,
            )
            store = store_for_destination(destination)
            mirror = CheckpointDurability(
                store, remote_prefix="durable_o1_records")
            return mirror.latest_row_count()
        except Exception:  # noqa: BLE001 - progress is advisory
            return None
    return probe


def run_session(*, authorization_path: str, out_dir: str,
                cli_args: list[str], base_url: str = "https://api.runpod.io",
                api_key: str | None = None,
                root: str | None = None, sleep=time.sleep,
                config: dict | None = None,
                spawn_watchdog_fn=None,
                progress_probe=None, spend_clock=None) -> dict:
    """progress_probe: optional zero-arg callable returning a monotonic
    progress marker (e.g. count of durably synced rows); used to abort a
    repeating zero-progress 'eviction' as a container defect."""
    os.makedirs(out_dir, exist_ok=True)
    # register every configured credential literally, BEFORE any step can
    # be recorded, so redaction never depends on a value matching a shape
    register_env_secrets()
    root = root or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    status: dict = {"schema": "o1b300.runpod_session_status.v2",
                    "utc": utcnow_iso(), "outcome": None, "steps": [],
                    "acquisitions": []}

    def step(name, detail=None):
        status["steps"].append({"step": name, "utc": utcnow_iso(),
                                "detail": redact(str(detail)) if detail else None})

    def finish(outcome, **extra):
        status["outcome"] = outcome
        status.update(extra)
        path = os.path.join(out_dir, "RUNPOD_SESSION_STATUS.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return status

    config = config or load_session_config(root)
    expected_identity = {
        "project": config["project"],
        "package_zip_sha256": config["package_zip_sha256"],
        "provider": "runpod",
        "budget_policy_sha256": config["budget_policy_sha256"],
        "deployment_spec_sha256": None,  # bound after render below
    }
    # 2. read-only preflight first (no authorization needed)
    pre = run_preflight(base_url=base_url, api_key=api_key)
    step("READONLY_PREFLIGHT", pre["verdict"])
    if pre["verdict"] != "PASS":
        return finish("REFUSED_PREFLIGHT", preflight=pre)
    owned = (pre.get("checks") or {}).get("owned_pods") or {}
    if owned.get("unexpected_active_billable_pod"):
        # An active pod we did not expect is either a leftover from an
        # earlier run or someone else's work.  Launching alongside it would
        # add a second billable resource to an account already spending, so
        # refuse and name it rather than proceed.
        return finish("REFUSED_UNEXPECTED_ACTIVE_POD", preflight=pre,
                      error=f"account already has active pod(s) "
                            f"{owned.get('active_pods')}; terminate or "
                            f"account for them before launching")
    if pre.get("capacity_unavailable_now"):
        # market state, not a software failure — but nothing can be rented
        # right now, so refuse cleanly before any authorization/mutation
        return finish("REFUSED_NO_CAPACITY", preflight=pre)

    # 3. canonical ALL-PROFILE deployment render + authorization binding
    rendered = _render_all_profiles(config)
    expected_identity["deployment_spec_sha256"] = rendered["request_sha256"]
    try:
        auth = LiveMutationAuthorization.verify(
            path=authorization_path, expected_identity=expected_identity,
            cli_args=cli_args,
            nonce_ledger=os.path.join(out_dir, "consumed_nonces.txt"))
    except AuthorizationError as exc:
        step("AUTHORIZATION_REFUSED", exc)
        return finish("LIVE_MUTATION_NOT_AUTHORIZED", error=str(exc))
    step("AUTHORIZED", "interlock satisfied")

    adapter_kw = {}
    if spend_clock is not None:
        adapter_kw["spend_clock"] = spend_clock
    # Optional operator ordering of the FROZEN profiles (e.g. put B200
    # first when B300 demand makes it unobtainable).  This is a preference,
    # not a new capability: the authorization already commits to every
    # frozen profile's canonical body, so no unrendered deployment becomes
    # possible.  Recorded in the session status and in every quote.
    if config.get("profile_preference"):
        adapter_kw["profile_preference"] = list(config["profile_preference"])
        status["profile_preference"] = list(config["profile_preference"])
    adapter = RunpodV2Adapter(base_url=base_url, api_key=api_key,
                              authorization=auth, sleep=sleep, **adapter_kw)
    controller_kw = {}
    if spawn_watchdog_fn is not None:
        controller_kw["spawn_watchdog_fn"] = spawn_watchdog_fn

    def _make_result_witness():
        """Durable second witness: did the pod publish its results archive?

        Independent of the pod's log endpoint, so a lost/500ing log API at
        the exact moment the container exits cannot be mistaken for
        "unfinished" and trigger a paid reacquisition.
        """
        destination = config.get("result_destination", "")
        if not destination:
            return None

        def witness():
            try:
                from o1_b200.runner.durability import store_for_destination
                store = store_for_destination(destination)
                return any(rel.endswith("o1_results.tar.gz")
                           for rel in store.list_prefix("results"))
            except Exception:  # noqa: BLE001 - advisory witness only
                return False
        return witness

    result_witness = _make_result_witness()

    def pod_done(pod, retries: int = 3):
        """Completion witness.

        A transient failure of the log endpoint must NOT be read as "not
        finished": that would classify a completed pod as evicted and pay
        for a whole redundant reacquisition.  The log probe is retried, and
        a lost log endpoint falls back to the durable result witness the pod
        publishes before exiting.
        """
        for attempt_i in range(retries):
            try:
                tail = adapter.get_container_logs(pod.id, tail=50)
                if "ZERO_TOUCH_ABORTED_AT_" in tail:
                    # The pod reached a DETERMINISTIC verdict and said so.
                    # Reacquiring cannot help — a fresh pod runs the same
                    # gates against the same artifacts and fails identically
                    # — so this must never be mistaken for an eviction.
                    marker = tail.split("ZERO_TOUCH_ABORTED_AT_")[-1].split()[0]
                    raise DeterministicPodFailure(
                        f"the pod aborted deterministically at {marker}; "
                        f"reacquisition would repeat it")
                return "ZERO_TOUCH_COMPLETE" in tail
            except DeterministicPodFailure:
                raise
            except Exception:  # noqa: BLE001 - log endpoint may lag
                if attempt_i + 1 < retries:
                    sleep(2.0)
        step("COMPLETION_WITNESS_LOG_UNAVAILABLE", pod.id)
        return bool(result_witness and result_witness())

    if progress_probe is None:
        destination = config.get("result_destination", "")
        if destination:
            progress_probe = durable_progress_probe(destination)

    last_progress = progress_probe() if progress_probe else None
    evictions_seen = 0

    def zero_progress_abort() -> bool:
        """True when a SECOND consecutive interruption arrives with zero new
        durable progress — treated as a container defect, not an eviction."""
        nonlocal last_progress, evictions_seen
        evictions_seen += 1
        if progress_probe is None:
            return False
        progress = progress_probe()
        if progress is None:            # unknown is not zero
            return False
        if progress == last_progress and evictions_seen > 1:
            return True
        last_progress = progress
        return False

    attempt = 0
    while True:
        attempt += 1
        if attempt > MAX_POD_ACQUISITIONS:
            return finish("ABORTED_ACQUISITION_LIMIT",
                          error=f"exceeded {MAX_POD_ACQUISITIONS} "
                                f"acquisitions")
        controller = PodLifecycleController(adapter, out_dir, sleep=sleep,
                                            attempt=attempt, **controller_kw)
        pod_id = None
        try:
            # fresh quote every acquisition: live spot rate, profile
            # re-selected under the frozen preference order
            quote = adapter.quote_instance(
                adapter_commit=config.get("adapter_commit", "UNKNOWN"))
            step("QUOTE", f"attempt {attempt}: {quote['profile']} "
                          f"({quote['profile_role']}) "
                          f"{quote['gpu_type_id']} @ "
                          f"{quote['bid_per_gpu_usd']}/h spot in "
                          f"{quote['datacenter_id']}")
            if quote["fallback_reason"]:
                step("FALLBACK_SELECTED", quote["fallback_reason"])
            status["acquisitions"].append(
                {"attempt": attempt, "profile": quote["profile"],
                 "fallback_reason": quote["fallback_reason"],
                 "operator_preference_applied": quote.get(
                     "operator_preference_applied", False),
                 "bid_per_gpu_usd": quote["bid_per_gpu_usd"]})
            profile = PROFILES_BY_KEY[quote["profile"]]
            adapter.validate_quote(quote)
            # the pod's own runtime allowance is the REMAINING allocation,
            # net of everything earlier pods in this session already spent
            limit = remaining_compute_seconds(
                quote["total_projected_hourly_usd"],
                adapter.session_spend_usd())
            if limit <= 0:
                return finish("ABORTED_BUDGET",
                              error="no compute allocation remains; refusing "
                                    "to acquire another pod")
            env_values = dict(identity_env_values(config))
            env_values.update({
                "O1_ACQUIRED_PROFILE": quote["profile"],
                "O1_SESSION_AUTHORIZED_SECONDS": str(limit),
                "O1_HOURLY_RATE_USD": quote["total_projected_hourly_usd"],
            })
            if os.environ.get("HF_TOKEN"):
                env_values["HF_TOKEN"] = os.environ["HF_TOKEN"]
                register_env_secrets()
            req = build_pod_request(
                profile=profile,
                image_digest_ref=config["image_digest_ref"],
                datacenter_id=quote["datacenter_id"],
                env_values=env_values)
            # the pod may not outlive its own paid allowance plus a
            # margin for startup and termination
            controller.monitor_timeout = limit + 1800
            # 4. provision (reconciled, duplicate-safe) + watchdogs
            pod_id = controller.provision(req, rendered)
            # 5. run
            startup = controller.wait_until_running(pod_id)
            if startup == "EVICTED":
                step("EVICTED_DURING_STARTUP", f"attempt {attempt}")
                if zero_progress_abort():
                    return finish(
                        "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
                        error="a second consecutive interruption with zero "
                              "durable progress is treated as a container "
                              "defect, not an eviction")
                continue
            outcome = controller.monitor(pod_id, until=pod_done)
            step("MONITOR_RESULT", f"attempt {attempt}: {outcome}")
            if outcome == "EVICTED":
                if status["acquisitions"]:
                    status["acquisitions"][-1]["outcome"] = "EVICTED"
                if zero_progress_abort():
                    return finish(
                        "ABORTED_REPEATED_FAILURE_NO_PROGRESS",
                        error="a second consecutive interruption with zero "
                              "durable progress is treated as a container "
                              "defect, not an eviction")
                continue
            if outcome != "COMPLETE":
                # SOFT_STOP (budget) and TERMINATED (external/watchdog) are
                # NOT completions: the pod never emitted its completion
                # witness, so no COMPLETE verdict may be written.  Collect
                # diagnostics, terminate, and report the real outcome.
                controller.collect_logs(pod_id)
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish(
                    f"ABORTED_{outcome}",
                    termination_confirmed=confirmed,
                    acquisitions_used=attempt,
                    error=f"monitor returned {outcome} without the pod's "
                          f"completion witness; no results are claimed")
            # 6/7. results: collect logs + download outputs, then verify the
            # downloaded archive against the digest the POD recorded — a
            # fixed result path with no cross-check would happily "complete"
            # against a stale archive from an earlier attempt
            controller.collect_logs(pod_id)
            dest = os.path.join(out_dir, "downloaded_results")
            got = adapter.download_results(
                config.get("result_source", dest), dest)
            step("RESULTS_DOWNLOADED", dest)
            witness_state, expected = _expected_result_digest(config)
            if witness_state == "VERIFIABLE" and got.get("sha256") != expected:
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish(
                    "ABORTED_RESULT_DIGEST_MISMATCH",
                    termination_confirmed=confirmed,
                    acquisitions_used=attempt,
                    error=f"downloaded archive digest "
                          f"{str(got.get('sha256'))[:16]} != the digest the "
                          f"pod recorded {str(expected)[:16]}; refusing to "
                          f"claim these results")
            status["result_sha256"] = got.get("sha256")
            status["result_witness"] = witness_state
            status["result_digest_verified"] = witness_state == "VERIFIABLE"
            if witness_state == "ERROR":
                step("RESULT_WITNESS_UNAVAILABLE", expected)
        except DeterministicPodFailure as exc:
            step("DETERMINISTIC_POD_FAILURE", exc)
            controller.collect_logs(pod_id)
            confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_DETERMINISTIC_POD_FAILURE",
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt, error=redact(str(exc)))
        except BudgetViolation as exc:
            step("BUDGET_STOP", exc)
            confirmed = True
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_BUDGET", error=redact(str(exc)),
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt)
        except AuthorizationError as exc:
            # creation-slot exhaustion or expiry mid-session
            step("AUTHORIZATION_STOP", exc)
            confirmed = True
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
            return finish("ABORTED_AUTHORIZATION_EXHAUSTED",
                          error=redact(str(exc)),
                          termination_confirmed=confirmed,
                          acquisitions_used=attempt)
        except (LifecycleError, Exception) as exc:  # noqa: BLE001
            step("SESSION_FAILURE", exc)
            if pod_id is not None:
                confirmed = controller.terminate_and_confirm(pod_id)
                return finish("ABORTED_TERMINATED" if confirmed
                              else "ABORTED_TERMINATION_UNCONFIRMED",
                              error=redact(str(exc)))
            return finish("ABORTED_BEFORE_CREATE", error=redact(str(exc)))
        # 8. terminate + confirm (always; stop is never final)
        confirmed = controller.terminate_and_confirm(pod_id)
        if not confirmed:
            outcome = "TERMINATION_UNCONFIRMED"
        elif status.get("result_witness") == "ERROR":
            # the results may well be correct, but nothing could confirm
            # them: that must not read as an ordinary verified COMPLETE
            outcome = "COMPLETE_RESULT_DIGEST_UNVERIFIED"
        else:
            outcome = "COMPLETE"
        return finish(outcome, termination_confirmed=confirmed,
                      acquisitions_used=attempt)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--authorization", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--execute-authorized-rental", action="store_true")
    p.add_argument("--base-url", default="https://api.runpod.io")
    a = p.parse_args()
    cli = [CLI_FLAG] if a.execute_authorized_rental else []
    status = run_session(authorization_path=a.authorization, out_dir=a.out,
                         cli_args=cli, base_url=a.base_url)
    print(json.dumps({"outcome": status["outcome"]}, indent=2))
    return 0 if status["outcome"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
