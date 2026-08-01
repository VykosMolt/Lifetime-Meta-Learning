"""Production RunPod API v2 adapter for the O1 B200 session.

Layering (each concern separated):

  request construction   pod_request.py / quote.py
  request validation     models.py (typed, fail-closed)
  network transport      transport.py (read-only vs authorized-mutating)
  response validation    models.py + schema_check.py
  lifecycle policy       lifecycle.py
  billing policy         billing.py
  termination confirm    here + watchdog_terminate.py (independent path)

The adapter is read-only unless constructed WITH a verified
LiveMutationAuthorization; without it every mutating operation raises
LIVE_MUTATION_NOT_AUTHORIZED before any network I/O.  It also conforms to
the runner's generic ProviderAdapter contract (o1_b200.runner.provider_adapter)
so the existing zero-touch state machine drives it unchanged.
"""
from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time

from .authorization import AuthorizationError, LiveMutationAuthorization
from .billing import SpendTracker, hard_compute_seconds, session_fits_policy
from .identityutil import utcnow_iso
from .models import (
    CreatePodRequestModel, GpuTypeModel, PodModel, SchemaIncompatibility,
)
from .policy import CLOUD, EXPECTED_GPU_ID, GPU_COUNT, QUOTE_VALIDITY_SECONDS
from .pod_request import build_pod_request, render_canonical_pod_request
from .quote import (
    CATALOG_PATH, QuoteError, build_quote, check_quote_fresh, find_b200,
    parse_catalog, validate_offer,
)
from .redaction import redact
from .schema_check import check_live_identity, pinned_schema_sha256, verify_pinned_schema
from .transport import (
    AmbiguousMutation, ApiHttpError, MutatingTransport, ReadOnlyTransport,
)


class RunpodAdapterError(RuntimeError):
    def __init__(self, msg: str):
        super().__init__(redact(msg))


def parse_hf_uri(uri: str) -> tuple[str, str]:
    """hf://<namespace>/<repo>/<path...> -> (namespace/repo, path)."""
    body = uri[len("hf://"):]
    parts = body.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise RunpodAdapterError(
            f"malformed hf:// result source {uri!r}; expected "
            f"hf://<namespace>/<repo>/<path>")
    return f"{parts[0]}/{parts[1]}", parts[2]


class RunpodV2Adapter:
    def __init__(self, *, base_url: str = "https://api.runpod.io",
                 authorization: LiveMutationAuthorization | None = None,
                 api_key: str | None = None, opener=None,
                 sleep=time.sleep, clock=time.time):
        verify_pinned_schema()
        self.readonly = ReadOnlyTransport(base_url=base_url, api_key=api_key,
                                          opener=opener, sleep=sleep)
        self.authorization = authorization
        self.mutating = None
        if authorization is not None:
            self.mutating = MutatingTransport(
                authorization, base_url=base_url, api_key=api_key,
                opener=opener, sleep=sleep)
        self.clock = clock
        self.accepted_quote: dict | None = None
        self.spend: SpendTracker | None = None

    # ---------------- read-only operations ----------------

    def authenticate_readonly(self) -> dict:
        if not self.readonly.has_credential():
            return {"authenticated": False,
                    "reason": "SKIPPED_NO_CREDENTIAL"}
        pods = self.list_owned_instances()
        return {"authenticated": True, "owned_pods": len(pods)}

    def get_api_schema_identity(self) -> dict:
        live = self.readonly.get("/v2/openapi.json")
        if not isinstance(live, dict) or "openapi" not in live:
            raise SchemaIncompatibility("live /v2/openapi.json is not an "
                                        "OpenAPI document")
        return check_live_identity(live)

    def list_gpu_types(self) -> list[GpuTypeModel]:
        return parse_catalog(self.readonly.get(CATALOG_PATH))

    def get_b200_offer(self) -> GpuTypeModel:
        return find_b200(self.list_gpu_types())

    def get_b200_availability(self) -> dict:
        gpu = self.get_b200_offer()
        v = validate_offer(gpu)
        return {"gpu_id": gpu.id, "availability": gpu.availability,
                "secure_rate_usd": str(v["rate"]),
                "datacenters": v["datacenters"]}

    def list_compatible_datacenters(self) -> list[str]:
        return self.get_b200_availability()["datacenters"]

    def quote_instance(self, *, disk_hourly_usd="0.0000",
                       adapter_commit: str = "UNKNOWN") -> dict:
        gpu = self.get_b200_offer()
        v = validate_offer(gpu)
        dc = sorted(v["datacenters"])[0]
        quote = build_quote(
            gpu, datacenter_id=dc, disk_gb=60,
            disk_hourly_usd=disk_hourly_usd,
            schema_sha256=pinned_schema_sha256(),
            adapter_commit=adapter_commit, now=self.clock)
        self.accepted_quote = quote
        return quote

    def validate_quote(self, quote: dict | None = None) -> dict:
        quote = quote or self.accepted_quote
        if quote is None:
            raise QuoteError("no quote to validate")
        check_quote_fresh(quote, now=self.clock)
        limit = hard_compute_seconds(quote["total_projected_hourly_usd"])
        session_fits_policy(quote["total_projected_hourly_usd"],
                            min(limit, 4 * 3600))
        return {"fresh": True, "hard_compute_seconds": limit}

    def list_owned_instances(self) -> list[PodModel]:
        payload = self.readonly.get("/v2/pods")
        items = payload if isinstance(payload, list) else (
            payload.get("pods") or payload.get("data") or [])
        return [PodModel.parse(p) for p in items]

    def get_instance(self, pod_id: str) -> PodModel:
        return PodModel.parse(self.readonly.get(f"/v2/pods/{pod_id}"))

    def get_container_logs(self, pod_id: str, tail: int = 200) -> str:
        return self._logs(pod_id, "container", tail)

    def get_system_logs(self, pod_id: str, tail: int = 200) -> str:
        return self._logs(pod_id, "system", tail)

    def _logs(self, pod_id: str, source: str, tail: int) -> str:
        raw = self.readonly.get(
            f"/v2/pods/{pod_id}/logs?source={source}&tail={int(tail)}")
        if isinstance(raw, dict):
            raw = raw.get("logs", raw)
        return redact(json.dumps(raw) if not isinstance(raw, str) else raw)

    def get_billing_usage(self, pod_id: str | None = None) -> dict:
        path = "/v2/billing/pods"
        if pod_id:
            path += f"?podId={pod_id}"
        payload = self.readonly.get(path)
        if not isinstance(payload, (dict, list)):
            raise SchemaIncompatibility("billing response malformed")
        return {"billing": payload, "retrieved_utc": utcnow_iso(self.clock)}

    # ---------------- request construction (no submission) ----------------

    def build_pod_request(self, *, image_digest_ref: str,
                          datacenter_id: str,
                          env_values: dict | None = None) -> CreatePodRequestModel:
        return build_pod_request(image_digest_ref=image_digest_ref,
                                 datacenter_id=datacenter_id,
                                 env_values=env_values)

    def render_canonical_pod_request(self, req: CreatePodRequestModel,
                                     identities: dict) -> dict:
        return render_canonical_pod_request(req, identities)

    # ---------------- mutating operations (interlocked) ----------------

    def _require_mutating(self) -> MutatingTransport:
        if self.mutating is None:
            raise AuthorizationError(
                "adapter constructed without a live-mutation authorization")
        return self.mutating

    def create_instance(self, req: CreatePodRequestModel,
                        rendered: dict) -> PodModel:
        m = self._require_mutating()
        req.validate()
        if self.accepted_quote is None:
            raise QuoteError("no accepted quote; quote_instance first")
        check_quote_fresh(self.accepted_quote, now=self.clock)
        if rendered.get("request_sha256") != \
                self.authorization.deployment_spec_sha256:
            raise AuthorizationError(
                "deployment specification hash does not match the "
                "authorization's committed deployment_spec_sha256")
        # duplicate protection: reconcile by canonical name + nonce first
        existing = self._reconcile_by_identity(req.name)
        if existing is not None:
            raise RunpodAdapterError(
                f"a pod named {req.name!r} already exists "
                f"({existing.id}); refusing duplicate creation")
        self.authorization.consume_nonce()
        body = req.to_json()
        body["env"] = dict(body.get("env", {}))
        body["env"]["O1_LAUNCH_NONCE"] = self.authorization.launch_nonce
        try:
            created = m.mutate("POST", "/v2/pods", body)
        except AmbiguousMutation:
            pod = self._reconcile_by_identity(req.name,
                                              self.authorization.launch_nonce)
            if pod is None:
                raise RunpodAdapterError(
                    "create response lost and no matching pod found; NOT "
                    "retrying create — operator-visible halt")
            return pod
        pod = PodModel.parse(created)
        self.spend = SpendTracker(
            self.accepted_quote["total_projected_hourly_usd"])
        return pod

    def _reconcile_by_identity(self, name: str,
                               nonce: str | None = None) -> PodModel | None:
        matches = []
        for pod in self.list_owned_instances():
            if pod.name != name:
                continue
            if nonce is not None:
                env = (pod.extra.get("env") or {})
                if isinstance(env, dict) and env.get("O1_LAUNCH_NONCE") not in (
                        None, nonce):
                    continue
            if pod.status != "TERMINATED":
                matches.append(pod)
        if len(matches) > 1:
            # ambiguous: terminate everything matching and halt
            for pod in matches:
                try:
                    self.terminate_instance(pod.id)
                except Exception:  # noqa: BLE001 - keep terminating others
                    pass
            raise RunpodAdapterError(
                f"reconciliation ambiguous: {len(matches)} pods named "
                f"{name!r}; all were sent terminate; halting")
        return matches[0] if matches else None

    def wait_for_state(self, pod_id: str, target: str, *,
                       timeout_seconds: float, poll_seconds: float = 5.0,
                       sleep=None) -> PodModel:
        sleep = sleep or time.sleep
        deadline = self.clock() + timeout_seconds
        last = None
        while self.clock() < deadline:
            pod = self.get_instance(pod_id)
            last = pod.status
            if pod.status == target:
                return pod
            if pod.status in ("ERROR", "EXITED") and target == "RUNNING":
                raise RunpodAdapterError(
                    f"pod {pod_id} reached {pod.status} before RUNNING")
            if pod.status == "TERMINATED" and target != "TERMINATED":
                raise RunpodAdapterError(f"pod {pod_id} terminated unexpectedly")
            sleep(poll_seconds)
        raise RunpodAdapterError(
            f"timeout waiting for {target} on {pod_id} (last={last})")

    def launch_job(self, pod_id: str, job: dict) -> dict:
        # the container's start command IS the job (zero-touch state machine);
        # launch_job records intent and verifies the pod is running it.
        pod = self.get_instance(pod_id)
        if pod.status != "RUNNING":
            raise RunpodAdapterError(f"pod {pod_id} is {pod.status}, not RUNNING")
        return {"job_id": f"zero-touch@{pod_id}", "job": job}

    def stop_instance(self, pod_id: str) -> None:
        m = self._require_mutating()
        m.mutate("POST", f"/v2/pods/{pod_id}/action", {"action": "stop"})

    def terminate_instance(self, pod_id: str) -> None:
        """Primary termination: permanent delete, never merely stop."""
        m = self._require_mutating()
        try:
            m.mutate("POST", f"/v2/pods/{pod_id}/action",
                     {"action": "terminate"})
        except ApiHttpError as exc:
            if exc.status not in (404, 409):
                # fall through to DELETE anyway; termination must be redundant
                pass
        try:
            m.mutate("DELETE", f"/v2/pods/{pod_id}")
        except ApiHttpError as exc:
            if exc.status != 404:
                raise

    def confirm_terminated(self, pod_id: str, *, timeout_seconds: float = 300,
                           poll_seconds: float = 5.0, sleep=None) -> bool:
        sleep = sleep or time.sleep
        deadline = self.clock() + timeout_seconds
        while self.clock() < deadline:
            try:
                pod = self.get_instance(pod_id)
            except ApiHttpError as exc:
                if exc.status == 404:
                    return True     # gone entirely: terminated
                raise
            if pod.status == "TERMINATED":
                return True
            sleep(poll_seconds)
        return False

    # ---------------- results ----------------

    def package_results(self, out_dir: str, archive_path: str) -> dict:
        from .redaction import redact as _r
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(out_dir, arcname=".")
        import hashlib
        h = hashlib.sha256()
        with open(archive_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"archive": archive_path, "sha256": h.hexdigest()}

    def download_results(self, source: str, dest_dir: str,
                         store=None) -> dict:
        os.makedirs(dest_dir, exist_ok=True)
        if source.startswith("hf://"):
            repo, filename = parse_hf_uri(source)
            from huggingface_hub import hf_hub_download
            from .artifact_store import sha256_file as _sha
            path = hf_hub_download(repo_id=repo, filename=filename,
                                   local_dir=dest_dir)
            return {"path": path, "sha256": _sha(path),
                    "bytes": os.path.getsize(path)}
        from .artifact_store import LocalArtifactStore
        store = store or LocalArtifactStore()
        return store.fetch(source, dest_dir)
