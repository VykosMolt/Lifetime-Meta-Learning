"""Production RunPod adapter for the O1 accelerator session (B300 target).

Layering (each concern separated):

  request construction   pod_request.py / quote.py
  request validation     models.py (typed, fail-closed)
  REST transport         transport.py (read-only vs authorized-mutating)
  spot acquisition       graphql_spot.py (the ONLY interruptible surface)
  response validation    models.py + schema_check.py
  lifecycle policy       lifecycle.py
  billing policy         billing.py
  termination confirm    here + watchdog_terminate.py (independent path)

Purchase mode is frozen INTERRUPTIBLE: pods are created through the pinned
GraphQL podRentInterruptable mutation (the REST v2 contract cannot express
spot capacity), while status polling, logs, billing and termination stay
on the pinned REST v2 surface — a spot pod is a pod.  Profile selection is
preference-ordered (B300 primary, B200 explicit fallback) and every
selection is recorded with its reason; nothing outside the two frozen
profiles is ever accepted.

The adapter is read-only unless constructed WITH a verified
LiveMutationAuthorization; without it every mutating operation raises
LIVE_MUTATION_NOT_AUTHORIZED before any network I/O.  It also conforms to
the runner's generic ProviderAdapter contract (o1_b200.runner.provider_adapter)
so the existing zero-touch state machine drives it unchanged.
"""
from __future__ import annotations

import json
import os
import tarfile
import time

from .authorization import AuthorizationError, LiveMutationAuthorization
from .billing import (
    BudgetViolation, SpendTracker, hard_compute_seconds,
    remaining_compute_seconds, session_fits_policy,
)
from .graphql_spot import PRODUCTION_GRAPHQL_URL, RunpodGraphQlClient
from .identityutil import utcnow_iso
from .models import (
    CreatePodRequestModel, GpuTypeModel, PodModel, SchemaIncompatibility,
)
from .policy import (
    MIN_CUDA_VERSION, PROFILES_BY_KEY, PURCHASE_MODE, QUOTE_VALIDITY_SECONDS,
)
from .pod_request import (
    build_pod_request, identity_body, render_canonical_deployment,
    verify_deployment_rendering,
)
from .quote import (
    CATALOG_PATH, QuoteError, build_quote, check_quote_fresh, parse_catalog,
    select_offer,
)
from .redaction import redact
from .schema_check import check_live_identity, pinned_schema_sha256, verify_pinned_schema
from .transport import (
    AmbiguousMutation, ApiHttpError, MutatingTransport, ReadOnlyTransport,
    TransportError,
)

REST_PRODUCTION_URL = "https://api.runpod.io"

# Safety margin added to the provider-side terminateAfter auto-terminate
# beyond the locally enforced hard budget deadline.
TERMINATE_AFTER_MARGIN_SECONDS = 30 * 60


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
    def __init__(self, *, base_url: str = REST_PRODUCTION_URL,
                 authorization: LiveMutationAuthorization | None = None,
                 api_key: str | None = None, opener=None,
                 sleep=time.sleep, clock=time.time,
                 spend_clock=time.monotonic, profile_preference=None):
        verify_pinned_schema()
        self.readonly = ReadOnlyTransport(base_url=base_url, api_key=api_key,
                                          opener=opener, sleep=sleep)
        graphql_url = (PRODUCTION_GRAPHQL_URL
                       if base_url.rstrip("/") == REST_PRODUCTION_URL
                       else base_url.rstrip("/") + "/graphql")
        self.graphql = RunpodGraphQlClient(
            url=graphql_url, api_key=api_key, authorization=authorization,
            opener=opener, sleep=sleep)
        self.authorization = authorization
        self.mutating = None
        if authorization is not None:
            self.mutating = MutatingTransport(
                authorization, base_url=base_url, api_key=api_key,
                opener=opener, sleep=sleep)
        self.clock = clock
        self.spend_clock = spend_clock
        # optional operator reordering of the frozen profiles (never an
        # addition: the authorization covers every frozen profile already)
        self.profile_preference = profile_preference
        self.accepted_quote: dict | None = None
        self.spend: SpendTracker | None = None
        #: audit trail for ownership decisions taken without a verifiable
        #: launch nonce (surfaced in the lifecycle report)
        self._reconciliation_notes: list[dict] = []

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

    def get_offer_selection(self) -> dict:
        """Preference-ordered profile selection against the live catalog."""
        return select_offer(self.list_gpu_types(), self.profile_preference)

    def get_availability(self) -> dict:
        """Availability + live spot pricing for the selected profile, with
        the full per-profile refusal record (fallbacks are never silent)."""
        sel = self.get_offer_selection()
        profile, gpu = sel["profile"], sel["gpu"]
        spot = self.graphql.spot_pricing(profile.gpu_type_id)
        return {"profile": profile.key,
                "profile_role": profile.role,
                "fallback_reason": sel["fallback_reason"],
                "profile_refusals": sel["refusals"],
                "gpu_id": gpu.id, "availability": gpu.availability,
                "purchase_mode": PURCHASE_MODE,
                "secure_list_usd": str(spot.get("secure_list_usd")),
                "secure_spot_usd": str(spot.get("secure_spot_usd")),
                "spot_stock_status": spot.get("stock_status"),
                "datacenters": sel["datacenters"]}

    def list_compatible_datacenters(self) -> list[str]:
        return self.get_offer_selection()["datacenters"]

    def quote_instance(self, *, disk_hourly_usd="0.0000",
                       adapter_commit: str = "UNKNOWN") -> dict:
        sel = self.get_offer_selection()
        spot = self.graphql.spot_pricing(sel["profile"].gpu_type_id)
        dc = sorted(sel["datacenters"])[0]
        quote = build_quote(
            sel, spot, datacenter_id=dc, disk_gb=60,
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

    def build_pod_request(self, *, profile_key: str, image_digest_ref: str,
                          datacenter_id: str,
                          env_values: dict | None = None) -> CreatePodRequestModel:
        return build_pod_request(profile=PROFILES_BY_KEY[profile_key],
                                 image_digest_ref=image_digest_ref,
                                 datacenter_id=datacenter_id,
                                 env_values=env_values)

    def render_canonical_deployment(self, reqs_by_profile: dict,
                                    identities: dict) -> dict:
        return render_canonical_deployment(reqs_by_profile, identities)

    # ---------------- mutating operations (interlocked) ----------------

    def _require_authorized(self) -> MutatingTransport:
        if self.mutating is None:
            raise AuthorizationError(
                "adapter constructed without a live-mutation authorization")
        return self.mutating

    def create_instance(self, req: CreatePodRequestModel,
                        rendered: dict) -> PodModel:
        """Create ONE interruptible pod via the pinned GraphQL surface."""
        self._require_authorized()
        req.validate()
        if self.accepted_quote is None:
            raise QuoteError("no accepted quote; quote_instance first")
        check_quote_fresh(self.accepted_quote, now=self.clock)
        if self.accepted_quote["gpu_type_id"] != req.gpu_type_id:
            raise RunpodAdapterError(
                f"quote is for {self.accepted_quote['gpu_type_id']!r} but "
                f"the request pins {req.gpu_type_id!r}; re-quote for the "
                f"selected profile")
        # Re-derive the hash from the rendering's CONTENTS before comparing:
        # trusting rendered["request_sha256"] would compare the operator's
        # commitment against a caller-supplied claim, letting an arbitrary
        # unrendered body deploy under a valid authorization.
        actual_spec_sha256 = verify_deployment_rendering(rendered)
        if actual_spec_sha256 != self.authorization.deployment_spec_sha256:
            raise AuthorizationError(
                "deployment specification hash does not match the "
                "authorization's committed deployment_spec_sha256")
        profile_key = self.accepted_quote["profile"]
        body = rendered.get("create_pod_bodies", {}).get(profile_key)
        if body is None or body != identity_body(req.to_json()):
            raise AuthorizationError(
                f"the request for profile {profile_key} is not the "
                f"authorized canonical body; refusing")
        # duplicate protection: reconcile by canonical name + nonce first
        existing = self._reconcile_by_identity(req.name)
        if existing is not None:
            raise RunpodAdapterError(
                f"a pod named {req.name!r} already exists "
                f"({existing.id}); refusing duplicate creation")
        # Budget FIRST, then burn the slot: a refusal here creates no pod,
        # so consuming an authorization slot for it would permanently spend
        # one of a small, deliberately bounded set for nothing.
        limit = remaining_compute_seconds(
            self.accepted_quote["total_projected_hourly_usd"],
            self.session_spend_usd())
        if limit <= 0:
            raise BudgetViolation(
                "no compute allocation remains; refusing to create a pod")
        self.authorization.consume_nonce()
        # provider-side auto-terminate: a redundant backstop derived from the
        # REMAINING allocation (never the full one), so the sum of all pods'
        # unattended horizons can never exceed the compute budget even if
        # this orchestrator dies mid-session
        terminate_after = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(self.clock() + limit + TERMINATE_AFTER_MARGIN_SECONDS))
        env = dict(req.env)
        env["O1_LAUNCH_NONCE"] = self.authorization.launch_nonce
        req_with_nonce = CreatePodRequestModel(
            name=req.name, image=req.image, cloud=req.cloud,
            gpu_type_id=req.gpu_type_id, gpu_count=req.gpu_count,
            container_disk_gb=req.container_disk_gb, env=env,
            purchase_mode=req.purchase_mode, ports=req.ports,
            args=req.args, datacenter_ids=req.datacenter_ids)
        rent_input = req_with_nonce.to_rent_input(
            self.accepted_quote["bid_per_gpu_usd"],
            min_cuda_version=MIN_CUDA_VERSION,
            terminate_after_utc=terminate_after)
        try:
            created = self.graphql.rent_interruptable(rent_input)
        except (AmbiguousMutation, TransportError, SchemaIncompatibility) as exc:
            # ANY uncertain outcome on a CREATE is ambiguous, not a clean
            # failure: a 5xx from the provider's gateway can arrive after
            # the pod was already created, and assuming otherwise leaves a
            # billing pod whose id nothing recorded.  Reconcile first; never
            # retry the create blindly.
            pod = self._reconcile_by_identity(req.name,
                                              self.authorization.launch_nonce)
            if pod is None:
                raise RunpodAdapterError(
                    f"spot create outcome unknown ({type(exc).__name__}) and "
                    f"no matching pod found; NOT retrying create — "
                    f"operator-visible halt") from None
            self._reconciliation_notes.append({
                "event": "ADOPTED_AFTER_AMBIGUOUS_CREATE",
                "pod_id": pod.id, "error": str(exc)[:200]})
            self._arm_spend()
            return pod
        pod_id = created.get("id")
        if not pod_id:
            raise RunpodAdapterError("spot create returned no pod id")
        self._arm_spend()
        # authoritative state comes from the REST surface
        return self.get_instance(pod_id)

    def session_spend_usd(self):
        """Cumulative session spend including every earlier evicted pod."""
        return self.spend.effective_spend() if self.spend is not None else "0"

    def _arm_spend(self) -> None:
        """Arm the meter at POD CREATION, not at RUNNING.

        RunPod bills a pod from provisioning onward, so a pod evicted before
        its container ever reached RUNNING still costs money; starting the
        meter at RUNNING recorded those attempts as free.
        """
        carry = self.session_spend_usd()
        self.spend = SpendTracker(
            self.accepted_quote["total_projected_hourly_usd"],
            clock=self.spend_clock, carryover_usd=carry)
        self.spend.mark_pod_started()

    def _reconcile_by_identity(self, name: str,
                               nonce: str | None = None) -> PodModel | None:
        """Reconcile owned pods by canonical name (+ launch nonce) across
        BOTH surfaces: REST /v2/pods and GraphQL myself.pods.  REST
        visibility of spot pods is a live-hardware-unvalidated assumption,
        so the GraphQL listing is consulted as a redundant second witness.
        """
        by_id: dict[str, PodModel] = {}
        for pod in self.list_owned_instances():
            by_id[pod.id] = pod
        try:
            for p in self.graphql.myself_pods():
                if p["id"] not in by_id and p.get("name") == name:
                    if p.get("desiredStatus") != "TERMINATED":
                        by_id[p["id"]] = self.get_instance(p["id"])
        except Exception:  # noqa: BLE001 - redundant surface; REST rules
            pass
        matches, unverified = [], []
        for pod in by_id.values():
            if pod.name != name or pod.status == "TERMINATED":
                continue
            if nonce is None:
                matches.append(pod)
                continue
            env = pod.extra.get("env")
            if isinstance(env, dict) and env.get("O1_LAUNCH_NONCE"):
                if env["O1_LAUNCH_NONCE"] != nonce:
                    continue        # provably a DIFFERENT launch: not ours
                matches.append(pod)
            else:
                # The live REST surface may omit env, in which case the
                # nonce cannot be checked.  Such a pod is NOT silently
                # treated as ours: it is held aside, and only used when no
                # nonce-verified pod exists at all.
                unverified.append(pod)
        if nonce is not None and not matches and unverified:
            # fall back to the newest by creation time and say so loudly,
            # rather than adopting an arbitrary same-named leftover
            unverified.sort(key=lambda p: str(p.created_at or ""),
                            reverse=True)
            matches = [unverified[0]]
            self._reconciliation_notes.append({
                "event": "NONCE_UNVERIFIED_ADOPTION",
                "pod_id": unverified[0].id,
                "candidates": len(unverified),
                "reason": "the pod listing carried no O1_LAUNCH_NONCE, so "
                          "ownership was inferred from the canonical name "
                          "and the newest creation time"})
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

    def classify_exit(self, pod: PodModel) -> str:
        """EXITED on interruptible capacity without completion is treated as
        eviction (SIGTERM/SIGKILL stop); ERROR is a container failure."""
        if pod.status == "ERROR":
            return "CONTAINER_FAILURE"
        if pod.status == "EXITED":
            return "EVICTION_SUSPECTED"
        return pod.status

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
                    f"pod {pod_id} reached {pod.status} before RUNNING "
                    f"({self.classify_exit(pod)})")
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
        m = self._require_authorized()
        m.mutate("POST", f"/v2/pods/{pod_id}/action", {"action": "stop"},
                 releasing=True)

    def terminate_instance(self, pod_id: str) -> None:
        """Primary termination: permanent delete, never merely stop."""
        m = self._require_authorized()
        try:
            m.mutate("POST", f"/v2/pods/{pod_id}/action",
                     {"action": "terminate"}, releasing=True)
        except ApiHttpError:
            # Deliberately swallowed for EVERY status: termination is
            # redundant by design and the DELETE below is the second,
            # independent path.  A raise here would skip it.
            pass
        try:
            m.mutate("DELETE", f"/v2/pods/{pod_id}", releasing=True)
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
