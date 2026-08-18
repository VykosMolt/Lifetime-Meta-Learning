"""Canonical Pod request: construction and rendering (never submitted here).

Freezes: name, image by immutable digest, Secure cloud, exactly one GPU of
a preference-ordered accelerator profile (B300 primary, B200 explicit
fallback), INTERRUPTIBLE purchase mode, container disk, no persistent or
network storage (durable state streams continuously to the result
destination; on-pod disk is treated as lost at eviction), no exposed ports
(no Jupyter, no HTTP service, no public SSH — the zero-touch runner needs
none; emergency shell access is intentionally ABSENT rather than
disabled), environment-variable NAMES only (secret values injected at
launch, never frozen into the template), the start command that launches
the sealed zero-touch state machine, health/shutdown behavior, artifact
locations, result destination, and package/container/budget identities.

The rental authorization commits to ONE deployment_spec_sha256 that covers
BOTH profiles' canonical bodies, so an authorized session may acquire the
fallback without a second operator ceremony while remaining unable to
deploy anything not rendered here.
"""
from __future__ import annotations

import json
import re

from .identityutil import canonical_sha256
from .models import CreatePodRequestModel
from .policy import (
    CLOUD, GPU_COUNT, MIN_CUDA_VERSION, PROFILE_PREFERENCE, PURCHASE_MODE,
    AcceleratorProfile,
)

POD_NAME = "o1-b300-calibration"
CONTAINER_DISK_GB = 60   # image + runtime scratch; checkpoint ~6 GB + records
ENV_NAMES = (
    "O1_B200_OUT", "O1_B200_ARTIFACT_SOURCE", "O1_B200_RESULT_DESTINATION",
    "RUNPOD_ALLOW_BILLABLE_MUTATIONS",
    # session facts injected per acquisition (fresh quote):
    "O1_ACQUIRED_PROFILE", "O1_SESSION_AUTHORIZED_SECONDS",
    "O1_HOURLY_RATE_USD", "O1_IMAGE_DIGEST",
    # durable-store credential for continuous row/checkpoint sync (secret;
    # never rendered into the canonical deployment)
    "HF_TOKEN",
)

#: Env values that legitimately differ per acquisition (fresh quote) or are
#: secrets.  These render as empty strings and are compared BY NAME only.
LAUNCH_VARIABLE_ENV_NAMES = (
    "O1_ACQUIRED_PROFILE", "O1_SESSION_AUTHORIZED_SECONDS",
    "O1_HOURLY_RATE_USD", "RUNPOD_ALLOW_BILLABLE_MUTATIONS",
    "HF_TOKEN", "O1_LAUNCH_NONCE",
)

#: Env values that ARE deployment identity: they decide what the pod fetches
#: and where it publishes, so they are frozen into the canonical rendering
#: and compared EXACTLY.  ("deployment identity" that excluded artifact
#: provenance would let an authorized launch read and publish anywhere.)
IDENTITY_ENV_NAMES = tuple(n for n in ENV_NAMES
                           if n not in LAUNCH_VARIABLE_ENV_NAMES)
START_ARGS = ("/opt/o1_b200/o1_b200/deploy/start_b300.sh")

DEPLOYMENT_SCHEMA = "o1b300.runpod_deployment_spec.v2"

_DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class PodRequestError(RuntimeError):
    pass


def build_pod_request(*, profile: AcceleratorProfile, image_digest_ref: str,
                      datacenter_id: str,
                      env_values: dict | None = None) -> CreatePodRequestModel:
    if not _DIGEST_RE.match(image_digest_ref):
        raise PodRequestError(
            f"image must be an immutable digest reference "
            f"(repo@sha256:<64 hex>), got {image_digest_ref!r} — mutable "
            f"tags are refused")
    env = {name: "" for name in ENV_NAMES}
    if env_values:
        unknown = sorted(set(env_values) - set(ENV_NAMES))
        if unknown:
            raise PodRequestError(f"unknown env names {unknown}")
        env.update({k: str(v) for k, v in env_values.items()})
    req = CreatePodRequestModel(
        name=POD_NAME,
        image=image_digest_ref,
        cloud=CLOUD,
        gpu_type_id=profile.gpu_type_id,
        gpu_count=GPU_COUNT,
        container_disk_gb=CONTAINER_DISK_GB,
        env=env,
        purchase_mode=PURCHASE_MODE,
        ports=(),              # no public application ports, no SSH, no Jupyter
        args=START_ARGS,
        datacenter_ids=(datacenter_id,),
    )
    req.validate()
    return req


def _behavior() -> dict:
    return {
        "start_command": START_ARGS,
        "starts": "sealed zero-touch state machine only",
        "purchase_mode": PURCHASE_MODE,
        "will_not_run_real_calibration_until": [
            "hardware validation", "non-O1 equivalence",
            "bounded benchmark", "backend selection",
            "replacement precommit", "external commit verification",
            "affordability gate"],
        "preemption": "eviction is normal: committed rows/checkpoints sync "
                      "continuously to the result destination; the evicted "
                      "remnant is terminated and a fresh qualifying pod "
                      "resumes from durable state (bounded by budget and "
                      "max_pod_creations)",
        "health_check": "state machine writes /outputs/FINAL_STATUS.json; "
                        "lifecycle controller polls Pod status + logs",
        "shutdown": "terminate (not stop) immediately after verified "
                    "result transfer; stop is never a final state",
        "artifact_locations": "/artifacts (hash-verified before model load)",
        "result_destination": "/outputs -> continuous sync + packaged "
                              "archive -> verified download -> terminate",
        "remote_shell": "none; no ports exposed",
        "min_cuda_version": MIN_CUDA_VERSION,
    }


def render_canonical_deployment(reqs_by_profile: dict, identities: dict) -> dict:
    """The exact frozen deployment specification covering every profile.

    reqs_by_profile: {profile_key: CreatePodRequestModel} — must cover the
    complete frozen preference order.  The sha256 of this rendering is what
    the rental authorization commits to (deployment_spec_sha256): the
    session can deploy the primary or the explicit fallback, and nothing
    else.
    """
    want = [p.key for p in PROFILE_PREFERENCE]
    if sorted(reqs_by_profile) != sorted(want):
        raise PodRequestError(
            f"deployment rendering must cover exactly the frozen profiles "
            f"{want}, got {sorted(reqs_by_profile)}")
    bodies = {}
    for key, req in reqs_by_profile.items():
        raw = req.to_json()
        # Check the RAW body, before identity_body() blanks anything: a
        # guard applied after blanking can never fire, so a newly added
        # secret that someone forgot to list in LAUNCH_VARIABLE_ENV_NAMES
        # would be frozen in cleartext into a document that is written to
        # disk and covered by the authorization hash.
        leaked = sorted(name for name, value in (raw.get("env") or {}).items()
                        if value and name not in IDENTITY_ENV_NAMES)
        if leaked:
            raise PodRequestError(
                f"canonical rendering must not contain non-identity env "
                f"VALUES: {leaked}; classify each name as identity-bearing "
                f"or launch-variable before rendering")
        bodies[key] = identity_body(raw)
    doc = {
        "schema": DEPLOYMENT_SCHEMA,
        "profile_preference": want,
        "create_pod_bodies": bodies,
        "behavior": _behavior(),
        "identities": dict(identities),
    }
    doc["request_sha256"] = canonical_sha256(DEPLOYMENT_SCHEMA, doc)
    return doc


def identity_body(body: dict) -> dict:
    """Project a createPod body onto its DEPLOYMENT-IDENTITY fields.

    Dropped: dataCenterIds (a launch-time fact from the fresh quote) and the
    VALUES of launch-variable/secret env names (kept as empty strings so the
    env NAME SET stays part of the identity).  Everything else — image
    digest, gpu id, count, cloud, purchase mode, disk, args, ports, name,
    and the identity-bearing env values (artifact source, result
    destination, output dir, image digest) — is identity.
    """
    out = {k: v for k, v in body.items() if k != "dataCenterIds"}
    env = dict(body.get("env") or {})
    for name in LAUNCH_VARIABLE_ENV_NAMES:
        if name in env:
            env[name] = ""
    out["env"] = env
    return out


def verify_deployment_rendering(rendered: dict) -> str:
    """Re-derive and return the rendering's own hash; raise on mismatch.

    The authorization commits to deployment_spec_sha256.  Comparing that
    commitment against ``rendered["request_sha256"]`` alone is comparing it
    against a caller-supplied CLAIM: a doc carrying an arbitrary body plus a
    matching-looking hash field would pass.  Every consumer must therefore
    recompute the hash over the rendering's CONTENTS before trusting it.
    """
    if not isinstance(rendered, dict):
        raise PodRequestError("deployment rendering is not an object")
    claimed = rendered.get("request_sha256")
    body = {k: v for k, v in rendered.items() if k != "request_sha256"}
    if body.get("schema") != DEPLOYMENT_SCHEMA:
        raise PodRequestError(
            f"deployment rendering schema {body.get('schema')!r} != "
            f"{DEPLOYMENT_SCHEMA!r}")
    actual = canonical_sha256(DEPLOYMENT_SCHEMA, body)
    if claimed != actual:
        raise PodRequestError(
            f"deployment rendering hash does not match its own contents "
            f"(claimed {str(claimed)[:16]}, actual {actual[:16]}); refusing")
    return actual


def render_canonical_pod_request(req: CreatePodRequestModel,
                                 identities: dict) -> dict:
    """Single-profile rendering kept for dry-run tooling; the authorization
    binds render_canonical_deployment (all profiles), not this."""
    body = req.to_json()
    if set(body["env"].values()) - {""}:
        raise PodRequestError(
            "canonical rendering must not contain secret env VALUES")
    doc = {
        "schema": "o1b300.runpod_pod_request.v2",
        "create_pod_body": body,
        "behavior": _behavior(),
        "identities": dict(identities),
    }
    doc["request_sha256"] = canonical_sha256(
        "o1b300.runpod_pod_request.v2", doc)
    return doc


def write_template(path: str, identities: dict) -> dict:
    """RUNPOD_POD_REQUEST.template.json with unresolved runtime fields."""
    bodies = {}
    for p in PROFILE_PREFERENCE:
        bodies[p.key] = {
            "name": POD_NAME,
            "image": "UNRESOLVED@sha256:" + "0" * 64 + "  (replace with the "
                     "built image digest; mutable tags refused)",
            "cloud": CLOUD,
            "gpu": {"id": p.gpu_type_id, "count": GPU_COUNT},
            "purchaseMode": PURCHASE_MODE,
            "disk": CONTAINER_DISK_GB,
            "env": {name: "" for name in ENV_NAMES},
            "ports": [],
            "args": START_ARGS,
            "dataCenterIds": ["UNRESOLVED (from the fresh pre-launch quote)"],
            "bidPerGpu": "UNRESOLVED (live secure spot rate at launch)",
        }
    doc = {
        "schema": "o1b300.runpod_pod_request.template.v2",
        "status": "TEMPLATE — the authorized session re-renders this with the "
                  "live datacenter, live spot bid and the immutable image "
                  "digest, then binds the all-profile deployment hash into "
                  "B300_RENTAL_AUTHORIZATION.json",
        "profile_preference": [p.key for p in PROFILE_PREFERENCE],
        "create_pod_bodies": bodies,
        "identities": dict(identities),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return doc
