"""Canonical Pod request: construction and rendering (never submitted here).

Freezes: name, image by immutable digest, Secure cloud, exactly one B200,
non-interruptible on-demand, container disk, no persistent/network storage
(results return via the output pipeline; minimum storage), no exposed ports
(no Jupyter, no HTTP service, no public SSH — the zero-touch runner needs
none; emergency shell access is intentionally ABSENT rather than disabled),
environment-variable NAMES only (secret values injected at launch, never
frozen into the template), the start command that launches the sealed
zero-touch state machine, health/shutdown behavior, artifact locations,
result destination, and package/container/budget identities.
"""
from __future__ import annotations

import json
import re

from .identityutil import canonical_sha256
from .models import CreatePodRequestModel
from .policy import CLOUD, EXPECTED_GPU_ID, GPU_COUNT

POD_NAME = "o1-b200-calibration"
CONTAINER_DISK_GB = 60   # image + runtime scratch; checkpoint ~6 GB + records
ENV_NAMES = (
    # names only; values are provided by the authorized launcher at runtime
    "O1_B200_OUT", "O1_B200_ARTIFACT_SOURCE", "O1_B200_RESULT_DESTINATION",
    "RUNPOD_ALLOW_BILLABLE_MUTATIONS",
)
START_ARGS = ("/opt/o1_b200/o1_b200/deploy/start_b200.sh")

_DIGEST_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class PodRequestError(RuntimeError):
    pass


def build_pod_request(*, image_digest_ref: str, datacenter_id: str,
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
        gpu_type_id=EXPECTED_GPU_ID,
        gpu_count=GPU_COUNT,
        container_disk_gb=CONTAINER_DISK_GB,
        env=env,
        ports=(),              # no public application ports, no SSH, no Jupyter
        args=START_ARGS,
        datacenter_ids=(datacenter_id,),
    )
    req.validate()
    return req


def render_canonical_pod_request(req: CreatePodRequestModel,
                                 identities: dict) -> dict:
    """The exact frozen deployment specification (dry-run rendering).

    identities: package/container/budget identity hashes bound alongside the
    raw createPod body; the sha256 of this rendering is what the rental
    authorization file must commit to (deployment_spec_sha256).
    """
    body = req.to_json()
    if set(body["env"].values()) - {""}:
        raise PodRequestError(
            "canonical rendering must not contain secret env VALUES")
    doc = {
        "schema": "o1b200.runpod_pod_request.v1",
        "create_pod_body": body,
        "behavior": {
            "start_command": START_ARGS,
            "starts": "sealed zero-touch state machine only",
            "will_not_run_real_calibration_until": [
                "hardware validation", "non-O1 equivalence",
                "bounded benchmark", "backend selection",
                "replacement precommit", "external commit verification",
                "affordability gate"],
            "health_check": "state machine writes /outputs/FINAL_STATUS.json; "
                            "lifecycle controller polls Pod status + logs",
            "shutdown": "terminate (not stop) immediately after verified "
                        "result transfer; stop is never a final state",
            "artifact_locations": "/artifacts (hash-verified before model load)",
            "result_destination": "/outputs -> packaged archive -> verified "
                                  "download -> terminate",
            "remote_shell": "none; no ports exposed",
        },
        "identities": dict(identities),
    }
    doc["request_sha256"] = canonical_sha256("o1b200.runpod_pod_request.v1", doc)
    return doc


def write_template(path: str, identities: dict) -> dict:
    """RUNPOD_B200_POD_REQUEST.template.json with unresolved runtime fields."""
    template_req = {
        "name": POD_NAME,
        "image": "UNRESOLVED@sha256:" + "0" * 64 + "  (replace with the built "
                 "image digest; mutable tags refused)",
        "cloud": CLOUD,
        "gpu": {"id": EXPECTED_GPU_ID, "count": GPU_COUNT},
        "disk": CONTAINER_DISK_GB,
        "env": {name: "" for name in ENV_NAMES},
        "ports": [],
        "args": START_ARGS,
        "dataCenterIds": ["UNRESOLVED (from the fresh pre-launch quote)"],
    }
    doc = {
        "schema": "o1b200.runpod_pod_request.template.v1",
        "status": "TEMPLATE — the authorized session re-renders this with the "
                  "live datacenter and the immutable image digest, then binds "
                  "its hash into B200_RENTAL_AUTHORIZATION.json",
        "create_pod_body": template_req,
        "identities": dict(identities),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return doc
