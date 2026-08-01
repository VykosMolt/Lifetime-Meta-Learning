"""Pinned-schema loading and runtime compatibility checking.

The canonical snapshot lives at openapi/runpod_v2_openapi.json with its
SHA-256 sidecar and provenance record.  ``verify_pinned_schema`` refuses to
run when the snapshot bytes do not match the pin.  ``check_live_identity``
compares a live copy of the schema (fetched read-only during preflight)
against the safety-relevant pinned surfaces and fails closed on drift.
"""
from __future__ import annotations

import hashlib
import json
import os

OPENAPI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openapi")
SCHEMA_PATH = os.path.join(OPENAPI_DIR, "runpod_v2_openapi.json")
SHA_PATH = os.path.join(OPENAPI_DIR, "runpod_v2_openapi.sha256")
PROVENANCE_PATH = os.path.join(OPENAPI_DIR, "SCHEMA_PROVENANCE.json")

# Safety-relevant surfaces: presence and exact semantics are required.
REQUIRED_PATHS = {
    "/v2/catalog/gpus": ["get"],
    "/v2/catalog/datacenters": ["get"],
    "/v2/pods": ["get", "post"],
    "/v2/pods/{id}": ["get", "delete"],
    "/v2/pods/{id}/action": ["post"],
    "/v2/pods/{id}/logs": ["get"],
    "/v2/billing/pods": ["get"],
}
REQUIRED_ENUMS = {
    "PodStatus": ["PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR",
                  "TERMINATED"],
    "Cloud": ["SECURE", "COMMUNITY"],
    "AvailabilityLevel": ["NONE", "LOW", "MEDIUM", "HIGH"],
    "PodAction": ["start", "stop", "restart", "terminate"],
}
REQUIRED_GPU_TYPE_FIELDS = ["id", "name", "memory", "secure", "community",
                            "price", "maxCount"]
REQUIRED_CREATE_POD_REQUIRED = ["name", "image"]


class SchemaPinError(RuntimeError):
    pass


def load_pinned_schema() -> dict:
    with open(SHA_PATH, encoding="utf-8") as fh:
        want = fh.read().split()[0]
    with open(SCHEMA_PATH, "rb") as fh:
        raw = fh.read()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SchemaPinError(
            f"pinned OpenAPI snapshot hash mismatch: {got[:16]} != {want[:16]}")
    return json.loads(raw)


def pinned_schema_sha256() -> str:
    with open(SHA_PATH, encoding="utf-8") as fh:
        return fh.read().split()[0]


def _surface_report(schema: dict) -> dict:
    problems = []
    paths = schema.get("paths", {})
    for path, methods in REQUIRED_PATHS.items():
        if path not in paths:
            problems.append(f"missing path {path}")
            continue
        for m in methods:
            if m not in paths[path]:
                problems.append(f"missing operation {m.upper()} {path}")
    comp = schema.get("components", {}).get("schemas", {})
    for enum_name, members in REQUIRED_ENUMS.items():
        live = comp.get(enum_name, {}).get("enum")
        if live is None:
            problems.append(f"missing enum {enum_name}")
        elif list(live) != members:
            problems.append(
                f"enum {enum_name} changed: pinned {members}, live {live}")
    gpu = comp.get("GpuType", {})
    for f in REQUIRED_GPU_TYPE_FIELDS:
        if f not in (gpu.get("required") or []):
            problems.append(f"GpuType.required lost field {f!r}")
    price = gpu.get("properties", {}).get("price", {})
    for f in ("secure", "community"):
        if f not in (price.get("required") or []):
            problems.append(f"GpuType.price lost required {f!r} (pricing field)")
    cpr = comp.get("CreatePodRequest", {})
    req = []
    for part in cpr.get("allOf", []):
        req += part.get("required") or []
    for f in REQUIRED_CREATE_POD_REQUIRED:
        if f not in req:
            problems.append(f"CreatePodRequest lost required {f!r}")
    if "interruptible" in json.dumps(schema):
        problems.append(
            "an 'interruptible' surface appeared in the schema; the pinned "
            "contract has none — semantics must be re-audited before use")
    return {"problems": problems, "compatible": not problems}


def verify_pinned_schema() -> dict:
    """Verify pin bytes + internal surface expectations (local, offline)."""
    schema = load_pinned_schema()
    rep = _surface_report(schema)
    if not rep["compatible"]:
        raise SchemaPinError(
            "pinned snapshot fails its own surface expectations: "
            + "; ".join(rep["problems"]))
    return {"sha256": pinned_schema_sha256(), "surfaces_ok": True}


def check_live_identity(live_schema: dict) -> dict:
    """Compare a live schema copy to the pinned surfaces (fail closed)."""
    rep = _surface_report(live_schema)
    raw = json.dumps(live_schema, sort_keys=True,
                     separators=(",", ":")).encode()
    rep["live_sha256_canonical"] = hashlib.sha256(raw).hexdigest()
    rep["pinned_sha256"] = pinned_schema_sha256()
    rep["byte_identical_to_pin"] = False  # canonicalization differs; surfaces rule
    if not rep["compatible"]:
        raise SchemaPinError(
            "LIVE RunPod v2 schema is incompatible with the pinned contract: "
            + "; ".join(rep["problems"]))
    return rep
