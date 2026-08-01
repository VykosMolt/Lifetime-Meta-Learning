"""Typed request/response models for the pinned RunPod API v2 contract.

Strict, fail-closed parsing: unknown REQUIRED semantics (missing required
fields, unknown lifecycle states, unknown enum members on safety-relevant
enums, absent pricing) raise SchemaIncompatibility.  Unknown DESCRIPTIVE
fields are retained in ``extra`` for forward compatibility but never feed a
safety decision.  No permissive dictionary access anywhere downstream: the
adapter consumes only these models.
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

POD_STATUSES = ("PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR",
                "TERMINATED")
CLOUDS = ("SECURE", "COMMUNITY")
AVAILABILITY_LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
POD_ACTIONS = ("start", "stop", "restart", "terminate")


class SchemaIncompatibility(RuntimeError):
    """The live response does not satisfy the pinned v2 contract."""


def _require(obj: dict, field: str, kind, ctx: str):
    if field not in obj:
        raise SchemaIncompatibility(f"{ctx}: required field {field!r} missing")
    v = obj[field]
    if kind is Decimal:
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise SchemaIncompatibility(f"{ctx}.{field}: not numeric: {type(v).__name__}")
        try:
            d = Decimal(str(v))
        except ArithmeticError as exc:
            raise SchemaIncompatibility(f"{ctx}.{field}: bad number {v!r}") from exc
        if not d.is_finite():
            raise SchemaIncompatibility(f"{ctx}.{field}: non-finite price")
        return d
    if kind is int and isinstance(v, bool):
        raise SchemaIncompatibility(f"{ctx}.{field}: bool where int expected")
    if not isinstance(v, kind):
        raise SchemaIncompatibility(
            f"{ctx}.{field}: expected {kind}, got {type(v).__name__}")
    return v


def _enum(obj: dict, field: str, allowed: tuple, ctx: str, *, required=True):
    if field not in obj:
        if required:
            raise SchemaIncompatibility(f"{ctx}: required enum {field!r} missing")
        return None
    v = obj[field]
    if v not in allowed:
        raise SchemaIncompatibility(
            f"{ctx}.{field}: unknown enum value {v!r}; pinned members are "
            f"{allowed} — treat as contract drift, fail closed")
    return v


@dataclasses.dataclass(frozen=True)
class GpuPrice:
    secure: Decimal
    community: Decimal


@dataclasses.dataclass(frozen=True)
class DataCenterAvailability:
    datacenter_id: str
    availability: str            # AvailabilityLevel
    extra: dict


@dataclasses.dataclass(frozen=True)
class GpuTypeModel:
    id: str
    name: str
    manufacturer: str
    memory_gb: int
    secure: bool
    community: bool
    price: GpuPrice
    max_count_secure: int
    availability: str | None     # AvailabilityLevel, present with include=AVAILABILITY
    datacenters: tuple           # tuple[DataCenterAvailability, ...]
    extra: dict

    @classmethod
    def parse(cls, obj: Any) -> "GpuTypeModel":
        ctx = "GpuType"
        if not isinstance(obj, dict):
            raise SchemaIncompatibility(f"{ctx}: not an object")
        price_obj = _require(obj, "price", dict, ctx)
        price = GpuPrice(
            secure=_require(price_obj, "secure", Decimal, f"{ctx}.price"),
            community=_require(price_obj, "community", Decimal, f"{ctx}.price"))
        max_count = _require(obj, "maxCount", dict, ctx)
        dcs = []
        for dc in obj.get("dataCenters") or []:
            dcs.append(DataCenterAvailability(
                datacenter_id=_require(dc, "id", str, f"{ctx}.dataCenters[]"),
                availability=_enum(dc, "availability", AVAILABILITY_LEVELS,
                                   f"{ctx}.dataCenters[]"),
                extra={k: v for k, v in dc.items()
                       if k not in ("id", "availability")}))
        known = {"id", "name", "manufacturer", "memory", "secure", "community",
                 "price", "maxCount", "availability", "dataCenters", "pool"}
        return cls(
            id=_require(obj, "id", str, ctx),
            name=_require(obj, "name", str, ctx),
            manufacturer=_require(obj, "manufacturer", str, ctx),
            memory_gb=_require(obj, "memory", int, ctx),
            secure=_require(obj, "secure", bool, ctx),
            community=_require(obj, "community", bool, ctx),
            price=price,
            max_count_secure=_require(max_count, "secure", int, f"{ctx}.maxCount"),
            availability=_enum(obj, "availability", AVAILABILITY_LEVELS, ctx,
                               required=False),
            datacenters=tuple(dcs),
            extra={k: v for k, v in obj.items() if k not in known},
        )


@dataclasses.dataclass(frozen=True)
class PodModel:
    id: str
    name: str
    status: str                  # PodStatus (pinned enum, fail closed)
    cloud: str | None
    gpu_type_id: str | None
    gpu_count: int | None
    image: str | None
    cost_per_hour: Decimal | None
    created_at: str | None
    started_at: str | None
    datacenter_id: str | None
    extra: dict

    @classmethod
    def parse(cls, obj: Any) -> "PodModel":
        ctx = "Pod"
        if not isinstance(obj, dict):
            raise SchemaIncompatibility(f"{ctx}: not an object")
        status = _enum(obj, "status", POD_STATUSES, ctx)
        gpu = obj.get("gpu")
        gpu_id = gpu_count = None
        if gpu is not None:
            if not isinstance(gpu, dict) or "id" not in gpu:
                raise SchemaIncompatibility(f"{ctx}.gpu: malformed GpuConfig")
            gpu_id = gpu["id"]
            gpu_count = int(gpu.get("count", 1))
        cloud = _enum(obj, "cloud", CLOUDS, ctx, required=False)
        cost = None
        if "cost" in obj and obj["cost"] is not None:
            cost = _require(obj, "cost", Decimal, ctx)
        known = {"id", "name", "status", "cloud", "gpu", "image", "cost",
                 "createdAt", "startedAt", "dataCenterId"}
        return cls(
            id=_require(obj, "id", str, ctx),
            name=_require(obj, "name", str, ctx),
            status=status,
            cloud=cloud,
            gpu_type_id=gpu_id,
            gpu_count=gpu_count,
            image=obj.get("image"),
            cost_per_hour=cost,
            created_at=obj.get("createdAt"),
            started_at=obj.get("startedAt"),
            datacenter_id=obj.get("dataCenterId"),
            extra={k: v for k, v in obj.items() if k not in known},
        )


@dataclasses.dataclass(frozen=True)
class CreatePodRequestModel:
    """Typed createPod body per the pinned CreatePodRequest schema."""
    name: str
    image: str
    cloud: str
    gpu_type_id: str
    gpu_count: int
    container_disk_gb: int
    env: dict
    ports: tuple = ()
    args: str | None = None
    datacenter_ids: tuple = ()

    def to_json(self) -> dict:
        body = {
            "name": self.name,
            "image": self.image,
            "cloud": self.cloud,
            "gpu": {"id": self.gpu_type_id, "count": int(self.gpu_count)},
            "disk": int(self.container_disk_gb),
            "env": dict(self.env),
        }
        if self.ports:
            body["ports"] = list(self.ports)
        if self.args is not None:
            body["args"] = self.args
        if self.datacenter_ids:
            body["dataCenterIds"] = list(self.datacenter_ids)
        return body

    def validate(self) -> None:
        if self.cloud != "SECURE":
            raise SchemaIncompatibility("pod request must pin cloud=SECURE")
        if self.gpu_count != 1:
            raise SchemaIncompatibility("pod request must pin exactly 1 GPU")
        if "@sha256:" not in self.image:
            raise ValueError(
                "image must be pinned by immutable digest (…@sha256:…), "
                "never a mutable tag")
        if not self.name:
            raise ValueError("pod name required")
        if self.container_disk_gb < 1:
            raise ValueError("container disk must be >= 1 GB")
        for key, value in self.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("env must be str->str")


def parse_error_response(obj: Any, http_status: int) -> str:
    if isinstance(obj, dict) and all(k in obj for k in ("title", "status", "detail")):
        return f"{obj['title']} ({obj['status']}): {obj['detail']}"
    return f"HTTP {http_status}: unrecognized error body (pinned ErrorResponse absent)"
