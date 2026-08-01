"""Catalog / availability / quote validation for the single-B200 Secure Pod.

The offer must be: product Pod, Secure cloud, count 1, GPU identity exactly
the intended B200, secure availability not NONE, a compatible datacenter,
finite quoted rate <= USD 5.89/h, no cluster minimum, no extra GPU, no
interruptible capacity (the pinned v2 contract has no interruptible surface;
its appearance is schema drift and fails closed upstream).
"""
from __future__ import annotations

import json
import time
from decimal import Decimal

from .billing import validate_gpu_rate
from .identityutil import canonical_sha256, utcnow_iso
from .models import GpuTypeModel, SchemaIncompatibility
from .policy import (
    CLOUD, EXPECTED_GPU_ID, EXPECTED_MANUFACTURER, GPU_COUNT,
    MIN_CUDA_VERSION, MIN_MEMORY_GB, PolicyViolation, PRODUCT,
    QUOTE_VALIDITY_SECONDS, assert_no_forbidden_substitute,
)

CATALOG_PATH = (f"/v2/catalog/gpus?include=AVAILABILITY&product={PRODUCT}"
                f"&count={GPU_COUNT}&cloud={CLOUD}"
                f"&minCudaVersion={MIN_CUDA_VERSION}")


class QuoteError(RuntimeError):
    pass


def find_b200(gpu_types: list[GpuTypeModel]) -> GpuTypeModel:
    exact = [g for g in gpu_types if g.id == EXPECTED_GPU_ID]
    if not exact:
        b200ish = [g.id for g in gpu_types if "B200" in g.id.upper()]
        raise QuoteError(
            f"no catalog entry with the exact intended id "
            f"{EXPECTED_GPU_ID!r}; B200-like ids present: {b200ish or 'none'}"
            f" — no substitution is permitted")
    if len(exact) > 1:
        raise QuoteError(f"catalog returned {len(exact)} entries for the id")
    return exact[0]


def validate_offer(gpu: GpuTypeModel) -> dict:
    assert_no_forbidden_substitute(gpu.id, gpu.name)
    if gpu.manufacturer != EXPECTED_MANUFACTURER:
        raise PolicyViolation(f"manufacturer {gpu.manufacturer!r} != NVIDIA")
    if gpu.memory_gb < MIN_MEMORY_GB:
        raise PolicyViolation(
            f"memory {gpu.memory_gb} GB below the B200 floor {MIN_MEMORY_GB}")
    if not gpu.secure:
        raise PolicyViolation(
            "offer is not available on Secure Cloud (Community-only offers "
            "are refused)")
    if gpu.availability is None:
        raise QuoteError("availability not present; query with "
                         "include=AVAILABILITY")
    if gpu.availability == "NONE":
        raise QuoteError("secure B200 availability is NONE right now")
    if gpu.max_count_secure < 1:
        raise PolicyViolation("secure maxCount < 1: no single-GPU offer")
    rate = validate_gpu_rate(gpu.price.secure)
    compatible_dcs = [d.datacenter_id for d in gpu.datacenters
                      if d.availability != "NONE"]
    if not compatible_dcs:
        raise QuoteError("no datacenter advertises non-NONE B200 availability")
    return {"rate": rate, "datacenters": compatible_dcs}


def build_quote(gpu: GpuTypeModel, *, datacenter_id: str,
                disk_gb: int, disk_hourly_usd, schema_sha256: str,
                adapter_commit: str, now=time.time) -> dict:
    v = validate_offer(gpu)
    if datacenter_id not in v["datacenters"]:
        raise QuoteError(
            f"datacenter {datacenter_id} does not advertise availability")
    rate = v["rate"]
    disk_rate = Decimal(str(disk_hourly_usd))
    total = (rate + disk_rate).quantize(Decimal("0.0001"))
    quote = {
        "schema": "o1b200.runpod_quote.v1",
        "retrieved_utc": utcnow_iso(now),
        "retrieved_monotonic_epoch": float(now()),
        "expiry_policy": f"invalid {QUOTE_VALIDITY_SECONDS}s after retrieval; "
                         f"must be re-quoted immediately before any "
                         f"authorized launch",
        "validity_seconds": QUOTE_VALIDITY_SECONDS,
        "gpu_type_id": gpu.id,
        "display_name": gpu.name,
        "memory_gb": gpu.memory_gb,
        "cloud": CLOUD,
        "product": PRODUCT,
        "gpu_count": GPU_COUNT,
        "interruptible": False,
        "datacenter_id": datacenter_id,
        "hourly_gpu_rate_usd": str(rate),
        "expected_disk_hourly_usd": str(disk_rate),
        "requested_disk_gb": int(disk_gb),
        "total_projected_hourly_usd": str(total),
        "api_schema_sha256": schema_sha256,
        "adapter_commit": adapter_commit,
    }
    quote["quote_sha256"] = canonical_sha256("o1b200.runpod_quote.v1", quote)
    return quote


def check_quote_fresh(quote: dict, now=time.time) -> None:
    age = float(now()) - float(quote["retrieved_monotonic_epoch"])
    if age > float(quote.get("validity_seconds", QUOTE_VALIDITY_SECONDS)):
        raise QuoteError(
            f"quote expired ({age:.0f}s old > "
            f"{quote.get('validity_seconds')}s); re-quote before provisioning "
            f"— cached prices are never used")


def parse_catalog(payload) -> list[GpuTypeModel]:
    if isinstance(payload, dict):
        items = payload.get("gpus") or payload.get("data") or payload.get("items")
        if items is None:
            raise SchemaIncompatibility(
                "catalog response object carries no recognized list field")
    elif isinstance(payload, list):
        items = payload
    else:
        raise SchemaIncompatibility("catalog response is neither list nor object")
    return [GpuTypeModel.parse(x) for x in items]
