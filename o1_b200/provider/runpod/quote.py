"""Catalog / availability / spot-quote validation for the single-GPU
INTERRUPTIBLE Secure Pod (B300 primary, B200 explicit fallback).

An offer must be: product Pod, Secure cloud, count 1, GPU identity exactly
one of the preference-ordered profiles, secure availability not NONE, a
compatible datacenter, and a live GraphQL spot offer with a finite bid
rate the committed budget can afford (billing.validate_gpu_rate — a
mechanical viability rule, never a static price).

Profile selection is preference-ordered and NEVER silent: when the primary
B300 is refused, the returned selection carries the exact refusal reason,
and that reason is frozen into the quote the authorization binds.
"""
from __future__ import annotations

import time
from decimal import Decimal

from .billing import validate_gpu_rate
from .identityutil import canonical_sha256, utcnow_iso
from .models import GpuTypeModel, SchemaIncompatibility
from .policy import (
    CLOUD, GPU_COUNT, MIN_CUDA_VERSION, PolicyViolation, PRODUCT,
    PROFILE_PREFERENCE, PURCHASE_MODE, QUOTE_VALIDITY_SECONDS,
    AcceleratorProfile, assert_no_forbidden_substitute,
)

CATALOG_PATH = (f"/v2/catalog/gpus?include=AVAILABILITY&product={PRODUCT}"
                f"&count={GPU_COUNT}&cloud={CLOUD}"
                f"&minCudaVersion={MIN_CUDA_VERSION}")

EXPECTED_MANUFACTURER = "NVIDIA"

QUOTE_SCHEMA = "o1b300.runpod_spot_quote.v2"


class QuoteError(RuntimeError):
    pass


def find_profile_offer(profile: AcceleratorProfile,
                       gpu_types: list[GpuTypeModel]) -> GpuTypeModel:
    exact = [g for g in gpu_types if g.id == profile.gpu_type_id]
    if not exact:
        similar = [g.id for g in gpu_types
                   if profile.display_name.upper() in g.id.upper()]
        raise QuoteError(
            f"no catalog entry with the exact intended id "
            f"{profile.gpu_type_id!r}; similar ids present: "
            f"{similar or 'none'} — no substitution is permitted")
    if len(exact) > 1:
        raise QuoteError(f"catalog returned {len(exact)} entries for "
                         f"{profile.gpu_type_id!r}")
    return exact[0]


def validate_offer(profile: AcceleratorProfile, gpu: GpuTypeModel) -> dict:
    assert_no_forbidden_substitute(profile, gpu.id, gpu.name)
    if gpu.manufacturer != EXPECTED_MANUFACTURER:
        raise PolicyViolation(f"manufacturer {gpu.manufacturer!r} != NVIDIA")
    if gpu.memory_gb < profile.min_memory_gb:
        raise PolicyViolation(
            f"memory {gpu.memory_gb} GB below the {profile.display_name} "
            f"floor {profile.min_memory_gb}")
    if not gpu.secure:
        raise PolicyViolation(
            "offer is not available on Secure Cloud (Community-only offers "
            "are refused)")
    if gpu.availability is None:
        raise QuoteError("availability not present; query with "
                         "include=AVAILABILITY")
    if gpu.availability == "NONE":
        raise QuoteError(
            f"secure {profile.display_name} availability is NONE right now")
    if gpu.max_count_secure < 1:
        raise PolicyViolation("secure maxCount < 1: no single-GPU offer")
    compatible_dcs = [d.datacenter_id for d in gpu.datacenters
                      if d.availability != "NONE"]
    if not compatible_dcs:
        raise QuoteError(
            f"no datacenter advertises non-NONE {profile.display_name} "
            f"availability")
    return {"datacenters": compatible_dcs}


def resolve_preference(preference=None) -> tuple:
    """The profile order to try, defaulting to the frozen preference.

    An operator may reorder the frozen profiles (e.g. put the B200 first
    when B300 demand makes it unobtainable), but may NOT introduce a
    profile outside the frozen set, and may not drop one silently: the
    order is a preference, not a new capability — the rental authorization
    already covers every frozen profile's canonical body.
    """
    if not preference:
        return PROFILE_PREFERENCE
    known = {p.key: p for p in PROFILE_PREFERENCE}
    unknown = [k for k in preference if k not in known]
    if unknown:
        raise PolicyViolation(
            f"unknown accelerator profile(s) {unknown}; the frozen set is "
            f"{sorted(known)}")
    duplicates = sorted({k for k in preference if preference.count(k) > 1})
    if duplicates:
        raise PolicyViolation(
            f"duplicate accelerator profile(s) {duplicates} in the "
            f"preference; the frozen set is a set, not a multiset")
    ordered = [known[k] for k in preference]
    ordered += [p for p in PROFILE_PREFERENCE if p.key not in preference]
    return tuple(ordered)


def select_offer(gpu_types: list[GpuTypeModel], preference=None) -> dict:
    """Preference-ordered profile selection with explicit refusal record.

    Returns {"profile", "gpu", "datacenters", "fallback_reason",
    "refusals"}; fallback_reason is None when the first-preference profile
    was selected.  Raises QuoteError with the complete per-profile refusal
    list when no profile qualifies.
    """
    refusals: dict[str, str] = {}
    for profile in resolve_preference(preference):
        try:
            gpu = find_profile_offer(profile, gpu_types)
            v = validate_offer(profile, gpu)
        except (QuoteError, PolicyViolation) as exc:
            refusals[profile.key] = str(exc)
            continue
        # "fallback" means something ahead of this profile was REFUSED —
        # not merely that this profile is the frozen secondary.  With an
        # operator-reordered preference, a first-choice B200 is a
        # deliberate selection, and the report must not call it a fallback.
        fallback_reason = None
        if refusals:
            fallback_reason = (
                "earlier-preference profile(s) refused: "
                + "; ".join(f"{k}: {r}" for k, r in refusals.items()))
        return {"profile": profile, "gpu": gpu,
                "datacenters": v["datacenters"],
                "fallback_reason": fallback_reason,
                "operator_preference_applied": bool(preference),
                "refusals": dict(refusals)}
    raise QuoteError(
        "no accelerator profile qualifies right now — "
        + "; ".join(f"{k}: {r}" for k, r in refusals.items()))


def validate_spot_offer(profile: AcceleratorProfile, spot: dict) -> Decimal:
    """Validate the live GraphQL spot offer and derive the bid rate.

    The bid is the current secure spot rate (never a guessed constant).
    Budget viability is enforced mechanically by validate_gpu_rate.
    """
    if spot.get("gpu_type_id") != profile.gpu_type_id:
        raise SchemaIncompatibility(
            f"spot offer is for {spot.get('gpu_type_id')!r}, expected "
            f"{profile.gpu_type_id!r}")
    if spot.get("secure_cloud") is not True:
        raise PolicyViolation(
            f"{profile.display_name} spot offer is not on Secure Cloud")
    rate = spot.get("secure_spot_usd")
    if rate is None:
        raise QuoteError(
            f"no live secure spot rate for {profile.display_name}; "
            f"interruptible capacity is not purchasable right now")
    bid = validate_gpu_rate(rate)
    minimum = spot.get("minimum_bid_usd")
    if minimum is not None:
        min_bid = Decimal(str(minimum))
        if min_bid > bid:
            # the market minimum exceeds the published spot rate; bid the
            # minimum if — and only if — the budget viability rule accepts it
            bid = validate_gpu_rate(min_bid)
    return bid


def build_quote(selection: dict, spot: dict, *, datacenter_id: str,
                disk_gb: int, disk_hourly_usd, schema_sha256: str,
                adapter_commit: str, now=time.time) -> dict:
    profile: AcceleratorProfile = selection["profile"]
    gpu: GpuTypeModel = selection["gpu"]
    if datacenter_id not in selection["datacenters"]:
        raise QuoteError(
            f"datacenter {datacenter_id} does not advertise availability")
    bid = validate_spot_offer(profile, spot)
    disk_rate = Decimal(str(disk_hourly_usd))
    total = (bid + disk_rate).quantize(Decimal("0.0001"))
    quote = {
        "schema": QUOTE_SCHEMA,
        "retrieved_utc": utcnow_iso(now),
        "retrieved_monotonic_epoch": float(now()),
        "expiry_policy": f"invalid {QUOTE_VALIDITY_SECONDS}s after retrieval; "
                         f"must be re-quoted immediately before any "
                         f"authorized launch or reacquisition",
        "validity_seconds": QUOTE_VALIDITY_SECONDS,
        "profile": profile.key,
        "profile_role": profile.role,
        "operator_preference_applied": selection.get(
            "operator_preference_applied", False),
        "fallback_reason": selection["fallback_reason"],
        "profile_refusals": selection["refusals"],
        "gpu_type_id": gpu.id,
        "display_name": gpu.name,
        "memory_gb": gpu.memory_gb,
        "cloud": CLOUD,
        "product": PRODUCT,
        "gpu_count": GPU_COUNT,
        "purchase_mode": PURCHASE_MODE,
        "interruptible": True,
        "datacenter_id": datacenter_id,
        "bid_per_gpu_usd": str(bid),
        "hourly_gpu_rate_usd": str(bid),
        "secure_list_rate_usd": str(spot.get("secure_list_usd")),
        "spot_stock_status": spot.get("stock_status"),
        "expected_disk_hourly_usd": str(disk_rate),
        "requested_disk_gb": int(disk_gb),
        "total_projected_hourly_usd": str(total),
        "api_schema_sha256": schema_sha256,
        "adapter_commit": adapter_commit,
    }
    quote["quote_sha256"] = canonical_sha256(QUOTE_SCHEMA, quote)
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
