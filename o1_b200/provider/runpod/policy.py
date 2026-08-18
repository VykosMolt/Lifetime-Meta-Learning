"""Frozen RunPod product policy for the O1 accelerator session (v2, B300).

Provider choice is frozen: RunPod / Pod / Secure Cloud / INTERRUPTIBLE
(spot) capacity / exactly one GPU / no serverless / no Community / no
savings plan / no cluster.

Accelerator profiles are preference-ordered:

  PRIMARY : NVIDIA B300 SXM6 AC  (Blackwell Ultra, sm_103, 288 GB HBM3e)
  FALLBACK: NVIDIA B200          (Blackwell,       sm_100, 180 GB HBM3e)

The B200 fallback is deliberate and explicit (operator-directed
2026-08-17), never silent: every acquisition records which profile was
used and, for the fallback, the precise reason the primary was refused.
The complete software stack was originally sized and validated for the
B200/180 GB part, and every capacity-dependent execution parameter is
derived by the on-pod mechanical benchmark, so the fallback changes no
scientific configuration.

There is NO static price assumption (the old USD 5.89/h B200 cap is
obsolete).  The live pre-launch quote is authoritative; a quote is refused
mechanically when the committed compute budget cannot buy a minimum viable
session at the quoted all-in rate.

All monetary values are decimal.Decimal — never binary floats.
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal

PROVIDER = "runpod"
PRODUCT = "POD"
CLOUD = "SECURE"
GPU_COUNT = 1

# Purchase mode is frozen: RunPod interruptible (spot) capacity.  The REST
# v2 contract exposes no interruptible surface; acquisition goes through the
# pinned GraphQL surface (graphql_spot.py) while all other lifecycle
# operations stay on REST v2.
PURCHASE_MODE = "INTERRUPTIBLE"


@dataclasses.dataclass(frozen=True)
class AcceleratorProfile:
    key: str                  # short stable key used in reports/configs
    role: str                 # PRIMARY | FALLBACK
    gpu_type_id: str          # EXACT catalog id; no fuzzy matching
    display_name: str
    min_memory_gb: int        # floor on the catalog "memory" (GB) field
    compute_capability: str   # CUDA compute capability the hardware gate expects
    sm_arch: str              # native SASS architecture torch must carry
    hbm_class: str


# Catalog ids read from the live RunPod API v2 catalog on 2026-08-17; the id
# must equal the catalog entry EXACTLY.  If RunPod renames a product,
# preflight reports the live catalog and THIS constant must be changed
# deliberately in a reviewed commit.
PROFILE_B300 = AcceleratorProfile(
    key="B300", role="PRIMARY",
    gpu_type_id="NVIDIA B300 SXM6 AC", display_name="B300",
    min_memory_gb=275, compute_capability="10.3", sm_arch="sm_103",
    hbm_class="HBM3e 288GB")

PROFILE_B200 = AcceleratorProfile(
    key="B200", role="FALLBACK",
    gpu_type_id="NVIDIA B200", display_name="B200",
    min_memory_gb=170, compute_capability="10.0", sm_arch="sm_100",
    hbm_class="HBM3e 180GB")

PROFILE_PREFERENCE: tuple[AcceleratorProfile, ...] = (PROFILE_B300,
                                                      PROFILE_B200)
PROFILES_BY_KEY = {p.key: p for p in PROFILE_PREFERENCE}

# Substitutes that must NEVER be accepted, under either profile.
FORBIDDEN_SUBSTITUTES = (
    "H100", "H200", "GB200", "GB300", "A100", "MI300", "RTX PRO", "RTX Pro",
    "L40", "L4", "V100", "A40", "A6000", "5090", "4090",
)

# CUDA compatibility floor for the rebuilt cu130 stack.  Live catalog
# 2026-08-17: both B200 and B300 hosts report CUDA 13.0/13.2 only; 12.8 is
# no longer offered on either.
MIN_CUDA_VERSION = "13.0"

# Quote validity: a stored quote older than this cannot authorize
# provisioning; re-quote immediately before any authorized launch or
# reacquisition.
QUOTE_VALIDITY_SECONDS = 15 * 60

# Budget (mirrors policies/BUDGET_POLICY; decimal).
TOTAL_AUTHORIZED_USD = Decimal("45.00")
MAX_COMPUTE_USD = Decimal("40.00")
RESERVED_NONCOMPUTE_USD = Decimal("5.00")

SOFT_STOP_FRACTION = Decimal("0.95")

# Mechanical rate viability: a quote is refused when MAX_COMPUTE_USD cannot
# buy at least this much runtime at the quoted all-in rate.  This replaces
# the obsolete static per-hour price cap with a rule derived from the
# committed budget, not from a guessed market price.
MIN_VIABLE_SESSION_SECONDS = 2 * 3600

# Preemption/reacquisition bounds: eviction of interruptible capacity is
# normal infrastructure behavior.  Reacquisition re-runs the full quote,
# authorization-binding, hardware and environment gates, and is bounded by
# BOTH the hard dollar budget and this attempt ceiling.  Additionally, a
# repeat failure with zero synced progress is treated as a container defect
# (loud abort), never retried as an eviction.
MAX_POD_ACQUISITIONS = 4


class PolicyViolation(RuntimeError):
    pass


def assert_no_forbidden_substitute(profile: AcceleratorProfile,
                                   gpu_id: str, gpu_name: str) -> None:
    hay = f"{gpu_id} {gpu_name}".upper()
    if profile.display_name.upper() not in hay:
        raise PolicyViolation(
            f"offer {gpu_id!r} is not the intended {profile.display_name} "
            f"product for profile {profile.key}")
    for bad in FORBIDDEN_SUBSTITUTES:
        if bad.upper() in hay:
            raise PolicyViolation(
                f"forbidden substitute {bad!r} offered as {gpu_id!r}; refusing")
    other = [p for p in PROFILE_PREFERENCE if p.key != profile.key]
    for p in other:
        if p.display_name.upper() in hay:
            raise PolicyViolation(
                f"offer {gpu_id!r} mixes profile identities "
                f"({profile.key} vs {p.key}); refusing")
