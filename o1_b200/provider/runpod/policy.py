"""Frozen RunPod product policy for the O1 B200 session.

Provider choice is frozen: RunPod / Pod / Secure Cloud / on-demand /
exactly one NVIDIA B200 / non-interruptible / no serverless / no Community /
no savings plan.  Maximum accepted quoted GPU rate: USD 5.89/hour.
All monetary values are decimal.Decimal — never binary floats.
"""
from __future__ import annotations

from decimal import Decimal

PROVIDER = "runpod"
PRODUCT = "POD"
CLOUD = "SECURE"
GPU_COUNT = 1

# The intended single-B200 product identity. The catalog id must equal this
# EXACTLY; no fuzzy matching and no substitution. If RunPod renames the
# product, preflight reports the live catalog and THIS constant must be
# changed deliberately in a reviewed commit.
EXPECTED_GPU_ID = "NVIDIA B200"
EXPECTED_MANUFACTURER = "NVIDIA"
# B200 carries 180 GB HBM3e (Runpod catalogs list usable GiB); anything
# below this is not the intended part.
MIN_MEMORY_GB = 170

# Substitutes that must NEVER be accepted silently.
FORBIDDEN_SUBSTITUTES = (
    "H100", "H200", "B300", "RTX PRO", "RTX Pro", "GB200", "A100",
)

MAX_GPU_HOURLY_USD = Decimal("5.89")

# CUDA compatibility floor for the sealed stack (torch cu128 / sm_100).
MIN_CUDA_VERSION = "12.8"

# Quote validity: a stored quote older than this cannot authorize
# provisioning; re-quote immediately before any authorized launch.
QUOTE_VALIDITY_SECONDS = 15 * 60

# Budget (mirrors policies/B200_BUDGET_POLICY.template.json; decimal).
TOTAL_AUTHORIZED_USD = Decimal("45.00")
MAX_COMPUTE_USD = Decimal("40.00")
RESERVED_NONCOMPUTE_USD = Decimal("5.00")

SOFT_STOP_FRACTION = Decimal("0.95")


class PolicyViolation(RuntimeError):
    pass


def assert_no_forbidden_substitute(gpu_id: str, gpu_name: str) -> None:
    hay = f"{gpu_id} {gpu_name}".upper()
    if "B200" not in hay:
        raise PolicyViolation(
            f"offer {gpu_id!r} is not the intended B200 product")
    for bad in FORBIDDEN_SUBSTITUTES:
        if bad.upper() in hay:
            raise PolicyViolation(
                f"forbidden substitute {bad!r} offered as {gpu_id!r}; refusing")
