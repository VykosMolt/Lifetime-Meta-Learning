"""Pinned RunPod GraphQL surface for INTERRUPTIBLE (spot) capacity.

The RunPod REST API v2 (pinned OpenAPI 2.0.0) exposes NO interruptible
surface: no spot price in the catalog, no purchase-mode field on
CreatePodRequest, no eviction state.  Spot pods are created only through
the GraphQL API's ``podRentInterruptable`` mutation.  This module pins the
minimal GraphQL contract the session depends on and nothing more:

  read-only (query operations only):
    * gpuTypes { id securePrice communitySpotPrice secureSpotPrice
                 lowestPrice { minimumBidPrice uninterruptablePrice
                               stockStatus } }
    * myself { pods { id name desiredStatus podType costPerHr } }
    * contract introspection (__type on PodRentInterruptableInput)

  mutating (requires the SAME LiveMutationAuthorization as REST mutations):
    * podRentInterruptable(input: {...}) { id name desiredStatus costPerHr }

Discipline mirrors transport.py exactly: read-only enforcement happens
locally BEFORE any socket (the request document must be a named ``query``
operation; anything else raises), ambient operator credentials are only
ever attached to the production endpoint (CredentialIsolationError
otherwise), every mutation rechecks the authorization, mutations are never
blind-retried (AmbiguousMutation -> reconcile), all strings pass through
redaction, and unknown response shapes fail closed.

Everything else about the pod (status polling, logs, billing, stop,
terminate, DELETE) stays on the pinned REST v2 surface: a spot pod is a
pod.  That REST-visibility assumption is exercised against the mock and
recorded as live-hardware-unvalidated until the first real spot pod.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request

from .models import SchemaIncompatibility
from .redaction import redact
from .transport import (
    AmbiguousMutation, CredentialIsolationError, ReadOnlyViolation,
    TransportError, resolve_credential,
)

PRODUCTION_GRAPHQL_URL = "https://api.runpod.io/graphql"
USER_AGENT = "o1-b300-runner/0.3 (pre-rental; read-only unless authorized)"
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_QUERY_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

# The exact input fields of PodRentInterruptableInput this session relies
# on, per the published RunPod GraphQL spec (graphql-spec.runpod.io,
# retrieved 2026-08-17; pinned in openapi/GRAPHQL_SPOT_CONTRACT.json).
#
# Live introspection is DISABLED on api.runpod.io/graphql (Apollo,
# INTROSPECTION_DISABLED — verified 2026-08-17), so the mutation input
# cannot be verified by a read-only call.  Enforcement is therefore:
#   1. this pin against the published spec, and
#   2. Apollo validates every document against the schema BEFORE execution,
#      so a drifted input field fails the authorized launch closed with a
#      validation error — no pod is created and nothing is billed.
# The read-only QUERY surface IS live-verified: executing the pinned
# queries successfully proves every referenced field still exists.
PINNED_RENT_INPUT_FIELDS = (
    "bidPerGpu", "gpuTypeId", "cloudType", "gpuCount", "containerDiskInGb",
    "env", "imageName", "dockerArgs", "dataCenterId", "ports", "name",
    "minCudaVersion", "terminateAfter", "startSsh", "startJupyter",
)
PINNED_GPU_TYPE_FIELDS = (
    "id", "securePrice", "communityPrice", "secureSpotPrice",
    "communitySpotPrice", "memoryInGb", "secureCloud",
)

_QUERY_RE = re.compile(r"^\s*query\b")

QUERY_SPOT_PRICING = """
query SpotPricing {
  gpuTypes {
    id
    memoryInGb
    secureCloud
    securePrice
    communityPrice
    secureSpotPrice
    communitySpotPrice
    lowestPrice(input: {gpuCount: 1, secureCloud: true}) {
      minimumBidPrice
      uninterruptablePrice
      stockStatus
    }
  }
}
"""

QUERY_MYSELF_PODS = """
query MyselfPods {
  myself {
    pods { id name desiredStatus podType costPerHr }
  }
}
"""


MUTATION_RENT_INTERRUPTABLE = """
mutation RentInterruptable($input: PodRentInterruptableInput!) {
  podRentInterruptable(input: $input) {
    id
    name
    desiredStatus
    costPerHr
  }
}
"""


class GraphQlError(TransportError):
    pass


class RunpodGraphQlClient:
    def __init__(self, *, url: str = PRODUCTION_GRAPHQL_URL,
                 api_key: str | None = None, authorization=None,
                 opener=None, sleep=time.sleep, rng=random.random):
        self.url = url
        self._api_key = resolve_credential(
            url, api_key, production_url=PRODUCTION_GRAPHQL_URL)
        self._authorization = authorization
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep
        self._rng = rng
        self.request_log: list[dict] = []   # redacted, safe to persist

    def has_credential(self) -> bool:
        return bool(self._api_key)

    # ---------------- transport internals ----------------

    def _send_once(self, document: str, variables: dict | None):
        body = json.dumps({"query": document,
                           "variables": variables or {}}).encode("utf-8")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json",
                   "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers,
                                     method="POST")
        try:
            with self._opener(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.request_log.append({"graphql": "network-error"})
            raise GraphQlError(
                f"GraphQL network failure: {type(exc).__name__}") from None
        self.request_log.append({"graphql": redact(document.split("{")[0]
                                                   .strip()),
                                 "status": status})
        return status, raw

    def _parse(self, status: int, raw: bytes) -> dict:
        if status != 200:
            raise GraphQlError(f"GraphQL HTTP {status}: "
                               f"{redact(raw.decode('utf-8', 'replace')[:300])}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            raise GraphQlError("GraphQL response is not JSON; failing closed")
        if not isinstance(payload, dict):
            raise GraphQlError("GraphQL response is not an object")
        if payload.get("errors"):
            raise GraphQlError(
                "GraphQL errors: "
                + redact(json.dumps(payload["errors"])[:500]))
        data = payload.get("data")
        if not isinstance(data, dict):
            raise GraphQlError("GraphQL response carries no data object")
        return data

    def query(self, document: str, variables: dict | None = None) -> dict:
        """Read-only path: the document MUST be a named query operation."""
        if not _QUERY_RE.match(document):
            raise ReadOnlyViolation(
                "GraphQL read-only path accepts only explicit `query` "
                "operations; mutations require mutate() with a verified "
                "LiveMutationAuthorization (LIVE_MUTATION_NOT_AUTHORIZED)")
        attempt = 0
        while True:
            status, raw = self._send_once(document, variables)
            if status in RETRYABLE_STATUS and attempt < MAX_QUERY_RETRIES:
                attempt += 1
                delay = min(BACKOFF_CAP_SECONDS,
                            BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                self._sleep(delay * (0.5 + self._rng()))
                continue
            return self._parse(status, raw)

    def mutate(self, document: str, variables: dict | None = None) -> dict:
        from .authorization import LiveMutationAuthorization
        if not isinstance(self._authorization, LiveMutationAuthorization):
            raise ReadOnlyViolation(
                "LIVE_MUTATION_NOT_AUTHORIZED: GraphQL mutation requires a "
                "verified LiveMutationAuthorization")
        self._authorization.recheck()
        # _parse MUST be inside the try.  _send_once turns an HTTPError into
        # a (status, raw) return, so a 5xx or an {"errors": ...} body from a
        # CREATE mutation surfaces here — and the pod may well exist anyway.
        # Treating that as a plain failure (rather than ambiguous) skipped
        # reconciliation entirely and could leave a billing pod whose id was
        # never recorded.
        try:
            status, raw = self._send_once(document, variables)
            return self._parse(status, raw)
        except AmbiguousMutation:
            raise
        except TransportError as exc:
            raise AmbiguousMutation(
                f"GraphQL mutation outcome unknown ({exc}); reconcile owned "
                f"resources before any retry") from None

    # ---------------- pinned operations ----------------

    def spot_pricing(self, gpu_type_id: str) -> dict:
        """Live spot offer for one GPU type; fail closed on shape drift."""
        data = self.query(QUERY_SPOT_PRICING)
        types = data.get("gpuTypes")
        if not isinstance(types, list):
            raise SchemaIncompatibility("gpuTypes is not a list")
        match = [g for g in types
                 if isinstance(g, dict) and g.get("id") == gpu_type_id]
        if not match:
            raise SchemaIncompatibility(
                f"GraphQL gpuTypes carries no entry {gpu_type_id!r}")
        if len(match) > 1:
            raise SchemaIncompatibility(
                f"GraphQL gpuTypes carries {len(match)} entries for the id")
        g = match[0]
        # lowestPrice is queried with secureCloud:true, so minimumBidPrice is
        # the SECURE market minimum — an unfiltered global/community minimum
        # must never be used to raise a Secure bid
        lowest = g.get("lowestPrice") or {}
        return {
            "gpu_type_id": g["id"],
            "memory_gb": g.get("memoryInGb"),
            "secure_cloud": g.get("secureCloud"),
            "secure_list_usd": g.get("securePrice"),
            "secure_spot_usd": g.get("secureSpotPrice"),
            "community_spot_usd": g.get("communitySpotPrice"),
            "minimum_bid_usd": lowest.get("minimumBidPrice"),
            "stock_status": lowest.get("stockStatus"),
        }

    def myself_pods(self) -> list[dict]:
        data = self.query(QUERY_MYSELF_PODS)
        myself = data.get("myself")
        if not isinstance(myself, dict) or not isinstance(
                myself.get("pods"), list):
            raise SchemaIncompatibility("myself.pods missing from response")
        out = []
        for p in myself["pods"]:
            if not isinstance(p, dict) or "id" not in p:
                raise SchemaIncompatibility("malformed myself.pods entry")
            out.append(p)
        return out

    def verify_contract(self) -> dict:
        """Verify the pinned GraphQL surface as far as read-only allows.

        Executing the pinned queries proves the query-side field contract
        live (Apollo validates the full document against the schema before
        execution — an unknown field errors instead of returning data).
        The mutation input cannot be introspected read-only
        (INTROSPECTION_DISABLED); it stays pinned-only and fails closed at
        launch time by the same pre-execution validation, with no billing.
        """
        self.query(QUERY_SPOT_PRICING)
        self.query(QUERY_MYSELF_PODS)
        return {
            "query_surface": "LIVE_VERIFIED",
            "mutation_surface": "PINNED_ONLY_INTROSPECTION_DISABLED"
                                "_FAILS_CLOSED_AT_LAUNCH",
            "pinned_rent_input_fields": sorted(PINNED_RENT_INPUT_FIELDS),
            "pinned_gpu_type_fields": sorted(PINNED_GPU_TYPE_FIELDS),
        }

    def rent_interruptable(self, rent_input: dict) -> dict:
        """Create ONE interruptible pod (authorization-gated)."""
        data = self.mutate(MUTATION_RENT_INTERRUPTABLE,
                           {"input": dict(rent_input)})
        pod = data.get("podRentInterruptable")
        if not isinstance(pod, dict) or not pod.get("id"):
            raise SchemaIncompatibility(
                "podRentInterruptable returned no pod object; treat as "
                "ambiguous and reconcile")
        return pod
