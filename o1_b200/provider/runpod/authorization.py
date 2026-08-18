"""Absolute live-mutation interlock.

Mutating RunPod operations require ALL of:

  1. env RUNPOD_ALLOW_BILLABLE_MUTATIONS == "YES_I_AUTHORIZE_THIS_RUN";
  2. an authorization file (B200_RENTAL_AUTHORIZATION.json);
  3. a valid authorization-file schema;
  4. matching project / package / provider / budget identities;
  5. an expiry timestamp in the future;
  6. a unique launch nonce (replay refused via a local nonce ledger);
  7. a SHA-256 commitment to the final deployment specification;
  8. allow_create_pod: true;
  9. allow_real_calibration: true;
 10. allow_confirmation: false  (must be literally false);
 11. the CLI flag --execute-authorized-rental;
 12. max_pod_creations: a small positive integer bounding how many pod
     creations (initial acquisition + eviction reacquisitions of
     INTERRUPTIBLE capacity) this one authorization may perform.  Each
     creation consumes one ledger slot ("nonce/seq"); exhaustion refuses
     further creation exactly like a replay.

Anything missing raises AuthorizationError("LIVE_MUTATION_NOT_AUTHORIZED …").
This module can only be satisfied by a real, deliberately created file — the
shipped template contains UNRESOLVED fields and fails validation by design.
"""
from __future__ import annotations

import json
import os
import time

from .redaction import redact

ENV_FLAG = "RUNPOD_ALLOW_BILLABLE_MUTATIONS"
ENV_FLAG_VALUE = "YES_I_AUTHORIZE_THIS_RUN"
CLI_FLAG = "--execute-authorized-rental"

AUTH_SCHEMA = "o1b300.rental_authorization.v2"

REQUIRED_FIELDS = (
    "schema", "project", "package_zip_sha256", "provider",
    "budget_policy_sha256", "expires_utc", "launch_nonce",
    "deployment_spec_sha256", "allow_create_pod", "allow_real_calibration",
    "allow_confirmation", "max_pod_creations",
)

MAX_ALLOWED_POD_CREATIONS = 8   # absolute ceiling regardless of the file


class AuthorizationError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(f"LIVE_MUTATION_NOT_AUTHORIZED: {redact(reason)}")


def _utcnow() -> float:
    return time.time()


class LiveMutationAuthorization:
    """Verified authorization context; construct via ``verify``."""

    def __init__(self, doc: dict, path: str, nonce_ledger: str,
                 now=_utcnow):
        self._doc = doc
        self._path = path
        self._nonce_ledger = nonce_ledger
        self._now = now

    @classmethod
    def verify(cls, *, path: str, expected_identity: dict,
               cli_args: list[str], nonce_ledger: str,
               environ: dict | None = None,
               now=_utcnow) -> "LiveMutationAuthorization":
        env = os.environ if environ is None else environ
        if env.get(ENV_FLAG) != ENV_FLAG_VALUE:
            raise AuthorizationError(
                f"environment variable {ENV_FLAG} is not set to the exact "
                f"authorization value")
        if CLI_FLAG not in cli_args:
            raise AuthorizationError(f"command-line flag {CLI_FLAG} absent")
        if not os.path.exists(path):
            raise AuthorizationError(f"authorization file missing: {path}")
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuthorizationError(f"authorization file unreadable: {exc}")
        missing = [f for f in REQUIRED_FIELDS if f not in doc]
        if missing:
            raise AuthorizationError(
                f"authorization schema invalid; missing {missing}")
        if doc["schema"] != AUTH_SCHEMA:
            raise AuthorizationError(
                f"authorization schema {doc['schema']!r} != {AUTH_SCHEMA!r}")
        unresolved = [k for k, v in doc.items()
                      if isinstance(v, str) and v.startswith("UNRESOLVED")]
        if unresolved:
            raise AuthorizationError(
                f"authorization contains unresolved template fields "
                f"{unresolved}; the template can never authorize anything")
        for key in ("project", "package_zip_sha256", "provider",
                    "budget_policy_sha256", "deployment_spec_sha256"):
            want = expected_identity.get(key)
            if want is None:
                raise AuthorizationError(f"expected identity lacks {key}")
            if doc[key] != want:
                raise AuthorizationError(
                    f"identity mismatch on {key}: authorization does not "
                    f"bind this project/package/provider/budget/deployment")
        try:
            expires = time.strptime(doc["expires_utc"], "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            raise AuthorizationError("expires_utc is not a valid UTC timestamp")
        import calendar
        if calendar.timegm(expires) <= now():
            raise AuthorizationError("authorization has expired")
        if doc["allow_create_pod"] is not True:
            raise AuthorizationError("allow_create_pod must be literally true")
        if doc["allow_real_calibration"] is not True:
            raise AuthorizationError(
                "allow_real_calibration must be literally true")
        if doc["allow_confirmation"] is not False:
            raise AuthorizationError(
                "allow_confirmation must be literally false; confirmatory "
                "generation is not authorized in any session")
        nonce = doc["launch_nonce"]
        if not isinstance(nonce, str) or len(nonce) < 16:
            raise AuthorizationError("launch_nonce must be >= 16 chars")
        maxc = doc["max_pod_creations"]
        if not isinstance(maxc, int) or isinstance(maxc, bool) \
                or not 1 <= maxc <= MAX_ALLOWED_POD_CREATIONS:
            raise AuthorizationError(
                f"max_pod_creations must be an integer in "
                f"[1, {MAX_ALLOWED_POD_CREATIONS}]")
        if cls._consumed_count(nonce_ledger, nonce) >= maxc:
            raise AuthorizationError(
                "launch_nonce creation budget exhausted (replay refused)")
        return cls(doc, path, nonce_ledger, now)

    @staticmethod
    def _consumed_count(nonce_ledger: str, nonce: str) -> int:
        """Count consumed slots as LINES, never as set members.

        Each consumption appends a globally unique line, so two concurrent
        creations can no longer collapse into one charged slot the way
        identical `nonce/seq` lines did under set-deduplication.
        """
        if not os.path.exists(nonce_ledger):
            return 0
        with open(nonce_ledger, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        # v1 ledgers recorded the bare nonce; count it as one consumption
        return sum(1 for ln in lines
                   if ln == nonce or ln.startswith(nonce + "/"))

    def consume_nonce(self) -> None:
        """Burn ONE creation slot (called immediately before each create).

        Refuses beyond max_pod_creations: eviction reacquisition is bounded
        by this ledger in addition to the hard dollar budget.  The
        read-check-append sequence holds an exclusive advisory lock on the
        ledger so two concurrent session drivers cannot both observe the
        same remaining count and each create a pod.
        """
        import fcntl
        import uuid
        nonce = self._doc["launch_nonce"]
        maxc = self._doc["max_pod_creations"]
        lock_path = self._nonce_ledger + ".lock"
        os.makedirs(os.path.dirname(os.path.abspath(lock_path)) or ".",
                    exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                seq = self._consumed_count(self._nonce_ledger, nonce)
                if seq >= maxc:
                    raise AuthorizationError(
                        f"pod-creation budget exhausted "
                        f"({seq}/{maxc} consumed); a new deliberate "
                        f"authorization is required")
                with open(self._nonce_ledger, "a", encoding="utf-8") as fh:
                    fh.write(f"{nonce}/{seq}/{uuid.uuid4().hex}\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def creations_remaining(self) -> int:
        return self._doc["max_pod_creations"] - self._consumed_count(
            self._nonce_ledger, self._doc["launch_nonce"])

    def recheck(self, *, releasing: bool = False) -> None:
        """Re-validated before EVERY mutating request.

        ``releasing=True`` marks a mutation that only ever RELEASES
        resources (stop/terminate/delete).  Expiry must not disable those:
        a session that outlives its authorization — routine once eviction
        and reacquisition stretch a run past a deliberately short expiry —
        would otherwise lose its primary termination path at exactly the
        moment it needs to shut a billing pod down, leaving only the
        watchdog's deadline. Acquisition stays strictly gated.
        """
        if os.environ.get(ENV_FLAG) != ENV_FLAG_VALUE:
            raise AuthorizationError(f"{ENV_FLAG} unset mid-run")
        import calendar
        expires = calendar.timegm(
            time.strptime(self._doc["expires_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        if expires <= self._now() and not releasing:
            raise AuthorizationError("authorization expired mid-run")

    @property
    def launch_nonce(self) -> str:
        return self._doc["launch_nonce"]

    @property
    def deployment_spec_sha256(self) -> str:
        return self._doc["deployment_spec_sha256"]
