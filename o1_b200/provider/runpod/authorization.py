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
 11. the CLI flag --execute-authorized-rental.

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

AUTH_SCHEMA = "o1b200.rental_authorization.v1"

REQUIRED_FIELDS = (
    "schema", "project", "package_zip_sha256", "provider",
    "budget_policy_sha256", "expires_utc", "launch_nonce",
    "deployment_spec_sha256", "allow_create_pod", "allow_real_calibration",
    "allow_confirmation",
)


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
        used = set()
        if os.path.exists(nonce_ledger):
            with open(nonce_ledger, encoding="utf-8") as fh:
                used = {ln.strip() for ln in fh if ln.strip()}
        if nonce in used:
            raise AuthorizationError(
                f"launch_nonce already consumed (replay refused)")
        return cls(doc, path, nonce_ledger, now)

    def consume_nonce(self) -> None:
        """Burn the nonce (called exactly once, immediately before create)."""
        with open(self._nonce_ledger, "a", encoding="utf-8") as fh:
            fh.write(self._doc["launch_nonce"] + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def recheck(self) -> None:
        """Re-validated before EVERY mutating request."""
        if os.environ.get(ENV_FLAG) != ENV_FLAG_VALUE:
            raise AuthorizationError(f"{ENV_FLAG} unset mid-run")
        import calendar
        expires = calendar.timegm(
            time.strptime(self._doc["expires_utc"], "%Y-%m-%dT%H:%M:%SZ"))
        if expires <= self._now():
            raise AuthorizationError("authorization expired mid-run")

    @property
    def launch_nonce(self) -> str:
        return self._doc["launch_nonce"]

    @property
    def deployment_spec_sha256(self) -> str:
        return self._doc["deployment_spec_sha256"]
