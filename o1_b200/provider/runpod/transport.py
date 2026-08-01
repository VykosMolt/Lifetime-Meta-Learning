"""HTTP transport for the RunPod v2 API with hard read-only enforcement.

Two transports:

  * ReadOnlyTransport — the ONLY transport constructible without the full
    live-mutation authorization.  It rejects every non-GET method BEFORE any
    socket is opened (enforced here at the transport layer, not by
    convention), so a mutating call attempted in read-only mode is a local
    exception, never a network event.
  * MutatingTransport — requires a verified LiveMutationAuthorization
    (authorization.py).  Even then, each individual mutating request
    re-checks the authorization before sending.

Both transports: bearer auth from redaction.load_api_key(), full redaction of
logs and exceptions, bounded retries with exponential backoff + jitter for
idempotent GETs (respecting Retry-After), fail-closed on unknown status codes
and malformed JSON.  Mutating requests are NEVER retried blindly: a timeout
or lost response surfaces as AmbiguousMutation so the caller reconciles.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

from .redaction import redact, load_api_key
from .models import parse_error_response

DEFAULT_BASE_URL = "https://api.runpod.io"
USER_AGENT = "o1-b200-runner/0.2 (pre-rental; read-only unless authorized)"

RETRYABLE_STATUS = (429, 500, 502, 503, 504)
MAX_GET_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_CAP_SECONDS = 30.0
REQUEST_TIMEOUT_SECONDS = 30.0


class TransportError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(redact(message))


class ReadOnlyViolation(TransportError):
    """A mutating request was attempted through the read-only transport."""


class ApiHttpError(TransportError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class AmbiguousMutation(TransportError):
    """A mutating request's outcome is unknown; reconcile before retrying."""


class MalformedResponse(TransportError):
    pass


class _BaseTransport:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 api_key: str | None = None,
                 sleep=time.sleep, rng=random.random,
                 opener=None):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key if api_key is not None else load_api_key()
        self._sleep = sleep
        self._rng = rng
        self._opener = opener or urllib.request.urlopen
        self.request_log: list[dict] = []   # redacted, safe to persist

    def has_credential(self) -> bool:
        return bool(self._api_key)

    # -------- internals --------

    def _headers(self) -> dict:
        h = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _record(self, method: str, url: str, status) -> None:
        self.request_log.append({
            "method": method, "url": redact(url), "status": status,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def _send_once(self, method: str, path: str, body: dict | None):
        url = self.base_url + path
        data = None
        headers = self._headers()
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            with self._opener(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                raw = resp.read()
                status = resp.status
                retry_after = resp.headers.get("Retry-After")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = exc.code
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._record(method, url, "network-error")
            raise TransportError(f"network failure on {method} {path}: "
                                 f"{type(exc).__name__}") from None
        self._record(method, url, status)
        return status, raw, retry_after

    def _parse(self, method: str, path: str, status: int, raw: bytes):
        if status in (200, 201):
            if not raw:
                raise MalformedResponse(f"{method} {path}: empty 2xx body")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise MalformedResponse(
                    f"{method} {path}: malformed JSON in {status} response")
        if status == 204:
            return None
        if status in (400, 401, 403, 404, 409, 422, 429) or 500 <= status < 600:
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            raise ApiHttpError(status, parse_error_response(body, status))
        raise TransportError(
            f"{method} {path}: unknown HTTP status {status}; the pinned "
            f"contract does not define it — failing closed")

    def get(self, path: str) -> dict | list | None:
        attempt = 0
        while True:
            status, raw, retry_after = self._send_once("GET", path, None)
            if status in RETRYABLE_STATUS and attempt < MAX_GET_RETRIES:
                attempt += 1
                delay = min(BACKOFF_CAP_SECONDS,
                            BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
                delay = delay * (0.5 + self._rng())
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                self._sleep(delay)
                continue
            return self._parse("GET", path, status, raw)


class ReadOnlyTransport(_BaseTransport):
    """GET-only.  Mutations are rejected locally before any network I/O."""

    def request(self, method: str, path: str, body: dict | None = None):
        if method.upper() != "GET":
            raise ReadOnlyViolation(
                f"READ-ONLY transport refuses {method.upper()} {path}: "
                f"mutating RunPod operations require the full live-mutation "
                f"authorization (LIVE_MUTATION_NOT_AUTHORIZED)")
        return self.get(path)

    # explicit named refusals so call sites cannot even express a mutation
    def post(self, *_a, **_kw):
        raise ReadOnlyViolation("LIVE_MUTATION_NOT_AUTHORIZED: POST refused "
                                "by the read-only transport")

    patch = put = delete = post


class MutatingTransport(_BaseTransport):
    """Requires a verified LiveMutationAuthorization for every mutation."""

    def __init__(self, authorization, **kw):
        from .authorization import LiveMutationAuthorization
        if not isinstance(authorization, LiveMutationAuthorization):
            raise ReadOnlyViolation(
                "LIVE_MUTATION_NOT_AUTHORIZED: MutatingTransport requires a "
                "verified LiveMutationAuthorization object")
        super().__init__(**kw)
        self._authorization = authorization

    def mutate(self, method: str, path: str, body: dict | None = None):
        self._authorization.recheck()   # raises LIVE_MUTATION_NOT_AUTHORIZED
        method = method.upper()
        if method == "GET":
            return self.get(path)
        if method not in ("POST", "PATCH", "DELETE", "PUT"):
            raise TransportError(f"unsupported method {method}")
        try:
            status, raw, _ = self._send_once(method, path, body)
        except TransportError as exc:
            # outcome unknown: the request may have been applied remotely
            raise AmbiguousMutation(
                f"{method} {path}: response lost ({exc}); reconcile owned "
                f"resources before any retry") from None
        return self._parse(method, path, status, raw)

    def get(self, path: str):
        return super().get(path)
