"""Local HTTP mock of the pinned RunPod v2 contract + GraphQL spot surface.

Drives the REAL production adapter over real HTTP (http.server) — not a
simplified client.  Scriptable scenario knobs cover: catalog variants
(B300/B200 present/absent, Secure/Community, price changes,
multi-datacenter, alternate-GPU-only, 2-GPU minimum), spot pricing
(changing rates, vanished/returned interruptible stock), interruptible pod
creation via /graphql podRentInterruptable, EVICTION of running spot pods
(stop-with-EXITED, matching RunPod's spot semantics), pod lifecycle
transitions, logs, billing, stop/terminate, 429 with Retry-After, 5xx,
malformed JSON, response loss after successful creation, delayed
termination, unknown lifecycle states, and schema drift on
/v2/openapi.json.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The one synthetic credential local mock tests use.  The mock server
# validates it strictly; it is never a real RunPod key, and real operator
# credentials must never replace it (transport.CredentialIsolationError
# enforces that at the transport layer).
MOCK_API_KEY = "rpa_MOCKKEY_1234567890abcdef"


def _gpu(gpu_id, name, memory, price_secure, availability="HIGH",
         secure=True, community=True, max_secure=8, dcs=None):
    return {
        "id": gpu_id, "name": name, "manufacturer": "NVIDIA",
        "memory": memory, "secure": secure, "community": community,
        "pool": None,
        "price": {"secure": price_secure, "community": 4.99},
        "maxCount": {"secure": max_secure, "community": 4},
        "availability": availability,
        "dataCenters": dcs if dcs is not None else [
            {"id": "US-KS-2", "availability": availability},
            {"id": "EU-RO-1", "availability": "LOW" if availability != "NONE"
             else "NONE"},
        ],
    }


def _b300(price_secure=7.89, availability="HIGH", **kw):
    return _gpu("NVIDIA B300 SXM6 AC", "B300", kw.pop("memory", 288),
                price_secure, availability=availability, **kw)


def _b200(price_secure=6.79, availability="HIGH", gpu_id="NVIDIA B200",
          name="B200", memory=180, **kw):
    return _gpu(gpu_id, name, memory, price_secure,
                availability=availability, **kw)


class Scenario:
    """Mutable scenario state shared with the handler."""

    def __init__(self):
        self.gpus = [_b300(), _b200()]
        self.auth_required = True
        self.api_key = MOCK_API_KEY
        self.pods: dict[str, dict] = {}
        self.lifecycle_plan = ["PROVISIONING", "STARTING", "RUNNING"]
        self.lifecycle_step_per_poll = True
        self.create_delay_terminate = 0     # polls TERMINATED lingers
        self.rate_limit_next = 0            # respond 429 to next N requests
        self.fail_next = []                 # list of (status, times)
        self.malformed_next = 0
        self.drop_create_response = False   # create succeeds, response lost
        self.unknown_status = None          # override status string
        self.openapi_drift = None           # None | "enum" | "path"
        self.terminate_requires_polls = 0   # delayed termination
        self.billing = {"pods": [{"podId": "none", "amountUsd": 0.0}]}
        self.log_text = "mock container log line\n"
        self.requests: list[tuple[str, str]] = []
        # ---- GraphQL spot surface ----
        # per-gpu spot overrides: {gpu_id: {"secureSpotPrice": .., "stockStatus": ..}}
        self.spot: dict[str, dict] = {}
        self.spot_stock_default = "High"
        self.graphql_fail_next = 0          # 5xx the next N graphql calls
        self.drop_rent_response = False     # rent succeeds, response lost
        self.rent_calls: list[dict] = []    # every rent input received
        self.graphql_documents: list[str] = []  # every document received
        self.pod_json_omit_env = False      # live REST may omit pod env
        self.evict_after_polls: int | None = None   # auto-evict running pods

    def spot_for(self, gpu_id: str) -> dict:
        entry = next((g for g in self.gpus if g["id"] == gpu_id), None)
        base = {
            "secureSpotPrice": (entry or {}).get("price", {}).get("secure"),
            "communitySpotPrice": (entry or {}).get("price", {}).get("community"),
            "minimumBidPrice": None,
            "stockStatus": self.spot_stock_default,
        }
        base.update(self.spot.get(gpu_id, {}))
        return base

    def evict(self, pod_id: str) -> None:
        """Spot eviction: SIGTERM/SIGKILL -> pod STOPPED (EXITED), never
        TERMINATED; container-disk state is considered lost."""
        pod = self.pods[pod_id]
        pod["plan"] = ["EXITED"]
        pod["poll_count"] = 0
        pod["desiredStatus"] = "EXITED"
        pod["evicted"] = True

    def next_pod_status(self, pod):
        plan = pod["plan"]
        if pod["terminated"]:
            if pod["terminate_polls_left"] > 0:
                pod["terminate_polls_left"] -= 1
                return pod["status"]
            return "TERMINATED"
        if self.unknown_status:
            return self.unknown_status
        idx = min(pod["poll_count"], len(plan) - 1)
        status = plan[idx]
        if (self.evict_after_polls is not None and status == "RUNNING"
                and not pod.get("evicted")
                and pod["poll_count"] >= self.evict_after_polls):
            self.evict(pod["id"])
            return "EXITED"
        return status


class _Handler(BaseHTTPRequestHandler):
    scenario: Scenario = None  # injected

    def log_message(self, *a):  # silence
        pass

    def _reply(self, status, obj=None, headers=None):
        body = b""
        if obj is not None:
            body = (obj if isinstance(obj, bytes)
                    else json.dumps(obj).encode())
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _err(self, status, title, detail):
        self._reply(status, {"title": title, "status": status,
                             "detail": detail})

    def _common(self, method):
        s = self.scenario
        s.requests.append((method, self.path))
        if s.rate_limit_next > 0:
            s.rate_limit_next -= 1
            self._reply(429, {"title": "Too Many Requests", "status": 429,
                              "detail": "rate limited"},
                        {"Retry-After": "0"})
            return False
        if s.fail_next:
            status, times = s.fail_next[0]
            if times <= 1:
                s.fail_next.pop(0)
            else:
                s.fail_next[0] = (status, times - 1)
            self._err(status, "Injected", "injected failure")
            return False
        if s.malformed_next > 0:
            s.malformed_next -= 1
            self._reply(200, b"{not json!!")
            return False
        if s.auth_required:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {s.api_key}":
                self._err(401, "Unauthorized", "bad or missing API key")
                return False
        return True

    # ---------------- GET ----------------

    def do_GET(self):
        s = self.scenario
        if self.path == "/v2/openapi.json":
            # openapi identity is public in the mock
            s.requests.append(("GET", self.path))
            self._reply(200, self._openapi())
            return
        if not self._common("GET"):
            return
        if self.path.startswith("/v2/catalog/gpus"):
            self._reply(200, s.gpus)
            return
        if self.path.startswith("/v2/catalog/datacenters"):
            self._reply(200, [{"id": "US-KS-2", "name": "Kansas",
                               "region": "NORTH_AMERICA",
                               "networkVolumeTypes": [], "compliance": [],
                               "globalNetwork": True}])
            return
        m = re.match(r"^/v2/pods/([^/?]+)/logs", self.path)
        if m:
            pod = s.pods.get(m.group(1))
            if pod is None:
                self._err(404, "Not Found", "no such pod")
                return
            self._reply(200, {"logs": s.log_text})
            return
        m = re.match(r"^/v2/pods/([^/?]+)$", self.path)
        if m:
            pod = s.pods.get(m.group(1))
            if pod is None:
                self._err(404, "Not Found", "no such pod")
                return
            pod["poll_count"] += 1
            pod["status"] = s.next_pod_status(pod)
            self._reply(200, self._pod_json(pod))
            return
        if self.path == "/v2/pods":
            self._reply(200, [self._pod_json(p) for p in s.pods.values()])
            return
        if self.path.startswith("/v2/billing/pods"):
            self._reply(200, s.billing)
            return
        if self.path.startswith("/v2/billing"):
            self._reply(200, {"totalUsd": 0.0})
            return
        self._err(404, "Not Found", f"no route {self.path}")

    # ---------------- POST / DELETE ----------------

    def do_POST(self):
        s = self.scenario
        if self.path == "/graphql":
            self._graphql()
            return
        if not self._common("POST"):
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v2/pods":
            pod_id = f"mockpod-{uuid.uuid4().hex[:12]}"
            pod = {
                "id": pod_id, "name": body.get("name", ""),
                "image": body.get("image"), "env": body.get("env", {}),
                "gpu": body.get("gpu"), "cloud": body.get("cloud", "SECURE"),
                "status": "PROVISIONING", "poll_count": 0,
                "plan": list(s.lifecycle_plan), "terminated": False,
                "terminate_polls_left": 0,
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
            }
            s.pods[pod_id] = pod
            if s.drop_create_response:
                s.drop_create_response = False
                # simulate a lost response: close without replying
                self.connection.close()
                return
            self._reply(201, self._pod_json(pod))
            return
        m = re.match(r"^/v2/pods/([^/?]+)/action$", self.path)
        if m:
            pod = s.pods.get(m.group(1))
            if pod is None:
                self._err(404, "Not Found", "no such pod")
                return
            action = body.get("action")
            if action not in ("start", "stop", "restart", "terminate"):
                self._err(422, "Invalid", f"unknown action {action}")
                return
            if action == "terminate":
                pod["terminated"] = True
                pod["terminate_polls_left"] = s.terminate_requires_polls
            elif action == "stop":
                pod["plan"] = ["EXITED"]
                pod["poll_count"] = 0
            self._reply(200, self._pod_json(pod))
            return
        self._err(404, "Not Found", f"no route {self.path}")

    def do_DELETE(self):
        s = self.scenario
        if not self._common("DELETE"):
            return
        m = re.match(r"^/v2/pods/([^/?]+)$", self.path)
        if m:
            pod = s.pods.get(m.group(1))
            if pod is None:
                self._err(404, "Not Found", "no such pod")
                return
            pod["terminated"] = True
            pod["terminate_polls_left"] = s.terminate_requires_polls
            self._reply(204)
            return
        self._err(404, "Not Found", f"no route {self.path}")

    do_PATCH = do_POST  # any PATCH hits _common auth and 404s

    # ---------------- GraphQL spot surface ----------------

    def _graphql(self):
        s = self.scenario
        s.requests.append(("POST", "/graphql"))
        if s.auth_required:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {s.api_key}":
                self._reply(200, {"errors": [{"message": "Unauthorized",
                                              "extensions":
                                              {"code": "UNAUTHENTICATED"}}]})
                return
        if s.graphql_fail_next > 0:
            s.graphql_fail_next -= 1
            self._err(503, "Injected", "graphql injected failure")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        doc = body.get("query", "")
        s.graphql_documents.append(doc)
        variables = body.get("variables") or {}
        if "podRentInterruptable" in doc:
            self._graphql_rent(variables.get("input") or {})
            return
        if "gpuTypes" in doc:
            out = []
            for g in s.gpus:
                spot = s.spot_for(g["id"])
                out.append({
                    "id": g["id"], "memoryInGb": g["memory"],
                    "secureCloud": g["secure"],
                    "securePrice": g["price"]["secure"],
                    "communityPrice": g["price"]["community"],
                    "secureSpotPrice": spot["secureSpotPrice"],
                    "communitySpotPrice": spot["communitySpotPrice"],
                    "lowestPrice": {
                        "minimumBidPrice": spot["minimumBidPrice"],
                        "uninterruptablePrice": g["price"]["secure"],
                        "stockStatus": spot["stockStatus"]},
                })
            self._reply(200, {"data": {"gpuTypes": out}})
            return
        if "myself" in doc:
            pods = [{"id": p["id"], "name": p["name"],
                     "desiredStatus": ("TERMINATED" if p["terminated"]
                                       else p.get("desiredStatus",
                                                  p["status"])),
                     "podType": p.get("podType", "INTERRUPTABLE"),
                     "costPerHr": p.get("bidPerGpu", 0.0)}
                    for p in s.pods.values()]
            self._reply(200, {"data": {"myself": {"pods": pods}}})
            return
        self._reply(200, {"errors": [{"message": f"unsupported operation"}]})

    def _graphql_rent(self, rent: dict):
        s = self.scenario
        s.rent_calls.append(dict(rent))
        gpu_id = rent.get("gpuTypeId")
        entry = next((g for g in s.gpus if g["id"] == gpu_id), None)
        if entry is None:
            self._reply(200, {"errors": [{"message":
                                          f"unknown gpuTypeId {gpu_id}"}]})
            return
        spot = s.spot_for(gpu_id)
        if spot["secureSpotPrice"] is None or spot["stockStatus"] is None:
            self._reply(200, {"errors": [{"message":
                                          "no interruptible capacity"}]})
            return
        bid = rent.get("bidPerGpu") or 0.0
        if bid < (spot["minimumBidPrice"] or spot["secureSpotPrice"] or 0.0):
            self._reply(200, {"errors": [{"message": "bid below market"}]})
            return
        env = {e["key"]: e["value"] for e in rent.get("env") or []}
        pod_id = f"mockspot-{uuid.uuid4().hex[:12]}"
        pod = {
            "id": pod_id, "name": rent.get("name", ""),
            "image": rent.get("imageName"), "env": env,
            "gpu": {"id": gpu_id, "count": rent.get("gpuCount", 1)},
            "cloud": rent.get("cloudType", "SECURE"),
            "status": "PROVISIONING", "poll_count": 0,
            "plan": list(s.lifecycle_plan), "terminated": False,
            "terminate_polls_left": 0, "podType": "INTERRUPTABLE",
            "desiredStatus": "RUNNING", "bidPerGpu": bid,
            "terminateAfter": rent.get("terminateAfter"),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        s.pods[pod_id] = pod
        if s.drop_rent_response:
            s.drop_rent_response = False
            self.connection.close()
            return
        self._reply(200, {"data": {"podRentInterruptable": {
            "id": pod_id, "name": pod["name"],
            "desiredStatus": "RUNNING", "costPerHr": bid}}})

    # ---------------- helpers ----------------

    def _pod_json(self, pod):
        out = {"id": pod["id"], "name": pod["name"], "status": pod["status"],
               "cloud": pod["cloud"], "gpu": pod["gpu"],
               "image": pod["image"], "cost": 5.49,
               "createdAt": pod["createdAt"],
               "startedAt": None, "dataCenterId": "US-KS-2",
               "env": pod["env"]}
        if self.scenario.pod_json_omit_env:
            # the pinned REST contract does not REQUIRE env on a Pod; a live
            # surface that omits it must not silently degrade nonce-based
            # ownership reconciliation into name-only matching
            out.pop("env")
        return out

    def _openapi(self):
        # minimal but surface-complete document mirroring the pinned contract
        doc = {
            "openapi": "3.1.0",
            "info": {"title": "Runpod REST API", "version": "2.0.0"},
            "paths": {p: {m: {} for m in ms} for p, ms in {
                "/v2/catalog/gpus": ["get"],
                "/v2/catalog/datacenters": ["get"],
                "/v2/pods": ["get", "post"],
                "/v2/pods/{id}": ["get", "delete"],
                "/v2/pods/{id}/action": ["post"],
                "/v2/pods/{id}/logs": ["get"],
                "/v2/billing/pods": ["get"],
            }.items()},
            "components": {"schemas": {
                "PodStatus": {"enum": ["PROVISIONING", "STARTING", "RUNNING",
                                       "EXITED", "ERROR", "TERMINATED"]},
                "Cloud": {"enum": ["SECURE", "COMMUNITY"]},
                "AvailabilityLevel": {"enum": ["NONE", "LOW", "MEDIUM",
                                               "HIGH"]},
                "PodAction": {"enum": ["start", "stop", "restart",
                                       "terminate"]},
                "GpuType": {"required": ["id", "name", "pool", "manufacturer",
                                         "memory", "secure", "community",
                                         "price", "maxCount"],
                            "properties": {"price": {"required":
                                                     ["secure", "community"]}}},
                "CreatePodRequest": {"allOf": [
                    {"properties": {}},
                    {"required": ["name", "image"], "properties": {}}]},
            }},
        }
        drift = self.scenario.openapi_drift
        if drift == "enum":
            doc["components"]["schemas"]["PodStatus"]["enum"].append("PAUSED")
        elif drift == "path":
            del doc["paths"]["/v2/pods/{id}/action"]
        return doc


class MockRunpodServer:
    def __init__(self, scenario: Scenario | None = None):
        self.scenario = scenario or Scenario()
        handler = type("Handler", (_Handler,), {"scenario": self.scenario})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
