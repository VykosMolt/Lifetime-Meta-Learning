"""Local HTTP mock of the pinned RunPod v2 contract.

Drives the REAL production adapter over real HTTP (http.server) — not a
simplified client.  Scriptable scenario knobs cover: catalog variants (B200
present/absent, Secure/Community, price changes, multi-datacenter,
alternate-GPU-only, 2-GPU minimum), pod creation and asynchronous lifecycle
transitions, logs, billing, stop/terminate, 429 with Retry-After, 5xx,
malformed JSON, response loss after successful creation, delayed
termination, unknown lifecycle states, and schema drift on /v2/openapi.json.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _b200(price_secure=5.49, availability="HIGH", secure=True,
          community=True, max_secure=8, gpu_id="NVIDIA B200",
          name="B200", memory=180, dcs=None):
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


class Scenario:
    """Mutable scenario state shared with the handler."""

    def __init__(self):
        self.gpus = [_b200()]
        self.auth_required = True
        self.api_key = "rpa_MOCKKEY_1234567890abcdef"
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
        return plan[idx]


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

    # ---------------- helpers ----------------

    def _pod_json(self, pod):
        return {"id": pod["id"], "name": pod["name"], "status": pod["status"],
                "cloud": pod["cloud"], "gpu": pod["gpu"],
                "image": pod["image"], "cost": 5.49,
                "createdAt": pod["createdAt"],
                "startedAt": None, "dataCenterId": "US-KS-2",
                "env": pod["env"]}

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
