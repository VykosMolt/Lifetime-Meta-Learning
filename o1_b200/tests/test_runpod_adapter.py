"""RunPod adapter: schema pin, catalog/quote validation, retries, redaction."""
from __future__ import annotations

import json
import os

from _h import Runner, fresh_dir

os.environ.setdefault("RUNPOD_API_KEY", "rpa_MOCKKEY_1234567890abcdef")

from o1_b200.provider.runpod import redaction
from o1_b200.provider.runpod.adapter import RunpodV2Adapter
from o1_b200.provider.runpod.billing import BudgetViolation, hard_compute_seconds
from o1_b200.provider.runpod.mock_server import MockRunpodServer, Scenario, _b200
from o1_b200.provider.runpod.models import SchemaIncompatibility
from o1_b200.provider.runpod.policy import PolicyViolation
from o1_b200.provider.runpod.quote import QuoteError
from o1_b200.provider.runpod.schema_check import SchemaPinError, verify_pinned_schema
from o1_b200.provider.runpod.transport import ApiHttpError, MalformedResponse, TransportError

NOSLEEP = lambda s: None  # noqa: E731


def _adapter(srv, **kw):
    return RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP, **kw)


def run() -> Runner:
    r = Runner("runpod_adapter")

    r.check("pinned OpenAPI snapshot verifies byte-exact with sane surfaces",
            verify_pinned_schema)

    def qualifying_quote():
        with MockRunpodServer() as srv:
            ad = _adapter(srv)
            q = ad.quote_instance(adapter_commit="test")
            assert q["gpu_type_id"] == "NVIDIA B200"
            assert q["cloud"] == "SECURE" and q["gpu_count"] == 1
            assert q["interruptible"] is False
            assert ad.validate_quote()["hard_compute_seconds"] == \
                hard_compute_seconds("5.49")
    r.check("1. qualifying Secure single-B200 quote accepted", qualifying_quote)

    def price_above_cap():
        sc = Scenario()
        sc.gpus = [_b200(price_secure=5.90)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except PolicyViolation as exc:
                assert "5.89" in str(exc)
                return
        raise AssertionError("USD 5.90 quote accepted over the 5.89 cap")
    r.check("2. price above USD 5.89 refused", price_above_cap)

    def community_only():
        sc = Scenario()
        sc.gpus = [_b200(secure=False)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except PolicyViolation as exc:
                assert "Community" in str(exc) or "Secure" in str(exc)
                return
        raise AssertionError("Community-only offer accepted")
    r.check("3. Community-only availability refused", community_only)

    def interruptible_drift():
        # the pinned contract has no interruptible surface; its appearance in
        # a live schema is drift and fails closed
        from o1_b200.provider.runpod.schema_check import check_live_identity, load_pinned_schema
        live = load_pinned_schema()
        live["components"]["schemas"]["CreatePodRequest"]["allOf"].append(
            {"properties": {"interruptible": {"type": "boolean"}}})
        try:
            check_live_identity(live)
        except SchemaPinError as exc:
            assert "interruptible" in str(exc)
            return
        raise AssertionError("interruptible surface drift accepted")
    r.check("4. interruptible offer surface = schema drift, fails closed",
            interruptible_drift)

    def two_gpu_only():
        sc = Scenario()
        sc.gpus = [_b200(max_secure=0)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except PolicyViolation as exc:
                assert "maxCount" in str(exc) or "single" in str(exc)
                return
        raise AssertionError("no-single-GPU offer accepted")
    r.check("5. cluster/multi-GPU-minimum offer refused (secure maxCount<1)",
            two_gpu_only)

    def b200_unavailable():
        sc = Scenario()
        sc.gpus = [_b200(availability="NONE",
                         dcs=[{"id": "US-KS-2", "availability": "NONE"}])]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "NONE" in str(exc)
                return
        raise AssertionError("NONE availability accepted")
    r.check("6. B200 unavailable (availability NONE) refused", b200_unavailable)

    def alternate_gpu_only():
        sc = Scenario()
        for sub in ("NVIDIA H200", "NVIDIA B300", "NVIDIA H100 80GB HBM3"):
            sc.gpus = [_b200(gpu_id=sub, name=sub)]
            with MockRunpodServer(sc) as srv:
                try:
                    _adapter(srv).quote_instance()
                except QuoteError:
                    continue
            raise AssertionError(f"substitute {sub} accepted")
    r.check("7. alternate GPU (H100/H200/B300) never substituted",
            alternate_gpu_only)

    def quote_expiry():
        clock = {"t": 1000.0}
        with MockRunpodServer() as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP,
                                 clock=lambda: clock["t"])
            ad.quote_instance()
            ad.validate_quote()
            clock["t"] += 15 * 60 + 1
            try:
                ad.validate_quote()
            except QuoteError as exc:
                assert "expired" in str(exc)
                return
        raise AssertionError("16-minute-old quote accepted")
    r.check("8. quote expires after the 15-minute validity interval",
            quote_expiry)

    def catalog_schema_drift():
        sc = Scenario()
        sc.gpus = [{k: v for k, v in _b200().items() if k != "price"}]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).list_gpu_types()
            except SchemaIncompatibility as exc:
                assert "price" in str(exc)
                return
        raise AssertionError("missing pricing field accepted")
    r.check("9. catalog schema drift (missing pricing) fails closed",
            catalog_schema_drift)

    def openapi_drift_detected():
        for kind in ("enum", "path"):
            sc = Scenario()
            sc.openapi_drift = kind
            with MockRunpodServer(sc) as srv:
                try:
                    _adapter(srv).get_api_schema_identity()
                except SchemaPinError:
                    continue
            raise AssertionError(f"live openapi drift {kind} accepted")
    r.check("live schema identity check fails closed on enum/path drift",
            openapi_drift_detected)

    def unknown_lifecycle_state():
        sc = Scenario()
        sc.unknown_status = "HIBERNATING"
        with MockRunpodServer(sc) as srv:
            ad = _adapter(srv)
            sc.pods["p1"] = {"id": "p1", "name": "x", "image": "i",
                             "env": {}, "gpu": {"id": "NVIDIA B200",
                                                "count": 1},
                             "cloud": "SECURE", "status": "RUNNING",
                             "poll_count": 0, "plan": ["RUNNING"],
                             "terminated": False, "terminate_polls_left": 0,
                             "createdAt": "2026-08-01T00:00:00Z"}
            try:
                ad.get_instance("p1")
            except SchemaIncompatibility as exc:
                assert "HIBERNATING" in str(exc)
                return
        raise AssertionError("unknown lifecycle state accepted")
    r.check("10. unknown lifecycle state fails closed", unknown_lifecycle_state)

    def rate_limit_429():
        sc = Scenario()
        sc.rate_limit_next = 2
        with MockRunpodServer(sc) as srv:
            ad = _adapter(srv)
            gpus = ad.list_gpu_types()   # retried through the 429s
            assert gpus[0].id == "NVIDIA B200"
    r.check("29. transient 429 with Retry-After is retried then succeeds",
            rate_limit_429)

    def transient_5xx():
        sc = Scenario()
        sc.fail_next = [(503, 2)]
        with MockRunpodServer(sc) as srv:
            assert _adapter(srv).list_gpu_types()[0].id == "NVIDIA B200"
    r.check("30. transient 5xx retried with backoff then succeeds",
            transient_5xx)

    def malformed_json_fails_closed():
        sc = Scenario()
        sc.malformed_next = 10   # exhaust retries
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).list_gpu_types()
            except (MalformedResponse, TransportError):
                return
        raise AssertionError("malformed JSON accepted")
    r.check("malformed JSON fails closed", malformed_json_fails_closed)

    def delayed_billing_is_supplementary():
        from o1_b200.provider.runpod.billing import SpendTracker
        clock = {"t": 0.0}
        st = SpendTracker("5.49", clock=lambda: clock["t"])
        st.mark_pod_started()
        clock["t"] += 3600.0            # 1h -> monotonic spend 5.49
        st.record_live_billing("0.10")  # delayed/lagging billing feed
        assert str(st.effective_spend()) == "5.4900", st.effective_spend()
        st.record_live_billing("39.99")  # live billing ahead of monotonic
        assert str(st.effective_spend()) == "39.99"
    r.check("28. delayed billing never extends the run; the monotonic "
            "watchdog is authoritative", delayed_billing_is_supplementary)

    def redaction_everywhere():
        fake = "rpa_FAKESECRET_abcdef1234567890"
        redaction.register_secret(fake)
        sc = Scenario()
        sc.api_key = fake
        os.environ["RUNPOD_API_KEY"] = fake
        try:
            with MockRunpodServer(sc) as srv:
                ad = RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP,
                                     api_key=fake)
                ad.list_gpu_types()
                sc.fail_next = [(401, 1)]
                try:
                    ad.list_gpu_types()
                except ApiHttpError as exc:
                    assert fake not in str(exc)
                log_text = json.dumps(ad.readonly.request_log)
                assert fake not in log_text, "API key leaked into request log"
                assert fake not in redaction.redact(
                    f"Authorization: Bearer {fake}")
        finally:
            os.environ["RUNPOD_API_KEY"] = "rpa_MOCKKEY_1234567890abcdef"
    r.check("31. API key never appears in logs or exception strings",
            redaction_everywhere)

    def budget_boundaries():
        from decimal import Decimal
        from o1_b200.provider.runpod.billing import (
            projected_session_cost, session_fits_policy)
        assert hard_compute_seconds("5.89") == int(
            Decimal("40.00") / Decimal("5.89") * 3600)
        # below / at / above USD 40
        session_fits_policy("4.00", 35999)                    # 39.9989 < 40
        session_fits_policy("4.00", 36000)                    # exactly 40.00
        try:
            session_fits_policy("4.00", 36001)                # 40.0011 > 40
        except BudgetViolation:
            pass
        else:
            raise AssertionError("cost above USD 40 accepted")
        # above USD 45 total: an hourly rate above the whole budget
        try:
            hard_compute_seconds("45.01")
        except BudgetViolation:
            pass
        else:
            raise AssertionError("rate above USD 45 accepted")
        assert str(projected_session_cost("5.49", 3600)) == "5.4900"
    r.check("budget arithmetic is decimal with boundary tests at/below/above "
            "USD 40 and USD 45", budget_boundaries)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
