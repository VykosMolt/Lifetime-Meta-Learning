"""RunPod adapter: schema pin, dual-profile catalog/spot-quote validation,
fallback honesty, retries, redaction."""
from __future__ import annotations

import json
import os

from _h import Runner, fresh_dir, hermetic_mock_credentials

MOCK_KEY = hermetic_mock_credentials()

from o1_b200.provider.runpod import redaction
from o1_b200.provider.runpod.adapter import RunpodV2Adapter
from o1_b200.provider.runpod.billing import BudgetViolation, hard_compute_seconds
from o1_b200.provider.runpod.mock_server import (
    MockRunpodServer, Scenario, _b200, _b300,
)
from o1_b200.provider.runpod.models import SchemaIncompatibility
from o1_b200.provider.runpod.policy import PolicyViolation
from o1_b200.provider.runpod.quote import QuoteError
from o1_b200.provider.runpod.schema_check import SchemaPinError, verify_pinned_schema
from o1_b200.provider.runpod.transport import ApiHttpError, MalformedResponse, TransportError

NOSLEEP = lambda s: None  # noqa: E731


def _adapter(srv, **kw):
    kw.setdefault("api_key", MOCK_KEY)
    return RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP, **kw)


def run() -> Runner:
    r = Runner("runpod_adapter")

    r.check("pinned OpenAPI snapshot verifies byte-exact with sane surfaces",
            verify_pinned_schema)

    def primary_b300_spot_quote():
        with MockRunpodServer() as srv:
            ad = _adapter(srv)
            q = ad.quote_instance(adapter_commit="test")
            assert q["gpu_type_id"] == "NVIDIA B300 SXM6 AC", q["gpu_type_id"]
            assert q["profile"] == "B300" and q["profile_role"] == "PRIMARY"
            assert q["fallback_reason"] is None
            assert q["cloud"] == "SECURE" and q["gpu_count"] == 1
            assert q["interruptible"] is True
            assert q["purchase_mode"] == "INTERRUPTIBLE"
            assert q["bid_per_gpu_usd"] == "7.89"
            assert ad.validate_quote()["hard_compute_seconds"] == \
                hard_compute_seconds("7.89")
    r.check("1. qualifying Secure single-B300 INTERRUPTIBLE quote accepted "
            "(primary, live spot bid, no fallback)", primary_b300_spot_quote)

    def explicit_b200_fallback():
        sc = Scenario()
        sc.gpus = [_b300(availability="NONE",
                         dcs=[{"id": "EU-NL-1", "availability": "NONE"}]),
                   _b200()]
        with MockRunpodServer(sc) as srv:
            q = _adapter(srv).quote_instance()
            assert q["profile"] == "B200" and q["profile_role"] == "FALLBACK"
            assert q["gpu_type_id"] == "NVIDIA B200"
            assert q["fallback_reason"] and "B300" in q["fallback_reason"], \
                "fallback must record WHY the primary was refused"
            assert "NONE" in q["profile_refusals"]["B300"]
            assert q["interruptible"] is True
    r.check("2. B200 fallback is explicit and honest: selected only when "
            "B300 is refused, with the exact recorded reason",
            explicit_b200_fallback)

    def b300_absent_entirely():
        sc = Scenario()
        sc.gpus = [_b200()]
        with MockRunpodServer(sc) as srv:
            q = _adapter(srv).quote_instance()
            assert q["profile"] == "B200"
            assert "no catalog entry" in q["profile_refusals"]["B300"]
    r.check("3. B300 vanished from the catalog -> explicit fallback, "
            "refusal recorded", b300_absent_entirely)

    def neither_profile_available():
        sc = Scenario()
        sc.gpus = [_b300(availability="NONE",
                         dcs=[{"id": "EU-NL-1", "availability": "NONE"}]),
                   _b200(availability="NONE",
                         dcs=[{"id": "US-KS-2", "availability": "NONE"}])]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "B300" in str(exc) and "B200" in str(exc)
                return
        raise AssertionError("no-capacity state did not fail closed")
    r.check("4. both profiles unavailable -> refused with both reasons",
            neither_profile_available)

    def unaffordable_rate_refused():
        sc = Scenario()
        sc.gpus = [_b300(price_secure=25.00)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except (PolicyViolation, BudgetViolation, QuoteError) as exc:
                assert "viable" in str(exc) or "budget" in str(exc)
                return
        raise AssertionError("a rate the budget cannot afford was accepted")
    r.check("5. mechanically unaffordable spot rate refused (no static "
            "price cap; budget-derived viability rule)", unaffordable_rate_refused)

    def spot_capacity_vanished():
        sc = Scenario()
        sc.spot["NVIDIA B300 SXM6 AC"] = {"secureSpotPrice": None,
                                          "stockStatus": None}
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "interruptible" in str(exc) or "spot" in str(exc)
                return
        raise AssertionError("spot-vanished B300 quote did not fail closed")
    r.check("6. interruptible capacity vanished (spot rate null) fails "
            "closed at quote time", spot_capacity_vanished)

    def spot_price_change_reflected():
        sc = Scenario()
        sc.spot["NVIDIA B300 SXM6 AC"] = {"secureSpotPrice": 6.50}
        with MockRunpodServer(sc) as srv:
            ad = _adapter(srv)
            assert ad.quote_instance()["bid_per_gpu_usd"] == "6.5"
            sc.spot["NVIDIA B300 SXM6 AC"] = {"secureSpotPrice": 7.10}
            assert ad.quote_instance()["bid_per_gpu_usd"] == "7.1", \
                "a fresh quote must reflect the changed live spot price"
    r.check("7. changing interruptible price is re-derived on every fresh "
            "quote (no caching)", spot_price_change_reflected)

    def minimum_bid_respected():
        sc = Scenario()
        sc.spot["NVIDIA B300 SXM6 AC"] = {"secureSpotPrice": 6.00,
                                          "minimumBidPrice": 6.75}
        with MockRunpodServer(sc) as srv:
            q = _adapter(srv).quote_instance()
            assert q["bid_per_gpu_usd"] == "6.75", q["bid_per_gpu_usd"]
    r.check("8. market minimum bid above spot rate raises the bid (still "
            "budget-validated)", minimum_bid_respected)

    def community_only():
        sc = Scenario()
        sc.gpus = [_b300(secure=False), _b200(secure=False)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "Community" in str(exc) or "Secure" in str(exc)
                return
        raise AssertionError("Community-only offer accepted")
    r.check("9. Community-only availability refused for both profiles",
            community_only)

    def wrong_memory_refused():
        sc = Scenario()
        sc.gpus = [_b300(memory=180)]   # not the intended 288 GB part
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "B300" in str(exc)
                return
        raise AssertionError("wrong-HBM B300 variant accepted")
    r.check("10. B300 variant below the 275 GB HBM floor refused",
            wrong_memory_refused)

    def multi_gpu_only_refused():
        sc = Scenario()
        sc.gpus = [_b300(max_secure=0), _b200(max_secure=0)]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).quote_instance()
            except QuoteError as exc:
                assert "maxCount" in str(exc) or "single" in str(exc)
                return
        raise AssertionError("no-single-GPU offer accepted")
    r.check("11. cluster/multi-GPU-minimum offer refused (secure maxCount<1)",
            multi_gpu_only_refused)

    def alternate_gpu_never_substituted():
        for sub in ("NVIDIA H200", "NVIDIA H100 80GB HBM3",
                    "NVIDIA GB300 NVL72", "NVIDIA RTX PRO 6000 Blackwell"):
            sc = Scenario()
            sc.gpus = [_b200(gpu_id=sub, name=sub, memory=288)]
            with MockRunpodServer(sc) as srv:
                try:
                    _adapter(srv).quote_instance()
                except (QuoteError, PolicyViolation):
                    continue
            raise AssertionError(f"substitute {sub} accepted")
    r.check("12. H100/H200/GB300/RTX-PRO never substituted under either "
            "profile", alternate_gpu_never_substituted)

    def quote_expiry():
        clock = {"t": 1000.0}
        with MockRunpodServer() as srv:
            ad = RunpodV2Adapter(base_url=srv.base_url, sleep=NOSLEEP,
                                 api_key=MOCK_KEY, clock=lambda: clock["t"])
            ad.quote_instance()
            ad.validate_quote()
            clock["t"] += 15 * 60 + 1
            try:
                ad.validate_quote()
            except QuoteError as exc:
                assert "expired" in str(exc)
                return
        raise AssertionError("16-minute-old quote accepted")
    r.check("13. quote expires after the 15-minute validity interval",
            quote_expiry)

    def catalog_schema_drift():
        sc = Scenario()
        sc.gpus = [{k: v for k, v in _b300().items() if k != "price"}]
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).list_gpu_types()
            except SchemaIncompatibility as exc:
                assert "price" in str(exc)
                return
        raise AssertionError("missing pricing field accepted")
    r.check("14. catalog schema drift (missing pricing) fails closed",
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
    r.check("15. live schema identity check fails closed on enum/path drift",
            openapi_drift_detected)

    def rest_interruptible_drift_fails_closed():
        # the pinned REST contract has no interruptible surface; its
        # appearance there is drift whose semantics must be re-audited
        from o1_b200.provider.runpod.schema_check import check_live_identity, load_pinned_schema
        live = load_pinned_schema()
        live["components"]["schemas"]["CreatePodRequest"]["allOf"].append(
            {"properties": {"interruptible": {"type": "boolean"}}})
        try:
            check_live_identity(live)
        except SchemaPinError as exc:
            assert "interruptible" in str(exc)
            return
        raise AssertionError("REST interruptible surface drift accepted")
    r.check("16. an interruptible field appearing in the REST contract is "
            "drift and fails closed (spot stays on the pinned GraphQL "
            "surface)", rest_interruptible_drift_fails_closed)

    def graphql_contract_check():
        with MockRunpodServer() as srv:
            rep = _adapter(srv).graphql.verify_contract()
            assert rep["query_surface"] == "LIVE_VERIFIED"
            assert "bidPerGpu" in rep["pinned_rent_input_fields"]
    r.check("17. GraphQL spot contract: query surface live-verified, "
            "mutation surface pinned fail-closed", graphql_contract_check)

    def graphql_readonly_enforced():
        from o1_b200.provider.runpod.graphql_spot import RunpodGraphQlClient
        from o1_b200.provider.runpod.transport import ReadOnlyViolation
        with MockRunpodServer() as srv:
            cl = RunpodGraphQlClient(url=srv.base_url + "/graphql",
                                     api_key=MOCK_KEY, sleep=NOSLEEP)
            for doc in ("mutation { podRentInterruptable(input: {}) { id } }",
                        "{ gpuTypes { id } }",
                        "subscription { x }"):
                try:
                    cl.query(doc)
                except ReadOnlyViolation:
                    continue
                raise AssertionError(f"non-query document accepted: {doc!r}")
            try:
                cl.mutate("mutation { podRentInterruptable(input: {}) { id } }")
            except ReadOnlyViolation as exc:
                assert "LIVE_MUTATION_NOT_AUTHORIZED" in str(exc)
                return
            raise AssertionError("unauthorized GraphQL mutation accepted")
    r.check("18. GraphQL read-only path refuses every non-query document "
            "BEFORE network; mutation requires authorization",
            graphql_readonly_enforced)

    def unknown_lifecycle_state():
        sc = Scenario()
        sc.unknown_status = "HIBERNATING"
        with MockRunpodServer(sc) as srv:
            ad = _adapter(srv)
            sc.pods["p1"] = {"id": "p1", "name": "x", "image": "i",
                             "env": {}, "gpu": {"id": "NVIDIA B300 SXM6 AC",
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
    r.check("19. unknown lifecycle state fails closed", unknown_lifecycle_state)

    def rate_limit_429():
        sc = Scenario()
        sc.rate_limit_next = 2
        with MockRunpodServer(sc) as srv:
            ad = _adapter(srv)
            gpus = ad.list_gpu_types()   # retried through the 429s
            assert gpus[0].id == "NVIDIA B300 SXM6 AC"
    r.check("20. transient 429 with Retry-After is retried then succeeds",
            rate_limit_429)

    def transient_5xx():
        sc = Scenario()
        sc.fail_next = [(503, 2)]
        with MockRunpodServer(sc) as srv:
            assert _adapter(srv).list_gpu_types()[0].id == \
                "NVIDIA B300 SXM6 AC"
    r.check("21. transient 5xx retried with backoff then succeeds",
            transient_5xx)

    def graphql_transient_5xx():
        sc = Scenario()
        sc.graphql_fail_next = 2
        with MockRunpodServer(sc) as srv:
            spot = _adapter(srv).graphql.spot_pricing("NVIDIA B300 SXM6 AC")
            assert spot["secure_spot_usd"] == 7.89
    r.check("22. GraphQL transient 5xx retried (queries only) then succeeds",
            graphql_transient_5xx)

    def malformed_json_fails_closed():
        sc = Scenario()
        sc.malformed_next = 10   # exhaust retries
        with MockRunpodServer(sc) as srv:
            try:
                _adapter(srv).list_gpu_types()
            except (MalformedResponse, TransportError):
                return
        raise AssertionError("malformed JSON accepted")
    r.check("23. malformed JSON fails closed", malformed_json_fails_closed)

    def delayed_billing_is_supplementary():
        from o1_b200.provider.runpod.billing import SpendTracker
        clock = {"t": 0.0}
        st = SpendTracker("5.49", clock=lambda: clock["t"])
        st.mark_pod_started()
        clock["t"] += 3600.0            # 1h -> monotonic spend 5.49
        st.record_live_billing("0.10")  # delayed/lagging billing feed
        assert str(st.effective_spend()) == "5.4900", st.effective_spend()
        st.record_live_billing("39.99")  # live billing ahead of monotonic
        assert str(st.effective_spend()) == "39.9900"
    r.check("24. delayed billing never extends the run; the monotonic "
            "watchdog is authoritative", delayed_billing_is_supplementary)

    def spend_carries_across_eviction():
        from o1_b200.provider.runpod.billing import SpendTracker
        clock = {"t": 0.0}
        st = SpendTracker("6.00", clock=lambda: clock["t"])
        st.mark_pod_started()
        clock["t"] += 1800.0             # 30 min -> 3.00
        carried = st.mark_pod_stopped()
        assert str(carried) == "3.0000"
        st2 = SpendTracker("8.00", clock=lambda: clock["t"],
                           carryover_usd=carried)
        st2.mark_pod_started()
        clock["t"] += 1800.0             # 30 min @8 -> 4.00; total 7.00
        assert str(st2.effective_spend()) == "7.0000"
    r.check("25. session spend carries across eviction/reacquisition; a "
            "fresh pod can never reset the budget", spend_carries_across_eviction)

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
                log_text = json.dumps(ad.readonly.request_log
                                      + ad.graphql.request_log)
                assert fake not in log_text, "API key leaked into request log"
                assert fake not in redaction.redact(
                    f"Authorization: Bearer {fake}")
        finally:
            os.environ["RUNPOD_API_KEY"] = MOCK_KEY
    r.check("26. API key never appears in logs or exception strings",
            redaction_everywhere)

    def budget_boundaries():
        from decimal import Decimal
        from o1_b200.provider.runpod.billing import (
            projected_session_cost, session_fits_policy)
        assert hard_compute_seconds("7.89") == int(
            Decimal("40.00") / Decimal("7.89") * 3600)
        session_fits_policy("4.00", 35999)                    # 39.9989 < 40
        session_fits_policy("4.00", 36000)                    # exactly 40.00
        try:
            session_fits_policy("4.00", 36001)                # 40.0011 > 40
        except BudgetViolation:
            pass
        else:
            raise AssertionError("cost above USD 40 accepted")
        try:
            hard_compute_seconds("45.01")
        except BudgetViolation:
            pass
        else:
            raise AssertionError("rate above USD 45 accepted")
        assert str(projected_session_cost("5.49", 3600)) == "5.4900"
    r.check("27. budget arithmetic is decimal with boundary tests at/below/"
            "above USD 40 and USD 45", budget_boundaries)

    def hf_result_uri_parsing():
        from o1_b200.provider.runpod.adapter import (
            RunpodAdapterError, parse_hf_uri)
        assert parse_hf_uri("hf://ns/repo/a/b/c.tar.gz") == \
            ("ns/repo", "a/b/c.tar.gz")
        for bad in ("hf://x", "hf://x/y", "hf:///y/z", "hf://x//z"):
            try:
                parse_hf_uri(bad)
            except RunpodAdapterError:
                continue
            raise AssertionError(f"malformed result source accepted: {bad}")
    r.check("28. hf:// result-destination URIs parse strictly; malformed "
            "refused", hf_result_uri_parsing)

    return r


if __name__ == "__main__":
    raise SystemExit(run().report())
