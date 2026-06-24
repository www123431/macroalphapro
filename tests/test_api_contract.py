"""tests/test_api_contract.py — API ↔ frontend contract guard.

The frontend's TypeScript interfaces (frontend/lib/api.ts) are hand-mirrored from these
endpoints' JSON. This test pins the response SHAPE so a backend field rename/removal fails CI
here — telling us to update the TS — instead of silently breaking the UI at runtime.

It asserts the keys the frontend actually consumes are present (a permissive superset check, so
adding backend fields never breaks it). Read-only endpoints only; the LLM-touching /api/chat is
exercised via its guard (empty message -> 422), never a real turn (no spend in tests).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _get(path: str) -> dict:
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:200]}"
    return r.json()


def test_health_contract():
    body = _get("/health")
    assert {"status", "service", "version"} <= set(body)


def test_agents_contract():
    body = _get("/api/agents")
    assert {"chief_of_staff", "specialists", "delegation_rule"} <= set(body)
    # AgentCard fields the /agents page reads
    card_keys = {"agent_id", "name", "kind", "role_id", "workload", "spec_ref",
                 "max_iterations", "tools", "scope"}
    assert card_keys <= set(body["chief_of_staff"])
    assert body["chief_of_staff"]["kind"] == "supervisor"
    assert len(body["specialists"]) == 7
    for s in body["specialists"]:
        assert card_keys <= set(s), f"specialist missing keys: {card_keys - set(s)}"
        assert s["kind"] == "specialist"
        assert isinstance(s["tools"], list)


def test_decay_report_contract():
    body = _get("/api/decay/report")
    # DecayReport (frontend) — top-level keys the dashboard/research read
    assert {"as_of", "overall", "realloc_action", "n_mechanisms",
            "mechanisms", "pairs", "alarms"} <= set(body)
    assert body["overall"] in {"HEALTHY", "WATCH", "ACTION"}
    if body["mechanisms"]:
        m = next(iter(body["mechanisms"].values()))
        assert {"role", "weight", "full_sharpe", "rolling_sharpe",
                "crisis_payoff", "signal_ic", "structural_decay"} <= set(m)
    for p in body["pairs"][:1]:
        assert {"pair", "rolling_corr", "downside_corr", "stress_corr"} <= set(p)


def test_book_state_contract():
    body = _get("/api/book/state")
    assert {"as_of", "strategies", "combined_gross", "combined_net",
            "combined_n", "sleeve_attribution"} <= set(body)
    assert isinstance(body["strategies"], list)


def test_brief_contract():
    body = _get("/api/brief")
    assert "as_of" in body  # may be null if no snapshot, but key present


def test_book_perf_contract():
    body = _get("/api/book/perf")
    assert {"n_weeks", "start", "end", "stats", "dates", "equity", "drawdown", "rolling_sharpe"} <= set(body)
    assert {"ann_ret", "ann_vol", "sharpe", "max_dd"} <= set(body["stats"])
    assert len(body["dates"]) == len(body["equity"]) == len(body["drawdown"])


def test_book_positions_contract():
    body = _get("/api/book/positions")
    assert {"as_of", "n", "n_long", "n_short", "gross", "net", "biggest", "positions"} <= set(body)
    for p in body["positions"][:1]:
        assert {"ticker", "weight", "side", "strategies"} <= set(p)
        assert p["side"] in {"long", "short"}


def test_book_overlay_contract():
    body = _get("/api/book/overlay")
    assert {"as_of", "positions", "gross", "net", "n", "single_name_cap", "gross_cap"} <= set(body)
    assert isinstance(body["positions"], list)
    for p in body["positions"][:1]:
        assert {"ticker", "weight"} <= set(p)
    assert "recent_trades" in body and isinstance(body["recent_trades"], list)


def test_book_tracking_contract():
    body = _get("/api/book/tracking")
    assert "available" in body
    if body.get("available"):
        assert {"n_live_days", "live", "backtest_expected", "tracking", "significant",
                "min_days_for_significance"} <= set(body)
        assert {"ann_ret", "ann_vol", "cum_return"} <= set(body["live"])
        assert isinstance(body["significant"], bool)


def test_book_combined_contract():
    body = _get("/api/book/combined")
    assert "available" in body
    if body.get("available"):
        assert {"carry_risk_weight", "combined", "equity_only", "dates", "equity_curve"} <= set(body)
        assert {"sharpe", "ann", "vol", "maxdd", "n"} <= set(body["combined"])
        assert len(body["dates"]) == len(body["equity_curve"])


def test_risk_contrib_contract():
    body = _get("/api/book/risk-contrib")
    assert "available" in body
    if body.get("available"):
        assert {"port_vol_annual", "coverage", "contributions"} <= set(body)
        assert {"n_covered", "n_total", "weight_covered"} <= set(body["coverage"])
        for c in body["contributions"][:1]:
            assert {"ticker", "weight", "pct_risk", "vol_annual"} <= set(c)


def test_factor_exposure_contract():
    body = _get("/api/book/factor-exposure")
    assert "available" in body
    if body.get("available"):
        assert {"r2", "idiosyncratic", "factors"} <= set(body)
        for f in body["factors"][:1]:
            assert {"factor", "beta", "risk_share"} <= set(f)


def test_scenarios_contract():
    body = _get("/api/book/scenarios")
    assert "available" in body
    if body.get("available"):
        assert {"worst", "best", "worst_day", "period"} <= set(body)
        assert {"1d", "5d", "20d"} <= set(body["worst"])
        if body.get("market"):
            assert {"book_beta", "shocks"} <= set(body["market"])


def test_book_dates_and_timetravel_contract():
    body = _get("/api/book/dates")
    assert {"dates", "latest"} <= set(body)
    assert isinstance(body["dates"], list)
    # as_of param is accepted on the artifact-backed views (time-travel)
    if body["dates"]:
        d = body["dates"][0]
        assert client.get(f"/api/book/state?as_of={d}").status_code == 200
        assert client.get(f"/api/book/positions?as_of={d}").status_code == 200
        assert client.get(f"/api/risk?as_of={d}").status_code == 200


def test_book_nav_contract():
    body = _get("/api/book/nav?days_back=120")
    assert "n_rows" in body
    if body.get("days"):
        d = body["days"][0]
        assert {"date", "nav_close", "daily_dietz", "external_flow"} <= set(d)


def test_pit_audit_contract():
    body = _get("/api/research/pit-audit")
    assert "available" in body
    if body.get("available"):
        assert body.get("book") or body.get("dpead")
        if body.get("book"):
            assert {"overall", "book_clean", "surfaces"} <= set(body["book"])
        if body.get("dpead"):
            assert {"overall", "critical_pass", "checks"} <= set(body["dpead"])


def test_gate_runs_contract():
    body = _get("/api/research/gate-runs")
    assert {"n", "runs"} <= set(body)
    assert isinstance(body["runs"], list)
    for r in body["runs"][:1]:
        assert {"name", "verdict", "n_months", "n_trials", "deflated_sr"} <= set(r)


def test_graveyard_contract():
    body = _get("/api/research/graveyard")
    assert {"as_of", "note", "entries"} <= set(body)
    assert len(body["entries"]) >= 1
    for e in body["entries"]:
        assert {"name", "family", "date", "verdict", "why"} <= set(e)


def test_ops_cost_contract():
    body = _get("/api/ops/cost")
    assert {"as_of", "today_usd", "last7_usd", "last30_usd", "lifetime_usd",
            "calls_total", "by_agent", "by_provider"} <= set(body)
    for a in body["by_agent"][:1]:
        assert {"agent_id", "total_usd", "calls", "last_ts", "providers"} <= set(a)
    for p in body["by_provider"][:1]:
        assert {"provider", "total_usd"} <= set(p)


def test_approvals_contract():
    body = _get("/api/approvals")
    assert {"n_pending", "approvals", "charter"} <= set(body)
    # charter (governance-queue routing) — both langs, the load-bearing claim present
    assert {"en", "zh"} <= set(body["charter"])
    assert "HARD-HALT" in body["charter"]["en"]
    # each item carries an effect (what Approve actually does) the UI renders
    for a in body["approvals"][:1]:
        assert {"effect_en", "effect_zh", "executes"} <= set(a)
        assert isinstance(a["executes"], bool)
    # resolve guards — never mutates a real row here
    assert client.post("/api/approvals/resolve", json={"ids": [], "approved": True, "rationale": "x"}).status_code == 422
    assert client.post("/api/approvals/resolve", json={"ids": [1], "approved": True, "rationale": "  "}).status_code == 422


def test_approval_detail_contract():
    # Unknown id -> 404 (server is up, route exists, row missing).
    assert client.get("/api/approvals/999999999").status_code == 404
    # Real row (if any pending) -> the deterministic decision-context shape the review page reads.
    listing = _get("/api/approvals")
    if not listing["approvals"]:
        pytest.skip("no approval rows to drill into")
    aid = listing["approvals"][0]["id"]
    body = _get(f"/api/approvals/{aid}")
    assert {"found", "approval_id", "base", "decision_context",
            "similar_past", "decision_replay", "review_categories"} <= set(body)
    assert body["found"] is True
    assert {"approval_id", "approval_type", "ticker"} <= set(body["base"])
    assert isinstance(body["decision_replay"], list)
    assert isinstance(body["review_categories"], list) and body["review_categories"]


def test_dq_contract():
    body = _get("/api/dq")
    assert "verdict" in body
    if body.get("verdict") not in ("UNKNOWN", None):
        assert {"as_of", "n_breaches", "checks", "rationale", "scope"} <= set(body)
        assert body["verdict"] in {"CLEAN", "WARN", "HALT"}
        assert isinstance(body["checks"], list)


def test_risk_contract():
    body = _get("/api/risk")
    assert {"as_of", "overall_severity", "halt", "n_breaches", "metrics", "modes"} <= set(body)
    assert {"gross", "net", "hhi", "max_weight", "short_ratio", "n_ok"} <= set(body["metrics"])
    assert len(body["modes"]) >= 11
    for r in body["modes"][:1]:
        assert {"mode_id", "name", "observed", "threshold", "verdict", "live"} <= set(r)


def test_alerts_contract():
    body = _get("/api/alerts?days_back=30")
    assert {"as_of", "days_back", "n_alerts", "alerts", "n_anomalies", "anomalies"} <= set(body)
    for a in body["alerts"][:1]:
        assert {"source", "date", "severity", "rule_description"} <= set(a)
    for a in body["anomalies"][:1]:
        assert {"scan_date", "ticker", "detector", "confidence_likert", "evidence"} <= set(a)


def test_ops_health_contract():
    body = _get("/api/ops/health")
    assert {"slo", "providers", "governance"} <= set(body)
    if "error" not in body["governance"]:
        assert {"clean", "agents", "eval_cases", "posture"} <= set(body["governance"])
    if "error" not in body["slo"]:
        assert {"n", "success_rate", "p50_ms", "p95_ms", "by_agent"} <= set(body["slo"])


def test_provenance_contract():
    body = _get("/api/provenance")
    assert {"as_of", "sources", "point_in_time"} <= set(body)
    for s in body["sources"][:1]:
        assert {"source", "kind"} <= set(s)


def test_freshness_contract():
    body = _get("/api/freshness")
    assert {"as_of", "sources", "overall", "worst_age_days"} <= set(body)
    assert body["overall"] in {"fresh", "stale"}
    for s in body["sources"]:
        assert {"source", "as_of", "age_days", "threshold_days", "stale"} <= set(s)


def test_ops_refresh_status_contract():
    # GET status only — POST would launch the real (minutes-long, mutating) daily job.
    body = _get("/api/ops/refresh")
    assert {"running", "exit_code", "ok", "message", "log_tail"} <= set(body)
    assert isinstance(body["running"], bool)


def test_ops_eval_contract():
    # eval-latest read (free) + eval-run STATUS only (POST would spend LLM $).
    body = _get("/api/ops/eval-latest")
    assert "found" in body
    if body["found"]:
        assert "static_all_pass" in body
        if body.get("live"):
            assert {"pass_rate", "runs", "total_cost_usd", "cases"} <= set(body["live"])
    st = _get("/api/ops/eval-run")
    assert {"running", "exit_code", "ok"} <= set(st)
    assert isinstance(st["running"], bool)


def test_metrics_and_request_id():
    r = client.get("/health")
    assert r.headers.get("x-request-id"), "every response should carry X-Request-ID"
    body = _get("/api/metrics")
    assert {"routes", "total_requests"} <= set(body)


def test_chat_guard_empty_message_422():
    # Guard path only — never triggers a real (billable) CoS turn.
    r = client.post("/api/chat", json={"message": "   ", "history": []})
    assert r.status_code == 422
