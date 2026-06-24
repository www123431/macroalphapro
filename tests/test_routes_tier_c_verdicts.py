"""tests/test_routes_tier_c_verdicts.py — L3-2 endpoint tests.

GET /api/research/tier_c_verdicts — unified Tier-C verdict feed
including L3-2 self_doubt assessment.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def _ev(**kw):
    base = dict(
        event_id="ev_default",
        event_type="factor_verdict_filed",
        subject_type="factor",
        subject_id="auto_default",
        family="PROFITABILITY",
        ts="2026-06-08T12:00:00Z",
        verdict="GREEN",
        summary="default summary",
        metrics={},
        parent_event_ids=(),
        artifacts=(),
        tags=("tier_c_auto",),
        actor="engine.test",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _full_metrics(with_self_doubt=True, with_replication=True,
                     with_anchor_ortho=False):
    m = {
        "auto_test_spec_hash":  "hash_abc",
        "signal_kind":          "cross_sectional_rank",
        "template_version":     "v1.1_2026-06-08",
        "sharpe":               0.67,
        "nw_t_stat":            3.57,
        "n_months":             395,
        "avg_turnover":         0.17,
        "cost_robust_verdict":  "GREEN",
    }
    if with_replication:
        m["replication"] = {
            "status": "REPLICATED", "our_t": 3.04,
            "paper_reported_t": 3.0, "t_gap": 0.044,
        }
    if with_self_doubt:
        m["self_doubt"] = {
            "confidence":              0.61,
            "confidence_reason":       "GREEN replicated but B2/B4 unresolved",
            "caveats":                 ["B2 survivorship",
                                        "B4 EW-only"],
            "methodological_concerns": ["B2 PARTIAL"],
            "suspicious_metrics":      ["our_t > paper_t suspicious"],
            "assessment_ts":           "2026-06-08T14:18:25Z",
            "model":                   "claude-sonnet-4-6",
        }
    if with_anchor_ortho:
        m["anchor_orthogonality"] = {
            "alpha_monthly":  0.0026, "alpha_annual": 0.0316,
            "alpha_nw_t":     1.88,   "alpha_nw_se": 0.0014,
            "betas":          {"RMW": 0.67, "HML": -0.35},
            "beta_nw_t":      {"RMW": 10.04, "HML": -4.95},
            "r2":             0.252, "r2_adj": 0.241,
            "n_overlap":      395, "anchor_names": ["MKT_RF", "SMB", "HML",
                                                       "RMW", "CMA", "MOM"],
            "nw_lag_used":    5, "window": "1992-02:2024-12",
            "anchor_library": "ken_french_ff5_mom",
        }
    return m


# ────────────────────────────────────────────────────────────────────
# Happy path: mix of GREEN/MARGINAL/RED with self_doubt
# ────────────────────────────────────────────────────────────────────
def test_returns_all_three_verdicts_by_default(monkeypatch):
    from engine.research_store import store as st
    events = [
        _ev(event_id="ev_g", verdict="GREEN",
             ts="2026-06-08T12:00:00Z",
             metrics=_full_metrics()),
        _ev(event_id="ev_m", verdict="MARGINAL",
             ts="2026-06-07T12:00:00Z",
             metrics=_full_metrics(with_self_doubt=False)),
        _ev(event_id="ev_r", verdict="RED",
             ts="2026-06-06T12:00:00Z",
             metrics=_full_metrics(with_replication=False)),
    ]
    monkeypatch.setattr(st, "filter_events", lambda **kw: list(events))
    r = client.get("/api/research/tier_c_verdicts?days=30")
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] == 3
    verdicts = [it["verdict"] for it in body["items"]]
    assert set(verdicts) == {"GREEN", "MARGINAL", "RED"}
    # Newest first
    assert body["items"][0]["verdict"] == "GREEN"


def test_self_doubt_payload_passed_through(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(event_id="ev_g", verdict="GREEN", metrics=_full_metrics()),
    ])
    r = client.get("/api/research/tier_c_verdicts")
    assert r.status_code == 200
    item = r.json()["items"][0]
    sd = item["self_doubt"]
    assert sd is not None
    assert sd["confidence"] == 0.61
    assert sd["model"] == "claude-sonnet-4-6"
    assert len(sd["caveats"]) == 2
    assert "B2 survivorship" in sd["caveats"]


def test_self_doubt_null_when_absent(monkeypatch):
    """Pre-L3-2 verdicts have no self_doubt in metrics → null."""
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(event_id="ev_old", verdict="GREEN",
             metrics=_full_metrics(with_self_doubt=False)),
    ])
    body = client.get("/api/research/tier_c_verdicts").json()
    assert body["items"][0]["self_doubt"] is None


def test_replication_null_when_absent(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="RED",
             metrics=_full_metrics(with_replication=False)),
    ])
    body = client.get("/api/research/tier_c_verdicts").json()
    assert body["items"][0]["replication"] is None


# ────────────────────────────────────────────────────────────────────
# verdicts query param filter
# ────────────────────────────────────────────────────────────────────
def test_verdicts_filter_green_only(monkeypatch):
    from engine.research_store import store as st
    events = [
        _ev(event_id="ev_g", verdict="GREEN", metrics=_full_metrics()),
        _ev(event_id="ev_r", verdict="RED", metrics=_full_metrics()),
    ]
    monkeypatch.setattr(st, "filter_events", lambda **kw: list(events))
    body = client.get("/api/research/tier_c_verdicts?verdicts=GREEN").json()
    assert body["n_total"] == 1
    assert body["items"][0]["verdict"] == "GREEN"
    assert body["verdicts"] == ["GREEN"]


def test_verdicts_filter_lowercase_accepted(monkeypatch):
    from engine.research_store import store as st
    events = [
        _ev(event_id="ev_g", verdict="GREEN", metrics=_full_metrics()),
        _ev(event_id="ev_r", verdict="RED", metrics=_full_metrics()),
    ]
    monkeypatch.setattr(st, "filter_events", lambda **kw: list(events))
    body = client.get(
        "/api/research/tier_c_verdicts?verdicts=green,red").json()
    assert body["n_total"] == 2


# ────────────────────────────────────────────────────────────────────
# tier_c_auto tag filter — non-tier-c verdicts NOT included
# ────────────────────────────────────────────────────────────────────
def test_non_tier_c_events_filtered_out(monkeypatch):
    """A factor_verdict_filed event WITHOUT tier_c_auto tag (e.g.
    legacy strict-gate verdicts) must NOT appear."""
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(event_id="ev_legacy", verdict="GREEN",
             tags=("strict_gate", "manual")),
        _ev(event_id="ev_tier_c", verdict="GREEN",
             tags=("tier_c_auto", "cross_sectional_rank")),
    ])
    body = client.get("/api/research/tier_c_verdicts").json()
    assert body["n_total"] == 1
    assert body["items"][0]["event_id"] == "ev_tier_c"


# ────────────────────────────────────────────────────────────────────
# Empty / validation
# ────────────────────────────────────────────────────────────────────
def test_empty_window(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [])
    body = client.get("/api/research/tier_c_verdicts").json()
    assert body["n_total"] == 0
    assert body["items"] == []


def test_rejects_invalid_days():
    assert client.get(
        "/api/research/tier_c_verdicts?days=0").status_code == 422


def test_rejects_invalid_limit():
    assert client.get(
        "/api/research/tier_c_verdicts?limit=99999").status_code == 422


# ────────────────────────────────────────────────────────────────────
# Schema stability
# ────────────────────────────────────────────────────────────────────
def test_response_schema(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN", metrics=_full_metrics()),
    ])
    body = client.get("/api/research/tier_c_verdicts").json()
    assert {"since", "verdicts", "n_total", "n_returned", "items"} \
        <= set(body.keys())
    item = body["items"][0]
    required = {
        "event_id", "subject_id", "family", "verdict", "verdict_ts",
        "summary", "signal_kind", "spec_hash", "template_version",
        "sharpe", "nw_t_stat", "n_months", "avg_turnover",
        "cost_robust_verdict", "replication", "self_doubt",
        "pnl_series_parquet", "anchor_orthogonality",
        "subsample_stability", "industry_extension",
    }
    assert required <= set(item.keys())


# ────────────────────────────────────────────────────────────────────
# L2-4 Commit 3: anchor_orthogonality in endpoint response
# ────────────────────────────────────────────────────────────────────
def test_anchor_orthogonality_pass_through(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN",
             metrics=_full_metrics(with_anchor_ortho=True)),
    ])
    item = client.get("/api/research/tier_c_verdicts").json()["items"][0]
    ao = item["anchor_orthogonality"]
    assert ao is not None
    assert ao["anchor_library"] == "ken_french_ff5_mom"
    assert ao["alpha_nw_t"] == 1.88
    assert ao["betas"]["RMW"] == 0.67
    assert ao["beta_nw_t"]["RMW"] == 10.04
    assert ao["r2"] == 0.252


def test_subsample_stability_pass_through(monkeypatch):
    from engine.research_store import store as st
    fake_ss = {
        "n_splits": 4, "n_total_months": 395, "windows": [],
        "worst_best_sharpe_ratio": 0.174,
        "institutional_stable": False,
        "monotone_decay": False, "monotone_growth": False,
        "decay_slope_per_year": 0.0001, "decay_slope_t": 0.4,
    }
    m = _full_metrics(with_anchor_ortho=True)
    m["subsample_stability"] = fake_ss
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN", metrics=m),
    ])
    item = client.get("/api/research/tier_c_verdicts").json()["items"][0]
    ss = item["subsample_stability"]
    assert ss is not None
    assert ss["n_splits"] == 4
    assert ss["worst_best_sharpe_ratio"] == 0.174
    assert ss["institutional_stable"] is False


def test_investment_role_filter_legacy_default_alpha(monkeypatch):
    """Phase 1 Commit 7: pre-v2 events (no investment_role in metrics)
    are matched by ?investment_role=alpha (legacy fallback)."""
    from engine.research_store import store as st
    # Two events: one pre-v2 (no investment_role), one explicit insurance
    pre_v2 = _ev(event_id="ev_legacy", verdict="GREEN",
                   metrics=_full_metrics())
    explicit_insurance = _ev(
        event_id="ev_insurance",
        verdict="GREEN",
        metrics={**_full_metrics(), "investment_role": "insurance"},
    )
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: [pre_v2, explicit_insurance])
    # alpha filter → pre-v2 (legacy default) matches
    items = client.get(
        "/api/research/tier_c_verdicts?investment_role=alpha"
    ).json()["items"]
    ids = {it["event_id"] for it in items}
    assert "ev_legacy" in ids
    assert "ev_insurance" not in ids


def test_investment_role_filter_explicit_insurance(monkeypatch):
    from engine.research_store import store as st
    pre_v2 = _ev(event_id="ev_legacy", verdict="GREEN",
                   metrics=_full_metrics())
    explicit_insurance = _ev(
        event_id="ev_insurance",
        verdict="GREEN",
        metrics={**_full_metrics(), "investment_role": "insurance"},
    )
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: [pre_v2, explicit_insurance])
    items = client.get(
        "/api/research/tier_c_verdicts?investment_role=insurance"
    ).json()["items"]
    ids = {it["event_id"] for it in items}
    assert ids == {"ev_insurance"}


def test_industry_extension_pass_through(monkeypatch):
    from engine.research_store import store as st
    fake_ix = {
        "alpha_full_monthly": -0.0019, "alpha_full_annual": -0.0234,
        "alpha_full_nw_t": -1.38, "alpha_full_nw_se": 0.0014,
        "alpha_ff5mom_only_nw_t": 1.88,
        "delta_alpha_nw_t_approx": 3.26,
        "industry_betas": {"BusEq": 0.54, "Shops": 0.34},
        "industry_beta_nw_t": {"BusEq": 8.46, "Shops": 6.07},
        "r2_full": 0.495,
        "industry_joint_f_test": {
            "f_stat": 8.39, "f_pvalue": 2.7e-31,
            "df_num": 12, "df_denom": 376,
        },
        "model_form": "joint_ff5mom_plus_12_industry",
    }
    m = _full_metrics(with_anchor_ortho=True)
    m["industry_extension"] = fake_ix
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN", metrics=m),
    ])
    item = client.get("/api/research/tier_c_verdicts").json()["items"][0]
    ix = item["industry_extension"]
    assert ix is not None
    assert ix["alpha_full_nw_t"] == -1.38
    assert ix["delta_alpha_nw_t_approx"] == 3.26
    assert ix["model_form"] == "joint_ff5mom_plus_12_industry"


def test_industry_extension_null_when_absent(monkeypatch):
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN",
             metrics=_full_metrics(with_anchor_ortho=False)),
    ])
    item = client.get("/api/research/tier_c_verdicts").json()["items"][0]
    assert item["industry_extension"] is None


def test_anchor_orthogonality_null_when_absent(monkeypatch):
    """Pre-L2-4-Commit-3 verdicts have no anchor_orthogonality → null."""
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev(verdict="GREEN",
             metrics=_full_metrics(with_anchor_ortho=False)),
    ])
    item = client.get("/api/research/tier_c_verdicts").json()["items"][0]
    assert item["anchor_orthogonality"] is None
