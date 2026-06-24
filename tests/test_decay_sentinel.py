"""tests/test_decay_sentinel.py — role-aware Decay Sentinel + narrator discipline.

Locks the refinement: insurance/trend judged on crisis-payoff (never flagged decayed
off a calm Sharpe), regime_premium on signal-IC (a drawdown with IC intact = HOLD),
role-aware re-allocation (freed weight to surviving RETURN sources only), no-decay ->
base verbatim, and the narrator never softens the deterministic verdict (banned-phrases).
"""
import numpy as np
import pandas as pd
import pytest

from engine.validation.decay_sentinel import (
    crisis_payoff, assess_structural_decay, recommend_allocation,
    sentinel_report, MechanismConfig,
)
from engine.agents.decay_sentinel.narrator import narrate_report, contains_banned_phrase


# ── crisis_payoff ──────────────────────────────────────────────────────────────
def test_crisis_payoff_positive_when_hedge_pays_in_stress():
    idx = pd.date_range("2014-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.005, 0.04, 120), index=idx)
    # a hedge: small calm drag, large positive in the market's worst months
    ret = -0.002 + np.where(mkt <= mkt.quantile(0.25), 0.05, 0.0)
    cp = crisis_payoff(pd.Series(ret, index=idx), mkt)
    assert cp > 0  # still protecting


# ── role-aware structural decay ──────────────────────────────────────────────
def _dead_sharpe():
    return pd.Series([0.05] * 30, index=pd.date_range("2014-01-31", periods=30, freq="ME"))


@pytest.mark.parametrize("role", ["insurance", "trend"])
def test_insurance_trend_never_flagged_off_calm_sharpe(role):
    d = assess_structural_decay("X", _dead_sharpe(), ic_roll=None, role=role)
    assert d["structural_decay"] is False
    assert role in d["reason"]


def test_regime_premium_holds_when_ic_intact():
    sr = _dead_sharpe()
    ic_intact = pd.Series([0.05] * 30, index=sr.index)
    ic_faded = pd.Series([-0.01] * 30, index=sr.index)
    # sustained low Sharpe but IC positive -> regime drawdown, HOLD
    assert assess_structural_decay("carry", sr, ic_roll=ic_intact, role="regime_premium")["structural_decay"] is False
    # sustained low Sharpe AND IC faded -> premium gone, decay
    assert assess_structural_decay("carry", sr, ic_roll=ic_faded, role="regime_premium")["structural_decay"] is True


def test_alpha_decays_on_sustained_low_sharpe_return_only():
    # an alpha with no signal panel: sustained low Sharpe alone flags (return-only caveat)
    assert assess_structural_decay("A", _dead_sharpe(), ic_roll=None, role="alpha")["structural_decay"] is True


# ── role-aware re-allocation ──────────────────────────────────────────────────
def test_no_decay_returns_base_verbatim():
    base = {"A": 0.4, "B": 0.4, "C": 0.3}  # intentionally sums > 1 (candidate overlay)
    flags = {k: False for k in base}
    assert recommend_allocation(base, flags, roles={"A": "alpha", "B": "alpha", "C": "regime_premium"}) == base


def test_freed_weight_goes_to_return_survivors_not_hedges():
    base = {"A": 0.4, "B": 0.4, "INS": 0.2}
    roles = {"A": "alpha", "B": "alpha", "INS": "insurance"}
    rec = recommend_allocation(base, {"A": True, "B": False, "INS": False}, roles=roles)
    # A halved (0.2), all freed (0.2) -> B (the only return survivor); INS untouched by freed weight
    assert rec["A"] == pytest.approx(0.2, abs=1e-9)
    assert rec["B"] == pytest.approx(0.6, abs=1e-9)
    assert rec["INS"] == pytest.approx(0.2, abs=1e-9)


def test_fallback_to_all_survivors_when_no_return_source_left():
    base = {"A": 0.5, "INS": 0.3, "TREND": 0.2}
    roles = {"A": "alpha", "INS": "insurance", "TREND": "trend"}
    rec = recommend_allocation(base, {"A": True, "INS": False, "TREND": False}, roles=roles)
    assert rec["A"] == pytest.approx(0.25, abs=1e-9)        # halved
    assert rec["INS"] + rec["TREND"] == pytest.approx(0.75, abs=1e-9)  # absorbed the freed weight
    assert rec["INS"] > 0.3 and rec["TREND"] > 0.2


# ── narrator discipline ───────────────────────────────────────────────────────
def _mini_report(overall, alarms, realloc=False):
    return dict(
        window=36, overall=overall, realloc_action=realloc, alarms=alarms,
        mechanisms={"D_PEAD": dict(full_sharpe=1.0, rolling_sharpe=1.1, rolling_t=2.0, decay_ratio=1.1),
                    "AC": dict(full_sharpe=0.5, rolling_sharpe=-0.01, rolling_t=-0.02, decay_ratio=-0.02)},
        roles={"D_PEAD": "alpha", "AC": "insurance"},
        crisis={"D_PEAD": float("nan"), "AC": 0.0018},
        decay={"D_PEAD": dict(structural_decay=False), "AC": dict(structural_decay=False, signal_ic=None)},
        base_weights={"D_PEAD": 0.24, "AC": 0.10},
        recommended_weights={"D_PEAD": 0.24, "AC": 0.10},
    )


@pytest.mark.parametrize("verdict,word", [("HEALTHY", "HEALTHY"), ("WATCH", "WATCH"), ("ACTION", "ACTION")])
def test_narrator_states_verdict_and_is_clean(verdict, word):
    alarms = [] if verdict == "HEALTHY" else [("WARN" if verdict == "WATCH" else "ALERT", "x")]
    res = narrate_report(_mini_report(verdict, alarms, realloc=(verdict == "ACTION")))
    assert word in res.text
    assert res.cost_usd == 0.0 and res.backend == "deterministic"
    assert contains_banned_phrase(res.text) is None       # no hedging — a verdict reads as a verdict
    # insurance crisis-payoff phrasing present (judged on hedging, not calm Sharpe)
    assert "crisis-payoff" in res.text


def test_sentinel_report_overall_is_deterministic():
    idx = pd.date_range("2010-01-31", periods=140, freq="ME")
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0.01, 0.03, 140), index=idx)   # healthy alpha
    b = pd.Series(rng.normal(0.008, 0.03, 140), index=idx)
    mkt = pd.Series(rng.normal(0.005, 0.04, 140), index=idx)
    mechs = {"A": MechanismConfig(name="A", returns=a, weight=0.5, role="alpha"),
             "B": MechanismConfig(name="B", returns=b, weight=0.5, role="alpha")}
    rep = sentinel_report(mechs, market=mkt)
    assert rep["overall"] in ("HEALTHY", "WATCH", "ACTION")
    assert rep["realloc_action"] is False                   # no structural decay on healthy synthetic alphas
    assert rep["recommended_weights"] == rep["base_weights"]
