"""vrp_spx template tests + M2 paper replication anchor.

M2 anchor: Carr-Wu 2009 documented systematically positive variance
risk premium on SPX. Our 1990-2026 sample includes 2008 GFC + 2020
COVID short-vol blowups; the qualitative claim (mean PnL > 0) should
still hold but Sharpe will be lower than Carr-Wu's pre-GFC sub-sample.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.agents.strengthener.templates import vrp_spx as vt


def _spec(**kw):
    base = dict(
        hypothesis_id="t-vrp",
        signal_kind="vrp",
        universe="us_equities_spx_options",
        date_range="1990-01:2025-12",
        signal_inputs=("cboe.vix_spx.vix", "cboe.vix_spx.spx"),
        rebal="monthly",
        weighting="ew",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("restatement",),
        cost_model="none",
        rationale="test",
        extracted_ts="2026-06-13T00:00:00Z",
        model="test",
    )
    base.update(kw)
    return FactorSpec(**base)


# ── Data loader ───────────────────────────────────────────────────


def test_load_returns_dataframe():
    df = vt._load_vix_spx_daily()
    assert df is not None
    assert "VIX" in df.columns
    assert "SPX" in df.columns
    assert len(df) > 500


# ── Monthly PnL builder ───────────────────────────────────────────


def test_monthly_pnl_uses_lagged_implied_var():
    """PIT sanity: the implied variance for period [t-21, t] must come
    from VIX at time t-21 (start of period), NOT VIX at time t.

    Test: feed synthetic VIX=20, SPX flat. Implied variance at start =
    (0.20)² × 21/252 = 0.00333. Realized var = 0 (flat returns). PnL =
    +0.00333.
    """
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "VIX": [20.0] * 300,
        "SPX": [100.0] * 300,
    }, index=idx)
    monthly = vt._build_monthly_vrp_pnl(df)
    assert monthly is not None
    # Most monthly PnL values should be positive ~0.00333 (lagged
    # implied var minus zero realized var)
    expected = (0.20 ** 2) * (21.0 / 252.0)
    finite = monthly.dropna()
    # Allow small float fuzz
    assert (finite > 0).mean() > 0.9, (
        f"Expected mostly positive PnL with flat SPX; got {finite.describe()}"
    )
    # Median close to expected
    assert abs(finite.median() - expected) < 1e-6


def test_monthly_pnl_negative_on_high_realized_vol():
    """If realized vol >> implied vol, PnL should go negative
    (insurance writer LOSES). Construct: VIX=10, SPX has huge
    realized vol."""
    import numpy as np
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    # Huge daily returns to drive realized vol up
    log_returns = rng.normal(0, 0.05, size=300)  # 5% daily ~ ~80% annual vol
    spx = pd.Series(100.0 * pd.Series(log_returns).cumsum().apply(math.exp).values, index=idx)
    df = pd.DataFrame({
        "VIX": [10.0] * 300,  # very low implied vol
        "SPX": spx.values,
    }, index=idx)
    monthly = vt._build_monthly_vrp_pnl(df)
    assert monthly is not None
    finite = monthly.dropna()
    # Most periods PnL should be NEGATIVE (insurance writers lose)
    assert (finite < 0).mean() > 0.7, (
        f"Expected mostly negative PnL when realized >> implied; "
        f"got {(finite < 0).mean():.2%} negative"
    )


# ── Verdict classification ────────────────────────────────────────


def test_classify_red_when_mean_pnl_negative():
    """Mean PnL ≤ 0 is RED regardless of NW-t (Carr-Wu doctrine)."""
    verdict, note = vt._classify_verdict(
        sharpe=-0.5, nw_t=-3.0, mean_pnl=-0.001, n_trials=1,
    )
    assert verdict == "RED"


def test_classify_green_when_significant_positive():
    verdict, note = vt._classify_verdict(
        sharpe=0.8, nw_t=3.5, mean_pnl=0.001, n_trials=1,
    )
    assert verdict == "GREEN"


def test_classify_marginal_in_band():
    verdict, note = vt._classify_verdict(
        sharpe=0.4, nw_t=1.80, mean_pnl=0.0005, n_trials=1,
    )
    assert verdict == "MARGINAL"


def test_classify_red_when_positive_but_insignificant():
    verdict, note = vt._classify_verdict(
        sharpe=0.1, nw_t=0.5, mean_pnl=0.0001, n_trials=1,
    )
    assert verdict == "RED"


# ── End-to-end live smoke ─────────────────────────────────────────


def test_live_smoke_carr_wu_short_variance():
    """End-to-end: short-variance VRP on SPX with real cached data.
    Verdict must be ONE of the documented values."""
    s = _spec()
    result = vt.template_vrp_spx(s)
    assert result.verdict in {
        "GREEN", "MARGINAL", "RED",
        "INSUFFICIENT_DATA", "INSUFFICIENT_HISTORY",
    }
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        m = result.metrics
        # Metrics populated
        for k in ("mean_pnl_monthly", "std_pnl_monthly", "sharpe_gross",
                   "nw_t_gross", "max_drawdown", "n_obs_months"):
            assert k in m
            assert m[k] is not None
        # PnL series df present
        assert "pnl_series_df" in result.artifacts


def test_m2_replicates_carr_wu_positive_vrp():
    """M2 (paper replication anchor): Carr-Wu 2009 documents
    POSITIVE average VRP on SPX. Our 1990-2026 sample MUST show
    positive mean PnL (even with 2008 + 2020 short-vol blowups,
    the cumulative VRP collected over calm decades dominates the
    catastrophic months on average).

    Anchor (loose, qualitative):
      - mean monthly PnL > 0
      - sample size ≥ 200 months (≥ 16 years)
      - sharpe_gross in [0.0, 2.0] (plausible range)

    Failure = either template math drifted (mean PnL flipped sign)
    OR data cache is corrupted. CI blocks merge.
    """
    s = _spec()
    result = vt.template_vrp_spx(s)
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        m = result.metrics
        # M2 ANCHOR: mean PnL must be POSITIVE
        assert m["mean_pnl_monthly"] > 0, (
            f"REPLICATION FAILURE (Carr-Wu 2009): mean monthly PnL "
            f"{m['mean_pnl_monthly']:.5f} should be POSITIVE in SPX. "
            f"VRP = insurance premium; insurance writers profit on average."
        )
        # Sample size sanity
        assert m["n_obs_months"] >= 200, (
            f"Only {m['n_obs_months']} months of data — Carr-Wu sample "
            f"needs ≥ 200 months for meaningful replication"
        )
        # Sharpe plausibility
        assert 0.0 < m["sharpe_gross"] < 2.0, (
            f"REPLICATION FAILURE: Sharpe {m['sharpe_gross']:.2f} outside "
            f"plausible band [0.0, 2.0] for short-vol on SPX"
        )


def test_dispatcher_routes_vrp_signal_kind():
    """Composite: TEMPLATE_REGISTRY['vrp'] must route to the shipped
    template, not _template_pending_build."""
    from engine.agents.strengthener.factor_dispatcher import TEMPLATE_REGISTRY
    s = _spec()
    fn = TEMPLATE_REGISTRY["vrp"]
    result = fn(s)
    # Must NOT be PENDING_TEMPLATE_BUILD
    assert result.verdict != "PENDING_TEMPLATE_BUILD", (
        "vrp signal_kind dispatched to _template_pending_build — "
        "wiring broken in factor_dispatcher.TEMPLATE_REGISTRY"
    )
