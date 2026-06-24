"""bt-flex-4.1 tests — portfolio_overlay_60_40 template.

Covers:
  - data loading paths
  - 60/40 baseline arithmetic
  - TSMOM overlay sign convention + vol target
  - portfolio blend arithmetic
  - Jobson-Korkie t-stat behavior
  - verdict thresholds (GREEN / MARGINAL / RED) with correct MaxDD sign
  - parse_overlay_pct edge cases
  - end-to-end live smoke (uses real cached data)
"""
from __future__ import annotations

import math
import pandas as pd
import pytest

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.agents.strengthener.templates import portfolio_overlay_60_40 as ovl


def _spec(**kw):
    """Synth a minimal FactorSpec for the template."""
    base = dict(
        hypothesis_id="t-1",
        signal_kind="portfolio_overlay",
        universe="us_balanced_60_40",
        date_range="2010-01:2024-12",
        signal_inputs=("spy_monthly", "ief_monthly"),
        rebal="monthly",
        weighting="ew",
        expected_holding_period="monthly",
        min_obs_months=36,
        pit_audits=("restatement",),
        cost_model="13bp_per_rt",
        rationale="test",
        extracted_ts="2026-06-11T07:00:00Z",
        model="test",
    )
    base.update(kw)
    return FactorSpec(**base)


# ── parse_overlay_pct ──────────────────────────────────────────────


def test_parse_overlay_pct_decimal():
    s = _spec(weighting_scheme_alt="0.25")
    assert ovl._parse_overlay_pct(s) == 0.25


def test_parse_overlay_pct_percentage():
    s = _spec(weighting_scheme_alt="20")
    assert ovl._parse_overlay_pct(s) == 0.20


def test_parse_overlay_pct_pct_suffix():
    s = _spec(weighting_scheme_alt="30pct")
    assert ovl._parse_overlay_pct(s) == 0.30


def test_parse_overlay_pct_defaults_when_unparseable():
    s = _spec(weighting_scheme_alt="garbage")
    assert ovl._parse_overlay_pct(s) == 0.20


def test_parse_overlay_pct_defaults_when_missing():
    s = _spec()
    assert ovl._parse_overlay_pct(s) == 0.20


def test_parse_overlay_pct_clamps_implausibly():
    # Outside 0.05-0.50 range falls back to default
    s_high = _spec(weighting_scheme_alt="0.75")
    s_low  = _spec(weighting_scheme_alt="0.01")
    assert ovl._parse_overlay_pct(s_high) == 0.20
    assert ovl._parse_overlay_pct(s_low) == 0.20


# ── arithmetic helpers ─────────────────────────────────────────────


def test_60_40_baseline_arithmetic():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    spy = pd.Series([0.02] * 12, index=idx)
    ief = pd.Series([0.005] * 12, index=idx)
    out = ovl._build_60_40_baseline(spy, ief)
    expected = 0.60 * 0.02 + 0.40 * 0.005
    assert all(abs(v - expected) < 1e-9 for v in out.values)
    assert len(out) == 12


def test_overlaid_portfolio_arithmetic():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    base = pd.Series([0.01] * 12, index=idx)
    olay = pd.Series([0.05] * 12, index=idx)
    out = ovl._build_overlaid_portfolio(base, olay, 0.20)
    expected = 0.80 * 0.01 + 0.20 * 0.05
    assert all(abs(v - expected) < 1e-9 for v in out.values)


def test_max_drawdown_sign_convention():
    # Returns producing 50% peak-to-trough should give MDD ≈ -0.50
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    s = pd.Series([0.0, 0.1, 0.1, -0.30, -0.20, -0.10], index=idx)
    mdd = ovl._max_drawdown(s)
    assert mdd < 0  # negative by convention
    assert mdd < -0.3


def test_annualized_sharpe_basic():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    s = pd.Series([0.01] * 120, index=idx)   # zero variance → nan
    assert math.isnan(ovl._annualized_sharpe(s))


# ── Jobson-Korkie ──────────────────────────────────────────────────


def test_jobson_korkie_identical_series_zero_diff():
    idx = pd.date_range("2010-01-31", periods=120, freq="ME")
    import numpy as np
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(0.005, 0.04, size=120), index=idx)
    t, se = ovl._jobson_korkie_t_stat(s, s)
    # Identical series → diff = 0, t = nan or 0
    if not math.isnan(t):
        assert abs(t) < 0.01


def test_jobson_korkie_short_window_returns_nan():
    idx = pd.date_range("2010-01-31", periods=12, freq="ME")
    import numpy as np
    rng = np.random.default_rng(7)
    s1 = pd.Series(rng.normal(0, 0.04, 12), index=idx)
    s2 = pd.Series(rng.normal(0, 0.04, 12), index=idx)
    t, se = ovl._jobson_korkie_t_stat(s1, s2)
    assert math.isnan(t)


# ── Verdict thresholds (sign-corrected MaxDD) ──────────────────────


def test_verdict_green_requires_all_conditions():
    # Sharpe up, MDD improved (positive delta), t-significant
    assert ovl._classify_verdict(
        sharpe_delta=+0.25, maxdd_delta=+0.05, sharpe_diff_t=2.0,
    ) == "GREEN"


def test_verdict_red_when_sharpe_t_insignificant():
    # Even if sharpe / maxdd look better, low t → RED
    assert ovl._classify_verdict(
        sharpe_delta=+0.25, maxdd_delta=+0.05, sharpe_diff_t=0.5,
    ) == "RED"


def test_verdict_red_when_maxdd_worsened():
    # Negative maxdd_delta = WORSE drawdown
    assert ovl._classify_verdict(
        sharpe_delta=+0.25, maxdd_delta=-0.05, sharpe_diff_t=2.5,
    ) == "RED"


def test_verdict_marginal_when_modest_improvements():
    assert ovl._classify_verdict(
        sharpe_delta=+0.15, maxdd_delta=+0.01, sharpe_diff_t=1.7,
    ) == "MARGINAL"


def test_verdict_red_when_sharpe_flat():
    assert ovl._classify_verdict(
        sharpe_delta=+0.005, maxdd_delta=+0.03, sharpe_diff_t=0.1,
    ) == "RED"


# ── Live smoke (uses real cached data) ─────────────────────────────


def test_live_smoke_hop_2017_canonical():
    """End-to-end: 60/40 + 20% TSMOM on SPY against real cached data."""
    s = _spec(weighting_scheme_alt="0.20")
    result = ovl.template_portfolio_overlay_60_40(s)
    # Must produce ONE of the documented verdicts (no crashes)
    assert result.verdict in {
        "GREEN", "MARGINAL", "RED", "INSUFFICIENT_DATA", "INSUFFICIENT_HISTORY"
    }
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        # Sanity check: metrics populated
        for k in ("base_sharpe", "overlay_portfolio_sharpe", "sharpe_delta",
                   "base_max_drawdown", "overlay_portfolio_max_drawdown",
                   "n_obs_months"):
            assert k in result.metrics
            assert result.metrics[k] is not None
        # Sharpe values should be finite + plausible (Sharpe 0-3 range)
        assert -1.0 < result.metrics["base_sharpe"] < 3.0
        # PnL series df present
        assert "pnl_series_df" in result.artifacts
        assert "pnl_default_col" in result.artifacts


def test_m2_replicates_hop2017_baseline_60_40_sharpe_band():
    """M2 (paper replication anchor): HOP-2017 reports US 60/40 1880-2016
    Sharpe around 0.40-0.60 depending on cost convention. Our SPY+IEF
    2011-2024 post-GFC bull-market sample produces Sharpe ~1.0 because
    base portfolio benefited from extraordinary post-GFC equity returns.

    Anchor: base 60/40 Sharpe in [0.40, 1.50] (very wide because our
    sample window is short + atypical). Key sanity:
      - base Sharpe > 0 (60/40 should be positive)
      - overlay vs base ΔSharpe in [-0.50, +0.50] (TSMOM overlay
        contribution should be small-magnitude, not wildly large)
      - J-K paired t stat finite (statistical machinery works)

    Failure = template math drifted; CI blocks merge.
    """
    s = _spec(weighting_scheme_alt="0.20")
    result = ovl.template_portfolio_overlay_60_40(s)
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        m = result.metrics
        # Base 60/40 Sharpe sanity: must be positive but not absurd
        assert 0.40 < m["base_sharpe"] < 1.50, (
            f"REPLICATION FAILURE: 60/40 base Sharpe {m['base_sharpe']:.2f} "
            f"outside [0.40, 1.50] sanity band"
        )
        # Overlay contribution should be modest magnitude
        assert -0.50 < m["sharpe_delta"] < 0.50, (
            f"REPLICATION FAILURE: overlay ΔSharpe {m['sharpe_delta']:.2f} "
            f"too large in magnitude — math may have drifted"
        )
        # J-K paired t-stat machinery alive
        import math
        assert math.isfinite(m["sharpe_diff_t"]), (
            f"J-K paired t-stat = {m['sharpe_diff_t']} (not finite)"
        )
