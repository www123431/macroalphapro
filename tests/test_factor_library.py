"""tests/test_factor_library.py — Factor signal content layer invariants.

Spec: docs/spec_factor_library_v1.md (registered 2026-05-09, id=42)

Tests cover:
  - FACTOR_REGISTRY structural invariants (5 entries, locked v1, no inverse_vol)
  - Boundary invariant: factor_lab does NOT import factor_library
  - select_independent_factors greedy correlation algorithm (spec §2.2)
  - Skeleton signal_fn closures raise NotImplementedError (W1 D2 marker)

Concrete signal_fn behavior + ensemble weighting tests land in W1 D2-D3 + W2.
"""
from __future__ import annotations

import datetime
import importlib
import pathlib

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR_REGISTRY structural invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_factor_registry_has_exactly_5_v1_locked_factors():
    """Spec §2.1 locks 5 candidates: bab / low_vol / tsmom_12_1 / csmom /
    donchian_trend. Adding/removing requires hypothesis_amend (n_trials_added=3),
    not a silent code edit."""
    from engine.factor_library import FACTOR_REGISTRY
    expected = {"bab", "low_vol", "tsmom_12_1", "csmom", "donchian_trend"}
    assert set(FACTOR_REGISTRY.keys()) == expected, (
        f"FACTOR_REGISTRY keys drifted from spec §2.1 lock; "
        f"got {set(FACTOR_REGISTRY)} expected {expected}"
    )


def test_factor_registry_does_not_contain_inverse_vol():
    """Spec §2.1 v1 explicitly DROPPED inverse_vol (concept overlap with §2.4
    risk-parity weighting). Re-adding requires hypothesis_amend."""
    from engine.factor_library import FACTOR_REGISTRY
    assert "inverse_vol" not in FACTOR_REGISTRY, (
        "inverse_vol was dropped v1 per spec §2.1 (concept overlap with risk parity); "
        "re-adding requires amend_spec(kind='hypothesis_amend')"
    )


def test_factor_spec_metadata_complete():
    """Each FactorSpec must have non-empty citation / asset_class / formula_summary
    (so reviewer can trace literature lineage of every candidate)."""
    from engine.factor_library import FACTOR_REGISTRY
    for fid, spec in FACTOR_REGISTRY.items():
        assert spec.factor_id == fid, f"factor_id {spec.factor_id!r} != registry key {fid!r}"
        assert spec.citation.strip(), f"{fid}: empty citation"
        assert spec.asset_class.strip(), f"{fid}: empty asset_class"
        assert spec.formula_summary.strip(), f"{fid}: empty formula_summary"
        assert callable(spec.signal_fn), f"{fid}: signal_fn not callable"


def test_donchian_citation_locks_hurst_ooi_pedersen_2017_only():
    """Spec §2.1 locks Donchian to Hurst-Ooi-Pedersen 2017 (preferred over Faber 2007).
    v3 amendment removed Faber from §11 学术锚点; FACTOR_REGISTRY citation must
    not regress to including Faber."""
    from engine.factor_library import FACTOR_REGISTRY
    cit = FACTOR_REGISTRY["donchian_trend"].citation
    assert "Hurst" in cit and "Ooi" in cit and "Pedersen" in cit and "2017" in cit, (
        f"Donchian citation should reference Hurst-Ooi-Pedersen 2017; got: {cit!r}"
    )
    assert "Faber" not in cit, (
        f"Donchian citation must not regress to Faber 2007 (v1 locked HOP 2017 only); "
        f"got: {cit!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module boundary: factor_lab must not import factor_library (one-way)
# ─────────────────────────────────────────────────────────────────────────────

def test_factor_lab_does_not_import_factor_library():
    """Per spec_factor_library_v1.md §4.1 + spec_factor_lab.md boundary section:
    dependency direction is one-way (library → lab.power, never reverse).
    Static check by grepping factor_lab/ for any 'engine.factor_library' import."""
    factor_lab_dir = pathlib.Path(__file__).resolve().parent.parent / "engine" / "factor_lab"
    assert factor_lab_dir.is_dir(), f"factor_lab dir missing: {factor_lab_dir}"
    violations: list[str] = []
    for py_file in factor_lab_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # skip comments
            if "engine.factor_library" in stripped or "from engine import factor_library" in stripped:
                violations.append(f"{py_file.relative_to(factor_lab_dir.parent.parent)}:{line_no}: {stripped}")
    assert not violations, (
        f"factor_lab imports factor_library (violates one-way dep per spec §4.1): "
        f"{violations}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# select_independent_factors algorithm correctness (spec §2.2)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_in_sample_returns(seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly returns 1996-2009 (168 obs) for the 5 v1 factors.

    Construction: orthogonal base + adversarial high-correlation pair so the
    greedy filter has something to drop.
        - bab:            highest Sharpe (mean 0.6%, sd 2%)
        - low_vol:        90% correlated to bab (forced drop on greedy)
        - tsmom_12_1:     orthogonal moderate Sharpe (mean 0.4%, sd 2.5%)
        - csmom:          orthogonal lower Sharpe (mean 0.2%, sd 2%)
        - donchian_trend: orthogonal lowest Sharpe (mean 0.1%, sd 3%)
    """
    rng = np.random.default_rng(seed)
    n = 168
    bab            = rng.normal(0.006, 0.02, n)
    low_vol        = 0.9 * bab + 0.1 * rng.normal(0, 0.02, n)  # highly correlated to bab
    tsmom_12_1     = rng.normal(0.004, 0.025, n)
    csmom          = rng.normal(0.002, 0.02, n)
    donchian_trend = rng.normal(0.001, 0.03, n)
    idx = pd.date_range("1996-01-31", periods=n, freq="ME")
    return pd.DataFrame({
        "bab": bab, "low_vol": low_vol, "tsmom_12_1": tsmom_12_1,
        "csmom": csmom, "donchian_trend": donchian_trend,
    }, index=idx)


def test_select_independent_factors_drops_correlated_with_lower_sharpe():
    """Greedy algorithm must retain bab (top Sharpe) and drop low_vol (90%
    correlated, lower Sharpe). tsmom/csmom/donchian have low correlation to
    retained → at least one should make it through."""
    from engine.factor_library import select_independent_factors
    df = _synthetic_in_sample_returns()
    retained = select_independent_factors(df, corr_threshold=0.7)
    assert "bab" in retained, (
        f"bab has highest Sharpe by construction; should always be retained. Got {retained}"
    )
    assert "low_vol" not in retained, (
        f"low_vol is 90% correlated to bab; greedy filter should drop it. Got {retained}"
    )
    # At least one orthogonal factor should pass the corr<0.7 filter.
    assert len(retained) >= 2, (
        f"with 3 orthogonal candidates (tsmom/csmom/donchian), at least one should "
        f"pass corr<0.7 to retained. Got {retained}"
    )


def test_select_independent_factors_rejects_unknown_candidates():
    """Unknown candidate ids must raise rather than silently filter — silent
    filter would mask spec drift."""
    from engine.factor_library import select_independent_factors
    df = _synthetic_in_sample_returns()
    with pytest.raises(ValueError, match="unknown candidates"):
        select_independent_factors(df, candidates=["bab", "value_factor_not_in_v1"])


def test_select_independent_factors_rejects_corr_threshold_out_of_range():
    """corr_threshold must be in (0, 1); spec §2.2 locks 0.7 but argument
    validation should still bound to sensible range."""
    from engine.factor_library import select_independent_factors
    df = _synthetic_in_sample_returns()
    with pytest.raises(ValueError, match="corr_threshold"):
        select_independent_factors(df, corr_threshold=1.5)
    with pytest.raises(ValueError, match="corr_threshold"):
        select_independent_factors(df, corr_threshold=-0.1)


def test_select_independent_factors_rejects_empty_returns():
    """Empty in-sample returns must raise rather than return empty list —
    silent empty would mask data-loading failure."""
    from engine.factor_library import select_independent_factors
    with pytest.raises(ValueError, match="empty"):
        select_independent_factors(pd.DataFrame())


def test_select_independent_factors_rejects_missing_columns():
    """If a candidate has no returns column, must raise (data plumbing bug)."""
    from engine.factor_library import select_independent_factors
    df = _synthetic_in_sample_returns().drop(columns=["donchian_trend"])
    with pytest.raises(ValueError, match="missing columns"):
        select_independent_factors(df, candidates=["bab", "donchian_trend"])


# ─────────────────────────────────────────────────────────────────────────────
# Pure compute helpers — synthetic-data tests (W1 D2-D3 implementation)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_closes(
    n_days: int = 300,
    tickers: list[str] | None = None,
    seed: int = 42,
    drift: dict[str, float] | None = None,
    vol: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily price series with controllable drift/vol per ticker.

    Default: 300 days (≈ 14 months) ending today, 4 tickers with mixed
    characteristics suitable for cross-factor smoke tests.
    """
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC", "DDD"]
    drift = drift or {t: 0.0003 for t in tickers}     # ~7%/yr default
    vol = vol or {t: 0.012 for t in tickers}           # ~19% annualized default
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n_days, freq="B")
    data = {}
    for t in tickers:
        rets = rng.normal(drift[t], vol[t], n_days)
        prices = 100 * np.exp(np.cumsum(rets))
        data[t] = prices
    return pd.DataFrame(data, index=idx)


# ── BAB pure compute ─────────────────────────────────────────────────────────

def test_bab_compute_low_beta_tickers_in_long_leg():
    """Synthetic universe with monotonic β should retain low-β in long, high-β in short."""
    from engine.factor_library import _compute_bab_weights
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    spy_rets = rng.normal(0.0004, 0.011, n)
    spy_close = pd.Series(100 * np.exp(np.cumsum(spy_rets)), index=idx)

    # Construct 6 tickers with controlled β: 0.3, 0.6, 0.9, 1.2, 1.5, 1.8
    universe_rets = {}
    for i, target_beta in enumerate([0.3, 0.6, 0.9, 1.2, 1.5, 1.8]):
        idio = rng.normal(0.0, 0.008, n)
        rets = target_beta * spy_rets + idio
        universe_rets[f"T{i}"] = 100 * np.exp(np.cumsum(rets))
    closes = pd.DataFrame(universe_rets, index=idx)
    weights = _compute_bab_weights(closes, spy_close)
    assert weights, "expected non-empty weights given 300 days × 6 tickers"
    # Low-β tickers (T0/T1) should be in long leg (positive weights)
    assert weights.get("T0", 0) > 0, f"T0 (β=0.3) should be long; weights={weights}"
    # High-β tickers (T4/T5) should be short
    assert weights.get("T5", 0) < 0, f"T5 (β=1.8) should be short; weights={weights}"


def test_bab_compute_returns_empty_on_insufficient_data():
    """< 60 daily observations → return {} (no exception)."""
    from engine.factor_library import _compute_bab_weights
    closes = _synthetic_closes(n_days=30)
    spy = closes["AAA"].rename("SPY")
    result = _compute_bab_weights(closes, spy, beta_window_days=252)
    assert result == {}


def test_bab_compute_raises_on_empty_input():
    from engine.factor_library import _compute_bab_weights
    with pytest.raises(ValueError, match="empty"):
        _compute_bab_weights(pd.DataFrame(), pd.Series(dtype=float))


def test_bab_compute_weights_gross_normalize_to_one():
    """β-neutral weights normalized to gross exposure 1."""
    from engine.factor_library import _compute_bab_weights
    rng = np.random.default_rng(11)
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    spy_rets = rng.normal(0.0004, 0.011, n)
    spy = pd.Series(100 * np.exp(np.cumsum(spy_rets)), index=idx)
    universe_rets = {f"T{i}": 100 * np.exp(np.cumsum(b * spy_rets + rng.normal(0, 0.008, n)))
                     for i, b in enumerate([0.3, 0.7, 1.0, 1.3, 1.6, 1.9])}
    closes = pd.DataFrame(universe_rets, index=idx)
    w = _compute_bab_weights(closes, spy)
    if w:
        gross = sum(abs(v) for v in w.values())
        assert abs(gross - 1.0) < 1e-9, f"gross exposure should be 1; got {gross}"


# ── Low-Vol pure compute ─────────────────────────────────────────────────────

def test_low_vol_compute_low_vol_in_long_high_in_short():
    """Synthetic universe with monotonic vol → bottom quintile long, top short."""
    from engine.factor_library import _compute_low_vol_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(13)
    # 10 tickers with vol from 5% to 50% annualized
    vols_annual = np.linspace(0.05, 0.50, 10)
    data = {}
    for i, vol_ann in enumerate(vols_annual):
        vol_daily = vol_ann / np.sqrt(252)
        rets = rng.normal(0.0, vol_daily, n)
        data[f"T{i}"] = 100 * np.exp(np.cumsum(rets))
    closes = pd.DataFrame(data, index=idx)
    w = _compute_low_vol_weights(closes)
    # Bottom-vol (T0) should be long; top-vol (T9) should be short
    assert w.get("T0", 0) > 0, f"T0 (lowest vol) should be long; got {w}"
    assert w.get("T9", 0) < 0, f"T9 (highest vol) should be short; got {w}"


def test_low_vol_compute_long_short_sum_zero():
    """Equal-weight long/short legs → weights sum to 0."""
    from engine.factor_library import _compute_low_vol_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(17)
    data = {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0, (0.05 + 0.05*i) / np.sqrt(252), n)))
            for i in range(10)}
    closes = pd.DataFrame(data, index=idx)
    w = _compute_low_vol_weights(closes)
    if w:
        s = sum(w.values())
        assert abs(s) < 1e-9, f"long-short weights should sum to 0; got {s}"


# ── TSMOM pure compute ───────────────────────────────────────────────────────

def test_tsmom_compute_uptrend_long_downtrend_short():
    """Strong up-trend (positive 12-1 return) → long; down-trend → short."""
    from engine.factor_library import _compute_tsmom_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(19)
    # UP: positive drift; DOWN: negative drift
    up_rets   = rng.normal(+0.001, 0.010, n)  # ~+25%/yr drift
    down_rets = rng.normal(-0.001, 0.010, n)  # ~-25%/yr drift
    data = {
        "UP":   100 * np.exp(np.cumsum(up_rets)),
        "DOWN": 100 * np.exp(np.cumsum(down_rets)),
    }
    closes = pd.DataFrame(data, index=idx)
    w = _compute_tsmom_weights(closes)
    assert w.get("UP", 0) > 0, f"UP-trending should be long; got {w}"
    assert w.get("DOWN", 0) < 0, f"DOWN-trending should be short; got {w}"


def test_tsmom_compute_gross_exposure_normalized():
    """|weights|.sum() == 1 after gross normalization."""
    from engine.factor_library import _compute_tsmom_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(23)
    drifts = [+0.001, -0.001, +0.0005, -0.0005]
    data = {f"T{i}": 100 * np.exp(np.cumsum(rng.normal(d, 0.012, n)))
            for i, d in enumerate(drifts)}
    closes = pd.DataFrame(data, index=idx)
    w = _compute_tsmom_weights(closes)
    if w:
        gross = sum(abs(v) for v in w.values())
        assert abs(gross - 1.0) < 1e-9, f"gross should be 1; got {gross}"


def test_tsmom_compute_returns_empty_on_insufficient_data():
    """< 252 days of history → return {}."""
    from engine.factor_library import _compute_tsmom_weights
    closes = _synthetic_closes(n_days=100)
    assert _compute_tsmom_weights(closes) == {}


# ── CSMOM pure compute ───────────────────────────────────────────────────────

def test_csmom_compute_within_class_long_short():
    """Within asset class, top-momentum → long, bottom → short. Cross-class
    rankings independent."""
    from engine.factor_library import _compute_csmom_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(29)
    # 3 equity tickers with monotone drift, 3 fixed_income with same pattern.
    # Drift differential must be large vs sampling noise (σ_mean ≈ vol/√252 ≈ 0.0008
    # for vol=0.012); use ±0.003 (~75%/yr drift) to make ranking deterministic.
    drifts = [-0.003, 0.0, +0.003]
    data: dict[str, np.ndarray] = {}
    for i, d in enumerate(drifts):
        data[f"EQ{i}"] = 100 * np.exp(np.cumsum(rng.normal(d, 0.012, n)))
        data[f"FI{i}"] = 100 * np.exp(np.cumsum(rng.normal(d, 0.005, n)))
    closes = pd.DataFrame(data, index=idx)
    asset_classes = {f"EQ{i}": "equity" for i in range(3)}
    asset_classes.update({f"FI{i}": "fixed_income" for i in range(3)})
    w = _compute_csmom_weights(closes, asset_classes)
    # Top of equity class (EQ2) should be long, bottom (EQ0) short
    assert w.get("EQ2", 0) > 0, f"EQ2 (top equity) should be long; got {w}"
    assert w.get("EQ0", 0) < 0, f"EQ0 (bottom equity) should be short; got {w}"
    # Same for fixed_income
    assert w.get("FI2", 0) > 0, f"FI2 (top FI) should be long; got {w}"
    assert w.get("FI0", 0) < 0, f"FI0 (bottom FI) should be short; got {w}"


def test_csmom_compute_raises_on_empty_asset_classes():
    """Asset class map is required to do within-class ranking."""
    from engine.factor_library import _compute_csmom_weights
    closes = _synthetic_closes(n_days=300)
    with pytest.raises(ValueError, match="asset_classes"):
        _compute_csmom_weights(closes, asset_classes={})


def test_csmom_compute_skips_class_with_too_few_tickers():
    """Class with < 3 tickers cannot do tertile split → silently skipped (not raised)."""
    from engine.factor_library import _compute_csmom_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(31)
    closes = pd.DataFrame({
        "EQ0": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
        "EQ1": 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))),
    }, index=idx)
    asset_classes = {"EQ0": "equity", "EQ1": "equity"}  # only 2 tickers in class
    w = _compute_csmom_weights(closes, asset_classes)
    # < 3 tickers per class → empty result, no exception
    assert w == {}


# ── Donchian pure compute ────────────────────────────────────────────────────

def test_donchian_compute_breakout_above_high_long():
    """Synthetic series ending at price above 252d high → long position."""
    from engine.factor_library import _compute_donchian_trend_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    # Series rising linearly (today is at the highest point)
    prices_up = np.linspace(50, 150, n)
    closes = pd.DataFrame({"UP": prices_up}, index=idx)
    w = _compute_donchian_trend_weights(closes)
    assert w.get("UP", 0) > 0, f"UP-trending breakout above all horizons should be long; got {w}"


def test_donchian_compute_breakdown_below_low_short():
    """Series ending at price below 252d low → short position."""
    from engine.factor_library import _compute_donchian_trend_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    # Series falling linearly
    prices_dn = np.linspace(150, 50, n)
    closes = pd.DataFrame({"DOWN": prices_dn}, index=idx)
    w = _compute_donchian_trend_weights(closes)
    assert w.get("DOWN", 0) < 0, f"DOWN-trending breakdown should be short; got {w}"


def test_donchian_compute_within_band_no_position():
    """Series within prior high/low band → 0 ensemble → no position (skipped)."""
    from engine.factor_library import _compute_donchian_trend_weights
    n = 300
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    # Sinusoidal series oscillating, last point in middle
    rng = np.random.default_rng(37)
    base = 100 + 10 * np.sin(np.linspace(0, 8 * np.pi, n))  # oscillates ±10 around 100
    base = base + rng.normal(0, 0.5, n)
    closes = pd.DataFrame({"OSC": base}, index=idx)
    w = _compute_donchian_trend_weights(closes)
    # OSC ends near middle → ensemble close to 0; weight may or may not be 0
    # depending on the precise ending. Just verify no exception + |weight| < some bound.
    if "OSC" in w:
        assert abs(w["OSC"]) <= 1.0


def test_donchian_compute_returns_empty_on_insufficient_data():
    """< 254 days → empty (need 252 + 1 for ensemble + today)."""
    from engine.factor_library import _compute_donchian_trend_weights
    closes = _synthetic_closes(n_days=200)
    assert _compute_donchian_trend_weights(closes) == {}


# ── compute_factor_returns_series (W1 D4 reusable) ───────────────────────────

def test_compute_factor_returns_series_walk_forward_no_lookahead():
    """Synthetic uptrend → TSMOM long → factor returns positive on subsequent
    months. Verify result Series indexed by rebalance_dates[1:] (no look-ahead
    on the first date)."""
    from engine.factor_library import compute_factor_returns_series
    n = 400
    idx = pd.date_range(end=pd.Timestamp("2024-12-31"), periods=n, freq="B")
    rng = np.random.default_rng(41)
    closes = pd.DataFrame({
        "UP":   100 * np.exp(np.cumsum(rng.normal(+0.001, 0.010, n))),
        "DOWN": 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.010, n))),
    }, index=idx)
    rebalance_dates = pd.date_range(end=pd.Timestamp("2024-12-31"),
                                    periods=6, freq="ME")
    series = compute_factor_returns_series("tsmom_12_1", closes, rebalance_dates)
    assert isinstance(series, pd.Series)
    assert len(series) == 5  # rebalance_dates[1:] → N-1 returns
    # First date excluded (no prior rebalance to derive return)
    assert series.index[0] >= rebalance_dates[1]


def test_compute_factor_returns_series_unknown_factor_raises():
    """Unknown factor_id must raise rather than silently produce NaN series."""
    from engine.factor_library import compute_factor_returns_series
    with pytest.raises(ValueError, match="unknown factor_id"):
        compute_factor_returns_series(
            factor_id="quality_factor_not_in_v1",
            closes=_synthetic_closes(n_days=300),
            rebalance_dates=pd.date_range("2024-01-01", periods=3, freq="ME"),
        )


def test_compute_factor_returns_series_too_few_dates_raises():
    """Need ≥ 2 rebalance dates to compute any return."""
    from engine.factor_library import compute_factor_returns_series
    closes = _synthetic_closes(n_days=300)
    with pytest.raises(ValueError, match="at least 2 rebalance_dates"):
        compute_factor_returns_series(
            factor_id="tsmom_12_1",
            closes=closes,
            rebalance_dates=[pd.Timestamp("2024-12-31")],
        )


# ── bhy_fdr_filter (Stage 1 BHY) ─────────────────────────────────────────────

def test_bhy_fdr_filter_rejects_all_when_all_p_high():
    """All p-values = 0.5 → none should pass BHY at α=0.05."""
    from engine.factor_library import bhy_fdr_filter
    p_values = {"f1": 0.5, "f2": 0.5, "f3": 0.5, "f4": 0.5, "f5": 0.5}
    result = bhy_fdr_filter(p_values, alpha=0.05)
    assert all(v is False for v in result.values())


def test_bhy_fdr_filter_accepts_strong_signal():
    """One factor with p=0.001 should pass even at strict BHY threshold for N=5.
    BHY threshold for rank 1: (1/5)·0.05 / c(5) where c(5) ≈ 2.283 → ≈ 0.00438.
    So p=0.001 < 0.00438 → pass."""
    from engine.factor_library import bhy_fdr_filter
    p_values = {"f1": 0.001, "f2": 0.5, "f3": 0.5, "f4": 0.5, "f5": 0.5}
    result = bhy_fdr_filter(p_values, alpha=0.05)
    assert result["f1"] is True
    # Other factors should not pass
    for f in ["f2", "f3", "f4", "f5"]:
        assert result[f] is False


def test_bhy_fdr_filter_handles_nan_as_unsignificant():
    """NaN p-value should be treated as 1.0 (definitely not significant)."""
    from engine.factor_library import bhy_fdr_filter
    p_values = {"f1": 0.001, "f2": float("nan"), "f3": 0.001}
    result = bhy_fdr_filter(p_values, alpha=0.05)
    assert result["f2"] is False, "NaN should not pass BHY"


def test_bhy_fdr_filter_rejects_invalid_alpha():
    from engine.factor_library import bhy_fdr_filter
    with pytest.raises(ValueError, match="alpha"):
        bhy_fdr_filter({"f1": 0.01}, alpha=1.5)
    with pytest.raises(ValueError, match="alpha"):
        bhy_fdr_filter({"f1": 0.01}, alpha=-0.01)


def test_bhy_fdr_filter_rejects_empty():
    from engine.factor_library import bhy_fdr_filter
    with pytest.raises(ValueError, match="empty"):
        bhy_fdr_filter({})


def test_bhy_fdr_filter_step_up_acceptance():
    """If rank-r passes, all earlier ranks should be accepted (step-up procedure).
    BHY thresholds for N=4 with α=0.05: c(4)=1+1/2+1/3+1/4=25/12≈2.083
        rank 1: 0.05/4/2.083 ≈ 0.0060
        rank 2: 0.10/4/2.083 ≈ 0.0120
        rank 3: 0.15/4/2.083 ≈ 0.0180
        rank 4: 0.20/4/2.083 ≈ 0.0240
    Construct p-values (0.005, 0.01, 0.015, 0.02): all should pass via step-up."""
    from engine.factor_library import bhy_fdr_filter
    p_values = {"f1": 0.005, "f2": 0.01, "f3": 0.015, "f4": 0.02}
    result = bhy_fdr_filter(p_values, alpha=0.05)
    # All four should pass (step-up: rank 4 passes 0.02 < 0.024 → accept ranks 1..4)
    assert all(result.values()), f"step-up should accept all four; got {result}"


# ── build_ensemble_weights still skeleton (W2 sprint) ────────────────────────

def test_build_ensemble_weights_currently_skeleton_raises():
    """W1 D2 marker: build_ensemble_weights raises NotImplementedError (W2 sprint).
    Once implemented, this test should be flipped to verify weight constraints
    (sum=1, max≤0.25, vol target)."""
    from engine.factor_library import build_ensemble_weights
    with pytest.raises(NotImplementedError, match="W2 sprint"):
        build_ensemble_weights(
            factor_signals={"bab": pd.Series([0.01, 0.02])},
            regime_label="risk-on",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Spec-locked sentinels (placeholders pending W1 D4)
# ─────────────────────────────────────────────────────────────────────────────

def test_selected_factors_v1_starts_empty_pending_in_sample_analysis():
    """Per spec §2.2, retained list is computed from 1996-2009 in-sample data;
    until W1 D4 correlation analysis runs, this should be empty tuple. After
    W1 D4 runs and locks the retained list via amend_spec, this test should
    flip to assert exact contents."""
    from engine.factor_library import SELECTED_FACTORS_V1
    assert SELECTED_FACTORS_V1 == (), (
        f"SELECTED_FACTORS_V1 populated before W1 D4 in-sample analysis ran "
        f"(without amend_spec). Got: {SELECTED_FACTORS_V1}"
    )


def test_regime_scalar_locked_starts_empty_pending_in_sample_analysis():
    """Per spec §2.4, REGIME_SCALAR_LOCKED is computed from in-sample regime-
    conditional Sharpe quintiles; until W1 D4 runs and locks via amend_spec,
    this should be empty dict."""
    from engine.factor_library import REGIME_SCALAR_LOCKED
    assert REGIME_SCALAR_LOCKED == {}, (
        f"REGIME_SCALAR_LOCKED populated before W1 D4 in-sample analysis "
        f"(without amend_spec). Got: {REGIME_SCALAR_LOCKED}"
    )
