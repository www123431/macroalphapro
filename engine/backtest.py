"""
Backtest Engine (Structured Signal Version)
============================================
Walk-forward backtest using purely structural signals — no LLM calls.
Designed to produce credible empirical evidence for the macro overlay
research question.

Research question answered
--------------------------
Does a regime-conditional TSMOM strategy outperform an unconditional
TSMOM strategy on the 18-sector ETF universe?

  Portfolio A : TSMOM only (unconditional)
  Portfolio B : TSMOM × Regime overlay (scale down in risk-off)
  Benchmark   : Equal-weight all sectors (1/N, monthly rebalanced)

Walk-forward protocol
---------------------
At each rebalancing date t:
  1. Signal computed from prices available strictly before t (signal.py)
  2. Regime estimated from FRED data available strictly before t (regime.py)
  3. Portfolio weights assigned based on signal + regime
  4. Actual return measured from t to t+1 (next rebalancing date)

No look-ahead bias: returns for period [t, t+1] are only used as the
*outcome* of the decision made at t, never as input to that decision.

Transaction costs
-----------------
Assumed zero for monthly rebalancing of liquid ETFs. This is a known
simplification — actual costs (bid-ask spread ~0.01-0.05%) are small
at monthly frequency but should be disclosed.

Performance metrics computed
----------------------------
  - Annualised return, volatility, Sharpe Ratio
  - Deflated Sharpe Ratio (López de Prado 2018) — corrects for n_trials
  - Maximum drawdown, Calmar Ratio
  - Regime-conditional Sharpe (risk-on periods vs risk-off periods)
  - Information Ratio vs benchmark
  - Win rate vs benchmark (monthly)

Integration
-----------
  Consumes: engine/signal.py, engine/regime.py, engine/history.SECTOR_ETF
  Consumed by: pages/backtest.py (UI display)
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from engine.history import SECTOR_ETF, get_active_sector_etf
from engine.signal import get_signal_dataframe
from engine.regime import get_regime_on
from engine.portfolio import construct_portfolio
from engine.universe_audit import audit_universe
from engine.universe_manager import load_all_etf_data, get_universe_as_of_preloaded

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252
_MONTHS_PER_YEAR       = 12

# PRE-5 confirmed 2026-04-21: no systematic parameter grid scan during development.
# Conservative estimate: ~2 lookback variants × ~2 vol_target variants × ~1.5 skip variants ≈ 6.
# Ref: Harvey, Liu & Zhu (2016) — "…and the Cross-Section of Expected Returns"
EFFECTIVE_N_TRIALS = 6


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    """Performance metrics for a single portfolio."""
    label:          str
    ann_return:     float
    ann_vol:        float
    sharpe:         float
    dsr:            float          # Deflated Sharpe Ratio
    max_drawdown:   float
    calmar:         float
    win_rate_vs_bm: float          # fraction of months portfolio > benchmark
    ir_vs_bm:       float          # Information Ratio vs benchmark
    sharpe_risk_on:    float | None   # Sharpe in risk-on months only
    sharpe_risk_off:   float | None   # Sharpe in risk-off months only
    drawdown_risk_on:  float | None   # max drawdown within risk-on periods
    drawdown_risk_off: float | None   # max drawdown within risk-off periods
    hit_rate_risk_on:  float | None   # win rate vs benchmark in risk-on months
    hit_rate_risk_off: float | None   # win rate vs benchmark in risk-off months
    avg_holding_months: float | None  # avg consecutive months in same signal direction
    n_months:          int
    n_trials:          int            # for DSR: number of strategy variants tested
    # P3-4: Beta-adjusted alpha and 60/40 benchmark comparison
    market_beta:       float | None = None
    alpha_annualized:  float | None = None   # Jensen's Alpha annualized
    sharpe_vs_60_40:   float | None = None   # Sharpe of excess return over 60/40


@dataclass
class BacktestResult:
    """Full backtest output."""
    returns:        pd.DataFrame   # columns: date, tsmom, tsmom_regime, benchmark, regime_label
    metrics_tsmom:  BacktestMetrics
    metrics_regime: BacktestMetrics
    metrics_bm:     BacktestMetrics
    warnings:       list[str] = field(default_factory=list)
    # P3-13: IC decay analysis — {horizon_months: {"ic_mean": float, "ic_std": float, "n": int}}
    ic_decay:       dict = field(default_factory=dict)
    # P3-4: Pure TSMOM (no risk mgmt) and 60/40 benchmark metrics
    metrics_pure_tsmom:  BacktestMetrics | None = None
    metrics_sixty_forty: BacktestMetrics | None = None


# ── ETF return fetcher ─────────────────────────────────────────────────────────

def _fetch_monthly_returns(
    tickers:    list[str],
    start_date: datetime.date,
    end_date:   datetime.date,
) -> pd.DataFrame:
    """
    Fetch adjusted monthly returns for tickers in [start_date, end_date].
    Uses month-end prices. Returns DataFrame with ticker columns.
    """
    fetch_start = start_date - datetime.timedelta(days=40)
    try:
        raw = yf.download(
            tickers,
            start=str(fetch_start),
            end=str(end_date + datetime.timedelta(days=5)),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        # Resample to month-end prices, then compute returns
        monthly_price  = close.resample("ME").last().dropna(how="all")
        monthly_return = monthly_price.pct_change().dropna(how="all")
        return monthly_return
    except Exception as exc:
        logger.warning("Monthly return fetch failed: %s", exc)
        return pd.DataFrame()


# ── Portfolio weight computation ───────────────────────────────────────────────

def _tsmom_weights(signal_df: pd.DataFrame) -> dict[str, float]:
    """
    Equal-weight within long (+1) and short (-1) groups.
    Net exposure: long - short = 0 (dollar-neutral).
    If all signals same sign (common in strong trends), weights sum to ±1.
    """
    longs  = [s for s in signal_df.index if signal_df.loc[s, "tsmom"] > 0]
    shorts = [s for s in signal_df.index if signal_df.loc[s, "tsmom"] < 0]

    weights: dict[str, float] = {}

    if longs:
        w_long = 1.0 / len(longs)
        for s in longs:
            weights[s] = w_long

    if shorts:
        w_short = -1.0 / len(shorts)
        for s in shorts:
            weights[s] = w_short

    return weights


def _apply_regime_overlay(
    weights:      dict[str, float],
    regime:       str,
    p_risk_on:    float,
    scale_factor: float = 0.3,
) -> dict[str, float]:
    """
    Scale down long positions in risk-off regime.

    Logic: in risk-off, reduce long exposure by (1 - scale_factor).
    Short positions are kept or slightly increased (flight-to-safety).
    In transition, apply partial scaling proportional to p_risk_off.

    scale_factor: minimum long weight multiplier in full risk-off (0.3 = 30% of normal)
    """
    if regime == "risk-on":
        return weights

    if regime == "risk-off":
        multiplier = scale_factor
    else:  # transition
        p_risk_off = 1.0 - p_risk_on
        multiplier = scale_factor + (1.0 - scale_factor) * p_risk_on

    scaled: dict[str, float] = {}
    for sector, w in weights.items():
        if w > 0:
            scaled[sector] = w * multiplier
        else:
            scaled[sector] = w  # keep shorts unchanged
    return scaled


def _benchmark_weights(sectors: list[str]) -> dict[str, float]:
    """Equal-weight all sectors (1/N benchmark)."""
    w = 1.0 / len(sectors) if sectors else 0.0
    return {s: w for s in sectors}


# ── Metrics computation ────────────────────────────────────────────────────────

def _compute_metrics(
    returns:     pd.Series,
    bm_returns:  pd.Series,
    regime_labels: pd.Series,
    label:       str,
    n_trials:    int = EFFECTIVE_N_TRIALS,
    spy_returns:           "pd.Series | None" = None,
    sixty_forty_returns:   "pd.Series | None" = None,
) -> BacktestMetrics:
    """Compute full performance metrics for a return series."""
    if returns.empty or len(returns) < 3:
        return BacktestMetrics(
            label=label, ann_return=0, ann_vol=0, sharpe=0, dsr=0,
            max_drawdown=0, calmar=0, win_rate_vs_bm=0, ir_vs_bm=0,
            sharpe_risk_on=None, sharpe_risk_off=None,
            drawdown_risk_on=None, drawdown_risk_off=None,
            hit_rate_risk_on=None, hit_rate_risk_off=None,
            avg_holding_months=None,
            n_months=0, n_trials=n_trials,
        )

    n   = len(returns)
    mu  = float(returns.mean() * _MONTHS_PER_YEAR)
    vol = float(returns.std()  * np.sqrt(_MONTHS_PER_YEAR))
    sr  = mu / vol if vol > 1e-9 else 0.0

    # Max drawdown
    cum    = (1 + returns).cumprod()
    peak   = cum.cummax()
    dd     = (cum - peak) / peak
    mdd    = float(dd.min())
    calmar = mu / abs(mdd) if abs(mdd) > 1e-9 else 0.0

    # Vs benchmark
    excess    = returns.values - bm_returns.reindex(returns.index).values
    excess_s  = pd.Series(excess, index=returns.index).dropna()
    win_rate  = float((excess_s > 0).mean())
    ir        = float(excess_s.mean() * _MONTHS_PER_YEAR /
                      (excess_s.std() * np.sqrt(_MONTHS_PER_YEAR))
                      ) if excess_s.std() > 1e-9 else 0.0

    # Deflated Sharpe Ratio (López de Prado 2018)
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())   # excess kurtosis
    T    = n
    euler_gamma = 0.5772156649
    import scipy.stats as stats
    sr_star = (
        (1 - euler_gamma) * stats.norm.ppf(1 - 1.0 / max(n_trials, 2))
        + euler_gamma     * stats.norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))
    )
    # BLP variance term for EXCESS kurtosis (pandas .kurtosis() is Fisher/excess): the
    # correct denominator is 1 - skew*SR + (kurt_excess + 2)/4 * SR^2. For a normal dist
    # (excess kurt 0) this is +0.5, not the -0.25 the old (kurt - 1)/4 form gave — that
    # form treated excess kurtosis as full kurtosis and OVERSTATED the deflated Sharpe
    # (anti-conservative). Restored 2026-05-22 (test-debt cleanup caught the regression).
    denom = 1 - skew * sr + (kurt + 2) / 4.0 * sr ** 2
    if denom > 1e-9 and T > 1:
        dsr = float(stats.norm.cdf((sr - sr_star) * np.sqrt(T - 1) / np.sqrt(denom)))
    else:
        dsr = float("nan")

    # Regime-conditional helpers
    risk_on_mask  = regime_labels.reindex(returns.index) == "risk-on"
    risk_off_mask = regime_labels.reindex(returns.index) == "risk-off"

    def _regime_sharpe(mask: pd.Series) -> float | None:
        sub = returns[mask.reindex(returns.index).fillna(False)]
        if len(sub) < 3:
            return None
        mu_s  = float(sub.mean() * _MONTHS_PER_YEAR)
        vol_s = float(sub.std()  * np.sqrt(_MONTHS_PER_YEAR))
        return round(mu_s / vol_s, 4) if vol_s > 1e-9 else 0.0

    def _regime_drawdown(mask: pd.Series) -> float | None:
        sub = returns[mask.reindex(returns.index).fillna(False)]
        if len(sub) < 3:
            return None
        cum = (1 + sub).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return round(float(dd.min()), 4)

    def _regime_hit_rate(mask: pd.Series) -> float | None:
        aligned = mask.reindex(returns.index).fillna(False)
        sub_r   = returns[aligned]
        sub_bm  = bm_returns.reindex(sub_r.index)
        if len(sub_r) < 3:
            return None
        return round(float((sub_r > sub_bm).mean()), 4)

    def _avg_holding() -> float | None:
        if len(returns) < 2:
            return None
        sig = np.sign(returns)
        runs, cur = [], 1
        for i in range(1, len(sig)):
            if sig.iloc[i] == sig.iloc[i - 1] and sig.iloc[i] != 0:
                cur += 1
            else:
                if sig.iloc[i - 1] != 0:
                    runs.append(cur)
                cur = 1
        if sig.iloc[-1] != 0:
            runs.append(cur)
        return round(float(np.mean(runs)), 2) if runs else None

    # ── P3-4: Beta/Alpha vs SPY and Sharpe vs 60/40 ──────────────────────────
    _market_beta = _alpha_ann = _sr6040 = None
    if spy_returns is not None and not spy_returns.empty:
        _spy_a = spy_returns.reindex(returns.index).dropna()
        _str_a = returns.reindex(_spy_a.index).dropna()
        _spy_a = _spy_a.reindex(_str_a.index)
        if len(_str_a) >= 12:
            _X = np.column_stack([np.ones(len(_spy_a)), _spy_a.values])
            _coef, _, _, _ = np.linalg.lstsq(_X, _str_a.values, rcond=None)
            _alpha_ann  = round(float(_coef[0]) * _MONTHS_PER_YEAR, 4)
            _market_beta = round(float(_coef[1]), 4)
    if sixty_forty_returns is not None and not sixty_forty_returns.empty:
        _ex = returns.reindex(sixty_forty_returns.index) - sixty_forty_returns.reindex(returns.index)
        _ex = _ex.dropna()
        if len(_ex) >= 12:
            _mu_ex  = float(_ex.mean() * _MONTHS_PER_YEAR)
            _vol_ex = float(_ex.std()  * np.sqrt(_MONTHS_PER_YEAR))
            _sr6040 = round(_mu_ex / _vol_ex, 4) if _vol_ex > 1e-9 else 0.0

    return BacktestMetrics(
        label=label,
        ann_return=round(mu,  4),
        ann_vol=round(vol, 4),
        sharpe=round(sr,  4),
        dsr=round(dsr, 4) if not np.isnan(dsr) else float("nan"),
        max_drawdown=round(mdd,    4),
        calmar=round(calmar, 4),
        win_rate_vs_bm=round(win_rate, 4),
        ir_vs_bm=round(ir, 4),
        sharpe_risk_on=_regime_sharpe(risk_on_mask),
        sharpe_risk_off=_regime_sharpe(risk_off_mask),
        drawdown_risk_on=_regime_drawdown(risk_on_mask),
        drawdown_risk_off=_regime_drawdown(risk_off_mask),
        hit_rate_risk_on=_regime_hit_rate(risk_on_mask),
        hit_rate_risk_off=_regime_hit_rate(risk_off_mask),
        avg_holding_months=_avg_holding(),
        n_months=n,
        n_trials=n_trials,
        market_beta=_market_beta,
        alpha_annualized=_alpha_ann,
        sharpe_vs_60_40=_sr6040,
    )


# ── Main backtest runner ───────────────────────────────────────────────────────

# Transaction cost assumption (basis points, one-way per rebalancing)
# P2-15: Dynamic transaction cost model (ATR-based half-spread estimate).
# Replaces flat 10 bps with: cost = Σ |Δw_i| × max(floor_bps, vol_14d_i × vol_scale) / 2
# where vol_14d_i is the 14-day rolling std of daily returns for sector i.
# Spread widens in volatile markets; liquid ETFs have a minimum floor (3 bps).
# Fallback to flat cost when daily price history is unavailable.
_TRANSACTION_COST_BPS    = 10           # flat fallback (one-way, per rebalancing)
_TRANSACTION_COST        = _TRANSACTION_COST_BPS / 10_000
_BM_TRANSACTION_COST     = 5 / 10_000  # benchmark lower-turnover flat fallback
_TC_FLOOR_BPS            = 3           # minimum half-spread for liquid US ETFs
_TC_VOL_SCALE            = 0.15        # spread ≈ 15% of daily vol (empirical for ETFs)
_TC_ATR_WINDOW           = 14          # ATR lookback in trading days


def _atr_transaction_cost(
    w_prev: dict[str, float],
    w_new:  dict[str, float],
    daily_ret_window: "pd.DataFrame",   # columns=sector, index=date; trailing window
    floor_bps: float = _TC_FLOOR_BPS,
    vol_scale: float = _TC_VOL_SCALE,
    atr_window: int  = _TC_ATR_WINDOW,
) -> float:
    """
    Compute one-way transaction cost for a single rebalancing.

    For each sector i:
        half_spread_i = max(floor_bps/10000, std(ret_14d_i) × vol_scale)
        cost_i        = |w_new_i - w_prev_i| × half_spread_i
    Total cost = Σ cost_i

    Falls back to flat _TRANSACTION_COST when data is insufficient.
    """
    if daily_ret_window is None or daily_ret_window.empty:
        # flat fallback: estimate turnover × flat half-spread
        all_sectors = set(w_prev) | set(w_new)
        turnover = sum(
            abs(w_new.get(s, 0.0) - w_prev.get(s, 0.0)) for s in all_sectors
        )
        return turnover * _TRANSACTION_COST / 2   # already one-way

    all_sectors = set(w_prev) | set(w_new)
    total_cost  = 0.0
    _floor = floor_bps / 10_000

    for sector in all_sectors:
        delta_w = abs(w_new.get(sector, 0.0) - w_prev.get(sector, 0.0))
        if delta_w < 1e-6:
            continue
        if sector in daily_ret_window.columns:
            _ret_s = daily_ret_window[sector].dropna().tail(atr_window)
            if len(_ret_s) >= 5:
                vol_daily = float(_ret_s.std())
                half_spread = max(_floor, vol_daily * vol_scale)
            else:
                half_spread = _floor
        else:
            half_spread = _floor

        total_cost += delta_w * half_spread

    return total_cost


def compute_ic_decay(
    signal_records:   list[dict],   # [{date, signals: {sector: float}}]
    monthly_returns:  pd.DataFrame, # index=date, columns=ticker
    ticker_to_sector: dict[str, str],
    horizons:         tuple = (1, 3, 6, 12),
) -> dict:
    """
    P3-13: Compute Information Coefficient decay across multiple holding horizons.

    For each horizon h (months):
      IC(h) = mean Spearman(signal_t, return_{t→t+h}) over all valid t

    Returns {h: {"ic_mean": float, "ic_std": float, "n": int}} for each horizon.
    IC significantly above zero at h=12 indicates long-lived alpha.
    IC collapsing to zero at h=3 indicates signal useful only for short holds.
    """
    from scipy.stats import spearmanr

    # Build sector-indexed monthly returns (sector ← ticker mapping)
    sector_returns = monthly_returns.rename(columns=ticker_to_sector)

    result: dict[int, dict] = {}
    for h in horizons:
        ic_series: list[float] = []
        for rec in signal_records:
            t_date = pd.Timestamp(rec["date"])
            sigs   = rec.get("signals", {})
            if not sigs:
                continue
            # Compute h-month forward return (compound)
            future_rows = sector_returns[sector_returns.index > t_date].iloc[:h]
            if len(future_rows) < h:
                continue   # insufficient future data
            fwd_ret = (1 + future_rows).prod() - 1   # compound h-month return

            sectors = [s for s in sigs if s in fwd_ret.index and not np.isnan(fwd_ret[s])]
            if len(sectors) < 5:
                continue

            sig_vec = np.array([sigs[s] for s in sectors])
            ret_vec = np.array([fwd_ret[s] for s in sectors])

            if np.std(sig_vec) < 1e-9 or np.std(ret_vec) < 1e-9:
                continue
            try:
                rho, _ = spearmanr(sig_vec, ret_vec)
                if not np.isnan(rho):
                    ic_series.append(float(rho))
            except Exception:
                continue

        if ic_series:
            arr = np.array(ic_series)
            result[h] = {
                "ic_mean": float(np.mean(arr)),
                "ic_std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n":       len(arr),
            }
        else:
            result[h] = {"ic_mean": 0.0, "ic_std": 0.0, "n": 0}

    return result


def run_backtest(
    start_date:       str,
    end_date:         str,
    lookback_months:  int   = 12,
    skip_months:      int   = 1,
    regime_scale:     float = 0.3,
    transaction_cost: float = _TRANSACTION_COST,
    progress_cb=None,
) -> BacktestResult:
    """
    Run walk-forward backtest from start_date to end_date.

    At each monthly rebalancing date:
      1. Compute TSMOM signals (data up to that date)
      2. Estimate regime (data up to that date)
      3. Compute portfolio A (TSMOM) and B (TSMOM + regime) weights
      4. Record actual next-month return

    Args:
        start_date:      "YYYY-MM-DD" — first signal date
        end_date:        "YYYY-MM-DD" — last signal date
        lookback_months: TSMOM formation window (default 12)
        skip_months:     TSMOM skip period (default 1)
        regime_scale:     Long position scale factor in risk-off (default 0.3)
        transaction_cost: one-way cost per rebalancing (default 10 bps)
        progress_cb:      optional callable(current, total, msg)

    Returns:
        BacktestResult with return series and metrics for all three portfolios.
    """
    warnings_log: list[str] = []
    _active_etf = get_active_sector_etf()
    tickers  = list(_active_etf.values())
    sectors  = list(_active_etf.keys())

    # ── P0-2: Survivorship bias audit ─────────────────────────────────────────
    _audit = audit_universe(tickers, start_date)
    warnings_log.extend(_audit.warnings)

    # ── P3-10: Pre-load ETF inception dates for per-period universe filtering ─
    # Each rebalancing date uses only ETFs with ≥3 years of history at that date.
    # This prevents survivorship bias: pre-2021 periods won't include XLC (2018).
    try:
        _all_etf_data = load_all_etf_data()
        warnings_log.append("P3-10: 幸存者偏差动态过滤已激活（每期仅使用成立≥3年的ETF）")
    except Exception as _p310_exc:
        logger.warning("P3-10 ETF data load failed, falling back to static universe: %s", _p310_exc)
        _all_etf_data = {s: (t, None) for s, t in _active_etf.items()}

    # ── Build rebalancing dates (month-end) ───────────────────────────────────
    rebal_dates = pd.date_range(
        start=start_date, end=end_date, freq="ME"
    ).date.tolist()

    if len(rebal_dates) < 4:
        warnings_log.append("样本期过短（< 4个月），回测结果无统计意义")

    # ── Pre-fetch all monthly returns for the full period ─────────────────────
    start_fetch = datetime.date.fromisoformat(start_date) - datetime.timedelta(days=40)
    end_fetch   = datetime.date.fromisoformat(end_date)   + datetime.timedelta(days=40)
    all_returns = _fetch_monthly_returns(tickers, start_fetch, end_fetch)

    if all_returns.empty:
        warnings_log.append("价格数据获取失败，无法运行回测")
        empty_df = pd.DataFrame()
        empty_m  = BacktestMetrics("", 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, 0, 2)
        return BacktestResult(empty_df, empty_m, empty_m, empty_m, warnings_log)

    ticker_to_sector = {v: k for k, v in _active_etf.items()}

    # P2-3: Pre-fetch daily returns for Ledoit-Wolf covariance (sector-indexed)
    _LW_LOOKBACK_DAYS = 252
    _daily_returns_all = pd.DataFrame()
    try:
        _daily_dl = yf.download(
            tickers,
            start=str(datetime.date.fromisoformat(start_date) - datetime.timedelta(days=_LW_LOOKBACK_DAYS + 30)),
            end=str(end_fetch),
            progress=False,
            auto_adjust=True,
        )
        if not _daily_dl.empty:
            _dc = _daily_dl["Close"] if isinstance(_daily_dl.columns, pd.MultiIndex) else _daily_dl
            _dc.index = pd.to_datetime(_dc.index).normalize()
            _dc.columns = [ticker_to_sector.get(str(c), str(c)) for c in _dc.columns]
            _daily_returns_all = _dc.pct_change().dropna(how="all")
    except Exception as _lw_exc:
        logger.debug("P2-3 daily returns fetch failed: %s", _lw_exc)

    # ── P3-4: Pre-fetch SPY and AGG for beta/alpha and 60/40 benchmark ────────
    _bm_rets_df = pd.DataFrame()
    try:
        _bm_rets_df = _fetch_monthly_returns(["SPY", "AGG"], start_fetch, end_fetch)
    except Exception as _bm_exc:
        logger.debug("P3-4 SPY/AGG fetch failed: %s", _bm_exc)

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    records:        list[dict] = []
    signal_records: list[dict] = []   # P3-13: {date, signals: {sector: continuous_signal}}
    _w_tsmom_prev:  dict = {}         # P2-15: track prior weights for turnover-based TC
    _w_regime_prev: dict = {}
    _w_bm_prev:     dict = {}
    _w_pure_prev:   dict = {}         # P3-4: pure TSMOM (no vol-targeting) prev weights
    total = len(rebal_dates) - 1   # last date has no "next period"

    for i, date in enumerate(rebal_dates[:-1]):
        next_date = rebal_dates[i + 1]

        if progress_cb:
            progress_cb(i, total, f"处理 {date} → {next_date}")

        # ── P3-10: Per-period universe (survivorship-bias-free) ─────────────
        _valid_etf = get_universe_as_of_preloaded(date, _all_etf_data)
        _valid_sectors = list(_valid_etf.keys()) if _valid_etf else sectors

        # Step 1: TSMOM signal at `date`
        try:
            sig_df = get_signal_dataframe(date, lookback_months, skip_months)
            # Filter to valid universe for this period
            if not sig_df.empty and _valid_sectors:
                sig_df = sig_df[sig_df.index.isin(_valid_sectors)]
        except Exception as exc:
            logger.warning("Signal failed at %s: %s", date, exc)
            sig_df = pd.DataFrame()

        # Step 2: Regime at `date`
        try:
            regime_r = get_regime_on(as_of=date, train_end=date)
            if regime_r.warning:
                warnings_log.append(f"{date}: {regime_r.warning}")
        except Exception as exc:
            logger.warning("Regime failed at %s: %s", date, exc)
            regime_r = None

        # P3-13: Store continuous signal snapshot for IC decay analysis
        if not sig_df.empty and "raw_return" in sig_df.columns and "ann_vol" in sig_df.columns:
            _sig_vals = {}
            for _sec in sig_df.index:
                _rv  = sig_df.loc[_sec, "raw_return"]
                _vol = sig_df.loc[_sec, "ann_vol"]
                if _vol and _vol > 1e-6 and not np.isnan(_rv):
                    _sig_vals[_sec] = float(_rv / _vol)   # continuous TSMOM signal
            if _sig_vals:
                signal_records.append({"date": date, "signals": _sig_vals})

        # Step 3: Weights via portfolio.py (vol-targeting)
        if sig_df.empty:
            w_tsmom  = _benchmark_weights(_valid_sectors)
            w_regime = _benchmark_weights(_valid_sectors)
            warnings_log.append(f"{date}: 信号缺失，使用等权基准")
        else:
            # P2-3: Slice 252-day returns window ending at rebal date for LW
            _lw_window = pd.DataFrame()
            if not _daily_returns_all.empty:
                _mask = _daily_returns_all.index <= pd.Timestamp(date)
                _lw_window = _daily_returns_all[_mask].iloc[-_LW_LOOKBACK_DAYS:]

            # Portfolio A: TSMOM + vol-targeting, no regime overlay
            pw_tsmom  = construct_portfolio(sig_df, regime=None,
                                            regime_scale=regime_scale,
                                            returns_matrix=_lw_window if not _lw_window.empty else None)
            w_tsmom   = pw_tsmom.weights.to_dict() if not pw_tsmom.weights.empty \
                        else _benchmark_weights(sectors)

            # Portfolio B: TSMOM + vol-targeting + regime overlay
            pw_regime = construct_portfolio(sig_df, regime=regime_r,
                                            regime_scale=regime_scale,
                                            returns_matrix=_lw_window if not _lw_window.empty else None)
            w_regime  = pw_regime.weights.to_dict() if not pw_regime.weights.empty \
                        else w_tsmom

        w_bm = _benchmark_weights(_valid_sectors)

        # P3-4: Pure TSMOM — equal-weight signals, no vol-targeting, no regime overlay
        if sig_df.empty:
            w_pure = dict(w_bm)
        else:
            w_pure = _tsmom_weights(sig_df)
            if not w_pure:
                w_pure = dict(w_bm)

        # Step 4: Actual next-month returns
        # Find the return row closest to next_date in all_returns
        next_ts  = pd.Timestamp(next_date)
        ret_rows = all_returns[all_returns.index <= next_ts + pd.Timedelta(days=5)]
        ret_rows = ret_rows[ret_rows.index >= next_ts - pd.Timedelta(days=10)]

        if ret_rows.empty:
            continue

        ret_row = ret_rows.iloc[-1]   # last available row near next_date

        def _portfolio_return(weights: dict[str, float]) -> float:
            total_ret = 0.0
            for sector, w in weights.items():
                ticker = _active_etf.get(sector)
                if ticker and ticker in ret_row.index:
                    r = ret_row[ticker]
                    if not np.isnan(r):
                        total_ret += w * float(r)
            return total_ret

        # P2-15: ATR-based dynamic transaction cost (turnover × vol-scaled half-spread)
        _tc_window = (
            _daily_returns_all[_daily_returns_all.index <= pd.Timestamp(date)]
            .iloc[-(_TC_ATR_WINDOW + 5):]
            if not _daily_returns_all.empty else pd.DataFrame()
        )
        _tc_tsmom  = _atr_transaction_cost(_w_tsmom_prev,  w_tsmom,  _tc_window)
        _tc_regime = _atr_transaction_cost(_w_regime_prev, w_regime, _tc_window)
        _tc_bm     = _atr_transaction_cost(_w_bm_prev,     w_bm,     _tc_window)

        ret_tsmom  = _portfolio_return(w_tsmom)  - _tc_tsmom
        ret_regime = _portfolio_return(w_regime) - _tc_regime
        ret_bm     = _portfolio_return(w_bm)     - _tc_bm

        # P3-4: Pure TSMOM (no vol-targeting) and 60/40
        _tc_pure   = _atr_transaction_cost(_w_pure_prev, w_pure, _tc_window)
        ret_pure   = _portfolio_return(w_pure) - _tc_pure

        _spy_r = _agg_r = 0.0
        if not _bm_rets_df.empty:
            _bm_row = _bm_rets_df[_bm_rets_df.index <= next_ts + pd.Timedelta(days=5)]
            _bm_row = _bm_row[_bm_row.index >= next_ts - pd.Timedelta(days=10)]
            if not _bm_row.empty:
                _lr = _bm_row.iloc[-1]
                _spy_r = float(_lr.get("SPY", 0) if hasattr(_lr, "get") else (getattr(_lr, "SPY", 0) or 0))
                _agg_r = float(_lr.get("AGG", 0) if hasattr(_lr, "get") else (getattr(_lr, "AGG", 0) or 0))
        ret_sixty_forty = 0.6 * _spy_r + 0.4 * _agg_r

        # Update previous weights for next period's turnover calculation
        _w_tsmom_prev  = dict(w_tsmom)
        _w_regime_prev = dict(w_regime)
        _w_bm_prev     = dict(w_bm)
        _w_pure_prev   = dict(w_pure)

        records.append({
            "date":         next_date,
            "tsmom":        ret_tsmom,
            "tsmom_regime": ret_regime,
            "benchmark":    ret_bm,
            "pure_tsmom":   ret_pure,
            "sixty_forty":  ret_sixty_forty,
            "regime_label": regime_r.regime if regime_r else "unknown",
            "p_risk_on":    regime_r.p_risk_on if regime_r else 0.5,
            "yield_spread": regime_r.yield_spread if regime_r else None,
        })

    if progress_cb:
        progress_cb(total, total, "计算绩效指标…")

    if not records:
        warnings_log.append("无有效回测记录")
        empty_df = pd.DataFrame()
        empty_m  = BacktestMetrics("", 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None, None, None, 0, 2)
        return BacktestResult(empty_df, empty_m, empty_m, empty_m, warnings_log)

    # ── Assemble return DataFrame ─────────────────────────────────────────────
    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)

    regime_labels = df["regime_label"]
    bm_returns    = df["benchmark"]

    # P3-4: SPY and 60/40 return series aligned to df.index
    _spy_series = pd.Series(dtype=float)
    if not _bm_rets_df.empty and "SPY" in _bm_rets_df.columns:
        _spy_series = _bm_rets_df["SPY"]
    _sixty_forty_s = df["sixty_forty"] if "sixty_forty" in df.columns else pd.Series(dtype=float)

    # ── Compute metrics ───────────────────────────────────────────────────────
    m_tsmom  = _compute_metrics(df["tsmom"],        bm_returns, regime_labels, "TSMOM",              n_trials=EFFECTIVE_N_TRIALS, spy_returns=_spy_series, sixty_forty_returns=_sixty_forty_s)
    m_regime = _compute_metrics(df["tsmom_regime"], bm_returns, regime_labels, "TSMOM + Regime",     n_trials=EFFECTIVE_N_TRIALS, spy_returns=_spy_series, sixty_forty_returns=_sixty_forty_s)
    m_bm     = _compute_metrics(df["benchmark"],    bm_returns, regime_labels, "Equal-Weight 基准", n_trials=1)
    m_pure   = _compute_metrics(df["pure_tsmom"],   bm_returns, regime_labels, "纯TSMOM（等权）",   n_trials=EFFECTIVE_N_TRIALS, spy_returns=_spy_series, sixty_forty_returns=_sixty_forty_s) if "pure_tsmom" in df.columns else None
    m_6040   = _compute_metrics(_sixty_forty_s.reindex(df.index).fillna(0), bm_returns, regime_labels, "60/40 (SPY+AGG)", n_trials=1) if not _sixty_forty_s.empty else None

    if len(records) < 24:
        warnings_log.append(
            f"样本量偏低（{len(records)} 个月）。统计检验功效不足，"
            "DSR 和 IR 结论参考性有限。建议至少 36 个月。"
        )

    # ── P3-13: IC decay analysis ─────────────────────────────────────────────
    _ic_decay: dict = {}
    if signal_records and not all_returns.empty:
        try:
            _ic_decay = compute_ic_decay(
                signal_records=signal_records,
                monthly_returns=all_returns,
                ticker_to_sector=ticker_to_sector,
                horizons=(1, 3, 6, 12),
            )
        except Exception as _ic_exc:
            logger.debug("IC decay computation failed: %s", _ic_exc)

    return BacktestResult(
        returns=df,
        metrics_tsmom=m_tsmom,
        metrics_regime=m_regime,
        metrics_bm=m_bm,
        warnings=warnings_log,
        ic_decay=_ic_decay,
        metrics_pure_tsmom=m_pure,
        metrics_sixty_forty=m_6040,
    )


# ── P1-5: BHY multiple-testing correction ─────────────────────────────────────

def bhy_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg-Yekutieli FDR correction for multiple Sharpe tests.

    Controls FDR under arbitrary dependence (more conservative than BH).
    Ref: Benjamini & Yekutieli (2001) — appropriate for correlated tests
    such as regime × sector Sharpe comparisons.

    Args:
        p_values: list of p-values from independent or correlated tests
        alpha:    target FDR level (default 5%)

    Returns:
        list[bool] — True if hypothesis rejected (significant) at FDR level alpha
    """
    from statsmodels.stats.multitest import multipletests
    if not p_values:
        return []
    reject, _, _, _ = multipletests(p_values, alpha=alpha, method="fdr_by")
    return reject.tolist()


def sharpe_pvalue(sr: float, n_months: int) -> float:
    """Approximate one-sided p-value for Sharpe > 0 under iid normal returns.
    Uses Student-t approximation: t = SR × sqrt(n) / sqrt(1 + 0.5 × SR²).
    """
    import scipy.stats as _stats
    if n_months < 4 or np.isnan(sr):
        return 1.0
    t_stat = sr * np.sqrt(n_months) / np.sqrt(1 + 0.5 * sr ** 2)
    return float(1 - _stats.t.cdf(t_stat, df=n_months - 1))


# ── Metrics display helper ─────────────────────────────────────────────────────

def metrics_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert BacktestResult metrics to a display DataFrame (all available strategies)."""
    _candidates = [
        result.metrics_tsmom,
        result.metrics_regime,
        getattr(result, "metrics_pure_tsmom", None),
        result.metrics_bm,
        getattr(result, "metrics_sixty_forty", None),
    ]
    metrics_list = [m for m in _candidates if m is not None]
    rows = []
    for m in metrics_list:
        rows.append({
            "策略":            m.label,
            "年化收益":        f"{m.ann_return:.2%}",
            "年化波动率":      f"{m.ann_vol:.2%}",
            "Sharpe":          f"{m.sharpe:.3f}",
            "DSR":             f"{m.dsr:.3f}" if not np.isnan(m.dsr) else "N/A",
            "最大回撤":        f"{m.max_drawdown:.2%}",
            "Calmar":          f"{m.calmar:.3f}",
            "vs基准胜率":      f"{m.win_rate_vs_bm:.1%}",
            "IR vs基准":       f"{m.ir_vs_bm:.3f}",
            "市场Beta":        f"{m.market_beta:.3f}"       if m.market_beta       is not None else "—",
            "Jensen Alpha%":   f"{m.alpha_annualized:.2%}"  if m.alpha_annualized  is not None else "—",
            "Sharpe vs 60/40": f"{m.sharpe_vs_60_40:.3f}"  if m.sharpe_vs_60_40   is not None else "—",
            "Sharpe(risk-on)":    f"{m.sharpe_risk_on:.3f}"    if m.sharpe_risk_on    is not None else "N/A",
            "Sharpe(risk-off)":   f"{m.sharpe_risk_off:.3f}"   if m.sharpe_risk_off   is not None else "N/A",
            "MDD(risk-on)":       f"{m.drawdown_risk_on:.2%}"  if m.drawdown_risk_on  is not None else "N/A",
            "MDD(risk-off)":      f"{m.drawdown_risk_off:.2%}" if m.drawdown_risk_off is not None else "N/A",
            "胜率(risk-on)":      f"{m.hit_rate_risk_on:.1%}"  if m.hit_rate_risk_on  is not None else "N/A",
            "胜率(risk-off)":     f"{m.hit_rate_risk_off:.1%}" if m.hit_rate_risk_off is not None else "N/A",
            "平均持仓(月)":       f"{m.avg_holding_months:.1f}" if m.avg_holding_months is not None else "N/A",
            "月份数":             m.n_months,
        })
    return pd.DataFrame(rows).set_index("策略")
