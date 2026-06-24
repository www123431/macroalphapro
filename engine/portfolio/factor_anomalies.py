"""
engine/portfolio/factor_anomalies.py — Path W (52-WH momentum) + Path X (IVOL).

Sibling pre-registered specs:
  Path W: docs/spec_path_w_52week_high_momentum_v1.md (spec_id=70, hash 758206c5)
  Path X: docs/spec_path_x_idiosyncratic_volatility_v1.md (spec_id=71, hash 0c71f9a5)

Shared universe / window / TC / rebalance / gates per Path V Amendment 1.
Differ only in signal mechanism — clean sibling test.

Path W signal (George-Hwang 2004 *JF*):
  high_52w_i(t) = max(price_i over t-52w to t-1w)  # excludes current week
  signal_i(t)   = price_i(t) / high_52w_i(t)        # ratio in [0, 1]
  rank ascending → top decile long-only (highest ratio = nearest 52WH)

Path X signal (Ang-Hodrick-Xing-Zhang 2006 *JF*):
  For each stock i and each rebalance week t:
    22 trailing daily returns (r_i, r_SPY)
    OLS: r_i = α + β × r_SPY + ε
    IVOL_i(t) = std(ε) × sqrt(252)  # annualized residual std
  rank ascending → bottom decile long-only (lowest IVOL = defensive)

Doctrine: no LLM, no spec parameter changes post-impl, pre-reg locked.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (shared with Path V; mutating requires amendment)
# ─────────────────────────────────────────────────────────────────────────────
CRSP_PANEL_PATH:    str   = "data/factor_ensemble_singlename/_crsp_dsf_panel.parquet"
WINDOW_START:       str   = "2014-09-12"
WINDOW_END:         str   = "2023-12-29"

# Path W: 52-week lookback signal
LOOKBACK_WEEKS_W:   int   = 52   # full 12-month rolling max
EXCLUDE_CURRENT_W:  int   = 1    # exclude current week (t-1 ... t-52)

# Path X: IVOL 22-day rolling regression
IVOL_DAYS:          int   = 22
MARKET_TICKER:      str   = "SPY"

# Shared (per spec §2.3-§2.6, same as Path V post-Amendment 1)
TOP_PCTILE:         float = 0.90   # Path W: top decile, rank >= 0.90
BOTTOM_PCTILE:      float = 0.10   # Path X: bottom decile, rank <= 0.10
PER_NAME_CAP:       float = 0.035
TC_DECIMAL_PER_SIDE: float = 10.0 / 10_000.0
EXCLUDE_TICKERS:    frozenset[str] = frozenset({"SPY", "BRK"})


@dataclass(frozen=True)
class FABacktestResult:
    """Output of backtest_factor_anomaly()."""
    spec_name:            str
    weekly_returns_gross: pd.Series
    weekly_returns_net:   pd.Series
    weekly_tc_drag:       pd.Series
    rebalance_dates:      list[pd.Timestamp]
    n_weeks:              int
    n_rebalances:         int
    avg_n_names_per_rebal: float
    avg_turnover:         float
    notes:                list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_crsp_panel_daily(
    panel_path:   str = CRSP_PANEL_PATH,
    window_start: str = WINDOW_START,
    window_end:   str = WINDOW_END,
    keep_spy:     bool = True,   # IVOL needs SPY column
) -> pd.DataFrame:
    """Load CRSP DSF panel daily, filter window. Optionally keep SPY for IVOL."""
    df = pd.read_parquet(panel_path)
    df = df.astype("float64")
    df = df.loc[window_start:window_end]
    # Exclude non-stock proxies (but SPY is needed as market proxy for Path X)
    to_drop = [t for t in EXCLUDE_TICKERS if t in df.columns and not (keep_spy and t == "SPY")]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def resample_weekly(daily_panel: pd.DataFrame) -> pd.DataFrame:
    return daily_panel.resample("W-FRI").last()


def build_rebalance_dates(weekly_panel: pd.DataFrame, min_idx: int) -> list[pd.Timestamp]:
    """First weekly bar of each calendar month, respecting min_idx warmup."""
    dates: list[pd.Timestamp] = []
    last_month = None
    for i, d in enumerate(weekly_panel.index):
        ym = (d.year, d.month)
        if ym != last_month:
            if i >= min_idx:
                dates.append(d)
            last_month = ym
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# Path W — 52-Week High Momentum
# ─────────────────────────────────────────────────────────────────────────────
def signal_path_w(
    weekly_panel: pd.DataFrame,
    rebal_date:   pd.Timestamp,
) -> pd.Series:
    """price_i(t) / max(price_i over t-52w to t-1w).

    Excluding current week t for the max (avoid trivial 1.0 at index high).
    """
    if rebal_date not in weekly_panel.index:
        prior = weekly_panel.index[weekly_panel.index <= rebal_date]
        if len(prior) == 0:
            return pd.Series(dtype=float)
        rebal_date = prior[-1]
    idx_t = weekly_panel.index.get_loc(rebal_date)
    if idx_t < LOOKBACK_WEEKS_W:
        return pd.Series(dtype=float)

    # Window: indices [idx_t - 52, idx_t - 1] inclusive
    window = weekly_panel.iloc[idx_t - LOOKBACK_WEEKS_W : idx_t]
    high_52w = window.max(axis=0)
    price_now = weekly_panel.iloc[idx_t]
    sig = price_now / high_52w
    # Valid only where both anchors are present and stock is currently listed
    valid = price_now.notna() & high_52w.notna()
    return sig.where(valid)


# ─────────────────────────────────────────────────────────────────────────────
# Path X — Idiosyncratic Volatility (Ang-Hodrick-Xing-Zhang 2006)
# ─────────────────────────────────────────────────────────────────────────────
def signal_path_x(
    daily_panel:  pd.DataFrame,
    rebal_date:   pd.Timestamp,
    ivol_days:    int = IVOL_DAYS,
) -> pd.Series:
    """22-day rolling residual std from regression r_i = α + β r_SPY + ε.

    For each stock with ≥ ivol_days valid daily returns ending at rebal_date,
    compute IVOL = std(residuals) × sqrt(252).
    """
    if MARKET_TICKER not in daily_panel.columns:
        raise RuntimeError(f"Path X requires {MARKET_TICKER} in panel")
    # Get daily returns up to rebal_date (snap to nearest <= date)
    avail = daily_panel.index[daily_panel.index <= rebal_date]
    if len(avail) < ivol_days + 2:
        return pd.Series(dtype=float)
    end_d = avail[-1]
    # Trailing ivol_days+1 daily prices → ivol_days returns
    window_prices = daily_panel.loc[:end_d].tail(ivol_days + 1)
    rets = window_prices.pct_change().iloc[1:]   # shape (ivol_days, n_tickers)
    if MARKET_TICKER not in rets.columns:
        return pd.Series(dtype=float)
    mkt = rets[MARKET_TICKER].to_numpy()
    if np.isnan(mkt).any():
        # SPY missing — skip this rebal
        return pd.Series(dtype=float)
    # Center for regression
    mkt_c = mkt - mkt.mean()
    mkt_var = float((mkt_c * mkt_c).sum())
    if mkt_var <= 0:
        return pd.Series(dtype=float)

    out = {}
    for ticker in rets.columns:
        if ticker == MARKET_TICKER:
            continue
        y = rets[ticker].to_numpy()
        if np.isnan(y).any():
            continue
        if len(y) != len(mkt):
            continue
        # OLS: β = cov(x, y) / var(x); α = mean(y) - β × mean(x)
        y_c = y - y.mean()
        beta = float((mkt_c * y_c).sum() / mkt_var)
        alpha = float(y.mean() - beta * mkt.mean())
        resid = y - (alpha + beta * mkt)
        ivol = float(np.std(resid, ddof=1)) * math.sqrt(252.0)
        if math.isfinite(ivol) and ivol > 0:
            out[ticker] = ivol
    return pd.Series(out, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Position selection helpers
# ─────────────────────────────────────────────────────────────────────────────
def select_top_decile(
    signal:           pd.Series,
    top_pctile:       float = TOP_PCTILE,
    per_name_cap:     float = PER_NAME_CAP,
) -> pd.Series:
    """Top decile equal-weight (used by Path W: high ratio = momentum)."""
    valid = signal.dropna()
    if len(valid) < 30:
        return pd.Series(dtype=float)
    rank = valid.rank(pct=True, method="average")
    top = valid[rank >= top_pctile]
    if len(top) == 0:
        return pd.Series(dtype=float)
    n = len(top)
    w = pd.Series(min(1.0 / n, per_name_cap), index=top.index)
    return w / w.sum()


def select_bottom_decile(
    signal:           pd.Series,
    bottom_pctile:    float = BOTTOM_PCTILE,
    per_name_cap:     float = PER_NAME_CAP,
) -> pd.Series:
    """Bottom decile equal-weight (used by Path X: low IVOL = defensive)."""
    valid = signal.dropna()
    if len(valid) < 30:
        return pd.Series(dtype=float)
    rank = valid.rank(pct=True, method="average", ascending=True)
    bottom = valid[rank <= bottom_pctile]
    if len(bottom) == 0:
        return pd.Series(dtype=float)
    n = len(bottom)
    w = pd.Series(min(1.0 / n, per_name_cap), index=bottom.index)
    return w / w.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward backtest engine (parameterized for W or X)
# ─────────────────────────────────────────────────────────────────────────────
def backtest_factor_anomaly(
    spec_name:           str,
    panel_path:          str = CRSP_PANEL_PATH,
    window_start:        str = WINDOW_START,
    window_end:          str = WINDOW_END,
) -> FABacktestResult:
    """Run Path W or Path X depending on spec_name in {'W','X'}."""
    if spec_name not in ("W", "X"):
        raise ValueError("spec_name must be 'W' or 'X'")

    daily = load_crsp_panel_daily(panel_path, window_start, window_end,
                                    keep_spy=(spec_name == "X"))
    weekly = resample_weekly(daily)

    # For Path W we need the weekly panel only; for Path X we need daily for IVOL
    weekly_for_returns = weekly.copy()
    if spec_name == "X" and MARKET_TICKER in weekly_for_returns.columns:
        # SPY not part of trading universe (already in EXCLUDE_TICKERS list for return calc)
        pass  # leave SPY in weekly_for_returns — will be filtered by position generation

    weekly_returns_per_stock = weekly_for_returns.pct_change()

    # Lookback warmup
    if spec_name == "W":
        min_idx = LOOKBACK_WEEKS_W
    else:
        # IVOL needs ≥22 daily returns ≈ 5 weekly bars
        min_idx = 5

    rebal_dates = build_rebalance_dates(weekly, min_idx)
    if not rebal_dates:
        raise RuntimeError("No rebalance dates after warmup exclusion")

    positions_history: dict[pd.Timestamp, pd.Series] = {}
    current_weights = pd.Series(dtype=float)
    weekly_gross_ret: dict[pd.Timestamp, float] = {}
    weekly_tc:        dict[pd.Timestamp, float] = {}

    rebal_set = set(rebal_dates)
    weeks = list(weekly.index)
    n_skipped = 0

    for i, week in enumerate(weeks):
        # 1) Return earned this week by prior positions
        if i > 0 and not current_weights.empty:
            r_t = weekly_returns_per_stock.iloc[i].reindex(current_weights.index)
            r_t = r_t.fillna(0.0)
            port_ret = float((current_weights * r_t).sum())
            weekly_gross_ret[week] = port_ret
            weekly_tc[week] = 0.0
        else:
            weekly_gross_ret[week] = 0.0
            weekly_tc[week] = 0.0

        # 2) On rebal days, compute signal + new weights + TC
        if week in rebal_set:
            if spec_name == "W":
                signal = signal_path_w(weekly, week)
                new_weights = select_top_decile(signal)
            else:  # X
                signal = signal_path_x(daily, week)
                # Drop SPY explicitly in case kept in panel
                if MARKET_TICKER in signal.index:
                    signal = signal.drop(MARKET_TICKER)
                new_weights = select_bottom_decile(signal)

            if new_weights.empty:
                n_skipped += 1
                continue
            all_tickers = current_weights.index.union(new_weights.index)
            w_old = current_weights.reindex(all_tickers, fill_value=0.0)
            w_new = new_weights.reindex(all_tickers, fill_value=0.0)
            turnover = float((w_new - w_old).abs().sum())
            tc = turnover * TC_DECIMAL_PER_SIDE * 2.0
            weekly_tc[week] = tc
            positions_history[week] = new_weights.copy()
            current_weights = new_weights

    gross = pd.Series(weekly_gross_ret, name=f"path_{spec_name.lower()}_gross")
    tcs   = pd.Series(weekly_tc,        name=f"path_{spec_name.lower()}_tc")
    net   = (gross - tcs).rename(f"path_{spec_name.lower()}_net")

    avg_n = (np.mean([len(s.dropna()) for s in positions_history.values()])
             if positions_history else 0.0)
    rebal_list = sorted(positions_history.keys())
    turnovers = []
    for i, d in enumerate(rebal_list):
        if i == 0:
            continue
        prev = positions_history[rebal_list[i - 1]]
        curr = positions_history[d]
        tk = prev.index.union(curr.index)
        turnovers.append(
            float((curr.reindex(tk, fill_value=0) - prev.reindex(tk, fill_value=0)).abs().sum())
        )
    avg_to = float(np.mean(turnovers)) if turnovers else 0.0

    notes: list[str] = []
    if n_skipped:
        notes.append(f"{n_skipped} rebalance(s) skipped due to insufficient names")

    return FABacktestResult(
        spec_name             = spec_name,
        weekly_returns_gross  = gross,
        weekly_returns_net    = net,
        weekly_tc_drag        = tcs,
        rebalance_dates       = rebal_list,
        n_weeks               = len(weeks),
        n_rebalances          = len(rebal_list),
        avg_n_names_per_rebal = float(avg_n),
        avg_turnover          = avg_to,
        notes                 = notes,
    )


def save_anomaly_returns(
    result:    FABacktestResult,
    save_dir:  str = "data/portfolio_replay",
) -> Path:
    p_dir = Path(save_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    out_path = p_dir / f"v1_path_{result.spec_name.lower()}_weekly.parquet"
    df = pd.DataFrame({
        "gross": result.weekly_returns_gross,
        "tc":    result.weekly_tc_drag,
        "net":   result.weekly_returns_net,
    })
    df.index.name = "week_end"
    df.to_parquet(out_path)
    return out_path
