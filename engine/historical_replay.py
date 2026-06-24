"""
Historical conditional replay (P-AUDIT v1 amendment v2, 2026-05-04).

Computes hit rate / mean active return / percentile distribution for a
(ticker × direction × regime × horizon) tuple, replayed from yfinance
historical prices over `lookback_years`. Implements Asness-Moskowitz-
Pedersen 2013 / Moskowitz-Ooi-Pedersen 2012 conditional event-study
methodology.

**0 LLM** anywhere in this module. Pure deterministic numpy / yfinance /
SQL.

Public API:
    get_historical_conditional_hit_rate(
        ticker, direction, target_regime, horizon_days=21,
        lookback_years=15, regime_proxy="msm_walk_forward",
        benchmark_ticker="SPY",
    ) -> dict

Two regime proxies:
    - "vix_simple":    VIX > 30 risk-off / <15 risk-on / else neutral
                       (fast; ex-post smoothed; UI must caveat)
    - "msm_walk_forward":  reuses engine.regime.get_regime_series which calls
                       get_regime_on(as_of=t, train_end=t) per t — proper
                       walk-forward MSM filter, no look-ahead by construction
                       (Diebold-Lee-Weinbach 1994 standard).

Anti-anchoring guards (UI-side, enforced in pages/orchestrator.py):
    1. Never sort sectors by hit rate
    2. Project ex-ante data renders before historical replay
    3. Disagreement flag is prominent when hist contradicts current quant signal
    4. Every render includes the boilerplate caveat
       "for context only — not a signal; strategy params are pre-registered
        and cannot be retro-tuned from this view"

Hard invariant (must hold at backend layer):
    The output dict is read-only context. It does NOT participate in any
    automated trade-signal pathway. Any caller that uses it to drive trade
    decisions violates the spec.

Spec: docs/decisions/historical_conditional_replay_v2.md
"""
from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers (best-effort — failures degrade gracefully)
# ─────────────────────────────────────────────────────────────────────────────

_PRICE_CACHE: dict[tuple, Any] = {}


def _fetch_daily_closes(ticker: str, lookback_years: int) -> Any:
    """yfinance daily Close series. Returns pandas Series indexed by date."""
    cache_key = (ticker, lookback_years)
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]
    try:
        import yfinance as yf
        end = datetime.date.today()
        start = end - datetime.timedelta(days=int(lookback_years * 365.25 + 30))
        df = yf.download(
            ticker, start=start, end=end, progress=False,
            auto_adjust=True, group_by="column",
        )
        if df is None or df.empty:
            return None
        col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        s = col.dropna().squeeze()
        if hasattr(s, "tz_localize"):
            try:
                s.index = s.index.tz_localize(None)
            except Exception:
                pass
        _PRICE_CACHE[cache_key] = s
        return s
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return None


def _fetch_vix(lookback_years: int) -> Any:
    return _fetch_daily_closes("^VIX", lookback_years)


# ─────────────────────────────────────────────────────────────────────────────
# Regime proxies
# ─────────────────────────────────────────────────────────────────────────────

def _vix_simple_regime_series(vix_series: Any) -> Any:
    """VIX > 30 risk-off / <15 risk-on / else neutral. Fast / ex-post."""
    import pandas as pd
    if vix_series is None or len(vix_series) == 0:
        return None
    out = pd.Series("neutral", index=vix_series.index)
    out[vix_series > 30] = "risk-off"
    out[vix_series < 15] = "risk-on"
    return out


def _msm_walk_forward_regime_series(month_end_dates: list[datetime.date]) -> dict:
    """
    Walk-forward MSM via existing engine.regime.get_regime_series.
    Returns dict {date: regime_label}.
    Note: get_regime_series is O(n) MSM fits — for 15y monthly = 180 fits ≈ slow.
    Cache aggressively.
    """
    cache_key = ("msm_walk_forward", tuple(month_end_dates[:1]) + tuple(month_end_dates[-1:]),
                 len(month_end_dates))
    if cache_key in _PRICE_CACHE:
        return _PRICE_CACHE[cache_key]
    try:
        from engine.regime import get_regime_series
        df = get_regime_series(month_end_dates)
        out = {d: row["regime"] for d, row in df.iterrows()}
        _PRICE_CACHE[cache_key] = out
        return out
    except Exception as e:
        logger.warning("MSM walk-forward failed (%s); falling back to vix_simple", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# TSMOM signal (12m formation – 1m skip), monthly
# ─────────────────────────────────────────────────────────────────────────────

def _tsmom_signal_series(closes: Any) -> Any:
    """sign(12m return - 1m return) at each month-end."""
    import pandas as pd
    if closes is None or len(closes) < 252:
        return None
    monthly = closes.resample("ME").last()
    if len(monthly) < 14:
        return None
    twelve_m = monthly.pct_change(12)
    one_m = monthly.pct_change(1)
    raw = (twelve_m - one_m)
    sig = raw.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

REGIME_LABELS = ("risk-on", "neutral", "risk-off", "transition")


def get_historical_conditional_hit_rate(
    ticker:           str,
    direction:        str,                  # "long" | "short"
    target_regime:    str,                  # "risk-on" | "neutral" | "risk-off"
    horizon_days:     int = 21,
    lookback_years:   int = 15,
    regime_proxy:     str = "msm_walk_forward",
    benchmark_ticker: str = "SPY",
) -> dict:
    """
    Returns dict:
        ticker / direction / target_regime / horizon_days / lookback_years /
        regime_proxy / benchmark_ticker /
        n_obs / hit_rate / mean_active_return / median_active_return /
        std_active_return / pct_5 / pct_95 /
        first_match_date / last_match_date /
        caveats: list[str]
        available: bool
    """
    import pandas as pd
    import numpy as np

    out: dict = {
        "ticker":           ticker,
        "direction":        direction,
        "target_regime":    target_regime,
        "horizon_days":     horizon_days,
        "lookback_years":   lookback_years,
        "regime_proxy":     regime_proxy,
        "benchmark_ticker": benchmark_ticker,
        "n_obs":            0,
        "hit_rate":         None,
        "mean_active_return":   None,
        "median_active_return": None,
        "std_active_return":    None,
        "pct_5":            None,
        "pct_95":            None,
        "first_match_date": None,
        "last_match_date":  None,
        "caveats":          [],
        "available":        False,
    }

    # Validate
    if direction not in ("long", "short"):
        out["caveats"].append(f"invalid direction {direction!r}")
        return out
    if target_regime not in ("risk-on", "neutral", "risk-off"):
        out["caveats"].append(f"invalid target_regime {target_regime!r}")
        return out

    # Fetch data
    closes_t = _fetch_daily_closes(ticker, lookback_years)
    closes_b = _fetch_daily_closes(benchmark_ticker, lookback_years)
    if closes_t is None or closes_b is None:
        out["caveats"].append(
            f"yfinance unavailable for {ticker} or {benchmark_ticker}"
        )
        return out

    # Align
    common_idx = closes_t.index.intersection(closes_b.index)
    if len(common_idx) < 252:
        out["caveats"].append("insufficient common history (<252 days)")
        return out
    closes_t = closes_t.loc[common_idx]
    closes_b = closes_b.loc[common_idx]

    # TSMOM monthly signal
    sig_t = _tsmom_signal_series(closes_t)
    if sig_t is None or sig_t.empty:
        out["caveats"].append("TSMOM signal computation failed")
        return out

    # Regime series
    if regime_proxy == "msm_walk_forward":
        # Walk-forward MSM at each month-end of signal series
        month_ends = [d.date() if hasattr(d, "date") else d for d in sig_t.index]
        regime_dict = _msm_walk_forward_regime_series(month_ends)
        if regime_dict:
            out["caveats"].append(
                "regime: walk-forward MSM (Diebold-Lee-Weinbach 1994 ex-ante filtered)"
            )
            regime_at = lambda dt: regime_dict.get(
                dt.date() if hasattr(dt, "date") else dt, "neutral"
            )
        else:
            # Fallback to vix_simple
            regime_proxy = "vix_simple"
            out["regime_proxy"] = "vix_simple"
            out["caveats"].append(
                "MSM walk-forward unavailable; fell back to vix_simple (ex-post smoothed)"
            )
    if regime_proxy == "vix_simple":
        vix = _fetch_vix(lookback_years)
        if vix is None:
            out["caveats"].append("VIX data unavailable")
            return out
        vix_aligned = vix.reindex(sig_t.index, method="ffill")
        regime_simple = _vix_simple_regime_series(vix_aligned)
        if "ex-post" not in " ".join(out["caveats"]):
            out["caveats"].append(
                "regime: vix_simple proxy (VIX>30 risk-off / <15 risk-on); ex-post smoothed"
            )
        regime_at = lambda dt: regime_simple.get(dt) if regime_simple is not None else "neutral"

    # Find condition-match dates: signal direction == requested AND regime == target
    direction_sign = 1 if direction == "long" else -1
    matches: list[tuple] = []
    for dt, sig_val in sig_t.dropna().items():
        if int(sig_val) != direction_sign:
            continue
        reg = regime_at(dt)
        if reg != target_regime:
            continue
        matches.append((dt, sig_val, reg))

    if not matches:
        out["caveats"].append("no historical matches for this conditional combination")
        return out

    # For each match, compute forward active return at horizon_days
    out["caveats"].append(
        f"forward active return = ticker / SPY excess at t+{horizon_days} trading days"
    )
    rets: list[float] = []
    for dt, _, _ in matches:
        try:
            # Find the index position in closes
            try:
                pos = closes_t.index.get_loc(dt)
            except KeyError:
                # Sometimes month-end dates fall on weekend; find next trading day
                pos_arr = closes_t.index.searchsorted(dt)
                if pos_arr >= len(closes_t):
                    continue
                pos = int(pos_arr)
            target_pos = pos + horizon_days
            if target_pos >= len(closes_t):
                continue
            r_t = float(closes_t.iloc[target_pos]) / float(closes_t.iloc[pos]) - 1.0
            r_b = float(closes_b.iloc[target_pos]) / float(closes_b.iloc[pos]) - 1.0
            rets.append(r_t - r_b)
        except Exception:
            continue

    if not rets:
        out["caveats"].append("matches found but no forward returns computable")
        return out

    arr = np.asarray(rets, dtype=float)
    out["n_obs"]                = int(len(arr))
    out["hit_rate"]             = float((arr > 0).mean())
    out["mean_active_return"]   = float(arr.mean())
    out["median_active_return"] = float(np.median(arr))
    out["std_active_return"]    = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    out["pct_5"]                = float(np.percentile(arr, 5))
    out["pct_95"]               = float(np.percentile(arr, 95))
    out["first_match_date"]     = str(matches[0][0])[:10]
    out["last_match_date"]      = str(matches[-1][0])[:10]
    out["available"]            = True

    out["caveats"].append(
        "for context only — not a signal; strategy params are pre-registered "
        "and cannot be retro-tuned from this view"
    )

    return out


def query_signature(
    ticker: str, direction: str, target_regime: str,
    horizon_days: int, lookback_years: int, regime_proxy: str,
    benchmark_ticker: str,
) -> str:
    """Stable signature for audit (each unique query counts toward
    EFFECTIVE_N_TRIALS multiple-testing budget per spec invariant)."""
    raw = "|".join([
        ticker, direction, target_regime,
        str(horizon_days), str(lookback_years), regime_proxy, benchmark_ticker,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
