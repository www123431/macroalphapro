"""engine/validation/risk_contribution.py — position-level risk decomposition (MCTR / CCTR).

Weight is NOT risk. A 13% position can be 5% or 40% of book volatility depending on its own vol and
its correlation with the rest of the book. This computes the institutional view a systematic risk
manager actually reads:

  portfolio vol   σ_p   = sqrt(wᵀ Σ w)
  marginal CTR    MCTRᵢ = (Σ w)ᵢ / σ_p              (∂σ_p/∂wᵢ)
  component CTR   CCTRᵢ = wᵢ · MCTRᵢ                 (Σᵢ CCTRᵢ = σ_p)
  % of book risk        = CCTRᵢ / σ_p² · σ_p = wᵢ(Σw)ᵢ / σ_p²   (Σ = 100%)

Signed weights are handled, so a short or negatively-correlated name can have a NEGATIVE % of risk
— it diversifies the book (the key insight weight-only views hide).

Σ is the sample daily covariance of the CURRENT holdings' returns, with light shrinkage toward its
diagonal for stability (a dependency-light Ledoit-Wolf-style regularizer). Returns are annualized
×√252. Coverage is reported honestly: names with no return history are dropped and flagged.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PANEL_PATH = Path("data/cache/_holdings_returns_panel.parquet")
# Always fetched alongside holdings so scenario_stress (equity-β shock) and factor_exposure
# (cross-asset 5-β: equity/rates/credit/commodity/dollar) can build their factors from the panel.
MARKET_PROXIES = ("SPY", "TLT", "HYG", "LQD", "DBC", "UUP")
ANN = 252
_MIN_OBS = 60          # need at least this many daily returns to trust a covariance
_SHRINK = 0.10         # δ: Σ_shrunk = (1-δ)Σ + δ·diag(Σ)


def _fetch_returns(tickers: list[str], start: str) -> pd.DataFrame:
    """Daily returns (date × ticker) from yfinance for `tickers`; drops names with no data."""
    import yfinance as yf
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.pct_change().dropna(how="all").dropna(axis=1, how="all")


def build_returns_panel(
    tickers: list[str], lookback_days: int = 420, *, max_age_hours: float = 24.0, force: bool = False,
) -> pd.DataFrame:
    """Daily-return panel (index=date, columns=ticker) for `tickers`, cached to parquet. Re-fetches
    via yfinance only when the cache is missing / older than max_age_hours / missing too many of the
    requested names. Network + slow on a cold build (~tens of seconds for ~180 names)."""
    want = sorted(set(t for t in tickers if t) | set(MARKET_PROXIES))
    if not force and PANEL_PATH.is_file():
        try:
            age_h = (datetime.datetime.now().timestamp() - PANEL_PATH.stat().st_mtime) / 3600.0
            cached = pd.read_parquet(PANEL_PATH)
            covered = sum(1 for t in want if t in cached.columns)
            if age_h <= max_age_hours and want and covered >= 0.9 * len(want):
                return cached
        except Exception as exc:
            logger.warning("risk_contribution: cache read failed: %s", exc)

    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).isoformat()
    rets = _fetch_returns(want, start)
    if rets.empty:
        raise RuntimeError("yfinance returned no data for the holdings universe")
    # yfinance drops random names on big batches. The factor proxies are critical (the factor /
    # scenario models depend on them) — retry any that the batch missed.
    miss = [p for p in MARKET_PROXIES if p not in rets.columns]
    if miss:
        extra = _fetch_returns(list(miss), start)
        if not extra.empty:
            rets = rets.join(extra[[c for c in extra.columns if c not in rets.columns]], how="outer")
    # Robustness: union with the prior cache so a transient fetch drop never SHRINKS coverage
    # (names that failed today keep their last-known history; the compute fills/ignores as needed).
    if PANEL_PATH.is_file():
        try:
            prev = pd.read_parquet(PANEL_PATH)
            keep = [c for c in prev.columns if c not in rets.columns]
            if keep:
                rets = rets.join(prev[keep], how="outer")
        except Exception:
            pass
    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(PANEL_PATH)
    logger.info("risk_contribution: built panel %d names × %d days (requested %d)",
                rets.shape[1], rets.shape[0], len(want))
    return rets


def _shrink_cov(cov: np.ndarray, delta: float = _SHRINK) -> np.ndarray:
    """Light shrinkage toward the diagonal (keeps Σ well-conditioned with ~180 names)."""
    d = np.diag(np.diag(cov))
    return (1.0 - delta) * cov + delta * d


def compute_risk_contributions(weights: dict, panel: pd.DataFrame, *, ann: int = ANN) -> dict:
    """Per-position risk decomposition for the combined book `weights` against the return `panel`."""
    if panel is None or panel.empty:
        return {"available": False, "reason": "no returns panel"}
    cols = [t for t, w in weights.items() if t in panel.columns and abs(w) > 1e-9]
    if len(cols) < 2:
        return {"available": False, "reason": "fewer than 2 holdings have return history"}
    R = panel[cols].dropna(how="all")
    R = R.tail(ann)                                   # most recent ~1y of daily returns
    if len(R) < _MIN_OBS:
        return {"available": False, "reason": f"only {len(R)} daily obs (<{_MIN_OBS})"}
    R = R.fillna(0.0)                                 # a name missing a day contributes 0 that day
    w = np.array([weights[t] for t in cols], dtype="float64")
    cov = _shrink_cov(np.cov(R.values, rowvar=False))
    sigma_w = cov @ w
    port_var = float(w @ sigma_w)
    if port_var <= 0:
        return {"available": False, "reason": "non-positive portfolio variance"}
    port_vol_d = port_var ** 0.5
    pct = (w * sigma_w) / port_var                    # component risk shares; sum to 1
    asset_vol_d = np.sqrt(np.clip(np.diag(cov), 0, None))
    sq = float(ann) ** 0.5

    contribs = [{
        "ticker": cols[i], "weight": round(float(w[i]), 6),
        "pct_risk": round(float(pct[i]), 6),
        "vol_annual": round(float(asset_vol_d[i] * sq), 6),
        "mctr_annual": round(float(sigma_w[i] / port_vol_d * sq), 6),
        "diversifying": bool(pct[i] < 0),
    } for i in range(len(cols))]
    contribs.sort(key=lambda c: -c["pct_risk"])

    all_w = sum(abs(v) for v in weights.values() if abs(v) > 1e-9) or 1.0
    cov_w = sum(abs(weights[t]) for t in cols)
    return {
        "available": True,
        "port_vol_annual": round(port_vol_d * sq, 6),
        "n_obs": int(len(R)),
        "lookback_start": str(R.index[0])[:10],
        "coverage": {"n_covered": len(cols), "n_total": sum(1 for v in weights.values() if abs(v) > 1e-9),
                     "weight_covered": round(cov_w / all_w, 4)},
        "contributions": contribs,
        "note": "Σ = shrunk sample daily covariance of current holdings; signed weights → a negative "
                "%-of-risk means the position diversifies (reduces) book volatility.",
    }
