"""
engine/forensic/strategy_decay.py — Memmel Z forward decay test.

Memmel 2003 *Mgmt Sci* Sharpe-ratio comparison Z-statistic. Tests H0:
"spec-locked in-sample Sharpe = realized forward Sharpe" vs H1: "Sharpe
significantly decayed" → flags strategy decay candidates.

Auto-gate: requires ≥30 trading days of forward data per strategy.
Memmel formula: Z = (S1 - S2) / sqrt(theta), theta = (1 + 0.5(S1^2 + S2^2) - rho*S1*S2)/n

DOCTRINE: forensic layer; flags decay candidates for human review, does NOT
auto-retire strategies. Tier 3 governance retains kill decision.
"""
from __future__ import annotations

import datetime
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Spec-locked in-sample Sharpe per strategy (from each spec's verdict file)
SPEC_LOCKED_SHARPE: dict[str, float] = {
    "K1_BAB":     0.779,    # spec id=61 (Path K1 size-expanded ETF PASS)
    "D_PEAD":     0.922,    # spec id=62 (Path D DHS PEAD-TS leg PASS)
    "PATH_N":     0.743,    # spec id=70 amend 1 (Chen-Noronha-Singal PASS strict)
    "CTA_PQTIX":  0.38,     # spec id=73 (CTA Path O long-term observed)
}

MIN_FORWARD_DAYS_FOR_TEST: int = 30
MEMMEL_Z_DECAY_THRESHOLD: float = 1.5    # |Z| > 1.5 → flag for review


def _compute_realized_sharpe(
    daily_returns: pd.Series,
    annualization: int = 252,
) -> tuple[Optional[float], int]:
    """Compute realized annualized Sharpe from daily return series."""
    rets = daily_returns.dropna()
    if len(rets) < 5:
        return None, len(rets)
    mu  = float(rets.mean()) * annualization
    sig = float(rets.std()) * math.sqrt(annualization)
    if sig <= 0:
        return None, len(rets)
    return mu / sig, len(rets)


def _memmel_z(s1: float, s2: float, n1: int, n2: int) -> float:
    """Memmel Z statistic. Assumes independence (rho=0) — conservative.

    Z = (s1 - s2) / sqrt((2 - 2*rho + 0.5*(s1^2 + s2^2 - s1*s2 - rho*s1*s2)) / n)
    Simplified rho=0, common n: Z = (s1 - s2) / sqrt((2 + 0.5*(s1^2 + s2^2)) / n)
    """
    n = min(n1, n2)
    if n <= 1:
        return 0.0
    variance_term = 2.0 + 0.5 * (s1 ** 2 + s2 ** 2)
    se = math.sqrt(variance_term / n)
    return (s1 - s2) / se if se > 0 else 0.0


def compute_memmel_z_per_strategy(
    as_of:    datetime.date,
    lookback: int = MIN_FORWARD_DAYS_FOR_TEST,
) -> dict:
    """Compute Memmel Z forward decay test per strategy.

    Reads PaperTradeStrategyLog daily aggregate returns for last `lookback`
    days, computes realized Sharpe, compares to spec-locked in-sample Sharpe
    via Memmel Z test.
    """
    from engine.db_models import PaperTradeStrategyLog, SessionFactory

    start = as_of - datetime.timedelta(days=int(lookback * 1.5))  # buffer for non-trading days
    end   = as_of

    s = SessionFactory()
    try:
        rows = (s.query(PaperTradeStrategyLog)
                  .filter(PaperTradeStrategyLog.date >= start)
                  .filter(PaperTradeStrategyLog.date <= end)
                  .all())
        if not rows:
            return {
                "status":     "INSUFFICIENT_DATA",
                "reason":     "no PaperTradeStrategyLog rows in lookback",
                "have":       0,
                "need":       MIN_FORWARD_DAYS_FOR_TEST,
                "eta_unlock": (as_of + datetime.timedelta(days=MIN_FORWARD_DAYS_FOR_TEST)).isoformat(),
            }
        data = pd.DataFrame([
            {"date": r.date, "strategy_name": r.strategy_name, "daily_net_return": r.daily_net_return}
            for r in rows if r.daily_net_return is not None
        ])
    finally:
        s.close()

    if data.empty:
        return {
            "status":     "INSUFFICIENT_DATA",
            "reason":     "PaperTradeStrategyLog rows have no daily_net_return populated yet",
            "have":       0,
            "need":       MIN_FORWARD_DAYS_FOR_TEST,
            "eta_unlock": (as_of + datetime.timedelta(days=MIN_FORWARD_DAYS_FOR_TEST)).isoformat(),
        }

    per_strategy: dict[str, dict] = {}
    for strat, sub in data.groupby("strategy_name"):
        rets = sub.sort_values("date")["daily_net_return"]
        realized_sharpe, n_obs = _compute_realized_sharpe(rets)
        spec_sharpe = SPEC_LOCKED_SHARPE.get(strat)
        if spec_sharpe is None:
            per_strategy[strat] = {"status": "SKIPPED", "reason": "no spec_locked Sharpe registered"}
            continue
        if realized_sharpe is None or n_obs < MIN_FORWARD_DAYS_FOR_TEST:
            per_strategy[strat] = {
                "status":     "INSUFFICIENT_DATA",
                "have":       n_obs,
                "need":       MIN_FORWARD_DAYS_FOR_TEST,
                "eta_unlock": (as_of + datetime.timedelta(
                    days=MIN_FORWARD_DAYS_FOR_TEST - n_obs)).isoformat(),
            }
            continue
        z = _memmel_z(spec_sharpe, realized_sharpe, MIN_FORWARD_DAYS_FOR_TEST, n_obs)
        per_strategy[strat] = {
            "status":             "OK",
            "spec_locked_sharpe": spec_sharpe,
            "realized_sharpe":    round(realized_sharpe, 4),
            "n_obs":              n_obs,
            "memmel_z":           round(z, 4),
            "decay_flag":         abs(z) > MEMMEL_Z_DECAY_THRESHOLD,
            "interpretation":     ("DECAY FLAG: realized << spec" if z > MEMMEL_Z_DECAY_THRESHOLD
                                   else "OUTPERFORM: realized >> spec" if z < -MEMMEL_Z_DECAY_THRESHOLD
                                   else "consistent with spec (within |Z| < 1.5)"),
        }

    # Overall status: OK if any strategy is OK, else INSUFFICIENT_DATA
    any_ok = any(d.get("status") == "OK" for d in per_strategy.values())
    return {
        "status":          "OK" if any_ok else "INSUFFICIENT_DATA",
        "as_of":           as_of.isoformat(),
        "lookback_days":   lookback,
        "per_strategy":    per_strategy,
        "math_anchor":     "Memmel 2003 Mgmt Sci Sharpe-ratio test",
        "decay_threshold": MEMMEL_Z_DECAY_THRESHOLD,
    }
