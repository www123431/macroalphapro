"""engine.research.pnl_diagnostics — N: the 4 parked senior gates.

Deferred from the 2026-06-09 senior施工建议 ("4 more senior gates
parked"); shipped 2026-06-10. Four diagnostics every institutional
referee asks about that Tier C computed nowhere (or computed but
never showed the LLM):

  N1  DSR-in-prompt   Bailey-LdP deflated Sharpe given the family's
                      n_trials. The dispatcher gate COUNTS trials but
                      the self_doubt LLM never saw the deflated
                      probability — it judged a t-stat with no
                      multiple-testing context.
  N2  ρ₁ check        Lag-1 autocorrelation of monthly PnL. Lo 2002:
                      positive serial correlation inflates the naive
                      Sharpe SE by ≈ sqrt((1+ρ₁)/(1-ρ₁)) under AR(1).
                      NW HAC already corrects the t-stat, but ρ₁ > 0.2
                      ALSO signals smoothed/stale pricing (illiquidity,
                      marks) — a data-quality smell worth surfacing.
  N3  paper-OOS ratio Sharpe AFTER the paper window ÷ Sharpe INSIDE
                      it. McLean-Pontiff 2016: 32-58% decay is normal;
                      a ratio < 0.30 means the factor effectively died
                      post-publication — replication success becomes
                      a NEGATIVE signal for forward returns.
  N4  power analysis  P(detect | true SR, n months) at T_GREEN=1.96.
                      Guards the OTHER tail: on short samples a RED
                      verdict is weak evidence of absence. At n=120
                      months, power to detect a true SR=0.5 is only
                      ~40% — the LLM should never read RED as "proven
                      dead" without seeing this number.

All pure functions over a monthly PnL series — IO-free, no store
access. Wiring: factor_dispatcher computes once per dispatch (right
where pnl_series_df + family n_trials are both in scope) and passes
the block to self_doubt + factor_verdict_emit like the lens outputs.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener._safety_constants import T_GREEN

logger = logging.getLogger(__name__)


# ρ₁ above this = serial-correlation smell (Lo 2002 §IV empirics:
# hedge-fund style smoothed series run 0.2-0.5; liquid futures ~0).
RHO1_SMELL_BAR = 0.20

# Paper-OOS ratio below this = factor effectively dead post-pub.
# McLean-Pontiff mean decay is 32-58% (ratio 0.42-0.68); 0.30 is
# beyond the bad end of the published distribution.
PAPER_OOS_DEAD_BAR = 0.30

# True-SR scenarios for the power table. 0.3 = weak factor,
# 0.5 = respectable, 0.8 = strong.
POWER_SCENARIO_SRS = (0.3, 0.5, 0.8)


# ────────────────────────────────────────────────────────────────────
# N2 — lag-1 autocorrelation + Lo 2002 SE inflation
# ────────────────────────────────────────────────────────────────────
def compute_rho1(pnl: pd.Series) -> Optional[dict]:
    """Lag-1 autocorrelation with asymptotic SE ≈ 1/sqrt(n) and the
    Lo 2002 AR(1) Sharpe-SE inflation factor sqrt((1+ρ)/(1-ρ))."""
    x = pnl.dropna().values
    n = len(x)
    if n < 24:
        return None
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return None
    rho1 = float(np.dot(x[:-1], x[1:]) / denom)
    se = 1.0 / math.sqrt(n)
    inflation = (math.sqrt((1 + rho1) / (1 - rho1))
                   if abs(rho1) < 1 else float("inf"))
    return {
        "rho1":                  rho1,
        "rho1_se":               se,
        "rho1_t":                rho1 / se,
        "sharpe_se_inflation":   inflation,
        "smell":                 bool(rho1 > RHO1_SMELL_BAR),
        "smell_bar":             RHO1_SMELL_BAR,
    }


# ────────────────────────────────────────────────────────────────────
# N3 — paper-window vs post-window (OOS) Sharpe ratio
# ────────────────────────────────────────────────────────────────────
def compute_paper_oos_ratio(
    pnl: pd.Series,
    paper_window: str,
    *,
    min_months_each: int = 24,
) -> Optional[dict]:
    """Sharpe AFTER the paper window ÷ Sharpe INSIDE it.

    Returns None when either segment is shorter than min_months_each
    (a noisy ratio is worse than no ratio) or in-window Sharpe ≤ 0
    (ratio undefined / meaningless).
    """
    from engine.research.ablation.metrics import annualized_sharpe
    try:
        p_start_str, p_end_str = paper_window.split(":")
        p_start = pd.Timestamp(f"{p_start_str.strip()}-01")
        p_end = (pd.Timestamp(f"{p_end_str.strip()}-01")
                   + pd.offsets.MonthEnd(0))
    except Exception:
        return None
    s = pnl.dropna()
    in_w  = s[(s.index >= p_start) & (s.index <= p_end)]
    post  = s[s.index > p_end]
    if len(in_w) < min_months_each or len(post) < min_months_each:
        return None
    sr_in   = annualized_sharpe(in_w)
    sr_post = annualized_sharpe(post)
    if not (math.isfinite(sr_in) and math.isfinite(sr_post)):
        return None
    if sr_in <= 0:
        return None
    ratio = sr_post / sr_in
    return {
        "sharpe_in_window":    float(sr_in),
        "sharpe_post_window":  float(sr_post),
        "oos_ratio":           float(ratio),
        "n_months_in":         int(len(in_w)),
        "n_months_post":       int(len(post)),
        "dead_bar":            PAPER_OOS_DEAD_BAR,
        "effectively_dead":    bool(ratio < PAPER_OOS_DEAD_BAR),
    }


# ────────────────────────────────────────────────────────────────────
# N4 — power of the T_GREEN gate at this sample length
# ────────────────────────────────────────────────────────────────────
def power_of_t_green(n_months: int, true_sr_annual: float) -> float:
    """P(NW t-stat ≥ T_GREEN | the factor truly has this annual SR).

    Approximation: t_hat ≈ N(true_t, 1) where
      true_t = true_SR_ann / SE(SR_ann),
      SE(SR_ann) ≈ sqrt((1 + SR²/2) / n_years)   (Lo 2002).
    """
    from scipy.stats import norm
    if n_months < 12:
        return float("nan")
    n_years = n_months / 12.0
    se = math.sqrt((1 + 0.5 * true_sr_annual ** 2) / n_years)
    true_t = true_sr_annual / se
    return float(1.0 - norm.cdf(T_GREEN - true_t))


def compute_power_table(n_months: int) -> dict:
    """Power at this sample length for the standard SR scenarios.
    Keys like 'sr_0.5' → power in [0,1]."""
    return {
        f"sr_{sr}": round(power_of_t_green(n_months, sr), 3)
        for sr in POWER_SCENARIO_SRS
    }


# ────────────────────────────────────────────────────────────────────
# Aggregate entry point — called once per dispatch
# ────────────────────────────────────────────────────────────────────
def compute_pnl_diagnostics(
    pnl: pd.Series,
    *,
    n_trials_family: int = 0,
    paper_window:    Optional[str] = None,
) -> Optional[dict]:
    """All 4 senior gates over a monthly net-PnL series. JSON-safe.

    Returns None when the series is too short for ANY diagnostic
    (< 24 months)."""
    s = pnl.dropna()
    n = len(s)
    if n < 24:
        return None

    out: dict = {"n_months": int(n)}

    # N1 — DSR (probability the observed Sharpe beats the expected
    # max of n_trials null strategies)
    try:
        from engine.research.ablation.metrics import (
            deflated_sharpe_ratio, annualized_sharpe,
        )
        trials = max(1, int(n_trials_family))
        dsr = deflated_sharpe_ratio(s, n_trials=trials)
        out["dsr"] = {
            "deflated_sr_prob":  (float(dsr)
                                    if math.isfinite(dsr) else None),
            "n_trials_family":   trials,
            "sharpe_ann":        float(annualized_sharpe(s)),
        }
    except Exception:
        logger.exception("pnl_diagnostics: DSR failed")
        out["dsr"] = None

    # N2 — ρ₁
    try:
        out["rho1"] = compute_rho1(s)
    except Exception:
        logger.exception("pnl_diagnostics: rho1 failed")
        out["rho1"] = None

    # N3 — paper OOS (only when a paper window is on the spec)
    out["paper_oos"] = None
    if paper_window:
        try:
            out["paper_oos"] = compute_paper_oos_ratio(s, paper_window)
        except Exception:
            logger.exception("pnl_diagnostics: paper_oos failed")

    # N4 — power table
    try:
        out["power"] = {
            "t_green":   T_GREEN,
            "n_months":  int(n),
            "table":     compute_power_table(n),
        }
    except Exception:
        logger.exception("pnl_diagnostics: power failed")
        out["power"] = None

    return out
