"""
Gate evaluation for horse race candidates (P/Q/S/T/U) vs PQTIX baseline.

Per spec §2.6 locked gates (identical across all 5 active specs):
  G1 PRIMARY · Sharpe (net 10bp TC) ≥ PQTIX Sharpe over same window
  G2         · max DD ≤ PQTIX max DD × 1.1
  G3         · ρ vs combined (K1 + D_PEAD + PATH_N) weekly ≤ 0.15
  G4         · Crisis-positive ≥ 2 of 3 windows (2018-Q4 / 2020-COVID / 2022)

Decision rule:
  PASS     = 4/4 gates pass
  MARGINAL = 3/4 pass
  FAIL     = ≤ 2/4 pass

Statistical augmentation per spec §2.7:
  - Newey-West HAC t-statistic with 8-lag on excess return for Sharpe non-zero test
  - Bootstrap 95% CI on Sharpe difference via 1000-sample stationary block bootstrap
    (Politis-Romano 1994, block_length=12 weeks)
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd

from engine.portfolio.macro_cta_research.crisis_windows import (
    CRISIS_WINDOWS, crisis_positive_count,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gate result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GateResult:
    """Per-spec gate evaluation outcome."""
    spec_label:        str
    spec_sharpe:       float
    spec_max_dd:       float
    spec_crisis_pos:   int       # n positive of 3
    spec_corr_other:   float

    pqtix_sharpe:      float
    pqtix_max_dd:      float

    g1_pass:           bool      # Sharpe ≥ PQTIX
    g2_pass:           bool      # max DD ≤ PQTIX max DD × 1.1
    g3_pass:           bool      # ρ vs other sleeves ≤ 0.15
    g4_pass:           bool      # crisis-positive ≥ 2/3

    n_gates_passed:    int       # 0..4
    verdict:           str       # PASS / MARGINAL / FAIL

    sharpe_delta:      float     # spec − pqtix
    sharpe_nw_t:       float     # Newey-West t-stat on excess vs PQTIX
    sharpe_ci_lo:      float     # 95% bootstrap CI lower
    sharpe_ci_hi:      float     # 95% bootstrap CI upper

    crisis_returns:    dict      # {crisis_key: cum_return}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(weekly_returns: pd.Series, rfr_ann: float = 0.04) -> float:
    if len(weekly_returns) < 4:
        return float("nan")
    ann_ret = float(weekly_returns.mean() * 52)
    ann_vol = float(weekly_returns.std() * np.sqrt(52))
    if ann_vol <= 1e-9:
        return float("nan")
    return (ann_ret - rfr_ann) / ann_vol


def _max_dd(weekly_returns: pd.Series) -> float:
    if weekly_returns.empty:
        return float("nan")
    nav = (1.0 + weekly_returns).cumprod()
    return float((nav / nav.cummax() - 1).min())


def _newey_west_t(returns: pd.Series, lag: int = 8) -> float:
    """Newey-West HAC t-stat on H0: mean = 0."""
    r = returns.dropna().values
    n = len(r)
    if n < 4 * lag:
        return float("nan")
    mean = r.mean()
    if abs(mean) < 1e-15:
        return 0.0
    # Long-run variance via Bartlett kernel
    gamma_0 = np.var(r, ddof=0)
    s_lr = gamma_0
    for k in range(1, lag + 1):
        w_k = 1.0 - k / (lag + 1)
        gamma_k = np.mean((r[k:] - mean) * (r[:-k] - mean))
        s_lr += 2 * w_k * gamma_k
    if s_lr <= 0:
        return float("nan")
    se = np.sqrt(s_lr / n)
    return float(mean / se)


def _stationary_bootstrap_sharpe_ci(returns: pd.Series,
                                     n_boot: int = 1000,
                                     block_len: int = 12,
                                     conf: float = 0.95,
                                     seed: int = 42) -> tuple[float, float]:
    """Stationary bootstrap (Politis-Romano 1994) 95% CI on Sharpe."""
    r = returns.dropna().values
    n = len(r)
    if n < block_len * 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / block_len   # geometric block length expectation
    samples = []
    for _ in range(n_boot):
        out = []
        idx = rng.integers(0, n)
        while len(out) < n:
            out.append(r[idx])
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
        boot = np.array(out)
        mean_ann = boot.mean() * 52
        vol_ann  = boot.std() * np.sqrt(52)
        if vol_ann > 1e-9:
            samples.append((mean_ann - 0.04) / vol_ann)
    if not samples:
        return float("nan"), float("nan")
    alpha = (1.0 - conf) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _aligned_corr(s1: pd.Series, s2: pd.Series) -> float:
    df = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
    if len(df) < 8:
        return float("nan")
    return float(df["a"].corr(df["b"]))


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gates(
    spec_weekly_returns:   pd.Series,
    pqtix_weekly_returns:  pd.Series,
    other_sleeves_weekly:  pd.Series,
    *,
    spec_label:            str = "",
    corr_threshold:        float = 0.15,
    dd_buffer:             float = 1.10,
    crisis_min_positive:   int   = 2,
) -> GateResult:
    """Apply 4 gates per spec §2.6.

    Args:
      spec_weekly_returns:  candidate strategy weekly returns (post-TC)
      pqtix_weekly_returns: PQTIX baseline weekly returns
      other_sleeves_weekly: combined K1 + D_PEAD + PATH_N weekly returns
      corr_threshold:       G3 cap (default 0.15 per spec)
      dd_buffer:            G2 buffer multiplier (default 1.10 per spec)
      crisis_min_positive:  G4 min positive crisis windows (default 2 of 3 per spec)
    """
    # Align all 3 series to common date index
    df = pd.concat({
        "spec":   spec_weekly_returns,
        "pqtix":  pqtix_weekly_returns,
        "other":  other_sleeves_weekly,
    }, axis=1).dropna(how="all")

    spec_r  = df["spec"].dropna()
    pqtix_r = df["pqtix"].dropna()
    other_r = df["other"].dropna()

    # Core metrics
    spec_sharpe   = _sharpe(spec_r)
    spec_max_dd_v = _max_dd(spec_r)
    pqtix_sharpe  = _sharpe(pqtix_r)
    pqtix_max_dd  = _max_dd(pqtix_r)

    # Crisis-positive count
    crisis_d = crisis_positive_count(spec_r)
    n_crisis_pos = crisis_d["n_positive"]
    crisis_returns = {k: v for k, v in crisis_d.items()
                      if k in CRISIS_WINDOWS}

    # Correlation vs other sleeves
    spec_corr = _aligned_corr(spec_r, other_r)

    # Gates
    g1_pass = spec_sharpe >= pqtix_sharpe if not (pd.isna(spec_sharpe) or pd.isna(pqtix_sharpe)) else False
    g2_pass = spec_max_dd_v >= (pqtix_max_dd * dd_buffer) if not (pd.isna(spec_max_dd_v) or pd.isna(pqtix_max_dd)) else False
    # Note: max_dd is negative; "≤ PQTIX × 1.1" means abs(spec_dd) ≤ abs(pqtix_dd) × 1.1
    # Equivalently spec_dd ≥ pqtix_dd × 1.1 (less negative)
    g3_pass = (not pd.isna(spec_corr)) and abs(spec_corr) <= corr_threshold
    g4_pass = n_crisis_pos >= crisis_min_positive

    n_passed = sum([g1_pass, g2_pass, g3_pass, g4_pass])
    if n_passed == 4:
        verdict = "PASS"
    elif n_passed == 3:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    # Statistical augmentation
    excess = spec_r - pqtix_r.reindex(spec_r.index).fillna(0)
    nw_t = _newey_west_t(excess, lag=8)
    ci_lo, ci_hi = _stationary_bootstrap_sharpe_ci(excess, n_boot=1000, block_len=12)

    return GateResult(
        spec_label       = spec_label,
        spec_sharpe      = spec_sharpe,
        spec_max_dd      = spec_max_dd_v,
        spec_crisis_pos  = n_crisis_pos,
        spec_corr_other  = spec_corr,
        pqtix_sharpe     = pqtix_sharpe,
        pqtix_max_dd     = pqtix_max_dd,
        g1_pass          = bool(g1_pass),
        g2_pass          = bool(g2_pass),
        g3_pass          = bool(g3_pass),
        g4_pass          = bool(g4_pass),
        n_gates_passed   = int(n_passed),
        verdict          = verdict,
        sharpe_delta     = float(spec_sharpe - pqtix_sharpe) if not (pd.isna(spec_sharpe) or pd.isna(pqtix_sharpe)) else float("nan"),
        sharpe_nw_t      = nw_t,
        sharpe_ci_lo     = ci_lo,
        sharpe_ci_hi     = ci_hi,
        crisis_returns   = crisis_returns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience — combined "other sleeves" weekly returns for G3
# ─────────────────────────────────────────────────────────────────────────────

def load_other_sleeves_combined_weekly(weights: Optional[dict] = None) -> pd.Series:
    """Combined K1 + D_PEAD + PATH_N weekly returns from Sprint B replay parquet.

    Default sleeve weights (book-level non-CTA): K1 36% + D_PEAD 27% + PATH_N 27%
    = 90% of book; reweighted to sum to 1.0 for ρ computation.
    """
    if weights is None:
        weights = {"K1_BAB": 0.36, "D_PEAD": 0.27, "PATH_N": 0.27}
    # Renormalize to sum 1 (CTA share excluded — we're testing replacement for CTA)
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    try:
        from engine.portfolio.replay_combined import load_all_strategy_returns_weekly
        df = load_all_strategy_returns_weekly()
        present = [c for c in weights if c in df.columns]
        if not present:
            return pd.Series(dtype=float)
        # Weighted sum
        out = pd.Series(0.0, index=df.index)
        for c in present:
            out = out + df[c].fillna(0) * weights[c]
        out.name = "other_sleeves_combined"
        return out
    except Exception as exc:
        logger.warning("other-sleeves load failed: %s", exc)
        return pd.Series(dtype=float)
