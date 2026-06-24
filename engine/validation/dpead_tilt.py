"""engine/validation/dpead_tilt.py — D_PEAD long/short tilt conditioning.

Phase 1 found D_PEAD is the one marginal-positive alpha. The leg
decomposition then showed the SHORT leg is the weak link (standalone
Sharpe 0.34 vs the long leg's 0.93): the current dollar-neutral 100/100
construction over-weights a noisy, costly-to-borrow short leg.

This module tests dialing the short leg DOWN (combined = long − w·short)
and — crucially — guards against in-sample tilt-mining with a
split-sample test: choose the optimal tilt on the FIRST half, then
measure whether it still beats the 100/100 baseline on the held-out
SECOND half. If the train-optimal tilt also wins out-of-sample, the
"reduce the short leg" finding is robust, not overfit.

Reducing the short leg also REDUCES borrow cost + squeeze risk, so the
net-of-cost benefit is larger than the gross numbers — a tailwind, not
a hidden cost.

Inputs are the daily leg series (r_long, r_short) from the DHS PEAD
walk-forward. Market series (for alpha/beta separation) is passed in so
the core functions stay pure / network-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TiltMetric:
    short_weight:  float
    sharpe:        float
    ann_return:    float
    market_beta:   float
    alpha_annual:  float
    alpha_tstat:   float
    max_drawdown:  float


def _max_dd(r: np.ndarray) -> float:
    curve = np.cumprod(1.0 + r)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def combined(r_long: pd.Series, r_short: pd.Series, short_weight: float) -> pd.Series:
    """Tilted combination: long − w·short. w=1 is the current dollar-
    neutral book; w=0 is long-only."""
    df = pd.concat([r_long.rename("l"), r_short.rename("s")], axis=1).dropna()
    return (df["l"] - short_weight * df["s"]).rename(f"tilt_{short_weight:.2f}")


def tilt_metric(
    combo:   pd.Series,
    market:  pd.Series,
    rf:      pd.Series,
    ppy:     int = 252,
) -> TiltMetric:
    """Sharpe + market beta + market-model alpha (Newey-West t) + MaxDD
    for one tilt series. short_weight is filled by the caller."""
    import statsmodels.api as sm
    r = combo.dropna().astype(float)
    sr  = float(r.mean() / r.std(ddof=1) * math.sqrt(ppy)) if r.std(ddof=1) > 0 else float("nan")
    ann = float(r.mean() * ppy)
    df = pd.concat([r.rename("y"), market.rename("m"), rf.rename("rf")],
                   axis=1).dropna().astype(float)
    y = np.asarray(df["y"] - df["rf"], dtype=float)
    X = sm.add_constant(np.asarray(df["m"], dtype=float))
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
    return TiltMetric(
        short_weight=float("nan"),
        sharpe=sr, ann_return=ann,
        market_beta=float(m.params[1]),
        alpha_annual=float(m.params[0] * ppy),
        alpha_tstat=float(m.tvalues[0]),
        max_drawdown=_max_dd(r.values),
    )


def tilt_sweep(
    r_long:  pd.Series,
    r_short: pd.Series,
    market:  pd.Series,
    rf:      pd.Series,
    weights: tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
    ppy:     int = 252,
) -> list[TiltMetric]:
    out = []
    for w in weights:
        tm = tilt_metric(combined(r_long, r_short, w), market, rf, ppy)
        out.append(TiltMetric(
            short_weight=w, sharpe=tm.sharpe, ann_return=tm.ann_return,
            market_beta=tm.market_beta, alpha_annual=tm.alpha_annual,
            alpha_tstat=tm.alpha_tstat, max_drawdown=tm.max_drawdown,
        ))
    return out


@dataclass(frozen=True)
class SplitSampleResult:
    train_optimal_w:     float
    train_sharpe_at_opt: float
    baseline_w:          float
    # out-of-sample (test half) Sharpes
    test_sharpe_at_opt:  float
    test_sharpe_baseline: float
    test_improvement:    float    # test_opt − test_baseline (>0 ⇒ robust)
    verdict:             str


def split_sample_robustness(
    r_long:    pd.Series,
    r_short:   pd.Series,
    weights:   tuple = (0.0, 0.3, 0.5, 0.7, 1.0),
    baseline_w: float = 1.0,
    ppy:       int = 252,
) -> SplitSampleResult:
    """The anti-overfitting test. Pick the Sharpe-maximizing short_weight
    on the FIRST half, then check whether it beats the 100/100 baseline
    on the held-out SECOND half. Sharpe here is the simple in-leg Sharpe
    (no market model) — we only need a consistent ranking metric.

    test_improvement > 0 ⇒ the "reduce the short leg" finding generalizes
    out-of-sample (robust). <= 0 ⇒ it was in-sample tilt-mining.
    """
    df = pd.concat([r_long.rename("l"), r_short.rename("s")], axis=1).dropna()
    n = len(df)
    mid = n // 2
    train, test = df.iloc[:mid], df.iloc[mid:]

    def _sr(sub, w):
        c = (sub["l"] - w * sub["s"]).values
        sd = c.std(ddof=1)
        return (c.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan")

    # Optimal tilt on TRAIN
    train_srs = {w: _sr(train, w) for w in weights}
    opt_w = max(train_srs, key=lambda w: (train_srs[w]
                                          if not math.isnan(train_srs[w]) else -1e9))

    test_opt  = _sr(test, opt_w)
    test_base = _sr(test, baseline_w)
    improvement = test_opt - test_base

    if math.isnan(improvement):
        verdict = "UNDEFINED"
    elif improvement > 0.10:
        verdict = (f"ROBUST — train-optimal short_w={opt_w:.1f} beats "
                   f"baseline OOS by {improvement:.2f} Sharpe")
    elif improvement > 0:
        verdict = (f"MILD — train-optimal short_w={opt_w:.1f} beats "
                   f"baseline OOS by only {improvement:.2f}")
    else:
        verdict = (f"OVERFIT — train-optimal short_w={opt_w:.1f} does NOT "
                   f"beat baseline OOS ({improvement:.2f})")

    return SplitSampleResult(
        train_optimal_w=opt_w, train_sharpe_at_opt=train_srs[opt_w],
        baseline_w=baseline_w, test_sharpe_at_opt=test_opt,
        test_sharpe_baseline=test_base, test_improvement=improvement,
        verdict=verdict,
    )
