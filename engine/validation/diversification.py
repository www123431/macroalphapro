"""engine/validation/diversification.py — is the "5-strategy" book really diversified?

Two structural truths a real fund must know:

  1. Effective number of bets. "5 strategies" is a count, not a
     diversification measure. D_PEAD and PATH_N are BOTH single-stock
     US-equity microstructure (the ss_sp500 sleeve, 48.6% combined). If
     they co-move, the book has fewer INDEPENDENT bets than 5. We
     measure this with the participation ratio of the correlation
     matrix eigenvalues (Meucci-style effective number of bets).

  2. Insurance contribution. CTA + AC (TLT/GLD) have low STANDALONE
     Sharpe — but they are insurance, not alpha. The correct test is:
     does adding them REDUCE the book's drawdown (especially in crisis
     windows) at acceptable Sharpe cost? This is the G7 portfolio-DD
     lens from the project's own gate framework, applied honestly.

Deterministic, read-only. Operates on the weekly per-strategy returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Book weights (from operating_model_v1 §sleeve allocation):
#   ss_sp500 48.6% split D_PEAD/PATH_N 50/50 → 24.3% each
#   etf_l1 32.4% (K1_BAB) · cta_defensive 9% · rms_crisis_hedge 10%
DEFAULT_BOOK_WEIGHTS = {
    "K1_BAB":               0.324,
    "D_PEAD":               0.243,
    "PATH_N":               0.243,
    "CTA_PQTIX":            0.090,
    "AC_proxy_AB_2014_23":  0.100,
}

# Which strategies are insurance/overlay (judged by contribution, not
# standalone alpha).
INSURANCE_STRATS = {"CTA_PQTIX", "AC_proxy_AB_2014_23"}


# ──────────────────────────────────────────────────────────────────────────────
# Diversification / effective bets
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DiversificationResult:
    correlation:          pd.DataFrame
    effective_bets:       float       # participation ratio of eigenvalues
    n_strategies:         int
    max_pair:             tuple       # (s1, s2, corr) most-correlated pair
    pead_pathn_corr:      float       # the same-sleeve concern, called out
    verdict:              str


def effective_number_of_bets(corr: np.ndarray) -> float:
    """Participation ratio of the correlation-matrix eigenvalues:
       ENB = (Σλ)² / Σλ². 5 uncorrelated → 5; 5 identical → 1."""
    eig = np.linalg.eigvalsh(corr)
    eig = eig[eig > 1e-10]
    return float((eig.sum() ** 2) / (eig ** 2).sum())


def analyze_diversification(strat_returns: pd.DataFrame) -> DiversificationResult:
    df = strat_returns.dropna()
    corr = df.corr()
    n = corr.shape[0]
    enb = effective_number_of_bets(corr.values)

    # Most-correlated off-diagonal pair
    max_pair = ("", "", 0.0)
    cols = list(corr.columns)
    for i in range(n):
        for j in range(i + 1, n):
            c = float(corr.iloc[i, j])
            if abs(c) > abs(max_pair[2]):
                max_pair = (cols[i], cols[j], c)

    pead_pathn = float("nan")
    if "D_PEAD" in corr.columns and "PATH_N" in corr.columns:
        pead_pathn = float(corr.loc["D_PEAD", "PATH_N"])

    ratio = enb / n
    if ratio >= 0.8:
        verdict = f"WELL diversified — {enb:.2f} effective bets of {n}"
    elif ratio >= 0.6:
        verdict = f"MODERATELY diversified — {enb:.2f} of {n}"
    else:
        verdict = f"CONCENTRATED — only {enb:.2f} effective bets of {n}"

    return DiversificationResult(
        correlation=corr, effective_bets=enb, n_strategies=n,
        max_pair=max_pair, pead_pathn_corr=pead_pathn, verdict=verdict,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Book metrics + insurance contribution
# ──────────────────────────────────────────────────────────────────────────────
def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of a return series (most-negative peak-to-trough)."""
    curve = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = curve / running_max - 1.0
    return float(dd.min())


def _book_metrics(returns: np.ndarray, ppy: int = 52) -> dict:
    r = returns[~np.isnan(returns)]
    sd = r.std(ddof=1)
    return {
        "ann_return": float(r.mean() * ppy),
        "ann_vol":    float(sd * math.sqrt(ppy)),
        "sharpe":     float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else float("nan"),
        "max_dd":     _max_drawdown(r),
    }


def _weighted_book(df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Weighted sum of strategy returns, weights renormalized to the
    columns present."""
    cols = [c for c in weights if c in df.columns]
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    return (df[cols].values * w).sum(axis=1)


@dataclass(frozen=True)
class InsuranceContribution:
    strategy:        str
    full_sharpe:     float
    full_maxdd:      float
    without_sharpe:  float
    without_maxdd:   float
    dd_reduction:    float    # full_maxdd - without_maxdd (positive = insurance HELPS, less negative DD)
    sharpe_cost:     float    # without_sharpe - full_sharpe (positive = insurance drags Sharpe)
    crisis_dd_reduction: float
    verdict:         str


def insurance_contribution(
    strat_returns: pd.DataFrame,
    weights:       dict = None,
    crisis_windows: list[tuple] = None,
) -> dict[str, InsuranceContribution]:
    """For each insurance sleeve, compare the book WITH vs WITHOUT it.

    The right lens (G7): insurance earns its place if it reduces book
    drawdown — most importantly in crisis windows — at acceptable Sharpe
    cost. Standalone Sharpe is the WRONG test for these sleeves.

    crisis_windows: list of (start_iso, end_iso) to measure DD reduction
    in stress periods specifically. Defaults to 2018-Q4, 2020-Q1, 2022.
    """
    weights = weights or DEFAULT_BOOK_WEIGHTS
    df = strat_returns.dropna()
    if crisis_windows is None:
        crisis_windows = [
            ("2018-10-01", "2018-12-31"),   # Q4 2018 selloff
            ("2020-02-15", "2020-04-15"),   # COVID crash
            ("2022-01-01", "2022-10-31"),   # rate-shock bear
        ]

    full = _weighted_book(df, weights)
    full_m = _book_metrics(full)

    def _crisis_dd(book_series_index, book_returns):
        s = pd.Series(book_returns, index=book_series_index)
        worst = 0.0
        for (a, b) in crisis_windows:
            seg = s.loc[(s.index >= a) & (s.index <= b)]
            if len(seg) > 1:
                worst = min(worst, _max_drawdown(seg.values))
        return worst

    full_crisis_dd = _crisis_dd(df.index, full)

    out: dict[str, InsuranceContribution] = {}
    for ins in INSURANCE_STRATS:
        if ins not in df.columns:
            continue
        w_without = {k: v for k, v in weights.items() if k != ins}
        without = _weighted_book(df, w_without)
        without_m = _book_metrics(without)
        without_crisis_dd = _crisis_dd(df.index, without)

        # dd_reduction > 0 means full book has SHALLOWER drawdown than
        # the book without insurance (insurance helped).
        dd_red = without_m["max_dd"] - full_m["max_dd"]   # both negative; without more negative ⇒ positive
        crisis_red = without_crisis_dd - full_crisis_dd
        sharpe_cost = without_m["sharpe"] - full_m["sharpe"]  # >0 ⇒ insurance dragged Sharpe

        if dd_red > 0.005 or crisis_red > 0.005:
            verdict = (f"EARNS its place — cuts {'crisis ' if crisis_red>dd_red else ''}"
                       f"DD by {max(dd_red, crisis_red)*100:.2f}pp "
                       f"(Sharpe cost {sharpe_cost*100:.0f}bp)")
        elif sharpe_cost > 0.05:
            verdict = (f"QUESTIONABLE — drags Sharpe {sharpe_cost:.2f} "
                       f"without meaningful DD reduction")
        else:
            verdict = "NEUTRAL — little DD help, little Sharpe cost"

        out[ins] = InsuranceContribution(
            strategy=ins, full_sharpe=full_m["sharpe"], full_maxdd=full_m["max_dd"],
            without_sharpe=without_m["sharpe"], without_maxdd=without_m["max_dd"],
            dd_reduction=dd_red, sharpe_cost=sharpe_cost,
            crisis_dd_reduction=crisis_red, verdict=verdict,
        )
    return out
