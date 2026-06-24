"""engine/risk/factor_budget.py — book-level factor risk decomposition.

Senior-quant standard: every institutional risk system (Aladdin / Axioma
/ MSCI BPM) produces a factor risk budget showing "of the X% portfolio
risk, Y% is from MOM exposure, Z% is from sector tilt, ..." . Without
this, deploy decisions are made on sleeve weights (70/25/5) when the
risk reality is factor weights (40% MOM, 30% sector, 25% risk-on, ...).

This is L1 of the post-Phase-3 improvement layer per user 2026-05-30
("the loop should optimize the strategy based on findings").

MATH:
  Each sleeve i has factor exposure beta vector β_i and idiosyncratic
  variance σ²_idio,i. Book weight w_i. Book factor exposure:
      B[f] = Σ_i w_i × β_i[f]
  Sample factor covariance from factor return panel:
      Σ_F[f,g] = cov(F_f, F_g)
  Book factor variance:
      σ²_fac = Σ_f Σ_g B[f] × Σ_F[f,g] × B[g] = B' Σ_F B
  Book idiosyncratic variance (sleeve-level, since we don't have stock
  weights):
      σ²_idio = Σ_i w_i² × σ²_idio,i
  Total book variance:
      σ²_book = σ²_fac + σ²_idio
  Per-factor variance contribution (Euler decomposition):
      VarContrib[f] = B[f] × [Σ_F × B]_f
      pct[f] = VarContrib[f] / σ²_book
  Verified: Σ_f VarContrib[f] = σ²_fac.

The output is the same shape as the institutional standard: a list of
factors sorted by % of total risk contribution, with sleeves contribu-
tions to each.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from engine.risk.barra_lite import (
    build_factor_returns,
    regress_sleeve_on_factors,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FactorBudgetReport:
    """Book-level factor risk decomposition output."""
    book_vol_annualized:      float
    factor_vol_annualized:    float    # systematic component
    idio_vol_annualized:      float    # specific component
    pct_factor:               float    # factor_var / total_var
    pct_idio:                 float    # idio_var / total_var
    factor_exposures:         dict[str, float]      # B[f] book-level
    factor_var_contrib_pct:   dict[str, float]      # % of total risk
    sleeve_idio_contrib_pct:  dict[str, float]      # per-sleeve idio share
    top_5_factors_by_risk:    list[tuple[str, float]]  # [(name, pct), ...]
    n_months_used:            int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def compute_factor_budget(
    sleeve_returns: dict[str, pd.Series],
    sleeve_weights: dict[str, float],
    factors: pd.DataFrame | None = None,
    phase: int = 3,
) -> FactorBudgetReport:
    """Decompose book risk into factor contributions.

    Args:
      sleeve_returns: dict {sleeve_name: monthly return Series} for each
        deployed sleeve in the book.
      sleeve_weights: dict {sleeve_name: weight in book} (must sum to 1).
      factors: factor return panel; defaults to build_factor_returns(phase).
      phase: 1/2/3 BARRA phase, used only if factors is None.

    Returns: FactorBudgetReport with the per-factor breakdown.
    """
    if factors is None:
        factors = build_factor_returns(phase=phase)

    if abs(sum(sleeve_weights.values()) - 1.0) > 0.01:
        logger.warning("sleeve weights sum to %.3f, not 1.0",
                          sum(sleeve_weights.values()))

    # Step 1: regress each sleeve on factors -> sleeve betas + idio variance
    factor_cols = list(factors.columns)
    sleeve_betas: dict[str, dict[str, float]] = {}
    sleeve_idio_var: dict[str, float] = {}
    sleeve_n_months: dict[str, int] = {}

    for name, ret in sleeve_returns.items():
        try:
            rep = regress_sleeve_on_factors(ret, factors, sleeve_name=name)
        except ValueError as exc:
            logger.warning("regression failed for %s: %s", name, exc)
            continue
        sleeve_betas[name] = rep.betas
        sleeve_n_months[name] = rep.n_months
        # Idiosyncratic variance = residual variance from regression.
        # Reconstruct: var(y) = factor_variance + idio_variance.
        # var(y) is known from y; factor_variance = β' Σ_F β.
        s = ret.copy()
        s.index = pd.to_datetime(s.index)
        s = s.resample("ME").last() if not s.index.equals(
            s.index.to_period("M").to_timestamp("M")) else s
        J = pd.concat([s.rename("y"), factors], axis=1).dropna()
        total_var = J["y"].var()
        beta_vec = np.array([rep.betas.get(c, 0.0) for c in factor_cols])
        cov_F = J[factor_cols].cov().values
        factor_var = float(beta_vec @ cov_F @ beta_vec)
        sleeve_idio_var[name] = max(0.0, total_var - factor_var)

    if not sleeve_betas:
        raise ValueError("no sleeve regressions succeeded")

    # Step 2: book-level factor exposures B[f] = Σ_i w_i × β_i[f]
    B = np.zeros(len(factor_cols))
    for i, c in enumerate(factor_cols):
        for name, betas in sleeve_betas.items():
            B[i] += sleeve_weights.get(name, 0.0) * betas.get(c, 0.0)

    # Step 3: factor covariance + book factor variance
    cov_F = factors.cov().values
    book_factor_var_monthly = float(B @ cov_F @ B)

    # Step 4: book idiosyncratic variance (sleeve-level approximation —
    # assumes sleeve idios are uncorrelated, which is the BARRA convention
    # for cross-sleeve idio).
    book_idio_var_monthly = float(sum(
        sleeve_weights.get(name, 0.0) ** 2 * v
        for name, v in sleeve_idio_var.items()
    ))

    total_var_monthly = book_factor_var_monthly + book_idio_var_monthly
    if total_var_monthly <= 0:
        raise ValueError("computed book variance non-positive")

    # Step 5: per-factor variance contribution via Euler decomposition
    # VarContrib[f] = B[f] × (Σ_F B)[f]
    SB = cov_F @ B
    var_contrib = {c: float(B[i] * SB[i]) for i, c in enumerate(factor_cols)}
    pct_contrib = {c: var_contrib[c] / total_var_monthly for c in factor_cols}

    # Per-sleeve idio share of total book variance
    sleeve_idio_pct = {
        name: (sleeve_weights.get(name, 0.0) ** 2 * v) / total_var_monthly
        for name, v in sleeve_idio_var.items()
    }

    # Annualize (sqrt(12) for monthly -> annual vol)
    sqrt12 = float(np.sqrt(12.0))
    book_vol_ann = float(np.sqrt(total_var_monthly)) * sqrt12
    factor_vol_ann = float(np.sqrt(book_factor_var_monthly)) * sqrt12
    idio_vol_ann = float(np.sqrt(book_idio_var_monthly)) * sqrt12

    # Top 5 by absolute contribution
    sorted_factors = sorted(pct_contrib.items(),
                                key=lambda kv: abs(kv[1]), reverse=True)
    top_5 = [(name, pct) for name, pct in sorted_factors[:5]]

    # Use the median n_months across sleeves as the audit basis
    n_months = int(np.median(list(sleeve_n_months.values())))

    return FactorBudgetReport(
        book_vol_annualized=book_vol_ann,
        factor_vol_annualized=factor_vol_ann,
        idio_vol_annualized=idio_vol_ann,
        pct_factor=book_factor_var_monthly / total_var_monthly,
        pct_idio=book_idio_var_monthly / total_var_monthly,
        factor_exposures={c: float(B[i]) for i, c in enumerate(factor_cols)},
        factor_var_contrib_pct=pct_contrib,
        sleeve_idio_contrib_pct=sleeve_idio_pct,
        top_5_factors_by_risk=top_5,
        n_months_used=n_months,
    )


# -- Orthogonality scoring for candidate vs current book -----------------

def factor_orthogonality_score(
    candidate_betas: dict[str, float],
    book_report: FactorBudgetReport,
) -> dict:
    """Score how orthogonal a candidate's factor exposures are to the
    current book's risk concentration.

    Method: project candidate β onto the unit-vector of book factor
    contributions. cosine = ±1 means perfectly aligned/opposite,
    cosine ≈ 0 means orthogonal.

    Returns: {
        cosine_to_book_risk:  in [-1, 1]; positive = same direction as
                                 book risk concentration; negative = opposite
        risk_diversifying_score: -cosine (positive when candidate reduces
                                 concentration; ranges [-1, 1])
        candidate_top_3_overlaps: factors where candidate has high beta
                                 AND book has high contribution
        candidate_top_3_diversifiers: factors where candidate has opposite
                                 sign vs book exposure
    }
    """
    # Align factor universe
    common = sorted(set(candidate_betas.keys())
                       & set(book_report.factor_exposures.keys()))
    if not common:
        return {
            "cosine_to_book_risk": 0.0,
            "risk_diversifying_score": 0.0,
            "candidate_top_3_overlaps": [],
            "candidate_top_3_diversifiers": [],
            "error": "no common factor columns",
        }
    cand_vec = np.array([candidate_betas[c] for c in common])
    # Book risk concentration vector = signed contribution (B[f] × σ_f)
    # We use the book FACTOR EXPOSURE B[f] directly (gives sign info).
    book_vec = np.array([book_report.factor_exposures[c] for c in common])

    norm_c = np.linalg.norm(cand_vec)
    norm_b = np.linalg.norm(book_vec)
    if norm_c < 1e-9 or norm_b < 1e-9:
        cosine = 0.0
    else:
        cosine = float((cand_vec @ book_vec) / (norm_c * norm_b))

    # Overlap = same sign + both large
    overlap_scores = {
        c: cand_vec[i] * book_vec[i]
        for i, c in enumerate(common)
    }
    sorted_overlap = sorted(overlap_scores.items(),
                                key=lambda kv: abs(kv[1]), reverse=True)
    top_overlaps = [(c, s) for c, s in sorted_overlap[:3] if s > 0]
    top_diversifiers = [(c, s) for c, s in sorted_overlap[:3] if s < 0]
    # Pad lists from full sort
    if len(top_overlaps) < 3:
        for c, s in sorted_overlap:
            if s > 0 and (c, s) not in top_overlaps:
                top_overlaps.append((c, s))
                if len(top_overlaps) >= 3:
                    break
    if len(top_diversifiers) < 3:
        for c, s in sorted_overlap:
            if s < 0 and (c, s) not in top_diversifiers:
                top_diversifiers.append((c, s))
                if len(top_diversifiers) >= 3:
                    break

    return {
        "cosine_to_book_risk": cosine,
        "risk_diversifying_score": -cosine,    # +1 = max diversifying
        "candidate_top_3_overlaps": top_overlaps,
        "candidate_top_3_diversifiers": top_diversifiers,
    }
