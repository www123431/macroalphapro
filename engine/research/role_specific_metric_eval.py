"""engine/research/role_specific_metric_eval.py — SLM Phase 2: maps
SleeveRole → metric-specific evaluator function.

Per the multi-role doctrine + ROLE_SPECIFIC_GATES in strategy_lifecycle:
each role is JUDGED on a DIFFERENT metric:

  alpha_seeker            → Sharpe ratio t-stat (sharpe_t_stat)
  risk_premium_harvester  → Sharpe ratio t-stat (HLZ-floor variant)
  insurance               → hedge correlation (negative β with risk
                            source, NOT Sharpe — insurance has
                            negative expected return BY DESIGN)
  regime_overlay          → switching attribution (PnL difference
                            vs static-weights counterfactual)
  diversifier             → cosine similarity with book (NOT Sharpe —
                            diversifier value is negative cosine,
                            return is secondary)

The evaluator returns a RoleMetricResult containing both the human-
readable metric value AND an equivalent t-statistic that the sequential
boundary can act on. For non-Sharpe roles, the t-stat is constructed
from the role's natural metric (e.g. hedge correlation t-stat from
regression β / SE).

Caller pattern (used by paper_trade_monitor in Phase 2 integration):

    result = evaluate_role_specific_metric(
        role=SleeveRole.ALPHA_SEEKER,
        sleeve_returns=trailing_returns_3mo,
    )
    decision = obf_boundary.decide(observed_t=result.t_stat, m=3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from engine.research.sequential_testing import sharpe_t_stat
from engine.research.strategy_lifecycle import SleeveRole


@dataclass(frozen=True)
class RoleMetricResult:
    """Output of a role-specific metric evaluation.

    Fields:
      role:           the SleeveRole evaluated
      metric_name:    human-readable label
      metric_value:   the natural metric (annualized Sharpe / cosine /
                      hedge correlation / etc.)
      t_stat:         equivalent t-stat for boundary.decide()
      n_observations: number of monthly observations used
      diagnostic:     extra info (e.g. regression R² for hedge_corr)
      evidence_passed: simple gate against role-specific minimum
                      threshold (None if no minimum applies)
    """

    role: SleeveRole
    metric_name: str
    metric_value: float
    t_stat: float
    n_observations: int
    diagnostic: dict[str, float]
    evidence_passed: Optional[bool] = None
    rationale: str = ""


# ── Per-role evaluators ─────────────────────────────────────────────────


def _evaluate_alpha_seeker(returns: pd.Series) -> RoleMetricResult:
    r = returns.dropna()
    if len(r) < 2:
        return RoleMetricResult(
            role=SleeveRole.ALPHA_SEEKER,
            metric_name="annualized_sharpe",
            metric_value=0.0, t_stat=0.0, n_observations=len(r),
            diagnostic={},
            rationale="insufficient data (n<2)",
        )
    sharpe_ann = float(r.mean() * 12 / (r.std(ddof=1) * math.sqrt(12)))
    t = sharpe_t_stat(r.values)
    # Minimum threshold for alpha_seeker is Sharpe ≥ 0.5 (HLZ-conservative
    # floor); this is INFORMATIVE only — the sequential boundary makes
    # the actual ACCEPT/REJECT call.
    return RoleMetricResult(
        role=SleeveRole.ALPHA_SEEKER,
        metric_name="annualized_sharpe",
        metric_value=sharpe_ann,
        t_stat=t,
        n_observations=len(r),
        diagnostic={"ann_return": float(r.mean() * 12),
                    "ann_vol": float(r.std(ddof=1) * math.sqrt(12))},
        evidence_passed=(sharpe_ann >= 0.5),
        rationale=f"Sharpe={sharpe_ann:+.3f} (HLZ floor 0.5), t={t:+.3f}",
    )


def _evaluate_risk_premium_harvester(returns: pd.Series) -> RoleMetricResult:
    """risk_premium_harvester uses same Sharpe-based metric as alpha_seeker
    but with a more permissive HLZ-floor interpretation. The role is paid
    for harvesting a known premium, not generating novel alpha."""
    r = returns.dropna()
    if len(r) < 2:
        return RoleMetricResult(
            role=SleeveRole.RISK_PREMIUM_HARVESTER,
            metric_name="annualized_sharpe",
            metric_value=0.0, t_stat=0.0, n_observations=len(r),
            diagnostic={},
            rationale="insufficient data (n<2)",
        )
    sharpe_ann = float(r.mean() * 12 / (r.std(ddof=1) * math.sqrt(12)))
    t = sharpe_t_stat(r.values)
    return RoleMetricResult(
        role=SleeveRole.RISK_PREMIUM_HARVESTER,
        metric_name="annualized_sharpe",
        metric_value=sharpe_ann,
        t_stat=t,
        n_observations=len(r),
        diagnostic={"ann_return": float(r.mean() * 12),
                    "ann_vol": float(r.std(ddof=1) * math.sqrt(12))},
        evidence_passed=(sharpe_ann >= 0.40),     # permissive HLZ floor
        rationale=f"Sharpe={sharpe_ann:+.3f} (HLZ-permissive 0.40), t={t:+.3f}",
    )


def _evaluate_insurance(
    returns: pd.Series, risk_source_returns: Optional[pd.Series],
) -> RoleMetricResult:
    """Insurance is judged on its hedge correlation against the risk
    source (e.g. SPY for crisis_hedge_tlt_gld; MTUM for mom_hedge).

    Hedge effectiveness metric:
      β = Cov(sleeve, risk_source) / Var(risk_source)
      t = β / SE(β)
    Insurance is BUYING the right to negative β; we require β ≤ -0.30
    sustained to pass.

    Note: Sharpe is INTENTIONALLY NOT consulted — insurance has negative
    expected return BY DESIGN; weak Sharpe is not a failure.
    """
    if risk_source_returns is None:
        return RoleMetricResult(
            role=SleeveRole.INSURANCE,
            metric_name="hedge_beta",
            metric_value=0.0, t_stat=0.0, n_observations=0,
            diagnostic={},
            evidence_passed=False,
            rationale="risk_source_returns required for insurance role",
        )
    joined = pd.concat([returns.rename("s"), risk_source_returns.rename("r")],
                       axis=1).dropna()
    if len(joined) < 3:
        return RoleMetricResult(
            role=SleeveRole.INSURANCE,
            metric_name="hedge_beta",
            metric_value=0.0, t_stat=0.0, n_observations=len(joined),
            diagnostic={},
            evidence_passed=False,
            rationale="insufficient overlap (n<3)",
        )
    var_r = joined["r"].var(ddof=1)
    cov_sr = joined["s"].cov(joined["r"])
    beta = float(cov_sr / var_r) if var_r > 0 else 0.0
    # Standard error of β (OLS, assuming homoskedastic residuals).
    fitted = beta * joined["r"]
    residuals = joined["s"] - fitted
    se_resid = residuals.std(ddof=1)
    n = len(joined)
    se_beta = se_resid / (joined["r"].std(ddof=1) * math.sqrt(n - 1)) \
              if joined["r"].std(ddof=1) > 0 else float("inf")
    # For insurance the SIGNED t = -β / SE (negate so a more-negative β
    # produces a HIGHER positive t — fits the boundary's "accept if t
    # large" semantics).
    t = -beta / se_beta if se_beta > 0 else 0.0
    return RoleMetricResult(
        role=SleeveRole.INSURANCE,
        metric_name="hedge_beta",
        metric_value=beta,
        t_stat=t,
        n_observations=n,
        diagnostic={"se_beta": se_beta, "var_risk_source": var_r},
        evidence_passed=(beta <= -0.30),    # standing role threshold
        rationale=f"hedge_β={beta:+.3f} (req ≤ -0.30), t={t:+.3f} (signed)",
    )


def _evaluate_regime_overlay(
    returns: pd.Series, static_baseline_returns: Optional[pd.Series],
) -> RoleMetricResult:
    """regime_overlay value = PnL DIFFERENCE vs static-weights counter-
    factual. We evaluate switching_attribution = mean(sleeve - static).

    The role's value is the SWITCHING contribution, not the absolute
    return. A regime_overlay that earns +10% but a static baseline
    would have earned +10% adds zero value.
    """
    if static_baseline_returns is None:
        return RoleMetricResult(
            role=SleeveRole.REGIME_OVERLAY,
            metric_name="switching_attribution",
            metric_value=0.0, t_stat=0.0, n_observations=0,
            diagnostic={},
            evidence_passed=False,
            rationale="static_baseline_returns required for regime_overlay role",
        )
    diff = (returns - static_baseline_returns).dropna()
    if len(diff) < 2:
        return RoleMetricResult(
            role=SleeveRole.REGIME_OVERLAY,
            metric_name="switching_attribution",
            metric_value=0.0, t_stat=0.0, n_observations=len(diff),
            diagnostic={},
            evidence_passed=False,
            rationale="insufficient overlap (n<2)",
        )
    mean_diff = float(diff.mean())
    se_diff = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    t = mean_diff / se_diff if se_diff > 0 else 0.0
    return RoleMetricResult(
        role=SleeveRole.REGIME_OVERLAY,
        metric_name="switching_attribution",
        metric_value=mean_diff,
        t_stat=t,
        n_observations=len(diff),
        diagnostic={"se_diff": se_diff},
        evidence_passed=(mean_diff > 0),
        rationale=f"switching_attribution={mean_diff:+.4f}/mo, t={t:+.3f}",
    )


def _evaluate_diversifier(
    returns: pd.Series, book_returns: Optional[pd.Series],
) -> RoleMetricResult:
    """diversifier value = cosine similarity with book ≤ -0.10 sustained.
    Sharpe NOT consulted — diversifier can have Sharpe ~0 and still add
    value via correlation profile.
    """
    if book_returns is None:
        return RoleMetricResult(
            role=SleeveRole.DIVERSIFIER,
            metric_name="cosine_with_book",
            metric_value=0.0, t_stat=0.0, n_observations=0,
            diagnostic={},
            evidence_passed=False,
            rationale="book_returns required for diversifier role",
        )
    joined = pd.concat([returns.rename("s"), book_returns.rename("b")],
                       axis=1).dropna()
    if len(joined) < 3:
        return RoleMetricResult(
            role=SleeveRole.DIVERSIFIER,
            metric_name="cosine_with_book",
            metric_value=0.0, t_stat=0.0, n_observations=len(joined),
            diagnostic={},
            evidence_passed=False,
            rationale="insufficient overlap (n<3)",
        )
    s = joined["s"].values
    b = joined["b"].values
    s_norm = float(np.linalg.norm(s))
    b_norm = float(np.linalg.norm(b))
    cosine = float(s @ b / (s_norm * b_norm)) if s_norm * b_norm > 0 else 0.0
    # T-stat for correlation: Pearson r → t = r * sqrt((n-2) / (1-r^2))
    r = joined["s"].corr(joined["b"])
    n = len(joined)
    if abs(r) < 1.0 and n > 2:
        t_raw = r * math.sqrt((n - 2) / (1 - r ** 2))
    else:
        t_raw = 0.0
    # Negate so that more-negative cosine → higher positive t
    t = -t_raw
    return RoleMetricResult(
        role=SleeveRole.DIVERSIFIER,
        metric_name="cosine_with_book",
        metric_value=cosine,
        t_stat=t,
        n_observations=n,
        diagnostic={"pearson_r": float(r)},
        evidence_passed=(cosine <= -0.10),    # standing diversifier threshold
        rationale=f"cosine={cosine:+.3f} (req ≤ -0.10), t={t:+.3f} (signed)",
    )


# ── Dispatch ────────────────────────────────────────────────────────────


def evaluate_role_specific_metric(
    *,
    role: SleeveRole,
    sleeve_returns: pd.Series,
    book_returns: Optional[pd.Series] = None,
    risk_source_returns: Optional[pd.Series] = None,
    static_baseline_returns: Optional[pd.Series] = None,
) -> RoleMetricResult:
    """Top-level dispatch — picks the right evaluator for the role.

    Callers don't need to know which auxiliary inputs each role needs;
    they should pass all available context and the evaluator picks the
    relevant ones. Missing required context → evidence_passed=False
    with a clear rationale.
    """
    if role == SleeveRole.ALPHA_SEEKER:
        return _evaluate_alpha_seeker(sleeve_returns)
    if role == SleeveRole.RISK_PREMIUM_HARVESTER:
        return _evaluate_risk_premium_harvester(sleeve_returns)
    if role == SleeveRole.INSURANCE:
        return _evaluate_insurance(sleeve_returns, risk_source_returns)
    if role == SleeveRole.REGIME_OVERLAY:
        return _evaluate_regime_overlay(sleeve_returns, static_baseline_returns)
    if role == SleeveRole.DIVERSIFIER:
        return _evaluate_diversifier(sleeve_returns, book_returns)
    raise ValueError(f"unknown role {role!r}")
