"""engine/validation/filter_counterfactual.py — Phase 5.5: A/B test
scaffold for any signal-to-trade filter (5.7 CA filter being the
first user, but the abstraction is general).

Senior design choice: separate EVALUATION from PRODUCTION. The
scaffold doesn't generate counterfactuals — caller (5.6 signal
taxonomy / 5.7 CA per-sleeve rules) supplies (baseline, filtered)
series. Scaffold ONLY runs the statistical test.

This separation means:
  - 5.5 scaffold can validate ANY execution-layer change (rebalance
    frequency, position sizing, vol-target k, signal smoothing, ...),
    not just CA
  - We commit to a single PBB-validated A/B harness; everything else
    is composable

Per [[project-paper-borrow-ml-btc-costs-2026-06-01]] item 5.5.

Public API:
  evaluate_filter_counterfactual(baseline, filtered, sleeve, descriptor)
      → FilterEvalResult { verdict, sharpe_before/after, PBB diff/CI/p, ... }

  evaluate_k_sweep(sleeve, counterfactual_factory, k_values)
      → list[FilterEvalResult] across k values; senior picks the
        DEPLOY-verdicts and chooses the cheapest k that ships
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterEvalResult:
    """One A/B comparison of a signal-to-trade filter."""
    sleeve_name:       str
    filter_descriptor: str
    n_obs_aligned:     int
    sharpe_before:     float
    sharpe_after:      float
    sharpe_diff:       float
    diff_ci_lo:        float
    diff_ci_hi:        float
    p_value:           float
    block_len:         float
    n_trades_before:   Optional[int]
    n_trades_after:    Optional[int]
    turnover_reduction_pct: Optional[float]
    verdict:           str            # DEPLOY / NO_EVIDENCE / WORSE
    reasons:           list[str]


def _count_trades(returns: pd.Series) -> Optional[int]:
    """Rough trade count = number of sign-change positions inferred
    from return series. None if not deducible.

    NOTE: this is best-effort heuristic; the proper count comes from
    upstream backtest logging the actual signal positions. Used here
    only as a Sleeve-level diagnostic for the scaffold."""
    r = returns.dropna()
    if len(r) < 3:
        return None
    # Heuristic: count rows where the return changed sign in a meaningful
    # way (>1bp absolute). Floor on the magnitude to filter numerical noise.
    sig = np.sign(r.values)
    sig[np.abs(r.values) < 1e-4] = 0
    transitions = int((np.diff(sig) != 0).sum())
    return transitions


def evaluate_filter_counterfactual(
    baseline_returns: pd.Series,
    filtered_returns: pd.Series,
    *,
    sleeve_name: str,
    filter_descriptor: str,
    n_iter: int = 5000,
    rng_seed: Optional[int] = None,
) -> FilterEvalResult:
    """Run a PBB-validated A/B test on (baseline, filtered) return
    series.

    Both series must be aligned (same index). Misalignment → drop
    non-overlapping observations.

    Verdict:
      - DEPLOY     : PBB p < 0.05 AND sharpe_diff > 0
      - NO_EVIDENCE: PBB p >= 0.05 (no statistically significant lift)
      - WORSE      : PBB p < 0.05 AND sharpe_diff <= 0
    """
    a = baseline_returns.copy()
    b = filtered_returns.copy()
    if not isinstance(a.index, pd.DatetimeIndex):
        a.index = pd.to_datetime(a.index)
    if not isinstance(b.index, pd.DatetimeIndex):
        b.index = pd.to_datetime(b.index)

    joined = pd.concat([a.rename("baseline"), b.rename("filtered")],
                          axis=1).dropna()
    n = len(joined)
    if n < 24:
        return FilterEvalResult(
            sleeve_name=sleeve_name,
            filter_descriptor=filter_descriptor,
            n_obs_aligned=n,
            sharpe_before=float("nan"),
            sharpe_after=float("nan"),
            sharpe_diff=float("nan"),
            diff_ci_lo=float("nan"),
            diff_ci_hi=float("nan"),
            p_value=float("nan"),
            block_len=float("nan"),
            n_trades_before=None,
            n_trades_after=None,
            turnover_reduction_pct=None,
            verdict="NO_EVIDENCE",
            reasons=[f"insufficient aligned observations ({n} < 24)"],
        )

    from engine.validation.block_bootstrap import pbb_sharpe_diff
    # NOTE: pbb_sharpe_diff order = (a, b) → diff = SR(a) - SR(b)
    # We want filtered minus baseline (positive = filter helps)
    pbb = pbb_sharpe_diff(
        joined["filtered"].values, joined["baseline"].values,
        n_iter=n_iter, rng_seed=rng_seed,
    )

    trades_before = _count_trades(joined["baseline"])
    trades_after = _count_trades(joined["filtered"])
    turnover_red_pct: Optional[float] = None
    if trades_before and trades_after is not None and trades_before > 0:
        turnover_red_pct = round(
            (1.0 - trades_after / trades_before) * 100.0, 1,
        )

    reasons: list[str] = []
    if pbb.p_value_two_sided >= 0.05:
        verdict = "NO_EVIDENCE"
        reasons.append(
            f"Sharpe diff {pbb.diff_point:+.3f} not statistically "
            f"significant (PBB p={pbb.p_value_two_sided:.3f}, "
            f"CI [{pbb.diff_ci_lo:.3f}, {pbb.diff_ci_hi:.3f}])"
        )
    elif pbb.diff_point > 0:
        verdict = "DEPLOY"
        reasons.append(
            f"Sharpe diff {pbb.diff_point:+.3f} significant at "
            f"alpha=0.05 (PBB p={pbb.p_value_two_sided:.3f}, "
            f"CI [{pbb.diff_ci_lo:.3f}, {pbb.diff_ci_hi:.3f}])"
        )
        if turnover_red_pct is not None and turnover_red_pct > 0:
            reasons.append(f"turnover -{turnover_red_pct:.0f}%")
    else:
        verdict = "WORSE"
        reasons.append(
            f"Filter SIGNIFICANTLY DEGRADES Sharpe: diff "
            f"{pbb.diff_point:+.3f} (PBB p={pbb.p_value_two_sided:.3f}). "
            "Do NOT deploy."
        )

    return FilterEvalResult(
        sleeve_name=sleeve_name,
        filter_descriptor=filter_descriptor,
        n_obs_aligned=n,
        sharpe_before=pbb.sharpe_b,
        sharpe_after=pbb.sharpe_a,
        sharpe_diff=pbb.diff_point,
        diff_ci_lo=pbb.diff_ci_lo,
        diff_ci_hi=pbb.diff_ci_hi,
        p_value=pbb.p_value_two_sided,
        block_len=pbb.block_len,
        n_trades_before=trades_before,
        n_trades_after=trades_after,
        turnover_reduction_pct=turnover_red_pct,
        verdict=verdict,
        reasons=reasons,
    )


def evaluate_k_sweep(
    *,
    sleeve_name:               str,
    counterfactual_factory:    Callable[[float], tuple[pd.Series, pd.Series]],
    k_values:                  Sequence[float] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    n_iter:                    int = 5000,
    rng_seed:                  Optional[int] = None,
) -> list[FilterEvalResult]:
    """Sweep filter strength k across multiple values; return A/B
    results for each. counterfactual_factory(k) MUST return (baseline,
    filtered) series.

    Senior chooses the SMALLEST k that ships DEPLOY — minimizing
    intervention while keeping statistical significance.
    Reports apply Hochberg correction across the swept family.
    """
    results: list[FilterEvalResult] = []
    for k in k_values:
        try:
            baseline, filtered = counterfactual_factory(k)
        except Exception as exc:
            logger.warning(
                "counterfactual_factory(k=%s) failed for sleeve %s: %s",
                k, sleeve_name, exc,
            )
            continue
        res = evaluate_filter_counterfactual(
            baseline_returns=baseline,
            filtered_returns=filtered,
            sleeve_name=sleeve_name,
            filter_descriptor=f"CA filter k={k}",
            n_iter=n_iter, rng_seed=rng_seed,
        )
        results.append(res)

    # Apply Hochberg correction across the k-sweep family. Senior may
    # have looked at the same sleeve under many k values — multiple-
    # comparison adjustment guards against k-fishing.
    from engine.validation.block_bootstrap import hochberg_adjust
    if results:
        raw_ps = [r.p_value for r in results]
        adj_ps = hochberg_adjust([
            p if (p == p) else 1.0  # NaN-safe
            for p in raw_ps
        ])
        adjusted: list[FilterEvalResult] = []
        for r, adj_p in zip(results, adj_ps):
            # Re-classify verdict using adjusted p
            if adj_p >= 0.05:
                new_verdict = "NO_EVIDENCE"
                new_reasons = list(r.reasons) + [
                    f"after Hochberg across {len(k_values)} k-values, "
                    f"adjusted p={adj_p:.3f} >= 0.05 (k-fishing guard)"
                ]
            elif r.sharpe_diff > 0:
                new_verdict = "DEPLOY"
                new_reasons = list(r.reasons) + [
                    f"survives Hochberg correction (adj p={adj_p:.3f})"
                ]
            else:
                new_verdict = "WORSE"
                new_reasons = list(r.reasons)
            adjusted.append(FilterEvalResult(
                sleeve_name=r.sleeve_name,
                filter_descriptor=r.filter_descriptor,
                n_obs_aligned=r.n_obs_aligned,
                sharpe_before=r.sharpe_before,
                sharpe_after=r.sharpe_after,
                sharpe_diff=r.sharpe_diff,
                diff_ci_lo=r.diff_ci_lo,
                diff_ci_hi=r.diff_ci_hi,
                p_value=adj_p,
                block_len=r.block_len,
                n_trades_before=r.n_trades_before,
                n_trades_after=r.n_trades_after,
                turnover_reduction_pct=r.turnover_reduction_pct,
                verdict=new_verdict,
                reasons=new_reasons,
            ))
        return adjusted

    return results
