"""
engine/factor_ensemble_v2/verdict.py — Extended v2 verdict module.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md §3 / §九

Per-baseline (4) + per-regime (4) verdict aggregation. Reuses v1's
bootstrap/Memmel infrastructure from engine.multivariate_msm_verdict.

NO date parameters (per v1 spec §4.5 amendment Nit #5 — apply same discipline).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.factor_ensemble_v2.tc import (
    TC_BPS_ROUNDTRIP_LOCKED,
    compute_tc_drag,
)
from engine.factor_ensemble_v2.beta_neutral import (
    BETA_NEUTRAL_FACTORS_LOCKED,
    compute_beta_panel,
    beta_neutralize_tsmom,
)
from engine.factor_ensemble_v2.regime import (
    REGIMES_LOCKED,
    classify_regime_series,
)
from engine.factor_ensemble_v2.multi_baseline import (
    BASELINE_DEFINITIONS_LOCKED,
    run_baseline,
)

logger = logging.getLogger(__name__)

DELTA_SHARPE_POSITIVE_THRESHOLD: float = 0.20
BOOTSTRAP_RESAMPLES:             int   = 1000
BOOTSTRAP_ALPHA:                 float = 0.05

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factor_ensemble_v2"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_VERDICT_JSON = _DATA_DIR / "v2_verdict.json"
_VERDICT_TXT  = _DATA_DIR / "v2_verdict.txt"
_PER_BASELINE_PARQUET = _DATA_DIR / "v2_per_baseline_diagnostics.parquet"
_PER_REGIME_PARQUET   = _DATA_DIR / "v2_per_regime_diagnostics.parquet"


@dataclasses.dataclass
class V2VerdictResult:
    """v2 verdict snapshot."""
    overall_decision:        str   # PASS / PARTIAL / FAIL with reason
    n_oos_months:            int
    ensemble_sharpe_net:     float
    per_baseline:            dict  # {baseline_id: {sharpe, delta, ci_lower, ci_upper, memmel_z, decision}}
    per_regime:              dict  # {regime: {n_obs, ensemble_sharpe, bab_sharpe, delta_sharpe}}
    n_baselines_positive:    int
    n_regimes_positive:      int
    abs_net_sharpe_above_zero: bool
    spec_hash:               str
    completed_at:            str


def _ensemble_with_v2_enhancements(
    rebalance_dates: list[datetime.date],
    universe_at_date_fn,
    asset_classes_fn,
    panel:           pd.DataFrame,
    bps_roundtrip:   float = TC_BPS_ROUNDTRIP_LOCKED,
) -> dict:
    """Run the v2 ensemble (TC + TSMOM β-neutral) walk-forward.

    Reuses v1's _compute_signal_at_date + _compute_weights_from_signal but
    with TSMOM signal substituted by β-neutralized version BEFORE the
    vol-parity ensemble combine.

    Returns dict identical schema to run_baseline (monthly_returns_net etc).
    """
    from engine.factor_ensemble_walk_forward import (
        _compute_weights_from_signal,
        _compute_realized_return,
    )
    from engine.factor_ensemble import compute_ensemble_signal, _compute_all_factor_signals
    from engine.factor_ensemble import _cross_section_z_score, _nan_aware_factor_average, ENSEMBLE_FACTORS

    monthly_records: list[dict] = []
    prev_weights: Optional[pd.Series] = None

    for i, rebal_date in enumerate(rebalance_dates):
        u_dict = universe_at_date_fn(rebal_date)
        if not u_dict:
            continue
        universe = list(u_dict.values())
        ac = asset_classes_fn(universe)

        # Step 1: compute all 4 raw factor signals
        try:
            raw_signals = _compute_all_factor_signals(
                as_of=rebal_date, universe=universe, asset_classes=ac, use_cache=True,
            )
        except Exception as exc:
            logger.warning("v2 ensemble: signal compute failed @ %s: %s", rebal_date, exc)
            continue

        # Step 2: β-neutralize TSMOM (per spec §2.3, ONLY TSMOM)
        if "tsmom" in raw_signals and not raw_signals["tsmom"].empty:
            beta_panel_at_date = compute_beta_panel(panel=panel, as_of=rebal_date, tickers=universe)
            raw_signals["tsmom"] = beta_neutralize_tsmom(
                tsmom_signal=raw_signals["tsmom"],
                beta_panel=beta_panel_at_date,
            )

        # Step 3: cross-section z-score per factor + NaN-aware average (same as v1)
        z_signals = {f: _cross_section_z_score(raw_signals.get(f)) for f in ENSEMBLE_FACTORS}
        ensemble_sig = _nan_aware_factor_average(z_signals, universe=universe)

        if ensemble_sig is None or ensemble_sig.empty:
            continue

        # Step 4: weights (same as v1)
        weights = _compute_weights_from_signal(ensemble_sig, rebal_date, panel=panel)
        if weights is None or weights.empty:
            continue

        # Step 5: realized return + TC
        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            break

        try:
            gross_return = _compute_realized_return(
                weights=weights, period_start=rebal_date,
                period_end=next_rebal, panel=panel,
            )
        except Exception as exc:
            logger.warning("v2 ensemble: realized return failed @ %s: %s", rebal_date, exc)
            continue

        tc = compute_tc_drag(weights_new=weights, weights_prev=prev_weights, bps_roundtrip=bps_roundtrip)
        monthly_records.append({
            "rebal_date":       rebal_date,
            "monthly_return_gross": gross_return,
            "tc_drag":          tc,
            "monthly_return_net":   gross_return - tc,
            "turnover":         tc / (bps_roundtrip / 10000.0) if bps_roundtrip > 0 else 0.0,
        })
        prev_weights = weights

    if not monthly_records:
        return {
            "monthly_returns_gross": pd.Series(dtype=float),
            "monthly_returns_net":   pd.Series(dtype=float),
            "turnover_per_period":   pd.Series(dtype=float),
            "n_successful_periods":  0,
        }
    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    return {
        "monthly_returns_gross": df["monthly_return_gross"],
        "monthly_returns_net":   df["monthly_return_net"],
        "turnover_per_period":   df["turnover"],
        "n_successful_periods":  len(df),
        "diagnostics_df":        df,
    }


def _annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(12))


def _decide_per_baseline(delta_sharpe: float, ci_lower: float, ci_upper: float) -> str:
    """Per-baseline decision label per spec §3.3.

    Bug fix 2026-05-09: previous default fallback returned INSUFFICIENT_POSITIVE_DIRECTION
    for ALL non-matching cases including delta < 0 with overlapping CI. This inflated
    n_baselines_positive count. Fixed: explicit branch for negative direction.
    """
    if not np.isfinite(delta_sharpe) or not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return "WITHDRAW"
    # Strong positive: magnitude + CI fully positive
    if delta_sharpe >= DELTA_SHARPE_POSITIVE_THRESHOLD and ci_lower > 0.0:
        return "DESCRIPTIVE_POSITIVE"
    # Strong negative: magnitude + CI fully negative
    if delta_sharpe < 0 and ci_upper < 0.0:
        return "DESCRIPTIVE_NEGATIVE"
    # Insufficient evidence cases
    if delta_sharpe > 0:
        if ci_lower > 0.0:
            return "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT"
        return "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION"
    if delta_sharpe < 0:
        if ci_upper < 0.0:
            return "DESCRIPTIVE_NEGATIVE"  # already caught above but defensive
        return "DESCRIPTIVE_INSUFFICIENT_NEGATIVE_DIRECTION"
    # delta_sharpe == 0
    return "DESCRIPTIVE_NEUTRAL"


def compute_v2_verdict(*, persist: bool = True) -> V2VerdictResult:
    """Run all 4 baselines + ensemble + 4-regime decomp; return aggregated verdict.

    No date parameters — reads OOS_START_DATE / DEFAULT_END_DATE from harness const
    (per v1 spec §4.5 Nit #5 discipline applied to v2 too).
    """
    from engine.factor_ensemble_walk_forward import (
        OOS_START_DATE, DEFAULT_END_DATE,
        _generate_monthend_dates, _get_universe_at_date,
        _build_asset_classes_lookup, _bulk_prefetch_panel,
    )
    from engine.multivariate_msm_verdict import (
        memmel_z_paired_sharpe_diff, bootstrap_sharpe_diff_ci,
    )

    rebalance_dates = _generate_monthend_dates(OOS_START_DATE, DEFAULT_END_DATE)

    # Bulk pre-fetch panel ONCE for all baselines + ensemble (massive speedup)
    all_tickers: set[str] = set()
    for d in rebalance_dates:
        u = _get_universe_at_date(d)
        if u:
            all_tickers.update(u.values())
    all_tickers.update(["SPY", "AGG"])  # for 60/40 + spy_buyhold
    panel = _bulk_prefetch_panel(
        tickers=sorted(all_tickers),
        start_date=rebalance_dates[0],
        end_date=rebalance_dates[-1],
    )

    # Run v2 ensemble
    logger.info("v2 ensemble: running walk-forward with TC + TSMOM β-neutral")
    ensemble = _ensemble_with_v2_enhancements(
        rebalance_dates=rebalance_dates,
        universe_at_date_fn=_get_universe_at_date,
        asset_classes_fn=_build_asset_classes_lookup,
        panel=panel,
    )
    ens_returns_net = ensemble["monthly_returns_net"]

    # Run 4 baselines
    baseline_results: dict[str, dict] = {}
    for bid in BASELINE_DEFINITIONS_LOCKED:
        logger.info("v2 baseline: %s", bid)
        baseline_results[bid] = run_baseline(
            baseline_id=bid,
            rebalance_dates=rebalance_dates,
            universe_at_date_fn=_get_universe_at_date,
            asset_classes_fn=_build_asset_classes_lookup,
            panel=panel,
        )

    # Per-baseline verdict
    per_baseline_out: dict[str, dict] = {}
    for bid, br in baseline_results.items():
        b_returns_net = br["monthly_returns_net"]
        common = ens_returns_net.index.intersection(b_returns_net.index)
        if len(common) < 12:
            per_baseline_out[bid] = {
                "ensemble_sharpe": float("nan"), "baseline_sharpe": float("nan"),
                "delta_sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "memmel_z": float("nan"), "decision": "WITHDRAW",
            }
            continue
        ens_aligned = ens_returns_net.loc[common]
        bas_aligned = b_returns_net.loc[common]
        s_ens = _annualized_sharpe(ens_aligned)
        s_bas = _annualized_sharpe(bas_aligned)
        delta = s_ens - s_bas if (np.isfinite(s_ens) and np.isfinite(s_bas)) else float("nan")
        ci_lo, ci_hi, _ = bootstrap_sharpe_diff_ci(
            returns_a=ens_aligned, returns_b=bas_aligned,
            n_resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
        )
        z, rho, _ = memmel_z_paired_sharpe_diff(returns_a=ens_aligned, returns_b=bas_aligned)
        per_baseline_out[bid] = {
            "ensemble_sharpe": round(float(s_ens), 4) if np.isfinite(s_ens) else float("nan"),
            "baseline_sharpe": round(float(s_bas), 4) if np.isfinite(s_bas) else float("nan"),
            "delta_sharpe":    round(float(delta), 4) if np.isfinite(delta) else float("nan"),
            "ci_lower":        round(float(ci_lo), 4) if np.isfinite(ci_lo) else float("nan"),
            "ci_upper":        round(float(ci_hi), 4) if np.isfinite(ci_hi) else float("nan"),
            "memmel_z":        round(float(z), 4) if np.isfinite(z) else float("nan"),
            "paired_corr":     round(float(rho), 4) if np.isfinite(rho) else float("nan"),
            "decision":        _decide_per_baseline(delta, ci_lo, ci_hi),
            "n_obs":           int(len(common)),
        }

    # Per-regime decomposition (using BAB as fixed comparison)
    regimes_series = classify_regime_series(panel=panel, rebalance_dates=list(ens_returns_net.index))
    bab_returns_net = baseline_results["bab_only"]["monthly_returns_net"]

    per_regime_out: dict[str, dict] = {}
    for regime in REGIMES_LOCKED:
        in_regime = regimes_series[regimes_series == regime].index
        ens_in = ens_returns_net.reindex(in_regime).dropna()
        bab_in = bab_returns_net.reindex(in_regime).dropna()
        n_obs = int(len(ens_in.index.intersection(bab_in.index)))
        if n_obs < 5:
            per_regime_out[regime] = {
                "n_obs": n_obs, "ensemble_sharpe": float("nan"),
                "bab_sharpe": float("nan"), "delta_sharpe": float("nan"),
            }
            continue
        common = ens_in.index.intersection(bab_in.index)
        s_ens = _annualized_sharpe(ens_in.loc[common])
        s_bab = _annualized_sharpe(bab_in.loc[common])
        delta = s_ens - s_bab if (np.isfinite(s_ens) and np.isfinite(s_bab)) else float("nan")
        per_regime_out[regime] = {
            "n_obs":           n_obs,
            "ensemble_sharpe": round(float(s_ens), 4) if np.isfinite(s_ens) else float("nan"),
            "bab_sharpe":      round(float(s_bab), 4) if np.isfinite(s_bab) else float("nan"),
            "delta_sharpe":    round(float(delta), 4) if np.isfinite(delta) else float("nan"),
        }

    # Aggregate verdict — count only truly positive direction labels
    # (POSITIVE / SMALL_EFFECT / POSITIVE_DIRECTION); NEGATIVE_DIRECTION + NEUTRAL excluded
    POSITIVE_LABELS = (
        "DESCRIPTIVE_POSITIVE",
        "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT",
        "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION",
    )
    n_baselines_positive = sum(
        1 for b in per_baseline_out.values() if b["decision"] in POSITIVE_LABELS
    )
    n_regimes_positive = sum(
        1 for r in per_regime_out.values()
        if np.isfinite(r["ensemble_sharpe"]) and r["ensemble_sharpe"] > 0
    )
    s_ens_overall = _annualized_sharpe(ens_returns_net)
    abs_above_zero = bool(np.isfinite(s_ens_overall) and s_ens_overall > 0)

    if n_baselines_positive >= 3 and n_regimes_positive >= 2 and abs_above_zero:
        overall = "PASS"
    elif n_baselines_positive >= 2 or n_regimes_positive >= 2:
        overall = f"PARTIAL (baselines={n_baselines_positive}/4, regimes_pos={n_regimes_positive}/4, abs_net_above_zero={abs_above_zero})"
    else:
        overall = f"FAIL (baselines={n_baselines_positive}/4, regimes_pos={n_regimes_positive}/4, abs_net_above_zero={abs_above_zero})"

    spec_hash = _read_spec_hash()
    result = V2VerdictResult(
        overall_decision=overall,
        n_oos_months=int(len(ens_returns_net)),
        ensemble_sharpe_net=round(float(s_ens_overall), 4) if np.isfinite(s_ens_overall) else float("nan"),
        per_baseline=per_baseline_out,
        per_regime=per_regime_out,
        n_baselines_positive=n_baselines_positive,
        n_regimes_positive=n_regimes_positive,
        abs_net_sharpe_above_zero=abs_above_zero,
        spec_hash=spec_hash,
        completed_at=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    if persist:
        _persist_v2_verdict(result, baseline_results, ensemble, regimes_series)

    return result


def _read_spec_hash() -> str:
    try:
        from engine.memory import SessionFactory, SpecRegistry
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter(
                SpecRegistry.spec_path == "docs/spec_factor_ensemble_v2_robust.md"
            ).first()
            if row and row.current_hash:
                return str(row.current_hash)
    except Exception:
        pass
    return "UNKNOWN"


def _persist_v2_verdict(result: V2VerdictResult, baseline_results, ensemble, regimes_series) -> None:
    payload = {
        "verdict_layer":                              "walk_forward_signal_only",
        "production_forward_required_for_swap":       True,
        "interpretation_caveat":                      (
            "v2 robust verdict aggregates 4 baselines + 4 regimes; descriptive case study, "
            "not statistical RCT (sample size 167 OOS months × 4 regimes too small). "
            "PASS authorizes supervisor PendingApproval workflow but does NOT itself flip "
            "PRODUCTION_SIGNAL — production swap requires 24mo forward live counterfactual."
        ),
        "overall_decision":                           result.overall_decision,
        "n_oos_months":                               result.n_oos_months,
        "ensemble_sharpe_net":                        result.ensemble_sharpe_net,
        "per_baseline":                               result.per_baseline,
        "per_regime":                                 result.per_regime,
        "n_baselines_positive_or_directional":        result.n_baselines_positive,
        "n_regimes_positive":                         result.n_regimes_positive,
        "abs_net_sharpe_above_zero":                  result.abs_net_sharpe_above_zero,
        "spec_hash":                                  result.spec_hash,
        "tc_bps_roundtrip_locked":                    TC_BPS_ROUNDTRIP_LOCKED,
        "beta_neutral_factors_locked":                list(BETA_NEUTRAL_FACTORS_LOCKED),
        "baselines_locked":                           list(BASELINE_DEFINITIONS_LOCKED),
        "regimes_locked":                             list(REGIMES_LOCKED),
        "delta_sharpe_threshold_locked":              DELTA_SHARPE_POSITIVE_THRESHOLD,
        "bootstrap_resamples_locked":                 BOOTSTRAP_RESAMPLES,
        "completed_at":                               result.completed_at,
    }
    _VERDICT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Persisted v2 verdict JSON to %s", _VERDICT_JSON)

    # Per-baseline diagnostics parquet
    rows = []
    for bid, br in baseline_results.items():
        if not br.get("monthly_returns_net", pd.Series()).empty:
            for d, ret in br["monthly_returns_net"].items():
                rows.append({"baseline_id": bid, "rebal_date": d, "monthly_return_net": ret})
    if rows:
        try:
            pd.DataFrame(rows).to_parquet(_PER_BASELINE_PARQUET)
        except Exception as exc:
            logger.warning("v2 per-baseline parquet persist failed: %s", exc)

    # Per-regime diagnostics parquet
    rows = []
    for d, regime in regimes_series.items():
        rows.append({"rebal_date": d, "regime": regime})
    if rows:
        try:
            pd.DataFrame(rows).to_parquet(_PER_REGIME_PARQUET)
        except Exception as exc:
            logger.warning("v2 per-regime parquet persist failed: %s", exc)

    _VERDICT_TXT.write_text(_render_human_summary(result), encoding="utf-8")


def _render_human_summary(r: V2VerdictResult) -> str:
    lines = [
        "Factor Ensemble v2 Robust — Walk-Forward Verdict",
        "=" * 78,
        f"Overall decision:                     {r.overall_decision}",
        f"OOS months:                           {r.n_oos_months}",
        f"Ensemble Sharpe (TC-net):             {r.ensemble_sharpe_net:+.4f}",
        f"# baselines positive/directional:     {r.n_baselines_positive} / 4",
        f"# regimes ensemble Sharpe > 0:        {r.n_regimes_positive} / 4",
        f"Absolute TC-net Sharpe ≥ 0:           {r.abs_net_sharpe_above_zero}",
        "",
        "Per-Baseline:",
    ]
    for bid, b in r.per_baseline.items():
        lines.append(
            f"  {bid:14s}: ens={b['ensemble_sharpe']:+.4f} bas={b['baseline_sharpe']:+.4f} "
            f"Δ={b['delta_sharpe']:+.4f} CI=[{b['ci_lower']:+.4f}, {b['ci_upper']:+.4f}] "
            f"Z={b['memmel_z']:+.4f} → {b['decision']}"
        )
    lines.append("")
    lines.append("Per-Regime:")
    for regime, rr in r.per_regime.items():
        lines.append(
            f"  {regime:15s}: n={rr['n_obs']:3d} ens={rr['ensemble_sharpe']:+.4f} "
            f"bab={rr['bab_sharpe']:+.4f} Δ={rr['delta_sharpe']:+.4f}"
        )
    lines.append("")
    lines.append(f"Spec hash: {r.spec_hash}")
    lines.append(f"Completed: {r.completed_at}")
    return "\n".join(lines) + "\n"
