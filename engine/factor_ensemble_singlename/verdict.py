"""
engine/factor_ensemble_singlename/verdict.py — Stage 2 Wave A/B verdict.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §3 + §九

Wave A labels: PRELIMINARY_PASS / PRELIMINARY_PARTIAL / PRELIMINARY_FAIL
              (cannot trigger production swap or forward-live)
Wave B labels: PASS / PARTIAL / FAIL  (publishable, can trigger PendingApproval)

Reuses:
  - engine.multivariate_msm_verdict (bootstrap CI + Memmel Z)
  - engine.factor_ensemble_v2.regime (4-regime classifier)
  - engine.factor_ensemble_v2.tc / multi_baseline framework patterns
  - engine.factor_ensemble_singlename.walk_forward (the run itself)

NO date parameters per v1 spec §4.5 amendment Nit #5 discipline.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DELTA_SHARPE_POSITIVE_THRESHOLD: float = 0.20
BOOTSTRAP_RESAMPLES:             int = 1000
BOOTSTRAP_ALPHA:                 float = 0.05

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "factor_ensemble_singlename"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclasses.dataclass
class SinglestockVerdictResult:
    """Stage 2 Wave A or B verdict snapshot."""
    wave:                       str   # "A" or "B"
    overall_decision:           str   # PRELIMINARY_PASS/PARTIAL/FAIL or PASS/PARTIAL/FAIL
    n_oos_months:               int
    ensemble_sharpe_net:        float
    per_baseline:               dict
    per_regime:                 dict
    n_baselines_positive:       int
    n_regimes_positive:         int
    abs_net_sharpe_above_zero:  bool
    spec_hash:                  str
    completed_at:               str
    metadata:                   dict = dataclasses.field(default_factory=dict)


def _decide_per_baseline(delta_sharpe: float, ci_lower: float, ci_upper: float) -> str:
    """Same logic as v2 (post-bug-fix): explicit positive / negative / insufficient."""
    if not np.isfinite(delta_sharpe) or not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return "WITHDRAW"
    if delta_sharpe >= DELTA_SHARPE_POSITIVE_THRESHOLD and ci_lower > 0.0:
        return "DESCRIPTIVE_POSITIVE"
    if delta_sharpe < 0 and ci_upper < 0.0:
        return "DESCRIPTIVE_NEGATIVE"
    if delta_sharpe > 0:
        if ci_lower > 0.0:
            return "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT"
        return "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION"
    if delta_sharpe < 0:
        return "DESCRIPTIVE_INSUFFICIENT_NEGATIVE_DIRECTION"
    return "DESCRIPTIVE_NEUTRAL"


def _annualized_sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(12))


def _run_baseline_singlestock(
    baseline_id:          str,
    rebalance_dates:      list[datetime.date],
    universe_at_date_fn:  Callable[[datetime.date], list[str]],
    panel:                pd.DataFrame,
) -> dict:
    """Compute one baseline's monthly returns (TC-net) for single-stock context.

    4 baselines:
      - bab_only:           single-stock BAB-only signal walk-forward
      - sixty_forty:        60% SPY + 40% AGG static rebalance
      - equal_weight_sp500: 1/N across vintage SP500 each month
      - spy_buyhold:        100% SPY buy-and-hold (1st period only, drift after)
    """
    from engine.factor_ensemble_singlename.walk_forward import (
        _construct_singlestock_weights, _compute_realized_return_panel,
        TC_BPS_LOCKED, MAX_NAME_WEIGHT_LOCKED,
    )
    from engine.factor_ensemble_v2.tc import compute_tc_drag

    if panel is None or panel.empty:
        return {
            "baseline_id": baseline_id,
            "monthly_returns_net": pd.Series(dtype=float),
            "n_periods": 0,
        }

    monthly_records: list[dict] = []
    prev_weights: Optional[pd.Series] = None
    establishment_done = False

    for i, rebal_date in enumerate(rebalance_dates):
        try:
            universe = universe_at_date_fn(rebal_date)
        except Exception:
            continue
        if not universe:
            continue

        # Compute baseline weights
        if baseline_id == "bab_only":
            from engine.factors_singlename import compute_bab_singlestock_signal
            from engine.factor_ensemble import _cross_section_z_score
            sig = compute_bab_singlestock_signal(rebal_date, universe, panel=panel)
            sig_z = _cross_section_z_score(sig)
            weights = _construct_singlestock_weights(sig_z, panel, rebal_date)
        elif baseline_id == "sixty_forty":
            # Bug fix 2026-05-09: SPY/AGG are benchmark ETFs, NOT SP500 constituents,
            # so don't require them in `universe` (vintage SP500 list). Only require
            # them in panel.columns (which they always are — we explicitly add them
            # to all_tickers in compute_singlestock_verdict).
            weights = pd.Series(dtype=float)
            spy_avail = "SPY" in panel.columns
            agg_avail = "AGG" in panel.columns
            if spy_avail and agg_avail:
                weights["SPY"] = 0.60
                weights["AGG"] = 0.40
            elif spy_avail:
                # AGG unavailable → fall back 100% SPY
                weights["SPY"] = 1.0
        elif baseline_id == "equal_weight_sp500":
            valid_tickers = [t for t in universe if t in panel.columns]
            if valid_tickers:
                w = 1.0 / len(valid_tickers)
                # Apply same 2% concentration cap as ensemble for fair comparison
                w_capped = min(w, MAX_NAME_WEIGHT_LOCKED)
                weights = pd.Series({t: w_capped for t in valid_tickers}, dtype=float)
            else:
                weights = pd.Series(dtype=float)
        elif baseline_id == "spy_buyhold":
            if not establishment_done and "SPY" in panel.columns:
                weights = pd.Series({"SPY": 1.0}, dtype=float)
                establishment_done = True
            else:
                # subsequent: drift, no rebalance
                weights = prev_weights if prev_weights is not None else pd.Series(dtype=float)
        else:
            raise ValueError(f"Unknown baseline_id: {baseline_id}")

        if weights is None or weights.empty:
            continue

        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            break

        try:
            gross_return = _compute_realized_return_panel(
                weights=weights, panel=panel,
                period_start=rebal_date, period_end=next_rebal,
            )
        except Exception:
            continue

        # TC: spy_buyhold subsequent periods have 0 turnover
        if baseline_id == "spy_buyhold" and i > 0:
            tc = 0.0
        else:
            tc = compute_tc_drag(weights_new=weights, weights_prev=prev_weights, bps_roundtrip=TC_BPS_LOCKED)

        monthly_records.append({
            "rebal_date": rebal_date,
            "monthly_return_net": gross_return - tc,
        })
        # Update prev_weights for spy_buyhold drift (don't overwrite first weights)
        if baseline_id != "spy_buyhold" or not establishment_done:
            pass
        if baseline_id == "spy_buyhold":
            # Keep prev_weights as the original SPY weights for drift
            if prev_weights is None:
                prev_weights = weights.copy()
        else:
            prev_weights = weights

    if not monthly_records:
        return {"baseline_id": baseline_id, "monthly_returns_net": pd.Series(dtype=float), "n_periods": 0}
    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    return {
        "baseline_id": baseline_id,
        "monthly_returns_net": df["monthly_return_net"],
        "n_periods": len(df),
    }


def compute_singlestock_verdict(
    universe_at_date_fn:  Callable[[datetime.date], list[str]],
    wave:                 str = "A",
    *,
    use_cache:            bool = True,
    persist:              bool = True,
) -> SinglestockVerdictResult:
    """Run Stage 2 verdict: ensemble + 4 baselines + 4 regimes aggregation.

    No date params (per v1 spec §4.5 Nit #5 discipline applied).
    """
    if wave not in ("A", "B"):
        raise ValueError(f"wave must be 'A' or 'B', got {wave!r}")

    from engine.factor_ensemble_singlename.walk_forward import (
        OOS_START_DATE_WAVE_A, OOS_END_DATE_WAVE_A,
        run_singlestock_walk_forward, _generate_monthend_dates,
    )
    from engine.factor_ensemble_singlename.panel_fetcher import bulk_fetch_singlestock_panel
    from engine.factor_ensemble_v2.regime import classify_regime_series, REGIMES_LOCKED
    from engine.multivariate_msm_verdict import (
        memmel_z_paired_sharpe_diff, bootstrap_sharpe_diff_ci,
    )

    if wave == "B":
        raise NotImplementedError("Wave B requires WRDS Compustat — use post-approval amend_spec path")

    start_date = OOS_START_DATE_WAVE_A
    end_date = OOS_END_DATE_WAVE_A
    rebalance_dates = _generate_monthend_dates(start_date, end_date)
    logger.info("singlestock verdict: wave=%s, %d rebalance dates", wave, len(rebalance_dates))

    # Pre-fetch panel once (reused across ensemble + 4 baselines)
    all_tickers: set[str] = {"SPY", "AGG"}
    for d in rebalance_dates:
        try:
            u = universe_at_date_fn(d)
            if u:
                all_tickers.update(u)
        except Exception:
            pass
    panel = bulk_fetch_singlestock_panel(
        tickers=sorted(all_tickers),
        start_date=rebalance_dates[0], end_date=rebalance_dates[-1],
        use_cache=use_cache,
    )

    # Run ensemble
    logger.info("running ensemble walk-forward (Wave %s)", wave)
    ensemble = run_singlestock_walk_forward(
        universe_at_date_fn=universe_at_date_fn,
        rebalance_dates=rebalance_dates,
        wave=wave, use_cache=use_cache,
    )
    ens_returns_net = ensemble.monthly_returns_net

    # Run 4 baselines
    baseline_ids = ("bab_only", "sixty_forty", "equal_weight_sp500", "spy_buyhold")
    baseline_results: dict[str, dict] = {}
    for bid in baseline_ids:
        logger.info("running baseline: %s", bid)
        baseline_results[bid] = _run_baseline_singlestock(
            baseline_id=bid, rebalance_dates=rebalance_dates,
            universe_at_date_fn=universe_at_date_fn, panel=panel,
        )

    # Per-baseline verdict
    per_baseline_out: dict[str, dict] = {}
    for bid, br in baseline_results.items():
        b_returns_net = br["monthly_returns_net"]
        if ens_returns_net.empty or b_returns_net.empty:
            per_baseline_out[bid] = {
                "ensemble_sharpe": float("nan"), "baseline_sharpe": float("nan"),
                "delta_sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "memmel_z": float("nan"), "decision": "WITHDRAW", "n_obs": 0,
            }
            continue
        common = ens_returns_net.index.intersection(b_returns_net.index)
        if len(common) < 12:
            per_baseline_out[bid] = {
                "ensemble_sharpe": float("nan"), "baseline_sharpe": float("nan"),
                "delta_sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "memmel_z": float("nan"), "decision": "WITHDRAW", "n_obs": int(len(common)),
            }
            continue
        ens_aligned = ens_returns_net.loc[common]
        bas_aligned = b_returns_net.loc[common]
        s_ens = _annualized_sharpe(ens_aligned)
        s_bas = _annualized_sharpe(bas_aligned)
        delta = s_ens - s_bas if (np.isfinite(s_ens) and np.isfinite(s_bas)) else float("nan")
        try:
            ci_lo, ci_hi, _ = bootstrap_sharpe_diff_ci(
                returns_a=ens_aligned, returns_b=bas_aligned,
                n_resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
            )
            z, rho, _ = memmel_z_paired_sharpe_diff(returns_a=ens_aligned, returns_b=bas_aligned)
        except Exception as exc:
            logger.warning("bootstrap/memmel failed for %s: %s", bid, exc)
            ci_lo = ci_hi = z = rho = float("nan")
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

    # Per-regime decomposition (using BAB single-stock as paired comparison)
    regimes_series = classify_regime_series(panel=panel, rebalance_dates=list(ens_returns_net.index))
    bab_returns_net = baseline_results["bab_only"]["monthly_returns_net"]
    per_regime_out: dict[str, dict] = {}
    for regime in REGIMES_LOCKED:
        in_regime = regimes_series[regimes_series == regime].index
        ens_in = ens_returns_net.reindex(in_regime).dropna()
        bab_in = bab_returns_net.reindex(in_regime).dropna()
        common = ens_in.index.intersection(bab_in.index)
        n_obs = int(len(common))
        if n_obs < 5:
            per_regime_out[regime] = {
                "n_obs": n_obs, "ensemble_sharpe": float("nan"),
                "bab_sharpe": float("nan"), "delta_sharpe": float("nan"),
            }
            continue
        s_ens = _annualized_sharpe(ens_in.loc[common])
        s_bab = _annualized_sharpe(bab_in.loc[common])
        delta = s_ens - s_bab if (np.isfinite(s_ens) and np.isfinite(s_bab)) else float("nan")
        per_regime_out[regime] = {
            "n_obs":           n_obs,
            "ensemble_sharpe": round(float(s_ens), 4) if np.isfinite(s_ens) else float("nan"),
            "bab_sharpe":      round(float(s_bab), 4) if np.isfinite(s_bab) else float("nan"),
            "delta_sharpe":    round(float(delta), 4) if np.isfinite(delta) else float("nan"),
        }

    # Aggregate
    POSITIVE_LABELS = (
        "DESCRIPTIVE_POSITIVE",
        "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT",
        "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION",
    )
    n_baselines_positive = sum(1 for b in per_baseline_out.values() if b["decision"] in POSITIVE_LABELS)
    n_regimes_positive = sum(
        1 for r in per_regime_out.values()
        if np.isfinite(r["ensemble_sharpe"]) and r["ensemble_sharpe"] > 0
    )
    s_ens_overall = _annualized_sharpe(ens_returns_net)
    abs_above_zero = bool(np.isfinite(s_ens_overall) and s_ens_overall > 0)

    # Decision label (Wave A: PRELIMINARY_ prefix)
    prefix = "PRELIMINARY_" if wave == "A" else ""
    if n_baselines_positive >= 3 and n_regimes_positive >= 2 and abs_above_zero:
        overall = f"{prefix}PASS"
    elif n_baselines_positive >= 2 or n_regimes_positive >= 2:
        overall = f"{prefix}PARTIAL (baselines={n_baselines_positive}/4, regimes_pos={n_regimes_positive}/4, abs_above_zero={abs_above_zero})"
    else:
        overall = f"{prefix}FAIL (baselines={n_baselines_positive}/4, regimes_pos={n_regimes_positive}/4, abs_above_zero={abs_above_zero})"

    spec_hash = _read_spec_hash()
    result = SinglestockVerdictResult(
        wave=wave,
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
        metadata={
            "ensemble_max_drawdown": ensemble.max_drawdown_net,
            "ensemble_ann_vol":      ensemble.annualized_vol_net,
            "n_total_tickers_seen":  ensemble.metadata.get("n_total_tickers_seen", 0),
        },
    )

    if persist:
        _persist_verdict(result, wave)
    return result


def _read_spec_hash() -> str:
    try:
        from engine.memory import SessionFactory, SpecRegistry
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter(
                SpecRegistry.spec_path == "docs/spec_factor_ensemble_singlename_v1.md"
            ).first()
            if row and row.current_hash:
                return str(row.current_hash)
    except Exception:
        pass
    return "UNKNOWN"


def _persist_verdict(result: SinglestockVerdictResult, wave: str) -> None:
    out_path = _DATA_DIR / f"v1_wave_{wave.lower()}_verdict.json"
    payload = {
        "verdict_layer": "walk_forward_signal_only",
        "production_forward_required_for_swap": (wave == "B"),  # Wave A cannot trigger
        "wave": wave,
        "wave_caveat": (
            "Wave A is PRELIMINARY case study using yfinance + best-effort historical SP500 "
            "constituents (mktcap-proxy primary, Wikipedia robustness check). NOT publishable; "
            "Wave B (post-WRDS Compustat+CRSP) is academic-grade replacement."
            if wave == "A" else
            "Wave B uses WRDS CRSP + Compustat vintage point-in-time data, publishable case study."
        ),
        "interpretation_caveat": (
            "Walk-forward signal-construction layer; production swap requires 24mo forward live "
            "counterfactual (Wave B PASS only) per spec §3.4."
        ),
        "overall_decision":             result.overall_decision,
        "n_oos_months":                 result.n_oos_months,
        "ensemble_sharpe_net":          result.ensemble_sharpe_net,
        "per_baseline":                 result.per_baseline,
        "per_regime":                   result.per_regime,
        "n_baselines_positive":         result.n_baselines_positive,
        "n_regimes_positive":           result.n_regimes_positive,
        "abs_net_sharpe_above_zero":    result.abs_net_sharpe_above_zero,
        "spec_hash":                    result.spec_hash,
        "completed_at":                 result.completed_at,
        "metadata":                     result.metadata,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("Wave %s verdict persisted to %s", wave, out_path)
