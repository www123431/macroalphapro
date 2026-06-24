"""
engine/factor_ensemble_verdict.py — Factor Ensemble v1 verdict computation.

Pre-registration: docs/spec_factor_ensemble_v1.md (id=50) §4.5

Spec lock (2026-05-09 amendments — pre-Sprint-Week-4 audit Issue #4 + Nit #5):
  • compute_verdict() takes NO date parameters; reads OOS_START_DATE /
    DEFAULT_END_DATE from engine.factor_ensemble_walk_forward module constants.
    Prevents HARKing R3 surface (silent window-shifting until ΔSharpe flips).
  • Output JSON schema is locked with mandatory verdict_layer +
    production_forward_required_for_swap + interpretation_caveat fields.
  • Verdict is at signal-construction layer, NOT production-live P&L forecast.
  • DESCRIPTIVE_POSITIVE authorizes supervisor PendingApproval workflow but
    does NOT itself flip PRODUCTION_SIGNAL.

Reuses engine.multivariate_msm_verdict for bootstrap CI + Memmel Z (paired).
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

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (per spec §3 + amendments)
# ─────────────────────────────────────────────────────────────────────────────

# ΔSharpe magnitude threshold for DESCRIPTIVE_POSITIVE per spec §3.2
DELTA_SHARPE_POSITIVE_THRESHOLD: float = 0.20

# Bootstrap config — locked per spec §3.1
BOOTSTRAP_RESAMPLES: int = 1000
BOOTSTRAP_ALPHA: float = 0.05  # 95% CI

# Output paths
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "factor_ensemble_v1"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_VERDICT_TXT = _DATA_DIR / "v1_verdict.txt"
_VERDICT_JSON = _DATA_DIR / "v1_verdict.json"

# Mandatory JSON schema fields (spec §4.5 amendment 2026-05-09 Issue #4)
_REQUIRED_VERDICT_FIELDS = (
    "verdict_layer",
    "production_forward_required_for_swap",
    "interpretation_caveat",
    "decision_label",
    "delta_sharpe_walk_forward",
    "ci_lower_95",
    "ci_upper_95",
    "memmel_z",
    "n_oos_months",
    "spec_hash",
    "harness_ensemble_only_baseline_consistency",
)

# Decision labels (spec §3.2)
_DECISION_LABELS = (
    "DESCRIPTIVE_POSITIVE",
    "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION",
    "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT",
    "DESCRIPTIVE_NEGATIVE",
    "WITHDRAW",
)


@dataclasses.dataclass(frozen=True)
class VerdictResult:
    """Locked verdict snapshot. All fields populated by compute_verdict()."""
    decision_label:                              str
    delta_sharpe_walk_forward:                   float
    ci_lower_95:                                 float
    ci_upper_95:                                 float
    memmel_z:                                    float
    n_oos_months:                                int
    paired_corr:                                 float
    ensemble_sharpe:                             float
    baseline_sharpe:                             float
    spec_hash:                                   str
    harness_ensemble_only_baseline_consistency:  str
    completed_at:                                str  # ISO UTC


# ─────────────────────────────────────────────────────────────────────────────
# Decision rule (spec §3.2)
# ─────────────────────────────────────────────────────────────────────────────


def _decide(
    delta_sharpe:  float,
    ci_lower:      float,
    ci_upper:      float,
    n_oos_months:  int,
) -> str:
    """Apply spec §3.2 decision rule. Pure function, no LLM."""
    if not np.isfinite(delta_sharpe) or not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return "WITHDRAW"
    if n_oos_months < 12:
        return "WITHDRAW"

    # Primary: descriptive positive if magnitude AND CI lower clearly above 0
    if delta_sharpe >= DELTA_SHARPE_POSITIVE_THRESHOLD and ci_lower > 0.0:
        return "DESCRIPTIVE_POSITIVE"

    # Negative: clearly worse than baseline
    if delta_sharpe < 0 and ci_upper < 0.0:
        return "DESCRIPTIVE_NEGATIVE"

    # Insufficient evidence variants
    if delta_sharpe > 0:
        # Direction positive but CI crosses 0 OR magnitude below threshold
        if delta_sharpe < DELTA_SHARPE_POSITIVE_THRESHOLD and ci_lower > 0.0:
            return "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT"
        return "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION"

    # delta == 0 or just barely negative without CI fully below
    return "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — NO date parameters (spec §4.5 amendment Nit #5)
# ─────────────────────────────────────────────────────────────────────────────


def compute_verdict(
    *,
    use_cache: bool = True,
    persist:   bool = True,
) -> VerdictResult:
    """
    Compute Factor Ensemble v1 walk-forward verdict.

    NOTE: this function takes NO date parameters. start_date and end_date are
    sourced from engine.factor_ensemble_walk_forward.OOS_START_DATE /
    DEFAULT_END_DATE module constants per spec §4.5 amendment 2026-05-09
    (Nit #5). Prevents HARKing R3 silent window-shifting.

    Args:
        use_cache: pass-through to walk-forward harness factor signal computation
        persist:   if True, write v1_verdict.txt + v1_verdict.json to _DATA_DIR

    Returns:
        VerdictResult with decision label + ΔSharpe + bootstrap CI + Memmel Z

    Raises:
        TypeError: if any caller passes positional args, start_date, or end_date
        ValueError: if mandatory verdict JSON schema fields would be missing
    """
    from engine.factor_ensemble_walk_forward import (
        OOS_START_DATE,
        DEFAULT_END_DATE,
        run_walk_forward,
    )
    from engine.multivariate_msm_verdict import (
        memmel_z_paired_sharpe_diff,
        bootstrap_sharpe_diff_ci,
        annualized_sharpe,
    )

    logger.info(
        "compute_verdict: harness window=[%s, %s] (locked from harness module constants)",
        OOS_START_DATE, DEFAULT_END_DATE,
    )

    # Step 1 — run BOTH legs through SAME harness, SAME dates, SAME universe
    ensemble_run = run_walk_forward(
        start_date=OOS_START_DATE,
        end_date=DEFAULT_END_DATE,
        baseline_only=False,
        use_cache=use_cache,
        persist=False,
    )
    baseline_run = run_walk_forward(
        start_date=OOS_START_DATE,
        end_date=DEFAULT_END_DATE,
        baseline_only=True,
        use_cache=use_cache,
        persist=False,
    )

    ens_returns: pd.Series = ensemble_run.monthly_returns
    base_returns: pd.Series = baseline_run.monthly_returns

    # Align on common dates (defensive — should be identical if harness is well-behaved)
    common = ens_returns.index.intersection(base_returns.index)
    ens_aligned = ens_returns.loc[common]
    base_aligned = base_returns.loc[common]
    n_oos_months = int(len(common))

    if n_oos_months < 12:
        logger.warning(
            "compute_verdict: only %d common OOS months; verdict will be WITHDRAW",
            n_oos_months,
        )

    # Step 2 — paired ΔSharpe (annualized) + bootstrap CI + Memmel Z
    s_ens = float(annualized_sharpe(ens_aligned))
    s_base = float(annualized_sharpe(base_aligned))
    delta_sharpe = s_ens - s_base if (np.isfinite(s_ens) and np.isfinite(s_base)) else float("nan")

    if n_oos_months >= 12:
        ci_lower, ci_upper, _block = bootstrap_sharpe_diff_ci(
            returns_a=ens_aligned, returns_b=base_aligned,
            n_resamples=BOOTSTRAP_RESAMPLES, alpha=BOOTSTRAP_ALPHA,
        )
        z, rho, _v = memmel_z_paired_sharpe_diff(
            returns_a=ens_aligned, returns_b=base_aligned,
        )
    else:
        ci_lower = ci_upper = float("nan")
        z = rho = float("nan")

    # Step 3 — decision label
    decision_label = _decide(delta_sharpe, ci_lower, ci_upper, n_oos_months)

    # Step 4 — Gate 0 baseline consistency status (read from gate0_baseline_check.json if present)
    gate0_status = "UNKNOWN"
    gate0_path = _DATA_DIR / "gate0_baseline_check.json"
    if gate0_path.exists():
        try:
            gate0_status = str(json.loads(gate0_path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))
        except Exception as exc:
            logger.warning("Could not read gate0 status: %s", exc)

    # Step 5 — current spec hash
    spec_hash = _read_current_spec_hash()

    result = VerdictResult(
        decision_label=decision_label,
        delta_sharpe_walk_forward=round(float(delta_sharpe), 4) if np.isfinite(delta_sharpe) else float("nan"),
        ci_lower_95=round(float(ci_lower), 4) if np.isfinite(ci_lower) else float("nan"),
        ci_upper_95=round(float(ci_upper), 4) if np.isfinite(ci_upper) else float("nan"),
        memmel_z=round(float(z), 4) if np.isfinite(z) else float("nan"),
        n_oos_months=n_oos_months,
        paired_corr=round(float(rho), 4) if np.isfinite(rho) else float("nan"),
        ensemble_sharpe=round(float(s_ens), 4) if np.isfinite(s_ens) else float("nan"),
        baseline_sharpe=round(float(s_base), 4) if np.isfinite(s_base) else float("nan"),
        spec_hash=spec_hash,
        harness_ensemble_only_baseline_consistency=gate0_status,
        completed_at=datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    if persist:
        _write_outputs(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# JSON schema enforcement (spec §4.5 amendment Issue #4)
# ─────────────────────────────────────────────────────────────────────────────


def build_verdict_json_payload(result: VerdictResult) -> dict:
    """Build the locked JSON payload from a VerdictResult.

    Raises:
        ValueError: if any required field is missing or null after assembly
                    (per spec §4.5 amendment 2026-05-09 Issue #4).
    """
    payload = {
        "verdict_layer": "walk_forward_signal_only",
        "production_forward_required_for_swap": True,
        "interpretation_caveat": (
            "Walk-forward harness ΔSharpe measures signal-construction layer "
            "alpha; production-forward 24mo+ live counterfactual (per §3.6) "
            "is the swap-decision criterion. DESCRIPTIVE_POSITIVE here "
            "authorizes the supervisor PendingApproval workflow but does NOT "
            "itself flip PRODUCTION_SIGNAL."
        ),
        "decision_label": result.decision_label,
        "delta_sharpe_walk_forward": result.delta_sharpe_walk_forward,
        "ci_lower_95": result.ci_lower_95,
        "ci_upper_95": result.ci_upper_95,
        "memmel_z": result.memmel_z,
        "n_oos_months": result.n_oos_months,
        "paired_corr": result.paired_corr,
        "ensemble_sharpe": result.ensemble_sharpe,
        "baseline_sharpe": result.baseline_sharpe,
        "spec_hash": result.spec_hash,
        "harness_ensemble_only_baseline_consistency": result.harness_ensemble_only_baseline_consistency,
        "completed_at": result.completed_at,
        "delta_sharpe_threshold_locked": DELTA_SHARPE_POSITIVE_THRESHOLD,
        "bootstrap_resamples_locked": BOOTSTRAP_RESAMPLES,
    }
    # Schema validation — mandatory fields present + non-null
    for fld in _REQUIRED_VERDICT_FIELDS:
        if fld not in payload:
            raise ValueError(f"verdict JSON schema violation: missing field '{fld}'")
        if payload[fld] is None:
            raise ValueError(f"verdict JSON schema violation: null value for required field '{fld}'")
    if payload["decision_label"] not in _DECISION_LABELS:
        raise ValueError(
            f"verdict JSON schema violation: decision_label "
            f"{payload['decision_label']!r} not in {_DECISION_LABELS}"
        )
    return payload


def _write_outputs(result: VerdictResult) -> None:
    payload = build_verdict_json_payload(result)
    _VERDICT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Persisted verdict JSON to %s", _VERDICT_JSON)
    _VERDICT_TXT.write_text(_render_human_summary(result), encoding="utf-8")
    logger.info("Persisted verdict TXT to %s", _VERDICT_TXT)


def _render_human_summary(r: VerdictResult) -> str:
    lines = [
        "Factor Ensemble v1 — Walk-Forward Verdict (signal-construction layer only)",
        "=" * 78,
        f"Decision label:                       {r.decision_label}",
        f"Walk-forward window:                  see harness OOS_START_DATE / DEFAULT_END_DATE",
        f"OOS months:                           {r.n_oos_months}",
        f"Ensemble annualized Sharpe:           {r.ensemble_sharpe:+.4f}",
        f"Baseline (BAB-only) Sharpe:           {r.baseline_sharpe:+.4f}",
        f"ΔSharpe (ensemble - baseline):        {r.delta_sharpe_walk_forward:+.4f}",
        f"95% bootstrap CI:                     [{r.ci_lower_95:+.4f}, {r.ci_upper_95:+.4f}]",
        f"Memmel Z (descriptive secondary):     {r.memmel_z:+.4f}",
        f"Paired ρ̂:                             {r.paired_corr:+.4f}",
        f"Gate 0 baseline consistency status:   {r.harness_ensemble_only_baseline_consistency}",
        f"Spec hash (at verdict run):           {r.spec_hash}",
        f"Completed at:                         {r.completed_at}",
        "",
        "Interpretation caveat:",
        "  Walk-forward harness ΔSharpe measures signal-construction layer alpha;",
        "  production-forward 24mo+ live counterfactual is the swap-decision criterion.",
        "  DESCRIPTIVE_POSITIVE authorizes PendingApproval workflow but does NOT",
        "  itself flip PRODUCTION_SIGNAL.",
    ]
    return "\n".join(lines) + "\n"


def _read_current_spec_hash() -> str:
    """Best-effort read of spec_factor_ensemble_v1.md current_hash from registry."""
    try:
        from engine.memory import SessionFactory, SpecRegistry
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter(
                SpecRegistry.spec_path == "docs/spec_factor_ensemble_v1.md"
            ).first()
            if row and row.current_hash:
                return str(row.current_hash)
    except Exception as exc:
        logger.warning("Could not read spec_hash: %s", exc)
    return "UNKNOWN"
