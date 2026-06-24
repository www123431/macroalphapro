"""
engine/d_pead_plus/verdict.py — 5-gate evaluation + STRICT/MARGINAL/FAIL decision.

Spec id=74 §3 LOCK:
  Gate 1 PRIMARY: Spearman IC delta > 0.02 AND NW-t > 1.96
  Gate 2 SECONDARY: Bootstrap 95% CI on Sharpe diff > 0
  Gate 3 SECONDARY: All 5 LLM features corr with SUE < 0.30
  Gate 4 SECONDARY: OOS Sharpe / dev Sharpe > 0.75
  Gate 5 OPERATIONAL: LLM API cost / expected ann return < 0.5%

  Decision matrix:
    STRICT_PASS: all 5 gates PASS
    MARGINAL:    PRIMARY PASS + 1-2 SECONDARY FAIL
    FAIL:        PRIMARY FAIL OR 3+ SECONDARY FAIL

DOCTRINE: Decision-layer module. ZERO LLM calls. Pure statistical computation.
Enforced by engine.d_pead_plus.doctrine.audit_decision_layer_imports().
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec id=74 §3 LOCKED gate thresholds
IC_DELTA_THRESHOLD:          float = 0.02
IC_NW_T_THRESHOLD:           float = 1.96
NW_LAG_LOCKED:               int   = 60       # cover holding period
BOOTSTRAP_N_ITER:            int   = 10000
BOOTSTRAP_BLOCK_SIZE:        int   = 21       # ~1 month
BOOTSTRAP_CI_ALPHA:          float = 0.05     # 95% CI
ORTHOGONALITY_THRESHOLD:     float = 0.30
DEV_OOS_RATIO_THRESHOLD:     float = 0.75
COST_RATIO_THRESHOLD:        float = 0.005

# Persistence
CACHE_DIR    = Path("data/d_pead_plus")
VERDICT_PATH = CACHE_DIR / "v1_verdict.json"


# ─────────────────────────────────────────────────────────────────────────────
# Statistical primitives
# ─────────────────────────────────────────────────────────────────────────────
def spearman_ic(signal: pd.Series, forward_return: pd.Series) -> float:
    """Rank correlation between signal and forward return."""
    common = signal.dropna().index.intersection(forward_return.dropna().index)
    if len(common) < 3:
        return float("nan")
    return float(signal.loc[common].rank().corr(forward_return.loc[common].rank()))


def newey_west_t_stat(series: pd.Series, lag: int = NW_LAG_LOCKED) -> float:
    """NW HAC t-stat for mean(series) = 0 null hypothesis.

    se² = γ₀ + 2 Σ_{j=1..L} (1 - j/(L+1)) γⱼ  (Bartlett kernel)
    t = mean / sqrt(se²/n)
    """
    x = series.dropna().values
    n = len(x)
    if n < lag + 2:
        return float("nan")
    mean = x.mean()
    e = x - mean

    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1)
        gamma_j = float(np.dot(e[j:], e[:-j]) / n)
        var += 2 * w * gamma_j
    se = math.sqrt(max(var / n, 1e-12))
    return float(mean / se) if se > 0 else float("nan")


def block_bootstrap_sharpe_diff(
    paired_returns: pd.DataFrame,
    n_iter:         int = BOOTSTRAP_N_ITER,
    block_size:     int = BOOTSTRAP_BLOCK_SIZE,
    seed:           int = 42,
) -> tuple[float, float, float]:
    """Block bootstrap CI on (Sharpe_plus - Sharpe_baseline).

    paired_returns: DataFrame with columns 'd_pead_baseline' and 'd_pead_plus'.
    Returns (point_estimate, ci_lower_5pct, ci_upper_95pct).
    """
    rng = np.random.default_rng(seed)
    n = len(paired_returns)
    if n < block_size * 2:
        return (float("nan"), float("nan"), float("nan"))

    baseline = paired_returns["d_pead_baseline"].fillna(0).values
    plus     = paired_returns["d_pead_plus"].fillna(0).values

    def _sharpe(r):
        sd = r.std()
        return (r.mean() / sd * math.sqrt(252)) if sd > 1e-9 else 0.0

    n_blocks = (n // block_size) + 1
    diffs = np.zeros(n_iter)
    for k in range(n_iter):
        # Sample block starting indices
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        s_base = _sharpe(baseline[idx])
        s_plus = _sharpe(plus[idx])
        diffs[k] = s_plus - s_base

    point   = float(_sharpe(plus) - _sharpe(baseline))
    ci_low  = float(np.percentile(diffs, 100 * BOOTSTRAP_CI_ALPHA / 2))
    ci_high = float(np.percentile(diffs, 100 * (1 - BOOTSTRAP_CI_ALPHA / 2)))
    return (point, ci_low, ci_high)


def memmel_z(paired_returns: pd.DataFrame) -> float:
    """Memmel 2003 Z-statistic for Sharpe ratio difference (paired samples).

    Z = (Sharpe1 - Sharpe2) / sqrt(theta / n)
    where theta = 2(1 - ρ) + 0.5 (Sh1² + Sh2² - 2 ρ Sh1 Sh2)
    (Memmel's simplified form assumes normal returns; bootstrap is robust alternative)
    """
    r1 = paired_returns["d_pead_baseline"].fillna(0).values
    r2 = paired_returns["d_pead_plus"].fillna(0).values
    n = len(r1)
    if n < 30:
        return float("nan")
    mu1, sd1 = r1.mean(), r1.std()
    mu2, sd2 = r2.mean(), r2.std()
    if sd1 < 1e-9 or sd2 < 1e-9:
        return float("nan")
    sh1 = mu1 / sd1
    sh2 = mu2 / sd2
    rho = float(np.corrcoef(r1, r2)[0, 1])
    theta = 2 * (1 - rho) + 0.5 * (sh1**2 + sh2**2 - 2 * rho * sh1 * sh2)
    if theta <= 0:
        return float("nan")
    se = math.sqrt(theta / n)
    return float((sh2 - sh1) / se)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict + gate evaluation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GateResult:
    name:        str
    tier:        str            # 'PRIMARY' | 'SECONDARY' | 'OPERATIONAL'
    passed:      bool
    value:       Optional[float]
    threshold:   Optional[float]
    description: str


@dataclass
class D_Pead_Plus_Verdict:
    decision:               str
    gate_1_primary:         GateResult
    gate_2_secondary:       GateResult
    gate_3_secondary:       GateResult
    gate_4_secondary:       GateResult
    gate_5_operational:     GateResult
    ic_delta:               float
    ic_delta_nw_t:          float
    sharpe_baseline_oos:    float
    sharpe_plus_oos:        float
    sharpe_diff_point:      float
    sharpe_diff_ci_low:     float
    sharpe_diff_ci_high:    float
    memmel_z:               float
    feature_corrs_with_sue: dict
    dev_sharpe_plus:        Optional[float]
    cost_ratio:             float
    notes:                  list[str] = field(default_factory=list)


def evaluate_verdict(
    oos_panel:             pd.DataFrame,         # full OOS panel with signal scores
    paired_daily_returns:  pd.DataFrame,         # cols d_pead_baseline / d_pead_plus
    dev_sharpe_plus:       Optional[float],      # from dev fit; None if dev backtest not run
    llm_api_cost_usd:      float,
    expected_ann_return_at_1m_usd: float = 75000.0,   # mid-point of $50K-$100K range
) -> D_Pead_Plus_Verdict:
    """Run 5-gate evaluation; produce STRICT_PASS / MARGINAL / FAIL verdict."""

    # ── Gate 1 PRIMARY: Spearman IC delta + NW-t ───────────────────────────
    if "sue" not in oos_panel.columns or "ret_60d_log" not in oos_panel.columns:
        raise ValueError("OOS panel missing 'sue' or 'ret_60d_log' columns")
    if "score" not in oos_panel.columns:
        raise ValueError("OOS panel missing 'score' (run feature_combiner first)")

    ic_baseline = spearman_ic(oos_panel["sue"],   oos_panel["ret_60d_log"])
    ic_plus     = spearman_ic(oos_panel["score"], oos_panel["ret_60d_log"])
    ic_delta    = ic_plus - ic_baseline

    # Compute time-series of quarterly IC deltas for NW-t
    quarterly_ic_deltas = []
    for q, grp in oos_panel.groupby("quarter"):
        ic_b = spearman_ic(grp["sue"],   grp["ret_60d_log"])
        ic_p = spearman_ic(grp["score"], grp["ret_60d_log"])
        if not (math.isnan(ic_b) or math.isnan(ic_p)):
            quarterly_ic_deltas.append(ic_p - ic_b)
    ic_delta_nw_t = newey_west_t_stat(pd.Series(quarterly_ic_deltas), lag=min(NW_LAG_LOCKED, len(quarterly_ic_deltas) // 3))

    gate_1 = GateResult(
        name="ic_delta_primary", tier="PRIMARY",
        passed=(ic_delta > IC_DELTA_THRESHOLD and (not math.isnan(ic_delta_nw_t)) and ic_delta_nw_t > IC_NW_T_THRESHOLD),
        value=float(ic_delta),
        threshold=IC_DELTA_THRESHOLD,
        description=f"ic_delta={ic_delta:+.4f} (NW-t={ic_delta_nw_t:+.3f}); threshold IC>{IC_DELTA_THRESHOLD} AND NW-t>{IC_NW_T_THRESHOLD}",
    )

    # ── Gate 2 SECONDARY: Bootstrap CI on Sharpe diff ──────────────────────
    point, ci_low, ci_high = block_bootstrap_sharpe_diff(paired_daily_returns)
    gate_2 = GateResult(
        name="bootstrap_sharpe_ci", tier="SECONDARY",
        passed=(not math.isnan(ci_low)) and ci_low > 0,
        value=float(point),
        threshold=0.0,
        description=f"Sharpe diff point={point:+.4f}, 95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]; pass if CI_low > 0",
    )

    # ── Gate 3 SECONDARY: feature orthogonality ────────────────────────────
    feature_corrs: dict[str, float] = {}
    if "sue" in oos_panel.columns:
        for feat in ("tone_score", "forward_confidence", "macro_headwind_flag", "evasion_score", "linguistic_complexity"):
            if feat in oos_panel.columns:
                feature_corrs[feat] = float(oos_panel[feat].corr(oos_panel["sue"]))
    max_abs_corr = max((abs(v) for v in feature_corrs.values()), default=0.0)
    gate_3 = GateResult(
        name="feature_orthogonality", tier="SECONDARY",
        passed=max_abs_corr < ORTHOGONALITY_THRESHOLD,
        value=float(max_abs_corr),
        threshold=ORTHOGONALITY_THRESHOLD,
        description=f"max |corr(feature, SUE)|={max_abs_corr:.4f}; threshold<{ORTHOGONALITY_THRESHOLD}",
    )

    # ── Gate 4 SECONDARY: Dev/OOS consistency ──────────────────────────────
    sharpe_baseline_oos = float(paired_daily_returns["d_pead_baseline"].mean() /
                                  (paired_daily_returns["d_pead_baseline"].std() + 1e-12) * math.sqrt(252))
    sharpe_plus_oos     = float(paired_daily_returns["d_pead_plus"].mean() /
                                  (paired_daily_returns["d_pead_plus"].std() + 1e-12) * math.sqrt(252))
    if dev_sharpe_plus is None or math.isnan(dev_sharpe_plus) or dev_sharpe_plus < 1e-9:
        dev_oos_ratio = float("nan")
        gate_4_pass = False
        gate_4_desc = f"dev_sharpe_plus N/A; OOS Sharpe={sharpe_plus_oos:+.3f}"
    else:
        dev_oos_ratio = sharpe_plus_oos / dev_sharpe_plus
        gate_4_pass = (not math.isnan(dev_oos_ratio)) and dev_oos_ratio > DEV_OOS_RATIO_THRESHOLD
        gate_4_desc = f"OOS/dev Sharpe ratio={dev_oos_ratio:.4f} (OOS={sharpe_plus_oos:+.3f}, dev={dev_sharpe_plus:+.3f}); threshold>{DEV_OOS_RATIO_THRESHOLD}"

    gate_4 = GateResult(
        name="dev_oos_consistency", tier="SECONDARY",
        passed=gate_4_pass,
        value=float(dev_oos_ratio) if not math.isnan(dev_oos_ratio) else None,
        threshold=DEV_OOS_RATIO_THRESHOLD,
        description=gate_4_desc,
    )

    # ── Gate 5 OPERATIONAL: Cost ratio ─────────────────────────────────────
    cost_ratio = (llm_api_cost_usd * 2) / expected_ann_return_at_1m_usd  # *2 for annual refresh estimate
    gate_5 = GateResult(
        name="cost_ratio_operational", tier="OPERATIONAL",
        passed=cost_ratio < COST_RATIO_THRESHOLD,
        value=float(cost_ratio),
        threshold=COST_RATIO_THRESHOLD,
        description=f"LLM cost ${llm_api_cost_usd:.2f} × 2 / expected ${expected_ann_return_at_1m_usd:.0f} = {cost_ratio:.4%}; threshold<{COST_RATIO_THRESHOLD:.2%}",
    )

    # ── Decision matrix (per spec §3.2) ─────────────────────────────────────
    primary_pass = gate_1.passed
    secondary_passes = sum([gate_2.passed, gate_3.passed, gate_4.passed])
    secondary_fails  = 3 - secondary_passes

    if primary_pass and secondary_fails == 0 and gate_5.passed:
        decision = "STRICT_PASS"
    elif primary_pass and secondary_fails <= 2:
        decision = "MARGINAL"
    else:
        decision = "FAIL"

    notes = [
        f"Spec id=74 hash d0532f8f; doctrine 0-LLM-in-DECISION enforced.",
        f"OOS sample n={len(oos_panel)} firm-quarters across {oos_panel['quarter'].nunique()} quarters.",
        f"Paired daily returns n={len(paired_daily_returns)} trading days.",
        f"Bootstrap: {BOOTSTRAP_N_ITER} iter, block_size={BOOTSTRAP_BLOCK_SIZE}, alpha={BOOTSTRAP_CI_ALPHA}.",
        f"Memmel Z (secondary diagnostic): {memmel_z(paired_daily_returns):+.3f}.",
        f"Verdict: {decision} ({sum([gate_1.passed, gate_2.passed, gate_3.passed, gate_4.passed, gate_5.passed])}/5 gates PASS).",
    ]

    return D_Pead_Plus_Verdict(
        decision               = decision,
        gate_1_primary         = gate_1,
        gate_2_secondary       = gate_2,
        gate_3_secondary       = gate_3,
        gate_4_secondary       = gate_4,
        gate_5_operational     = gate_5,
        ic_delta               = float(ic_delta),
        ic_delta_nw_t          = float(ic_delta_nw_t),
        sharpe_baseline_oos    = sharpe_baseline_oos,
        sharpe_plus_oos        = sharpe_plus_oos,
        sharpe_diff_point      = float(point),
        sharpe_diff_ci_low     = float(ci_low),
        sharpe_diff_ci_high    = float(ci_high),
        memmel_z               = memmel_z(paired_daily_returns),
        feature_corrs_with_sue = feature_corrs,
        dev_sharpe_plus        = dev_sharpe_plus,
        cost_ratio             = float(cost_ratio),
        notes                  = notes,
    )


def save_verdict(verdict: D_Pead_Plus_Verdict, save_path: Optional[Path] = None) -> dict:
    """Save verdict to JSON."""
    if save_path is None:
        save_path = VERDICT_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert_gate(g: GateResult) -> dict:
        return {
            "name":        g.name,
            "tier":        g.tier,
            "passed":      g.passed,
            "value":       g.value,
            "threshold":   g.threshold,
            "description": g.description,
        }

    payload = {
        "spec_id":   74,
        "spec_hash": "6d8e614ebd68ec42d071949bfd4299b0e4a7a363",   # post-amendment 1
        "doctrine":  "0-LLM-in-DECISION (amendment 2026-05-13)",
        "decision":  verdict.decision,
        "run_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "gate_1_primary":      _convert_gate(verdict.gate_1_primary),
        "gate_2_secondary":    _convert_gate(verdict.gate_2_secondary),
        "gate_3_secondary":    _convert_gate(verdict.gate_3_secondary),
        "gate_4_secondary":    _convert_gate(verdict.gate_4_secondary),
        "gate_5_operational":  _convert_gate(verdict.gate_5_operational),
        "metrics": {
            "ic_delta":            verdict.ic_delta,
            "ic_delta_nw_t":       verdict.ic_delta_nw_t,
            "sharpe_baseline_oos": verdict.sharpe_baseline_oos,
            "sharpe_plus_oos":     verdict.sharpe_plus_oos,
            "sharpe_diff_point":   verdict.sharpe_diff_point,
            "sharpe_diff_ci_95":   [verdict.sharpe_diff_ci_low, verdict.sharpe_diff_ci_high],
            "memmel_z":            verdict.memmel_z,
            "dev_sharpe_plus":     verdict.dev_sharpe_plus,
            "cost_ratio":          verdict.cost_ratio,
        },
        "feature_corrs_with_sue":  verdict.feature_corrs_with_sue,
        "notes":                    verdict.notes,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def get_locked_constants() -> dict:
    return {
        "IC_DELTA_THRESHOLD":      IC_DELTA_THRESHOLD,
        "IC_NW_T_THRESHOLD":       IC_NW_T_THRESHOLD,
        "NW_LAG_LOCKED":           NW_LAG_LOCKED,
        "BOOTSTRAP_N_ITER":        BOOTSTRAP_N_ITER,
        "BOOTSTRAP_BLOCK_SIZE":    BOOTSTRAP_BLOCK_SIZE,
        "BOOTSTRAP_CI_ALPHA":      BOOTSTRAP_CI_ALPHA,
        "ORTHOGONALITY_THRESHOLD": ORTHOGONALITY_THRESHOLD,
        "DEV_OOS_RATIO_THRESHOLD": DEV_OOS_RATIO_THRESHOLD,
        "COST_RATIO_THRESHOLD":    COST_RATIO_THRESHOLD,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== D-PEAD-Plus Verdict — Locked Constants ===")
    for k, v in get_locked_constants().items():
        print(f"  {k}: {v}")
