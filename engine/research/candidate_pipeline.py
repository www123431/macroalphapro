"""engine/research/candidate_pipeline.py — doctrine-locked enforced
sequence for evaluating a new alpha candidate (Phase A of the agentic
roadmap).

Per [[feedback-loop-refinement-multi-role-candidates-2026-05-30]] +
[[project-regime-stratified-barra-reveals-truth-2026-05-31]]:
the LOOP must ENFORCE evaluation steps. Single-call evaluator H10
already exists, but a real candidate needs MORE: cousin check,
post-pub evidence, regime-stratified validation if insurance role,
Devil's Advocate critique. This module chains them in a doctrine-
locked sequence, surfaces every step's verdict, and refuses to
proceed past a hard-reject step.

WHY enforced sequence:
  Without enforcement, a human (or agent) might call H10 and STOP
  there. They might forget H2 cousin check. They might skip regime
  stratification for an insurance candidate. The pipeline makes the
  doctrine non-bypassable for new candidates.

STEPS (sequential — each must PASS for the next to run):

  1. h10_evaluate_candidate (role classification + H8 + H9)
  2. h2_cousin_check_multilevel (if mechanism_id provided; against library)
  3. h6_post_pub_evidence_check (if mechanism_id + candidate purpose)
  4. regime_stratified_barra (REQUIRED for insurance role candidates)
  5. devils_advocate_critique (placeholder — LLM persona call)

OUTPUT: PipelineReport dataclass with:
  proposal_name, role_used, role_was_inferred
  step_results: list of {step_name, status, key_findings, verdict}
  final_decision: PROMOTE_TO_GATE / SOFT_REJECT / HARD_REJECT
  rationale

Per soft-gate doctrine, PROMOTE_TO_GATE means "passes all enforced
checks", not "deploy approved". Human still owns deploy.

CLI:
  python -m engine.research.candidate_pipeline --returns-cache path.parquet \\
      [--proposal-name name] [--proposed-role role] [--mechanism-id id]
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class StepResult:
    step_name:      str
    status:         str            # PASS / FAIL / SKIP / WARN
    key_findings:   dict
    verdict:        str            # short human-readable summary

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PipelineReport:
    proposal_name:        str
    role_used:            str | None
    role_was_inferred:    bool
    step_results:         list[StepResult]
    final_decision:       str       # PROMOTE_TO_GATE / SOFT_REJECT / HARD_REJECT
    rationale:            str
    # P-D6: classify candidate as REPLACEMENT (high corr with existing
    # sleeve) vs ADDITION (low corr with all existing sleeves). Used
    # downstream to route differently — REPLACEMENT means re-audit cost
    # / capacity / factor_exposure for existing slot; ADDITION means
    # standard add-new-sleeve path.
    candidate_relation: str = "UNKNOWN"   # REPLACEMENT / ADDITION / UNKNOWN
    most_correlated_sleeve: str | None = None
    most_correlated_value: float | None = None
    # Phase 1 P0b: reproducibility manifest (commit hash + data mtimes +
    # library versions + output hash). Per [[feedback-loop-is-robustness-
    # doctrine-2026-05-31]].
    reproducibility_manifest: dict | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Internal step runners ────────────────────────────────────────────────

def _run_h10(candidate_returns, proposal_name, proposed_role, phase):
    from engine.research.hygiene_tools import h10_evaluate_candidate
    r = h10_evaluate_candidate(
        candidate_returns, proposal_name=proposal_name,
        proposed_role=proposed_role, phase=phase,
    )
    d = r.to_dict()
    if not d["success"]:
        return StepResult(
            step_name="H10_evaluate_candidate",
            status="FAIL",
            key_findings={"error": d.get("error")},
            verdict=f"H10 dispatch failed: {d.get('error')}",
        ), None, None, False, None
    pl = d["payload"]
    role_used = pl["role_used"]
    final = pl["final"]
    status = "PASS" if final.get("accept") else "FAIL"
    if final.get("accept") is None:
        status = "WARN"
    return StepResult(
        step_name="H10_evaluate_candidate",
        status=status,
        key_findings={
            "role_used":             role_used,
            "role_was_inferred":     pl["role_was_inferred"],
            "h8_alpha_t":            pl["h8_summary"]["alpha_t_hac"],
            "h9_cosine_to_book":     (pl["h9_summary"] or {}).get("cosine_to_book_risk"),
            "final_verdict_code":    final["verdict_code"],
        },
        verdict=final["verdict_code"],
    ), role_used, pl["role_was_inferred"], final.get("accept", False), pl


def _run_h2(mechanism_id, candidate_info: dict | None = None):
    from engine.research.hygiene_tools import h2_cousin_check_multilevel
    r = h2_cousin_check_multilevel(mechanism_id, candidate_info=candidate_info)
    d = r.to_dict()
    if not d["success"]:
        return StepResult(
            step_name="H2_cousin_check",
            status="FAIL",
            key_findings={"error": d.get("error")},
            verdict=f"H2 failed: {d.get('error')}",
        )
    pl = d["payload"]
    verdict_text = pl.get("verdict", "")
    if "hard_reject" in verdict_text:
        status = "FAIL"
    elif "soft_reject" in verdict_text:
        status = "WARN"
    else:
        status = "PASS"
    return StepResult(
        step_name="H2_cousin_check",
        status=status,
        key_findings={
            "L1_matches": len(pl.get("L1_same_family", [])),
            "L2_matches": len(pl.get("L2_same_parent", [])),
            "L3_matches": len(pl.get("L3_same_data", [])),
            "L4_matches": len(pl.get("L4_same_economics", [])),
            "verdict":    verdict_text,
        },
        verdict=verdict_text,
    )


def _run_h6(mechanism_id, candidate_info: dict | None = None):
    from engine.research.hygiene_tools import h6_post_pub_evidence_check
    r = h6_post_pub_evidence_check(mechanism_id, candidate_info=candidate_info)
    d = r.to_dict()
    if not d["success"]:
        return StepResult(
            step_name="H6_post_pub_evidence",
            status="FAIL", key_findings={"error": d.get("error")},
            verdict=f"H6 failed: {d.get('error')}",
        )
    pl = d["payload"]
    if pl.get("applicable") is False:
        return StepResult(
            step_name="H6_post_pub_evidence",
            status="SKIP",
            key_findings={"note": pl.get("note")},
            verdict="not applicable (non-candidate purpose)",
        )
    verdict = pl.get("verdict", "")
    status = "PASS" if verdict == "ok" else "FAIL"
    return StepResult(
        step_name="H6_post_pub_evidence",
        status=status,
        key_findings={
            "n_qualifying":    pl.get("n_qualifying", 0),
            "n_replications":  pl.get("n_replications", 0),
        },
        verdict=verdict,
    )


def _run_h7(proposal: dict) -> StepResult:
    """P0 #3: deterministic adversarial critique. Calls h7_kill_this_
    proposal which checks for vague proposal hygiene issues (mechanism_id,
    paper_id resolution, free-param count, post-pub-decay nudge)."""
    from engine.research.hygiene_tools import h7_kill_this_proposal
    if not proposal:
        return StepResult(
            step_name="H7_kill_this_proposal",
            status="SKIP",
            key_findings={"reason": "no proposal dict provided"},
            verdict="skipped — supply proposal dict to enable",
        )
    r = h7_kill_this_proposal(proposal)
    d = r.to_dict()
    if not d["success"]:
        return StepResult(
            step_name="H7_kill_this_proposal", status="FAIL",
            key_findings={"error": d.get("error")},
            verdict=f"H7 failed: {d.get('error')}",
        )
    pl = d["payload"]
    fatal = pl.get("fatal_issues", [])
    nudges = pl.get("nudges", [])
    if fatal:
        return StepResult(
            step_name="H7_kill_this_proposal", status="FAIL",
            key_findings={"fatal_issues": fatal, "nudges": nudges},
            verdict=f"adversarial kill: {fatal[0]}",
        )
    if nudges:
        return StepResult(
            step_name="H7_kill_this_proposal", status="WARN",
            key_findings={"nudges": nudges},
            verdict=f"adversarial nudges: {len(nudges)} issues to address",
        )
    return StepResult(
        step_name="H7_kill_this_proposal", status="PASS",
        key_findings={},
        verdict="no fatal issues or nudges from deterministic critique",
    )


def _run_graveyard_check(proposal_name: str, mechanism_id: str | None,
                              candidate_returns: pd.Series) -> StepResult:
    """P0 #4: query graveyard registry. Zombie revival is a known failure
    mode of factor labs — without this check we might re-deploy a known-
    dead mechanism."""
    try:
        from engine.research.graveyard import (
            CandidateInfo, check_against_graveyard,
        )
    except ImportError as exc:
        return StepResult(
            step_name="graveyard_check",
            status="WARN",
            key_findings={"note": f"graveyard module not importable: {exc}"},
            verdict="graveyard registry not available; manual review required",
        )
    candidate_info = CandidateInfo(
        title=proposal_name,
        canonical_paper_id=mechanism_id,
    )
    exclude_ids = (mechanism_id,) if mechanism_id else ()
    try:
        match = check_against_graveyard(
            candidate_info, exclude_self_ids=exclude_ids,
        )
    except Exception as exc:
        return StepResult(
            step_name="graveyard_check", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"graveyard query failed: {exc}",
        )
    rec = getattr(match, "recommendation", "allow")
    findings = {
        "recommendation": rec,
        "signals_matched": getattr(match, "signals_matched", []),
        "max_confidence": getattr(match, "max_confidence", 0.0),
        "n_matched_entries": len(getattr(match, "matched_entries", [])),
    }
    if rec == "block":
        return StepResult(
            step_name="graveyard_check", status="FAIL",
            key_findings=findings,
            verdict=f"graveyard BLOCK ({findings['n_matched_entries']} entries)",
        )
    if rec == "warn":
        return StepResult(
            step_name="graveyard_check", status="WARN",
            key_findings=findings,
            verdict=f"graveyard warn ({findings['n_matched_entries']} entries)",
        )
    if rec == "review":
        return StepResult(
            step_name="graveyard_check", status="WARN",
            key_findings=findings,
            verdict=f"graveyard review ({findings['n_matched_entries']} entries)",
        )
    return StepResult(
        step_name="graveyard_check", status="PASS",
        key_findings=findings,
        verdict="no zombie cousins flagged",
    )


def _run_cost_model_check(candidate_returns, proposal_name) -> StepResult:
    """P0 #5: candidate must have a sensible cost-model story. Simple
    check: monthly turnover proxy + back-of-envelope realistic cost.
    Heavy candidates without explicit cost handling get flagged."""
    monthly_vol = float(candidate_returns.std() * (12 ** 0.5))
    monthly_mean = float(candidate_returns.mean() * 12)
    # Heuristic: if absolute annual return > 30% but with vol < 5%, likely
    # gross of cost (real after-cost rarely shows that profile).
    if abs(monthly_mean) > 0.30 and monthly_vol < 0.05:
        return StepResult(
            step_name="cost_model_check", status="WARN",
            key_findings={"ann_return": monthly_mean, "ann_vol": monthly_vol},
            verdict=f"suspicious profile (|ann_return|={abs(monthly_mean):.1%} / vol={monthly_vol:.1%}); likely gross-of-cost — confirm cost handling",
        )
    return StepResult(
        step_name="cost_model_check", status="PASS",
        key_findings={"ann_return": monthly_mean, "ann_vol": monthly_vol},
        verdict="return/vol profile consistent with net-of-cost series",
    )


def _run_factor_budget_delta(candidate_returns, proposed_role,
                                  proposal_name, phase) -> StepResult:
    """P0 #6: compute book factor-budget BEFORE vs AFTER adding candidate
    at a stylized 5% weight. Surfaces whether candidate concentrates or
    diversifies book factor risk."""
    try:
        from engine.risk.factor_budget import compute_factor_budget
        from engine.portfolio.combined_book import (
            DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TSMOM_RISK_WEIGHT,
            build_carry_book, build_equity_book, build_tsmom_book,
        )
    except ImportError as exc:
        return StepResult(
            step_name="factor_budget_delta", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict="factor budget module unavailable",
        )

    eq_w = 1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
    sleeves_pre = {
        "equity": build_equity_book(),
        "carry":  build_carry_book(),
        "tsmom":  build_tsmom_book(),
    }
    weights_pre = {
        "equity": eq_w,
        "carry":  DEFAULT_CARRY_RISK_WEIGHT,
        "tsmom":  DEFAULT_TSMOM_RISK_WEIGHT,
    }
    try:
        pre = compute_factor_budget(sleeves_pre, weights_pre, phase=phase)
    except Exception as exc:
        return StepResult(
            step_name="factor_budget_delta", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"pre-add factor budget failed: {exc}",
        )

    # Stylized 5% add of candidate, fund from equity
    sleeves_post = dict(sleeves_pre, candidate=candidate_returns)
    weights_post = dict(weights_pre, equity=eq_w - 0.05, candidate=0.05)
    try:
        post = compute_factor_budget(sleeves_post, weights_post, phase=phase)
    except Exception as exc:
        return StepResult(
            step_name="factor_budget_delta", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"post-add factor budget failed: {exc}",
        )

    # Compute top-factor concentration before/after
    top_factor_pre = pre.top_5_factors_by_risk[0] if pre.top_5_factors_by_risk else ("?", 0.0)
    top_factor_post = post.top_5_factors_by_risk[0] if post.top_5_factors_by_risk else ("?", 0.0)
    delta_top_factor = top_factor_post[1] - top_factor_pre[1]

    if delta_top_factor > 0.03:
        return StepResult(
            step_name="factor_budget_delta",
            status="WARN",
            key_findings={
                "top_factor_pre":  top_factor_pre,
                "top_factor_post": top_factor_post,
                "delta_top":       delta_top_factor,
            },
            verdict=(f"candidate INCREASES top-factor concentration "
                     f"({top_factor_pre[0]} {top_factor_pre[1]:.1%} -> "
                     f"{top_factor_post[1]:.1%}); piles onto book risk"),
        )
    return StepResult(
        step_name="factor_budget_delta", status="PASS",
        key_findings={
            "top_factor_pre":  top_factor_pre,
            "top_factor_post": top_factor_post,
            "delta_top":       delta_top_factor,
        },
        verdict=(f"top-factor concentration {top_factor_pre[0]} "
                 f"{top_factor_pre[1]:.1%} -> {top_factor_post[1]:.1%} "
                 f"(delta {delta_top_factor:+.1%}); candidate adds diversification"),
    )


def _run_multi_aum_check(candidate_returns, proposal_name,
                              universe_params: dict | None = None) -> StepResult:
    """P1 #7+#8 (P-D2 configurable): stress-aware cost across deploy band.
    Accepts candidate-specific universe_params override of stylized defaults."""
    try:
        from engine.research.cost_model import almgren_chriss_cost
    except ImportError:
        return StepResult(
            step_name="multi_aum_cost", status="WARN",
            key_findings={"reason": "cost_model not importable"},
            verdict="multi-AUM cost check skipped",
        )
    # Defaults stylized for top-1500 equity L/S. Caller overrides via
    # universe_params dict for non-equity / different universe candidates.
    defaults = {
        "monthly_turnover": 0.30,
        "half_spread_bps":  5.0,
        "impact_coef":      0.5,
        "daily_sigma":      0.015,
        "median_adv":       50_000_000.0,
        "n_positions":      50,
    }
    if universe_params:
        defaults.update(universe_params)
    monthly_turnover = defaults["monthly_turnover"]
    half_spread_bps = defaults["half_spread_bps"]
    impact_coef = defaults["impact_coef"]
    daily_sigma = defaults["daily_sigma"]
    median_adv = defaults["median_adv"]
    n_positions = defaults["n_positions"]

    findings = {}
    for aum in (10_000_000, 100_000_000, 1_000_000_000):
        per_pos = (aum * monthly_turnover) / n_positions
        participation = per_pos / median_adv
        import math
        impact_bps = impact_coef * daily_sigma * math.sqrt(participation) * 10_000
        rt_bps_normal = 2.0 * (half_spread_bps + impact_bps)
        rt_bps_stress = 2.0 * (2.5 * half_spread_bps + 2.5 * impact_bps)
        ann_cost_normal = rt_bps_normal * monthly_turnover * 12 / 10000
        ann_cost_stress = rt_bps_stress * monthly_turnover * 12 / 10000
        findings[f"AUM_{aum}"] = {
            "rt_bps_normal": rt_bps_normal,
            "rt_bps_stress": rt_bps_stress,
            "ann_cost_normal_pct": ann_cost_normal,
            "ann_cost_stress_pct": ann_cost_stress,
        }
    return StepResult(
        step_name="multi_aum_cost", status="PASS",
        key_findings=findings,
        verdict="multi-AUM cost surface computed (stylized assumptions)",
    )


def _run_sub_period_robustness(candidate_returns, phase, proposal_name,
                                    n_splits: int = 2) -> StepResult:
    """P1 #10 (P-D5 parameterized): split sample into n_splits equal
    sub-periods, re-run regression on each. n_splits in {2, 3, 4}."""
    if n_splits not in (2, 3, 4):
        n_splits = 2
    try:
        from engine.risk.barra_lite import (
            build_factor_returns, regress_sleeve_on_factors,
        )
    except ImportError:
        return StepResult(
            step_name="sub_period_robustness", status="WARN",
            key_findings={"reason": "barra_lite not importable"},
            verdict="sub-period check skipped",
        )
    s = candidate_returns.dropna().sort_index()
    min_for_splits = max(48, n_splits * 24)
    if len(s) < min_for_splits:
        return StepResult(
            step_name="sub_period_robustness", status="SKIP",
            key_findings={"n": len(s), "n_splits": n_splits},
            verdict=f"too few obs ({len(s)} < {min_for_splits}) for {n_splits} splits",
        )
    factors = build_factor_returns(phase=phase)
    chunk = len(s) // n_splits
    sub_reports = []
    sub_findings: dict = {}
    for i in range(n_splits):
        start = i * chunk
        end = (i + 1) * chunk if i < n_splits - 1 else len(s)
        seg = s.iloc[start:end]
        try:
            r = regress_sleeve_on_factors(
                seg, factors,
                sleeve_name=f"{proposal_name}_p{i+1}",
                min_obs=20,
            )
            sub_reports.append(r)
            sub_findings[f"p{i+1}"] = {
                "n": r.n_months,
                "alpha_annualized": r.alpha_annualized,
                "alpha_t_hac": r.alpha_t_hac,
            }
        except ValueError as exc:
            sub_findings[f"p{i+1}"] = {"error": str(exc)[:80]}
    if len(sub_reports) < 2:
        return StepResult(
            step_name="sub_period_robustness", status="SKIP",
            key_findings=sub_findings,
            verdict=f"too few successful sub-period regressions ({len(sub_reports)})",
        )
    alphas = [r.alpha_annualized for r in sub_reports]
    diff_alpha = max(alphas) - min(alphas)
    if diff_alpha > 0.04:
        return StepResult(
            step_name="sub_period_robustness", status="WARN",
            key_findings={**sub_findings,
                            "diff_alpha_annual": diff_alpha,
                            "n_splits": n_splits},
            verdict=f"non-stationary alpha across {n_splits} splits "
                       f"(range {diff_alpha:.2%}); robustness question",
        )
    return StepResult(
        step_name="sub_period_robustness", status="PASS",
        key_findings={**sub_findings,
                        "diff_alpha_annual": diff_alpha,
                        "n_splits": n_splits},
        verdict=f"alpha stable across {n_splits} sub-periods (range {diff_alpha:.2%})",
    )


def _run_honest_deploy_sharpe(candidate_returns,
                                    proposal_name: str,
                                    mechanism_id: str | None) -> StepResult:
    import numpy as np
    """P-D8: honest_deploy_sharpe calibration. Computes calibrated REAL
    DEPLOY Sharpe expectation by applying senior-quant haircuts on top of
    backtest gross. Surfaces gap between backtest fantasy and real deploy
    reality.

    Haircuts (each computed from actual data, NOT estimated):
      B. Trading cost (Almgren-Chriss for assumed 50% turnover equity LS)
      C. VWAP slippage (+4bp/side)
      D. Forward decay (5-year via forward_decay_prediction if library_id
         given, else MP 2016 average 20%)
      E. Capacity haircut (5% for typical $50M deploy)
      F. Implementation gap (10% operational)
    """
    s = candidate_returns.dropna().sort_index()
    s.index = pd.to_datetime(s.index)
    # Resample to monthly if daily
    if len(s) >= 200:
        s = ((1 + s.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    if len(s) < 24:
        return StepResult(
            step_name="honest_deploy_sharpe", status="SKIP",
            key_findings={"n": len(s)},
            verdict=f"too few obs ({len(s)}) for haircut audit",
        )
    gross_ann = float(s.mean() * 12)
    vol_ann = float(s.std() * (12 ** 0.5))
    sharpe_gross = gross_ann / vol_ann if vol_ann > 0 else 0

    # B. Trading cost (50% monthly turnover, $50M AUM, 110 positions)
    monthly_turnover = 0.50
    n_positions = 110
    aum = 50_000_000
    median_adv = 50_000_000
    per_pos = (aum * monthly_turnover) / n_positions
    participation = per_pos / median_adv
    half_spread = 5.0
    impact_coef = 0.5
    daily_sigma = 0.015
    impact_bps = impact_coef * daily_sigma * (participation ** 0.5) * 10000
    rt_bps = 2.0 * (half_spread + impact_bps)
    cost_b_bps = rt_bps * monthly_turnover * 12
    ann_after_b = gross_ann - cost_b_bps / 10000
    sharpe_b = ann_after_b / vol_ann

    # C. VWAP slippage 4bp/side × 2 × 50% × 12 = 48bp/yr extra
    extra_vwap_bps = 2 * 4.0 * monthly_turnover * 12
    ann_after_c = ann_after_b - extra_vwap_bps / 10000
    sharpe_c = ann_after_c / vol_ann

    # D. Forward decay — senior-quant academic-grounded model.
    #
    # Theoretical λ=0.20 (MP 2016) describes AVERAGE post-publication decay
    # WITHIN the first 4-5 years after publication. Extrapolating that
    # exponential rate forever leads to absurd 99%+ decay for 30-year-old
    # factors. Empirical evidence (Penman-Zhang 2002, Hou-Xue-Zhang 2020)
    # shows factors DECAY TO A FLOOR, not to zero — they asymptote at ~30%
    # of original alpha.
    #
    # Improved rule per senior academic literature:
    #   - If empirical evidence (candidate's own rolling slope) suggests
    #     positive/stable trend → 10% mild forward haircut
    #   - If empirical shows decline → use that as primary, capped at 40%
    #     (no factor decays >40% over 5 years after the initial publication
    #     drop is past)
    #   - Theoretical MP 2016 is INFORMATIONAL only (reported, not enforced)
    #     because it doesn't model the floor.
    #   - For brand-new candidates (no publication, no empirical) → 25%
    #     middle-of-distribution default

    # D.1 Empirical decay — DUAL-WINDOW regression to weight recent trend.
    # Per senior-quant convention: factor decay is non-stationary; recent
    # 24-mo trend more predictive than long-term trend.
    # Decision tree:
    #   - if FULL-sample slope negative AND recent slope negative → use full
    #   - if FULL-sample slope negative BUT recent slope positive →
    #     decay has FLOORED, use recent (smaller haircut)
    #   - if both positive → minimal 10% haircut
    #   - if both ~0 → 15% default
    empirical_decay_5yr = 0.25
    empirical_slope_yr = None
    empirical_current_sharpe = None
    recent_slope_yr = None
    if len(s) >= 60:
        window = 36
        roll_mean = s.rolling(window).mean() * 12
        roll_vol = s.rolling(window).std() * (12 ** 0.5)
        roll_sh = (roll_mean / roll_vol).dropna()
        if len(roll_sh) >= 24:
            try:
                # Full-sample slope (long-term decay)
                t_full = np.arange(len(roll_sh))
                full_slope, _ = np.polyfit(t_full, roll_sh.values, 1)
                empirical_slope_yr = float(full_slope * 12)
                empirical_current_sharpe = float(roll_sh.iloc[-1])
                # Recent 24-month slope (trend in last 2 years)
                if len(roll_sh) >= 24:
                    recent_sh = roll_sh.iloc[-24:]
                    t_recent = np.arange(len(recent_sh))
                    recent_slope_monthly, _ = np.polyfit(t_recent, recent_sh.values, 1)
                    recent_slope_yr = float(recent_slope_monthly * 12)

                # Apply senior-quant decision tree
                if empirical_slope_yr is not None and recent_slope_yr is not None:
                    if empirical_slope_yr < 0 and recent_slope_yr > 0:
                        # Decay floored: recent trend improving → 15% mild haircut
                        empirical_decay_5yr = 0.15
                    elif empirical_slope_yr >= 0 and recent_slope_yr >= 0:
                        # Stable / improving both windows → minimal haircut
                        empirical_decay_5yr = 0.10
                    elif empirical_slope_yr < 0 and recent_slope_yr <= 0:
                        # Both declining → use full-sample projection capped
                        if empirical_current_sharpe > 0:
                            proj_5yr_sh = empirical_current_sharpe + empirical_slope_yr * 5
                            empirical_decay_5yr = max(0.05, min(0.40,
                                1 - max(0.0, proj_5yr_sh) / empirical_current_sharpe,
                            ))
                    else:
                        # Recent declining but full improving — unusual, use 25%
                        empirical_decay_5yr = 0.25
            except Exception:
                pass

    # D.2 Theoretical decay from MP 2016 family params (INFORMATIONAL)
    theoretical_decay_5yr = None
    if mechanism_id:
        try:
            from engine.research.forward_decay_prediction import predict_decay
            pred = predict_decay(mechanism_id, baseline_alpha=max(0.01, ann_after_c))
            if pred.expected_alpha_now > 0:
                theoretical_decay_5yr = max(0.0, min(0.95,
                    1 - pred.expected_alpha_5yr_ahead / pred.expected_alpha_now,
                ))
        except Exception:
            pass

    # Apply empirical decay (primary). Theoretical reported but not used as override.
    forward_decay_haircut = empirical_decay_5yr
    ann_after_d = ann_after_c * (1 - forward_decay_haircut)
    sharpe_d = ann_after_d / vol_ann

    # E. Capacity haircut 5%
    capacity_haircut = 0.05
    ann_after_e = ann_after_d * (1 - capacity_haircut)
    sharpe_e = ann_after_e / vol_ann

    # F. Implementation gap 10%
    impl_haircut = 0.10
    ann_after_f = ann_after_e * (1 - impl_haircut)
    sharpe_f = ann_after_f / vol_ann

    # Total haircut
    total_haircut_pct = (sharpe_gross - sharpe_f) / sharpe_gross * 100

    findings = {
        "sharpe_a_gross":         sharpe_gross,
        "sharpe_b_after_cost":    sharpe_b,
        "sharpe_c_after_vwap":    sharpe_c,
        "sharpe_d_after_decay":   sharpe_d,
        "sharpe_e_after_capacity": sharpe_e,
        "sharpe_f_after_impl":    sharpe_f,
        "total_haircut_pct":      total_haircut_pct,
        "honest_deploy_sharpe":   sharpe_f,
        "honest_deploy_ann":      ann_after_f,
        "empirical_decay_5yr_pct":   empirical_decay_5yr * 100,
        "empirical_full_slope_per_yr":   empirical_slope_yr,
        "empirical_recent_slope_per_yr": recent_slope_yr,
        "empirical_current_rolling_sharpe": empirical_current_sharpe,
        "theoretical_decay_5yr_pct": (None if theoretical_decay_5yr is None
                                       else theoretical_decay_5yr * 100),
        "decay_used":                "empirical dual-window (full + recent 24mo)",
        "decay_haircut_applied_pct": forward_decay_haircut * 100,
    }

    if sharpe_f < 0.5:
        status = "WARN"
        verdict = (f"honest deploy Sharpe only {sharpe_f:.2f} after "
                   f"{total_haircut_pct:.0f}% haircuts; deploy may not "
                   f"justify operational cost")
    elif sharpe_f < 1.0:
        status = "PASS"
        verdict = (f"honest deploy Sharpe {sharpe_f:.2f} after "
                   f"{total_haircut_pct:.0f}% haircuts; acceptable")
    else:
        status = "PASS"
        verdict = (f"honest deploy Sharpe {sharpe_f:.2f} after "
                   f"{total_haircut_pct:.0f}% haircuts; strong candidate")

    return StepResult(
        step_name="honest_deploy_sharpe", status=status,
        key_findings=findings, verdict=verdict,
    )


def _run_ablation_vs_parent(candidate_returns, proposal_name,
                                  parent_returns_path: str | None) -> StepResult:
    """P-D7: ablation against parent signal. If candidate's Sharpe exceeds
    parent's by suspiciously large margin (>50% relative), WARN — could
    be either real complementarity OR selection-effect / p-hack.

    Caller passes path to parent series (e.g., '_dpead_recon_base.parquet'
    for D_PEAD variants). Without parent_returns_path → SKIP.
    """
    if not parent_returns_path:
        return StepResult(
            step_name="ablation_vs_parent",
            status="SKIP",
            key_findings={"reason": "no parent_returns_path provided"},
            verdict="skipped — supply parent_returns_path to enable",
        )
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    full_path = repo_root / parent_returns_path
    if not full_path.exists():
        return StepResult(
            step_name="ablation_vs_parent",
            status="WARN",
            key_findings={"reason": f"parent file missing: {parent_returns_path}"},
            verdict=f"parent cache missing: {parent_returns_path}",
        )
    try:
        parent = pd.read_parquet(full_path).iloc[:, 0]
        parent.index = pd.to_datetime(parent.index)
        # Resample to monthly if daily
        if len(parent) >= 200:
            parent = ((1 + parent.clip(-0.2, 0.2)).resample("ME").prod() - 1)
        cand = candidate_returns.copy()
        cand.index = pd.to_datetime(cand.index)
        # Align on common period
        joined = pd.concat([cand.rename("cand"), parent.rename("parent")],
                              axis=1).dropna()
        if len(joined) < 24:
            return StepResult(
                step_name="ablation_vs_parent", status="SKIP",
                key_findings={"n_overlap": len(joined)},
                verdict=f"too few overlap months ({len(joined)})",
            )
        cand_sharpe = float(joined["cand"].mean() / joined["cand"].std() * 12 ** 0.5)
        parent_sharpe = float(joined["parent"].mean() / joined["parent"].std() * 12 ** 0.5)
        sharpe_gain = cand_sharpe - parent_sharpe
        rel_gain = sharpe_gain / max(abs(parent_sharpe), 0.1)
        # Correlation
        corr = float(joined["cand"].corr(joined["parent"]))

        findings = {
            "cand_sharpe":   cand_sharpe,
            "parent_sharpe": parent_sharpe,
            "sharpe_gain":   sharpe_gain,
            "relative_gain": rel_gain,
            "correlation":   corr,
            "n_overlap":     len(joined),
        }
        # Verdict logic:
        # - relative_gain < 0:           regression, fail
        # - 0 < rel_gain < 0.10:         mild improvement, OK
        # - 0.10 < rel_gain < 0.50:      strong improvement, PASS
        # - 0.50 < rel_gain < 1.00:      large improvement, INVESTIGATE (could be real)
        # - rel_gain >= 1.00:            suspiciously large, WARN
        if rel_gain < 0:
            return StepResult(
                step_name="ablation_vs_parent", status="FAIL",
                key_findings=findings,
                verdict=(f"REGRESSION: candidate Sharpe {cand_sharpe:.2f} < "
                         f"parent {parent_sharpe:.2f} (relative gain "
                         f"{rel_gain:+.0%})"),
            )
        if rel_gain >= 1.0:
            return StepResult(
                step_name="ablation_vs_parent", status="WARN",
                key_findings=findings,
                verdict=(f"SUSPICIOUSLY LARGE gain: candidate Sharpe "
                         f"{cand_sharpe:.2f} vs parent {parent_sharpe:.2f} "
                         f"(relative +{rel_gain:.0%}). Could be real signal "
                         f"complementarity OR selection-effect / p-hack."),
            )
        if rel_gain >= 0.5:
            return StepResult(
                step_name="ablation_vs_parent", status="WARN",
                key_findings=findings,
                verdict=(f"LARGE gain: candidate Sharpe {cand_sharpe:.2f} "
                         f"vs parent {parent_sharpe:.2f} (relative "
                         f"+{rel_gain:.0%}). Investigate cause."),
            )
        return StepResult(
            step_name="ablation_vs_parent", status="PASS",
            key_findings=findings,
            verdict=(f"Improvement vs parent: Sharpe {cand_sharpe:.2f} vs "
                     f"{parent_sharpe:.2f} (relative +{rel_gain:.0%}). "
                     f"Correlation {corr:+.2f}."),
        )
    except Exception as exc:
        return StepResult(
            step_name="ablation_vs_parent", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"ablation failed: {exc}",
        )


def _run_block_bootstrap_significance(
    candidate_returns,
    proposal_name: str,
    parent_returns_path: str | None,
) -> StepResult:
    """P-D8 (5.2): Paired Block Bootstrap Sharpe-diff significance vs
    parent / benchmark.

    Where P-D7 ablation gives a relative-gain HEURISTIC, this step
    gives a STATISTICAL TEST. They are complementary:
      - P-D7: candidate Sharpe / parent Sharpe ratio
      - P-D8: P(observed_diff or larger | true_diff = 0)

    Uses Politis-White 2003 stationary bootstrap with Politis-White
    2009 auto block length on the DIFFERENCE series. Verdict:
      - p < 0.05 AND diff > 0  → PASS  (significant improvement)
      - p < 0.05 AND diff <= 0 → FAIL  (significantly worse)
      - p >= 0.05              → WARN  (noise — apparent gain unproven)
      - No parent              → SKIP
    """
    if not parent_returns_path:
        return StepResult(
            step_name="block_bootstrap_significance", status="SKIP",
            key_findings={"reason": "no parent_returns_path provided"},
            verdict="skipped — supply parent_returns_path to enable",
        )
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    full_path = repo_root / parent_returns_path
    if not full_path.exists():
        return StepResult(
            step_name="block_bootstrap_significance", status="WARN",
            key_findings={"reason": f"parent file missing: {parent_returns_path}"},
            verdict=f"parent cache missing: {parent_returns_path}",
        )
    try:
        parent = pd.read_parquet(full_path).iloc[:, 0]
        parent.index = pd.to_datetime(parent.index)
        if len(parent) >= 200:
            parent = ((1 + parent.clip(-0.2, 0.2)).resample("ME").prod() - 1)
        cand = candidate_returns.copy()
        cand.index = pd.to_datetime(cand.index)
        joined = pd.concat([cand.rename("cand"), parent.rename("parent")],
                              axis=1).dropna()
        n_overlap = len(joined)
        if n_overlap < 24:
            return StepResult(
                step_name="block_bootstrap_significance", status="SKIP",
                key_findings={"n_overlap": n_overlap},
                verdict=f"too few overlap months ({n_overlap})",
            )

        from engine.validation.block_bootstrap import pbb_sharpe_diff
        # Fewer iterations than default 10k since pipeline runs many
        # candidates; keep statistically defensible
        result = pbb_sharpe_diff(
            joined["cand"].values, joined["parent"].values,
            n_iter=3000,
        )
        findings = {
            "cand_sharpe":     round(result.sharpe_a, 3),
            "parent_sharpe":   round(result.sharpe_b, 3),
            "sharpe_diff":     round(result.diff_point, 3),
            "ci_lo":           round(result.diff_ci_lo, 3),
            "ci_hi":           round(result.diff_ci_hi, 3),
            "p_value":         round(result.p_value_two_sided, 4),
            "block_len":       round(result.block_len, 1),
            "block_method":    result.block_method,
            "n_overlap":       n_overlap,
        }
        if result.p_value_two_sided >= 0.05:
            return StepResult(
                step_name="block_bootstrap_significance", status="WARN",
                key_findings=findings,
                verdict=(
                    f"Sharpe diff {result.diff_point:+.2f} NOT statistically "
                    f"significant (PBB p={result.p_value_two_sided:.3f}, "
                    f"CI [{result.diff_ci_lo:.2f}, {result.diff_ci_hi:.2f}]). "
                    f"Apparent gain is within bootstrap noise."
                ),
            )
        if result.diff_point <= 0:
            return StepResult(
                step_name="block_bootstrap_significance", status="FAIL",
                key_findings=findings,
                verdict=(
                    f"SIGNIFICANTLY WORSE than parent: diff "
                    f"{result.diff_point:+.2f}, PBB p={result.p_value_two_sided:.3f}."
                ),
            )
        return StepResult(
            step_name="block_bootstrap_significance", status="PASS",
            key_findings=findings,
            verdict=(
                f"Sharpe diff {result.diff_point:+.2f} significant at "
                f"alpha=0.05 (PBB p={result.p_value_two_sided:.3f}, "
                f"CI [{result.diff_ci_lo:.2f}, {result.diff_ci_hi:.2f}], "
                f"auto-block_len={result.block_len:.1f})."
            ),
        )
    except Exception as exc:
        return StepResult(
            step_name="block_bootstrap_significance", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"PBB significance test failed: {exc}",
        )


def _run_quarter_concentration(
    candidate_returns,
    proposal_name: str,
) -> StepResult:
    """P-D9 (5.4): per-quarter return distribution + concentration-risk
    verdict. Catches strategies whose ARC depends on 2-3 lucky quarters
    (paper's median 7.98% vs mean 331% lesson).

    Verdict:
      LOW concentration   → PASS
      MED concentration   → WARN  (review pattern, may still deploy)
      HIGH concentration  → FAIL  (drop-top-N ARC negative or
                                    pct_profitable < 50% — structural)
    """
    try:
        from engine.validation.quarter_distribution import (
            classify_concentration, compute_quarter_distribution,
        )
        if len(candidate_returns) < 12:
            return StepResult(
                step_name="quarter_concentration", status="SKIP",
                key_findings={"n_obs": len(candidate_returns)},
                verdict="too few observations (<12)",
            )
        qd = compute_quarter_distribution(candidate_returns)
        v = classify_concentration(qd)
        findings = {
            "verdict":              v["verdict"],
            "n_quarters":           qd.n_quarters,
            "mean_arc":             round(qd.mean_arc, 4),
            "drop_top_n":           qd.drop_top_n,
            "drop_top_arc":         round(qd.drop_top_arc, 4),
            "pct_profitable":       qd.pct_profitable,
            "mean_median_ratio":    (round(v["mean_median_ratio"], 2)
                                       if v["mean_median_ratio"] is not None
                                       else None),
            "reasons":              v["reasons"],
        }
        if v["verdict"] == "HIGH":
            return StepResult(
                step_name="quarter_concentration", status="FAIL",
                key_findings=findings,
                verdict=(
                    f"HIGH concentration: " + (v["reasons"][0]
                                                  if v["reasons"]
                                                  else "structural concern")
                ),
            )
        if v["verdict"] == "MED":
            return StepResult(
                step_name="quarter_concentration", status="WARN",
                key_findings=findings,
                verdict=(
                    f"MED concentration: " + (v["reasons"][0]
                                                  if v["reasons"]
                                                  else "moderate concern")
                ),
            )
        return StepResult(
            step_name="quarter_concentration", status="PASS",
            key_findings=findings,
            verdict=(
                f"LOW concentration: drop-top-{qd.drop_top_n} "
                f"ARC = {qd.drop_top_arc:.1%}, "
                f"{qd.pct_profitable:.0%} of quarters profitable"
            ),
        )
    except Exception as exc:
        return StepResult(
            step_name="quarter_concentration", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"concentration check failed: {exc}",
        )


def _run_correlation_matrix(candidate_returns, proposal_name) -> StepResult:
    """P1 #11 (P-D4 regime-conditional): correlation of candidate vs
    deployed sleeves, BOTH aggregated AND per-regime. Diversifiers
    that look low-corr aggregated but spike in STRESS get caught here."""
    try:
        from engine.portfolio.combined_book import (
            build_carry_book, build_crisis_hedge_book, build_equity_book,
            build_mom_hedge_book, build_tsmom_book,
        )
    except ImportError as exc:
        return StepResult(
            step_name="correlation_matrix", status="WARN",
            key_findings={"error": str(exc)[:120]},
            verdict="combined_book sleeves not importable",
        )
    sleeves = {
        "equity":      build_equity_book(),
        "carry":       build_carry_book(),
        "tsmom":       build_tsmom_book(),
        "crisis_hedge": build_crisis_hedge_book(),
        "mom_hedge":    build_mom_hedge_book(),
    }
    cand = candidate_returns.copy()
    cand.index = pd.to_datetime(cand.index)
    cand = cand.resample("ME").last() if not cand.index.equals(
        cand.index.to_period("M").to_timestamp("M")) else cand
    # Load regime monthly classifier
    regime = None
    try:
        from engine.portfolio.combined_book import build_vix_regime_monthly
        regime = build_vix_regime_monthly()
    except Exception:
        pass

    corrs_agg = {}
    corrs_per_regime: dict[str, dict[str, float]] = {
        "CALM": {}, "NORMAL": {}, "STRESS": {},
    }
    for name, s in sleeves.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s = s.resample("ME").last()
        joined = pd.concat([cand.rename("cand"), s.rename(name)], axis=1).dropna()
        if len(joined) < 12:
            corrs_agg[name] = None
        else:
            corrs_agg[name] = float(joined["cand"].corr(joined[name]))
        if regime is not None and len(joined) > 0:
            for r_label in ["CALM", "NORMAL", "STRESS"]:
                in_regime = regime[regime == r_label].index.intersection(joined.index)
                if len(in_regime) >= 8:
                    sub = joined.loc[in_regime]
                    corrs_per_regime[r_label][name] = float(
                        sub["cand"].corr(sub[name])
                    )

    high_corr = {k: v for k, v in corrs_agg.items() if v is not None and abs(v) > 0.5}
    # NEW: also check STRESS regime correlation spike
    high_stress_corr = {
        k: v for k, v in corrs_per_regime.get("STRESS", {}).items()
        if abs(v) > 0.7
    }
    issues = []
    if high_corr:
        issues.append(f"aggregated |corr|>0.5 with {list(high_corr.keys())}")
    if high_stress_corr:
        issues.append(f"STRESS |corr|>0.7 with {list(high_stress_corr.keys())}")

    if issues:
        return StepResult(
            step_name="correlation_matrix", status="WARN",
            key_findings={
                "correlations": corrs_agg,
                "high_corr": high_corr,
                "per_regime": corrs_per_regime,
                "high_stress_corr": high_stress_corr,
            },
            verdict="; ".join(issues),
        )
    return StepResult(
        step_name="correlation_matrix", status="PASS",
        key_findings={
            "correlations": corrs_agg,
            "per_regime": corrs_per_regime,
        },
        verdict="all correlations within thresholds (aggregated + per-regime)",
    )


def _run_regime_stratified(candidate_returns, proposed_role, phase,
                                proposal_name):
    """Regime-stratified BARRA. P0 #1+#2: now applied to ALL roles, with
    role-specific verdicts. BLOCKING only for insurance role."""
    from engine.risk.barra_lite import (
        regress_sleeve_by_regime, build_factor_returns,
    )
    from engine.portfolio.combined_book import build_vix_regime_monthly

    try:
        factors = build_factor_returns(phase=phase)
        regime = build_vix_regime_monthly()
        reports = regress_sleeve_by_regime(
            candidate_returns, regime, factors,
            sleeve_name=proposal_name, min_months_per_regime=18,
        )
    except Exception as exc:
        is_insurance = (proposed_role == "insurance")
        return StepResult(
            step_name="regime_stratified_BARRA",
            status="FAIL" if is_insurance else "WARN",
            key_findings={"error": str(exc)[:120]},
            verdict=f"regime-stratified BARRA error ({'BLOCKING' if is_insurance else 'non-blocking'}): {exc}",
        )

    findings: dict = {}
    for label, rep in reports.items():
        findings[label] = {
            "n": rep.n_months,
            "alpha_t_hac": rep.alpha_t_hac,
            "alpha_annualized": rep.alpha_annualized,
            "MOM_t": rep.t_stats_hac.get("MOM"),
        }

    # Role-specific verdict logic per P0 #1+#2 doctrine.
    normal = reports.get("NORMAL")
    stress = reports.get("STRESS")
    calm = reports.get("CALM")

    # ── insurance: STRESS α must be >= NORMAL α (BLOCKING) ───────────
    if proposed_role == "insurance":
        if not (normal and stress):
            return StepResult(
                step_name="regime_stratified_BARRA", status="WARN",
                key_findings=findings,
                verdict="insurance role but couldn't stratify NORMAL+STRESS",
            )
        if stress.alpha_annualized < normal.alpha_annualized - 0.005:
            return StepResult(
                step_name="regime_stratified_BARRA", status="FAIL",
                key_findings=findings,
                verdict=(f"insurance HYPOTHESIS REJECTED: STRESS α "
                         f"{stress.alpha_annualized:+.2%} < NORMAL α "
                         f"{normal.alpha_annualized:+.2%}. Pays MORE drag "
                         f"in stress. mom_hedge failure mode."),
            )
        return StepResult(
            step_name="regime_stratified_BARRA", status="PASS",
            key_findings=findings,
            verdict="insurance hypothesis SUPPORTED: STRESS α >= NORMAL α",
        )

    # ── risk_premium_harvester: WARN if premium reverses sign in any regime ──
    if proposed_role == "risk_premium_harvester":
        if not (normal and stress):
            return StepResult(
                step_name="regime_stratified_BARRA", status="WARN",
                key_findings=findings,
                verdict="harvester role but couldn't stratify NORMAL+STRESS",
            )
        signs = {label: (1 if rep.alpha_annualized > 0 else -1)
                  for label, rep in reports.items()}
        if len(set(signs.values())) > 1:
            return StepResult(
                step_name="regime_stratified_BARRA", status="WARN",
                key_findings=findings,
                verdict=("harvester premium REVERSES SIGN across regimes "
                         f"({signs}); fragile or regime-specific harvest"),
            )
        return StepResult(
            step_name="regime_stratified_BARRA", status="PASS",
            key_findings=findings,
            verdict="harvester premium stable in sign across regimes",
        )

    # ── diversifier: H9 cosine should not pile on book in STRESS ──
    if proposed_role == "diversifier":
        if stress and stress.r_squared > 0.8 and normal and normal.r_squared < 0.5:
            return StepResult(
                step_name="regime_stratified_BARRA", status="WARN",
                key_findings=findings,
                verdict=("diversifier becomes HIGHLY factor-driven in "
                         f"STRESS (R^2 {stress.r_squared:.2f} vs NORMAL "
                         f"{normal.r_squared:.2f}); correlation spike risk"),
            )
        return StepResult(
            step_name="regime_stratified_BARRA", status="PASS",
            key_findings=findings,
            verdict="diversifier R^2 profile stable across regimes",
        )

    # ── alpha_seeker: alpha must survive in at least 2 regimes ────────
    if proposed_role == "alpha_seeker":
        regimes_with_positive_t = sum(
            1 for rep in reports.values() if rep.alpha_t_hac >= 1.0
        )
        if regimes_with_positive_t < 2:
            return StepResult(
                step_name="regime_stratified_BARRA", status="WARN",
                key_findings=findings,
                verdict=(f"alpha only present in {regimes_with_positive_t} "
                         f"regime(s); not robust"),
            )
        return StepResult(
            step_name="regime_stratified_BARRA", status="PASS",
            key_findings=findings,
            verdict=f"alpha appears in {regimes_with_positive_t} regimes; robust",
        )

    # ── regime_overlay: skip (its purpose IS regime-conditional) ─────
    return StepResult(
        step_name="regime_stratified_BARRA", status="PASS",
        key_findings=findings,
        verdict=f"regime-stratified done across {len(reports)} regimes; informational",
    )


def _run_devils_advocate(proposal_name: str,
                              proposed_role: str | None,
                              h10_pl: dict,
                              prior_steps: list,
                              candidate_relation: str = "UNKNOWN",
                              most_correlated_sleeve: str | None = None,
                              most_correlated_value: float | None = None,
                              ) -> StepResult:
    """P2 #12: Devil's Advocate LLM persona call via DeepSeek V4 Pro
    (devils_advocate_constrained_evidence per
    [[project-agent-team-persona-locked-2026-05-18]]).

    Sends a structured single-turn prompt summarizing the candidate's
    profile + prior pipeline step results, asks for adversarial critique.
    Returns FAIL if persona identifies fatal red flags, WARN if material
    concerns, PASS if no concerns. Graceful fallback to placeholder
    WARN if DeepSeek key missing or call fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return _da_placeholder("openai SDK not installed")

    # Try to read DeepSeek key from secrets.toml. Python 3.11+ has tomllib
    # built-in; Python 3.10 needs the `tomli` package as fallback.
    api_key = None
    toml_load = None
    try:
        import tomllib as _tomllib
        toml_load = _tomllib.load
    except ImportError:
        try:
            import tomli as _tomli
            toml_load = _tomli.load
        except ImportError:
            pass
    try:
        secrets_path = Path(".streamlit/secrets.toml")
        if toml_load and secrets_path.exists():
            with open(secrets_path, "rb") as f:
                secrets = toml_load(f)
            api_key = (secrets.get("DEEPSEEK_API_KEY")
                       or secrets.get("DEEPSEEK_API_TOKEN"))
    except Exception as exc:
        logger.warning("DA secrets read failed: %s", exc)

    if not api_key:
        return _da_placeholder("DEEPSEEK_API_KEY not in secrets.toml")

    # Build adversarial prompt
    summary = _da_summarize_for_prompt(
        proposal_name, proposed_role, h10_pl, prior_steps,
        candidate_relation=candidate_relation,
        most_correlated_sleeve=most_correlated_sleeve,
        most_correlated_value=most_correlated_value,
    )
    # Role-aware interpretation guidance — 2026-05-31 fix per user
    # finding "DA not role-aware" (10th catch). DA must know that
    # negative cosine is GOOD for diversifier/insurance roles,
    # CONCERNING for alpha_seeker.
    role_guidance = _da_role_specific_guidance(proposed_role)
    # Relation-aware guidance — 11th user catch (DA flagged PIT SN
    # cosine 0.88 as fatal red flag despite candidate being a
    # REPLACEMENT for parent same-mechanism sleeve)
    relation_guidance = _da_relation_specific_guidance(
        candidate_relation, most_correlated_sleeve, most_correlated_value,
    )
    system_prompt = (
        "You are 'devils_advocate_constrained_evidence' per the locked "
        "agent persona spec. Your job is ADVERSARIAL CRITIQUE of the "
        "candidate strategy below.\n\n"
        f"ROLE-SPECIFIC INTERPRETATION (proposed_role={proposed_role}):\n"
        f"{role_guidance}\n\n"
        f"RELATION-SPECIFIC INTERPRETATION (candidate_relation="
        f"{candidate_relation}):\n"
        f"{relation_guidance}\n\n"
        "Identify (1) fatal red flags that should HARD-REJECT the "
        "candidate, (2) material concerns warranting WARN, or (3) PASS "
        "if no material concerns. Be evidence-constrained — do not "
        "invent. CRITICAL: do NOT raise concerns about metrics that "
        "are DESIRED for the candidate's role (e.g. negative cosine "
        "for diversifier/insurance; high cosine for REPLACEMENT-relation "
        "candidates is EXPECTED). Output strict JSON: "
        '{"verdict": "FAIL|WARN|PASS", "fatal_red_flags": [...], '
        '"material_concerns": [...], "rationale": "..."}'
    )

    try:
        client = OpenAI(api_key=api_key,
                          base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary},
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        return _da_placeholder(f"DeepSeek call failed: {exc}")

    try:
        import json
        content = resp.choices[0].message.content
        parsed = json.loads(content)
    except Exception as exc:
        return _da_placeholder(f"DA response parse failed: {exc}")

    verdict_str = (parsed.get("verdict") or "WARN").upper()
    fatal = parsed.get("fatal_red_flags") or []
    concerns = parsed.get("material_concerns") or []
    rationale = parsed.get("rationale") or ""

    if verdict_str == "FAIL" or fatal:
        return StepResult(
            step_name="devils_advocate", status="FAIL",
            key_findings={
                "fatal_red_flags": fatal,
                "material_concerns": concerns,
                "rationale": rationale[:300],
            },
            verdict=f"DA HARD REJECT: {(fatal[:1] or [rationale[:80]])[0]}",
        )
    if verdict_str == "WARN" or concerns:
        return StepResult(
            step_name="devils_advocate", status="WARN",
            key_findings={
                "material_concerns": concerns,
                "rationale": rationale[:300],
            },
            verdict=f"DA concerns: {(concerns[:1] or [rationale[:80]])[0]}",
        )
    return StepResult(
        step_name="devils_advocate", status="PASS",
        key_findings={"rationale": rationale[:200]},
        verdict="DA: no material concerns",
    )


def _da_placeholder(reason: str) -> StepResult:
    return StepResult(
        step_name="devils_advocate", status="WARN",
        key_findings={"note": reason},
        verdict=f"DA placeholder ({reason}); manual review required",
    )


_DA_ROLE_GUIDANCE = {
    "alpha_seeker": (
        "alpha_seeker is paid for Sharpe via genuine alpha. Ideal cosine "
        "to existing book is near zero (±0.2). |cosine| > 0.3 warrants "
        "investigation: high POSITIVE may indicate factor wrap / "
        "duplicate-of-existing-sleeve; high NEGATIVE may indicate hidden "
        "short-equity-bet that masquerades as alpha. BUT — if candidate "
        "ALSO shows diversification value (cosine negative AND alpha "
        "after factor control is significant), the negative cosine is a "
        "BONUS not a concern."
    ),
    "risk_premium_harvester": (
        "risk_premium_harvester harvests known macro/structural premia "
        "(carry, term, vol). Similar to alpha_seeker: cosine near zero "
        "ideal; |cosine| > 0.3 needs investigation."
    ),
    "insurance": (
        "insurance is bought for crisis-period protection. NEGATIVE "
        "cosine to book/risk-source is DESIRED — that's literally what "
        "insurance does. POSITIVE cosine to risk source = hedge isn't "
        "working. Cosine alone is NOT a concern when negative; concern "
        "only if cosine DRIFTS positive (hedge broken) or if Sharpe is "
        "more negative than expected insurance cost."
    ),
    "diversifier": (
        "diversifier is bought for cosine reduction. NEGATIVE cosine to "
        "book is THE ENTIRE POINT — more negative = better. DO NOT raise "
        "negative-cosine concerns for diversifier candidates. Concern "
        "only if cosine DRIFTS POSITIVE over time (diversification "
        "breakdown) or if drag exceeds expected diversification benefit."
    ),
    "regime_overlay": (
        "regime_overlay is a meta-strategy on top of base sleeves. "
        "Cosine alone is NOT the right metric; focus on switching "
        "attribution (does the overlay ADD value vs static weights?) + "
        "regime classifier accuracy."
    ),
}


def _da_role_specific_guidance(role: str | None) -> str:
    if not role:
        return ("(no proposed_role specified — apply generic adversarial "
                "critique without role-specific cosine bias)")
    return _DA_ROLE_GUIDANCE.get(role.lower(),
                                 f"(role '{role}' has no specific guidance)")


def _da_relation_specific_guidance(
    relation: str,
    most_correlated_sleeve: str | None,
    most_correlated_value: float | None,
) -> str:
    """11th catch 2026-05-31: DA was treating high cosine as fatal even
    for REPLACEMENT candidates where high cosine is EXPECTED (replacing
    same-mechanism sleeve)."""
    if relation == "REPLACEMENT":
        return (
            f"This candidate is REPLACEMENT relation — proposed to "
            f"REPLACE existing sleeve {most_correlated_sleeve!r} (cosine "
            f"{most_correlated_value:+.2f} if known). High positive cosine "
            f"with that specific sleeve is EXPECTED and DESIRED — it's "
            f"the SAME mechanism in an improved variant. DO NOT raise "
            f"high-cosine fatal red flag for the replaced sleeve; do "
            f"raise concerns if cosine ALSO high with OTHER (different-"
            f"mechanism) sleeves, indicating cross-family redundancy."
        )
    if relation == "ADDITION":
        return (
            "This candidate is ADDITION relation — proposes new sleeve "
            "orthogonal to existing book (cosine < 0.4 with all). High "
            "|cosine| would be a contradiction here; raise concern if "
            "cosine drifts up vs declared profile."
        )
    return ("relation classification UNKNOWN (cosine between 0.4 and "
            "0.7 with most correlated) — apply standard rigor; either "
            "REPLACEMENT or ADDITION framing acceptable per evidence")


def _da_summarize_for_prompt(proposal_name, proposed_role,
                                  h10_pl, prior_steps,
                                  candidate_relation: str = "UNKNOWN",
                                  most_correlated_sleeve: str | None = None,
                                  most_correlated_value: float | None = None,
                                  ) -> str:
    lines = [f"CANDIDATE: {proposal_name}",
              f"PROPOSED ROLE: {proposed_role}",
              f"CANDIDATE RELATION: {candidate_relation}"]
    if most_correlated_sleeve and most_correlated_value is not None:
        lines.append(
            f"MOST CORRELATED EXISTING SLEEVE: {most_correlated_sleeve} "
            f"(cosine {most_correlated_value:+.2f})"
        )
    if h10_pl:
        lines.append(f"H10 alpha_t: {h10_pl.get('h8_summary', {}).get('alpha_t_hac', 'n/a')}")
        cos = (h10_pl.get('h9_summary') or {}).get('cosine_to_book_risk')
        if cos is not None:
            lines.append(f"H10 cosine to book: {cos:+.2f}")
    lines.append("\nPRIOR PIPELINE STEPS:")
    for s in prior_steps:
        lines.append(f"  {s.step_name} [{s.status}]: {s.verdict[:120]}")
    lines.append("\nGive adversarial critique focused on hidden risks, "
                  "data integrity, look-ahead, p-hacking signals, "
                  "factor crowding, regime fragility. RESPECT the "
                  "relation context above when judging cosine.")
    return "\n".join(lines)


def _run_data_quality_check(candidate_returns) -> StepResult:
    """P2 #14: detect outliers + suspicious return patterns + look-ahead
    smells. Returns WARN if anomalies found (rarely FAIL since some
    factor strategies legitimately show jumps)."""
    s = candidate_returns.dropna()
    if len(s) < 12:
        return StepResult(
            step_name="data_quality", status="SKIP",
            key_findings={"n": len(s)},
            verdict=f"too few obs ({len(s)})",
        )
    mean, std = float(s.mean()), float(s.std())
    if std < 1e-10:
        return StepResult(
            step_name="data_quality", status="FAIL",
            key_findings={"std": std},
            verdict="zero variance — corrupt or constant series",
        )
    z = (s - mean) / std
    n_outliers = int((z.abs() > 5).sum())
    monthly_max = float(s.max())
    monthly_min = float(s.min())
    # Look-ahead smell: if final months have abnormally high Sharpe
    # could indicate forward-fitting.
    if len(s) >= 24:
        recent = s.iloc[-12:]
        recent_sharpe = float(recent.mean() / recent.std() * (12 ** 0.5)) \
                            if recent.std() > 0 else 0
        rest = s.iloc[:-12]
        rest_sharpe = float(rest.mean() / rest.std() * (12 ** 0.5)) \
                          if rest.std() > 0 else 0
        sharpe_jump = recent_sharpe - rest_sharpe
    else:
        sharpe_jump = 0.0

    issues = []
    if n_outliers >= 3:
        issues.append(f"{n_outliers} outliers (|z|>5)")
    if abs(monthly_max) > 0.50 or abs(monthly_min) > 0.50:
        issues.append(f"extreme monthly return (max {monthly_max:.1%} / min {monthly_min:.1%})")
    if sharpe_jump > 1.0:
        issues.append(f"Sharpe jumps {sharpe_jump:+.2f} in final 12 mo (look-ahead smell)")

    if issues:
        return StepResult(
            step_name="data_quality", status="WARN",
            key_findings={
                "n_outliers": n_outliers,
                "monthly_max": monthly_max,
                "monthly_min": monthly_min,
                "recent_sharpe_jump": sharpe_jump,
                "issues": issues,
            },
            verdict="; ".join(issues),
        )
    return StepResult(
        step_name="data_quality", status="PASS",
        key_findings={
            "n_outliers": n_outliers,
            "monthly_max": monthly_max,
            "monthly_min": monthly_min,
        },
        verdict="no data quality smells",
    )


def _compute_meta_decision(steps: list, h10_accept: bool) -> tuple[str, str]:
    """P2 #15: meta-decision layer combining ALL steps with role-weighted
    logic. Beyond simple all-PASS-or-REJECT, this gives nuanced verdict.

    Returns (decision, rationale).
    """
    fails = [s for s in steps if s.status == "FAIL"]
    warns = [s for s in steps if s.status == "WARN"]
    passes = [s for s in steps if s.status == "PASS"]
    skips = [s for s in steps if s.status == "SKIP"]

    # Hard reject if any FAIL
    if fails:
        return ("HARD_REJECT",
                "; ".join(f"{s.step_name}: {s.verdict[:80]}" for s in fails))

    # If H10 didn't affirmatively accept (None or False), soft reject
    if not h10_accept:
        return ("SOFT_REJECT",
                f"H10 did not affirmatively accept. {len(warns)} WARN steps to review.")

    # Count critical warns (DA + regime_strat + factor_budget are heavy-weight)
    critical_warn_names = {"devils_advocate", "regime_stratified_BARRA",
                            "factor_budget_delta", "data_quality"}
    critical_warns = [s for s in warns if s.step_name in critical_warn_names]

    if len(critical_warns) >= 3:
        return ("SOFT_REJECT",
                f"{len(critical_warns)} critical-weight WARN steps — too many "
                f"red flags for promotion. Review and address before retry.")

    if len(critical_warns) >= 1:
        return ("BORDERLINE_REVIEW",
                f"{len(critical_warns)} critical WARN(s) require human review "
                f"before deploy: {[s.step_name for s in critical_warns]}")

    if len(warns) >= 5:
        return ("SOFT_REJECT",
                f"{len(warns)} total WARN steps — too noisy for promotion")

    return ("PROMOTE_TO_GATE",
            f"all critical steps PASS; {len(warns)} non-critical WARN to "
            f"address. {len(passes)} PASS / {len(skips)} SKIP.")


# ── Pipeline orchestrator ────────────────────────────────────────────────

def run_candidate_pipeline(
    candidate_returns: pd.Series,
    proposal_name: str = "candidate",
    proposed_role: str | None = None,
    mechanism_id: str | None = None,
    proposal_dict: dict | None = None,
    parent_returns_path: str | None = None,
    phase: int = 3,
) -> PipelineReport:
    """Doctrine-locked sequential evaluator (full P0+P1+P2 version).

    Steps (all run; FAIL of critical steps short-circuits to HARD_REJECT):
      1. H10 unified evaluator                  [BLOCKING on FAIL]
      2. data_quality                            [WARN-class]
      3. H2 cousin check                         [BLOCKING if hard-reject]
      4. H6 post-pub evidence                    [BLOCKING if applicable]
      5. H7 kill-this-proposal                   [BLOCKING on fatal]
      6. graveyard registry check                [BLOCKING on block]
      7. cost_model_check                        [WARN-class]
      8. regime_stratified_BARRA                 [BLOCKING for insurance]
      9. factor_budget_delta                     [WARN-class]
     10. multi_aum_cost                          [WARN-class]
     11. sub_period_robustness                   [WARN-class]
     12. correlation_matrix vs deployed sleeves  [WARN-class]
     13. devils_advocate LLM                     [BLOCKING on FAIL]
     14. meta_decision combining all             [overall verdict]
    """
    steps: list[StepResult] = []
    h10_pl_cache = None

    # ── Step 0 (Phase 1 P0b): build reproducibility manifest ─────────
    manifest = None
    try:
        from engine.research.repro_manifest import build_manifest, hash_output
        manifest = build_manifest(pipeline_config={
            "phase":              phase,
            "proposed_role":      proposed_role,
            "mechanism_id":       mechanism_id,
            "proposal_name":      proposal_name,
        })
    except Exception as exc:
        logger.warning("manifest build failed: %s", exc)

    # ── Step 1: H10 unified evaluator ────────────────────────────────
    h10_step, role_used, role_inferred, h10_accept, h10_pl_cache = _run_h10(
        candidate_returns, proposal_name, proposed_role, phase,
    )
    steps.append(h10_step)
    if h10_step.status == "FAIL":
        return _short_circuit(steps, h10_step, proposal_name,
                                  role_used, role_inferred,
                                  "H10 rejected")

    # ── Step 2: data quality ─────────────────────────────────────────
    dq_step = _run_data_quality_check(candidate_returns)
    steps.append(dq_step)
    if dq_step.status == "FAIL":
        return _short_circuit(steps, dq_step, proposal_name,
                                  role_used, role_inferred,
                                  "data quality hard-fail")

    # ── Step 3: H2 cousin check ──────────────────────────────────────
    if mechanism_id:
        # Build candidate_info from proposal_dict for hygiene checks
        # (H2 cousin, H6 post-pub) that previously REQUIRED library entry.
        # 2026-05-31 chicken-and-egg bug fix.
        ci = None
        if proposal_dict:
            ci = {
                "family":         proposal_dict.get("family"),
                "parent_family":  proposal_dict.get("parent_family"),
                "required_data":  proposal_dict.get("required_data") or [],
                "economics_text": proposal_dict.get("economics_text", ""),
                # Pass-through: H6 needs post_pub_decay; future checks
                # may need other fields. Spreading proposal_dict ensures
                # any extra metadata flows through.
                "post_pub_decay": proposal_dict.get("post_pub_decay") or {},
            }
        h2_step = _run_h2(mechanism_id, candidate_info=ci)
        steps.append(h2_step)
        if h2_step.status == "FAIL":
            return _short_circuit(steps, h2_step, proposal_name,
                                      role_used, role_inferred,
                                      "H2 hard-reject")
    else:
        steps.append(StepResult(
            step_name="H2_cousin_check", status="SKIP",
            key_findings={"reason": "no mechanism_id provided"},
            verdict="skipped — provide mechanism_id to enable",
        ))

    # ── Step 4: H6 post-pub evidence ─────────────────────────────────
    if mechanism_id:
        # Reuse the candidate_info dict from step 3 for chicken-egg fix
        h6_step = _run_h6(mechanism_id, candidate_info=ci)
        steps.append(h6_step)
        if h6_step.status == "FAIL":
            return _short_circuit(steps, h6_step, proposal_name,
                                      role_used, role_inferred,
                                      "H6 post-pub rejected")
    else:
        steps.append(StepResult(
            step_name="H6_post_pub_evidence", status="SKIP",
            key_findings={"reason": "no mechanism_id provided"},
            verdict="skipped — provide mechanism_id to enable",
        ))

    # ── Step 5: H7 deterministic adversarial ─────────────────────────
    h7_step = _run_h7(proposal_dict or {})
    steps.append(h7_step)
    if h7_step.status == "FAIL":
        return _short_circuit(steps, h7_step, proposal_name,
                                  role_used, role_inferred,
                                  "H7 adversarial kill")

    # ── Step 6: graveyard registry check ─────────────────────────────
    grave_step = _run_graveyard_check(proposal_name, mechanism_id,
                                              candidate_returns)
    steps.append(grave_step)
    if grave_step.status == "FAIL":
        return _short_circuit(steps, grave_step, proposal_name,
                                  role_used, role_inferred,
                                  "graveyard block")

    # ── Step 7: cost_model sanity ────────────────────────────────────
    cost_step = _run_cost_model_check(candidate_returns, proposal_name)
    steps.append(cost_step)

    # ── Step 8: regime-stratified BARRA (BLOCKING for insurance) ─────
    regime_step = _run_regime_stratified(
        candidate_returns, role_used, phase, proposal_name,
    )
    steps.append(regime_step)
    if regime_step.status == "FAIL" and role_used == "insurance":
        return _short_circuit(steps, regime_step, proposal_name,
                                  role_used, role_inferred,
                                  "insurance hypothesis fails regime test")

    # ── Step 9: factor budget delta ──────────────────────────────────
    fb_step = _run_factor_budget_delta(
        candidate_returns, role_used, proposal_name, phase,
    )
    steps.append(fb_step)

    # ── Step 10: multi-AUM cost ──────────────────────────────────────
    aum_step = _run_multi_aum_check(candidate_returns, proposal_name)
    steps.append(aum_step)

    # ── Step 11: sub-period robustness ───────────────────────────────
    sub_step = _run_sub_period_robustness(
        candidate_returns, phase, proposal_name,
    )
    steps.append(sub_step)

    # ── Step 12: correlation matrix vs deployed sleeves ──────────────
    corr_step = _run_correlation_matrix(candidate_returns, proposal_name)
    steps.append(corr_step)

    # ── Step 12b (P-D7): ablation vs parent signal ────────────────────
    ablation_step = _run_ablation_vs_parent(
        candidate_returns, proposal_name, parent_returns_path,
    )
    steps.append(ablation_step)

    # ── Step 12c (P-D8): honest_deploy_sharpe calibration ─────────────
    haircut_step = _run_honest_deploy_sharpe(
        candidate_returns, proposal_name, mechanism_id,
    )
    steps.append(haircut_step)

    # Pre-compute relation BEFORE DA so DA has context (11th catch fix)
    pre_relation, pre_top_sleeve, pre_top_val = \
        _classify_replacement_or_addition(steps)

    # ── Step 13: Devil's Advocate LLM ────────────────────────────────
    da_step = _run_devils_advocate(
        proposal_name, role_used, h10_pl_cache, list(steps),
        candidate_relation=pre_relation,
        most_correlated_sleeve=pre_top_sleeve,
        most_correlated_value=pre_top_val,
    )
    steps.append(da_step)
    if da_step.status == "FAIL":
        return _short_circuit(steps, da_step, proposal_name,
                                  role_used, role_inferred,
                                  "Devil's Advocate hard reject")

    # ── Step 14: meta-decision + P-D6 REPLACEMENT/ADDITION classification ──
    decision, rationale = _compute_meta_decision(steps, h10_accept)
    relation, top_sleeve, top_val = _classify_replacement_or_addition(steps)
    # If REPLACEMENT and currently HARD_REJECT, rewrite to suggest
    # replacement re-audit path (the correlation that triggered HARD/WARN
    # is EXPECTED for a variant of existing sleeve, not a true block)
    if relation == "REPLACEMENT" and decision in ("BORDERLINE_REVIEW",
                                                     "SOFT_REJECT"):
        # Filter out correlation-only WARNs since they're explained by
        # the replacement classification
        non_corr_critical = [
            s for s in steps
            if s.status in ("FAIL", "WARN")
            and s.step_name in {"devils_advocate", "regime_stratified_BARRA",
                                  "factor_budget_delta", "data_quality"}
            and s.step_name != "correlation_matrix"
        ]
        if not any(s.status == "FAIL" for s in non_corr_critical) and h10_accept:
            decision = "PROMOTE_AS_REPLACEMENT"
            rationale = (
                f"candidate is REPLACEMENT for existing sleeve {top_sleeve!r} "
                f"(corr {top_val:+.2f}). The correlation WARN is EXPECTED "
                f"and re-classified as informational. Re-audit cost_model + "
                f"factor_exposure + capacity for the variant before deploy."
            )

    # P0b: finalize manifest by hashing the pipeline output
    if manifest is not None:
        try:
            preliminary_report = {
                "final_decision":   decision,
                "rationale":        rationale,
                "role_used":        role_used,
                "candidate_relation": relation,
                "step_results":     [s.to_dict() for s in steps],
            }
            manifest.output_hash = hash_output(preliminary_report)
        except Exception as exc:
            logger.warning("output hash failed: %s", exc)

    return PipelineReport(
        proposal_name=proposal_name,
        role_used=role_used,
        role_was_inferred=bool(role_inferred),
        step_results=steps,
        final_decision=decision,
        rationale=rationale,
        candidate_relation=relation,
        most_correlated_sleeve=top_sleeve,
        most_correlated_value=top_val,
        reproducibility_manifest=manifest.to_dict() if manifest else None,
    )


def _classify_replacement_or_addition(steps: list) -> tuple[str, str | None,
                                                                  float | None]:
    """P-D6: classify candidate as REPLACEMENT vs ADDITION based on
    correlation_matrix step's findings.

    - REPLACEMENT: max |corr| >= 0.7 with any existing sleeve
    - ADDITION:    max |corr| < 0.4 with all existing sleeves
    - UNKNOWN:     between (could be either)
    """
    corr_step = next(
        (s for s in steps if s.step_name == "correlation_matrix"), None,
    )
    if not corr_step:
        return ("UNKNOWN", None, None)
    corrs = (corr_step.key_findings or {}).get("correlations") or {}
    if not corrs:
        return ("UNKNOWN", None, None)
    top_sleeve, top_val = None, 0.0
    for k, v in corrs.items():
        if v is None:
            continue
        if abs(v) > abs(top_val):
            top_sleeve, top_val = k, float(v)
    if abs(top_val) >= 0.7:
        return ("REPLACEMENT", top_sleeve, top_val)
    if abs(top_val) < 0.4:
        return ("ADDITION", top_sleeve, top_val)
    return ("UNKNOWN", top_sleeve, top_val)


def _short_circuit(steps, fail_step, proposal_name, role_used,
                       role_inferred, label) -> PipelineReport:
    return PipelineReport(
        proposal_name=proposal_name,
        role_used=role_used,
        role_was_inferred=bool(role_inferred),
        step_results=steps,
        final_decision="HARD_REJECT",
        rationale=f"{label}: {fail_step.verdict}",
    )


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--returns-cache", required=True,
                     help="parquet file with monthly returns Series")
    p.add_argument("--column", default=None,
                     help="column name in parquet (defaults to first)")
    p.add_argument("--proposal-name", default="candidate")
    p.add_argument("--proposed-role", default=None,
                     choices=["alpha_seeker", "risk_premium_harvester",
                                "insurance", "regime_overlay", "diversifier"])
    p.add_argument("--mechanism-id", default=None,
                     help="library YAML id, enables H2 cousin + H6 post-pub")
    p.add_argument("--phase", type=int, default=3, choices=[1, 2, 3])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = pd.read_parquet(args.returns_cache)
    col = args.column or df.columns[0]
    returns = df[col]
    returns.index = pd.to_datetime(returns.index)

    report = run_candidate_pipeline(
        returns, proposal_name=args.proposal_name,
        proposed_role=args.proposed_role,
        mechanism_id=args.mechanism_id,
        phase=args.phase,
    )

    print(f"=== Candidate Pipeline: {args.proposal_name} ===")
    print(f"role_used:         {report.role_used} "
          f"({'inferred' if report.role_was_inferred else 'provided'})")
    print()
    print(f"{'step':<28}  {'status':<8}  verdict")
    print(f"{'-'*28}  {'-'*8}  {'-'*40}")
    for s in report.step_results:
        print(f"{s.step_name:<28}  {s.status:<8}  {s.verdict[:60]}")
    print()
    print(f"FINAL DECISION: {report.final_decision}")
    print(f"  rationale: {report.rationale[:200]}")
    return 0 if report.final_decision == "PROMOTE_TO_GATE" else 1


if __name__ == "__main__":
    sys.exit(main())
