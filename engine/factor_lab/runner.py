"""
engine/factor_lab/runner.py — Single-factor BHY test wrapper.

Spec: docs/spec_factor_lab.md §3.5 (decision logic) + §5 (ship gate)
Boundary invariant: zero LLM imports — pure deterministic test execution.

Wraps engine.b_plus_search.run_single_strategy_weekly so the lab UI can
trigger a single-factor weekly backtest, derive a verdict from the
Newey-West t-statistic, and atomically transition the SpecRegistry row's
lab_state with a corresponding amendment_log entry.

Verdict thresholds (per spec §3.5)
----------------------------------
  PASS               nw_t_stat ≥ 1.96    (raw two-sided 5% sig)
  MARGINAL           1.65 ≤ nw_t < 1.96  (10% sig, sub-5%)
  FAIL               nw_t < 1.65 and achieved power ≥ 0.50
  FAIL_UNDERPOWERED  nw_t < 1.65 and achieved power < 0.50
                     (would re-block at register-time check)

These thresholds are statistical convention values, not magic numbers
amenable to tuning. Changing them requires amend_spec(threshold_tweak)
on docs/spec_factor_lab.md.

What this module does NOT do
----------------------------
- BHY FDR over a candidate batch — that lives in b_plus_search.run_mass_search
  for true multiple-testing pre-registered runs. Single-factor lab tests
  are individually pre-registered (one spec per candidate), so per-test
  FDR correction is not needed; the EFFECTIVE_N_TRIALS denominator already
  carries the project-wide multiple-testing burden via SpecRegistry.
- β-neutral confirmation (ship gate condition #2) — deferred to
  comparison view, post-thesis polish per spec §10.
- Auto-ship to PRODUCTION_SIGNAL — ALWAYS goes through PendingApproval
  governance, never direct config edit.
"""
from __future__ import annotations

import datetime
import logging
import math
import os
from typing import Any, Optional

from engine.factor_lab.types import FactorState
from engine.factor_lab.power import achieved_power_at_n
from engine.factor_lab import registry as _registry

logger = logging.getLogger(__name__)


# Verdict thresholds — locked per spec §3.5
_T_PASS_5PCT  = 1.96    # raw 5% two-sided
_T_MARGINAL   = 1.65    # raw 10% two-sided
_POWER_FLOOR  = 0.50    # below this → underpowered FAIL


def _classify_verdict(
    nw_t_stat:        float,
    achieved_power:   float,
) -> FactorState:
    """Map (NW t-stat, achieved power) → terminal FactorState.

    Power gate matters only on FAIL: a high-t result is PASS regardless
    of underpowered status (we have stronger-than-expected evidence).
    """
    if not math.isfinite(nw_t_stat):
        return FactorState.FAIL
    if abs(nw_t_stat) >= _T_PASS_5PCT:
        return FactorState.PASS
    if abs(nw_t_stat) >= _T_MARGINAL:
        return FactorState.MARGINAL
    if achieved_power < _POWER_FLOOR:
        return FactorState.FAIL_UNDERPOWERED
    return FactorState.FAIL


def run_factor_lab_test(
    *,
    spec_id:               int,
    strategy_id:           str,
    start_date:            str = "2010-01-01",
    end_date:              Optional[str] = None,
    universe_tier:         int   = 1,
    expected_sharpe_lift:  float = 0.50,
    baseline_sharpe:       float = 1.00,
    decisions_dir:         str = "docs/decisions",
) -> dict[str, Any]:
    """Run single-factor BHY test for a registered candidate.

    Workflow (atomic per-step state transitions):
        REGISTERED → TESTING                 (entry)
        TESTING    → PASS|MARGINAL|FAIL|FAIL_UNDERPOWERED   (verdict)

    Args:
        spec_id: SpecRegistry.id of a candidate in REGISTERED state.
        strategy_id: identifier of a StrategySpec already implemented in
            engine.b_plus_search.STRATEGY_REGISTRY. Lab MVP requires the
            user to register the strategy code separately — this is by
            design (no LLM-generated signal_fn execution).
        start_date / end_date: backtest window (ISO date strings). End
            defaults to today.
        universe_tier: 1 (35 ETF) or 2 (45 ETF).
        expected_sharpe_lift / baseline_sharpe: used to compute achieved
            power for the FAIL_UNDERPOWERED disambiguation. Should match
            the values the candidate was registered with.
        decisions_dir: where to write the verdict markdown.

    Returns:
        {
            'spec_id':       int,
            'verdict':       FactorState value,
            'nw_t_stat':     float,
            'sharpe':        float,
            'n_obs':         int,
            'achieved_power': float,
            'decision_path': str,    # path of the verdict markdown
            'raw':           dict,   # full b_plus_search result
        }

    Raises:
        LookupError: spec_id not found.
        ValueError: spec not in REGISTERED state, or strategy_id not in registry.
        engine.factor_lab.IllegalTransition: state machine violation.
    """
    # ── Validate candidate is in REGISTERED state ───────────────────────────
    candidate = _registry.get_candidate(spec_id)
    if candidate is None:
        raise LookupError(f"SpecRegistry id={spec_id} not found")
    if candidate["lab_state"] != FactorState.REGISTERED.value:
        raise ValueError(
            f"spec_id={spec_id} state={candidate['lab_state']!r} — "
            f"BHY test requires state=REGISTERED. Run power_check at "
            f"register time first."
        )

    # ── Defense in depth: spec_hash drift check ────────────────────────────
    # If the spec markdown was silently edited between register and run,
    # Tier R rule_spec_hash_vs_code_drift catches it on next sweep — but
    # we add a runner-time check too because BHY runs are computationally
    # expensive (~minutes) and cited in verdicts. No point burning compute
    # on a candidate whose hash chain is broken.
    from engine.preregistration import _compute_git_blob_hash, _resolve_to_abs
    try:
        live_hash = _compute_git_blob_hash(_resolve_to_abs(candidate["spec_path"]))
    except Exception as exc:
        raise ValueError(
            f"Cannot read spec markdown {candidate['spec_path']!r}: {exc}. "
            f"Refuse to run BHY without an intact pre-registration anchor."
        )
    if live_hash != candidate["current_hash"]:
        raise ValueError(
            f"spec_hash drift detected for {candidate['spec_path']!r}: "
            f"stored={candidate['current_hash'][:16]} vs live={live_hash[:16]}. "
            f"Spec markdown changed without amend_spec — HARKing R1 violation. "
            f"Run `python -m engine.preregistration amend_spec` to formalize "
            f"the change before re-running."
        )

    # ── Look up StrategySpec ────────────────────────────────────────────────
    from engine.b_plus_search import (
        get_strategy, get_universe_tier, run_single_strategy_weekly,
    )
    try:
        strategy = get_strategy(strategy_id)
    except KeyError:
        raise ValueError(
            f"strategy_id={strategy_id!r} not in b_plus_search.STRATEGY_REGISTRY. "
            f"Implement signal_fn + register before running lab test."
        )

    universe = get_universe_tier(universe_tier)
    end_iso  = end_date or datetime.date.today().isoformat()

    # ── Transition REGISTERED → TESTING (atomic, with amendment_log) ───────
    _registry.transition_state(
        spec_id    = spec_id,
        new_state  = FactorState.TESTING,
        reason     = (
            f"Run BHY test: strategy_id={strategy_id}, "
            f"window=[{start_date}, {end_iso}], universe_tier={universe_tier}"
        ),
        actor      = "factor_lab.runner",
    )

    # ── Run single-factor weekly backtest ───────────────────────────────────
    try:
        raw = run_single_strategy_weekly(
            spec       = strategy,
            universe   = universe,
            start_date = start_date,
            end_date   = end_iso,
        )
    except Exception as exc:
        # Execution failure: write a FAIL verdict with execution_error reason
        # rather than leaving the row stuck in TESTING.
        _registry.transition_state(
            spec_id    = spec_id,
            new_state  = FactorState.FAIL,
            reason     = f"execution_error: {type(exc).__name__}: {exc}",
            actor      = "factor_lab.runner",
        )
        logger.error("factor_lab runner: spec_id=%d strategy=%s exec failed: %s",
                     spec_id, strategy_id, exc)
        raise

    # ── Classify verdict ────────────────────────────────────────────────────
    nw_t   = float(raw.get("nw_t_stat") or 0.0)
    sharpe = float(raw.get("sharpe")    or 0.0)
    n_obs  = int  (raw.get("n_obs")     or 0)

    achieved = achieved_power_at_n(
        expected_sharpe_lift  = expected_sharpe_lift,
        baseline_sharpe       = baseline_sharpe,
        n_available           = max(n_obs, 1),
        observations_per_year = 52,   # b_plus_search runs weekly
    )
    verdict = _classify_verdict(nw_t_stat=nw_t, achieved_power=achieved)

    # ── Transition TESTING → terminal verdict ───────────────────────────────
    _registry.transition_state(
        spec_id    = spec_id,
        new_state  = verdict,
        reason     = (
            f"BHY single-factor: nw_t={nw_t:+.3f}, sharpe={sharpe:+.3f}, "
            f"n_obs={n_obs}, achieved_power={achieved:.2f}. "
            f"Threshold pass={_T_PASS_5PCT} marginal={_T_MARGINAL} "
            f"power_floor={_POWER_FLOOR}."
        ),
        actor      = "factor_lab.runner",
    )

    # ── Write verdict markdown to decisions/ ────────────────────────────────
    spec_short = candidate["spec_path"].replace("docs/spec_", "").replace(".md", "")
    today      = datetime.date.today().isoformat()
    md_path    = os.path.join(decisions_dir, f"lab_{spec_short}_{today}.md")
    md_text    = _render_verdict_md(
        spec_path  = candidate["spec_path"],
        spec_hash  = candidate["current_hash"],
        strategy_id = strategy_id,
        verdict    = verdict,
        nw_t       = nw_t,
        sharpe     = sharpe,
        n_obs      = n_obs,
        achieved   = achieved,
        start_date = start_date,
        end_date   = end_iso,
        universe_tier = universe_tier,
        expected_sharpe_lift = expected_sharpe_lift,
        baseline_sharpe      = baseline_sharpe,
    )
    try:
        os.makedirs(decisions_dir, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(md_text)
    except Exception as exc:
        logger.warning("factor_lab runner: failed to write verdict md: %s", exc)
        md_path = ""

    return {
        "spec_id":        spec_id,
        "verdict":        verdict.value,
        "nw_t_stat":      nw_t,
        "sharpe":         sharpe,
        "n_obs":          n_obs,
        "achieved_power": achieved,
        "decision_path":  md_path,
        "raw":            {k: v for k, v in raw.items()
                           if k not in ("weekly_returns", "cum_nav",
                                        "signal_history")},  # truncate big DFs
    }


def _render_verdict_md(
    *,
    spec_path:            str,
    spec_hash:            str,
    strategy_id:          str,
    verdict:              FactorState,
    nw_t:                 float,
    sharpe:               float,
    n_obs:                int,
    achieved:             float,
    start_date:           str,
    end_date:             str,
    universe_tier:        int,
    expected_sharpe_lift: float,
    baseline_sharpe:      float,
) -> str:
    """Render the verdict markdown to docs/decisions/lab_<id>_<date>.md.

    Schema mirrors existing decision evidence docs: header + one-line
    verdict + parameter / result tables. No prose padding; this is an
    audit-trail document, not a research narrative. Authors revise the
    spec markdown for narrative; verdicts here are immutable.
    """
    return (
        f"# Factor Lab Verdict — {strategy_id} ({verdict.value})\n\n"
        f"**Spec**: [{spec_path}](../{spec_path})\n"
        f"**Spec hash**: `{spec_hash[:16]}`\n"
        f"**Verdict**: **{verdict.value}**\n"
        f"**Generated by**: `engine.factor_lab.runner.run_factor_lab_test` "
        f"on {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
        f"## Run parameters\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Strategy id | `{strategy_id}` |\n"
        f"| Window | {start_date} → {end_date} |\n"
        f"| Universe tier | {universe_tier} |\n"
        f"| Frequency | weekly (b_plus_search.run_single_strategy_weekly) |\n\n"
        f"## Result\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| NW t-stat | {nw_t:+.3f} |\n"
        f"| Sharpe (annualized) | {sharpe:+.3f} |\n"
        f"| n_obs (weekly) | {n_obs} |\n"
        f"| Achieved power at n | {achieved:.2f} |\n\n"
        f"## Power audit\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| Expected Sharpe lift (annualized) | {expected_sharpe_lift:+.2f} |\n"
        f"| Baseline Sharpe (annualized) | {baseline_sharpe:+.2f} |\n"
        f"| Achieved power | {achieved:.2f} |\n"
        f"| Power floor for FAIL_UNDERPOWERED | {_POWER_FLOOR} |\n\n"
        f"## Verdict thresholds (locked per docs/spec_factor_lab.md §3.5)\n\n"
        f"| Threshold | Value |\n|---|---|\n"
        f"| PASS (raw 5% two-sided) | nw_t ≥ {_T_PASS_5PCT} |\n"
        f"| MARGINAL (raw 10%) | {_T_MARGINAL} ≤ nw_t < {_T_PASS_5PCT} |\n"
        f"| FAIL_UNDERPOWERED | nw_t < {_T_MARGINAL} and achieved_power < {_POWER_FLOOR} |\n\n"
        f"## Ship gate (next step if PASS / MARGINAL)\n\n"
        f"Per docs/spec_factor_lab.md §5, PASS verdict does NOT auto-swap "
        f"production_signal. Required manual gates before swap:\n\n"
        f"1. ≥10y external literature support OR raw 5% sig + BHY-FDR pass at batch level\n"
        f"2. β-neutral confirmation (Carhart 4-factor decomposition)\n"
        f"3. Tier 1 audit clean (47 PASS / 0 FAIL)\n"
        f"4. Pre-registered spec_hash + amendment ledger complete\n\n"
        f"All four gated through PendingApproval (`approval_type='production_signal_swap'`).\n"
    )
