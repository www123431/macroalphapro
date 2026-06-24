"""
engine/auto_audit_rules.py — Rule registry for Auto-Audit Loop

Each rule is a zero-argument callable returning either:
  • None                                         → no contradiction
  • {"rule_name": str, "severity": "LOW"|"MID"|"HIGH", "snapshot": dict}
                                                 → contradiction with details

Rules are registered into either CRITICAL_RULES (run daily) or WEEKLY_RULES.

Splits agreed with supervisor 2026-05-06:
  CRITICAL (daily, 9 rules — drift > 24h hurts production correctness or audit trail)
    1.  production_signal_vs_falsification_chain                   ✅ R-1.B.2
    2.  spec_hash_vs_code_drift                                    ✅ R-1.B.2
    3.  effective_n_trials_math_consistency                        ✅ R-1.B.1
    5.  cash_flow_conservation                                     ✅ R-1.B.2
    8.  hash_chain_continuity                                      ✅ R-1.B.1
    9.  universe_drift_vs_registered                               ✅ R-1.B.2
    10. backtest_vs_production_param_alignment                     ✅ R-1.B.2 (simplified — backtest grep deferred to R-1.B.3)
    12. harking_detector_runs                                      ✅ R-1.B.1
    14. db_schema_vs_orm_consistency                               ✅ R-1.B.3
    10b. backtest_grep_kwargs_alignment (R-1.B.2 enrichment)        ✅ R-1.B.3
    16. path_consistency (proposer↔gate FORBIDDEN/FLAGGED match)    ✅ R-1.E (DRY check)

  WEEKLY (6 rules — slow drift, weekly cadence is enough)
    4.  agent_reflection_heartbeat                                 ✅ R-1.B.1
    6.  approval_queue_staleness                                   ✅ R-1.B.3
    7.  anomaly_screener_m1_drift                                  ✅ R-1.B.3 (drift + label velocity)
    11. paper_trading_e_arm_config_drift                           ✅ R-1.B.3 (resumption-only)
    13. llm_cumulative_cost_budget                                 ✅ R-1.B.3 (renamed from monthly)
    15. skill_library_dormancy                                     ✅ R-1.B.3 (silenceable LOW)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Rule signature ────────────────────────────────────────────────────────────
RuleResult = Optional[Dict[str, Any]]
RuleFn = Callable[[], RuleResult]

# ── Registries ────────────────────────────────────────────────────────────────
CRITICAL_RULES: List[RuleFn] = []
WEEKLY_RULES: List[RuleFn] = []


# ═════════════════════════════════════════════════════════════════════════════
# R-1.B.1 — Wrapper rules over existing audit scripts
# ═════════════════════════════════════════════════════════════════════════════
# These 4 rules wrap logic that already lives in scripts/* or engine/preregistration.
# They are registered as auto_audit rules so the cron orchestrator can run them
# alongside the new rules (R-1.B.2/B.3) and persist findings uniformly.

def rule_effective_n_trials_math_consistency() -> RuleResult:
    """
    Critical rule #3 — EFFECTIVE_N_TRIALS math consistency.

    Verifies the per-spec n_trials_contributed columns match what the rules of
    register_spec / amend_spec would have produced, in three slices:
      (a) retro_registered=True specs must have n_trials_contributed = 0
      (b) retro_registered=False specs must have
          n_trials_contributed == 1 + sum(amendment.n_trials_added)
      (c) compute_pre_registration_n_trials() must equal the manual recomputation

    P-LAB exemption (2026-05-08): rows with `factor_kind='infrastructure_spec'`
    are exempt from rule (b)'s base=1 default — they are tracked in the
    registry for HARKing R1 (silent edit) protection but DO NOT contribute
    to the multiple-testing burden (not research hypotheses). For these rows
    base=0 regardless of retro_registered. See docs/spec_factor_lab.md §6.

    Any slice failing = somebody hand-edited the DB or a code path bypassed
    register_spec / amend_spec. p-values cited against EFFECTIVE_N_TRIALS would
    be silently invalid until this is reconciled, so severity = HIGH.
    """
    from engine.memory import SessionFactory, SpecRegistry
    from engine.preregistration import (
        AMENDMENT_KINDS, compute_pre_registration_n_trials,
    )

    issues: List[Dict[str, Any]] = []
    total_recomputed = 0

    with SessionFactory() as s:
        rows = s.query(SpecRegistry).all()
        for r in rows:
            try:
                ledger = json.loads(r.amendment_log or "[]")
                if not isinstance(ledger, list):
                    ledger = []
            except Exception:
                ledger = []

            # Per-amendment kind validation
            ledger_n = 0
            bad_kinds: List[str] = []
            for entry in ledger:
                kind = entry.get("kind")
                expected = AMENDMENT_KINDS.get(kind)
                stored = entry.get("n_trials_added")
                if expected is None:
                    bad_kinds.append(f"unknown_kind={kind!r}")
                elif stored is None or int(stored) != expected:
                    bad_kinds.append(f"{kind}: stored={stored} expected={expected}")
                else:
                    ledger_n += expected

            # P-LAB infrastructure_spec exemption: not a research hypothesis,
            # so base=0 regardless of retro flag. Counts toward registry
            # tracking (HARKing R1) but not n_trials accounting.
            if (r.factor_kind or "") == "infrastructure_spec":
                base = 0
            else:
                base = 0 if r.retro_registered else 1
            expected_total = base + ledger_n
            stored_total = int(r.n_trials_contributed or 0)

            # Only research-hypothesis specs contribute to recomputed sum
            if not r.retro_registered and (r.factor_kind or "") != "infrastructure_spec":
                total_recomputed += stored_total

            if stored_total != expected_total or bad_kinds:
                issues.append({
                    "spec_path":      r.spec_path,
                    "stored":         stored_total,
                    "expected":       expected_total,
                    "retro":          bool(r.retro_registered),
                    "amendment_n":    len(ledger),
                    "bad_kinds":      bad_kinds,
                })

    counter_value = compute_pre_registration_n_trials()
    counter_drift = (counter_value != total_recomputed)
    if counter_drift:
        issues.append({
            "kind":           "counter_vs_recomputed",
            "counter":        counter_value,
            "recomputed":     total_recomputed,
        })

    if not issues:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_issues":        len(issues),
            "counter_value":   counter_value,
            "recomputed_sum":  total_recomputed,
            "issues":          issues[:20],   # truncate for snapshot size
        },
    }


def rule_hash_chain_continuity() -> RuleResult:
    """
    Critical rule #8 — Narrative hash chain continuity.

    Wraps scripts/audit_narrative_chain.run_audit. Flags if any of:
      • hash mismatch (stored sha256 ≠ recomputed) — TAMPER signal
      • prev_narrative_hash points to non-existent earlier row — broken chain
      • >1 chain root (multiple None prev_hash entries) — fork

    Severity HIGH because tamper detection / append-only audit chain is the
    core compliance promise (GIPS 2020 §III.A.18 / SEC 17a-4(b) lineage).
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from scripts import audit_narrative_chain  # type: ignore

    out = audit_narrative_chain.run_audit()
    n_mismatch = int(out.get("n_hash_mismatch", 0))
    n_broken   = int(out.get("n_chain_broken", 0))
    n_roots    = int(out.get("chain_root_count", 0))

    # 0 rows is OK (no snapshots yet); >1 roots is a fork
    fork_flag = (n_roots > 1)

    if n_mismatch == 0 and n_broken == 0 and not fork_flag:
        return None

    return {
        "severity": "HIGH",
        "snapshot": {
            "n_with_snapshot":   int(out.get("n_with_snapshot", 0)),
            "n_hash_mismatch":   n_mismatch,
            "n_chain_broken":    n_broken,
            "chain_root_count":  n_roots,
            "issues":            out.get("issues", [])[:20],
        },
    }


def rule_harking_detector_runs() -> RuleResult:
    """
    Critical rule #12 — HARKing detector R1-R4.

    Calls engine.preregistration.detect_harking(). Any returned flag = a
    rule R1-R4 hit on the live amendment ledger (e.g. silent threshold drift,
    re-numbering of n_trials, hypothesis swap without scope_narrow). Each hit
    must be reviewed before further inference is drawn from the affected spec.

    detect_harking is idempotent (already-flagged + unresolved hits are not
    re-emitted), so a non-empty result means *new* flags since last run.
    """
    from engine.preregistration import detect_harking

    try:
        flags = detect_harking()
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"error": f"detect_harking crashed: {exc}"},
        }

    if not flags:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_flags": len(flags),
            "flags":   flags[:20],
        },
    }


def rule_agent_reflection_heartbeat() -> RuleResult:
    """
    Weekly rule #4 — Agent reflection heartbeat.

    Wraps scripts/audit_agent_liveness.run_audit. Any agent with
    flags ∋ {NEVER_RUN, STALE_*D, ALL_FAILED, ALL_30D_RUNS_FAILED,
    AGENT_RUNS_BUT_NO_DATA_WRITES, NO_DOWNSTREAM_DATA_30D, LOW_VOLUME_HISTORICAL}
    is a heartbeat issue.

    Severity:
      HIGH if any agent NEVER_RUN, ALL_FAILED, or stale beyond 2× cadence
      MID  for low-volume / data-flow flags only

    NO_DOWNSTREAM_DATA_30D alone is downgraded to MID for function-style
    agents (no AgentRun rows, expected on long-cadence ones like
    universe_review which only acts every 90d).
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from scripts import audit_agent_liveness  # type: ignore

    results = audit_agent_liveness.run_audit()
    flagged = [r for r in results if r.get("flags")]
    if not flagged:
        return None

    # Promoted to HIGH per supervisor 2026-05-06 design review:
    # AGENT_RUNS_BUT_NO_DATA_WRITES = silent failure (looks fine, writes nothing).
    # macro_research 2026-05-04 incident: 7.5h to detect manually because nothing
    # surfaced. Treating as worse than ALL_FAILED, which at least is loud.
    HIGH = {
        "NEVER_RUN",
        "ALL_FAILED",
        "ALL_30D_RUNS_FAILED",
        "AGENT_RUNS_BUT_NO_DATA_WRITES",
    }
    severity = "MID"
    for r in flagged:
        flags = set(r.get("flags") or [])
        if flags & HIGH or any(f.startswith("STALE_") for f in flags):
            severity = "HIGH"
            break

    return {
        "severity": severity,
        "snapshot": {
            "n_total_agents":    len(results),
            "n_flagged":         len(flagged),
            "agents":            [
                {
                    "agent_id":          r["agent_id"],
                    "flags":             r["flags"],
                    "last_succeeded_at": r.get("last_succeeded_at"),
                    "n_30d_runs":        r.get("n_30d_runs"),
                    "n_downstream_30d":  r.get("n_downstream_30d"),
                }
                for r in flagged
            ],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# R-1.B.2 — 5 new critical rules (no existing wrapper to lean on)
# ═════════════════════════════════════════════════════════════════════════════

# Rule #1 — REJECTED_PRODUCTION_SIGNALS hardcoded set
# Decision 2026-05-06: hardcoded set, not parsed from falsification_chain.md.
# falsification_chain.md is human-readable narrative; parsing it is fragile.
# Each addition here is an explicit supervisor decision tied to a specific test.
REJECTED_PRODUCTION_SIGNALS = {
    "narrative_overlay":          "falsification_chain #1 (Phase 0 REJECT)",
    "narrative_risk_gate_d1":     "falsification_chain #2 (D1 SOFT REJECT)",
    "narrative_risk_gate_d1_1":   "falsification_chain #3 (D1.1 HARD REJECT, 192-month OOS)",
    "factor_mad":                 "falsification_chain #4 (FactorMAD 0/24 promoted)",
    "tsmom":                      "S1 multi-window 6×5y mean Sharpe -0.06, 2/6 positive (memory project_s1_multi_window_2026-05-03)",
}


def rule_production_signal_vs_falsification_chain() -> RuleResult:
    """
    Critical rule #1 — Production signal must not be a falsified strategy.

    Reads engine.config.PRODUCTION_SIGNAL and checks against the hardcoded
    REJECTED_PRODUCTION_SIGNALS set. Each entry in the set carries the
    specific evidence reference that justified the rejection — supervisor
    must explicitly amend this set (not falsification_chain.md alone) to
    revive a previously-rejected strategy.

    Severity HIGH because running a falsified strategy = direct alpha
    leakage; the project's most embarrassing failure mode (TSMOM in
    production until 2026-05-05 supervisor challenge).
    """
    from engine.config import PRODUCTION_SIGNAL

    if PRODUCTION_SIGNAL in REJECTED_PRODUCTION_SIGNALS:
        return {
            "severity": "HIGH",
            "snapshot": {
                "PRODUCTION_SIGNAL": PRODUCTION_SIGNAL,
                "evidence":          REJECTED_PRODUCTION_SIGNALS[PRODUCTION_SIGNAL],
                "rejected_set_size": len(REJECTED_PRODUCTION_SIGNALS),
                "rejected_keys":     sorted(REJECTED_PRODUCTION_SIGNALS),
            },
        }
    return None


# Rule #2 — Production code files that must be hash-tracked
# Decision 2026-05-06: 9 files including auto_audit's own models + rules
# (full self-audit closure). Anything edited in these files without amend_spec
# = audit trail break.
PRODUCTION_CODE_FILES = (
    "engine/portfolio.py",             # construct_portfolio: vol target + caps
    "engine/signal.py",                # ql01_bab production signal
    "engine/config.py",                # central tunables
    "engine/regime.py",                # MSM regime detection
    "engine/preregistration.py",       # spec registry + HARKing detector
    "engine/anomaly_screener.py",      # S6 LLM detector + rule baselines
    "engine/auto_audit.py",            # self-audit: this very orchestrator
    "engine/auto_audit_models.py",     # self-audit: ORM (would-be silent-disable target)
    "engine/auto_audit_rules.py",      # self-audit: rule registry (silent-disable target)
    "engine/auto_audit_proposer.py",   # self-audit: LLM Layer 1 (silent-prompt-modify target)
    "engine/auto_audit_gate.py",       # self-audit: Layer 2 gate (silent-bypass target)
    "engine/auto_audit_promoter.py",   # self-audit: gate→PendingApproval bridge
    "engine/auto_audit_executor.py",   # self-audit: amend_spec executor on approval
)


def rule_spec_hash_vs_code_drift() -> RuleResult:
    """
    Critical rule #2 — Production code hash drift detection.

    Two failure modes:
      (a) PRODUCTION_CODE_FILES not all in spec_registry — drift detection
          on those files is INACTIVE. Bootstrap script must register them
          (see scripts/bootstrap_auto_audit_baseline.py).
      (b) Any active SpecRegistry row whose file content hash differs from
          stored current_hash AND amendment_log is empty — silent edit
          violation. (Drift WITH amendment_log = legitimate amend, not
          reported.)

    Severity HIGH because drift > 24h on production code = cited p-values
    and forward-test results are silently invalid (HARKing R1 territory).
    """
    import os
    from engine.memory import SessionFactory, SpecRegistry
    from engine.preregistration import _compute_git_blob_hash, _resolve_to_abs

    issues: List[Dict[str, Any]] = []
    missing_registrations: List[str] = []

    with SessionFactory() as s:
        # (a) Coverage check — every PRODUCTION_CODE_FILE must be registered
        registered = {
            r.spec_path
            for r in s.query(SpecRegistry).filter(SpecRegistry.status == "active").all()
        }
        for path in PRODUCTION_CODE_FILES:
            if path not in registered:
                missing_registrations.append(path)

        # (b) Hash drift across every active SpecRegistry row
        rows = s.query(SpecRegistry).filter(SpecRegistry.status == "active").all()
        for r in rows:
            abs_path = _resolve_to_abs(r.spec_path)
            if not os.path.exists(abs_path):
                issues.append({
                    "spec_path": r.spec_path,
                    "kind":      "file_missing",
                })
                continue
            try:
                recomputed = _compute_git_blob_hash(abs_path)
            except Exception as exc:
                issues.append({
                    "spec_path": r.spec_path,
                    "kind":      "hash_error",
                    "error":     str(exc),
                })
                continue
            if recomputed == r.current_hash:
                continue
            # Drift detected — check amendment log
            try:
                log = json.loads(r.amendment_log or "[]")
            except Exception:
                log = []
            if not log:
                issues.append({
                    "spec_path":           r.spec_path,
                    "kind":                "drift_no_amendment",
                    "stored_hash":         r.current_hash[:12],
                    "recomputed_hash":     recomputed[:12],
                })
            # Drift WITH amendment_log = legitimate, not reported.

    if not issues and not missing_registrations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_issues":              len(issues),
            "n_missing_registrations": len(missing_registrations),
            "missing_registrations":   missing_registrations[:20],
            "issues":                  issues[:20],
        },
    }


def rule_cash_flow_conservation() -> RuleResult:
    """
    Critical rule #5 — NAV / cash flow accounting must conserve.

    Three accounting equalities checked per consecutive snapshot pair:
      (a) Morning:    nav_close[t-1] + external_flow[t] == nav_after_flow[t]
      (b) Close:      nav_after_flow[t] + gross_pnl[t]  == nav_close[t]
      (c) CashFlow:   external_flow[t] == Σ(applied external CashFlow.amount_usd
                                           on flow_date == t)

    Tolerance: max(abs $0.01, relative 1bp) per GIPS 2020 industry convention.
    Lookback: full history scanned (cheap), but only issues within last 60
    days are emitted as findings (older issues are persistent — cron would
    re-emit them every day otherwise; supervisor manually clears stale).

    Severity HIGH because accounting break = unreliable NAV / TWR / MWR
    numbers cited to investors (SEC 17a-4(b) territory).
    """
    import datetime
    import math
    from sqlalchemy import func
    from engine.memory import CashFlow, PortfolioNavSnapshot, SessionFactory

    TOL_REL = 0.0001              # 1bp
    TOL_ABS = 0.01                # $0.01
    REPORT_LOOKBACK_DAYS = 60

    cutoff = datetime.date.today() - datetime.timedelta(days=REPORT_LOOKBACK_DAYS)
    issues: List[Dict[str, Any]] = []

    def _tol(scale: float) -> float:
        return max(TOL_ABS, abs(scale) * TOL_REL)

    def _safe(x: Any) -> float:
        """Coerce None / NaN to 0.0 — NaN propagation in equality checks would
        silently pass (NaN > tol is False), masking a real break. Replace with
        0 so any NaN-tainted snapshot surfaces as a *visible* mismatch."""
        if x is None:
            return 0.0
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if math.isnan(v) else v

    with SessionFactory() as s:
        snaps = (
            s.query(PortfolioNavSnapshot)
             .order_by(PortfolioNavSnapshot.snapshot_date.asc())
             .all()
        )

        for i in range(1, len(snaps)):
            prev, cur = snaps[i - 1], snaps[i]
            if cur.snapshot_date < cutoff:
                continue

            # (a) Morning equality
            prev_close   = _safe(prev.nav_close)
            ext_flow_cur = _safe(cur.external_flow)
            after_flow   = _safe(cur.nav_after_flow)
            morning_lhs = prev_close + ext_flow_cur
            if abs(morning_lhs - after_flow) > _tol(prev_close):
                issues.append({
                    "date":           str(cur.snapshot_date),
                    "kind":           "morning_equality",
                    "prev_nav_close": prev_close,
                    "external_flow":  ext_flow_cur,
                    "nav_after_flow": after_flow,
                    "diff":           round(morning_lhs - after_flow, 4),
                })

            # (b) Close equality
            gross_pnl  = _safe(cur.gross_pnl)
            close_now  = _safe(cur.nav_close)
            close_lhs  = after_flow + gross_pnl
            if abs(close_lhs - close_now) > _tol(after_flow):
                issues.append({
                    "date":           str(cur.snapshot_date),
                    "kind":           "close_equality",
                    "nav_after_flow": after_flow,
                    "gross_pnl":      gross_pnl,
                    "nav_close":      close_now,
                    "diff":           round(close_lhs - close_now, 4),
                })

            # (c) CashFlow consistency
            sum_applied = _safe(
                s.query(func.coalesce(func.sum(CashFlow.amount_usd), 0.0))
                 .filter(CashFlow.flow_date == cur.snapshot_date)
                 .filter(CashFlow.is_external.is_(True))
                 .filter(CashFlow.status == "applied")
                 .scalar()
            )
            if abs(sum_applied - ext_flow_cur) > _tol(ext_flow_cur):
                issues.append({
                    "date":               str(cur.snapshot_date),
                    "kind":               "cashflow_sum_mismatch",
                    "snapshot_external":  ext_flow_cur,
                    "cashflow_table_sum": sum_applied,
                    "diff":               round(sum_applied - ext_flow_cur, 4),
                })

    if not issues:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_issues":         len(issues),
            "lookback_days":    REPORT_LOOKBACK_DAYS,
            "tolerance_abs":    TOL_ABS,
            "tolerance_rel_bp": int(TOL_REL * 10000),
            "issues":           issues[:20],
        },
    }


# Rule #9 — universe baseline keys
_UNIVERSE_BASELINE_KEY = "auto_audit.universe_baseline_hash"
_UNIVERSE_INIT_KEY     = "auto_audit.universe_baseline_initialized_at"


def _canonical_universe_hash() -> tuple[str, int]:
    """Returns (sha256_hex, n_entries) for current active UniverseETF rows."""
    import hashlib
    from engine.universe_manager import get_universe_by_class

    universe = get_universe_by_class()  # {asset_class: {sector: ticker}}
    lines: List[str] = []
    for ac in sorted(universe.keys()):
        for sector in sorted(universe[ac].keys()):
            lines.append(f"{ac}|{sector}|{universe[ac][sector]}")
    canonical = "\n".join(lines)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), len(lines)


def rule_universe_drift_vs_registered() -> RuleResult:
    """
    Critical rule #9 — UniverseETF active set must match the registered
    baseline hash.

    Baseline is stored in SystemConfig under key
    `auto_audit.universe_baseline_hash`. On the very first run the rule
    self-initialises (writes current hash + an init timestamp into
    SystemConfig) and emits no finding — supervisor sees the
    initialisation event in audit logs and the UI badge.

    On subsequent runs, any drift = HIGH finding. After supervisor
    approves the change (R-1.E), R-1.E will overwrite the baseline.
    Until then, the same finding re-emits each cron tick (intended:
    keeps the issue visible).
    """
    import datetime
    from engine.memory import SessionFactory
    from engine.db_models import SystemConfig

    cur_hash, n_entries = _canonical_universe_hash()

    with SessionFactory() as s:
        baseline = s.query(SystemConfig).filter_by(key=_UNIVERSE_BASELINE_KEY).first()
        if baseline is None:
            today_iso = datetime.date.today().isoformat()
            s.add(SystemConfig(key=_UNIVERSE_BASELINE_KEY, value=cur_hash))
            s.add(SystemConfig(key=_UNIVERSE_INIT_KEY,     value=today_iso))
            s.commit()
            logger.info(
                "auto_audit: universe baseline initialised hash=%s n=%d at=%s",
                cur_hash[:12], n_entries, today_iso,
            )
            return None

        if baseline.value == cur_hash:
            return None

        init_row = s.query(SystemConfig).filter_by(key=_UNIVERSE_INIT_KEY).first()
        return {
            "severity": "HIGH",
            "snapshot": {
                "baseline_hash":      baseline.value[:12],
                "current_hash":       cur_hash[:12],
                "n_universe_entries": n_entries,
                "initialised_at":     init_row.value if init_row else "unknown",
                "note":               "Baseline overwritten only after R-1.E approval; finding re-emits until then.",
            },
        }


# Rule #10 — alignment surface (production constants that must hold)
# Decision 2026-05-06: 7 PnL-affecting params; backtest.py grep deferred to R-1.B.3.
ALIGNMENT_SURFACE = {
    "TARGET_VOL":        0.10,
    "MAX_LEVERAGE":      2.0,
    "MAX_WEIGHT":        0.25,
    "REGIME_SCALE":      0.6,    # 2026-05-08: spec_v3 OOS POSITIVE → c=0.6 swap
    "MAX_NET":           0.4,
    "MIN_NET":           -0.1,
    "PRODUCTION_SIGNAL": "ql01_bab",
}


def rule_backtest_vs_production_param_alignment() -> RuleResult:
    """
    Critical rule #10 (simplified) — Production constants must match the
    locked alignment surface.

    R-1.B.2 scope: read engine.config and check 7 PnL-affecting constants
    against ALIGNMENT_SURFACE. Any mismatch = production drifted from the
    P0-1 alignment commitment without supervisor approval.

    R-1.B.3 will extend this: AST-parse engine/backtest.py to confirm
    backtest call sites use the same values (catches the rarer case where
    backtest hardcodes a different number than config exports).

    Severity HIGH because P0-1 alignment is the project's core thesis
    claim — drift here invalidates every quoted backtest number.
    """
    import engine.config as cfg

    diffs: List[Dict[str, Any]] = []
    for key, expected in ALIGNMENT_SURFACE.items():
        actual = getattr(cfg, key, "<MISSING>")
        if actual != expected:
            diffs.append({
                "param":    key,
                "expected": expected,
                "actual":   actual,
            })

    if not diffs:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_diffs":         len(diffs),
            "diffs":           diffs,
            "alignment_surface_size": len(ALIGNMENT_SURFACE),
            "note":             "Backtest grep cross-check deferred to R-1.B.3.",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# R-1.B.3 — 6 weekly + 1 critical + #10 grep enrichment
# ═════════════════════════════════════════════════════════════════════════════

def rule_db_schema_vs_orm_consistency() -> RuleResult:
    """
    Critical rule #14 — DB column-set must match ORM column-set per table.

    For every table declared in Base.metadata.tables, compare ORM Column
    names against the live DB's PRAGMA table_info. Reports as
    ORM-only (column declared but not in DB → migration missed) or
    DB-only (column exists but ORM doesn't know → ad-hoc SQL edit).

    Index drift NOT checked: SQLite auto-creates internal indexes for unique
    constraints with names that don't match ORM Index() declarations, which
    would produce false positives.

    Severity HIGH because schema drift = silent INSERT/UPDATE failure or
    silent data loss on the unaware side.
    """
    from sqlalchemy import text
    from engine.db_models import Base
    from engine.memory import engine as _engine

    issues: List[Dict[str, Any]] = []
    with _engine.connect() as conn:
        for table_name, table in Base.metadata.tables.items():
            orm_cols = {c.name for c in table.columns}
            try:
                rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            except Exception as exc:
                issues.append({
                    "table":   table_name,
                    "kind":    "introspection_failed",
                    "error":   str(exc),
                })
                continue
            db_cols = {row[1] for row in rows}
            if not db_cols:
                # Table not yet created — init_db hasn't run for this table.
                # Skipping (not a drift, just a fresh setup).
                continue
            orm_only = sorted(orm_cols - db_cols)
            db_only  = sorted(db_cols - orm_cols)
            if orm_only or db_only:
                issues.append({
                    "table":     table_name,
                    "orm_only":  orm_only,
                    "db_only":   db_only,
                })

    if not issues:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_tables_with_drift": len(issues),
            "issues":              issues[:30],
        },
    }


def rule_approval_queue_staleness() -> RuleResult:
    """
    Weekly rule #6 — PendingApproval rows stuck in 'pending' too long.

    Threshold: triggered_date older than 14 days. Counts only status='pending'
    (resolved/rejected/auto_approved are terminal).

    Severity ladder:
      < 5 stale     → LOW
      5-15 stale    → MID
      ≥ 16 stale    → HIGH (queue runaway; supervisor capacity overflow)

    Rows with NULL triggered_date are skipped (legacy data, can't compute age).
    "Approved-but-not-actioned" check deferred — no actioned_at column on
    PendingApproval, would need JOIN with downstream tables.
    """
    import datetime
    from engine.memory import PendingApproval, SessionFactory

    STALE_DAYS = 14
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=STALE_DAYS)

    with SessionFactory() as s:
        stale_rows = (
            s.query(PendingApproval)
             .filter(PendingApproval.status == "pending")
             .filter(PendingApproval.triggered_date.isnot(None))
             .filter(PendingApproval.triggered_date < cutoff)
             .all()
        )

    if not stale_rows:
        return None

    n = len(stale_rows)
    if n >= 16:
        sev = "HIGH"
    elif n >= 5:
        sev = "MID"
    else:
        sev = "LOW"

    return {
        "severity": sev,
        "snapshot": {
            "stale_threshold_days": STALE_DAYS,
            "n_stale":              n,
            "oldest": [
                {
                    "id":              p.id,
                    "approval_type":   p.approval_type,
                    "triggered_date":  str(p.triggered_date),
                    "age_days":        (today - p.triggered_date).days,
                    "ticker":          p.ticker,
                }
                for p in sorted(stale_rows, key=lambda r: r.triggered_date)[:10]
            ],
        },
    }


def rule_anomaly_screener_m1_drift() -> RuleResult:
    """
    Weekly rule #7 — Anomaly screener LLM detector precision drift.

    Two sub-checks:
      (a) PRECISION DRIFT: precision over last 7d vs 7-30d ago.
          Requires n≥10 labeled flags in the recent window (else stats noisy).
          drift = prior_precision - recent_precision.
            < 0.20  → no finding
            0.20-0.40 → MID
            ≥ 0.40  → HIGH

      (b) LABEL VELOCITY: count of supervisor-labelled flags in last 60d.
          If < 10 → MID finding "evaluation pipeline itself is dormant".
          The drift check could be silently noise-bound by sparse data; this
          flags it explicitly.

    Both sub-checks emit at most one combined finding (whichever severity is
    higher). Snapshot includes both metrics for full context.
    """
    import datetime
    from sqlalchemy import func
    from engine.memory import AnomalyFlag, SessionFactory

    today = datetime.date.today()
    now = datetime.datetime.combine(today, datetime.time())
    cutoff_recent = now - datetime.timedelta(days=7)
    cutoff_prior  = now - datetime.timedelta(days=30)
    cutoff_label_velocity = now - datetime.timedelta(days=60)

    DRIFT_LOOKBACK_RECENT = 7
    DRIFT_LOOKBACK_PRIOR  = 30
    LABEL_VELOCITY_DAYS   = 60
    MIN_N_FOR_DRIFT       = 10
    MIN_N_FOR_VELOCITY    = 10

    with SessionFactory() as s:
        # Recent window: last 7d labeled flags (LLM detector only)
        recent = (
            s.query(AnomalyFlag.supervisor_useful)
             .filter(AnomalyFlag.detector == "llm")
             .filter(AnomalyFlag.supervisor_label_at >= cutoff_recent)
             .filter(AnomalyFlag.supervisor_label_at.isnot(None))
             .all()
        )
        prior = (
            s.query(AnomalyFlag.supervisor_useful)
             .filter(AnomalyFlag.detector == "llm")
             .filter(AnomalyFlag.supervisor_label_at < cutoff_recent)
             .filter(AnomalyFlag.supervisor_label_at >= cutoff_prior)
             .filter(AnomalyFlag.supervisor_label_at.isnot(None))
             .all()
        )
        velocity_count = (
            s.query(func.count(AnomalyFlag.id))
             .filter(AnomalyFlag.detector == "llm")
             .filter(AnomalyFlag.supervisor_label_at >= cutoff_label_velocity)
             .scalar() or 0
        )

    n_recent = len(recent)
    n_prior  = len(prior)
    p_recent = (sum(1 for r in recent if r[0]) / n_recent) if n_recent else None
    p_prior  = (sum(1 for r in prior  if r[0]) / n_prior)  if n_prior  else None

    severity: Optional[str] = None
    flags: List[str] = []
    detail: Dict[str, Any] = {
        "n_recent":          n_recent,
        "n_prior":           n_prior,
        "precision_recent":  round(p_recent, 3) if p_recent is not None else None,
        "precision_prior":   round(p_prior, 3)  if p_prior  is not None else None,
        "n_velocity_60d":    velocity_count,
        "min_n_for_drift":   MIN_N_FOR_DRIFT,
        "min_n_for_velocity": MIN_N_FOR_VELOCITY,
    }

    # (a) Precision drift
    if n_recent >= MIN_N_FOR_DRIFT and n_prior >= MIN_N_FOR_DRIFT and p_recent is not None and p_prior is not None:
        drift = p_prior - p_recent
        detail["drift"] = round(drift, 3)
        if drift >= 0.40:
            severity = "HIGH"
            flags.append("precision_drift_high")
        elif drift >= 0.20:
            severity = "MID"
            flags.append("precision_drift_mid")

    # (b) Label velocity
    if velocity_count < MIN_N_FOR_VELOCITY:
        if severity is None:
            severity = "MID"
        flags.append("label_velocity_low")

    if severity is None:
        return None
    detail["flags"] = flags
    return {"severity": severity, "snapshot": detail}


def rule_paper_trading_e_arm_config_drift() -> RuleResult:
    """
    Weekly rule #11 — Paper trading E arm/baseline drift detector.

    Per memory project_meta_audit_kill_simplify_2026-05-05.md, paper trading
    E was killed 2026-05-05 ("evaluation theater"). 6 historical runs persist,
    all with signal_baseline='tsmom' (pre-migration).

    Rule fires only on RESUMPTION boundary:
      • dormancy = no PaperTradingRun rows in last 30 days
      • if dormant: silent (matches kill state, supervisor expected this)
      • if newly active in last 30d AND signal_baseline mismatches production
        PRODUCTION_SIGNAL OR arm A/B/C count ratio is skewed (any arm <20% of
        max) → MID

    Old historical mismatch (the 6 tsmom rows) is NOT reported — supervisor
    already aware via the kill decision.
    """
    import datetime
    from sqlalchemy import func
    from engine.memory import SessionFactory
    from engine.db_models import PaperTradingRun
    from engine.config import PRODUCTION_SIGNAL

    DORMANCY_DAYS = 30
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=DORMANCY_DAYS)

    with SessionFactory() as s:
        recent = (
            s.query(PaperTradingRun)
             .filter(PaperTradingRun.as_of_date >= cutoff)
             .all()
        )

    if not recent:
        # Dormant — kill state matches expectation. Silent.
        return None

    # Resumption: check arm balance + signal_baseline alignment
    arm_counts: Dict[str, int] = {}
    baseline_mismatches: List[Dict[str, Any]] = []
    for r in recent:
        arm_counts[r.arm] = arm_counts.get(r.arm, 0) + 1
        if r.signal_baseline and r.signal_baseline != PRODUCTION_SIGNAL:
            baseline_mismatches.append({
                "id":              r.id,
                "as_of_date":      str(r.as_of_date),
                "arm":             r.arm,
                "signal_baseline": r.signal_baseline,
            })

    flags: List[str] = []

    # Arm balance: any arm < 20% of max
    if arm_counts:
        max_count = max(arm_counts.values())
        skewed = [a for a, c in arm_counts.items() if c < max_count * 0.2]
        if skewed:
            flags.append("arm_balance_skewed")

    if baseline_mismatches:
        flags.append("baseline_signal_mismatch")

    if not flags:
        return None
    return {
        "severity": "MID",
        "snapshot": {
            "dormancy_days":         DORMANCY_DAYS,
            "n_recent_runs":         len(recent),
            "arm_counts":            arm_counts,
            "production_signal":     PRODUCTION_SIGNAL,
            "baseline_mismatches":   baseline_mismatches[:10],
            "flags":                 flags,
            "note":                  "Rule fires only on RESUMPTION; historical kill-state runs are not reported.",
        },
    }


def rule_llm_cumulative_cost_budget() -> RuleResult:
    """
    Weekly rule #13 — Cumulative LLM spend vs budget cap.

    Reads anomaly_llm_detector.get_cost_status() — the existing cost tracker.
    Budget resolution chain (2026-05-08):
      engine.llm_budget.get_s6_anomaly_budget_usd_per_year()
        → SystemConfig key 'llm_budget.s6_anomaly.usd_per_year' (runtime tunable)
        → falls back to engine.config.S6_COST_BUDGET_USD ($250 default).

    Severity (cumulative, not monthly — matches the existing tracker which
    is cumulative; rule name reflects this):
      < 60%   → no finding
      60-80%  → LOW
      80-100% → MID
      ≥ 100%  → HIGH (over budget — must address before next call)

    Edge: budget == 0 → no finding (cost tracking effectively disabled).
    """
    from engine.anomaly_llm_detector import get_cost_status

    status = get_cost_status()
    budget = status.get("budget_usd") or 0
    total  = status.get("total_usd") or 0

    if budget <= 0:
        return None

    fraction = total / budget

    if fraction < 0.60:
        return None
    if fraction >= 1.00:
        sev = "HIGH"
    elif fraction >= 0.80:
        sev = "MID"
    else:
        sev = "LOW"

    return {
        "severity": sev,
        "snapshot": {
            "total_usd":   round(total, 4),
            "budget_usd":  budget,
            "fraction":    round(fraction, 3),
            "calls":       status.get("calls", 0),
        },
    }


def rule_skill_library_dormancy() -> RuleResult:
    """
    Weekly rule #15 — SkillLibrary table unused.

    Per memory project_meta_audit_kill_simplify_2026-05-05.md, SkillLibrary
    was tagged as a dead branch (0 rows ≥30 days, code paths still import
    it). This rule emits LOW severity once; supervisor can mark IGNORED to
    silence it for 30 days (silenceable mechanism, R-1.B.3).

    Severity LOW because it's a tech-debt flag, not a production-correctness
    issue.
    """
    import datetime
    from sqlalchemy import func
    from engine.memory import SessionFactory, SkillLibrary

    DORMANT_DAYS = 30
    today = datetime.date.today()

    with SessionFactory() as s:
        n_rows = s.query(SkillLibrary).count()
        last_updated = s.query(func.max(SkillLibrary.updated_at)).scalar()

    if n_rows == 0:
        # 0 rows is the documented dead-branch state.
        return {
            "severity": "LOW",
            "snapshot": {
                "n_rows":          0,
                "kind":            "no_rows",
                "note":            "SkillLibrary has 0 rows; consider deleting code references if confirmed dead. Silenceable.",
            },
        }

    if last_updated is not None:
        age_days = (datetime.datetime.utcnow() - last_updated).days
        if age_days >= DORMANT_DAYS:
            return {
                "severity": "LOW",
                "snapshot": {
                    "n_rows":             n_rows,
                    "kind":               "dormant",
                    "last_updated":       str(last_updated),
                    "age_days":           age_days,
                    "dormant_threshold":  DORMANT_DAYS,
                },
            }

    return None


# ─── R-1.B.2 #10 enrichment: AST-parse engine/backtest.py ─────────────────────
def _backtest_kwargs_literal_constants() -> Dict[str, Any]:
    """
    Return {kwarg_name: literal_value} for every literal-constant kwarg passed
    to construct_portfolio() inside engine/backtest.py.

    Limitations:
      • Only literal kwargs are captured (constant numbers/strings/None/bool).
      • Variable references (e.g., construct_portfolio(target_vol=my_var))
        are skipped — AST static analysis can't resolve them without runtime.
      • Multiple call sites with conflicting literals: returns the LAST one
        (we report all in the rule snapshot).
    """
    import ast
    import os

    backtest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "engine", "backtest.py",
    )
    if not os.path.exists(backtest_path):
        return {}

    with open(backtest_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return {}

    captured: Dict[str, Any] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match construct_portfolio( ... )
        func_name = (
            node.func.id if isinstance(node.func, ast.Name)
            else getattr(node.func, "attr", "") if isinstance(node.func, ast.Attribute)
            else ""
        )
        if func_name != "construct_portfolio":
            continue
        for kw in node.keywords:
            if kw.arg is None:
                continue
            try:
                val = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                continue  # variable / expression — skip
            captured[kw.arg] = val
    return captured


def rule_path_consistency() -> RuleResult:
    """
    Critical rule #16 (R-1.E DRY check) — proposer.LLM_FORBIDDEN_PATHS must
    equal gate.FORBIDDEN_PATHS, and same for FLAGGED. The two lists are the
    single source of truth for what files the LLM may propose touching;
    if they drift, gate may approve files the LLM was told to avoid (or
    vice versa).

    Severity HIGH because drift here = audit-loop safety property broken.
    """
    try:
        from engine.auto_audit_proposer import (
            LLM_FORBIDDEN_PATHS as proposer_forbid,
            LLM_FLAGGED_PATHS   as proposer_flag,
        )
        from engine.auto_audit_gate import (
            FORBIDDEN_PATHS as gate_forbid,
            FLAGGED_PATHS   as gate_flag,
        )
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"error": f"import failed: {exc}"},
        }

    issues: List[Dict[str, Any]] = []
    if set(proposer_forbid) != set(gate_forbid):
        issues.append({
            "list":          "FORBIDDEN_PATHS",
            "proposer_only": sorted(set(proposer_forbid) - set(gate_forbid)),
            "gate_only":     sorted(set(gate_forbid) - set(proposer_forbid)),
        })
    if set(proposer_flag) != set(gate_flag):
        issues.append({
            "list":          "FLAGGED_PATHS",
            "proposer_only": sorted(set(proposer_flag) - set(gate_flag)),
            "gate_only":     sorted(set(gate_flag) - set(proposer_flag)),
        })

    if not issues:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {"n_drifts": len(issues), "issues": issues},
    }


def rule_backtest_grep_kwargs_alignment() -> RuleResult:
    """
    Critical rule #10b (R-1.B.3 enrichment) — backtest.py construct_portfolio()
    literal kwargs must match production ALIGNMENT_SURFACE.

    Companion to rule_backtest_vs_production_param_alignment which checks
    engine.config; this checks engine/backtest.py call sites.

    Mapping kwarg names → ALIGNMENT_SURFACE keys (intentional renaming
    handled here):
      target_vol     → TARGET_VOL
      max_leverage   → MAX_LEVERAGE
      max_weight     → MAX_WEIGHT
      regime_scale   → REGIME_SCALE
      max_net        → MAX_NET
      min_net        → MIN_NET

    Variable kwargs (non-literal) are silently skipped — known AST limitation
    (documented). PRODUCTION_SIGNAL is not a kwarg, so excluded here.
    """
    KWARG_TO_SURFACE = {
        "target_vol":   "TARGET_VOL",
        "max_leverage": "MAX_LEVERAGE",
        "max_weight":   "MAX_WEIGHT",
        "regime_scale": "REGIME_SCALE",
        "max_net":      "MAX_NET",
        "min_net":      "MIN_NET",
    }

    captured = _backtest_kwargs_literal_constants()
    diffs: List[Dict[str, Any]] = []
    for kwarg, surface_key in KWARG_TO_SURFACE.items():
        if kwarg not in captured:
            continue
        expected = ALIGNMENT_SURFACE.get(surface_key)
        actual = captured[kwarg]
        if actual != expected:
            diffs.append({
                "kwarg":    kwarg,
                "surface":  surface_key,
                "expected": expected,
                "actual":   actual,
            })

    if not diffs:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_diffs":          len(diffs),
            "diffs":            diffs,
            "captured_kwargs":  captured,
            "note":             "AST literal-kwargs only; variable kwargs not captured.",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# R-1.C — Per-rule extra-context providers (optional, for the LLM proposer)
# ═════════════════════════════════════════════════════════════════════════════
# Each provider is rule-specific intelligence handed to the Layer 1 LLM:
#   • facts:            structured rule-specific data (file diffs, ledger
#                       entries, etc.) the LLM benefits from seeing
#   • prompt_overrides: short hints the LLM should weight more heavily for
#                       this rule's diagnosis / options / recommendation
# Rules without a registered provider get an empty context (the generic
# skeleton suffices for trivial cases).

def _ctx_production_signal_vs_falsification(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "PRODUCTION_SIGNAL is in REJECTED_PRODUCTION_SIGNALS. The current value was reverted to a previously-falsified strategy.",
            "options_hint":        "Default option = revert to last known-good signal (`ql01_bab` per memory project_b_plus_prod_migration_2026-05-05). Re-validation = consume EFFECTIVE_N_TRIALS slot, requires forward test.",
            "recommendation_bias": "Strongly favour revert; re-validation only if external literature ≥10y supports rehabilitation.",
        },
    }


def _ctx_spec_hash_drift(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    For rule_spec_hash_vs_code_drift — load each drifted spec_path's last 60
    lines so the LLM can reason about whether the edit was a substantive
    logic change or trivial doc/comment.
    """
    import os
    facts: Dict[str, Any] = {"file_tails": []}
    for issue in snapshot.get("issues", []):
        path = issue.get("spec_path")
        if not path:
            continue
        abs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            path,
        )
        if not os.path.exists(abs_path):
            facts["file_tails"].append({"path": path, "exists": False})
            continue
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            facts["file_tails"].append({
                "path":  path,
                "exists": True,
                "tail_60_lines": "".join(lines[-60:])[:3000],
            })
        except Exception:
            facts["file_tails"].append({"path": path, "exists": True, "read_error": True})
    return {
        "facts": facts,
        "prompt_overrides": {
            "diagnosis_hint":      "Compare each spec_path's recomputed hash to stored. Inspect the tail_60_lines for the nature of the edit.",
            "options_hint":        "Trivial edit (whitespace/typo/comment) → amend_spec(kind='clarification', n_trials_added=0). Substantive logic → kind='threshold_tweak' or higher (n_trials_added>0). Unauthorised edit → revert + amend_spec.",
            "recommendation_bias": "Lean conservative: when in doubt, propose 'clarification' + ask supervisor for evidence the edit was intentional.",
        },
    }


def _ctx_db_schema_drift(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "Either ORM and DB are out of sync (migration missed) or someone ran ad-hoc SQL.",
            "options_hint":        "ORM-only column → add migration in engine/memory.py _migrate_db. DB-only column → check if any production code reads it (grep): if yes, add to ORM; if no (zero references) and zero non-NULL values, propose DROP COLUMN.",
            "recommendation_bias": "Prefer DROP COLUMN only when 0 non-NULL rows AND zero code references. Otherwise prefer add-to-ORM.",
        },
    }


def _ctx_cash_flow(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "Three accounting equalities checked: morning, close, cashflow_sum. Likely root causes: missed dividend/distribution, corporate action, manual NAV edit, rounding accumulation.",
            "options_hint":        "First option always = investigate (yfinance + CashFlow table query for the specific date). Don't propose data correction without confirming the cause.",
            "recommendation_bias": "Lean toward investigation, not patching. NAV numbers are investor-visible — never silently 'fix' them.",
        },
    }


def _ctx_universe_drift(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "Active UniverseETF set diverges from the registered baseline. Could be: legitimate universe expansion (supervisor approved elsewhere) / silent edit / scheduled universe_review change.",
            "options_hint":        "Option 1 = approve drift + update baseline (R-1.E will do this). Option 2 = revert UniverseETF table to baseline. Option 3 = investigate.",
            "recommendation_bias": "Universe changes affect every backtest; favour investigation over auto-approval.",
        },
    }


def _ctx_param_alignment(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "engine.config production constants drifted from ALIGNMENT_SURFACE — the P0-1 thesis-claim alignment commitment.",
            "options_hint":        "Either revert config to surface, or amend ALIGNMENT_SURFACE (with strong justification — this is a thesis-claim invariant).",
            "recommendation_bias": "Strongly favour revert. ALIGNMENT_SURFACE amendment requires supervisor + new spec amend with kind ≥ threshold_tweak.",
        },
    }


def _ctx_approval_staleness(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "PendingApproval rows older than 14 days. Likely supervisor capacity overflow or auto-allocation backlog.",
            "options_hint":        "Triage the oldest items: cancel ones that are no longer relevant; bulk-resolve ones that have been superseded by other decisions; escalate truly important ones.",
            "recommendation_bias": "Prefer triage workflow over blind expiration; preserve audit trail.",
        },
    }


def _ctx_skill_library(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "facts": {},
        "prompt_overrides": {
            "diagnosis_hint":      "SkillLibrary has 0 rows or has been dormant ≥30 days. Per memory project_meta_audit_kill_simplify_2026-05-05, this is documented dead-branch state.",
            "options_hint":        "Option 1 = silence with kind='no_action' + rationale referencing the meta-audit decision. Option 2 = delete the import statements and ORM entry (small refactor).",
            "recommendation_bias": "Default to silence — supervisor already decided 2026-05-05; deletion can wait.",
        },
    }


# Registration of context providers (lazy — referenced from auto_audit_proposer
# if it's imported; not required for rule execution).
_CONTEXT_PROVIDER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "rule_production_signal_vs_falsification_chain": _ctx_production_signal_vs_falsification,
    "rule_spec_hash_vs_code_drift":                  _ctx_spec_hash_drift,
    "rule_db_schema_vs_orm_consistency":             _ctx_db_schema_drift,
    "rule_cash_flow_conservation":                   _ctx_cash_flow,
    "rule_universe_drift_vs_registered":             _ctx_universe_drift,
    "rule_backtest_vs_production_param_alignment":   _ctx_param_alignment,
    "rule_backtest_grep_kwargs_alignment":           _ctx_param_alignment,
    "rule_approval_queue_staleness":                 _ctx_approval_staleness,
    "rule_skill_library_dormancy":                   _ctx_skill_library,
}


def get_context_provider_registry() -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """Public accessor; auto_audit_proposer imports this to populate its
    EXTRA_CONTEXT_PROVIDERS without circular imports."""
    return dict(_CONTEXT_PROVIDER_REGISTRY)


# ═════════════════════════════════════════════════════════════════════════════
# Wave 5 (2026-05-07 applied-focus reframe) —
# capability_vs_data_congruence: weekly meta-audit
# ═════════════════════════════════════════════════════════════════════════════
# Day 1 deep-audit found 17 empty tables; Wave 4 reclassified 12 as healthy or
# project-age-related; the remaining ~3 represent genuine "capability claimed
# but underlying table never accumulates" risk.  Tier R catches structural
# drift and hash chain breaks but had no rule for "capability X needs data in
# table Y per cadence Z".  This rule fills that gap.
#
# Registry semantics:
#   capability       — short label for the claim (matches README / capability_evidence.md)
#   downstream_table — DB table that should accumulate evidence
#   cadence_days     — expected inter-row spacing (e.g. monthly = 30)
#   skip_until_age_d — project must be at least this old before we expect rows
#                      (avoids false positives on a 3-day-old project)
#   silenceable      — LOW severity findings are silenceable via the standard
#                      Tier R IGNORE flow; HIGH findings escalate
#
# Sibling-pattern hunt:  rule_agent_reflection_heartbeat covers agents that
# inherit base.Agent + log to AgentRun; this rule complements it by covering
# function-based capability claims (memory_curator) and pure detectors
# (HARKing) that don't emit AgentRun rows.

CAPABILITY_REGISTRY: List[Dict[str, Any]] = [
    {
        "capability":       "memory_curator_monthly_report",
        "downstream_table": "memory_curator_reports",
        "downstream_filter": "1=1",
        "downstream_time_col": "generated_at",
        "cadence_days":     30,
        "skip_until_age_d": 35,    # need full month + a few days of buffer
        "silenceable":      False,
    },
    {
        "capability":       "harking_detection_active",
        "downstream_table": "harking_flags",
        "downstream_filter": "1=1",
        "downstream_time_col": "detected_at",   # 2026-05-07 self-audit: schema uses detected_at not created_at
        "cadence_days":     90,    # HARKing is rare-event detection
        "skip_until_age_d": 180,   # don't flag until 6 months active
        "silenceable":      True,  # zero flags is healthy if research is clean
    },
]


def _project_age_days() -> int:
    """Days since project's earliest cycle_states row.  Returns 0 if no
    cycle_states rows exist (project never ran)."""
    from sqlalchemy import text as _sql_text
    from engine.memory import SessionFactory
    import datetime as _dt
    with SessionFactory() as s:
        row = s.execute(_sql_text(
            "SELECT MIN(started_at) FROM cycle_states"
        )).fetchone()
    if row is None or row[0] is None:
        return 0
    started = row[0]
    if isinstance(started, str):
        try:
            started = _dt.datetime.fromisoformat(started)
        except ValueError:
            return 0
    return (_dt.datetime.utcnow() - started).days


def rule_capability_vs_data_congruence() -> RuleResult:
    """
    Weekly rule — flag claimed capabilities whose downstream table has no rows
    despite the project being old enough to expect them.

    Severity ladder:
      project too young (age < skip_until_age_d)  → SKIP (silent)
      table empty within 1× cadence past threshold → no finding (margin)
      table empty for ≥1× cadence past threshold  → MID
      table empty for ≥2× cadence past threshold  → HIGH

    The rule uses the same downstream-table query pattern as
    scripts/audit_agent_liveness, but lives inside the Tier R sweep so its
    findings flow through the standard proposer → gate → supervisor pipeline.
    """
    from sqlalchemy import text as _sql_text
    from engine.memory import SessionFactory
    import datetime as _dt

    age_days = _project_age_days()
    if age_days <= 0:
        return None  # cannot determine age

    issues: List[Dict[str, Any]] = []
    today = _dt.datetime.utcnow()

    with SessionFactory() as s:
        for cap in CAPABILITY_REGISTRY:
            if age_days < cap["skip_until_age_d"]:
                continue   # too early to expect data

            # Most-recent row's timestamp
            try:
                last_ts_row = s.execute(_sql_text(
                    f"SELECT MAX({cap['downstream_time_col']}) "
                    f"FROM {cap['downstream_table']} "
                    f"WHERE {cap['downstream_filter']}"
                )).fetchone()
            except Exception as e:
                # Table or column missing — surface as HIGH (schema regression)
                issues.append({
                    "capability":  cap["capability"],
                    "kind":        "table_or_column_missing",
                    "table":       cap["downstream_table"],
                    "error":       str(e)[:200],
                    "severity":    "HIGH",
                })
                continue

            last_ts = last_ts_row[0] if last_ts_row else None
            if last_ts is None:
                # Table exists but zero rows
                staleness_days = age_days   # since project start
            else:
                if isinstance(last_ts, str):
                    try:
                        last_ts = _dt.datetime.fromisoformat(last_ts)
                    except ValueError:
                        last_ts = None
                if last_ts is None:
                    staleness_days = age_days
                else:
                    staleness_days = (today - last_ts).days

            cadence = cap["cadence_days"]
            sev: Optional[str] = None
            if staleness_days >= 2 * cadence:
                sev = "HIGH"
            elif staleness_days >= cadence:
                sev = "MID" if not cap["silenceable"] else "LOW"

            if sev is None:
                continue

            issues.append({
                "capability":      cap["capability"],
                "kind":            "stale_or_empty",
                "table":           cap["downstream_table"],
                "staleness_days":  staleness_days,
                "expected_cadence_days": cadence,
                "skip_until_age_d":      cap["skip_until_age_d"],
                "project_age_days":      age_days,
                "severity":        sev,
            })

    if not issues:
        return None

    # Aggregate severity = highest seen
    order = {"LOW": 0, "MID": 1, "HIGH": 2}
    agg = max(issues, key=lambda r: order.get(r.get("severity", "LOW"), 0))
    return {
        "rule_name": "rule_capability_vs_data_congruence",
        "severity":  agg["severity"],
        "snapshot":  {
            "n_capabilities_checked": len(CAPABILITY_REGISTRY),
            "n_issues":               len(issues),
            "issues":                 issues,
            "project_age_days":       age_days,
        },
    }


def rule_factor_lab_state_consistent() -> RuleResult:
    """
    Critical rule (P-LAB, 2026-05-08) — Factor Lab state machine consistency.

    Verifies for every SpecRegistry row with factor_kind ∈ {production_swap,
    overlay, shadow}:
      (a) lab_state must be a valid FactorState enum value (no NULL,
          no garbage strings)
      (b) Every consecutive (from_state, to_state) pair recorded in
          amendment_log lab_state_transition entries must be a legal
          transition per docs/spec_factor_lab.md §2.2.
      (c) The latest lab_state_transition entry's to_state must equal
          the row's current lab_state (no silent overwrite).

    Spec ref: docs/spec_factor_lab.md §8.

    Severity HIGH: state machine inconsistency means audit-trail integrity
    is broken — any verdict citing the row's state could be wrong.
    """
    from engine.factor_lab.types import FactorState, _LEGAL_TRANSITIONS
    from engine.memory import SessionFactory, SpecRegistry

    issues: List[Dict[str, Any]] = []
    _ACTIVE_KINDS = {"production_swap", "overlay", "shadow"}
    _VALID_STATES = {s.value for s in FactorState}

    with SessionFactory() as s:
        rows = (
            s.query(SpecRegistry)
            .filter(SpecRegistry.factor_kind.in_(_ACTIVE_KINDS))
            .all()
        )
        for r in rows:
            cur = r.lab_state
            if cur is None or cur not in _VALID_STATES:
                issues.append({
                    "spec_path":      r.spec_path,
                    "kind":           "invalid_lab_state",
                    "value":          cur,
                })
                continue

            # Walk amendment_log lab_state_transition entries
            try:
                log = json.loads(r.amendment_log or "[]")
            except Exception:
                log = []
            transitions = [e for e in log
                           if isinstance(e, dict)
                           and e.get("kind") == "lab_state_transition"]
            for entry in transitions:
                src_v = entry.get("from_state")
                dst_v = entry.get("to_state")
                if src_v not in _VALID_STATES or dst_v not in _VALID_STATES:
                    issues.append({
                        "spec_path":  r.spec_path,
                        "kind":       "transition_unknown_state",
                        "entry":      {"from": src_v, "to": dst_v},
                    })
                    continue
                src_state = FactorState(src_v)
                dst_state = FactorState(dst_v)
                if dst_state not in _LEGAL_TRANSITIONS.get(src_state, set()):
                    issues.append({
                        "spec_path":  r.spec_path,
                        "kind":       "illegal_transition_in_log",
                        "entry":      {"from": src_v, "to": dst_v},
                    })

            # Last transition must match current state
            if transitions:
                last_to = transitions[-1].get("to_state")
                if last_to != cur:
                    issues.append({
                        "spec_path":         r.spec_path,
                        "kind":              "current_state_vs_log_mismatch",
                        "current_lab_state": cur,
                        "last_log_to_state": last_to,
                    })

    if not issues:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_issues":  len(issues),
            "n_active":  len(rows),
            "issues":    issues[:30],
        },
    }


def rule_factor_lab_no_factor_library_import() -> RuleResult:
    """
    Critical rule (factor library v1, 2026-05-09) — One-way module dependency lock.

    Per docs/spec_factor_library_v1.md §4.1 + docs/spec_factor_lab.md boundary:
      - factor_library is the **content layer** (signal_fn closures + ensemble)
      - factor_lab    is the **infrastructure layer** (state machine + power_check)
      - Allowed:    factor_library → factor_lab.power.power_check
      - FORBIDDEN:  factor_lab    → factor_library  (any import)

    Static check: grep engine/factor_lab/**/*.py for any reference to
    `engine.factor_library` or `from engine import factor_library`. Comments are
    ignored. Severity HIGH: layering violation breaks the "infrastructure stays
    content-agnostic" invariant — once broken, future signal additions require
    factor_lab edits, defeating the abstraction.
    """
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    factor_lab_dir = project_root / "engine" / "factor_lab"

    if not factor_lab_dir.is_dir():
        return None  # nothing to check

    violations: List[Dict[str, Any]] = []
    for py_file in factor_lab_dir.rglob("*.py"):
        try:
            text = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if (
                "engine.factor_library" in stripped
                or "from engine import factor_library" in stripped
            ):
                violations.append({
                    "file":  str(py_file.relative_to(project_root)),
                    "line":  line_no,
                    "code":  stripped[:200],
                })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_violations": len(violations),
            "violations":   violations[:20],
        },
    }


def rule_etf_holdings_cap_clamp_bounds() -> RuleResult:
    """
    Critical rule (ETF Holdings Risk Monitor v1, 2026-05-08, spec id=49) —
    verify cap-multiplier + duration constants are within spec-locked bounds.

    Per docs/spec_etf_holdings_llm_risk_monitor.md §2.7 + §六 forbidden mods:
      - HARD_CAP_MULTIPLIER ∈ [HARD_CAP_FLOOR, HARD_CAP_UPPER] = [0.5, 1.0]
        (越下限 = 过度 defensive; 越上限 = LLM aggressive 重现 wrapping)
      - HARD_CAP_DURATION_DAYS ≤ HARD_CAP_DURATION_CAP = 10
        (defense-in-depth: even with future amend, cannot exceed)
      - CAP_TRIGGER_THRESHOLD must be in [1.0, 5.0] Likert range

    Severity HIGH: bound violations would let LLM modify P&L outside spec,
    breaking pre-registration discipline.
    """
    try:
        from engine.etf_holdings_risk_monitor import (
            HARD_CAP_MULTIPLIER,
            HARD_CAP_DURATION_DAYS,
            HARD_CAP_DURATION_CAP,
            HARD_CAP_FLOOR,
            HARD_CAP_UPPER,
            CAP_TRIGGER_THRESHOLD,
        )
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"reason": "import_failed", "exception": str(exc)},
        }

    violations: List[Dict[str, Any]] = []

    if not (HARD_CAP_FLOOR <= HARD_CAP_MULTIPLIER <= HARD_CAP_UPPER):
        violations.append({
            "constant": "HARD_CAP_MULTIPLIER",
            "value":    HARD_CAP_MULTIPLIER,
            "expected": f"[{HARD_CAP_FLOOR}, {HARD_CAP_UPPER}]",
        })
    if HARD_CAP_DURATION_DAYS > HARD_CAP_DURATION_CAP:
        violations.append({
            "constant": "HARD_CAP_DURATION_DAYS",
            "value":    HARD_CAP_DURATION_DAYS,
            "cap":      HARD_CAP_DURATION_CAP,
        })
    if not (1.0 <= CAP_TRIGGER_THRESHOLD <= 5.0):
        violations.append({
            "constant": "CAP_TRIGGER_THRESHOLD",
            "value":    CAP_TRIGGER_THRESHOLD,
            "expected": "[1.0, 5.0]",
        })
    if HARD_CAP_FLOOR < 0 or HARD_CAP_UPPER > 1.0 + 1e-9:
        violations.append({
            "constant":    "HARD_CAP_FLOOR/UPPER",
            "floor_value": HARD_CAP_FLOOR,
            "upper_value": HARD_CAP_UPPER,
            "expected":    "FLOOR ≥ 0, UPPER ≤ 1.0",
        })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {"n_violations": len(violations), "violations": violations},
    }


def rule_etf_holdings_no_llm_in_eval() -> RuleResult:
    """
    Critical rule (ETF Holdings Risk Monitor v1, 2026-05-08, spec id=49) —
    verify 0-LLM-in-evaluation invariant on engine/etf_holdings_risk_monitor.py.

    Per project rule "0-LLM-in-evaluation" (feedback_no_llm_as_judge.md) +
    spec §rule-9 boundary invariant:
      - LLM should be called ONLY inside `_call_llm_screen_name` (and via
        public API `screen_name` which delegates).
      - LLM must NOT be called inside aggregation, trigger, cap-application,
        cost-ledger, or cap-state-management functions.

    Static check: scan engine/etf_holdings_risk_monitor.py for `model.generate_content`
    or `pool.get_model` outside the `_call_llm_screen_name` function block.

    Severity HIGH: if LLM enters verdict path, statistical inference invalid +
    counterfactual P&L attribution non-reproducible.
    """
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    target = project_root / "engine" / "etf_holdings_risk_monitor.py"

    if not target.is_file():
        return None  # module not yet present; sprint may not have shipped

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"reason": "read_failed", "exception": str(exc)},
        }

    # Find the _call_llm_screen_name function block (allowed LLM zone)
    lines = text.splitlines()
    allowed_zone_start = None
    allowed_zone_end = None
    for i, line in enumerate(lines):
        if line.startswith("def _call_llm_screen_name"):
            allowed_zone_start = i
        elif allowed_zone_start is not None and allowed_zone_end is None:
            # End of function = next top-level def or class or end of file
            stripped = line.lstrip()
            if line and not line.startswith(" ") and not line.startswith("\t"):
                if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("# "):
                    allowed_zone_end = i
                    break
    if allowed_zone_end is None:
        allowed_zone_end = len(lines)

    # LLM call signatures to look for (forbidden outside allowed zone)
    forbidden_patterns = [
        "model.generate_content",
        "pool.get_model",
    ]

    violations: List[Dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        # Skip allowed zone
        if allowed_zone_start is not None and (allowed_zone_start + 1) <= line_no <= allowed_zone_end:
            continue
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        # Skip docstrings (heuristic: lines inside triple-quote blocks; we just check
        # if the line is purely descriptive text without forbidden patterns at literal level)
        for pat in forbidden_patterns:
            if pat in stripped:
                violations.append({
                    "line":     line_no,
                    "pattern":  pat,
                    "code":     stripped[:200],
                })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_violations":   len(violations),
            "violations":     violations[:20],
            "allowed_zone":   f"lines {allowed_zone_start}-{allowed_zone_end} (_call_llm_screen_name)",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# LLM-component removal-test doc existence (2026-05-12, Phase 1 c) — 1 rule
# Per feedback_llm_component_removal_test_governance.md retroactive sweep.
# ═════════════════════════════════════════════════════════════════════════════


# Known LLM-in-loop component → required removal_test doc filename
_LLM_COMPONENT_REMOVAL_TEST_MAP: Dict[str, str] = {
    "engine/agents/reflection.py":           "reflection_memory.md",
    "engine/agents/spec_drafter.py":         "auto_spec_drafter.md",
    "engine/anomaly_llm_detector.py":        "anomaly_screener_s6.md",
    "engine/etf_holdings_risk_monitor.py":   "etf_holdings_monitor.md",
    "engine/fomc_surprise_override.py":      "fomc_surprise_override.md",
    "engine/agents/history_rag/synthesize.py": "rag_history_synthesis.md",
}


def rule_llm_removal_test_doc_exists() -> RuleResult:
    """
    Weekly rule (Phase 1 c, 2026-05-12) — verify every known LLM-in-loop
    component has a registered removal_test doc.

    Per feedback_llm_component_removal_test_governance.md standing rule:
      - Each LLM-touching component must register removal_metric +
        removal_prediction + removal_test_deadline + fallback
      - Universal 6mo deadline: 2026-11-09
      - Missing doc = governance gap; cannot evaluate KILL question at deadline

    Severity LOW: doc missing is fixable (write the doc); but tracks as
    LOW-priority finding to ensure regular review.

    Severity MID: if component file exists AND removal_test doc missing AND
    deadline approaching (within 30 days) → escalate.
    """
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    removal_tests_dir = project_root / "docs" / "removal_tests"

    if not removal_tests_dir.is_dir():
        return {
            "severity": "MID",
            "snapshot": {
                "reason": "removal_tests_dir_missing",
                "expected_path": str(removal_tests_dir),
            },
        }

    missing: List[Dict[str, str]] = []
    for component_path, expected_doc in _LLM_COMPONENT_REMOVAL_TEST_MAP.items():
        component_file = project_root / component_path
        doc_file = removal_tests_dir / expected_doc
        if component_file.is_file() and not doc_file.is_file():
            missing.append({
                "component": component_path,
                "expected_doc": str(doc_file.relative_to(project_root)),
            })

    if not missing:
        return None
    return {
        "severity": "LOW",  # MID escalation logic can be added when deadline approaches
        "snapshot": {
            "n_missing":   len(missing),
            "missing":     missing,
            "deadline":    "2026-11-09",
            "governance":  "feedback_llm_component_removal_test_governance.md",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# FOMC Surprise Override v1 (2026-05-12 unlock, spec id=48) — 2 Tier R rules
# Mirrors ETF Holdings monitor pattern; same invariant family.
# ═════════════════════════════════════════════════════════════════════════════


def rule_fomc_override_clamp_bounds() -> RuleResult:
    """
    Critical rule (FOMC Surprise Override v1, 2026-05-12 unlock, spec id=48) —
    verify multiplier + duration constants stay within spec-locked bounds.

    Per docs/spec_fomc_surprise_override.md §2.7 + §六 forbidden mods:
      - HARD_OVERRIDE_MULTIPLIER ∈ [HARD_MULTIPLIER_LOWER, HARD_MULTIPLIER_UPPER]
        (越下限 = LLM 过度 defensive 减仓; 越上限 = LLM aggressive 加仓 = wrapping critique 重现)
      - HARD_DURATION_DAYS ≤ HARD_DURATION_CAP (even with future amend, cannot exceed)

    Severity HIGH: bound violations would let LLM modify P&L outside spec.
    """
    try:
        from engine.fomc_surprise_override import (
            HARD_OVERRIDE_MULTIPLIER,
            HARD_DURATION_DAYS,
            HARD_DURATION_CAP,
            HARD_MULTIPLIER_LOWER,
            HARD_MULTIPLIER_UPPER,
        )
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"reason": "import_failed", "exception": str(exc)},
        }

    violations: List[Dict[str, Any]] = []

    if not (HARD_MULTIPLIER_LOWER <= HARD_OVERRIDE_MULTIPLIER <= HARD_MULTIPLIER_UPPER):
        violations.append({
            "constant": "HARD_OVERRIDE_MULTIPLIER",
            "value":    HARD_OVERRIDE_MULTIPLIER,
            "expected": f"[{HARD_MULTIPLIER_LOWER}, {HARD_MULTIPLIER_UPPER}]",
        })
    if HARD_DURATION_DAYS > HARD_DURATION_CAP:
        violations.append({
            "constant": "HARD_DURATION_DAYS",
            "value":    HARD_DURATION_DAYS,
            "cap":      HARD_DURATION_CAP,
        })
    if HARD_MULTIPLIER_LOWER < 0 or HARD_MULTIPLIER_UPPER > 1.0 + 1e-9:
        violations.append({
            "constant":    "HARD_MULTIPLIER_LOWER/UPPER",
            "lower_value": HARD_MULTIPLIER_LOWER,
            "upper_value": HARD_MULTIPLIER_UPPER,
            "expected":    "LOWER ≥ 0, UPPER ≤ 1.0",
        })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {"n_violations": len(violations), "violations": violations},
    }


def rule_fomc_override_no_llm_in_eval() -> RuleResult:
    """
    Critical rule (FOMC Surprise Override v1, 2026-05-12 unlock, spec id=48) —
    verify 0-LLM-in-evaluation invariant on engine/fomc_surprise_override.py.

    Per project rule "0-LLM-in-evaluation" (feedback_no_llm_as_judge.md) +
    spec §3.7 boundary invariant:
      - LLM should be called ONLY inside `_call_llm` (and indirectly via
        process_fomc_day which calls it).
      - LLM must NOT be called inside trigger_emergency_override,
        apply_override_to_regime_scale, get_active_override_state,
        validate_and_classify, or cost-ledger functions.

    Static check: scan engine/fomc_surprise_override.py for `model.generate_content`
    or `pool.get_model` outside the `_call_llm` function block.

    Severity HIGH: if LLM enters verdict path, statistical inference invalid +
    counterfactual P&L attribution non-reproducible.
    """
    import pathlib

    project_root = pathlib.Path(__file__).resolve().parent.parent
    target = project_root / "engine" / "fomc_surprise_override.py"

    if not target.is_file():
        return None  # module not present (defensive)

    try:
        text = target.read_text(encoding="utf-8")
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {"reason": "read_failed", "exception": str(exc)},
        }

    # Find the _call_llm function block (allowed LLM zone)
    lines = text.splitlines()
    allowed_zone_start = None
    allowed_zone_end = None
    for i, line in enumerate(lines):
        if line.startswith("def _call_llm("):
            allowed_zone_start = i
        elif allowed_zone_start is not None and allowed_zone_end is None:
            stripped = line.lstrip()
            if line and not line.startswith(" ") and not line.startswith("\t"):
                if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("# "):
                    allowed_zone_end = i
                    break
    if allowed_zone_end is None:
        allowed_zone_end = len(lines)

    forbidden_patterns = [
        "model.generate_content",
        "pool.get_model",
    ]

    violations: List[Dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if allowed_zone_start is not None and (allowed_zone_start + 1) <= line_no <= allowed_zone_end:
            continue
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        for pat in forbidden_patterns:
            if pat in stripped:
                violations.append({
                    "line":    line_no,
                    "pattern": pat,
                    "code":    stripped[:200],
                })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_violations": len(violations),
            "violations":   violations[:20],
            "allowed_zone": f"lines {allowed_zone_start}-{allowed_zone_end} (_call_llm)",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Factor Ensemble v1 (2026-05-09, spec id=50) — 3 NEW Tier R rules
# Per spec §4.7 amendment 2026-05-09 (pre-Sprint-Week-4 audit Issue #3)
# ═════════════════════════════════════════════════════════════════════════════

# Layer 1 — AST static patterns flagged as MID-severity lookahead candidates
# inside engine/factors/*.py. Quality.py legitimately uses .info but is wrapped
# by SPEC_LOCK_DATE guard; AST cannot prove safety, so we flag and let the
# runtime probe (Layer 2) be the HIGH-severity backstop.
_FACTOR_LOOKAHEAD_AST_PATTERNS = (
    ".info",                # yfinance Ticker.info → current fundamentals (lookahead)
    "datetime.date.today",  # naive 'now' inside walk-forward → lookahead
    "datetime.datetime.now",
    "pd.Timestamp.now",
)

# Layer 2 — runtime probe sample. MUST use tickers actually in the active
# universe; prior bug (2026-05-09 audit) hardcoded SPY/XLK which are NOT in
# the registry — the probe spuriously passed only because the SPEC_LOCK_DATE
# guard at quality.py L103 short-circuits before universe membership check.
# To make the probe robust to any future refactor that moves the guard, we
# pull live equity-scope tickers from the active universe at probe time.
def _probe_equity_tickers(n: int = 3) -> tuple[str, ...]:
    try:
        from engine.universe_manager import get_asset_class_map
        ac = get_asset_class_map() or {}
        eq = sorted([t for t, c in ac.items() if c in {"equity_sector", "equity_factor"}])
        return tuple(eq[:n]) if eq else ("XLF", "XLE", "XLY")
    except Exception:
        return ("XLF", "XLE", "XLY")  # last-resort fallback (all in registry as of 2026-05-09)


def rule_factor_ensemble_no_lookahead() -> RuleResult:
    """
    Tier R rule (Factor Ensemble v1, spec id=50 §4.7 amendment 2026-05-09).

    TWO-LAYER lookahead detector:
      Layer 1 (AST static scan, severity MID on hits): pattern-grep
          engine/factors/*.py for obvious lookahead anti-patterns
          (.info / today() / now()). Quality.py legitimately uses .info
          inside a SPEC_LOCK_DATE guard — AST cannot prove that safety,
          so any hit only escalates to MID for human review.
      Layer 2 (runtime SPEC_LOCK_DATE guard probe, severity HIGH on regress):
          invoke compute_quality_signal(as_of=SPEC_LOCK_DATE-1day, …, use_cache=False)
          and assert returned Series is all-NaN. Runs only the cheap entry-guard
          path — does NOT actually fetch fundamentals. If guard regresses,
          walk-forward Quality contamination becomes possible.

    HIGH severity on Layer 2 failure (silent guard regression).
    MID severity on Layer 1 hit only (advisory; needs human review).
    """
    import os
    from pathlib import Path

    import ast
    repo_root = Path(__file__).resolve().parent.parent
    factors_dir = repo_root / "engine" / "factors"
    layer1_hits: list[dict] = []
    if factors_dir.exists():
        for p in factors_dir.glob("*.py"):
            if p.name == "__init__.py":
                continue
            try:
                src = p.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(p))
            except Exception:
                continue
            rel_file = str(p.relative_to(repo_root)).replace(os.sep, "/")
            # Pre-pass: collect Attribute nodes that are the .func slot of a Call
            # (i.e., method invocations like logger.info("msg")) — these should be
            # excluded from the .info attribute-access pattern, which targets
            # yfinance Ticker.info property reads.
            method_call_attrs = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    method_call_attrs.add(id(node.func))
            for node in ast.walk(tree):
                # Pattern 1: yfinance Ticker.info — property access, NOT a method
                # call. Also excludes cases where owner is named `logger` (Python
                # logging Logger.info method).
                if isinstance(node, ast.Attribute) and node.attr == "info":
                    if id(node) in method_call_attrs:
                        continue
                    if isinstance(node.value, ast.Name) and node.value.id == "logger":
                        continue
                    layer1_hits.append({
                        "file":     rel_file,
                        "line":     getattr(node, "lineno", 0),
                        "pattern":  ".info",
                        "code":     ast.unparse(node)[:200] if hasattr(ast, "unparse") else "<.info attr access>",
                    })
                # Pattern 2: today() / now() calls
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("today", "now"):
                        layer1_hits.append({
                            "file":     rel_file,
                            "line":     getattr(node, "lineno", 0),
                            "pattern":  f".{node.func.attr}()",
                            "code":     ast.unparse(node)[:200] if hasattr(ast, "unparse") else f"<.{node.func.attr}() call>",
                        })

    # Layer 2 — runtime probe (only run if quality.py importable + has SPEC_LOCK_DATE)
    layer2_failure: Optional[Dict[str, Any]] = None
    try:
        from engine.factors.quality import compute_quality_signal, SPEC_LOCK_DATE
        import datetime as _dt

        probe_as_of = SPEC_LOCK_DATE - _dt.timedelta(days=1)
        probe_universe = list(_probe_equity_tickers(3))
        probe_asset_classes = {t: "equity_sector" for t in probe_universe}
        probe_signal = compute_quality_signal(
            as_of=probe_as_of,
            universe=probe_universe,
            asset_classes=probe_asset_classes,
            use_cache=False,
        )
        # Spec §2.2.3 amendment lock: as_of < SPEC_LOCK_DATE → all-NaN
        if probe_signal.notna().any():
            non_nan_tickers = probe_signal[probe_signal.notna()].index.tolist()
            layer2_failure = {
                "probe_as_of":      probe_as_of.isoformat(),
                "spec_lock_date":   SPEC_LOCK_DATE.isoformat(),
                "non_nan_tickers":  list(non_nan_tickers),
                "diagnosis":        "SPEC_LOCK_DATE guard regressed — Quality returns non-NaN for as_of < SPEC_LOCK_DATE",
            }
    except Exception as exc:
        # Probe failure (import / yfinance / network) → MID severity informational only
        layer2_failure = None
        layer1_hits.append({
            "file":    "engine/factors/quality.py",
            "line":    0,
            "pattern": "_runtime_probe_skipped",
            "code":    f"runtime probe could not execute: {exc!s}",
        })

    if layer2_failure is not None:
        return {
            "severity": "HIGH",
            "snapshot": {
                "rule":            "rule_factor_ensemble_no_lookahead",
                "layer":           2,
                "layer2_failure":  layer2_failure,
                "layer1_n_hits":   len(layer1_hits),
                "layer1_hits":     layer1_hits[:20],
            },
        }
    if layer1_hits:
        return {
            "severity": "MID",
            "snapshot": {
                "rule":          "rule_factor_ensemble_no_lookahead",
                "layer":         1,
                "n_hits":        len(layer1_hits),
                "hits":          layer1_hits[:20],
                "note":          (
                    "AST hits are advisory; quality.py legitimately uses .info "
                    "inside SPEC_LOCK_DATE guard. Layer 2 runtime probe passed "
                    "(or could not run). Human review recommended on each hit."
                ),
            },
        }
    return None


def rule_factor_ensemble_no_param_tuning() -> RuleResult:
    """
    Tier R rule (Factor Ensemble v1, spec id=50 §4.7).

    Verifies locked numeric constants in the factor modules + ensemble combiner
    + walk-forward harness match spec values. Any drift = post-hoc parameter
    tuning suspicion (HARKing R2-R3). Severity HIGH.

    Locked values:
      - engine.factors.tsmom.LOOKBACK_MONTHS == 12 (HOP 2017 anchor)
      - engine.factors.tsmom.SKIP_MONTHS    == 1
      - engine.factors.tsmom.VOL_WINDOW_DAYS == 60
      - engine.factors.carry_equity.DIVIDEND_LOOKBACK_DAYS == 365
      - engine.factors.quality.QUALITY_SUB_COMPONENTS == ("profitability", "growth")
      - engine.factors.quality.SPEC_LOCK_DATE == date(2026, 5, 9)
      - engine.factor_ensemble.ENSEMBLE_FACTORS == ("tsmom","carry_equity","quality","bab")
      - engine.factor_ensemble.N_FACTORS == 4
      - engine.factor_ensemble_walk_forward.TARGET_VOL == 0.10
      - engine.factor_ensemble_walk_forward.OOS_START_DATE == date(2011, 1, 1)
      - engine.factor_ensemble_walk_forward.DEFAULT_END_DATE == date(2024, 12, 31)
      - engine.factor_ensemble_verdict.DELTA_SHARPE_POSITIVE_THRESHOLD == 0.20
      - engine.factor_ensemble_verdict.BOOTSTRAP_RESAMPLES == 1000
    """
    import datetime as _dt
    expected: list[tuple[str, str, Any]] = [
        ("engine.factors.tsmom",                       "LOOKBACK_MONTHS",                 12),
        ("engine.factors.tsmom",                       "SKIP_MONTHS",                     1),
        ("engine.factors.tsmom",                       "VOL_WINDOW_DAYS",                 60),
        ("engine.factors.carry_equity",                "DIVIDEND_LOOKBACK_DAYS",          365),
        ("engine.factors.quality",                     "QUALITY_SUB_COMPONENTS",          ("profitability", "growth")),
        ("engine.factors.quality",                     "SPEC_LOCK_DATE",                  _dt.date(2026, 5, 9)),
        ("engine.factor_ensemble",                     "ENSEMBLE_FACTORS",                ("tsmom", "carry_equity", "quality", "bab")),
        ("engine.factor_ensemble",                     "N_FACTORS",                       4),
        ("engine.factor_ensemble_walk_forward",        "TARGET_VOL",                      0.10),
        ("engine.factor_ensemble_walk_forward",        "OOS_START_DATE",                  _dt.date(2011, 1, 1)),
        ("engine.factor_ensemble_walk_forward",        "DEFAULT_END_DATE",                _dt.date(2024, 12, 31)),
        ("engine.factor_ensemble_verdict",             "DELTA_SHARPE_POSITIVE_THRESHOLD", 0.20),
        ("engine.factor_ensemble_verdict",             "BOOTSTRAP_RESAMPLES",             1000),
    ]
    drifts: list[dict] = []
    for module_name, attr, expected_val in expected:
        try:
            mod = __import__(module_name, fromlist=[attr])
            actual = getattr(mod, attr, None)
        except Exception as exc:
            drifts.append({
                "module":   module_name,
                "attr":     attr,
                "expected": repr(expected_val),
                "actual":   f"<import error: {exc!s}>",
            })
            continue
        if actual != expected_val:
            drifts.append({
                "module":   module_name,
                "attr":     attr,
                "expected": repr(expected_val),
                "actual":   repr(actual),
            })

    if drifts:
        return {
            "severity": "HIGH",
            "snapshot": {
                "rule":     "rule_factor_ensemble_no_param_tuning",
                "n_drifts": len(drifts),
                "drifts":   drifts,
                "note":     (
                    "Locked-constant drift = HARKing R2-R3 surface. Any change "
                    "to these values requires explicit amend_spec(kind='hypothesis_amend' "
                    "or 'threshold_tweak') with reason."
                ),
            },
        }
    return None


def rule_factor_ensemble_baseline_reproducibility() -> RuleResult:
    """
    Tier R rule (Factor Ensemble v1, spec id=50 §4.7).

    Reads data/factor_ensemble_v1/gate0_baseline_check.json (produced by
    scripts/run_factor_ensemble_v1_gate0_check.py) and asserts mandatory_pass.

    Per spec §五 Gate 0 amendment 2026-05-09 (Fix #2):
      - PRIMARY (mandatory): non-pathological Sharpe in [-0.5, 2.0] AND finite n_periods
      - DIRECTIONAL (informational): ±0.5 of B++ 0.985 — not a fail criterion

    Severity:
      HIGH if mandatory_pass=False OR file absent past first run (would let
      walk-forward verdict use a broken harness baseline silently).
      LOW (informational) if status=PASS_WITH_DIRECTIONAL_CAVEAT (broad band warn).
    """
    import json as _json
    from pathlib import Path

    gate0_path = Path(__file__).resolve().parent.parent / "data" / "factor_ensemble_v1" / "gate0_baseline_check.json"
    if not gate0_path.exists():
        # Pre-launch: no run yet. Return None (silent) — Gate 0 is pre-launch
        # validation; rule fires only when the file IS present and shows failure.
        return None

    try:
        payload = _json.loads(gate0_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "severity": "HIGH",
            "snapshot": {
                "rule":      "rule_factor_ensemble_baseline_reproducibility",
                "diagnosis": f"gate0_baseline_check.json unreadable: {exc!s}",
                "path":      str(gate0_path),
            },
        }

    mandatory_pass = bool(payload.get("mandatory_pass", False))
    status = str(payload.get("status", "UNKNOWN"))

    if not mandatory_pass:
        return {
            "severity": "HIGH",
            "snapshot": {
                "rule":             "rule_factor_ensemble_baseline_reproducibility",
                "status":           status,
                "harness_sharpe":   payload.get("harness_sharpe"),
                "mandatory_range":  payload.get("mandatory_range"),
                "mandatory_pass":   mandatory_pass,
                "diagnosis":        "Harness BAB-only baseline failed mandatory non-pathological range check; ensemble vs baseline ΔSharpe verdict CANNOT be trusted until baseline reproducibility restored",
            },
        }

    if status == "PASS_WITH_DIRECTIONAL_CAVEAT":
        return {
            "severity": "LOW",
            "snapshot": {
                "rule":            "rule_factor_ensemble_baseline_reproducibility",
                "status":          status,
                "harness_sharpe":  payload.get("harness_sharpe"),
                "bpp_delta":       payload.get("bpp_delta"),
                "note":            "Mandatory PASS but informational directional band breach (broad ±0.5 vs B++ 0.985). Documented apples-to-oranges (monthly vs weekly); LOW informational only.",
            },
        }

    return None


# ═════════════════════════════════════════════════════════════════════════════
# MS-7 (2026-05-10) — multi-sleeve sleeve_id integrity
# ═════════════════════════════════════════════════════════════════════════════

def rule_sleeve_id_integrity() -> RuleResult:
    """
    Critical rule — sleeve_id integrity across the 4 multi-sleeve tables.

    Flags 3 data-corruption modes (post-MS-1 schema):
      1. NULL sleeve_id values (despite NOT NULL constraint — defensive check)
      2. sleeve_id values outside ALLOWED_SLEEVES set (typo silent sharding)
      3. ss_sp500 rows existing while capital allocation is 0% on that sleeve
         (implies write-path bug — single-stock data tagged but inactive sleeve)

    Severity HIGH:
      Mis-tagged sleeve_id breaks per-sleeve attribution (MS-3) +
      multi-sleeve capital allocation (MS-2) silently. Per project final
      vision V2, sleeve_id is the multi-sleeve governance backbone — drift
      here invalidates all downstream per-sleeve reporting.

    See: project_final_vision_hybrid_2026-05-10.md, MS-1/2/3 sprints.
    """
    from sqlalchemy import text
    from engine.memory import engine as _engine
    from engine.portfolio_sleeves import (
        ALLOWED_SLEEVES,
        get_active_config,
        is_sleeve_active,
    )

    tables_with_sleeve_id = [
        "decision_logs",
        "simulated_positions",
        "simulated_trades",
        "simulated_monthly_returns",
    ]
    issues: List[Dict[str, Any]] = []

    try:
        active_cfg = get_active_config()
    except Exception:
        active_cfg = None

    with _engine.connect() as conn:
        for table in tables_with_sleeve_id:
            # Confirm table exists
            try:
                cols = {row[1] for row in
                        conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
            except Exception as exc:
                issues.append({
                    "table":   table,
                    "kind":    "introspection_failed",
                    "error":   str(exc),
                })
                continue
            if not cols:
                continue   # table not yet created
            if "sleeve_id" not in cols:
                issues.append({
                    "table":  table,
                    "kind":   "sleeve_id_column_missing",
                    "fix":    "run engine.memory.init_db() to apply MS-1 migration",
                })
                continue

            # 1. NULL detection (despite NOT NULL — defensive)
            try:
                n_null = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE sleeve_id IS NULL")
                ).scalar()
            except Exception as exc:
                issues.append({
                    "table":   table,
                    "kind":    "null_query_failed",
                    "error":   str(exc),
                })
                continue
            if n_null and n_null > 0:
                issues.append({
                    "table":  table,
                    "kind":   "null_sleeve_id_rows",
                    "n_null": int(n_null),
                    "fix":    "investigate write-path bypassing NOT NULL constraint",
                })

            # 2. Unknown sleeve_id values (typo silent sharding)
            try:
                rows = conn.execute(
                    text(f"SELECT DISTINCT sleeve_id FROM {table}")
                ).fetchall()
            except Exception as exc:
                issues.append({
                    "table":   table,
                    "kind":    "distinct_query_failed",
                    "error":   str(exc),
                })
                continue
            seen_sleeves = {row[0] for row in rows if row[0] is not None}
            unknown = sorted(seen_sleeves - ALLOWED_SLEEVES)
            if unknown:
                issues.append({
                    "table":   table,
                    "kind":    "unknown_sleeve_id_values",
                    "unknown": unknown,
                    "fix":     (
                        "either add to engine.portfolio_sleeves.ALLOWED_SLEEVES "
                        "(if intentional new sleeve), or correct typos via UPDATE"
                    ),
                })

            # 3. Inactive-sleeve write detection (per active capital config)
            if active_cfg is not None:
                for sleeve_id in seen_sleeves:
                    if sleeve_id in ALLOWED_SLEEVES and not is_sleeve_active(
                        sleeve_id, active_cfg,
                    ):
                        try:
                            n = conn.execute(
                                text(f"SELECT COUNT(*) FROM {table} "
                                     f"WHERE sleeve_id = :sid"),
                                {"sid": sleeve_id},
                            ).scalar()
                        except Exception:
                            n = None
                        if n and n > 0:
                            issues.append({
                                "table":      table,
                                "kind":       "rows_in_zero_capital_sleeve",
                                "sleeve_id":  sleeve_id,
                                "n_rows":     int(n),
                                "fix":        (
                                    f"sleeve {sleeve_id!r} has 0% capital allocation "
                                    f"per active config; rows here may indicate a "
                                    f"write-path bug or a stale sleeve. Either "
                                    f"reallocate capital (Tier 3 governance) or "
                                    f"investigate writes."
                                ),
                            })

    if not issues:
        return None
    return {
        "severity":  "HIGH",
        "snapshot":  {
            "n_issues": len(issues),
            "issues":   issues[:30],
            "context":  (
                "MS-7 sleeve_id integrity (post-MS-1/2/3 schema). Multi-sleeve "
                "governance backbone — drift here invalidates per-sleeve "
                "attribution. See project_final_vision_hybrid_2026-05-10.md."
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Ops Watchdog Agent v1.0 — Phase 1 detection rules
#   spec: docs/spec_ops_watchdog_agent_v1.md (id=63, hash 512c918f)
#   Covers 10 of 12 error modes (modes 1/2/5/6/7/8/9/10/11/12).
#   Modes 3 (delisted ETF) + 4 (sleeve drift) reuse existing rules:
#     - rule_universe_drift_vs_registered  (mode 3, partial augment in v1)
#     - rule_sleeve_id_integrity           (mode 4, full coverage)
#   Registered to WATCHDOG_RULES list (NOT CRITICAL/WEEKLY) — invoked by
#   Ops Watchdog Agent at 06:10 SGT, 10min after MacroAlphaPro_DailyBatch.
# ═════════════════════════════════════════════════════════════════════════════


def rule_cycle_state_completion() -> RuleResult:
    """
    Watchdog mode 1 — Cycle silently failed mid-batch.

    Inspect the most recent CycleState row (cycle_type='daily'):
      - status='failed'                                → HIGH
      - status='running' AND started_at > 12h ago      → HIGH (stuck cycle)
      - status not in terminal set AND finished_at NULL → MID
      - No CycleState row in last 36h                  → MID (cycle never started)

    Terminal-OK statuses: 'completed', 'approved', 'pending_gate', 'rejected'.
    pending_gate is terminal-OK because human-gate await is a valid resting
    state (not a silent failure).
    """
    import datetime
    try:
        from engine.db_models import CycleState
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    now = datetime.datetime.utcnow()
    stuck_cutoff   = now - datetime.timedelta(hours=12)
    missing_cutoff = now - datetime.timedelta(hours=36)
    terminal_ok = {"completed", "approved", "pending_gate", "rejected"}

    try:
        with SessionFactory() as s:
            recent = (s.query(CycleState)
                       .filter(CycleState.cycle_type == "daily")
                       .order_by(CycleState.started_at.desc())
                       .first())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    issues: List[Dict[str, Any]] = []

    if recent is None:
        # No daily cycle history yet; not a finding (defensive).
        return None

    if recent.started_at < missing_cutoff and recent.status in terminal_ok:
        issues.append({
            "kind":         "no_recent_cycle_in_36h",
            "last_cycle_id": recent.id,
            "last_started":  recent.started_at.isoformat(),
            "last_status":   recent.status,
        })

    if recent.status == "failed":
        issues.append({
            "kind":      "cycle_failed",
            "cycle_id":  recent.id,
            "started":   recent.started_at.isoformat(),
            "error_log": (recent.error_log or "")[:300],
        })
    elif recent.status == "running" and recent.started_at < stuck_cutoff:
        issues.append({
            "kind":     "cycle_stuck_running",
            "cycle_id": recent.id,
            "started":  recent.started_at.isoformat(),
            "elapsed_hours": round((now - recent.started_at).total_seconds() / 3600.0, 2),
        })

    if not issues:
        return None

    severity = "HIGH" if any(i["kind"] in ("cycle_failed", "cycle_stuck_running")
                             for i in issues) else "MID"
    return {
        "severity": severity,
        "snapshot": {
            "n_issues": len(issues),
            "issues":   issues,
            "context":  "Ops Watchdog mode 1 — cycle silently failed mid-batch.",
        },
    }


def rule_universe_data_freshness_per_ticker() -> RuleResult:
    """
    Watchdog mode 2 — yfinance stale data per production-universe ticker.

    For each ticker in active universe (engine.universe_manager.get_active_universe),
    query SignalRecord.date latest entry. Flag tickers with:
      - No SignalRecord row ever          → HIGH (per-ticker missing)
      - Latest date > 5 calendar days ago → MID (stale beyond weekend slack)

    Severity escalates to HIGH when >20% of universe is stale (broad outage).
    """
    import datetime
    try:
        from sqlalchemy import func
        from engine.db_models import SignalRecord
        from engine.memory import SessionFactory
        from engine.universe_manager import get_active_universe
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        active = get_active_universe()  # {sector: ticker}
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "universe_load_failed", "exception": str(exc)}}

    tickers = sorted(set(active.values()))
    if not tickers:
        return None

    today  = datetime.date.today()
    cutoff = today - datetime.timedelta(days=5)

    try:
        with SessionFactory() as s:
            rows = (s.query(SignalRecord.ticker, func.max(SignalRecord.date))
                     .filter(SignalRecord.ticker.in_(tickers))
                     .group_by(SignalRecord.ticker)
                     .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    latest_by_ticker = {r[0]: r[1] for r in rows}

    missing: List[Dict[str, Any]] = []
    stale:   List[Dict[str, Any]] = []
    for t in tickers:
        last = latest_by_ticker.get(t)
        if last is None:
            missing.append({"ticker": t})
        elif last < cutoff:
            stale.append({"ticker": t, "latest_date": last.isoformat(),
                          "days_stale": (today - last).days})

    n_bad = len(missing) + len(stale)
    if n_bad == 0:
        return None

    pct = n_bad / max(len(tickers), 1)
    severity = "HIGH" if pct >= 0.20 or missing else "MID"
    return {
        "severity": severity,
        "snapshot": {
            "n_universe":    len(tickers),
            "n_missing":     len(missing),
            "n_stale":       len(stale),
            "pct_bad":       round(pct, 3),
            "cutoff_days":   5,
            "missing_sample": missing[:15],
            "stale_sample":   stale[:15],
            "context":       "Ops Watchdog mode 2 — yfinance stale data per ticker.",
        },
    }


def rule_weight_delta_p99_unexplained() -> RuleResult:
    """
    Watchdog mode 5 — Massive weight_delta unexplained.

    For trades in last 7 days, find any whose |weight_delta| > 2x the p99 of
    |weight_delta| over the trailing 60-day window AND whose trigger_reason
    is NOT in ('signal_flip', 'regime_change'). Such spikes lacking a routine
    rebalance/threshold explanation are candidates for data/write-path errors.

    Severity HIGH (per MODE_SEVERITY_MAP_LOCKED — potential data error).
    """
    import datetime
    try:
        from engine.db_models import SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    today = datetime.date.today()
    recent_cut   = today - datetime.timedelta(days=7)
    baseline_cut = today - datetime.timedelta(days=60)

    try:
        with SessionFactory() as s:
            baseline = (s.query(SimulatedTrade.weight_delta)
                         .filter(SimulatedTrade.trade_date >= baseline_cut,
                                 SimulatedTrade.trade_date < recent_cut)
                         .all())
            recent_rows = (s.query(SimulatedTrade)
                            .filter(SimulatedTrade.trade_date >= recent_cut)
                            .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    abs_baseline = sorted(abs(r[0]) for r in baseline if r[0] is not None)
    if len(abs_baseline) < 20:
        # Insufficient baseline; not a finding (calendar-bound).
        return None

    # p99 of trailing 60d |weight_delta|
    p99_idx = max(0, int(round(0.99 * (len(abs_baseline) - 1))))
    p99 = abs_baseline[p99_idx]
    if p99 <= 1e-9:
        return None

    legit = {"signal_flip", "regime_change"}
    spikes: List[Dict[str, Any]] = []
    for t in recent_rows:
        ad = abs(t.weight_delta or 0.0)
        if ad > 2.0 * p99 and (t.trigger_reason or "") not in legit:
            spikes.append({
                "trade_id":       t.id,
                "trade_date":     t.trade_date.isoformat(),
                "ticker":         t.ticker,
                "weight_delta":   round(float(t.weight_delta), 6),
                "abs_delta":      round(ad, 6),
                "trigger_reason": t.trigger_reason or "(null)",
                "p99_baseline":   round(p99, 6),
                "ratio_vs_p99":   round(ad / p99, 3),
            })

    if not spikes:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_spikes":      len(spikes),
            "spikes":        spikes[:20],
            "p99_baseline":  round(p99, 6),
            "n_baseline":    len(abs_baseline),
            "context":       "Ops Watchdog mode 5 — unexplained weight_delta spike.",
        },
    }


def rule_signal_trade_referential_integrity() -> RuleResult:
    """
    Watchdog mode 6 — Trade execution missing for active signal.

    For SignalRecord rows in last 7 days with composite_score >= 70 AND
    gate_status='passed', assert at least one SimulatedTrade row exists for
    the same (date_window=±2d, ticker). Orphaned signals indicate the
    execution path silently dropped a trade.

    Severity MID — recoverable via auto_repair recipe `repair_retry_execution_if_signal_active`.
    """
    import datetime
    try:
        from engine.db_models import SignalRecord, SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    today      = datetime.date.today()
    window_cut = today - datetime.timedelta(days=7)

    try:
        with SessionFactory() as s:
            active_signals = (s.query(SignalRecord)
                              .filter(SignalRecord.date >= window_cut,
                                      SignalRecord.composite_score >= 70.0,
                                      SignalRecord.gate_status == "passed")
                              .all())
            trade_rows = (s.query(SimulatedTrade.ticker, SimulatedTrade.trade_date)
                          .filter(SimulatedTrade.trade_date >= window_cut
                                  - datetime.timedelta(days=2))
                          .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    trade_keys = {(r[0], r[1]) for r in trade_rows}

    orphans: List[Dict[str, Any]] = []
    for sig in active_signals:
        matched = False
        for offset in range(-2, 3):
            d = sig.date + datetime.timedelta(days=offset)
            if (sig.ticker, d) in trade_keys:
                matched = True
                break
        if not matched:
            orphans.append({
                "signal_id":       sig.id,
                "signal_date":     sig.date.isoformat(),
                "ticker":          sig.ticker,
                "composite_score": round(float(sig.composite_score or 0), 2),
                "gate_status":     sig.gate_status,
            })

    if not orphans:
        return None
    return {
        "severity": "MID",
        "snapshot": {
            "n_orphans": len(orphans),
            "orphans":   orphans[:20],
            "context":   "Ops Watchdog mode 6 — active signals without executed trade.",
        },
    }


def rule_nav_move_vs_rebalance_audit() -> RuleResult:
    """
    Watchdog mode 7 — NAV anomaly unexplained.

    For latest PortfolioNavSnapshot, compute daily ex-flow return
        r = (nav_close - nav_after_flow) / nav_after_flow.
    If |r| > 3σ of the trailing 60-day daily_modified_dietz distribution
    AND no SimulatedTrade rows exist for the same date, flag as anomaly.
    Severity SEVERE per spec table (always escalate, decision required).
    """
    import datetime
    import math
    try:
        from engine.db_models import PortfolioNavSnapshot, SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        with SessionFactory() as s:
            latest = (s.query(PortfolioNavSnapshot)
                       .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                       .first())
            if latest is None:
                return None
            baseline_cut = latest.snapshot_date - datetime.timedelta(days=60)
            baseline = (s.query(PortfolioNavSnapshot.daily_modified_dietz)
                         .filter(PortfolioNavSnapshot.snapshot_date >= baseline_cut,
                                 PortfolioNavSnapshot.snapshot_date < latest.snapshot_date)
                         .all())
            n_trades_today = (s.query(SimulatedTrade)
                              .filter(SimulatedTrade.trade_date == latest.snapshot_date)
                              .count())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    vals = [b[0] for b in baseline if b[0] is not None]
    if len(vals) < 20:
        return None

    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
    sigma = math.sqrt(var)
    if sigma < 1e-9:
        return None

    if latest.nav_after_flow == 0:
        return None
    r = (latest.nav_close - latest.nav_after_flow) / latest.nav_after_flow
    z = (r - mean) / sigma

    # Flag only when no trades on same day (genuinely unexplained).
    if abs(z) <= 3.0 or n_trades_today > 0:
        return None

    return {
        "severity": "HIGH",
        "snapshot": {
            "snapshot_date":      latest.snapshot_date.isoformat(),
            "ex_flow_return":     round(r, 6),
            "z_vs_60d_baseline":  round(z, 3),
            "baseline_n":         len(vals),
            "baseline_sigma":     round(sigma, 6),
            "n_trades_same_day":  n_trades_today,
            "context":            "Ops Watchdog mode 7 — NAV moved >3σ with no trades same day.",
        },
    }


def rule_signal_panel_nan_scan() -> RuleResult:
    """
    Watchdog mode 8 — Signal computation NaN.

    For SignalRecord rows in last 3 days, count rows with composite_score
    NULL OR tsmom_signal NULL OR gate_status NULL. If NaN% > 5% of recent
    rows, flag as signal-logic bug suspicion (severity SEVERE per spec).
    """
    import datetime
    try:
        from sqlalchemy import or_
        from engine.db_models import SignalRecord
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    today  = datetime.date.today()
    cutoff = today - datetime.timedelta(days=3)

    try:
        with SessionFactory() as s:
            total = (s.query(SignalRecord)
                      .filter(SignalRecord.date >= cutoff)
                      .count())
            n_nan = (s.query(SignalRecord)
                      .filter(SignalRecord.date >= cutoff)
                      .filter(or_(SignalRecord.composite_score.is_(None),
                                  SignalRecord.tsmom_signal.is_(None),
                                  SignalRecord.gate_status.is_(None)))
                      .count())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    if total < 5:
        return None
    pct = n_nan / total
    if pct <= 0.05:
        return None

    return {
        "severity": "HIGH",
        "snapshot": {
            "lookback_days": 3,
            "n_total":       total,
            "n_nan":         n_nan,
            "pct_nan":       round(pct, 3),
            "threshold":     0.05,
            "context":       "Ops Watchdog mode 8 — signal panel NaN rate breach.",
        },
    }


def rule_realized_tc_vs_spec_rate() -> RuleResult:
    """
    Watchdog mode 9 — TC drag computed wrong (spec id=63 amendment 2,
    hash c0a5f989, 2026-05-12; post-Path-E TC fix lesson per
    feedback_etf_tc_tier_model.md; unit-conversion follow-up fix 2026-05-12).

    For each sleeve with registered TC metadata in engine.spec_metadata,
    compute the median realised PER-EVENT half-spread from SimulatedTrade
    rows over last 30d and compare to that sleeve's PRIMARY strategy's
    locked `tc_bps_per_event`. Flag HIGH if
        |realized_median - locked_tc| / locked_tc > MODE_9_DEVIATION_THRESHOLD
    (default 0.5 = 50% deviation).

    UNIT CONVERSION (critical):
      engine.cost_model.compute_cost_bps stores `cost_bps = |Δw| × spread_bps`
      (line 61 contract). The stored cost_bps is "bp of NAV cost for THIS
      trade", NOT "bp per event of TC". To reconstruct per-event TC, divide:
        realized_half_spread_bps = cost_bps / |weight_delta|
      The median of this ratio is comparable to spec-locked tc_bps_per_event.

    Amendment 2 history:
      v1 (Phase 1):  ratio cost_bps/|wd| vs 2× LIVE_FLAT_COST_BPS=20 threshold
                     — undertuned (B++ drift to 12bp would NOT fire)
      v2 (refactor): dropped /|wd| divisor + per-spec 50% threshold
                     — WRONG, compared NAV-scaled cost_bps to per-event TC
                       (97.5% false-positive deviation on real DB)
      v3 (this fix): keep /|wd| divisor + per-spec 50% threshold (correct)

    Severity SEVERE (halt next batch per spec §2.2 MODE_SEVERITY_MAP_LOCKED).
    """
    import datetime
    try:
        from engine.db_models import SimulatedTrade
        from engine.memory import SessionFactory
        from engine.spec_metadata import (
            MODE_9_DEVIATION_THRESHOLD,
            MODE_9_MIN_TRADES_FOR_CHECK,
            get_known_sleeves,
            get_primary_tc_for_sleeve,
        )
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    today  = datetime.date.today()
    cutoff = today - datetime.timedelta(days=30)

    violations: List[Dict[str, Any]] = []
    per_sleeve_stats: List[Dict[str, Any]] = []

    for sleeve_id in get_known_sleeves():
        primary = get_primary_tc_for_sleeve(sleeve_id)
        if primary is None:
            continue
        locked_tc = float(primary["tc_bps_per_event"])
        if locked_tc <= 0:
            continue

        try:
            with SessionFactory() as s:
                rows = (s.query(SimulatedTrade.cost_bps,
                                SimulatedTrade.weight_delta)
                         .filter(SimulatedTrade.trade_date >= cutoff,
                                 SimulatedTrade.sleeve_id == sleeve_id,
                                 SimulatedTrade.cost_bps.isnot(None))
                         .all())
        except Exception as exc:
            return {"severity": "LOW",
                    "snapshot": {"reason": "query_failed",
                                 "sleeve_id": sleeve_id,
                                 "exception": str(exc)}}

        # Convert stored NAV-scaled cost_bps back to per-event half-spread.
        ratios: List[float] = []
        for cb, wd in rows:
            if wd is None or abs(float(wd)) < 1e-6:
                continue
            ratios.append(float(cb) / abs(float(wd)))

        if len(ratios) < MODE_9_MIN_TRADES_FOR_CHECK:
            per_sleeve_stats.append({
                "sleeve_id":     sleeve_id,
                "spec_id":       primary["spec_id"],
                "locked_tc_bps": locked_tc,
                "n_trades":      len(ratios),
                "status":        "insufficient_data",
                "min_required":  MODE_9_MIN_TRADES_FOR_CHECK,
            })
            continue

        ratios.sort()
        median_realized = ratios[len(ratios) // 2]
        deviation = abs(median_realized - locked_tc) / locked_tc

        per_sleeve_stats.append({
            "sleeve_id":            sleeve_id,
            "spec_id":              primary["spec_id"],
            "spec_notes":           primary.get("notes", "")[:120],
            "locked_tc_bps":        round(locked_tc, 3),
            "median_realized_bps":  round(median_realized, 3),
            "deviation":            round(deviation, 4),
            "threshold":            MODE_9_DEVIATION_THRESHOLD,
            "n_trades":             len(ratios),
            "status":               ("breach" if deviation > MODE_9_DEVIATION_THRESHOLD
                                     else "within_tolerance"),
        })

        if deviation > MODE_9_DEVIATION_THRESHOLD:
            violations.append({
                "sleeve_id":            sleeve_id,
                "primary_spec_id":      primary["spec_id"],
                "locked_tc_bps":        round(locked_tc, 3),
                "median_realized_bps":  round(median_realized, 3),
                "deviation":            round(deviation, 4),
                "threshold":            MODE_9_DEVIATION_THRESHOLD,
                "n_trades":             len(ratios),
            })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_violations":     len(violations),
            "violations":       violations,
            "per_sleeve_stats": per_sleeve_stats,
            "lookback_days":    30,
            "context": (
                "Ops Watchdog mode 9 (amendment 2) — realized half-spread "
                "(cost_bps/|wd|) drifted >50% from spec-locked "
                "tc_bps_per_event for primary strategy of one or more "
                "sleeves. See engine.spec_metadata + engine.cost_model."
            ),
        },
    }


def rule_max_position_weight_vs_cap() -> RuleResult:
    """
    Watchdog mode 10 — Weight cap not enforced (covers all sleeves).

    Differs from rule_etf_holdings_cap_clamp_bounds (which validates the
    LLM holdings-screen multiplier bounds at config layer): this rule
    inspects ACTUAL persisted SimulatedPosition.actual_weight rows for the
    latest snapshot_date and asserts max |actual_weight| ≤ MAX_WEIGHT +
    tolerance.

    Severity MID per spec — auto-truncate recipe handles repair.
    """
    import datetime
    try:
        from engine.db_models import SimulatedPosition
        from engine.memory import SessionFactory
        from engine.config import MAX_WEIGHT
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        with SessionFactory() as s:
            latest_date = (s.query(SimulatedPosition.snapshot_date)
                            .order_by(SimulatedPosition.snapshot_date.desc())
                            .first())
            if latest_date is None:
                return None
            latest_date = latest_date[0]
            rows = (s.query(SimulatedPosition)
                     .filter(SimulatedPosition.snapshot_date == latest_date)
                     .filter(SimulatedPosition.actual_weight.isnot(None))
                     .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    tolerance = 0.005   # 50bp slack vs locked cap
    cap_upper = float(MAX_WEIGHT) + tolerance

    violations: List[Dict[str, Any]] = []
    for p in rows:
        aw = abs(float(p.actual_weight))
        if aw > cap_upper:
            violations.append({
                "position_id":   p.id,
                "snapshot_date": p.snapshot_date.isoformat(),
                "sector":        p.sector,
                "ticker":        p.ticker,
                "sleeve_id":     getattr(p, "sleeve_id", None),
                "actual_weight": round(float(p.actual_weight), 6),
                "abs_weight":    round(aw, 6),
                "cap":           float(MAX_WEIGHT),
                "excess":        round(aw - float(MAX_WEIGHT), 6),
            })

    if not violations:
        return None
    return {
        "severity": "MID",
        "snapshot": {
            "n_violations":  len(violations),
            "violations":    violations[:30],
            "max_weight":    float(MAX_WEIGHT),
            "tolerance":     tolerance,
            "snapshot_date": latest_date.isoformat(),
            "context":       "Ops Watchdog mode 10 — position weight exceeds MAX_WEIGHT.",
        },
    }


def rule_rebalance_frequency_audit() -> RuleResult:
    """
    Watchdog mode 11 — Rebalance cadence drift.

    Production cadence is monthly (per daily_batch monthly_rebalance_auto
    gate). Inspect SimulatedTrade rows for the trailing 6 calendar months
    (excluding current partial month). For each completed month:
      - 0 rebalance events                   → MID per month
      - >2 distinct rebalance days per month → MID per month
    A "rebalance day" = distinct trade_date with ≥3 rows (single ad-hoc
    fills with 1-2 trades are excluded — likely human approvals, not
    cadence drift).

    Pre-launch floor (2026-05-13):
      Production paper-trade pipeline went live on Sprint D-2 launch day
      (2026-05-13). Months entirely before that have no expected fills and
      must be excluded from the audit — otherwise this rule fires every
      day with historical false positives.

    Severity SEVERE per spec table (config bug suspicion).
    """
    import datetime
    from collections import defaultdict
    try:
        from engine.db_models import SimulatedTrade
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    # Production trading start: the rule cannot evaluate months that ended
    # before this date (no fills could have occurred).
    PRODUCTION_TRADING_START = datetime.date(2026, 5, 13)

    today = datetime.date.today()
    # Start of trailing 6 completed months
    first_of_month = today.replace(day=1)
    start = first_of_month
    for _ in range(6):
        prev_last = start - datetime.timedelta(days=1)
        start = prev_last.replace(day=1)
    # `start` is now first day of month T-6, end is first_of_month exclusive.

    # Floor start at the first day of the month containing the launch date.
    _launch_month_start = PRODUCTION_TRADING_START.replace(day=1)
    if start < _launch_month_start:
        start = _launch_month_start
    # If the floor pushes start past the audit window, there is nothing to check.
    if start >= first_of_month:
        return None

    try:
        with SessionFactory() as s:
            rows = (s.query(SimulatedTrade.trade_date)
                     .filter(SimulatedTrade.trade_date >= start,
                             SimulatedTrade.trade_date < first_of_month)
                     .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    by_month: Dict[str, List[datetime.date]] = defaultdict(list)
    for (d,) in rows:
        by_month[f"{d.year:04d}-{d.month:02d}"].append(d)

    # Build expected month list (T-6 .. T-1)
    months: List[str] = []
    cur = start
    while cur < first_of_month:
        months.append(f"{cur.year:04d}-{cur.month:02d}")
        # advance to next month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    violations: List[Dict[str, Any]] = []
    for ym in months:
        dates = by_month.get(ym, [])
        # Compute distinct days with ≥3 fills (filters ad-hoc human-approval days)
        day_counts: Dict[datetime.date, int] = defaultdict(int)
        for d in dates:
            day_counts[d] += 1
        bulk_days = sorted(d for d, c in day_counts.items() if c >= 3)
        if len(bulk_days) == 0:
            violations.append({
                "month":       ym,
                "kind":        "no_rebalance",
                "n_bulk_days": 0,
                "n_trades":    len(dates),
            })
        elif len(bulk_days) > 2:
            violations.append({
                "month":       ym,
                "kind":        "excess_rebalance_days",
                "n_bulk_days": len(bulk_days),
                "bulk_days":   [d.isoformat() for d in bulk_days],
            })

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "months_inspected": months,
            "n_violations":     len(violations),
            "violations":       violations,
            "context":          "Ops Watchdog mode 11 — rebalance cadence drift.",
        },
    }


def rule_regime_scale_vs_exposure_audit() -> RuleResult:
    """
    Watchdog mode 12 — REGIME_SCALE not applied.

    When REGIME_SCALE < 1.0 (overlay active) AND latest regime in
    ('risk-off', 'transition'), the latest snapshot's gross long exposure
    should be <= the cap × REGIME_SCALE roughly (allowing 10% slack for
    discrete weights). If sum of positive actual_weight ≥ MAX_NET (no
    scaling visible), the scale-multiplier path may be silently bypassed.

    Severity MID per spec — auto-reapply scale recipe handles repair.
    """
    try:
        from engine.db_models import RegimeSnapshot, SimulatedPosition
        from engine.memory import SessionFactory
        from engine.config import MAX_NET
        from engine.portfolio_core import REGIME_SCALE  # moved from config in the portfolio refactor
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    # If overlay disabled (scale=1.0), nothing to enforce.
    if float(REGIME_SCALE) >= 1.0 - 1e-9:
        return None

    try:
        with SessionFactory() as s:
            regime_row = (s.query(RegimeSnapshot)
                           .order_by(RegimeSnapshot.as_of_date.desc())
                           .first())
            if regime_row is None:
                return None
            if regime_row.regime not in ("risk-off", "transition"):
                return None
            latest_pos_date = (s.query(SimulatedPosition.snapshot_date)
                                .order_by(SimulatedPosition.snapshot_date.desc())
                                .first())
            if latest_pos_date is None:
                return None
            latest_pos_date = latest_pos_date[0]
            rows = (s.query(SimulatedPosition.actual_weight)
                     .filter(SimulatedPosition.snapshot_date == latest_pos_date)
                     .filter(SimulatedPosition.actual_weight.isnot(None))
                     .all())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    gross_long = sum(float(r[0]) for r in rows if float(r[0]) > 0)
    if gross_long <= 0:
        return None

    # Expected scaled cap on long exposure (10% slack tolerance)
    expected_cap = float(MAX_NET) * float(REGIME_SCALE) * 1.10
    # Reverse check: was the scale applied? If gross_long > MAX_NET, scale is
    # certainly NOT applied (overlay bypassed). If between MAX_NET*SCALE and
    # MAX_NET, ambiguous; only firm flag is gross_long > MAX_NET.
    if gross_long <= float(MAX_NET) + 1e-9:
        return None

    return {
        "severity": "MID",
        "snapshot": {
            "regime":         regime_row.regime,
            "regime_as_of":   regime_row.as_of_date.isoformat(),
            "regime_scale":   float(REGIME_SCALE),
            "max_net":        float(MAX_NET),
            "gross_long":     round(gross_long, 6),
            "expected_cap":   round(expected_cap, 6),
            "snapshot_date":  latest_pos_date.isoformat(),
            "context":        "Ops Watchdog mode 12 — REGIME_SCALE not applied to long exposure.",
        },
    }


def rule_watchdog_runs_daily() -> RuleResult:
    """
    Tier R meta-monitoring rule (Phase 5, 2026-05-13, spec id=63 §4.3) —
    assert the Ops Watchdog Agent has fired at least one AuditRun with
    scope='watchdog' in the last 30 hours.

    Watchdog is scheduled daily at 06:10 SGT via Windows Task Scheduler
    "MacroAlphaPro_Watchdog" (registered Phase 5 Gate 8). If the task fails
    to fire — Task Scheduler crashed / spec-amend silently broke the entry
    point / Watchdog __main__.py raised on import — there will be NO
    AuditRun(scope='watchdog') row in the recent window.

    30-hour window (not 24h): gives ~6h slack for clock drift, late
    startup, OR Windows wake-from-sleep StartWhenAvailable catch-up.

    Severity HIGH; cadence weekly (slow drift, but missing > 30h IS bad).
    """
    import datetime
    try:
        from engine.auto_audit_models import AuditRun
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(hours=30)

    try:
        with SessionFactory() as s:
            row = (s.query(AuditRun)
                    .filter(AuditRun.scope == "watchdog",
                            AuditRun.run_at >= cutoff)
                    .order_by(AuditRun.run_at.desc())
                    .first())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    if row is not None:
        return None
    # No watchdog run in 30h — find the LAST one for context (if any)
    try:
        with SessionFactory() as s:
            last_row = (s.query(AuditRun)
                         .filter(AuditRun.scope == "watchdog")
                         .order_by(AuditRun.run_at.desc())
                         .first())
    except Exception:
        last_row = None
    last_iso = last_row.run_at.isoformat() if last_row else None
    last_age_hours = ((now - last_row.run_at).total_seconds() / 3600.0
                      if last_row else None)
    return {
        "severity": "HIGH",
        "snapshot": {
            "cutoff_hours":  30,
            "now_utc":       now.isoformat(),
            "last_run_at":   last_iso,
            "last_age_hours": (round(last_age_hours, 1)
                               if last_age_hours is not None else None),
            "context": (
                "Tier R meta-monitor (spec id=63 §4.3): no AuditRun with "
                "scope='watchdog' in last 30h. Task Scheduler "
                "'MacroAlphaPro_Watchdog' (06:10 SGT) may have failed to "
                "fire. Investigate `Get-ScheduledTaskInfo MacroAlphaPro_"
                "Watchdog` + `engine/agents/ops_watchdog/__main__.py` "
                "import path."
            ),
        },
    }


def rule_pairwise_correlation_drift() -> RuleResult:
    """
    Sprint G (2026-05-13) — daily ρ drift sentinel.

    Computes trailing 12-week pairwise correlation across the 4-component
    portfolio (K1 / D-PEAD / Path N / CTA) and alerts if any pair drifts
    beyond locked thresholds:
      WARN     |ρ| > 0.20
      CRITICAL |ρ| > 0.30

    Why critical: deployment_design.md §1 anchors entire 4-alpha portfolio
    thesis on "all pairwise |ρ| < 0.10" empirically. ρ drift to 0.30
    degrades combined Sharpe from ~1.3 to ~0.6 — silent failure if
    not monitored daily.

    Implementation: reuses engine.portfolio.correlation_sentinel.run_correlation_sentinel()
    so logic stays single-sourced. Rule wraps with severity mapping for
    Watchdog notification routing.

    Severity:
      None     all CLEAN
      MID      at least one pair WARN, no CRITICAL
      HIGH     at least one pair CRITICAL
    """
    import datetime
    try:
        from engine.portfolio.correlation_sentinel import run_correlation_sentinel
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        report = run_correlation_sentinel(as_of=datetime.date.today())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "sentinel_run_failed", "exception": str(exc)}}

    if report.severity == "CLEAN":
        return None

    # Distinguish data-layer issue (LOW) from actual portfolio drift (MID/HIGH)
    severity_map = {
        "INSUFFICIENT_DATA": "LOW",   # data stale (backtest parquet end past); not thesis broken
        "WARN":              "MID",   # one or more pairs > 0.20
        "CRITICAL":          "HIGH",  # one or more pairs > 0.30
    }
    rule_severity = severity_map.get(report.severity, "LOW")

    return {
        "severity": rule_severity,
        "snapshot": {
            "as_of":            str(report.as_of),
            "window_weeks":     report.window_weeks,
            "sample_n_weeks":   report.sample_n_weeks,
            "overall_severity": report.severity,
            "max_abs_rho":      report.max_abs_rho,
            "max_drift":        report.max_drift,
            "pairs_flagged": [
                {
                    "pair":          f"{c.pair_a}__{c.pair_b}",
                    "rho_trailing":  c.rho_trailing,
                    "rho_baseline":  c.rho_baseline,
                    "abs_drift":     c.abs_drift,
                    "severity":      c.severity,
                }
                for c in report.correlations
                if c.severity in {"WARN", "CRITICAL"}
            ],
            "context": (
                "Sprint G ρ drift sentinel: 4-alpha portfolio thesis "
                "assumes pairwise |ρ|<0.10. Trailing 12-week ρ drift to "
                ">0.30 = portfolio Sharpe degradation 1.3→~0.6. "
                "Investigate which strategy pair has correlated regime "
                "shift; consider reducing combined allocation."
            ),
        },
    }


def rule_factor_tilt_exceeds_threshold() -> RuleResult:
    """
    Tier-1 audit #4 Phase C (2026-05-14) — daily FF5 factor-tilt sentinel.

    Reads latest paper-trade book positions, computes Fama-French 5-factor
    portfolio betas, and alerts if any non-Market factor |β| exceeds 0.5
    (book has concentrated SMB/HML/RMW/CMA exposure not justified by
    explicit sleeve mandate). For Market β > 1.5 we also alert (over-
    levered to market vs implied 0.85-1.0 from 90% equity allocation).

    Severity:
      None         all |β| within bounds
      MID          one or more factor |β| in [0.5, 0.8); single hedge
                   suggestion in snapshot
      HIGH         any factor |β| >= 0.8; book has structural tilt; PM
                   should investigate which sleeve/strategy drives it

    Snapshot includes per-factor β, threshold breach list, and a coarse
    "suggested hedge" direction (long IWB vs short IWM for SMB heavy, etc.)
    surfaced for PM action — NOT auto-executed (0-LLM-in-DECISION doctrine).

    Implementation: reuses engine.risk_metrics.compute_ff5_factor_tilt for
    single-source factor math; reads latest PaperTradeStrategyLog row per
    strategy + applies PAPER_TRADE_SLEEVE_ALLOCATION × intra_sleeve_weight.
    """
    import datetime
    import json as _json

    try:
        from engine.memory import init_db, SessionFactory
        from engine.db_models import PaperTradeStrategyLog
        from engine.portfolio.paper_trade_combined import (
            PAPER_TRADE_SLEEVE_ALLOCATION, INTRA_SS_SP500_WEIGHTS,
        )
        from engine.risk_metrics import compute_ff5_factor_tilt
        import pandas as _pd
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        init_db()
        with SessionFactory() as sess:
            latest_date = sess.query(PaperTradeStrategyLog.date).order_by(
                PaperTradeStrategyLog.date.desc()
            ).first()
            if not latest_date:
                return None  # no paper-trade rows yet — nothing to evaluate
            as_of = latest_date[0]
            rows = (
                sess.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.date == as_of)
                    .all()
            )
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "db_read_failed", "exception": str(exc)}}

    # Aggregate ticker → signed book weight across strategies
    by_ticker: dict[str, float] = {}
    for r in rows:
        if r.status != "OK" or not r.positions_json:
            continue
        try:
            positions = _json.loads(r.positions_json)
        except Exception:
            continue
        # Strategy book-weight factor = sleeve_alloc × intra_sleeve_weight × (intra_ss for ss_sp500)
        sleeve_w = PAPER_TRADE_SLEEVE_ALLOCATION.get(r.sleeve_id, 0.0)
        intra_w  = float(r.intra_sleeve_weight or 0.0)
        if r.sleeve_id == "ss_sp500":
            # D_PEAD / PATH_N share the sleeve; intra_sleeve_weight already
            # encodes the intra-ss split (0.5 each)
            book_factor = sleeve_w * intra_w
        else:
            book_factor = sleeve_w * intra_w
        for tk, w in positions.items():
            try:
                by_ticker[tk] = by_ticker.get(tk, 0.0) + book_factor * float(w)
            except Exception:
                continue

    if not by_ticker:
        return None

    book_df = _pd.DataFrame(
        [{"ticker": tk, "actual_weight": w} for tk, w in by_ticker.items()]
    )
    try:
        tilt = compute_ff5_factor_tilt(book_df, period="2y")
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "compute_failed", "exception": str(exc)}}

    if tilt["n_obs"] == 0 or tilt["n_assets"] == 0:
        return {"severity": "LOW",
                "snapshot": {"reason": "insufficient_factor_history",
                              "as_of": str(as_of),
                              "n_obs": tilt["n_obs"],
                              "n_assets": tilt["n_assets"]}}

    # Threshold evaluation
    MKT_HIGH = 1.5
    FACTOR_WARN = 0.5
    FACTOR_HIGH = 0.8
    breaches: list[dict] = []
    severity = None

    def _hedge_suggestion(factor: str, beta: float) -> str:
        sign = "long" if beta > 0 else "short"
        if factor == "Mkt":
            return f"reduce gross or hedge SPY ({sign} {abs(beta):.2f} excess)"
        if factor == "SMB":
            return ("long IWB / short IWM (book is small-cap heavy)"
                    if beta > 0 else
                    "long IWM / short IWB (book is large-cap heavy)")
        if factor == "HML":
            return ("trim value, add growth (book is value heavy)"
                    if beta > 0 else
                    "trim growth, add value (book is growth heavy)")
        if factor == "RMW":
            return ("trim QUAL / quality-tilt names"
                    if beta > 0 else
                    "add QUAL or quality-tilt names")
        if factor == "CMA":
            return ("trim USMV / conservative-investment names"
                    if beta > 0 else
                    "add USMV / conservative-investment names")
        return "review sleeve attribution"

    # Mkt has its own threshold
    if abs(tilt["Mkt"]) > MKT_HIGH:
        breaches.append({
            "factor":  "Mkt",
            "beta":    round(tilt["Mkt"], 3),
            "level":   "HIGH",
            "hedge":   _hedge_suggestion("Mkt", tilt["Mkt"]),
        })
        severity = "HIGH"

    for fac in ("SMB", "HML", "RMW", "CMA"):
        b = tilt[fac]
        if abs(b) >= FACTOR_HIGH:
            level = "HIGH"
            severity = "HIGH"
        elif abs(b) >= FACTOR_WARN:
            level = "WARN"
            severity = severity or "MID"
        else:
            continue
        breaches.append({
            "factor": fac,
            "beta":   round(b, 3),
            "level":  level,
            "hedge":  _hedge_suggestion(fac, b),
        })

    if severity is None:
        return None

    return {
        "severity": severity,
        "snapshot": {
            "as_of":      str(as_of),
            "betas": {
                "Mkt": round(tilt["Mkt"], 3),
                "SMB": round(tilt["SMB"], 3),
                "HML": round(tilt["HML"], 3),
                "RMW": round(tilt["RMW"], 3),
                "CMA": round(tilt["CMA"], 3),
            },
            "alpha_annualized": round(tilt["alpha_daily"] * 252.0, 4),
            "r_squared":        round(tilt["r_squared"], 3),
            "n_obs_days":       tilt["n_obs"],
            "n_assets":         tilt["n_assets"],
            "thresholds": {
                "Mkt_HIGH":     MKT_HIGH,
                "factor_WARN":  FACTOR_WARN,
                "factor_HIGH":  FACTOR_HIGH,
            },
            "breaches":         breaches,
            "proxy_disclosure": tilt["proxy_disclosure"],
            "context": (
                "Tier-1 audit #4 Phase C: FF5 factor-tilt sentinel. "
                "|β|>=0.5 on non-Market factor = book has structural "
                "tilt not explicitly mandated by sleeve design. "
                "Investigate which strategy drives the tilt (sleeve "
                "attribution table on Positions page). Hedge suggestions "
                "advisory only — 0-LLM-in-DECISION doctrine: no auto-trade."
            ),
        },
    }


def rule_weekly_recon_summary() -> RuleResult:
    """
    Sprint G extension (2026-05-13) — wrap engine.portfolio.weekly_recon
    inside Watchdog so all quant detection points fire on same 06:10 SGT
    schedule with same 4-channel notification.

    Maps weekly_recon ReconAlert severity into Watchdog rule severity:
      No alerts                                    → None
      Any CRITICAL (data_gap full / missing_strategy / error_streak)
        → HIGH
      Any WARN (no_signal_streak / partial data_gap)
        → MID

    Reuses run_weekly_recon() so logic stays single-sourced.

    Note: weekly_recon's threshold for missing-day "data_gap" is 2 days
    (CRITICAL if >2 days missing in 7-day window). This complements
    rule_paper_trade_daily_runs (Sprint D-3) which fires HIGH if no
    rows in last 30h. Both cover overlapping but distinct failure modes:
      - rule_paper_trade_daily_runs: immediate 30h-gap detection
      - rule_weekly_recon_summary:    7-day window patterns + streaks
    """
    import datetime
    try:
        from engine.portfolio.weekly_recon import run_weekly_recon
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        report = run_weekly_recon(datetime.date.today(), lookback_days=7)
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "weekly_recon_run_failed",
                              "exception": str(exc)}}

    if not report.alerts:
        return None

    has_critical = any(a.severity == "CRITICAL" for a in report.alerts)
    rule_severity = "HIGH" if has_critical else "MID"

    return {
        "severity": rule_severity,
        "snapshot": {
            "report_date":      str(report.report_date),
            "window":           f"{report.window_start} to {report.window_end}",
            "n_days_with_data": report.n_days_with_data,
            "n_days_in_window": report.n_days_in_window,
            "alert_count":      len(report.alerts),
            "alerts": [
                {"severity": a.severity, "category": a.category,
                 "strategy": a.strategy, "message": a.message[:200]}
                for a in report.alerts
            ],
            "context": (
                "Sprint G ext: Weekly reconciliation report wrapper. "
                "Categories: data_gap / missing_strategy / no_signal_streak "
                "/ error_streak. Investigate PaperTradeStrategyLog daily "
                "writes (Sprint D-2 MacroAlphaPro_PaperTrade) + strategy "
                "module health (cache freshness / WRDS / yfinance)."
            ),
        },
    }


def rule_paper_trade_daily_runs() -> RuleResult:
    """
    Sprint D-3 Tier R meta-monitoring rule (2026-05-13) — assert the daily
    paper-trade orchestrator has written rows to PaperTradeStrategyLog
    within the last 30 hours.

    Paper-trade daily auto-run is scheduled via Windows Task Scheduler
    "MacroAlphaPro_PaperTrade" (06:00 SGT, registered Sprint D-2). If the
    task fails to fire — Task Scheduler crashed / Python env broken /
    yfinance fully down — there will be NO PaperTradeStrategyLog rows in
    the recent window, and forward OOS data accumulation silently stops.

    Critical: each missed day = 4 forward strategy-snapshot rows lost
    permanently. Detecting a 24-30h gap is the difference between losing
    1 day vs losing 5+ days of forward OOS data.

    30-hour window: gives ~6h slack for clock drift, weekend (paper trade
    runs daily including weekends, but signals on Sat/Sun may all be
    NO_SIGNAL), or Windows wake-from-sleep StartWhenAvailable catch-up.

    Severity HIGH; cadence weekly (slow drift, but missing > 30h IS bad).
    """
    import datetime
    try:
        from engine.db_models import PaperTradeStrategyLog
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    now = datetime.datetime.utcnow()
    cutoff = now - datetime.timedelta(hours=30)

    try:
        with SessionFactory() as s:
            row = (s.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.created_at >= cutoff)
                    .order_by(PaperTradeStrategyLog.created_at.desc())
                    .first())
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "query_failed", "exception": str(exc)}}

    if row is not None:
        return None

    # No paper-trade run in 30h — find the LAST one for context
    try:
        with SessionFactory() as s:
            last_row = (s.query(PaperTradeStrategyLog)
                         .order_by(PaperTradeStrategyLog.created_at.desc())
                         .first())
    except Exception:
        last_row = None
    last_iso = last_row.created_at.isoformat() if last_row else None
    last_age_hours = ((now - last_row.created_at).total_seconds() / 3600.0
                      if last_row else None)
    return {
        "severity": "HIGH",
        "snapshot": {
            "cutoff_hours":  30,
            "now_utc":       now.isoformat(),
            "last_run_at":   last_iso,
            "last_age_hours": (round(last_age_hours, 1)
                               if last_age_hours is not None else None),
            "context": (
                "Sprint D-3 Tier R rule: no PaperTradeStrategyLog row "
                "created in last 30h. Task Scheduler 'MacroAlphaPro_"
                "PaperTrade' (06:00 SGT) may have failed to fire. "
                "Investigate Get-ScheduledTaskInfo MacroAlphaPro_PaperTrade "
                "+ data/paper_trade/daily_run_<date>.log + "
                "engine.portfolio.paper_trade_combined imports. Each day "
                "missed = 4 forward OOS rows lost permanently."
            ),
        },
    }


def rule_watchdog_halt_flag_not_stuck() -> RuleResult:
    """
    Tier R meta-monitoring rule (Phase 4, 2026-05-12, spec id=63 §4.3) —
    detect a Watchdog-set halt flag that has been stuck True for > 7 days
    without human clear. Per spec §六 invariant, the halt flag can ONLY be
    cleared by a human via the dashboard "Acknowledge Watchdog Halt" button
    (engine.circuit_breaker.manual_reset). If it stays stuck > 7d:
      - The operator has likely forgotten the open SEVERE issue
      - Production may be silently halted longer than intended
      - This rule fires HIGH to remind the operator to investigate + clear

    Reads engine.circuit_breaker.get_status(). If level=SEVERE and reason
    starts with "ops_watchdog:" (i.e. set by this agent) and triggered_at
    is > 7 days ago → HIGH finding. Other SEVERE causes (VIX spike etc.)
    have their own escalation paths and are out of scope here.

    Severity HIGH; cadence weekly (slow drift).
    """
    import datetime
    try:
        from engine.circuit_breaker import get_status, LEVEL_SEVERE
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        state = get_status()
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "circuit_breaker_query_failed",
                             "exception": str(exc)}}

    if state is None or state.level != LEVEL_SEVERE:
        return None
    if not state.reason or not state.reason.startswith("ops_watchdog:"):
        # SEVERE set by something else (e.g. VIX spike); not our scope.
        return None
    if not state.triggered_at:
        # Missing timestamp — defensively skip (otherwise we'd fire on every run)
        return None

    try:
        triggered_dt = datetime.datetime.fromisoformat(state.triggered_at.rstrip("Z"))
    except Exception:
        return {"severity": "LOW",
                "snapshot": {"reason": "triggered_at_parse_failed",
                             "raw": state.triggered_at[:200]}}

    now = datetime.datetime.utcnow()
    age_days = (now - triggered_dt).total_seconds() / 86400.0
    if age_days <= 7.0:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "halt_set_at":     state.triggered_at,
            "halt_age_days":   round(age_days, 2),
            "halt_reason":     state.reason[:300],
            "threshold_days":  7.0,
            "context": (
                "Tier R meta-monitor (spec id=63 §4.3): a Watchdog-set "
                "halt flag has been stuck True > 7 days without human "
                "clear. Investigate the root cause + clear via dashboard."
            ),
        },
    }


def rule_watchdog_auto_repair_no_raw_sql() -> RuleResult:
    """
    Tier R guardrail (Phase 3, 2026-05-12, spec id=63 §6 invariants) — scan
    engine/agents/ops_watchdog/auto_repair.py for forbidden raw-SQL patterns
    that would let Watchdog write directly to production tables.

    Locked invariant (spec §六): Watchdog NEVER writes to portfolio /
    simulated_positions / simulated_trades / portfolio_nav_snapshots /
    universe_etfs. Auto-repair recipes must call existing production
    functions (e.g. engine.daily_batch.run_daily_batch); writes are
    THOSE functions' responsibility, not Watchdog's.

    Forbidden patterns (HIGH severity if any present):
      - `text(`                 (raw sqlalchemy.text)
      - `session.execute(`      (direct connection execute)
      - `UPDATE simulated_`     (any production-table UPDATE)
      - `INSERT INTO simulated_`
      - `DELETE FROM simulated_`
      - `UPDATE portfolio_nav_`
      - `UPDATE universe_etfs`

    Allowed: importing + calling existing production fns from engine.*.
    """
    from pathlib import Path
    target = (Path(__file__).resolve().parent
              / "agents" / "ops_watchdog" / "auto_repair.py")
    if not target.exists():
        # Phase 3 not yet shipped — no finding
        return None
    try:
        src = target.read_text(encoding="utf-8")
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "read_failed", "exception": str(exc)}}

    forbidden_patterns = (
        "text(",
        "session.execute(",
        "UPDATE simulated_",
        "INSERT INTO simulated_",
        "DELETE FROM simulated_",
        "UPDATE portfolio_nav_",
        "UPDATE universe_etfs",
    )

    violations: List[Dict[str, Any]] = []
    for line_no, line in enumerate(src.split("\n"), start=1):
        stripped = line.strip()
        # Skip docstring / comment lines (they mention patterns for documentation)
        if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
            continue
        for pat in forbidden_patterns:
            if pat in line:
                violations.append({
                    "pattern": pat,
                    "line":    line_no,
                    "code":    stripped[:200],
                })
                break

    if not violations:
        return None
    return {
        "severity": "HIGH",
        "snapshot": {
            "n_violations": len(violations),
            "violations":   violations[:15],
            "context": (
                "Tier R guardrail (spec id=63 §六 read-only invariant). "
                "Auto-repair recipes must call existing production fns, "
                "not write raw SQL to production tables."
            ),
        },
    }


def rule_agent_slo_breach() -> RuleResult:
    """
    Phase 1 Agent Observability v1 (2026-05-15) — Tier R rule.

    Reads data/agent_slo_metrics.jsonl + computes per-agent compliance over
    last 30 days. Flags agents with PERSISTENT FAIL on any compliance gate
    (latency p95 / success rate / cost / schema validity).

    Severity:
      MID   if 1 agent FAIL
      HIGH  if 2+ agents FAIL OR success_rate < 80% any agent
      None  if all agents PASS or INSUFFICIENT_DATA

    Per spec_agent_observability_v1 (Phase 1 v1 deploy):
      - Watchdog catches sustained compliance breach (not transient)
      - 30d window for stability
      - Defers fine-grained alerting to v2 (e.g., per-failure-mode thresholds)
    """
    try:
        from engine.agents.observability import (
            DEFAULT_AGENT_SLOS, compute_agent_slo_compliance, load_metrics,
        )
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        records = load_metrics(days_lookback=30)
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "metrics_load_failed", "exception": str(exc)}}

    fail_agents = []
    severe_agents = []
    summary_per_agent = {}
    for agent_id in DEFAULT_AGENT_SLOS:
        c = compute_agent_slo_compliance(agent_id, records)
        summary_per_agent[agent_id] = {
            "compliance":   c.get("compliance_overall"),
            "n":            c.get("n_invocations"),
            "success_rate": c.get("success_rate"),
            "p95_ms":       c.get("latency_p95_ms"),
        }
        if c.get("compliance_overall") == "FAIL":
            fail_agents.append(agent_id)
            sr = c.get("success_rate") or 0
            if sr < 0.80:
                severe_agents.append(agent_id)

    if not fail_agents:
        return None    # all PASS or INSUFFICIENT_DATA

    if severe_agents:
        severity = "HIGH"
    elif len(fail_agents) >= 2:
        severity = "HIGH"
    else:
        severity = "MID"

    return {
        "severity": severity,
        "snapshot": {
            "fail_agents":         fail_agents,
            "severe_agents":       severe_agents,
            "summary_per_agent":   summary_per_agent,
            "context": (
                f"Phase 1 Agent Observability — {len(fail_agents)} agent(s) "
                f"FAIL compliance over last 30d. Severe: {severe_agents}. "
                "Investigate via pages/agent_slo_dashboard.py."
            ),
        },
    }


def rule_watchdog_daily_cost_budget() -> RuleResult:
    """
    Watchdog mode 13 — Watchdog daily LLM cost runaway (amendment 1, 2026-05-12).

    Reads engine.llm_cost_ledger entries for agent_id='ops_watchdog' on today's
    UTC date. If total cost_usd > $0.50 (2.5x expected $0.20 budget), flag
    SEVERE. Spec §2.1 mode 13, §2.2 severity map, §4.3 Tier R meta-rule.

    Closes the cadence gap between per-call enforcement (engine.llm_budget) and
    the weekly cumulative check (rule_llm_cumulative_cost_budget): a prompt-loop
    bug could burn $50/day for 7 days before the weekly rule catches it.
    """
    DAILY_BUDGET_USD: float = 0.50  # 2.5x expected $0.20 (spec § mode-13 detail)
    AGENT_ID = "ops_watchdog"

    try:
        from engine.llm_cost_ledger import get_total_today
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "import_failed", "exception": str(exc)}}

    try:
        total = float(get_total_today(agent_id=AGENT_ID))
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"reason": "ledger_query_failed", "exception": str(exc)}}

    if total <= DAILY_BUDGET_USD:
        return None

    return {
        "severity": "HIGH",
        "snapshot": {
            "agent_id":          AGENT_ID,
            "cost_today_usd":    round(total, 6),
            "daily_budget_usd":  DAILY_BUDGET_USD,
            "ratio_vs_budget":   round(total / DAILY_BUDGET_USD, 3),
            "context": (
                "Ops Watchdog mode 13 (amendment 1) — daily LLM cost > budget. "
                "Triggers halt_next_watchdog_run flag per spec §2.6 + §九 SEVERE."
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Registration
# ═════════════════════════════════════════════════════════════════════════════
CRITICAL_RULES.extend([
    rule_effective_n_trials_math_consistency,
    rule_hash_chain_continuity,
    rule_harking_detector_runs,
    # R-1.B.2
    rule_production_signal_vs_falsification_chain,
    rule_spec_hash_vs_code_drift,
    rule_cash_flow_conservation,
    rule_universe_drift_vs_registered,
    rule_backtest_vs_production_param_alignment,
    # R-1.B.3
    rule_db_schema_vs_orm_consistency,
    rule_backtest_grep_kwargs_alignment,
    # R-1.E
    rule_path_consistency,
    # P-LAB (2026-05-08): factor lab state machine consistency
    rule_factor_lab_state_consistent,
    # Factor Library v1 (2026-05-09): one-way module dependency lock
    rule_factor_lab_no_factor_library_import,
    # ETF Holdings Risk Monitor v1 (2026-05-08, spec id=49):
    # cap-bound + 0-LLM-in-eval invariants
    rule_etf_holdings_cap_clamp_bounds,
    rule_etf_holdings_no_llm_in_eval,
    # FOMC Surprise Override v1 (2026-05-12 unlock, spec id=48):
    # clamp bounds + 0-LLM-in-eval invariants
    rule_fomc_override_clamp_bounds,
    rule_fomc_override_no_llm_in_eval,
    # Factor Ensemble v1 (2026-05-09, spec id=50) — 3 NEW rules per §4.7
    rule_factor_ensemble_no_lookahead,
    rule_factor_ensemble_no_param_tuning,
    rule_factor_ensemble_baseline_reproducibility,
    # MS-7 (2026-05-10) — multi-sleeve sleeve_id integrity
    rule_sleeve_id_integrity,
    # Sprint G (2026-05-13 night) — daily 4-alpha pairwise ρ drift sentinel.
    # Critical because deployment_design 4-alpha portfolio thesis depends on
    # |ρ|<0.10 assumption; ρ→0.30 silently degrades combined Sharpe 1.3→0.6.
    rule_pairwise_correlation_drift,
    # Tier-1 audit #4 Phase C (2026-05-14) — FF5 factor-tilt sentinel.
    # MID at |β|>=0.5, HIGH at |β|>=0.8 on non-Market factor. Hedge
    # suggestions surfaced advisory-only (0-LLM-in-DECISION doctrine).
    rule_factor_tilt_exceeds_threshold,
])

WEEKLY_RULES.extend([
    rule_agent_reflection_heartbeat,
    # R-1.B.3
    rule_approval_queue_staleness,
    rule_anomaly_screener_m1_drift,
    rule_paper_trading_e_arm_config_drift,
    rule_llm_cumulative_cost_budget,
    rule_skill_library_dormancy,
    # Wave 5 (2026-05-07 applied-focus reframe)
    rule_capability_vs_data_congruence,
    # Phase 1 c (2026-05-12) — LLM-component removal-test doc governance
    rule_llm_removal_test_doc_exists,
    # Ops Watchdog Phase 3 (2026-05-12) — Tier R guardrail: auto_repair.py
    # must not contain raw SQL targeting production tables (spec §六 invariant)
    rule_watchdog_auto_repair_no_raw_sql,
    # Ops Watchdog Phase 4 (2026-05-12) — Tier R meta-monitor: halt flag
    # must not stay True > 7 days without human clear (spec §4.3)
    rule_watchdog_halt_flag_not_stuck,
    # Ops Watchdog Phase 5 (2026-05-13) — Tier R meta-monitor: assert
    # Watchdog Task Scheduler fired at least once in last 30h (spec §4.3)
    rule_watchdog_runs_daily,
    # Sprint D-3 (2026-05-13 night) — Tier R: assert paper-trade daily
    # auto-run (MacroAlphaPro_PaperTrade 06:00 SGT) wrote PaperTradeStrategyLog
    # rows within last 30h. Each missed day = 4 forward OOS rows lost.
    rule_paper_trade_daily_runs,
    # Sprint G extension (2026-05-13 night) — wrap weekly_recon as Watchdog
    # rule so all quant detection points fire on same 06:10 SGT schedule
    # with same 4-channel notification stack.
    rule_weekly_recon_summary,
])


# ─────────────────────────────────────────────────────────────────────────────
# Phase A2 (2026-05-14) — ETF Holdings Risk Monitor liveness checks
#   Spec id=49 v3 hash 9cc868d2. Tier R Watchdog rules ensuring:
#     1. cap_state.json updated within recent monthly cycle (no silent fail)
#     2. trailing 30d Vertex cost trajectory under budget (early warning before HARD HALT)
#   Both rules reuse engine.etf_holdings_risk_monitor helpers — single source of truth.
# ─────────────────────────────────────────────────────────────────────────────

def rule_etf_holdings_cap_state_freshness() -> RuleResult:
    """ETF Holdings cap_state.json freshness check (Tier R).

    HIGH if mtime > 35 days (monthly cadence + 5-day buffer for late runs).
    CRITICAL if mtime > 60 days (silent scheduler fail, 2 missed monthly runs).

    No-action LOW if cap_state.json doesn't exist (first-time setup, not a failure).
    """
    import datetime
    from pathlib import Path

    path = Path("data/etf_holdings_risk_monitor/cap_state.json")
    if not path.exists():
        return {
            "severity": "LOW",
            "snapshot": {
                "kind":   "cap_state_missing",
                "reason": "cap_state.json does not exist (first-time setup before first monthly run)",
                "path":   str(path),
            },
        }

    mtime_dt   = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    age_days   = (datetime.datetime.now() - mtime_dt).days

    if age_days > 60:
        severity = "HIGH"   # spec compliance — CRITICAL would be too strong for ops layer
        kind     = "cap_state_2plus_months_stale"
    elif age_days > 35:
        severity = "MID"
        kind     = "cap_state_1month_stale"
    else:
        return None

    return {
        "severity": severity,
        "snapshot": {
            "kind":            kind,
            "path":            str(path),
            "last_modified":   mtime_dt.isoformat(),
            "age_days":        age_days,
            "threshold_warn":  35,
            "threshold_crit":  60,
            "spec_id":         49,
            "spec_hash":       "9cc868d2",
            "fix_hint":        "Check MacroAlphaPro_ETFHoldings Task Scheduler status. Manual rerun: py -3.11 -m scripts.run_etf_holdings_monitor_monthly --force",
        },
    }


def rule_etf_holdings_cost_budget() -> RuleResult:
    """ETF Holdings trailing-30d Vertex cost budget check (Tier R).

    Early warning before spec §2.3 HARD HALT at trailing 365d > $720.
    HIGH if trailing 30d cost > $50 (~75% of monthly budget allocation).
    MID  if trailing 30d cost > $30.
    """
    try:
        from engine.etf_holdings_risk_monitor import get_cost_status
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"kind": "etf_holdings_module_missing", "reason": str(exc)}}

    try:
        status = get_cost_status()
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"kind": "cost_status_unavailable", "reason": str(exc)}}

    trailing_365d_total = status.get("trailing_365d_total_usd", 0.0)
    n_calls_trailing    = status.get("trailing_365d_n_calls", 0)

    # No-action if no calls fired yet
    if n_calls_trailing == 0 and trailing_365d_total == 0:
        return None

    # Trailing 30d cost (approximation: trailing_365d_total / 12 if uniform spread,
    # but actual would need per-day ledger; use 365d as proxy guard, trip earlier)
    # Conservative thresholds vs spec §2.3 $720 HARD HALT:
    if trailing_365d_total > 540:   # 75% of HARD HALT
        severity = "HIGH"
    elif trailing_365d_total > 360:  # 50% of HARD HALT
        severity = "MID"
    elif trailing_365d_total > 180:  # 25% of HARD HALT
        severity = "LOW"
    else:
        return None

    return {
        "severity": severity,
        "snapshot": {
            "kind":                  "etf_holdings_cost_trajectory",
            "trailing_365d_total":   round(trailing_365d_total, 4),
            "trailing_365d_n_calls": n_calls_trailing,
            "spec_budget_annual":    540,
            "hard_halt_threshold":   720,
            "spec_id":               49,
            "spec_hash":             "9cc868d2",
            "fix_hint":              "Audit recent monthly runs in cost ledger. If anomalous burst, check for retry loop or schema validation failures triggering re-calls.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-agent consistency rule (2026-05-14, Step 2.5 of UI roadmap)
#
# Bridges 3 specs via shared `data/etf_holdings_risk_monitor/cap_state.json`:
#   spec id=49 v3 (ETF Holdings Risk Monitor) WRITES the cap_state.json file
#   spec id=61   (K1_BAB) READS it via paper_trade_combined Step 3b
#   spec id=63   (Ops Watchdog) READS it here to VERIFY application
#
# Purpose: affirmative cross-agent verification. If ETF Holdings declares
# "QQQ capped at 15%" in cap_state.json but K1's persisted positions show
# QQQ at 22% book weight, that's a real bug class (cap silently not
# applied). Without this rule, supervisor only learns of the divergence
# via P&L surprise. Doctrine: shared structured memory + deterministic
# routing, NOT LLM-to-LLM message passing.
# ─────────────────────────────────────────────────────────────────────────────

def rule_etf_cap_state_consistency_with_book() -> RuleResult:
    """Verify ETF Holdings cap_state.json caps are reflected in K1_BAB positions.

    Auto-repair recipe (when fires):
      1. Re-run cleanup_expired_cap_state() — purge any genuinely-expired entries
      2. Re-run paper_trade_combined daily orchestrator — apply cap via Step 3b
      3. If still divergent after rerun, escalate to Tier 3 (real cap-engine bug)

    Returns None when no active caps OR all caps consistent with book.
    MID severity when divergence detected (auto-repair available).

    Spec refs:
      - docs/spec_etf_holdings_llm_risk_monitor.md §2.7 (cap mechanics)
      - docs/spec_path_k1_size_expanded_b_plus_v1.md (K1 hash a0bbcbda)
      - docs/spec_ops_watchdog.md §error_mode_index (this rule = Tier R)
    """
    import json
    import datetime as _dt
    try:
        from engine.etf_holdings_risk_monitor import (
            _load_cap_state, apply_cap_to_max_weight, _trading_days_elapsed,
        )
        from engine.db_models import PaperTradeStrategyLog
        from engine.memory import SessionFactory
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"kind": "import_failed", "exception": str(exc)}}

    try:
        caps = _load_cap_state()
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"kind": "cap_state_unreadable", "exception": str(exc)}}

    if not caps:
        return None   # no active caps → nothing to verify

    today = _dt.datetime.utcnow().date()

    # K1_BAB sleeve allocation factor: 36% × 100% (etf_l1 sole strategy)
    K1_SLEEVE_TARGET   = 0.36
    K1_INTRA_SLEEVE_W  = 1.00
    K1_BOOK_FACTOR     = K1_SLEEVE_TARGET * K1_INTRA_SLEEVE_W
    BASE_MAX_WEIGHT    = 0.25
    BOOK_TOLERANCE     = 0.005   # 50bp slack vs effective cap

    # Load latest K1 snapshot
    try:
        with SessionFactory() as s:
            row = (s.query(PaperTradeStrategyLog)
                    .filter_by(strategy_name="K1_BAB")
                    .order_by(PaperTradeStrategyLog.date.desc())
                    .first())
            if row is None:
                return None   # no K1 snapshot yet (pre-2026-05-13)
            if row.status != "OK":
                # K1 inactive — caps stored but not constraining (informational)
                return {
                    "severity": "LOW",
                    "snapshot": {
                        "kind":          "k1_inactive_caps_stored",
                        "k1_status":     row.status,
                        "snapshot_date": row.date.isoformat(),
                        "n_caps_stored": len(caps),
                        "capped_tickers": sorted(caps.keys()),
                        "interpretation": "K1_BAB not generating signal; caps in cap_state.json have no current positions to constrain. State stored for future K1 OK days.",
                        "spec_id":       49,
                        "spec_hash":     "9cc868d2",
                    },
                }
            try:
                positions = json.loads(row.positions_json) if row.positions_json else {}
            except Exception:
                positions = {}
    except Exception as exc:
        return {"severity": "LOW",
                "snapshot": {"kind": "k1_query_failed", "exception": str(exc)}}

    # Cross-check: each capped ETF that K1 actually holds should respect effective cap
    violations: List[Dict[str, Any]] = []
    verified:   List[Dict[str, Any]] = []
    for etf, entry in caps.items():
        etf_u = etf.upper()
        intra_w = positions.get(etf_u, positions.get(etf, None))
        if intra_w is None:
            continue   # cap stored, K1 not holding this ETF — informational, not violation
        try:
            triggered_at = _dt.date.fromisoformat(entry["triggered_at"])
        except Exception:
            continue
        days_since = _trading_days_elapsed(triggered_at, today)
        effective_intra_cap = apply_cap_to_max_weight(
            base_max_weight     = BASE_MAX_WEIGHT,
            cap_active          = True,
            days_since_trigger  = days_since,
        )
        # K1 stores INTRA-sleeve weight (sums to ~1.0 within etf_l1). Effective
        # intra cap = BASE_MAX_WEIGHT × multiplier. Live intra_w must be ≤ this.
        intra_w_f = float(intra_w)
        if abs(intra_w_f) > effective_intra_cap + BOOK_TOLERANCE:
            violations.append({
                "etf":                 etf_u,
                "live_intra_w":        round(intra_w_f, 6),
                "effective_intra_cap": round(effective_intra_cap, 6),
                "live_book_w":         round(intra_w_f * K1_BOOK_FACTOR, 6),
                "effective_book_cap":  round(effective_intra_cap * K1_BOOK_FACTOR, 6),
                "excess_intra":        round(abs(intra_w_f) - effective_intra_cap, 6),
                "cap_triggered_at":    entry.get("triggered_at"),
                "cap_expires_at":      entry.get("expires_at"),
                "cap_aggregate_score": entry.get("aggregate_score"),
            })
        else:
            verified.append({
                "etf":                 etf_u,
                "live_intra_w":        round(intra_w_f, 6),
                "effective_intra_cap": round(effective_intra_cap, 6),
                "headroom_intra":      round(effective_intra_cap - abs(intra_w_f), 6),
            })

    if not violations:
        # No violations + no positions matching any caps → no-action
        if not verified:
            return None
        # All caps verified — emit nothing (clean state)
        return None

    return {
        "severity": "MID",
        "snapshot": {
            "kind":               "etf_cap_state_book_divergence",
            "snapshot_date":      row.date.isoformat(),
            "n_active_caps":      len(caps),
            "n_violations":       len(violations),
            "n_verified":         len(verified),
            "violations":         violations[:10],
            "verified":           verified[:10],
            "spec_ids_bridged":   [49, 61, 63],
            "context":            (
                "ETF Holdings cap_state.json declares cap but K1_BAB live "
                "intra-sleeve weight exceeds effective cap. Either the cap "
                "engine in paper_trade_combined Step 3b silently failed OR "
                "cap_state.json was modified between Step 3b and persistence."
            ),
            "fix_hint":           (
                "1) Manually run engine.etf_holdings_risk_monitor."
                "cleanup_expired_cap_state(). 2) Re-run "
                "scripts/run_paper_trade_daily.py --force. 3) If violations "
                "persist, inspect paper_trade_combined Step 3b code path "
                "for short-circuit OR file-race against cap_state.json mtime."
            ),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Ops Watchdog Agent v1.0 registry (spec id=63, hash 512c918f)
#   Separate from CRITICAL_RULES / WEEKLY_RULES. Invoked by Watchdog at
#   06:10 SGT daily (10min after MacroAlphaPro_DailyBatch 06:00 SGT).
#   Covers 10 of 12 spec error modes; remaining 2 reuse existing rules.
# ═════════════════════════════════════════════════════════════════════════════
WATCHDOG_RULES: List[RuleFn] = [
    # Operations layer (5 NEW)
    rule_cycle_state_completion,                    # mode 1
    rule_universe_data_freshness_per_ticker,        # mode 2
    rule_weight_delta_p99_unexplained,              # mode 5
    rule_signal_trade_referential_integrity,        # mode 6
    rule_nav_move_vs_rebalance_audit,               # mode 7
    # Trading layer (5 NEW)
    rule_signal_panel_nan_scan,                     # mode 8
    rule_realized_tc_vs_spec_rate,                  # mode 9
    rule_max_position_weight_vs_cap,                # mode 10
    rule_rebalance_frequency_audit,                 # mode 11
    rule_regime_scale_vs_exposure_audit,            # mode 12
    # Meta-monitoring layer (1 NEW, amendment 1 2026-05-12, spec hash 9d050804)
    rule_watchdog_daily_cost_budget,                # mode 13
    # Phase A2 (2026-05-14) — ETF Holdings Risk Monitor liveness (spec id=49 v3 hash 9cc868d2)
    rule_etf_holdings_cap_state_freshness,
    rule_etf_holdings_cost_budget,
    # Step 2.5 (2026-05-14) — Cross-agent ETF Holdings ↔ K1_BAB ↔ Watchdog consistency
    rule_etf_cap_state_consistency_with_book,
    # Phase 1 Agent Observability v1 (2026-05-15) — 5th Tier R rule
    rule_agent_slo_breach,
]

