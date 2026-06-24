"""
engine/agents/ops_watchdog/triage.py — Severity classification (LOCKED).

Hardcoded mapping: AuditFinding rule_name → mode_key → mode_severity.
PURE PYTHON, NO LLM. Per spec §2.2 + §6 forbidden modifications: the LLM
contributes context (which modes co-occur, baseline comparison) but NEVER
changes the severity mapping. Adding a mode requires spec_amend with
HARKing R1-R4 review.

Severity tiers (REUSE engine.circuit_breaker 4-level scheme per spec §2.2):
  - NONE   = "none"     all checks green; silent log
  - LIGHT  = "light"    1 non-critical anomaly; dashboard widget yellow
  - MEDIUM = "medium"   multiple findings OR 1 ops failure; toast + dashboard
  - SEVERE = "severe"   critical failure; toast persist + email + halt flag

Decision tree (§九):
  - 0 findings                                                    → NONE
  - findings only on auto-repairable ops modes (1/2/4/6/10/12)    → MEDIUM
  - 1 finding on critical mode (3/5/7/8/9/11/13)                  → SEVERE
  - findings only on "light" modes (mode 4 sleeve_drift)          → LIGHT
                                                                    (escalates
                                                                     to MEDIUM
                                                                     if 2+)
"""
from __future__ import annotations

from typing import Iterable, Optional

# ── Severity constants (mirror engine.circuit_breaker LEVEL_* values) ────────
SEVERITY_NONE:   str = "none"
SEVERITY_LIGHT:  str = "light"
SEVERITY_MEDIUM: str = "medium"
SEVERITY_SEVERE: str = "severe"

# Ordering for "highest wins" aggregation
_SEVERITY_RANK: dict[str, int] = {
    SEVERITY_NONE:   0,
    SEVERITY_LIGHT:  1,
    SEVERITY_MEDIUM: 2,
    SEVERITY_SEVERE: 3,
}


# ─────────────────────────────────────────────────────────────────────────────
# LOCKED constants (spec §2.2 + §6) — modifying requires spec amendment
# ─────────────────────────────────────────────────────────────────────────────

MODE_SEVERITY_MAP_LOCKED: dict[str, str] = {
    "mode_1_cycle_failed":                   SEVERITY_MEDIUM,  # auto-repair attempt
    "mode_2_yfinance_stale":                 SEVERITY_MEDIUM,  # auto-repair attempt
    "mode_3_etf_delisted":                   SEVERITY_SEVERE,  # decision required + halt
    "mode_4_sleeve_drift":                   SEVERITY_LIGHT,   # cosmetic; backfillable
    "mode_5_weight_delta_unexplained":       SEVERITY_SEVERE,  # potential data error
    "mode_6_trade_execution_missing":        SEVERITY_MEDIUM,  # auto-repair attempt
    "mode_7_nav_anomaly":                    SEVERITY_SEVERE,  # always escalate
    "mode_8_signal_nan":                     SEVERITY_SEVERE,  # signal logic bug
    "mode_9_tc_drag_wrong":                  SEVERITY_SEVERE,  # halt next batch
    "mode_10_weight_cap_violation":          SEVERITY_MEDIUM,  # auto-truncate
    "mode_11_cadence_drift":                 SEVERITY_SEVERE,  # config bug
    "mode_12_regime_scale_misapplied":       SEVERITY_MEDIUM,  # auto-reapply
    "mode_13_watchdog_cost_runaway":         SEVERITY_SEVERE,  # halt next watchdog
}

# AuditFinding.rule_name → mode_key. Covers all 11 NEW Watchdog rules from
# Phase 1 + 2 REUSED existing rules (modes 3 / 4). Rules outside this map
# do NOT contribute to Watchdog severity aggregation (out of scope).
RULE_TO_MODE_LOCKED: dict[str, str] = {
    # NEW rules (Phase 1, 11 rules)
    "rule_cycle_state_completion":             "mode_1_cycle_failed",
    "rule_universe_data_freshness_per_ticker": "mode_2_yfinance_stale",
    "rule_weight_delta_p99_unexplained":       "mode_5_weight_delta_unexplained",
    "rule_signal_trade_referential_integrity": "mode_6_trade_execution_missing",
    "rule_nav_move_vs_rebalance_audit":        "mode_7_nav_anomaly",
    "rule_signal_panel_nan_scan":              "mode_8_signal_nan",
    "rule_realized_tc_vs_spec_rate":           "mode_9_tc_drag_wrong",
    "rule_max_position_weight_vs_cap":         "mode_10_weight_cap_violation",
    "rule_rebalance_frequency_audit":          "mode_11_cadence_drift",
    "rule_regime_scale_vs_exposure_audit":     "mode_12_regime_scale_misapplied",
    "rule_watchdog_daily_cost_budget":         "mode_13_watchdog_cost_runaway",
    # REUSED rules (existing in CRITICAL_RULES, spec §2.1 reuse mapping)
    "rule_universe_drift_vs_registered":       "mode_3_etf_delisted",
    "rule_sleeve_id_integrity":                "mode_4_sleeve_drift",
}

# Modes that have hardcoded auto-repair recipes (Phase 3). Triage flags which
# findings are repair-candidates; the recipe functions live in auto_repair.py.
AUTO_REPAIRABLE_MODES_LOCKED: frozenset[str] = frozenset({
    "mode_1_cycle_failed",
    "mode_2_yfinance_stale",
    "mode_4_sleeve_drift",
    "mode_6_trade_execution_missing",
    "mode_10_weight_cap_violation",
    "mode_12_regime_scale_misapplied",
})


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers — no DB / file / LLM I/O
# ─────────────────────────────────────────────────────────────────────────────

def rule_name_to_mode(rule_name: str) -> Optional[str]:
    """Resolve a Watchdog-relevant rule_name to its mode_key, or None."""
    return RULE_TO_MODE_LOCKED.get(rule_name)


def mode_to_severity(mode_key: str) -> str:
    """Return locked severity for a known mode, or NONE for unknown modes."""
    return MODE_SEVERITY_MAP_LOCKED.get(mode_key, SEVERITY_NONE)


def is_auto_repairable(mode_key: str) -> bool:
    """Whether a mode has a hardcoded auto-repair recipe (Phase 3)."""
    return mode_key in AUTO_REPAIRABLE_MODES_LOCKED


def aggregate_severity(rule_names: Iterable[str]) -> dict:
    """
    Aggregate a list of fired AuditFinding rule_names into a single Watchdog
    severity decision. Pure function — no side effects.

    Algorithm (spec §九 decision tree):
      1. Map each rule_name → mode_key (drop unknown).
      2. Look up mode_severity per LOCKED map.
      3. Aggregate: highest severity wins.
      4. Escalation: if base severity is LIGHT and 2+ findings → MEDIUM
         (multiple-cosmetic-issue rule from §九 row "2+ findings → MEDIUM").

    Args:
        rule_names: iterable of AuditFinding.rule_name strings that fired

    Returns:
        {
            "severity":            str,           # "none"|"light"|"medium"|"severe"
            "n_findings":          int,
            "n_watchdog_findings": int,           # subset matching a mode
            "modes_fired":         list[str],     # unique mode_keys, sorted
            "modes_severity":      dict[str, str], # mode → severity
            "auto_repairable_modes": list[str],   # subset in AUTO_REPAIRABLE
            "escalation_applied":  bool,          # True if multi-finding escalated
        }
    """
    rule_list = list(rule_names)
    mode_keys: list[str] = []
    for rn in rule_list:
        mk = rule_name_to_mode(rn)
        if mk is not None:
            mode_keys.append(mk)

    modes_unique = sorted(set(mode_keys))
    modes_severity = {m: mode_to_severity(m) for m in modes_unique}

    if not modes_unique:
        return {
            "severity":              SEVERITY_NONE,
            "n_findings":            len(rule_list),
            "n_watchdog_findings":   0,
            "modes_fired":           [],
            "modes_severity":        {},
            "auto_repairable_modes": [],
            "escalation_applied":    False,
        }

    # Highest severity wins.
    base_severity = max(modes_severity.values(),
                        key=lambda s: _SEVERITY_RANK[s])
    escalation_applied = False

    # Spec §九: 2+ findings → MEDIUM floor (escalates LIGHT to MEDIUM).
    if base_severity == SEVERITY_LIGHT and len(mode_keys) >= 2:
        base_severity = SEVERITY_MEDIUM
        escalation_applied = True

    repairable = sorted(m for m in modes_unique if is_auto_repairable(m))

    return {
        "severity":              base_severity,
        "n_findings":            len(rule_list),
        "n_watchdog_findings":   len(mode_keys),
        "modes_fired":           modes_unique,
        "modes_severity":        modes_severity,
        "auto_repairable_modes": repairable,
        "escalation_applied":    escalation_applied,
    }
