"""engine.agents.strengthener.factor_dispatcher — Tier C-2a.

Receives a SPEC-extracted FactorSpec (from factor_spec_extractor)
and routes it to the matching deterministic backtest template. This
piece (C-2a) is the dispatcher's outer shell + gates; the template
implementations land in C-2b (tsmom on sector_etf, first end-to-end
loop), C-2e (cross_sectional_rank with WRDS), C-2f (carry).

Per docs/spec_tier_c_factor_backtest_auto_dispatcher.md "CRITICAL
guardrails":

  1. Constrained DSL — signal_kind is a controlled enum (validated
     in C-1's extract_factor_spec; this dispatcher re-validates as
     defense in depth)
  2. n_trials += 1 per auto-test — increment when a verdict actually
     emits; refuse to dispatch when family already at HARD threshold
     (15) per Bailey-LdP §3
  3. Hard cost gate — max 5 auto-tests per WEEK (rolling 7 days
     across all families). Above that, human override required.
  4. PIT audit forced PRE-backtest — template-implementer
     responsibility; dispatcher enforces "spec must list pit_audits"
     at validation time (must NOT be empty unless escape hatch)
  5. Spec approval is the gate — dispatcher refuses if spec hasn't
     been approved (approval_state checked at dispatch_factor_spec
     entry; surfaced by /approvals C-2d UI)
  6. GREEN doesn't auto-promote — handled by event-emission layer
     (C-2c): factor_verdict_filed events carry auto_test_* tags so
     downstream consumers know to require human action before
     paper_trade.
  7. Provenance tags — every dispatch log row carries spec_hash +
     llm_model + extractor_workload + dispatcher_version
  8. signal_inputs whitelist — PIT_CORRECT_SOURCES; dispatcher
     refuses if any spec.signal_inputs token outside the whitelist

This module is PURE dispatch + audit. It does NOT:
  - Call the LLM (spec extractor's job; C-1)
  - Run the actual backtest (template's job; C-2b/e/f)
  - Emit factor_verdict_filed (event emitter's job; C-2c)

Architecture (sketch):

  FactorSpec (from C-1 extractor)
      │
      ▼
  pre_dispatch_check(spec)
      │  ── refuses if: not approved | week_count ≥ 5 |
      │     family n_trials ≥ HARD | signal_inputs outside
      │     whitelist | escape hatch (route to manual)
      │
      ▼
  TEMPLATE_REGISTRY[signal_kind](spec) -> TemplateResult
      │  ── deterministic Python; NO LLM
      │  ── C-2a stubs: every template returns
      │     "pending_template_build" until C-2b/e/f land
      │
      ▼
  persist dispatch_log row + (C-2c) emit factor_verdict_filed

Per [[project-a-plus-b-substrate-first-roadmap-2026-06-05]]:
  Output is RESEARCH data, NOT capital action. A GREEN auto-test
  verdict still requires manual paper_trade promotion. The
  "research auto, capital human" hard line stands.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Callable, Optional

from engine.agents.strengthener.factor_spec_extractor import (
    FactorSpec, SIGNAL_KINDS,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Cost-gate + n_trials thresholds — load-bearing constants
# ────────────────────────────────────────────────────────────────────
# L2-1 Phase 2.5 (2026-06-08): A-class safety constants moved to
# _safety_constants module. Re-export here for backward-compat
# (some tests import these names from factor_dispatcher).
from engine.agents.strengthener._safety_constants import (
    MAX_AUTO_DISPATCHES_PER_WEEK,
    N_TRIALS_CAUTION,
    N_TRIALS_HARD,
)


# L2-1 Phase 2.6 (2026-06-08): B-class parameter typed ranges.
# FactorSpec v2 exposes these design-choice parameters as Optional
# fields; this dict defines the SAFE typed range for each. Out-of-
# range values are rejected at dispatcher gate #9
# (B_CLASS_OUT_OF_RANGE) before any template invocation.
#
# Philosophy: "freedom within safe rails" — LLM has full design
# space within the range, cannot escape. Range bounds set per
# literature (e.g. quintile L/S well-studied; 20-bucket sorts are
# statistically thin per Lo 2002).
# flex-4 (2026-06-10, F-gap-2 fix): each range carries `step` — the
# literature-conventional ablation increment for the specification-
# robustness neighborhood. The neighborhood is DERIVED from this
# table (±1·step, ±2·step clipped to [min,max]); the old parallel
# NEIGHBORHOOD_DELTAS param-name list is gone. One table owns both
# the safe bounds AND the ablation geometry.
B_CLASS_RANGES = {
    "universe_size":     {"type": int,   "min": 100,   "max": 5000, "step": 250},
    "n_buckets":         {"type": int,   "min": 3,     "max": 10,   "step": 1},
    "signal_lookback_m": {"type": int,   "min": 1,     "max": 120,  "step": 1},
    "signal_skip_m":     {"type": int,   "min": 0,     "max": 12,   "step": 1},
    "vol_target_annual": {"type": float, "min": 0.03,  "max": 0.30, "step": 0.02},
}
B_CLASS_ENUMS = {
    "weighting_scheme_alt": {"ew", "vw", "rank", None},
}


# ────────────────────────────────────────────────────────────────────
# PIT_CORRECT_SOURCES — signal_inputs whitelist
# ────────────────────────────────────────────────────────────────────
# Every spec.signal_inputs entry MUST be a prefix-match against this
# whitelist. Forces human to manually add a new source path (rare
# event) before any new data domain enters auto-dispatch — prevents
# the LLM from silently referencing a cache path that has known
# look-ahead bugs / restatement issues.
#
# C-2a baseline: covers the inputs the 3 initial templates need
# (TSMOM on sector ETFs first; cross_sec + carry expand list when
# their templates land in C-2e/f). Adding a prefix is a human
# decision that should reference the source's PIT-audit status.
PIT_CORRECT_SOURCES: frozenset[str] = frozenset({
    # CRSP (PIT-clean for prices; survivorship handled by full universe)
    "crsp.msf.",                # monthly stock file (prices, returns)
    "crsp.dsf.",                # daily stock file
    "crsp.msenames.",           # security master (point-in-time names)
    "crsp.msedelist.",          # delisting events

    # Compustat fundamentals (PIT-clean ONLY when used with rdq/datadate
    # lag; templates must enforce this — flagged in C-2e cross_sec build)
    "compustat.funda.",         # annual funda
    "compustat.fundq.",         # quarterly funda

    # FF/Hou-Xue-Zhang factors (already public; no restatement issue)
    "ff.factors_monthly.",
    "ff.factors_daily.",
    "ff.factors_weekly.",        # bt-flex-4.2: Ken French FF5+Mom weekly

    # Sector ETF closes (PIT — adjusted closes from yfinance / data_ops)
    "etf.adj_close.",

    # FX (G10 spot + interest rates — Bloomberg / FRED feeds)
    "fx.spot.",
    "fx.interest_rate.",

    # Treasury curve (Fed H.15)
    "treasury.constant_maturity.",

    # OptionMetrics (PIT for options; VRP templates need this)
    "optionmetrics.standardized_options.",
    "optionmetrics.volsurf.",

    # CBOE VIX + SPX daily index (no restatement; PIT-clean)
    "cboe.vix_spx.",            # vrp template (Carr-Wu 2009 short-vol)

    # OptionMetrics IV surface (fetched 2026-06-14 via ${WRDS_USER_2}; PIT-clean —
    # end-of-day IV is published next day at the latest)
    "optionm.vsurfd.",          # skew_premium template (Bollerslev-Todorov 2011)

    # Compustat fundq (PIT via rdq lag; smallcap subset 2011-2025)
    "compustat.fundq.",         # event_drift_pead template (Bernard-Thomas 1989)

    # IBES analyst EPS estimates (1990-2024 via ${WRDS_USER_2}, monthly summary)
    "ibes.statsumu_epsus.",     # event_drift_revision template (CJL 1996)

    # FRED / macro (PIT via vintage — but most templates use latest, OK
    # for non-realtime research)
    "fred.",

    # ICE BofA MOVE index (1-month forward implied vol on US Treasury
    # futures, basis points of yield). PIT-clean: published end-of-day,
    # no restatement. Vol indexes are like VIX — observed at close.
    # Used by vrp_treasury template (W6-rigor-A-validate-loop-closed,
    # 2026-06-22). Cache: data/cache/_move_tlt_daily.parquet.
    "move.",                     # Bond-VRP template (MVP 2026-06-22)

    # iShares 20+y Treasury ETF (TLT) adjusted close from yfinance.
    # PIT-clean: yfinance returns adjusted-for-dividend close that's
    # final at session close. Used as long-duration Treasury price
    # proxy in vrp_treasury template (MOVE-vs-realized-vol pair).
    "tlt.",                      # Bond-VRP template (MVP 2026-06-22)
})


# ────────────────────────────────────────────────────────────────────
# Template registry (C-2a: all stubs; C-2b/e/f land real impls)
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class TemplateResult:
    """What every template function returns. Deterministic shape so
    the dispatcher's downstream (C-2c event emission) is uniform
    regardless of which template ran."""
    verdict:        str      # "GREEN" | "MARGINAL" | "RED" | "EXECUTION_ERROR"
                              # | "PENDING_TEMPLATE_BUILD" | "PIT_AUDIT_FAIL"
    summary:        str      # 1-line human summary
    metrics:        dict     # template-specific (sharpe, nw_t_stat, n_obs,
                              # ann_return, ann_vol, ic_mean, etc.)
    artifacts:      dict     # paths to capability evidence (figures, parquet)
    template_version: str    # "v0_stub" | "v1" — bumps invalidate cache


def _template_pending_build(spec: FactorSpec) -> TemplateResult:
    """C-2a stub returned for every signal_kind until its real
    template lands. Lets us ship the dispatcher + cost-gate + audit
    plumbing now and validate it before implementing templates.

    flex-3 (2026-06-10): routing slip, not a dead wall — per
    [[feedback-dead-wall-monitoring-standing-2026-06-10]] the result
    carries gap-tier classification + live data probe + next action,
    and the demand is logged to the capability-gap ledger.
    """
    guidance: dict = {}
    try:
        from engine.research.capability_gaps import (
            guidance_unsupported_universe, log_gap,
        )
        guidance = guidance_unsupported_universe(
            spec.signal_kind, spec.universe)
        log_gap(hypothesis_id=spec.hypothesis_id, guidance=guidance)
    except Exception:
        logger.exception("pending_build: guidance failed")
    return TemplateResult(
        verdict          = "PENDING_TEMPLATE_BUILD",
        summary          = (f"signal_kind={spec.signal_kind} × "
                              f"universe={spec.universe} has no template. "
                              + (guidance.get("next_action", "") or "")[:220]),
        metrics          = {"template_build_status": "pending",
                              "guidance": guidance},
        artifacts        = {},
        template_version = "v0_stub",
    )


def _template_custom_code_escape(spec: FactorSpec) -> TemplateResult:
    """Escape hatch: LLM signaled the hypothesis doesn't fit any
    template. Surfaces to /approvals (C-2d) as a custom-code
    reminder so the human knows to take over manually.

    flex-3: demand logged — repeated custom-code escapes for the same
    shape of work ARE the signal that a new template class is due.
    """
    guidance: dict = {}
    try:
        from engine.research.capability_gaps import (
            GAP_EFFORT, GAP_TIER_3_TEMPLATE, log_gap,
        )
        guidance = {
            "gap_class":  GAP_TIER_3_TEMPLATE,
            "data_check": {"escape_hatch": True},
            "next_action": (
                "Hand-write this test in a research_new session. If "
                "the same shape of custom work recurs (check the "
                "capability-gap digest), promote it to a template."),
            "effort":     GAP_EFFORT[GAP_TIER_3_TEMPLATE],
            "requested":  {"signal_kind": spec.signal_kind,
                            "universe": spec.universe},
        }
        log_gap(hypothesis_id=spec.hypothesis_id, guidance=guidance)
    except Exception:
        logger.exception("custom_code_escape: guidance failed")
    return TemplateResult(
        verdict          = "CUSTOM_CODE_REQUIRED",
        summary          = ("requires_custom_code: no dispatcher "
                              "template applies — human must write "
                              "this test by hand"),
        metrics          = {"escape_hatch": True, "guidance": guidance},
        artifacts        = {},
        template_version = "n/a",
    )


# Registry — single source of truth.
# Adding a template = (a) implement the function, (b) replace the
# stub here. Removing a template requires also removing the matching
# signal_kind from SIGNAL_KINDS (which is a breaking change).
#
# Lazy import for time_series_momentum so the dispatcher module
# itself stays import-cheap (templates pull in pandas / numpy /
# DB; we don't want every cost-gate check to drag those in).
def _tsmom_template_lazy(spec: FactorSpec) -> "TemplateResult":
    """Lazy-import wrapper around template_tsmom_sector_etf so
    dispatcher module import stays light. The actual template is in
    engine.agents.strengthener.templates.tsmom_sector_etf."""
    from engine.agents.strengthener.templates.tsmom_sector_etf import (
        template_tsmom_sector_etf,
    )
    # Universe scope guard belongs to dispatcher level so we can
    # route between multiple TSMOM templates as universes expand
    # (us_equities_sector_etf vs commodity_futures_27 etc.). Today
    # only sector_etf has a template — others escape-hatch.
    if spec.universe == "us_equities_sector_etf":
        return template_tsmom_sector_etf(spec)
    return _template_pending_build(spec)


def _cross_sec_template_lazy(spec: FactorSpec) -> "TemplateResult":
    """Lazy-import wrapper around template_cross_sec_us_equities so
    dispatcher module import stays light."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    # Universe scope guard at dispatcher level — only top_3000 has a
    # template in C-2e.1; sp500 + other universes fall back to stub.
    if spec.universe == "us_equities_top_3000":
        return template_cross_sec_us_equities(spec)
    return _template_pending_build(spec)


def _carry_template_lazy(spec: FactorSpec) -> "TemplateResult":
    """Lazy-import wrapper around template_carry_g10_fx (C-2f).
    Other carry universes (commodity_futures_27, us_treasury_curve)
    escape-hatch to _template_pending_build until their fetchers ship."""
    from engine.agents.strengthener.templates.carry_g10_fx import (
        template_carry_g10_fx,
    )
    if spec.universe == "fx_g10":
        return template_carry_g10_fx(spec)
    return _template_pending_build(spec)


def _portfolio_overlay_lazy(spec: FactorSpec) -> "TemplateResult":
    """bt-flex-4.1: portfolio_overlay routing. Only us_balanced_60_40 is
    wired today; future overlay universes (us_balanced_70_30, etc.)
    escape-hatch to pending_build."""
    from engine.agents.strengthener.templates.portfolio_overlay_60_40 import (
        template_portfolio_overlay_60_40,
    )
    if spec.universe == "us_balanced_60_40":
        return template_portfolio_overlay_60_40(spec)
    return _template_pending_build(spec)


def _factor_combination_lazy(spec: FactorSpec) -> "TemplateResult":
    """bt-flex-4.2: factor_combination routing. Only ken_french_ff5_mom
    universe wired today (Asness-Moskowitz-Pedersen 2013 50/50 V+M etc).
    Future combination universes (cross_sec_us_equities decile pairs
    etc.) escape-hatch to pending_build."""
    from engine.agents.strengthener.templates.factor_combination_ff import (
        template_factor_combination_ff,
    )
    if spec.universe == "ken_french_ff5_mom":
        return template_factor_combination_ff(spec)
    return _template_pending_build(spec)


def _event_drift_lazy(spec: FactorSpec) -> "TemplateResult":
    """event_drift router: routes by universe.
      - us_equities_pead     → event_drift_pead (Bernard-Thomas 1989)
      - us_equities_revision → event_drift_revision (CJL 1996)
    """
    if spec.universe == "us_equities_pead":
        from engine.agents.strengthener.templates.event_drift_pead import (
            template_event_drift_pead,
        )
        return template_event_drift_pead(spec)
    if spec.universe == "us_equities_revision":
        from engine.agents.strengthener.templates.event_drift_revision import (
            template_event_drift_revision,
        )
        return template_event_drift_revision(spec)
    return _template_pending_build(spec)


def _spx_skew_premium_lazy(spec: FactorSpec) -> "TemplateResult":
    """SPX skew premium template (2026-06-14): tests Bollerslev-Todorov 2011
    canonical claim that option-implied skew predicts SPX excess returns.
    Single universe: us_equities_spx_options."""
    from engine.agents.strengthener.templates.spx_skew_premium import (
        template_spx_skew_premium,
    )
    if spec.universe == "us_equities_spx_options":
        return template_spx_skew_premium(spec)
    return _template_pending_build(spec)


def _vrp_lazy(spec: FactorSpec) -> "TemplateResult":
    """VRP shipped (2026-06-13): variance risk premium.
    Universes:
      - us_equities_spx_options (Carr-Wu 2009 SPX, VIX/SPX, shipped 06-13)
      - us_treasury_options     (Bond-VRP MVP, MOVE/TLT, shipped 06-22)
    Other universes fall through to CUSTOM_CODE_REQUIRED.
    """
    from engine.agents.strengthener.templates.vrp_spx import (
        template_vrp_spx,
    )
    from engine.agents.strengthener.templates.vrp_treasury import (
        template_vrp_treasury,
    )
    if spec.universe == "us_equities_spx_options":
        return template_vrp_spx(spec)
    if spec.universe == "us_treasury_options":
        return template_vrp_treasury(spec)
    return _template_pending_build(spec)


def _spanning_test_lazy(spec: FactorSpec) -> "TemplateResult":
    """BUG-2 fix (2026-06-13): spanning_test routes 'is X spanned by M?'
    claims to the dedicated regression template. Only ken_french_ff5_mom
    wired today — broader test asset universes (CRSP decile portfolios,
    industry portfolios) are future work."""
    from engine.agents.strengthener.templates.spanning_test_ff import (
        template_spanning_test_ff,
    )
    if spec.universe == "ken_french_ff5_mom":
        return template_spanning_test_ff(spec)
    return _template_pending_build(spec)


TEMPLATE_REGISTRY: dict[str, Callable[[FactorSpec], TemplateResult]] = {
    "cross_sectional_rank":   _cross_sec_template_lazy,  # C-2e.1 SHIPPED (CRSP-only signals)
    "time_series_momentum":   _tsmom_template_lazy,      # C-2b SHIPPED
    "carry":                  _carry_template_lazy,      # C-2f SHIPPED (fx_g10)
    "portfolio_overlay":      _portfolio_overlay_lazy,   # bt-flex-4.1 SHIPPED (60_40 + TSMOM)
    "factor_combination":     _factor_combination_lazy,  # bt-flex-4.2 SHIPPED (FF5+Mom blends)
    "spanning_test":          _spanning_test_lazy,       # BUG-2 fix SHIPPED (single-asset spanning)
    "vrp":                    _vrp_lazy,                 # SHIPPED 2026-06-13 (Carr-Wu 2009 short-vol on SPX)
    "skew_premium":           _spx_skew_premium_lazy,    # SHIPPED 2026-06-14 (Bollerslev-Todorov 2011 SPX skew)
    "event_drift":            _event_drift_lazy,         # SHIPPED 2026-06-13 (Bernard-Thomas 1989 PEAD on smallcap fundq)
    "requires_custom_code":   _template_custom_code_escape,
}


# ────────────────────────────────────────────────────────────────────
# Pre-dispatch checks (return None = OK; return DispatchRefusal = abort)
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class DispatchRefusal:
    """Why the dispatcher REFUSED to run a spec. Surfaces in the
    dispatch_log + /approvals so the user can act on it."""
    reason_code: str    # WEEKLY_CAP / N_TRIALS_HARD / SIGNAL_INPUT_UNKNOWN /
                         # NOT_APPROVED / UNKNOWN_SIGNAL_KIND
    detail:      str    # human message
    metrics:     dict   # supporting numbers (count, threshold, etc.)


def _count_dispatches_last_week(log_path: Path) -> int:
    """Count MANUAL auto-dispatch rows (any verdict, including refusals)
    in the last 7 days. Used by the WEEKLY_CAP cost gate.

    burn-1b (2026-06-11): cron-tagged rows (cron_run_id non-null) are
    EXCLUDED — cron runs through its own caps in engine.research.
    burndown_caps (FAMILY_WEEKLY_CAP + GLOBAL_SOFT/HARD). Mixing them
    would double-cap and starve cron of slots even when manual usage
    is light.
    """
    if not log_path.exists():
        return 0
    cutoff = (_dt.datetime.utcnow()
              - _dt.timedelta(days=7)
              ).strftime("%Y-%m-%dT%H:%M:%SZ")
    count = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("cron_run_id"):
                # Cron rows live under burndown_caps, not WEEKLY_CAP.
                continue
            ts = row.get("ts", "")
            if ts and ts >= cutoff:
                count += 1
    return count


def _family_n_trials_now(family: str) -> int:
    """Count Tier C factor research trials in `family` for Bailey-Lopez
    de Prado §3 deflated-Sharpe multi-testing accounting.

    Definition (2026-06-12 design-flaw fix): the `family` argument is
    now interpreted as a **strategy_family** (canonical Bailey-LdP
    family identifier derived from spec content) rather than a claim-
    origin mechanism_family. The lookup matches events where:

      - event.family == requested strategy_family, OR
      - tag `strategy_family:<X>` matches requested X  (forward compat)
      - tag 'tier_c_auto' present  (gate excludes manual / non-tier-C)

    The OR with event.family preserves backward compat for events
    written before the strategy_family migration: their event.family
    was the mechanism_family inherited from hypothesis claim. For
    legacy strategy families that happen to share the name (VALUE /
    MOMENTUM / SIZE / PROFITABILITY / etc.) the lookup still works.
    The design-flaw exposure was the COMBINATION_X_Y family which
    didn't previously exist; new events get the correct counter, old
    events stay in their legacy buckets.

    Background — pre-rewrite this function read a separate ledger
    keyed by workflow_executor tags; the 2026-06-08 audit caught
    that dispatcher passes MechanismFamily enum values and got
    n_trials=0 every time. The 2026-06-08 rewrite linked it to
    factor_verdict_filed events. The 2026-06-12 rewrite (this
    revision) fixes the claim/strategy family conflation.

    Returns 0 if research store missing — gate fails open, which is
    the right behavior for a brand-new system or import-time failure.
    """
    try:
        from engine.research_store.store import filter_events
    except Exception as exc:
        logger.warning("n_trials count: filter_events import failed: %s", exc)
        return 0
    try:
        events = filter_events(event_type="factor_verdict_filed")
    except Exception as exc:
        logger.warning("n_trials count: filter_events raised: %s", exc)
        return 0
    fam_lower = (family or "").lower()
    fam_tag = f"strategy_family:{fam_lower}"
    n = 0
    for ev in events:
        if "tier_c_auto" not in (ev.tags or ()):
            continue
        # MATCH if either path identifies this event as belonging to
        # the requested family (canonical event.family OR explicit tag).
        ev_family_lower = (ev.family or "").lower()
        if ev_family_lower == fam_lower:
            n += 1
            continue
        if any((t or "").lower() == fam_tag for t in (ev.tags or ())):
            n += 1
    return n


def _signal_inputs_in_whitelist(
    signal_inputs: tuple[str, ...],
) -> tuple[bool, list[str]]:
    """True iff every signal_inputs entry prefix-matches a path in
    PIT_CORRECT_SOURCES. Returns (ok, list_of_violators)."""
    violators = [
        s for s in signal_inputs
        if not any(s.startswith(p) for p in PIT_CORRECT_SOURCES)
    ]
    return (len(violators) == 0, violators)


def pre_dispatch_check(
    spec:              FactorSpec,
    *,
    spec_approved:     bool,
    family_hint:       str,
    log_path:          Path,
    human_override:    Optional[str] = None,
    cron_run_id:       Optional[str] = None,
) -> Optional[DispatchRefusal]:
    """Apply all pre-dispatch gates. Returns None on PASS; a
    DispatchRefusal otherwise.

    Args:
      spec: the FactorSpec from C-1 extractor (already enum-validated)
      spec_approved: True if human approved this spec in /approvals
                     (gate #5). C-2a stub: caller passes True for
                     dry-run testing; C-2d wires the real approval
                     check from the strengthener approval store.
      family_hint: mechanism_family from the source Hypothesis (the
                   FactorSpec doesn't carry it — caller knows it).
                   Used for the family-aware n_trials check.
      log_path: dispatch log file (for the weekly-cap count).
      human_override: first-real-use fix (2026-06-10). The WEEKLY_CAP
                     and N_TRIALS_HARD refusals always said "human
                     override required" but no mechanism existed.
                     A reason string (≥ 10 chars, institutional ack
                     standard) bypasses gates #3 and #2 ONLY — never
                     approval, enum, PIT-whitelist, B-class range, or
                     template-cert gates. The reason is recorded in
                     the dispatch log row for audit.
    """
    override_active = bool(human_override
                             and len(human_override.strip()) >= 10)
    # Gate (defense in depth): signal_kind in controlled enum
    if spec.signal_kind not in SIGNAL_KINDS:
        return DispatchRefusal(
            reason_code = "UNKNOWN_SIGNAL_KIND",
            detail      = (f"signal_kind={spec.signal_kind!r} not in "
                            f"controlled enum (defense-in-depth — "
                            "extractor should have caught this)"),
            metrics     = {"signal_kind": spec.signal_kind},
        )

    # Gate #5: spec approval
    if not spec_approved:
        return DispatchRefusal(
            reason_code = "NOT_APPROVED",
            detail      = ("spec not yet approved in /approvals — "
                            "human review required before dispatch"),
            metrics     = {},
        )

    # Gate #3: weekly cost cap (flex-7: routed through the refusal
    # guidance registry — site only declares reason_code + context).
    # burn-1b: cron dispatches (cron_run_id non-null) are exempt from
    # WEEKLY_CAP — they run through burndown_caps which enforces its own
    # family rotation + global throughput limits.
    week_count = _count_dispatches_last_week(log_path)
    if cron_run_id:
        # Skip WEEKLY_CAP for cron — burndown_caps is the cron gate
        pass
    elif week_count >= MAX_AUTO_DISPATCHES_PER_WEEK and not override_active:
        from engine.research.capability_gaps import build_refusal
        return build_refusal(
            "WEEKLY_CAP",
            detail        = (f"{week_count} auto-dispatches in last 7d "
                              f">= cap {MAX_AUTO_DISPATCHES_PER_WEEK}; "
                              "human override required (pass "
                              "human_override='<reason>')."),
            metrics       = {"week_count": week_count,
                              "cap":        MAX_AUTO_DISPATCHES_PER_WEEK},
            hypothesis_id = spec.hypothesis_id,
            context       = {"week_count": week_count,
                              "cap":        MAX_AUTO_DISPATCHES_PER_WEEK},
        )
    if week_count >= MAX_AUTO_DISPATCHES_PER_WEEK and override_active:
        logger.warning("pre_dispatch: WEEKLY_CAP (%d/%d) bypassed by "
                          "human override: %s",
                          week_count, MAX_AUTO_DISPATCHES_PER_WEEK,
                          human_override)

    # Gate #2: family-aware n_trials HARD threshold (flex-7 factory)
    # 2026-06-12 fix: count by strategy_family (canonical Bailey-LdP
    # denominator) instead of mechanism_family (paper-claim taxonomy).
    # See engine.research.strategy_family_classifier for why these
    # must not be conflated. family_hint is still recorded on the
    # refusal for principal-facing diagnostics; metrics include
    # both fields so audit can see the divergence.
    from engine.research.strategy_family_classifier import (
        strategy_family_for_spec,
    )
    strategy_family = strategy_family_for_spec(spec)
    fam_n = _family_n_trials_now(strategy_family)
    if fam_n >= N_TRIALS_HARD and not override_active:
        from engine.research.capability_gaps import build_refusal
        return build_refusal(
            "N_TRIALS_HARD",
            detail        = (f"strategy_family {strategy_family!r} "
                              f"(claim_family={family_hint!r}) "
                              f"n_trials={fam_n} >= HARD threshold "
                              f"{N_TRIALS_HARD}; Bailey-LdP DSR penalty "
                              f"too high — human override required."),
            metrics       = {"strategy_family": strategy_family,
                              "claim_family":   family_hint,
                              "n_trials":       fam_n,
                              "threshold":      N_TRIALS_HARD},
            hypothesis_id = spec.hypothesis_id,
            context       = {"family":          strategy_family,
                              "n_trials":       fam_n,
                              "threshold":      N_TRIALS_HARD},
        )
    if fam_n >= N_TRIALS_HARD and override_active:
        logger.warning("pre_dispatch: N_TRIALS_HARD (%d/%d) bypassed "
                          "by human override: %s",
                          fam_n, N_TRIALS_HARD, human_override)

    # Gate #8: signal_inputs whitelist (skip for escape hatch)
    if spec.signal_kind != "requires_custom_code":
        ok, bad = _signal_inputs_in_whitelist(spec.signal_inputs)
        if not ok:
            # 2026-06-13: route through build_refusal so the gap emits to
            # capability_gaps demand ledger — the previous direct
            # DispatchRefusal return swallowed the demand signal silently.
            # Now each PIT-whitelist miss → 1 demand row → next paper
            # curator / data-fetch pass sees the unmet demand.
            from engine.research.capability_gaps import build_refusal
            return build_refusal(
                "SIGNAL_INPUT_UNKNOWN",
                detail        = (f"signal_inputs reference sources "
                                  f"outside PIT_CORRECT_SOURCES whitelist: "
                                  f"{bad}"),
                metrics       = {"violators": bad},
                hypothesis_id = spec.hypothesis_id,
                context       = {"signal_inputs": list(spec.signal_inputs or ()),
                                  "violators":      bad,
                                  "signal_kind":    spec.signal_kind,
                                  "universe":       spec.universe},
            )

    # Gate #10 (L2-1 Phase 5): Template audit certification.
    # Any dispatch path must have a fresh TemplateContract covering
    # (signal_kind, universe). Stub-template paths (PENDING_TEMPLATE_
    # BUILD, escape hatch) bypass this gate — they're not making
    # research claims.
    if spec.signal_kind not in ("requires_custom_code",):
        from engine.agents.strengthener.templates._template_contract import (
            contract_for_scope,
        )
        contract = contract_for_scope(spec.signal_kind, spec.universe)
        if contract is None:
            # No certified template — but allow stub-build paths
            # (e.g. vrp/event_drift not yet shipped). Detect via
            # template registry returning the lazy/stub function.
            stub_kinds: set = set()
            if spec.signal_kind in stub_kinds:
                # Stub path acceptable — caller gets
                # PENDING_TEMPLATE_BUILD verdict from template
                pass
            else:
                # flex-7: routed through the refusal guidance registry —
                # next_action / data_check / effort / demand log all
                # come from the provider keyed by reason_code.
                from engine.research.capability_gaps import build_refusal
                return build_refusal(
                    "TEMPLATE_NOT_CERTIFIED",
                    detail        = (f"no certified TemplateContract for "
                                      f"signal_kind={spec.signal_kind!r} + "
                                      f"universe={spec.universe!r}."),
                    metrics       = {"signal_kind": spec.signal_kind,
                                      "universe":    spec.universe},
                    hypothesis_id = spec.hypothesis_id,
                    context       = {"signal_kind": spec.signal_kind,
                                      "universe":    spec.universe},
                )
        elif not contract.is_fresh():
            from engine.research.capability_gaps import build_refusal
            return build_refusal(
                "TEMPLATE_CERT_STALE",
                detail        = (f"template {contract.template_name} "
                                  f"PIT audit cert dated "
                                  f"{contract.pit_audit_date} > 365d "
                                  "stale; re-audit required."),
                metrics       = {"template":   contract.template_name,
                                  "audit_date": contract.pit_audit_date},
                hypothesis_id = spec.hypothesis_id,
                context       = {"template":   contract.template_name,
                                  "audit_date": contract.pit_audit_date},
            )

    # Gate #9 (L2-1 Phase 2.6): B-class parameter typed-range check.
    # Each B-class parameter MUST be within its declared range or
    # None (= use template default). Out-of-range = refuse.
    for fname, rule in B_CLASS_RANGES.items():
        val = getattr(spec, fname, None)
        if val is None:
            continue
        if not isinstance(val, rule["type"]):
            return DispatchRefusal(
                reason_code = "B_CLASS_OUT_OF_RANGE",
                detail      = (f"{fname}={val!r} wrong type "
                                f"(expected {rule['type'].__name__})"),
                metrics     = {"field": fname, "value": val},
            )
        if not (rule["min"] <= val <= rule["max"]):
            return DispatchRefusal(
                reason_code = "B_CLASS_OUT_OF_RANGE",
                detail      = (f"{fname}={val} outside safe range "
                                f"[{rule['min']}, {rule['max']}]"),
                metrics     = {"field": fname, "value": val,
                                "range": [rule["min"], rule["max"]]},
            )
    for fname, allowed in B_CLASS_ENUMS.items():
        val = getattr(spec, fname, None)
        if val in allowed:
            continue
        # 2026-06-14: weighting_scheme_alt is dual-purpose — enum for
        # cross_sec ("ew"/"vw"/"rank") + numeric for factor_combination
        # (weight on first factor, e.g. "0.50") + portfolio_overlay
        # (overlay pct, e.g. "0.20"). Accept numeric strings parseable
        # in [0.0, 1.0]. Refuse anything else.
        if fname == "weighting_scheme_alt" and isinstance(val, str):
            try:
                fv = float(val)
                if 0.0 <= fv <= 1.0:
                    continue
            except ValueError:
                pass
        return DispatchRefusal(
            reason_code = "B_CLASS_OUT_OF_RANGE",
            detail      = (f"{fname}={val!r} not in enum {sorted(x for x in allowed if x)} "
                            f"AND not a [0.0,1.0] numeric string"),
            metrics     = {"field": fname, "value": val},
        )

    # Gate #11 (2026-06-14): SPEC_CONTENT_DUPLICATE — short-circuit if a
    # prior dispatch produced a verdict for the SAME spec content (same
    # signal_kind / universe / inputs / dates etc., different
    # hypothesis_id wrapper). Saves template + audit + rigor cost on
    # re-runs of the SAME backtest just because a different paper
    # produced a hypothesis around the same canonical spec.
    #
    # Belief-layer integrity rationale: 4 same-spec hypotheses should
    # count as 1 obs, not 4 obs. Without this gate, the autopsy ledger
    # over-counts spec-duplicate verdicts and skews belief prior.
    #
    # human_override bypasses (e.g. you really want to re-run the
    # identical spec for fresh data window — though the cleaner path
    # is to bump the date_range which changes the content_hash).
    if not human_override:
        try:
            seen_hashes = _load_dispatched_content_hashes()
            content_hash = _spec_content_hash(spec)
            if content_hash in seen_hashes:
                return DispatchRefusal(
                    reason_code = "SPEC_CONTENT_DUPLICATE",
                    detail      = (f"identical spec content_hash="
                                    f"{content_hash} already dispatched + "
                                    f"verdict-emitted by a prior hypothesis. "
                                    f"Skipping to keep belief_layer "
                                    f"autopsy counts honest."),
                    metrics     = {"content_hash": content_hash},
                )
        except Exception:
            pass   # if load fails, fall through to normal dispatch

    # All gates passed
    return None


# ────────────────────────────────────────────────────────────────────
# Dispatch log — append-only audit trail
# ────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
FACTOR_DISPATCH_LOG_PATH = (_REPO_ROOT / "data" / "strengthener"
                              / "factor_dispatch_log.jsonl")


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_dispatch_log(record: dict, log_path: Path) -> str:
    """Append one factor-dispatch record. Returns the
    dispatch_event_id for callers + audit JOINs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record["dispatch_event_id"]


def record_extraction_failure(
    *,
    hypothesis_id:  str,
    family_hint:    str,
    error_code:     str,
    error_detail:   str,
    cron_run_id:    Optional[str] = None,
    cron_source:    Optional[str] = None,
    log_path:       Optional[Path] = None,
) -> str:
    """burn-1c.1 (2026-06-11) — write a dispatch_log row when the LLM spec
    extractor fails BEFORE the dispatcher's pre-check sees the spec.

    Without this, the executor's extraction-failure path never writes to
    dispatch_log, so burndown_ranker.load_dispatched_hypothesis_ids
    doesn't include the failed hypothesis_id — same hypothesis gets
    re-picked + re-extracted every cron round, burning ~$0.03 LLM
    each time. Caught after force-run round 5 showed hypothesis
    78f8dc8a re-extracted in rounds 5/6/7 consecutively.

    The row uses the same shape as a refusal:
      refusal = {reason_code: "EXTRACT_RETURNED_NONE", ...}
      template_result = None
    so the existing ranker dedup + burndown_caps quota-exclusion
    code paths work unchanged. burndown_caps._row_is_successful_dispatch
    correctly excludes these (refusal != None → not counted).
    """
    path = log_path or FACTOR_DISPATCH_LOG_PATH
    return _append_dispatch_log({
        "dispatch_event_id":     str(uuid.uuid4()),
        "ts":                    _utc_iso(),
        "hypothesis_id":         hypothesis_id,
        "spec_hash":             None,
        "auto_test_spec_hash":   None,
        "auto_test_llm_model":   None,
        "extractor_workload":    "strengthener_factor_spec",
        "dispatcher_version":    DISPATCHER_VERSION,
        "family_hint":           family_hint,
        "refusal":               {
            "reason_code": error_code,
            "detail":      error_detail,
            "metrics":     {},
        },
        "template_result":       None,
        "cron_run_id":           cron_run_id,
        "cron_source":           cron_source,
        "actor":                 "engine.agents.strengthener.factor_spec_extractor",
    }, path)


# ────────────────────────────────────────────────────────────────────
# Spec hash — provenance tag per gate #7
# ────────────────────────────────────────────────────────────────────
def _spec_hash(spec: FactorSpec) -> str:
    """Stable 16-char hex hash of the SPEC's controlled fields. Used
    as auto_test_spec_hash in dispatch_log + (C-2c) event metrics.
    Excludes diagnostics (extracted_ts, model) so the hash is stable
    across re-runs of the same spec."""
    import hashlib
    payload = json.dumps({
        "hypothesis_id":          spec.hypothesis_id,
        "signal_kind":            spec.signal_kind,
        "universe":               spec.universe,
        "date_range":             spec.date_range,
        "signal_inputs":          list(spec.signal_inputs),
        "rebal":                  spec.rebal,
        "weighting":              spec.weighting,
        "expected_holding_period": spec.expected_holding_period,
        "min_obs_months":         spec.min_obs_months,
        "pit_audits":             list(spec.pit_audits),
        "cost_model":             spec.cost_model,
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _spec_content_hash(spec: "FactorSpec | dict") -> str:
    """SPEC CONTENT hash — EXCLUDES hypothesis_id so two different
    hypotheses producing identical specs (same signal_kind / universe /
    inputs / rebal / weighting / dates / cost_model) collapse to the
    SAME hash.

    Used by dispatch_factor_spec's SPEC_CONTENT_DUPLICATE gate
    (2026-06-14) to skip re-running the SAME backtest just because a
    different paper produced a hypothesis-id wrapper around the same
    spec. Saves template + audit + rigor cost AND keeps belief_layer
    autopsy counts honest (4 same-spec hypotheses with 1 backtest
    output should = 1 obs, not 4 obs)."""
    import hashlib
    if isinstance(spec, dict):
        # Used for backfill from dispatch_log rows (spec_inputs dict)
        d = spec
        signal_inputs = d.get("signal_inputs") or []
        pit_audits    = d.get("pit_audits") or []
        payload = json.dumps({
            "signal_kind":            d.get("signal_kind"),
            "universe":               d.get("universe"),
            "date_range":             d.get("date_range"),
            "signal_inputs":          list(signal_inputs) if signal_inputs else [],
            "rebal":                  d.get("rebal"),
            "weighting":              d.get("weighting"),
            "expected_holding_period": d.get("expected_holding_period"),
            "min_obs_months":         d.get("min_obs_months"),
            "pit_audits":             list(pit_audits) if pit_audits else [],
            "cost_model":             d.get("cost_model"),
        }, sort_keys=True).encode("utf-8")
    else:
        payload = json.dumps({
            "signal_kind":            spec.signal_kind,
            "universe":               spec.universe,
            "date_range":             spec.date_range,
            "signal_inputs":          list(spec.signal_inputs),
            "rebal":                  spec.rebal,
            "weighting":              spec.weighting,
            "expected_holding_period": spec.expected_holding_period,
            "min_obs_months":         spec.min_obs_months,
            "pit_audits":             list(spec.pit_audits),
            "cost_model":             spec.cost_model,
        }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_dispatched_content_hashes(
    dispatch_log_path: Optional[Path] = None,
) -> set[str]:
    """Return the set of content_hashes for all PRIOR successful
    dispatches (those that produced a verdict, not refusals). Refusals
    don't count — they didn't consume the test, so the same content
    should still be run when the gate clears."""
    p = dispatch_log_path or FACTOR_DISPATCH_LOG_PATH
    if not p.is_file():
        return set()
    out: set[str] = set()
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except Exception:
                continue
            tr = r.get("template_result") or {}
            if not tr.get("verdict") in ("GREEN", "MARGINAL", "RED"):
                continue
            si = r.get("spec_inputs") or {}
            if not si.get("signal_kind"):
                continue
            try:
                out.add(_spec_content_hash(si))
            except Exception:
                continue
    except Exception:
        return out
    return out


# ────────────────────────────────────────────────────────────────────
# Main entry — pre-checks, dispatch, audit log
# ────────────────────────────────────────────────────────────────────
DISPATCHER_VERSION = "0.1.0"   # bump when behavior changes


def dispatch_factor_spec(
    spec:              FactorSpec,
    *,
    family_hint:       str,
    spec_approved:     bool,
    dry_run:           bool                = False,
    log_path:          Optional[Path]      = None,
    human_override:    Optional[str]       = None,
    cron_run_id:       Optional[str]       = None,
    cron_source:       Optional[str]       = None,
) -> dict:
    """Take ONE FactorSpec + run all gates + invoke the matching
    template + write to dispatch log. Returns the structured result.

    Args:
      spec: from factor_spec_extractor.extract_factor_spec
      family_hint: mechanism_family of the source Hypothesis (for
                   the n_trials check). Caller pulls from the
                   Hypothesis.
      spec_approved: True iff human approved the SPEC in /approvals.
                     C-2d wires this from the approval store.
      dry_run: skip writing the dispatch log + skip template invoke;
               useful for the /approvals preview surface.
      log_path: override (tests use tmp; prod uses
                FACTOR_DISPATCH_LOG_PATH).

    Returns:
      {
        hypothesis_id:       str,
        spec_hash:           str,
        refusal:             {reason_code, detail, metrics} | None,
        template_result:     {verdict, summary, metrics, ...} | None,
        dispatch_event_id:   str | None,
        actor:               "engine.agents.strengthener.factor_dispatcher",
      }
    """
    path = log_path or FACTOR_DISPATCH_LOG_PATH
    spec_hash = _spec_hash(spec)

    out: dict = {
        "hypothesis_id":     spec.hypothesis_id,
        "spec_hash":         spec_hash,
        "refusal":           None,
        "template_result":   None,
        "dispatch_event_id": None,
        "actor":             "engine.agents.strengthener.factor_dispatcher",
    }

    # Pre-dispatch gates
    refusal = pre_dispatch_check(
        spec,
        spec_approved  = spec_approved,
        family_hint    = family_hint,
        log_path       = path,
        human_override = human_override,
        cron_run_id    = cron_run_id,
    )
    if human_override:
        # Audit: the override reason travels with the result + log row
        out["human_override"] = human_override
    if cron_run_id:
        out["cron_run_id"] = cron_run_id
    if cron_source:
        out["cron_source"] = cron_source
    if refusal is not None:
        out["refusal"] = _dc.asdict(refusal)
        if not dry_run:
            try:
                eid = _append_dispatch_log({
                    "dispatch_event_id":     str(uuid.uuid4()),
                    "ts":                    _utc_iso(),
                    "hypothesis_id":         spec.hypothesis_id,
                    "spec_hash":             spec_hash,
                    "auto_test_spec_hash":   spec_hash,
                    "auto_test_llm_model":   spec.model,
                    "extractor_workload":    "strengthener_factor_spec",
                    "dispatcher_version":    DISPATCHER_VERSION,
                    "family_hint":           family_hint,
                    "refusal":               _dc.asdict(refusal),
                    "template_result":       None,
                    "cron_run_id":           cron_run_id,
            "cron_source":           cron_source,
                    "actor":                 out["actor"],
                }, path)
                out["dispatch_event_id"] = eid
            except OSError as exc:
                logger.error("factor_dispatch: log write failed: %s", exc)
        return out

    # Belief Layer Phase 1 (belief-1, 2026-06-11): commit a predicted
    # verdict distribution BEFORE the strict gate runs. Writes to
    # data/research/predictions.jsonl which is AIR-GAPPED from verdict
    # logic — no lens / strict_gate / template module imports belief
    # (structural invariant in tests/test_belief.py). Pure deterministic
    # prior; LLM-free; failure is logged but does NOT block dispatch
    # (prediction is quality control, not a gate).
    try:
        from engine.research.belief import predict_and_log
        # 2026-06-12: use strategy_family (not mechanism_family) so
        # belief-1 posterior aggregates over canonical strategy
        # space, not paper-claim taxonomy. Same fix as n_trials gate.
        from engine.research.strategy_family_classifier import (
            strategy_family_for_spec,
        )
        _belief_pred = predict_and_log(
            subject_id   = spec.hypothesis_id,
            family       = strategy_family_for_spec(spec),
            signal_kind  = spec.signal_kind,
            extra_inputs = {"spec_hash":      spec_hash,
                              "claim_family":  family_hint},
        )
        out["prediction_id"] = _belief_pred.prediction_id
    except Exception as _belief_exc:
        logger.warning(
            "belief: predict_and_log failed for %s: %s",
            spec.hypothesis_id, _belief_exc,
        )

    # Dispatch via template registry
    tpl = TEMPLATE_REGISTRY.get(spec.signal_kind, _template_pending_build)
    try:
        tr = tpl(spec)
    except Exception as exc:
        logger.exception("factor_dispatch: template %s raised for %s",
                          spec.signal_kind, spec.hypothesis_id)
        tr = TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = f"template raised: {type(exc).__name__}: {exc}",
            metrics          = {"error": str(exc)[:200]},
            artifacts        = {},
            template_version = "n/a",
        )
    # L2-4 prep (2026-06-08): artifacts may contain non-JSON values
    # (e.g. pnl_series_df is a pandas DataFrame consumed by
    # write_pnl_parquet downstream). Strip before serializing for
    # log + caller return — file references are added back later.
    def _tr_to_jsonable(_tr) -> dict:
        d = _dc.asdict(_tr)
        d["artifacts"] = {
            k: v for k, v in (d.get("artifacts") or {}).items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
        return d

    # bt-flex-1 (2026-06-11): auto in-paper / post-paper / full-sample
    # triple decay analysis. Slices the template's PnL series — no
    # template re-run, < 1s overhead. Only fires when paper_window is
    # available; failure logs but does NOT block dispatch.
    _oos = None
    paper_window = getattr(spec, "paper_original_window", None)
    if paper_window and tr.verdict in ("GREEN", "MARGINAL", "RED"):
        try:
            from engine.research.oos_triple import compute_oos_triple
            _pnl_df = tr.artifacts.get("pnl_series_df") if isinstance(tr.artifacts, dict) else None
            _pnl_col = tr.artifacts.get("pnl_default_col", "pnl_net_13bp") if isinstance(tr.artifacts, dict) else "pnl_net_13bp"
            _oos = compute_oos_triple(
                _pnl_df,
                full_window  = spec.date_range,
                paper_window = paper_window,
                pnl_column   = _pnl_col,
            )
            if _oos is not None:
                tr.metrics["oos_triple"] = _oos.to_dict()
        except Exception as _oos_exc:
            logger.warning(
                "bt-flex-1 oos_triple failed for %s: %s",
                spec.hypothesis_id, _oos_exc,
            )

    out["template_result"] = _tr_to_jsonable(tr)
    # 2026-06-14 bug fix: rigor pipeline (Phase 4.1) needs the IN-MEMORY
    # tr.artifacts containing pnl_series_df (DataFrame). _tr_to_jsonable
    # strips DataFrames because dispatch_log is JSONL. Pass the raw tr
    # alongside under an underscore-prefixed key so in-process consumers
    # (burndown_executor._maybe_run_post_green_rigor) can access the
    # full artifacts dict. Caller decides whether to read this; JSON
    # consumers (UI, ledger readers) ignore underscore-prefixed keys.
    out["_template_result_obj"] = tr

    if dry_run:
        return out

    try:
        eid = _append_dispatch_log({
            "dispatch_event_id":     str(uuid.uuid4()),
            "ts":                    _utc_iso(),
            "hypothesis_id":         spec.hypothesis_id,
            "spec_hash":             spec_hash,
            "auto_test_spec_hash":   spec_hash,
            "auto_test_llm_model":   spec.model,
            "extractor_workload":    "strengthener_factor_spec",
            "dispatcher_version":    DISPATCHER_VERSION,
            "family_hint":           family_hint,
            "refusal":               None,
            "human_override":        human_override,
            "cron_run_id":           cron_run_id,
            "cron_source":           cron_source,
            # belief-1 prediction_id already set by predict_and_log
            # call above (line ~770); include in audit row so
            # downstream consumers don't need cross-file join via
            # subject_id to find the paired prediction.
            "prediction_id":         out.get("prediction_id"),
            "template_result":       _tr_to_jsonable(tr),
            "spec_inputs": {
                "signal_kind":            spec.signal_kind,
                "universe":               spec.universe,
                "date_range":             spec.date_range,
                "rebal":                  spec.rebal,
                "weighting":              spec.weighting,
                # 2026-06-14: store full spec content so future
                # SPEC_CONTENT_DUPLICATE dedup gate can compute the
                # full content_hash. Previously only 5 fields stored
                # → dedup couldn't catch same-spec wrappers across
                # different hypotheses.
                "signal_inputs":          list(spec.signal_inputs or ()),
                "expected_holding_period": spec.expected_holding_period,
                "min_obs_months":         spec.min_obs_months,
                "pit_audits":             list(spec.pit_audits or ()),
                "cost_model":             spec.cost_model,
            },
            "actor":                 out["actor"],
        }, path)
        out["dispatch_event_id"] = eid
    except OSError as exc:
        logger.error("factor_dispatch: log write failed: %s", exc)

    # L3-2 Self-Doubt (2026-06-08): post-template Sonnet call scores
    # system confidence in this verdict + lists caveats. Surfaces in
    # event metrics + (Tier E) /approvals UI. Anti-trust UX. Lazy
    # import + try/except (graceful degradation — failure doesn't
    # block emit).
    # Phase 1 Commit 4 (2026-06-09): Tier D pre-routing per spec
    # §15.A3. Non-alpha investment roles (insurance / diversifier
    # / hedge) bypass Tier C entirely and route to Tier D
    # (role-specific review queue). Tier D produces diagnostic
    # metrics + human review trigger but NO verdict, NO confidence.
    #
    # Phase 3 (deferred) fills in insurance / diversifier
    # methodology per Bondarenko 2014 / Kelly-Pruitt 2014 /
    # Asness 2017 / Ilmanen 2011 / Engle DCC 2002.
    if tr.verdict in ("GREEN", "MARGINAL", "RED"):
        try:
            from engine.agents.strengthener.tier_d_review import (
                should_route_to_tier_d, dispatch_tier_d,
            )
            if should_route_to_tier_d(spec):
                tier_d_result = dispatch_tier_d(
                    spec, family_hint, tr,
                    dispatch_event_id=out.get("dispatch_event_id"),
                )
                out["tier_d_result"] = tier_d_result
                # Tier D dispatch: return WITHOUT running Tier C lenses
                # or self_doubt or emit (per A3, no automated verdict
                # for non-alpha roles).
                logger.info(
                    "factor_dispatch: %s routed to Tier D "
                    "(investment_role=%s); skipping Tier C",
                    spec.hypothesis_id,
                    tier_d_result.get("investment_role"),
                )
                return out
        except Exception:
            logger.exception(
                "factor_dispatch: Tier D routing failed for %s; "
                "falling through to Tier C as conservative default",
                spec.hypothesis_id,
            )

    # Phase 1 Commit 3 (2026-06-09): role-aware declarative lens
    # routing per docs/spec_role_aware_test_routing.md v2.
    #
    # Replaces 5 hardcoded lens calls (L2-4 anchor, L2-5 subsample,
    # L2-6 industry, cross-asset macro, and the hardcoded
    # cross_asset_signals whitelist) with iteration over the lens
    # registry. Each lens declares its own applicability + DAG
    # dependencies + conditional skip predicates.
    #
    # Per spec §15.A5: every lens decision (execute / skip / fail) is
    # recorded in routing_decisions for audit trail.
    #
    # Backwards compat: legacy FactorSpecs without the 7 role-axis
    # fields fall back to infer_legacy_axes() heuristics. The chosen
    # fallback values are logged in routing_decisions so the principal
    # can see "we ran X because we inferred asset_class=equity from
    # signal_kind=cross_sectional_rank".
    anchor_orthogonality = None
    subsample_stability = None
    industry_extension = None
    cross_asset_extension = None
    specification_robustness = None
    routing_decisions: list = []
    if tr.verdict in ("GREEN", "MARGINAL", "RED"):
        pnl_df = (tr.artifacts or {}).get("pnl_series_df")
        try:
            from engine.research.lens_registry import (
                discover_lenses, applicable_lenses,
                resolve_lens_dag, should_execute,
            )
            from engine.agents.strengthener.factor_spec_extractor import (
                infer_legacy_axes,
            )
            fallback_axes = infer_legacy_axes(spec)
            registry = discover_lenses()
            applicable = applicable_lenses(registry, spec, fallback_axes)
            ordered = resolve_lens_dag(applicable)
            applicable_names = {l.name for l in applicable}
            for declared in registry.values():
                if declared.name not in applicable_names:
                    routing_decisions.append({
                        "lens":      declared.name,
                        "action":    "skipped_inapplicable",
                        "reason":    (f"applicable_to does not match "
                                       f"spec metadata (after legacy "
                                       f"fallback)"),
                        "applicable_required": declared.applicable_to,
                    })

            lens_outputs: dict = {}
            for lens in ordered:
                proceed, skip_reason = should_execute(lens, lens_outputs)
                if not proceed:
                    routing_decisions.append({
                        "lens":   lens.name,
                        "action": "skipped_conditional",
                        "reason": skip_reason,
                    })
                    continue
                try:
                    result = lens.runner(spec, tr, lens_outputs)
                except Exception as exc:
                    logger.exception(
                        "factor_dispatch: lens %s raised for %s",
                        lens.name, spec.hypothesis_id,
                    )
                    routing_decisions.append({
                        "lens":   lens.name,
                        "action": "failed_exception",
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                if result is None:
                    routing_decisions.append({
                        "lens":   lens.name,
                        "action": "returned_none",
                        "reason": "lens runner returned None "
                                  "(insufficient overlap or data)",
                    })
                    continue
                lens_outputs[lens.name] = result
                routing_decisions.append({
                    "lens":   lens.name,
                    "action": "executed",
                })

            # Map lens outputs to legacy variable names for downstream
            # self_doubt / emit signature compatibility (Commits 4-5
            # will refactor to a single lens_outputs dict).
            #
            # B.1 (2026-06-09): fx_carry_anchor_regression shares the
            # `anchor_orthogonality` slot with anchor_regression. By
            # applicable_to design they're mutually exclusive
            # (equity → FF5+MOM; FX → LRV HML_FX+DOL) so this never
            # silently picks one over the other. self_doubt renders
            # the section header dynamically from the `anchor_library`
            # field in the output dict.
            anchor_orthogonality  = (
                lens_outputs.get("anchor_regression")
                or lens_outputs.get("fx_carry_anchor_regression")
            )
            subsample_stability   = lens_outputs.get("subsample_stability")
            industry_extension    = lens_outputs.get("industry_extension")
            cross_asset_extension = lens_outputs.get("cross_asset_extension")
            specification_robustness = (
                lens_outputs.get("specification_robustness")
            )
            if anchor_orthogonality is not None:
                out["anchor_orthogonality"]   = anchor_orthogonality
            if subsample_stability is not None:
                out["subsample_stability"]    = subsample_stability
            if industry_extension is not None:
                out["industry_extension"]     = industry_extension
            if cross_asset_extension is not None:
                out["cross_asset_extension"]  = cross_asset_extension
            if specification_robustness is not None:
                out["specification_robustness"] = specification_robustness
        except Exception:
            logger.exception(
                "factor_dispatch: lens_registry iteration failed for %s",
                spec.hypothesis_id,
            )
            routing_decisions.append({
                "lens":   "_registry",
                "action": "failed_exception",
                "reason": "registry discovery / DAG resolution failed",
            })

        if routing_decisions:
            out["routing_decisions"] = routing_decisions

    # N (2026-06-10): the 4 senior gates over the net PnL — DSR given
    # family n_trials, ρ₁ serial-correlation smell, paper-OOS decay
    # ratio, and statistical power of T_GREEN at this sample length.
    # Computed HERE (not as a registry lens) because it applies
    # unconditionally to every PnL and needs dispatcher-level context
    # (family n_trials) that lens runners don't receive.
    pnl_diagnostics = None
    fam_n_self_doubt = 0
    if tr.verdict in ("GREEN", "MARGINAL", "RED"):
        try:
            # 2026-06-12: strategy_family, not claim mechanism_family
            from engine.research.strategy_family_classifier import (
                strategy_family_for_spec,
            )
            _sf = strategy_family_for_spec(spec)
            fam_n_self_doubt = _family_n_trials_now(_sf)
        except Exception:
            logger.exception("factor_dispatch: fam_n recompute failed")
        try:
            from engine.research.pnl_diagnostics import (
                compute_pnl_diagnostics,
            )
            from engine.research.lens_helpers import (
                resolve_default_net_col,
            )
            _arts = tr.artifacts or {}
            _pnl_df = _arts.get("pnl_series_df")
            if _pnl_df is not None and len(_pnl_df) > 0:
                _net_col = resolve_default_net_col(_arts)
                if _net_col and _net_col in _pnl_df.columns:
                    pnl_diagnostics = compute_pnl_diagnostics(
                        _pnl_df[_net_col],
                        n_trials_family = fam_n_self_doubt,
                        paper_window    = spec.paper_original_window,
                    )
            if pnl_diagnostics is not None:
                out["pnl_diagnostics"] = pnl_diagnostics
        except Exception:
            logger.exception("factor_dispatch: pnl_diagnostics failed "
                              "for %s", spec.hypothesis_id)

    self_doubt = None
    if tr.verdict in ("GREEN", "MARGINAL", "RED"):
        try:
            from engine.agents.strengthener.self_doubt import (
                assess_self_doubt,
            )
            self_doubt = assess_self_doubt(
                spec, tr,
                family_hint              = family_hint,
                n_trials_family          = fam_n_self_doubt,
                anchor_orthogonality     = anchor_orthogonality,
                subsample_stability      = subsample_stability,
                industry_extension       = industry_extension,
                cross_asset_extension    = cross_asset_extension,
                specification_robustness = specification_robustness,
                pnl_diagnostics          = pnl_diagnostics,
                routing_decisions        = routing_decisions or None,
            )
            if self_doubt:
                out["self_doubt"] = _dc.asdict(self_doubt)
        except Exception:
            logger.exception("factor_dispatch: self_doubt failed for %s",
                              spec.hypothesis_id)

    # C-2c: emit factor_verdict_filed for GREEN/MARGINAL/RED.
    # Internal verdicts (PENDING_TEMPLATE_BUILD, DATA_ERROR, etc.)
    # stay in audit log only — emit_tier_c_verdict short-circuits.
    # Lazy import to keep dispatcher module light + break the
    # potential dispatcher↔emitter import cycle.
    try:
        from engine.agents.strengthener.factor_verdict_emit import (
            emit_tier_c_verdict,
        )
        event_id = emit_tier_c_verdict(
            spec, family_hint, tr,
            dispatch_event_id        = out["dispatch_event_id"],
            self_doubt               = self_doubt,
            anchor_orthogonality     = anchor_orthogonality,
            subsample_stability      = subsample_stability,
            industry_extension       = industry_extension,
            cross_asset_extension    = cross_asset_extension,
            specification_robustness = specification_robustness,
            pnl_diagnostics          = pnl_diagnostics,
            routing_decisions        = routing_decisions or None,
        )
        if event_id:
            out["verdict_event_id"] = event_id
            # belief-2 autopsy (2026-06-12): join the just-emitted verdict
            # event back to the belief-1 prediction (both keyed by
            # subject_id ~ hypothesis_id) and write a surprise diagnostic
            # row to data/research/autopsies.jsonl. Lazy import + try/
            # except — autopsy is monitoring infrastructure, never gates
            # the verdict emit it depends on. Air-gap preserved: autopsy
            # reads BOTH predictions + verdicts AFTER they're produced;
            # never writes to predictions.jsonl.
            try:
                from engine.research.belief_autopsy import (
                    run_autopsy_for_verdict_event,
                )
                _autopsy = run_autopsy_for_verdict_event(event_id)
                if _autopsy is not None:
                    out["autopsy_id"] = _autopsy.autopsy_id
            except Exception:
                logger.exception("belief_autopsy failed for %s",
                                   spec.hypothesis_id)
            # External adversarial audit (Mitigation #1 from self-audit
            # blind-spots doctrine 2026-06-13). Routes verdict event
            # through a non-Anthropic LLM for independent review.
            # Defaults to stub provider (no API call) until
            # EXTERNAL_AUDIT_PROVIDER env var is set. Failure non-fatal.
            try:
                from engine.research.external_audit import (
                    audit_verdict_event,
                )
                # Re-load the just-emitted event for full audit payload
                from engine.research_store.store import by_event_id
                _ev_obj = by_event_id(event_id)
                if _ev_obj is not None:
                    _audit = audit_verdict_event(_ev_obj.to_dict())
                    if _audit.severity != "skipped":
                        out["external_audit_id"] = _audit.audit_id
                        out["external_audit_severity"] = _audit.severity
            except Exception:
                logger.exception("external_audit failed for %s",
                                   spec.hypothesis_id)
            # 2026-06-11 back-fill: dispatch_log row was written BEFORE
            # emit_tier_c_verdict ran, so the row doesn't carry the
            # resulting event_id. Append a "lineage_update" sidecar row
            # so downstream consumers can join dispatch_event_id →
            # verdict_event_id by a single dispatch_log scan instead of
            # needing the subject_id round-trip via events.jsonl.
            try:
                _append_dispatch_log({
                    "dispatch_event_id":  out["dispatch_event_id"],
                    "ts":                 _utc_iso(),
                    "hypothesis_id":      spec.hypothesis_id,
                    "kind":               "lineage_update",
                    "verdict_event_id":   event_id,
                    "prediction_id":      out.get("prediction_id"),
                    "autopsy_id":         out.get("autopsy_id"),
                    "actor":              out["actor"],
                }, path)
            except OSError as _exc:
                logger.warning("factor_dispatch: lineage_update append "
                                 "failed: %s", _exc)
    except Exception as exc:
        # Emit failure must NOT block audit log write — that already
        # happened above. Log + carry on.
        logger.exception("factor_dispatch: verdict emit failed for %s",
                          spec.hypothesis_id)

    return out
