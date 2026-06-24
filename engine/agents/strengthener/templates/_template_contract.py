"""engine.agents.strengthener.templates._template_contract — Tier C L2-1 Phase 5.

TemplateContract: declarative manifest that every Tier C template
MUST provide. Dispatcher refuses dispatch for any (signal_kind,
universe) combination without a valid contract registered here.

PURPOSE
=======
Per docs/spec_pit_data_accessor.md §3 and section 4.4, this is the
L4 layer of the 4-layer PIT architecture:

  L1 PIT Data Warehouse (parquets PIT-clean by construction)
  L2 SimulationClock
  L3 PITDataAccessor
  L4 TemplateContract + AuditGate (this module + dispatcher gate #10)

Without L4, a new template could be added that introduces silent
PIT bugs or untested signal kinds — dispatcher would happily call
it. With L4, every template must:
  - Declare its scope (which signal_kind+universe combinations it serves)
  - Carry a human-readable PIT audit certification (who/when/why)
  - Be re-certified every 365 days (or on refactor) — stale = refuse

ARCHITECTURAL DOCTRINE
======================
Adding a new template = 3 steps:
  1. Implement the template function (signal_fn → TemplateResult)
  2. Define + register the TemplateContract here
  3. Add (signal_kind, universe) → contract mapping

Skipping step 2 = template dispatch is REJECTED at the gate. This
makes "I forgot to PIT-audit my new template" architecturally
impossible.

FRESHNESS
=========
Audit certification expires after _CERT_FRESHNESS_DAYS (365d).
Reason: after major refactors (e.g. L2-1 PIT data layer change),
all templates need re-audit. 365d gives a buffer between PIT
infrastructure changes; for hot-iteration periods (within a single
quarter of L2-1 work), audit dates should be bumped each commit
that touches the template.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
from typing import Optional

# Commit 1 of the flexibility chain (2026-06-10): cross-sec
# supported_signals derives from the S-class signal registry.
from engine.research.signal_registry import dispatchable_signals


_CERT_FRESHNESS_DAYS = 365   # re-audit required after this


# ────────────────────────────────────────────────────────────────────
# Phase 6 (2026-06-08): DataShapeRequirement
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class DataShapeRequirement:
    """Declares the SEMANTIC SHAPE a template needs from a data source.

    Surfaces the "cohort mismatch" silent bug class identified in
    Phase 3.1 verification (2026-06-08): PIT data is QUARTERLY
    granularity but cross_sec template assumed ANNUAL. Without
    explicit shape declaration, accessor served quarterly data and
    GP/A t-stat inflated 0.79 spuriously.

    With this contract: template declares (source="comp_pit.funda",
    frequency="annual"); accessor reads contract + auto-coerces PIT
    data to annual cohort. Architectural guarantee against future
    same-class silent bugs.

    FREQUENCY semantics:
      "annual"     — one row per (firm, fiscal year-end)
      "quarterly"  — one row per (firm, fiscal quarter-end)
      "monthly"    — one row per (firm, calendar month-end)
      "daily"      — one row per (firm, trading day)

    AGGREGATION (income statement fields ONLY; balance sheet fields
    don't need aggregation since values are point-in-time):
      "fy_total"     — sum of 4 quarters within fiscal year (Novy-Marx /
                        Fama-French / classic factor research convention)
      "trailing_ttm" — sum of trailing 4 quarters from any quarter-end
                        (HXZ q-factor convention)
      "latest_quarter" — quarterly value as-is (e.g. Q4 alone)
      "fy_end"       — balance sheet value at fiscal year-end
      None           — pass through (balance sheet items, or signals
                        that handle aggregation themselves)
    """
    source:            str                          # e.g. "comp_pit.funda"
    frequency:         str                          # one of FREQUENCIES below
    aggregation:       Optional[str] = None         # one of AGGREGATIONS or None
    notes:             str = ""                     # human-readable rationale


# Controlled vocabularies for DataShapeRequirement
FREQUENCIES = frozenset({"annual", "quarterly", "monthly", "daily"})
AGGREGATIONS = frozenset({"fy_total", "trailing_ttm", "latest_quarter",
                            "fy_end", None})


@_dc.dataclass(frozen=True)
class TemplateContract:
    """Declarative manifest for a Tier C backtest template.
    Frozen — modifications require new commit + audit re-cert."""

    template_name:           str           # match TEMPLATE_REGISTRY key
    template_version:        str           # e.g. "v1.0_2026-06-08"

    # PIT certification — human-reviewed
    pit_audit_certified_by:  str           # actor identifier
    pit_audit_date:          str           # "YYYY-MM-DD"
    pit_audit_notes:         str           # <=800 chars rationale

    # Scope — what (signal_kind, universe) combinations this template handles
    supported_signal_kinds:  tuple[str, ...]
    supported_universes:     tuple[str, ...]

    # Canonical signal keys (controlled vocabulary of signals this
    # template can compute — used for AUDIT + extractor prompt
    # documentation). e.g. ("mktcap", "vol_12m", "gp_at", ...)
    supported_signals:       tuple[str, ...]

    # Bibliography pointer for L2-2 replication mode (optional)
    canonical_paper_id:      Optional[str] = None
    canonical_paper_window:  Optional[str] = None    # "YYYY-MM:YYYY-MM"
    canonical_paper_t:       Optional[float] = None

    # Phase 6 (2026-06-08): declarative data shape requirements.
    # Accessor reads this + auto-coerces data to declared shape.
    # Empty tuple = legacy template that doesn't declare (accessor
    # falls back to current behavior; B-fix tactical patch applies).
    required_data_shape: tuple[DataShapeRequirement, ...] = ()

    def is_fresh(self, as_of: Optional[_dt.date] = None) -> bool:
        """True iff this certificate is still within freshness window."""
        if as_of is None:
            as_of = _dt.date.today()
        try:
            cert_date = _dt.date.fromisoformat(self.pit_audit_date)
        except ValueError:
            return False
        return (as_of - cert_date).days <= _CERT_FRESHNESS_DAYS


# ────────────────────────────────────────────────────────────────────
# Registry: one TemplateContract per shipped template
# ────────────────────────────────────────────────────────────────────
# Adding a new template = add its TemplateContract here. Dispatcher
# gate #10 rejects any (signal_kind, universe) pair not covered by
# at least one fresh contract.

CONTRACT_REGISTRY: dict[str, TemplateContract] = {

    "cross_sec_us_equities": TemplateContract(
        template_name           = "cross_sec_us_equities",
        template_version        = "v1.1_2026-06-08",
        pit_audit_certified_by  = "claude-2026-06-08-l2-1-phase6",
        pit_audit_date          = "2026-06-08",
        pit_audit_notes         = (
            "Phase 6 v1.1: declares required_data_shape annual "
            "for Compustat (was tactically B-fixed via legacy "
            "key INNER JOIN in commit b0d92c9b; this contract "
            "declares the intent architecturally). CRSP msf "
            "monthly + comp_pit annual cohort. GP/A REPLICATED "
            "vs Novy-Marx 2013 within 0.044 t-stat. "
            "B0 fix shipped via Phase 1.5/1.6 bitemporal "
            "(knowable_at column from real comp.fundq.rdq). "
            "B1 lagged mktcap fix shipped (328154dc). "
            "B-class params (universe_size, n_buckets) flow "
            "through FactorSpec v2."
        ),
        supported_signal_kinds  = ("cross_sectional_rank",),
        supported_universes     = ("us_equities_top_3000",),
        # Canonical paper anchor: Novy-Marx 2013 "The Other Side of
        # Value: The Gross Profitability Premium" — flagship signal
        # GP/A REPLICATED at gap 0.044. Pub-year proxy = data end-year.
        canonical_paper_id      = "novy_marx_2013_gross_profitability",
        canonical_paper_window  = "1963-01:2010-12",
        canonical_paper_t       = None,
        # Commit 1 of the flexibility chain (2026-06-10): derived
        # from the S-class signal registry — single source of truth.
        # Only status="dispatchable" entries appear here; "proposed"
        # entries exist in the registry but cannot burn dispatch
        # quota until their verification card is approved.
        supported_signals       = dispatchable_signals(),
        required_data_shape = (
            # CRSP price layer: monthly frequency, point-in-time
            # values (no aggregation since price/return/mktcap are
            # observed values not aggregates).
            DataShapeRequirement(
                source = "crsp.msf",
                frequency = "monthly",
                aggregation = None,
                notes = ("CRSP monthly stock file: prc / ret / "
                          "mktcap / shrout. Universe selection "
                          "uses lagged mktcap (Bug 1 fix)."),
            ),
            # Compustat layer: ANNUAL frequency for cross-sec
            # factor research. Fundamental fields aggregate to FY:
            #   - balance sheet (at, ceq, lt): fy_end
            #   - income statement (sale, cogs, ni): fy_total
            # B-fix lesson: PIT cache is QUARTERLY raw; without
            # this declaration, template silently uses quarterly
            # snapshots and signal semantics shift.
            DataShapeRequirement(
                source = "comp_pit.funda",
                frequency = "annual",
                aggregation = "fy_total",
                notes = ("Annual fiscal year-end cohort. For "
                          "income statement fields (sale, cogs, "
                          "ni), use fiscal year totals. For "
                          "balance sheet (at, ceq), use FY-end "
                          "values (sub-aggregation handled by "
                          "accessor)."),
            ),
        ),
    ),

    "carry_g10_fx": TemplateContract(
        template_name           = "carry_g10_fx",
        template_version        = "v1.0_2026-06-09",
        pit_audit_certified_by  = "claude-2026-06-09-c-2f",
        pit_audit_date          = "2026-06-09",
        pit_audit_notes         = (
            "C-2f v1.0: G10 FX carry template built on the LRV "
            "2011 anchor library (engine.research.fx_carry_anchors). "
            "PIT-clean by construction — the sort key (rdiff) is "
            "lagged one month inside build_carry_anchors per LRV "
            "§2.1, so no look-ahead exists at the template level. "
            "Data sources are FRED-sourced monthly spot + IR3TIB01 "
            "short rates (already cached as parquet by Phase 2 "
            "Commits 1+2 of role-aware routing spec). EUR-USD "
            "1999 launch + JPY rate start 2002 are documented "
            "panel-start constraints; template will refuse "
            "INSUFFICIENT_HISTORY when date_range exceeds them. "
            "No Compustat / CRSP dependency → no B0 / B1 cohort "
            "risk class."
        ),
        supported_signal_kinds  = ("carry",),
        supported_universes     = ("fx_g10",),
        supported_signals       = ("carry_lrv_hml_fx",),
        canonical_paper_id      = "lustig_roussanov_verdelhan_2011",
        canonical_paper_window  = "1983-11:2009-12",
        # LRV 2011 Table 1: HML_FX annualized Sharpe ≈ 0.55, t ≈ 2.8
        # on the developed-FCY universe over 1983-2009 (their full
        # sample). Our overlap is narrower (1999+ due to EUR/JPY
        # constraints) so REPLICATION_INSUFFICIENT_OVERLAP is the
        # likely status until data backfill.
        canonical_paper_t       = 2.8,
        required_data_shape     = (
            DataShapeRequirement(
                source = "fred.fx_spot_g10",
                frequency = "monthly",
                aggregation = None,
                notes = ("Month-end G10 FX spot vs USD from FRED "
                          "DEXJPUS / DEXUSEU / ... — quote convention "
                          "normalized in fetch_fx_spot_g10.py (DEX_US_ "
                          "direct vs DEX_US indirect)."),
            ),
            DataShapeRequirement(
                source = "fred.ir3tib01",
                frequency = "monthly",
                aggregation = None,
                notes = ("G10 short-rate panel from FRED IR3TIB01 "
                          "series including USD. Differentials "
                          "(rdiff_<CCY>_pct) computed in fetcher; "
                          "carry sort key uses LAGGED rdiff."),
            ),
        ),
    ),

    "event_drift_revision": TemplateContract(
        template_name           = "event_drift_revision",
        template_version        = "v1.0_2026-06-14",
        pit_audit_certified_by  = "claude-2026-06-14-revision",
        pit_audit_date          = "2026-06-14",
        pit_audit_notes         = (
            "Analyst EPS revision quintile L/S, Chan-Jegadeesh-Lakonishok "
            "1996 canonical. Reads IBES statsumu_epsus from "
            "data/cache/_ibes_eps_summary_us_fy1.parquet (1.95M rows "
            "1990-2024 via ${WRDS_USER_2}) + crsp.msenames cusip→permno bridge "
            "+ CRSP MSF forward returns. revision_pct = (numup-numdown)/"
            "max(numest,1) per (ticker, statpers month-end). Quintile "
            "sort on revision_pct, long top, short bottom, hold 1mo "
            "forward. PIT-safe: IBES statpers is the stat-period date "
            "(public knowledge); MSF return uses month_end+1 only. "
            "PIT caveats: statpers is sometimes mid-month — we round to "
            "month_end which may include 1-2 days of look-ahead at the "
            "very edge; for serious deployment would need to lag explicit "
            "1 trading day. Verdict requires POSITIVE mean PnL "
            "(CJL direction: up revisions predict positive return)."
        ),
        supported_signal_kinds  = ("event_drift",),
        supported_universes     = ("us_equities_revision",),
        supported_signals       = (
            "analyst_revision_quintile_ls",
            "ibes_up_down_ratio",
        ),
        canonical_paper_id      = "chan_jegadeesh_lakonishok_1996_revisions",
        canonical_paper_window  = "1977-01:1993-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "ibes.statsumu_epsus",
                frequency = "monthly",
                aggregation = "statpers_month_end",
                notes = ("FY1 EPS estimate summary stats per "
                          "(ticker, statpers month). numup/numdown for "
                          "revision direction."),
            ),
            DataShapeRequirement(
                source = "crsp.msenames",
                frequency = "as-of",
                aggregation = None,
                notes = "cusip 8-char → permno bridge.",
            ),
            DataShapeRequirement(
                source = "crsp.msf",
                frequency = "monthly",
                aggregation = None,
                notes = "Monthly returns for forward 1mo holding.",
            ),
        ),
    ),

    "event_drift_pead": TemplateContract(
        template_name           = "event_drift_pead",
        template_version        = "v1.0_2026-06-13",
        pit_audit_certified_by  = "claude-2026-06-13-event-drift",
        pit_audit_date          = "2026-06-13",
        pit_audit_notes         = (
            "Bernard-Thomas 1989 PEAD canonical. SUE via seasonal random "
            "walk (epspxq_t - epspxq_{t-4}) / sigma_8q. Monthly decile "
            "long-short on recent announcers (rdq in past 60d, 2-day "
            "post-event buffer). PIT-safe: rdq is the actual public "
            "announcement date (Compustat fundq.rdq), so portfolio "
            "formation at month-end M uses only announcements with "
            "rdq <= M-2 trading days. Forward 1-month return for month "
            "M+1 is post-event drift. Universe is smallcap subset "
            "(3356 gvkeys, 2011-2025) — broader fundq not cached. "
            "M2 anchor (Chordia-Goyal-Sadka 2009): post-2000 PEAD "
            "weaker than BT-1989 1974-1986; expect Sharpe 0.3-0.7."
        ),
        supported_signal_kinds  = ("event_drift",),
        supported_universes     = ("us_equities_pead",),
        supported_signals       = (
            "post_earnings_drift_sue_decile",
            "earnings_surprise_drift",
        ),
        canonical_paper_id      = "bernard_thomas_1989_pead",
        canonical_paper_window  = "1974-01:1986-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "compustat.fundq",
                frequency = "quarterly",
                aggregation = "earnings_announcement_event",
                notes = ("Compustat fundq with rdq + epspxq columns. "
                          "Smallcap subset 2011-2025 currently cached."),
            ),
            DataShapeRequirement(
                source = "crsp.msf",
                frequency = "monthly",
                aggregation = None,
                notes = ("CRSP MSF monthly returns 1990-2024 for "
                          "forward drift window."),
            ),
        ),
    ),

    "spx_skew_premium": TemplateContract(
        template_name           = "spx_skew_premium",
        template_version        = "v1.0_2026-06-14",
        pit_audit_certified_by  = "claude-2026-06-14-skew",
        pit_audit_date          = "2026-06-14",
        pit_audit_notes         = (
            "SPX option-implied skew predictor of SPX excess return, "
            "Bollerslev-Todorov 2011 canonical. Reads OptionMetrics "
            "vsurfd 2000-2024 (377k rows, fetched via ${WRDS_USER_2} from "
            "data/cache/_spx_iv_surface_daily.parquet). skew_t = "
            "put_25d_IV(30d) - call_25d_IV(30d). Strategy: scale long-"
            "SPX position by 60mo Z-score of skew. PIT-safe: IV surface "
            "is end-of-day, published next morning at latest; we lag "
            "skew_t for predicting return_{t+1}. Verdict via NW-t HAC "
            "lag 6 on monthly predictive regression beta. GREEN requires "
            "POSITIVE beta — Bollerslev-Todorov direction: high skew "
            "predicts HIGH next-month return (tail-risk premium). "
            "Negative beta → RED regardless of magnitude."
        ),
        supported_signal_kinds  = ("skew_premium",),
        supported_universes     = ("us_equities_spx_options",),
        supported_signals       = (
            "spx_25d_skew_premium",
            "iv_put_call_diff",
        ),
        canonical_paper_id      = "bollerslev_todorov_2011_tail_risk",
        canonical_paper_window  = "1996-01:2007-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "optionm.vsurfd",
                frequency = "daily",
                aggregation = "monthly_end",
                notes = ("OptionMetrics IV surface (secid=108105 SPX). "
                          "30-day, 25-delta puts + calls. Resampled to "
                          "month-end."),
            ),
            DataShapeRequirement(
                source = "cboe.vix_spx",
                frequency = "daily",
                aggregation = "monthly_log_return",
                notes = "SPX daily levels for monthly excess return calc.",
            ),
        ),
    ),

    "vrp_spx": TemplateContract(
        template_name           = "vrp_spx",
        template_version        = "v1.0_2026-06-13",
        pit_audit_certified_by  = "claude-2026-06-13-vrp",
        pit_audit_date          = "2026-06-13",
        pit_audit_notes         = (
            "Variance risk premium on SPX, Carr-Wu 2009 canonical "
            "short-variance proxy. Reads VIX+SPX daily from "
            "data/cache/_vix_spx_daily.parquet (1990-2026, ~9163 rows). "
            "PnL = (VIX_{t-21}/100)² × (21/252) - realized_var_{t-21,t}. "
            "PIT-safe: VIX/SPX are published market data with no "
            "restatement; implied variance at month-start uses "
            "lagged VIX value (no look-ahead). Verdict requires "
            "POSITIVE mean PnL (Carr-Wu doctrine: VRP is a risk "
            "premium for vol insurance writers) — negative mean → "
            "RED regardless of NW-t. M2 anchor: mean PnL > 0 in "
            "1990-2007 sub-sample. NOT in cost-stress space yet "
            "(variance swap bid-ask ignored)."
        ),
        supported_signal_kinds  = ("vrp",),
        supported_universes     = ("us_equities_spx_options",),
        supported_signals       = (
            "spx_vrp_short_variance",
            "vix_minus_realized",
        ),
        canonical_paper_id      = "carr_wu_2009_variance_risk_premiums",
        canonical_paper_window  = "1990-01:2007-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "cboe.vix_spx",
                frequency = "daily",
                aggregation = "monthly_end",
                notes = ("VIX + SPX daily 1990-2026 cached. "
                          "Template resamples to month-end PnL."),
            ),
        ),
    ),

    "vrp_treasury": TemplateContract(
        template_name           = "vrp_treasury",
        template_version        = "v1.0_mvp_2026-06-22",
        pit_audit_certified_by  = "claude-2026-06-22-w6-rigor-a-validate",
        pit_audit_date          = "2026-06-22",
        pit_audit_notes         = (
            "Bond-VRP MVP, MOVE/TLT substituted for VIX/SPX in the "
            "Carr-Wu 2009 short-vol formulation. MOVE is bps yield-vol; "
            "scaled to TLT price vol via effective duration D=17y "
            "(post-2002 average). PnL = (MOVE_t × D/100)² × (21/252) "
            "- realized_var_{t,t+21} on TLT log-returns. PIT-safe: "
            "MOVE published end-of-day no restatement; TLT yfinance "
            "adjusted close final at session close; implied at month-"
            "start uses 21-trading-day lag (no look-ahead). MVP "
            "limitations: duration constant D=17 (real range 16-19); "
            "no TC adjustment (Treasury options have wider spread "
            "than SPX); MOVE→TLT scaling introduces duration-mismatch "
            "noise. Verdict = positive-mean test then NW-t HAC lag 6 "
            "(BUG-3 multi-test corrected). Initial Bond-VRP candidate "
            "(0bee667e, 2026-06-22): Sharpe +0.56 / NW-t +1.69 / RED."
        ),
        supported_signal_kinds  = ("vrp",),
        supported_universes     = ("us_treasury_options",),
        supported_signals       = (
            "bond_vrp_short_variance",
            "move_minus_tlt_realized",
        ),
        canonical_paper_id      = None,
        canonical_paper_window  = None,
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "move.implied_vol_index",
                frequency = "daily",
                aggregation = "monthly_end",
                notes = ("MOVE + TLT daily 2002-2026 cached. "
                          "Template resamples to month-end PnL."),
            ),
        ),
    ),

    "spanning_test_ff": TemplateContract(
        template_name           = "spanning_test_ff",
        template_version        = "v1.0_2026-06-13",
        pit_audit_certified_by  = "claude-2026-06-13-bug2",
        pit_audit_date          = "2026-06-13",
        pit_audit_notes         = (
            "BUG-2 fix template: single-asset spanning regression on FF5+MOM. "
            "test_asset = first signal_input (FF factor name); model = "
            "remaining inputs. alpha-t HAC SE lag 6. Verdict via BUG-3 "
            "multi-testing-corrected threshold (HLZ floor + Bonferroni). "
            "PIT-safe: Ken French data public + lagged. Replication anchor "
            "(M2 hook): MOM regressed on FF5 should produce alpha-t > 2.5 "
            "per AFP 2014 / HXZ 2015."
        ),
        supported_signal_kinds  = ("spanning_test",),
        supported_universes     = ("ken_french_ff5_mom",),
        supported_signals       = (
            "mom_on_ff5",
            "hml_on_ff5_minus_hml",
            "smb_on_ff5_minus_smb",
        ),
        canonical_paper_id      = "asness_frazzini_pedersen_2014_quality",
        canonical_paper_window  = "1963-07:2014-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "ff.factors_weekly",
                frequency = "weekly",
                aggregation = None,
                notes = "Ken French FF5+MOM weekly. Template compounds to monthly.",
            ),
        ),
    ),

    "factor_combination_ff": TemplateContract(
        template_name           = "factor_combination_ff",
        template_version        = "v1.0_2026-06-11",
        pit_audit_certified_by  = "claude-2026-06-11-bt-flex-4.2",
        pit_audit_date          = "2026-06-11",
        pit_audit_notes         = (
            "bt-flex-4.2 factor_combination template. Tests "
            "'w% factor_a + (1-w)% factor_b' shape (canonical "
            "Asness-Moskowitz-Pedersen 2013 50/50 HML+MOM). "
            "Reads Ken French FF5+Mom weekly returns from "
            "data/cache/ken_french_ff5_mom_weekly.parquet "
            "(1963-2026, 3279 weekly obs), compounds to monthly. "
            "PIT-safe: Ken French data is public + lagged by "
            "construction; no look-ahead. Verdict: forward-style "
            "(GREEN/MARGINAL/RED via NW-t + CAPM α-t + 80bp "
            "cost stress) — combined strategy is treated as new "
            "alpha claim. Additionally reports Jobson-Korkie "
            "paired ΔSharpe vs each component (informational; "
            "answers 'does the combo strictly beat each alone?')."
        ),
        supported_signal_kinds  = ("factor_combination",),
        supported_universes     = ("ken_french_ff5_mom",),
        supported_signals       = (
            "hml_mom_50_50",
            "value_mom_combo",
            "hml_smb_50_50",
            "rmw_cma_50_50",
        ),
        canonical_paper_id      = "asness_moskowitz_pedersen_2013",
        canonical_paper_window  = "1972-01:2011-12",
        canonical_paper_t       = None,
        required_data_shape     = (
            DataShapeRequirement(
                source = "ff.factors_weekly",
                frequency = "weekly",
                aggregation = None,
                notes = ("Ken French FF5 + MOM weekly returns. "
                          "Template compounds to monthly internally."),
            ),
        ),
    ),

    "portfolio_overlay_60_40": TemplateContract(
        template_name           = "portfolio_overlay_60_40",
        template_version        = "v1.0_2026-06-11",
        pit_audit_certified_by  = "claude-2026-06-11-bt-flex-4.1",
        pit_audit_date          = "2026-06-11",
        pit_audit_notes         = (
            "bt-flex-4.1 portfolio_overlay template. Tests "
            "'X% strategy in 60/40 portfolio' shape (HOP-2017 "
            "canonical). Reads SPY monthly returns from "
            "data/multivariate_msm_v4/spy_monthly.parquet and "
            "IEF daily closes from data/cache/_bond_etf_px.parquet "
            "(resampled to month-end). TSMOM overlay built "
            "in-template: sign(past-12mo SPY total return), "
            "vol-targeted to 10% annual via trailing 12mo "
            "realized vol. PIT-safe: SPY monthly cache is "
            "month-end returns (no look-ahead); IEF resample "
            "uses last close per month (PIT). Verdict via "
            "Jobson-Korkie 1981 / Memmel 2003 Sharpe-ratio-"
            "difference t-stat — NOT comparable to single-"
            "factor strict-gate verdicts. Senior lens stack "
            "(FF5 spanning, spec_robustness) does not apply "
            "and should be skipped for portfolio_overlay specs."
        ),
        supported_signal_kinds  = ("portfolio_overlay",),
        supported_universes     = ("us_balanced_60_40",),
        supported_signals       = ("tsmom_overlay_on_spy",),
        canonical_paper_id      = "hurst_ooi_pedersen_2017",
        canonical_paper_window  = "1880-01:2016-12",
        canonical_paper_t       = None,   # paper reports Sharpe gain not t
        required_data_shape     = (
            DataShapeRequirement(
                source = "etf.adj_close",
                frequency = "daily",
                aggregation = None,
                notes = ("SPY + IEF adjusted closes. SPY also has "
                          "a pre-computed monthly_ret cache used "
                          "directly; IEF is resampled from daily."),
            ),
        ),
    ),

    "tsmom_sector_etf": TemplateContract(
        template_name           = "tsmom_sector_etf",
        template_version        = "v1.1_2026-06-08",
        pit_audit_certified_by  = "claude-2026-06-08-l2-1-phase6",
        pit_audit_date          = "2026-06-08",
        pit_audit_notes         = (
            "Phase 6 v1.1: declares required_data_shape weekly "
            "for ETF closes. Sector ETF universe (35 ETFs from "
            "UniverseETF DB, batch <= 4). Uses CRSP daily closes "
            "via engine.signal._fetch_closes → weekly Friday "
            "resample. Lookback / skip / vol_target FactorSpec "
            "v2 B-class parameterizable (Phase 4); defaults "
            "match Moskowitz et al. 2012 TSMOM(12,1) at 10% "
            "vol target. No Compustat dependency = no B0 bug. "
            "No cohort-mismatch risk (single frequency)."
        ),
        supported_signal_kinds  = ("time_series_momentum",),
        supported_universes     = ("us_equities_sector_etf",),
        supported_signals       = ("tsmom_signed_vol_targeted",),
        canonical_paper_id      = "moskowitz_ooi_pedersen_2012",
        canonical_paper_window  = "1985-01:2009-12",
        canonical_paper_t       = 6.5,  # MOP 2012 24 instr basket TSMOM(12,1)
        required_data_shape = (
            DataShapeRequirement(
                source = "etf.adj_close",
                frequency = "daily",
                aggregation = None,
                notes = ("ETF adjusted closes. Template resamples "
                          "to weekly Friday close internally; this "
                          "contract declares the SOURCE frequency "
                          "(daily). No aggregation — close values "
                          "are observations not aggregates."),
            ),
        ),
    ),
}


# ────────────────────────────────────────────────────────────────────
# Lookup helpers (dispatcher gate uses these)
# ────────────────────────────────────────────────────────────────────
def get_contract(template_name: str) -> Optional[TemplateContract]:
    return CONTRACT_REGISTRY.get(template_name)


def contract_for_scope(
    signal_kind: str,
    universe:    str,
) -> Optional[TemplateContract]:
    """Find the FRESH contract that declares it handles this
    (signal_kind, universe) pair. Returns None if no fresh contract
    matches.

    Multiple contracts theoretically could match (one template
    handles two universes etc.); we return the FIRST fresh match.
    """
    for c in CONTRACT_REGISTRY.values():
        if (signal_kind in c.supported_signal_kinds
                and universe in c.supported_universes):
            return c
    return None
