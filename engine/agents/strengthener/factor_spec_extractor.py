"""engine.agents.strengthener.factor_spec_extractor — Tier C-1.

Translates a B-approved factor Hypothesis (predicted_direction !=
zero) into a STRUCTURED backtest SPEC JSON. The dispatcher (C-2)
then maps SPEC.signal_kind to a pre-built engine.factor_lab
template and runs the backtest deterministically.

Per docs/spec_tier_c_factor_backtest_auto_dispatcher.md:

  "Spec-first, code never": the LLM never writes executable Python.
  It only fills slots in a structured SPEC. The dispatcher maps
  SPEC to a pre-built engine.factor_lab template function. Human
  approves the SPEC (not LLM-written code).

Pattern-5-compliant design (same as procedural_dispatcher):
  - Single Sonnet call with strict JSON tool_use schema
  - LLM picks ONE signal_kind from a CONTROLLED enum; can NOT add
    new kinds
  - 'requires_custom_code' escape hatch is the legitimate "no
    template fits" verdict — surfaces to /approvals as a reminder
    for human to take over

Why this exists — found 2026-06-08:
  90% of A's recent factor candidates need backtesting against
  historical data, which currently requires the principal to
  hand-write the strategy_fn + register the StrategySpec + invoke
  run_single_strategy_weekly. 5 controlled signal_kind templates
  (cross_sectional_rank, time_series_momentum, carry, vrp,
  event_drift) cover ~80% of A's expected output per audit on
  207 hypotheses. The remaining 20% legitimately needs custom
  code; the escape hatch surfaces them clearly.

Per [[project-a-plus-b-substrate-first-roadmap-2026-06-05]] capital
line: this spec extraction is RESEARCH automation only. A GREEN
verdict from auto-test does NOT auto-promote to paper_trade — the
principal still owns the capital decision.

Architectural boundary:
  - input:  ONE Hypothesis (B-approved, factor-shape)
  - output: Optional[FactorSpec] (None on LLM failure or escape
            hatch; caller surfaces to /approvals either way)
  - NO I/O in this module: caller persists / surfaces / dispatches
  - NO factor_lab import: this module is pure spec extraction
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import re
from typing import Optional

# Top-level import for monkeypatch in tests (same pattern as review.py)
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Controlled signal_kind enum — single source of truth
# ────────────────────────────────────────────────────────────────────
# Adding a kind = building the matching template in factor_dispatcher
# (C-2). LLM can ONLY pick from these names; cannot invent new ones.
#
# Coverage estimate per 2026-06-08 audit of 207 factor candidates:
#   cross_sectional_rank   ~30% (PROFITABILITY + VALUE + LOW_VOL + SIZE)
#   carry                  ~19% (CARRY across FX/rates/commodity/equity)
#   time_series_momentum   ~7%  (MOMENTUM + CROSS_ASSET_MOMENTUM)
#   vrp                    ~8%  (VOL_RISK_PREMIUM)
#   event_drift            ~16% (EARNINGS_DRIFT + ATTENTION + SENTIMENT)
#   requires_custom_code   ~20% (microstructure, methodology, novel)
# Total templates coverage: ~80% per spec's "70%+ → DO Tier C" gate.
SIGNAL_KINDS = (
    "cross_sectional_rank",   # equity X-sec factor (size/value/BAB/NMC)
    "time_series_momentum",   # TSMOM single-asset or cross-asset basket
    "carry",                  # FX/rates/commodity/equity carry premium
    "vrp",                    # variance risk premium / vol carry
    "event_drift",            # PEAD / analyst-revision / attention drift
    "portfolio_overlay",      # bt-flex-4.1 — strategy X overlaid at K%
                              # weight on a base portfolio (e.g. 20% TSMOM
                              # in 60/40 per Hurst-Ooi-Pedersen 2017)
    "factor_combination",     # bt-flex-4.2 — w% factor_a + (1-w)% factor_b
                              # tested as a single combined strategy
                              # (e.g. 50/50 HML+MOM per Asness-Moskowitz-
                              # Pedersen 2013 "Value & Momentum Everywhere")
    "skew_premium",           # 2026-06-14 — SPX option-implied skew as
                              # predictor of next-month SPX excess return
                              # (Bollerslev-Todorov 2011); uses OptionMetrics
                              # vsurfd via ${WRDS_USER_2} account
    "spanning_test",          # BUG-2 fix (2026-06-13) — "is X spanned by
                              # model M?" Avoids Sonnet drift that turned
                              # FF5-vs-FF3 claims into combo specs.
    "requires_custom_code",   # escape hatch — no template fits cleanly
)


# Universe identifiers the dispatcher knows how to materialize.
# Adding a universe = wiring the matching loader in factor_dispatcher
# (C-2). LLM can ONLY pick from these names.
UNIVERSES = (
    "us_equities_top_3000",   # CRSP top-3000-by-cap monthly rebal
    "us_equities_sp500",      # S&P 500 constituents (point-in-time)
    "us_equities_sector_etf", # 11 SPDR sector ETFs (for sector momentum)
    "fx_g10",                 # G10 FX crosses vs USD
    "commodity_futures_27",   # 27-contract continuous futures basket
    "us_treasury_curve",      # 2y/5y/10y/30y on-the-run
    "global_equity_indices",  # MSCI country indices (cross-country mom)
    "us_balanced_60_40",      # bt-flex-4.1 — 60% SPY + 40% IEF (7-10y
                              # treasury) base portfolio for overlay tests
    "ken_french_ff5_mom",     # bt-flex-4.2 — Ken French FF5+Mom weekly
                              # factor returns 1963-2026 (HML/MOM/SMB/
                              # RMW/CMA/MKT_RF) for factor_combination
    "us_equities_spx_options",# vrp template — SPX index + VIX daily
                              # 1990-2026 (Carr-Wu 2009 short-vol)
    "us_equities_pead",       # event_drift template — smallcap Compustat
                              # fundq + CRSP MSF (Bernard-Thomas 1989 PEAD)
    "us_equities_revision",   # event_drift template (revision variant) —
                              # IBES statsumu_epsus + CRSP MSF + msenames
                              # bridge (Chan-Jegadeesh-Lakonishok 1996)
    "unknown_universe",       # LLM signals "spec needs a universe I don't have"
)


# Rebalance frequencies the dispatcher understands. Anything else =
# requires_custom_code.
REBAL_FREQS = ("daily", "weekly", "monthly", "quarterly")


# Weighting schemes the dispatcher implements deterministically.
WEIGHTINGS = (
    "decile_long_short_dollar_neutral",  # top D10 long, bottom D10 short
    "quintile_long_short_dollar_neutral",
    "tercile_long_short_dollar_neutral",
    "long_only_topN",
    "rank_weighted",
    "signed_signal_volatility_targeted",  # TSMOM-style
    "equal_weight_basket",
)


# ────────────────────────────────────────────────────────────────────
# Output dataclass — what the dispatcher receives
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class FactorSpec:
    """LLM-extracted factor backtest SPEC. Frozen post-extraction.
    Dispatcher (C-2) reads this + invokes the matching template."""

    hypothesis_id:       str        # the source Hypothesis row
    signal_kind:         str        # CONTROLLED enum (see SIGNAL_KINDS)
    universe:            str        # CONTROLLED enum (see UNIVERSES)
    date_range:          str        # "YYYY-MM:YYYY-MM" inclusive
    signal_inputs:       tuple[str, ...]   # cache-path identifiers
    rebal:               str        # CONTROLLED enum (see REBAL_FREQS)
    weighting:           str        # CONTROLLED enum (see WEIGHTINGS)
    expected_holding_period: str    # "weekly" / "monthly" / "quarterly"
    min_obs_months:      int        # minimum sample size for valid test
    pit_audits:          tuple[str, ...]   # ("restatement","lookahead","survivorship")
    cost_model:          str        # cost model identifier
    rationale:           str        # 1-2 sentences why this signal_kind fits

    # Diagnostics — populated by extractor, NOT by LLM
    extracted_ts:        str
    model:               str

    # L2-2 Replication Mode (2026-06-08): if the hypothesis references
    # a known academic paper, these fields enable template-level
    # replication check (run subsample stats on the overlap window,
    # flag MISMATCH if our t deviates >0.5σ from the paper's reported
    # t). Both optional + default None so existing FactorSpec
    # constructions stay valid. The extractor prompt is updated in a
    # separate piece (L2-2.1) to populate them when the hypothesis
    # has source_paper_id in the registry.
    paper_original_window: "str | None" = None    # "YYYY-MM:YYYY-MM"
    paper_reported_t:      "float | None" = None  # paper's |t| stat

    # L2-1 Phase 2.6 FactorSpec v2 (2026-06-08): B-class parameters
    # — research design choices the LLM extractor SHOULD be able to
    # propose so Layer 3 critic/variant/self-doubt can suggest
    # actionable variants. All Optional + default None so the
    # template falls back to its module-level default when not set
    # (parity-preserving vs pre-v2 FactorSpec).
    #
    # Dispatcher enforces typed-range validation via gate #9
    # B_CLASS_OUT_OF_RANGE (see factor_dispatcher.pre_dispatch_check).
    # LLM has design freedom WITHIN the safe range — cannot escape
    # the rails.
    #
    # Ranges per literature (cross-sec defaults in []):
    universe_size:       "int | None"   = None   # 100-5000   [3000]
    n_buckets:           "int | None"   = None   # 3-10       [5]
    signal_lookback_m:   "int | None"   = None   # 1-120 mo   [12 for vol/mom]
    signal_skip_m:       "int | None"   = None   # 0-12 mo    [1 for mom_12_1]
    vol_target_annual:   "float | None" = None   # 0.03-0.30  [0.10 for tsmom]
    weighting_scheme_alt: "str | None"  = None   # "ew"|"vw"|"rank"  ["ew"]

    # ─────────────────────────────────────────────────────────────
    # Role-aware test routing axes (Phase 1, 2026-06-09)
    # Per docs/spec_role_aware_test_routing.md v2 (commit 2ca50bf2):
    # 7 Optional axes that allow declarative lens dispatch instead
    # of hardcoded signal_kind whitelists. All Optional — legacy
    # dispatch (pre-2026-06-09) falls back to inferred values via
    # infer_legacy_axes(). When LLM extractor leaves a field None
    # (paper unclear), fallback inference fills from signal_kind /
    # universe / mechanism_family.
    #
    # Schemas locked in spec §3 + §15.A1 + §15.A2.
    # ─────────────────────────────────────────────────────────────

    # A2: ROLE split — investment vs statistical (per spec §15.A2)
    # investment_role: what the sleeve DOES in portfolio
    #   alpha / insurance / diversifier / hedge / overlay
    # statistical_role: the return process's statistical character
    #   directional / mean_reverting / arbitrage / market_making
    #   / event_driven
    # These are INDEPENDENT (merger arb = alpha + arbitrage).
    investment_role:        "str | None" = None
    statistical_role:       "str | None" = None

    # asset_class: equity / fixed_income / fx / commodity /
    # multi_asset / cross_asset
    asset_class:            "str | None" = None

    # mechanism: behavioral / risk_premium / arbitrage / structural
    # / event_driven / microstructure
    # (Distinct from family_hint passed via Hypothesis. family_hint
    # is used for n_trials DSR counter; mechanism is a finer-grained
    # finance taxonomy.)
    mechanism:              "str | None" = None

    # horizon: high_frequency / daily / weekly / monthly / quarterly
    horizon:                "str | None" = None

    # A1 NEW axes (per spec §15.A1):

    # capacity_tier: under_100m / 100m_to_1b / over_1b / unknown
    # Influences cost_stress thresholds (over_1b factors need market
    # impact modeling) + crowdedness anchor selection.
    capacity_tier:          "str | None" = None

    # data_dependency_type: fundamental / price / microstructure /
    # alternative / macro
    # Distinct staleness/PIT/granularity requirements per type.
    # Mixing them under one abstraction caused B0/B1/B2 bugs.
    data_dependency_type:   "str | None" = None

    # regime_sensitivity: stationary / known_regime_break / unknown
    # If known_regime_break, L2-5 subsample window must cut at
    # regime boundaries, NOT equal 4-split.
    regime_sensitivity:     "str | None" = None

    def is_escape_hatch(self) -> bool:
        """True if LLM signaled 'no template fits'. Caller should
        surface to /approvals as a custom_code reminder, NOT
        auto-dispatch."""
        return self.signal_kind == "requires_custom_code"


# ────────────────────────────────────────────────────────────────────
# Legacy axis inference (Phase 1, 2026-06-09)
#
# Pre-spec-v2 dispatch produced FactorSpec without any of the 7
# role-routing axes. To preserve dispatch correctness for those
# (and for any future spec where LLM leaves fields None because
# the paper doesn't say), infer_legacy_axes() fills the most
# probable values from signal_kind + universe + family_hint.
#
# Inference quality is "best-effort heuristic" — NOT the same
# rigor as LLM extraction from paper. When inferred values cause
# routing differences, audit trail (see spec §15.A5) records
# action="inferred" so user can review.
# ────────────────────────────────────────────────────────────────────

# Signal kind → (asset_class, statistical_role) heuristic mapping
# Derived from current SIGNAL_KINDS taxonomy + institutional convention.
_SIGNAL_KIND_INFERENCE = {
    # equity factors
    "cross_sectional_rank": ("equity",      "directional"),
    "size_factor":          ("equity",      "directional"),
    "value_factor":         ("equity",      "directional"),
    "mom_12_1":             ("equity",      "directional"),
    "reversal_1m":          ("equity",      "mean_reverting"),
    "ibes_revisions":       ("equity",      "event_driven"),
    "pead":                 ("equity",      "event_driven"),
    # cross-asset
    "time_series_momentum": ("cross_asset", "directional"),
    "tsmom_signal":         ("cross_asset", "directional"),
    "carry":                ("cross_asset", "directional"),
    "fx_carry":             ("fx",          "directional"),
    "commodity_carry":      ("commodity",   "directional"),
    # arbitrage / event
    "merger_arb":           ("equity",      "arbitrage"),
    "convertible_arb":      ("multi_asset", "arbitrage"),
    "vrp":                  ("multi_asset", "arbitrage"),
    "fomc_drift":           ("equity",      "event_driven"),
}


def infer_legacy_axes(spec: FactorSpec) -> dict:
    """Return a dict of inferred axis values for fields the spec leaves
    None. Helper for dispatcher backwards-compat; not invoked when LLM
    extraction filled the field.

    Inference is heuristic — explicit fields in spec WIN. Output is
    only meant to fill gaps.
    """
    out: dict = {}
    sk_lookup = _SIGNAL_KIND_INFERENCE.get(spec.signal_kind, (None, None))
    inferred_ac, inferred_sr = sk_lookup

    if spec.asset_class is None and inferred_ac is not None:
        out["asset_class"] = inferred_ac
    if spec.statistical_role is None and inferred_sr is not None:
        out["statistical_role"] = inferred_sr

    # investment_role: default to alpha unless escape_hatch
    if spec.investment_role is None:
        out["investment_role"] = "alpha"

    # mechanism: default to None (Hypothesis.family_hint covers
    # n_trials accounting; mechanism is a finer-grained label
    # the LLM should populate when the paper allows)
    # → leave None if extractor didn't fill

    # horizon: infer from rebal frequency
    if spec.horizon is None:
        if spec.rebal == "monthly":
            out["horizon"] = "monthly"
        elif spec.rebal == "weekly":
            out["horizon"] = "weekly"
        elif spec.rebal == "quarterly":
            out["horizon"] = "quarterly"
        elif spec.rebal == "daily":
            out["horizon"] = "daily"

    # capacity_tier: default unknown — too risky to guess from
    # universe size alone (a top-3000 backtest can be deployed at
    # any AUM the user picks)
    if spec.capacity_tier is None:
        out["capacity_tier"] = "unknown"

    # data_dependency_type: heuristic from signal_inputs paths
    if spec.data_dependency_type is None:
        inputs_str = " ".join(spec.signal_inputs).lower()
        if "funda" in inputs_str or "fundq" in inputs_str:
            out["data_dependency_type"] = "fundamental"
        elif "msf" in inputs_str or "dsf" in inputs_str or "adj_close" in inputs_str:
            out["data_dependency_type"] = "price"
        elif "fred" in inputs_str or "macro" in inputs_str:
            out["data_dependency_type"] = "macro"
        elif "ibes" in inputs_str:
            out["data_dependency_type"] = "alternative"  # analyst forecasts

    # regime_sensitivity: default to unknown — true classification
    # requires academic familiarity with the factor history
    if spec.regime_sensitivity is None:
        out["regime_sensitivity"] = "unknown"

    return out


# ────────────────────────────────────────────────────────────────────
# Eligibility check — same shape as is_procedural_hypothesis
# ────────────────────────────────────────────────────────────────────
# Procedural / methodology / overfit-research subtypes that should
# go through procedural_dispatcher, NOT factor_spec_extractor. Caught
# 2026-06-08 audit: 40+ "OTHER family" candidates are actually
# methodology research that fell through A's mechanism_family enum
# (no METHODOLOGY family yet). They have predicted_direction=positive
# from the LLM but are NOT factor backtests.
_METHODOLOGY_SUBTYPE_RE = re.compile(
    r"(multiple_testing|backtest_overfit|false_discovery|"
    r"data_snooping|optimal_stopping|factor_zoo|"
    r"transaction_cost_estim|price_impact|liquidity_estim)",
    re.IGNORECASE,
)


def is_factor_hypothesis(h) -> bool:
    """True if a Hypothesis is a candidate for factor-spec extraction:
      - predicted_direction != zero (zero = procedural, goes elsewhere)
      - mechanism_subtype is NOT methodology/overfit/microstructure
        research (those legitimately need custom code, route directly
        to escape hatch — no need to spend Sonnet $0.03)
      - has at least one source_chunk OR synthesizes_event_ids OR
        synthesizes_paper_ids (so spec extractor has provenance)
    """
    try:
        if h.predicted_direction.value == "zero":
            return False
    except AttributeError:
        return False
    if _METHODOLOGY_SUBTYPE_RE.search(h.mechanism_subtype or ""):
        return False
    if not (h.source_chunk_ids or h.synthesizes_paper_ids
            or h.synthesizes_event_ids):
        return False
    return True


# ────────────────────────────────────────────────────────────────────
# Prompt assembly + tool schema
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are translating ONE B-approved factor research hypothesis into a
STRUCTURED backtest SPEC for the deterministic dispatcher. Your role
is NOT to write code or invent new test methodologies; only to MAP
the hypothesis to ONE of the CONTROLLED signal_kind templates +
extract the universe/dates/inputs.

The dispatcher is deterministic Python — it reads your SPEC + invokes
a pre-built engine.factor_lab template. The principal will approve
your SPEC (universe, dates, inputs) before the dispatcher runs. NO
human-reads-Python step. If the hypothesis doesn't fit ANY template
cleanly, pick signal_kind='requires_custom_code' — the principal will
write the test by hand. Do NOT force-fit.

CONTROLLED signal_kind options:

  cross_sectional_rank
    Rank-and-bucket a cross-section of assets by a signal (e.g.
    profitability, book-to-market, idiosyncratic vol, BAB). Long
    top decile/quintile, short bottom. Dollar-neutral. Use for:
    PROFITABILITY, VALUE, LOW_VOL, SIZE, QUALITY composite, etc.

  time_series_momentum
    TSMOM on a single asset OR cross-asset basket: long if past
    return positive, short if negative, signal magnitude proportional
    to past return. Use for: MOMENTUM, CROSS_ASSET_MOMENTUM, trend
    on commodity/FX/index baskets.

  carry
    Carry premium: long high-carry, short low-carry within an asset
    class. Use for: FX carry (interest rate differential), commodity
    carry (front/back roll), equity carry (dividend yield), rates
    carry (slope of curve).

  vrp
    Variance risk premium / vol carry: short volatility, long realized
    vol or vice-versa. SHIPPED 2026-06-13 — Carr-Wu 2009 canonical
    short-variance proxy on SPX index + VIX.
    Required:
      - universe='us_equities_spx_options'
      - signal_inputs = ('cboe.vix_spx.vix', 'cboe.vix_spx.spx')
        (notional inputs; template reads cached VIX+SPX daily 1990-2026)
      - rebal='monthly'
      - weighting_scheme_alt: unused (variance-swap convention is fixed)
    CHOOSE THIS for: "implied vol exceeds realized vol systematically"
    / "short variance / short straddle earns positive risk premium"
    / "VRP is significantly positive" claims on US equity index vol.
    NOT for: cross-sectional vol (e.g. "low-vol anomaly" → use
    cross_sectional_rank), or VIX futures basis-trading (no template yet).

  event_drift
    Buy/sell triggered by an EVENT (earnings surprise, analyst
    revision, attention spike, sentiment flip). Hold for K days/weeks
    then close. SHIPPED 2026-06-13 — Bernard-Thomas 1989 PEAD
    canonical, smallcap Compustat fundq + CRSP MSF 2011-2024.
    Required:
      - universe='us_equities_pead'
      - signal_inputs = ('compustat.fundq.epspxq', 'compustat.fundq.rdq')
        (notional inputs; template auto-loads cached fundq + CRSP MSF)
      - rebal='monthly'
      - weighting='quintile_long_short_dollar_neutral' (template enforces
        decile sort internally regardless)
    CHOOSE THIS for: PEAD claims, SUE-sorted long-short, earnings-
    announcement-driven drift, analyst-revision drift (latter would
    need IBES data, currently RED-routed to capability_gaps).
    NOT for: methodology claims about WHY PEAD exists (use
    requires_custom_code) or DECAY_STUDY of PEAD over time
    (router refuses with NEEDS_NEW_TEMPLATE).

  spanning_test
    Tests claims like "Is anomaly X spanned/subsumed by model M?" /
    "Does adding factor F to model M improve spanning?" / "Anomaly X
    is subsumed by FF5". Regression of test asset's excess return on
    a set of competing model factors with HAC SE; alpha-t against
    HLZ multi-testing-corrected threshold.
    Required:
      - universe='ken_french_ff5_mom'
      - signal_inputs structure: FIRST entry = test asset, REST = model
        factors. All entries are 'ff.factors_weekly.<x>' (x in
        {hml, mom, smb, rmw, cma, mkt_rf}).
        Need ≥3 entries (1 test + ≥2 model). Model can't include the
        test asset.
        Example: ('ff.factors_weekly.mom', 'ff.factors_weekly.mkt_rf',
                   'ff.factors_weekly.smb', 'ff.factors_weekly.hml',
                   'ff.factors_weekly.rmw', 'ff.factors_weekly.cma')
        tests "is MOM spanned by FF5?"
      - weighting_scheme_alt: unused
    CHOOSE THIS for SPANNING / SUBSUMPTION claims. NOT factor_combination
    (which tests a NEW STRATEGY built from blending). Do NOT stretch
    spanning claims into combo specs — that is the documented BUG-2
    Sonnet drift the spanning_test template exists to prevent.

  factor_combination
    Blends two factor return series at w / (1-w) weights and tests the
    COMBINATION as a single new strategy (e.g. "50/50 value+momentum"
    per Asness-Moskowitz-Pedersen 2013 "Value and Momentum Everywhere").
    Outputs Sharpe / NW-t / 80bp cost-stressed verdict AND Jobson-Korkie
    paired ΔSharpe vs each component (does the combo strictly beat
    either alone?).
    Required:
      - universe='ken_french_ff5_mom'
      - signal_inputs = two factor names from
        {hml, mom, smb, rmw, cma, mkt_rf} as
        ('ff.factors_weekly.<a>', 'ff.factors_weekly.<b>')
      - weighting_scheme_alt parsed as weight on first factor (0.05-0.95);
        default 0.50 if absent or unparseable
    CHOOSE THIS when claim describes "w% factor_a + (1-w)% factor_b combo"
    where the inputs are KNOWN academic factors (HML/MOM/SMB/RMW/CMA/MKT)
    NOT raw signals.

  portfolio_overlay
    Blends a strategy at K% weight into a fixed base portfolio (e.g.
    "20% TSMOM in 60/40"). Outputs portfolio-level metrics (Sharpe,
    MaxDD, Vol) AND a Jobson-Korkie t-stat for Sharpe-ratio difference
    vs the base. NOT a single-factor alpha test. Use for: portfolio
    allocation studies (HOP-2017 60/40 + TSMOM), risk-overlay
    proposals, fixed-weight strategy blending.
    Required:
      - universe='us_balanced_60_40' (the only base portfolio wired today)
      - signal_inputs MUST start with 'etf.adj_close.' (the template
        reads SPY + IEF from the ETF close cache; use
        ('etf.adj_close.spy', 'etf.adj_close.ief'))
      - overlay pct: the template defaults to 0.20 (canonical HOP).
        For non-default pct, leave weighting='ew' but set the rationale
        to mention the target pct — the template's pct parser falls
        back gracefully if extractor cannot encode the pct.
    CHOOSE THIS, NOT requires_custom_code, whenever the claim describes
    "add X% strategy Y to portfolio Z" structure.

  requires_custom_code
    No template fits — examples: novel microstructure research,
    methodology-only papers (multiple-testing corrections,
    overfit detection), conditional/regime-switching with non-
    standard estimators, papers requiring data sources outside the
    dispatcher's whitelist. Set signal_kind='requires_custom_code'
    + leave universe='unknown_universe' if data source isn't covered.
    REQUIRED: rationale must name WHY no template fits.

CONTROLLED universe options:

  us_equities_top_3000     — CRSP top-3000-by-cap monthly rebal
  us_equities_sp500        — S&P 500 constituents (point-in-time)
  us_equities_sector_etf   — 11 SPDR sector ETFs
  fx_g10                   — G10 FX crosses vs USD
  commodity_futures_27     — 27-contract continuous futures basket
  us_treasury_curve        — 2y / 5y / 10y / 30y on-the-run
  global_equity_indices    — MSCI country indices
  us_balanced_60_40        — 60% SPY + 40% IEF (7-10y treasury) base
                             portfolio; pair with signal_kind=
                             portfolio_overlay
  ken_french_ff5_mom       — Ken French FF5+MOM weekly factor returns
                             1963-2026; pair with signal_kind=
                             factor_combination
  us_equities_spx_options  — SPX index level + VIX daily 1990-2026;
                             pair with signal_kind=vrp for variance
                             risk premium / short-vol claims
                             (Carr-Wu 2009 canonical)
  us_equities_pead         — smallcap Compustat fundq (epspxq + rdq,
                             2011-2025) + CRSP MSF returns; pair with
                             signal_kind=event_drift for post-earnings-
                             announcement drift claims (Bernard-Thomas
                             1989 canonical)
  unknown_universe         — spec needs a universe the dispatcher
                             doesn't have; pair with signal_kind=
                             requires_custom_code

CONTROLLED rebal options: daily / weekly / monthly / quarterly

CONTROLLED weighting options:
  decile_long_short_dollar_neutral
  quintile_long_short_dollar_neutral
  tercile_long_short_dollar_neutral
  long_only_topN
  rank_weighted
  signed_signal_volatility_targeted   (TSMOM-style)
  equal_weight_basket

CONTROLLED pit_audits (pick all that apply):
  restatement, lookahead, survivorship

date_range format: "YYYY-MM:YYYY-MM" inclusive. Default to the
hypothesis's required_data window if specified; otherwise pick a
reasonable window matching the universe's data coverage (e.g.
us_equities_* default 2000-01:2024-12, fx_g10 default 1999-01:
2024-12).

min_obs_months: floor of 60 (5 years). Most factor research needs
≥120 (10 years) for credible Sharpe; ≥180 (15 years) for credible
out-of-sample. Pick based on hypothesis's expected_holding_period
and statistical power.

cost_model: always "engine.execution.cost_model.basic" for now
(default 2-3 bp per round-trip; revisit per signal_kind in C-2).

OPTIONAL B-CLASS PARAMETERS (Phase 2.6 FactorSpec v2)
======================================================
These let you propose specific research-design variants when the
hypothesis or paper SPECIFIES a non-default value. LEAVE NULL unless
the hypothesis text directly calls for a specific value. Defaults
match canonical academic conventions.

  universe_size: Optional[int] in [100, 5000]
    Override the default universe size (e.g. 3000 for top_3000).
    POPULATE when:
      - Hypothesis says "small-cap" / "micro-cap" → 500 or 1000
      - Hypothesis says "Russell 1000" → 1000
      - Hypothesis says "top 500" / "S&P 500-like" → 500
      - Paper specifies a sub-universe by mktcap rank
    LEAVE NULL when: hypothesis just says "US equities" / no size constraint

  n_buckets: Optional[int] in [3, 10]
    Override the default 5 (quintile L/S). Other common values:
    3 (tercile), 10 (decile).
    POPULATE when:
      - Paper used "decile" or "10-bucket" → 10
      - Paper used "tertile" or "3-bucket" → 3
      - Hypothesis emphasizes extreme buckets → 10
    LEAVE NULL when: no specific bucketing mentioned, or "quintile" / "5"

  signal_lookback_m: Optional[int] in [1, 120]
    Lookback window in MONTHS for momentum / vol / other lookback-
    based signals.
    POPULATE when:
      - "12-1 momentum" / "12-month lookback" → 12
      - "6-month momentum" → 6
      - "60-month low-vol" → 60
    LEAVE NULL when: no specific lookback mentioned

  signal_skip_m: Optional[int] in [0, 12]
    Skip period (months) for momentum-style signals.
    POPULATE when:
      - "skip last month" / "12-1 momentum" → 1
      - "skip 2 months" / "12-2 momentum" → 2
    LEAVE NULL when: no skip mentioned

  vol_target_annual: Optional[float] in [0.03, 0.30]
    Per-asset vol target for TSMOM-style sizing.
    POPULATE when:
      - "15% vol target" → 0.15
      - "5% vol target" → 0.05
    LEAVE NULL when: no specific vol target

  weighting_scheme_alt: Optional in {"ew", "vw", "rank", null}
    For cross-sec L/S: alternative weighting WITHIN each bucket.
    POPULATE when:
      - "value-weighted" / "VW" → "vw"
      - "equal-weighted" / "EW" → "ew"
      - "rank-weighted" → "rank"
    LEAVE NULL when: no specific scheme (default behavior used)

CONSERVATIVE PRINCIPLE: prefer NULL over guessing. If you populate
a B-class field, the dispatcher will REPLACE the safe default.
Wrong value here = silent miscalibration. ONLY populate when the
hypothesis text or paper title/abstract explicitly indicates the
value.

ROLE-AWARE ROUTING AXES (Phase 1, 2026-06-09)
==============================================
Per docs/spec_role_aware_test_routing.md v2: 7 Optional axes that
the dispatcher uses to decide WHICH rigor lenses apply to this
sleeve. Same NULL-prefers-NULL rule as B-class — when the paper
doesn't say, leave NULL and let dispatcher fall back to inferred
heuristics.

ROLE SPLIT (per spec §15.A2)
  investment_role: what the sleeve DOES in portfolio
    Optional in {"alpha", "insurance", "diversifier", "hedge", "overlay"}
    POPULATE when paper clearly signals:
      - "outperformance" / "anomaly" / "risk premium" → "alpha"
      - "tail hedge" / "crash insurance" / "convex payoff" → "insurance"
      - "uncorrelated stream" / "diversification benefit" → "diversifier"
      - "hedges X factor exposure" → "hedge"
      - "overlay strategy" / "modifies existing book" → "overlay"
    LEAVE NULL when: paper doesn't clearly say (most academic anomalies
                       default to "alpha" via legacy inference anyway)

  statistical_role: the return process's statistical character
    Optional in {"directional", "mean_reverting", "arbitrage",
                  "market_making", "event_driven"}
    POPULATE based on the return process:
      - long-only / long-bias OR L/S that bets on trend → "directional"
      - L/S that bets on convergence to fair value → "mean_reverting"
      - capture deal-spread / convergence between linked securities → "arbitrage"
      - inventory / liquidity provision → "market_making"
      - bet on specific event resolution (earnings, M&A, FOMC) → "event_driven"
    KEY: investment_role and statistical_role are INDEPENDENT.
    Example: merger arb = alpha (investment_role) + arbitrage
    (statistical_role).

ASSET STRUCTURE
  asset_class: Optional in {"equity", "fixed_income", "fx",
                              "commodity", "multi_asset", "cross_asset"}
    POPULATE when the paper specifies:
      - US equities / global stocks → "equity"
      - Treasury / corporate / TIPS / sovereign bonds → "fixed_income"
      - FX spot / forwards → "fx"
      - commodity futures → "commodity"
      - portfolio across MULTIPLE asset classes → "multi_asset"
      - SAME signal applied across asset classes for cross-section
        diversification → "cross_asset" (e.g., G10 carry across stocks
        + bonds + FX + commodities)
    LEAVE NULL when: paper unclear

  mechanism: Optional in {"behavioral", "risk_premium", "arbitrage",
                            "structural", "event_driven",
                            "microstructure"}
    finer-grained economic explanation:
      - "underreaction" / "overreaction" / "investor psychology" → "behavioral"
      - "compensation for risk" / "ICAPM extension" → "risk_premium"
      - "structural feature of market" (e.g., demand for index funds) → "structural"
      - bid-ask spread / order flow → "microstructure"
    LEAVE NULL when: paper doesn't propose a mechanism

  horizon: Optional in {"high_frequency", "daily", "weekly",
                          "monthly", "quarterly"}
    POPULATE from rebal frequency or paper's stated horizon
    LEAVE NULL: dispatcher infers from rebal field

A1 NEW AXES (capacity / data / regime)
  capacity_tier: Optional in {"under_100m", "100m_to_1b",
                                "over_1b", "unknown"}
    Estimated AUM capacity before alpha decay. POPULATE only if paper
    explicitly addresses capacity (rare in academic papers). LEAVE
    NULL → defaults to "unknown" (safe).

  data_dependency_type: Optional in {"fundamental", "price",
                                        "microstructure", "alternative",
                                        "macro"}
    Type of underlying data the signal needs. POPULATE:
      - Compustat fundamentals → "fundamental"
      - CRSP prices / returns → "price"
      - tick / quote data → "microstructure"
      - analyst estimates / news / sentiment → "alternative"
      - macro time series (FRED, etc.) → "macro"
    LEAVE NULL: dispatcher infers from signal_inputs paths.

  regime_sensitivity: Optional in {"stationary", "known_regime_break",
                                      "unknown"}
    Known regime structure of the factor. POPULATE:
      - paper documents stable performance across decades → "stationary"
      - paper documents crash regime (carry crash, momentum crash,
        2008, 2020) → "known_regime_break"
    LEAVE NULL: defaults to "unknown" (safe — uses equal subsample
                 splits).

CONSERVATIVE PRINCIPLE FOR ALL ROLE-AXIS FIELDS: prefer NULL over
guessing. Legacy inference will fill from signal_kind + universe +
data paths. When in doubt, NULL.

Output: invoke the emit_factor_spec tool EXACTLY ONCE.
"""


_SPEC_TOOL = {
    "name": "emit_factor_spec",
    "description": ("Emit ONE structured factor backtest SPEC for "
                    "this B-approved hypothesis. Pick signal_kind "
                    "from the controlled enum; if no template fits "
                    "cleanly, pick requires_custom_code."),
    "input_schema": {
        "type": "object",
        "properties": {
            "signal_kind": {
                "type": "string",
                "enum": list(SIGNAL_KINDS),
            },
            "universe": {
                "type": "string",
                "enum": list(UNIVERSES),
            },
            "date_range": {
                "type": "string",
                "pattern": r"^\d{4}-\d{2}:\d{4}-\d{2}$",
            },
            "signal_inputs": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "rebal": {
                "type": "string",
                "enum": list(REBAL_FREQS),
            },
            "weighting": {
                "type": "string",
                "enum": list(WEIGHTINGS),
            },
            "expected_holding_period": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly"],
            },
            "min_obs_months": {
                "type": "integer",
                "minimum": 60,
                "maximum": 600,
            },
            "pit_audits": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["restatement", "lookahead", "survivorship"],
                },
                "minItems": 0,
                "maxItems": 3,
            },
            "cost_model": {"type": "string"},
            "rationale": {
                "type": "string",
                "minLength": 20,
                "maxLength": 600,
            },
            # Phase 6 (2026-06-08): optional B-class FactorSpec v2 fields.
            # LLM populates ONLY when hypothesis text directly indicates
            # a non-default value. Conservative null-preferring.
            "universe_size": {
                "type": ["integer", "null"],
                "minimum": 100,
                "maximum": 5000,
                "description": (
                    "Override default universe size (e.g. 500 for "
                    "small-cap restriction). NULL = template default."),
            },
            "n_buckets": {
                "type": ["integer", "null"],
                "minimum": 3,
                "maximum": 10,
                "description": (
                    "Override quintile L/S default (5). Use 3 for "
                    "tercile, 10 for decile. NULL = template default."),
            },
            "signal_lookback_m": {
                "type": ["integer", "null"],
                "minimum": 1,
                "maximum": 120,
                "description": (
                    "Signal lookback in months (e.g. 12 for 12-1 mom). "
                    "NULL = template default."),
            },
            "signal_skip_m": {
                "type": ["integer", "null"],
                "minimum": 0,
                "maximum": 12,
                "description": (
                    "Months to skip in signal (e.g. 1 for skip-last-month). "
                    "NULL = template default."),
            },
            "vol_target_annual": {
                "type": ["number", "null"],
                "minimum": 0.03,
                "maximum": 0.30,
                "description": (
                    "Per-asset vol target for TSMOM sizing. NULL = "
                    "template default 10%."),
            },
            "weighting_scheme_alt": {
                "type": ["string", "null"],
                "enum": ["ew", "vw", "rank", None],
                "description": (
                    "Alternative within-bucket weighting (ew/vw/rank). "
                    "NULL = default."),
            },
            # Role-aware routing axes (Phase 1, 2026-06-09)
            # Per docs/spec_role_aware_test_routing.md v2
            "investment_role": {
                "type": ["string", "null"],
                "enum": ["alpha", "insurance", "diversifier",
                          "hedge", "overlay", None],
                "description": (
                    "What sleeve DOES in portfolio. NULL when paper "
                    "unclear → dispatcher infers 'alpha' as default."),
            },
            "statistical_role": {
                "type": ["string", "null"],
                "enum": ["directional", "mean_reverting", "arbitrage",
                          "market_making", "event_driven", None],
                "description": (
                    "Return process character. Independent from "
                    "investment_role (merger arb = alpha + arbitrage). "
                    "NULL when paper doesn't say."),
            },
            "asset_class": {
                "type": ["string", "null"],
                "enum": ["equity", "fixed_income", "fx", "commodity",
                          "multi_asset", "cross_asset", None],
                "description": (
                    "Asset class traded. NULL → dispatcher infers from "
                    "signal_kind."),
            },
            "mechanism": {
                "type": ["string", "null"],
                "enum": ["behavioral", "risk_premium", "arbitrage",
                          "structural", "event_driven",
                          "microstructure", None],
                "description": (
                    "Fine-grained mechanism. NULL = use Hypothesis "
                    "family_hint."),
            },
            "horizon": {
                "type": ["string", "null"],
                "enum": ["high_frequency", "daily", "weekly",
                          "monthly", "quarterly", None],
                "description": "Time horizon. NULL → infer from rebal.",
            },
            "capacity_tier": {
                "type": ["string", "null"],
                "enum": ["under_100m", "100m_to_1b", "over_1b",
                          "unknown", None],
                "description": (
                    "AUM capacity tier. NULL → 'unknown' (safe)."),
            },
            "data_dependency_type": {
                "type": ["string", "null"],
                "enum": ["fundamental", "price", "microstructure",
                          "alternative", "macro", None],
                "description": (
                    "Data type. NULL → dispatcher infers from "
                    "signal_inputs paths."),
            },
            "regime_sensitivity": {
                "type": ["string", "null"],
                "enum": ["stationary", "known_regime_break",
                          "unknown", None],
                "description": (
                    "Regime structure. NULL → 'unknown' (safe — uses "
                    "equal subsample splits)."),
            },
        },
        "required": [
            "signal_kind", "universe", "date_range", "signal_inputs",
            "rebal", "weighting", "expected_holding_period",
            "min_obs_months", "pit_audits", "cost_model", "rationale",
        ],
        "additionalProperties": False,
    },
}


def _format_user(h) -> str:
    """Render the Hypothesis as the user message body. Keep it tight
    — every field load-bearing for the spec decision; nothing else."""
    parts = [
        f"HYPOTHESIS_ID:       {h.hypothesis_id}",
        f"MECHANISM_FAMILY:    {h.mechanism_family.value}",
        f"MECHANISM_SUBTYPE:   {h.mechanism_subtype}",
        f"PREDICTED_DIRECTION: {h.predicted_direction.value}",
        f"PREDICTED_MAGNITUDE: {h.predicted_magnitude}",
        f"EXTRACTION_METHOD:   {h.extraction_method.value}",
        "",
        "CLAIM:",
        h.claim.strip(),
        "",
        "TEST_METHODOLOGY:",
        h.test_methodology.strip(),
        "",
        "REQUIRED_DATA:",
    ]
    for rd in (h.required_data or ()):
        parts.append(f"  - {rd}")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class ExtractionResult:
    """Phase 2.1 (2026-06-13): Stage-0 router verdict + Stage-1 spec.

    Either field can be None:
      - router_verdict None: Stage 0 skipped (h ineligible per
        is_factor_hypothesis); spec is also None.
      - router_verdict.refusal set, spec None: router refused to
        proceed (LOW_CONFIDENCE_CLASSIFY / NEEDS_NEW_TEMPLATE /
        WRONG_HYPOTHESIS_TYPE / CLASSIFIER_UNAVAILABLE).
      - router_verdict.is_actionable + spec None: Stage 1 LLM call
        failed or returned bad payload.
      - router_verdict.is_actionable + spec set: happy path.
    """
    spec:           Optional[FactorSpec]
    router_verdict: Optional["ClaimShapeVerdict"]  # forward ref to claim_shape_router

    @property
    def refusal_reason(self) -> Optional[str]:
        rv = self.router_verdict
        if rv is None:
            return None
        return rv.refusal


def extract_factor_spec_with_routing(h) -> ExtractionResult:
    """Phase 2.1: two-stage extraction with Stage-0 claim-shape router.

    Stage 0 classifies the claim into one canonical shape. If the shape
    is TESTABLE_FUTURE / NOT_TESTABLE / UNCLEAR, returns ExtractionResult
    with spec=None and router_verdict.refusal set — caller surfaces the
    refusal_reason rather than running Stage 1 ($0.005 saved per skip).

    Stage 1 is the existing extractor LLM call, now receiving a
    shape_hint guardrail in the user prompt.
    """
    import datetime as _dt
    from engine.agents.strengthener.claim_shape_router import (
        classify_claim_shape, shape_hint_for_extractor,
    )

    if not is_factor_hypothesis(h):
        return ExtractionResult(spec=None, router_verdict=None)

    # Stage 0 — claim-shape classification
    router_verdict = classify_claim_shape(h)
    if router_verdict.refusal is not None:
        logger.info(
            "factor_spec: router refused %s shape=%s reason=%s conf=%.2f",
            h.hypothesis_id, router_verdict.shape.value,
            router_verdict.refusal, router_verdict.confidence,
        )
        return ExtractionResult(spec=None, router_verdict=router_verdict)

    # Stage 1 — existing extractor, with shape hint prepended
    hint_line = shape_hint_for_extractor(router_verdict.shape)
    user_body = _format_user(h)
    if hint_line:
        user_body = hint_line + "\n\n" + user_body

    try:
        result = llm_call(
            workload   = "strengthener_factor_spec",
            system     = _SYSTEM_PROMPT,
            user       = user_body,
            agent_id   = "strengthener_factor_spec",
            tools      = [_SPEC_TOOL],
            max_tokens = 2048,
            scope      = "tier_c_factor_spec_extractor",
        )
    except Exception as exc:
        logger.warning("factor_spec: llm_call failed for %s: %s",
                        h.hypothesis_id, exc)
        return ExtractionResult(spec=None, router_verdict=router_verdict)

    spec = _build_spec_from_llm_result(h, result)
    return ExtractionResult(spec=spec, router_verdict=router_verdict)


def extract_factor_spec(h) -> Optional[FactorSpec]:
    """Single LLM call. Returns:
      - FactorSpec with signal_kind in SIGNAL_KINDS — caller routes
        to dispatcher (or escape hatch path if requires_custom_code)
      - None on hard failure (tool not called, invalid enum, parse
        error). Caller treats None as "extractor unavailable" and
        falls back to manual.

    No I/O in this function. Caller is responsible for:
      - eligibility check via is_factor_hypothesis(h) BEFORE calling
      - persisting the FactorSpec
      - surfacing approval atom to /approvals
      - invoking dispatcher (C-2) post-approval

    Phase 2.1 (2026-06-13): backward-compat wrapper around
    extract_factor_spec_with_routing. New callers should use the
    routing-aware version for richer refusal observability.
    """
    return extract_factor_spec_with_routing(h).spec


def _build_spec_from_llm_result(h, result) -> Optional[FactorSpec]:
    """Stage 1 result → FactorSpec. Internal helper extracted from the
    legacy extract_factor_spec body during the Phase 2.1 split. Returns
    None on bad payload (caller already logs context)."""
    import datetime as _dt

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_factor_spec":
            payload = tc.input
            break
    if payload is None:
        logger.warning("factor_spec: %s did not call emit_factor_spec "
                        "tool", h.hypothesis_id)
        return None

    signal_kind = str(payload.get("signal_kind") or "")
    if signal_kind not in SIGNAL_KINDS:
        logger.warning("factor_spec: %s emitted unknown signal_kind=%r",
                        h.hypothesis_id, signal_kind)
        return None

    universe = str(payload.get("universe") or "")
    if universe not in UNIVERSES:
        logger.warning("factor_spec: %s emitted unknown universe=%r",
                        h.hypothesis_id, universe)
        return None

    rebal = str(payload.get("rebal") or "")
    if rebal not in REBAL_FREQS:
        logger.warning("factor_spec: %s emitted unknown rebal=%r",
                        h.hypothesis_id, rebal)
        return None

    weighting = str(payload.get("weighting") or "")
    if weighting not in WEIGHTINGS:
        logger.warning("factor_spec: %s emitted unknown weighting=%r",
                        h.hypothesis_id, weighting)
        return None

    date_range = str(payload.get("date_range") or "")
    if not re.match(r"^\d{4}-\d{2}:\d{4}-\d{2}$", date_range):
        logger.warning("factor_spec: %s emitted bad date_range=%r",
                        h.hypothesis_id, date_range)
        return None

    # Phase 6 (2026-06-08): read Optional B-class FactorSpec v2 fields
    # from payload. LLM populates ONLY when hypothesis specifies a
    # non-default; otherwise None → template default applies.
    def _opt_int(v):
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            logger.warning("factor_spec: bad int payload %r ignored", v)
            return None

    def _opt_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            logger.warning("factor_spec: bad float payload %r ignored", v)
            return None

    def _opt_str_enum(v, allowed):
        if v is None or v == "":
            return None
        s = str(v)
        if s not in allowed:
            logger.warning("factor_spec: bad enum payload %r not in "
                              "%s ignored", v, sorted(allowed))
            return None
        return s

    return FactorSpec(
        hypothesis_id       = h.hypothesis_id,
        signal_kind         = signal_kind,
        universe            = universe,
        date_range          = date_range,
        signal_inputs       = tuple(str(x) for x in
                                     (payload.get("signal_inputs") or ())),
        rebal               = rebal,
        weighting           = weighting,
        expected_holding_period = str(payload.get("expected_holding_period")
                                         or "monthly"),
        min_obs_months      = int(payload.get("min_obs_months") or 120),
        pit_audits          = tuple(str(x) for x in
                                     (payload.get("pit_audits") or ())),
        cost_model          = str(payload.get("cost_model")
                                    or "engine.execution.cost_model.basic"),
        rationale           = str(payload.get("rationale") or ""),
        extracted_ts        = _dt.datetime.utcnow()
                                .strftime("%Y-%m-%dT%H:%M:%SZ"),
        model               = result.model,
        # Phase 6 B-class (Optional — null-preferring)
        universe_size       = _opt_int(payload.get("universe_size")),
        n_buckets           = _opt_int(payload.get("n_buckets")),
        signal_lookback_m   = _opt_int(payload.get("signal_lookback_m")),
        signal_skip_m       = _opt_int(payload.get("signal_skip_m")),
        vol_target_annual   = _opt_float(payload.get("vol_target_annual")),
        weighting_scheme_alt = _opt_str_enum(
            payload.get("weighting_scheme_alt"), {"ew", "vw", "rank"}),
        # Role-aware routing axes (Phase 1, 2026-06-09)
        investment_role     = _opt_str_enum(
            payload.get("investment_role"),
            {"alpha", "insurance", "diversifier", "hedge", "overlay"}),
        statistical_role    = _opt_str_enum(
            payload.get("statistical_role"),
            {"directional", "mean_reverting", "arbitrage",
             "market_making", "event_driven"}),
        asset_class         = _opt_str_enum(
            payload.get("asset_class"),
            {"equity", "fixed_income", "fx", "commodity",
             "multi_asset", "cross_asset"}),
        mechanism           = _opt_str_enum(
            payload.get("mechanism"),
            {"behavioral", "risk_premium", "arbitrage", "structural",
             "event_driven", "microstructure"}),
        horizon             = _opt_str_enum(
            payload.get("horizon"),
            {"high_frequency", "daily", "weekly", "monthly", "quarterly"}),
        capacity_tier       = _opt_str_enum(
            payload.get("capacity_tier"),
            {"under_100m", "100m_to_1b", "over_1b", "unknown"}),
        data_dependency_type = _opt_str_enum(
            payload.get("data_dependency_type"),
            {"fundamental", "price", "microstructure",
             "alternative", "macro"}),
        regime_sensitivity  = _opt_str_enum(
            payload.get("regime_sensitivity"),
            {"stationary", "known_regime_break", "unknown"}),
    )
