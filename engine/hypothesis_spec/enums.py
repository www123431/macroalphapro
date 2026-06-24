"""hypothesis_spec.enums — controlled vocabularies.

Every enum here is part of the spec contract. Adding a value = a
deliberate vocabulary extension (typically requires extending one
or more Composer components to handle it).

Pattern: every enum is a str-Enum so it serializes cleanly to JSON
and round-trips through the LLM extractor.
"""
from __future__ import annotations
import enum


# ── Claim type (level 0 — what KIND of claim) ─────────────────

class ClaimType(str, enum.Enum):
    """B.2-A1 (2026-06-05): the FIRST classification a claim gets.
    Distinguishes tradable factor hypotheses from meta-claims about
    methodology, microstructure, capacity, decay, or domain facts.

    Only FACTOR_HYPOTHESIS claims flow into the Composer / direction_proposer /
    strategy lifecycle. The other types are still stored as research evidence
    (consumed by audit_verifier, chat RAG, etc.) but don't pollute the
    strategy pipeline.

    Rationale: pre-B.2, the 206-spec backfill put 49 claims (24%) into
    family=OTHER because the FamilyV2 vocabulary only had factor families.
    Real-world papers contain methodology critiques (HLZ multiple-testing,
    BHY FDR thresholds), microstructure measurements (market impact bps),
    capacity studies (break-even fund size), decay studies (post-publication
    Sharpe decline), and domain facts (customer-supplier income correlation)
    that aren't tradable but ARE research-valuable. They need a home.
    """
    FACTOR_HYPOTHESIS = "FACTOR_HYPOTHESIS"   # tradable strategy claim
    METHODOLOGY       = "METHODOLOGY"         # research-method claim (HLZ, BHY, sample-size theory)
    MICROSTRUCTURE    = "MICROSTRUCTURE"      # market impact, bid-ask, implementation cost
    CAPACITY          = "CAPACITY"            # AUM ceiling, break-even fund size, scalability
    DECAY_STUDY       = "DECAY_STUDY"         # post-publication Sharpe decay, robustness over time
    FACTOR_STRUCTURE  = "FACTOR_STRUCTURE"    # factor correlations, model explanatory power
    DOMAIN_FACT       = "DOMAIN_FACT"         # real-economy observation (not directly tradable)
    OTHER             = "OTHER"
    UNKNOWN           = "UNKNOWN"


# ── Family (level 1 of the hierarchy) ─────────────────────────

class FamilyV2(str, enum.Enum):
    """Top-level mechanism family. Maps roughly 1:1 to the existing
    mechanism_family enum in the older hypothesis store, but is a
    fresh enum so the spec layer can evolve independently."""
    CARRY              = "CARRY"
    MOMENTUM           = "MOMENTUM"
    REVERSAL           = "REVERSAL"
    VALUE              = "VALUE"
    QUALITY            = "QUALITY"
    LOW_VOL            = "LOW_VOL"
    SIZE               = "SIZE"
    PROFITABILITY      = "PROFITABILITY"
    INVESTMENT         = "INVESTMENT"
    VOL_RISK_PREMIUM   = "VOL_RISK_PREMIUM"
    TERM_STRUCTURE     = "TERM_STRUCTURE"
    SHORT_INTEREST     = "SHORT_INTEREST"
    ATTENTION          = "ATTENTION"
    EARNINGS_DRIFT     = "EARNINGS_DRIFT"
    SENTIMENT          = "SENTIMENT"
    SUPPLY_CHAIN       = "SUPPLY_CHAIN"
    OPTIONS_IMPLIED    = "OPTIONS_IMPLIED"
    HOLDINGS_BASED     = "HOLDINGS_BASED"
    CROSS_ASSET_MOMENTUM = "CROSS_ASSET_MOMENTUM"
    OTHER              = "OTHER"


# ── Asset class (level 2) ─────────────────────────────────────

class AssetClass(str, enum.Enum):
    """The asset class the hypothesis operates on. Multi-asset hypotheses
    use COMBINED + populate Universe.sub_classes."""
    EQUITY        = "EQUITY"
    FX            = "FX"
    RATES         = "RATES"
    COMMODITY     = "COMMODITY"
    CREDIT        = "CREDIT"
    OPTIONS       = "OPTIONS"
    DIGITAL       = "DIGITAL"
    COMBINED      = "COMBINED"
    UNKNOWN       = "UNKNOWN"


# ── Signal type (level 3) ─────────────────────────────────────

class SignalType(str, enum.Enum):
    """How the cross-sectional / time-series ranking is computed.

    Naming convention: <method>_<base>_<modifier?> where modifier is
    typically a horizon (e.g. mom_12_1 = 12-1 month momentum)."""
    # Carry family
    CARRY_FORWARD_DISCOUNT       = "CARRY_FORWARD_DISCOUNT"
    CARRY_ROLL_YIELD             = "CARRY_ROLL_YIELD"
    CARRY_DIV_YIELD              = "CARRY_DIV_YIELD"
    CARRY_TERM_PREMIUM           = "CARRY_TERM_PREMIUM"

    # Momentum family
    MOMENTUM_12_1                = "MOMENTUM_12_1"
    MOMENTUM_6_1                 = "MOMENTUM_6_1"
    MOMENTUM_TSMOM_12            = "MOMENTUM_TSMOM_12"
    MOMENTUM_RESIDUAL            = "MOMENTUM_RESIDUAL"

    # Reversal
    REVERSAL_SHORT_TERM_1M       = "REVERSAL_SHORT_TERM_1M"
    REVERSAL_LONG_TERM_60_13     = "REVERSAL_LONG_TERM_60_13"

    # Value
    VALUE_BOOK_TO_MARKET         = "VALUE_BOOK_TO_MARKET"
    VALUE_EARNINGS_YIELD         = "VALUE_EARNINGS_YIELD"
    VALUE_PURCHASING_POWER_PARITY = "VALUE_PURCHASING_POWER_PARITY"

    # Low-vol / BAB
    BAB                          = "BAB"
    LOW_VOL_RESIDUAL             = "LOW_VOL_RESIDUAL"

    # Quality / profitability
    QUALITY_QMJ                  = "QUALITY_QMJ"
    PROFITABILITY_GROSS          = "PROFITABILITY_GROSS"
    PROFITABILITY_NET            = "PROFITABILITY_NET"

    # Vol risk premium / options
    VRP_VARIANCE_SWAP            = "VRP_VARIANCE_SWAP"
    VRP_DELTA_HEDGED_STRADDLE    = "VRP_DELTA_HEDGED_STRADDLE"
    OPTIONS_SKEW                 = "OPTIONS_SKEW"
    OPTIONS_TERM_STRUCTURE       = "OPTIONS_TERM_STRUCTURE"

    # Event-driven
    PEAD_SUE                     = "PEAD_SUE"
    EARNINGS_REVISION            = "EARNINGS_REVISION"
    SHORT_INTEREST_DAYS_TO_COVER = "SHORT_INTEREST_DAYS_TO_COVER"

    # Other
    HOLDINGS_13F_BREADTH         = "HOLDINGS_13F_BREADTH"
    SENTIMENT_NEWS               = "SENTIMENT_NEWS"
    UNKNOWN                      = "UNKNOWN"


# ── Sign (direction) ──────────────────────────────────────────

class Sign(str, enum.Enum):
    """How the signal is taken. LONG_SHORT = standard XS; LONG_ONLY
    keeps just the top tercile; SHORT_ONLY keeps just the bottom."""
    LONG_SHORT  = "LONG_SHORT"
    LONG_ONLY   = "LONG_ONLY"
    SHORT_ONLY  = "SHORT_ONLY"
    TIMESERIES  = "TIMESERIES"    # TSMOM-style; signal IS the position
    UNKNOWN     = "UNKNOWN"


# ── Universe subset ───────────────────────────────────────────

class UniverseSubset(str, enum.Enum):
    """Which slice of the asset_class universe."""
    ALL                = "ALL"
    G10                = "G10"
    G3                 = "G3"
    EM                 = "EM"
    DM                 = "DM"
    US_LARGE           = "US_LARGE"
    US_RUSSELL_1000    = "US_RUSSELL_1000"
    US_RUSSELL_3000    = "US_RUSSELL_3000"
    US_SP500           = "US_SP500"
    INTERNATIONAL      = "INTERNATIONAL"
    COMMODITY_LIQUID   = "COMMODITY_LIQUID"
    CUSTOM             = "CUSTOM"
    UNKNOWN            = "UNKNOWN"


# ── Weighting ─────────────────────────────────────────────────

class Weighting(str, enum.Enum):
    """How positions within a leg are sized."""
    EQUAL          = "EQUAL"
    VALUE          = "VALUE"           # market-cap weighted
    RISK_PARITY    = "RISK_PARITY"
    INV_VOL        = "INV_VOL"
    MEAN_VARIANCE  = "MEAN_VARIANCE"
    SIGNAL_RANK    = "SIGNAL_RANK"     # weight ∝ rank score
    UNKNOWN        = "UNKNOWN"


# ── Rebalance ─────────────────────────────────────────────────

class Rebalance(str, enum.Enum):
    """Position rebalancing cadence."""
    DAILY      = "DAILY"
    WEEKLY     = "WEEKLY"
    MONTHLY    = "MONTHLY"
    QUARTERLY  = "QUARTERLY"
    THRESHOLD  = "THRESHOLD"           # drift-triggered
    UNKNOWN    = "UNKNOWN"


# ── Direction (predicted outcome) ─────────────────────────────

class PredictedDirection(str, enum.Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    ZERO     = "ZERO"
    UNKNOWN  = "UNKNOWN"
