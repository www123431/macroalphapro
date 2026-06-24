"""engine.research_store.red_lessons.mechanism_families — mechanism family taxonomy.

Mechanism family is the second classification axis (orthogonal to failure mode).
Two RED Lessons in the same family share the underlying ECONOMIC MECHANISM
even if their signals, data sources, and verdicts differ.

Why a taxonomy and not free-form labels:
  - 23 + 44 existing RED records would generate ~30 ad-hoc family strings
    if free-form, which destroys query-ability ("attention" / "Attention" /
    "media_attention" / "investor_attention" all mean the same family)
  - controlled families let the Layer 1-3 retrieval (same-family ∩ same-mode
    etc.) actually JOIN
  - new families should be added DELIBERATELY — if you can't find a fit in
    the existing taxonomy, that itself is a research insight ("we discovered
    a genuinely new mechanism family"); rare event

Each family has:
  - canonical label
  - definition (one-line economic interpretation)
  - representative anchor paper (the classic citation)
  - what differentiates it from adjacent families

Families are NOT mutually exclusive at the abstract level (e.g. PEAD is
both event-driven AND momentum-class), but each LESSON picks ONE primary
family. Use `mechanism_subtype` (free-form string) for finer detail.
"""
from __future__ import annotations

from enum import Enum


class MechanismFamily(str, Enum):
    """Controlled mechanism-family taxonomy. ~16 families covers the post-1990
    academic factor literature. Don't widen without deliberate review."""

    # Cross-sectional equity classics
    VALUE                       = "VALUE"
    MOMENTUM                    = "MOMENTUM"
    SIZE                        = "SIZE"
    PROFITABILITY               = "PROFITABILITY"
    INVESTMENT                  = "INVESTMENT"
    LOW_VOL                     = "LOW_VOL"

    # Event-driven (firm-level)
    EARNINGS_DRIFT              = "EARNINGS_DRIFT"           # PEAD class
    ANALYST_REVISION            = "ANALYST_REVISION"
    GUIDANCE                    = "GUIDANCE"
    INSIDER_TRADING             = "INSIDER_TRADING"
    SHORT_INTEREST              = "SHORT_INTEREST"

    # Text / attention / behavioral
    ATTENTION                   = "ATTENTION"                # media / Google trends / Wikipedia
    SENTIMENT                   = "SENTIMENT"                # text-derived tone
    NEWS_SHOCK                  = "NEWS_SHOCK"

    # Cross-asset / macro
    CARRY                       = "CARRY"                    # FX / rates / commodity / equity
    CROSS_ASSET_MOMENTUM        = "CROSS_ASSET_MOMENTUM"     # TSMOM / XSMOM in futures
    TERM_STRUCTURE              = "TERM_STRUCTURE"           # yield curve / commodity term
    MACRO_SURPRISE              = "MACRO_SURPRISE"           # Fed / NFP / CPI

    # Options / volatility
    VOL_RISK_PREMIUM            = "VOL_RISK_PREMIUM"
    OPTIONS_IMPLIED             = "OPTIONS_IMPLIED"          # IV skew / put-call / IV-RV gap

    # Reversal (short horizon)
    REVERSAL                    = "REVERSAL"

    # Other
    HOLDINGS_BASED              = "HOLDINGS_BASED"           # 13F / fund flows
    SUPPLY_CHAIN                = "SUPPLY_CHAIN"
    OTHER                       = "OTHER"                    # escape hatch — must justify


MECHANISM_FAMILY_DOCS: dict[MechanismFamily, dict[str, str]] = {
    MechanismFamily.VALUE: {
        "definition": "Cross-sectional return predictability from price-to-fundamentals ratios "
                      "(book/price, earnings/price, sales/price, cash flow/price).",
        "anchor_paper": "Fama & French 1992, 'Cross-Section of Expected Returns', JF",
    },
    MechanismFamily.MOMENTUM: {
        "definition": "Continuation of past relative returns (3-12 month formation, "
                      "1-month skip, 1-12 month holding).",
        "anchor_paper": "Jegadeesh & Titman 1993, 'Returns to Buying Winners and Selling Losers', JF",
    },
    MechanismFamily.SIZE: {
        "definition": "Small-cap excess return over large-cap (now widely considered "
                      "subsumed and/or fragile).",
        "anchor_paper": "Banz 1981, JFE",
    },
    MechanismFamily.PROFITABILITY: {
        "definition": "High gross-profit / high operating-profit firms outperform.",
        "anchor_paper": "Novy-Marx 2013, 'The Other Side of Value', JFE",
    },
    MechanismFamily.INVESTMENT: {
        "definition": "Low-investment firms outperform high-investment firms "
                      "(CAPEX, asset growth, share issuance).",
        "anchor_paper": "Cooper-Gulen-Schill 2008, 'Asset Growth and Returns', JF",
    },
    MechanismFamily.LOW_VOL: {
        "definition": "Low-beta / low-idiosyncratic-vol stocks earn higher risk-adjusted "
                      "returns (BAB, MIN-VOL).",
        "anchor_paper": "Frazzini & Pedersen 2014, 'Betting Against Beta', JFE",
    },
    MechanismFamily.EARNINGS_DRIFT: {
        "definition": "Post-earnings-announcement drift: prices under-react to earnings "
                      "surprises and drift 60-90 days in the surprise direction.",
        "anchor_paper": "Bernard & Thomas 1989, 'Post-Earnings-Announcement Drift', JAR",
    },
    MechanismFamily.ANALYST_REVISION: {
        "definition": "Sell-side EPS / target-price revisions predict near-term returns; "
                      "drift mechanism similar to PEAD.",
        "anchor_paper": "Stickel 1991, 'Common Stock Returns Surrounding Earnings Forecast Revisions', JF",
    },
    MechanismFamily.GUIDANCE: {
        "definition": "Management-issued forward guidance (issue, withdrawal, beat/miss vs "
                      "prior guide) generates drift effects.",
        "anchor_paper": "Anilowski-Feng-Skinner 2007, JAE",
    },
    MechanismFamily.INSIDER_TRADING: {
        "definition": "Form 4 insider buys / sells predict future returns; cluster size "
                      "and insider role matter.",
        "anchor_paper": "Jeng-Metrick-Zeckhauser 2003, RFS",
    },
    MechanismFamily.SHORT_INTEREST: {
        "definition": "High short interest predicts negative future returns; "
                      "Days-to-cover and short fee variants.",
        "anchor_paper": "Asquith-Pathak-Ritter 2005, JFE",
    },
    MechanismFamily.ATTENTION: {
        "definition": "Investor-attention proxies (media coverage, Google/Wikipedia "
                      "search, news count) predict short-term returns + reversals.",
        "anchor_paper": "Da-Engelberg-Gao 2011, 'In Search of Attention', JF",
    },
    MechanismFamily.SENTIMENT: {
        "definition": "Text-derived sentiment / tone (LM dictionary, FinBERT, custom) "
                      "predicts returns; PEAD-adjacent in event windows.",
        "anchor_paper": "Tetlock 2007, 'Giving Content to Investor Sentiment', JF",
    },
    MechanismFamily.NEWS_SHOCK: {
        "definition": "Specific news events (M&A rumors, lawsuits, regulatory) generate "
                      "discrete return jumps + drift.",
        "anchor_paper": "Chan 2003, 'Stock Price Reaction to News and No-News', JFE",
    },
    MechanismFamily.CARRY: {
        "definition": "Forward / spot price differential earns positive return on average "
                      "across FX, rates, commodity, equity-index futures.",
        "anchor_paper": "Koijen-Moskowitz-Pedersen-Vrugt 2018, 'Carry', JFE",
    },
    MechanismFamily.CROSS_ASSET_MOMENTUM: {
        "definition": "Time-series and cross-sectional momentum applied to liquid futures "
                      "across asset classes (CTAs / managed futures).",
        "anchor_paper": "Moskowitz-Ooi-Pedersen 2012, 'Time Series Momentum', JFE",
    },
    MechanismFamily.TERM_STRUCTURE: {
        "definition": "Yield-curve slope / commodity term structure as predictor of "
                      "future returns (CMS / convenience yield trades).",
        "anchor_paper": "Cochrane-Piazzesi 2005, 'Bond Risk Premia', AER",
    },
    MechanismFamily.MACRO_SURPRISE: {
        "definition": "Pre-scheduled macro releases (Fed, NFP, CPI) generate immediate "
                      "+ drift return responses keyed to surprise direction.",
        "anchor_paper": "Lucca-Moench 2015, 'The Pre-FOMC Announcement Drift', JF",
    },
    MechanismFamily.VOL_RISK_PREMIUM: {
        "definition": "Realized < implied vol on average; selling vol earns a premium "
                      "(VRP, variance swaps, short-vol ETPs).",
        "anchor_paper": "Carr-Wu 2009, 'Variance Risk Premiums', RFS",
    },
    MechanismFamily.OPTIONS_IMPLIED: {
        "definition": "Options-implied quantities (skew, IV rank, IV-RV spread, put-call "
                      "ratio) predict returns / vol.",
        "anchor_paper": "Bali-Hovakimian 2009, 'Volatility Spreads and Expected Stock Returns', MS",
    },
    MechanismFamily.REVERSAL: {
        "definition": "Short-horizon (1-week, 1-month) reversal of past returns; opposite "
                      "of momentum at shorter horizons.",
        "anchor_paper": "Jegadeesh 1990, 'Evidence of Predictable Behavior of Security Returns', JF",
    },
    MechanismFamily.HOLDINGS_BASED: {
        "definition": "Quarterly 13F holdings changes, fund flows, mutual fund "
                      "ownership concentration as predictors.",
        "anchor_paper": "Chen-Jegadeesh-Wermers 2000, 'The Value of Active Mutual Fund Management', JFQA",
    },
    MechanismFamily.SUPPLY_CHAIN: {
        "definition": "Customer-supplier link return predictability; gradual information "
                      "diffusion along the supply chain.",
        "anchor_paper": "Cohen-Frazzini 2008, 'Economic Links and Predictable Returns', JF",
    },
    MechanismFamily.OTHER: {
        "definition": "Escape hatch. Use only when none of the above fits, AND document "
                      "WHY in the lesson's failure_evidence field.",
        "anchor_paper": "(none — must be filled in lesson-specific context)",
    },
}


# Sanity check at import time.
assert set(MECHANISM_FAMILY_DOCS.keys()) == set(MechanismFamily), (
    f"MECHANISM_FAMILY_DOCS missing entries for: "
    f"{set(MechanismFamily) - set(MECHANISM_FAMILY_DOCS.keys())}"
)
