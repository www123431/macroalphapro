"""
ETF → CFTC futures contract mapping (P3a step 3, 2026-05-07).

Many of our 45 production ETFs track an underlying that has a corresponding
CFTC futures contract — speculator vs commercial positioning on the
futures is a forward-looking sentiment proxy for the spot ETF.

Mapping discipline
------------------
1. Each ETF maps to AT MOST one canonical contract. When CFTC publishes
   a "Consolidated" code that aggregates legacy + electronic versions
   (e.g. ``13874+`` for "S&P 500 Consolidated"), prefer it — it's the
   stable long-run series and aggregates trading from all venues.
2. Sector ETFs (XLF / XLE / XLV / etc.) map to the corresponding
   E-MINI S&P sector index futures.
3. ETFs with no clean futures analogue (factor ETFs like USMV, QUAL;
   thematic ETFs like ICLN, SMH; international ETFs like KWEB, INDA)
   are intentionally NOT mapped here — they're left to ``None`` so
   the consumer can render "no COT" rather than show a mismatched proxy.
4. Verified against actual ``cftc_cot_weekly`` content as of 2026-05-07
   covering 2020-01-07 → 2024-12-31.

When extending: probe DB for the contract first
    ``SELECT DISTINCT market_name, contract_market_code FROM cftc_cot_weekly
     WHERE market_name LIKE '%<your-pattern>%' AND report_type=...``
to confirm the exact code BEFORE adding to the mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CotMapping:
    """One ETF → CFTC contract pairing."""
    contract_market_code: str          # CFTC primary key
    report_type:          Literal["disagg_fut", "tff_fut"]
    canonical_market_name: str         # human-readable for citations
    rationale:            str          # why this proxy + caveats


# ── 45-ETF universe → CFTC mapping ──────────────────────────────────────────
# Ordered by asset class for readability.
ETF_TO_COT: dict[str, CotMapping] = {

    # ── Broad equity index (TFF) ─────────────────────────────────────────
    "SPY": CotMapping(
        contract_market_code = "13874+",
        report_type          = "tff_fut",
        canonical_market_name= "S&P 500 Consolidated",
        rationale            = "Consolidated S&P 500 futures (E-MINI + full + micro). 5y stable.",
    ),
    "QQQ": CotMapping(
        contract_market_code = "20974+",
        report_type          = "tff_fut",
        canonical_market_name= "NASDAQ-100 Consolidated",
        rationale            = "Consolidated NASDAQ-100 futures (E-MINI + micro). 5y stable.",
    ),
    "DIA": CotMapping(
        contract_market_code = "12460+",
        report_type          = "tff_fut",
        canonical_market_name= "DJIA Consolidated",
        rationale            = "Consolidated Dow Jones Industrial Avg futures. 5y stable.",
    ),

    # ── Sector ETFs (TFF) — E-MINI S&P Select Sector futures ─────────────
    "XLE": CotMapping(
        contract_market_code = "138749",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P ENERGY INDEX",
        rationale            = "Sector E-MINI matches XLE Select Sector composition.",
    ),
    "XLF": CotMapping(
        contract_market_code = "13874C",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P FINANCIAL INDEX",
        rationale            = "Sector E-MINI matches XLF Select Sector composition.",
    ),
    "XLV": CotMapping(
        contract_market_code = "13874E",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P HEALTH CARE INDEX",
        rationale            = "Sector E-MINI matches XLV Select Sector composition.",
    ),
    "XLI": CotMapping(
        contract_market_code = "13874F",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P INDUSTRIAL INDEX",
        rationale            = "Sector E-MINI matches XLI Select Sector composition.",
    ),
    "XLP": CotMapping(
        contract_market_code = "138748",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P CONSU STAPLES INDEX",
        rationale            = "Sector E-MINI matches XLP Select Sector composition.",
    ),
    "XLU": CotMapping(
        contract_market_code = "13874J",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P UTILITIES INDEX",
        rationale            = "Sector E-MINI matches XLU Select Sector composition.",
    ),
    "XLK": CotMapping(
        contract_market_code = "13874I",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P TECHNOLOGY INDEX",
        rationale            = "Sector E-MINI matches XLK Select Sector composition.",
    ),
    "XLB": CotMapping(
        contract_market_code = "13874H",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P MATERIALS INDEX",
        rationale            = "Sector E-MINI matches XLB Select Sector composition.",
    ),
    "XLRE": CotMapping(
        contract_market_code = "13874R",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P REAL ESTATE INDEX",
        rationale            = "Sector E-MINI matches XLRE Select Sector composition.",
    ),
    "XLY": CotMapping(
        # CFTC publishes XLY equivalent under code "13874D" (Cons Disc).
        # Verified non-empty in cftc_cot_weekly 2020-2024.
        contract_market_code = "13874D",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P CONSUMER DISCRETIONARY INDEX",
        rationale            = "Sector E-MINI matches XLY Select Sector composition.",
    ),
    "XLC": CotMapping(
        contract_market_code = "13874P",
        report_type          = "tff_fut",
        canonical_market_name= "E-MINI S&P COMMUNICATION INDEX",
        rationale            = "Sector E-MINI matches XLC Select Sector composition.",
    ),

    # ── Treasury / rates (TFF) ───────────────────────────────────────────
    "TLT": CotMapping(
        contract_market_code = "020601",
        report_type          = "tff_fut",
        canonical_market_name= "U.S. TREASURY BONDS",
        rationale            = "TLT tracks 20+ Year Treasuries; CBOT T-Bond futures cover 15-25y deliverable basket.",
    ),
    "IEF": CotMapping(
        contract_market_code = "043602",
        report_type          = "tff_fut",
        canonical_market_name= "10-YEAR U.S. TREASURY NOTES",
        rationale            = "IEF tracks 7-10y T-Notes; CBOT 10y futures direct match.",
    ),
    "SHY": CotMapping(
        contract_market_code = "042601",
        report_type          = "tff_fut",
        canonical_market_name= "2-YEAR U.S. TREASURY NOTES",
        rationale            = "SHY tracks 1-3y T-Notes; CBOT 2y futures closest analogue.",
    ),

    # ── Commodity (Disaggregated) ────────────────────────────────────────
    "GLD": CotMapping(
        contract_market_code = "088691",
        report_type          = "disagg_fut",
        canonical_market_name= "GOLD - COMMODITY EXCHANGE INC.",
        rationale            = "GLD direct gold-bullion proxy; CME Comex gold futures direct match.",
    ),
    "SLV": CotMapping(
        contract_market_code = "084691",
        report_type          = "disagg_fut",
        canonical_market_name= "SILVER - COMMODITY EXCHANGE INC.",
        rationale            = "SLV direct silver-bullion proxy; CME Comex silver futures direct match.",
    ),
    "USO": CotMapping(
        contract_market_code = "067651",
        report_type          = "disagg_fut",
        canonical_market_name= "CRUDE OIL, LIGHT SWEET",
        rationale            = "USO holds front-month WTI; CME WTI light-sweet direct match.",
    ),

    # ── Currency (TFF) ───────────────────────────────────────────────────
    "UUP": CotMapping(
        contract_market_code = "098662",
        report_type          = "tff_fut",
        canonical_market_name= "USD INDEX - ICE FUTURES U.S.",
        rationale            = "UUP tracks DXY; ICE USD INDEX futures direct match.",
    ),
    "FXE": CotMapping(
        contract_market_code = "099741",
        report_type          = "tff_fut",
        canonical_market_name= "EURO FX",
        rationale            = "FXE tracks EUR/USD; CME EURO FX futures direct match.",
    ),
    "FXY": CotMapping(
        contract_market_code = "097741",
        report_type          = "tff_fut",
        canonical_market_name= "JAPANESE YEN",
        rationale            = "FXY tracks JPY/USD inverted; CME JPY futures direct match.",
    ),
    "FXC": CotMapping(
        contract_market_code = "090741",
        report_type          = "tff_fut",
        canonical_market_name= "CANADIAN DOLLAR",
        rationale            = "FXC tracks CAD/USD; CME CAD futures direct match.",
    ),

    # ── Volatility (TFF) ─────────────────────────────────────────────────
    "VXX": CotMapping(
        contract_market_code = "1170E1",
        report_type          = "tff_fut",
        canonical_market_name= "VIX FUTURES",
        rationale            = "VXX tracks short-dated VIX futures roll; CFE VIX direct match.",
    ),
    # SVXY (inverse VIX) intentionally NOT mapped — same underlying as VXX
    # but inverse exposure; consumer should derive it from VXX positioning
    # rather than have two ETFs map to the same contract.
}


# ── Public API ───────────────────────────────────────────────────────────────

def get_mapping(ticker: str) -> CotMapping | None:
    """Return the CFTC contract mapping for ``ticker``, or None if unmapped."""
    return ETF_TO_COT.get(ticker.upper())


def mapped_tickers() -> list[str]:
    """Return all tickers with a CFTC mapping."""
    return list(ETF_TO_COT.keys())


def coverage_report(universe: list[str]) -> dict:
    """Diagnostic: which tickers in ``universe`` have / lack a mapping.

    Use to surface coverage in the supervisor UI: "X of N ETFs have COT
    proxy; Y are intentionally unmapped".
    """
    universe = [t.upper() for t in universe]
    mapped   = [t for t in universe if t in ETF_TO_COT]
    unmapped = [t for t in universe if t not in ETF_TO_COT]
    return {
        "n_universe":  len(universe),
        "n_mapped":    len(mapped),
        "n_unmapped":  len(unmapped),
        "mapped":      sorted(mapped),
        "unmapped":    sorted(unmapped),
        "coverage_pct": round(len(mapped) * 100 / max(1, len(universe)), 1),
    }
