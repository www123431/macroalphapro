"""engine/execution/futures_specs.py — contract specs for the carry/CTA futures universe.

For a FAITHFUL futures sim we must round to WHOLE contracts at real contract notionals (you cannot
hold 0.3 of a CL contract). This table maps each carry/B instrument (sym from commodity_carry.
COMMODITIES / crossasset_carry.FX / RATES) to: a yfinance continuous-futures ticker (free forward
prices), an APPROX USD contract notional (recent-price ballpark, for lumpiness rounding), and
whether a MICRO contract exists (1/10 notional → finer granularity for a small book).

NOTE: notionals are APPROXIMATE (recent-price round numbers) — fine for quantifying granularity /
lumpiness, but EXACT exchange multipliers must be used before any real-capital deployment.
"""
from __future__ import annotations

# sym -> (yfinance ticker, approx USD contract notional, has_micro)
SPECS: dict[str, tuple[str, float, bool]] = {
    # ── energy ──
    "CL_WTI":      ("CL=F", 75_000, True),    # MCL micro = 100 bbl
    "BRN_Brent":   ("BZ=F", 75_000, False),
    "HO_HeatOil":  ("HO=F", 110_000, False),  # 42k gal × ~2.6
    "RB_Gasoline": ("RB=F", 90_000, False),
    "NG_NatGas":   ("NG=F", 30_000, True),    # MNG micro
    "G_GasOil":    ("",      75_000, False),  # ICE gasoil — no clean yf ticker
    # ── metals ──
    "GC_Gold":     ("GC=F", 220_000, True),   # MGC micro = 10 oz
    "SI_Silver":   ("SI=F", 150_000, True),   # SIL micro = 1000 oz
    "HG_Copper":   ("HG=F", 110_000, True),   # MHG micro
    "PL_Platinum": ("PL=F", 50_000, False),
    "PA_Palladium":("PA=F", 100_000, False),
    # ── grains / softs / meats (CENTS-quoted on yf; notional ≈) ──
    "ZC_Corn":     ("ZC=F", 25_000, False),
    "ZS_Soybean":  ("ZS=F", 65_000, False),
    "ZM_SoyMeal":  ("ZM=F", 40_000, False),
    "ZL_SoyOil":   ("ZL=F", 30_000, False),
    "CC_Cocoa":    ("CC=F", 80_000, False),
    "OJ_OrangeJuice":("OJ=F", 35_000, False),
    "LE_LiveCattle":("LE=F", 80_000, False),
    "GF_FeederCattle":("GF=F", 130_000, False),
    "HE_LeanHogs": ("HE=F", 35_000, False),
    # ── FX (CME currency futures) ──
    "EUR": ("6E=F", 135_000, True),   # M6E micro = 12,500 EUR
    "JPY": ("6J=F", 130_000, True),   # MJY micro
    "GBP": ("6B=F", 80_000, True),    # M6B micro
    "AUD": ("6A=F", 65_000, True),    # M6A micro
    "CAD": ("6C=F", 73_000, False),
    "CHF": ("6S=F", 140_000, False),
    "NZD": ("6N=F", 60_000, False),
    "MXN": ("6M=F", 28_000, False),
    "BRL": ("6L=F", 20_000, False),
    # ── rates (CBOT Treasury futures) ──
    "UST30": ("ZB=F", 120_000, False),
    "UST10": ("ZN=F", 115_000, True),  # MTN? (10Y micro exists)
    "UST5":  ("ZF=F", 110_000, False),
    "UST2":  ("ZT=F", 215_000, False),
}

MICRO_DIVISOR = 10.0   # micro contracts are ~1/10 the standard notional


def contract_notional(sym: str, use_micro: bool = False) -> float | None:
    spec = SPECS.get(sym)
    if not spec:
        return None
    _, notional, has_micro = spec
    if use_micro and has_micro:
        return notional / MICRO_DIVISOR
    return notional


def yf_ticker(sym: str) -> str | None:
    spec = SPECS.get(sym)
    return spec[0] if spec and spec[0] else None
