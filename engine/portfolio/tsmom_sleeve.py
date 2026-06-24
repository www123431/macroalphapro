"""engine/portfolio/tsmom_sleeve.py — deployable cross-asset TSMOM sleeve return engine.

The deployable, vol-targeted MONTHLY return series for the 5-leg Futures TSMOM sleeve
(cmdty / fx / rates_us / rates_xc / eqidx), spec 77 §11+§12 amendment 2026-05-29.

The strict-gate validation in `engine/validation/crossasset_tsmom.py` proved:
  Sharpe-t 3.12 (≥HLZ 3.0)        ✓
  Deflated SR 0.910 (≥0.90)        ✓
  OOS Sharpe 0.351                 ✓
  Subperiods both positive (0.85 / 0.37) ✓
  Book correlation 0.37 (<0.5)    ✓
  FF5+UMD α-t 1.70 (<|2|)         ✓
  Per-instrument sign 100% / 89% / 100% / 100% / 100% ✓
  Net Sharpe 0.62                  ✓
GREEN — third mechanism family. See
docs/decisions/crossasset_tsmom_GREEN_2026-05-29.md.

Construction (Moskowitz-Ooi-Pedersen 2012, MOP):
  Per instrument: signal_t = sign(cumulative_return over [t-12, t-1])
  Position size: signal × min(40%/realized_vol, 2.0)
  Per leg:       equal-weight average across instruments
  5-leg combine: risk-parity (inverse-vol) — same `risk_parity_combine` as carry

This module is the sleeve's P&L ENGINE. It does NOT touch combined_book NAV
directly; combined_book.py is the integrator.
"""
from __future__ import annotations

import pandas as pd

# Mirror carry_sleeve conventions
DEFAULT_TARGET_ANNUAL_VOL: float = 0.10
MONTHS_PER_YEAR: int = 12


def build_tsmom_sleeve_returns(
    target_annual_vol: float = DEFAULT_TARGET_ANNUAL_VOL,
    include_cmdty: bool = True,
    include_fx: bool = True,
    include_rates_us: bool = True,
    include_rates_xc: bool = True,
    include_eqidx: bool = True,
) -> pd.Series:
    """Vol-targeted 5-leg TSMOM sleeve, monthly return series.

    Returns the GROSS return series; cost deduction (n_legs × RT_CY) is the
    caller's responsibility — combined_book.py applies it consistently with the
    carry sleeve convention.
    """
    from engine.portfolio.carry_sleeve import vol_target
    from engine.validation.crossasset_tsmom import build_tsmom_sleeve_returns as _vbuild

    raw = _vbuild(
        include_cmdty=include_cmdty,
        include_fx=include_fx,
        include_rates_us=include_rates_us,
        include_rates_xc=include_rates_xc,
        include_eqidx=include_eqidx,
    )
    return vol_target(raw, target_annual_vol, periods_per_year=MONTHS_PER_YEAR)
