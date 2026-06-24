"""composer.components.universes — UNIVERSE atomic components.

Each component returns a boolean DataFrame [date × asset] indicating
which assets are in the universe at each date. Composer uses this to
mask out signals / positions for out-of-universe assets.

Coverage map
------------
  FX__G10            G10 currency pairs (existing carry sleeve panel)
  FX__G3             top-3 (USD vs EUR/JPY/GBP) — subset of G10
  FX__EM             EM currency pairs (MXN/BRL only currently; minimal)
  COMMODITY__LIQUID  commodity carry panel (existing)
  RATES__G10         G10 government bond futures
  EQUITY__US_SP500   S&P 500 large-cap (placeholder; loads when needed)
"""
from __future__ import annotations

import logging
import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, register_component,
)

logger = logging.getLogger(__name__)


# ── FX G10 ────────────────────────────────────────────────


@register_component(ComponentRole.UNIVERSE, "FX__G10")
class UniverseFXG10(Component):
    """G10 FX universe: JPY, CHF, EUR, GBP, AUD, NZD, CAD + reference USD.

    Data backed by engine.validation.crossasset_carry FX panel (CME
    currency futures). Returns a DataFrame indexed by month-end dates,
    columns = currency code, values = boolean (always True for in-panel
    assets after first quote date)."""
    description = "G10 currency pairs from CME FX futures panel"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.validation.crossasset_carry import build_fx_carry
        # build_fx_carry returns (carry_wide, returns_wide, ls)
        # We grab the returns_wide (rw) and use its columns as the universe
        cw, rw, _ls = build_fx_carry()
        # Restrict to the 9-currency panel; convert non-null returns to membership
        membership = rw.notna()
        return ComponentResult(
            data=membership,
            metadata={
                "asset_class":  "FX",
                "subset":       "G10",
                "n_assets":     int(membership.shape[1]),
                "date_start":   str(membership.index.min())[:10],
                "date_end":     str(membership.index.max())[:10],
                "source":       "engine.validation.crossasset_carry.build_fx_carry",
                "assets":       list(membership.columns),
            },
        )


# ── Commodity liquid ─────────────────────────────────────


@register_component(ComponentRole.UNIVERSE, "COMMODITY__LIQUID")
@register_component(ComponentRole.UNIVERSE, "COMMODITY__ALL")
class UniverseCommodityLiquid(Component):
    """Liquid commodity futures (the carry sleeve panel)."""
    description = "Liquid commodity futures from carry sleeve panel"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.validation.crossasset_carry import build_commodity_carry_ls
        # build_commodity_carry_ls returns the LS Series; we need the
        # underlying wide panel — fall back to the carry-sleeve helper
        # that exposes the per-contract returns.
        try:
            from engine.portfolio.carry_sleeve import build_carry_contract_panels
            cwide, rwide = build_carry_contract_panels()
        except Exception as exc:
            # Minimum-fallback: derive from the LS series so the
            # component never silently fails — it still has the
            # date index, just no per-asset breakdown.
            ls = build_commodity_carry_ls()
            df = pd.DataFrame({"_ls_aggregate": ls.notna()})
            return ComponentResult(
                data=df,
                metadata={
                    "asset_class":  "COMMODITY",
                    "subset":       "LIQUID",
                    "n_assets":     1,
                    "warning":      f"per-asset panel unavailable: {exc}",
                    "source":       "ls aggregate fallback",
                },
            )
        membership = rwide.notna()
        return ComponentResult(
            data=membership,
            metadata={
                "asset_class":  "COMMODITY",
                "subset":       "LIQUID",
                "n_assets":     int(membership.shape[1]),
                "date_start":   str(membership.index.min())[:10],
                "date_end":     str(membership.index.max())[:10],
                "source":       "engine.portfolio.carry_sleeve.build_carry_contract_panels",
            },
        )


# ── EQUITY (C1 substrate, 2026-06-05) ────────────────────


@register_component(ComponentRole.UNIVERSE, "EQUITY__ALL")
@register_component(ComponentRole.UNIVERSE, "EQUITY__US_RUSSELL_3000")
class UniverseEquityAll(Component):
    """All available equity assets in the cached CRSP universe.

    Backing data: data/cache/_crsp_msf_insider_universe.parquet (~7000
    permnos, 2013-10 onwards). This is NOT full CRSP — it's the panel
    cached for insider/value/quality work. Specs noting "full universe"
    or "Russell 3000" use this as the achievable proxy.
    """
    description = "Full cached CRSP equity universe (~7000 permnos)"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.composer.components.equity_data import crsp_returns_wide
        rw = crsp_returns_wide()
        membership = rw.notna()
        return ComponentResult(
            data=membership,
            metadata={
                "asset_class":  "EQUITY",
                "subset":       spec.universe.subset.value,
                "n_assets":     int(membership.shape[1]),
                "date_start":   str(membership.index.min())[:10],
                "date_end":     str(membership.index.max())[:10],
                "source":       "data/cache/_crsp_msf_insider_universe.parquet",
                "caveat":       "cached insider-study panel, ~7000 permnos; "
                                  "NOT full CRSP for periods outside 2013-2024",
            },
        )


@register_component(ComponentRole.UNIVERSE, "EQUITY__US_LARGE")
@register_component(ComponentRole.UNIVERSE, "EQUITY__US_RUSSELL_1000")
@register_component(ComponentRole.UNIVERSE, "EQUITY__US_SP500")
class UniverseEquityLargeCap(Component):
    """Top-1000 by month-end market cap from the cached CRSP universe.

    Used for US_LARGE / RUSSELL_1000 / SP500 subset specs. The size
    cutoff is approximate (S&P 500 is more like top 500; Russell 1000
    is top 1000) but the same top-N-by-mcap mechanic produces a
    monotonically-large-cap-biased subset that satisfies the "large-
    cap equity" intent of the spec.
    """
    description = "Top-1000 by mcap proxy for SP500/RUSSELL_1000/US_LARGE"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.composer.components.equity_data import (
            universe_top_by_mcap, crsp_returns_wide,
        )
        rw = crsp_returns_wide()
        # Top-N depends on subset semantics
        sub = spec.universe.subset.value
        if sub == "US_SP500":
            top_n = 500
        elif sub == "US_RUSSELL_1000":
            top_n = 1000
        else:
            top_n = 1000  # US_LARGE default
        membership = universe_top_by_mcap(top_n)
        # Restrict to the CRSP date index (in case mcap has more dates)
        membership = membership.reindex(rw.index).fillna(False)
        return ComponentResult(
            data=membership,
            metadata={
                "asset_class":  "EQUITY",
                "subset":       sub,
                "n_assets":     int(membership.any().sum()),
                "top_n":        top_n,
                "date_start":   str(membership.index.min())[:10],
                "date_end":     str(membership.index.max())[:10],
                "source":       "top-N-by-mcap from _crsp_msf_insider_mcap.parquet",
            },
        )


# ── Rates G10 ────────────────────────────────────────────


@register_component(ComponentRole.UNIVERSE, "RATES__G10")
class UniverseRatesG10(Component):
    """G10 government bond futures (10Y) panel."""
    description = "G10 10Y government bond futures"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.validation.crossasset_carry import build_rates_xc_carry
        cw, rw, _ls = build_rates_xc_carry()
        membership = rw.notna()
        return ComponentResult(
            data=membership,
            metadata={
                "asset_class":  "RATES",
                "subset":       "G10",
                "n_assets":     int(membership.shape[1]),
                "date_start":   str(membership.index.min())[:10],
                "date_end":     str(membership.index.max())[:10],
                "source":       "engine.validation.crossasset_carry.build_rates_xc_carry",
            },
        )
