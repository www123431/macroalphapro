"""composer.components.signals — SIGNAL atomic components.

Each component produces a signal DataFrame [date × asset] where
positive values predict outperformance (long) and negative values
predict underperformance (short). Composer's WEIGHTING component
transforms this signal into position weights.

Sign convention (LdP §3): higher value = stronger long signal.

Coverage map (initial)
----------------------
  CARRY_FORWARD_DISCOUNT    forward-discount signed by currency
  CARRY_ROLL_YIELD          futures roll yield (commodity / rates)
  MOMENTUM_12_1             12-month return excluding most recent month
  MOMENTUM_TSMOM_12         TSMOM sign(12m cumulative return)
"""
from __future__ import annotations

import logging
import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, register_component,
)

logger = logging.getLogger(__name__)


# ── CARRY_FORWARD_DISCOUNT (FX) ────────────────────────


@register_component(ComponentRole.SIGNAL, "CARRY_FORWARD_DISCOUNT")
class SignalCarryForwardDiscount(Component):
    """Forward-discount signed signal for FX.

    For each currency, the signal value is the forward discount (in
    annualized %) — high for high-yielders (long), low/negative for
    low-yielders (short). This is the load-bearing carry signal
    documented in KMPV2017 and the canonical UIP-failure literature.

    Lookback: 1 month (standard). Quantile applied downstream by
    weighting/rebalance.
    """
    description = "FX forward-discount carry signal (KMPV2017 canonical form)"

    def build(self, spec, context: dict) -> ComponentResult:
        from engine.validation.crossasset_carry import build_fx_carry
        carry_wide, _rw, _ls = build_fx_carry()
        # carry_wide is already a date × currency panel of carry signals
        return ComponentResult(
            data=carry_wide,
            metadata={
                "signal_type":     "CARRY_FORWARD_DISCOUNT",
                "asset_class":     "FX",
                "n_assets":        int(carry_wide.shape[1]),
                "date_start":      str(carry_wide.index.min())[:10],
                "date_end":        str(carry_wide.index.max())[:10],
                "lookback_months": 1,
                "source":          "engine.validation.crossasset_carry.build_fx_carry (carry_wide)",
                "sign_convention": "higher = long; lower = short (high-yielders > low-yielders)",
            },
        )


# ── CARRY_ROLL_YIELD ────────────────────────────────────


@register_component(ComponentRole.SIGNAL, "CARRY_ROLL_YIELD")
class SignalCarryRollYield(Component):
    """Futures roll yield (cmdty / rates) — the futures-curve analogue
    of FX forward discount."""
    description = "Futures roll yield (commodity or rates curve)"

    def build(self, spec, context: dict) -> ComponentResult:
        # Decide which carry to use based on the universe asset_class
        ac = spec.universe.asset_class.value
        if ac == "COMMODITY":
            from engine.portfolio.carry_sleeve import build_carry_contract_panels
            cwide, _rwide = build_carry_contract_panels()
            source = "engine.portfolio.carry_sleeve.build_carry_contract_panels (cwide)"
        elif ac == "RATES":
            from engine.validation.crossasset_carry import build_rates_xc_carry
            cwide, _rw, _ls = build_rates_xc_carry()
            source = "engine.validation.crossasset_carry.build_rates_xc_carry (carry_wide)"
        else:
            raise ValueError(
                f"CARRY_ROLL_YIELD needs asset_class COMMODITY or RATES, "
                f"got {ac}")
        return ComponentResult(
            data=cwide,
            metadata={
                "signal_type":     "CARRY_ROLL_YIELD",
                "asset_class":     ac,
                "n_assets":        int(cwide.shape[1]),
                "date_start":      str(cwide.index.min())[:10],
                "date_end":        str(cwide.index.max())[:10],
                "source":          source,
                "sign_convention": "higher = long (steeper roll yield)",
            },
        )


# ── VALUE_BOOK_TO_MARKET (C1, 2026-06-05) ────────────────


@register_component(ComponentRole.SIGNAL, "VALUE_BOOK_TO_MARKET")
class SignalValueBookToMarket(Component):
    """Fama-French style book-to-market signal for equity.

    For each (date, permno):
      B/M = book_equity (Compustat ceq, lagged 180d for PIT) / mcap_at_date

    Sign convention: HIGHER B/M = MORE value = LONG. Negative book
    equity dropped per FF convention.

    Data join chain (all offline, no WRDS at run time):
      CRSP returns / mcap (data/cache/_crsp_msf_insider_*)
        ↔ Compustat funda (data/cache/_compustat_funda.parquet)
      via CCM link (data/cache/_crsp_ccm_link.parquet, F-substrate 2026-06-05).

    Universe coverage caveat: the cached Compustat panel covers ~32%
    of the CRSP universe per cross-section (large-cap biased). The
    signal panel will be NaN for permnos without a Compustat join —
    downstream weighting components correctly skip NaN per row.
    """
    description = "Book-to-market value signal (Fama-French canonical, PIT-disciplined)"

    def build(self, spec, context: dict) -> ComponentResult:
        ac = spec.universe.asset_class.value
        if ac != "EQUITY":
            raise ValueError(
                f"VALUE_BOOK_TO_MARKET expects EQUITY asset_class, got {ac}")
        from engine.composer.components.equity_data import book_to_market_wide
        bm = book_to_market_wide()
        n_resolved = bm.notna().sum().sum()
        return ComponentResult(
            data=bm,
            metadata={
                "signal_type":   "VALUE_BOOK_TO_MARKET",
                "asset_class":   "EQUITY",
                "n_assets":      int(bm.shape[1]),
                "n_resolved":    int(n_resolved),
                "date_start":    str(bm.index.min())[:10],
                "date_end":      str(bm.index.max())[:10],
                "compustat_lag_days": 180,
                "convention":    "FF: drop negative book equity",
                "source":        "engine.composer.components.equity_data.book_to_market_wide",
                "sign_convention": "higher B/M = long (value premium)",
                "caveat":        "join coverage ~32% per cross-section "
                                  "(cached Compustat subset is large-cap biased)",
            },
        )


# ── BAB (Betting-Against-Beta, 2026-06-05) ────────────────


@register_component(ComponentRole.SIGNAL, "BAB")
class SignalBAB(Component):
    """Betting-Against-Beta signal (Frazzini-Pedersen 2014).

    Signal = -beta(36-month rolling vs FF Mkt-RF). HIGHER signal
    (i.e. LOWER beta) = LONG. Captures the BAB anomaly: low-beta
    assets historically outperform their CAPM-predicted return,
    high-beta assets underperform.

    PIT discipline: beta computed using returns up to t-1, signal
    used for trading from t onward (shift handled in equity_data.beta_wide).

    Combined with INV_VOL weighting, this is the canonical FP2014
    construction. With EQUAL weighting it's the simplified anomaly form.
    """
    description = "Betting-Against-Beta = negated 36m beta vs market (Frazzini-Pedersen 2014)"

    def build(self, spec, context: dict) -> ComponentResult:
        ac = spec.universe.asset_class.value
        if ac != "EQUITY":
            raise ValueError(
                f"BAB expects EQUITY asset_class, got {ac}")
        from engine.composer.components.equity_data import beta_wide
        beta = beta_wide(window_months=36, min_periods=24)
        # Sign: BAB = -beta (low beta = long)
        signal = -beta
        n_resolved = int(signal.notna().sum().sum())
        return ComponentResult(
            data=signal,
            metadata={
                "signal_type":   "BAB",
                "asset_class":   "EQUITY",
                "n_assets":      int(signal.shape[1]),
                "n_resolved":    n_resolved,
                "date_start":    str(signal.index.min())[:10],
                "date_end":      str(signal.index.max())[:10],
                "beta_window":   36,
                "min_periods":   24,
                "market_source": "Fama-French MKT_RF weekly resampled to monthly",
                "sign_convention": "higher signal (lower beta) = long; BAB = bet against high beta",
                "reference":     "Frazzini-Pedersen 2014 JFE",
                "caveat":        "raw beta (no industry adjustment); FP2014 also "
                                  "centers beta cross-sectionally — that's done "
                                  "implicitly by downstream weighting's L/S construction",
            },
        )


# ── PROFITABILITY_GROSS (C3, 2026-06-05) ─────────────────


@register_component(ComponentRole.SIGNAL, "PROFITABILITY_GROSS")
class SignalProfitabilityGross(Component):
    """Novy-Marx 2013 JFE gross profitability: GP/A = (revt - cogs) / at,
    PIT-disciplined via 180-day Compustat lag.

    Sign: HIGHER = LONG (gross profitability premium). Novy-Marx shows
    GP/A has predictive power comparable to (or stronger than) book-
    to-market for cross-sectional equity returns. Often used as a
    "value-with-quality" filter that strengthens HML by avoiding
    value traps (cheap-but-unprofitable firms).

    Raw signal (not z-scored). Downstream weighting component does
    cross-sectional ranking.
    """
    description = "Gross profitability GP/A (Novy-Marx 2013)"

    def build(self, spec, context: dict) -> ComponentResult:
        ac = spec.universe.asset_class.value
        if ac != "EQUITY":
            raise ValueError(
                f"PROFITABILITY_GROSS expects EQUITY asset_class, got {ac}")
        from engine.composer.components.equity_data import gross_profitability_wide
        gp = gross_profitability_wide()
        n_resolved = int(gp.notna().sum().sum())
        return ComponentResult(
            data=gp,
            metadata={
                "signal_type":      "PROFITABILITY_GROSS",
                "asset_class":      "EQUITY",
                "n_assets":         int(gp.shape[1]),
                "n_resolved":       n_resolved,
                "date_start":       str(gp.index.min())[:10],
                "date_end":         str(gp.index.max())[:10],
                "compustat_lag_days": 180,
                "source":           "engine.composer.components.equity_data.gross_profitability_wide",
                "sign_convention":  "higher = long (more profitable)",
                "reference":        "Novy-Marx 2013 JFE",
            },
        )


# ── QUALITY_QMJ (C2, 2026-06-05) ─────────────────────────


@register_component(ComponentRole.SIGNAL, "QUALITY_QMJ")
class SignalQualityQMJ(Component):
    """Quality-Minus-Junk (Asness-Frazzini-Pedersen 2019) signal —
    simplified 2-dim composite (profitability + safety) per cached
    Compustat fields.

    SIMPLIFIED CAVEAT: full AFP 2019 has 4 dims (profitability / growth /
    safety / payout). We have cached compustat for profitability +
    safety; growth + payout require additional fields not in current
    cache. Equity_data.quality_qmj_wide docstring details the math.

    Sign: HIGHER quality = LONG (going long high-Q, short junk).
    """
    description = "Quality-Minus-Junk composite (AFP 2019, simplified 2-dim)"

    def build(self, spec, context: dict) -> ComponentResult:
        ac = spec.universe.asset_class.value
        if ac != "EQUITY":
            raise ValueError(
                f"QUALITY_QMJ expects EQUITY asset_class, got {ac}")
        from engine.composer.components.equity_data import quality_qmj_wide
        q = quality_qmj_wide()
        n_resolved = int(q.notna().sum().sum())
        return ComponentResult(
            data=q,
            metadata={
                "signal_type":      "QUALITY_QMJ",
                "asset_class":      "EQUITY",
                "n_assets":         int(q.shape[1]),
                "n_resolved":       n_resolved,
                "date_start":       str(q.index.min())[:10],
                "date_end":         str(q.index.max())[:10],
                "compustat_lag_days": 180,
                "components_used":  ["profitability (GP/A)", "safety (1 - debt/at)"],
                "afp_2019_full":    False,
                "missing_dimensions": ["growth", "payout"],
                "source":           "engine.composer.components.equity_data.quality_qmj_wide",
                "sign_convention":  "higher = long (high quality)",
                "caveat":           "simplified 2-dim composite, not full AFP 2019; "
                                     "cached compustat lacks growth/payout fields",
            },
        )


# ── MOMENTUM_12_1 ───────────────────────────────────────


@register_component(ComponentRole.SIGNAL, "MOMENTUM_12_1")
class SignalMomentum12_1(Component):
    """12-1 month cross-sectional momentum: t-12 to t-2 cumulative return,
    excluding the most recent month (avoids short-term reversal contamination).

    Requires the asset universe's returns_wide DataFrame (the universe
    component output via context['universe']).
    """
    description = "12-1 month XS momentum (Jegadeesh-Titman canonical lag)"

    def build(self, spec, context: dict) -> ComponentResult:
        # Look up universe to get returns wide
        # The composer passes universe result in context.
        u_result = context.get("universe")
        if u_result is None:
            raise ValueError("MOMENTUM_12_1 needs the universe component output")
        universe_data = u_result.data
        # Need returns wide; we approximate from the universe's returns
        # Most universe components produce membership DataFrames. For
        # momentum we need to reload the returns panel from the same source.
        ac = spec.universe.asset_class.value
        if ac == "FX":
            from engine.validation.crossasset_carry import build_fx_carry
            _cw, rw, _ls = build_fx_carry()
        elif ac == "RATES":
            from engine.validation.crossasset_carry import build_rates_xc_carry
            _cw, rw, _ls = build_rates_xc_carry()
        elif ac == "COMMODITY":
            from engine.portfolio.carry_sleeve import build_carry_contract_panels
            _cwide, rw = build_carry_contract_panels()
        else:
            raise ValueError(f"MOMENTUM_12_1 needs returns for asset_class={ac}")
        # 12-1 momentum: cumulative return from t-12 to t-2 (exclude t-1)
        cum = (1 + rw).rolling(12).apply(lambda x: x.prod(), raw=False) - 1
        # Shift by 1 to exclude the most recent month
        sig = cum.shift(1)
        return ComponentResult(
            data=sig,
            metadata={
                "signal_type":     "MOMENTUM_12_1",
                "asset_class":     ac,
                "lookback_months": 12,
                "exclude_recent":  1,
                "n_assets":        int(sig.shape[1]),
                "source":          f"computed from {ac} returns panel",
                "sign_convention": "higher = long (winners)",
            },
        )
