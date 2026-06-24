"""composer.components.weightings — WEIGHTING atomic components.

Each component reads the signal panel + universe membership + spec
quantile and produces a weights DataFrame [date × asset] where each
row sums to 0 (long-short market neutral) or 1 (long-only).

Sign convention from signal: higher = long, lower = short.
"""
from __future__ import annotations

import logging
import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, register_component,
)
from engine.hypothesis_spec.enums import Sign

logger = logging.getLogger(__name__)


def _xs_long_short_top_bottom(signal: pd.DataFrame, q: float) -> pd.DataFrame:
    """Standard cross-sectional long-short: rank per date, weight = +1/n
    in top quantile, -1/n in bottom quantile, 0 elsewhere. Then divide
    by per-side count so each side weight sums to ±1 and net = 0."""
    out = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < 4:
            continue
        n_top = max(1, int(round(len(valid) * q)))
        n_bot = max(1, int(round(len(valid) * q)))
        ranked = valid.sort_values(ascending=False)
        top    = ranked.index[:n_top]
        bot    = ranked.index[-n_bot:]
        out.loc[dt, top] = 1.0 / n_top
        out.loc[dt, bot] = -1.0 / n_bot
    return out


# ── EQUAL ─────────────────────────────────────────────


@register_component(ComponentRole.WEIGHTING, "EQUAL")
class WeightingEqual(Component):
    """Equal-weight on top quantile minus equal-weight on bottom quantile.
    The simplest XS construction; baseline against which other weightings
    are compared."""
    description = "Equal-weight top quantile minus equal-weight bottom quantile"

    def build(self, spec, context: dict) -> ComponentResult:
        signals = context.get("signals")
        if not signals:
            raise ValueError("EQUAL weighting needs signals in context")
        # context["signals"] is a list of ComponentResult per leg
        primary = next((s for s in signals if s.metadata.get("role", "primary") == "primary"),
                       signals[0])
        sig = primary.data
        q = max(0.1, min(0.5, spec.legs[0].quantile))
        w = _xs_long_short_top_bottom(sig, q)
        return ComponentResult(
            data=w,
            metadata={
                "weighting": "EQUAL",
                "quantile":  q,
                "side":      "long_short",
                "method":    "rank top-q minus rank bottom-q",
            },
        )


# ── INV_VOL ───────────────────────────────────────────


@register_component(ComponentRole.WEIGHTING, "INV_VOL")
class WeightingInvVol(Component):
    """Inverse-volatility weighting on top/bottom quantiles. Inside each
    side, weight ∝ 1 / 12m rolling vol so high-vol assets get smaller
    positions. This is the BAB-style construction (Frazzini-Pedersen 2014)."""
    description = "Inverse-vol within top/bottom quantile (BAB-style)"

    def build(self, spec, context: dict) -> ComponentResult:
        signals = context.get("signals")
        universe = context.get("universe")
        if not signals or universe is None:
            raise ValueError("INV_VOL needs signals + universe in context")

        primary = signals[0]
        sig = primary.data
        q = max(0.1, min(0.5, spec.legs[0].quantile))

        # Need a returns panel to compute vol. Pull from same source as
        # the universe component used (heuristic: re-derive).
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
        elif ac == "EQUITY":
            # C1 substrate (2026-06-05): equity uses cached CRSP universe
            from engine.composer.components.equity_data import crsp_returns_wide
            rw = crsp_returns_wide()
        else:
            raise ValueError(f"INV_VOL needs returns for asset_class={ac}")

        vol = rw.rolling(12).std().shift(1)
        inv = 1.0 / vol.replace(0, pd.NA)

        out = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
        for dt, row in sig.iterrows():
            valid = row.dropna()
            if len(valid) < 4 or dt not in inv.index:
                continue
            n_top = max(1, int(round(len(valid) * q)))
            n_bot = max(1, int(round(len(valid) * q)))
            ranked = valid.sort_values(ascending=False)
            top    = ranked.index[:n_top]
            bot    = ranked.index[-n_bot:]

            inv_row_top = inv.loc[dt, top].dropna()
            inv_row_bot = inv.loc[dt, bot].dropna()
            if len(inv_row_top) and inv_row_top.sum() > 0:
                w_top = inv_row_top / inv_row_top.sum()
                out.loc[dt, w_top.index] = w_top
            if len(inv_row_bot) and inv_row_bot.sum() > 0:
                w_bot = inv_row_bot / inv_row_bot.sum()
                out.loc[dt, w_bot.index] = -w_bot

        return ComponentResult(
            data=out,
            metadata={
                "weighting": "INV_VOL",
                "quantile":  q,
                "vol_window_months": 12,
                "lagged_by": 1,
                "side":      "long_short",
            },
        )


# ── SIGNAL_RANK (2026-06-05) ──────────────────────────


@register_component(ComponentRole.WEIGHTING, "SIGNAL_RANK")
class WeightingSignalRank(Component):
    """Rank-weighted long-short: weight ∝ (rank - median) so the top
    and bottom of the cross-section get the largest positive / negative
    weights, with smooth tail-off toward the middle. Used in academic
    factor specs where the signal is treated as a continuous score
    (e.g. price-of-quality regressions) rather than a top/bottom-q
    discrete sort.

    Sign convention: HIGHER signal = LONG. Normalized so the long side
    sums to +1 and the short side sums to -1 (market-neutral).
    """
    description = "Rank-proportional XS long-short (continuous-score signals)"

    def build(self, spec, context: dict) -> ComponentResult:
        signals = context.get("signals")
        if not signals:
            raise ValueError("SIGNAL_RANK weighting needs signals in context")
        primary = next(
            (s for s in signals if s.metadata.get("role", "primary") == "primary"),
            signals[0],
        )
        sig = primary.data
        out = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
        for dt, row in sig.iterrows():
            valid = row.dropna()
            n = len(valid)
            if n < 4:
                continue
            # rank 1..n, center on (n+1)/2 so top gets positive, bottom negative
            ranks = valid.rank(method="average")
            centered = ranks - (n + 1) / 2.0
            # normalize so long side sums to +1 (and short side to -1)
            pos = centered.clip(lower=0)
            neg = centered.clip(upper=0)
            pos_sum = pos.sum()
            neg_sum = -neg.sum()   # positive number
            w = pd.Series(0.0, index=valid.index)
            if pos_sum > 0:
                w[pos.index] = pos / pos_sum
            if neg_sum > 0:
                w[neg.index] = w[neg.index] + neg / neg_sum
            out.loc[dt, w.index] = w
        return ComponentResult(
            data=out,
            metadata={
                "weighting":     "SIGNAL_RANK",
                "method":        "centered-rank, long+short normalized to ±1",
                "side":          "long_short",
                "uses_quantile": False,   # uses the FULL cross-section
            },
        )


# ── RISK_PARITY ───────────────────────────────────────


@register_component(ComponentRole.WEIGHTING, "RISK_PARITY")
class WeightingRiskParity(Component):
    """Risk-parity: each leg's weight inversely proportional to its
    realized vol. For a single-signal hypothesis this collapses to
    EQUAL × vol-target; the cross-asset 4-leg sleeve uses it across
    asset classes."""
    description = "Risk-parity: weight inversely proportional to leg vol"

    def build(self, spec, context: dict) -> ComponentResult:
        # For a single-signal hypothesis we delegate to EQUAL and apply
        # vol-target downstream — that's the canonical RP-for-single-signal.
        signals = context.get("signals")
        if not signals:
            raise ValueError("RISK_PARITY needs signals in context")
        primary = signals[0]
        sig = primary.data
        q = max(0.1, min(0.5, spec.legs[0].quantile))
        w = _xs_long_short_top_bottom(sig, q)
        return ComponentResult(
            data=w,
            metadata={
                "weighting":  "RISK_PARITY",
                "note":       "single-signal RP = EQUAL XS + downstream vol-target",
                "quantile":   q,
            },
        )
