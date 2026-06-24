"""composer.components.risk_filters — RISK_FILTER atomic components.

Each component produces a per-date scalar multiplier applied to the
returns series before final output. Composer multiplies the raw
weight×return series by this multiplier element-wise.
"""
from __future__ import annotations

import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, register_component,
)


@register_component(ComponentRole.RISK_FILTER, "VOL_TARGET")
class RiskFilterVolTarget(Component):
    """Vol-targeting: rescale realized monthly returns to hit the spec's
    target annual vol. Computes 12-month trailing realized vol of the
    pre-target returns, then multiplier = target_ann_vol / (12m_vol × √12).

    Lagged by 1 to avoid look-ahead. Returns multipliers ≥ 0.
    """
    description = "Constant annualized vol target via trailing 12m vol"

    def build(self, spec, context: dict) -> ComponentResult:
        raw_returns = context.get("pre_filter_returns")
        if raw_returns is None:
            raise ValueError("VOL_TARGET needs pre_filter_returns in context")
        target = spec.risk.vol_target_annual
        if target is None:
            return ComponentResult(
                data=pd.Series(1.0, index=raw_returns.index),
                metadata={"risk_filter": "VOL_TARGET", "skipped": True},
            )

        # Realized 12m annualized vol (lagged by 1 month)
        vol12 = raw_returns.rolling(12).std().shift(1) * (12 ** 0.5)
        with pd.option_context("mode.use_inf_as_na", True):
            mult = (target / vol12).fillna(0.0)
        # Cap multiplier at spec.risk.max_leverage if set, else unbounded
        cap = spec.risk.max_leverage
        if cap is not None and cap > 0:
            mult = mult.clip(upper=float(cap))
        return ComponentResult(
            data=mult,
            metadata={
                "risk_filter":         "VOL_TARGET",
                "target_annual":       target,
                "max_leverage_cap":    cap,
                "vol_lookback_months": 12,
                "vol_lagged_by":       1,
            },
        )
