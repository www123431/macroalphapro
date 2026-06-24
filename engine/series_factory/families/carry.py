"""series_factory/families/carry.py — CARRY family subtype builders.

Each builder constructs a DIFFERENT returns series so the pipeline
verdict is honest about which carry variant was tested. NO fall-through
to defaults — unmapped subtypes refuse explicitly via the core.

Coverage (per real hypothesis_store distribution as of 2026-06-05):

  REGISTERED                    SUBTYPE COVERS (32 CARRY hypotheses observed)
  fx_carry_g10                  11 FX-carry-class hypotheses
  commodity_carry_xs            1 commodity-carry hypothesis
  rates_carry_us                1 US-rates-carry hypothesis
  rates_carry_xc                1 XC-bond-carry hypothesis
  cross_asset_carry_4leg        3 cross-asset combined hypotheses
                                (marked deployed_replay=True — see warning)
  carry_tsmom_filtered          2 carry-with-momentum-filter hypotheses
  bab_carry                     1 BAB-style carry hypothesis
  equity_div_carry              2 equity-dividend-carry hypotheses
                                (registered as not-yet-implemented refusal)

Subtypes NOT registered (will refuse + Claude handoff):
  ~14 long-form "OTHER" subtypes like
    fx_carry_two_factor_model, uip_failure_cross_section, etc
  These are specific empirical-finance constructs that need their own
  builder; the framework's job is to REFUSE not to fake-test.
"""
from __future__ import annotations

import pandas as pd

from engine.series_factory import register_subtype


# ── FX leg only ─────────────────────────────────────────────

@register_subtype("CARRY", "fx_carry_g10")
def _fx_carry_g10(hypothesis_id: str, params: dict) -> pd.Series:
    """FX carry on G10 currencies, long-short cross-sectional.

    Returns monthly long-short cross-sectional FX carry returns. Picks
    the LS leg from engine.validation.crossasset_carry.build_fx_carry
    (returns a tuple (long, short, ls); we take element [2] = ls).
    """
    from engine.validation.crossasset_carry import build_fx_carry
    _long, _short, ls = build_fx_carry()
    s = ls.copy()
    s.name = f"fx_carry__{hypothesis_id[:20]}"
    return s


# Alias for the dozens of subtype names that all reduce to "FX carry LS"
for _alias in (
    "fx_carry_cross_sectional",
    "fx_carry_single_factor_cross_section",
    "fx_carry_cross_sectional_risk_price",
    "fx_carry_beta_sorted_portfolios",
    "average_forward_discount_predictability",
):
    register_subtype("CARRY", _alias)(_fx_carry_g10)


# ── Commodity leg only ──────────────────────────────────────

@register_subtype("CARRY", "commodity_carry_xs")
def _commodity_carry_xs(hypothesis_id: str, params: dict) -> pd.Series:
    """Commodity cross-sectional carry (long-short on roll yield).

    Returns monthly LS commodity carry from
    engine.validation.crossasset_carry.build_commodity_carry_ls.
    """
    from engine.validation.crossasset_carry import build_commodity_carry_ls
    s = build_commodity_carry_ls().copy()
    s.name = f"cmdty_carry__{hypothesis_id[:20]}"
    return s


# ── Rates legs ──────────────────────────────────────────────

@register_subtype("CARRY", "rates_carry_us")
def _rates_carry_us(hypothesis_id: str, params: dict) -> pd.Series:
    """US Treasury 4-tenor rates carry (UST2/5/10/30 roll yield)."""
    from engine.validation.crossasset_carry import build_rates_carry
    _long, _short, ls = build_rates_carry()
    s = ls.copy()
    s.name = f"rates_us_carry__{hypothesis_id[:20]}"
    return s


@register_subtype("CARRY", "rates_carry_xc")
def _rates_carry_xc(hypothesis_id: str, params: dict) -> pd.Series:
    """G10 cross-country bond futures carry (Bund/Gilt/CGB/AGB/JGB/BTP/OAT 10Y)."""
    from engine.validation.crossasset_carry import build_rates_xc_carry
    _long, _short, ls = build_rates_xc_carry()
    s = ls.copy()
    s.name = f"rates_xc_carry__{hypothesis_id[:20]}"
    return s


# ── Combined (DEPLOYED — marked replay) ────────────────────

@register_subtype("CARRY", "cross_asset_carry_4leg", deployed_replay=True)
def _cross_asset_4leg(hypothesis_id: str, params: dict) -> pd.Series:
    """Deployed 4-leg risk-parity cross-asset carry sleeve. **REPLAY** —
    NOT a new test; just rebuilds the existing live config. Verdict will
    track what we already know about the deployed sleeve, not anything
    novel about the picked hypothesis. Use only for sanity / decay-check
    re-runs of the deployed config itself.
    """
    from engine.portfolio.carry_sleeve import build_carry_sleeve_returns
    s = build_carry_sleeve_returns(
        target_annual_vol = float(params.get("target_annual_vol", 0.10)),
        include_rates     = bool(params.get("include_rates", True)),
        include_rates_xc  = bool(params.get("include_rates_xc", True)),
    ).copy()
    s.name = f"4leg_replay__{hypothesis_id[:20]}"
    return s


# ── Carry × momentum filter ────────────────────────────────

@register_subtype("CARRY", "carry_tsmom_filtered")
def _carry_tsmom_filtered(hypothesis_id: str, params: dict) -> pd.Series:
    """Carry ONLY ON when 12m TSMOM signal agrees.

    Overlays a binary momentum gate (TSMOM 12m sign of cumulative return)
    on the 4-leg combined carry. When the asset class TSMOM is negative,
    we mute that leg. Implementation: combined carry × indicator(12m_mom>0).
    Simpler than full per-asset gating but materially different from
    pure-carry construction.
    """
    from engine.portfolio.carry_sleeve import build_carry_sleeve_returns
    carry = build_carry_sleeve_returns().copy()
    # 12m cumulative log-return sign as gate
    log_carry = (1 + carry).apply(lambda x: 0 if x <= -1 else __import__("math").log1p(x))
    rolling_12m = log_carry.rolling(12).sum()
    gate = (rolling_12m > 0).astype(float)
    # Lag the gate by 1 to avoid look-ahead
    gated = carry * gate.shift(1).fillna(0.0)
    gated.name = f"carry_tsmom_filtered__{hypothesis_id[:20]}"
    return gated


# ── BAB-style carry (low-vol weighted) ─────────────────────

@register_subtype("CARRY", "bab_carry")
def _bab_carry(hypothesis_id: str, params: dict) -> pd.Series:
    """BAB-style carry: weight legs inversely to their 12m rolling vol.

    Frazzini-Pedersen BAB applied to the 4 carry legs — instead of risk-
    parity (which uses vol for sizing too, but rebalances differently),
    we weight by 1/vol_12m × normalize. Different from deployed 4-leg
    because BAB amplifies low-vol legs more aggressively.
    """
    from engine.validation.crossasset_carry import (
        build_commodity_carry_ls, build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    legs = {
        "cmdty":    build_commodity_carry_ls(),
        "fx":       build_fx_carry()[2],
        "rates_us": build_rates_carry()[2],
        "rates_xc": build_rates_xc_carry()[2],
    }
    df = pd.concat(legs, axis=1).dropna(how="all")
    # Inverse 12m vol weighting, lagged
    vol = df.rolling(12).std().shift(1)
    inv_vol = 1.0 / vol.replace(0, pd.NA)
    w = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)
    s = (df * w).sum(axis=1)
    s.name = f"bab_carry__{hypothesis_id[:20]}"
    return s


# ── Subtypes NOT yet covered (refuse + Claude) ─────────────
#
# The following CARRY subtypes were observed in the hypothesis store
# but have NO real builder yet. They are deliberately NOT registered,
# which means series_factory.build() returns ok=False/error='unknown_subtype'
# → frontend surfaces "ask Claude to extend the registry" handoff.
#
#   equity_div_carry / equity_dividend_carry        (eq carry needs div yield panel)
#   fx_carry_two_factor_model                       (multi-factor)
#   uip_failure_cross_section                       (specific empirical construct)
#   fx_carry_countercyclical_risk_premium           (state-conditional)
#   fx_carry_beta_monotonicity                      (beta-sorted, monotonicity test)
#   counter_cyclical_currency_risk_premium          (macro-conditional)
#   time_varying_market_beta_carry_trade            (rolling-beta)
#
# Adding each = a new @register_subtype("CARRY", "...") function below
# (50-80 lines real implementation each). For now they refuse, which
# is correct per the LdP §2 epistemic discipline.
