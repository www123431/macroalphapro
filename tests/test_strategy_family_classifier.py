"""Strategy family classifier (2026-06-12 design fix) tests.

Covers:
  - cross_sectional_rank dispatches by signal_inputs (mktcap → SIZE, etc.)
  - factor_combination sorts FF factors canonically (HML+MOM == MOM+HML)
  - portfolio_overlay routes by universe (us_balanced_60_40 → OVERLAY_60_40)
  - carry distinguishes universe (fx_g10 vs commodity)
  - Unknown signal_kinds produce honest UNKNOWN_<X> labels
  - canonical_strategy_family_tag + claim_family_tag wrap consistently
"""
from __future__ import annotations

import pytest

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.research.strategy_family_classifier import (
    canonical_strategy_family_tag,
    claim_family_tag,
    strategy_family_for_spec,
)


def _spec(**kw):
    base = dict(
        hypothesis_id="t-sf",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="2010-01:2024-12",
        signal_inputs=("crsp.msf.mktcap",),
        rebal="monthly",
        weighting="ew",
        expected_holding_period="monthly",
        min_obs_months=36,
        pit_audits=("restatement",),
        cost_model="13bp_per_rt",
        rationale="test",
        extracted_ts="2026-06-12T00:00:00Z",
        model="test",
    )
    base.update(kw)
    return FactorSpec(**base)


# ── cross_sectional_rank ─────────────────────────────────────────


def test_cross_sec_mktcap_classifies_size():
    s = _spec(signal_inputs=("crsp.msf.mktcap",))
    assert strategy_family_for_spec(s) == "SIZE"


def test_cross_sec_vol_12m_classifies_low_vol():
    s = _spec(signal_inputs=("crsp.msf.vol_12m",))
    assert strategy_family_for_spec(s) == "LOW_VOL"


def test_cross_sec_gp_at_classifies_profitability():
    s = _spec(signal_inputs=("compustat.funda.gp_at",))
    assert strategy_family_for_spec(s) == "PROFITABILITY"


def test_cross_sec_book_to_market_classifies_value():
    s = _spec(signal_inputs=("compustat.funda.book_to_market",))
    assert strategy_family_for_spec(s) == "VALUE"


def test_cross_sec_momentum_classifies_momentum():
    s = _spec(signal_inputs=("crsp.msf.ret_12_1",))
    assert strategy_family_for_spec(s) == "MOMENTUM"


def test_cross_sec_unknown_signal_returns_unknown_label():
    s = _spec(signal_inputs=("some.unknown.signal",))
    assert strategy_family_for_spec(s) == "CROSS_SEC_UNKNOWN"


# ── factor_combination (THE DESIGN FLAW THIS FIXES) ──────────────


def test_factor_combination_hml_mom_canonical():
    s = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"),
    )
    assert strategy_family_for_spec(s) == "COMBINATION_HML_MOM"


def test_factor_combination_mom_hml_same_family_as_hml_mom():
    """Order doesn't matter — sorted canonically."""
    s1 = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"),
    )
    s2 = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.mom", "ff.factors_weekly.hml"),
    )
    assert strategy_family_for_spec(s1) == strategy_family_for_spec(s2)


def test_factor_combination_rmw_cma_canonical():
    s = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.rmw", "ff.factors_weekly.cma"),
    )
    # Sorted: CMA < RMW alphabetically
    assert strategy_family_for_spec(s) == "COMBINATION_CMA_RMW"


def test_factor_combination_short_prefix():
    s = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.hml", "ff.smb"),
    )
    assert strategy_family_for_spec(s) == "COMBINATION_HML_SMB"


def test_factor_combination_unrecognized_factor_returns_unknown():
    s = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.invalid", "ff.factors_weekly.hml"),
    )
    # Only 1 recognized factor → < 2 → UNKNOWN
    assert strategy_family_for_spec(s) == "COMBINATION_UNKNOWN"


# ── carry distinguishes universe ─────────────────────────────────


def test_carry_fx_g10_classifies_carry_fx():
    s = _spec(signal_kind="carry", universe="fx_g10")
    assert strategy_family_for_spec(s) == "CARRY_FX"


def test_carry_commodity_classifies_carry_commodity():
    s = _spec(signal_kind="carry", universe="commodity_futures_27")
    assert strategy_family_for_spec(s) == "CARRY_COMMODITY"


def test_carry_treasury_classifies_carry_rates():
    s = _spec(signal_kind="carry", universe="us_treasury_curve")
    assert strategy_family_for_spec(s) == "CARRY_RATES"


# ── other signal kinds ──────────────────────────────────────────


def test_tsmom_classifies_tsmom():
    s = _spec(signal_kind="time_series_momentum",
                universe="us_equities_sector_etf")
    assert strategy_family_for_spec(s) == "TSMOM"


def test_portfolio_overlay_60_40_classifies_overlay_60_40():
    s = _spec(signal_kind="portfolio_overlay", universe="us_balanced_60_40")
    assert strategy_family_for_spec(s) == "OVERLAY_60_40"


def test_vrp_classifies_vrp():
    s = _spec(signal_kind="vrp", universe="us_equities_top_3000")
    assert strategy_family_for_spec(s) == "VRP"


def test_requires_custom_code_classifies_custom_code():
    s = _spec(signal_kind="requires_custom_code", universe="unknown_universe")
    assert strategy_family_for_spec(s) == "CUSTOM_CODE"


# ── tag wrappers ─────────────────────────────────────────────────


def test_canonical_strategy_family_tag_format():
    s = _spec(
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"),
    )
    assert canonical_strategy_family_tag(s) == "strategy_family:COMBINATION_HML_MOM"


def test_claim_family_tag_format():
    assert claim_family_tag("VALUE") == "claim_family:VALUE"
    assert claim_family_tag("momentum") == "claim_family:MOMENTUM"
    assert claim_family_tag(None) == "claim_family:UNKNOWN"
    assert claim_family_tag("") == "claim_family:UNKNOWN"
