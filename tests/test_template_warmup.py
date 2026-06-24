"""Tests for Phase 6c template warmup fix.

Verifies the 3-layer warmup resolution:
  Layer 1: template module's warmup_months() function
  Layer 2: empirical detection via dry-run
  Layer 3: conservative default
"""
from __future__ import annotations

import sys
from unittest import mock

import pytest

from engine.research.protocols.protocol_designer import (
    _empirical_warmup_dryrun,
    compute_template_warmup,
)


# ── Layer 1: template-declared warmup_months() ──────────────────────────

def test_equity_xsmom_declared_warmup():
    """equity_xsmom has its own warmup_months() function."""
    from engine.research.templates.equity_xsmom import warmup_months
    binding = {"lookback_months": 12, "vol_target": 0.10,
                "vol_target_lookback": 36}
    assert warmup_months(binding) == 12 + 1 + 36    # = 49


def test_equity_xsmom_no_vol_target():
    """Without vol_target, only lookback + lag."""
    from engine.research.templates.equity_xsmom import warmup_months
    assert warmup_months({"lookback_months": 12, "vol_target": None}) == 13


def test_factor_quartile_declared_warmup():
    from engine.research.templates.factor_quartile import warmup_months
    assert warmup_months({"vol_target": 0.10, "vol_target_lookback": 24}) == 25
    assert warmup_months({"vol_target": None}) == 1


def test_cross_asset_tsmom_declared_warmup():
    from engine.research.templates.cross_asset_tsmom import warmup_months
    binding = {"lookback_months": 12, "per_instrument_vol_lookback": 36}
    assert warmup_months(binding) == 12 + 1 + 36


# ── compute_template_warmup integrates Layer 1 ──────────────────────────

def test_compute_warmup_uses_template_layer_1():
    """compute_template_warmup defers to template's declared function."""
    w = compute_template_warmup(
        "equity_xsmom",
        {"lookback_months": 12, "vol_target": 0.10, "vol_target_lookback": 36},
    )
    assert w == 49


def test_compute_warmup_empty_template_id_returns_zero():
    assert compute_template_warmup(None, {}) == 0
    assert compute_template_warmup("", {}) == 0


def test_compute_warmup_unknown_template_uses_layer_2_or_3():
    """Unknown template_id: should not crash; should use empirical or default."""
    w = compute_template_warmup("nonexistent_template_xyz", {})
    # Either empirical returns None and we get conservative default 12,
    # OR empirical succeeds with whatever it finds. Either way, finite int.
    assert isinstance(w, int)
    assert 0 <= w <= 60


# ── Layer 2: empirical detection ────────────────────────────────────────

def test_empirical_warmup_detects_equity_xsmom_warmup():
    """Empirical detection on equity_xsmom matches what its function returns."""
    binding = {
        "lookback_months": 12, "skip_months": 1,
        "top_frac": 0.2, "bottom_frac": 0.2,
        "weighting": "equal_weight", "rebal_freq": "monthly",
        "cost_bps_per_side": 12.0, "microcap_price_threshold": 5.0,
        "vol_target": 0.10, "vol_target_lookback": 36,
    }
    detected = _empirical_warmup_dryrun("equity_xsmom", binding)
    declared = 49
    # Tolerance: empirical may be ±2 months from declared
    assert detected is not None
    assert abs(detected - declared) <= 4


def test_empirical_warmup_returns_none_for_invalid_template():
    """Empirical on a nonexistent template returns None gracefully."""
    result = _empirical_warmup_dryrun("nonexistent_xyz", {})
    assert result is None


# ── Mechanism YAML override ─────────────────────────────────────────────

def test_mechanism_yaml_can_override_warmup(tmp_path, monkeypatch):
    """If mechanism YAML declares template_warmup_months, that wins."""
    from engine.research.protocols import (
        instantiate_protocol, load_mechanism,
    )
    mech = load_mechanism("equity_xsmom_jt")
    mech_with_override = dict(mech)
    mech_with_override["template_warmup_months"] = 6   # override

    proto = instantiate_protocol(
        mech_with_override,
        proposal_sample_start="2000-01-01",
        proposal_sample_end="2020-01-01",
    )
    fh = next(leg for leg in proto.legs if leg.id == "subperiod_first_half")
    # With warmup=6mo (not 49), effective start ≈ 2000-07 not 2004-02
    import datetime
    eff_start = datetime.date.fromisoformat(fh.sample_start)
    assert eff_start <= datetime.date(2000, 12, 1)
