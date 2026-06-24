"""tests/test_alpha_factory_decide_verdict.py — audit A2 F1 regression tests.

Tests the pure _decide_verdict function extracted from screen_candidate
during the 2026-06-03 audit A2 fix. Covers:

  - F1 bug regression: DEAD/WEAK decay → RED even with strong stats
  - happy path: strong stats + alive → GREEN
  - stat failures → RED
  - marginal alive → YELLOW
  - undefined inputs → YELLOW
"""
from __future__ import annotations

import math

import pytest

from engine.validation.alpha_factory import _decide_verdict


# ─────────────────────── F1 regression — the actual bug ───────────────


def test_F1_dead_decay_forces_RED_even_with_strong_stats():
    """F1 bug regression (audit A2 2026-06-03): a strategy with strong
    DSR AND strong residual alpha but DEAD recent window must be RED.

    Pre-fix this returned YELLOW with a footnote, which made decay-
    failure indistinguishable from marginal-stats YELLOW. A dead
    strategy is not iterating-worthy."""
    light, reasons = _decide_verdict(
        net_dsr      = 0.95,                        # strong (≥ 0.90)
        alpha_t      = 3.5,                         # strong (≥ 2.0)
        alpha_ann    = 0.07,                        # positive
        bench_used   = "ff5_umd",
        decay_verdict= "DEAD — recent-window Sharpe <= 0",
    )
    assert light == "RED", (
        f"expected RED for DEAD decay, got {light}; reasons={reasons}"
    )
    assert any("decay" in r.lower() for r in reasons)


def test_F1_weak_decay_forces_RED_even_with_strong_stats():
    """Same as DEAD but WEAK variant."""
    light, reasons = _decide_verdict(
        net_dsr      = 0.95,
        alpha_t      = 3.5,
        alpha_ann    = 0.07,
        bench_used   = "ff5_umd",
        decay_verdict= "WEAK recently — recent Sharpe < 0.3",
    )
    assert light == "RED"


def test_F1_dead_decay_RED_even_when_other_logic_would_have_returned_YELLOW():
    """If stats are marginal AND decay is dead, still RED (decay takes
    priority over marginal-stats YELLOW)."""
    light, reasons = _decide_verdict(
        net_dsr      = 0.75,                        # ok but not strong
        alpha_t      = 1.80,                        # weak alpha (1.65 < t < 2.0)
        alpha_ann    = 0.04,
        bench_used   = "ff5_umd",
        decay_verdict= "DEAD — recent-window Sharpe <= 0",
    )
    assert light == "RED"


# ─────────────────────── happy paths preserved ────────────────────────


def test_GREEN_with_strong_stats_and_alive():
    light, reasons = _decide_verdict(
        net_dsr      = 0.95, alpha_t = 3.5, alpha_ann = 0.07,
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "GREEN"
    assert any("survives" in r for r in reasons)


def test_GREEN_with_strong_stats_and_alive_but_front_loaded():
    """ALIVE_BUT_FRONT_LOADED is not DEAD/WEAK; current contract says
    alive=True → can still be GREEN."""
    light, _ = _decide_verdict(
        net_dsr      = 0.92, alpha_t = 2.5, alpha_ann = 0.05,
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE but FRONT-LOADED — early period far stronger",
    )
    assert light == "GREEN"


# ─────────────────────── stat-failure paths ───────────────────────────


def test_RED_when_dsr_collapses_under_cost():
    light, reasons = _decide_verdict(
        net_dsr      = 0.50,                        # below ok bar (0.70)
        alpha_t      = 3.0, alpha_ann = 0.06,
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "RED"
    assert any("deflated SR" in r for r in reasons)


def test_RED_when_no_residual_alpha():
    light, reasons = _decide_verdict(
        net_dsr      = 0.95,
        alpha_t      = 0.50,                        # below weak bar (1.65)
        alpha_ann    = 0.01,
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "RED"
    assert any("residual alpha" in r for r in reasons)


def test_RED_when_alpha_negative():
    light, _ = _decide_verdict(
        net_dsr      = 0.95,
        alpha_t      = 2.5,                         # |t| OK but...
        alpha_ann    = -0.03,                       # negative alpha
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "RED"   # real_alpha = False when alpha_ann <= 0


# ─────────────────────── YELLOW marginal-alive path ───────────────────


def test_YELLOW_marginal_stats_alive():
    """ok DSR + weak alpha + alive → YELLOW (iterate)."""
    light, reasons = _decide_verdict(
        net_dsr      = 0.80,                        # ok but not strong
        alpha_t      = 1.75,                        # weak (1.65 ≤ t < 2.0)
        alpha_ann    = 0.03,
        bench_used   = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "YELLOW"
    assert any("marginal" in r for r in reasons)


# ─────────────────────── undefined inputs ──────────────────────────────


def test_YELLOW_undefined_net_dsr():
    light, _ = _decide_verdict(
        net_dsr      = float("nan"),
        alpha_t      = 3.0, alpha_ann = 0.05, bench_used = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "YELLOW"


def test_YELLOW_undefined_alpha_t():
    light, _ = _decide_verdict(
        net_dsr      = 0.95,
        alpha_t      = float("nan"), alpha_ann = 0.05, bench_used = "ff5_umd",
        decay_verdict= "ALIVE — recent edge intact",
    )
    assert light == "YELLOW"


# ─────────────────────── F7 regime-gate regression tests ──────────────


def test_F7_worst_regime_below_floor_forces_RED():
    """F7 (audit A2 2026-06-03): if worst-regime Sharpe is below the
    floor, the strategy is regime-fragile and cannot pass to GREEN,
    even with strong overall stats. Same doctrine as F1: a strategy
    that collapses in a crisis regime is not iterating-worthy.

    Pre-fix: mom_hedge_overlay passed alpha_factory gate because gate
    was regime-blind; killed by separate ablation later.
    """
    light, reasons = _decide_verdict(
        net_dsr             = 0.95,
        alpha_t             = 3.0,
        alpha_ann           = 0.05,
        bench_used          = "ff5_umd",
        decay_verdict       = "ALIVE — recent edge intact",
        worst_regime_sharpe = -3.5,         # 2018 vol-mageddon level
        worst_regime_label  = "2018_volmageddon_Q1",
        regime_floor        = -1.0,
    )
    assert light == "RED"
    assert any("regime-fragile" in r for r in reasons)
    assert any("2018_volmageddon_Q1" in r for r in reasons)


def test_F7_worst_regime_above_floor_does_not_block():
    """Worst-regime Sharpe at -0.5 (above floor of -1.0) should NOT
    block GREEN; other tests determine the verdict."""
    light, reasons = _decide_verdict(
        net_dsr             = 0.95,
        alpha_t             = 3.0,
        alpha_ann           = 0.05,
        bench_used          = "ff5_umd",
        decay_verdict       = "ALIVE — recent edge intact",
        worst_regime_sharpe = -0.5,
        worst_regime_label  = "2018_q4_drawdown",
        regime_floor        = -1.0,
    )
    assert light == "GREEN"


def test_F7_none_disables_regime_gate():
    """When regime info is None (opt-out / not computed), gate is
    silent — verdict driven entirely by other layers."""
    light, _ = _decide_verdict(
        net_dsr             = 0.95,
        alpha_t             = 3.0,
        alpha_ann           = 0.05,
        bench_used          = "ff5_umd",
        decay_verdict       = "ALIVE — recent edge intact",
        worst_regime_sharpe = None,        # disabled
        worst_regime_label  = None,
    )
    assert light == "GREEN"


def test_F7_strict_floor_zero_blocks_negative_regime():
    """If user sets regime_floor=0.0, any negative-Sharpe regime forces RED.
    Stricter than default -1.0 floor."""
    light, _ = _decide_verdict(
        net_dsr             = 0.95,
        alpha_t             = 3.0,
        alpha_ann           = 0.05,
        bench_used          = "ff5_umd",
        decay_verdict       = "ALIVE — recent edge intact",
        worst_regime_sharpe = -0.2,
        worst_regime_label  = "2018_q4_drawdown",
        regime_floor        = 0.0,
    )
    assert light == "RED"


def test_F7_decay_takes_priority_over_regime():
    """If BOTH decay fails AND regime fails, RED with decay reason
    (decay is checked first; both findings would be valid)."""
    light, reasons = _decide_verdict(
        net_dsr             = 0.95,
        alpha_t             = 3.0,
        alpha_ann           = 0.05,
        bench_used          = "ff5_umd",
        decay_verdict       = "DEAD — recent-window Sharpe <= 0",
        worst_regime_sharpe = -3.5,
        worst_regime_label  = "2018_volmageddon_Q1",
        regime_floor        = -1.0,
    )
    assert light == "RED"
    # decay reason fired first
    assert any("decay" in r.lower() for r in reasons)


def test_regime_breakdown_helper_filters_low_obs():
    """_regime_breakdown skips regimes with n_obs < min_obs."""
    import pandas as pd
    from engine.validation.alpha_factory import _regime_breakdown

    # Synthetic weekly returns covering 2014-09 → 2020-12
    idx = pd.date_range("2014-09-05", "2020-12-31", freq="W-FRI")
    rng_seed = pd.Series(
        [0.001 * ((i % 7) - 3) for i in range(len(idx))],   # deterministic pattern
        index=idx,
    )
    breakdown, worst_label, worst_sharpe = _regime_breakdown(
        rng_seed, annualization=52, min_obs=13,
    )
    # Regimes that fall outside the data range should NOT appear
    assert "2023_recovery" not in breakdown
    # Regimes inside range should be present with computed Sharpe
    assert len(breakdown) > 0
    # Worst regime should be in the breakdown
    assert worst_label in breakdown
    assert worst_sharpe == breakdown[worst_label]


def test_regime_breakdown_returns_empty_on_no_data():
    """When returns span no regime windows, breakdown is empty and
    worst_label/sharpe are None."""
    import pandas as pd
    from engine.validation.alpha_factory import _regime_breakdown

    # Returns from a window completely outside REGIMES_V1 (pre-2014)
    idx = pd.date_range("2010-01-01", "2010-12-31", freq="W-FRI")
    r = pd.Series([0.001] * len(idx), index=idx)
    breakdown, worst_label, worst_sharpe = _regime_breakdown(
        r, annualization=52, min_obs=13,
    )
    assert breakdown == {}
    assert worst_label is None
    assert worst_sharpe is None
