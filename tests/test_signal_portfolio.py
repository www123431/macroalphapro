"""
tests/test_signal_portfolio.py — production signal + portfolio invariants.

Spot-check: vol target / leverage cap / weight cap / config constants.
Not exhaustive — backtest-level tests live in S-3 reproducibility freeze.
"""
import pytest


def test_alignment_surface_matches_live_config():
    """The 7 keys in ALIGNMENT_SURFACE must match engine.config production
    constants. Drift here = thesis-claim invariant violation."""
    from engine.auto_audit_rules import ALIGNMENT_SURFACE
    import engine.config as cfg
    for key, expected in ALIGNMENT_SURFACE.items():
        actual = getattr(cfg, key, None)
        assert actual == expected, (
            f"engine.config.{key} = {actual!r}, ALIGNMENT_SURFACE expects {expected!r}"
        )


def test_target_vol_positive_and_reasonable():
    from engine.config import TARGET_VOL
    assert 0.0 < TARGET_VOL < 1.0


def test_max_leverage_within_bounds():
    from engine.config import MAX_LEVERAGE
    assert 0.5 <= MAX_LEVERAGE <= 5.0


def test_max_weight_below_one():
    """Single-position weight cap must be a fraction; >1.0 = leverage bug."""
    from engine.config import MAX_WEIGHT
    assert 0.0 < MAX_WEIGHT < 1.0


def test_net_exposure_bounds_consistent():
    """MIN_NET ≤ MAX_NET; both within [-1, 1]."""
    from engine.config import MIN_NET, MAX_NET
    assert MIN_NET <= MAX_NET
    assert -1.0 <= MIN_NET <= 1.0
    assert -1.0 <= MAX_NET <= 1.0


def test_production_signal_is_known_string():
    """PRODUCTION_SIGNAL must be one of {ql01_bab, tsmom} string literals
    that signal.py implements."""
    from engine.config import PRODUCTION_SIGNAL
    assert PRODUCTION_SIGNAL in ("ql01_bab", "tsmom"), \
        f"PRODUCTION_SIGNAL={PRODUCTION_SIGNAL!r} is not implemented"


def test_regime_scale_within_spec_bounds():
    """REGIME_SCALE production value must lie within spec_v1 §3.6 procedural
    bounds [0.3, 0.7]. 2026-05-08 production swap: was 1.0 (disabled),
    now 0.6 (supervisor pick after spec_multivariate_msm_v3.md OOS verdict
    DESCRIPTIVE_POSITIVE)."""
    from engine.config import REGIME_SCALE
    assert 0.3 <= REGIME_SCALE <= 0.7, (
        f"REGIME_SCALE={REGIME_SCALE} outside spec_v1 §3.6 bounds [0.3, 0.7]; "
        f"changes outside this range require hypothesis_amend on multivariate_msm_v3."
    )


def test_amendment_kinds_n_trials_monotone():
    """Higher-impact amendment kinds must consume ≥ trial count of lower ones."""
    from engine.preregistration import AMENDMENT_KINDS
    # clarification/scope_narrow/superseded = 0 (no trial cost)
    assert AMENDMENT_KINDS["clarification"] == 0
    # threshold_tweak < hypothesis_amend < endpoint_swap (semantic ordering)
    assert AMENDMENT_KINDS["threshold_tweak"] < AMENDMENT_KINDS["hypothesis_amend"]
    assert AMENDMENT_KINDS["hypothesis_amend"] < AMENDMENT_KINDS["endpoint_swap"]
