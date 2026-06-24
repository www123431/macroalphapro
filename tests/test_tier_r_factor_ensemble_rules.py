"""
tests/test_tier_r_factor_ensemble_rules.py — Sprint Week 4 (spec id=50 §4.7).

Pre-registration: docs/spec_factor_ensemble_v1.md §4.7 amendment 2026-05-09
(pre-Sprint-Week-4 audit Issue #3).

Verifies:
  • rule_factor_ensemble_no_lookahead — Layer 1 AST scan + Layer 2 runtime probe
    -- Current production: at most MID severity (AST hit on legitimate guarded
       yfinance .info usage); MUST NOT fire HIGH because guard is in place.
    -- Layer 2 detects guard regression: simulate by patching SPEC_LOCK_DATE to
       a past date so probe returns non-NaN → HIGH severity expected.
  • rule_factor_ensemble_no_param_tuning — passes on current code; flags drift
    when locked constants are mutated.
  • rule_factor_ensemble_baseline_reproducibility — silent (None) when no
    gate0 file; HIGH when mandatory_pass=False; LOW on PASS_WITH_DIRECTIONAL_CAVEAT.
  • All 3 rules registered in CRITICAL_RULES.
"""
from __future__ import annotations

import datetime
import json

import pytest

from engine import auto_audit_rules as ar


# ─────────────────────────────────────────────────────────────────────────────
# Registration sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_three_factor_ensemble_rules_registered_in_critical():
    expected = {
        ar.rule_factor_ensemble_no_lookahead,
        ar.rule_factor_ensemble_no_param_tuning,
        ar.rule_factor_ensemble_baseline_reproducibility,
    }
    actual_set = set(ar.CRITICAL_RULES)
    missing = expected - actual_set
    assert not missing, f"rules not registered in CRITICAL_RULES: {missing}"


# ─────────────────────────────────────────────────────────────────────────────
# rule_factor_ensemble_no_lookahead
# ─────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_layer2_passes_on_current_quality_guard():
    """Current quality.py ships with SPEC_LOCK_DATE guard intact.
    Layer 2 runtime probe must NOT fire HIGH severity.
    AST Layer 1 may legitimately flag MID (yfinance .info access guarded by
    the runtime check), but that's advisory only."""
    result = ar.rule_factor_ensemble_no_lookahead()
    if result is None:
        return  # PASS
    assert result["severity"] != "HIGH", (
        f"current quality.py SPEC_LOCK_DATE guard should keep severity ≤ MID; "
        f"got {result['severity']} snapshot={result['snapshot']}"
    )
    # Layer 1 hit must be advisory MID only
    assert result["severity"] == "MID"
    assert result["snapshot"]["layer"] == 1


def test_no_lookahead_layer2_high_when_quality_guard_regressed(monkeypatch):
    """Simulate guard regression: monkey-patch SPEC_LOCK_DATE to a future
    date that the probe's `as_of - 1day` is still in the past — the guard
    should still trigger and return all-NaN. Then patch the function to
    bypass the guard entirely → probe returns non-NaN → rule fires HIGH."""
    import pandas as pd
    import numpy as np

    # Force compute_quality_signal to return non-NaN regardless of as_of
    # — simulating guard regression
    def regressed_quality(as_of, universe, asset_classes, use_cache=False):
        return pd.Series([0.5] * len(universe), index=universe, dtype=float)

    monkeypatch.setattr("engine.factors.quality.compute_quality_signal", regressed_quality)

    result = ar.rule_factor_ensemble_no_lookahead()
    assert result is not None, "rule should fire when guard regressed"
    assert result["severity"] == "HIGH", f"regressed guard must fire HIGH, got {result['severity']}"
    assert result["snapshot"]["layer"] == 2
    layer2 = result["snapshot"]["layer2_failure"]
    assert layer2 is not None
    assert "non_nan_tickers" in layer2 and len(layer2["non_nan_tickers"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# rule_factor_ensemble_no_param_tuning
# ─────────────────────────────────────────────────────────────────────────────

def test_no_param_tuning_passes_on_locked_constants():
    """Current locked constants match expected values."""
    result = ar.rule_factor_ensemble_no_param_tuning()
    assert result is None, f"locked constants should pass; got {result}"


def test_no_param_tuning_flags_drift_when_constant_mutated(monkeypatch):
    """Mutate one locked constant → rule should fire HIGH with drift snapshot."""
    monkeypatch.setattr("engine.factors.tsmom.LOOKBACK_MONTHS", 99)
    result = ar.rule_factor_ensemble_no_param_tuning()
    assert result is not None
    assert result["severity"] == "HIGH"
    drift_attrs = [d["attr"] for d in result["snapshot"]["drifts"]]
    assert "LOOKBACK_MONTHS" in drift_attrs


# ─────────────────────────────────────────────────────────────────────────────
# rule_factor_ensemble_baseline_reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def _gate0_path():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "data" / "factor_ensemble_v1" / "gate0_baseline_check.json"


def test_baseline_reproducibility_silent_when_no_gate0_file(tmp_path, monkeypatch):
    """No gate0 file → rule returns None (silent; pre-launch state)."""
    # Redirect repo root to tmp_path so the file is absent
    fake_root = tmp_path
    (fake_root / "data" / "factor_ensemble_v1").mkdir(parents=True)
    monkeypatch.setattr(
        "engine.auto_audit_rules.__file__",
        str(fake_root / "engine" / "auto_audit_rules.py"),
    )
    # auto_audit_rules computes path via Path(__file__).resolve().parent.parent;
    # simulate by ensuring the actual gate0 file under that path does not exist.
    # Since we can't fully reroot, we just ensure the actual project file path doesn't have it.
    # This test covers the no-file branch via a different mechanism: temporarily move
    # the real file aside if it exists.
    real_path = _gate0_path()
    backup = None
    if real_path.exists():
        backup = real_path.read_text(encoding="utf-8")
        real_path.unlink()
    try:
        result = ar.rule_factor_ensemble_baseline_reproducibility()
        assert result is None, f"absent gate0 file should be silent; got {result}"
    finally:
        if backup is not None:
            real_path.write_text(backup, encoding="utf-8")


def test_baseline_reproducibility_high_when_mandatory_pass_false():
    """gate0 file present with mandatory_pass=False → HIGH severity."""
    real_path = _gate0_path()
    backup = real_path.read_text(encoding="utf-8") if real_path.exists() else None
    real_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        real_path.write_text(json.dumps({
            "status":           "FAIL_PATHOLOGICAL",
            "harness_sharpe":   float("nan"),
            "mandatory_range":  [-0.5, 2.0],
            "mandatory_pass":   False,
        }), encoding="utf-8")
        result = ar.rule_factor_ensemble_baseline_reproducibility()
        assert result is not None
        assert result["severity"] == "HIGH"
        assert result["snapshot"]["mandatory_pass"] is False
    finally:
        if backup is None:
            real_path.unlink(missing_ok=True)
        else:
            real_path.write_text(backup, encoding="utf-8")


def test_baseline_reproducibility_low_on_directional_caveat():
    """gate0 with mandatory_pass=True but PASS_WITH_DIRECTIONAL_CAVEAT → LOW."""
    real_path = _gate0_path()
    backup = real_path.read_text(encoding="utf-8") if real_path.exists() else None
    real_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        real_path.write_text(json.dumps({
            "status":          "PASS_WITH_DIRECTIONAL_CAVEAT",
            "harness_sharpe":  0.20,
            "mandatory_pass":  True,
            "bpp_delta":       -0.78,
        }), encoding="utf-8")
        result = ar.rule_factor_ensemble_baseline_reproducibility()
        assert result is not None
        assert result["severity"] == "LOW"
    finally:
        if backup is None:
            real_path.unlink(missing_ok=True)
        else:
            real_path.write_text(backup, encoding="utf-8")
