"""Tests for tunable_bindings whitelist (Huatai 借鉴 ①)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import yaml


# ── build_mechanism_stub includes tunable_bindings + locked_logic_anchor ──

def test_stub_includes_tunable_bindings():
    from engine.research.discovery.queue_actions import build_mechanism_stub
    entry = {
        "title": "X", "routing": {"family": "carry"},
        "extraction": {"family_guess": "carry"},
    }
    stub = build_mechanism_stub(entry)
    assert "tunable_bindings" in stub
    assert isinstance(stub["tunable_bindings"], list)
    # carry family should have predefined tunables
    assert "top_frac" in stub["tunable_bindings"]
    assert "cost_bps_per_side" in stub["tunable_bindings"]


def test_stub_includes_locked_logic_anchor():
    from engine.research.discovery.queue_actions import build_mechanism_stub
    entry = {"title": "X", "extraction": {"family_guess": "momentum"}}
    stub = build_mechanism_stub(entry)
    assert "locked_logic_anchor" in stub
    assert "momentum" in stub["locked_logic_anchor"]


def test_stub_unknown_family_gets_empty_tunables():
    """Unknown family → no auto-gate tuning until human classifies."""
    from engine.research.discovery.queue_actions import build_mechanism_stub
    entry = {"title": "X"}      # no family
    stub = build_mechanism_stub(entry)
    assert stub["tunable_bindings"] == []


def test_stub_family_case_normalized():
    """Family stored lowercase regardless of input case."""
    from engine.research.discovery.queue_actions import build_mechanism_stub
    entry = {"extraction": {"family_guess": "CARRY"}}
    stub = build_mechanism_stub(entry)
    assert stub["family"] == "carry"
    assert "top_frac" in stub["tunable_bindings"]


# ── validate_binding_changes_against_whitelist ───────────────────────────

def test_validate_allows_whitelisted():
    from engine.research.discovery.queue_actions import (
        validate_binding_changes_against_whitelist,
    )
    yaml_doc = {"tunable_bindings": ["top_frac", "vol_target"]}
    proposed = {"top_frac": 0.1}
    ok, violations = validate_binding_changes_against_whitelist(yaml_doc, proposed)
    assert ok is True
    assert violations == []


def test_validate_catches_non_whitelisted():
    from engine.research.discovery.queue_actions import (
        validate_binding_changes_against_whitelist,
    )
    yaml_doc = {"tunable_bindings": ["top_frac"]}
    proposed = {"top_frac": 0.1, "weighting": "value_weight"}    # weighting not whitelisted
    ok, violations = validate_binding_changes_against_whitelist(yaml_doc, proposed)
    assert ok is False
    assert "weighting" in violations


def test_validate_empty_whitelist_rejects_everything():
    from engine.research.discovery.queue_actions import (
        validate_binding_changes_against_whitelist,
    )
    yaml_doc = {"tunable_bindings": []}
    proposed = {"top_frac": 0.1}
    ok, violations = validate_binding_changes_against_whitelist(yaml_doc, proposed)
    assert ok is False
    assert "top_frac" in violations


def test_validate_no_proposed_changes_passes():
    from engine.research.discovery.queue_actions import (
        validate_binding_changes_against_whitelist,
    )
    yaml_doc = {"tunable_bindings": ["top_frac"]}
    ok, violations = validate_binding_changes_against_whitelist(yaml_doc, {})
    assert ok is True


# ── auto_gate respects whitelist ─────────────────────────────────────────

def test_autogate_ignores_non_whitelisted_bindings(tmp_path, monkeypatch, caplog):
    """Stub with bindings NOT in whitelist → auto_gate ignores them
    (uses template defaults instead) and logs."""
    from engine.research.discovery import auto_gate as ag

    yaml_doc = {
        "id": "test_carry",
        "title": "Test",
        "family": "carry",
        "tunable_bindings": ["top_frac"],       # only top_frac allowed
        "bindings": {
            "top_frac":  0.15,
            "weighting": "value_weight",        # NOT in whitelist
            "vol_target": 0.05,                  # NOT in whitelist
        },
    }
    stub_path = tmp_path / "test_carry.yaml"
    stub_path.write_text(yaml.safe_dump(yaml_doc), encoding="utf-8")

    # Mock run_gate so we can capture effective kwargs
    captured_kwargs = {}
    def _mock_template(*, factor_panel, price_panel, return_panel, **kw):
        captured_kwargs.update(kw)
        # return a stub series matching expected shape
        import pandas as pd
        return pd.Series([0.01] * 100, index=pd.date_range("2010-01-01", periods=100, freq="ME"))

    monkeypatch.setattr(
        "engine.research.templates.TEMPLATES",
        {"factor_quartile": _mock_template},
    )
    from engine.research import pipeline as _p
    monkeypatch.setattr(_p, "run_gate", lambda *a, **kw: {
        "name": "x", "available": True, "verdict": "RED",
        "standalone_sharpe": 0.0,
    })

    import logging
    with caplog.at_level(logging.INFO):
        result = ag.auto_gate(stub_path, write_ledger=False)

    assert result.ok is True
    # top_frac IS in whitelist → should be applied
    assert captured_kwargs.get("top_frac") == 0.15
    # weighting NOT in whitelist → should fall back to default
    assert captured_kwargs.get("weighting") == "equal_weight"
    # vol_target NOT in whitelist → should fall back to default None
    assert captured_kwargs.get("vol_target") is None
    # Log should mention the ignored keys
    assert any("ignored non-whitelisted bindings" in r.message for r in caplog.records)


def test_autogate_whitelist_all_uses_all(tmp_path, monkeypatch):
    """When YAML whitelists ALL keys, auto_gate applies all overrides."""
    from engine.research.discovery import auto_gate as ag

    yaml_doc = {
        "id": "test_x",
        "title": "Test",
        "family": "carry",
        "tunable_bindings": ["top_frac", "vol_target", "cost_bps_per_side"],
        "bindings": {
            "top_frac": 0.05,
            "vol_target": 0.08,
            "cost_bps_per_side": 5.0,
        },
    }
    stub_path = tmp_path / "test_x.yaml"
    stub_path.write_text(yaml.safe_dump(yaml_doc), encoding="utf-8")

    captured = {}
    def _mock_template(*, factor_panel, price_panel, return_panel, **kw):
        captured.update(kw)
        import pandas as pd
        return pd.Series([0.01] * 100, index=pd.date_range("2010-01-01", periods=100, freq="ME"))
    monkeypatch.setattr("engine.research.templates.TEMPLATES",
                          {"factor_quartile": _mock_template})
    from engine.research import pipeline as _p
    monkeypatch.setattr(_p, "run_gate", lambda *a, **kw: {
        "name": "x", "available": True, "verdict": "RED",
        "standalone_sharpe": 0.0,
    })
    result = ag.auto_gate(stub_path, write_ledger=False)
    assert result.ok is True
    assert captured["top_frac"] == 0.05
    assert captured["vol_target"] == 0.08
    assert captured["cost_bps_per_side"] == 5.0
