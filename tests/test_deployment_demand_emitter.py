"""Tests for engine.research.deployment_demand_emitter (2026-06-17).

Coverage:
  - Direction → family rules fire for canonical phrases
  - Idempotent re-emission (signature dedup against existing rows)
  - Multiple families per sleeve OK
  - Sleeve with no direction emits 0 demands
  - Real active_deployment.yaml round-trip (smoke)
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest
import yaml

from engine.research.deployment_demand_emitter import (
    DeploymentDemand, _direction_to_families, _signature,
    emit_deployment_demand, parse_active_deployment,
)


SYNTH_YAML = """\
active_config_id: test_synth_config
configs:
  - id: test_synth_config
    deploy_date: "2026-06-17"
    label: "synthetic test config"
    expected_stats: { sharpe: 1.0, ann_ret: 0.10, ann_vol: 0.10, max_dd: -0.07,
                       backtest_window: "2014-09-12 to 2023-12-22" }
    sleeves:
      - name: equity_book
        role: alpha
        base_weight: 0.63
        builder: x.build_equity_book
        signing_spec_ids: [4, 12]
        research_keywords:
          - "post-earnings announcement drift"
          - "PEAD"
        improvement_directions:
          - "low-vol BAB variants"
          - "PIT FF12 within-sector ranking variants"
      - name: cross_asset_carry
        role: alpha
        base_weight: 0.25
        builder: x.build_carry_book
        signing_spec_ids: [77]
        research_keywords:
          - "FX carry"
          - "Koijen carry"
        improvement_directions:
          - "G10 rate carry expansion"
          - "commodity convenience yield expansion"
      - name: cross_asset_tsmom
        role: alpha
        base_weight: 0.05
        builder: x.build_tsmom_book
        signing_spec_ids: [77]
        research_keywords:
          - "time-series momentum"
          - "TSMOM"
        improvement_directions:
          - "fast vs slow trend blend"
      - name: no_improvement_sleeve
        role: diversifier
        base_weight: 0.02
        builder: x.build
        signing_spec_ids: [80]
        research_keywords: []
        improvement_directions: []
"""


def test_direction_to_families_low_vol():
    fams = _direction_to_families("low-vol BAB variants", [])
    fam_set = {f for f, _ in fams}
    assert "LOW_VOL" in fam_set


def test_direction_to_families_carry_via_keywords():
    # The direction itself doesn't say "carry"; the keyword bag carries the signal.
    fams = _direction_to_families("G10 rate carry expansion", ["FX carry"])
    fam_set = {f for f, _ in fams}
    assert "CARRY" in fam_set


def test_direction_to_families_tsmom():
    fams = _direction_to_families("fast vs slow trend blend",
                                    ["time-series momentum", "TSMOM"])
    fam_set = {f for f, _ in fams}
    assert "CROSS_ASSET_MOMENTUM" in fam_set


def test_direction_to_families_pead_within_equity():
    fams = _direction_to_families("sector-neutralization mechanisms",
                                    ["post-earnings announcement drift", "PEAD"])
    fam_set = {f for f, _ in fams}
    assert "EARNINGS_DRIFT" in fam_set


def test_direction_to_families_empty_when_no_match():
    fams = _direction_to_families("regime classifier refinements", [])
    # 'regime classifier' is not a canonical mechanism_family; emitter
    # SHOULD ignore (no row emitted) rather than misclassify.
    assert fams == []


def test_direction_to_families_no_false_positive_flight_to_quality():
    # Regression test 2026-06-17: the bare \bquality\b regex in
    # PROFITABILITY rule was matching "flight to quality" (a
    # crisis-hedge keyword) and producing spurious PROFITABILITY demand
    # rows. The rule now requires a factor-y qualifier (quality factor /
    # quality minus junk / QMJ / RMW / novy-marx) before firing.
    fams = _direction_to_families("VIX regime classifier refinements",
                                    ["crisis alpha", "gold flight to quality",
                                     "safe haven"])
    fam_set = {f for f, _ in fams}
    assert "PROFITABILITY" not in fam_set


def test_direction_to_families_quality_factor_does_fire():
    # Make sure tightening the QUALITY rule didn't break legitimate
    # quality-factor matches.
    fams = _direction_to_families("explore QMJ quality factor variants", [])
    fam_set = {f for f, _ in fams}
    assert "PROFITABILITY" in fam_set


def test_signature_is_deterministic_and_collision_resistant():
    s1 = _signature("equity_book", "LOW_VOL", "low-vol BAB variants")
    s2 = _signature("equity_book", "LOW_VOL", "low-vol BAB variants")
    s3 = _signature("equity_book", "LOW_VOL", "min-variance variants")
    assert s1 == s2
    assert s1 != s3
    assert s1.startswith("DEPLOYMENT_DEMAND::equity_book::LOW_VOL::")


def test_parse_active_deployment_synth(tmp_path):
    yaml_path = tmp_path / "active_deployment.yaml"
    yaml_path.write_text(SYNTH_YAML, encoding="utf-8")

    cfg_info, demands = parse_active_deployment(yaml_path=yaml_path)
    assert cfg_info["active_config_id"] == "test_synth_config"

    # Expected matches:
    # equity_book × "low-vol BAB variants"        → LOW_VOL  ✓
    # equity_book × "PIT FF12 within-sector..."   → EARNINGS_DRIFT via keywords ✓
    # cross_asset_carry × "G10 rate carry"       → CARRY ✓
    # cross_asset_carry × "commodity convenience" → CARRY (same family, dup-dropped)
    # cross_asset_tsmom × "fast vs slow trend"   → CROSS_ASSET_MOMENTUM ✓
    # no_improvement_sleeve × []                  → none
    fams_by_sleeve = {(d.sleeve_name, d.family) for d in demands}
    assert ("equity_book",        "LOW_VOL")              in fams_by_sleeve
    assert ("cross_asset_carry",  "CARRY")                in fams_by_sleeve
    assert ("cross_asset_tsmom",  "CROSS_ASSET_MOMENTUM") in fams_by_sleeve
    # No-improvement sleeve emits nothing
    no_imp = [d for d in demands if d.sleeve_name == "no_improvement_sleeve"]
    assert no_imp == []


def test_emit_writes_jsonl_and_is_idempotent(tmp_path):
    yaml_path = tmp_path / "active_deployment.yaml"
    yaml_path.write_text(SYNTH_YAML, encoding="utf-8")
    gaps_path = tmp_path / "capability_gaps.jsonl"

    now = _dt.datetime(2026, 6, 17, 12, 0, tzinfo=_dt.timezone.utc)
    result1 = emit_deployment_demand(
        yaml_path=yaml_path, gaps_path=gaps_path,
        dry_run=False, now=now,
    )
    n1 = result1["written"]
    assert n1 > 0
    rows1 = [json.loads(ln) for ln in gaps_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows1) == n1
    for r in rows1:
        assert r["gap_class"]  == "DEPLOYMENT_DEMAND"
        assert r["source"]     == "deployment_improvement"
        assert r["family"]     != ""
        assert r["signature"].startswith("DEPLOYMENT_DEMAND::")

    # Re-emit: nothing new should be written
    result2 = emit_deployment_demand(
        yaml_path=yaml_path, gaps_path=gaps_path,
        dry_run=False, now=now,
    )
    assert result2["written"]         == 0
    assert result2["already_present"] == n1
    rows2 = [json.loads(ln) for ln in gaps_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows2) == n1   # no duplicates appended


def test_real_active_deployment_yaml_smoke():
    """Smoke test against the real deployment yaml — verify it parses and
    produces non-empty demands across the expected families.
    """
    repo_root = Path(__file__).resolve().parents[1]
    yaml_path = repo_root / "data" / "portfolio" / "active_deployment.yaml"
    if not yaml_path.is_file():
        pytest.skip("real active_deployment.yaml not present in this checkout")
    cfg_info, demands = parse_active_deployment(yaml_path=yaml_path)
    families = {d.family for d in demands}
    # At minimum: equity book should yield EARNINGS_DRIFT or ANALYST_REVISION
    # (via PEAD / analyst revision keywords) and the carry sleeve CARRY.
    assert any(f in families for f in ("EARNINGS_DRIFT", "ANALYST_REVISION")), (
        f"expected at least one EARNINGS_DRIFT/ANALYST_REVISION demand from "
        f"equity_book, got families={families}"
    )
    assert "CARRY" in families
