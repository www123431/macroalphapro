"""Tests for engine.research.library_cost_model_audit."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.research import library_cost_model_audit as lcm


def _write_yaml(tmp_dir: Path, name: str, body: str) -> Path:
    fp = tmp_dir / name
    fp.write_text(body, encoding="utf-8")
    return fp


# ── _check_one ───────────────────────────────────────────────────────────

def test_check_missing_block():
    fp = Path("/tmp/x.yaml")
    res = lcm._check_one(fp, {"id": "x"})
    assert res["status"] == "MISSING_BLOCK"
    assert not res["pass"]


def test_check_pending_minimal_ok():
    fp = Path("/tmp/x.yaml")
    entry = {
        "cost_model": {
            "audit_status": "pending",
            "audit_priority": "high",
            "current_default_in_gate": {"type": "scalar_bps", "bps_per_side": 12},
        }
    }
    res = lcm._check_one(fp, entry)
    assert res["pass"] is True
    assert res["audit_status"] == "pending"
    assert res["audit_priority"] == "high"


def test_check_pending_missing_priority():
    fp = Path("/tmp/x.yaml")
    entry = {
        "cost_model": {
            "audit_status": "pending",
            "current_default_in_gate": {"type": "scalar_bps", "bps_per_side": 12},
        }
    }
    res = lcm._check_one(fp, entry)
    assert res["pass"] is False
    assert "audit_priority" in res["missing"]


def test_check_audited_complete():
    fp = Path("/tmp/x.yaml")
    entry = {
        "cost_model": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc123",
            "type": "almgren_chriss",
            "half_spread_bps": 5.0,
            "impact_coef": 0.5,
            "daily_sigma_estimate": 0.015,
            "universe_median_adv_usd": 50_000_000,
            "monthly_turnover_estimate": 0.40,
            "stress_multiplier": 2.5,
            "rationale": "Russell-1000 universe per FIM 2015 + Almgren 2005, 50+ chars",
            "multi_aum_sharpe_sleeve": {"at_10M": 1.1, "at_100M": 1.05, "at_1B": 1.0},
            "capacity": {"hard_capacity_usd": 1_900_000_000},
        }
    }
    res = lcm._check_one(fp, entry)
    assert res["pass"] is True


def test_check_audited_missing_capacity():
    fp = Path("/tmp/x.yaml")
    entry = {
        "cost_model": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "type": "almgren_chriss",
            "half_spread_bps": 5.0,
            "impact_coef": 0.5,
            "daily_sigma_estimate": 0.015,
            "universe_median_adv_usd": 50_000_000,
            "monthly_turnover_estimate": 0.40,
            "stress_multiplier": 2.5,
            "rationale": "x" * 60,
            "multi_aum_sharpe_sleeve": {"at_100M": 1.0},
            # capacity missing
        }
    }
    res = lcm._check_one(fp, entry)
    assert res["pass"] is False
    assert "capacity" in res["missing"]


def test_check_audited_rationale_too_short():
    fp = Path("/tmp/x.yaml")
    entry = {
        "cost_model": {
            "audit_status": "audited",
            "audit_date": "2026-05-30",
            "audit_script": "scripts/x.py",
            "audit_commit": "abc",
            "type": "almgren_chriss",
            "half_spread_bps": 5.0,
            "impact_coef": 0.5,
            "daily_sigma_estimate": 0.015,
            "universe_median_adv_usd": 50_000_000,
            "monthly_turnover_estimate": 0.40,
            "stress_multiplier": 2.5,
            "rationale": "too short",   # < 50 chars
            "multi_aum_sharpe_sleeve": {"at_100M": 1.0},
            "capacity": {"hard_capacity_usd": 1_000_000_000},
        }
    }
    res = lcm._check_one(fp, entry)
    assert res["pass"] is False
    assert any("rationale" in m for m in res["missing"])


def test_check_invalid_audit_status():
    fp = Path("/tmp/x.yaml")
    entry = {"cost_model": {"audit_status": "made_up"}}
    res = lcm._check_one(fp, entry)
    assert res["status"] == "INVALID_AUDIT_STATUS"
    assert not res["pass"]


# ── audit_library on real library ────────────────────────────────────────

def test_audit_library_real():
    """Real library should currently have all blocks present (1 audited + 8 pending)."""
    summary = lcm.audit_library()
    assert summary["total"] >= 9
    assert summary["missing_block"] == 0
    assert summary["audited"] >= 1
    # All entries must validate
    failed = [r for r in summary["results"] if not r["pass"]]
    assert not failed, f"library entries failing cost_model audit: {failed}"


def test_audit_library_post_earnings_drift_is_audited():
    """post_earnings_drift.yaml is the canonical audited example."""
    summary = lcm.audit_library()
    by_name = {r["name"]: r for r in summary["results"]}
    assert "post_earnings_drift" in by_name
    assert by_name["post_earnings_drift"]["audit_status"] == "audited"
    assert by_name["post_earnings_drift"]["pass"] is True


def test_pending_highs_flagged():
    """cross_asset_carry + time_series_momentum are DEPLOYED → must be high priority."""
    summary = lcm.audit_library()
    by_name = {r["name"]: r for r in summary["results"]}
    for nm in ("cross_asset_carry", "time_series_momentum"):
        if nm in by_name and by_name[nm]["audit_status"] == "pending":
            assert by_name[nm]["audit_priority"] == "high", \
                f"{nm} is DEPLOYED — pending audit must be high priority"
