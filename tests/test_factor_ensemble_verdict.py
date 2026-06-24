"""
tests/test_factor_ensemble_verdict.py — Sprint Week 4 (spec id=50 §4.5).

Pre-registration: docs/spec_factor_ensemble_v1.md §4.5 amendment 2026-05-09
(pre-Sprint-Week-4 audit Issue #4 + Nit #5).

Verifies:
  • compute_verdict signature has NO date parameters (Nit #5).
  • Mandatory JSON schema fields present + non-null (Issue #4).
  • Decision rule applies thresholds correctly (spec §3.2).
  • Underpowered (n<12) → WITHDRAW.
  • build_verdict_json_payload raises ValueError on schema violation.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from engine import factor_ensemble_verdict as fev


# ─────────────────────────────────────────────────────────────────────────────
# 1. compute_verdict NO date parameters (Nit #5)
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_verdict_signature_has_no_date_params():
    """Spec §4.5 amendment Nit #5: compute_verdict MUST NOT accept date params."""
    sig = inspect.signature(fev.compute_verdict)
    forbidden = {"start_date", "end_date", "oos_start_date", "default_end_date"}
    for fld in sig.parameters:
        assert fld not in forbidden, (
            f"compute_verdict({fld}=...) parameter forbidden per spec §4.5 Nit #5 "
            "(prevents HARKing R3 silent window-shifting)"
        )


def test_compute_verdict_rejects_positional_date_args():
    """Calling compute_verdict with positional date args must raise TypeError."""
    import datetime as _dt
    with pytest.raises(TypeError):
        fev.compute_verdict(_dt.date(2011, 1, 1))  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 2. Decision rule (spec §3.2)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delta,ci_lo,ci_hi,n,expected", [
    # DESCRIPTIVE_POSITIVE: magnitude ≥ 0.20 AND ci_lower > 0
    (0.30, 0.05, 0.55, 168, "DESCRIPTIVE_POSITIVE"),
    # Below threshold magnitude but CI fully positive
    (0.10, 0.02, 0.18, 168, "DESCRIPTIVE_INSUFFICIENT_SMALL_EFFECT"),
    # Positive direction but CI crosses zero
    (0.15, -0.10, 0.40, 168, "DESCRIPTIVE_INSUFFICIENT_POSITIVE_DIRECTION"),
    # Negative + CI fully negative
    (-0.25, -0.45, -0.05, 168, "DESCRIPTIVE_NEGATIVE"),
    # Underpowered (n<12)
    (0.30, 0.05, 0.55, 6,   "WITHDRAW"),
    # NaN delta → WITHDRAW
    (float("nan"), 0.0, 0.5, 168, "WITHDRAW"),
])
def test_decide_label(delta, ci_lo, ci_hi, n, expected):
    label = fev._decide(delta, ci_lo, ci_hi, n)
    assert label == expected, f"_decide({delta}, [{ci_lo},{ci_hi}], n={n}) = {label} != {expected}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. JSON schema lock (Issue #4) — mandatory fields + types
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(decision="DESCRIPTIVE_POSITIVE"):
    return fev.VerdictResult(
        decision_label=decision,
        delta_sharpe_walk_forward=0.30,
        ci_lower_95=0.05,
        ci_upper_95=0.55,
        memmel_z=2.10,
        n_oos_months=168,
        paired_corr=0.65,
        ensemble_sharpe=1.10,
        baseline_sharpe=0.80,
        spec_hash="abc123",
        harness_ensemble_only_baseline_consistency="PASS",
        completed_at="2026-05-09T00:00:00Z",
    )


def test_build_verdict_json_payload_schema_complete():
    payload = fev.build_verdict_json_payload(_make_result())
    for fld in fev._REQUIRED_VERDICT_FIELDS:
        assert fld in payload, f"missing required field {fld}"
        assert payload[fld] is not None, f"required field {fld} is null"
    # Locked semantics
    assert payload["verdict_layer"] == "walk_forward_signal_only"
    assert payload["production_forward_required_for_swap"] is True
    assert "PendingApproval" in payload["interpretation_caveat"]


def test_build_verdict_json_payload_invalid_decision_label_rejected():
    bad = _make_result(decision="MAGIC_OK")
    with pytest.raises(ValueError, match="decision_label"):
        fev.build_verdict_json_payload(bad)


def test_build_verdict_json_payload_threshold_constants_locked():
    payload = fev.build_verdict_json_payload(_make_result())
    assert payload["delta_sharpe_threshold_locked"] == 0.20  # spec §3.2
    assert payload["bootstrap_resamples_locked"] == 1000     # spec §3.1


# ─────────────────────────────────────────────────────────────────────────────
# 4. compute_verdict integration (mocked walk-forward) — schema enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_verdict_persisted_json_round_trips_schema(tmp_path, monkeypatch):
    """End-to-end: compute_verdict with mocked harness produces JSON file
    matching schema lock (when persist=True)."""
    from engine import factor_ensemble_walk_forward as wf_mod

    # Mock walk-forward returns
    idx = pd.date_range("2011-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(42)
    ens_returns = pd.Series(rng.normal(0.012, 0.04, 24), index=idx)
    base_returns = pd.Series(rng.normal(0.008, 0.04, 24), index=idx)

    fake_ens = wf_mod.WalkForwardResult(
        n_periods=24, monthly_returns=ens_returns,
        cumulative_return=ens_returns.add(1).prod() - 1,
        annualized_sharpe=1.0, annualized_vol=0.10, max_drawdown=-0.05,
        n_etfs_per_period=pd.Series(),
        gross_exposure=pd.Series(),
    )
    fake_base = wf_mod.WalkForwardResult(
        n_periods=24, monthly_returns=base_returns,
        cumulative_return=base_returns.add(1).prod() - 1,
        annualized_sharpe=0.7, annualized_vol=0.10, max_drawdown=-0.05,
        n_etfs_per_period=pd.Series(),
        gross_exposure=pd.Series(),
    )
    monkeypatch.setattr(
        "engine.factor_ensemble_verdict.run_walk_forward",
        lambda **kw: fake_ens if not kw.get("baseline_only") else fake_base,
        raising=False,
    )
    # Re-export — real import is inside compute_verdict, patch the source module too
    monkeypatch.setattr("engine.factor_ensemble_walk_forward.run_walk_forward",
                        lambda **kw: fake_ens if not kw.get("baseline_only") else fake_base)

    # Redirect output dir
    monkeypatch.setattr(fev, "_VERDICT_JSON", tmp_path / "v1_verdict.json")
    monkeypatch.setattr(fev, "_VERDICT_TXT", tmp_path / "v1_verdict.txt")
    monkeypatch.setattr(fev, "_DATA_DIR", tmp_path)

    result = fev.compute_verdict(use_cache=False, persist=True)

    # Persisted files exist
    assert (tmp_path / "v1_verdict.json").exists()
    assert (tmp_path / "v1_verdict.txt").exists()

    # JSON round-trip — schema fields present + non-null
    payload = json.loads((tmp_path / "v1_verdict.json").read_text(encoding="utf-8"))
    for fld in fev._REQUIRED_VERDICT_FIELDS:
        assert fld in payload and payload[fld] is not None, f"missing/null {fld}"
    # n_oos_months must equal common-date count
    assert payload["n_oos_months"] == 24
    # Decision label must be one of the locked options
    assert payload["decision_label"] in fev._DECISION_LABELS
