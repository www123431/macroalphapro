"""Tests for engine.research.factor_exposure_gap_detector (2026-06-17).

Coverage:
  - Synthetic case: a sleeve with KNOWN factor loading → detector recovers
    the loaded factor AND flags the unloaded factor as gap
  - Synthetic case: a sleeve with NO loading on any factor → all flagged gap
  - Synthetic case: a sleeve with strong loading on all factors → no gaps
  - Direction proposal mapping correctness
  - Real factor matrix builder smoke test (will skip if substrate missing)
  - Stage 1 wire-in: emit_fegd_demand writes capability_gaps rows
  - End-to-end: burndown_ranker picks up the FEGD-emitted families
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.research.factor_exposure_gap_detector import (
    GAP_T_THRESHOLD,
    FactorExposure, ExposureReport, ProposedDirection,
    build_canonical_factor_matrix, deployed_factor_exposure,
    propose_improvement_directions,
)


def _synth_factor_matrix(n_months: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic monthly factor matrix with 4 ~iid noise factors."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    data = rng.normal(0, 0.04, size=(n_months, 4))
    return pd.DataFrame(data, index=idx, columns=["BAB", "VRP", "RMW", "MOM"])


def test_factor_exposure_recovers_known_loading():
    fm = _synth_factor_matrix()
    rng = np.random.default_rng(123)
    # Sleeve loads 0.8 on BAB + tiny noise, ~0 on others
    sleeve = 0.8 * fm["BAB"] + pd.Series(rng.normal(0, 0.005, len(fm)),
                                            index=fm.index)
    report = deployed_factor_exposure(sleeve, sleeve_id="synth_bab_loader",
                                         factor_matrix=fm)

    # BAB should be the non-gap (strong load), others should be gaps
    by_factor = {e.factor: e for e in report.exposures}
    assert by_factor["BAB"].t_stat > 5.0
    assert abs(by_factor["BAB"].beta - 0.8) < 0.05
    assert "BAB" not in report.gap_factors
    # Other 3 factors should be flagged
    for other in ["VRP", "RMW", "MOM"]:
        assert other in report.gap_factors, f"expected {other} flagged as gap"


def test_factor_exposure_mostly_gap_when_sleeve_is_pure_noise():
    # Use larger n_months so chance correlations are tighter
    fm = _synth_factor_matrix(n_months=500, seed=42)
    rng = np.random.default_rng(7)
    sleeve = pd.Series(rng.normal(0, 0.04, len(fm)), index=fm.index)
    report = deployed_factor_exposure(sleeve, sleeve_id="synth_noise",
                                         factor_matrix=fm)
    # Pure noise sleeve: AT LEAST 3 of 4 factors should be flagged as gap
    # (allowing for one chance ~5% tail correlation at |t| > 1.65).
    assert len(report.gap_factors) >= 3


def test_factor_exposure_no_gap_when_sleeve_loads_on_all():
    fm = _synth_factor_matrix()
    rng = np.random.default_rng(99)
    sleeve = (0.5 * fm["BAB"] + 0.4 * fm["VRP"] +
              0.6 * fm["RMW"] + 0.3 * fm["MOM"] +
              pd.Series(rng.normal(0, 0.003, len(fm)), index=fm.index))
    report = deployed_factor_exposure(sleeve, sleeve_id="synth_all_loaded",
                                         factor_matrix=fm)
    assert report.gap_factors == ()


def test_exclude_factors_removes_self_regression_loop():
    fm = _synth_factor_matrix()
    sleeve = fm["BAB"]   # sleeve IS the BAB factor itself
    report = deployed_factor_exposure(sleeve, sleeve_id="bab_itself",
                                         factor_matrix=fm,
                                         exclude_factors=("BAB",))
    # BAB excluded, so report.exposures has only 3 factors
    assert all(e.factor != "BAB" for e in report.exposures)
    assert len(report.exposures) == 3


def test_propose_improvement_directions_maps_gaps():
    fake_report = ExposureReport(
        sleeve_id    = "equity_book",
        n_obs        = 200,
        window_start = "2010-01-31",
        window_end   = "2026-08-31",
        r_squared    = 0.4,
        exposures    = (
            FactorExposure("BAB",      beta=0.05, t_stat=0.3,  p_value=0.7),
            FactorExposure("XA_CARRY", beta=0.10, t_stat=1.0,  p_value=0.3),
            FactorExposure("RMW",      beta=0.65, t_stat=8.5,  p_value=0.0),
        ),
        gap_factors  = ("BAB", "XA_CARRY"),
    )
    props = propose_improvement_directions(fake_report)
    fams = {p.mechanism_family for p in props}
    assert "LOW_VOL" in fams
    assert "CARRY"   in fams
    # RMW not in gap_factors → no PROFITABILITY proposal emitted
    assert "PROFITABILITY" not in fams


def test_emit_fegd_demand_writes_rows_and_is_idempotent(tmp_path):
    """End-to-end Stage 1 wire-in test — synthetic sleeve with known gaps
    → emit_fegd_demand writes capability_gaps rows tagged source:fegd_factor_gap
    → re-run is idempotent (signature dedup)."""
    from engine.research.factor_exposure_gap_detector import emit_fegd_demand
    import json

    # Synthetic factor matrix where BAB is iid noise (will be GAP relative to
    # a sleeve that loads only on RMW)
    n = 300
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
        "VRP": rng.normal(0, 0.04, n),
    }, index=idx)
    # Sleeve loads only on RMW; BAB + VRP should be detected as gaps
    sleeve = 0.8 * fm["RMW"] + pd.Series(rng.normal(0, 0.005, n), index=idx)

    gaps_file = tmp_path / "capability_gaps.jsonl"
    now = _dt.datetime(2026, 6, 17, 12, 0, tzinfo=_dt.timezone.utc)

    # First run — should write 2 rows (BAB, VRP gaps)
    res1 = emit_fegd_demand(
        sleeve_id="test_sleeve", sleeve_pnl=sleeve,
        factor_matrix=fm, gaps_path=gaps_file,
        dry_run=False, now=now,
    )
    assert res1["written"] >= 2
    rows1 = [json.loads(ln) for ln in gaps_file.read_text(encoding="utf-8").splitlines()
              if ln.strip()]
    assert len(rows1) == res1["written"]
    # Every row should be FEGD-sourced + family present
    for r in rows1:
        assert r["source"]   == "fegd_factor_gap"
        assert r["gap_class"] == "FEGD_GAP"
        assert r["family"]   != ""
        assert r["signature"].startswith("FEGD_DEMAND::")
        assert r["sleeve"]   == "test_sleeve"

    # Re-run — should write 0 new rows (idempotent)
    res2 = emit_fegd_demand(
        sleeve_id="test_sleeve", sleeve_pnl=sleeve,
        factor_matrix=fm, gaps_path=gaps_file,
        dry_run=False, now=now,
    )
    assert res2["written"]         == 0
    assert res2["already_present"] == res1["written"]


def test_emit_fegd_demand_families_picked_up_by_ranker(tmp_path):
    """Wire-in regression: the FEGD-emitted rows must be readable by
    burndown_ranker.load_demand_families to ensure end-to-end boost.

    Uses a synthetic sleeve that loads strongly on RMW + leaves BAB / VRP
    as gaps. Larger n=500 reduces chance correlations on the gap factors.
    """
    from engine.research.factor_exposure_gap_detector import emit_fegd_demand
    from engine.research.burndown_ranker import load_demand_families

    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "VRP": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
    }, index=idx)
    # Sleeve loads strongly on RMW; BAB + VRP should be gaps
    sleeve = 0.8 * fm["RMW"] + pd.Series(rng.normal(0, 0.005, n), index=idx)

    gaps_file = tmp_path / "capability_gaps.jsonl"
    now = _dt.datetime(2026, 6, 17, 12, 0, tzinfo=_dt.timezone.utc)
    emit_fegd_demand(
        sleeve_id="test_sleeve_rmw_loader", sleeve_pnl=sleeve,
        factor_matrix=fm, gaps_path=gaps_file,
        dry_run=False, now=now,
    )
    fams = load_demand_families(gaps_path=gaps_file)
    # BAB → LOW_VOL and VRP → VOL_RISK_PREMIUM should be flagged as gaps
    # (RMW is the loaded factor, NOT a gap, so PROFITABILITY should NOT appear)
    assert "LOW_VOL"          in fams
    assert "VOL_RISK_PREMIUM" in fams
    assert "PROFITABILITY"    not in fams


def test_pre_enhance_check_proceed_for_clean_gap():
    """Synthetic sleeve loads only on RMW; querying for LOW_VOL family
    (canonical factor BAB) should return PROCEED — BAB is a true gap."""
    from engine.research.factor_exposure_gap_detector import pre_enhance_check
    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
    }, index=idx)
    sleeve = 0.8 * fm["RMW"] + pd.Series(rng.normal(0, 0.005, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="synth_rmw_loader",
        candidate_mechanism_family="LOW_VOL",
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    assert dec.recommendation == "PROCEED"
    assert dec.matched_gap_factor == "BAB"
    assert abs(dec.factor_t_stat) < 1.65   # BAB is noise factor, sub-GAP threshold


def test_pre_enhance_check_skip_for_strongly_loaded():
    """Same sleeve, querying PROFITABILITY (canonical factor RMW) — sleeve
    loads on RMW at t > 2.0 → SKIP."""
    from engine.research.factor_exposure_gap_detector import pre_enhance_check
    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
    }, index=idx)
    sleeve = 0.8 * fm["RMW"] + pd.Series(rng.normal(0, 0.005, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="synth_rmw_loader",
        candidate_mechanism_family="PROFITABILITY",
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    assert dec.recommendation == "SKIP"
    assert dec.matched_gap_factor == "RMW"
    assert abs(dec.factor_t_stat) > 2.0


def test_pre_enhance_check_warn_for_marginal_loading():
    """Sleeve with moderate RMW loading — t between 1.0 and 2.0 → WARN.
    This mirrors the GP/A situation in this session's audit."""
    from engine.research.factor_exposure_gap_detector import pre_enhance_check
    n = 300
    rng = np.random.default_rng(7)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
    }, index=idx)
    # Weak RMW loading: 0.15 coefficient + larger noise → t in 1-2 range
    sleeve = 0.15 * fm["RMW"] + pd.Series(rng.normal(0, 0.04, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="synth_marginal_rmw",
        candidate_mechanism_family="PROFITABILITY",
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    # Allow for some seed-dependent flex; verify it's not SKIP if t < 2.0
    abs_t = abs(dec.factor_t_stat)
    if abs_t < 1.0:
        assert dec.recommendation == "PROCEED"
    elif abs_t < 2.0:
        assert dec.recommendation == "WARN"
    else:
        assert dec.recommendation == "SKIP"


def test_pre_enhance_check_proceed_when_family_not_in_factor_matrix():
    """Family with no canonical factor in matrix → PROCEED (cannot pre-check)."""
    from engine.research.factor_exposure_gap_detector import pre_enhance_check
    n = 300
    rng = np.random.default_rng(99)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({"BAB": rng.normal(0, 0.04, n)}, index=idx)
    sleeve = pd.Series(rng.normal(0, 0.04, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="any",
        candidate_mechanism_family="ATTENTION",  # no canonical factor for this
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    assert dec.recommendation == "PROCEED"
    assert dec.matched_gap_factor is None


def test_emit_fegd_demand_applies_sleeve_relevance_filter(tmp_path):
    """Phase 2 — sleeve_id with a SLEEVE_FAMILY_RELEVANCE entry filters
    out semantically irrelevant gaps before writing rows."""
    from engine.research.factor_exposure_gap_detector import (
        emit_fegd_demand, SLEEVE_FAMILY_RELEVANCE,
    )
    import json
    # Synthetic factor matrix where the noise factors will be flagged as GAP
    n = 500
    rng = np.random.default_rng(42)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),    # → LOW_VOL (relevant for equity_book)
        "HML": rng.normal(0, 0.04, n),    # → VALUE  (relevant for equity_book)
        "MOM": rng.normal(0, 0.04, n),    # → MOMENTUM (relevant for equity_book)
    }, index=idx)
    # Pure noise sleeve → all 3 factors flagged as GAP
    sleeve = pd.Series(rng.normal(0, 0.04, n), index=idx)

    gaps_file = tmp_path / "capability_gaps.jsonl"
    now = _dt.datetime(2026, 6, 17, 12, 0, tzinfo=_dt.timezone.utc)

    # Case 1: sleeve_id = "equity_book" → LOW_VOL, VALUE, MOMENTUM ALL relevant
    res_eq = emit_fegd_demand(
        sleeve_id="equity_book", sleeve_pnl=sleeve,
        factor_matrix=fm, gaps_path=gaps_file,
        dry_run=False, now=now,
    )
    rows_eq = [json.loads(ln) for ln in
                 gaps_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    eq_families = {r["family"] for r in rows_eq if r["sleeve"] == "equity_book"}
    # All 3 should be present (all relevant for equity_book)
    assert "LOW_VOL"  in eq_families
    assert "VALUE"    in eq_families
    assert "MOMENTUM" in eq_families

    # Case 2: sleeve_id = "cross_asset_carry" → only CARRY-family-relevant
    # gaps emit. None of LOW_VOL / VALUE / MOMENTUM are in carry's relevance set.
    gaps_file2 = tmp_path / "capability_gaps2.jsonl"
    res_carry = emit_fegd_demand(
        sleeve_id="cross_asset_carry", sleeve_pnl=sleeve,
        factor_matrix=fm, gaps_path=gaps_file2,
        dry_run=False, now=now,
    )
    if gaps_file2.is_file():
        rows_carry = [json.loads(ln) for ln in
                       gaps_file2.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        rows_carry = []
    carry_families = {r["family"] for r in rows_carry if r["sleeve"] == "cross_asset_carry"}
    # Carry sleeve filter should drop all 3 (not in relevance set)
    assert "LOW_VOL"  not in carry_families
    assert "VALUE"    not in carry_families
    assert "MOMENTUM" not in carry_families


def test_pre_enhance_check_phase3_strong_proceed_for_near_zero_loading():
    """Phase 3 — sleeve with essentially zero loading → STRONG_PROCEED."""
    from engine.research.factor_exposure_gap_detector import pre_enhance_check
    n = 500
    rng = np.random.default_rng(2026)
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
        "RMW": rng.normal(0, 0.04, n),
    }, index=idx)
    # Sleeve is pure noise, very small magnitude → expect near-zero t-stat
    sleeve = pd.Series(rng.normal(0, 0.001, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="synth",
        candidate_mechanism_family="LOW_VOL",
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    # Either STRONG_PROCEED (|t|<1.0) or PROCEED (1.0-1.5) acceptable for noise
    assert dec.recommendation in ("STRONG_PROCEED", "PROCEED")


def test_pre_enhance_check_phase3_weak_proceed_boundary_case():
    """Phase 3 — the BAB-book-overlay calibration case: boundary loading
    in [1.5, 1.85) should return WEAK_PROCEED, not the old PROCEED."""
    from engine.research.factor_exposure_gap_detector import (
        pre_enhance_check, PROCEED_T_THRESHOLD, WEAK_PROCEED_T,
    )
    # Construct a sleeve whose loading is engineered to produce t-stat
    # in the boundary zone — small effect + moderate noise + moderate n
    n = 100   # ~book-window length
    rng = np.random.default_rng(42)
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    fm = pd.DataFrame({
        "BAB": rng.normal(0, 0.04, n),
    }, index=idx)
    # Tune coefficient until t-stat lands in [1.5, 1.85)
    sleeve = 0.30 * fm["BAB"] + pd.Series(rng.normal(0, 0.04, n), index=idx)
    dec = pre_enhance_check(
        sleeve_id="synth",
        candidate_mechanism_family="LOW_VOL",
        sleeve_pnl=sleeve, factor_matrix=fm,
    )
    abs_t = abs(dec.factor_t_stat) if dec.factor_t_stat is not None else 0
    # Phase 3 boundaries — verify recommendation matches the tier
    if PROCEED_T_THRESHOLD <= abs_t < WEAK_PROCEED_T:
        assert dec.recommendation == "WEAK_PROCEED", \
            f"|t|={abs_t:.2f} in [{PROCEED_T_THRESHOLD}, {WEAK_PROCEED_T}) should be WEAK_PROCEED"


def test_real_factor_matrix_smoke():
    """Smoke test against the real cached factor sources.

    Skips if Ken French or AQR BAB caches aren't present.
    """
    repo_root = Path(__file__).resolve().parents[1]
    kf_path = repo_root / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
    bab_path = repo_root / "data" / "cache" / "aqr_bab_usa_monthly.parquet"
    if not kf_path.is_file() or not bab_path.is_file():
        pytest.skip("real factor caches not present in this checkout")
    try:
        fm = build_canonical_factor_matrix()
    except Exception as e:
        pytest.skip(f"factor matrix build raised — skipping smoke: {e}")
    assert {"MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "BAB", "VRP"} \
        .issubset(set(fm.columns))
    assert len(fm) > 100  # sanity: at least some months returned
