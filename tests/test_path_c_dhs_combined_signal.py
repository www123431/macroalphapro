"""
tests/test_path_c_dhs_combined_signal.py — Sprint D-4 COMBINED signal tests.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §2.5
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.path_c.dhs_combined_signal import (
    build_combined_panel,
    assign_combined_decile_legs,
)


def _make_pead_row(ticker: str, gvkey: int, q: str, sue: float, mcap: float = 10_000.0):
    return {
        "permno":          1000 + gvkey,
        "ticker":          ticker,
        "gvkey":           gvkey,
        "fiscal_yearq":    q,
        "rdq":             pd.Timestamp(q.replace("Q1", "-03-01").replace("Q2", "-06-01")
                                          .replace("Q3", "-09-01").replace("Q4", "-12-01")).date(),
        "sue":             sue,
        "market_cap_at_q": mcap,
    }


def _make_fin_row(ticker: str, gvkey: int, q: str, nsi: float, acc: float):
    return {
        "ticker":       ticker,
        "gvkey":        gvkey,
        "fiscal_yearq": q,
        "nsi":          nsi,
        "acc_scaled":   acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# build_combined_panel
# ─────────────────────────────────────────────────────────────────────────────

def test_build_combined_panel_inner_join():
    """Only firms with BOTH SUE and FIN should appear (intersection)."""
    pead = pd.DataFrame([
        _make_pead_row("A", 1, "2014Q1", 1.5),
        _make_pead_row("B", 2, "2014Q1", -0.5),
        _make_pead_row("C", 3, "2014Q1", 0.0),
    ])
    fin = pd.DataFrame([
        _make_fin_row("A", 1, "2014Q1", 0.05, 0.02),
        _make_fin_row("B", 2, "2014Q1", -0.10, -0.03),
        # C has SUE but no FIN — should be excluded
    ])
    joined = build_combined_panel(pead, fin)
    assert len(joined) == 2  # only A + B (intersection)
    assert set(joined["ticker"]) == {"A", "B"}


def test_build_combined_panel_empty_input():
    pead = pd.DataFrame({"permno": [], "ticker": [], "gvkey": [], "fiscal_yearq": [],
                         "rdq": [], "sue": [], "market_cap_at_q": []})
    fin = pd.DataFrame({"ticker": [], "gvkey": [], "fiscal_yearq": [],
                        "nsi": [], "acc_scaled": []})
    assert build_combined_panel(pead, fin).empty


def test_build_combined_panel_dedupes_restated_rows():
    """If fundq has a restated row for same (gvkey, fiscal_yearq), dedupe keeps first
    (deterministic by rdq ASC). Tests A2 fix: real-data robustness."""
    pead = pd.DataFrame([
        # firm A has TWO rows for 2014Q1 — restatement scenario
        _make_pead_row("A", 1, "2014Q1", 1.5),
        {**_make_pead_row("A", 1, "2014Q1", 1.8),
         "rdq": pd.Timestamp("2014-05-15").date()},   # later rdq → restatement
        _make_pead_row("B", 2, "2014Q1", -0.5),
    ])
    fin = pd.DataFrame([
        _make_fin_row("A", 1, "2014Q1", 0.05, 0.02),
        _make_fin_row("B", 2, "2014Q1", -0.10, -0.03),
    ])
    joined = build_combined_panel(pead, fin)
    # Should not raise on duplicate gvkey-fiscal_yearq; dedupes deterministically
    assert len(joined) == 2  # A + B, no duplicate
    a_row = joined[joined["ticker"] == "A"]
    # Kept earliest rdq (sort ASC, keep first)
    assert a_row["sue"].iloc[0] == 1.5


def test_build_combined_panel_keys_match_across_quarters():
    """Same firm in different quarters should join correctly."""
    pead = pd.DataFrame([
        _make_pead_row("A", 1, "2014Q1", 1.0),
        _make_pead_row("A", 1, "2014Q2", -1.0),
    ])
    fin = pd.DataFrame([
        _make_fin_row("A", 1, "2014Q1", 0.05, 0.01),
        _make_fin_row("A", 1, "2014Q2", -0.05, -0.01),
    ])
    joined = build_combined_panel(pead, fin)
    assert len(joined) == 2
    assert set(joined["fiscal_yearq"]) == {"2014Q1", "2014Q2"}


# ─────────────────────────────────────────────────────────────────────────────
# assign_combined_decile_legs
# ─────────────────────────────────────────────────────────────────────────────

def test_assign_combined_decile_legs_basic():
    """20 firms in 1 quarter; assignment produces long + short + flat."""
    n = 20
    rng = np.random.default_rng(123)
    pead_rows = [_make_pead_row(f"T{i:02d}", i, "2014Q1", float(rng.normal(0, 2.0)))
                 for i in range(n)]
    fin_rows = [_make_fin_row(f"T{i:02d}", i, "2014Q1",
                              float(rng.normal(0, 0.05)),
                              float(rng.normal(0, 0.02)))
                for i in range(n)]
    out = assign_combined_decile_legs(pd.DataFrame(pead_rows), pd.DataFrame(fin_rows))
    assert "leg" in out.columns
    assert "combined" in out.columns
    assert "r_pead" in out.columns
    assert "r_fin" in out.columns
    leg_counts = out["leg"].value_counts()
    assert leg_counts.get("long", 0) >= 1
    assert leg_counts.get("short", 0) >= 1


def test_assign_combined_decile_legs_directionality():
    """Firm with highest SUE + lowest NSI/ACC (bullish on BOTH) should be long.
    Firm with lowest SUE + highest NSI/ACC (bearish on BOTH) should be short."""
    n = 20
    sue_vals = np.linspace(-3.0, +3.0, n)        # T00 lowest sue, T19 highest sue
    nsi_vals = np.linspace(+0.3, -0.3, n)        # T00 high nsi, T19 low nsi
    acc_vals = np.linspace(+0.1, -0.1, n)        # T00 high acc, T19 low acc

    pead_rows = [_make_pead_row(f"T{i:02d}", i, "2014Q1", sue_vals[i]) for i in range(n)]
    fin_rows  = [_make_fin_row(f"T{i:02d}", i, "2014Q1", nsi_vals[i], acc_vals[i])
                 for i in range(n)]
    out = assign_combined_decile_legs(pd.DataFrame(pead_rows), pd.DataFrame(fin_rows))

    # T19: high SUE (bullish), low NSI+ACC (bullish) → should be LONG
    # T00: low SUE (bearish), high NSI+ACC (bearish) → should be SHORT
    leg19 = out.loc[out["ticker"] == "T19", "leg"].iloc[0]
    leg00 = out.loc[out["ticker"] == "T00", "leg"].iloc[0]
    assert leg19 == "long"
    assert leg00 == "short"


def test_assign_combined_decile_legs_empty():
    pead = pd.DataFrame(columns=["permno", "ticker", "gvkey", "fiscal_yearq",
                                  "rdq", "sue", "market_cap_at_q"])
    fin = pd.DataFrame(columns=["ticker", "gvkey", "fiscal_yearq", "nsi", "acc_scaled"])
    out = assign_combined_decile_legs(pead, fin)
    assert "leg" in out.columns
    assert out.empty


def test_assign_combined_decile_legs_separates_quarters():
    """Same firms in 2 quarters; rank within each independently."""
    n = 20
    rng = np.random.default_rng(7)
    pead_rows = []
    fin_rows = []
    for i in range(n):
        for q in ("2014Q1", "2014Q2"):
            pead_rows.append(_make_pead_row(f"T{i:02d}", i, q, float(rng.normal(0, 2.0))))
            fin_rows.append(_make_fin_row(f"T{i:02d}", i, q,
                                           float(rng.normal(0, 0.05)),
                                           float(rng.normal(0, 0.02))))
    out = assign_combined_decile_legs(pd.DataFrame(pead_rows), pd.DataFrame(fin_rows))
    for q in ("2014Q1", "2014Q2"):
        sub = out[out["fiscal_yearq"] == q]
        # Each quarter should have its own L/S
        assert sub["leg"].nunique() >= 2


def test_assign_combined_decile_legs_rpead_rfin_in_range():
    """r_pead, r_fin should be in [0, 1] (percentile)."""
    n = 20
    rng = np.random.default_rng(42)
    pead_rows = [_make_pead_row(f"T{i:02d}", i, "2014Q1", float(rng.normal(0, 2.0)))
                 for i in range(n)]
    fin_rows  = [_make_fin_row(f"T{i:02d}", i, "2014Q1",
                                float(rng.normal(0, 0.05)),
                                float(rng.normal(0, 0.02)))
                 for i in range(n)]
    out = assign_combined_decile_legs(pd.DataFrame(pead_rows), pd.DataFrame(fin_rows))
    assert (out["r_pead"] >= 0.0).all() and (out["r_pead"] <= 1.0).all()
    assert (out["r_fin"]  >= 0.0).all() and (out["r_fin"]  <= 1.0).all()
    np.testing.assert_allclose(out["combined"].values,
                                ((out["r_pead"] + out["r_fin"]) / 2.0).values,
                                rtol=1e-9)
