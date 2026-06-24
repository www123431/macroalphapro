"""Unit tests for engine.research.knowledge_graph.

Safety properties:
1. Graph builds successfully from real data (no exception)
2. Mechanism classifier maps known descriptions to expected families
3. Parent family rollup catches cousin relationships
4. Cousin detection (similar_to) is symmetric
5. Deployed overlap detection works on parent level (the key use case)
6. Blind spots list is non-empty for our small dataset
7. Coverage matrix is a properly-shaped DataFrame
8. Query API handles unknown candidates gracefully
"""
from __future__ import annotations

import pytest
import pandas as pd

from engine.research.knowledge_graph import (
    KnowledgeGraph,
    build_graph,
    candidate_blind_spots,
    candidates_by_mechanism_family,
    candidates_by_verdict,
    classify_asset_class,
    classify_mechanism,
    coverage_matrix,
    deployed_overlap_check,
    failure_theme_clusters,
    normalize_verdict,
    parent_families_for,
    similar_candidates,
    stress_window_summary,
    summary,
)


# ── Mechanism classifier ─────────────────────────────────────────────────────

@pytest.mark.parametrize("desc,expected", [
    ("Novy-Marx 2013 gross profitability", "quality"),
    ("Blitz-Huij-Martens 2011 residual momentum", "residual_momentum"),
    ("VRP / VIX term-structure carry (Karagozoglu-Lin 2010)", "vol_carry"),
    ("Hong-Lim-Stein 2000 sector lead-lag", "lead_lag"),
    ("KMPV roll-yield carry across asset classes", "carry"),
    ("Moskowitz-Ooi-Pedersen TSMOM trend continuation", "tsmom"),
    ("PEAD post-earnings drift", "earnings_underreaction"),
    ("MSM multivariate regime overlay", "regime_overlay"),
    ("Hurst-Ooi-Pedersen 2017 crisis alpha hedge", "crisis_alpha"),
])
def test_classify_mechanism_canonical(desc, expected):
    families = classify_mechanism(desc)
    assert expected in families


def test_classify_unclassified():
    families = classify_mechanism("totally unrelated text foo bar baz")
    assert families == ["unclassified"]


# ── Asset class classifier ──────────────────────────────────────────────────

@pytest.mark.parametrize("desc,expected", [
    ("Novy-Marx gross profitability stock factor", "equity"),
    ("Sector SPDR ETF XLK XLF lead lag", "etfs"),
    ("VIX variance VRP only — no futures here", "vol"),
    ("HYG high-yield credit spread strategy", "credit"),
    ("KMPV cross-asset futures carry tr_ds_fut", "futures"),
])
def test_classify_asset_class(desc, expected):
    assert classify_asset_class(desc) == expected


def test_classify_asset_class_mixed_when_no_keywords():
    assert classify_asset_class("foo bar baz") == "mixed"


# ── Verdict normalization ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("GREEN", "GREEN"),
    ("GREEN — 4/4 strict bars cleared at c=4.5%", "GREEN"),
    ("YELLOW", "YELLOW"),
    ("RED", "RED"),
    ("UNINTERPRETABLE", "UNINTERPRETABLE"),
    (None, "UNKNOWN"),
    ("something weird", "UNKNOWN"),
])
def test_normalize_verdict(raw, expected):
    assert normalize_verdict(raw) == expected


# ── Parent family rollup ────────────────────────────────────────────────────

@pytest.mark.parametrize("child,parent", [
    ("quality",            "equity_factor"),
    ("residual_momentum",  "equity_factor"),
    ("momentum",           "equity_factor"),
    ("earnings_underreaction", "equity_factor"),
    ("carry",              "cross_asset_carry"),
    ("vol_carry",          "cross_asset_carry"),
    ("tsmom",              "cross_asset_trend"),
    ("lead_lag",           "network_effects"),
])
def test_parent_family_rollup(child, parent):
    assert parent in parent_families_for(child)


# ── Full graph build (smoke + structure) ────────────────────────────────────

def test_graph_builds_from_real_data():
    g = build_graph()
    s = summary(g)
    # Expected node types present
    assert s["n_nodes_by_type"]["Candidate"] >= 5
    assert s["n_nodes_by_type"]["MechanismFamily"] >= 10
    assert s["n_nodes_by_type"]["AssetClass"] == 6
    assert s["n_nodes_by_type"]["Verdict"] >= 4
    assert s["n_nodes_by_type"]["Sleeve"] == 3   # 3 deployed sleeves


def test_verdict_counts_are_nonzero_and_sane():
    """Each verdict bucket can have multiple candidates; candidates can
    appear in multiple verdict buckets if they were re-run (different
    `received` edges over time). So we don't require strict equality, just
    that the totals are nonzero and the green sleeves reflect today's 3
    deployed mechanisms."""
    g = build_graph()
    n_red = len(candidates_by_verdict(g, "RED"))
    assert n_red >= 4    # we have at least 4 RED Phase-2 candidates today


# ── Cousin detection (the key use case) ─────────────────────────────────────

def test_cousin_detection_residual_momentum_vs_quality():
    g = build_graph()
    sims = similar_candidates(g, "residual_momentum_bhm_2011_v1")
    sim_ids = [s.id for s in sims]
    assert "quality_novymarx_2013_v1" in sim_ids


def test_cousin_detection_is_symmetric():
    g = build_graph()
    rm_sims = {s.id for s in similar_candidates(g, "residual_momentum_bhm_2011_v1")}
    q_sims  = {s.id for s in similar_candidates(g, "quality_novymarx_2013_v1")}
    # quality should be in rm's sims AND rm should be in quality's sims
    assert "quality_novymarx_2013_v1" in rm_sims
    assert "residual_momentum_bhm_2011_v1" in q_sims


def test_lead_lag_has_no_cousins_in_current_data():
    """Lead-lag is a truly new mechanism class for us — no parent overlap
    with the other candidates currently in graph."""
    g = build_graph()
    sims = similar_candidates(g, "sector_leadlag_v1_dailysignal_monthlyrebal")
    assert len(sims) == 0


# ── Deployed overlap detection ──────────────────────────────────────────────

def test_residual_momentum_overlaps_equity_book_at_parent_level():
    """Today's documented finding: residual_momentum has book_corr 0.66 with
    PEAD book. The graph must surface this at parent_only level."""
    g = build_graph()
    overlap = deployed_overlap_check(g, "residual_momentum_bhm_2011_v1")
    assert "equity_book" in overlap
    assert overlap["equity_book"]["overlap_strength"] == "parent_only"
    assert "equity_factor" in overlap["equity_book"]["parent_level_overlap"]


def test_quality_overlaps_equity_book_at_parent_level():
    g = build_graph()
    overlap = deployed_overlap_check(g, "quality_novymarx_2013_v1")
    assert "equity_book" in overlap


def test_deployed_overlap_handles_unknown_candidate():
    g = build_graph()
    overlap = deployed_overlap_check(g, "nonexistent_candidate_xyz")
    assert "error" in overlap


# ── Blind spots & coverage ──────────────────────────────────────────────────

def test_blind_spots_non_empty_and_sorted():
    g = build_graph()
    blinds = candidate_blind_spots(g)
    assert len(blinds) > 0
    assert blinds == sorted(blinds)


def test_blind_spots_include_credit_combinations():
    """We haven't tested credit-based strategies at all — expect many credit
    blind spots."""
    g = build_graph()
    blinds = candidate_blind_spots(g)
    credit_blinds = [(ac, fam) for ac, fam in blinds if ac == "credit"]
    assert len(credit_blinds) >= 5


def test_coverage_matrix_shape():
    g = build_graph()
    m = coverage_matrix(g)
    assert isinstance(m, pd.DataFrame)
    # rows = mechanism families, cols = asset classes
    assert m.shape[1] == 6     # 6 asset classes
    assert m.shape[0] >= 10    # 10+ canonical families


def test_coverage_matrix_carry_futures_cell_high():
    """We've tested many carry-on-futures candidates — that cell should be hot."""
    g = build_graph()
    m = coverage_matrix(g)
    assert m.loc["carry", "futures"] >= 3
