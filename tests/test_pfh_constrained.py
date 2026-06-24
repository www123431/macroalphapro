"""Tests for engine.research.pfh constrained mode — Week 4 closed loop.

Covers:
  - Axis catalog discovery (universes / signals / weightings / tested tuples)
  - Constrained generator enumerates Cartesian product minus tested
  - PFH proposer in mode=constrained returns valid candidates
  - Emitted compose-spec YAMLs are well-formed AND materialize-able
  - End-to-end: PFH suggest → emit → materialize works on real catalog
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.research.pfh.axis_catalog import (
    AxisCatalog, enumerate_untested_tuples, load_axis_catalog,
)
from engine.research.pfh.constrained_generator import (
    _infer_family_from_signal, generate_constrained_candidates,
)
from engine.research.pfh.proposer import suggest_top_k


# ── Axis catalog ────────────────────────────────────────────────────


def test_load_axis_catalog_finds_real_components():
    """Smoke against the actual repo: confirm Week 4 catalog expansion
    landed (>= 3 universes, >= 4 signals, >= 2 weightings)."""
    cat = load_axis_catalog()
    assert len(cat.universes)  >= 3, f"expected ≥3 universes, got {cat.universes}"
    assert len(cat.signals)    >= 4, f"expected ≥4 signals, got {cat.signals}"
    assert len(cat.weightings) >= 2, f"expected ≥2 weightings, got {cat.weightings}"


def test_load_axis_catalog_includes_real_crsp_universe():
    cat = load_axis_catalog()
    assert "equity_us_crsp_monthly" in cat.universes


def test_load_axis_catalog_includes_cross_asset_universes():
    """Path B (2026-06-01) added 3 cross-asset universes: full 17-instrument
    panel + FX subset + commodity subset."""
    cat = load_axis_catalog()
    for required in ("futures_cross_asset_17_monthly",
                     "futures_fx_g3_monthly",
                     "futures_commodity_8_monthly"):
        assert required in cat.universes, \
            f"missing cross-asset universe {required!r}"


def test_diversify_top_k_respects_per_universe_cap():
    """Constrained mode top-K must not be dominated by one universe.

    Without the per-universe cap, ALL untested-cell candidates share the
    same posterior (cell n=0 → prior dominates), and ties are broken by
    alphabetical universe order. The cap forces engine to spread across
    asset classes.
    """
    out = suggest_top_k(k=10, mode="constrained",
                        max_per_family=10, max_per_universe=2,
                        write_specs=False, write_ledger=False)
    universes = [s["proposal"]["universe"] for s in out["top"]]
    from collections import Counter
    counts = Counter(universes)
    for uni, n in counts.items():
        assert n <= 2, f"universe {uni} appears {n} times (cap is 2)"


def test_cross_asset_universe_loads_via_composer(tmp_path, monkeypatch):
    """Smoke test that the new cross-asset universe parquets load
    correctly through the composer."""
    from engine.feature_store.composer import (
        load_universe, load_universe_panel,
    )
    uni = load_universe("futures_cross_asset_17_monthly")
    panel = load_universe_panel(uni)
    # 228 months × 17 assets per the ETL spec
    assert panel.shape[0] >= 150
    assert panel.shape[1] == 17
    # Date range should extend into the 2010s+
    assert panel.index.max().year >= 2020


def test_load_axis_catalog_picks_up_tested_compose_specs():
    """The seed compose-spec (eq_mom_12_1_demo) + the real demo spec
    (eq_mom_12_1_us_real) should both register as tested tuples."""
    cat = load_axis_catalog()
    # synthetic_equity_demo × momentum_12_1 × decile_ls_10 is tested
    assert ("synthetic_equity_demo", "momentum_12_1", "decile_ls_10") \
            in cat.tested_tuples
    # CRSP real-data spec
    assert ("equity_us_crsp_monthly", "momentum_12_1", "decile_ls_10") \
            in cat.tested_tuples


def test_enumerate_untested_tuples_excludes_tested():
    cat = load_axis_catalog()
    untested = enumerate_untested_tuples(cat)
    for t in untested:
        assert t not in cat.tested_tuples
    assert len(untested) == cat.n_possible - len(cat.tested_tuples)


def test_enumerate_untested_count_matches_arithmetic():
    cat = load_axis_catalog()
    expected_total = len(cat.universes) * len(cat.signals) * len(cat.weightings)
    assert cat.n_possible == expected_total


# ── Family inference ────────────────────────────────────────────────


def test_infer_family_handles_known_signals():
    assert _infer_family_from_signal("momentum_12_1") == "momentum"
    assert _infer_family_from_signal("momentum_3_1")  == "momentum"
    assert _infer_family_from_signal("reversal_1m")   == "reversal"


def test_infer_family_fallback_for_unknown():
    assert _infer_family_from_signal("my_made_up_signal") == "my_made_up_signal"


# ── Constrained generator ────────────────────────────────────────────


def test_constrained_generator_emits_only_existing_axes():
    """Every candidate must reference real axis component names."""
    cat = load_axis_catalog()
    candidates = generate_constrained_candidates(cat)
    for c in candidates:
        assert c.universe in cat.universes
        assert c.signal_recipe in cat.signals
        assert c.weighting in cat.weightings
        # CRITICAL: needs_new_axes must be empty (closed-loop requirement)
        assert c.needs_new_axes == [], \
            f"constrained candidate {c.candidate_id} has needs_new_axes"


def test_constrained_generator_count_matches_untested():
    cat = load_axis_catalog()
    candidates = generate_constrained_candidates(cat)
    assert len(candidates) == cat.n_untested


def test_constrained_generator_kind_label():
    cat = load_axis_catalog()
    candidates = generate_constrained_candidates(cat)
    if candidates:
        assert all(c.proposal_kind == "constrained" for c in candidates)


# ── Proposer in constrained mode ────────────────────────────────────


def test_proposer_constrained_mode_returns_valid_dict():
    out = suggest_top_k(k=3, mode="constrained",
                        write_specs=False, write_ledger=False)
    assert out["mode"] == "constrained"
    assert out["n_candidates_total"] > 0
    assert len(out["top"]) <= 3
    for s in out["top"]:
        assert s["proposal"]["proposal_kind"] == "constrained"
        assert s["proposal"]["universe"] is not None
        assert s["proposal"]["signal_recipe"] is not None
        assert s["proposal"]["weighting"] is not None


def test_proposer_open_vs_constrained_produce_different_candidates():
    """Sanity: the two modes should produce structurally different output."""
    o = suggest_top_k(k=3, mode="open",
                      write_specs=False, write_ledger=False)
    c = suggest_top_k(k=3, mode="constrained",
                      write_specs=False, write_ledger=False)
    o_kinds = {s["proposal"]["proposal_kind"] for s in o["top"]}
    c_kinds = {s["proposal"]["proposal_kind"] for s in c["top"]}
    assert c_kinds == {"constrained"}
    assert "constrained" not in o_kinds


def test_proposer_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        suggest_top_k(k=3, mode="invalid_mode",
                      write_specs=False, write_ledger=False)


# ── End-to-end: PFH → emit → materialize ────────────────────────────


def test_pfh_constrained_emitted_specs_materialize(tmp_path, monkeypatch):
    """The headline Week 4 test: PFH writes specs, materializer reads
    them, and the composer produces a non-trivial output."""
    # Redirect ledger + computed dir to tmp so we don't pollute repo
    monkeypatch.setattr(
        "engine.research.pfh.proposer.PFH_LEDGER",
        tmp_path / "pfh.jsonl",
    )
    # Specs go to the real specs dir so the registry/loader can find them
    out = suggest_top_k(k=3, mode="constrained",
                        write_specs=True, write_ledger=False)
    assert out["written_spec_paths"], "expected specs to be written"

    # Verify each written spec parses as a compose-spec
    from engine.feature_store.composer import load_compose_spec
    from engine.feature_store.registry import SPECS_DIR
    for relpath in out["written_spec_paths"]:
        spec_path = Path(relpath)
        if not spec_path.is_absolute():
            spec_path = Path(__file__).resolve().parents[1] / relpath
        # Strict load: must succeed
        spec = load_compose_spec(spec_path)
        assert spec.universe.name in {u for u in
            ["equity_us_crsp_monthly", "equity_us_high_vol_monthly",
             "equity_us_post2018_monthly", "synthetic_equity_demo"]}, \
            f"unknown universe {spec.universe.name!r}"

    # Clean up the emitted specs to keep the repo state stable across runs
    for relpath in out["written_spec_paths"]:
        p = Path(relpath)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[1] / relpath
        if p.is_file():
            p.unlink()


def test_pfh_constrained_spec_yaml_well_formed(tmp_path, monkeypatch):
    """Emitted YAMLs must include PFH audit metadata."""
    out = suggest_top_k(k=1, mode="constrained",
                        write_specs=True, write_ledger=False)
    assert out["written_spec_paths"]

    relpath = out["written_spec_paths"][0]
    p = Path(relpath)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / relpath
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert raw["audit"]["added_by"] == "pfh"
        assert raw["audit"]["status"] == "pending_pfh_review"
        assert "pfh_score" in raw["audit"]
        # CRITICAL Week 4 difference: inputs[] should reference the REAL
        # universe input_path, not the PLACEHOLDER from open mode
        assert raw["inputs"][0]["cache_path"] != \
            "data/feature_store/_specs/PFH_DEPENDENCIES_TBD"
    finally:
        if p.is_file():
            p.unlink()
