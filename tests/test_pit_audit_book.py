"""tests/test_pit_audit_book.py — book-level PIT register covers every deployed mechanism."""
from pathlib import Path

import pytest

from engine.validation.pit_audit_book import (
    SURFACES, CANDIDATE_SURFACES, run_book_pit_audit,
)


def test_every_deployed_strategy_has_a_documented_surface():
    from engine.strategies import get_registry
    live = set(get_registry().names())
    missing = live - set(SURFACES)
    assert not missing, f"deployed strategies with no PIT surface entry: {missing}"


def test_surface_entries_are_well_formed():
    for s in list(SURFACES.values()) + list(CANDIDATE_SURFACES.values()):
        assert s.surface and s.control and s.anchor
        assert s.verification in ("data", "construction")
        assert s.status in ("PASS", "FLAG", "INFO")
    # D_PEAD must be the DATA-verified one (the deep audit), not construction-only
    assert SURFACES["D_PEAD"].verification == "data"


def test_carry_candidate_is_marked_and_not_in_live_surfaces():
    assert "cross_asset_carry" in CANDIDATE_SURFACES
    assert "cross_asset_carry" not in SURFACES


def test_book_audit_clean_on_real_book():
    if not Path("data/cache/_pead_ts_panel_2014_2023.parquet").exists():
        pytest.skip("deployed D_PEAD panel cache not present")
    rep = run_book_pit_audit()
    assert rep.undocumented == []
    assert rep.dpead_data_verified is True
    assert rep.book_clean is True
    assert "BOOK LOOK-AHEAD CLEAN" in rep.overall
