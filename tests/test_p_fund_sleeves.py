"""
tests/test_p_fund_sleeves.py — MS-3 per-sleeve attribution tests.

Coverage:
  - SleevePeriodSummary dataclass invariants
  - compute_per_sleeve_summary math correctness (geometric linking)
  - Empty period / no rows → zero return
  - Cross-sleeve summary structure
  - Reject unknown sleeve_id
  - 0-LLM-imports invariant
"""
from __future__ import annotations

import datetime
from typing import Iterable

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ── In-memory DB fixture (mirrors test_ms1_sleeve_id pattern) ──────────────
@pytest.fixture
def session_factory(monkeypatch: pytest.MonkeyPatch):
    """Fresh in-memory DB with ORM tables created from db_models.Base.metadata."""
    import engine.memory as mem
    import engine.db_models as dbm

    eng = create_engine("sqlite:///:memory:", future=True)
    dbm.Base.metadata.create_all(eng)
    SF = sessionmaker(bind=eng, future=True)
    monkeypatch.setattr(mem, "engine", eng)
    monkeypatch.setattr(mem, "SessionFactory", SF)
    return SF


def _seed_monthly_returns(
    session_factory,
    sleeve_id:      str,
    monthly_data:   Iterable[tuple[datetime.date, str, float, float]],
) -> None:
    """Helper: insert SimulatedMonthlyReturn rows.

    monthly_data: iterable of (return_month, sector, weight_held, sector_return)
    contribution = weight_held × sector_return is computed automatically.
    """
    from engine.memory import SimulatedMonthlyReturn
    SF = session_factory
    with SF() as s:
        for month, sector, w, r in monthly_data:
            contrib = w * r
            s.add(SimulatedMonthlyReturn(
                return_month=month, sector=sector,
                weight_held=w, sector_return=r,
                contribution=contrib,
                is_profitable=contrib > 0,
                sleeve_id=sleeve_id,
            ))
        s.commit()


# ── compute_per_sleeve_summary basic ────────────────────────────────────────
def test_per_sleeve_empty_when_no_rows(session_factory) -> None:
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    assert out.sleeve_id == "etf_l1"
    assert out.n_months == 0
    assert out.twr_period == 0.0
    assert out.twr_annualized == 0.0
    assert "no SimulatedMonthlyReturn rows" in out.note


def test_per_sleeve_rejects_unknown_sleeve(session_factory) -> None:
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    with pytest.raises(ValueError, match="not in ALLOWED_SLEEVES"):
        compute_per_sleeve_summary(
            sleeve_id="not_real",
            start=datetime.date(2024, 1, 1),
            end=datetime.date(2024, 12, 31),
        )


def test_per_sleeve_single_month_single_sector(session_factory) -> None:
    """One month, one sector: weight 1.0 × return 0.05 → twr_period = 0.05."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 6, 1), "Tech", 1.0, 0.05),
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    assert out.n_months == 1
    assert abs(out.twr_period - 0.05) < 1e-9


def test_per_sleeve_multi_sector_aggregates_within_month(session_factory) -> None:
    """Multiple sectors in same month: contributions sum per month."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 6, 1), "Tech",   0.5, 0.10),  # contrib = 0.05
        (datetime.date(2024, 6, 1), "Finance", 0.5, 0.04), # contrib = 0.02
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    # June total = 0.05 + 0.02 = 0.07; one month → twr_period = 0.07
    assert out.n_months == 1
    assert abs(out.twr_period - 0.07) < 1e-9


def test_per_sleeve_geometric_link_two_months(session_factory) -> None:
    """Two months: (1+r1)*(1+r2) - 1 = 0.10*0.05 + 0.10 + 0.05 = 0.155."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 5, 1), "Tech", 1.0, 0.10),
        (datetime.date(2024, 6, 1), "Tech", 1.0, 0.05),
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    expected = 1.10 * 1.05 - 1.0   # = 0.155
    assert abs(out.twr_period - expected) < 1e-9
    assert out.n_months == 2


def test_per_sleeve_filters_date_range(session_factory) -> None:
    """Only rows within [start, end] count."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2023, 11, 1), "Tech", 1.0, 0.50),  # OUT of range
        (datetime.date(2024, 5, 1),  "Tech", 1.0, 0.10),
        (datetime.date(2024, 12, 1), "Tech", 1.0, 0.05),
        (datetime.date(2025, 1, 1),  "Tech", 1.0, 0.99),  # OUT of range
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1),
        end=datetime.date(2024, 12, 31),
    )
    expected = 1.10 * 1.05 - 1.0
    assert out.n_months == 2
    assert abs(out.twr_period - expected) < 1e-9


def test_per_sleeve_filters_by_sleeve_id(session_factory) -> None:
    """ETF rows shouldn't bleed into ss_sp500 query and vice versa."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 6, 1), "Tech", 1.0, 0.10),
    ])
    _seed_monthly_returns(session_factory, "ss_sp500", [
        (datetime.date(2024, 6, 1), "AAPL", 1.0, 0.20),
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out_etf = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31),
    )
    out_ss = compute_per_sleeve_summary(
        sleeve_id="ss_sp500",
        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31),
    )
    assert abs(out_etf.twr_period - 0.10) < 1e-9
    assert abs(out_ss.twr_period - 0.20) < 1e-9


def test_per_sleeve_annualization_makes_sense(session_factory) -> None:
    """6 months of identical 1% monthly → annualized ≈ 12.68% (1.01^12 - 1)."""
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, m, 1), "Tech", 1.0, 0.01)
        for m in range(1, 7)  # Jan-Jun
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31),
    )
    assert out.n_months == 6
    period_expected = 1.01 ** 6 - 1
    assert abs(out.twr_period - period_expected) < 1e-9
    # Annualized: (1 + period)^(12/6) - 1 = (1.01^6)^2 - 1 = 1.01^12 - 1
    annual_expected = 1.01 ** 12 - 1
    assert abs(out.twr_annualized - annual_expected) < 1e-9


def test_per_sleeve_contributing_period_correct(session_factory) -> None:
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 3, 1), "Tech", 1.0, 0.02),
        (datetime.date(2024, 7, 1), "Tech", 1.0, 0.03),
        (datetime.date(2024, 11, 1), "Tech", 1.0, 0.04),
    ])
    from engine.p_fund_sleeves import compute_per_sleeve_summary
    out = compute_per_sleeve_summary(
        sleeve_id="etf_l1",
        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31),
    )
    assert out.contributing_period == (datetime.date(2024, 3, 1), datetime.date(2024, 11, 1))


# ── Cross-sleeve summary structure ─────────────────────────────────────────
def test_cross_sleeve_summary_has_both_sleeves(session_factory) -> None:
    _seed_monthly_returns(session_factory, "etf_l1", [
        (datetime.date(2024, 6, 1), "Tech", 1.0, 0.10),
    ])
    from engine.p_fund_sleeves import compute_cross_sleeve_summary
    out = compute_cross_sleeve_summary(
        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31),
    )
    # Should have BOTH sleeve summaries even if one is empty
    assert "etf_l1" in out["per_sleeve"]
    assert "ss_sp500" in out["per_sleeve"]
    assert out["per_sleeve"]["etf_l1"].n_months == 1
    assert out["per_sleeve"]["ss_sp500"].n_months == 0
    assert "caveat" in out
    assert "synthetic" in out["caveat"].lower() or "GIPS" in out["caveat"]


def test_cross_sleeve_summary_window_recorded(session_factory) -> None:
    from engine.p_fund_sleeves import compute_cross_sleeve_summary
    out = compute_cross_sleeve_summary(
        start=datetime.date(2024, 3, 15), end=datetime.date(2024, 9, 30),
    )
    assert out["window"]["start"] == "2024-03-15"
    assert out["window"]["end"] == "2024-09-30"


# ── 0-LLM-imports invariant ────────────────────────────────────────────────
def test_module_has_no_llm_imports() -> None:
    import engine.p_fund_sleeves as mod
    src = open(mod.__file__, encoding="utf-8").read()
    forbidden = ["google.generativeai", "google.genai",
                 "from engine.deepseek_client", "from engine.key_pool"]
    for pattern in forbidden:
        assert pattern not in src, (
            f"p_fund_sleeves violates 0-LLM-imports invariant: found {pattern!r}"
        )
