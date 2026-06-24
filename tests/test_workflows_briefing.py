"""tests/test_workflows_briefing.py — Wave A Morning Briefing tests.

Covers:
  - Each section gatherer returns BriefingItems with valid shape
  - Curation respects severity-then-recency ordering + MAX cap
  - Headline summary count is correct
  - Markdown renderer produces a deterministic, parseable artifact
  - persist_briefing writes the file at the documented path
  - Empty-DB case: generate_briefing returns a Briefing with 0 items
    and a "nothing to report" headline (must not crash)
  - Section gatherer failure modes are caught (one bad table doesn't
    sink the whole briefing)

Tests deliberately use real but empty tables — relies on the
test_persona_session_store conftest fixture that create_all-s the
full schema. Each test that inserts rows cleans up its own data.
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest


# Use the same fixture pattern as test_persona_session_store so the
# alert / audit tables exist in the temp DB.
@pytest.fixture(autouse=True)
def _ensure_briefing_tables():
    from engine.db_models import Base as DBModelsBase, engine
    from engine.auto_audit_models import Base as AuditBase
    DBModelsBase.metadata.create_all(engine)
    AuditBase.metadata.create_all(engine)
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Data-shape invariants
# ──────────────────────────────────────────────────────────────────────────────
def test_briefing_item_dataclass_shape():
    from engine.workflows import BriefingItem
    item = BriefingItem(
        section="risk", title="t", body="b", severity="URGENT",
        source_table="x", source_ids=(1, 2),
    )
    # Frozen dataclass — must reject mutation
    with pytest.raises(Exception):
        item.severity = "INFO"


def test_generate_briefing_on_empty_db():
    """Empty alert / NAV / audit tables → no items, friendly headline,
    no exception."""
    from engine.workflows import generate_briefing
    b = generate_briefing()
    assert b.n_total == 0
    assert len(b.items) == 0
    assert "All quiet" in b.headline


def test_curation_caps_to_max_items():
    from engine.workflows import BriefingItem
    from engine.workflows.briefing import _curate, MAX_ITEMS_PER_BRIEFING

    many = [
        BriefingItem(
            section="risk", title=f"t{i}", body="b",
            severity="INFO", source_table="x", source_ids=(i,),
        )
        for i in range(20)
    ]
    out = _curate(many)
    assert len(out) == MAX_ITEMS_PER_BRIEFING


def test_curation_ranks_urgent_before_info():
    from engine.workflows import BriefingItem
    from engine.workflows.briefing import _curate

    items = [
        BriefingItem(section="risk", title="info", body="", severity="INFO",
                     source_table="x", source_ids=(1,)),
        BriefingItem(section="risk", title="urgent", body="", severity="URGENT",
                     source_table="x", source_ids=(2,)),
        BriefingItem(section="risk", title="notable", body="", severity="NOTABLE",
                     source_table="x", source_ids=(3,)),
    ]
    out = _curate(items, max_k=3)
    assert [it.title for it in out] == ["urgent", "notable", "info"]


def test_curation_ranks_recent_before_old_within_severity():
    from engine.workflows import BriefingItem
    from engine.workflows.briefing import _curate

    older = datetime.datetime(2020, 1, 1, 12, 0)
    newer = datetime.datetime(2026, 5, 19, 12, 0)
    items = [
        BriefingItem(section="risk", title="old",   body="", severity="NOTABLE",
                     source_table="x", source_ids=(1,), observed_at=older),
        BriefingItem(section="risk", title="newer", body="", severity="NOTABLE",
                     source_table="x", source_ids=(2,), observed_at=newer),
    ]
    out = _curate(items, max_k=2)
    assert [it.title for it in out] == ["newer", "old"]


def test_headline_counts_correctly():
    from engine.workflows import BriefingItem
    from engine.workflows.briefing import _headline

    items = [
        BriefingItem(section="risk", title="a", body="", severity="URGENT",
                     source_table="x", source_ids=()),
        BriefingItem(section="risk", title="b", body="", severity="URGENT",
                     source_table="x", source_ids=()),
        BriefingItem(section="data", title="c", body="", severity="NOTABLE",
                     source_table="x", source_ids=()),
        BriefingItem(section="data", title="d", body="", severity="INFO",
                     source_table="x", source_ids=()),
    ]
    h = _headline(items)
    assert "2 URGENT" in h
    assert "1 NOTABLE" in h
    assert "1 INFO" in h


# ──────────────────────────────────────────────────────────────────────────────
# Section gatherers — verified by inserting real rows
# ──────────────────────────────────────────────────────────────────────────────
def test_section_risk_picks_up_recent_alert():
    from engine.db_models import RiskManagerAlert, SessionFactory
    from engine.workflows.briefing import _section_risk

    today = datetime.date.today()
    with SessionFactory() as s:
        s.add(RiskManagerAlert(
            date              = today,
            alert_id          = "test-briefing-risk-uuid",
            mode_id           = "TEST_MODE",
            severity          = "HARD_HALT",
            cb_severity       = "SEVERE",
            halt_decision     = True,
            phase             = "pre_trade",
            rule_description  = "Test briefing risk item",
            affected_json     = '["TEST_TICKER"]',
            extra_json        = "{}",
            spec_anchor       = "test",
            generated_at_utc  = datetime.datetime.utcnow(),
        ))
        s.commit()

    try:
        items = _section_risk(today)
        titles = [it.title for it in items]
        assert any("TEST_MODE" in t for t in titles), (
            "_section_risk did not pick up the inserted alert row"
        )
        # The matching item should be URGENT (halt_decision=True)
        match = [it for it in items if "TEST_MODE" in it.title][0]
        assert match.severity == "URGENT"
        assert match.suggested_followup is not None
        assert "Risk Manager" in match.suggested_followup
    finally:
        with SessionFactory() as s:
            s.query(RiskManagerAlert).filter(
                RiskManagerAlert.alert_id == "test-briefing-risk-uuid"
            ).delete()
            s.commit()


def test_section_attribution_summarizes_nav_path():
    from engine.db_models import PortfolioNavSnapshot, SessionFactory
    from engine.workflows.briefing import _section_attribution

    today = datetime.date.today()
    # Insert a 3-day NAV path: 1.000 → 1.005 → 1.010 (+1%)
    rows = []
    for i, nav in enumerate([1.000, 1.005, 1.010]):
        d = today - datetime.timedelta(days=2 - i)
        rows.append(PortfolioNavSnapshot(
            snapshot_date         = d,
            nav_open              = nav - 0.001,
            external_flow         = 0.0,
            nav_after_flow        = nav - 0.001,
            nav_close             = nav,
            gross_pnl             = 0.001,
            daily_modified_dietz  = (nav / (nav - 0.001)) - 1 if i > 0 else 0.0,
        ))
    with SessionFactory() as s:
        for r in rows:
            s.merge(r)
        s.commit()

    try:
        items = _section_attribution(today)
        assert len(items) >= 1, "_section_attribution returned nothing"
        item = items[0]
        assert item.section == "attribution"
        # Body should reference the NAV values
        assert "1.0000" in item.title or "1.0100" in item.title
    finally:
        with SessionFactory() as s:
            for r in rows:
                s.query(PortfolioNavSnapshot).filter(
                    PortfolioNavSnapshot.snapshot_date == r.snapshot_date
                ).delete()
            s.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Render + persist
# ──────────────────────────────────────────────────────────────────────────────
def test_render_as_markdown_empty_briefing():
    from engine.workflows import Briefing, render_as_markdown

    b = Briefing(
        as_of        = datetime.date(2026, 5, 19),
        generated_at = datetime.datetime(2026, 5, 19, 14, 30),
        items        = (),
        headline     = "All quiet.",
        n_total      = 0,
    )
    md = render_as_markdown(b)
    assert "Morning Briefing — 2026-05-19" in md
    assert "All quiet." in md
    assert "Nothing to report" in md


def test_render_as_markdown_includes_followups():
    from engine.workflows import Briefing, BriefingItem, render_as_markdown

    item = BriefingItem(
        section="risk", title="Test Risk Title",
        body="Test risk body sentence.",
        severity="URGENT",
        source_table="risk_manager_alerts",
        source_ids=(1, 2),
        suggested_followup="Ask Risk Manager: 'detail X'",
        observed_at=datetime.datetime(2026, 5, 19, 6, 30),
    )
    b = Briefing(
        as_of        = datetime.date(2026, 5, 19),
        generated_at = datetime.datetime(2026, 5, 19, 6, 35),
        items        = (item,),
        headline     = "1 URGENT",
        n_total      = 1,
    )
    md = render_as_markdown(b)
    assert "URGENT" in md
    assert "Test Risk Title" in md
    assert "Test risk body sentence." in md
    assert "Ask Risk Manager: 'detail X'" in md
    assert "risk_manager_alerts" in md


def test_persist_briefing_writes_file(tmp_path):
    from engine.workflows import Briefing, persist_briefing

    b = Briefing(
        as_of        = datetime.date(2026, 5, 19),
        generated_at = datetime.datetime(2026, 5, 19, 6, 35),
        items        = (),
        headline     = "All quiet.",
        n_total      = 0,
    )
    out_path = persist_briefing(b, out_dir=tmp_path)
    assert out_path.exists()
    assert out_path.name == "2026-05-19.md"
    assert "Morning Briefing" in out_path.read_text(encoding="utf-8")


def test_persist_briefing_is_idempotent(tmp_path):
    """Same-date persist must overwrite (not create -2 -3 -N suffixes)."""
    from engine.workflows import Briefing, persist_briefing

    b1 = Briefing(
        as_of        = datetime.date(2026, 5, 19),
        generated_at = datetime.datetime(2026, 5, 19, 6, 35),
        items        = (),
        headline     = "v1",
        n_total      = 0,
    )
    b2 = Briefing(
        as_of        = datetime.date(2026, 5, 19),
        generated_at = datetime.datetime(2026, 5, 19, 7, 0),
        items        = (),
        headline     = "v2",
        n_total      = 0,
    )
    persist_briefing(b1, out_dir=tmp_path)
    p2 = persist_briefing(b2, out_dir=tmp_path)
    assert p2.read_text(encoding="utf-8").count("Morning Briefing") == 1
    assert "v2" in p2.read_text(encoding="utf-8")
    # And only one file exists
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1


def test_generate_briefing_does_not_crash_on_partial_table_failure(monkeypatch):
    """If one section gatherer raises, the rest must still return —
    a missing table cannot sink the morning briefing."""
    import engine.workflows.briefing as bm

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated table missing")

    monkeypatch.setattr(bm, "_section_risk", _boom)
    # Should still produce a Briefing (with items from other sections,
    # which on an empty DB will be 0 — but the call must not raise)
    b = bm.generate_briefing()
    assert b is not None
    assert b.headline   # any non-empty string


def test_followup_strings_are_copyable():
    """Every BriefingItem with suggested_followup must produce a string
    the user can copy-paste into Chief of Staff. Verifies the format
    looks like an Anthropic-style instruction prompt — starts with
    'Ask X:' followed by the actual question in quotes."""
    from engine.workflows.briefing import _section_risk

    # Use a known good test row to exercise the section
    today = datetime.date.today()
    from engine.db_models import RiskManagerAlert, SessionFactory
    with SessionFactory() as s:
        s.add(RiskManagerAlert(
            date              = today,
            alert_id          = "test-briefing-followup-uuid",
            mode_id           = "1",
            severity          = "SOFT_WARN",
            cb_severity       = "LIGHT",
            halt_decision     = False,
            phase             = "pre_trade",
            rule_description  = "Test followup",
            affected_json     = "[]",
            extra_json        = "{}",
            spec_anchor       = "test",
            generated_at_utc  = datetime.datetime.utcnow(),
        ))
        s.commit()

    try:
        items = _section_risk(today)
        target = [it for it in items if "Mode 1" in it.title][0]
        fu = target.suggested_followup
        assert fu is not None
        assert fu.startswith("Ask ")
        assert ":" in fu
        # No emoji per project doctrine
        import re
        emoji_chars = re.findall(r"[\U0001F300-\U0001FAFF]", fu)
        assert emoji_chars == []
    finally:
        with SessionFactory() as s:
            s.query(RiskManagerAlert).filter(
                RiskManagerAlert.alert_id == "test-briefing-followup-uuid"
            ).delete()
            s.commit()
