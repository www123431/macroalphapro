"""tests/test_papers_curator_synthesis_context.py — Phase 2.0 step 3b.

Context gatherer tests. Uses monkeypatch on the module-level path
constants so each test runs in a fresh tmp_path — no risk of polluting
real data dirs.

Verifies:
  - All four loaders return () gracefully when source missing/empty
  - Summaries are joined correctly from cache+summaries by (source, source_id)
  - Recency window filters fetched_ts < cutoff
  - INGEST > READ_AND_DISCARD > SKIP ordering
  - Hard cap on summaries (max_rows)
  - DEPLOYED filter on sleeves (skip non-deployed)
  - KPI fields handled gracefully when None/missing
  - Recent events filtered to synthesis-relevant types
  - Newest-first ordering on events
  - Hard cap on events (max_rows)
  - Top-level build_synthesis_input wires everything + emits snapshot_ts
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _ts(days_ago: int = 0) -> str:
    """ISO timestamp `days_ago` days before now."""
    return (_dt.datetime.utcnow()
            - _dt.timedelta(days=days_ago)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_jsonl(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _patch_paths(monkeypatch, tmp_path: Path):
    """Redirect every gatherer source to tmp_path/<subdir>."""
    from engine.agents.papers_curator import synthesis_context as sc
    papers   = tmp_path / "papers_curator"
    library  = tmp_path / "library"
    events   = tmp_path / "events.jsonl"
    monkeypatch.setattr(sc, "_PAPERS_DIR", papers)
    monkeypatch.setattr(sc, "_LIBRARY_DIR", library)
    monkeypatch.setattr(sc, "_EVENTS_PATH", events)
    return papers, library, events


def _patch_doctrine(monkeypatch, *, hits=None, raises=False):
    """Patch the doctrine_index module so synthesis_context's
    _load_doctrine_snippets path is testable without chroma.
    hits is a list of (name, description, snippet) tuples."""
    from engine.agents.papers_curator import doctrine_index as di

    class _FakeHit:
        def __init__(self, name, description, snippet):
            self.name = name
            self.description = description
            self.snippet = snippet
            self.entry_type = "feedback"
            self.distance = 0.1
            self.file_path = f"/fake/{name}.md"

    def _fake_query(topic_hint, **kw):
        if raises:
            raise RuntimeError("chroma exploded")
        if not hits:
            return ()
        return tuple(_FakeHit(*h) for h in hits)

    monkeypatch.setattr(di, "query_doctrine", _fake_query)


# ──────────────────────────────────────────────────────────────────────
# Empty / missing — graceful ()
# ──────────────────────────────────────────────────────────────────────
def test_loaders_return_empty_when_sources_missing(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _patch_paths(monkeypatch, tmp_path)
    _patch_doctrine(monkeypatch)   # no hits
    assert sc._load_recent_summaries() == ()
    assert sc._load_deployed_sleeves() == ()
    assert sc._load_recent_events() == ()
    # Tier-2: empty topic_hint → () (cost discipline)
    assert sc._load_doctrine_snippets(topic_hint="") == ()


def test_build_synthesis_input_on_empty_world(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _patch_paths(monkeypatch, tmp_path)
    _patch_doctrine(monkeypatch)   # no hits
    si = sc.build_synthesis_input()
    assert si.recent_summaries == ()
    assert si.deployed_sleeves == ()
    assert si.recent_events == ()
    assert si.doctrine_snippets == ()
    assert si.snapshot_ts.endswith("Z")  # ISO Z-suffixed


# ──────────────────────────────────────────────────────────────────────
# Summaries — join + recency + ordering
# ──────────────────────────────────────────────────────────────────────
def test_summaries_join_cache_and_summaries(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    papers, _, _ = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": "2606.11111",
         "title": "EM bond carry refresh",
         "authors": ["Asness", "Moskowitz"],
         "fetched_ts": _ts(2)},
    ])
    _write_jsonl(papers / "summaries.jsonl", [
        {"source": "arxiv", "source_id": "2606.11111",
         "thesis": "Carry persists in EM sovereign",
         "testable_hypothesis": "CARRY/EM_SOV decile sort",
         "why_matters_for_us": "orthogonal to deployed",
         "risk_flags": ["data not free", "short sample"],
         "recommended_action": "INGEST",
         "summarized_ts": _ts(2)},
    ])
    out = sc._load_recent_summaries()
    assert len(out) == 1
    r = out[0]
    assert r.paper_id == "arxiv/2606.11111"
    assert r.title == "EM bond carry refresh"
    assert "Asness" in r.authors_short
    assert r.recommended_action == "INGEST"
    assert r.risk_flags_short == ("data not free", "short sample")


def test_summaries_recency_window_filters_old_rows(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    papers, _, _ = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": "new",
         "title": "fresh", "fetched_ts": _ts(2)},
        {"source": "arxiv", "source_id": "old",
         "title": "stale", "fetched_ts": _ts(30)},
    ])
    _write_jsonl(papers / "summaries.jsonl", [
        {"source": "arxiv", "source_id": "new",
         "thesis": "x", "recommended_action": "INGEST",
         "summarized_ts": _ts(2)},
        {"source": "arxiv", "source_id": "old",
         "thesis": "y", "recommended_action": "INGEST",
         "summarized_ts": _ts(30)},
    ])
    out = sc._load_recent_summaries(days=14)
    assert {r.paper_id for r in out} == {"arxiv/new"}


def test_summaries_skip_cache_rows_without_summary(monkeypatch, tmp_path):
    """A cache row without a summary should NOT enter synthesis input
    — we want the 5-field summary, not the raw abstract."""
    from engine.agents.papers_curator import synthesis_context as sc
    papers, _, _ = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": "only_cache",
         "title": "x", "fetched_ts": _ts(1)},
    ])
    # no summaries.jsonl written
    out = sc._load_recent_summaries()
    assert out == ()


def test_summaries_ordered_ingest_first(monkeypatch, tmp_path):
    """INGEST should sort before SKIP — synthesizer prompt prioritizes
    actionable candidates."""
    from engine.agents.papers_curator import synthesis_context as sc
    papers, _, _ = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": "skip_one",
         "title": "skip", "fetched_ts": _ts(1)},
        {"source": "arxiv", "source_id": "ingest_one",
         "title": "ingest", "fetched_ts": _ts(1)},
    ])
    _write_jsonl(papers / "summaries.jsonl", [
        {"source": "arxiv", "source_id": "skip_one",
         "thesis": "x", "recommended_action": "SKIP",
         "summarized_ts": _ts(1)},
        {"source": "arxiv", "source_id": "ingest_one",
         "thesis": "y", "recommended_action": "INGEST",
         "summarized_ts": _ts(1)},
    ])
    out = sc._load_recent_summaries()
    assert out[0].recommended_action == "INGEST"
    assert out[1].recommended_action == "SKIP"


def test_summaries_caps_at_max_rows(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    papers, _, _ = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": f"p{i}",
         "title": f"t{i}", "fetched_ts": _ts(1)}
        for i in range(50)
    ])
    _write_jsonl(papers / "summaries.jsonl", [
        {"source": "arxiv", "source_id": f"p{i}",
         "thesis": "x", "recommended_action": "INGEST",
         "summarized_ts": _ts(1)}
        for i in range(50)
    ])
    out = sc._load_recent_summaries(max_rows=10)
    assert len(out) == 10


# ──────────────────────────────────────────────────────────────────────
# Sleeves — DEPLOYED filter + kpi handling
# ──────────────────────────────────────────────────────────────────────
def test_sleeves_only_deployed(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, library, _ = _patch_paths(monkeypatch, tmp_path)
    library.mkdir(parents=True, exist_ok=True)
    (library / "deployed_one.yaml").write_text(
        "id: deployed_one\nfamily: CARRY\nstatus_in_our_book: DEPLOYED\n",
        encoding="utf-8")
    (library / "pending.yaml").write_text(
        "id: pending_one\nfamily: PEAD\nstatus_in_our_book: PENDING_DEPLOY\n",
        encoding="utf-8")
    out = sc._load_deployed_sleeves()
    assert len(out) == 1
    assert out[0].sleeve_id == "deployed_one"
    assert out[0].family == "CARRY"


def test_sleeves_skip_underscore_files(monkeypatch, tmp_path):
    """Files starting with _ are doctrine/audit not sleeves."""
    from engine.agents.papers_curator import synthesis_context as sc
    _, library, _ = _patch_paths(monkeypatch, tmp_path)
    library.mkdir(parents=True, exist_ok=True)
    (library / "_schema.yaml").write_text(
        "id: _schema\nstatus_in_our_book: DEPLOYED\n", encoding="utf-8")
    (library / "real.yaml").write_text(
        "id: real\nfamily: X\nstatus_in_our_book: DEPLOYED\n",
        encoding="utf-8")
    out = sc._load_deployed_sleeves()
    assert {s.sleeve_id for s in out} == {"real"}


def test_sleeves_kpi_optional(monkeypatch, tmp_path):
    """A DEPLOYED sleeve without live_kpi block should still load,
    with sharpe/months/decay set to None."""
    from engine.agents.papers_curator import synthesis_context as sc
    _, library, _ = _patch_paths(monkeypatch, tmp_path)
    library.mkdir(parents=True, exist_ok=True)
    (library / "nokpi.yaml").write_text(
        "id: nokpi\nfamily: X\nstatus_in_our_book: DEPLOYED\n",
        encoding="utf-8")
    out = sc._load_deployed_sleeves()
    assert len(out) == 1
    s = out[0]
    assert s.ann_sharpe_live is None
    assert s.months_since_deploy is None
    assert s.last_decay_alert is None


def test_sleeves_kpi_parsed_when_present(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, library, _ = _patch_paths(monkeypatch, tmp_path)
    library.mkdir(parents=True, exist_ok=True)
    (library / "carry.yaml").write_text(
        "id: carry_g10\n"
        "family: CARRY\n"
        "status_in_our_book: DEPLOYED\n"
        "live_kpi:\n"
        "  ann_sharpe_live: 0.83\n"
        "  months_since_deploy: 7\n"
        "  last_decay_alert_ts: 2026-05-12T00:00:00Z\n",
        encoding="utf-8")
    out = sc._load_deployed_sleeves()
    assert len(out) == 1
    s = out[0]
    assert s.ann_sharpe_live == 0.83
    assert s.months_since_deploy == 7
    assert s.last_decay_alert == "2026-05-12 00:00:00+00:00" \
        or s.last_decay_alert.startswith("2026-05-12")


# ──────────────────────────────────────────────────────────────────────
# Events — relevance + recency + ordering
# ──────────────────────────────────────────────────────────────────────
def test_events_filter_to_relevant_types(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, _, events = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(events, [
        {"event_id": "a", "event_type": "factor_verdict_filed",
         "subject_id": "foo", "family": "PEAD", "verdict": "GREEN",
         "summary": "deploys", "ts": _ts(2)},
        {"event_id": "b", "event_type": "session_started",
         "subject_id": "ses1", "family": "—", "verdict": "—",
         "summary": "opens", "ts": _ts(2)},
        {"event_id": "c", "event_type": "decay_alert",
         "subject_id": "qmj", "family": "QUALITY", "verdict": "AMBER",
         "summary": "soft", "ts": _ts(3)},
    ])
    out = sc._load_recent_events()
    assert {e.event_id for e in out} == {"a", "c"}


def test_events_recency_window(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, _, events = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(events, [
        {"event_id": "fresh", "event_type": "factor_verdict_filed",
         "subject_id": "x", "ts": _ts(5)},
        {"event_id": "stale", "event_type": "factor_verdict_filed",
         "subject_id": "y", "ts": _ts(90)},
    ])
    out = sc._load_recent_events(days=30)
    assert {e.event_id for e in out} == {"fresh"}


def test_events_newest_first(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, _, events = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(events, [
        {"event_id": "older", "event_type": "factor_verdict_filed",
         "ts": _ts(10)},
        {"event_id": "newer", "event_type": "factor_verdict_filed",
         "ts": _ts(1)},
    ])
    out = sc._load_recent_events()
    assert out[0].event_id == "newer"
    assert out[1].event_id == "older"


def test_events_cap_at_max_rows(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    _, _, events = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(events, [
        {"event_id": f"e{i}", "event_type": "factor_verdict_filed",
         "ts": _ts(1)}
        for i in range(100)
    ])
    out = sc._load_recent_events(max_rows=20)
    assert len(out) == 20


# ──────────────────────────────────────────────────────────────────────
# Malformed lines tolerated
# ──────────────────────────────────────────────────────────────────────
def test_malformed_jsonl_line_skipped_not_raises(monkeypatch, tmp_path):
    """One bad line should not kill the whole gather."""
    from engine.agents.papers_curator import synthesis_context as sc
    _, _, events = _patch_paths(monkeypatch, tmp_path)
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        '{"event_id": "good", "event_type": "factor_verdict_filed", "ts": "%s"}\n'
        'this is not json at all\n'
        '{"event_id": "good2", "event_type": "decay_alert", "ts": "%s"}\n'
        % (_ts(1), _ts(1)),
        encoding="utf-8",
    )
    out = sc._load_recent_events()
    assert {e.event_id for e in out} == {"good", "good2"}


# ──────────────────────────────────────────────────────────────────────
# Top-level: full snapshot wired
# ──────────────────────────────────────────────────────────────────────
def test_build_synthesis_input_full_wire(monkeypatch, tmp_path):
    from engine.agents.papers_curator import synthesis_context as sc
    papers, library, events = _patch_paths(monkeypatch, tmp_path)
    _write_jsonl(papers / "cache.jsonl", [
        {"source": "arxiv", "source_id": "x1",
         "title": "EM carry", "fetched_ts": _ts(2)}
    ])
    _write_jsonl(papers / "summaries.jsonl", [
        {"source": "arxiv", "source_id": "x1",
         "thesis": "carry", "recommended_action": "INGEST",
         "summarized_ts": _ts(2)}
    ])
    library.mkdir(parents=True, exist_ok=True)
    (library / "c.yaml").write_text(
        "id: cross_asset_carry\nfamily: CARRY\nstatus_in_our_book: DEPLOYED\n",
        encoding="utf-8")
    _write_jsonl(events, [
        {"event_id": "ev1", "event_type": "factor_verdict_filed",
         "subject_id": "foo", "family": "PEAD",
         "verdict": "RED", "summary": "decays", "ts": _ts(3)}
    ])
    _patch_doctrine(monkeypatch, hits=[
        ("project-carry-doctrine-2026", "Carry sleeve sizing rule",
         "Vol-target 6% on cross-asset carry."),
    ])
    si = sc.build_synthesis_input()
    assert len(si.recent_summaries) == 1
    assert len(si.deployed_sleeves) == 1
    assert len(si.recent_events) == 1
    assert len(si.doctrine_snippets) == 1
    assert si.doctrine_snippets[0].memory_file_id == "project-carry-doctrine-2026"
    assert si.snapshot_ts.endswith("Z")


# ──────────────────────────────────────────────────────────────────────
# Tier-2 Q2: topic_hint composition + doctrine wiring
# ──────────────────────────────────────────────────────────────────────
def test_topic_hint_includes_paper_theses(monkeypatch, tmp_path):
    from engine.agents.papers_curator.synthesis_context import (
        _build_topic_hint,
    )
    from engine.agents.papers_curator.synthesis import (
        PaperSummaryRef, SleeveStateRef, RecentEventRef,
    )
    summaries = (
        PaperSummaryRef(
            paper_id="p1", title="t1", authors_short="x",
            thesis="EM sovereign carry refresh",
            testable_hypothesis="x", why_matters_for_us="x",
            risk_flags_short=(), recommended_action="INGEST",
        ),
    )
    hint = _build_topic_hint(summaries, (), ())
    assert "EM sovereign carry refresh" in hint


def test_topic_hint_includes_deployed_families(monkeypatch):
    from engine.agents.papers_curator.synthesis_context import (
        _build_topic_hint,
    )
    from engine.agents.papers_curator.synthesis import SleeveStateRef
    sleeves = (
        SleeveStateRef(sleeve_id="cross_asset_carry", family="CARRY",
                        status="DEPLOYED", ann_sharpe_live=None,
                        months_since_deploy=None, last_decay_alert=None),
        SleeveStateRef(sleeve_id="ts_momentum", family="MOMENTUM",
                        status="DEPLOYED", ann_sharpe_live=None,
                        months_since_deploy=None, last_decay_alert=None),
    )
    hint = _build_topic_hint((), sleeves, ())
    assert "CARRY" in hint
    assert "MOMENTUM" in hint
    assert "Deployed families" in hint


def test_topic_hint_includes_flagged_event_families(monkeypatch):
    """Only doctrine_signal_detected + decay_alert events surface their
    family — NOT factor_verdict_filed (those are too numerous to use
    as topic anchor)."""
    from engine.agents.papers_curator.synthesis_context import (
        _build_topic_hint,
    )
    from engine.agents.papers_curator.synthesis import RecentEventRef
    events = (
        RecentEventRef(event_id="e1", event_type="doctrine_signal_detected",
                        subject_id="x", family="EARNINGS_DRIFT",
                        verdict="MARGINAL", summary="cluster", ts=_ts(1)),
        RecentEventRef(event_id="e2", event_type="decay_alert",
                        subject_id="x", family="CARRY",
                        verdict="MARGINAL", summary="decay", ts=_ts(1)),
        RecentEventRef(event_id="e3", event_type="factor_verdict_filed",
                        subject_id="x", family="MOMENTUM",
                        verdict="RED", summary="x", ts=_ts(1)),
    )
    hint = _build_topic_hint((), (), events)
    assert "EARNINGS_DRIFT" in hint
    assert "CARRY" in hint
    assert "MOMENTUM" not in hint   # factor_verdict_filed excluded


def test_topic_hint_empty_when_no_inputs():
    from engine.agents.papers_curator.synthesis_context import (
        _build_topic_hint,
    )
    assert _build_topic_hint((), (), ()) == ""


def test_doctrine_snippets_empty_when_topic_hint_blank(monkeypatch):
    """Cost discipline: blank topic → don't fire chroma."""
    from engine.agents.papers_curator import synthesis_context as sc
    _patch_doctrine(monkeypatch, hits=[
        ("would-not-be-called", "x", "x")
    ])
    assert sc._load_doctrine_snippets(topic_hint="") == ()
    assert sc._load_doctrine_snippets(topic_hint="   ") == ()


def test_doctrine_snippets_returns_empty_on_chroma_failure(monkeypatch):
    """Chroma down → A degrades to no-doctrine, doesn't crash."""
    from engine.agents.papers_curator import synthesis_context as sc
    _patch_doctrine(monkeypatch, raises=True)
    assert sc._load_doctrine_snippets(topic_hint="real topic") == ()


def test_doctrine_snippets_adapts_index_hit_to_synthesis_hit(monkeypatch):
    """doctrine_index.DoctrineHit and synthesis.DoctrineHit have
    different field names — the adapter must map correctly."""
    from engine.agents.papers_curator import synthesis_context as sc
    _patch_doctrine(monkeypatch, hits=[
        ("feedback-carry-2026", "Carry rule",
         "Vol-target 6%. Politis-Romano bootstrap calibration..."),
    ])
    hits = sc._load_doctrine_snippets(topic_hint="cross-asset carry")
    assert len(hits) == 1
    h = hits[0]
    assert h.memory_file_id == "feedback-carry-2026"
    assert h.headline == "Carry rule"
    assert "Vol-target" in h.snippet
