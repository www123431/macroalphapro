"""Tests for source_health tracker + arxiv RSS fallback."""
from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest

from engine.data import source_health


@pytest.fixture(autouse=True)
def isolated_health_file(tmp_path, monkeypatch):
    monkeypatch.setattr(source_health, "HEALTH_FILE",
                          tmp_path / "source_health.json")
    yield


# ── source_health basics ────────────────────────────────────────────────

def test_unrecorded_source_is_healthy():
    healthy, reason = source_health.is_healthy("brand_new_source")
    assert healthy is True
    assert reason is None


def test_mark_failure_sets_cooldown():
    source_health.mark_failure("arxiv_api", "rate_limited", "test 429")
    healthy, reason = source_health.is_healthy("arxiv_api")
    assert healthy is False
    assert "rate_limited" in reason
    assert "cooldown" in reason


def test_mark_success_clears_record():
    source_health.mark_failure("arxiv_api", "rate_limited", "test")
    source_health.mark_success("arxiv_api")
    healthy, _ = source_health.is_healthy("arxiv_api")
    assert healthy is True


def test_cooldown_expires_auto_clears(monkeypatch):
    """When cooldown expires, source becomes healthy + record auto-cleared."""
    source_health.mark_failure("test_src", "network", "test")
    # Forcibly age the record by editing it
    state = source_health._read_state()
    state["test_src"]["cooldown_until_ts"] = (
        datetime.datetime.utcnow() - datetime.timedelta(minutes=1)
    ).isoformat(timespec="seconds") + "Z"
    source_health._write_state(state)
    # Now is_healthy should return True AND clear the entry
    healthy, _ = source_health.is_healthy("test_src")
    assert healthy is True
    # Re-read state — entry should be gone
    state_after = source_health._read_state()
    assert "test_src" not in state_after


def test_repeated_failures_exponential_backoff():
    """Consecutive failures double the cooldown."""
    source_health.mark_failure("flaky", "network", "1st")
    state1 = source_health._read_state()["flaky"]
    cooldown_1 = state1["cooldown_minutes"]

    source_health.mark_failure("flaky", "network", "2nd")
    state2 = source_health._read_state()["flaky"]
    cooldown_2 = state2["cooldown_minutes"]

    assert state2["consecutive_failures"] == 2
    assert cooldown_2 > cooldown_1


def test_auth_missing_no_cooldown():
    """auth_missing has 0 cooldown — user fixes credentials immediately."""
    source_health.mark_failure("paid_src", "auth_missing", "no key")
    healthy, _ = source_health.is_healthy("paid_src")
    assert healthy is True    # not tracked


def test_list_unhealthy_returns_currently_blocked():
    source_health.mark_failure("a", "rate_limited", "x")
    source_health.mark_failure("b", "access_denied", "y")
    source_health.mark_success("a")
    unhealthy = source_health.list_unhealthy()
    assert "a" not in unhealthy
    assert "b" in unhealthy


def test_clear_all_removes_everything():
    source_health.mark_failure("x", "rate_limited", "test")
    source_health.clear_all()
    healthy, _ = source_health.is_healthy("x")
    assert healthy is True


# ── arxiv fetcher uses source_health ────────────────────────────────────

def test_arxiv_fetcher_skips_when_unhealthy(monkeypatch):
    """arxiv fetcher consults source_health first; returns empty if unhealthy."""
    from engine.research.discovery import arxiv_qfin_fetcher
    source_health.mark_failure("arxiv_api", "rate_limited", "test")
    # Now call fetch — should return empty without hitting network
    df = arxiv_qfin_fetcher.fetch_qfin_papers(
        "2024-01-01", "2024-01-31", max_results=5,
    )
    assert df.empty


def test_arxiv_fetcher_skip_health_check_bypasses(monkeypatch):
    """skip_health_check=True bypasses (for tests)."""
    from engine.research.discovery import arxiv_qfin_fetcher
    source_health.mark_failure("arxiv_api", "rate_limited", "test")

    captured = []
    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            captured.append(url)
            class _R:
                status_code = 200
                content = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(arxiv_qfin_fetcher, "POLITE_DELAY_SEC", 0)
    arxiv_qfin_fetcher.fetch_qfin_papers(
        "2024-01-01", "2024-01-31", max_results=5,
        skip_health_check=True,
    )
    assert len(captured) >= 1    # network was attempted


def test_arxiv_fetcher_marks_unhealthy_on_429(monkeypatch):
    """When API returns 429, source_health gets marked."""
    from engine.research.discovery import arxiv_qfin_fetcher

    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            class _R:
                status_code = 429
                content = b""
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(arxiv_qfin_fetcher, "POLITE_DELAY_SEC", 0)
    df = arxiv_qfin_fetcher.fetch_qfin_papers(
        "2024-01-01", "2024-01-31", max_results=5,
        skip_health_check=True,
    )
    assert df.empty
    # source_health should now have arxiv_api marked
    healthy, _ = source_health.is_healthy("arxiv_api")
    assert healthy is False


# ── RSS fallback ────────────────────────────────────────────────────────

SAMPLE_RSS = b"""<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns="http://purl.org/rss/1.0/"
          xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>q-fin.PR</title>
  </channel>
  <item>
    <title>Foo Mechanism. (arXiv:2401.99999v1 [q-fin.PR])</title>
    <description>This paper describes a foo factor.</description>
    <link>http://arxiv.org/abs/2401.99999v1</link>
    <dc:creator>Researcher A; Researcher B</dc:creator>
  </item>
</rdf:RDF>"""


def test_rss_fetcher_parses_items(monkeypatch):
    """RSS fallback should parse item format correctly."""
    from engine.research.discovery import arxiv_qfin_fetcher

    class _FakeSession:
        def __init__(self): self.headers = {}
        def get(self, url, **kw):
            class _R:
                status_code = 200
                content = SAMPLE_RSS
            return _R()
    monkeypatch.setattr("requests.Session", lambda: _FakeSession())
    monkeypatch.setattr("time.sleep", lambda x: None)
    monkeypatch.setattr(arxiv_qfin_fetcher, "POLITE_DELAY_SEC", 0)
    df = arxiv_qfin_fetcher.fetch_qfin_rss(categories=["q-fin.PR"])
    assert not df.empty
    assert df.iloc[0]["arxiv_id"] == "2401.99999v1"
    assert "Foo Mechanism" in df.iloc[0]["title"]
    assert "foo factor" in df.iloc[0]["abstract"]


def test_fetch_with_fallback_uses_rss_when_api_empty(monkeypatch):
    """When API returns empty, fetch_qfin_with_fallback tries RSS."""
    from engine.research.discovery import arxiv_qfin_fetcher

    # API returns empty
    monkeypatch.setattr(
        arxiv_qfin_fetcher, "fetch_qfin_papers",
        lambda *args, **kw: pd.DataFrame(),
    )
    # RSS returns 1 paper
    def _mock_rss(**kw):
        return pd.DataFrame([{
            "arxiv_id": "rss_id", "title": "RSS Paper",
            "authors": "x", "abstract": "y",
            "categories": "q-fin.PR",
            "submitted_date": None, "updated_date": None,
            "pdf_url": None, "abs_url": None,
        }])
    monkeypatch.setattr(arxiv_qfin_fetcher, "fetch_qfin_rss", _mock_rss)

    df = arxiv_qfin_fetcher.fetch_qfin_with_fallback(
        "2024-01-01", "2024-01-31", max_results=5,
    )
    assert not df.empty
    assert df.iloc[0]["arxiv_id"] == "rss_id"
