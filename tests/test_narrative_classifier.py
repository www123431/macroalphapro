"""
tests/test_narrative_classifier.py — W3 D2a + D2b unit tests (2026-05-08).

Pre-registration: docs/spec_multivariate_msm_v4_narrative.md §2.7
Single-source amendment: 2026-05-08 (FOMC-only; ECB dropped, see spec §2.7.6).

Coverage matrix:
  • Lexicon lock      : 24 + 24 entries, uniqueness, expected words present.
  • compute_raw_score : hawkish-only / dovish-only / neutral / mixed / empty.
  • compute_narrative_score : raises before (μ, σ) lock; correct after.
  • aggregate_monthly : in-window mean, forward-fill, NaN at series start.
  • aggregate_monthly_series : month-end wrapper carries forward correctly.
  • _strip_html       : tag stripping, entity decode, whitespace collapse.
  • fetch_fomc_press_statement : type check + URL construction + parse path
                                 (mocked HTTP) + live network sanity (skip
                                 if offline).
  • LLM tag stub      : returns valid enum ('other' placeholder for D2a).
"""
from __future__ import annotations

import datetime
import math
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from engine import narrative_classifier as nc


# ─────────────────────────────────────────────────────────────────────────────
# Lexicon lock tests (spec §2.7.1)
# ─────────────────────────────────────────────────────────────────────────────


def test_hawkish_lexicon_locked():
    """24 entries, unique, includes literature-anchor terms."""
    assert len(nc.HAWKISH_WORDS_V4) == 24
    assert len(set(nc.HAWKISH_WORDS_V4)) == 24
    # spot-check signature terms from HM 2016 + ABG 2022
    for term in ("tighten", "hike", "raise", "inflationary", "restrictive"):
        assert term in nc.HAWKISH_WORDS_V4


def test_dovish_lexicon_locked():
    """24 entries, unique, includes literature-anchor terms."""
    assert len(nc.DOVISH_WORDS_V4) == 24
    assert len(set(nc.DOVISH_WORDS_V4)) == 24
    for term in ("accommodative", "easing", "patient", "cut", "support"):
        assert term in nc.DOVISH_WORDS_V4


def test_lexicons_disjoint():
    """No word appears in both hawkish and dovish lists."""
    overlap = set(nc.HAWKISH_WORDS_V4) & set(nc.DOVISH_WORDS_V4)
    assert overlap == set(), f"unexpected lexicon overlap: {overlap}"


# ─────────────────────────────────────────────────────────────────────────────
# compute_raw_score (spec §2.7.2)
# ─────────────────────────────────────────────────────────────────────────────


def test_raw_score_hawkish_only():
    """Pure hawkish text yields positive raw score."""
    text = "The Committee will tighten policy and hike rates further increase."
    raw = nc.compute_raw_score(text)
    assert raw > 0.0, f"hawkish text should have positive score, got {raw}"


def test_raw_score_dovish_only():
    """Pure dovish text yields negative raw score."""
    text = "The Committee remains accommodative and patient with easing supportive stance."
    raw = nc.compute_raw_score(text)
    assert raw < 0.0, f"dovish text should have negative score, got {raw}"


def test_raw_score_neutral_yields_zero():
    """Text with no lexicon hits → 0.0."""
    text = "The meeting reviewed banana republic data over the quarter."
    raw = nc.compute_raw_score(text)
    assert raw == 0.0


def test_raw_score_balanced_cancels():
    """Equal hawkish and dovish counts cancel to zero."""
    # 1 hawkish ("hike") + 1 dovish ("cut")
    text = "Some members would hike and others would cut according to data."
    raw = nc.compute_raw_score(text)
    assert raw == 0.0


def test_raw_score_empty_text():
    """Empty / blank → 0.0 (no division by zero)."""
    assert nc.compute_raw_score("") == 0.0
    assert nc.compute_raw_score("   \n\t  ") == 0.0


def test_raw_score_case_insensitive():
    """Case should not matter."""
    a = nc.compute_raw_score("Tighten and HIKE")
    b = nc.compute_raw_score("tighten and hike")
    assert a == b


def test_raw_score_multi_word_phrase_counts():
    """'inflation pressure' should match as a phrase, not require both words."""
    text = "The rising inflation pressure remains a concern for the committee."
    # phrases hit: "inflation pressure", "concern" → 2 hawkish; 0 dovish
    # "raise" not present; counting substring: nothing else matches
    raw = nc.compute_raw_score(text)
    assert raw > 0.0


def test_raw_score_type_check():
    """Non-string input raises."""
    with pytest.raises(TypeError):
        nc.compute_raw_score(42)
    with pytest.raises(TypeError):
        nc.compute_raw_score(None)


# ─────────────────────────────────────────────────────────────────────────────
# compute_narrative_score (spec §2.7.2 z-norm)
# ─────────────────────────────────────────────────────────────────────────────


def test_narrative_score_locked_post_d3():
    """After W3 D3 (2026-05-08) the in-sample (μ, σ) are locked. Confirm the
    locked values are populated and `compute_narrative_score` is callable.
    Per spec §6 these values must NEVER change again post-lock; this test
    serves as a tripwire if accidental re-locks happen."""
    assert nc._RAW_SCORE_INSAMPLE_MEAN is not None
    assert nc._RAW_SCORE_INSAMPLE_STD is not None
    assert nc._RAW_SCORE_INSAMPLE_STD > 0.0
    # Spot-check that values are in the expected magnitude range for a
    # per-word ratio (raw_score ∈ ~[-0.05, +0.05]).
    assert abs(nc._RAW_SCORE_INSAMPLE_MEAN) < 0.05
    assert nc._RAW_SCORE_INSAMPLE_STD < 0.05
    # Callable end-to-end:
    z = nc.compute_narrative_score("The Committee remains accommodative.")
    assert isinstance(z, float)


def test_narrative_score_raises_if_unlock_attempted(monkeypatch):
    """If μ/σ are reset to None (forbidden per spec §6), the score function
    must refuse to run. Guards against accidental re-unlock."""
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_MEAN", None)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_STD", None)
    with pytest.raises(RuntimeError, match="not yet locked"):
        nc.compute_narrative_score("any text")


def test_narrative_score_after_mocked_lock(monkeypatch):
    """With (μ, σ) locked, z-score formula is applied correctly."""
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_MEAN", 0.001)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_STD", 0.002)
    text = "The Committee will tighten policy."  # hawkish
    raw = nc.compute_raw_score(text)
    expected = (raw - 0.001) / 0.002
    actual = nc.compute_narrative_score(text)
    assert math.isclose(actual, expected, rel_tol=1e-9)


def test_narrative_score_zero_sigma_raises(monkeypatch):
    """σ ≤ 0 is degenerate; must raise."""
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_MEAN", 0.0)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_STD", 0.0)
    with pytest.raises(RuntimeError, match="σ must be > 0"):
        nc.compute_narrative_score("text")


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_monthly (spec §2.7.4 forward-fill)
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_in_window_mean():
    """Meetings inside [t-30, t] return their mean."""
    scores = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2024-01-15", "2024-01-25", "2024-02-05"]),
    )
    # window for 2024-02-10 is [2024-01-11, 2024-02-10] → all 3 meetings in
    val = nc.aggregate_monthly(scores, datetime.date(2024, 2, 10))
    assert math.isclose(val, 2.0)


def test_aggregate_forward_fill_when_no_meetings():
    """No meetings in window → forward-fill prior_value."""
    scores = pd.Series(
        [1.5],
        index=pd.to_datetime(["2024-01-15"]),
    )
    # window for 2024-04-30 is [2024-03-31, 2024-04-30] → no meetings
    val = nc.aggregate_monthly(scores, datetime.date(2024, 4, 30), prior_value=0.7)
    assert val == 0.7


def test_aggregate_first_month_no_prior_returns_nan():
    """No meetings + no prior → NaN (start of series, before first meeting)."""
    scores = pd.Series(
        [1.5],
        index=pd.to_datetime(["2024-06-15"]),
    )
    val = nc.aggregate_monthly(scores, datetime.date(2024, 1, 31), prior_value=None)
    assert math.isnan(val)


def test_aggregate_only_in_window_meetings_used():
    """Out-of-window meetings ignored; only window mean used."""
    scores = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.to_datetime(["2024-01-01", "2024-02-15", "2024-04-01"]),
    )
    # window for 2024-03-01 is [2024-01-31, 2024-03-01] → only 2024-02-15 (=20)
    val = nc.aggregate_monthly(scores, datetime.date(2024, 3, 1), prior_value=99.0)
    assert math.isclose(val, 20.0)


def test_aggregate_empty_series():
    """Empty meeting series with prior → forward-fill; without → NaN."""
    empty = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    assert nc.aggregate_monthly(empty, datetime.date(2024, 1, 31), prior_value=0.5) == 0.5
    assert math.isnan(nc.aggregate_monthly(empty, datetime.date(2024, 1, 31)))


def test_aggregate_type_check():
    with pytest.raises(TypeError):
        nc.aggregate_monthly([1.0, 2.0], datetime.date(2024, 1, 31))


# ─────────────────────────────────────────────────────────────────────────────
# aggregate_monthly_series (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────


def test_aggregate_series_carries_forward():
    """Wrapper carries last-known value through gap months."""
    scores = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2024-01-15", "2024-03-15", "2024-06-15"]),
    )
    month_ends = pd.date_range("2024-01-31", "2024-06-30", freq="ME")
    out = nc.aggregate_monthly_series(scores, month_ends)
    assert len(out) == 6
    # 2024-01: meeting on 1/15 in window → 1.0
    assert math.isclose(out.iloc[0], 1.0)
    # 2024-02: no meeting in [1/31..2/29] → forward-fill 1.0
    assert math.isclose(out.iloc[1], 1.0)
    # 2024-03: meeting on 3/15 in [2/29..3/31] → 2.0
    assert math.isclose(out.iloc[2], 2.0)
    # 2024-04: no meeting → forward-fill 2.0
    assert math.isclose(out.iloc[3], 2.0)
    # 2024-05: no meeting → forward-fill 2.0
    assert math.isclose(out.iloc[4], 2.0)
    # 2024-06: meeting on 6/15 in [5/31..6/30] → 3.0
    assert math.isclose(out.iloc[5], 3.0)


def test_aggregate_series_initial_nan_until_first_meeting():
    """Months before first meeting yield NaN (no prior to forward-fill)."""
    scores = pd.Series([5.0], index=pd.to_datetime(["2024-04-15"]))
    month_ends = pd.date_range("2024-01-31", "2024-04-30", freq="ME")
    out = nc.aggregate_monthly_series(scores, month_ends)
    assert math.isnan(out.iloc[0])
    assert math.isnan(out.iloc[1])
    assert math.isnan(out.iloc[2])
    # Apr: meeting in [3/31..4/30] → 5.0
    assert math.isclose(out.iloc[3], 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# _strip_html (D2b)
# ─────────────────────────────────────────────────────────────────────────────


def test_strip_html_removes_script_and_style():
    html = (
        "<html><head><style>a{color:red}</style>"
        "<script>alert('x')</script></head>"
        "<body><p>Committee will tighten policy</p></body></html>"
    )
    text = nc._strip_html(html)
    assert "alert" not in text
    assert "color:red" not in text
    assert "tighten policy" in text


def test_strip_html_decodes_entities_and_collapses_whitespace():
    html = "<p>The&nbsp;Committee&nbsp;remains  patient&mdash;data&hellip;</p>"
    text = nc._strip_html(html)
    assert "&nbsp;" not in text
    assert "&mdash;" not in text
    assert "Committee remains patient" in text
    # collapsed whitespace
    assert "  " not in text


def test_strip_html_type_check():
    with pytest.raises(TypeError):
        nc._strip_html(123)


def test_strip_html_handles_real_fed_like_markup():
    """Smoke check that the stripper handles a Fed-style snippet."""
    html = """
        <div class="col-xs-12 col-sm-8 col-md-8">
          <h3 class="title">Federal Reserve issues FOMC statement</h3>
          <p>Recent indicators suggest that economic activity has been
             expanding at a solid pace.</p>
          <p>In support of these goals, the Committee decided to
             <em>raise</em> the target range for the federal funds rate.</p>
        </div>
    """
    text = nc._strip_html(html)
    assert "FOMC statement" in text
    assert "raise" in text
    assert "target range" in text


# ─────────────────────────────────────────────────────────────────────────────
# fetch_fomc_press_statement (D2b)
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_fomc_type_check():
    with pytest.raises(TypeError):
        nc.fetch_fomc_press_statement("2024-01-31")


def test_fetch_fomc_template_count_matches_4_eras():
    """4 distinct URL eras documented in module docstring; templates tuple
    must contain at least one entry per era (some eras use 2 variants)."""
    assert len(nc._FOMC_URL_TEMPLATES) >= 4
    joined = " ".join(nc._FOMC_URL_TEMPLATES)
    # Era 4 (2008+)
    assert "newsevents/pressreleases/monetary{ymd}a.htm" in joined
    # Era 3 (2003-2007)
    assert "boarddocs/press/monetary/{yyyy}/{ymd}" in joined
    # Era 2 (1996-2002)
    assert "boarddocs/press/general/{yyyy}/{ymd}/default.htm" in joined
    # Era 1 (1994-1995)
    assert "fomc/{ymd}default.htm" in joined


def test_fetch_fomc_era4_first_attempt_wins(monkeypatch):
    """When the primary (most-recent era 4) URL returns a healthy body,
    fetcher must NOT call any fallback templates."""
    calls = []
    healthy_body = "<p>" + " ".join(
        ["The Committee decided to raise the target range"] + ["word"] * 100
    ) + "</p>"

    def fake_fetch(url, *, timeout):
        calls.append(url)
        if "newsevents/pressreleases" in url:
            return healthy_body
        raise AssertionError(f"unexpected fallback call: {url}")

    monkeypatch.setattr(nc, "_fetch_url", fake_fetch)
    text = nc.fetch_fomc_press_statement(datetime.date(2024, 1, 31))
    assert "raise" in text
    assert len(calls) == 1
    assert "monetary20240131a.htm" in calls[0]


def test_fetch_fomc_falls_back_through_eras_until_match(monkeypatch):
    """
    Era 1 (pre-1996) date: every newer pattern must 404 in turn before the
    Era 1 URL succeeds. Verifies the cascading fallback walks all 4 eras.
    """
    import urllib.error
    calls = []
    healthy_body = "<p>" + " ".join(
        ["The Committee decided"] + ["text"] * 100
    ) + "</p>"

    def fake_fetch(url, *, timeout):
        calls.append(url)
        # Only the era-1 URL succeeds; everything else 404s.
        if url.endswith("/fomc/19940517default.htm"):
            return healthy_body
        raise urllib.error.HTTPError(
            url, 404, "Not Found", hdrs={}, fp=None,
        )

    monkeypatch.setattr(nc, "_fetch_url", fake_fetch)
    text = nc.fetch_fomc_press_statement(datetime.date(1994, 5, 17))
    assert "Committee" in text
    # Walked through all templates; final win on era-1.
    assert len(calls) == len(nc._FOMC_URL_TEMPLATES)
    assert calls[-1].endswith("/fomc/19940517default.htm")


def test_fetch_fomc_short_body_triggers_next_era(monkeypatch):
    """A 200 with too-short body must NOT count as a hit; fallback continues."""
    calls = []

    def fake_fetch(url, *, timeout):
        calls.append(url)
        if "newsevents/pressreleases" in url:
            return "<html><body>Page not found</body></html>"  # ~3 words
        # First fallback returns plausibly long body
        return "<p>" + " ".join(["text"] * 100) + "</p>"

    monkeypatch.setattr(nc, "_fetch_url", fake_fetch)
    out = nc.fetch_fomc_press_statement(datetime.date(2005, 5, 3))
    assert len(calls) == 2
    assert len(out.split()) >= 50


def test_fetch_fomc_propagates_url_error_after_all_eras_fail(monkeypatch):
    """If every era 404s/URLErrors, the last error should propagate."""
    import urllib.error

    def fake_fetch(url, *, timeout):
        raise urllib.error.URLError(f"network down for {url}")

    monkeypatch.setattr(nc, "_fetch_url", fake_fetch)
    with pytest.raises(urllib.error.URLError):
        nc.fetch_fomc_press_statement(datetime.date(2023, 11, 1))


def test_fetch_fomc_return_url_includes_winning_template(monkeypatch):
    """`return_url=True` returns (text, url); url is the template that won."""
    body = "<p>" + " ".join(["text"] * 100) + "</p>"

    def fake_fetch(url, *, timeout):
        return body

    monkeypatch.setattr(nc, "_fetch_url", fake_fetch)
    text, url = nc.fetch_fomc_press_statement(
        datetime.date(2024, 1, 31), return_url=True
    )
    assert isinstance(text, str)
    assert "monetary20240131a.htm" in url


# ── Live network sanity tests (one per era; skip by default) ─────────────────


@pytest.mark.skipif(
    os.environ.get("MAP_RUN_NETWORK_TESTS", "0") != "1",
    reason="Live FOMC fetch — set MAP_RUN_NETWORK_TESTS=1 to run.",
)
def test_fetch_fomc_live_era4_2024():
    """Era 4 (2008-2024+): 2024-01-31 FOMC statement."""
    text = nc.fetch_fomc_press_statement(datetime.date(2024, 1, 31))
    assert len(text.split()) >= 200
    lower = text.lower()
    assert "committee" in lower
    assert "federal funds rate" in lower or "target range" in lower


@pytest.mark.skipif(
    os.environ.get("MAP_RUN_NETWORK_TESTS", "0") != "1",
    reason="Live FOMC fetch — set MAP_RUN_NETWORK_TESTS=1 to run.",
)
def test_fetch_fomc_live_era3_2003():
    """Era 3 (2003-2007): 2003-06-25 FOMC statement (boarddocs/press/monetary path)."""
    text, url = nc.fetch_fomc_press_statement(
        datetime.date(2003, 6, 25), return_url=True
    )
    assert "boarddocs/press/monetary" in url, f"expected era 3 path, got {url}"
    assert len(text.split()) >= 50


@pytest.mark.skipif(
    os.environ.get("MAP_RUN_NETWORK_TESTS", "0") != "1",
    reason="Live FOMC fetch — set MAP_RUN_NETWORK_TESTS=1 to run.",
)
def test_fetch_fomc_live_era2_1999():
    """Era 2 (1996-2002): 1999-06-30 FOMC statement (boarddocs/press/general path)."""
    text, url = nc.fetch_fomc_press_statement(
        datetime.date(1999, 6, 30), return_url=True
    )
    assert "boarddocs/press/general" in url, f"expected era 2 path, got {url}"
    assert len(text.split()) >= 50


@pytest.mark.skipif(
    os.environ.get("MAP_RUN_NETWORK_TESTS", "0") != "1",
    reason="Live FOMC fetch — set MAP_RUN_NETWORK_TESTS=1 to run.",
)
def test_fetch_fomc_live_era1_1994():
    """Era 1 (1994-1995): 1994-05-17 FOMC statement (/fomc/{YMD}default.htm)."""
    text, url = nc.fetch_fomc_press_statement(
        datetime.date(1994, 5, 17), return_url=True
    )
    assert url.endswith("/fomc/19940517default.htm"), f"expected era 1 path, got {url}"
    assert len(text.split()) >= 30  # era-1 statements are SHORT (~150 words)


# ─────────────────────────────────────────────────────────────────────────────
# LLM tag stub
# ─────────────────────────────────────────────────────────────────────────────


def test_llm_tag_returns_valid_enum():
    """D2a stub returns 'other'; D2b will hook to Gemini. Result must be in enum."""
    tag = nc.llm_narrative_tag("any monetary policy text")
    assert tag in nc._LLM_TAG_ENUM or tag == "lookup_failed"


def test_llm_tag_type_check():
    with pytest.raises(TypeError):
        nc.llm_narrative_tag(123)


# ─────────────────────────────────────────────────────────────────────────────
# Single-source amendment lock (no fetch_ecb_press_conference)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_ecb_fetcher_after_single_source_amendment():
    """
    Per 2026-05-08 amendment (spec §2.7.6 + amend_log id=47), ECB fetcher must be
    fully removed. Re-introducing it requires amend_spec(superseded), so a guard
    test is the cheapest enforcement.
    """
    assert not hasattr(nc, "fetch_ecb_press_conference"), (
        "fetch_ecb_press_conference must remain removed; re-adding requires "
        "amend_spec(superseded) per spec §6 forbidden modifications"
    )
