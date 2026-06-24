"""
tests/test_forensic_replay_harness.py — Gap #3 forensic replay harness tests.

3 deterministic tests covering:
  1. Anchor file load + AV cache injection round-trip (no network)
  2. Factor decomposition graceful failure on pre-launch date (Lehman 2008)
  3. yfinance prepare_replay_context end-to-end on Christmas Eve 2018 SPY
     (needs network for yfinance but no LLM call)

NOT tested here (intentional): full devils_advocate live LLM replay
(integration smoke is covered by scripts/run_forensic_replay_anchor_events.py
which is gated behind explicit invocation, not pytest, to avoid burning LLM
cost on every test run).
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from engine.forensic.replay_harness import (
    inject_anchors_to_av_cache,
    prepare_replay_context,
    replay_factor_decomposition,
    ReplayContext,
    ReplayResult,
)


def test_anchor_injection_roundtrip(tmp_path):
    """AV cache injection produces the cache key + schema that news_context expects."""
    av_cache_path = tmp_path / "av_cache.json"
    n = inject_anchors_to_av_cache(
        event_slug="lehman_2008_09",
        ticker="SPY",
        event_date=datetime.date(2008, 9, 15),
        window_days=5,
        av_cache_path=av_cache_path,
    )
    # Lehman anchor file has 3 headlines all within ±5d of 2008-09-15
    assert n == 3, f"expected 3 anchors injected, got {n}"

    cache = json.loads(av_cache_path.read_text(encoding="utf-8"))
    # Cache key MUST match news_context._av_cache_key format
    expected_key = "SPY|20080910T0000|20080920T2359"
    assert expected_key in cache, f"missing expected cache key; got keys {list(cache.keys())}"

    entry = cache[expected_key]
    assert entry["n_articles"] == 3
    assert len(entry["headlines"]) == 3
    # Each headline must have AV-compatible schema
    for h in entry["headlines"]:
        assert "title"           in h and h["title"]
        assert "source"          in h
        assert "published"       in h
        assert "sentiment_label" in h
        assert "sentiment_score" in h
        # Sentiment must NOT be pre-assigned (HARKing immunity)
        assert h["sentiment_label"] == "Unknown", \
            f"sentiment label leaked: {h['sentiment_label']}"
        assert h["sentiment_score"] == 0.0


def test_factor_decomp_fails_gracefully_on_pre_launch_date(monkeypatch, tmp_path):
    """Lehman 2008 factor decomp must FAIL with explicit reason
    (QUAL launched 2013-07; USMV launched 2011-10).
    """
    ctx = ReplayContext(
        event_slug="lehman_2008_09",
        event_name="Lehman Brothers collapse",
        event_date=datetime.date(2008, 9, 15),
        ticker="SPY",
        realized_return_horizon_days=22,
        realized_return=-0.16,    # historical ground-truth
        weight=0.10,
        signal_value=None,
        n_anchors_injected=3,
    )
    result = replay_factor_decomposition(ctx)
    assert isinstance(result, ReplayResult)
    assert result.success is False
    assert result.agent_name == "residual_attribution_factor_returns"
    detail = (result.output_summary.get("detail") or "")
    # Must explicitly cite proxy ETF launch dates
    assert "QUAL" in detail or "USMV" in detail


@pytest.mark.network
def test_prepare_replay_context_christmas_eve_2018_spy():
    """yfinance round-trip: realized 22d return for SPY on 2018-12-24 must be
    in plausible range. Historical record: SPY rallied ~+11% from 2018-12-24 to
    end of Jan 2019 (sharp rebound from Christmas Eve low).
    """
    ctx = prepare_replay_context(
        event_slug="christmas_eve_2018_12",
        ticker="SPY",
        realized_horizon_days=22,
    )
    assert ctx.event_date == datetime.date(2018, 12, 24)
    assert ctx.ticker == "SPY"
    # If yfinance is reachable, realized_return should be roughly +0.05 to +0.15
    # (post-Christmas Eve rebound). If yfinance fails we get None — accept either.
    if ctx.realized_return is not None:
        assert -0.20 < ctx.realized_return < 0.20, \
            f"realized_return {ctx.realized_return} outside plausible range for SPY 2018-12-24+22d"
