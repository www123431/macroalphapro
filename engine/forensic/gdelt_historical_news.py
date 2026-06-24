"""
engine/forensic/gdelt_historical_news.py — Historical news corpus for
forensic replay harness (Gap #3 fix, 2026-05-15).

Background: AlphaVantage free tier only retains ~2 years of news. To validate
the forensic suite on historical crisis events (Lehman 2008-09 / COVID 2020-03 /
Christmas Eve 2018-12) we need a free, programmatic, reproducible source of
historical headlines.

Solution: GDELT 2.0 DOC API (https://api.gdeltproject.org/api/v2/doc/doc).
  - Free, no auth needed
  - Coverage: 2015-02-19 + (GDELT 2.0); GDELT 1.0 extends to 1979 but different schema
  - Reasonable rate limit (a few queries / sec without auth)
  - Returns JSON with title / url / source / domain / language / sentiment tone
  - Used in 3000+ academic papers (Leetaru-Schrodt 2013)

Caveat: GDELT 2.0 starts 2015-02-19. For events BEFORE that date (Lehman 2008),
we fall back to a manual curation step that records "GDELT coverage not
available — anchor headlines used as input directly". This is honestly disclosed
in the capability evidence doc.

Doctrine compliance:
  - 0-LLM-in-DECISION preserved (this layer is data fetching only)
  - LLM-risk-side preserved (validates forensic agents which are already risk-side)
  - No new agents added (this is data infrastructure, not an agent)

Validation criterion (A+B hybrid pattern):
  - A (this module): primary historical news corpus from GDELT
  - B (data/forensic_replay_anchors/*.json): 3 pre-registered must-include anchors per event
  - Recall = |GDELT_results ∩ anchors| / |anchors| → measure of A's retrieval quality
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────
GDELT_DOC_API_URL    = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_2_0_START_DATE = datetime.date(2015, 2, 19)    # GDELT 2.0 coverage start
DEFAULT_MAX_RECORDS  = 50                             # per query (API max 250)
DEFAULT_TIMEOUT_S    = 20
DEFAULT_RETRY_DELAY_S = 5.0


# ── Public dataclass for replay-harness consumption ─────────────────────────
def gdelt_2_0_covers(event_date: datetime.date) -> bool:
    """Whether GDELT 2.0 has coverage for this date."""
    return event_date >= GDELT_2_0_START_DATE


# ── Query construction ──────────────────────────────────────────────────────
def build_query_for_ticker(ticker: str, event_keywords: Optional[Sequence[str]] = None) -> str:
    """Build a GDELT DOC API query for a ticker + optional crisis-event keywords.

    Query strategy: combine ticker-specific terms with event keywords using OR.
    All terms quoted to enforce phrase matching where multi-word.

    Examples:
      build_query_for_ticker("SPY")
        → '("S&P 500" OR "stock market" OR equity) sourcecountry:US'
      build_query_for_ticker("TLT", ["banking crisis"])
        → '(Treasury OR "bond yield" OR "banking crisis") sourcecountry:US'
    """
    ticker_terms: dict[str, list[str]] = {
        "SPY": ['"S&P 500"', '"stock market"', "equity", '"Dow Jones"'],
        "TLT": ["Treasury", '"bond yield"', '"long-term bonds"'],
        "GLD": ['"gold price"', "gold", '"precious metals"'],
        "QQQ": ['"Nasdaq"', '"technology stocks"'],
        "VIX": ['"VIX"', '"volatility index"', '"market volatility"'],
    }
    terms = list(ticker_terms.get(ticker.upper(), [ticker]))
    if event_keywords:
        terms.extend(f'"{kw}"' if " " in kw else kw for kw in event_keywords)
    or_clause = " OR ".join(terms)
    # Restrict to US-sourced English-language news (most relevant for US tickers)
    return f"({or_clause}) sourcecountry:US sourcelang:eng"


# ── Core API call ───────────────────────────────────────────────────────────
def query_gdelt_doc_api(
    query:        str,
    start_dt:     datetime.datetime,
    end_dt:       datetime.datetime,
    maxrecords:   int = DEFAULT_MAX_RECORDS,
    timeout_s:    float = DEFAULT_TIMEOUT_S,
    retry_on_429: bool = True,
) -> list[dict]:
    """Query GDELT DOC 2.0 API and return list of article dicts.

    Args:
        query: GDELT query string (see build_query_for_ticker)
        start_dt: UTC datetime, inclusive
        end_dt: UTC datetime, inclusive (DOC API uses datetime granularity)
        maxrecords: 1-250 (API hard cap)
        timeout_s: HTTP timeout
        retry_on_429: retry once on rate limit

    Returns:
        List of dicts {url, title, source, domain, seendate, language, socialimage, tone}
        Empty list if query fails / no results.
    """
    params = {
        "query":         query,
        "mode":          "ArtList",
        "format":        "json",
        "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end_dt.strftime("%Y%m%d%H%M%S"),
        "maxrecords":    min(maxrecords, 250),
        "sort":          "DateDesc",
    }

    def _call_once() -> Optional[dict]:
        try:
            resp = requests.get(GDELT_DOC_API_URL, params=params, timeout=timeout_s)
            if resp.status_code == 200:
                # GDELT sometimes returns empty body or HTML on no-results — be defensive
                txt = resp.text.strip()
                if not txt or txt.startswith("<"):
                    return {"articles": []}
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    logger.debug("GDELT non-JSON response (first 200): %s", txt[:200])
                    return {"articles": []}
            if resp.status_code == 429 and retry_on_429:
                logger.warning("GDELT rate-limit 429; retrying in %.1fs", DEFAULT_RETRY_DELAY_S)
                time.sleep(DEFAULT_RETRY_DELAY_S)
                resp2 = requests.get(GDELT_DOC_API_URL, params=params, timeout=timeout_s)
                if resp2.status_code == 200:
                    return resp2.json()
            logger.warning("GDELT API HTTP %d: %s", resp.status_code, resp.text[:160])
            return None
        except Exception as exc:
            logger.warning("GDELT API call failed: %s", exc)
            return None

    data = _call_once()
    if data is None:
        return []
    return data.get("articles", []) or []


# ── Format conversion: GDELT → AV-cache headline schema ─────────────────────
def gdelt_articles_to_av_headlines(articles: list[dict]) -> list[dict]:
    """Convert GDELT articles to the AV-compatible headline schema that
    news_context expects (see news_context._av_cache_set headlines list).

    Schema target:
        {title, source, published, sentiment_label, sentiment_score}

    GDELT tone field is V2Tone formatted "tone,positive,negative,polarity,
    activity_density,self_density" — we use first comma-separated value
    (overall tone in [-100, +100]) and re-scale to AV's [-1, +1] convention.
    """
    out = []
    for art in articles:
        # GDELT 2.0 DOC API returns 'seendate' like '20200316T120000Z'
        seen = art.get("seendate", "")
        try:
            dt = datetime.datetime.strptime(seen, "%Y%m%dT%H%M%SZ")
            published = dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            published = "n/a"

        tone_raw = art.get("tone", "0")
        try:
            tone = float(str(tone_raw).split(",")[0])  # first value is overall tone
        except Exception:
            tone = 0.0
        # Map GDELT tone [-100, 100] → AV sentiment_score [-1, 1]
        sentiment_score = max(-1.0, min(1.0, tone / 100.0))
        if sentiment_score >= 0.15:
            sentiment_label = "Bullish"
        elif sentiment_score <= -0.15:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        out.append({
            "title":            (art.get("title") or "").strip(),
            "source":           art.get("domain") or art.get("sourcecountry") or "GDELT",
            "published":        published,
            "sentiment_label":  sentiment_label,
            "sentiment_score":  sentiment_score,
            "_provenance":      "gdelt-2.0-doc-api",
            "_url":             art.get("url", ""),
        })
    return out


# ── Populate AV cache (so news_context.investigate_trade reads transparently) ─
def populate_av_cache_for_replay(
    ticker:           str,
    event_date:       datetime.date,
    window_days:      int,
    event_keywords:   Optional[Sequence[str]] = None,
    maxrecords:       int = DEFAULT_MAX_RECORDS,
    av_cache_path:    Optional[Path] = None,
) -> dict:
    """Fetch historical headlines via GDELT and write them into the AV cache
    structure that news_context expects. Returns retrieval summary.

    Idempotent: re-running for same (ticker, event_date, window) overwrites cache.

    Returns dict {n_articles, query, window_start_iso, window_end_iso,
                  gdelt_covered, cache_key}
    """
    window_start = event_date - datetime.timedelta(days=window_days)
    window_end   = event_date + datetime.timedelta(days=window_days)

    # Honest disclosure if event predates GDELT 2.0
    if not gdelt_2_0_covers(window_start):
        logger.warning(
            "GDELT 2.0 does not cover %s — coverage starts %s. "
            "Returning empty corpus; replay should fall back to anchor-only mode.",
            window_start.isoformat(), GDELT_2_0_START_DATE.isoformat(),
        )
        return {
            "n_articles":      0,
            "query":           None,
            "window_start_iso": window_start.isoformat(),
            "window_end_iso":   window_end.isoformat(),
            "gdelt_covered":   False,
            "cache_key":       None,
        }

    query = build_query_for_ticker(ticker, event_keywords)
    start_dt = datetime.datetime.combine(window_start, datetime.time(0, 0, 0))
    end_dt   = datetime.datetime.combine(window_end, datetime.time(23, 59, 59))

    raw_articles = query_gdelt_doc_api(query, start_dt, end_dt, maxrecords=maxrecords)
    av_format = gdelt_articles_to_av_headlines(raw_articles)

    # Write into news_context's AV cache structure with matching key format.
    # AV cache key format: f"{ticker}|{time_from}|{time_to}" where time_from
    # is "YYYYMMDDTHHMM" (see news_context._av_cache_key).
    time_from = window_start.strftime("%Y%m%dT0000")
    time_to   = window_end.strftime("%Y%m%dT2359")
    cache_key = f"{ticker.upper()}|{time_from}|{time_to}"

    if av_cache_path is None:
        _repo_root = Path(__file__).resolve().parent.parent.parent
        av_cache_path = _repo_root / "data" / "forensic" / "av_news_cache.json"

    av_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if av_cache_path.exists():
        try:
            cache = json.loads(av_cache_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("AV cache corrupt; starting fresh")
            cache = {}

    cache[cache_key] = {
        "headlines":     av_format,
        "cached_at_iso": datetime.datetime.utcnow().isoformat(),
        "n_articles":    len(av_format),
        "_provenance":   "gdelt-2.0-doc-api",
    }
    av_cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    return {
        "n_articles":      len(av_format),
        "query":           query,
        "window_start_iso": window_start.isoformat(),
        "window_end_iso":   window_end.isoformat(),
        "gdelt_covered":   True,
        "cache_key":       cache_key,
        "raw_articles":    raw_articles,    # passed through for recall computation
        "av_format":       av_format,
    }


# ── Recall check: A ∩ B (GDELT retrieved vs pre-registered anchors) ─────────
def compute_anchor_recall(
    gdelt_articles:  list[dict],
    anchor_records:  list[dict],
) -> dict:
    """Pre-registered ground-truth recall measurement.

    For each anchor in anchor_records, check whether GDELT retrieved at least
    one article that matches via search_keywords (any keyword present in title,
    case-insensitive substring match) AND date within ±2 days of anchor date.

    Returns:
        {anchors_total, anchors_recalled, recall_rate, per_anchor_details}

    Per-anchor detail dict: {anchor_title, anchor_date_iso, recalled, matched_articles}
    """
    per_anchor: list[dict] = []
    n_recalled = 0
    for anchor in anchor_records:
        a_title = anchor.get("title", "")
        a_date  = anchor.get("date_iso", "")
        keywords = [k.lower() for k in anchor.get("search_keywords", [])]

        try:
            a_dt = datetime.date.fromisoformat(a_date)
        except Exception:
            a_dt = None

        matches: list[dict] = []
        for art in gdelt_articles:
            art_title = (art.get("title") or "").lower()
            if not any(k in art_title for k in keywords):
                continue
            # Date proximity check
            seen = art.get("seendate", "")
            try:
                art_dt = datetime.datetime.strptime(seen, "%Y%m%dT%H%M%SZ").date()
                if a_dt is not None and abs((art_dt - a_dt).days) > 2:
                    continue
            except Exception:
                pass    # if date unparseable, accept the title-keyword match
            matches.append({
                "title":     art.get("title"),
                "url":       art.get("url"),
                "seendate":  seen,
            })

        recalled = len(matches) > 0
        if recalled:
            n_recalled += 1
        per_anchor.append({
            "anchor_title":     a_title,
            "anchor_date_iso":  a_date,
            "recalled":         recalled,
            "n_matched_articles": len(matches),
            "first_match":      matches[0] if matches else None,
        })

    total = len(anchor_records)
    return {
        "anchors_total":     total,
        "anchors_recalled":  n_recalled,
        "recall_rate":       round(n_recalled / total, 3) if total else None,
        "per_anchor":        per_anchor,
    }


def load_anchor_file(event_slug: str, anchors_dir: Optional[Path] = None) -> dict:
    """Load a pre-registered anchor file from data/forensic_replay_anchors/.

    event_slug examples: "lehman_2008_09", "covid_2020_03", "christmas_eve_2018_12"
    """
    if anchors_dir is None:
        _repo_root = Path(__file__).resolve().parent.parent.parent
        anchors_dir = _repo_root / "data" / "forensic_replay_anchors"
    path = anchors_dir / f"{event_slug}.json"
    if not path.exists():
        raise FileNotFoundError(f"anchor file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
