"""engine.agents.papers_curator.nber_rss_crawler — Stage A piece 5.

Fourth substrate stream: NBER working papers RSS feed.

Joins arxiv RSS + watchlist (SS authors) + forward citations at
cache.jsonl. Anti-rut design (per [[project-anti-rut-doctrine-2026-06-07]]):
NBER captures the formal-economics frontier (Brunnermeier, Adrian,
Welch, et al. publish working papers BEFORE arxiv preprint and OFTEN
before SSRN). That's where institutional / regulatory-driven finance
research surfaces first.

NBER feed shape (verified 2026-06-07):
  url:      https://back.nber.org/rss/new.xml  (old www.nber.org/papers.rss
                                                  301-redirects here)
  entries:  ~35 per fetch (all-topics, not topic-filtered — NBER's
              per-topic RSS endpoints all 404 as of 2026-06-07)
  id:       https://www.nber.org/papers/wNNNNN#fromrss
  summary:  abstract (truncated to ~200 chars typically)
  authors:  NOT in feed (filed empty; LLM summarizer recovers from PDF)
  date:     NOT in feed (filed as fetched_ts)

Topic filtering at substrate layer: NO. Per anti-rut doctrine, we
DON'T pre-filter inputs by keyword (lossy). The downstream LLM
filter is the topical gate. NBER costs ~$0.035/day at filter; fine.

SSRN: paywalled. SSRN public RSS endpoints (FEN, top-ten, journal-
specific) all 403 as of 2026-06-07. Re-attempt when SSRN exposes
free RSS or we find a documented workaround (RePEc NEP-FMK was
DNS-flaky too). Not blocking piece 5 ship.

Returns a structured result the chief_of_staff orchestrator can fold
into the weekly memo ('NBER substrate: 35 fetched, 12 new this week').
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from pathlib import Path
from typing import Optional

from engine.agents.papers_curator.crawler import PaperCandidate

logger = logging.getLogger(__name__)


_NBER_RSS_URL = "https://back.nber.org/rss/new.xml"
_WP_NUMBER_RE = re.compile(r"/papers/(w\d+)", re.IGNORECASE)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_wp_number(link: str) -> str:
    """NBER entry link is like https://www.nber.org/papers/w35254#fromrss.
    Extract the 'w35254' identifier as source_id. Returns '' if pattern
    not found (defensive — NBER may someday change URL shape)."""
    if not link:
        return ""
    m = _WP_NUMBER_RE.search(link)
    return m.group(1).lower() if m else ""


def _to_paper_candidate(entry, *, fetched_ts: str) -> Optional[PaperCandidate]:
    """Adapt one feedparser entry → PaperCandidate.

    Returns None when the entry is malformed (no WP number / no title) —
    caller filters None.
    """
    link = (entry.get("link") or "").strip()
    wp_number = _extract_wp_number(link)
    title = (entry.get("title") or "").strip()
    if not wp_number or not title:
        return None

    summary = (entry.get("summary") or "").strip()
    # Use feed's published date if available, else fetched_ts
    published = (entry.get("published") or "").strip() or fetched_ts
    # NBER doesn't expose authors in RSS — leave empty; LLM summarizer
    # recovers them from the PDF.
    authors: tuple[str, ...] = ()

    # NBER WP PDF convention: https://www.nber.org/papers/wNNNNN/wNNNNN.pdf
    # NOT the /system/files/working_papers/... pattern used by some
    # older NBER tooling (those URLs require auth).
    pdf_url = f"https://www.nber.org/papers/{wp_number}/{wp_number}.pdf"
    abs_url = f"https://www.nber.org/papers/{wp_number}"

    return PaperCandidate(
        source       = "nber",
        source_id    = wp_number,
        title        = title,
        authors      = authors,
        abstract     = summary,
        abs_url      = abs_url,
        pdf_url      = pdf_url,
        published_ts = published,
        categories   = ("nber_working_paper",),
        fetched_ts   = fetched_ts,
    )


def crawl_nber_rss(
    *,
    url:               str = _NBER_RSS_URL,
    timeout:           float = 30.0,
    max_retries:       int = 3,
) -> list[PaperCandidate]:
    """Fetch NBER working-papers RSS, parse + adapt to PaperCandidate.

    Errors are isolated — network failure returns []; the daily crawl
    keeps the other sources running.
    """
    import time as _time
    import feedparser  # heavy import, defer

    last_exc: Exception | None = None
    parsed = None
    for attempt in range(max_retries + 1):
        try:
            # feedparser handles HTTP itself + tolerates encoding quirks
            parsed = feedparser.parse(
                url,
                request_headers={"User-Agent": "MacroAlphaPro-PapersCurator/1.0"},
            )
            status = parsed.get("status", 0)
            if status >= 400:
                # 404 / 403 — treat as deterministic; no retry
                logger.warning("nber: HTTP %d on %s — abandoning fetch",
                                status, url)
                return []
            if parsed.bozo and not parsed.entries:
                # Malformed AND no entries — likely DNS / transport
                last_exc = parsed.get("bozo_exception")
                if attempt < max_retries:
                    backoff = 2.0 * (attempt + 1)
                    logger.info("nber: feed bozo (attempt %d/%d): %s — "
                                  "retry in %.0fs",
                                  attempt + 1, max_retries + 1, last_exc,
                                  backoff)
                    _time.sleep(backoff)
                    continue
                logger.warning("nber: bozo after %d retries: %s",
                                max_retries, last_exc)
                return []
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                backoff = 2.0 * (attempt + 1)
                logger.info("nber: transport error (attempt %d/%d): %s "
                              "— retry in %.0fs",
                              attempt + 1, max_retries + 1, exc, backoff)
                _time.sleep(backoff)
                continue
            logger.warning("nber: fetch failed after %d retries: %s",
                            max_retries, last_exc)
            return []

    if parsed is None or not parsed.entries:
        return []

    fetched_ts = _utc_iso()
    out: list[PaperCandidate] = []
    for entry in parsed.entries:
        cand = _to_paper_candidate(entry, fetched_ts=fetched_ts)
        if cand is not None:
            out.append(cand)
    logger.info("nber: parsed %d candidates from %d entries",
                 len(out), len(parsed.entries))
    return out


def crawl_and_persist_nber(*, url: str = _NBER_RSS_URL) -> dict:
    """One-shot wrapper: fetch + dedup-write. Returns structured result
    matching watchlist_crawler.crawl_watchlist + forward_citation_crawler.
    crawl_forward_citations so chief_of_staff can compose them
    uniformly.
    """
    from engine.agents.papers_curator.store import save_new_candidates

    result = {
        "run_ts":           _utc_iso(),
        "source":           "nber",
        "n_fetched":        0,
        "n_new":            0,
        "errors":           [],
    }
    try:
        cands = crawl_nber_rss(url=url)
    except Exception as exc:
        logger.exception("nber: crawl_nber_rss raised unexpectedly")
        result["errors"].append(f"crawl: {exc}")
        return result

    result["n_fetched"] = len(cands)
    if not cands:
        return result

    try:
        n_new = save_new_candidates(cands)
        result["n_new"] = n_new
    except Exception as exc:
        logger.exception("nber: save_new_candidates failed")
        result["errors"].append(f"persist: {exc}")
    return result
