"""engine.agents.papers_curator.crawler — fetch latest papers per source.

Sources implemented:
  - arxiv q-fin (all subcategories, last 7 days) via the arxiv Atom API
    https://export.arxiv.org/api/query — stable, no auth, no rate-limit
    issues at the volumes we crawl (~30 papers/day).

NOT YET implemented (future commits):
  - NBER WP RSS (https://www.nber.org/papers.rss?topic=2 — finance)
  - SSRN FEN (paywalled, messy; skip for v1)

Design:
  - One PaperCandidate dataclass — the common schema all sources map to
  - Each source = one function returning list[PaperCandidate]
  - crawl_all() iterates all enabled sources, returns merged list
  - NO dedup here (store layer owns that)
  - NO LLM call here (cheap fetch + parse only)
  - Errors per source are isolated; one source down doesn't kill the others

Wall: ~2s per source. Cron-friendly.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import xml.etree.ElementTree as _ET
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@_dc.dataclass(frozen=True)
class PaperCandidate:
    """Common-schema representation of a newly-published paper.

    Stable contract — adding fields is OK, removing/renaming is not.
    """
    source:        str                  # "arxiv" | "nber" | ...
    source_id:     str                  # source-native id; dedup key with source
    title:         str
    authors:       tuple[str, ...]
    abstract:      str
    abs_url:       str                  # landing page
    pdf_url:       str                  # direct PDF (if available)
    published_ts:  str                  # iso UTC; source-reported publish ts
    categories:    tuple[str, ...]      # source-native category tags
    fetched_ts:    str                  # iso UTC; when WE crawled it

    def to_dict(self) -> dict:
        return _dc.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaperCandidate":
        return cls(
            source        = str(d.get("source", "")),
            source_id     = str(d.get("source_id", "")),
            title         = str(d.get("title", "")),
            authors       = tuple(d.get("authors") or ()),
            abstract      = str(d.get("abstract", "")),
            abs_url       = str(d.get("abs_url", "")),
            pdf_url       = str(d.get("pdf_url", "")),
            published_ts  = str(d.get("published_ts", "")),
            categories    = tuple(d.get("categories") or ()),
            fetched_ts    = str(d.get("fetched_ts", "")),
        )


# ──────────────────────────────────────────────────────────────────────
# arxiv q-fin
# ──────────────────────────────────────────────────────────────────────
def _parse_arxiv_entry(entry: _ET.Element, fetched_ts: str) -> Optional[PaperCandidate]:
    """Parse one <atom:entry> element into a PaperCandidate. Returns
    None on malformed entries (caller continues with next)."""
    try:
        # id is like "http://arxiv.org/abs/2401.12345v2"
        full_id = entry.findtext("atom:id", default="", namespaces=_ARXIV_NS).strip()
        # strip version + URL prefix
        source_id = full_id.rsplit("/", 1)[-1]
        if "v" in source_id:
            source_id = source_id.split("v", 1)[0]

        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").strip()
        title = " ".join(title.split())   # collapse whitespace
        abstract = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").strip()
        abstract = " ".join(abstract.split())
        published_ts = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS).strip()
        abs_url = full_id   # arxiv id IS the landing url

        # authors
        authors: list[str] = []
        for a in entry.findall("atom:author", _ARXIV_NS):
            name = (a.findtext("atom:name", default="", namespaces=_ARXIV_NS) or "").strip()
            if name:
                authors.append(name)

        # categories (q-fin.PR, q-fin.ST, etc.)
        cats: list[str] = []
        for c in entry.findall("atom:category", _ARXIV_NS):
            term = c.attrib.get("term", "").strip()
            if term:
                cats.append(term)

        # pdf url — arxiv convention is /pdf/<id>
        pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

        if not source_id or not title:
            return None
        return PaperCandidate(
            source       = "arxiv",
            source_id    = source_id,
            title        = title,
            authors      = tuple(authors),
            abstract     = abstract,
            abs_url      = abs_url,
            pdf_url      = pdf_url,
            published_ts = published_ts,
            categories   = tuple(cats),
            fetched_ts   = fetched_ts,
        )
    except Exception as exc:
        logger.warning("arxiv: failed to parse entry: %s", exc)
        return None


def crawl_arxiv_qfin(*, max_results: int = 50, timeout: float = 60.0,
                      max_retries: int = 3) -> list[PaperCandidate]:
    """Fetch the latest `max_results` q-fin.* papers from arxiv, sorted
    by submission date desc. Returns parsed PaperCandidate list.

    Query covers ALL q-fin subcategories (PR, ST, RM, CP, PM, MF, GN, EC).

    Retries on transport errors (arxiv from CN can be slow / occasionally
    DNS-flaky). Failed runs return [] so the daily crawl as a whole
    keeps working.
    """
    import time as _time
    params = {
        "search_query": "cat:q-fin.*",
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
        "max_results":  str(max_results),
        "start":        "0",
    }
    # arxiv API guideline: keep requests >= 3 sec apart. With one daily
    # call this never matters, but during dev when we hammer it, prior
    # failed attempts can flip the upstream into a 429 cool-down for
    # ~minutes. Retry with growing backoff on 429 / 503 / transport.
    _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.get(_ARXIV_API_URL, params=params, timeout=timeout,
                              headers={"User-Agent": "MacroAlphaPro-PapersCurator/1.0"})
            if resp.status_code in _RETRYABLE_STATUS:
                if attempt == max_retries:
                    logger.warning("arxiv: status %d after %d retries — giving up",
                                    resp.status_code, max_retries)
                    return []
                # Longer backoff for 429 (rate-limit) than for timeouts —
                # arxiv's cool-down is usually 1-2 minutes.
                backoff = 30.0 if resp.status_code == 429 else 5.0 * (attempt + 1)
                logger.info("arxiv: HTTP %d (attempt %d/%d) — retry in %.0fs",
                             resp.status_code, attempt + 1, max_retries + 1, backoff)
                _time.sleep(backoff)
                continue
            resp.raise_for_status()
            break
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == max_retries:
                logger.warning("arxiv: fetch failed after %d retries: %s",
                                max_retries, exc)
                return []
            backoff = 2.0 * (attempt + 1)
            logger.info("arxiv: transport error (attempt %d/%d): %s — retry in %.0fs",
                         attempt + 1, max_retries + 1, exc, backoff)
            _time.sleep(backoff)
        except Exception as exc:
            logger.exception("arxiv: non-retryable failure: %s", exc)
            return []
    if resp is None or resp.status_code >= 400:
        return []

    try:
        root = _ET.fromstring(resp.text)
    except _ET.ParseError as exc:
        logger.exception("arxiv: XML parse failed: %s", exc)
        return []

    fetched_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[PaperCandidate] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        cand = _parse_arxiv_entry(entry, fetched_ts)
        if cand is not None:
            out.append(cand)
    logger.info("arxiv: parsed %d candidates from %d entries", len(out),
                 len(root.findall("atom:entry", _ARXIV_NS)))
    return out


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def crawl_all(*, arxiv_max: int = 50) -> list[PaperCandidate]:
    """Fan out to all enabled sources. Returns merged candidate list.
    Per-source failures are logged but do not abort the run.
    """
    out: list[PaperCandidate] = []
    try:
        out.extend(crawl_arxiv_qfin(max_results=arxiv_max))
    except Exception as exc:
        logger.exception("crawl_arxiv_qfin failed (continuing): %s", exc)
    # Future sources slot in here: NBER WP, SSRN FEN, etc.
    return out
