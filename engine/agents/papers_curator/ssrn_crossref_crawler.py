"""engine.agents.papers_curator.ssrn_crossref_crawler — Stage A piece 5
follow-up: pull SSRN papers via CrossRef DOI registry workaround.

Why this exists (per [[project-anti-rut-doctrine-2026-06-07]]):
SSRN's public RSS endpoints are all 403-gated as of 2026-06-07 — FEN,
top-ten, journal-specific, and even basic browser-UA requests return
403 (CloudFlare bot detection). Direct scraping is forbidden by TOS.

The workaround: SSRN registers ALL papers with CrossRef under DOI
prefix 10.2139 (e.g. `10.2139/ssrn.5007147`). CrossRef's free API
indexes 1.4M SSRN papers and supports filtering by deposit date.
Sort by `deposited` desc surfaces recently-uploaded SSRN papers.

This is INDIRECT but legitimate — CrossRef is the public DOI
registry; SSRN voluntarily deposits its DOIs there. No scraping,
no TOS violation, no auth needed.

Trade-offs vs direct SSRN access:
  - Pros: free, fast, well-documented API; works today
  - Cons: ~780 SSRN deposits/day across ALL topics. Most are NOT
    finance (CrossRef indexes everything SSRN cross-lists).
    Downstream LLM filter handles topical relevance (same pattern
    as NBER). Cost: max_results × $0.001 per fetch = $0.10/fetch
    at default 100 papers, $0.70/week at weekly cadence — fine.
  - 'published' date is unreliable (CrossRef shows 2103/2104 for
    many SSRN papers — SSRN deposits dates wrong). 'deposited'
    date is reliable and reflects when CrossRef received it.

Cache schema integration:
  source       = "ssrn"
  source_id    = full DOI (e.g. "10.2139/ssrn.5007147")
  categories   = ("ssrn_via_crossref",)

The DOI naturally serves as dedup key — re-fetches don't double-write
because CACHE_PATH dedup is on (source, source_id) per Stage A piece 1.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

from engine.agents.papers_curator.crawler import PaperCandidate

logger = logging.getLogger(__name__)


_CROSSREF_API = "https://api.crossref.org/prefixes/10.2139/works"
_USER_AGENT = ("MacroAlphaPro-PapersCurator/1.0 "
                "(mailto:${USER_EMAIL})")

# CrossRef polite pool: <50 req/sec is fine, but we want gentle.
# 1 req every 2s leaves ample headroom and matches the calling pattern
# of NBER/SS (one-shot daily fetch).
_MIN_INTERVAL_S = 0.5


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# Strip CrossRef JATS / HTML markup that occasionally appears in
# titles + abstracts. Conservative regex — only strips well-formed
# <tag> and </tag> patterns.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_ENT_RE = re.compile(r"&(lt|gt|amp|quot|nbsp);")
_HTML_ENT_MAP = {"&lt;": "<", "&gt;": ">", "&amp;": "&",
                  "&quot;": '"', "&nbsp;": " "}


def _strip_markup(s: str) -> str:
    """Strip <jats:p>, <span>, &lt; etc. that leak into CrossRef text."""
    if not s:
        return ""
    s = _HTML_TAG_RE.sub(" ", s)
    for ent, ch in _HTML_ENT_MAP.items():
        s = s.replace(ent, ch)
    # Collapse whitespace
    return " ".join(s.split())


def _build_query_url(*, from_deposit_date: str, rows: int) -> str:
    params = {
        "filter": f"from-deposit-date:{from_deposit_date}",
        "rows":   str(rows),
        "sort":   "deposited",
        "order":  "desc",
    }
    return f"{_CROSSREF_API}?{urllib.parse.urlencode(params)}"


def _crossref_get(url: str, *, timeout: float = 20.0,
                    max_retries: int = 3) -> Optional[dict]:
    """One GET with retry on 429/5xx. None on hard failure."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        time.sleep(_MIN_INTERVAL_S)
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT,
                                "Accept":     "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                backoff = 2.0 * (2 ** attempt)   # 2, 4, 8s
                logger.warning("crossref: HTTP %d (attempt %d/%d) — "
                                "retry in %.0fs",
                                exc.code, attempt + 1, max_retries,
                                backoff)
                if attempt + 1 < max_retries:
                    time.sleep(backoff)
                    continue
            logger.warning("crossref: HTTP %d on %s — giving up",
                            exc.code, url[:80])
            return None
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                backoff = 2.0 * (2 ** attempt)
                logger.info("crossref: transport error (attempt %d/%d): "
                              "%s — retry in %.0fs",
                              attempt + 1, max_retries, exc, backoff)
                time.sleep(backoff)
                continue
    logger.warning("crossref: failed after %d retries: %s",
                    max_retries, last_exc)
    return None


def _extract_authors(item: dict) -> tuple[str, ...]:
    """CrossRef author item shape: [{'given': 'Foo', 'family': 'Bar'},
    ...]. Returns ('Foo Bar', ...) or () if absent."""
    out: list[str] = []
    for a in (item.get("author") or []):
        given = (a.get("given") or "").strip()
        family = (a.get("family") or "").strip()
        name = (f"{given} {family}".strip()
                if given or family else "")
        if name:
            out.append(name)
    return tuple(out)


def _to_paper_candidate(item: dict, *, fetched_ts: str
                           ) -> Optional[PaperCandidate]:
    """CrossRef item → PaperCandidate. Returns None on malformed
    (no DOI or no title)."""
    doi = (item.get("DOI") or "").strip()
    titles = item.get("title") or []
    title = _strip_markup(titles[0] if titles else "")
    if not doi or not title:
        return None

    abstract = _strip_markup(item.get("abstract") or "")
    authors = _extract_authors(item)

    # 'deposited' is when CrossRef received the metadata; we use as
    # published_ts proxy because CrossRef 'published' is unreliable
    # for SSRN deposits (often shows 2103/2104 garbage).
    dep = item.get("deposited") or {}
    dep_parts = (dep.get("date-parts") or [[]])[0]
    if len(dep_parts) >= 3:
        published_ts = (f"{dep_parts[0]:04d}-{dep_parts[1]:02d}-"
                         f"{dep_parts[2]:02d}T00:00:00Z")
    else:
        published_ts = fetched_ts

    # Use SSRN landing URL convention — extract numeric SSRN id from
    # the DOI's tail (after 'ssrn.')
    abs_url = ""
    pdf_url = ""
    if "ssrn." in doi:
        ssrn_num = doi.split("ssrn.", 1)[1]
        if ssrn_num.isdigit():
            abs_url = (f"https://papers.ssrn.com/sol3/papers.cfm"
                        f"?abstract_id={ssrn_num}")
            # Note: SSRN PDF URLs require auth — we leave empty.
            # LLM summarizer reads abstract from cache (already here).

    return PaperCandidate(
        source       = "ssrn",
        source_id    = doi,
        title        = title,
        authors      = authors,
        abstract     = abstract,
        abs_url      = abs_url,
        pdf_url      = pdf_url,
        published_ts = published_ts,
        categories   = ("ssrn_via_crossref",),
        fetched_ts   = fetched_ts,
    )


def crawl_ssrn_via_crossref(
    *,
    lookback_days: int = 7,
    max_results:   int = 100,
    timeout:       float = 20.0,
    max_retries:   int = 3,
) -> list[PaperCandidate]:
    """Fetch SSRN papers deposited in CrossRef within the last
    `lookback_days`, up to `max_results`. Returns adapted candidates.

    Default 7d / 100 results matches the chief_of_staff weekly
    cadence — covers the gap since last fetch with comfortable margin.
    """
    cutoff = (_dt.datetime.utcnow()
              - _dt.timedelta(days=lookback_days)
              ).strftime("%Y-%m-%d")
    url = _build_query_url(from_deposit_date=cutoff,
                            rows=min(max_results, 1000))
    data = _crossref_get(url, timeout=timeout,
                          max_retries=max_retries)
    if not data:
        return []
    items = ((data.get("message") or {}).get("items") or [])[:max_results]
    fetched_ts = _utc_iso()
    out: list[PaperCandidate] = []
    for item in items:
        cand = _to_paper_candidate(item, fetched_ts=fetched_ts)
        if cand is not None:
            out.append(cand)
    logger.info("ssrn(crossref): parsed %d candidates from %d items",
                 len(out), len(items))
    return out


def crawl_and_persist_ssrn(*, lookback_days: int = 7,
                              max_results: int = 100) -> dict:
    """One-shot wrapper: fetch + dedup-write. Structured result dict
    matches watchlist_crawler / forward_citation_crawler / nber so
    chief_of_staff can compose them uniformly."""
    from engine.agents.papers_curator.store import save_new_candidates

    result = {
        "run_ts":     _utc_iso(),
        "source":     "ssrn",
        "n_fetched":  0,
        "n_new":      0,
        "errors":     [],
    }
    try:
        cands = crawl_ssrn_via_crossref(
            lookback_days=lookback_days, max_results=max_results,
        )
    except Exception as exc:
        logger.exception("ssrn(crossref): crawl raised unexpectedly")
        result["errors"].append(f"crawl: {exc}")
        return result

    result["n_fetched"] = len(cands)
    if not cands:
        return result

    try:
        n_new = save_new_candidates(cands)
        result["n_new"] = n_new
    except Exception as exc:
        logger.exception("ssrn(crossref): save_new_candidates failed")
        result["errors"].append(f"persist: {exc}")
    return result
