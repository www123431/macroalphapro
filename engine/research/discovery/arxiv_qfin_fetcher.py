"""engine/research/discovery/arxiv_qfin_fetcher.py — arXiv q-fin API fetcher.

arXiv Atom API is free, no auth, polite usage (1 req/3sec).
We pull from q-fin (Quantitative Finance) category with date filter.

Categories searched (q-fin):
  q-fin.PR  — Pricing of Securities
  q-fin.ST  — Statistical Finance
  q-fin.PM  — Portfolio Management
  q-fin.GN  — General Finance
  q-fin.MF  — Mathematical Finance
  q-fin.CP  — Computational Finance
  q-fin.TR  — Trading and Market Microstructure
  q-fin.EC  — Economics

Output: long-format DataFrame [arxiv_id, title, authors, abstract,
                                 categories, submitted_date, updated_date,
                                 pdf_url, abs_url]

NOT in scope of this fetcher:
- PDF download / parsing
- LLM extraction
- Quality filtering
That's the next stage (paper_extractor.py + hygiene_gate.py).
"""
from __future__ import annotations

import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

ARXIV_BASE = "http://export.arxiv.org/api/query"
QFIN_CATEGORIES = [
    "q-fin.PR", "q-fin.ST", "q-fin.PM", "q-fin.GN",
    "q-fin.MF", "q-fin.CP", "q-fin.TR", "q-fin.EC",
]
NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv asks for 1 req/3sec polite use
POLITE_DELAY_SEC = 3.5
RESULTS_PER_PAGE = 100


def fetch_qfin_papers(
    start_date: str,
    end_date: str,
    *,
    max_results: int = 500,
    categories: list[str] | None = None,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch q-fin arXiv papers submitted within date range.

    Args:
      start_date: YYYY-MM-DD
      end_date:   YYYY-MM-DD
      max_results: cap total pulled (default 500; arXiv pagination)
      categories: subset of QFIN_CATEGORIES (default: all)
      skip_health_check: bypass source_health (testing only)

    Returns:
      DataFrame with one row per paper. EMPTY if source unhealthy
      (caller should check + try fallback like RSS or other source).
    """
    from engine.data import source_health

    # Pre-flight health check (per WRDS-care + arxiv-429 incident)
    if not skip_health_check:
        healthy, reason = source_health.is_healthy("arxiv_api")
        if not healthy:
            logger.info("arxiv_api unhealthy: %s; returning empty", reason)
            return pd.DataFrame(columns=[
                "arxiv_id", "title", "authors", "abstract", "categories",
                "submitted_date", "updated_date", "pdf_url", "abs_url",
            ])

    import requests
    cats = categories or QFIN_CATEGORIES
    # Don't use the retry-enabled session — arXiv 429 should back off
    # politely, not retry-storm into more 429s
    session = requests.Session()
    session.headers.update({
        "User-Agent":      "macro-alpha-research/1.0 (research@local; arxiv-qfin)",
        "Accept-Encoding": "gzip, deflate",
    })
    # Initial polite delay before first request
    time.sleep(POLITE_DELAY_SEC)

    # arXiv API uses OR for categories
    cat_query = " OR ".join(f"cat:{c}" for c in cats)
    # Date filter via submittedDate
    date_filter = (
        f"submittedDate:[{start_date.replace('-', '')}0000 "
        f"TO {end_date.replace('-', '')}2359]"
    )
    search_query = f"({cat_query}) AND {date_filter}"

    rows: list[dict] = []
    start_idx = 0
    consecutive_failures = 0
    while len(rows) < max_results:
        batch_size = min(RESULTS_PER_PAGE, max_results - len(rows))
        params = {
            "search_query": search_query,
            "start":        start_idx,
            "max_results":  batch_size,
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }
        url = f"{ARXIV_BASE}?{urllib.parse.urlencode(params)}"
        try:
            resp = session.get(url, timeout=30)
        except Exception as exc:
            logger.warning("arxiv fetch failed at offset %d: %s", start_idx, exc)
            break
        # Handle 429 — mark source unhealthy + abort (don't retry-storm)
        if resp.status_code == 429:
            from engine.data import source_health
            source_health.mark_failure(
                "arxiv_api", "rate_limited",
                f"HTTP 429 at offset {start_idx}",
            )
            logger.warning(
                "arxiv 429 — marking source unhealthy (24h cooldown); aborting"
            )
            break
        if resp.status_code != 200:
            from engine.data import source_health
            source_health.mark_failure(
                "arxiv_api", "network",
                f"HTTP {resp.status_code} at offset {start_idx}",
            )
            logger.warning("arxiv HTTP %d at offset %d", resp.status_code, start_idx)
            break
        consecutive_failures = 0

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            logger.warning("arxiv XML parse failed: %s", exc)
            break

        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for entry in entries:
            row = _parse_entry(entry)
            if row:
                rows.append(row)

        if len(entries) < batch_size:
            break
        start_idx += batch_size
        time.sleep(POLITE_DELAY_SEC)

    if not rows:
        return pd.DataFrame(columns=[
            "arxiv_id", "title", "authors", "abstract", "categories",
            "submitted_date", "updated_date", "pdf_url", "abs_url",
        ])
    # Success — clear any prior unhealthy mark
    from engine.data import source_health
    source_health.mark_success("arxiv_api")
    return pd.DataFrame(rows)


# ── Layer 2 fallback: arXiv RSS endpoint (more generous rate limit) ─────

# arXiv RSS feeds: http://arxiv.org/rss/<category>
# These are paged by day and historically have looser rate limits than
# the API. Used as fallback when API marks unhealthy.

ARXIV_RSS_BASE = "http://export.arxiv.org/rss"
RSS_NS = {
    "rdf":   "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rss":   "http://purl.org/rss/1.0/",
    "dc":    "http://purl.org/dc/elements/1.1/",
}


def fetch_qfin_rss(
    *,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """RSS fallback. Returns LATEST papers (no date range — RSS feeds the
    most recent batch). Lower granularity than the API but works when API
    is throttled.

    Best used when api unhealthy via `source_health.is_healthy('arxiv_api')`.
    """
    from engine.data import source_health
    healthy, reason = source_health.is_healthy("arxiv_rss")
    if not healthy:
        logger.info("arxiv_rss unhealthy: %s; returning empty", reason)
        return pd.DataFrame(columns=[
            "arxiv_id", "title", "authors", "abstract", "categories",
            "submitted_date", "updated_date", "pdf_url", "abs_url",
        ])

    import requests
    cats = categories or QFIN_CATEGORIES
    session = requests.Session()
    session.headers.update({
        "User-Agent": "macro-alpha-research/1.0 (research@local; arxiv-qfin-rss)",
    })

    rows: list[dict] = []
    for cat in cats:
        time.sleep(POLITE_DELAY_SEC)
        url = f"{ARXIV_RSS_BASE}/{cat}"
        try:
            resp = session.get(url, timeout=30)
        except Exception as exc:
            logger.warning("arxiv RSS %s failed: %s", cat, exc)
            continue
        if resp.status_code == 429:
            source_health.mark_failure(
                "arxiv_rss", "rate_limited", f"HTTP 429 on {cat}"
            )
            break
        if resp.status_code != 200:
            continue
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            continue
        # RSS format uses different namespace
        items = root.findall(".//rss:item", RSS_NS)
        for item in items:
            row = _parse_rss_item(item, cat)
            if row:
                rows.append(row)

    if rows:
        source_health.mark_success("arxiv_rss")
    if not rows:
        return pd.DataFrame(columns=[
            "arxiv_id", "title", "authors", "abstract", "categories",
            "submitted_date", "updated_date", "pdf_url", "abs_url",
        ])
    # Dedup by arxiv_id (a paper may appear in multiple categories)
    df = pd.DataFrame(rows).drop_duplicates(subset=["arxiv_id"], keep="first")
    return df.reset_index(drop=True)


def _parse_rss_item(item, category: str) -> dict | None:
    """Parse one RSS item. Different namespace from Atom."""
    try:
        title = (item.findtext("rss:title", default="", namespaces=RSS_NS)
                  .strip().replace("\n", " "))
        # arXiv RSS title format: "Title. (arXiv:2401.12345v1 [q-fin.PR])"
        arxiv_id = ""
        if "arXiv:" in title:
            try:
                aid_part = title.split("arXiv:")[1].split(" ")[0].rstrip(")]")
                arxiv_id = aid_part
                # Clean title
                title = title.split(". (arXiv:")[0].strip()
            except Exception:
                pass
        desc = (item.findtext("rss:description", default="", namespaces=RSS_NS)
                   .strip().replace("\n", " "))
        creator = item.findtext("dc:creator", default="", namespaces=RSS_NS)
        link = item.findtext("rss:link", default="", namespaces=RSS_NS)
        return {
            "arxiv_id":       arxiv_id,
            "title":          title,
            "authors":        creator,
            "abstract":       desc,
            "categories":     category,
            "submitted_date": None,    # RSS doesn't include date directly
            "updated_date":   None,
            "pdf_url":        link.replace("/abs/", "/pdf/") if link else None,
            "abs_url":        link,
        }
    except Exception as exc:
        logger.warning("RSS item parse failed: %s", exc)
        return None


def fetch_qfin_with_fallback(
    start_date: str, end_date: str,
    *, max_results: int = 500,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    """Try API first; fall back to RSS if API unhealthy (or returns empty).

    Returns the merged-best-available DataFrame.
    """
    api_df = fetch_qfin_papers(
        start_date, end_date,
        max_results=max_results, categories=categories,
    )
    if not api_df.empty:
        return api_df
    logger.info("arxiv API returned empty; trying RSS fallback")
    rss_df = fetch_qfin_rss(categories=categories)
    return rss_df


def _parse_entry(entry) -> dict | None:
    """Parse one Atom entry into a row dict."""
    try:
        arxiv_id_url = entry.findtext("atom:id", default="", namespaces=NS)
        arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else ""
        title = (entry.findtext("atom:title", default="", namespaces=NS)
                  .strip().replace("\n", " "))
        abstract = (entry.findtext("atom:summary", default="", namespaces=NS)
                       .strip().replace("\n", " "))

        authors = []
        for author in entry.findall("atom:author", NS):
            name = author.findtext("atom:name", default="", namespaces=NS)
            if name:
                authors.append(name)

        categories = []
        for cat in entry.findall("atom:category", NS):
            term = cat.attrib.get("term", "")
            if term:
                categories.append(term)

        submitted = entry.findtext("atom:published", default="", namespaces=NS)
        updated = entry.findtext("atom:updated", default="", namespaces=NS)

        # Find PDF link
        pdf_url = None
        abs_url = arxiv_id_url
        for link in entry.findall("atom:link", NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
                break

        return {
            "arxiv_id":       arxiv_id,
            "title":          title,
            "authors":        "; ".join(authors),
            "abstract":       abstract,
            "categories":     "; ".join(categories),
            "submitted_date": submitted[:10] if submitted else None,
            "updated_date":   updated[:10] if updated else None,
            "pdf_url":        pdf_url,
            "abs_url":        abs_url,
        }
    except Exception as exc:
        logger.warning("entry parse failed: %s", exc)
        return None
