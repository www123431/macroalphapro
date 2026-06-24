"""engine/research/discovery/nber_fetcher.py — NBER Working Papers.

Free official NBER feed. Two access paths:
  Layer 1: RSS feed at https://www.nber.org/rss/new.xml (recent papers only)
  Layer 2: JSON API at https://www.nber.org/api/v1/working_papers (paginated)

Both polite-paced; NBER is academic-friendly but still avoid hammering.

Source-health integrated (same pattern as arxiv_qfin_fetcher).

Output schema: same as arxiv fetcher (arxiv_id replaced by nber_id):
  nber_id (e.g. "w32123"), title, authors, abstract, categories (JEL codes),
  submitted_date, updated_date, pdf_url, abs_url
"""
from __future__ import annotations

import logging
import time
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd

from engine.data import source_health

logger = logging.getLogger(__name__)

NBER_API_BASE = "https://www.nber.org/api/v1/working_papers"
NBER_RSS_URL = "https://www.nber.org/rss/new.xml"

POLITE_DELAY_SEC = 3.0
RESULTS_PER_PAGE = 50

# JEL codes for finance/economics research most relevant to factor research
TARGET_JEL_CODES = ("G10", "G11", "G12", "G14", "G15")
RSS_NS = {
    "rss": "",
    "dc":  "http://purl.org/dc/elements/1.1/",
}


def fetch_nber_recent(
    *,
    max_results: int = 50,
    target_jel_codes: tuple[str, ...] = TARGET_JEL_CODES,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch latest NBER WPs via RSS feed (Layer 1).

    RSS is most-recent-first; max ~50 entries. Use for daily polling.
    """
    if not skip_health_check:
        healthy, reason = source_health.is_healthy("nber_rss")
        if not healthy:
            logger.info("nber_rss unhealthy: %s", reason)
            return _empty_df()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "macro-alpha-research/1.0 (research@local; nber-recent)",
    })
    time.sleep(POLITE_DELAY_SEC)

    try:
        resp = session.get(NBER_RSS_URL, timeout=30)
    except Exception as exc:
        source_health.mark_failure("nber_rss", "network", str(exc))
        return _empty_df()
    if resp.status_code == 429:
        source_health.mark_failure("nber_rss", "rate_limited",
                                      f"HTTP 429")
        return _empty_df()
    if resp.status_code != 200:
        source_health.mark_failure("nber_rss", "network",
                                      f"HTTP {resp.status_code}")
        return _empty_df()

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        source_health.mark_failure("nber_rss", "schema_unknown",
                                      f"XML parse: {exc}")
        return _empty_df()

    rows = []
    for item in root.findall(".//item"):
        row = _parse_rss_item(item, target_jel_codes)
        if row:
            rows.append(row)
    if rows:
        source_health.mark_success("nber_rss")
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows).head(max_results)


def fetch_nber_api(
    start_date: str, end_date: str,
    *,
    max_results: int = 200,
    target_jel_codes: tuple[str, ...] = TARGET_JEL_CODES,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch NBER WPs via JSON API for a date range (Layer 2).

    KNOWN LIMITATION 2026-05-30: NBER does not expose a public REST API
    (api/v1/working_papers returns 404; browse pages are Vue/Nuxt-rendered
    with paper data injected client-side). This function is kept as a
    stub for two reasons:
      (1) future option of HTML scraping with Selenium/Playwright;
      (2) future option if NBER ever publishes a real API.
    For now, historical NBER backfill is NOT AVAILABLE via this fetcher.
    Use the RSS feed (fetch_nber_recent) for new-flow only.

    Per [[feedback-no-brittle-hardcoding-2026-05-30]] (3-layer pattern):
    NBER historical = Layer 1 (RSS) unavailable past ~50 recent entries;
    Layer 2 (HTML scrape) deferred until a concrete need emerges.
    """
    logger.warning(
        "fetch_nber_api: NBER has no public JSON API as of 2026-05-30. "
        "Use fetch_nber_recent (RSS) for new-flow; backfill NOT available."
    )
    # Mark unhealthy so the rest of the orchestrator skips it gracefully.
    if not skip_health_check:
        source_health.mark_failure(
            "nber_api", "schema_unknown",
            "NBER has no public JSON API; HTML pages are JS-rendered",
        )
    return _empty_df()


def _fetch_nber_api_legacy_impl(
    start_date: str, end_date: str,
    *,
    max_results: int = 200,
    target_jel_codes: tuple[str, ...] = TARGET_JEL_CODES,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Original implementation kept for reference if NBER publishes an
    API in the future. Currently 404s on all known endpoints."""
    if not skip_health_check:
        healthy, reason = source_health.is_healthy("nber_api")
        if not healthy:
            logger.info("nber_api unhealthy: %s", reason)
            return _empty_df()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "macro-alpha-research/1.0 (research@local; nber-api)",
        "Accept":     "application/json",
    })

    rows = []
    page = 1
    fetched = 0
    while fetched < max_results:
        time.sleep(POLITE_DELAY_SEC)
        params = {
            "page":     page,
            "per_page": min(RESULTS_PER_PAGE, max_results - fetched),
        }
        url = f"{NBER_API_BASE}?{urllib.parse.urlencode(params)}"
        try:
            resp = session.get(url, timeout=30)
        except Exception as exc:
            source_health.mark_failure("nber_api", "network", str(exc))
            break
        if resp.status_code == 429:
            source_health.mark_failure("nber_api", "rate_limited", "HTTP 429")
            break
        if resp.status_code != 200:
            source_health.mark_failure("nber_api", "network",
                                          f"HTTP {resp.status_code}")
            break
        try:
            payload = resp.json()
        except Exception as exc:
            source_health.mark_failure("nber_api", "schema_unknown",
                                          f"JSON parse: {exc}")
            break

        # NBER API response shape can vary; handle common patterns
        results = payload.get("results", payload if isinstance(payload, list) else [])
        if not results:
            break

        for item in results:
            row = _parse_api_item(item, start_date, end_date, target_jel_codes)
            if row:
                rows.append(row)

        fetched += len(results)
        if len(results) < RESULTS_PER_PAGE:
            break
        page += 1

    if rows:
        source_health.mark_success("nber_api")
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows)


# ── parsers ─────────────────────────────────────────────────────────────

def _parse_rss_item(item, target_jel_codes: tuple) -> dict | None:
    """Parse one NBER RSS item."""
    try:
        title = (item.findtext("title", default="") or "").strip()
        desc = (item.findtext("description", default="") or "").strip()
        link = (item.findtext("link", default="") or "").strip()
        pub_date = item.findtext("pubDate", default="")
        creator = item.findtext("{http://purl.org/dc/elements/1.1/}creator",
                                  default="")

        # Extract NBER ID from link (e.g. https://www.nber.org/papers/w32123)
        nber_id = ""
        if "/papers/" in link:
            nber_id = link.split("/papers/")[-1].rstrip("/")

        # Filter by JEL if categories present (RSS may include in description)
        # For now, include everything from RSS; let downstream LLM filter
        return {
            "nber_id":        nber_id,
            "title":          title,
            "authors":        creator,
            "abstract":       desc,
            "categories":     "",   # RSS doesn't usually include JEL
            "submitted_date": _parse_pub_date(pub_date),
            "updated_date":   None,
            "pdf_url":        f"https://www.nber.org/system/files/working_papers/{nber_id}/{nber_id}.pdf"
                                if nber_id else None,
            "abs_url":        link,
        }
    except Exception as exc:
        logger.warning("NBER RSS parse failed: %s", exc)
        return None


def _parse_api_item(item: dict, start_date: str, end_date: str,
                      target_jel_codes: tuple) -> dict | None:
    """Parse one NBER API JSON item."""
    try:
        nber_id = str(item.get("paper", "") or item.get("number", ""))
        if nber_id and not nber_id.startswith("w"):
            nber_id = f"w{nber_id}"
        title = str(item.get("title", "")).strip()
        abstract = str(item.get("abstract", "")).strip()

        # Authors: list of names or list of dicts
        authors_raw = item.get("authors", []) or item.get("author", [])
        if isinstance(authors_raw, list):
            author_names = []
            for a in authors_raw:
                if isinstance(a, dict):
                    name = a.get("full_name") or a.get("name", "")
                else:
                    name = str(a)
                if name:
                    author_names.append(name)
            authors = "; ".join(author_names)
        else:
            authors = str(authors_raw)

        # JEL codes: filter if target set non-empty
        jel = item.get("jel_codes", []) or item.get("jel", [])
        jel_str = "; ".join(str(j) for j in jel) if isinstance(jel, list) else str(jel)
        if target_jel_codes:
            matching = [c for c in (jel if isinstance(jel, list) else [])
                          if any(str(c).startswith(tc) for tc in target_jel_codes)]
            if not matching:
                return None    # filter out non-finance

        # Date
        pub_date_raw = (item.get("public_date") or item.get("issued_date")
                          or item.get("created_at") or "")
        sub_date = pub_date_raw[:10] if pub_date_raw else None
        if sub_date and (sub_date < start_date or sub_date > end_date):
            return None

        return {
            "nber_id":        nber_id,
            "title":          title,
            "authors":        authors,
            "abstract":       abstract,
            "categories":     jel_str,
            "submitted_date": sub_date,
            "updated_date":   None,
            "pdf_url":        item.get("pdf_url"),
            "abs_url":        item.get("url") or f"https://www.nber.org/papers/{nber_id}",
        }
    except Exception as exc:
        logger.warning("NBER API parse failed: %s", exc)
        return None


def _parse_pub_date(s: str) -> str | None:
    """RSS pubDate like 'Mon, 15 Jan 2024 00:00:00 GMT' → YYYY-MM-DD."""
    if not s:
        return None
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(s).date().isoformat()
    except Exception:
        return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "nber_id", "title", "authors", "abstract", "categories",
        "submitted_date", "updated_date", "pdf_url", "abs_url",
    ])
