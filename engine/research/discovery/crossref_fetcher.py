"""engine/research/discovery/crossref_fetcher.py — fast-indexing fallback
to OpenAlex via Crossref DOI registration API.

SENIOR DESIGN NOTE — why Crossref instead of "SSRN RSS":
  User asked for "SSRN RSS lightweight fallback" to catch papers
  before OpenAlex's 2-4 week indexing lag. Probe 2026-05-30 confirmed
  SSRN exposes NO public RSS (all candidate URLs 403/404/500; SSRN
  bot-blocks even plain abstract pages with 403).

  Crossref is the correct lightweight fallback because:
    - Indexes within DAYS of DOI registration, not weeks
    - Free public REST API, no auth needed
    - Per-venue filter via ISSN (built into our venue YAML)
    - Returns title + authors + venue + DOI + publication date

  Crossref does NOT return abstracts for most records (publisher
  copyright). This is OK because:
    - Title + venue + author still drives credibility scorer ranking
    - LLM extraction (Stage 1) fetches the abstract from the DOI
      via separate publisher API call (downstream concern)
    - Crossref's purpose here is the FRESHNESS SIGNAL — "new paper
      just registered" — not full content delivery.

Architecture (mirrors openalex_fetcher pattern):
  - Reads ISSN field from cross_disciplinary_venues.yaml
  - Per-venue source_health (key: crossref_<issn>)
  - 1.0/sec polite delay (Crossref's etiquette guideline)
  - Cursor pagination via 'cursor' parameter
"""
from __future__ import annotations

import logging
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import yaml

from engine.data import source_health

logger = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org/works"
POLITE_DELAY_SEC = 1.0
DEFAULT_ROWS = 50

REPO_ROOT = Path(__file__).resolve().parents[3]
VENUES_YAML = REPO_ROOT / "data" / "research" / "cross_disciplinary_venues.yaml"


# ── Venue config loader (shared with openalex_fetcher) ─────────────────────

_VENUES_CACHE: dict[str, dict] | None = None


def load_venues() -> dict[str, dict]:
    """{openalex_source_id: {name, short, category, issn, ...}}"""
    global _VENUES_CACHE
    if _VENUES_CACHE is not None:
        return _VENUES_CACHE
    if not VENUES_YAML.exists():
        logger.warning("venue YAML not found: %s", VENUES_YAML)
        _VENUES_CACHE = {}
        return _VENUES_CACHE
    with VENUES_YAML.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _VENUES_CACHE = {k: v for k, v in raw.items()
                       if isinstance(v, dict) and v.get("issn")}
    return _VENUES_CACHE


def get_venues_with_issn() -> list[dict]:
    """List of {issn, name, short, category, credibility_tier,
                 graveyard_routing} for each venue with ISSN set."""
    out = []
    for sid, cfg in load_venues().items():
        if cfg.get("issn"):
            out.append({
                "openalex_id":          sid,
                "issn":                 cfg["issn"],
                "name":                 cfg.get("name", ""),
                "short":                cfg.get("short", ""),
                "category":             cfg.get("category", ""),
                "credibility_tier":     cfg.get("credibility_tier"),
                "graveyard_routing":    cfg.get("graveyard_routing"),
            })
    return out


# ── Single-venue Crossref fetch ───────────────────────────────────────────

def fetch_crossref_venue(
    issn: str,
    start_date: str, end_date: str,
    *,
    max_results: int = 100,
    skip_health_check: bool = False,
    venue_cfg: dict | None = None,
) -> pd.DataFrame:
    """Fetch papers from one Crossref venue (by ISSN) within date range.

    Args:
      issn: e.g. "0025-1909" for Management Science
      start_date / end_date: YYYY-MM-DD
      max_results: per-venue cap
      venue_cfg: pre-loaded venue config dict (for category /
                  credibility_tier / graveyard_routing tagging)
    """
    health_key = f"crossref_{issn}"
    if not skip_health_check:
        healthy, reason = source_health.is_healthy(health_key)
        if not healthy:
            logger.info("crossref_%s unhealthy: %s", issn, reason)
            return _empty_df()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "macro-alpha-research/1.0 (research@local; crossref)",
    })

    venue_cfg = venue_cfg or {}
    rows: list[dict] = []
    cursor = "*"
    fetched = 0

    while fetched < max_results:
        time.sleep(POLITE_DELAY_SEC)
        rows_param = min(DEFAULT_ROWS, max_results - fetched)
        # Crossref query: type=journal-article + ISSN + date range
        filter_str = (
            f"type:journal-article,"
            f"from-pub-date:{start_date},"
            f"until-pub-date:{end_date},"
            f"issn:{issn}"
        )
        params = {
            "filter":  filter_str,
            "rows":    rows_param,
            "cursor":  cursor,
            "select":  ("title,author,issued,DOI,container-title,"
                        "published-online,abstract"),
        }
        url = f"{CROSSREF_BASE}?{urllib.parse.urlencode(params)}"
        try:
            resp = session.get(url, timeout=30)
        except Exception as exc:
            source_health.mark_failure(health_key, "network", str(exc))
            break
        if resp.status_code == 429:
            source_health.mark_failure(health_key, "rate_limited", "HTTP 429")
            break
        if resp.status_code != 200:
            source_health.mark_failure(health_key, "network",
                                          f"HTTP {resp.status_code}")
            break
        try:
            payload = resp.json()
        except Exception as exc:
            source_health.mark_failure(health_key, "schema_unknown",
                                          f"JSON: {exc}")
            break

        msg = payload.get("message", {})
        items = msg.get("items", [])
        if not items:
            break

        for item in items:
            row = _normalize_crossref_record(item, venue_cfg)
            if row:
                rows.append(row)

        fetched += len(items)
        next_cursor = msg.get("next-cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    if rows:
        source_health.mark_success(health_key)
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows)


def _normalize_crossref_record(item: dict, venue_cfg: dict) -> dict | None:
    """Map one Crossref work → unified discovery schema."""
    try:
        doi = item.get("DOI", "").strip()
        if not doi:
            return None

        titles = item.get("title") or []
        title = (titles[0] if titles else "").strip()
        if not title:
            return None

        # Authors: list of {given, family, ...}
        authors = []
        affs = []
        for au in (item.get("author") or []):
            given = (au.get("given") or "").strip()
            family = (au.get("family") or "").strip()
            if family:
                authors.append(f"{family}, {given}" if given else family)
            for aff in (au.get("affiliation") or []):
                aff_name = (aff.get("name") or "").strip()
                if aff_name:
                    affs.append(aff_name)
        authors_str = "; ".join(authors)
        affs_str = "; ".join(sorted(set(affs)))

        # Publication date — Crossref gives "issued" as date-parts
        # or "published-online" — prefer earliest
        sub_date = None
        for date_field in ("published-online", "issued"):
            dp = (item.get(date_field) or {}).get("date-parts") or []
            if dp and dp[0]:
                parts = dp[0]
                if len(parts) >= 3:
                    sub_date = f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
                elif len(parts) >= 2:
                    sub_date = f"{parts[0]:04d}-{parts[1]:02d}-01"
                elif len(parts) >= 1:
                    sub_date = f"{parts[0]:04d}-01-01"
                if sub_date:
                    break

        # Venue
        venue_titles = item.get("container-title") or []
        venue_name = (venue_titles[0] if venue_titles
                        else venue_cfg.get("name", ""))

        # Abstract — most records lack it (publisher copyright)
        abstract = (item.get("abstract") or "").strip()
        # Strip <jats:p> tags if present (Crossref returns JATS XML sometimes)
        if abstract.startswith("<"):
            import re
            abstract = re.sub(r"<[^>]+>", " ", abstract).strip()

        return {
            "source":          "crossref",
            "source_id":       doi,
            "title":           title,
            "abstract":        abstract,
            "authors":         authors_str,
            "affiliations":    affs_str,
            "venue":           venue_name,
            "venue_category":  venue_cfg.get("category", ""),
            "credibility_tier_hint":  venue_cfg.get("credibility_tier"),
            "graveyard_routing":      venue_cfg.get("graveyard_routing"),
            "submitted_date":  sub_date,
            "doi":             doi,
            "abs_url":         f"https://doi.org/{doi}",
            "pdf_url":         None,
            "categories":      venue_cfg.get("category", ""),
            "citation_count":  None,    # Crossref doesn't return this
        }
    except Exception as exc:
        logger.warning("crossref record parse failed: %s", exc)
        return None


# ── Multi-venue fetch ─────────────────────────────────────────────────────

def fetch_crossref_recent(
    days_back: int = 14,
    *,
    max_results_per_venue: int = 50,
    category_filter: tuple[str, ...] = (),
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch recent papers across all curated venues with ISSN.

    Args:
      days_back: how many days back to fetch (default 14 = catches
                  freshly-registered papers OpenAlex hasn't indexed yet)
      max_results_per_venue: per-venue cap
      category_filter: optionally restrict to subset of venue categories
    """
    import datetime
    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=days_back)
              ).isoformat()

    venues = get_venues_with_issn()
    if category_filter:
        venues = [v for v in venues if v["category"] in category_filter]
    if not venues:
        return _empty_df()

    all_rows: list[pd.DataFrame] = []
    for v in venues:
        logger.info("crossref fetch: %s (%s) %s..%s",
                       v["short"], v["issn"], start, end)
        try:
            df = fetch_crossref_venue(
                v["issn"], start, end,
                max_results=max_results_per_venue,
                skip_health_check=skip_health_check,
                venue_cfg=v,
            )
            if len(df):
                all_rows.append(df)
        except Exception as exc:
            logger.warning("venue %s (%s) fetch failed: %s",
                              v["short"], v["issn"], exc)

    if not all_rows:
        return _empty_df()
    return pd.concat(all_rows, ignore_index=True)


# ── Empty schema ──────────────────────────────────────────────────────────

def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "source", "source_id", "title", "abstract", "authors",
        "affiliations", "venue", "venue_category",
        "credibility_tier_hint", "graveyard_routing",
        "submitted_date", "doi", "abs_url", "pdf_url",
        "categories", "citation_count",
    ])


# ── CLI ────────────────────────────────────────────────────────────────────

def _cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--max-per-venue", type=int, default=20)
    parser.add_argument("--category", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cats = (args.category,) if args.category else ()
    df = fetch_crossref_recent(
        days_back=args.days_back,
        max_results_per_venue=args.max_per_venue,
        category_filter=cats,
    )
    print(f"Total recent papers: {len(df)}")
    if len(df):
        print(df.groupby(["venue_category", "venue"]).size().to_string())


if __name__ == "__main__":
    _cli()
