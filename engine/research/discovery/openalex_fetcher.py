"""engine/research/discovery/openalex_fetcher.py — cross-disciplinary
venue fetcher via OpenAlex API.

Per [[project-senior-pipeline-roadmap-2026-05-30]] roadmap #3:
quant alpha hides in accounting / operations / marketing literatures
that finance practitioners don't systematically read. OpenAlex is the
right transport because it's:
  - Free, public, no API key required
  - Indexes 200M+ scholarly works including TAR/JAR/MS/JMR + SSRN
    preprint mirror
  - Returns structured JSON: title, abstract (inverted index),
    authors, venue, DOI, cite count
  - Supports per-source filter (we have curated 7 venue IDs)

Architecture (Layer 1 primary, falls back gracefully on error):
  1. Iterate curated venue list from cross_disciplinary_venues.yaml
  2. For each venue, fetch papers within date range via API
  3. Normalize abstract from inverted-index format
  4. Tag papers with category + credibility_tier from YAML
  5. Critical Finance Review papers carry graveyard_routing="auto_negative_evidence"
     flag (downstream discovery_pipeline routes them to graveyard
     instead of normal LLM extraction)

Source-health integrated. Polite rate limit (1 req/sec — OpenAlex's
guideline for unauthenticated use).
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

OPENALEX_BASE = "https://api.openalex.org/works"
POLITE_DELAY_SEC = 1.1     # OpenAlex unauthenticated guideline = 1/sec
DEFAULT_PER_PAGE = 50      # max 200; smaller = lower memory + better
                            # interleaving of source-health failures

REPO_ROOT = Path(__file__).resolve().parents[3]
VENUES_YAML = REPO_ROOT / "data" / "research" / "cross_disciplinary_venues.yaml"


# ── Venue config loader ───────────────────────────────────────────────────

_VENUES_CACHE: dict[str, dict] | None = None


def load_venues() -> dict[str, dict]:
    """{openalex_source_id: {name, short, category, credibility_tier, ...}}"""
    global _VENUES_CACHE
    if _VENUES_CACHE is not None:
        return _VENUES_CACHE
    if not VENUES_YAML.exists():
        logger.warning("venue YAML not found: %s", VENUES_YAML)
        _VENUES_CACHE = {}
        return _VENUES_CACHE
    with VENUES_YAML.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _VENUES_CACHE = {k: v for k, v in raw.items() if isinstance(v, dict)}
    return _VENUES_CACHE


# ── Abstract decoder ──────────────────────────────────────────────────────

def _abstract_from_inverted_index(idx: dict | None) -> str:
    """OpenAlex stores abstracts as inverted-index for licensing reasons.
    Reconstruct plain text: position → word."""
    if not idx:
        return ""
    pos_word = []
    for word, positions in idx.items():
        for p in positions:
            pos_word.append((p, word))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)


# ── Single-venue fetch ────────────────────────────────────────────────────

def fetch_openalex_venue(
    source_id: str,
    start_date: str, end_date: str,
    *,
    max_results: int = 200,
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch papers from one OpenAlex venue within date range.

    Args:
      source_id:  OpenAlex source ID, e.g. "S160506855" (TAR)
      start_date: YYYY-MM-DD
      end_date:   YYYY-MM-DD
      max_results: per-venue cap
      skip_health_check: bypass source_health (testing only)

    Returns: DataFrame with normalized schema (title/abstract/authors/
      venue/submitted_date/doi/abs_url/citation_count + graveyard_routing
      flag from venue config).
    """
    health_key = f"openalex_{source_id}"
    if not skip_health_check:
        healthy, reason = source_health.is_healthy(health_key)
        if not healthy:
            logger.info("openalex_%s unhealthy: %s", source_id, reason)
            return _empty_df()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "macro-alpha-research/1.0 (research@local; openalex)",
        "Accept":     "application/json",
    })

    venue_cfg = load_venues().get(source_id, {})
    rows: list[dict] = []
    cursor = "*"      # OpenAlex cursor-based pagination
    fetched = 0

    while fetched < max_results:
        time.sleep(POLITE_DELAY_SEC)
        per_page = min(DEFAULT_PER_PAGE, max_results - fetched)
        params = {
            "filter":    f"primary_location.source.id:{source_id},"
                           f"from_publication_date:{start_date},"
                           f"to_publication_date:{end_date}",
            "per-page":  per_page,
            "cursor":    cursor,
            "select":    ("id,title,abstract_inverted_index,authorships,"
                          "primary_location,publication_date,doi,"
                          "cited_by_count"),
        }
        url = f"{OPENALEX_BASE}?{urllib.parse.urlencode(params)}"
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

        results = payload.get("results", [])
        if not results:
            break

        for item in results:
            row = _normalize_openalex_record(item, venue_cfg)
            if row:
                rows.append(row)

        fetched += len(results)
        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor

    if rows:
        source_health.mark_success(health_key)
    if not rows:
        return _empty_df()
    return pd.DataFrame(rows)


def _normalize_openalex_record(item: dict, venue_cfg: dict) -> dict | None:
    """Map one OpenAlex Work → unified discovery schema."""
    try:
        # ID — strip URL prefix
        openalex_id = (item.get("id") or "").split("/")[-1]
        if not openalex_id:
            return None

        title = (item.get("title") or "").strip()
        if not title:
            return None

        abstract = _abstract_from_inverted_index(
            item.get("abstract_inverted_index")
        )

        # Authors — flatten authorship list to "First Last; First Last"
        authors = []
        affs = []
        for au in (item.get("authorships") or []):
            person = (au.get("author") or {})
            name = (person.get("display_name") or "").strip()
            if name:
                authors.append(name)
            for inst in (au.get("institutions") or []):
                affs.append((inst.get("display_name") or "").strip())
        authors_str = "; ".join(authors)
        affs_str = "; ".join(sorted(set(affs)))

        # Venue
        primary_loc = item.get("primary_location") or {}
        source_info = primary_loc.get("source") or {}
        venue_name = source_info.get("display_name") or venue_cfg.get("name", "")

        # DOI
        doi = item.get("doi", "")
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        return {
            "source":          "openalex",
            "source_id":       openalex_id,
            "title":           title,
            "abstract":        abstract,
            "authors":         authors_str,
            "affiliations":    affs_str,
            "venue":           venue_name,
            "venue_category":  venue_cfg.get("category", ""),
            "credibility_tier_hint":  venue_cfg.get("credibility_tier"),
            "graveyard_routing":      venue_cfg.get("graveyard_routing"),
            "submitted_date":  item.get("publication_date"),
            "doi":             doi or None,
            "abs_url":         item.get("id"),    # openalex URL is the canonical ref
            "pdf_url":         None,
            "categories":      venue_cfg.get("category", ""),
            "citation_count":  int(item.get("cited_by_count", 0) or 0),
        }
    except Exception as exc:
        logger.warning("openalex record parse failed: %s", exc)
        return None


# ── Multi-venue fetch ─────────────────────────────────────────────────────

def fetch_cross_disciplinary(
    start_date: str, end_date: str,
    *,
    max_results_per_venue: int = 100,
    category_filter: tuple[str, ...] = (),
    skip_health_check: bool = False,
) -> pd.DataFrame:
    """Fetch from all curated cross-disciplinary venues + concatenate.

    Args:
      category_filter: if non-empty, restrict to venues with matching
                        category (e.g. ('accounting', 'replication'))
      max_results_per_venue: cap per individual venue
    """
    venues = load_venues()
    if category_filter:
        venues = {sid: cfg for sid, cfg in venues.items()
                    if cfg.get("category") in category_filter}

    if not venues:
        return _empty_df()

    all_rows: list[pd.DataFrame] = []
    for source_id, cfg in venues.items():
        logger.info("openalex fetch: %s (%s)", cfg.get("short"), source_id)
        try:
            df = fetch_openalex_venue(
                source_id, start_date, end_date,
                max_results=max_results_per_venue,
                skip_health_check=skip_health_check,
            )
            if len(df):
                all_rows.append(df)
        except Exception as exc:
            logger.warning("venue %s fetch failed: %s", source_id, exc)

    if not all_rows:
        return _empty_df()
    return pd.concat(all_rows, ignore_index=True)


# ── empty schema ──────────────────────────────────────────────────────────

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
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--max-per-venue", type=int, default=20)
    parser.add_argument("--category", default="", help="filter to one category")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cats = (args.category,) if args.category else ()
    df = fetch_cross_disciplinary(
        args.start, args.end,
        max_results_per_venue=args.max_per_venue,
        category_filter=cats,
    )
    print(f"Total papers: {len(df)}")
    if len(df):
        print(df.groupby(["venue_category", "venue"]).size().to_string())


if __name__ == "__main__":
    _cli()
