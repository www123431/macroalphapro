"""engine/research/discovery/multi_source_dispatch.py — orchestrator across all
discovery sources.

Two modes, BOTH supported per user 2026-05-30: "过去的也要啊...双线推进"
  HISTORICAL BACKFILL: fetch year-by-year going backwards in time
  NEW FLOW (DAILY/WEEKLY): fetch recent papers across all sources

Sources orchestrated (Phase 8 plan):
  - arxiv q-fin (API + RSS fallback) — historical via date range
  - NBER (API + RSS) — historical via date range
  - Tier-1 RSS feeds (recent only — backfill not via RSS)

Output: unified DataFrame with normalized schema:
  source, source_id, title, authors, abstract, categories, submitted_date,
  updated_date, pdf_url, abs_url
Where source ∈ {arxiv, nber, tier1_rss_<id>}, source_id = arxiv_id/nber_id/etc.

Cross-source dedup: title token-overlap ≥ 0.85 (slightly tighter than
discovery_pipeline's library dedup because intra-source duplicates are
common across NBER ↔ arxiv ↔ journal versions of the same paper).
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

POLITE_INTER_SOURCE_DELAY = 5.0   # between source calls


def _normalize_paper_row(source: str, row: dict) -> dict:
    """Map per-source schema to unified schema.

    All fetchers populate `source_id` (OpenAlex: Work ID, Crossref: DOI,
    arxiv: arxiv_id via arxiv_id field, NBER: nber_id). Per-fetcher
    legacy fields (arxiv_id / nber_id / paper_id) are also checked as
    fallback so each fetcher's original schema still works.

    Propagates metadata downstream (venue/affiliations/doi/
    graveyard_routing) so credibility_scorer + CFR auto-routing
    keep functioning through the dispatcher.
    """
    source_id = (row.get("source_id") or row.get("arxiv_id")
                  or row.get("nber_id") or row.get("paper_id") or "")
    return {
        "source":               source,
        "source_id":             source_id,
        # Legacy arxiv_id field for back-compat with discovery_pipeline
        # which historically read arxiv_id; populated for ANY source now.
        "arxiv_id":              source_id,
        "title":                 row.get("title", ""),
        "authors":                row.get("authors", ""),
        "abstract":              row.get("abstract", ""),
        "affiliations":          row.get("affiliations", ""),
        "venue":                 row.get("venue", ""),
        "venue_category":        row.get("venue_category", ""),
        "credibility_tier_hint": row.get("credibility_tier_hint"),
        "graveyard_routing":      row.get("graveyard_routing"),
        "doi":                    row.get("doi"),
        "categories":            row.get("categories", ""),
        "submitted_date":        row.get("submitted_date"),
        "updated_date":          row.get("updated_date"),
        "pdf_url":                row.get("pdf_url"),
        "abs_url":                row.get("abs_url"),
        "citation_count":        row.get("citation_count"),
    }


def _dedup_across_sources(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Remove cross-source duplicates by title token-overlap."""
    if df.empty:
        return df
    df = df.copy()
    df["_title_tokens"] = df["title"].fillna("").str.lower().apply(
        lambda s: set(s.split()) - {"a", "the", "of", "and", "in", "on", "for", "to"}
    )
    keep_indices = []
    seen_token_sets: list[set] = []
    for idx, row in df.iterrows():
        tokens = row["_title_tokens"]
        if not tokens:
            keep_indices.append(idx)
            continue
        is_dup = False
        for seen in seen_token_sets:
            if not seen:
                continue
            overlap = len(tokens & seen) / max(len(tokens | seen), 1)
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep_indices.append(idx)
            seen_token_sets.append(tokens)
    out = df.loc[keep_indices].drop(columns=["_title_tokens"])
    return out.reset_index(drop=True)


def fetch_new_flow(*, max_results_per_source: int = 50) -> pd.DataFrame:
    """RECENT papers from all sources. For daily/weekly cron.

    Per [[feedback-iterate-and-solve-inflight-2026-05-29]]: each source
    independent — one failing doesn't break the rest.
    """
    rows: list[dict] = []

    # arxiv: try API first, fall back to RSS
    try:
        from engine.research.discovery import arxiv_qfin_fetcher
        time.sleep(POLITE_INTER_SOURCE_DELAY)
        end = datetime.date.today().isoformat()
        start = (datetime.date.today() - datetime.timedelta(days=14)).isoformat()
        arxiv_df = arxiv_qfin_fetcher.fetch_qfin_with_fallback(
            start, end, max_results=max_results_per_source,
        )
        for _, row in arxiv_df.iterrows():
            rows.append(_normalize_paper_row("arxiv", row.to_dict()))
    except Exception as exc:
        logger.warning("arxiv new-flow failed: %s", exc)

    # NBER: RSS endpoint is sufficient for new flow
    try:
        from engine.research.discovery import nber_fetcher
        time.sleep(POLITE_INTER_SOURCE_DELAY)
        nber_df = nber_fetcher.fetch_nber_recent(max_results=max_results_per_source)
        for _, row in nber_df.iterrows():
            rows.append(_normalize_paper_row("nber", row.to_dict()))
    except Exception as exc:
        logger.warning("nber new-flow failed: %s", exc)

    # Tier-1 RSS
    try:
        from engine.research.discovery import tier1_rss_fetcher
        time.sleep(POLITE_INTER_SOURCE_DELAY)
        rss_df = tier1_rss_fetcher.fetch_tier1_rss(
            max_per_feed=max_results_per_source,
        )
        for _, row in rss_df.iterrows():
            rows.append(_normalize_paper_row(
                f"tier1_rss_{row.get('journal', 'unknown')}",
                row.to_dict(),
            ))
    except Exception as exc:
        logger.warning("tier1 RSS new-flow failed: %s", exc)

    # Cross-disciplinary (OpenAlex) — Senior roadmap #3
    try:
        from engine.research.discovery import openalex_fetcher
        time.sleep(POLITE_INTER_SOURCE_DELAY)
        end = datetime.date.today().isoformat()
        start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        xd_df = openalex_fetcher.fetch_cross_disciplinary(
            start, end, max_results_per_venue=max_results_per_source,
        )
        for _, row in xd_df.iterrows():
            rows.append(_normalize_paper_row(
                f"openalex_{row.get('venue_category', 'xd')}",
                row.to_dict(),
            ))
    except Exception as exc:
        logger.warning("openalex cross-disciplinary new-flow failed: %s", exc)

    # Crossref FRESH-pass — catches DOI-registered papers that OpenAlex's
    # 2-4 week indexer hasn't picked up yet. Last 14 days, capped low.
    try:
        from engine.research.discovery import crossref_fetcher
        time.sleep(POLITE_INTER_SOURCE_DELAY)
        cr_df = crossref_fetcher.fetch_crossref_recent(
            days_back=14, max_results_per_venue=max_results_per_source // 2,
        )
        for _, row in cr_df.iterrows():
            rows.append(_normalize_paper_row(
                f"crossref_{row.get('venue_category', 'xd')}",
                row.to_dict(),
            ))
    except Exception as exc:
        logger.warning("crossref fresh-pass new-flow failed: %s", exc)

    if not rows:
        return pd.DataFrame(columns=[
            "source", "source_id", "title", "authors", "abstract",
            "categories", "submitted_date", "updated_date",
            "pdf_url", "abs_url",
        ])
    df = pd.DataFrame(rows)
    df = _dedup_across_sources(df)
    return df


def fetch_historical_backfill(
    start_date: str, end_date: str,
    *,
    max_results_per_year: int = 500,
    sources: list[str] | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> pd.DataFrame:
    """HISTORICAL backfill — slice the date range by year + pull each year.

    Per user 2026-05-30: "过去的也要啊...双线推进". This is the historical-
    track companion to fetch_new_flow.

    Args:
      start_date: YYYY-MM-DD (earliest)
      end_date:   YYYY-MM-DD (most recent)
      max_results_per_year: per-source per-year cap
      sources: subset of {'arxiv', 'nber'}. NBER historical is currently
               UNAVAILABLE (NBER has no public JSON API and pages are
               JS-rendered — see nber_fetcher.fetch_nber_api docstring),
               so backfill default is 'arxiv' only. Tier-1 RSS NOT
               included because it only serves recent.
      progress_callback: optional fn(source_name, papers_count) for UI

    Returns: unified deduped DataFrame.
    """
    sources = sources or ["arxiv"]
    rows: list[dict] = []

    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    for year in range(end_year, start_year - 1, -1):    # most recent first
        year_start = max(start_date, f"{year}-01-01")
        year_end = min(end_date, f"{year}-12-31")
        if year_start > year_end:
            continue

        if "arxiv" in sources:
            try:
                from engine.research.discovery import arxiv_qfin_fetcher
                time.sleep(POLITE_INTER_SOURCE_DELAY)
                arxiv_df = arxiv_qfin_fetcher.fetch_qfin_papers(
                    year_start, year_end, max_results=max_results_per_year,
                )
                for _, row in arxiv_df.iterrows():
                    rows.append(_normalize_paper_row("arxiv", row.to_dict()))
                if progress_callback:
                    progress_callback(f"arxiv {year}", len(arxiv_df))
            except Exception as exc:
                logger.warning("arxiv backfill %s failed: %s", year, exc)

        if "nber" in sources:
            try:
                from engine.research.discovery import nber_fetcher
                time.sleep(POLITE_INTER_SOURCE_DELAY)
                nber_df = nber_fetcher.fetch_nber_api(
                    year_start, year_end, max_results=max_results_per_year,
                )
                for _, row in nber_df.iterrows():
                    rows.append(_normalize_paper_row("nber", row.to_dict()))
                if progress_callback:
                    progress_callback(f"nber {year}", len(nber_df))
            except Exception as exc:
                logger.warning("nber backfill %s failed: %s", year, exc)

    if not rows:
        return pd.DataFrame(columns=[
            "source", "source_id", "title", "authors", "abstract",
            "categories", "submitted_date", "updated_date",
            "pdf_url", "abs_url",
        ])
    df = pd.DataFrame(rows)
    df = _dedup_across_sources(df)
    return df
