"""engine.agents.papers_curator.watchlist_crawler — Stage A piece 2.

Walks the adversarial author watchlist (engine.agents.papers_curator.
watchlist) and ingests recent papers per author into the existing
cache.jsonl via the existing store.save_new_candidates dedup pipe.

Downstream is unchanged: filter (Deepseek) + summarize (Deepseek) +
A's synthesis all read cache.jsonl. They don't care whether a row
came from arxiv RSS or Semantic Scholar via watchlist — the source
field just changes.

Design choices per project_anti_rut_doctrine_2026-06-07.md:
  - Lazy author_id resolution: watchlist YAML can ship with name +
    rationale only; the crawler resolves SS author_ids on first run
    and persists them back. Lets the principal edit the watchlist
    without knowing SS internal ids.
  - Per-author cap: max papers_per_author per crawl (default 10) so
    one prolific author doesn't dominate the substrate.
  - Recency filter: only papers from last `lookback_years` (default
    2). Older work isn't anti-rut signal — it's stale.
  - Skip cost discipline: empty watchlist → 0 SS calls. Same for
    authors with last_crawled_ts within the last day (don't re-hit
    the API for someone we just polled).

Returns a structured result so chief_of_staff orchestrator can
include it in the weekly memo ('this week's substrate enrichment').
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

from engine.agents.papers_curator.crawler import PaperCandidate
from engine.agents.papers_curator.watchlist import (
    WatchlistAuthor,
    load_watchlist,
    resolve_author_id,
    update_after_crawl,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _was_crawled_within(author: WatchlistAuthor, *, hours: int = 24) -> bool:
    """True if this author was crawled within the last N hours.
    Cost discipline: don't re-hit SS for someone polled today."""
    if not author.last_crawled_ts:
        return False
    try:
        last = _dt.datetime.fromisoformat(
            author.last_crawled_ts.replace("Z", "")
        )
    except ValueError:
        return False
    return (_dt.datetime.utcnow() - last).total_seconds() < hours * 3600


def _to_paper_candidate(ss_paper, *, fetched_ts: str) -> PaperCandidate:
    """Adapt Semantic Scholar PaperSummary → cache.jsonl PaperCandidate.

    Some SS papers don't have an arxiv id (working papers / journals).
    For those we use the SS paper_id as source_id, with source=
    'semantic_scholar'. Dedup with arxiv-source records still works
    because (source, source_id) is the dedup key.
    """
    return PaperCandidate(
        source        = "semantic_scholar",
        source_id     = ss_paper.paper_id,
        title         = ss_paper.title,
        authors       = ss_paper.author_names,
        abstract      = ss_paper.abstract,
        abs_url       = ss_paper.url,
        # SS doesn't give a direct PDF; leave empty
        pdf_url       = "",
        published_ts  = f"{ss_paper.year}-01-01T00:00:00Z" if ss_paper.year
                         else "",
        # Use venue as a single-element category; downstream filter+summary
        # ignores categories
        categories    = (ss_paper.venue,) if ss_paper.venue else (),
        fetched_ts    = fetched_ts,
    )


def crawl_watchlist(
    *,
    papers_per_author:  int = 10,
    lookback_years:     int = 2,
    skip_recent_hours:  int = 24,
    watchlist_path:     Optional[Path] = None,
) -> dict:
    """Run one watchlist crawl. Returns a structured result for
    chief_of_staff to surface in the weekly memo.

    Returns:
      {
        run_ts:                iso
        n_authors_total:       int  # in watchlist
        n_authors_crawled:     int  # actually hit SS this run
        n_authors_skipped:     int  # crawled recently OR unresolvable
        n_papers_fetched:      int  # before dedup
        n_papers_new:          int  # after store.save_new_candidates dedup
        unresolved_names:      list[str]  # couldn't find via SS search
        errors:                list[str]
      }
    """
    from engine.agents.papers_curator.semantic_scholar import author_papers
    from engine.agents.papers_curator.store import save_new_candidates

    result = {
        "run_ts":                _utc_iso(),
        "n_authors_total":       0,
        "n_authors_crawled":     0,
        "n_authors_skipped":     0,
        "n_papers_fetched":      0,
        "n_papers_new":          0,
        "unresolved_names":      [],
        "errors":                [],
    }

    authors = load_watchlist(path=watchlist_path)
    result["n_authors_total"] = len(authors)
    if not authors:
        return result

    min_year = _dt.datetime.utcnow().year - lookback_years
    candidates: list[PaperCandidate] = []
    fetched_ts = _utc_iso()

    for author in authors:
        # 1. Skip if recently crawled
        if _was_crawled_within(author, hours=skip_recent_hours):
            result["n_authors_skipped"] += 1
            continue

        # 2. Resolve author_id if missing
        aid = author.author_id
        if not aid:
            try:
                aid = resolve_author_id(author.name, path=watchlist_path)
            except Exception as exc:
                logger.warning("watchlist: resolve failed for %s: %s",
                                author.name, exc)
                aid = None
            if not aid:
                result["unresolved_names"].append(author.name)
                result["n_authors_skipped"] += 1
                continue

        # 3. Fetch recent papers
        try:
            ss_papers = author_papers(
                aid,
                limit    = papers_per_author,
                min_year = min_year,
            )
        except Exception as exc:
            logger.exception("watchlist: SS author_papers failed for %s",
                              author.name)
            result["errors"].append(f"author:{author.name}: {exc}")
            continue

        result["n_authors_crawled"] += 1
        result["n_papers_fetched"]  += len(ss_papers)

        # 4. Adapt → PaperCandidate
        for sp in ss_papers:
            if not sp.paper_id:
                continue
            candidates.append(_to_paper_candidate(sp, fetched_ts=fetched_ts))

        # 5. Persist last_crawled_ts so next run prioritizes others
        try:
            update_after_crawl(aid, path=watchlist_path)
        except Exception as exc:
            logger.warning("watchlist: update_after_crawl failed: %s", exc)

    # 6. Bulk write with existing dedup
    if candidates:
        try:
            n_new = save_new_candidates(candidates)
            result["n_papers_new"] = n_new
        except Exception as exc:
            logger.exception("watchlist: save_new_candidates failed")
            result["errors"].append(f"persist: {exc}")

    # 7. Outage detection — if every author failed to crawl AND none
    # were skipped because of skip_recent_hours, that's an upstream
    # SS outage / auth failure / network down. Silent unresolveds
    # would otherwise mask this from chief_of_staff (caught
    # 2026-06-07 failure-surface walk).
    n_total       = result["n_authors_total"]
    n_crawled     = result["n_authors_crawled"]
    n_unresolved  = len(result["unresolved_names"])
    if n_total > 0 and n_crawled == 0 and n_unresolved == n_total:
        result["errors"].append(
            f"outage_suspected: 0/{n_total} authors crawled; "
            f"all {n_unresolved} unresolved via SS — check API key "
            "or network"
        )

    return result
