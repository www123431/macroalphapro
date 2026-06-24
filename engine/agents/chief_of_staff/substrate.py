"""engine.agents.chief_of_staff.substrate — Stage A piece 7a.

Weekly substrate refresh: compose all 5 crawlers in sequence and
write a structured weekly result.

Why this lives in chief_of_staff (not its own agent):
The substrate IS the precondition for A's synthesis — without fresh
papers, A has nothing to synthesize from. Per the 6-step weekly
flow in [[spec-research-session-orchestrator-2026-06-06]], this is
'step 0' (precedes D/A/B). chief_of_staff owns the weekly rhythm;
substrate refresh belongs in its module.

Why standalone (not wired into run_weekly_session yet):
Piece-by-piece doctrine (see [[feedback-piece-by-piece-not-batch-2026-06-05]]).
This commit ships the substrate-refresh primitive + persistence + tests.
Piece 7b will wire it into run_weekly_session as a new step 0.
Splitting lets the user walk failure surfaces between commits.

The 5 crawlers + their entry points:
  arxiv         engine.agents.papers_curator.crawler.crawl_arxiv_qfin
                (returns list[PaperCandidate]; we wrap with persist)
  nber          engine.agents.papers_curator.nber_rss_crawler.crawl_and_persist_nber
  ssrn          engine.agents.papers_curator.ssrn_crossref_crawler.crawl_and_persist_ssrn
  watchlist     engine.agents.papers_curator.watchlist_crawler.crawl_watchlist
  forward       engine.agents.papers_curator.forward_citation_crawler.crawl_forward_citations

Each crawler returns a structured result dict already; we collect
them into a unified weekly result + persist to disk + return.

Error isolation: per-crawler exceptions are caught + recorded; the
next crawler still runs. Same fail-safe contract as run_weekly_session.

dry_run: if True, crawlers either skip their writes or run only the
fetch step. arxiv has no dry-run primitive so we skip it entirely
when dry_run; the others honor it through their existing wrappers
(or skip persistence).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUBSTRATE_DIR = (_REPO_ROOT / "data" / "agents" / "chief_of_staff"
                  / "weekly_substrate")


ALL_SOURCES = ("arxiv", "nber", "ssrn", "watchlist", "forward_citations")


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_str() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


@_dc.dataclass(frozen=True)
class SubstrateRunResult:
    """Typed result of one weekly substrate refresh.

    Each source has its own result dict (the same shape the source's
    own runner produced). Roll-up totals are convenience fields.
    """
    run_ts:                       str
    run_date:                     str
    dry_run:                      bool
    enabled_sources:              tuple[str, ...]
    arxiv_result:                 dict
    nber_result:                  dict
    ssrn_result:                  dict
    watchlist_result:             dict
    forward_citation_result:      dict
    total_fetched:                int
    total_new:                    int
    errors:                       list

    def to_dict(self) -> dict:
        d = _dc.asdict(self)
        # Tuples → lists for JSON
        d["enabled_sources"] = list(self.enabled_sources)
        return d


# ────────────────────────────────────────────────────────────────────
# Per-source wrappers — translate each crawler's return into a unified
# {"n_fetched", "n_new", "errors", ...source_specific} dict
# ────────────────────────────────────────────────────────────────────
def _run_arxiv(*, max_results: int) -> dict:
    """arxiv has no built-in persist wrapper — we do the dedup-write
    inline so the result dict matches the others."""
    from engine.agents.papers_curator.crawler import crawl_arxiv_qfin
    from engine.agents.papers_curator.store import save_new_candidates

    result = {
        "source":    "arxiv",
        "n_fetched": 0,
        "n_new":     0,
        "errors":    [],
    }
    try:
        cands = crawl_arxiv_qfin(max_results=max_results)
    except Exception as exc:
        logger.exception("substrate: arxiv fetch raised")
        result["errors"].append(f"fetch: {exc}")
        return result
    result["n_fetched"] = len(cands)
    if not cands:
        return result
    try:
        result["n_new"] = save_new_candidates(cands)
    except Exception as exc:
        logger.exception("substrate: arxiv persist raised")
        result["errors"].append(f"persist: {exc}")
    return result


def _run_nber() -> dict:
    from engine.agents.papers_curator.nber_rss_crawler import (
        crawl_and_persist_nber,
    )
    return crawl_and_persist_nber()


def _run_ssrn(*, lookback_days: int, max_results: int) -> dict:
    from engine.agents.papers_curator.ssrn_crossref_crawler import (
        crawl_and_persist_ssrn,
    )
    return crawl_and_persist_ssrn(
        lookback_days=lookback_days,
        max_results=max_results,
    )


def _run_watchlist(*, papers_per_author: int,
                    lookback_years: int,
                    skip_recent_hours: int) -> dict:
    from engine.agents.papers_curator.watchlist_crawler import (
        crawl_watchlist,
    )
    return crawl_watchlist(
        papers_per_author = papers_per_author,
        lookback_years    = lookback_years,
        skip_recent_hours = skip_recent_hours,
    )


def _run_forward_citations(*, max_per_seed: int,
                             lookback_years: int,
                             skip_recent_hours: int) -> dict:
    from engine.agents.papers_curator.forward_citation_crawler import (
        crawl_forward_citations,
    )
    return crawl_forward_citations(
        max_per_seed      = max_per_seed,
        lookback_years    = lookback_years,
        skip_recent_hours = skip_recent_hours,
    )


# ────────────────────────────────────────────────────────────────────
# Roll-up helper — different crawlers report different field names
# ────────────────────────────────────────────────────────────────────
def _extract_counts(r: dict) -> tuple[int, int]:
    """(n_fetched, n_new) from a source result dict. Each source's
    runner uses slightly different field names; this normalizes.

    Returns (0, 0) on an empty result so missing sources don't
    poison the roll-up totals."""
    fetched_keys = ("n_fetched",          # nber, ssrn, arxiv (wrapped)
                     "n_papers_fetched",   # watchlist
                     "n_citations_fetched")   # forward_citations
    new_keys     = ("n_new",
                     "n_papers_new",
                     "n_citations_new")
    n_fetched = 0
    n_new     = 0
    for k in fetched_keys:
        if k in r:
            n_fetched = int(r.get(k) or 0)
            break
    for k in new_keys:
        if k in r:
            n_new = int(r.get(k) or 0)
            break
    return n_fetched, n_new


def _persist(result: SubstrateRunResult,
              *, dir_path: Optional[Path] = None) -> Path:
    """Write the result to disk. Two writes per call:

    1. weekly_substrate/<run_date>.json — latest snapshot for that day
       (overwritten on same-day re-runs; what UI / smoke reads)
    2. weekly_substrate/_history.jsonl — append-only audit log, one
       line per run forever (so re-running on the same day doesn't
       erase earlier audit data — caught 2026-06-07 surface walk)

    Returns the path to the snapshot file (the .json). Audit log is
    a side effect; failures on it log + continue rather than abort
    the run.
    """
    d = dir_path or SUBSTRATE_DIR
    d.mkdir(parents=True, exist_ok=True)

    payload = result.to_dict()

    # Latest snapshot (overwrite OK — UI wants the freshest)
    out = d / f"{result.run_date}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")

    # Append to history log (one JSON object per line)
    history = d / "_history.jsonl"
    try:
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("substrate: history append failed: %s "
                        "(snapshot still written to %s)",
                        exc, out.name)
    return out


# ────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────
def run_weekly_substrate(
    *,
    dry_run:                    bool = False,
    enabled_sources:            tuple[str, ...] = ALL_SOURCES,
    arxiv_max:                  int = 50,
    ssrn_lookback_days:         int = 7,
    ssrn_max_results:           int = 100,
    watchlist_papers_per_author:int = 10,
    watchlist_lookback_years:   int = 2,
    watchlist_skip_recent_hours:int = 24,
    forward_max_per_seed:       int = 20,
    forward_lookback_years:     int = 3,
    forward_skip_recent_hours:  int = 24,
    persist_dir:                Optional[Path] = None,
) -> SubstrateRunResult:
    """Run all enabled crawlers in sequence, collect results, persist
    + return.

    Per-source exceptions are isolated. Source-specific configuration
    is exposed via kwargs (sensible defaults match each crawler's
    weekly-cadence-friendly values).

    enabled_sources lets the principal disable a broken source
    without code change (e.g. 'NBER down today, run the rest').
    Unknown source names are logged + skipped.
    """
    run_ts   = _utc_iso()
    run_date = _today_str()
    enabled  = tuple(s.lower() for s in enabled_sources)
    errors: list[str] = []

    arxiv_result:    dict = {}
    nber_result:     dict = {}
    ssrn_result:     dict = {}
    watchlist_result:dict = {}
    forward_result:  dict = {}

    # arxiv
    if "arxiv" in enabled:
        if dry_run:
            arxiv_result = {"source": "arxiv", "n_fetched": 0, "n_new": 0,
                             "errors": [], "dry_run": True}
        else:
            try:
                arxiv_result = _run_arxiv(max_results=arxiv_max)
            except Exception as exc:
                logger.exception("substrate: arxiv raised at top level")
                arxiv_result = {"source": "arxiv", "errors": [str(exc)],
                                 "n_fetched": 0, "n_new": 0}
            errors.extend(f"arxiv: {e}" for e in
                           arxiv_result.get("errors", []))

    # nber
    if "nber" in enabled:
        if dry_run:
            nber_result = {"source": "nber", "n_fetched": 0, "n_new": 0,
                            "errors": [], "dry_run": True}
        else:
            try:
                nber_result = _run_nber()
            except Exception as exc:
                logger.exception("substrate: nber raised at top level")
                nber_result = {"source": "nber", "errors": [str(exc)],
                                "n_fetched": 0, "n_new": 0}
            errors.extend(f"nber: {e}" for e in
                           nber_result.get("errors", []))

    # ssrn
    if "ssrn" in enabled:
        if dry_run:
            ssrn_result = {"source": "ssrn", "n_fetched": 0, "n_new": 0,
                            "errors": [], "dry_run": True}
        else:
            try:
                ssrn_result = _run_ssrn(
                    lookback_days = ssrn_lookback_days,
                    max_results   = ssrn_max_results,
                )
            except Exception as exc:
                logger.exception("substrate: ssrn raised at top level")
                ssrn_result = {"source": "ssrn", "errors": [str(exc)],
                                "n_fetched": 0, "n_new": 0}
            errors.extend(f"ssrn: {e}" for e in
                           ssrn_result.get("errors", []))

    # watchlist
    if "watchlist" in enabled:
        if dry_run:
            watchlist_result = {"source": "watchlist",
                                 "n_papers_fetched": 0,
                                 "n_papers_new": 0,
                                 "errors": [], "dry_run": True}
        else:
            try:
                watchlist_result = _run_watchlist(
                    papers_per_author = watchlist_papers_per_author,
                    lookback_years    = watchlist_lookback_years,
                    skip_recent_hours = watchlist_skip_recent_hours,
                )
            except Exception as exc:
                logger.exception("substrate: watchlist raised at top "
                                  "level")
                watchlist_result = {"source": "watchlist",
                                     "errors": [str(exc)],
                                     "n_papers_fetched": 0,
                                     "n_papers_new": 0}
            errors.extend(f"watchlist: {e}" for e in
                           watchlist_result.get("errors", []))

    # forward_citations
    if "forward_citations" in enabled:
        if dry_run:
            forward_result = {"source": "forward_citations",
                                "n_citations_fetched": 0,
                                "n_citations_new": 0,
                                "errors": [], "dry_run": True}
        else:
            try:
                forward_result = _run_forward_citations(
                    max_per_seed      = forward_max_per_seed,
                    lookback_years    = forward_lookback_years,
                    skip_recent_hours = forward_skip_recent_hours,
                )
            except Exception as exc:
                logger.exception("substrate: forward_citations raised "
                                  "at top level")
                forward_result = {"source": "forward_citations",
                                    "errors": [str(exc)],
                                    "n_citations_fetched": 0,
                                    "n_citations_new": 0}
            errors.extend(f"forward_citations: {e}" for e in
                           forward_result.get("errors", []))

    # Warn on unknown source names
    unknown = set(enabled) - set(ALL_SOURCES)
    for u in sorted(unknown):
        msg = f"unknown source: {u!r}"
        errors.append(msg)
        logger.warning("substrate: %s", msg)

    # Roll-up totals
    total_fetched = 0
    total_new     = 0
    for r in (arxiv_result, nber_result, ssrn_result,
                watchlist_result, forward_result):
        if not r:
            continue
        f, n = _extract_counts(r)
        total_fetched += f
        total_new     += n

    result = SubstrateRunResult(
        run_ts                  = run_ts,
        run_date                = run_date,
        dry_run                 = dry_run,
        enabled_sources         = enabled,
        arxiv_result            = arxiv_result,
        nber_result             = nber_result,
        ssrn_result             = ssrn_result,
        watchlist_result        = watchlist_result,
        forward_citation_result = forward_result,
        total_fetched           = total_fetched,
        total_new               = total_new,
        errors                  = errors,
    )

    # Persist unless dry-run (no point polluting the audit trail with
    # preview runs)
    if not dry_run:
        try:
            _persist(result, dir_path=persist_dir)
        except Exception as exc:
            logger.exception("substrate: persist raised (returning "
                              "result without disk write)")
            # Append to errors but don't fail the function — result
            # is still useful in-memory
            errors.append(f"persist: {exc}")

    return result
