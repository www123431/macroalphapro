"""engine.agents.papers_curator.forward_citation_crawler — Stage A piece 4.

Walks 1 hop forward from each seed paper (papers that cite the seed)
via Semantic Scholar's /paper/{id}/citations endpoint and ingests
descendants into cache.jsonl alongside arxiv RSS + watchlist substrate.

Anti-rut design (per [[project-anti-rut-doctrine-2026-06-07]]):
  Three input streams cross at cache.jsonl:
    - arxiv RSS:  topical recency
    - watchlist:  adversarial-author cognitive diversity
    - forward citations: 'who's working ON the paper our deployed
                          sleeve was built FROM' — surfaces
                          methodology refiners + critiques without
                          requiring topical-keyword overlap

Cost discipline:
  - skip_recent_hours guard so re-running doesn't burn quota for
    seeds we just hit
  - per-seed cap (max_per_seed, default 20) so one widely-cited
    methodology paper doesn't dominate the substrate
  - min_year filter (default = current year - 3) — older citers
    aren't anti-rut signal

Lookup priority per seed: doi > arxiv > paper_id (SS native).

Returns a structured result the chief_of_staff orchestrator can fold
into the weekly memo ('substrate enrichment: 12 seeds → 47 new
candidates this week').
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from engine.agents.papers_curator.crawler import PaperCandidate
from engine.agents.papers_curator.semantic_scholar import (
    forward_citations,
    lookup_paper_by_arxiv,
    lookup_paper_by_doi,
    search_paper_by_title,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEEDS_PATH = _REPO_ROOT / "data" / "papers_curator" / "forward_seeds.yaml"
SEED_STATE_PATH = (_REPO_ROOT / "data" / "papers_curator"
                    / "forward_seeds_state.json")


# ────────────────────────────────────────────────────────────────────
# Seed dataclasses
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class ForwardSeed:
    """One entry in forward_seeds.yaml."""
    slug:        str
    kind:        str            # sleeve_seed / methodology_seed / red_verdict_seed
    doi:         str = ""       # priority 1
    arxiv_id:    str = ""       # priority 2
    paper_id:    str = ""       # priority 3 (SS native id)
    title_hint:  str = ""       # for logging only
    sleeve:      str = ""       # if kind=sleeve_seed
    notes:       str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ForwardSeed":
        return cls(
            slug       = str(d.get("slug")  or ""),
            kind       = str(d.get("kind")  or "sleeve_seed"),
            doi        = str(d.get("doi")   or ""),
            arxiv_id   = str(d.get("arxiv") or d.get("arxiv_id") or ""),
            paper_id   = str(d.get("paper_id") or ""),
            title_hint = str(d.get("title_hint") or ""),
            sleeve     = str(d.get("sleeve") or ""),
            notes      = str(d.get("notes") or ""),
        )


def load_seeds(path: Optional[Path] = None) -> tuple[ForwardSeed, ...]:
    p = path or SEEDS_PATH
    if not p.is_file():
        return ()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rows = data.get("seeds") or []
    out = tuple(ForwardSeed.from_dict(r) for r in rows if isinstance(r, dict))
    # Drop seeds with no resolvable id at all (defensive — yaml typo
    # shouldn't crash the crawler)
    out = tuple(s for s in out if s.doi or s.arxiv_id or s.paper_id)
    return out


# ────────────────────────────────────────────────────────────────────
# Seed state — tracks last_crawled_ts so re-runs don't re-hit SS
# ────────────────────────────────────────────────────────────────────
def _load_seed_state(path: Optional[Path] = None) -> dict[str, str]:
    """{slug: iso_ts_last_crawled}. Missing file → {}."""
    import json
    p = path or SEED_STATE_PATH
    if not p.is_file():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("forward_seeds_state load failed: %s", exc)
        return {}


def _save_seed_state(state: dict[str, str], path: Optional[Path] = None
                       ) -> None:
    import json
    p = path or SEED_STATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _was_crawled_within(slug: str, state: dict[str, str], *,
                          hours: int = 24) -> bool:
    """True if this seed was crawled within the last N hours."""
    ts = state.get(slug, "")
    if not ts:
        return False
    try:
        last = _dt.datetime.fromisoformat(ts.replace("Z", ""))
    except ValueError:
        return False
    return (_dt.datetime.utcnow() - last).total_seconds() < hours * 3600


# ────────────────────────────────────────────────────────────────────
# SS lookup: seed → SS paper_id
# ────────────────────────────────────────────────────────────────────
_SLUG_YEAR_RE = re.compile(r"(\d{4})")


def _parse_slug(slug: str) -> tuple[tuple[str, ...], Optional[int]]:
    """Pull (authors_lastnames, year) out of our canonical slug convention
    `lastname_lastname_..._YYYY_venue`. Returns ((), None) on malformed.

    Examples:
      'koijen_moskowitz_pedersen_vrugt_2018_jfe'
         → (('koijen','moskowitz','pedersen','vrugt'), 2018)
      'harvey_liu_zhu_2016_rfs'
         → (('harvey','liu','zhu'), 2016)
    """
    parts = (slug or "").split("_")
    year_idx = None
    for i, p in enumerate(parts):
        if _SLUG_YEAR_RE.fullmatch(p):
            year_idx = i
            break
    if year_idx is None or year_idx == 0:
        return ((), None)
    return (tuple(parts[:year_idx]), int(parts[year_idx]))


def _strict_match(result, *, slug_authors: tuple[str, ...],
                   slug_year: Optional[int],
                   title_hint: str) -> bool:
    """Strict accept rule for SS title-search fallback.

    Three conditions ALL must hold:
      1. title_hint is a case-insensitive substring of result.title
      2. ≥2 slug-authors appear (case-insensitive) in result.author_names
         (or all of them if slug has fewer than 2 authors)
      3. result.year within ±1 of slug_year (when both known)

    The 2-author floor catches single-author-coincidence (a different
    'Carry' paper by an unrelated author) while still tolerating
    citation-style name variations (initials vs full names)."""
    title = (result.title or "").lower()
    if title_hint and title_hint.lower() not in title:
        return False

    if slug_authors:
        result_names = " ".join(result.author_names or ()).lower()
        n_hits = sum(1 for a in slug_authors
                      if a and a.lower() in result_names)
        floor = min(2, len(slug_authors))
        if n_hits < floor:
            return False

    if slug_year and result.year:
        if abs(int(result.year) - int(slug_year)) > 1:
            return False

    return True


def _build_title_search_query(seed: ForwardSeed) -> str:
    """Compose a free-text SS query from slug authors + title hint.
    More descriptive query → better SS BM25 relevance hit."""
    authors, year = _parse_slug(seed.slug)
    parts = list(authors)
    if year:
        parts.append(str(year))
    if seed.title_hint:
        parts.append(seed.title_hint)
    return " ".join(p for p in parts if p)


def _resolve_seed_paper_id(seed: ForwardSeed) -> Optional[str]:
    """Return SS paper_id for this seed.

    Priority: paper_id (literal) > doi > arxiv > title-search fallback.
    Title-search ONLY runs as last resort (older DOIs sometimes absent
    from SS index — McLean-Pontiff DOI was OK but HLZ 2016, KMPV 2018,
    Blitz 2011 weren't). Strict match prevents fuzzy contamination —
    if no result passes _strict_match, the seed stays unresolved.
    """
    if seed.paper_id:
        return seed.paper_id
    if seed.doi:
        ps = lookup_paper_by_doi(seed.doi)
        if ps and ps.paper_id:
            return ps.paper_id
    if seed.arxiv_id:
        ps = lookup_paper_by_arxiv(seed.arxiv_id)
        if ps and ps.paper_id:
            return ps.paper_id

    # Title-search fallback — needs title_hint to run
    if not seed.title_hint:
        return None
    query = _build_title_search_query(seed)
    results = search_paper_by_title(query, limit=5)
    slug_authors, slug_year = _parse_slug(seed.slug)
    for r in results:
        if _strict_match(r, slug_authors=slug_authors,
                          slug_year=slug_year,
                          title_hint=seed.title_hint):
            logger.info("forward_citations: title-search resolved %s "
                          "→ %s (%s, %s)",
                          seed.slug, r.paper_id, r.year,
                          (r.title or "")[:60])
            return r.paper_id
    return None


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_paper_candidate(ss_paper, *, fetched_ts: str,
                          seed_slug: str) -> PaperCandidate:
    """Adapt SS PaperSummary → cache.jsonl PaperCandidate.

    Tag with categories=('forward_citation', f'seed:{seed_slug}') so
    downstream filter/summary can attribute the candidate's source
    (different from watchlist's venue-as-category convention)."""
    categories = ("forward_citation", f"seed:{seed_slug}")
    if ss_paper.venue:
        categories = categories + (ss_paper.venue,)
    return PaperCandidate(
        source       = "semantic_scholar",
        source_id    = ss_paper.paper_id,
        title        = ss_paper.title,
        authors      = ss_paper.author_names,
        abstract     = ss_paper.abstract,
        abs_url      = ss_paper.url,
        pdf_url      = "",
        published_ts = (f"{ss_paper.year}-01-01T00:00:00Z"
                          if ss_paper.year else ""),
        categories   = categories,
        fetched_ts   = fetched_ts,
    )


# ────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────
def crawl_forward_citations(
    *,
    max_per_seed:      int = 20,
    lookback_years:    int = 3,
    skip_recent_hours: int = 24,
    seeds_path:        Optional[Path] = None,
    state_path:        Optional[Path] = None,
) -> dict:
    """Run one forward-citation crawl over all seeds in forward_seeds.yaml.

    Returns:
      {
        run_ts:                  iso
        n_seeds_total:           int
        n_seeds_crawled:         int
        n_seeds_skipped:         int   # recently crawled OR unresolvable
        n_citations_fetched:     int   # before dedup
        n_citations_new:         int   # after store.save_new_candidates dedup
        unresolved_seeds:        list[str]  # couldn't resolve doi/arxiv → SS
        errors:                  list[str]
      }
    """
    from engine.agents.papers_curator.store import save_new_candidates

    result = {
        "run_ts":              _utc_iso(),
        "n_seeds_total":       0,
        "n_seeds_crawled":     0,
        "n_seeds_skipped":     0,
        "n_citations_fetched": 0,
        "n_citations_new":     0,
        "unresolved_seeds":    [],
        "errors":              [],
    }

    seeds = load_seeds(path=seeds_path)
    result["n_seeds_total"] = len(seeds)
    if not seeds:
        return result

    state = _load_seed_state(path=state_path)
    min_year = _dt.datetime.utcnow().year - lookback_years
    candidates: list[PaperCandidate] = []
    fetched_ts = _utc_iso()

    for seed in seeds:
        # 1. Skip if recently crawled
        if _was_crawled_within(seed.slug, state, hours=skip_recent_hours):
            result["n_seeds_skipped"] += 1
            continue

        # 2. Resolve SS paper_id
        try:
            sid = _resolve_seed_paper_id(seed)
        except Exception as exc:
            logger.warning("forward_citations: lookup failed for %s: %s",
                            seed.slug, exc)
            sid = None
        if not sid:
            result["unresolved_seeds"].append(seed.slug)
            result["n_seeds_skipped"] += 1
            continue

        # 3. Fetch forward citations
        try:
            citers = forward_citations(
                sid, limit=max_per_seed, min_year=min_year,
            )
        except Exception as exc:
            logger.exception("forward_citations: SS call failed for %s",
                              seed.slug)
            result["errors"].append(f"seed:{seed.slug}: {exc}")
            continue

        result["n_seeds_crawled"]      += 1
        result["n_citations_fetched"]  += len(citers)

        # 4. Adapt → PaperCandidate
        for sp in citers:
            if not sp.paper_id:
                continue
            candidates.append(_to_paper_candidate(
                sp, fetched_ts=fetched_ts, seed_slug=seed.slug,
            ))

        # 5. Persist state so next run prioritizes others
        state[seed.slug] = fetched_ts

    # 6. Save updated state (only if anything changed)
    if result["n_seeds_crawled"] > 0:
        try:
            _save_seed_state(state, path=state_path)
        except Exception as exc:
            logger.warning("forward_seeds_state save failed: %s", exc)

    # 7. Bulk write via existing dedup
    if candidates:
        try:
            n_new = save_new_candidates(candidates)
            result["n_citations_new"] = n_new
        except Exception as exc:
            logger.exception("forward_citations: save_new_candidates failed")
            result["errors"].append(f"persist: {exc}")

    # 8. Outage detection — if every seed failed to resolve, that's
    # an upstream SS outage / auth failure / index drift, not a
    # 'seeds list bad' situation. Silent unresolveds would otherwise
    # mask this from chief_of_staff (caught 2026-06-07 failure-surface
    # walk).
    n_total      = result["n_seeds_total"]
    n_crawled    = result["n_seeds_crawled"]
    n_unresolved = len(result["unresolved_seeds"])
    if n_total > 0 and n_crawled == 0 and n_unresolved == n_total:
        result["errors"].append(
            f"outage_suspected: 0/{n_total} seeds resolved; "
            f"all {n_unresolved} unresolved via SS — check API key "
            "or network"
        )

    return result
