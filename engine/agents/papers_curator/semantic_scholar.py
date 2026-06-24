"""engine.agents.papers_curator.semantic_scholar — Tier-2 Stage A piece 1.

Thin client over Semantic Scholar's free Graph API:
  https://api.semanticscholar.org/graph/v1/

Why this exists:
  The institutional research lib's core tool. Without it, A reasons
  over a single-source (arxiv RSS) corpus and has no quality signal
  beyond keyword match. Semantic Scholar provides:

    - Author h-index, paper count, affiliations
    - Paper venue + year + citation count (the institutional quality
      signal)
    - Forward citations: "papers that CITED this paper" — the bridge
      to current conversation about any baseline paper
    - Reference traversal: "papers this paper CITED" — for completeness
      walks

  Free tier: 100 req/sec hard cap, no daily limit in practice. We use
  conservative ~1 req/sec to be polite.

Per project memory project_anti_rut_doctrine_2026-06-07: this client
is the FOUNDATION for adversarial author watchlist, forward citation
traversal, and multi-dim gap detection. Stage A's other pieces all
depend on metadata this surface returns.

Fail-OPEN: API down / rate-limited / paper not found → return None
(callers degrade gracefully — paper just isn't enriched, not dropped).
"""
from __future__ import annotations

import dataclasses as _dc
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_USER_AGENT = "MacroAlphaPro/0.1 (solo-quant research; +e1521244@u.nus.edu)"

# Rate limiting per Semantic Scholar policy (verified 2026-06-07 from
# the official key-grant email):
#   Unauthenticated (no API key)  ~ 1 req / 3 sec (shared pool, often
#                                    triggers 429 even at 1 req/sec)
#   With free API key             1 req / sec, cumulative across ALL
#                                    endpoints (NOT 100 req/sec — the
#                                    docs page mis-states this; the
#                                    actual grant letter says 1/s).
#
# Get a free key at
#   https://www.semanticscholar.org/product/api#api-key-form
# then set SEMANTIC_SCHOLAR_API_KEY env var (or in
# .streamlit/secrets.toml). The client auto-detects + uses it.
_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
# Try .streamlit/secrets.toml as a fallback (existing convention in
# this project for API keys — see ANTHROPIC_API_KEY / DEEPSEEK_API_KEY).
# Try multiple TOML parsers in priority order:
#   1. tomllib (stdlib, py 3.11+)
#   2. toml (third-party, py 3.10 deps)
#   3. tomli (third-party, py < 3.11 backport)
# If ALL three fail we log a one-line warning so the silent-no-key
# failure mode (which cost ~20 min on 2026-06-07) can't recur silently.
if not _API_KEY:
    from pathlib import Path as _P
    _SECRETS_PATH = (_P(__file__).resolve().parent.parent.parent.parent
                       / ".streamlit" / "secrets.toml")
    if _SECRETS_PATH.is_file():
        _parsed: dict | None = None
        _parser_used = ""
        try:
            import tomllib   # py 3.11+
            with _SECRETS_PATH.open("rb") as _f:
                _parsed = tomllib.load(_f)
                _parser_used = "tomllib"
        except ImportError:
            try:
                import toml as _toml
                with _SECRETS_PATH.open("r", encoding="utf-8") as _f:
                    _parsed = _toml.load(_f)
                    _parser_used = "toml"
            except ImportError:
                try:
                    import tomli as _tomli
                    with _SECRETS_PATH.open("rb") as _f:
                        _parsed = _tomli.load(_f)
                        _parser_used = "tomli"
                except ImportError:
                    logger.warning(
                        "semantic_scholar: secrets.toml found but no TOML "
                        "parser available (tomllib / toml / tomli all "
                        "missing). API key cannot be loaded; running "
                        "unauthenticated. Run: pip install toml"
                    )
        except Exception as _exc:
            logger.warning("semantic_scholar: secrets.toml parse failed "
                            "(%s): %s — running unauthenticated",
                            _parser_used or "unknown", _exc)
            _parsed = None
        if _parsed is not None:
            _API_KEY = str(_parsed.get("SEMANTIC_SCHOLAR_API_KEY", "")
                            or "").strip()

# Throttle: 3.0s unauthenticated (conservative — SS shared pool burns
# fast); 1.2s with API key. Grant is exactly 1 req/sec — we leave 20%
# headroom (1.2s) so clock skew + network jitter don't push us over.
_MIN_INTERVAL_S = 1.2 if _API_KEY else 3.0
_last_request_ts = 0.0


def _polite_sleep() -> None:
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _last_request_ts = time.time()


def _http_get(path: str, *, params: Optional[dict] = None,
               timeout: float = 15.0,
               max_retries: int = 3) -> Optional[dict]:
    """Internal: one GET to the API. Returns parsed JSON dict or None.

    On 429, applies exponential backoff up to max_retries (default 3):
    waits 5s, 10s, 20s before re-attempting. After max_retries → None
    (caller treats as 'API unreachable for now'). 404 / network /
    parse errors return None immediately (no retry — those are
    deterministic failures)."""
    url = f"{_BASE_URL}{path}"
    if params:
        # urlencode handles None values cleanly (drops them)
        clean = {k: v for k, v in params.items() if v is not None}
        url = f"{url}?{urllib.parse.urlencode(clean)}"
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept":     "application/json",
    }
    if _API_KEY:
        # SS auth header — unlocks 100 req/sec
        headers["x-api-key"] = _API_KEY

    for attempt in range(max_retries):
        _polite_sleep()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Rate limited — exponential backoff (5s, 10s, 20s)
                backoff = 5 * (2 ** attempt)
                logger.warning("semantic_scholar: 429 on %s, retry %d/%d "
                                "after %ds backoff",
                                path, attempt + 1, max_retries, backoff)
                if attempt + 1 < max_retries:
                    time.sleep(backoff)
                    continue
                return None
            elif exc.code == 404:
                logger.debug("semantic_scholar: 404 on %s", path)
                return None
            else:
                logger.warning("semantic_scholar: HTTP %d on %s: %s",
                                exc.code, path, exc.reason)
                return None
        except (urllib.error.URLError, json.JSONDecodeError, Exception) as exc:
            logger.warning("semantic_scholar: request failed %s: %s", path, exc)
            return None
    return None


# ────────────────────────────────────────────────────────────────────
# Dataclasses — projections of Semantic Scholar fields we actually use
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class AuthorSummary:
    """Subset of Semantic Scholar author record we use for quality
    signals. `h_index` is the key institutional reputation proxy;
    `paper_count` complements it for productivity context."""
    author_id:    str             # SS author id
    name:         str
    h_index:      Optional[int]
    paper_count:  Optional[int]
    affiliations: tuple[str, ...]


@_dc.dataclass(frozen=True)
class PaperSummary:
    """Subset of SS paper record + the SS-internal id (needed for
    forward citation traversal)."""
    paper_id:       str            # SS paperId (canonical)
    title:          str
    abstract:       str            # may be empty if SS doesn't have it
    year:           Optional[int]
    venue:          str            # journal/conf name; empty if working paper
    citation_count: Optional[int]
    author_ids:     tuple[str, ...]
    author_names:   tuple[str, ...]
    doi:            str            # may be empty
    arxiv_id:       str            # may be empty
    url:            str


# ────────────────────────────────────────────────────────────────────
# Author lookup — by name or by id
# ────────────────────────────────────────────────────────────────────
def search_author_by_name(name: str, *, limit: int = 5
                            ) -> tuple[AuthorSummary, ...]:
    """Free-text author search. Returns top matches; pick by name match
    + h-index. Use this once to populate watchlist with author_ids,
    then `lookup_author_by_id` for subsequent calls (more accurate).

    Returns () on API failure or no matches."""
    name = (name or "").strip()
    if not name:
        return ()
    data = _http_get(
        "/author/search",
        params={
            "query":  name,
            "limit":  limit,
            "fields": "name,hIndex,paperCount,affiliations",
        },
    )
    if not data or "data" not in data:
        return ()
    out: list[AuthorSummary] = []
    for r in (data.get("data") or [])[:limit]:
        out.append(AuthorSummary(
            author_id    = str(r.get("authorId") or ""),
            name         = str(r.get("name") or ""),
            h_index      = r.get("hIndex"),
            paper_count  = r.get("paperCount"),
            affiliations = tuple(
                a.get("name", "") if isinstance(a, dict) else str(a)
                for a in (r.get("affiliations") or [])
            ),
        ))
    return tuple(out)


def lookup_author_by_id(author_id: str) -> Optional[AuthorSummary]:
    """Authoritative author record for a known SS author_id. Used by
    the watchlist to refresh metadata."""
    author_id = (author_id or "").strip()
    if not author_id:
        return None
    data = _http_get(
        f"/author/{urllib.parse.quote(author_id)}",
        params={"fields": "name,hIndex,paperCount,affiliations"},
    )
    if not data:
        return None
    return AuthorSummary(
        author_id    = str(data.get("authorId") or author_id),
        name         = str(data.get("name") or ""),
        h_index      = data.get("hIndex"),
        paper_count  = data.get("paperCount"),
        affiliations = tuple(
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in (data.get("affiliations") or [])
        ),
    )


# ────────────────────────────────────────────────────────────────────
# Author's recent papers
# ────────────────────────────────────────────────────────────────────
def author_papers(author_id: str, *, limit: int = 10,
                    min_year: Optional[int] = None
                    ) -> tuple[PaperSummary, ...]:
    """Return up to `limit` most recent papers for an author. Used by
    the adversarial author watchlist to ingest 'what is this author
    publishing this year that I haven't seen.'

    min_year filter applied client-side (SS API doesn't support it
    directly on this endpoint)."""
    author_id = (author_id or "").strip()
    if not author_id:
        return ()
    data = _http_get(
        f"/author/{urllib.parse.quote(author_id)}/papers",
        params={
            "limit":  min(limit, 100),
            "fields": ("title,abstract,year,venue,citationCount,"
                        "externalIds,authors,url"),
        },
    )
    if not data or "data" not in data:
        return ()
    out: list[PaperSummary] = []
    for r in (data.get("data") or [])[:limit]:
        year = r.get("year")
        if min_year is not None and (year is None or year < min_year):
            continue
        out.append(_to_paper_summary(r))
    return tuple(out)


# ────────────────────────────────────────────────────────────────────
# Forward citation traversal — papers that CITE paper X
# ────────────────────────────────────────────────────────────────────
def forward_citations(paper_id: str, *, limit: int = 25,
                        min_year: Optional[int] = None
                        ) -> tuple[PaperSummary, ...]:
    """Papers that cite the given SS paperId. The institutional
    research lib's core tool: 'what's the conversation about paper X
    since publication?'

    Sorted by citation_count desc on the SS side; we cap at `limit`."""
    paper_id = (paper_id or "").strip()
    if not paper_id:
        return ()
    data = _http_get(
        f"/paper/{urllib.parse.quote(paper_id)}/citations",
        params={
            "limit":  min(limit, 100),
            "fields": ("title,abstract,year,venue,citationCount,"
                        "externalIds,authors,url"),
        },
    )
    if not data or "data" not in data:
        return ()
    out: list[PaperSummary] = []
    for r in (data.get("data") or [])[:limit]:
        # SS wraps each citation as {"citingPaper": {...}}
        cp = r.get("citingPaper") if isinstance(r, dict) else None
        if not cp:
            continue
        year = cp.get("year")
        if min_year is not None and (year is None or year < min_year):
            continue
        out.append(_to_paper_summary(cp))
    return tuple(out)


# ────────────────────────────────────────────────────────────────────
# Paper lookup by external id (DOI / arxiv)
# ────────────────────────────────────────────────────────────────────
def lookup_paper_by_doi(doi: str) -> Optional[PaperSummary]:
    """Look up a paper by DOI. Used to bridge from our papers_registry
    (which has doi) to SS's paper_id (which we need for forward
    citation traversal)."""
    doi = (doi or "").strip()
    if not doi:
        return None
    data = _http_get(
        f"/paper/DOI:{urllib.parse.quote(doi)}",
        params={"fields": ("title,abstract,year,venue,citationCount,"
                             "externalIds,authors,url")},
    )
    if not data:
        return None
    return _to_paper_summary(data)


def search_paper_by_title(query: str, *, limit: int = 5
                             ) -> tuple[PaperSummary, ...]:
    """Free-text search over SS's paper corpus. Used as a last-resort
    fallback when DOI/arxiv lookups fail (some older + non-Elsevier/RFS
    DOIs aren't indexed by SS — see forward_citation_crawler's
    title-search fallback).

    Caller is responsible for verifying the match — this returns the
    top-N relevance-ranked hits as SS provides them. SS relevance is
    approximate (BM25-like) so fuzzy matches WILL appear. Don't accept
    a result without an explicit substring + author/year check
    downstream.
    """
    query = (query or "").strip()
    if not query:
        return ()
    data = _http_get(
        "/paper/search",
        params={
            "query":  query,
            "limit":  min(limit, 100),
            "fields": ("title,abstract,year,venue,citationCount,"
                        "externalIds,authors,url"),
        },
    )
    if not data or "data" not in data:
        return ()
    return tuple(_to_paper_summary(r)
                  for r in (data.get("data") or [])[:limit])


def lookup_paper_by_arxiv(arxiv_id: str) -> Optional[PaperSummary]:
    """Look up by arxiv id (e.g. '2606.12345'). Useful for our existing
    arxiv-source cache to attach SS metadata."""
    arxiv_id = (arxiv_id or "").strip()
    if not arxiv_id:
        return None
    data = _http_get(
        f"/paper/arXiv:{urllib.parse.quote(arxiv_id)}",
        params={"fields": ("title,abstract,year,venue,citationCount,"
                             "externalIds,authors,url")},
    )
    if not data:
        return None
    return _to_paper_summary(data)


# ────────────────────────────────────────────────────────────────────
# Internals
# ────────────────────────────────────────────────────────────────────
def _to_paper_summary(r: dict) -> PaperSummary:
    """Normalize an SS paper record into our PaperSummary projection.
    Defensive against missing fields — SS sometimes returns None for
    abstract / year / venue on older or low-quality records."""
    ext = r.get("externalIds") or {}
    authors_raw = r.get("authors") or []
    author_ids:   list[str] = []
    author_names: list[str] = []
    for a in authors_raw:
        if isinstance(a, dict):
            if a.get("authorId"):
                author_ids.append(str(a["authorId"]))
            if a.get("name"):
                author_names.append(str(a["name"]))
    return PaperSummary(
        paper_id       = str(r.get("paperId") or ""),
        title          = str(r.get("title") or ""),
        abstract       = str(r.get("abstract") or ""),
        year           = r.get("year"),
        venue          = str(r.get("venue") or ""),
        citation_count = r.get("citationCount"),
        author_ids     = tuple(author_ids),
        author_names   = tuple(author_names),
        doi            = str(ext.get("DOI") or ""),
        arxiv_id       = str(ext.get("ArXiv") or ""),
        url            = str(r.get("url") or ""),
    )
