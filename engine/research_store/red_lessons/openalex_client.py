"""engine.research_store.red_lessons.openalex_client — OpenAlex paper lookup + cache.

Wraps the OpenAlex REST API (https://api.openalex.org) for paper metadata
acquisition. Used by P2 paper-anchor backfill.

Why OpenAlex over Crossref / Semantic Scholar:
  - Free, 100k requests/day, no API key required
  - Full bibliographic metadata + DOI + author list + abstract + citation
    counts in a single endpoint
  - Search endpoint returns ranked results so we can pick best match
    without fuzzy matching ourselves

Cache: `data/research_store/openalex_cache.json` — keyed by query string.
Re-runs of the lookup script hit the cache; only NEW queries cost network.

Doctrine:
  - Cache is content-addressable by raw query string + entity type. No
    expiry — paper metadata is effectively immutable.
  - On API failure (network, rate-limit, malformed response), return None
    and log; never raise. Caller decides skip-vs-retry.
  - We store the FULL OpenAlex response in cache (not just our extracted
    fields) so future schema changes don't require re-fetching.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_PATH = _REPO_ROOT / "data" / "research_store" / "openalex_cache.json"
OPENALEX_BASE = "https://api.openalex.org"
USER_AGENT = "macroalphapro/0.1 (research; mailto:${USER_EMAIL})"


# ─────────────────────── cache layer ──────────────────────────────────


def _load_cache() -> dict[str, Any]:
    if not CACHE_PATH.is_file():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("openalex cache load failed (%s); starting fresh", e)
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                          encoding="utf-8")


# ─────────────────────── anchor-paper string parser ───────────────────


@dataclass(frozen=True)
class ParsedAnchor:
    """Parsed components of a MECHANISM_FAMILY_DOCS[...]['anchor_paper'] string.

    Example input strings:
      "Bernard & Thomas 1989, 'Post-Earnings-Announcement Drift', JAR"
      "Da-Engelberg-Gao 2011, 'In Search of Attention', JF"
      "Frazzini & Pedersen 2014, 'Betting Against Beta', JFE"
      "Koijen-Moskowitz-Pedersen-Vrugt 2018, 'Carry', JFE"
    """
    authors:   tuple[str, ...]
    year:      int | None
    title:     str
    venue:     str
    raw:       str


def parse_anchor_string(s: str) -> ParsedAnchor | None:
    """Parse an anchor-paper string into structured components.

    Returns None if the string doesn't match the expected shape.
    """
    if not s or s.strip() == "(none — must be filled in lesson-specific context)":
        return None

    # year
    m_year = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    year = int(m_year.group(1)) if m_year else None

    # title: single-quoted segment
    m_title = re.search(r"'([^']+)'", s)
    title = (m_title.group(1) if m_title else "").strip()

    # venue: last comma-separated chunk (after title)
    venue = ""
    if m_title:
        tail = s[m_title.end():].strip()
        # strip leading comma + whitespace
        venue = re.sub(r"^[,\s]+", "", tail).strip()

    # authors: prefix before the year
    if m_year:
        head = s[: m_year.start()].strip()
        # Split on " & ", " and ", " - ", " — " (em dash also), commas
        # but be careful — author surnames can have hyphens. Use a soft
        # split: try " & " and " and " first (definite separators), then
        # fall back to splitting on "-" or commas.
        if " & " in head:
            authors = tuple(a.strip() for a in head.split(" & ") if a.strip())
        elif " and " in head:
            authors = tuple(a.strip() for a in head.split(" and ") if a.strip())
        elif "-" in head and head.count("-") <= 5:
            # multi-author hyphenated paper (KMPV style)
            authors = tuple(a.strip() for a in head.split("-") if a.strip())
        elif "," in head:
            authors = tuple(a.strip() for a in head.split(",") if a.strip())
        else:
            authors = (head,)
    else:
        authors = ()

    return ParsedAnchor(
        authors=authors,
        year=year,
        title=title,
        venue=venue,
        raw=s,
    )


# ─────────────────────── OpenAlex search ──────────────────────────────


def _http_get_json(url: str, timeout: int = 15) -> dict[str, Any] | None:
    """Issue a GET request, return parsed JSON. None on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.warning("openalex GET failed for %s: %s", url[:120], e)
        return None


def search_works(query: str,
                 year: int | None = None,
                 cache: dict[str, Any] | None = None,
                 polite_sleep_s: float = 0.15) -> dict[str, Any] | None:
    """Search OpenAlex works endpoint. Returns the top result dict, or None.

    Args:
        query: free-text search string (title + authors)
        year: optional publication year filter (±1 year tolerance)
        cache: shared cache dict (mutated)
        polite_sleep_s: polite delay between live API calls
    """
    cache = cache if cache is not None else {}
    cache_key = f"search::{query}::{year}"
    if cache_key in cache:
        return cache[cache_key]

    # Build query
    params = {
        "search":   query,
        "per_page": "5",
    }
    if year is not None:
        # ±1 year tolerance — publication can land in the JF the year after
        # working-paper draft
        params["filter"] = f"publication_year:{year-1}|{year}|{year+1}"

    qs = urllib.parse.urlencode(params)
    url = f"{OPENALEX_BASE}/works?{qs}"

    time.sleep(polite_sleep_s)
    result = _http_get_json(url)
    if result is None:
        cache[cache_key] = None
        return None

    # Pick top-1 (OpenAlex relevance-sorts by default)
    works = result.get("results") or []
    top = works[0] if works else None
    cache[cache_key] = top
    return top


# ─────────────────────── result → PaperRef field extraction ───────────


def extract_doi(work: dict[str, Any]) -> str:
    doi = work.get("doi") or ""
    # OpenAlex returns DOIs as full URLs; strip prefix
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi


def extract_authors(work: dict[str, Any]) -> tuple[str, ...]:
    out = []
    for a in (work.get("authorships") or []):
        author = a.get("author") or {}
        name = author.get("display_name") or ""
        # last name only (per our anchor-string convention)
        if name:
            parts = name.strip().split()
            out.append(parts[-1] if parts else name)
    return tuple(out)


def extract_year(work: dict[str, Any]) -> int | None:
    y = work.get("publication_year")
    return int(y) if y else None


def extract_title(work: dict[str, Any]) -> str:
    return (work.get("title") or work.get("display_name") or "").strip()


def extract_venue(work: dict[str, Any]) -> str:
    # Modern OpenAlex: primary_location.source.display_name
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return (src.get("display_name") or "").strip()


def extract_abstract(work: dict[str, Any]) -> str:
    """OpenAlex returns abstract as inverted_index (word → positions).
    Reconstruct flat text."""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    # Collect (position, word) pairs and sort
    positioned: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for p in positions:
            positioned.append((p, word))
    positioned.sort()
    return " ".join(w for _, w in positioned)


# ─────────────────────── public lookup function ───────────────────────


def _result_authors_match_expected(work: dict[str, Any],
                                   expected_authors: tuple[str, ...]) -> bool:
    """Validate: at least 1 expected author surname is present in result authors.

    Defense against OpenAlex returning a wrong-domain paper for a generic
    title (e.g. searching 'Carry' returns a medical paper about plasmid
    'carrying'). If authors don't overlap, the match is wrong.
    """
    if not expected_authors:
        # No author info to validate against; accept result optimistically
        return True
    result_authors_lower = [a.lower() for a in extract_authors(work)]
    expected_lower = [a.lower() for a in expected_authors]
    return any(e in result_authors_lower for e in expected_lower)


def lookup_anchor(anchor_str: str,
                  cache: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Top-level: anchor string → OpenAlex work dict (or None).

    Tries the title as the main query (highest precision); falls back to
    author + title combined. Validates author overlap to reject
    wrong-domain matches on generic titles.
    """
    parsed = parse_anchor_string(anchor_str)
    if parsed is None:
        return None

    cache = cache if cache is not None else _load_cache()

    # Strategy 1: title-driven query (works best for distinctive titles)
    if parsed.title:
        result = search_works(parsed.title, year=parsed.year, cache=cache)
        if result is not None and _result_authors_match_expected(result, parsed.authors):
            return result

    # Strategy 2: first-author + title + year combined
    if parsed.authors and parsed.title:
        q = f"{parsed.authors[0]} {parsed.title}"
        result = search_works(q, year=parsed.year, cache=cache)
        if result is not None and _result_authors_match_expected(result, parsed.authors):
            return result

    # Strategy 3: authors + year (last resort)
    if parsed.authors and parsed.year:
        q = " ".join(parsed.authors[:3])
        result = search_works(q, year=parsed.year, cache=cache)
        if result is not None and _result_authors_match_expected(result, parsed.authors):
            return result

    return None


# ─────────────────────── cache I/O helpers ────────────────────────────


def load_cache() -> dict[str, Any]:
    return _load_cache()


def save_cache(cache: dict[str, Any]) -> None:
    _save_cache(cache)
