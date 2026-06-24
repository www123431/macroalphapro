"""engine/research/library_integrity.py — deterministic crossref-based
verification of the Mechanism Library paper master index.

Catches the failure mode that produced 2 wrong-DOI incidents in the prior
session: correct title + correct authors + correct year + correct journal,
but INVENTED DOI string. LLMs share this failure mode at runtime; the only
robust defense is to treat crossref.org as ground truth and never trust an
in-memory DOI.

Doctrine (STANDING — see feedback memory):
  Never trust an LLM-recalled DOI (including my own, the seed author).
  Always crossref-verify before any paper-cite becomes load-bearing.

Used by:
- CLI (this file): `python -m engine.research.library_integrity`
- Pre-commit hook: stops a commit if any master-index entry with
  verified=true fails the crossref check.
- (future) H4 hygiene tool: same check at generator runtime.
- (future) library_writer.py: pre-flight check before YAML write.

Verification semantics:
- title:   token-set Jaccard ≥ 0.7 against crossref title
- authors: family-name set intersection ≥ min(my_count, crossref_count) - 1
- year:    exact match
- journal: token-set Jaccard ≥ 0.5 (lower bar — journals vary in formal naming)
- DOI:     must resolve via crossref API

A 30-day local cache lives at `data/research/_crossref_cache.json` to keep
the pre-commit hook fast.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
MASTER_INDEX = LIBRARY_DIR / "_canonical_papers_tier1_2.yaml"
CACHE_PATH = REPO_ROOT / "data" / "research" / "_crossref_cache.json"
CACHE_TTL_DAYS = 30
CROSSREF_TIMEOUT = 10.0


# ── crossref cache + fetch ──────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _extract_year(msg: dict) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = msg.get(key, {}).get("date-parts", [[]])
        if parts and parts[0]:
            return parts[0][0]
    return None


def _fetch_crossref(doi: str) -> dict | None:
    """Pull crossref metadata for DOI. Returns simplified dict or None on
    network/parse failure. Uses 30-day local cache."""
    cache = _load_cache()
    cache_key = doi.lower()
    if cache_key in cache:
        entry = cache[cache_key]
        age_days = (datetime.datetime.utcnow().timestamp() - entry["fetched_at"]) / 86400
        if age_days < CACHE_TTL_DAYS:
            return entry["data"]

    url = f"https://api.crossref.org/works/{doi}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "library-integrity-check/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=CROSSREF_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        logger.warning("crossref HTTP %d for %s", e.code, doi)
        return None
    except Exception as e:
        logger.warning("crossref fetch failed for %s: %s", doi, e)
        return None

    msg = payload.get("message", {})
    simplified = {
        "title":   (msg.get("title") or [None])[0],
        "authors": [a.get("family") for a in msg.get("author", []) if a.get("family")],
        "year":    _extract_year(msg),
        "journal": (msg.get("container-title") or [None])[0],
        "volume":  msg.get("volume"),
        "issue":   msg.get("issue"),
        "doi":     (msg.get("DOI") or "").lower(),
    }
    cache[cache_key] = {
        "fetched_at": datetime.datetime.utcnow().timestamp(),
        "data": simplified,
    }
    _save_cache(cache)
    time.sleep(0.3)  # be polite to crossref
    return simplified


# ── token normalization for fuzzy match ─────────────────────────────────

_STOPWORDS = {"the", "a", "an", "and", "of", "for", "in", "on", "to"}


def _normalize_tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return {t for t in s.split() if t and t not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _parse_expected_authors(author_str: str | None) -> set[str]:
    """Handle "Bernard, Thomas" → {"bernard", "thomas"} and
    "Koijen, Moskowitz, Pedersen, Vrugt" → {all four}."""
    if not author_str:
        return set()
    parts = [p.strip().lower() for p in author_str.split(",") if p.strip()]
    return {p.split()[-1] for p in parts}  # last-token = family name


# ── core verify ─────────────────────────────────────────────────────────

def verify_paper(paper_id: str, expected: dict) -> dict:
    """Verify one master-index entry against crossref.

    Returns dict with keys: paper_id, doi, doi_resolves, title_match,
    authors_match, year_match, journal_match, overall_pass, errors[]"""
    doi = expected.get("doi")
    result = {
        "paper_id":      paper_id,
        "doi":           doi,
        "doi_resolves":  False,
        "title_match":   False,
        "authors_match": False,
        "year_match":    False,
        "journal_match": False,
        "overall_pass":  False,
        "errors":        [],
    }
    if not doi:
        result["errors"].append("no DOI in master index entry")
        return result

    crossref = _fetch_crossref(doi)
    if not crossref:
        result["errors"].append("crossref fetch failed or DOI does not resolve")
        return result
    result["doi_resolves"] = True

    # Title
    expected_t = _normalize_tokens(expected.get("title"))
    actual_t = _normalize_tokens(crossref.get("title"))
    title_j = _jaccard(expected_t, actual_t)
    result["title_match"] = title_j >= 0.7
    if not result["title_match"]:
        result["errors"].append(
            f"title mismatch (jaccard={title_j:.2f}): "
            f"expected={expected.get('title')!r} "
            f"actual={crossref.get('title')!r}"
        )

    # Authors
    expected_lastnames = _parse_expected_authors(expected.get("author"))
    actual_lastnames = {a.lower() for a in crossref.get("authors") or []}
    intersection = expected_lastnames & actual_lastnames
    min_count = min(len(expected_lastnames), len(actual_lastnames))
    min_required = max(min_count - 1, 1) if min_count else 0
    result["authors_match"] = len(intersection) >= min_required and min_count > 0
    if not result["authors_match"]:
        result["errors"].append(
            f"author mismatch: expected={sorted(expected_lastnames)} "
            f"actual={sorted(actual_lastnames)}"
        )

    # Year
    result["year_match"] = expected.get("year") == crossref.get("year")
    if not result["year_match"]:
        result["errors"].append(
            f"year mismatch: expected={expected.get('year')} "
            f"actual={crossref.get('year')}"
        )

    # Journal
    expected_j = _normalize_tokens(expected.get("journal"))
    actual_j = _normalize_tokens(crossref.get("journal"))
    journal_j = _jaccard(expected_j, actual_j)
    result["journal_match"] = journal_j >= 0.5
    if not result["journal_match"]:
        result["errors"].append(
            f"journal mismatch (jaccard={journal_j:.2f}): "
            f"expected={expected.get('journal')!r} "
            f"actual={crossref.get('journal')!r}"
        )

    result["overall_pass"] = (
        result["doi_resolves"]
        and result["title_match"]
        and result["authors_match"]
        and result["year_match"]
        and result["journal_match"]
    )
    return result


def verify_master_index(strict: bool = False) -> dict:
    """Verify all entries in `_canonical_papers_tier1_2.yaml` against crossref.

    strict=True  → also verify entries with verified=false (catches the
                    pre-commit-hook case where author tries to flip verified
                    to true without actually verifying)
    strict=False → only verify entries with verified=true (audit mode)

    Returns: {total, passed, failed, skipped, results, all_pass}
    """
    if not MASTER_INDEX.exists():
        return {"error": f"master index not found at {MASTER_INDEX}"}

    master = yaml.safe_load(MASTER_INDEX.read_text(encoding="utf-8"))
    papers = master.get("papers", {})
    results = []

    for paper_id, entry in papers.items():
        if not entry.get("verified") and not strict:
            results.append({
                "paper_id": paper_id, "skipped": True,
                "reason":   "verified=false (run with --strict to enforce)",
            })
            continue
        if not entry.get("doi"):
            if entry.get("ssrn_id"):
                results.append({
                    "paper_id": paper_id, "skipped": True,
                    "reason":   "SSRN-only entry; manual verification (SG4 TBD)",
                })
                continue
            results.append({
                "paper_id":     paper_id,
                "overall_pass": False,
                "errors":       ["no DOI and no SSRN ID"],
            })
            continue
        results.append(verify_paper(paper_id, entry))

    fails = [r for r in results
             if not r.get("skipped") and not r.get("overall_pass")]
    return {
        "total":    len(results),
        "passed":   sum(1 for r in results if r.get("overall_pass")),
        "failed":   len(fails),
        "skipped":  sum(1 for r in results if r.get("skipped")),
        "results":  results,
        "all_pass": len(fails) == 0,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Mechanism Library master index against crossref"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Verify even entries with verified=false (pre-commit hook mode)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress info logs; output only the summary line + failures"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s"
    )

    summary = verify_master_index(strict=args.strict)
    if "error" in summary:
        print(f"ERROR: {summary['error']}", file=sys.stderr)
        return 2

    print(
        f"[library_integrity] checked={summary['total']} "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"skipped={summary['skipped']} strict={args.strict}"
    )

    if summary["failed"] > 0:
        print("\nFAILURES:")
        for r in summary["results"]:
            if r.get("skipped") or r.get("overall_pass"):
                continue
            print(f"\n  {r['paper_id']} (DOI {r.get('doi')}):")
            for err in r.get("errors", []):
                print(f"    - {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
