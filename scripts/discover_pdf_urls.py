"""scripts/discover_pdf_urls.py — Stage C piece 1 helper.

For every papers_registry entry with fulltext_status=metadata_only,
query Semantic Scholar's openAccessPdf field via the existing SS
API client. Output a markdown report with auto-discovered PDF URLs
+ candidate LIBRARY_PDF_OVERRIDES entries for human review.

Manual verification (per library_pdf_overrides.py doctrine) still
needed before adding entries:
  1. WebFetch the URL — confirm it's actually a PDF
  2. Verify first-page text matches the expected paper
  3. Check authors line up
  4. Add to LIBRARY_PDF_OVERRIDES with verified_first_page_quote

Usage:
  python scripts/discover_pdf_urls.py             # human-readable
  python scripts/discover_pdf_urls.py --json      # JSON output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.papers_curator import semantic_scholar as ss


def _load_metadata_only_papers() -> list[dict]:
    """All papers in registry whose latest version has
    fulltext_status=metadata_only."""
    p = (_REPO_ROOT / "data" / "research_store"
          / "papers_registry.jsonl")
    by_id: dict[str, dict] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = row.get("paper_id", "")
            if not pid:
                continue
            prior = by_id.get(pid)
            if (prior is None
                or row.get("version", 1) > prior.get("version", 1)):
                by_id[pid] = row
    return [r for r in by_id.values()
            if r.get("fulltext_status") == "metadata_only"]


def _lookup_ss_open_access_pdf(doi: str) -> dict:
    """Single SS API call → openAccessPdf + externalIds. Returns
    {found: bool, url: str, arxiv_id: str, error: str}."""
    result = {"found": False, "url": "", "arxiv_id": "", "error": ""}
    if not doi:
        result["error"] = "no_doi"
        return result
    url = (f"https://api.semanticscholar.org/graph/v1/paper/DOI:"
            f"{urllib.parse.quote(doi)}"
            f"?fields=title,year,openAccessPdf,externalIds")
    h = {"User-Agent": "MacroAlphaPro-PdfDiscover/1.0",
          "Accept": "application/json"}
    if ss._API_KEY:
        h["x-api-key"] = ss._API_KEY
    try:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read().decode("utf-8"))
        oapdf = d.get("openAccessPdf") or {}
        eids = d.get("externalIds") or {}
        if oapdf.get("url"):
            result["found"] = True
            result["url"] = oapdf["url"]
        if eids.get("ArXiv"):
            result["arxiv_id"] = eids["ArXiv"]
    except urllib.error.HTTPError as e:
        result["error"] = f"http_{e.code}"
    except Exception as e:
        result["error"] = f"err:{type(e).__name__}"
    return result


import urllib.parse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true",
                    help="Output JSON")
    args = p.parse_args()

    papers = _load_metadata_only_papers()
    out_rows: list[dict] = []

    for paper in papers:
        time.sleep(1.3)   # respect SS 1 req/sec
        doi = paper.get("doi", "")
        lookup = _lookup_ss_open_access_pdf(doi)
        row = {
            "paper_id":  paper.get("paper_id", "")[:8],
            "title":     (paper.get("title") or "")[:70],
            "authors":   (paper.get("authors") or [])[:3],
            "year":      paper.get("year"),
            "doi":       doi,
            "ss_found":  lookup["found"],
            "pdf_url":   lookup["url"],
            "arxiv_id":  lookup["arxiv_id"],
            "ss_error":  lookup["error"],
        }
        out_rows.append(row)

    n_found  = sum(1 for r in out_rows if r["ss_found"])
    n_arxiv  = sum(1 for r in out_rows if r["arxiv_id"])
    n_failed = sum(1 for r in out_rows if r["ss_error"])

    if args.json:
        sys.stdout.write(json.dumps({
            "n_total":   len(out_rows),
            "n_found":   n_found,
            "n_arxiv":   n_arxiv,
            "n_failed":  n_failed,
            "rows":      out_rows,
        }, indent=2, ensure_ascii=False))
        return 0

    print(f"PDF discovery for {len(out_rows)} metadata_only papers")
    print(f"  SS openAccessPdf found:  {n_found}")
    print(f"  SS provided arxiv id:    {n_arxiv}")
    print(f"  SS lookup failed (404):  {n_failed}")
    print()
    print("=" * 72)
    print("AUTO-DISCOVERED (candidate manual_pdf_url entries):")
    print("=" * 72)
    for r in out_rows:
        if not r["ss_found"]:
            continue
        au = ', '.join(r["authors"])
        print(f"  [{r['year']}] {au}")
        print(f"    title: {r['title']}")
        print(f"    doi:   {r['doi']}")
        print(f"    pdf:   {r['pdf_url']}")
        print()
    print("=" * 72)
    print("NEEDS MANUAL SEARCH (SS had no openAccessPdf):")
    print("=" * 72)
    for r in out_rows:
        if r["ss_found"]:
            continue
        au = ', '.join(r["authors"])
        flag = f" [{r['ss_error']}]" if r["ss_error"] else ""
        print(f"  [{r['year']}] {au}{flag}")
        print(f"    title: {r['title']}")
        print(f"    doi:   {r['doi']}")
        if r["arxiv_id"]:
            print(f"    arxiv: {r['arxiv_id']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
