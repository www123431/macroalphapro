"""engine.inbox.paper_fetcher — L2.1 academic paper fetcher + pre-filter.

⚠️  DEPRECATED 2026-06-04 (PR-A+B consolidation).

This module is no longer wired into any UI surface or cron. Replaced by
the T7 PAPER → HYPOTHESIS → TEST → VERDICT chain:

  - Discovery (find new ArXiv/NBER candidates):
      engine.research.discovery → data/research/discovery_queue.jsonl

  - Ingestion (PDF → registry + chunks + hypotheses):
      scripts/extract_paper_hypotheses.py +
      planned /research/papers/new UI →
      data/research_store/papers_registry.jsonl + hypotheses.jsonl

  - Reading queue (/lab/literature):
      engine.inbox.composer.source_papers_from_t7 reads the T7 registry.

Why retired: the legacy Haiku-scored flow ran independently of the
PAPER → HYPOTHESIS → TEST → VERDICT chain, producing scored entries
that never became hypothesis-traced or chain-locked. The shelf
assignment + hypothesis density signal in the T7 registry replaces
the numeric Haiku score with a typed, auditable relevance proxy.

Code kept (not deleted) for:
  - History of the keyword pre-filter approach
  - Reference if we ever want to re-introduce an automated ArXiv
    fetcher feeding into the T7 ingestion side (via discovery_queue,
    NOT directly into the registry)

────────────────────────────────────────────────────────────────────
(Original docstring follows for the archaeological record.)

Pulls from three free RSS feeds (ArXiv q-fin, SSRN Finance, NBER WP), runs
the keyword pre-filter (built from active_deployment.yaml sleeves), and
writes survivors to data/research_ops/papers_pre_filtered.jsonl.

L2.2 (paper_scorer.py) reads the survivors and applies Haiku-4.5 relevance
scoring. Then L2.3 (weekly_digest.py) does the cross-paper summary.

Doctrine (key invariant): the keyword map DERIVES from
data/portfolio/active_deployment.yaml — never hardcoded. When a sleeve
is added/replaced, the filter automatically follows. See
project_l2_paper_filter_active_deploy_derived_2026-06-02.

Cost: zero (all RSS sources are free). Cron: daily 06:30 UTC.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_PAPERS_DIR     = _REPO_ROOT / "data" / "research_ops"
_PRE_FILTERED   = _PAPERS_DIR / "papers_pre_filtered.jsonl"


# ── RSS sources (free, no API keys needed) ────────────────────────


# ArXiv q-fin RSS 2.0 — last ~50 new submissions. Verified live 2026-06-02.
ARXIV_QFIN_RSS = "http://export.arxiv.org/rss/q-fin"

# NBER Working Papers — latest new (all programs). Verified live 2026-06-02.
NBER_NEW_RSS = "https://www.nber.org/rss/new.xml"

# SSRN — DEFERRED (Q3 2024+ SSRN deprecated most free RSS endpoints;
# their JEL-classified feeds return 404. Would need their API. For now
# ArXiv + NBER give ~80 papers/week, enough for L2 to be useful.)


# ── Keyword map derivation (THE anti-drift mechanism) ─────────────


def derive_keyword_map() -> dict[str, dict[str, Any]]:
    """Build the paper-filter keyword map from ACTIVE deploy.

    DO NOT hardcode an alternative version. Per Item 7 doctrine, every
    derived structure must trace back to active_deployment.yaml. This
    function reads that file via the registry; when the deploy changes,
    this map automatically follows.

    Returns:
        {
          "sleeve_name": {
            "lane":      "direction" | "methodology",
            "keywords":  [...],
            "anchors":   [...],
          }
        }
    """
    try:
        from engine.portfolio.deployed_registry import load_active
        cfg = load_active()
    except Exception:
        logger.exception("derive_keyword_map: failed to load active deploy")
        return {}

    out: dict[str, dict[str, Any]] = {}
    for s in cfg.sleeves:
        if not s.research_keywords:
            continue
        out[s.name] = {
            "lane":     "direction",
            "keywords": list(s.research_keywords),
            "anchors":  list(s.academic_anchors),
            "role":     s.role,
        }

    # Hardcoded methodology entries that aren't tied to any one sleeve.
    # These are methods we USE across sleeves, not strategies we deploy.
    # They could move to a separate yaml in a future cleanup, but the
    # set is small + slow-moving so inlining is acceptable.
    methodology = {
        "deflated_sharpe": {
            "lane":     "methodology",
            "keywords": [
                "deflated Sharpe",
                "multiple testing finance",
                "Bailey Lopez de Prado",
                "Harvey Liu Zhu",
                "HLZ multiple testing",
                "false discovery finance",
            ],
            "anchors":  ["Bailey-LdP 2014 (JPM)", "HLZ 2016 (RFS)"],
            "role":     "methodology",
        },
        "purged_cv": {
            "lane":     "methodology",
            "keywords": [
                "CPCV",
                "combinatorial purged",
                "walk-forward finance",
                "PBO",
                "probability of backtest overfitting",
            ],
            "anchors":  ["Lopez de Prado 2018"],
            "role":     "methodology",
        },
        "risk_parity_construction": {
            "lane":     "methodology",
            "keywords": [
                "risk parity",
                "Ledoit-Wolf shrinkage",
                "Black-Litterman",
                "Bayesian portfolio",
                "covariance shrinkage",
            ],
            "anchors":  ["Asness-Frazzini 2013", "Ledoit-Wolf 2003 (JEF)"],
            "role":     "methodology",
        },
    }
    out.update(methodology)
    return out


# ── RSS fetchers ───────────────────────────────────────────────────


def _http_get(url: str, timeout: float = 12.0) -> Optional[bytes]:
    """Minimal HTTP GET. Returns body bytes or None on failure."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "MacroAlphaPro-ResearchOps/1.0 (academic paper aggregator)",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as e:
        logger.warning("HTTP GET %s failed: %s", url, e)
        return None
    except Exception:
        logger.exception("HTTP GET %s unexpected error", url)
        return None


def _parse_arxiv(body: bytes) -> list[dict[str, Any]]:
    """ArXiv q-fin RSS 2.0 (since Q3 2024). Returns list of papers.

    Schema:
      <rss version="2.0">
        <channel>
          <item>
            <title>...</title>
            <link>...</link>
            <description>... abstract HTML ...</description>
            <pubDate>...</pubDate>
          </item>
        </channel>
      </rss>
    """
    return _parse_rss_2(body, "arxiv_qfin")


def _parse_rss_2(body: bytes, source: str) -> list[dict[str, Any]]:
    """Generic RSS 2.0 parser for NBER / SSRN-style feeds."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        logger.warning("%s RSS parse error", source)
        return []
    out: list[dict[str, Any]] = []
    # RSS 2.0 puts items under channel
    for item in root.findall(".//item"):
        title    = (item.findtext("title", default="") or "").strip()
        link     = (item.findtext("link",  default="") or "").strip()
        abstract = (item.findtext("description", default="") or "").strip()
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()
        ts       = (item.findtext("pubDate", default="") or "").strip()
        if not title:
            continue
        out.append({
            "title":    title,
            "abstract": abstract,
            "link":     link,
            "ts":       ts or _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source":   source,
        })
    return out


def fetch_all_sources() -> list[dict[str, Any]]:
    """Fetch papers from active RSS sources. Returns a flat list."""
    out: list[dict[str, Any]] = []

    body = _http_get(ARXIV_QFIN_RSS)
    if body:
        out.extend(_parse_arxiv(body))

    body = _http_get(NBER_NEW_RSS)
    if body:
        out.extend(_parse_rss_2(body, "nber_new"))

    return out


# ── Keyword pre-filter ────────────────────────────────────────────


def _match_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    """Return the subset of `keywords` that appear in `text` (case-insensitive)."""
    if not text:
        return []
    tlow = text.lower()
    return [k for k in keywords if k.lower() in tlow]


def pre_filter(papers: list[dict[str, Any]],
               keyword_map: Optional[dict[str, dict[str, Any]]] = None,
               ) -> list[dict[str, Any]]:
    """Apply the keyword pre-filter.

    For each paper, scan title + abstract against EVERY entry in the
    derived keyword_map. Drop papers that match no entry. For survivors,
    attach matched-entry list as `family_match` field.
    """
    if keyword_map is None:
        keyword_map = derive_keyword_map()
    if not keyword_map:
        logger.warning("pre_filter: empty keyword map, nothing will survive")
        return []

    out: list[dict[str, Any]] = []
    for p in papers:
        text = f"{p.get('title','')} {p.get('abstract','')}"
        family_match: list[dict[str, Any]] = []
        for family_id, info in keyword_map.items():
            matches = _match_keywords(text, info["keywords"])
            if matches:
                family_match.append({
                    "family":  family_id,
                    "lane":    info["lane"],
                    "role":    info.get("role"),
                    "matches": matches,
                })
        if not family_match:
            continue
        survivor = dict(p)
        survivor["family_match"] = family_match
        out.append(survivor)
    return out


# ── Persistence ───────────────────────────────────────────────────


def _stable_id(source: str, link: str, title: str) -> str:
    """Stable id across runs so dedupe across daily fetches works."""
    h = hashlib.blake2b((link or title).encode("utf-8"), digest_size=6).hexdigest()
    return f"px_{source}_{h}"


def write_pre_filtered(survivors: list[dict[str, Any]]) -> int:
    """Append survivors to data/research_ops/papers_pre_filtered.jsonl.

    Dedupes against existing rows in the file via stable id. Returns the
    number of NEW rows actually written.
    """
    _PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if _PRE_FILTERED.is_file():
        try:
            with _PRE_FILTERED.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        if "id" in r:
                            existing_ids.add(r["id"])
                    except Exception:
                        continue
        except Exception:
            logger.exception("write_pre_filtered: failed to read existing")

    written = 0
    with _PRE_FILTERED.open("a", encoding="utf-8") as fh:
        for s in survivors:
            pid = _stable_id(s["source"], s.get("link", ""), s.get("title", ""))
            if pid in existing_ids:
                continue
            row = dict(s)
            row["id"] = pid
            row["fetched_ts"] = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            row["scored"] = False     # L2.2 will set this to True after scoring
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def run() -> dict[str, Any]:
    """Full L2.1 pipeline: fetch → pre-filter → persist. Returns stats."""
    papers = fetch_all_sources()
    survivors = pre_filter(papers)
    written = write_pre_filtered(survivors)
    return {
        "fetched":  len(papers),
        "survived": len(survivors),
        "new":      written,
        "ts":       _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="L2.1 paper fetcher + pre-filter")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch + filter but don't write to ledger")
    args = ap.parse_args()
    if args.dry_run:
        ps = fetch_all_sources()
        survivors = pre_filter(ps)
        print(f"fetched={len(ps)} survived={len(survivors)}")
        for s in survivors[:5]:
            fams = ", ".join(f["family"] for f in s["family_match"])
            print(f"  - [{s['source']}] {s['title'][:80]}  ({fams})")
    else:
        stats = run()
        print(json.dumps(stats, indent=2))
