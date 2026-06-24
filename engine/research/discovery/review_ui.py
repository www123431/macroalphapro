"""engine/research/discovery/review_ui.py — Local web UI for paper
nomination + queue browsing.

Per user 2026-05-30: "CLI 太不自然了,有没有更用户友好的交互方式".
We're NOT using Streamlit (banned), so this uses pure stdlib
http.server + htmx via CDN. Zero new dependencies.

USAGE:
  python -m engine.research.discovery.review_ui          # default port 8765
  python -m engine.research.discovery.review_ui --port 9999
  Then open http://localhost:8765/ in browser.

ENDPOINTS:
  GET  /                — main dashboard (queue + nominate + bookmarklet)
  POST /nominate        — accept URL/DOI/arxiv_id/OpenAlex Work ID
                          → fetch metadata → score → add to queue
  POST /action          — promote / skip / dismiss queue items
  GET  /api/queues      — JSON dump of current queues (for htmx refresh)

The KEY UX win is the BOOKMARKLET on the homepage: drag-to-bookmark-bar,
then on any paper page (arxiv / SSRN / DOI / OpenAlex) click → instant
add. No copy-paste of IDs, no CLI.
"""
from __future__ import annotations

import argparse
import datetime
import http.server
import json
import logging
import re
import socketserver
import sys
import urllib.parse
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
REPO_ROOT = Path(__file__).resolve().parents[3]
DISCOVERY_QUEUE = REPO_ROOT / "data" / "research" / "discovery_queue.jsonl"
DISCOVERY_BORDERLINE = REPO_ROOT / "data" / "research" / "discovery_borderline.jsonl"


# ── Identifier extraction ─────────────────────────────────────────────────

_RX_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?",
                              re.IGNORECASE)
_RX_ARXIV_BARE = re.compile(r"^\s*(\d{4}\.\d{4,5})(?:v\d+)?\s*$")
_RX_DOI = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)
_RX_OPENALEX = re.compile(r"(?:openalex\.org/)?(W\d+)", re.IGNORECASE)
_RX_SSRN = re.compile(r"(?:ssrn\.com/(?:abstract|sol3/papers\.cfm\?abstract_id))[_=]?(\d+)",
                          re.IGNORECASE)
_RX_SSRN_BARE = re.compile(r"^\s*(\d{6,8})\s*$")    # raw SSRN abstract_id


def extract_identifier(url_or_id: str) -> dict | None:
    """Parse arxiv ID, DOI, OpenAlex Work ID, or SSRN abstract ID."""
    if not url_or_id:
        return None
    u = url_or_id.strip()

    # OpenAlex Work ID — high specificity, check first
    m = _RX_OPENALEX.search(u)
    if m and ("openalex" in u.lower() or len(u) < 20):
        return {"type": "openalex", "id": m.group(1).upper()}

    # arxiv URL — must contain arxiv.org/abs or /pdf
    m = _RX_ARXIV_URL.search(u)
    if m:
        return {"type": "arxiv", "id": m.group(1)}

    # arxiv bare — entire input must be JUST the arxiv ID
    # (prevents matching 2024.1234 embedded inside a DOI)
    m = _RX_ARXIV_BARE.match(u)
    if m:
        return {"type": "arxiv", "id": m.group(1)}

    # SSRN URL
    m = _RX_SSRN.search(u)
    if m:
        return {"type": "ssrn", "id": m.group(1)}

    # SSRN bare 7-digit
    m = _RX_SSRN_BARE.match(u)
    if m:
        return {"type": "ssrn", "id": m.group(1)}

    # DOI
    m = _RX_DOI.search(u)
    if m:
        return {"type": "doi", "id": m.group(1).rstrip("/.;)>")}

    return None


# ── Metadata fetchers ─────────────────────────────────────────────────────

METADATA_CACHE_DIR = REPO_ROOT / "data" / "cache" / "nominate_metadata"
METADATA_CACHE_TTL_HOURS = 24


def _metadata_cache_path(ident: dict) -> Path:
    """Hash the identifier into a stable cache file path."""
    import hashlib
    raw = f"{ident.get('type', '')}|{ident.get('id', '')}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return METADATA_CACHE_DIR / f"{h}.json"


def _load_metadata_cached(ident: dict) -> dict | None:
    """Load cached metadata if within TTL. Returns None on miss/stale."""
    import datetime as _dt
    cache_path = _metadata_cache_path(ident)
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        ts_str = data.get("_cached_at", "")
        ts = _dt.datetime.fromisoformat(ts_str.rstrip("Z"))
        age = _dt.datetime.utcnow() - ts
        if age.total_seconds() > METADATA_CACHE_TTL_HOURS * 3600:
            return None
        # Strip cache metadata; return original record
        data.pop("_cached_at", None)
        return data
    except Exception:
        return None


def _save_metadata_cached(ident: dict, meta: dict) -> None:
    """Save metadata with cache timestamp. Best-effort."""
    import datetime as _dt
    METADATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _metadata_cache_path(ident)
    try:
        record = {**meta, "_cached_at": _dt.datetime.utcnow().isoformat() + "Z"}
        cache_path.write_text(json.dumps(record, ensure_ascii=False),
                                  encoding="utf-8")
    except Exception as exc:
        logger.warning("metadata cache save failed: %s", exc)


def fetch_metadata(ident: dict) -> dict | None:
    """Fetch normalized metadata for an identifier.

    Senior 漏洞 6: cache by (ident_type, ident_id) for 24h so re-nominate
    of same DOI doesn't burn API quota. Cache hit returns immediately.
    """
    cached = _load_metadata_cached(ident)
    if cached is not None:
        return cached
    try:
        result = None
        if ident["type"] == "openalex":
            result = _fetch_openalex(ident["id"])
        elif ident["type"] == "doi":
            result = _fetch_crossref(ident["id"])
        elif ident["type"] == "arxiv":
            result = _fetch_arxiv(ident["id"])
        elif ident["type"] == "ssrn":
            result = _fetch_openalex_by_ssrn(ident["id"])
        if result:
            _save_metadata_cached(ident, result)
        return result
    except Exception as exc:
        logger.warning("metadata fetch failed for %s: %s", ident, exc)
        return None


def _fetch_openalex(work_id: str) -> dict | None:
    import requests
    r = requests.get(
        f"https://api.openalex.org/works/{work_id}",
        timeout=15,
        headers={"User-Agent": "macro-alpha-research/1.0"},
    )
    if r.status_code != 200:
        return None
    item = r.json()
    from engine.research.discovery.openalex_fetcher import _normalize_openalex_record
    return _normalize_openalex_record(item, {})


def _fetch_openalex_by_ssrn(ssrn_id: str) -> dict | None:
    """SSRN abstract IDs are sometimes searchable in OpenAlex by URL."""
    import requests
    # Try searching OpenAlex for SSRN URL match
    ssrn_url = f"https://ssrn.com/abstract={ssrn_id}"
    r = requests.get(
        "https://api.openalex.org/works",
        params={"filter": f"locations.landing_page_url:{ssrn_url}",
                  "per-page": 1},
        timeout=15,
        headers={"User-Agent": "macro-alpha-research/1.0"},
    )
    if r.status_code != 200:
        return None
    items = r.json().get("results", [])
    if not items:
        return None
    from engine.research.discovery.openalex_fetcher import _normalize_openalex_record
    return _normalize_openalex_record(items[0], {})


def _fetch_crossref(doi: str) -> dict | None:
    """Fetch metadata for a DOI via 3-tier fallback chain.

    Per [[feedback-no-brittle-hardcoding-2026-05-30]] layered resilience:
    L1: Crossref (fast, canonical title/authors/venue, often NO abstract)
    L2: OpenAlex by DOI (inverted-index abstract for ~70% of papers)
    L3: Semantic Scholar by DOI (best abstract coverage including
        publisher-locked papers — many JFE/JF entries surface here)

    Merge strategy: take canonical title/authors/venue from L1, take
    abstract from whichever tier has the longest non-empty.
    """
    import requests
    cr_result = None
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=15,
            headers={"User-Agent": "macro-alpha-research/1.0"},
        )
        if r.status_code == 200:
            msg = r.json().get("message", {})
            from engine.research.discovery.crossref_fetcher import _normalize_crossref_record
            cr_result = _normalize_crossref_record(msg, {})
    except Exception as exc:
        logger.warning("crossref fetch failed for %s: %s", doi, exc)

    if cr_result and len(cr_result.get("abstract") or "") > 50:
        return cr_result

    # L2: OpenAlex by DOI
    oa_result = None
    try:
        oa_r = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            timeout=15,
            headers={"User-Agent": "macro-alpha-research/1.0"},
        )
        if oa_r.status_code == 200:
            from engine.research.discovery.openalex_fetcher import _normalize_openalex_record
            oa_result = _normalize_openalex_record(oa_r.json(), {})
    except Exception as exc:
        logger.warning("openalex DOI fallback failed for %s: %s", doi, exc)

    # L3: Semantic Scholar by DOI (best abstract coverage)
    s2_result = None
    try:
        s2_r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "title,abstract,authors,year,venue,citationCount,externalIds"},
            timeout=15,
            headers={"User-Agent": "macro-alpha-research/1.0"},
        )
        if s2_r.status_code == 200:
            s2 = s2_r.json()
            s2_abstract = (s2.get("abstract") or "").strip()
            if s2_abstract:
                s2_authors = "; ".join(
                    a.get("name", "") for a in (s2.get("authors") or [])
                )
                s2_result = {
                    "source":          "semantic_scholar",
                    "source_id":       doi,
                    "title":           s2.get("title", ""),
                    "abstract":        s2_abstract,
                    "authors":         s2_authors,
                    "venue":           s2.get("venue", ""),
                    "submitted_date":  (str(s2.get("year")) + "-01-01"
                                          if s2.get("year") else None),
                    "doi":             doi,
                    "abs_url":         f"https://doi.org/{doi}",
                    "citation_count":  s2.get("citationCount"),
                    "credibility_tier_hint": None,
                    "graveyard_routing": None,
                    "venue_category":  "",
                }
    except Exception as exc:
        logger.warning("S2 DOI fallback failed for %s: %s", doi, exc)

    # Merge: title/authors/venue from Crossref (canonical), abstract
    # from whichever has the longest non-empty.
    candidates = [x for x in (cr_result, oa_result, s2_result) if x]
    if not candidates:
        return None
    # Start from Crossref canonical metadata if available
    base = cr_result or candidates[0]
    merged = dict(base)
    # Find best abstract across all tiers
    abstracts = [(x.get("abstract") or "") for x in candidates]
    best_abstract = max(abstracts, key=len)
    if best_abstract:
        merged["abstract"] = best_abstract
    # Use citation_count from S2 if Crossref doesn't have it
    if not merged.get("citation_count") and s2_result and s2_result.get("citation_count"):
        merged["citation_count"] = s2_result["citation_count"]
    return merged


def _fetch_arxiv(arxiv_id: str) -> dict | None:
    """arxiv has its own Atom-formatted API."""
    import requests
    import xml.etree.ElementTree as ET
    r = requests.get(
        f"http://export.arxiv.org/api/query?id_list={arxiv_id}",
        timeout=15,
        headers={"User-Agent": "macro-alpha-research/1.0"},
    )
    if r.status_code != 200:
        return None
    try:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.content)
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="",
                                       namespaces=ns) or "")[:10]
        authors = []
        for a in entry.findall("atom:author/atom:name", ns):
            if a.text:
                authors.append(a.text.strip())
        return {
            "source":          "arxiv",
            "source_id":       arxiv_id,
            "title":           title.replace("\n", " "),
            "abstract":        summary.replace("\n", " "),
            "authors":         "; ".join(authors),
            "venue":           "arxiv",
            "submitted_date":  published or None,
            "doi":             None,
            "abs_url":         f"https://arxiv.org/abs/{arxiv_id}",
            "graveyard_routing": None,
            "credibility_tier_hint": None,
            "venue_category":  "",
        }
    except Exception:
        return None


# ── Nominate (the core action) ────────────────────────────────────────────

def nominate(url_or_id: str) -> dict:
    """Parse identifier → fetch metadata → score → queue.

    Confidence routing strategy:
      - Abstract present (≥ 50 chars): run deterministic calculator
      - Abstract missing or too short: fall back to venue-tier routing
        (a JF/JFE/RFS paper without abstract is more likely production-
        ready than an arxiv preprint without abstract)
    Manual nominates always land in primary queue regardless of routing
    tier — user judgment trumps auto-classifier.
    """
    ident = extract_identifier(url_or_id)
    if not ident:
        return {"error": f"could not parse identifier from {url_or_id!r}",
                "input": url_or_id}

    meta = fetch_metadata(ident)
    if not meta:
        return {"error": f"could not fetch metadata for {ident}",
                "ident": ident}

    title = meta.get("title", "")
    abstract = meta.get("abstract", "") or ""
    venue = meta.get("venue", "")
    pdf_url = meta.get("pdf_url") or meta.get("abs_url") or ""

    from engine.research.discovery.confidence_calculator import compute_confidence
    from engine.research.discovery.family_thresholds import explain_routing

    scoring_method = "text"
    gemini_extraction_dict = None
    if len(abstract.strip()) >= 50:
        det = compute_confidence(title, abstract)
        confidence_value = det.confidence
        confidence_dict = det.to_dict()
    else:
        # Senior 漏洞 8: abstract missing. Try Gemini PDF body first
        # (real content extraction), fall back to venue-tier if PDF
        # not downloadable / Gemini not available.
        gemini_pdf_ok = False
        if pdf_url and ".pdf" in pdf_url.lower() or pdf_url.endswith(".pdf"):
            try:
                from engine.research.discovery.gemini_pdf_extractor import (
                    extract_from_pdf,
                )
                gem = extract_from_pdf(pdf_url, doi=meta.get("doi"))
                gemini_extraction_dict = gem.to_dict()
                if gem.ok and gem.reconstructed_abstract:
                    abstract = gem.reconstructed_abstract
                    scoring_method = "gemini_pdf_extract"
                    det = compute_confidence(
                        title, abstract,
                        family_guess=gem.family_guess,
                    )
                    confidence_value = det.confidence
                    confidence_dict = det.to_dict()
                    gemini_pdf_ok = True
            except Exception as exc:
                logger.warning("gemini PDF fallback failed: %s", exc)
        if not gemini_pdf_ok:
            scoring_method = "venue_tier_fallback"
            from engine.research.discovery.credibility_scorer import (
                PaperMetadata, score_paper,
            )
            cred = score_paper(PaperMetadata(
                title=title, abstract="",
                authors=meta.get("authors", ""), venue=venue,
                submitted_date=meta.get("submitted_date"),
                doi=meta.get("doi"),
            ))
            # Use venue feature as confidence proxy
            confidence_value = cred.features.get("venue_tier", 0.4)
            confidence_dict = {
                "confidence":     round(confidence_value, 4),
                "method":         "venue_tier_fallback",
                "abstract_chars": len(abstract.strip()),
                "venue":          venue,
                "venue_tier":     cred.features.get("venue_tier"),
                "note":           "no abstract available; routing by venue tier",
            }

    routing = explain_routing(confidence_value, "unknown")

    entry = {
        "source":               "manual_nominate",
        "source_id":            ident["id"],
        "ident_type":           ident["type"],
        "title":                title,
        "abstract":             abstract,
        "authors":              meta.get("authors", ""),
        "venue":                venue,
        "doi":                  meta.get("doi"),
        "submitted_date":       meta.get("submitted_date"),
        "citation_count":       meta.get("citation_count"),
        "confidence":           confidence_dict,
        "scoring_method":       scoring_method,
        "routing":              routing,
        "nominated_via":        "ui",
        "ts":                   datetime.datetime.utcnow().isoformat() + "Z",
    }

    # Manual nominates ALWAYS land in primary review queue — user
    # explicitly signaled interest.
    DISCOVERY_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with DISCOVERY_QUEUE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    return {
        "ok":             True,
        "title":          title,
        "venue":          venue,
        "confidence":     confidence_value,
        "routing":        routing["routing"],
        "scoring_method": scoring_method,
        "queued_to":      DISCOVERY_QUEUE.name,
        "ident_type":     ident["type"],
        "ident_id":       ident["id"],
    }


# ── Queue I/O ─────────────────────────────────────────────────────────────

def read_queue(path: Path, limit: int = 20) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for raw in lines[-limit:][::-1]:
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


# ── HTML rendering ────────────────────────────────────────────────────────

def _render_queue_item(item: dict, tier: str) -> str:
    title = item.get("title", "")[:200]
    venue = item.get("venue", "") or item.get("source", "")
    routing = item.get("routing", {})
    conf = routing.get("adjusted_confidence")
    if conf is None:
        conf_d = item.get("confidence", {})
        if isinstance(conf_d, dict):
            conf = conf_d.get("confidence", 0)
        else:
            conf = conf_d or 0
    family = routing.get("family", "?")
    doi = item.get("doi") or item.get("ident_id", "")
    ts = item.get("ts", "")[:19]
    abs_url = item.get("abs_url") or (f"https://doi.org/{doi}" if doi else "#")

    return f"""
<div class="queue-item tier-{tier}">
  <div class="qi-title"><a href="{abs_url}" target="_blank">{title}</a></div>
  <div class="qi-meta">
    <span class="badge">{venue}</span>
    <span class="badge">family={family}</span>
    <span class="badge conf">conf={conf:.2f}</span>
    <span class="ts">{ts}</span>
  </div>
</div>
"""


def render_home() -> str:
    review = read_queue(DISCOVERY_QUEUE, limit=15)
    border = read_queue(DISCOVERY_BORDERLINE, limit=15)

    review_html = ("".join(_render_queue_item(r, "review") for r in review)
                     or "<p class=empty>No reviews queued yet.</p>")
    border_html = ("".join(_render_queue_item(r, "borderline") for r in border)
                     or "<p class=empty>No borderline papers yet.</p>")

    bookmarklet = (
        "javascript:(function(){"
        "fetch('http://localhost:8765/nominate',{method:'POST',"
        "headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({url:location.href,title:document.title})"
        "}).then(r=>r.json()).then(d=>{"
        "if(d.error)alert('Error: '+d.error);"
        "else alert('Added: '+(d.title||'').slice(0,80)+"
        "' (conf='+d.confidence.toFixed(2)+', '+d.routing+')');"
        "}).catch(e=>alert('UI server not running? '+e));"
        "})();"
    )

    return HOME_TEMPLATE.replace("{{REVIEW_QUEUE}}", review_html) \
                            .replace("{{BORDERLINE_QUEUE}}", border_html) \
                            .replace("{{BOOKMARKLET}}", bookmarklet)


HOME_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<title>Research Review Queue</title>
<meta charset="utf-8">
<script src="https://unpkg.com/htmx.org@1.9.10" crossorigin></script>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 960px;
         margin: 32px auto; color: #1a1a1a; padding: 0 16px; }
  h1 { font-size: 22px; margin: 0 0 24px 0; }
  h2 { font-size: 16px; margin: 24px 0 12px 0; color: #444; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  section { margin-bottom: 28px; }
  .nominate-form { display: flex; gap: 8px; }
  .nominate-form input[type=text] { flex: 1; padding: 8px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
  .nominate-form button { padding: 8px 16px; background: #2a6df4; color: white; border: none; border-radius: 4px; cursor: pointer; }
  .nominate-form button:hover { background: #1a5de4; }
  #nominate-result { margin-top: 8px; font-size: 13px; color: #555; min-height: 18px; }
  #nominate-result.ok { color: #1a7a3c; }
  #nominate-result.err { color: #b03a2e; }
  .queue-item { padding: 10px 12px; margin: 6px 0; border-radius: 4px; background: #fafafa; border-left: 4px solid #ccc; }
  .queue-item.tier-review { border-left-color: #1a7a3c; }
  .queue-item.tier-borderline { border-left-color: #d18a1a; }
  .qi-title { font-size: 14px; margin-bottom: 4px; }
  .qi-title a { color: #1a1a1a; text-decoration: none; }
  .qi-title a:hover { text-decoration: underline; }
  .qi-meta { font-size: 11px; color: #777; }
  .badge { display: inline-block; padding: 2px 6px; background: #eee; border-radius: 3px; margin-right: 6px; font-family: monospace; }
  .badge.conf { background: #d9eedd; }
  .ts { font-family: monospace; color: #999; }
  .empty { color: #999; font-style: italic; font-size: 13px; }
  pre.bookmarklet { background: #f4f4f4; padding: 10px; font-size: 10px; word-break: break-all;
                    white-space: pre-wrap; border: 1px solid #ddd; border-radius: 4px; }
  .bm-link { display: inline-block; padding: 6px 12px; background: #ffe28a; color: #555;
             border: 1px dashed #aa8a1a; border-radius: 4px; text-decoration: none;
             font-weight: bold; margin: 8px 0; cursor: move; }
  .hint { color: #777; font-size: 12px; }
</style>
</head>
<body>

<h1>Research Review Queue</h1>

<section>
<h2>Nominate Paper</h2>
<form class="nominate-form" hx-post="/nominate"
         hx-ext="json-enc" hx-target="#nominate-result" hx-swap="innerHTML">
  <input type="text" name="url"
            placeholder="paste DOI / arxiv URL / OpenAlex Work ID / SSRN URL"
            autofocus required>
  <button type="submit">Add</button>
</form>
<div id="nominate-result"></div>
<p class="hint">Or use the bookmarklet below for one-click add from any paper page.</p>
</section>

<section>
<h2>Bookmarklet</h2>
<p>Drag this link to your bookmarks bar, then click it while on any
paper page (arxiv / SSRN / DOI / OpenAlex) to instantly add:</p>
<a class="bm-link" href="{{BOOKMARKLET}}">[+] Add to Research Queue</a>
<p class="hint">If drag doesn't work, copy the JS below as a manual bookmark URL:</p>
<pre class="bookmarklet">{{BOOKMARKLET}}</pre>
</section>

<section>
<h2>Primary Review Queue</h2>
{{REVIEW_QUEUE}}
</section>

<section>
<h2>Borderline Queue (spot-check)</h2>
{{BORDERLINE_QUEUE}}
</section>

<script>
// Show nominate result inline (htmx returns JSON, we format it)
document.body.addEventListener("htmx:afterRequest", function(e) {
  if (e.detail.elt && e.detail.elt.matches("form.nominate-form")) {
    let div = document.getElementById("nominate-result");
    try {
      let d = JSON.parse(e.detail.xhr.responseText);
      if (d.error) {
        div.className = "err";
        div.innerText = "Error: " + d.error;
      } else {
        div.className = "ok";
        div.innerText = `Added "${d.title.slice(0,80)}" → ${d.routing} (conf=${d.confidence.toFixed(2)}, ${d.ident_type})`;
        document.querySelector("input[name=url]").value = "";
        setTimeout(() => location.reload(), 800);
      }
    } catch(err) {
      div.className = "err";
      div.innerText = "Server returned malformed response";
    }
  }
});
</script>

</body></html>
"""


# ── HTTP handler ──────────────────────────────────────────────────────────

class ReviewHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(render_home())
        elif self.path == "/api/queues":
            self._send_json({
                "review":     read_queue(DISCOVERY_QUEUE, limit=20),
                "borderline": read_queue(DISCOVERY_BORDERLINE, limit=20),
            })
        else:
            self.send_error(404, "Not found")

    def do_OPTIONS(self):
        """CORS preflight for bookmarklet from any origin."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == "/nominate":
            self._handle_nominate()
        else:
            self.send_error(404, "Not found")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        ct = self.headers.get("Content-Type", "")
        try:
            if "json" in ct:
                return json.loads(raw)
            return dict(urllib.parse.parse_qsl(raw))
        except Exception:
            return {}

    def _handle_nominate(self):
        body = self._read_body()
        url_or_id = (body.get("url") or body.get("id")
                       or body.get("doi") or body.get("arxiv_id") or "")
        if not url_or_id:
            self._send_json({"error": "missing url/id field"}, status=400)
            return
        try:
            result = nominate(url_or_id)
            status = 200 if result.get("ok") else 400
            self._send_json(result, status=status)
        except Exception as exc:
            logger.exception("nominate failed")
            self._send_json({"error": str(exc)}, status=500)

    def _send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status: int = 200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        # Bookmarklet can come from any origin — explicit allow
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


# ── Server entry ─────────────────────────────────────────────────────────

def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1"):
    """Run the review UI server on localhost. Blocks until Ctrl+C."""
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), ReviewHandler) as httpd:
        print(f"[review_ui] serving http://{host}:{port}/")
        print(f"[review_ui] queue file:      {DISCOVERY_QUEUE}")
        print(f"[review_ui] borderline file: {DISCOVERY_BORDERLINE}")
        print(f"[review_ui] press Ctrl+C to shutdown")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("[review_ui] shutting down")
            httpd.shutdown()


def _cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1, localhost only)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    serve(port=args.port, host=args.host)


if __name__ == "__main__":
    _cli()
