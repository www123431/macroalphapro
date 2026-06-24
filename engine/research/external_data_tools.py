"""engine/research/external_data_tools.py — Phase post-5.7: external
data tools (arxiv / sec_edgar / fred) as MCP-callable functions.

Per [[feedback-pre-implementation-fitness-check-2026-06-01]]: NO
generic web-search. 3 specialized read-only tools target the actual
gaps in our existing 11-tool registry:

  arxiv_search       — academic paper discovery beyond verified
                        master_index (which only carries our hand-
                        curated tier 1/2 set)
  sec_edgar_search   — company filing full-text + metadata
                        (issuance / buyback / risk factor / MD&A)
  fred_query         — macro time series (regime context, FED rate,
                        VIX history, unemployment, term structure)

Each tool: read-only, zero-write, no API key burden beyond what's
already configured. fred_query reuses existing FRED_API_KEY.
arxiv + sec_edgar use public endpoints (no key).

Senior scope notes:
  - For factual / pricing / financial data, Bloomberg/Refinitiv/WRDS
    remain authoritative. These tools complement, don't replace.
  - All 3 return STRUCTURED dicts (title / url / abstract / data series)
    suitable for LLM tool-use consumption.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)


# ── arxiv ──────────────────────────────────────────────────────────────

_ARXIV_API = "https://export.arxiv.org/api/query"
_ARXIV_HEADERS = {
    # arxiv requests a descriptive User-Agent per their robot policy
    "User-Agent": "MacroAlphaPro/1.0 (https://github.com/; research)",
}


def arxiv_search(query: str, max_results: int = 5) -> dict:
    """Search arxiv.org for academic papers. Returns top results with
    metadata + abstract. Free API, no key required.

    Args:
      query: free-text search (title, author, abstract). Use quant
        finance terminology: "cross-sectional momentum", "term
        structure carry", "post earnings drift", etc.
      max_results: 1-20, default 5.
    """
    max_results = max(1, min(int(max_results), 20))
    try:
        resp = requests.get(
            _ARXIV_API,
            params={
                "search_query": query,
                "start":        0,
                "max_results":  max_results,
            },
            headers=_ARXIV_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        return {"error": f"arxiv fetch failed: {exc}", "n": 0,
                "results": []}

    # arxiv returns Atom XML; we parse minimally without lxml dep
    text = resp.text
    entries: list[dict] = []
    # Split on <entry>...</entry>
    for raw in re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL):
        def _find(tag: str) -> str:
            m = re.search(f"<{tag}[^>]*>(.*?)</{tag}>", raw, re.DOTALL)
            return (m.group(1).strip() if m else "")
        title = re.sub(r"\s+", " ", _find("title")).strip()
        summary = re.sub(r"\s+", " ", _find("summary")).strip()
        published = _find("published")
        url_m = re.search(r'<id>(http[^<]+)</id>', raw)
        url = url_m.group(1).strip() if url_m else ""
        # Extract authors
        authors = re.findall(r"<name>([^<]+)</name>", raw)
        entries.append({
            "title":     title[:300],
            "url":       url,
            "published": published[:10],
            "authors":   authors[:6],
            "abstract":  summary[:1200],
        })
        if len(entries) >= max_results:
            break
    return {"n": len(entries), "results": entries, "query": query}


# ── SEC EDGAR ──────────────────────────────────────────────────────────

_EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
# SEC fair-access policy requires identifying User-Agent with a real
# contact email. They block / 500 if missing or boilerplate.
_EDGAR_HEADERS = {
    "User-Agent": "MacroAlphaPro research ${USER_EMAIL}",
    "Accept":     "application/json",
}


def sec_edgar_search(
    query: str,
    forms: Optional[list[str]] = None,
    n_results: int = 10,
) -> dict:
    """Full-text search SEC EDGAR filings. Returns matched filings
    with metadata. Free, no key. Rate-limited per SEC fair-access.

    Args:
      query: free-text (in filing body). Quote phrases for exact match.
      forms: filter by form type ["10-K", "10-Q", "8-K", "DEF 14A", ...]
      n_results: 1-50, default 10.
    """
    n_results = max(1, min(int(n_results), 50))
    # SEC API 500s on bare multi-word queries; wrap them in quotes if
    # the caller didn't already (single word or already-quoted passes
    # through). This is an EDGAR quirk, not arxiv-style OR-of-terms.
    if " " in query and not (query.strip().startswith('"') and
                              query.strip().endswith('"')):
        query_safe = f'"{query.strip()}"'
    else:
        query_safe = query
    params: dict[str, Any] = {"q": query_safe}
    if forms:
        params["forms"] = ",".join(forms)

    try:
        resp = requests.get(
            _EDGAR_BASE,
            params=params,
            headers=_EDGAR_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"error": f"EDGAR fetch failed: {exc}", "n": 0,
                "results": []}

    hits = (data.get("hits") or {}).get("hits") or []
    results: list[dict] = []
    for h in hits[:n_results]:
        src = h.get("_source") or {}
        # EDGAR full-text doc id format: <accno>:<filename>
        adsh = ((h.get("_id") or "").split(":")[0]) or ""
        cik  = src.get("ciks", [None])[0] if src.get("ciks") else None
        results.append({
            "adsh":       adsh,
            "cik":        cik,
            "form":       src.get("form"),
            "filed_date": src.get("file_date"),
            "company":    (src.get("display_names") or [None])[0],
            "snippet":    (src.get("xsl") or "")[:400],
            "score":      h.get("_score"),
        })
    return {
        "n":       len(results),
        "results": results,
        "query":   query,
        "forms":   forms or "any",
    }


# ── FRED macro query ───────────────────────────────────────────────────


def _get_fred_key() -> Optional[str]:
    """Resolve FRED API key: env first, then secrets.toml direct read.
    Streamlit's get_secret() helper relies on streamlit being running;
    outside that runtime (e.g. MCP server / pytest), it can return None
    silently. Direct file read is the reliable path."""
    import os
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
    except ImportError:
        return None
    from pathlib import Path
    secrets_path = Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return None
    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data.get("FRED_API_KEY") or data.get("fred_api_key")


def fred_query(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Fetch a FRED macroeconomic time series.

    Common series_ids:
      UNRATE        — unemployment rate
      VIXCLS        — VIX close
      DGS10         — 10y treasury yield (daily)
      T10Y2Y        — 10y - 2y term spread
      CPIAUCSL      — CPI all urban consumers
      FEDFUNDS      — federal funds rate
      DTB3          — 3-month T-bill
      GDP           — real GDP
      M2SL          — M2 money supply
      WALCL         — Fed total assets

    Args:
      series_id: FRED series identifier (case-sensitive)
      start_date: YYYY-MM-DD optional
      end_date: YYYY-MM-DD optional
    """
    key = _get_fred_key()
    if not key:
        return {
            "error": "FRED_API_KEY not configured (checked env + "
                     ".streamlit/secrets.toml)",
            "series_id": series_id, "n_obs": 0, "observations": [],
        }
    try:
        from fredapi import Fred
        client = Fred(api_key=key)
        kwargs = {}
        if start_date:
            kwargs["observation_start"] = start_date
        if end_date:
            kwargs["observation_end"] = end_date
        s = client.get_series(series_id, **kwargs)
    except Exception as exc:
        return {"error": f"FRED fetch failed: {exc}",
                "series_id": series_id, "n_obs": 0, "observations": []}

    if s is None or len(s) == 0:
        return {"error": "empty series", "series_id": series_id,
                "n_obs": 0, "observations": []}

    # Cap to first/last + sample for very long series
    n = len(s)
    if n > 200:
        # Decimate to ~200 points evenly spaced, preserve first / last
        step = max(1, n // 200)
        decimated = s.iloc[::step]
        obs = [{"date": str(d)[:10], "value": (None if v != v else float(v))}
                for d, v in decimated.items()]
    else:
        obs = [{"date": str(d)[:10], "value": (None if v != v else float(v))}
                for d, v in s.items()]

    return {
        "series_id":    series_id,
        "n_obs":        n,
        "n_returned":   len(obs),
        "start_date":   str(s.index[0])[:10],
        "end_date":     str(s.index[-1])[:10],
        "latest_value": (None if s.iloc[-1] != s.iloc[-1] else float(s.iloc[-1])),
        "observations": obs,
    }
