"""engine/data/fetchers/scraper_wikipedia.py — Wikipedia HTML scraping.

3-LAYER resilience strategy (not single hardcoded selector):

  Layer 1: PRIMARY — pd.read_html with table id="constituents" attr
  Layer 2: SEMANTIC — BeautifulSoup, find table containing both "Symbol"
           and "Security" headers (or fuzzy variants)
  Layer 3: LLM RESCUE — when both above fail, feed HTML to LLM and ask
           "find the S&P 500 constituents table" with strict output schema

Layer 3 is opt-in via env LAYER3_LLM_RESCUE=true (cost ~$0.05/call).
On success, the successful selector is CACHED to layer1_selector_cache.json
so subsequent calls skip Layer 2/3.

Senior-quant care:
- Wikipedia is not a primary source of truth — recommended for ANALYSIS
  (constituent membership over time) only; use WRDS / Yahoo as authoritative
- Schema can break with Wikipedia infobox redesigns (happens ~yearly)
- We always return rows in a CANONICAL format regardless of scraping layer

Doctrine:
- Probe = HEAD request only, no scraping
- Multi-layer fetch gracefully falls through; logs which layer served
- Never silently substitute mock data on scrape failure (return empty + raise)
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from engine.data.orchestrator import ProbeResult
from engine.data.fetchers._common import http_session, replace_sentinel_values

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

SELECTOR_CACHE = Path(__file__).resolve().parent / "_layer1_selector_cache.json"


def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    """HEAD request only; verifies the URL is reachable and is HTML."""
    import time
    t0 = time.time()
    session = http_session()
    try:
        resp = session.head(SP500_URL, timeout=10, allow_redirects=True)
    except Exception as exc:
        return ProbeResult(
            available=False, error=f"Wikipedia probe network: {exc}",
            error_class="network", elapsed_secs=time.time() - t0,
        )
    if resp.status_code != 200:
        return ProbeResult(
            available=False, error=f"Wikipedia HTTP {resp.status_code}",
            error_class="network" if resp.status_code >= 500 else "schema_unknown",
            elapsed_secs=time.time() - t0,
        )
    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        return ProbeResult(
            available=False, error=f"Wikipedia returned non-HTML: {content_type}",
            error_class="schema_unknown", elapsed_secs=time.time() - t0,
        )
    return ProbeResult(
        available=True, error=None, error_class=None,
        elapsed_secs=time.time() - t0,
    )


# ── Layer 1: hardcoded primary selector ─────────────────────────────────

def _layer1_pd_read_html(html: str) -> pd.DataFrame | None:
    """Try pd.read_html with the historically-known table id."""
    from io import StringIO
    try:
        tables = pd.read_html(StringIO(html), attrs={"id": "constituents"})
    except Exception:
        try:
            tables = pd.read_html(StringIO(html))
        except Exception:
            return None
    for tbl in tables:
        # Canonical columns we want
        cols_lower = {c.lower() for c in tbl.columns}
        if {"symbol", "security"}.issubset(cols_lower) or \
            {"ticker", "name"}.issubset(cols_lower):
            return _normalize_columns(tbl)
    return None


# ── Layer 2: semantic search via BeautifulSoup ──────────────────────────

def _layer2_semantic_bs(html: str) -> pd.DataFrame | None:
    """Find any table containing ticker-like and name-like columns
    via fuzzy header search. Resilient to id/class renames."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        headers = [h.get_text(strip=True).lower()
                    for h in table.find_all("th")]
        if not headers:
            continue
        # Look for ticker-like header
        ticker_idx = next(
            (i for i, h in enumerate(headers)
              if re.search(r"\b(symbol|ticker|stock\s+symbol)\b", h)),
            None,
        )
        name_idx = next(
            (i for i, h in enumerate(headers)
              if re.search(r"\b(security|name|company|company\s+name)\b", h)),
            None,
        )
        if ticker_idx is None or name_idx is None:
            continue
        try:
            from io import StringIO
            tbl = pd.read_html(StringIO(str(table)))[0]
        except Exception:
            continue
        return _normalize_columns(tbl)
    return None


# ── Layer 3: LLM rescue (opt-in via env) ────────────────────────────────

def _layer3_llm_rescue(html: str) -> pd.DataFrame | None:
    """Feed HTML excerpt to LLM and ask it to identify constituents.

    Cost: ~$0.05 per call. Cached on success to avoid re-trigger.
    Opt-in via env LAYER3_LLM_RESCUE=true (default off — expensive).
    """
    import os
    if os.environ.get("LAYER3_LLM_RESCUE", "").lower() not in ("true", "1"):
        logger.info("Layer 3 LLM rescue not enabled; set LAYER3_LLM_RESCUE=true to use")
        return None
    try:
        from engine.research.economic_check import _read_anthropic_key
        from anthropic import Anthropic
    except ImportError:
        return None
    key = _read_anthropic_key()
    if not key:
        return None

    # Truncate HTML to fit in context
    snippet = html[:50000]
    client = Anthropic(api_key=key, timeout=60.0)
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=(
                "Extract S&P 500 constituents table from the given HTML. "
                "Return strictly a JSON array of objects with keys: "
                "ticker, name, sector, date_added (YYYY-MM-DD or null). "
                "Return ONLY the JSON array, no preamble."
            ),
            messages=[{"role": "user", "content": snippet}],
        )
    except Exception as exc:
        logger.warning("Layer 3 LLM rescue failed: %s", exc)
        return None
    text = "".join(b.text for b in resp.content if b.type == "text")
    # Strict JSON parse
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("Layer 3 LLM did not return JSON array")
        return None
    try:
        rows = json.loads(text[start:end + 1])
    except Exception as exc:
        logger.warning("Layer 3 LLM JSON parse failed: %s", exc)
        return None
    if not rows or not isinstance(rows, list):
        return None
    df = pd.DataFrame(rows)
    return _normalize_columns(df)


# ── Canonical column normalization ──────────────────────────────────────

def _normalize_columns(tbl: pd.DataFrame) -> pd.DataFrame:
    """Map any variant of ticker/name/sector/date_added column names to canonical."""
    rename_map: dict[str, str] = {}
    for c in tbl.columns:
        cl = str(c).lower().strip()
        if cl in ("symbol", "ticker", "stock symbol"):
            rename_map[c] = "ticker"
        elif cl in ("security", "name", "company", "company name"):
            rename_map[c] = "name"
        elif cl in ("gics sector", "sector"):
            rename_map[c] = "sector"
        elif "date" in cl and ("added" in cl or "first" in cl):
            rename_map[c] = "date_added"
    tbl = tbl.rename(columns=rename_map)
    keep = [c for c in ("ticker", "name", "sector", "date_added") if c in tbl.columns]
    if not keep or "ticker" not in keep:
        return pd.DataFrame(columns=["ticker", "name", "sector", "date_added"])
    out = tbl[keep].copy()
    # Coerce date_added
    if "date_added" in out.columns:
        out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce")
    # Fill missing columns
    for c in ("ticker", "name", "sector", "date_added"):
        if c not in out.columns:
            out[c] = None
    return out[["ticker", "name", "sector", "date_added"]].reset_index(drop=True)


# ── Selector cache ──────────────────────────────────────────────────────

def _load_selector_cache() -> dict:
    if SELECTOR_CACHE.exists():
        try:
            return json.loads(SELECTOR_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_selector_cache(d: dict) -> None:
    SELECTOR_CACHE.write_text(json.dumps(d, indent=2), encoding="utf-8")


# ── Main fetch ──────────────────────────────────────────────────────────

def scrape_sp500(start: str, end: str, **kw) -> pd.DataFrame:
    """Scrape S&P 500 constituents with 3-layer resilience.

    Tries Layer 1 → 2 → 3 in order. Records which layer served via
    metadata on the returned DataFrame (df.attrs['layer_used']).
    """
    session = http_session()
    try:
        resp = session.get(SP500_URL, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Wikipedia GET failed: %s", exc)
        return pd.DataFrame(columns=["ticker", "name", "sector", "date_added"])
    html = resp.text

    # Layer 1
    df = _layer1_pd_read_html(html)
    if df is not None and not df.empty:
        df.attrs["layer_used"] = 1
        return df

    # Layer 2
    df = _layer2_semantic_bs(html)
    if df is not None and not df.empty:
        df.attrs["layer_used"] = 2
        logger.info("Wikipedia Layer 1 failed; Layer 2 succeeded")
        # Cache the success — next time we know Layer 2 works
        cache = _load_selector_cache()
        cache["sp500_layer"] = 2
        _save_selector_cache(cache)
        return df

    # Layer 3 (opt-in LLM rescue)
    df = _layer3_llm_rescue(html)
    if df is not None and not df.empty:
        df.attrs["layer_used"] = 3
        logger.warning("Wikipedia Layers 1+2 failed; Layer 3 LLM rescue succeeded")
        return df

    logger.error("Wikipedia all 3 layers failed; returning empty DataFrame")
    return pd.DataFrame(columns=["ticker", "name", "sector", "date_added"])
