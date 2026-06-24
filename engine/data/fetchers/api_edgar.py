"""engine/data/fetchers/api_edgar.py — SEC EDGAR 8-K and filings metadata.

Free public source. Strict User-Agent requirement (SEC fair access policy)
— if UA is missing or wrong, SEC will block. Rate limit: 10 req/sec.

Senior-quant care:
- Filing date in EST (NYSE-aligned); convert to UTC.
- 8-K item codes matter for event-study work (Item 2.02 = earnings; Item 5.02 = exec change; Item 8.01 = misc).
- Restated filings appear as later filings — must filter by original_filing_date for PIT.
"""
from __future__ import annotations

import logging
from urllib.parse import urlencode

import pandas as pd

from engine.data.orchestrator import ProbeResult
from engine.data.fetchers._common import http_session, to_utc_dates

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

EDGAR_BASE = "https://efts.sec.gov/LATEST/search-index"
EDGAR_USER_AGENT = (
    "macro-alpha-research/1.0 (academic; research@local) "
    "Mozilla/5.0 (compatible; ResearchBot/1.0)"
)


def probe(start: str, end: str, *, target_function: str | None = None,
            **kw) -> ProbeResult:
    """Probe EDGAR via single-result search."""
    import time
    t0 = time.time()
    session = http_session(user_agent=EDGAR_USER_AGENT)
    try:
        url = f"{EDGAR_BASE}?q=&forms=8-K&dateRange=custom" \
              f"&startdt={start}&enddt={start}&hits=1"
        resp = session.get(url, timeout=10)
        if resp.status_code in (403, 429):
            return ProbeResult(
                available=False, error=f"EDGAR rate-limit/auth: HTTP {resp.status_code}",
                error_class="rate_limited" if resp.status_code == 429
                              else "access_denied",
                elapsed_secs=time.time() - t0,
            )
        if resp.status_code != 200:
            return ProbeResult(
                available=False, error=f"EDGAR HTTP {resp.status_code}",
                error_class="network", elapsed_secs=time.time() - t0,
            )
    except Exception as exc:
        return ProbeResult(
            available=False, error=f"EDGAR probe network error: {exc}",
            error_class="network", elapsed_secs=time.time() - t0,
        )
    return ProbeResult(
        available=True, error=None, error_class=None,
        elapsed_secs=time.time() - t0,
    )


def fetch_8k_meta(start: str, end: str, *,
                   max_results: int = 1000, **kw) -> pd.DataFrame:
    """Fetch 8-K filing metadata in date range.

    Returns columns: filing_date, cik, accession_no, filing_type, link.

    Pagination via EDGAR's `from` parameter; we cap at max_results.
    """
    session = http_session(user_agent=EDGAR_USER_AGENT)
    rows = []
    fetched = 0
    batch_size = 100
    while fetched < max_results:
        params = {
            "q": "",
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": start,
            "enddt": end,
            "hits": batch_size,
            "from": fetched,
        }
        url = f"{EDGAR_BASE}?{urlencode(params)}"
        try:
            resp = session.get(url, timeout=15)
        except Exception as exc:
            logger.warning("EDGAR fetch failed at offset %d: %s", fetched, exc)
            break
        if resp.status_code != 200:
            logger.warning("EDGAR HTTP %d at offset %d", resp.status_code, fetched)
            break
        try:
            payload = resp.json()
        except Exception:
            logger.warning("EDGAR returned non-JSON at offset %d", fetched)
            break
        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            src = h.get("_source", {})
            ciks = src.get("ciks") or []
            rows.append({
                "filing_date":  src.get("file_date"),
                "cik":          ciks[0] if ciks else None,
                "accession_no": src.get("adsh", h.get("_id", "")),
                "filing_type":  src.get("form", "8-K"),
                "link":         f"https://www.sec.gov/Archives/edgar/data/"
                                  f"{(ciks[0] if ciks else '')}/"
                                  f"{src.get('adsh', '').replace('-', '')}",
            })
        fetched += len(hits)
        if len(hits) < batch_size:
            break
    if not rows:
        return pd.DataFrame(columns=["filing_date", "cik", "accession_no",
                                       "filing_type", "link"])
    df = pd.DataFrame(rows)
    df["filing_date"] = to_utc_dates(df["filing_date"])
    return df.sort_values("filing_date").reset_index(drop=True)
