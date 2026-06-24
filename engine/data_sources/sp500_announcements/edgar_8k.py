"""
engine/data_sources/sp500_announcements/edgar_8k.py — SEC EDGAR 8-K monitor.

Sprint D-1 secondary source. Queries SEC EDGAR full-text search API for
8-K filings containing S&P 500 inclusion announcement language.

EDGAR API:
  https://efts.sec.gov/LATEST/search-index?q=<query>&forms=8-K&dateRange=custom&startdt=YYYY-MM-DD&enddt=YYYY-MM-DD

Returns: structured JSON with filing list including filer name (with CIK +
ticker symbols if available), file_date (filing date ≈ announcement date),
and link to filing.

Purpose: refine `announcement_date` for Wikipedia-detected events. Wikipedia
gives effective_date; EDGAR gives 8-K filing date which is typically closer
to actual announcement.
"""
from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


EDGAR_SEARCH_API = "https://efts.sec.gov/LATEST/search-index"

DEFAULT_USER_AGENT = (
    "MacroAlphaPro Research ${USER_EMAIL} "
    "(quant research; respects robots.txt; rate-limited)"
)

DEFAULT_TIMEOUT_SECONDS = 15

# Phrases commonly used in 8-K when company is added to S&P 500
SP500_INCLUSION_QUERIES = [
    '"added to the S&P 500"',
    '"will be added to the S&P 500"',
    '"inclusion in the S&P 500"',
    '"join the S&P 500"',
]


@dataclass(frozen=True)
class Edgar8KFiling:
    """One 8-K filing matching S&P 500 inclusion query."""
    cik:           str
    filer_name:    str
    tickers:       list[str]    # may be empty
    file_date:     datetime.date
    accession_no:  str
    query_matched: str
    raw_display:   str          # raw display_names field for audit


def _parse_filer_display(display_names: list[str]) -> tuple[str, list[str], str]:
    """Parse EDGAR display_names like:
    'REALTY INCOME CORP  (O, O-P)  (CIK 0000726728)' →
        name='REALTY INCOME CORP', tickers=['O', 'O-P'], cik='0000726728'
    """
    if not display_names:
        return ("", [], "")
    raw = display_names[0]
    # Extract tickers in parens before CIK
    tick_match = re.search(r"\(([^)]+)\)\s+\(CIK\s+(\d+)\)", raw)
    cik = ""
    tickers: list[str] = []
    name = raw
    if tick_match:
        ticker_str = tick_match.group(1)
        cik = tick_match.group(2)
        tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
        # Remove the ticker + CIK section from name
        name = raw[:tick_match.start()].strip()
    else:
        # Sometimes only CIK is in parens
        cik_match = re.search(r"\(CIK\s+(\d+)\)", raw)
        if cik_match:
            cik = cik_match.group(1)
            name = raw[:cik_match.start()].strip()
    return (name, tickers, cik)


def fetch_edgar_8k_sp500_filings(
    start_date:     datetime.date,
    end_date:       Optional[datetime.date]    = None,
    user_agent:     str                         = DEFAULT_USER_AGENT,
    timeout_seconds: int                        = DEFAULT_TIMEOUT_SECONDS,
    max_results_per_query: int                  = 50,
) -> list[Edgar8KFiling]:
    """Query EDGAR 8-K filings matching S&P 500 inclusion language.

    Tries each query in SP500_INCLUSION_QUERIES; dedupes by accession_no.

    Args:
        start_date: search from this date (inclusive)
        end_date:   search to this date (inclusive, default today)
        user_agent: required by SEC (include contact email per their policy)
    """
    if end_date is None:
        end_date = datetime.date.today()

    headers = {"User-Agent": user_agent}
    seen_accessions: set[str] = set()
    filings: list[Edgar8KFiling] = []

    for query in SP500_INCLUSION_QUERIES:
        params = {
            "q":         query,
            "forms":     "8-K",
            "dateRange": "custom",
            "startdt":   start_date.isoformat(),
            "enddt":     end_date.isoformat(),
        }
        try:
            r = requests.get(
                EDGAR_SEARCH_API, headers=headers, params=params,
                timeout=timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("EDGAR query %r failed: %s", query, exc)
            continue

        if r.status_code != 200:
            logger.warning("EDGAR query %r HTTP %d", query, r.status_code)
            continue

        try:
            j = r.json()
        except Exception:
            logger.warning("EDGAR query %r non-JSON response", query)
            continue

        hits = j.get("hits", {}).get("hits", [])
        for hit in hits[:max_results_per_query]:
            src = hit.get("_source", {})
            accession = src.get("adsh", "") or hit.get("_id", "")
            if accession in seen_accessions:
                continue
            seen_accessions.add(accession)

            display_names = src.get("display_names", [])
            name, tickers, cik = _parse_filer_display(display_names)

            file_date_str = src.get("file_date", "")
            try:
                file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d").date()
            except ValueError:
                logger.warning("EDGAR result has bad file_date %r", file_date_str)
                continue

            filings.append(Edgar8KFiling(
                cik           = cik,
                filer_name    = name,
                tickers       = tickers,
                file_date     = file_date,
                accession_no  = accession,
                query_matched = query,
                raw_display   = display_names[0] if display_names else "",
            ))

    logger.info(
        "fetch_edgar_8k_sp500_filings %s to %s: %d unique filings across %d queries",
        start_date, end_date, len(filings), len(SP500_INCLUSION_QUERIES),
    )
    return filings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Smoke test: query last 3 months
    end = datetime.date.today()
    start = end - datetime.timedelta(days=90)
    filings = fetch_edgar_8k_sp500_filings(start, end)
    print(f"\nFetched {len(filings)} EDGAR 8-K filings (S&P 500 inclusion) "
          f"{start} to {end}")
    for f in filings[:15]:
        tk = "/".join(f.tickers) if f.tickers else "—"
        print(f"  {f.file_date} {tk:<8} CIK={f.cik:<10} "
              f"{f.filer_name[:45]:<45} (matched: {f.query_matched})")
