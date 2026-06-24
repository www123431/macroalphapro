"""
engine/data_sources/sp500_announcements/wikipedia.py — Wikipedia S&P 500 changes parser.

Sprint D-1 primary source. Parses the "Selected changes" table on
https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

The table has 4 columns:
  - Effective Date (e.g. "May 7, 2026")
  - Added: Ticker | Security
  - Removed: Ticker | Security
  - Reason (free text)

Output: list of SP500ChangeEvent (one per add or one per remove).
"""
from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


WIKIPEDIA_API_BASE      = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SP500_PAGE    = "List of S&P 500 companies"
WIKIPEDIA_SP500_URL     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_SELECTED_CHANGES_SECTION = "2"   # "Selected changes" is the 2nd section

DEFAULT_USER_AGENT = (
    "MacroAlphaPro Research ${USER_EMAIL} "
    "(quant research; respects robots.txt; rate-limited 1 req/day)"
)

DEFAULT_TIMEOUT_SECONDS = 15

# Standard S&P announcement-to-effective lag (per S&P Global standing practice)
ANNOUNCEMENT_LAG_TRADING_DAYS = 5


@dataclass(frozen=True)
class SP500ChangeEvent:
    """One S&P 500 reconstitution event (add OR remove).

    For paper-trade Path N alpha capture, only ADD events with future
    effective_date matter (long T-5 to T-1).
    """
    effective_date:     datetime.date
    announcement_date:  Optional[datetime.date]   # heuristic (eff - 5 trading days) or refined by EDGAR
    ticker:             str                       # symbol (e.g. "VEEV")
    company_name:       str                       # full name (e.g. "Veeva Systems")
    action:             str                       # "ADD" | "REMOVE"
    reason:             str                       # free text from Wikipedia
    source:             str = "wikipedia"


def _parse_effective_date(text: str) -> Optional[datetime.date]:
    """Wikipedia uses formats like 'May 7, 2026' or 'September 18, 2025'."""
    text = text.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Try removing footnote markers like [1][2]
    cleaned = re.sub(r"\[\d+\]", "", text).strip()
    if cleaned != text:
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
    return None


def _estimate_announcement_date(
    effective_date: datetime.date,
    lag_trading_days: int = ANNOUNCEMENT_LAG_TRADING_DAYS,
) -> datetime.date:
    """Heuristic: announcement ≈ effective_date minus 5 trading days.

    For Sprint D-1 we use simple calendar approximation (subtract 7 calendar days
    to cover the typical 5 NYSE trading days with weekend buffer). Sprint D-2
    will integrate pandas_market_calendars for exact NYSE day arithmetic.
    """
    return effective_date - datetime.timedelta(days=lag_trading_days + 2)


def _clean_cell_text(text: str) -> str:
    """Strip Wikipedia footnote markers like [1], [2]; collapse whitespace."""
    if not text:
        return ""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_wikipedia_sp500_changes(
    user_agent:        str = DEFAULT_USER_AGENT,
    timeout_seconds:   int = DEFAULT_TIMEOUT_SECONDS,
    section:           str = WIKIPEDIA_SELECTED_CHANGES_SECTION,
) -> list[SP500ChangeEvent]:
    """Fetch + parse the Wikipedia 'Selected changes' table.

    Returns list of SP500ChangeEvent (separate events for ADD and REMOVE in
    each row of the source table).

    Raises RuntimeError on HTTP failure or schema-breaking HTML change.
    """
    headers = {"User-Agent": user_agent}
    params = {
        "action":  "parse",
        "page":    WIKIPEDIA_SP500_PAGE,
        "format":  "json",
        "section": section,
        "prop":    "text",
    }
    r = requests.get(
        WIKIPEDIA_API_BASE, headers=headers, params=params,
        timeout=timeout_seconds,
    )
    if r.status_code != 200:
        raise RuntimeError(
            f"Wikipedia API HTTP {r.status_code}: {r.text[:200]}"
        )

    j = r.json()
    if "error" in j:
        raise RuntimeError(f"Wikipedia API error: {j['error']}")

    html = j.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        raise RuntimeError("Wikipedia API returned empty section HTML")

    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError(
            "Wikipedia 'Selected changes' section has no table; "
            "HTML structure may have changed"
        )

    # The first table is the Selected Changes table
    table = tables[0]
    rows = table.find_all("tr")

    events: list[SP500ChangeEvent] = []
    parse_errors: list[str] = []

    # First 2 rows are headers (group + sub-header); data starts at row 2
    for row_idx, row in enumerate(rows[2:], start=2):
        cells = row.find_all(["td", "th"])
        cell_texts = [_clean_cell_text(c.get_text()) for c in cells]
        if len(cell_texts) < 5:
            continue

        eff_date_text = cell_texts[0]
        added_ticker   = cell_texts[1]
        added_name     = cell_texts[2]
        removed_ticker = cell_texts[3]
        removed_name   = cell_texts[4]
        reason         = cell_texts[5] if len(cell_texts) > 5 else ""

        eff_date = _parse_effective_date(eff_date_text)
        if eff_date is None:
            parse_errors.append(f"row {row_idx}: bad date {eff_date_text!r}")
            continue

        ann_date = _estimate_announcement_date(eff_date)

        if added_ticker and added_ticker.strip():
            events.append(SP500ChangeEvent(
                effective_date    = eff_date,
                announcement_date = ann_date,
                ticker            = added_ticker.strip().upper(),
                company_name      = added_name,
                action            = "ADD",
                reason            = reason,
            ))
        if removed_ticker and removed_ticker.strip():
            events.append(SP500ChangeEvent(
                effective_date    = eff_date,
                announcement_date = ann_date,
                ticker            = removed_ticker.strip().upper(),
                company_name      = removed_name,
                action            = "REMOVE",
                reason            = reason,
            ))

    if parse_errors:
        logger.warning(
            "fetch_wikipedia_sp500_changes parsed %d events with %d errors: %s",
            len(events), len(parse_errors), parse_errors[:3],
        )

    logger.info("fetch_wikipedia_sp500_changes: %d events parsed", len(events))
    return events


if __name__ == "__main__":
    import json as _json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    events = fetch_wikipedia_sp500_changes()
    print(f"Fetched {len(events)} S&P 500 change events from Wikipedia")
    for e in events[:10]:
        print(f"  {e.effective_date} [{e.action:<6}] {e.ticker:<6} {e.company_name[:40]:<40} "
              f"(ann ≈ {e.announcement_date}) — {e.reason[:60]}")
