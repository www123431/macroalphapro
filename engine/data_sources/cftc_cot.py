"""
CFTC Commitments of Traders fetcher (Disaggregated + TFF, futures-only).

P3a deliverable (2026-05-07). Free weekly data source from the US CFTC.
Each row reports the long / short / spread positioning of one futures
contract on one Tuesday-of-record.

Two report types, both ingested into ``cftc_cot_weekly`` with a
``report_type`` discriminator:

    'disagg_fut'   Disaggregated, futures-only         ← commodities
        URL: https://www.cftc.gov/files/dea/history/fut_disagg_xls_<year>.zip
        Trader cats: prod_merc / swap / m_money / other_rept / non_rept

    'tff_fut'      Traders in Financial Futures        ← equity / rates / VIX / FX
        URL: https://www.cftc.gov/files/dea/history/fut_fin_xls_<year>.zip
        Trader cats: dealer / asset_mgr / lev_money / other_rept / non_rept

Public methods
--------------
    fetch_year(year, report_type)        → pd.DataFrame
    parse_zip(blob, report_type)         → pd.DataFrame
    upsert_year(year, report_type)       → dict
    upsert_year_both(year)               → dict   (calls both, returns merged counters)
    canonical_columns(report_type)       → list[str]

Idempotency
-----------
Natural key is (contract_market_code, report_date, report_type). Re-running
upsert is safe; CFTC late revisions overwrite the prior write for the same key.
"""
from __future__ import annotations

import datetime
import io
import logging
import urllib.error
import urllib.request
import zipfile
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)

ReportType = Literal["disagg_fut", "tff_fut"]

# ── URLs (verified live 2026-05-07) ──────────────────────────────────────────
_URLS: dict[ReportType, str] = {
    "disagg_fut": "https://www.cftc.gov/files/dea/history/fut_disagg_xls_{year}.zip",
    "tff_fut":    "https://www.cftc.gov/files/dea/history/fut_fin_xls_{year}.zip",
}
_USER_AGENT = "Mozilla/5.0 (research; macro-alpha-pro)"

# ── Column maps (raw → canonical), per report type ───────────────────────────
#
# CFTC has an annoying case inconsistency between the two archives:
#   Disaggregated commodity file: uses ``_ALL`` suffix (e.g. Other_Rept_*_ALL)
#   TFF financial file:           uses ``_All`` suffix (e.g. Other_Rept_*_All)
# The shared metadata block below uses the spellings that ARE shared between
# both files; per-report-type maps override / add the column-suffix variant
# of the trader-category fields that differ.

_SHARED_META = {
    "Market_and_Exchange_Names":      "market_name",
    "As_of_Date_In_Form_YYMMDD":      "as_of_date_yymmdd",
    "Report_Date_as_MM_DD_YYYY":      "report_date",
    "CFTC_Contract_Market_Code":      "contract_market_code",
    "CFTC_Market_Code":               "market_code",
    "CFTC_Region_Code":               "region_code",
    "CFTC_Commodity_Code":            "commodity_code",
    "Open_Interest_All":              "open_interest",
}

# Disaggregated commodity-specific columns (note _ALL all-caps).
_DISAGG_SPECIFIC = {
    "Prod_Merc_Positions_Long_ALL":   "prod_merc_long",
    "Prod_Merc_Positions_Short_ALL":  "prod_merc_short",
    "Swap_Positions_Long_All":        "swap_long",
    "Swap__Positions_Short_All":      "swap_short",
    "Swap__Positions_Spread_All":     "swap_spread",
    "M_Money_Positions_Long_ALL":     "m_money_long",
    "M_Money_Positions_Short_ALL":    "m_money_short",
    "M_Money_Positions_Spread_ALL":   "m_money_spread",
    "Other_Rept_Positions_Long_ALL":  "other_rept_long",
    "Other_Rept_Positions_Short_ALL": "other_rept_short",
    "Other_Rept_Positions_Spread_ALL":"other_rept_spread",
    "Tot_Rept_Positions_Long_All":    "tot_rept_long",
    "Tot_Rept_Positions_Short_All":   "tot_rept_short",
    "NonRept_Positions_Long_All":     "non_rept_long",
    "NonRept_Positions_Short_All":    "non_rept_short",
}

# TFF financial-specific columns (note _All title-case).
_TFF_SPECIFIC = {
    "Dealer_Positions_Long_All":      "dealer_long",
    "Dealer_Positions_Short_All":     "dealer_short",
    "Dealer_Positions_Spread_All":    "dealer_spread",
    "Asset_Mgr_Positions_Long_All":   "asset_mgr_long",
    "Asset_Mgr_Positions_Short_All":  "asset_mgr_short",
    "Asset_Mgr_Positions_Spread_All": "asset_mgr_spread",
    "Lev_Money_Positions_Long_All":   "lev_money_long",
    "Lev_Money_Positions_Short_All":  "lev_money_short",
    "Lev_Money_Positions_Spread_All": "lev_money_spread",
    "Other_Rept_Positions_Long_All":  "other_rept_long",
    "Other_Rept_Positions_Short_All": "other_rept_short",
    "Other_Rept_Positions_Spread_All":"other_rept_spread",
    "Tot_Rept_Positions_Long_All":    "tot_rept_long",
    "Tot_Rept_Positions_Short_All":   "tot_rept_short",
    "NonRept_Positions_Long_All":     "non_rept_long",
    "NonRept_Positions_Short_All":    "non_rept_short",
}


def _column_map(report_type: ReportType) -> dict[str, str]:
    if report_type == "disagg_fut":
        return {**_SHARED_META, **_DISAGG_SPECIFIC}
    if report_type == "tff_fut":
        return {**_SHARED_META, **_TFF_SPECIFIC}
    raise ValueError(f"unknown report_type: {report_type}")


def canonical_columns(report_type: ReportType) -> list[str]:
    """Return canonical column names for a given report type."""
    return list(_column_map(report_type).values())


# ── Fetch + parse ────────────────────────────────────────────────────────────

def _download(year: int, report_type: ReportType, timeout: int = 30) -> bytes:
    url = _URLS[report_type].format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    logger.info("cftc_cot: downloading %s [%s]", url, report_type)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_zip(blob: bytes, report_type: ReportType) -> pd.DataFrame:
    """Parse already-downloaded ZIP bytes into a canonical DataFrame.

    Drops rows where contract_market_code is missing. Coerces dtypes:
      - report_date     → datetime64[ns]
      - integer columns → int64 with NaN-safe coercion
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xls")]
        if not names:
            raise ValueError(f"CFTC {report_type} ZIP missing .xls file")
        with z.open(names[0]) as f:
            xls_bytes = f.read()

    df = pd.read_excel(io.BytesIO(xls_bytes), sheet_name=0)

    col_map = _column_map(report_type)
    keep = {raw: canon for raw, canon in col_map.items() if raw in df.columns}
    missing = [raw for raw in col_map if raw not in df.columns]
    if missing:
        logger.warning(
            "cftc_cot.parse_zip[%s]: %d expected columns missing: %s",
            report_type, len(missing), missing[:6],
        )
    df = df.rename(columns=keep)[list(keep.values())]

    df = df.dropna(subset=["contract_market_code"]).copy()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"])

    int_cols = [c for c in df.columns if c not in {
        "market_name", "as_of_date_yymmdd", "report_date",
        "contract_market_code", "market_code", "region_code", "commodity_code",
    }]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in ("contract_market_code", "market_code", "commodity_code", "region_code"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df.reset_index(drop=True)


def fetch_year(year: int, report_type: ReportType = "disagg_fut",
               timeout: int = 30) -> pd.DataFrame:
    blob = _download(year, report_type, timeout=timeout)
    return parse_zip(blob, report_type)


# ── Persist ──────────────────────────────────────────────────────────────────

# Columns we copy from the parsed DataFrame into the ORM model. Keyed by
# report_type because TFF and Disaggregated have disjoint trader-category
# columns (the other side gets NULL).
_DISAGG_PERSIST_COLS = (
    "prod_merc_long", "prod_merc_short",
    "swap_long", "swap_short", "swap_spread",
    "m_money_long", "m_money_short", "m_money_spread",
)
_TFF_PERSIST_COLS = (
    "dealer_long", "dealer_short", "dealer_spread",
    "asset_mgr_long", "asset_mgr_short", "asset_mgr_spread",
    "lev_money_long", "lev_money_short", "lev_money_spread",
)
_SHARED_PERSIST_COLS = (
    "open_interest",
    "other_rept_long", "other_rept_short", "other_rept_spread",
    "non_rept_long", "non_rept_short",
)


def _build_record(rec: dict, report_type: ReportType) -> dict:
    """Project a parsed row dict into the ORM-shaped kwargs."""
    out: dict[str, Any] = {
        "contract_market_code": rec["contract_market_code"],
        "report_date":          rec["report_date"],
        "report_type":          report_type,
        "market_name":          rec.get("market_name"),
        "market_code":          rec.get("market_code"),
        "commodity_code":       rec.get("commodity_code"),
    }
    for c in _SHARED_PERSIST_COLS:
        out[c] = int(rec.get(c, 0) or 0)

    specific = _DISAGG_PERSIST_COLS if report_type == "disagg_fut" else _TFF_PERSIST_COLS
    for c in specific:
        v = rec.get(c)
        out[c] = int(v) if v is not None else None
    return out


def upsert_year(year: int, report_type: ReportType = "disagg_fut",
                batch_size: int = 500) -> dict[str, Any]:
    """Fetch one year of one report type + bulk-upsert into ``cftc_cot_weekly``.

    Returns counter dict {year, report_type, n_rows_fetched, n_inserted,
    n_updated, elapsed_seconds}. Idempotent.
    """
    import time
    t0 = time.time()
    df = fetch_year(year, report_type=report_type)
    n_fetched = len(df)
    if n_fetched == 0:
        return {"year": year, "report_type": report_type, "n_rows_fetched": 0,
                "n_inserted": 0, "n_updated": 0,
                "elapsed_seconds": round(time.time() - t0, 2)}

    from engine.memory import SessionFactory
    from engine.db_models import CftcCotWeekly

    n_inserted = 0
    n_updated  = 0

    with SessionFactory() as s:
        year_start = datetime.datetime(year, 1, 1)
        year_end   = datetime.datetime(year + 1, 1, 1)
        existing = {
            (code, dt) for code, dt in s.query(
                CftcCotWeekly.contract_market_code,
                CftcCotWeekly.report_date,
            ).filter(
                CftcCotWeekly.report_type == report_type,
                CftcCotWeekly.report_date >= year_start,
                CftcCotWeekly.report_date <  year_end,
            ).all()
        }

        records = df.to_dict(orient="records")
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            for rec in batch:
                rdt = rec["report_date"]
                if not isinstance(rdt, datetime.datetime):
                    rdt = pd.Timestamp(rdt).to_pydatetime()
                rec["report_date"] = rdt
                code = rec["contract_market_code"]
                key  = (code, rdt)
                kwargs = _build_record(rec, report_type)
                if key in existing:
                    s.query(CftcCotWeekly).filter(
                        CftcCotWeekly.contract_market_code == code,
                        CftcCotWeekly.report_date == rdt,
                        CftcCotWeekly.report_type == report_type,
                    ).update({k: v for k, v in kwargs.items()
                              if k not in ("contract_market_code", "report_date", "report_type")})
                    n_updated += 1
                else:
                    s.add(CftcCotWeekly(**kwargs))
                    n_inserted += 1
            s.commit()

    return {
        "year":            year,
        "report_type":     report_type,
        "n_rows_fetched":  n_fetched,
        "n_inserted":      n_inserted,
        "n_updated":       n_updated,
        "elapsed_seconds": round(time.time() - t0, 2),
    }


def upsert_year_both(year: int) -> dict[str, Any]:
    """Convenience: fetch + upsert both Disaggregated and TFF for a year.

    Returns merged counters with per-report-type breakdown.
    """
    out: dict[str, Any] = {"year": year, "by_report_type": {}}
    for rt in ("disagg_fut", "tff_fut"):
        out["by_report_type"][rt] = upsert_year(year, report_type=rt)
    return out


# ── Public read API for downstream signal consumers ──────────────────────────

def get_cot_positioning(
    ticker: str,
    as_of:  datetime.datetime | datetime.date | None = None,
    *,
    max_staleness_days: int = 14,
) -> dict[str, Any] | None:
    """
    Return the most recent COT positioning at-or-before ``as_of`` for the
    futures contract that proxies ``ticker``.

    Returns ``None`` when:
      - The ticker has no CFTC mapping (factor / thematic / international ETFs).
      - No row exists at-or-before as_of (date predates DB coverage).
      - The most recent row is older than ``max_staleness_days`` (CFTC publishes
        every Friday → typical staleness ≤ 7 days; >14 days indicates a fetch lag).

    The returned dict is shape-stable for both report types — TFF-only fields
    (dealer / asset_mgr / lev_money) are present with None values for
    Disaggregated rows, and vice-versa. This lets downstream signal code
    consume one schema regardless of which report the ticker maps to.

    Example
    -------
        >>> pos = get_cot_positioning("SPY")
        >>> pos["asset_mgr_net_pct"], pos["lev_money_net_pct"]
        (0.478, -0.158)   # institutional bullish; hedge funds bearish

    Args
    ----
    ticker : ETF ticker (case-insensitive); resolved via ETF_TO_COT mapping.
    as_of  : reference date. ``None`` = latest available.
    max_staleness_days : reject rows older than this. Default 14 (≈ 2 reports).
    """
    from engine.data_sources.cftc_etf_mapping import get_mapping
    from engine.memory import SessionFactory
    from engine.db_models import CftcCotWeekly

    mapping = get_mapping(ticker)
    if mapping is None:
        return None

    if as_of is None:
        as_of_dt = datetime.datetime.utcnow()
    elif isinstance(as_of, datetime.datetime):
        as_of_dt = as_of
    else:
        as_of_dt = datetime.datetime.combine(as_of, datetime.time.min)

    with SessionFactory() as s:
        row = (
            s.query(CftcCotWeekly)
             .filter(
                 CftcCotWeekly.contract_market_code == mapping.contract_market_code,
                 CftcCotWeekly.report_type          == mapping.report_type,
                 CftcCotWeekly.report_date          <= as_of_dt,
             )
             .order_by(CftcCotWeekly.report_date.desc())
             .first()
        )
        if row is None:
            return None

        n_days_stale = (as_of_dt - row.report_date).days
        if n_days_stale > max_staleness_days:
            return None

        oi = max(1, row.open_interest)   # avoid div-by-zero

        out: dict[str, Any] = {
            "ticker":               ticker.upper(),
            "contract_market_code": row.contract_market_code,
            "market_name":          row.market_name,
            "report_type":          row.report_type,
            "report_date":          row.report_date,
            "n_days_stale":         n_days_stale,
            "open_interest":        row.open_interest,

            # Disaggregated trader categories
            "prod_merc_long":       row.prod_merc_long,
            "prod_merc_short":      row.prod_merc_short,
            "swap_long":            row.swap_long,
            "swap_short":           row.swap_short,
            "m_money_long":         row.m_money_long,
            "m_money_short":        row.m_money_short,

            # TFF trader categories
            "dealer_long":          row.dealer_long,
            "dealer_short":         row.dealer_short,
            "asset_mgr_long":       row.asset_mgr_long,
            "asset_mgr_short":      row.asset_mgr_short,
            "lev_money_long":       row.lev_money_long,
            "lev_money_short":      row.lev_money_short,

            # Shared
            "other_rept_long":      row.other_rept_long,
            "other_rept_short":     row.other_rept_short,
            "non_rept_long":        row.non_rept_long,
            "non_rept_short":       row.non_rept_short,
        }

        # Derived: net positioning as % of OI. NULL when underlying is None.
        def _net_pct(long_v, short_v):
            if long_v is None or short_v is None:
                return None
            return round((long_v - short_v) / oi, 4)

        out["m_money_net_pct"]   = _net_pct(row.m_money_long,   row.m_money_short)
        out["asset_mgr_net_pct"] = _net_pct(row.asset_mgr_long, row.asset_mgr_short)
        out["lev_money_net_pct"] = _net_pct(row.lev_money_long, row.lev_money_short)
        out["dealer_net_pct"]    = _net_pct(row.dealer_long,    row.dealer_short)
        out["prod_merc_net_pct"] = _net_pct(row.prod_merc_long, row.prod_merc_short)

        return out
