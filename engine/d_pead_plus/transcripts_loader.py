"""
engine/d_pead_plus/transcripts_loader.py — WRDS CIQ earnings call pipeline.

Spec id=74 §2.4 + §2.1 locks:
  - Source: ciq_transcripts.wrds_transcript_detail + ciqtranscriptcomponent
  - Event filter: keydeveventtypename = 'Earnings Calls' ONLY (LOCK #2)
  - Date alignment: ±5 calendar days of rdq; closest if >1 (LOCK #1)
  - CIQ companyid → ticker via ciq_common.wrds_ticker (primaryflag=1)
  - Ticker → CRSP permno via crsp.msenames (point-in-time)
  - Universe filter: top-1500 mcap NYSE/NASDAQ US common (LOCK #3)

DOCTRINE: this module is part of FEATURE EXTRACTION context, not decision.
LLM is NOT called here; this only prepares data for engine.d_pead_plus.llm_extractor.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Spec id=74 §2.4 LOCKED
EVENT_TYPE_LOCKED:         str = "Earnings Calls"
DATE_ALIGNMENT_WINDOW_DAYS: int = 5

# Cache path
CACHE_DIR = Path("data/d_pead_plus")
TRANSCRIPTS_INDEX_PATH = CACHE_DIR / "_transcripts_index.parquet"
TRANSCRIPTS_TEXT_PATH  = CACHE_DIR / "_transcripts_text.parquet"


@dataclass(frozen=True)
class TranscriptRecord:
    """One earnings call transcript matched to firm-quarter."""
    permno:           int                  # CRSP permno (decision-layer identifier)
    rdq:              datetime.date        # Compustat earnings announcement date
    transcript_id:    int                  # CIQ transcript id
    company_id:       int                  # CIQ companyid
    company_name:     str
    ticker:           str
    call_date:        datetime.date        # mostimportantdateutc
    date_diff_days:   int                  # call_date - rdq
    n_components:     int
    total_chars:      int


def fetch_transcript_index(
    rdq_panel: pd.DataFrame,
    *,
    conn = None,
) -> pd.DataFrame:
    """Match earnings calls to firm-quarter (permno, rdq) pairs.

    Args:
        rdq_panel: DataFrame with columns (permno, ticker, rdq) for our universe
        conn: optional pre-opened WRDS connection

    Returns DataFrame with columns:
        permno, rdq, transcript_id, company_id, company_name, ticker,
        call_date, date_diff_days, n_components, total_chars

    Algorithm:
        1. For each (ticker, rdq) row:
            a. Find CIQ companyid via ciq_common.wrds_ticker (primaryflag=1)
            b. Find earnings call within ±5 days of rdq
            c. If multiple matches: take closest to rdq
            d. Record transcript metadata
    """
    if rdq_panel.empty:
        return pd.DataFrame()

    own_conn = conn is None
    if conn is None:
        from engine.universe_singlename.crsp_loader import _open_wrds_connection
        conn = _open_wrds_connection()

    try:
        # Build ticker list for batch lookup
        tickers = rdq_panel["ticker"].dropna().astype(str).str.upper().unique().tolist()
        if not tickers:
            logger.warning("fetch_transcript_index: no tickers in rdq_panel")
            return pd.DataFrame()
        ticker_csv = ",".join(f"'{t}'" for t in tickers)

        # Step 1: ticker → companyid lookup (primary ticker only)
        sql_link = f"""
        SELECT DISTINCT companyid, ticker, companyname
        FROM ciq_common.wrds_ticker
        WHERE primaryflag = 1
          AND UPPER(ticker) IN ({ticker_csv})
        """
        df_link = conn.raw_sql(sql_link)
        df_link["companyid"] = df_link["companyid"].astype("Int64")
        df_link["ticker"]    = df_link["ticker"].astype(str).str.upper()
        logger.info("ticker→companyid linked: %d / %d tickers found",
                    df_link["companyid"].notna().sum(), len(tickers))

        if df_link.empty:
            return pd.DataFrame()

        # Build companyid list
        companyids = df_link["companyid"].dropna().astype(int).unique().tolist()
        companyid_csv = ",".join(str(c) for c in companyids)

        # Step 2: pull earnings call metadata for these companyids over window
        window_start = rdq_panel["rdq"].min()
        window_end   = rdq_panel["rdq"].max()
        sql_calls = f"""
        SELECT t.transcriptid, t.companyid, t.companyname, t.mostimportantdateutc AS call_date
        FROM ciq_transcripts.wrds_transcript_detail t
        WHERE t.keydeveventtypename = '{EVENT_TYPE_LOCKED}'
          AND t.companyid IN ({companyid_csv})
          AND t.mostimportantdateutc BETWEEN
              '{(window_start - datetime.timedelta(days=DATE_ALIGNMENT_WINDOW_DAYS)).isoformat()}' AND
              '{(window_end + datetime.timedelta(days=DATE_ALIGNMENT_WINDOW_DAYS)).isoformat()}'
        """
        df_calls = conn.raw_sql(sql_calls)
        df_calls["companyid"]  = df_calls["companyid"].astype(int)
        df_calls["transcriptid"] = df_calls["transcriptid"].astype(int)
        df_calls["call_date"]  = pd.to_datetime(df_calls["call_date"]).dt.date
        df_calls["n_components"] = 0  # diagnostic deferred to text-fetch stage
        logger.info("earnings calls fetched: %d", len(df_calls))

        # Step 3: join rdq_panel with companyid mapping
        rdq = rdq_panel.copy()
        rdq["ticker"] = rdq["ticker"].astype(str).str.upper()
        rdq["rdq"]    = pd.to_datetime(rdq["rdq"]).dt.date
        rdq_linked = rdq.merge(df_link[["companyid", "ticker"]], on="ticker", how="inner")
        rdq_linked["companyid"] = rdq_linked["companyid"].astype(int)

        # Step 4: vectorized merge_asof match within ±5 days
        # merge_asof requires both sides sorted by the on key (date), with by groups
        rdq_linked["rdq_ts"] = pd.to_datetime(rdq_linked["rdq"])
        df_calls["call_date_ts"] = pd.to_datetime(df_calls["call_date"])
        rdq_linked = rdq_linked.sort_values("rdq_ts").reset_index(drop=True)
        df_calls_sorted = df_calls.sort_values("call_date_ts").reset_index(drop=True)

        # merge_asof per companyid group with tolerance
        tolerance = pd.Timedelta(days=DATE_ALIGNMENT_WINDOW_DAYS)
        merged_left = pd.merge_asof(
            rdq_linked, df_calls_sorted,
            left_on="rdq_ts", right_on="call_date_ts",
            by="companyid", direction="nearest", tolerance=tolerance,
        )
        # Drop unmatched (NaN transcriptid)
        merged_left = merged_left[merged_left["transcriptid"].notna()].copy()

        merged_left["date_diff_days"] = (merged_left["call_date_ts"] - merged_left["rdq_ts"]).dt.days
        merged_left["transcript_id"]  = merged_left["transcriptid"].astype(int)
        merged_left["permno"]         = merged_left["permno"].astype(int)
        merged_left["company_id"]     = merged_left["companyid"].astype(int)
        merged_left["call_date"]      = merged_left["call_date_ts"].dt.date
        merged_left["rdq"]            = merged_left["rdq_ts"].dt.date
        # CIQ has "companyname" (no underscore) in wrds_transcript_detail
        if "companyname" in merged_left.columns:
            merged_left["company_name"] = merged_left["companyname"].astype(str)
        elif "company_name" not in merged_left.columns:
            merged_left["company_name"] = ""

        result = merged_left[[
            "permno", "rdq", "transcript_id", "company_id", "company_name",
            "ticker", "call_date", "date_diff_days", "n_components",
        ]].copy()
        result["n_components"] = result["n_components"].fillna(0).astype(int)
        logger.info("Matched %d firm-quarters to earnings calls (of %d possible)",
                    len(result), len(rdq_linked))
        return result
    finally:
        if own_conn:
            conn.close()


def fetch_transcript_text(transcript_ids: list[int], *, conn = None) -> pd.DataFrame:
    """Pull full transcript text for given transcript_ids.

    Returns DataFrame: transcript_id, full_text, total_chars
    """
    if not transcript_ids:
        return pd.DataFrame()

    own_conn = conn is None
    if conn is None:
        from engine.universe_singlename.crsp_loader import _open_wrds_connection
        conn = _open_wrds_connection()

    try:
        ids_csv = ",".join(str(int(i)) for i in transcript_ids)
        sql = f"""
        SELECT transcriptid,
               STRING_AGG(componenttext, ' ' ORDER BY componentorder) AS full_text,
               SUM(LENGTH(componenttext)) AS total_chars
        FROM ciq_transcripts.ciqtranscriptcomponent
        WHERE transcriptid IN ({ids_csv})
        GROUP BY transcriptid
        """
        df = conn.raw_sql(sql)
        df["transcript_id"] = df["transcriptid"].astype(int)
        df = df[["transcript_id", "full_text", "total_chars"]]
        df["total_chars"] = df["total_chars"].astype(int)
        return df
    finally:
        if own_conn:
            conn.close()


def cache_transcripts(
    index_df: pd.DataFrame,
    text_df:  pd.DataFrame,
) -> None:
    """Save transcript index + text to parquet cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(TRANSCRIPTS_INDEX_PATH)
    text_df.to_parquet(TRANSCRIPTS_TEXT_PATH)
    logger.info("Cached %d index rows + %d text rows to %s",
                len(index_df), len(text_df), CACHE_DIR)


def load_cached_transcripts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached transcript index + text from parquet."""
    if not TRANSCRIPTS_INDEX_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    idx = pd.read_parquet(TRANSCRIPTS_INDEX_PATH)
    txt = pd.read_parquet(TRANSCRIPTS_TEXT_PATH) if TRANSCRIPTS_TEXT_PATH.exists() else pd.DataFrame()
    return idx, txt


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--rdq-panel", required=True, help="path to firm-quarter rdq panel parquet")
    p.add_argument("--cache", action="store_true", help="save results to parquet cache")
    args = p.parse_args()
    rdq_panel = pd.read_parquet(args.rdq_panel)
    print(f"Loaded rdq_panel: {len(rdq_panel)} rows")
    idx = fetch_transcript_index(rdq_panel)
    print(f"Index shape: {idx.shape}")
    if not idx.empty:
        text_df = fetch_transcript_text(idx["transcript_id"].tolist())
        print(f"Text shape: {text_df.shape}")
        if args.cache:
            cache_transcripts(idx, text_df)
            print(f"Cached.")
