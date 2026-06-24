"""engine/line_c/deep_transcripts.py — deep-history CIQ transcript pull (${WRDS_USER_1}).

Mirrors engine.d_pead_plus.transcripts_loader (spec id=74 LOCKS: 'Earnings Calls'
only; ±5 calendar-day rdq alignment, nearest; primary ticker) but drives every
query through ONE reused ${WRDS_USER_1} psycopg2 connection (engine.line_c.wrds_direct),
non-interactive, no auth hammering. Text pull is BATCHED + RESUMABLE.

Outputs (data/line_c/):
  _transcripts_index_2011_2024.parquet  permno, rdq, transcript_id, ticker, call_date, ...
  _transcripts_text_2011_2024.parquet   transcript_id, full_text, total_chars (grows incrementally)
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import pandas as pd

from engine.line_c import wrds_direct

logger = logging.getLogger(__name__)

CACHE = Path("data/line_c")
CACHE.mkdir(parents=True, exist_ok=True)
INDEX_PATH = CACHE / "_transcripts_index_2011_2024.parquet"
TEXT_PATH  = CACHE / "_transcripts_text_2011_2024.parquet"

EVENT_TYPE = "Earnings Calls"
ALIGN_DAYS = 5


def match_index(conn, rdq_panel: pd.DataFrame) -> pd.DataFrame:
    """Match earnings-call transcripts to (permno, ticker, rdq) rows, ±5d nearest."""
    rdq = rdq_panel.copy()
    rdq["ticker"] = rdq["ticker"].astype(str).str.upper()
    rdq["rdq"] = pd.to_datetime(rdq["rdq"])
    tickers = sorted(rdq["ticker"].dropna().unique().tolist())

    # 1) ticker -> companyid (primary ticker only)
    link = pd.read_sql(
        "SELECT DISTINCT companyid, UPPER(ticker) AS ticker, companyname "
        "FROM ciq_common.wrds_ticker WHERE primaryflag = 1 AND UPPER(ticker) IN %(t)s",
        conn, params={"t": tuple(tickers)},
    )
    link = link.dropna(subset=["companyid"])
    link["companyid"] = link["companyid"].astype(int)
    companyids = sorted(link["companyid"].unique().tolist())
    logger.info("ticker->companyid: %d tickers -> %d companyids", len(tickers), len(companyids))
    if not companyids:
        return pd.DataFrame()

    # 2) earnings-call metadata for those companyids over the window
    wstart = (rdq["rdq"].min() - pd.Timedelta(days=ALIGN_DAYS)).date().isoformat()
    wend   = (rdq["rdq"].max() + pd.Timedelta(days=ALIGN_DAYS)).date().isoformat()
    calls = pd.read_sql(
        "SELECT transcriptid, companyid, companyname, mostimportantdateutc AS call_date "
        "FROM ciq_transcripts.wrds_transcript_detail "
        "WHERE keydeveventtypename = %(ev)s AND companyid IN %(c)s "
        "AND mostimportantdateutc BETWEEN %(s)s AND %(e)s",
        conn, params={"ev": EVENT_TYPE, "c": tuple(companyids), "s": wstart, "e": wend},
        parse_dates=["call_date"],
    )
    calls["companyid"] = calls["companyid"].astype(int)
    calls["transcriptid"] = calls["transcriptid"].astype(int)
    logger.info("earnings calls fetched: %d", len(calls))
    if calls.empty:
        return pd.DataFrame()

    # 3) join companyid onto rdq panel, 4) merge_asof nearest within ±5 days, by companyid
    rdq_l = rdq.merge(link[["companyid", "ticker"]], on="ticker", how="inner")
    rdq_l["companyid"] = rdq_l["companyid"].astype(int)
    rdq_l = rdq_l.sort_values("rdq").reset_index(drop=True)
    calls_s = calls.sort_values("call_date").reset_index(drop=True)

    merged = pd.merge_asof(
        rdq_l, calls_s, left_on="rdq", right_on="call_date",
        by="companyid", direction="nearest", tolerance=pd.Timedelta(days=ALIGN_DAYS),
    )
    merged = merged[merged["transcriptid"].notna()].copy()
    merged["transcript_id"] = merged["transcriptid"].astype(int)
    merged["permno"] = merged["permno"].astype(int)
    merged["company_id"] = merged["companyid"].astype(int)
    merged["date_diff_days"] = (merged["call_date"] - merged["rdq"]).dt.days
    merged["call_date"] = merged["call_date"].dt.date
    merged["rdq"] = merged["rdq"].dt.date
    merged["company_name"] = merged.get("companyname_y", merged.get("companyname", "")).astype(str)

    out = merged[["permno", "rdq", "transcript_id", "company_id", "company_name",
                  "ticker", "call_date", "date_diff_days"]].copy()
    # one transcript per firm-quarter: keep the closest call to rdq
    out["abs_diff"] = out["date_diff_days"].abs()
    out = out.sort_values("abs_diff").drop_duplicates(["permno", "rdq"], keep="first")
    out = out.drop(columns="abs_diff").sort_values(["rdq", "ticker"]).reset_index(drop=True)
    logger.info("matched %d firm-quarters to transcripts (of %d panel rows)", len(out), len(rdq_panel))
    return out


def fetch_text_batched(conn, transcript_ids: list[int], *, batch_size=800, flush_every=5) -> pd.DataFrame:
    """Pull full transcript text in resumable batches; skip already-cached ids."""
    ids = sorted(set(int(i) for i in transcript_ids))
    done: set[int] = set()
    if TEXT_PATH.exists():
        cached = pd.read_parquet(TEXT_PATH, columns=["transcript_id"])
        done = set(cached["transcript_id"].astype(int).tolist())
    todo = [i for i in ids if i not in done]
    logger.info("text pull: %d total, %d cached, %d to fetch", len(ids), len(done), len(todo))
    if not todo:
        return pd.read_parquet(TEXT_PATH)

    pending: list[pd.DataFrame] = []
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        sub = todo[bi * batch_size:(bi + 1) * batch_size]
        t0 = time.time()
        df = pd.read_sql(
            "SELECT transcriptid AS transcript_id, "
            "STRING_AGG(componenttext, ' ' ORDER BY componentorder) AS full_text, "
            "SUM(LENGTH(componenttext)) AS total_chars "
            "FROM ciq_transcripts.ciqtranscriptcomponent "
            "WHERE transcriptid IN %(ids)s GROUP BY transcriptid",
            conn, params={"ids": tuple(sub)},
        )
        df["transcript_id"] = df["transcript_id"].astype(int)
        df["total_chars"] = df["total_chars"].fillna(0).astype(int)
        pending.append(df)
        logger.info("  batch %d/%d: %d rows in %.1fs", bi + 1, n_batches, len(df), time.time() - t0)
        if (bi + 1) % flush_every == 0 or bi == n_batches - 1:
            _append_text(pending)
            pending = []
    return pd.read_parquet(TEXT_PATH)


def _append_text(frames: list[pd.DataFrame]) -> None:
    if not frames:
        return
    new = pd.concat(frames, ignore_index=True)
    if TEXT_PATH.exists():
        old = pd.read_parquet(TEXT_PATH)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates("transcript_id", keep="last")
    new.to_parquet(TEXT_PATH)
    logger.info("  flushed -> %d total text rows cached", len(new))


def build_index():
    import warnings, json
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    sue = pd.read_parquet(CACHE / "_sue_panel_2011_2024.parquet")
    rdq_panel = sue[["permno", "ticker", "rdq"]].drop_duplicates()
    conn = wrds_direct.connect("${WRDS_USER_1}")
    try:
        idx = match_index(conn, rdq_panel)
        idx.to_parquet(INDEX_PATH)
        rdq = pd.to_datetime(idx["rdq"])
        meta = {
            "n_matched": int(len(idx)),
            "n_panel_rows": int(len(rdq_panel)),
            "match_rate": round(len(idx) / max(len(rdq_panel), 1), 3),
            "rdq_min": str(rdq.min().date()), "rdq_max": str(rdq.max().date()),
            "n_permno": int(idx["permno"].nunique()),
            "median_abs_date_diff": int(idx["date_diff_days"].abs().median()),
        }
        print(json.dumps(meta, indent=2))
        print("\nby year:")
        print(idx.assign(yr=rdq.dt.year).groupby("yr").size().to_string())
    finally:
        conn.close()


def build_text():
    """Resumable full-text pull for all matched transcripts (ONE ${WRDS_USER_1} conn)."""
    import warnings, json
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    idx = pd.read_parquet(INDEX_PATH)
    ids = idx["transcript_id"].astype(int).unique().tolist()
    conn = wrds_direct.connect("${WRDS_USER_1}")
    try:
        txt = fetch_text_batched(conn, ids)
        print(json.dumps({
            "n_text_rows": int(len(txt)),
            "median_chars": int(txt["total_chars"].median()),
            "total_gb": round(txt["total_chars"].sum() / 1e9, 2),
        }, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    if "--text" in sys.argv:
        build_text()
    else:
        build_index()
