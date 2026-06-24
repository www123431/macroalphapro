"""scripts/fetch_optionm_full_skew_surface.py — pull full OptionMetrics
IVOL skew surface from WRDS optionm.vsurfd for Path C tail-hedge work.

Per user direction 2026-05-31: don't stub the data — fetch the actual
10/25/50/75/90 delta × 30/60/91/182/365 days surface so put-spread
backtest can be done rigorously.

WRDS table: optionm.volatility_surface_view (unified across years).
Column names (verified via describe_table):
  securityid    standardized security id (was 'secid' in older docs)
  date          trade date
  days          days to expiration (30/60/91/122/152/182/273/365/547/730)
  callput       'C' or 'P'
  delta         -90/-75/-50/-25/-10 (puts) or 10/25/50/75/90 (calls)
  impliedvol    annualized IVOL
  strike        strike at this delta+IV
  premium       option price
  dispersion    surface fit dispersion

Coverage strategy: keep this script focused on a SHORT secid list of
index/big-ETF underlyings (SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VXX);
fetching the whole CRSP universe would explode the row count.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_skew")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "cache" / "_optionm_full_skew.parquet"

# Active WRDS user (per [[feedback-wrds-care-and-probe-pattern-2026-05-30]];
# ${WRDS_USER_2} active in pgpass; ${WRDS_USER_1} stale in older secrets)
_WRDS_USER = "${WRDS_USER_2}"

# Underlying secids we care about for hedging / vol-carry work.
# Looked up via _optionm_secid_permno.parquet + manual cross-ref.
DEFAULT_SECIDS = {
    # CORRECTED 2026-05-31: prior cache fetched secid=8957 = penny stock
    # (permno 89911), NOT SPY. Verified via optionm.securd1 lookup.
    108105: "SPX",     # S&P 500 INDEX (CBOE) — preferred tail hedge underlying
                       #   (European exercise + cash-settled + most liquid)
    100155: "SPY",     # SPDR TRUST SERIES 1 — equity ETF on S&P 500
    # Other major index underlyings to add on demand:
    # 102265: "QQQ",   # Nasdaq 100 ETF
    # 102456: "IWM",   # Russell 2000 ETF
}

# Deltas to pull — full skew at standard OM grid
DELTAS = [-90, -75, -50, -25, -10, 10, 25, 50, 75, 90]

# Maturities (days) — focus on monthly + quarterly + half-year + annual
DAYS = [30, 60, 91, 182, 365]

START_DATE = "2014-01-01"
END_DATE = "2024-03-31"


def _connect():
    import wrds
    logger.info(f"opening WRDS connection (user={_WRDS_USER})")
    return wrds.Connection(wrds_username=_WRDS_USER)


def verify_secids(conn, secids: list[int]) -> dict[int, dict]:
    """Sanity check: look up each secid in optionm.securd1 and return
    {secid: {ticker, issuer, issue_type}}. ABORTS fetch if any secid
    doesn't exist or has a sus ticker mismatch.

    Lesson from 2026-05-31: hard-coding secid=8957 as "SPY" was wrong
    — actually a penny stock (permno 89911). Always verify via
    optionm.securd1 before fetch.
    """
    if not secids:
        return {}
    secid_list = "(" + ",".join(str(s) for s in secids) + ")"
    q = f"""
    SELECT secid, ticker, issuer, issue_type
    FROM optionm.securd1
    WHERE secid IN {secid_list}
    """
    df = conn.raw_sql(q)
    found: dict[int, dict] = {}
    for _, row in df.iterrows():
        found[int(row["secid"])] = {
            "ticker":     row["ticker"],
            "issuer":     row["issuer"],
            "issue_type": row.get("issue_type"),
        }
    return found


def _year_range() -> list[int]:
    return list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))


def probe_row_count(conn, secid: int) -> int:
    """Sum COUNT(*) across yearly tables vsurfdYYYY. Loop because the
    unified volatility_surface_view UNIONs in optionm_europe which
    ${WRDS_USER_2} lacks permission for."""
    delta_list_sql = "(" + ",".join(str(d) for d in DELTAS) + ")"
    days_list_sql = "(" + ",".join(str(d) for d in DAYS) + ")"
    total = 0
    t0 = time.time()
    for year in _year_range():
        tbl = f"optionm.vsurfd{year}"
        q = f"""
        SELECT COUNT(*) AS n FROM {tbl}
        WHERE secid = {secid}
          AND date BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND days IN {days_list_sql}
          AND delta IN {delta_list_sql}
        """
        try:
            total += int(conn.raw_sql(q).iloc[0, 0])
        except Exception as exc:
            logger.warning(f"  {tbl} probe failed: {exc}")
    elapsed = time.time() - t0
    logger.info(f"  probe secid={secid}: {total:,} rows total ({elapsed:.1f}s)")
    return total


def fetch_secid(conn, secid: int) -> pd.DataFrame:
    """Pull the full skew surface across yearly tables."""
    delta_list_sql = "(" + ",".join(str(d) for d in DELTAS) + ")"
    days_list_sql = "(" + ",".join(str(d) for d in DAYS) + ")"
    frames = []
    t0 = time.time()
    for year in _year_range():
        tbl = f"optionm.vsurfd{year}"
        q = f"""
        SELECT secid, date, days, cp_flag, delta,
               impl_volatility, impl_strike, impl_premium
        FROM {tbl}
        WHERE secid = {secid}
          AND date BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND days IN {days_list_sql}
          AND delta IN {delta_list_sql}
        ORDER BY date, days, cp_flag, delta
        """
        try:
            yr_df = conn.raw_sql(q)
            frames.append(yr_df)
            logger.info(f"    {tbl}: {len(yr_df):,} rows")
        except Exception as exc:
            logger.warning(f"  {tbl} fetch failed: {exc}")
    elapsed = time.time() - t0
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    logger.info(f"  pulled secid={secid} total: {len(df):,} rows ({elapsed:.1f}s)")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-only", action="store_true",
                     help="only run COUNT(*) probe, no data fetch")
    p.add_argument("--secids", type=int, nargs="+", default=list(DEFAULT_SECIDS.keys()),
                     help="secids to fetch (default: SPY only)")
    args = p.parse_args()

    print("=" * 88)
    print(" OptionMetrics FULL skew surface fetch")
    print("=" * 88)
    print(f"  secids:     {args.secids}")
    print(f"  deltas:     {DELTAS}")
    print(f"  maturities: {DAYS} days")
    print(f"  window:     {START_DATE} → {END_DATE}")

    try:
        conn = _connect()
    except Exception as exc:
        logger.exception("WRDS connection failed")
        print(f"\n  CONNECTION ERROR: {exc}")
        return 1

    print(f"\n[verify_secids] (per 2026-05-31 penny-stock lesson)")
    verified = verify_secids(conn, args.secids)
    for sid in args.secids:
        meta = verified.get(sid)
        expected_name = DEFAULT_SECIDS.get(sid, "<unknown>")
        if meta is None:
            print(f"  secid={sid}: NOT FOUND in optionm.securd1 — ABORT")
            return 1
        actual_ticker = (meta["ticker"] or "").strip()
        actual_issuer = (meta["issuer"] or "").strip()
        print(f"  secid={sid}  ticker={actual_ticker!r}  "
              f"issuer={actual_issuer!r}  issue_type={meta['issue_type']!r}")
        # If we declared an expected ticker in DEFAULT_SECIDS, assert it matches
        if expected_name != "<unknown>" and actual_ticker.upper() != expected_name.upper():
            print(f"    TICKER MISMATCH: expected {expected_name!r} got "
                  f"{actual_ticker!r} — ABORT (check DEFAULT_SECIDS)")
            return 1

    print(f"\n[probe]")
    total_rows = 0
    for secid in args.secids:
        total_rows += probe_row_count(conn, secid)
    print(f"  total expected rows: {total_rows:,}")

    if args.probe_only:
        print("\n  --probe-only: exiting without fetch")
        return 0

    print(f"\n[fetch]")
    frames = []
    for secid in args.secids:
        try:
            frames.append(fetch_secid(conn, secid))
        except Exception as exc:
            logger.exception(f"fetch failed for secid={secid}")

    if not frames:
        print("\n  no data fetched — aborting")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["secid"] = combined["secid"].astype(int)
    combined["days"] = combined["days"].astype(int)
    combined["delta"] = combined["delta"].astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH)
    print(f"\n[saved] {OUT_PATH}")
    print(f"  rows:   {len(combined):,}")
    print(f"  range:  {combined['date'].min().date()} → "
          f"{combined['date'].max().date()}")
    print(f"  deltas: {sorted(combined['delta'].unique())}")
    print(f"  days:   {sorted(combined['days'].unique())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
