"""
engine/universe_singlename/constituents_loader.py — vintage S&P 500 constituents.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.1

Wave A: 3 historical SP500 sources (Wikipedia + GitHub + proxy) with sensitivity
        check per pre-Wave-A audit Issue #2. Disk-cached.
Wave B: CRSP authoritative (post-WRDS approval, separate module).

Honest Wave A limitations:
  - Wikipedia archive: covers ~2010+ adds/drops well; pre-2010 incomplete
  - fja05680/sp500: maintained by community; quality variable
  - current-mktcap-top-500 proxy: today's top 500 mkt cap, applied historically
    (PURE survivorship — included as worst-case lower bound of sensitivity)

If 3 sources agree within ±0.10 Sharpe → preliminary verdict reasonable.
If diverge > ±0.10 → universe-source-sensitive flag in PRELIMINARY verdict.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.1 + amendments 2026-05-09 (smoke-driven, GitHub dropped) +
# 2026-05-10 (W-B-2: crsp_vintage skeleton, real path activated post-WRDS approval).
UNIVERSE_SOURCES_LOCKED: tuple[str, ...] = (
    "wikipedia_archive",      # Best-effort reconstruction from current + historical changes
    "mktcap_top500_proxy",    # Today's top 500 — PURE SURVIVORSHIP, used as Wave A primary
                              # (honest acknowledgement: Wave A is preliminary, NOT publishable;
                              #  Wave B CRSP is the academic-grade vintage replacement)
    "crsp_vintage",           # Wave B authoritative source — CRSP S&P 500 historical
                              # constituents (1925+, fully vintage point-in-time, includes
                              # delisted/inactive). Skeleton + mock until WRDS approved.
    "russell2000_proxy",      # 2026-05-12 Path J: CRSP market-cap-rank synthesis
                              # (rank 1001-3000 = Russell 2000 proxy). Pure CRSP-based,
                              # no external Russell index license needed. Survivor-bias
                              # free via point-in-time msf rank.
)

# Cache directory (disk-backed)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "universe_singlename"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_WIKIPEDIA_CACHE = _CACHE_DIR / "_wikipedia_sp500_history.parquet"
_GITHUB_CACHE = _CACHE_DIR / "_github_fja05680_sp500.parquet"
_PROXY_CACHE = _CACHE_DIR / "_mktcap_top500_proxy.parquet"


@dataclasses.dataclass(frozen=True)
class SP500ConstituentsResult:
    """Result of loading constituents at a specific date."""
    as_of:           datetime.date
    source:          str
    tickers:         list[str]
    n_constituents:  int
    metadata:        dict


def _fetch_wikipedia_sp500_history() -> pd.DataFrame:
    """Fetch Wikipedia 'List of S&P 500 companies' page + history of changes.

    Returns DataFrame with columns: ticker, action (add/remove), date.
    Cached to disk; returns cached if exists.
    """
    if _WIKIPEDIA_CACHE.exists():
        try:
            return pd.read_parquet(_WIKIPEDIA_CACHE)
        except Exception as exc:
            logger.warning("wikipedia cache load failed: %s — refetching", exc)

    # Wikipedia URL: en.wikipedia.org/wiki/List_of_S%26P_500_companies
    # 2 tables: current constituents + selected changes
    # Use pandas.read_html
    try:
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(html)
    except Exception as exc:
        logger.error("wikipedia fetch failed: %s", exc)
        return pd.DataFrame(columns=["ticker", "action", "date"])

    # Table 0: current S&P 500 constituents
    # Table 1: history of changes (added / removed)
    if len(tables) < 2:
        logger.error("wikipedia returned < 2 tables")
        return pd.DataFrame(columns=["ticker", "action", "date"])

    current = tables[0]
    changes = tables[1]

    # Normalize current constituents into "added at very early date" rows
    # Use 1990-01-01 as proxy for "always in universe" baseline
    current_tickers = current["Symbol"].tolist() if "Symbol" in current.columns else current.iloc[:, 0].tolist()
    current_rows = pd.DataFrame({
        "ticker": current_tickers,
        "action": "current",
        "date": pd.Timestamp("1990-01-01"),
    })

    # Normalize changes table — varies in column structure; handle defensively
    change_rows = []
    if not changes.empty:
        # Common Wikipedia format: multi-index with "Added"/"Removed" + Date column
        try:
            # Flatten multi-index columns if present
            if isinstance(changes.columns, pd.MultiIndex):
                changes.columns = ["_".join([str(c) for c in col]).strip() for col in changes.columns]
            for _, row in changes.iterrows():
                # Try to extract date
                date_val = None
                for col in changes.columns:
                    if "Date" in col or "date" in col:
                        try:
                            date_val = pd.to_datetime(row[col], errors="coerce")
                            if pd.notna(date_val):
                                break
                        except Exception:
                            continue
                if date_val is None or pd.isna(date_val):
                    continue
                # Try to extract added + removed tickers
                for col in changes.columns:
                    if "Added" in col and ("Ticker" in col or "Symbol" in col):
                        ticker = row[col]
                        if isinstance(ticker, str) and len(ticker) <= 10:
                            change_rows.append({"ticker": ticker, "action": "added", "date": date_val})
                    if "Removed" in col and ("Ticker" in col or "Symbol" in col):
                        ticker = row[col]
                        if isinstance(ticker, str) and len(ticker) <= 10:
                            change_rows.append({"ticker": ticker, "action": "removed", "date": date_val})
        except Exception as exc:
            logger.warning("wikipedia changes parse partial fail: %s", exc)

    df = pd.concat([current_rows, pd.DataFrame(change_rows)], ignore_index=True) if change_rows else current_rows
    try:
        df.to_parquet(_WIKIPEDIA_CACHE)
    except Exception as exc:
        logger.warning("wikipedia cache persist failed: %s", exc)
    return df


def _fetch_github_fja05680() -> pd.DataFrame:
    """Fetch fja05680/sp500 GitHub dataset.

    Repo: https://github.com/fja05680/sp500
    Provides historical SP500 by month going back ~2000.
    Format: CSV with date + ticker columns.
    """
    if _GITHUB_CACHE.exists():
        try:
            return pd.read_parquet(_GITHUB_CACHE)
        except Exception as exc:
            logger.warning("github cache load failed: %s — refetching", exc)

    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes(09-09-2024).csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            csv = resp.read().decode("utf-8")
        from io import StringIO
        df = pd.read_csv(StringIO(csv))
    except Exception as exc:
        logger.error("github fja05680 fetch failed: %s", exc)
        return pd.DataFrame(columns=["date", "tickers"])

    # Format: typically "date" + "tickers" comma-separated
    if "date" not in df.columns and len(df.columns) >= 2:
        df.columns = ["date"] + list(df.columns[1:])
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    except Exception as exc:
        logger.warning("github date parse fail: %s", exc)

    try:
        df.to_parquet(_GITHUB_CACHE)
    except Exception as exc:
        logger.warning("github cache persist failed: %s", exc)
    return df


def _fetch_mktcap_top500_proxy() -> list[str]:
    """Worst-case-survivorship proxy: today's S&P 500 from yfinance.

    This is PURE SURVIVORSHIP (only includes today's survivors), used as the
    LOWER bound of universe sensitivity. If Wave A Sharpe with this proxy is
    similar to other 2 sources → strategy robust to survivorship; if very
    different → universe-source-sensitive flag.
    """
    if _PROXY_CACHE.exists():
        try:
            df = pd.read_parquet(_PROXY_CACHE)
            return df["ticker"].tolist()
        except Exception as exc:
            logger.warning("proxy cache load failed: %s — refetching", exc)

    # Fall back to current Wikipedia table (which lists today's S&P 500)
    wiki_df = _fetch_wikipedia_sp500_history()
    current = wiki_df[wiki_df["action"] == "current"]
    tickers = current["ticker"].tolist()
    if not tickers:
        return []

    df = pd.DataFrame({"ticker": tickers})
    try:
        df.to_parquet(_PROXY_CACHE)
    except Exception as exc:
        logger.warning("proxy cache persist failed: %s", exc)
    return tickers


def load_sp500_constituents_at_date(
    as_of:  datetime.date,
    source: str = "wikipedia_archive",
) -> SP500ConstituentsResult:
    """Return SP500 constituents at a specific date from a specific source.

    Per spec Wave A Issue #2: caller should run all 3 sources separately for
    sensitivity check, NOT pick one and trust.

    Args:
        as_of:  date for which to query constituents
        source: one of UNIVERSE_SOURCES_LOCKED

    Returns:
        SP500ConstituentsResult with tickers list + metadata.
    """
    if source not in UNIVERSE_SOURCES_LOCKED:
        raise ValueError(f"source {source!r} not in {UNIVERSE_SOURCES_LOCKED}")
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")

    if source == "wikipedia_archive":
        wiki_df = _fetch_wikipedia_sp500_history()
        if wiki_df.empty:
            return SP500ConstituentsResult(as_of=as_of, source=source, tickers=[],
                                            n_constituents=0, metadata={"note": "wikipedia fetch failed"})
        # Reconstruct constituents at as_of (correct logic, fix 2026-05-09):
        #   universe(at as_of) = today_set
        #     PLUS  {removed AFTER as_of}  (still in at as_of, removed later)
        #     MINUS {added   AFTER as_of}  (not in at as_of, added later)
        current_tickers = set(wiki_df[wiki_df["action"] == "current"]["ticker"].tolist())
        added_after = wiki_df[(wiki_df["action"] == "added") & (wiki_df["date"] > pd.Timestamp(as_of))]
        removed_after = wiki_df[(wiki_df["action"] == "removed") & (wiki_df["date"] > pd.Timestamp(as_of))]
        # Companies added after as_of: were NOT in universe at as_of → remove from set
        for _, row in added_after.iterrows():
            current_tickers.discard(row["ticker"])
        # Companies removed after as_of: WERE in universe at as_of → add to set
        for _, row in removed_after.iterrows():
            current_tickers.add(row["ticker"])
        tickers = sorted(current_tickers)
        return SP500ConstituentsResult(
            as_of=as_of, source=source, tickers=tickers,
            n_constituents=len(tickers),
            metadata={
                "reconstruction_method": "today_set ∪ removed_after − added_after",
                "wikipedia_total_changes_rows": int(((wiki_df["action"] == "added") | (wiki_df["action"] == "removed")).sum()),
                "honest_caveat": "Wikipedia 'selected changes' table is incomplete pre-2010; coverage best for 2010+",
            },
        )

    elif source == "mktcap_top500_proxy":
        tickers = sorted(_fetch_mktcap_top500_proxy())
        return SP500ConstituentsResult(
            as_of=as_of, source=source, tickers=tickers, n_constituents=len(tickers),
            metadata={"warning": "PURE SURVIVORSHIP proxy — today's top 500, applied historically"},
        )

    elif source == "crsp_vintage":
        tickers, meta = _fetch_crsp_vintage_constituents(as_of)
        return SP500ConstituentsResult(
            as_of=as_of, source=source, tickers=sorted(tickers),
            n_constituents=len(tickers), metadata=meta,
        )

    raise RuntimeError(f"unreachable source branch: {source}")


# ── Wave B: CRSP vintage constituents (W-B-2, 2026-05-10) ──────────────────
# Per spec §2.1 Wave B: CRSP S&P 500 historical constituents (gold standard,
# 1925+, no survivorship, includes delisted/inactive). Activated when WRDS
# account is approved + configured. Until then, mock-mode falls back to
# mktcap_top500_proxy + tags metadata so downstream code can detect skeleton
# vs real Wave B data.

_CRSP_SP500_CONSTITUENTS_SQL = """
SELECT permno, ticker, start_date, end_date
FROM crsp.dsp500list
WHERE start_date <= %(as_of)s
  AND (end_date >= %(as_of)s OR end_date IS NULL)
"""


def _fetch_crsp_vintage_constituents(
    as_of:     datetime.date,
    mock_mode: Optional[bool] = None,
) -> tuple[list[str], dict]:
    """Wave B CRSP vintage constituents at as_of.

    Args:
        as_of:     date for which to query S&P 500 membership
        mock_mode: None (default) auto-detects via crsp_loader.is_wrds_available();
                   True forces mock fallback; False raises if WRDS unavailable.

    Returns:
        (tickers, metadata) — metadata flags WAVE_B_FALLBACK when mock_mode is on
        so downstream verdict / UI can label results as preliminary.
    """
    if mock_mode is None:
        try:
            from engine.universe_singlename.crsp_loader import is_wrds_available
            mock_mode = not is_wrds_available()
        except ImportError:
            mock_mode = True

    if mock_mode:
        # Mock fallback: return mktcap_top500_proxy with metadata flag so
        # downstream code knows Wave B real data is NOT yet active.
        proxy_tickers = sorted(_fetch_mktcap_top500_proxy())
        return proxy_tickers, {
            "WAVE_B_FALLBACK": True,
            "fallback_source": "mktcap_top500_proxy",
            "fallback_warning": (
                "Wave B CRSP vintage constituents NOT YET ACTIVE — using "
                "mktcap_top500_proxy as skeleton fallback. Real CRSP path "
                "activates on WRDS account approval (NUS BizFDB application)."
            ),
        }

    # Real path (activated 2026-05-11) — query crsp.dsp500list + map permno→ticker
    try:
        from engine.universe_singlename.crsp_loader import is_wrds_available
    except ImportError as exc:
        raise RuntimeError(f"crsp_loader unavailable: {exc}")

    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing, "
            "or install wrds + configure credentials for real-data path."
        )

    return _crsp_vintage_real_query(as_of)


# Apply retry wrapper at module-bottom (function defined below; closure capture works)
def _crsp_vintage_real_query(as_of: datetime.date) -> tuple[list[str], dict]:
    """Retry-wrapped real-path WRDS query for CRSP vintage S&P 500 constituents."""
    from engine.universe_singlename.wrds_retry import with_wrds_retry
    wrapped = with_wrds_retry(max_attempts=3, base_delay=5.0)(
        _crsp_vintage_real_query_inner
    )
    return wrapped(as_of)


def _crsp_vintage_real_query_inner(as_of: datetime.date) -> tuple[list[str], dict]:
    """Inner real-path WRDS query, wrapped externally with retry decorator."""
    from engine.universe_singlename.crsp_loader import _open_wrds_connection
    conn = _open_wrds_connection()
    try:
        # Step 1: get permnos active in S&P 500 at as_of
        # crsp.dsp500list schema: permno, start (DATE), ending (DATE)
        result = conn.raw_sql(
            """
            SELECT permno, start, ending
            FROM crsp.dsp500list
            WHERE start <= %(as_of)s
              AND ending >= %(as_of)s
            """,
            params={"as_of": as_of.isoformat()},
            date_cols=["start", "ending"],
        )
        if result.empty:
            logger.warning(
                "crsp_vintage: dsp500list returned 0 permnos at %s "
                "(date may predate CRSP S&P 500 coverage 1957+)", as_of,
            )
            return [], {
                "WAVE_B_FALLBACK": False,
                "source": "crsp.dsp500list",
                "as_of":  as_of.isoformat(),
                "warning": "0 constituents at as_of (CRSP S&P 500 list pre-1957 sparse)",
            }
        permnos = sorted(int(p) for p in result["permno"].dropna().unique())

        # Step 2: resolve permno → ticker via crsp.msenames (point-in-time)
        names = conn.raw_sql(
            """
            SELECT permno, ticker, comnam, namedt, nameendt
            FROM crsp.msenames
            WHERE permno IN %(permnos)s
              AND namedt  <= %(as_of)s
              AND nameendt >= %(as_of)s
            """,
            params={
                "permnos": tuple(permnos),
                "as_of":   as_of.isoformat(),
            },
            date_cols=["namedt", "nameendt"],
        )

        tickers = sorted(
            str(t).strip()
            for t in names["ticker"].dropna().unique()
            if str(t).strip()
        )

        meta = {
            "WAVE_B_FALLBACK":  False,
            "source":           "crsp.dsp500list + crsp.msenames",
            "as_of":            as_of.isoformat(),
            "n_permnos":        len(permnos),
            "n_tickers":        len(tickers),
            "publishable":      True,
            "note":             (
                "Vintage S&P 500 constituents from CRSP — gold standard, "
                "no survivorship bias, includes delisted/inactive companies"
            ),
        }
        logger.info(
            "crsp_vintage: %d permnos → %d tickers at %s",
            len(permnos), len(tickers), as_of,
        )
        return tickers, meta
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Russell 2000 proxy loader (Path J spec id=60, 2026-05-12) ──────────────
def load_russell2000_proxy_at_date(
    as_of:    datetime.date,
    rank_min: int = 1001,
    rank_max: int = 3000,
) -> SP500ConstituentsResult:
    """Russell-2000-proxy universe via CRSP market-cap rank.

    Returns SP500ConstituentsResult shape (reused dataclass) with tickers ranked
    `rank_min`..`rank_max` by market cap at `as_of` (or nearest CRSP msf month-end).

    Implementation:
      1. Round `as_of` down to nearest CRSP month-end (last business day of month)
      2. Query crsp.msf JOIN crsp.msenames for shrcd∈{10,11} exchcd∈{1,2,3}
      3. ROW_NUMBER OVER (ORDER BY abs(prc) × shrout DESC) — point-in-time mkt cap
      4. Filter rank ∈ [rank_min, rank_max]
      5. Return tickers (sorted alphabetically; rank order in metadata)

    Probe 2026-05-12: 2014-01-31 → 2000 firms mkt cap $94M-$2.38B avg $722M;
    single quarter query ~0.7s. No survivorship bias (rank computed fresh per
    quarter; firms drop in/out as mkt cap changes).

    Args:
        as_of:    date for which to query universe
        rank_min: minimum rank (default 1001 — top of Russell 2000)
        rank_max: maximum rank (default 3000 — bottom of Russell 2000)

    Returns:
        SP500ConstituentsResult with .tickers, .source="russell2000_proxy"
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if rank_min < 1 or rank_max < rank_min:
        raise ValueError(f"rank_min ({rank_min}) and rank_max ({rank_max}) invalid")

    from engine.universe_singlename.wrds_retry import with_wrds_retry
    from engine.universe_singlename.crsp_loader import _open_wrds_connection

    @with_wrds_retry(max_attempts=3, base_delay=5.0)
    def _inner():
        conn = _open_wrds_connection()
        try:
            # Query crsp.msf for the latest available date ≤ as_of (CRSP msf is
            # month-end snapshots; we accept whatever date is available within
            # the as_of month — falls back to most-recent prior month if needed).
            sql = """
            WITH msf_at AS (
                SELECT *
                FROM crsp.msf
                WHERE date BETWEEN %(window_start)s AND %(window_end)s
                  AND prc IS NOT NULL AND shrout IS NOT NULL
            ),
            latest_date AS (
                SELECT MAX(date) AS dt FROM msf_at
            ),
            ranked AS (
                SELECT
                    m.permno,
                    m.ticker,
                    m.comnam,
                    ABS(s.prc) * s.shrout AS mkt_cap_thousands,
                    ROW_NUMBER() OVER (ORDER BY ABS(s.prc) * s.shrout DESC NULLS LAST) AS rnk
                FROM msf_at s
                JOIN crsp.msenames m
                  ON s.permno = m.permno
                 AND s.date >= m.namedt
                 AND s.date <= COALESCE(m.nameendt, '9999-12-31')
                WHERE s.date = (SELECT dt FROM latest_date)
                  AND m.shrcd IN (10, 11)
                  AND m.exchcd IN (1, 2, 3)
            )
            SELECT permno, ticker, comnam, mkt_cap_thousands, rnk
            FROM ranked
            WHERE rnk BETWEEN %(rank_min)s AND %(rank_max)s
            ORDER BY rnk
            """
            # Window: 60 days back from as_of (catches month-end if as_of mid-month)
            window_start = (as_of - datetime.timedelta(days=60)).isoformat()
            window_end   = as_of.isoformat()
            result = conn.raw_sql(
                sql,
                params={
                    "window_start": window_start,
                    "window_end":   window_end,
                    "rank_min":     int(rank_min),
                    "rank_max":     int(rank_max),
                },
            )
            if result.empty:
                logger.warning(
                    "russell2000_proxy: no firms in rank %d-%d at %s",
                    rank_min, rank_max, as_of,
                )
                return [], {
                    "source":   "russell2000_proxy",
                    "as_of":    as_of.isoformat(),
                    "rank_min": rank_min, "rank_max": rank_max,
                    "warning":  "0 firms returned (check window or rank range)",
                }
            tickers = sorted(
                str(t).strip()
                for t in result["ticker"].dropna().unique()
                if str(t).strip()
            )
            meta = {
                "source":     "russell2000_proxy",
                "as_of":      as_of.isoformat(),
                "rank_min":   rank_min, "rank_max":   rank_max,
                "n_firms":    len(result),
                "n_tickers":  len(tickers),
                "min_mkt_cap_M": float(result["mkt_cap_thousands"].min()) / 1000,
                "max_mkt_cap_M": float(result["mkt_cap_thousands"].max()) / 1000,
                "avg_mkt_cap_M": float(result["mkt_cap_thousands"].mean()) / 1000,
                "publishable": True,
                "note":       (
                    "Russell-2000-proxy via CRSP market-cap rank synthesis; "
                    "shrcd∈{10,11} exchcd∈{1,2,3}; point-in-time, no "
                    "survivorship bias (rank recomputed each as_of)"
                ),
            }
            logger.info(
                "russell2000_proxy: rank %d-%d at %s → %d firms (mkt cap "
                "$%.0fM-$%.0fM avg $%.0fM)",
                rank_min, rank_max, as_of,
                len(tickers),
                meta["min_mkt_cap_M"], meta["max_mkt_cap_M"], meta["avg_mkt_cap_M"],
            )
            return tickers, meta
        finally:
            try:
                conn.close()
            except Exception:
                pass

    tickers, meta = _inner()
    return SP500ConstituentsResult(
        as_of=as_of,
        source="russell2000_proxy",
        tickers=tickers,
        n_constituents=len(tickers),
        metadata=meta,
    )


def load_sp500_constituents_panel(
    rebalance_dates: list[datetime.date],
    sources:         tuple[str, ...] = UNIVERSE_SOURCES_LOCKED,
) -> dict[str, dict[datetime.date, list[str]]]:
    """Load constituents across all rebalance dates from all sources.

    Returns:
        {source: {date: [tickers]}}
    """
    out: dict[str, dict[datetime.date, list[str]]] = {s: {} for s in sources}
    for source in sources:
        for d in rebalance_dates:
            res = load_sp500_constituents_at_date(as_of=d, source=source)
            out[source][d] = res.tickers
    return out
