"""
engine/universe_singlename/crsp_loader.py — Wave B WRDS / CRSP loader skeleton.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §4.1.

Status (2026-05-10): SKELETON + MOCK MODE.
  - WRDS account approval pending (NUS BizFDB application 2026-05-09)
  - Real WRDS path stubs implemented; will be activated when:
      1. `pip install wrds` (to be added to requirements.txt at activation)
      2. ~/.pgpass configured with WRDS credentials
      3. Test connection: `wrds.Connection().raw_sql("SELECT 1")`
  - Until then: mock mode returns deterministic synthetic panels for
    skeleton-level integration testing of the Wave B walk-forward harness.

Public API:
  - bulk_fetch_crsp_daily_panel() — mirrors panel_fetcher.bulk_fetch_singlestock_panel()
    so Wave A → Wave B swap is single-line at the call site.
  - is_wrds_available() — feature flag for tests + Wave B activation gate.

Wave A vs Wave B distinction:
  - Wave A uses yfinance (Yahoo Finance) — CSP corruption + survivorship bias risk
  - Wave B uses CRSP daily file `dsf` — institutional gold standard since 1925,
    fully vintage point-in-time, includes inactive/delisted companies
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Feature flag: WRDS availability ─────────────────────────────────────────
def is_wrds_available() -> bool:
    """Lazy check — returns True only if `wrds` Python lib is installed AND
    user has configured credentials.

    Credential sources checked (any one suffices):
      - Unix:    ~/.pgpass
      - Windows: %APPDATA%/postgresql/pgpass.conf  (WRDS standard on Windows)
      - Env:     WRDS_USER or WRDS_USERNAME
      - Project: .streamlit/secrets.toml [WRDS] section

    Used by:
      - Wave B walk-forward to gate real-data path
      - Tests to skip integration tests when WRDS unconfigured
      - UI to show Wave B status (PENDING / READY / ACTIVE)
    """
    try:
        import wrds  # noqa: F401
    except ImportError:
        return False
    import os
    # Unix pgpass
    if (Path.home() / ".pgpass").exists():
        return True
    # Windows pgpass (WRDS standard location)
    appdata = os.environ.get("APPDATA")
    if appdata and (Path(appdata) / "postgresql" / "pgpass.conf").exists():
        return True
    # Env var fallback
    if os.environ.get("WRDS_USER") or os.environ.get("WRDS_USERNAME"):
        return True
    # Project secrets fallback
    if _get_wrds_username():
        # Username configured but pgpass not yet created — semi-available
        # (will need first-time interactive setup). Return True so callers
        # know wrds path is intended; first connect will prompt.
        return True
    return False


# ── Storage ─────────────────────────────────────────────────────────────────
_CACHE_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "factor_ensemble_singlename"
)
_CRSP_PANEL_CACHE_PATH = _CACHE_DIR / "_crsp_dsf_panel.parquet"


# ── Connection helper ───────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class WRDSConnectionInfo:
    """Diagnostic info about the active WRDS connection (logged on first call)."""
    library_version: str
    user:            Optional[str]
    server:          Optional[str]


def _get_wrds_username() -> Optional[str]:
    """Read WRDS username from .streamlit/secrets.toml [WRDS] section.

    Falls back to env var WRDS_USERNAME, then None. Returning None lets
    wrds.Connection() prompt interactively (acceptable for first-time setup).
    """
    # Try streamlit secrets first (project's secret-config pattern)
    try:
        import streamlit as st
        return st.secrets["WRDS"]["USERNAME"]
    except Exception:
        pass
    # Fallback: TOML direct read (for non-Streamlit contexts e.g. CLI)
    try:
        import toml
        secrets_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            data = toml.load(secrets_path)
            return data.get("WRDS", {}).get("USERNAME")
    except Exception:
        pass
    # Fallback: env var
    import os
    return os.environ.get("WRDS_USERNAME")


def _open_wrds_connection():
    """Open a WRDS database connection.

    Lazy import + lazy connect — only invoked when `mock_mode=False` is requested.
    Reads username from .streamlit/secrets.toml [WRDS] section; password comes
    from auto-generated ~/AppData/Roaming/postgresql/pgpass.conf (Windows) or
    ~/.pgpass (Unix). First-time setup: `wrds.Connection()` interactive prompt
    auto-creates pgpass file.

    Raises RuntimeError if wrds library or credentials not configured (caller
    falls back to mock_mode or aborts).
    """
    try:
        import wrds
    except ImportError as exc:
        raise RuntimeError(
            "wrds Python library not installed. To activate Wave B real-data "
            "path: `pip install wrds` + configure pgpass.conf with WRDS "
            "credentials. Until then, set mock_mode=True for skeleton testing."
        ) from exc

    username = _get_wrds_username()
    kwargs = {"wrds_username": username} if username else {}

    try:
        conn = wrds.Connection(**kwargs)
        info = WRDSConnectionInfo(
            library_version=getattr(wrds, "__version__", "unknown"),
            user=getattr(conn, "_username", None) or username or "unknown",
            server=getattr(conn, "_hostname", None) or "wrds-pgdata.wharton.upenn.edu",
        )
        logger.info(
            "WRDS connection opened: lib=%s user=%s server=%s",
            info.library_version, info.user, info.server,
        )
        return conn
    except Exception as exc:
        raise RuntimeError(
            f"WRDS connection failed: {exc}. Check ~/.pgpass credentials + "
            "WRDS account approval status (NUS BizFDB application)."
        ) from exc


# ── Mock-mode panel generator ───────────────────────────────────────────────
def _mock_crsp_panel(
    tickers: list[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    """Generate a deterministic synthetic price panel for skeleton testing.

    Uses ticker hash as random seed so panel is reproducible across runs +
    across machines. Prices follow geometric Brownian motion with drift 8% pa
    + vol 25% pa (typical large-cap stock Sharpe-irrelevant baseline).

    Returns a DataFrame in the same shape as bulk_fetch_singlestock_panel()
    so downstream walk_forward / factor modules can exercise their entire
    pipeline pre-WRDS.
    """
    dates = pd.bdate_range(start=start, end=end, freq="B")
    if len(dates) == 0:
        return pd.DataFrame(columns=tickers)

    n = len(dates)
    panel = pd.DataFrame(index=dates, columns=sorted(set(tickers)), dtype=float)

    annual_drift = 0.08
    annual_vol   = 0.25
    daily_drift  = annual_drift / 252.0
    daily_vol    = annual_vol / np.sqrt(252.0)

    for ticker in panel.columns:
        seed = int(hashlib.md5(ticker.encode("utf-8")).hexdigest()[:8], 16)
        rng  = np.random.default_rng(seed)
        rets = rng.normal(loc=daily_drift, scale=daily_vol, size=n)
        prices = 100.0 * np.exp(np.cumsum(rets))
        panel[ticker] = prices

    return panel


# ── CRSP query (real-mode, activated 2026-05-11) ────────────────────────────
_CRSP_NAMES_PERMNO_SQL = """
SELECT permno, ticker, comnam, namedt, nameendt
FROM crsp.msenames
WHERE ticker IN %(tickers)s
  AND nameendt >= %(start_date)s
  AND namedt  <= %(end_date)s
ORDER BY ticker, namedt
"""

_CRSP_DSF_PRICE_SQL = """
SELECT
    permno,
    date,
    prc,
    cfacpr
FROM crsp.dsf
WHERE date BETWEEN %(start_date)s AND %(end_date)s
  AND permno IN %(permnos)s
ORDER BY permno, date
"""


def _resolve_tickers_to_permnos(
    conn,
    tickers: list[str],
    start:   datetime.date,
    end:     datetime.date,
) -> dict[str, list[tuple[int, datetime.date, datetime.date]]]:
    """Resolve ticker → (permno, namedt, nameendt) periods active in window.

    crsp.msenames stores name history; a single ticker symbol may correspond
    to multiple permnos across time (rare ticker reuse). For Wave B walk-
    forward we pull all overlapping rows; downstream pivot uses the latest
    active permno on each date.

    Returns:
        {ticker: [(permno, namedt, nameendt), ...]}  — empty list if ticker
        not in CRSP universe for window.
    """
    if not tickers:
        return {}
    result = conn.raw_sql(
        _CRSP_NAMES_PERMNO_SQL,
        params={
            "tickers":    tuple(tickers),
            "start_date": start.isoformat(),
            "end_date":   end.isoformat(),
        },
        date_cols=["namedt", "nameendt"],
    )
    if result.empty:
        return {}
    out: dict[str, list[tuple[int, datetime.date, datetime.date]]] = {}
    for _, row in result.iterrows():
        t = str(row["ticker"]).strip()
        namedt = row["namedt"] if not pd.isna(row["namedt"]) else start
        nameendt = row["nameendt"] if not pd.isna(row["nameendt"]) else end
        if hasattr(namedt, "date"):
            namedt = namedt.date()
        if hasattr(nameendt, "date"):
            nameendt = nameendt.date()
        out.setdefault(t, []).append((int(row["permno"]), namedt, nameendt))
    return out


from engine.universe_singlename.wrds_retry import with_wrds_retry


@with_wrds_retry(max_attempts=3, base_delay=5.0)
def _real_crsp_panel(
    tickers: list[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    """Real WRDS-CRSP query: ticker → permno → daily adjusted close.

    Pipeline:
      1. Open WRDS connection (`_open_wrds_connection()`)
      2. Resolve ticker → permno via `crsp.msenames` (handle ticker reuse
         via namedt/nameendt window overlap)
      3. Query `crsp.dsf` for daily prices over [start, end] for those permnos
      4. Apply CRSP adjustment:  adj_close = abs(prc) / cfacpr
         (negative prc = bid-ask midpoint per CRSP convention — take abs;
          cfacpr divides for split/dividend adjustment to back-cast prices)
      5. Map permno → ticker (latest active in window) + pivot long → wide
      6. Return DataFrame indexed by date, columns = tickers, NaN where missing

    Activated 2026-05-11 per `project_wave_b_wrds_activation_checklist_2026-05-10.md`.
    """
    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing, "
            "or install wrds + configure credentials for real-data path."
        )

    if not tickers:
        return pd.DataFrame()

    needed_tickers = sorted(set(tickers))
    conn = _open_wrds_connection()
    try:
        # Step 1: ticker → permno mapping
        ticker_to_permnos = _resolve_tickers_to_permnos(
            conn, needed_tickers, start, end,
        )
        all_permnos = sorted({p for periods in ticker_to_permnos.values()
                              for (p, _, _) in periods})
        if not all_permnos:
            logger.warning(
                "crsp_loader: no CRSP permno matches for %d tickers in [%s, %s]",
                len(needed_tickers), start, end,
            )
            return pd.DataFrame()

        # Step 2: query CRSP daily prices
        logger.info(
            "crsp_loader: querying crsp.dsf for %d permnos × [%s, %s]",
            len(all_permnos), start, end,
        )
        dsf = conn.raw_sql(
            _CRSP_DSF_PRICE_SQL,
            params={
                "start_date": start.isoformat(),
                "end_date":   end.isoformat(),
                "permnos":    tuple(all_permnos),
            },
            date_cols=["date"],
        )
        if dsf.empty:
            logger.warning("crsp_loader: crsp.dsf returned 0 rows")
            return pd.DataFrame()

        # Step 3: apply CRSP adjustment
        # prc may be negative when CRSP records bid-ask average (no closing
        # trade that day) — abs() is the canonical CRSP convention.
        dsf["prc"] = dsf["prc"].abs()
        # cfacpr is the cumulative price adjustment factor; divide prc by
        # cfacpr to get adjusted-for-splits-and-dividends backward-cast price.
        # When cfacpr is NaN or 0, fall back to raw prc (rare; usually means
        # missing factor).
        cfacpr = dsf["cfacpr"].fillna(1.0).replace(0, 1.0)
        dsf["adj_close"] = dsf["prc"] / cfacpr

        # Step 4: build permno → ticker mapping (latest active in window)
        permno_to_ticker: dict[int, str] = {}
        for ticker, periods in ticker_to_permnos.items():
            # When ticker maps to multiple permnos in window, pick the one
            # with the latest nameendt (most recent ID).
            best = max(periods, key=lambda x: x[2])
            permno_to_ticker[best[0]] = ticker
            # Also map other permnos to same ticker (rare; both periods active)
            for (p, _, _) in periods:
                permno_to_ticker.setdefault(p, ticker)

        dsf["ticker"] = dsf["permno"].map(permno_to_ticker)
        dsf = dsf.dropna(subset=["ticker"])

        # Step 5: pivot to date × ticker
        panel = dsf.pivot_table(
            index="date",
            columns="ticker",
            values="adj_close",
            aggfunc="last",   # in case of cross-permno conflict, prefer latest
        )
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()
        logger.info(
            "crsp_loader: real panel built: %d dates × %d tickers (requested %d)",
            panel.shape[0], panel.shape[1], len(needed_tickers),
        )
        return panel
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Public API ──────────────────────────────────────────────────────────────
def bulk_fetch_crsp_daily_panel(
    tickers:     list[str],
    start_date:  datetime.date,
    end_date:    datetime.date,
    *,
    mock_mode:   Optional[bool] = None,
    use_cache:   bool = True,
) -> pd.DataFrame:
    """Bulk-fetch CRSP daily adjusted-close price panel.

    Mirror of `engine.factor_ensemble_singlename.panel_fetcher.bulk_fetch_singlestock_panel`
    so Wave B walk-forward can call this in place of the yfinance path with no
    other changes (single-line swap at the harness call site).

    Args:
        tickers:     S&P 500 ticker symbols (will be ticker-period-resolved
                     to CRSP permno on real path; mock path treats as opaque IDs)
        start_date:  inclusive panel start
        end_date:    inclusive panel end
        mock_mode:   None (default) = auto-detect via is_wrds_available()
                     True           = force synthetic panel (skeleton testing)
                     False          = require real WRDS path (raises if not configured)
        use_cache:   read/write parquet cache (real path only;
                     mock path is reproducible from seed and does not cache)

    Returns:
        pd.DataFrame indexed by date (B-day frequency), columns = sorted unique
        tickers, NaN for missing/delisted/pre-IPO. Values are
        adjusted-close prices.
    """
    if mock_mode is None:
        mock_mode = not is_wrds_available()

    needed_tickers = sorted(set(tickers))

    if mock_mode:
        logger.info(
            "crsp_loader: MOCK MODE — synthetic panel for %d tickers, [%s, %s]",
            len(needed_tickers), start_date, end_date,
        )
        return _mock_crsp_panel(needed_tickers, start_date, end_date)

    # Real path with cache
    if use_cache and _CRSP_PANEL_CACHE_PATH.exists():
        try:
            cache_df = pd.read_parquet(_CRSP_PANEL_CACHE_PATH)
            cache_ok = (
                not cache_df.empty
                and cache_df.index.min() <= pd.Timestamp(start_date)
                and cache_df.index.max() >= pd.Timestamp(end_date)
                and all(t in cache_df.columns for t in needed_tickers)
            )
            if cache_ok:
                logger.info(
                    "crsp_loader: cache HIT for %d tickers, [%s, %s]",
                    len(needed_tickers), start_date, end_date,
                )
                return cache_df
        except Exception as exc:
            logger.warning("crsp_loader: cache read failed: %s — refetching", exc)

    panel = _real_crsp_panel(needed_tickers, start_date, end_date)

    # Persist
    if use_cache:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(_CRSP_PANEL_CACHE_PATH)
            logger.info(
                "crsp_loader: cache persisted: %d tickers × %d dates",
                panel.shape[1], panel.shape[0],
            )
        except Exception as exc:
            logger.warning("crsp_loader: cache persist failed: %s", exc)

    return panel
