"""
engine/data_snapshot.py — Backtest data freeze + reload (S-3, 2026-05-06).

Reproducibility infrastructure for the project's backtest pipeline.
Captures the four external data dependencies (yfinance monthly + daily ETF
prices, ^VIX series, FRED macros) into a `data/snapshots/<snapshot_id>/`
parquet bundle so any examiner can re-run `engine.backtest.run_backtest`
with identical inputs years later — yfinance / FRED upstream changes
notwithstanding.

Storage layout (per `docs/reproducibility.md`):
    data/snapshots/<snapshot_id>/
        manifest.json           # metadata + per-file sha256
        yf_monthly_etf.parquet  # 18 ETFs × monthly returns
        yf_daily_etf.parquet    # 18 ETFs × daily prices
        yf_vix.parquet          # ^VIX daily
        fred_macros.parquet     # FRED series stacked

Usage:
    # 1) Freeze: capture current live data into a named snapshot
    snap = freeze_snapshot(
        start_date=datetime.date(2009, 1, 1),
        end_date=datetime.date.today(),
        name="thesis_v1",
        tickers=[...],
        fred_series=["DGS10", "DGS2", "BAMLH0A0HYM2"],
    )

    # 2) Reload later for reproducibility:
    snap = load_snapshot("thesis_v1_2026-05-06")
    run_backtest(start, end, snapshot=snap)  # zero live network calls
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


SNAPSHOT_ROOT = Path(__file__).parent.parent / "data" / "snapshots"
SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DataSnapshot dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataSnapshot:
    """In-memory bundle of frozen data for one backtest reproducibility run."""
    snapshot_id:     str
    created_at:      datetime.datetime
    fetch_start:     datetime.date
    fetch_end:       datetime.date
    tickers:         List[str]
    fred_series:     List[str]

    yf_monthly_etf:  pd.DataFrame   # ticker columns × monthly DatetimeIndex (returns)
    yf_daily_etf:    pd.DataFrame   # ticker columns × daily DatetimeIndex (prices)
    yf_vix:          pd.DataFrame   # 1-col 'close' × daily DatetimeIndex
    fred_macros:     pd.DataFrame   # FRED series id columns × daily DatetimeIndex

    manifest:        Dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return SNAPSHOT_ROOT / self.snapshot_id


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────
_FILE_NAMES = {
    "yf_monthly_etf": "yf_monthly_etf.parquet",
    "yf_daily_etf":   "yf_daily_etf.parquet",
    "yf_vix":         "yf_vix.parquet",
    "fred_macros":    "fred_macros.parquet",
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_parquet(df: pd.DataFrame, dst: Path) -> None:
    """Write to .tmp then rename — guards against partial-write corruption."""
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    df.to_parquet(tmp, compression="snappy", engine="pyarrow")
    os.replace(tmp, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Live data fetchers (separated so freeze_snapshot can drive them)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_yf_monthly(tickers: List[str],
                      start: datetime.date,
                      end:   datetime.date) -> pd.DataFrame:
    """Monthly returns; mirrors engine/backtest._fetch_monthly_returns logic."""
    import yfinance as yf
    fetch_start = start - datetime.timedelta(days=40)
    raw = yf.download(
        tickers,
        start=str(fetch_start),
        end=str(end + datetime.timedelta(days=5)),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=tickers)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    monthly_price  = close.resample("ME").last().dropna(how="all")
    monthly_return = monthly_price.pct_change().dropna(how="all")
    return monthly_return


def _fetch_yf_daily(tickers: List[str],
                    start: datetime.date,
                    end:   datetime.date) -> pd.DataFrame:
    """Daily close prices for Ledoit-Wolf covariance lookback."""
    import yfinance as yf
    raw = yf.download(
        tickers,
        start=str(start),
        end=str(end + datetime.timedelta(days=5)),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=tickers)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close.index = pd.to_datetime(close.index).normalize()
    return close


def _fetch_yf_vix(start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """^VIX series for regime detection."""
    import yfinance as yf
    raw = yf.download(
        "^VIX",
        start=str(start),
        end=str(end + datetime.timedelta(days=5)),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        return pd.DataFrame(columns=["close"])
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return pd.DataFrame(columns=["close"])
        s = raw["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
    elif "Close" in raw.columns:
        s = raw["Close"]
    else:
        return pd.DataFrame(columns=["close"])
    df = s.to_frame(name="close")
    df.index = pd.to_datetime(df.index).normalize()
    return df


def _fetch_fred(series: List[str],
                start: datetime.date,
                end:   datetime.date) -> pd.DataFrame:
    """FRED CSV pull, joined into one DataFrame keyed by series id."""
    import io
    import requests
    out_frames: Dict[str, pd.Series] = {}
    for sid in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), parse_dates=[0])
            df.columns = ["date", sid]
            df = df.set_index("date").sort_index()
            df.index = df.index.normalize()
            df = df.loc[
                pd.Timestamp(start) - pd.Timedelta(days=10):
                pd.Timestamp(end) + pd.Timedelta(days=10)
            ]
            # Coerce non-numeric to NaN (FRED uses '.' for missing data)
            df[sid] = pd.to_numeric(df[sid], errors="coerce")
            out_frames[sid] = df[sid]
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", sid, exc)
    if not out_frames:
        return pd.DataFrame(columns=series)
    return pd.concat(out_frames, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def freeze_snapshot(
    *,
    start_date:  datetime.date,
    end_date:    datetime.date,
    name:        str,
    tickers:     List[str],
    fred_series: List[str],
    notes:       Optional[str] = None,
) -> DataSnapshot:
    """
    Pull all 4 data sources for the requested date range, write parquet bundle
    + manifest.json with per-file sha256. Returns the loaded DataSnapshot.

    snapshot_id is `<name>_<YYYY-MM-DD>`. If a snapshot of the same id already
    exists, raises FileExistsError to avoid silent overwrite of historical
    reproducibility anchors.
    """
    today = datetime.date.today()
    snapshot_id = f"{name}_{today.isoformat()}"
    dst = SNAPSHOT_ROOT / snapshot_id
    if dst.exists():
        raise FileExistsError(
            f"snapshot {snapshot_id} already exists at {dst}. "
            "Pick a different name or rm -rf the existing dir."
        )
    dst.mkdir(parents=True, exist_ok=False)

    logger.info("freeze_snapshot %s: fetching yf monthly...", snapshot_id)
    yf_monthly = _fetch_yf_monthly(tickers, start_date, end_date)
    logger.info("freeze_snapshot %s: fetching yf daily...", snapshot_id)
    yf_daily   = _fetch_yf_daily(tickers,   start_date, end_date)
    logger.info("freeze_snapshot %s: fetching ^VIX...", snapshot_id)
    yf_vix_df  = _fetch_yf_vix(start_date,  end_date)
    logger.info("freeze_snapshot %s: fetching FRED...", snapshot_id)
    fred_df    = _fetch_fred(fred_series,   start_date, end_date)

    files: Dict[str, Dict[str, Any]] = {}
    for key, df in [
        ("yf_monthly_etf", yf_monthly),
        ("yf_daily_etf",   yf_daily),
        ("yf_vix",         yf_vix_df),
        ("fred_macros",    fred_df),
    ]:
        path = dst / _FILE_NAMES[key]
        _atomic_write_parquet(df, path)
        files[_FILE_NAMES[key]] = {
            "sha256":  _file_sha256(path),
            "n_rows":  int(len(df)),
            "n_cols":  int(len(df.columns)),
            "columns": list(map(str, df.columns)),
        }

    code_version = _try_get_git_rev()
    manifest = {
        "snapshot_id":  snapshot_id,
        "name":         name,
        "created_at":   datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "fetch_start":  start_date.isoformat(),
        "fetch_end":    end_date.isoformat(),
        "tickers":      list(tickers),
        "fred_series":  list(fred_series),
        "vix_symbol":   "^VIX",
        "files":        files,
        "code_version": code_version,
        "notes":        notes,
        "format_version": 1,
    }
    (dst / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "freeze_snapshot %s: done (%d files; total ~%s bytes)",
        snapshot_id, len(files),
        sum(int((dst / k).stat().st_size) for k in files),
    )
    return DataSnapshot(
        snapshot_id    = snapshot_id,
        created_at     = datetime.datetime.utcnow(),
        fetch_start    = start_date,
        fetch_end      = end_date,
        tickers        = list(tickers),
        fred_series    = list(fred_series),
        yf_monthly_etf = yf_monthly,
        yf_daily_etf   = yf_daily,
        yf_vix         = yf_vix_df,
        fred_macros    = fred_df,
        manifest       = manifest,
    )


def load_snapshot(snapshot_id: str, *, verify_hashes: bool = True) -> DataSnapshot:
    """Reload a snapshot bundle from disk. Verifies sha256 by default."""
    src = SNAPSHOT_ROOT / snapshot_id
    if not src.exists():
        raise FileNotFoundError(f"snapshot {snapshot_id} not at {src}")
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing in {src}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if verify_hashes:
        for fname, meta in manifest.get("files", {}).items():
            path = src / fname
            if not path.exists():
                raise FileNotFoundError(f"{path} missing per manifest")
            actual = _file_sha256(path)
            expected = meta.get("sha256")
            if actual != expected:
                raise ValueError(
                    f"snapshot {snapshot_id}: {fname} sha256 mismatch "
                    f"(stored {expected[:12]} vs actual {actual[:12]}) — TAMPER"
                )

    yf_monthly = pd.read_parquet(src / _FILE_NAMES["yf_monthly_etf"])
    yf_daily   = pd.read_parquet(src / _FILE_NAMES["yf_daily_etf"])
    yf_vix_df  = pd.read_parquet(src / _FILE_NAMES["yf_vix"])
    fred_df    = pd.read_parquet(src / _FILE_NAMES["fred_macros"])

    return DataSnapshot(
        snapshot_id    = snapshot_id,
        created_at     = datetime.datetime.fromisoformat(
            manifest["created_at"].rstrip("Z")
        ),
        fetch_start    = datetime.date.fromisoformat(manifest["fetch_start"]),
        fetch_end      = datetime.date.fromisoformat(manifest["fetch_end"]),
        tickers        = list(manifest.get("tickers", [])),
        fred_series    = list(manifest.get("fred_series", [])),
        yf_monthly_etf = yf_monthly,
        yf_daily_etf   = yf_daily,
        yf_vix         = yf_vix_df,
        fred_macros    = fred_df,
        manifest       = manifest,
    )


def list_snapshots() -> List[Dict[str, Any]]:
    """Return list of all snapshot manifests sorted by created_at descending."""
    out: List[Dict[str, Any]] = []
    if not SNAPSHOT_ROOT.exists():
        return out
    for d in SNAPSHOT_ROOT.iterdir():
        if not d.is_dir():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            out.append(json.loads(mf.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("Skipping unparseable manifest at %s", mf)
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used by run_backtest / regime to read snapshot slices on demand
# ─────────────────────────────────────────────────────────────────────────────
def get_monthly_returns_from_snapshot(
    snap:    DataSnapshot,
    tickers: List[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    """Slice snapshot's monthly returns to the requested window + ticker set."""
    df = snap.yf_monthly_etf
    if df.empty:
        return df
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    sliced = df.loc[mask]
    common = [t for t in tickers if t in sliced.columns]
    return sliced[common]


def get_daily_prices_from_snapshot(
    snap:    DataSnapshot,
    tickers: List[str],
    start:   datetime.date,
    end:     datetime.date,
) -> pd.DataFrame:
    df = snap.yf_daily_etf
    if df.empty:
        return df
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    sliced = df.loc[mask]
    common = [t for t in tickers if t in sliced.columns]
    return sliced[common]


def get_vix_from_snapshot(
    snap:  DataSnapshot,
    start: datetime.date,
    end:   datetime.date,
) -> pd.Series:
    df = snap.yf_vix
    if df.empty:
        return pd.Series(dtype=float, name="close")
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask, "close"] if "close" in df.columns else pd.Series(dtype=float)


def get_fred_from_snapshot(
    snap:        DataSnapshot,
    series_id:   str,
    start:       datetime.date,
    end:         datetime.date,
) -> pd.Series:
    df = snap.fred_macros
    if df.empty or series_id not in df.columns:
        return pd.Series(dtype=float, name=series_id)
    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask, series_id]


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────
def _try_get_git_rev() -> Optional[str]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return None
