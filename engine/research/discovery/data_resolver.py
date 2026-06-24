"""engine/research/discovery/data_resolver.py — map mechanism YAML
required_data tokens to real fetched panels.

Senior gap ⑤ per [[project-end-to-end-vision-2026-05-30]]: until now
forward_oos_runner used synthetic data, making "real" verdicts
indistinguishable from noise. This module is the real-data path.

DESIGN:
  1. Read mechanism YAML → extract required_data tokens + binding +
     execution_template.template_id
  2. Resolve each token to a fetcher call (via _TOKEN_FETCHERS map)
  3. Reshape to template-expected panels (factor_panel / price_panel
     / return_panel)
  4. Return panels dict ready to pass into template_fn(**panels, **binding)

CURRENT TOKEN COVERAGE (v1, equity-focused):
  crsp_dsf / crsp_msf  — WRDS CRSP daily/monthly  → falls back yfinance
  ret_daily / ret_monthly — derived from CRSP/yfinance
  vix_index / vix3m_index / vxx_etn / vxz_etn — yfinance
  fred_macro — FRED API

NOT YET COVERED (returns NotImplementedError with clear message):
  compustat_*  — fundamentals, needs wrds_compustat fetcher
  ibes_*       — analyst data, needs wrds_ibes fetcher
  futures (tr_ds_fut_settle / cmdty_settle / fx_settle / rates_settle
            / eqidx_settle) — needs wrds_direct futures fetcher
  tr13f_holdings / edgar_8k_meta / dera_insider — niche, defer

CACHING:
  Real data fetches are expensive (WRDS / yfinance / FRED rate limits +
  network time). Cache by (token, start_date, end_date, ticker_hash)
  → parquet file in data/cache/data_resolver/. 7-day TTL.
"""
from __future__ import annotations

import datetime
import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "data_resolver"
CACHE_TTL_DAYS = 7


# ── Default universe + window ────────────────────────────────────────────

DEFAULT_EQUITY_UNIVERSE = [
    # Liquid mega-cap subset — sufficient for factor signal tests on a
    # promoted candidate. Real production binding can override universe.
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS",
    "JNJ", "PFE", "MRK", "UNH", "LLY",
    "PG", "KO", "PEP", "WMT", "HD",
    "XOM", "CVX", "CAT", "BA", "GE",
    "SPY", "QQQ", "IWM",   # benchmarks
]
DEFAULT_SAMPLE_YEARS = 7      # enough for monthly gate with 24-month floor


# ── Cache helpers ────────────────────────────────────────────────────────

def _cache_key(token: str, start: str, end: str,
                  universe: list[str] | None) -> Path:
    universe_hash = hashlib.sha256(
        ",".join(sorted(universe or [])).encode("utf-8")
    ).hexdigest()[:8]
    name = f"{token}_{start}_{end}_{universe_hash}.parquet"
    return CACHE_DIR / name


def _load_cached(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        age_days = (datetime.datetime.utcnow().timestamp()
                      - path.stat().st_mtime) / 86400
        if age_days > CACHE_TTL_DAYS:
            return None
        return pd.read_parquet(path)
    except Exception:
        return None


def _save_cached(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception as exc:
        logger.warning("cache save failed %s: %s", path.name, exc)


# ── Token → fetcher dispatch ─────────────────────────────────────────────

def _fetch_crsp_daily(start: str, end: str,
                        universe: list[str] | None = None) -> pd.DataFrame:
    """CRSP daily — falls back to yfinance if WRDS unavailable.

    Returns long format: date, ticker, prc, ret
    """
    try:
        from engine.data.fetchers.wrds_crsp import fetch_dsf
        df = fetch_dsf(start, end, tickers=universe)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.info("WRDS CRSP daily unavailable, falling back to yfinance: %s",
                       exc)
    from engine.data.fetchers.api_yfinance import fetch_equity_daily
    return fetch_equity_daily(start, end, tickers=universe or DEFAULT_EQUITY_UNIVERSE)


def _fetch_crsp_monthly(start: str, end: str,
                          universe: list[str] | None = None) -> pd.DataFrame:
    """CRSP monthly — falls back to yfinance if WRDS unavailable."""
    try:
        from engine.data.fetchers.wrds_crsp import fetch_msf
        df = fetch_msf(start, end, tickers=universe)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.info("WRDS CRSP monthly unavailable, falling back to yfinance: %s",
                       exc)
    from engine.data.fetchers.api_yfinance import fetch_equity_monthly
    return fetch_equity_monthly(start, end, tickers=universe or DEFAULT_EQUITY_UNIVERSE)


def _fetch_vix_family(start: str, end: str, *, symbol: str) -> pd.DataFrame:
    """VIX-family from yfinance (^VIX, ^VIX3M, VXX, VXZ)."""
    from engine.data.fetchers.api_yfinance import fetch_index
    return fetch_index(start, end, symbol=symbol)


def _fetch_fred_macro(start: str, end: str,
                        series: list[str] | None = None) -> pd.DataFrame:
    """FRED macro series. Defaults to T10Y2Y, VIXCLS, DGS10 if not specified."""
    from engine.data.fetchers.api_fred import fetch_series
    series = series or ["T10Y2Y", "DGS10", "VIXCLS"]
    frames = []
    for s in series:
        try:
            df = fetch_series(s, start, end)
            if df is not None and not df.empty:
                df["series"] = s
                frames.append(df)
        except Exception as exc:
            logger.warning("FRED %s failed: %s", s, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


_TOKEN_FETCHERS = {
    "crsp_dsf":      lambda s, e, u: _fetch_crsp_daily(s, e, u),
    "crsp_msf":      lambda s, e, u: _fetch_crsp_monthly(s, e, u),
    "ret_daily":     lambda s, e, u: _fetch_crsp_daily(s, e, u),
    "ret_monthly":   lambda s, e, u: _fetch_crsp_monthly(s, e, u),
    "vix_index":     lambda s, e, u: _fetch_vix_family(s, e, symbol="^VIX"),
    "vix3m_index":   lambda s, e, u: _fetch_vix_family(s, e, symbol="^VIX3M"),
    "vxx_etn":       lambda s, e, u: _fetch_vix_family(s, e, symbol="VXX"),
    "vxz_etn":       lambda s, e, u: _fetch_vix_family(s, e, symbol="VXZ"),
    "fred_macro":    lambda s, e, u: _fetch_fred_macro(s, e),
}


# Tokens we DECLARE but cannot fetch yet — caller gets NotImplementedError
# with clear message
_NOT_YET_FETCHABLE = {
    "compustat_quarterly", "compustat_annual",
    "ibes_summary", "ibes_guidance",
    "SUE_panel", "ann_dates", "ret_60d",
    "tr13f_holdings", "edgar_8k_meta", "dera_insider",
    "tr_ds_fut_settle", "cmdty_contracts", "cmdty_settle",
    "fx_contracts", "fx_settle", "rates_contracts", "rates_settle",
    "rates_xc_settle", "eqidx_contracts", "eqidx_settle",
    "trace_bond_monthly",
    "rpna_daily_sentiment", "rpna_entity_map",
}


def fetch_token(
    token: str, *,
    start: str, end: str,
    universe: list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Resolve one token to a real DataFrame. Raises NotImplementedError
    for tokens that have no fetcher wired yet."""
    # IMPORTANT: check _NOT_YET_FETCHABLE BEFORE _TOKEN_FETCHERS membership;
    # otherwise tokens in IMPLEMENTED_DATA-but-not-wired would raise KeyError
    # instead of the more informative NotImplementedError.
    from engine.research.hygiene_tools import IMPLEMENTED_DATA, DECLARED_DATA
    if token in _NOT_YET_FETCHABLE or token in DECLARED_DATA:
        raise NotImplementedError(
            f"token {token!r} is declared in IMPLEMENTED_DATA but no real "
            f"fetcher is wired in data_resolver. Wire engine/data/fetchers/"
            f"<source>.py first."
        )
    if token not in _TOKEN_FETCHERS:
        # Unknown to both fetcher map AND not-yet-wired list
        if token in IMPLEMENTED_DATA:
            raise NotImplementedError(
                f"token {token!r} is in IMPLEMENTED_DATA but no fetcher "
                f"registered in _TOKEN_FETCHERS."
            )
        raise KeyError(f"unknown token {token!r}; valid: {sorted(_TOKEN_FETCHERS.keys())}")

    if use_cache:
        cache_path = _cache_key(token, start, end, universe)
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached

    fetcher = _TOKEN_FETCHERS[token]
    df = fetcher(start, end, universe)
    if use_cache:
        _save_cached(_cache_key(token, start, end, universe), df)
    return df


# ── Panel construction (long → wide for templates) ───────────────────────

def _long_to_wide_prc(df: pd.DataFrame) -> pd.DataFrame:
    """Long (date, ticker, prc) → wide (dates × tickers) price panel."""
    if df.empty or "prc" not in df.columns:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="prc").sort_index()


def _long_to_wide_ret(df: pd.DataFrame) -> pd.DataFrame:
    """Long (date, ticker, ret) → wide (dates × tickers) return panel."""
    if df.empty or "ret" not in df.columns:
        return pd.DataFrame()
    return df.pivot(index="date", columns="ticker", values="ret").sort_index()


def resolve_panels_for_template(
    mechanism_yaml: dict,
    *,
    end_date: str | None = None,
    sample_years: int = DEFAULT_SAMPLE_YEARS,
    universe: list[str] | None = None,
) -> dict:
    """Build the panels dict the mechanism's template expects, using
    real data fetched from required_data tokens.

    Returns: {panel_name: DataFrame} ready to pass into the template
    function as keyword args.

    Raises:
      ValueError if execution_template missing
      NotImplementedError if any required token has no fetcher
    """
    exec_tpl = mechanism_yaml.get("execution_template") or {}
    template_id = exec_tpl.get("template_id")
    if not template_id:
        raise ValueError("mechanism YAML has no execution_template.template_id")
    required_data = mechanism_yaml.get("required_data") or []
    if not required_data:
        raise ValueError("mechanism YAML has no required_data tokens")

    end_date = end_date or datetime.date.today().isoformat()
    start_date = (
        datetime.date.fromisoformat(end_date)
        - datetime.timedelta(days=int(sample_years * 365.25))
    ).isoformat()

    universe = universe or DEFAULT_EQUITY_UNIVERSE

    # Fetch all tokens
    fetched: dict[str, pd.DataFrame] = {}
    for token in required_data:
        df = fetch_token(token, start=start_date, end=end_date,
                            universe=universe)
        fetched[token] = df

    # Build template-specific panels
    panels: dict[str, pd.DataFrame] = {}
    if template_id in ("equity_xsmom", "factor_quartile",
                          "dispersion"):
        # Need price_panel + return_panel (+ factor_panel for factor_quartile/dispersion)
        # Use crsp_msf if available, else crsp_dsf re-aggregated.
        # Note: DataFrame truthiness is ambiguous so use explicit isinstance check.
        def _first_nonempty(*keys):
            for k in keys:
                v = fetched.get(k)
                if isinstance(v, pd.DataFrame) and not v.empty:
                    return v
            return None
        msf = _first_nonempty("crsp_msf", "ret_monthly")
        dsf = _first_nonempty("crsp_dsf", "ret_daily")
        if msf is not None:
            panels["price_panel"] = _long_to_wide_prc(msf)
            panels["return_panel"] = _long_to_wide_ret(msf)
        elif dsf is not None:
            # Re-aggregate to monthly
            d = dsf.copy()
            d["month"] = pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp("M")
            monthly = (d.groupby(["ticker", "month"], as_index=False)
                          .agg(ret=("ret", lambda r: (1 + r).prod() - 1),
                                 prc=("prc", "last")))
            monthly = monthly.rename(columns={"month": "date"})
            panels["price_panel"] = _long_to_wide_prc(monthly)
            panels["return_panel"] = _long_to_wide_ret(monthly)
        else:
            raise ValueError(
                f"template {template_id} needs equity data but neither "
                f"crsp_msf / crsp_dsf / ret_* was fetched"
            )
        # For factor_quartile + dispersion, we ALSO need factor_panel.
        # In v1, fall back to using returns lag as a stand-in factor (NOT
        # real production — production should fetch the actual factor
        # data via a binding param).
        if template_id in ("factor_quartile", "dispersion"):
            # Use trailing 12-1 momentum as default factor panel (mirrors
            # equity_xsmom signal — gives equality with what auto-gate
            # would do on synthetic data, just on real)
            ret = panels["return_panel"]
            mom = ret.shift(1).rolling(12, min_periods=6).sum()
            panels["factor_panel"] = mom
    elif template_id == "event_study":
        # event_study needs return_panel + event_panel. Event panel
        # without real corporate-action data isn't doable yet.
        raise NotImplementedError(
            "event_study real-data path needs event_panel (e.g. SUE_panel + "
            "ann_dates) which has no fetcher wired yet."
        )
    elif template_id == "term_structure":
        # term_structure needs yield_panel — need rates_settle which is
        # in _NOT_YET_FETCHABLE.
        raise NotImplementedError(
            "term_structure real-data path needs yield_panel from "
            "rates_settle (not yet wired)."
        )
    elif template_id == "cross_asset_tsmom":
        # Needs futures settle panels — _NOT_YET_FETCHABLE.
        raise NotImplementedError(
            "cross_asset_tsmom needs futures settle data (not yet wired)."
        )
    else:
        raise NotImplementedError(
            f"template {template_id!r} has no panel-builder in data_resolver yet."
        )

    return panels


# ── Public: resolve + simulate ──────────────────────────────────────────

def can_resolve(mechanism_yaml: dict) -> tuple[bool, str | None]:
    """Pre-flight check: can this mechanism's required_data be fetched?
    Returns (yes_we_can, reason_if_not). Cheap — no actual fetches."""
    exec_tpl = mechanism_yaml.get("execution_template") or {}
    if not exec_tpl.get("template_id"):
        return False, "no execution_template.template_id"
    required = mechanism_yaml.get("required_data") or []
    if not required:
        return False, "empty required_data"
    template_id = exec_tpl["template_id"]
    if template_id not in ("equity_xsmom", "factor_quartile", "dispersion"):
        return False, (
            f"template {template_id!r} has no real-data path yet "
            f"(v1 covers equity_xsmom / factor_quartile / dispersion only)"
        )
    blocked = [t for t in required if t in _NOT_YET_FETCHABLE]
    if blocked:
        return False, f"tokens {blocked} have no fetcher wired"
    unknown = [t for t in required
                  if t not in _TOKEN_FETCHERS and t not in _NOT_YET_FETCHABLE]
    if unknown:
        return False, f"unknown tokens {unknown}"
    return True, None
