"""engine/data/dsl_adapter.py — bridge fetcher long-format → DSL wide-panel.

Layer between Layer 0 fetchers (which return canonical long-format DataFrames
with date/ticker/value columns) and Layer 1 templates (which expect wide-panel
DataFrames indexed by date with tickers as columns).

This adapter lives BETWEEN data engine (Phase 6) and DSL (Phase 3). Adding
a new token-shape combo requires registering a new adapter function.

Doctrine:
- Pure functions — no state, no I/O
- Per-token-and-template-id routing (different templates may want different shapes)
- Returns dict {dsl_kwarg_name: pd.DataFrame} ready to merge into data_kwargs
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _equity_long_to_wide(df: pd.DataFrame) -> dict:
    """Equity long format (date, ticker|permno_or_ticker, prc, ret) → wide panels.

    Returns:
      price_panel:  DataFrame (date × ticker) of adjusted prices
      return_panel: DataFrame (date × ticker) of monthly/daily returns
    """
    if df.empty:
        return {"price_panel": pd.DataFrame(), "return_panel": pd.DataFrame()}
    ticker_col = ("ticker" if "ticker" in df.columns
                   else "permno_or_ticker" if "permno_or_ticker" in df.columns
                   else None)
    if ticker_col is None:
        raise ValueError("equity DataFrame must have 'ticker' or 'permno_or_ticker' column")
    price_panel = df.pivot_table(index="date", columns=ticker_col,
                                    values="prc", aggfunc="last")
    return_panel = df.pivot_table(index="date", columns=ticker_col,
                                     values="ret", aggfunc="last")
    return {"price_panel": price_panel, "return_panel": return_panel}


def _futures_returns_to_wide(df: pd.DataFrame) -> dict:
    """Futures long format (date, futcode, settle) → wide settle panel and
    return panel (computed from settle).
    """
    if df.empty:
        return {"return_panel": pd.DataFrame()}
    settle_panel = df.pivot_table(index="date", columns="futcode",
                                     values="settle", aggfunc="last")
    return_panel = settle_panel.pct_change()
    return {"return_panel": return_panel}


def _passthrough_long(df: pd.DataFrame, kwarg_name: str) -> dict:
    """Long-format passthrough for templates that take the data as-is."""
    return {kwarg_name: df}


# ── Token → adapter routing ─────────────────────────────────────────────

# Each (token, template_id) → adapter function
# If a template_id is None, the adapter applies for ANY template
# More specific (token, template_id) wins over (token, None)
_ADAPTERS = {
    # Equity (CRSP DSF / MSF / yfinance) → equity_xsmom + factor_quartile
    ("crsp_dsf", None):  _equity_long_to_wide,
    ("crsp_msf", None):  _equity_long_to_wide,

    # Futures (TR_DS_FUT settle) → cross_asset_tsmom
    ("tr_ds_fut_settle", "cross_asset_tsmom"): _futures_returns_to_wide,
    ("cmdty_settle", "cross_asset_tsmom"):     _futures_returns_to_wide,
    ("fx_settle", "cross_asset_tsmom"):        _futures_returns_to_wide,
    ("rates_settle", "cross_asset_tsmom"):     _futures_returns_to_wide,
    ("rates_xc_settle", "cross_asset_tsmom"):  _futures_returns_to_wide,
    ("eqidx_settle", "cross_asset_tsmom"):     _futures_returns_to_wide,

    # FRED macro / VIX → passthrough
    ("fred_macro", None): lambda df: _passthrough_long(df, "macro_panel"),
    ("vix_index", None):  lambda df: _passthrough_long(df, "vix_panel"),
    ("vix3m_index", None): lambda df: _passthrough_long(df, "vix3m_panel"),
}


def adapt_for_dsl(token: str, df: pd.DataFrame,
                    template_id: str | None = None) -> dict:
    """Convert one fetcher output to DSL template input shape.

    Args:
      token:       data inventory token (e.g. crsp_dsf)
      df:          long-format DataFrame from fetcher
      template_id: the Layer 1 template this will feed (optional, for
                    template-specific routing)

    Returns:
      dict of {dsl_kwarg_name: DataFrame} to merge into data_kwargs
    """
    # 1. Try (token, template_id) specific
    if (token, template_id) in _ADAPTERS:
        return _ADAPTERS[(token, template_id)](df)
    # 2. Fall back to (token, None) generic
    if (token, None) in _ADAPTERS:
        return _ADAPTERS[(token, None)](df)
    # 3. No adapter — passthrough as a kwarg named after the token
    logger.warning("no adapter for token %r template %r; passthrough as kwarg %r",
                    token, template_id, token)
    return {token: df}


def assemble_dsl_kwargs(token_dfs: dict[str, pd.DataFrame],
                          template_id: str | None = None) -> dict:
    """Convert multiple token DataFrames into a merged DSL data_kwargs.

    Args:
      token_dfs:   {token: DataFrame} from orchestrator's assemble_data_kwargs
      template_id: the Layer 1 template id

    Returns:
      data_kwargs dict ready to pass as **data_kwargs to DSL runner
    """
    out: dict = {}
    for token, df in token_dfs.items():
        adapted = adapt_for_dsl(token, df, template_id)
        # Later tokens override earlier on key collision
        # (e.g. crsp_dsf and yfinance both provide price_panel — last wins)
        out.update(adapted)
    return out


def list_adapters() -> list[tuple[str, str | None]]:
    """List registered (token, template_id) adapter keys."""
    return sorted(_ADAPTERS.keys(), key=lambda x: (x[0], x[1] or ""))
