"""
External data sources — one module per provider.

Modules
-------
  cftc_cot   — CFTC Commitments of Traders (Disaggregated, futures-only)
               weekly speculator vs commercial positioning. Free; weekly.
               Used by P3c (COT-conditional BAB extension test).

Sources NOT in this package (kept at engine/ root for historical reasons):
  macro_fetcher.py     - FRED macro indicators
  news_fetcher.py      - news headlines (GNews / SEC EDGAR / RSS)
  news_fetchers.py     - sentiment (Alpha Vantage)
  data_snapshot.py     - yfinance + cache (per-ticker)

New sources go here. Convention: one module per provider, public API
exposes (fetch_*, parse_*, upsert_*) functions plus any provider-specific
constants. Do not import streamlit here — modules must be runnable from
plain CLI scripts.
"""
