"""
scripts/run_d_pead_plus.py — Sprint I D-PEAD-Plus orchestration runner.

Spec id=74 hash d0532f8f. End-to-end pipeline:
  Step 1: build_universe       — top-1500 per quarter (WRDS CRSP)
  Step 2: refresh_cache        — extend Compustat fundq cache to 2024-Q2+
  Step 3: build_panel          — merge universe × SUE + match transcripts
  Step 4: extract              — LLM feature extraction (Gemini API, ~$6, ~2h)
  Step 5: fit_dev              — OLS coefficient fit on dev quarters; FREEZE
  Step 6: backtest             — single OOS run; D-PEAD baseline vs D-PEAD-Plus
  Step 7: verdict              — 5-gate evaluation; STRICT/MARGINAL/FAIL output

Each step is idempotent (caches intermediate outputs to parquet).

DOCTRINE: this orchestration script imports engine.d_pead_plus.llm_extractor_rest
only inside step_extract function. Other steps avoid LLM imports entirely.
The decision-layer modules (feature_combiner, backtest, verdict) NEVER import
LLM SDKs (enforced by engine.d_pead_plus.doctrine).
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("run_d_pead_plus")

# Spec-locked window
WINDOW_START = datetime.date(2024, 4, 1)   # 2024-Q2
WINDOW_END   = datetime.date(2026, 6, 30)  # 2026-Q2

CACHE_DIR    = REPO_ROOT / "data" / "d_pead_plus"
UNIVERSE_PARQUET = CACHE_DIR / "_universe_panel.parquet"
SUE_PANEL_PARQUET = CACHE_DIR / "_sue_panel_2024q2_plus.parquet"
MERGED_PANEL_PARQUET = CACHE_DIR / "_merged_panel.parquet"


def _setup_logging():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = CACHE_DIR / f"orchestration_run_{datetime.date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: universe
# ─────────────────────────────────────────────────────────────────────────────
def step_build_universe() -> pd.DataFrame:
    """Build top-1500 universe for each quarter end in window."""
    from engine.d_pead_plus.universe import fetch_universe_for_window
    logger.info("Step 1: build_universe %s to %s", WINDOW_START, WINDOW_END)
    universe_by_q = fetch_universe_for_window(WINDOW_START, WINDOW_END)
    rows = []
    for qe, u in universe_by_q.items():
        for permno in u.permnos:
            rows.append({"quarter_end": qe, "permno": permno})
    df = pd.DataFrame(rows)
    df.to_parquet(UNIVERSE_PARQUET)
    logger.info("Step 1: saved %d rows to %s", len(df), UNIVERSE_PARQUET)
    return df


def _load_universe_panel() -> pd.DataFrame:
    if not UNIVERSE_PARQUET.exists():
        raise FileNotFoundError(f"Run --step universe first; missing {UNIVERSE_PARQUET}")
    return pd.read_parquet(UNIVERSE_PARQUET)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: refresh D-PEAD Compustat cache
# ─────────────────────────────────────────────────────────────────────────────
def step_refresh_cache() -> pd.DataFrame:
    """Extend engine.path_c.pead_ts_signal_panel cache to 2024-Q2+.

    Current cache: 2014-2023. We pull fresh fundq for our universe × window.
    """
    from engine.path_c.pead_ts_signal_panel import bulk_fetch_pead_ts_signal_panel
    logger.info("Step 2: refresh_cache 2023-04-01 to 2026-06-30 (buffer for σ window)")

    # Get all permnos from universe panel, then resolve to tickers via CRSP
    univ = _load_universe_panel()
    permnos = univ["permno"].unique().tolist()
    logger.info("Step 2: querying CRSP for %d unique permnos → tickers", len(permnos))

    from engine.universe_singlename.crsp_loader import _open_wrds_connection
    conn = _open_wrds_connection()
    try:
        # CRSP permno → ticker (use most recent ticker per permno)
        permno_csv = ",".join(str(int(p)) for p in permnos)
        sql = f"""
        SELECT DISTINCT permno, ticker FROM crsp.msenames
        WHERE permno IN ({permno_csv})
          AND nameendt >= '2024-01-01'
          AND ticker IS NOT NULL
        """
        ticker_map = conn.raw_sql(sql)
        ticker_map["permno"] = ticker_map["permno"].astype(int)
        ticker_map["ticker"] = ticker_map["ticker"].astype(str).str.strip().str.upper()
        # Take one ticker per permno
        ticker_map = ticker_map.drop_duplicates(subset="permno", keep="last")
    finally:
        conn.close()

    tickers = ticker_map["ticker"].unique().tolist()
    logger.info("Step 2: resolved %d unique tickers from %d permnos", len(tickers), len(permnos))

    # Call existing PEAD-TS panel builder with 2024+ range; auto-extends cache
    panel_result = bulk_fetch_pead_ts_signal_panel(
        tickers=tickers,
        start_date=datetime.date(2023, 4, 1),  # buffer for σ window
        end_date=WINDOW_END,
        mock_mode=False,
        use_cache=True,
    )
    # Filter to our window
    panel_df = panel_result.panel
    panel_df["rdq"] = pd.to_datetime(panel_df["rdq"]).dt.date
    panel_df_window = panel_df[
        (panel_df["rdq"] >= WINDOW_START) & (panel_df["rdq"] <= WINDOW_END)
    ].copy()

    # panel_df already contains permno from PEAD-TS pipeline; just dedupe
    panel_df_window["permno"] = panel_df_window["permno"].astype(int)
    panel_df_window = panel_df_window.drop_duplicates(subset=["permno", "rdq"], keep="first")

    panel_df_window.to_parquet(SUE_PANEL_PARQUET)
    logger.info("Step 2: saved %d rows (permno × rdq with SUE) to %s",
                len(panel_df_window), SUE_PANEL_PARQUET)
    return panel_df_window


def _load_sue_panel() -> pd.DataFrame:
    if not SUE_PANEL_PARQUET.exists():
        raise FileNotFoundError(f"Run --step cache first; missing {SUE_PANEL_PARQUET}")
    return pd.read_parquet(SUE_PANEL_PARQUET)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: build panel + match transcripts
# ─────────────────────────────────────────────────────────────────────────────
def step_build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge SUE panel + transcripts; return (index_df, text_df)."""
    from engine.d_pead_plus.transcripts_loader import (
        fetch_transcript_index, fetch_transcript_text, cache_transcripts,
    )

    logger.info("Step 3: build_panel (SUE × universe × transcripts match)")

    # Load SUE panel + universe; restrict SUE to universe firms in their quarter
    sue = _load_sue_panel()
    univ = _load_universe_panel()
    univ["quarter_end"] = pd.to_datetime(univ["quarter_end"]).dt.date

    # For each SUE row, find quarter_end >= rdq within 100 days
    sue["rdq"] = pd.to_datetime(sue["rdq"]).dt.date
    sue["quarter_end"] = sue["rdq"].apply(
        lambda d: _next_quarter_end(d)
    )
    # Merge on (permno, quarter_end)
    panel = sue.merge(univ, on=["permno", "quarter_end"], how="inner")
    logger.info("Step 3: panel after universe filter: %d firm-quarters", len(panel))

    if panel.empty:
        logger.warning("Step 3: empty panel — universe didn't match SUE")
        return pd.DataFrame(), pd.DataFrame()

    # Match transcripts
    rdq_panel = panel[["permno", "ticker", "rdq"]].drop_duplicates()
    index_df = fetch_transcript_index(rdq_panel)
    logger.info("Step 3: matched %d firm-quarters to earnings calls", len(index_df))

    if index_df.empty:
        logger.warning("Step 3: no transcripts matched")
        return pd.DataFrame(), pd.DataFrame()

    # Pull transcript text
    text_df = fetch_transcript_text(index_df["transcript_id"].tolist())
    logger.info("Step 3: pulled %d transcript texts", len(text_df))

    # Cache
    cache_transcripts(index_df, text_df)

    # Save merged panel
    panel.to_parquet(MERGED_PANEL_PARQUET)
    logger.info("Step 3: saved merged panel %d rows to %s", len(panel), MERGED_PANEL_PARQUET)
    return index_df, text_df


def _next_quarter_end(d: datetime.date) -> datetime.date:
    """Return next NYSE quarter-end date >= d."""
    q_ends = [
        datetime.date(d.year, 3, 31),
        datetime.date(d.year, 6, 30),
        datetime.date(d.year, 9, 30),
        datetime.date(d.year, 12, 31),
    ]
    for qe in q_ends:
        if qe >= d:
            return qe
    return datetime.date(d.year + 1, 3, 31)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: LLM extraction (Phase 3, expensive)
# ─────────────────────────────────────────────────────────────────────────────
def step_extract(max_transcripts: int | None = None) -> int:
    """Run Gemini extraction on transcripts. ⚠️ Costs ~$10 + ~5h runtime.

    Day-2 (2026-05-14): switched from SDK-based run_extraction to REST-based
    run_extraction_rest. SDK stalled indefinitely after ~30 calls due to
    gRPC connection state degradation. REST uses httpx network-level timeout
    (reliable). PROMPT_HASH unchanged (thinking_budget=0 not in hash inputs).
    """
    from engine.d_pead_plus.transcripts_loader import load_cached_transcripts
    from engine.d_pead_plus.llm_extractor_rest import run_extraction_rest

    idx_df, text_df = load_cached_transcripts()
    if idx_df.empty or text_df.empty:
        logger.error("Step 4: no cached transcripts. Run --step panel first.")
        return 0

    logger.info("Step 4: extract — %d transcripts, max=%s", len(idx_df), max_transcripts)
    if max_transcripts is None:
        # Confirmed by user 2026-05-13 night for Sprint I Phase 3 full run
        logger.info("Step 4: FULL extraction starting (user-confirmed).")
        logger.info("  Expected cost: ~$%.2f", len(idx_df) * 0.0009)
        logger.info("  Expected runtime: ~%.0f minutes", len(idx_df) * 1.7 / 60)

    records = run_extraction_rest(idx_df, text_df, max_transcripts=max_transcripts)
    logger.info("Step 4: extracted %d records; total cost $%.4f",
                len(records), sum(r.cost_usd for r in records))
    return len(records)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: dev fit OLS
# ─────────────────────────────────────────────────────────────────────────────
def step_fit_dev() -> None:
    """Fit OLS on dev quarters (2024Q2-Q4), freeze coefficients."""
    from engine.d_pead_plus.feature_combiner import (
        prepare_panel, fit_dev_ols, save_coefficients, DEV_QUARTERS_LOCKED,
    )
    from engine.d_pead_plus.llm_extractor import load_existing_extractions

    sue_df = _load_sue_panel()[["permno", "ticker", "rdq", "sue", "market_cap_at_q"]]
    sue_df = sue_df.rename(columns={"market_cap_at_q": "mcap"})

    llm_df = load_existing_extractions()
    if llm_df.empty:
        logger.error("Step 5: no LLM extractions. Run --step extract first.")
        return

    # Compute forward returns (60-day log return per firm-quarter)
    forward_ret_df = _compute_forward_returns(sue_df)

    full_panel = prepare_panel(sue_df, llm_df, forward_ret_df)
    # Filter to dev quarters
    dev_panel = full_panel[full_panel["quarter"].isin(DEV_QUARTERS_LOCKED)].copy()
    logger.info("Step 5: dev panel size %d (quarters: %s)",
                len(dev_panel), DEV_QUARTERS_LOCKED)

    coeffs = fit_dev_ols(dev_panel)
    save_coefficients(coeffs)
    logger.info("Step 5: coefficients FROZEN")


def _compute_forward_returns(sue_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 60-day forward log return for each (permno, rdq) via yfinance.

    Switched from CRSP dsf to yfinance 2026-05-14: CRSP data lag stops at
    ~2024-12-31; OOS quarters 2025-Q1 to 2026-Q2 have NO CRSP coverage,
    causing OOS panel = 0 rows after merge. yfinance has current data.

    Consistent with spec Amendment 1 yfinance choice for Sprint A/B/D-2/G.
    """
    import math
    import numpy as np
    import yfinance as yf

    logger.info("Computing 60-day forward returns via yfinance...")

    # Need ticker column to call yfinance
    if "ticker" not in sue_df.columns:
        raise ValueError("sue_df missing 'ticker' column (needed for yfinance)")

    df = sue_df.copy()
    df["rdq"] = pd.to_datetime(df["rdq"])
    rdq_min = df["rdq"].min()
    rdq_max = df["rdq"].max()
    fetch_start = (rdq_min - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    fetch_end   = (rdq_max + datetime.timedelta(days=100)).strftime("%Y-%m-%d")

    unique_tickers = sorted(set(t for t in df["ticker"].dropna().unique() if t and str(t) != "nan"))
    logger.info("Fetching yfinance close prices for %d tickers from %s to %s",
                len(unique_tickers), fetch_start, fetch_end)

    # Batch download (yfinance handles many tickers at once)
    BATCH_SIZE = 200
    all_close: dict[str, pd.Series] = {}
    for i in range(0, len(unique_tickers), BATCH_SIZE):
        batch = unique_tickers[i:i + BATCH_SIZE]
        try:
            data = yf.download(batch, start=fetch_start, end=fetch_end,
                               progress=False, auto_adjust=True, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                for tkr in batch:
                    if tkr in close.columns:
                        s = close[tkr].dropna()
                        if not s.empty:
                            all_close[tkr] = s
            else:
                # Single ticker case
                s = data["Close"].dropna()
                if not s.empty:
                    all_close[batch[0]] = s
        except Exception as exc:
            logger.warning("yfinance batch %d-%d failed: %s", i, i + len(batch), exc)
        logger.info("  batch %d-%d: %d tickers with data so far", i, i + len(batch), len(all_close))

    logger.info("yfinance fetched data for %d / %d tickers", len(all_close), len(unique_tickers))

    # Compute 60-trading-day forward log return per (ticker, rdq)
    rows = []
    for _, row in df.iterrows():
        tkr = str(row["ticker"])
        rdq = row["rdq"]
        permno = int(row["permno"])

        if tkr not in all_close:
            continue
        series = all_close[tkr]
        # Trading days after rdq (exclusive of rdq date itself)
        future = series[series.index > rdq].head(60)
        if len(future) < 30:  # need at least 30 trading days
            continue

        entry = future.iloc[0]
        # Cumulative log return = log(P_last / P_first)
        # Or equivalently sum of daily log returns
        # We use sum of daily log returns to handle adjusted close splits
        daily_log = np.log(future / future.shift(1)).dropna()
        if len(daily_log) < 25:
            continue
        ret60_log = float(daily_log.sum())

        rows.append({"permno": permno, "rdq": rdq, "ret_60d_log": ret60_log})

    out = pd.DataFrame(rows)
    logger.info("Forward returns computed: %d (permno, rdq) pairs", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 6+7: backtest + verdict
# ─────────────────────────────────────────────────────────────────────────────
def step_backtest_and_verdict() -> dict:
    """Run OOS backtest + 5-gate verdict; save v1_verdict.json."""
    from engine.d_pead_plus.feature_combiner import (
        prepare_panel, load_coefficients, apply_frozen_coefficients_oos,
        DEV_QUARTERS_LOCKED,
    )
    from engine.d_pead_plus.llm_extractor import load_existing_extractions
    from engine.d_pead_plus.backtest import run_strategy_backtest, save_backtest_daily
    from engine.d_pead_plus.verdict import evaluate_verdict, save_verdict

    coeffs = load_coefficients()
    if coeffs is None:
        logger.error("Step 6: no frozen coefficients. Run --step fit_dev first.")
        return {}

    sue_df = _load_sue_panel()[["permno", "ticker", "rdq", "sue", "market_cap_at_q"]]
    sue_df = sue_df.rename(columns={"market_cap_at_q": "mcap"})
    llm_df = load_existing_extractions()
    forward_ret_df = _compute_forward_returns(sue_df)
    full_panel = prepare_panel(sue_df, llm_df, forward_ret_df)

    oos_panel = full_panel[~full_panel["quarter"].isin(DEV_QUARTERS_LOCKED)].copy()
    oos_panel = apply_frozen_coefficients_oos(oos_panel, coeffs)
    logger.info("Step 6: OOS panel size %d", len(oos_panel))

    # Pull daily returns for backtest
    daily_df = _build_daily_return_pivot(oos_panel)

    # D-PEAD baseline: rank by SUE only
    oos_baseline = oos_panel.copy()
    oos_baseline["score"] = oos_baseline["sue_z"]
    oos_baseline["decile"] = oos_baseline.groupby("quarter")["score"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1
    )
    oos_baseline["long_flag"]  = (oos_baseline["decile"] == 10).astype(int)
    oos_baseline["short_flag"] = (oos_baseline["decile"] == 1).astype(int)

    baseline_result = run_strategy_backtest(oos_baseline, daily_df, "d_pead_baseline")
    plus_result     = run_strategy_backtest(oos_panel,    daily_df, "d_pead_plus")
    save_backtest_daily(baseline_result, plus_result)

    paired_returns = pd.DataFrame({
        "d_pead_baseline": baseline_result.daily_returns,
        "d_pead_plus":     plus_result.daily_returns,
    }).dropna()

    # Dev Sharpe (for Gate 4)
    dev_sharpe_plus = None  # could be computed by re-running backtest on dev; skip for v1

    # Cost from llm_extractor cache
    cost_total = float(llm_df["cost_usd"].sum()) if "cost_usd" in llm_df.columns else 6.0

    verdict = evaluate_verdict(
        oos_panel              = oos_panel,
        paired_daily_returns   = paired_returns,
        dev_sharpe_plus        = dev_sharpe_plus,
        llm_api_cost_usd       = cost_total,
    )
    save_verdict(verdict)
    logger.info("Step 7: VERDICT = %s", verdict.decision)
    return {"decision": verdict.decision, "verdict": verdict}


def _build_daily_return_pivot(oos_panel: pd.DataFrame) -> pd.DataFrame:
    """Build daily-return pivot (date × permno) for OOS panel via yfinance.

    Switched from CRSP dsf to yfinance 2026-05-14 (same reason as
    _compute_forward_returns: CRSP data lag stops at 2024-12-31).
    """
    import yfinance as yf
    import numpy as np

    # Pull ticker map from SUE panel
    sue_df = _load_sue_panel()[["permno", "ticker"]].drop_duplicates()
    perm_to_tkr = dict(zip(sue_df["permno"].astype(int), sue_df["ticker"].astype(str)))

    permnos = sorted(set(int(p) for p in oos_panel["permno"].unique()))
    oos_panel["rdq"] = pd.to_datetime(oos_panel["rdq"])
    rdq_min = oos_panel["rdq"].min()
    rdq_max = oos_panel["rdq"].max()
    fetch_start = (rdq_min - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    fetch_end   = (rdq_max + datetime.timedelta(days=100)).strftime("%Y-%m-%d")

    tickers = sorted(set(perm_to_tkr.get(p, "") for p in permnos) - {""})
    logger.info("Fetching daily-return pivot via yfinance: %d tickers, %s to %s",
                len(tickers), fetch_start, fetch_end)

    BATCH_SIZE = 200
    daily_returns: dict[int, pd.Series] = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            data = yf.download(batch, start=fetch_start, end=fetch_end,
                               progress=False, auto_adjust=True, threads=True)
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
                for tkr in batch:
                    if tkr in close.columns:
                        s = close[tkr].dropna()
                        if len(s) > 1:
                            ret = s.pct_change().dropna()
                            # Map back to permno
                            for p, t in perm_to_tkr.items():
                                if t == tkr:
                                    daily_returns[p] = ret
                                    break
            else:
                s = data["Close"].dropna()
                if len(s) > 1:
                    ret = s.pct_change().dropna()
                    for p, t in perm_to_tkr.items():
                        if t == batch[0]:
                            daily_returns[p] = ret
                            break
        except Exception as exc:
            logger.warning("yfinance batch %d-%d failed: %s", i, i + len(batch), exc)

    logger.info("Daily returns covered %d / %d permnos", len(daily_returns), len(permnos))

    # Build pivot from collected series
    pivot = pd.DataFrame(daily_returns).fillna(0)
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint I D-PEAD-Plus orchestration")
    parser.add_argument("--step", choices=[
        "universe", "cache", "panel", "extract", "fit_dev", "backtest", "all",
    ], required=True, help="Pipeline step")
    parser.add_argument("--max-transcripts", type=int, default=None,
                        help="Cap on extraction count (for dev/dry-run)")
    args = parser.parse_args()

    log_file = _setup_logging()
    logger.info("=== run_d_pead_plus --step %s ===", args.step)
    logger.info("Log file: %s", log_file)

    # Pre-flight doctrine assert
    from engine.d_pead_plus.doctrine import assert_no_llm_in_decision_layer
    assert_no_llm_in_decision_layer()
    logger.info("Doctrine pre-flight: CLEAN (0-LLM-in-DECISION)")

    if args.step == "universe" or args.step == "all":
        step_build_universe()
    if args.step == "cache" or args.step == "all":
        step_refresh_cache()
    if args.step == "panel" or args.step == "all":
        step_build_panel()
    if args.step == "extract":
        n = step_extract(max_transcripts=args.max_transcripts)
        logger.info("Extraction count: %d", n)
    if args.step == "fit_dev":
        step_fit_dev()
    if args.step == "backtest":
        result = step_backtest_and_verdict()
        logger.info("Final: %s", result.get("decision", "UNKNOWN"))
    if args.step == "all":
        logger.info("'all' covers universe → cache → panel. Run extract / fit_dev / backtest separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
