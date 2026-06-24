"""
scripts/investigate_dd.py — One-command DD investigation tool.

Phase 1 MVP (today): Queries Sprint H PaperTradeTradeLog + fetches yfinance
realized returns → finds top-N worst trades → calls per-trade LLM forensic
verdict → renders integrated markdown report → writes to docs/.

Phase 2 deterministic helpers (auto-gate when data insufficient):
  - Brinson 3-layer attribution
  - FF5 factor decomposition
  - Memmel Z forward decay (needs 30d trade data)
  - Forward IC degradation (needs 60d trade data)
  - P&L 30/60/90d timeseries

Phase 2 bootstrap baseline + Phase 3 components DEFERRED — see
project_investigate_dd_phase23_deferred_2026-05-13.md for trigger conditions.

DOCTRINE: All LLM calls happen in forensic layer (engine.forensic.*), output
never feeds back into decision layer. 0-LLM-in-DECISION preserved.

USAGE:
  py -3.11 scripts/investigate_dd.py --date 2026-05-13
  py -3.11 scripts/investigate_dd.py --date 2026-05-13 --top-n 5 --output docs/
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("investigate_dd")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Query Sprint H trade log + fetch realized returns
# ─────────────────────────────────────────────────────────────────────────────

def step_query_trades(as_of: datetime.date) -> pd.DataFrame:
    """Query Sprint H PaperTradeTradeLog for the target date."""
    from engine.portfolio.attribution_logger import query_trade_log
    df = query_trade_log(date_start=as_of, date_end=as_of)
    if df.empty:
        logger.warning("No Sprint H rows for %s. Daily orchestrator may not have run.", as_of)
    return df


def step_fetch_realized_returns(
    trade_df: pd.DataFrame,
    as_of:    datetime.date,
    horizon:  int = 5,
) -> pd.DataFrame:
    """Augment trade_df with realized N-day return per ticker.

    Returns trade_df with new column `realized_N_d_return` (NaN if not yet computable).
    """
    if trade_df.empty:
        return trade_df

    import yfinance as yf

    df = trade_df.copy()
    df[f"realized_{horizon}d_return"] = float("nan")

    # Filter actual tickers (skip permno_* placeholders from Path N backtest mode)
    real_tickers = sorted({
        t for t in df["ticker"].unique()
        if t and not t.startswith("permno_") and t != "PQTIX"
    })

    if not real_tickers:
        logger.info("No real tickers to fetch returns for (only permno/PQTIX entries)")
        return df

    start = as_of
    end   = as_of + datetime.timedelta(days=horizon + 7)  # buffer for weekends

    try:
        prices = yf.download(
            real_tickers, start=start, end=end, progress=False, auto_adjust=True,
        )
        # yfinance returns multi-index columns for multi-ticker; single-ticker returns flat
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            close = prices[["Close"]].rename(columns={"Close": real_tickers[0]})
    except Exception as exc:
        logger.warning("yfinance fetch failed: %s; realized returns unavailable", exc)
        return df

    # For each ticker, get first available open price and N-day forward
    for ticker in real_tickers:
        if ticker not in close.columns:
            continue
        series = close[ticker].dropna()
        if len(series) < 2:
            continue
        entry = float(series.iloc[0])
        if len(series) < horizon + 1:
            # Not enough forward data yet → leave NaN (insufficient data)
            continue
        exit_price = float(series.iloc[min(horizon, len(series) - 1)])
        if entry > 0:
            ret = (exit_price - entry) / entry
            df.loc[df["ticker"] == ticker, f"realized_{horizon}d_return"] = ret

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Strategy contribution + top-N worst trades
# ─────────────────────────────────────────────────────────────────────────────

def step_strategy_contributions(trade_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Aggregate weight × realized_return per strategy."""
    ret_col = f"realized_{horizon}d_return"
    if ret_col not in trade_df.columns or trade_df.empty:
        return pd.DataFrame(columns=["strategy_name", "n_trades", "contribution"])

    df = trade_df.copy()
    df["contribution"] = df["weight"] * df[ret_col].fillna(0)
    out = df.groupby("strategy_name").agg(
        n_trades=("ticker", "count"),
        contribution=("contribution", "sum"),
        n_with_returns=(ret_col, lambda s: s.notna().sum()),
    ).reset_index().sort_values("contribution")
    return out


def step_top_n_worst(
    trade_df: pd.DataFrame,
    top_n:    int = 3,
    horizon:  int = 5,
) -> pd.DataFrame:
    """Find top-N worst trades by weight × realized_return."""
    ret_col = f"realized_{horizon}d_return"
    if ret_col not in trade_df.columns or trade_df.empty:
        return pd.DataFrame()

    df = trade_df.copy()
    df = df.dropna(subset=[ret_col])
    if df.empty:
        return df

    df["contribution"] = df["weight"] * df[ret_col]
    return df.nsmallest(top_n, "contribution")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Per-trade forensic LLM verdict
# ─────────────────────────────────────────────────────────────────────────────

def step_forensic_per_trade(worst_df: pd.DataFrame, horizon: int = 5) -> list:
    """Call engine.forensic.news_context.investigate_trade() per worst trade."""
    if worst_df.empty:
        return []

    from engine.forensic.news_context import investigate_trade

    summaries = []
    for _, row in worst_df.iterrows():
        ret_col = f"realized_{horizon}d_return"
        try:
            summary = investigate_trade(
                date                  = pd.to_datetime(row["date"]).date(),
                ticker                = str(row["ticker"]),
                signal_value          = (None if pd.isna(row["signal_value"]) else float(row["signal_value"])),
                weight                = float(row["weight"]),
                realized_return       = (None if pd.isna(row.get(ret_col)) else float(row[ret_col])),
                strategy_name         = str(row["strategy_name"]),
                expected_horizon_days = int(row["expected_horizon_days"]),
            )
            summaries.append(summary)
        except Exception as exc:
            logger.warning("forensic per-trade %s failed: %s", row["ticker"], exc)
    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Phase 2 deterministic helpers (auto-gate if data insufficient)
# ─────────────────────────────────────────────────────────────────────────────

def step_phase2_helpers(as_of: datetime.date, trade_df: pd.DataFrame) -> dict:
    """Invoke each Phase 2 helper; collect results (OK or INSUFFICIENT_DATA)."""
    results: dict = {}

    # Brinson 3-layer attribution
    try:
        from engine.forensic.brinson import compute_brinson_attribution
        results["brinson"] = compute_brinson_attribution(trade_df)
    except ImportError:
        results["brinson"] = {"status": "MODULE_MISSING"}

    # FF5 factor decomposition
    try:
        from engine.forensic.factor_decomp import compute_ff5_decomp
        results["ff5_decomp"] = compute_ff5_decomp(trade_df, as_of)
    except ImportError:
        results["ff5_decomp"] = {"status": "MODULE_MISSING"}

    # Memmel Z forward decay
    try:
        from engine.forensic.strategy_decay import compute_memmel_z_per_strategy
        results["memmel_z"] = compute_memmel_z_per_strategy(as_of)
    except ImportError:
        results["memmel_z"] = {"status": "MODULE_MISSING"}

    # Forward IC degradation
    try:
        from engine.forensic.forward_ic import compute_forward_ic_per_strategy
        results["forward_ic"] = compute_forward_ic_per_strategy(as_of)
    except ImportError:
        results["forward_ic"] = {"status": "MODULE_MISSING"}

    # P&L 30/60/90d timeseries
    try:
        from engine.forensic.pnl_timeseries import compute_pnl_trailing
        results["pnl_trailing"] = compute_pnl_trailing(as_of)
    except ImportError:
        results["pnl_trailing"] = {"status": "MODULE_MISSING"}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Render markdown report
# ─────────────────────────────────────────────────────────────────────────────

def render_report(
    as_of:           datetime.date,
    trade_df:        pd.DataFrame,
    contribs:        pd.DataFrame,
    worst_df:        pd.DataFrame,
    forensic_list:   list,
    phase2_results:  dict,
    horizon:         int = 5,
) -> str:
    """Render final markdown report."""
    lines = [f"# DD Investigation Report — {as_of.isoformat()}\n"]

    # Section 1: Executive summary placeholder (real synthesis below)
    if trade_df.empty:
        lines.append("> **NO DATA** — Sprint H `paper_trade_trade_log` empty for this date.")
        lines.append("> Daily orchestrator likely did not run. Check `scripts/run_paper_trade_daily.py` log.\n")
        return "\n".join(lines)

    n_trades_total = len(trade_df)
    ret_col = f"realized_{horizon}d_return"
    n_with_returns = trade_df[ret_col].notna().sum() if ret_col in trade_df.columns else 0
    lines.append(
        f"_Sprint H trade log: **{n_trades_total} trades** across "
        f"**{trade_df['strategy_name'].nunique()} strategies**; "
        f"**{n_with_returns}/{n_trades_total}** have T+{horizon}d realized returns._\n"
    )

    # Section 2: Strategy contribution table
    lines.append("## 1. Strategy Contribution Decomposition\n")
    if contribs.empty or n_with_returns == 0:
        lines.append(f"_No realized {horizon}d returns yet. Returns unlock once T+{horizon}d data available._\n")
    else:
        lines.append("| Strategy | n trades | Contribution | Realized cover |")
        lines.append("|---|---|---|---|")
        for _, row in contribs.iterrows():
            lines.append(
                f"| {row['strategy_name']} | {int(row['n_trades'])} | "
                f"{row['contribution']:+.4%} | {int(row['n_with_returns'])}/{int(row['n_trades'])} |"
            )
        lines.append("")

    # Section 3: Top-N worst trades + forensic verdicts
    lines.append(f"## 2. Top-{len(worst_df)} Worst Trades (Forensic Analysis)\n")
    if worst_df.empty:
        lines.append("_Insufficient realized return data to rank worst trades yet._\n")
    else:
        for i, (_, row) in enumerate(worst_df.iterrows(), 1):
            forensic = forensic_list[i - 1] if i - 1 < len(forensic_list) else None
            lines.append(f"### #{i} {row['ticker']} ({row['strategy_name']})")
            lines.append(
                f"- **Signal:** {row['signal_value']}  |  **Weight:** {row['weight']:+.4f}  "
                f"|  **Event:** {row['event_trigger']}  |  **Horizon:** {row['expected_horizon_days']}d"
            )
            realized = row.get(ret_col)
            if not pd.isna(realized):
                lines.append(f"- **Realized {horizon}d:** {realized:+.2%}  "
                             f"|  **Contribution:** {row['contribution']:+.4%}")

            if forensic is not None:
                verdict_label = {
                    "case_a": "Signal Wrong / Over-fit",
                    "case_b": "Horizon Incomplete",
                    "case_c": "Exogenous Shock",
                }.get(forensic.forensic_verdict, forensic.forensic_verdict)
                lines.append(f"- **Forensic verdict:** `{forensic.forensic_verdict}` ({verdict_label})")
                if forensic.material_events:
                    lines.append("- **Material events:**")
                    for e in forensic.material_events[:3]:
                        lines.append(f"  - {e}")
                lines.append(f"- **Signal alignment:** {forensic.signal_alignment}")
                if forensic.key_quotes:
                    lines.append(f"- **Quote:** > {forensic.key_quotes[0]}")
            else:
                lines.append("- _Forensic verdict unavailable (LLM call failed or skipped)_")
            lines.append("")

    # Section 4: Phase 2 deterministic helpers
    lines.append("## 3. Phase 2 Deterministic Analysis (Auto-Gated)\n")
    for name, result in phase2_results.items():
        status = result.get("status", "UNKNOWN") if isinstance(result, dict) else "UNKNOWN"
        if status == "OK":
            lines.append(f"### {name.upper()} — OK")
            lines.append(f"```json\n{json.dumps(result, indent=2, default=str)[:1500]}\n```\n")
        elif status == "INSUFFICIENT_DATA":
            lines.append(f"### {name.upper()} — ⏳ INSUFFICIENT_DATA")
            have = result.get("have", "?")
            need = result.get("need", "?")
            eta = result.get("eta_unlock", "n/a")
            lines.append(f"_Have {have}, need {need}. ETA unlock: {eta}._\n")
        elif status == "MODULE_MISSING":
            lines.append(f"### {name.upper()} — module not yet built\n")
        else:
            lines.append(f"### {name.upper()} — status: {status}\n")

    # Section 4.5: Synthesis pass (Phase 1.5 — single LLM consolidation)
    if forensic_list:
        try:
            from engine.forensic.news_context import synthesize_dd_report
            # Build contribs list for synthesis (need dicts not DataFrame rows)
            contribs_list = [
                {"strategy_name": r["strategy_name"],
                 "n_trades":      int(r["n_trades"]),
                 "n_with_returns":int(r["n_with_returns"]),
                 "contribution":  float(r["contribution"])}
                for _, r in contribs.iterrows()
            ] if not contribs.empty else []
            synthesis = synthesize_dd_report(
                as_of                = as_of,
                strategy_contribs    = contribs_list,
                forensic_summaries   = forensic_list,
                brinson_result       = phase2_results.get("brinson"),
                factor_decomp_result = phase2_results.get("ff5_decomp"),
            )
        except Exception:
            logger.exception("synthesis pass failed (non-fatal)")
            synthesis = None
    else:
        synthesis = None

    if synthesis is not None:
        # Insert executive summary at top of report (after metadata line)
        executive_section = [
            "## Executive Summary\n",
            f"**TL;DR:** {synthesis.tl_dr}\n",
            f"**Cross-trade pattern:** {synthesis.cross_trade_pattern}\n",
            "**Action priority (rule-mapped):**",
        ]
        for i, a in enumerate(synthesis.action_priority, 1):
            executive_section.append(f"  {i}. {a}")
        executive_section.append("")
        # Insert after the metadata line (line index 1 — after H1 + 1 blank)
        insert_at = 2 if len(lines) > 2 else len(lines)
        for offset, esline in enumerate(executive_section):
            lines.insert(insert_at + offset, esline)

    # Section 5: Action guidance (rule-mapped, not LLM-decided)
    lines.append("## 4. Action Guidance (Rule-Mapped)\n")
    if forensic_list:
        verdicts = [s.forensic_verdict for s in forensic_list]
        n_a = verdicts.count("case_a")
        n_b = verdicts.count("case_b")
        n_c = verdicts.count("case_c")
        lines.append(f"- **case_a** (signal wrong / over-fit): **{n_a}** — flag for Forward IC validation; no new adds on same direction")
        lines.append(f"- **case_b** (horizon incomplete): **{n_b}** — **hold position**, wait until horizon completes before reclassifying")
        lines.append(f"- **case_c** (exogenous shock): **{n_c}** — **hold position**; assess cluster risk in Phase 2 Brinson decomp")
        lines.append("")
        if n_a >= 3:
            lines.append("⚠️ **Pattern flag:** 3+ case_a in one investigation → consider strategy decay audit (Sprint E E-1 forward Sharpe bootstrap)\n")
    else:
        lines.append("_No forensic verdicts available; check upstream errors._\n")

    # Footer
    forensic_cost  = sum(getattr(s, "cost_usd", 0.0) for s in forensic_list)
    synthesis_cost = synthesis.cost_usd if synthesis else 0.0
    total_cost     = forensic_cost + synthesis_cost
    lines.append("---")
    lines.append(
        f"_Generated by `scripts/investigate_dd.py` on {datetime.datetime.utcnow().isoformat()} UTC. "
        f"Sprint H rows: {n_trades_total}. LLM cost: ${total_cost:.4f} "
        f"({len(forensic_list)} forensic + {'1' if synthesis else '0'} synthesis × Gemini 2.5 Flash). "
        f"Doctrine: 0-LLM-in-DECISION preserved (forensic layer only)._"
    )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="One-command DD investigation tool (Sprint H follow-up).")
    p.add_argument("--date", required=True, help="Investigation date YYYY-MM-DD")
    p.add_argument("--top-n", type=int, default=3, help="Number of worst trades to investigate (default 3)")
    p.add_argument("--horizon", type=int, default=5, help="Realized return horizon in days (default 5)")
    p.add_argument("--output", default="docs", help="Output directory for markdown (default docs/)")
    args = p.parse_args()

    as_of = datetime.date.fromisoformat(args.date)
    logger.info("DD investigation — date=%s top_n=%d horizon=%dd", as_of, args.top_n, args.horizon)

    # Step 1: query Sprint H
    trade_df = step_query_trades(as_of)
    logger.info("Step 1: %d Sprint H rows", len(trade_df))

    # Step 2: fetch realized returns
    if not trade_df.empty:
        trade_df = step_fetch_realized_returns(trade_df, as_of, horizon=args.horizon)
        n_with_ret = trade_df[f"realized_{args.horizon}d_return"].notna().sum()
        logger.info("Step 2: realized returns covered %d/%d trades", n_with_ret, len(trade_df))
    else:
        n_with_ret = 0

    # Step 3: contributions + top-N worst
    contribs = step_strategy_contributions(trade_df, horizon=args.horizon)
    worst    = step_top_n_worst(trade_df, top_n=args.top_n, horizon=args.horizon)
    logger.info("Step 3: contribs %d strategies, top-%d worst identified", len(contribs), len(worst))

    # Step 4: per-trade forensic LLM
    forensic_list = step_forensic_per_trade(worst, horizon=args.horizon)
    logger.info("Step 4: forensic LLM × %d (cost $%.4f)",
                len(forensic_list), sum(getattr(s, "cost_usd", 0) for s in forensic_list))

    # Step 5: Phase 2 deterministic helpers
    phase2_results = step_phase2_helpers(as_of, trade_df)
    n_ok        = sum(1 for r in phase2_results.values() if r.get("status") == "OK")
    n_pending   = sum(1 for r in phase2_results.values() if r.get("status") == "INSUFFICIENT_DATA")
    n_missing   = sum(1 for r in phase2_results.values() if r.get("status") == "MODULE_MISSING")
    logger.info("Step 5: phase2 helpers — %d OK / %d pending / %d module-missing",
                n_ok, n_pending, n_missing)

    # Step 6: render + write
    report_md = render_report(as_of, trade_df, contribs, worst, forensic_list, phase2_results, horizon=args.horizon)
    output_path = Path(args.output) / f"dd_investigation_{as_of.isoformat()}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_md, encoding="utf-8")
    logger.info("Step 6: report written to %s (%d bytes)", output_path, len(report_md))

    print(f"\n[OK] Report: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
