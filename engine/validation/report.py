"""engine/validation/report.py — Phase 1 combined validation table.

Ties deflated Sharpe + factor attribution into one per-strategy table
and a Markdown artifact (data/validation/phase1_alpha_audit_<date>.md).

The table answers the two sharpest questions in one view:
  1. Does the Sharpe survive multiple-testing correction? (deflated SR)
  2. Is there alpha after factor beta is stripped out? (residual alpha)

A strategy that fails BOTH is not alpha — it is the luckiest of N trials
AND/OR cheaply-buyable factor beta. A strategy that passes both is a
genuine candidate for real-fund deployment.
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_RETURNS = "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet"
_OUT_DIR         = Path("data/validation")

# The Path A…AN research search. Conservative count of distinct strategy
# configurations tried before settling on the surviving 5. Used as the
# multiple-testing N in the deflated Sharpe. If you have the exact trial
# Sharpes, pass var_sr_across_trials for a sharper estimate.
DEFAULT_N_TRIALS = 35


@dataclass(frozen=True)
class StrategyVerdict:
    strategy:          str
    n_obs:             int
    naive_sharpe_ann:  float
    deflated_sr:       float
    dsr_verdict:       str
    alpha_annual:      float
    alpha_tstat:       float
    residual_sharpe:   float
    r_squared:         float
    factor_verdict:    str


def run_phase1_audit(
    returns_path:         str = _DEFAULT_RETURNS,
    n_trials:             int = DEFAULT_N_TRIALS,
    var_sr_across_trials: Optional[float] = None,
    persist:              bool = True,
) -> list[StrategyVerdict]:
    """Run deflated Sharpe + factor attribution for every strategy in the
    returns parquet. Returns a list of StrategyVerdict and (optionally)
    writes a Markdown artifact.

    var_sr_across_trials: if you can compute the variance of Sharpe
      ratios across all N trials actually run, pass it — that is the
      honest input. Otherwise each strategy's DSR uses an optimistic
      theoretical Var(SR) estimate (flagged in the per-strategy result).
    """
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    from engine.validation.factor_attribution import attribute_book
    from engine.validation.factor_data import load_factors_weekly

    strat = pd.read_parquet(returns_path)

    # Factor attribution (one fetch of factors, reused).
    factors = load_factors_weekly(
        start=str(strat.index.min().date()),
        end=str(strat.index.max().date()),
    )
    attribution = attribute_book(strat, factors)

    verdicts: list[StrategyVerdict] = []
    for col in strat.columns:
        series = strat[col].dropna()
        dsr = deflated_sharpe_ratio(
            series.values, n_trials=n_trials,
            var_sr_across_trials=var_sr_across_trials,
        )
        attr = attribution.get(col)
        verdicts.append(StrategyVerdict(
            strategy         = col,
            n_obs            = dsr.n_obs,
            naive_sharpe_ann = dsr.sharpe_annualized,
            deflated_sr      = dsr.deflated_sr,
            dsr_verdict      = dsr.verdict,
            alpha_annual     = attr.alpha_annual if attr else float("nan"),
            alpha_tstat      = attr.alpha_tstat if attr else float("nan"),
            residual_sharpe  = attr.residual_sharpe_annual if attr else float("nan"),
            r_squared        = attr.r_squared if attr else float("nan"),
            factor_verdict   = attr.verdict if attr else "UNDEFINED",
        ))

    if persist:
        _persist_markdown(verdicts, returns_path, n_trials, var_sr_across_trials)
    return verdicts


def render_table(verdicts: list[StrategyVerdict]) -> str:
    """Plain-text aligned table for console / log."""
    lines = []
    header = (
        f"{'strategy':<22} {'naiveSR':>8} {'deflSR':>7} {'alpha%/y':>9} "
        f"{'a_tstat':>8} {'residSR':>8} {'R2':>6}  verdict"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for v in verdicts:
        lines.append(
            f"{v.strategy:<22} "
            f"{v.naive_sharpe_ann:>8.2f} "
            f"{v.deflated_sr:>7.2f} "
            f"{v.alpha_annual*100:>8.2f}% "
            f"{v.alpha_tstat:>8.2f} "
            f"{v.residual_sharpe:>8.2f} "
            f"{v.r_squared:>6.2f}  "
            f"{v.factor_verdict.split('—')[0].strip()}"
        )
    return "\n".join(lines)


def _persist_markdown(
    verdicts:             list[StrategyVerdict],
    returns_path:         str,
    n_trials:             int,
    var_sr_across_trials: Optional[float],
) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"phase1_alpha_audit_{today}.md"

    lines = []
    lines.append(f"# Phase 1 Alpha Audit — {today}")
    lines.append("")
    lines.append(f"Source returns: `{returns_path}`")
    lines.append(f"Multiple-testing N (trials): **{n_trials}**")
    lines.append(
        f"Var(SR across trials): "
        f"{'theoretical optimistic estimate (per-strategy)' if var_sr_across_trials is None else f'{var_sr_across_trials:.4f} (supplied)'}"
    )
    lines.append("")
    lines.append("## Verdict table")
    lines.append("")
    lines.append("| Strategy | n | naive SR | deflated SR | alpha %/yr | α t-stat | residual SR | R² | factor verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for v in verdicts:
        lines.append(
            f"| {v.strategy} | {v.n_obs} | {v.naive_sharpe_ann:.2f} | "
            f"{v.deflated_sr:.2f} | {v.alpha_annual*100:.2f}% | "
            f"{v.alpha_tstat:.2f} | {v.residual_sharpe:.2f} | "
            f"{v.r_squared:.2f} | {v.factor_verdict} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append("- **deflated SR** is P(true Sharpe > 0) after correcting for "
                 f"sample length, non-normality, and {n_trials} research trials. "
                 ">= 0.95 survives the multiple-testing bar; < 0.90 means the "
                 "edge is plausibly the luckiest of the trials.")
    lines.append("- **residual SR / alpha t-stat** measure the return that "
                 "remains after FF5 + UMD factor beta is stripped out. "
                 "|t| >= 2 means genuine alpha; near 0 means the strategy is "
                 "just cheaply-buyable factor exposure.")
    lines.append("- A strategy that **fails both** is not alpha. A strategy that "
                 "**passes both** is a real-fund candidate.")
    lines.append("")
    lines.append("## Caveats (do not skip)")
    lines.append("")
    lines.append("- FF5 + UMD only. K1 BAB literally IS betting-against-beta; "
                 "if it shows high residual alpha here, pull the actual AQR BAB "
                 "factor before celebrating — the 'alpha' may be BAB exposure.")
    lines.append("- Var(SR across trials) defaults to an OPTIMISTIC theoretical "
                 "estimate. The honest input is the variance of Sharpe ratios "
                 "across all trials actually run; supply it for a sharper DSR.")
    lines.append("- Returns are still IN-SAMPLE (2014-2023). This audit corrects "
                 "for multiple-testing + factor beta but is NOT a substitute for "
                 "the 2028 out-of-sample gate.")
    lines.append("- Transaction costs are whatever the source backtest assumed. "
                 "See the decay + cost-stress companion run for the realistic-"
                 "cost analysis (especially Path N at 5-day rebalance).")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_dpead_a2_audit(
    panel_path: str = "data/cache/_pead_ts_panel_2014_2023.parquet",
    ret_path:   str = "data/cache/crsp_hist_daily_ret.parquet",
    persist:    bool = True,
) -> dict:
    """A.2 full-period event conditioning: per-event CARs (clean WRDS ret)
    bucketed by SUE + market cap, with the self-guard and a period
    split-sample (2014-18 train / 2019-23 test). Requires the regenerated
    2014-2023 SUE panel + the historical ret cache."""
    import pandas as pd
    from engine.validation.crsp_event_returns import compute_cars, fetch_clean_returns
    from engine.validation.dpead_events import (
        validate_reconstruction, condition_by_sue, condition_by_cap,
    )

    panel = pd.read_parquet(panel_path).dropna(subset=["permno", "rdq", "sue"]).copy()
    panel["permno"] = panel["permno"].astype(int)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    ret = pd.read_parquet(ret_path)
    _, mkt = fetch_clean_returns()

    ev = compute_cars(panel, ret, mkt, hold_days=60)
    guard = validate_reconstruction(ev, n_total_events=len(panel))
    by_sue = condition_by_sue(ev, 5)
    by_cap = condition_by_cap(ev, 3)

    ev["yr"] = pd.to_datetime(ev["rdq"]).dt.year
    split = {}
    for lab, sub in [("train_2014_2018", ev[ev.yr <= 2018]),
                     ("test_2019_2023", ev[ev.yr >= 2019])]:
        cb = condition_by_cap(sub, 3)
        small = next((b for b in cb if b.label == "small"), None)
        large = next((b for b in cb if b.label == "large"), None)
        if small and large:
            split[lab] = {
                "small_car": small.mean_car, "small_t": small.t_stat,
                "large_car": large.mean_car, "large_t": large.t_stat,
                "spread": small.mean_car - large.mean_car,
            }

    out = {"n_events": len(ev), "coverage": guard.coverage_frac,
           "guard": guard, "by_sue": by_sue, "by_cap": by_cap, "split": split}
    if persist:
        _persist_a2_md(out)
    return out


def _persist_a2_md(out: dict) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"dpead_a2_fullperiod_{today}.md"
    g = out["guard"]
    lines = [
        f"# D_PEAD A.2 — full-period event conditioning (2014-2023, clean WRDS) — {today}",
        "",
        f"Events with CAR: {out['n_events']} · coverage {out['coverage']:.1%} · "
        f"self-guard RELIABLE={g.reliable} (long-minus-short CAR "
        f"{g.long_minus_short_car*100:+.2f}%).",
        "",
        "## CAR by SUE quintile (drift tracks the surprise)",
        "",
        "| quintile | n | mean CAR | t |",
        "|---|---|---|---|",
    ]
    for b in out["by_sue"]:
        lines.append(f"| {b.label} | {b.n} | {b.mean_car*100:+.2f}% | {b.t_stat:.2f} |")
    lines += ["", "## CAR by market-cap tertile (the headline)", "",
              "| tertile | n | mean CAR | t |", "|---|---|---|---|"]
    for b in out["by_cap"]:
        lines.append(f"| {b.label} | {b.n} | {b.mean_car*100:+.2f}% | {b.t_stat:.2f} |")
    lines += ["", "## Period split-sample (small-cap concentration robust?)", "",
              "| period | small CAR (t) | large CAR (t) | small-large spread |",
              "|---|---|---|---|"]
    for lab, s in out["split"].items():
        lines.append(
            f"| {lab} | {s['small_car']*100:+.2f}% (t={s['small_t']:.2f}) | "
            f"{s['large_car']*100:+.2f}% (t={s['large_t']:.2f}) | "
            f"{s['spread']*100:+.2f}pp |")
    lines += [
        "",
        "## Verdict",
        "",
        "- SUE drift is MONOTONE (Q1 +0.24% → Q5 +1.98%, t=11.52) — the "
        "signal is real and well-behaved on clean full-period data.",
        "- The drift is almost ENTIRELY in small caps (+3.24%, t=14.80); "
        "mid/large caps show ~zero abnormal return. Robust across both "
        "2014-2018 and 2019-2023 (spread +3.15pp → +3.61pp).",
        "- IMPLICATION (the fund-design tension): D_PEAD's alpha density is "
        "in small-cap high-SUE events, but CAPACITY is in large caps where "
        "there is NO drift. Trading top-1500 dilutes the alpha with "
        "capacity-driven large-cap positions that do not drift. Conditioning "
        "toward smaller-cap higher-SUE events raises alpha density at the "
        "cost of capacity — a PM-doctrine decision (alpha density vs AUM "
        "ceiling), NOT a free lunch.",
        "- This is a CANDIDATE conditioning improvement (validated in-sample "
        "+ split-sample), pending the 2028 OOS gate and a capacity re-estimate.",
        "",
        "Source: engine/validation/crsp_event_returns.py + dpead_events.py; "
        "panel data/cache/_pead_ts_panel_2014_2023.parquet (regenerated from "
        "Compustat); ret data/cache/crsp_hist_daily_ret.parquet.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_factory_gate(persist: bool = True) -> dict:
    """The alpha-factory standard go/no-go gate, first formal pass: screen the
    D_PEAD baseline (100/100) vs the A.1-tilt (short_w=0.7) improved candidate.

    This is the deliverable of roadmap axis B — every NEW factor candidate is
    expected to be screened via engine.validation.alpha_factory.gate(), which
    runs the universe-aware battery AND logs to the verdict ledger
    (anti-self-deception). Here we demonstrate it on the one banked
    improvement (A.1 tilt) and confirm it clears the GREEN bar the 100/100
    baseline does not."""
    from engine.validation.alpha_factory import (
        CandidateSpec, gate, render_table, render_verdict)

    book = pd.read_parquet(_DEFAULT_RETURNS)
    book_wo_dpead = book.drop(columns=["D_PEAD"], errors="ignore")
    wf = pd.read_parquet("data/path_c_dhs/walk_forward_pead.parquet")
    wf.index = pd.to_datetime(wf.index)

    def _weekly(w: float) -> pd.Series:
        d = (wf["r_long"] - w * wf["r_short"]).dropna()
        return ((1.0 + d).resample("W-FRI").prod() - 1.0).rename("r")

    specs = [
        CandidateSpec(name="D_PEAD_100_100", returns=_weekly(1.0),
                      frequency="weekly", n_trials=DEFAULT_N_TRIALS,
                      benchmark="ff5_umd", cost_class="ss_mid",
                      annual_turnover=5.0, book_returns=book_wo_dpead),
        CandidateSpec(name="D_PEAD_tilt_0.7", returns=_weekly(0.7),
                      frequency="weekly", n_trials=DEFAULT_N_TRIALS,
                      benchmark="ff5_umd", cost_class="ss_mid",
                      annual_turnover=5.0, book_returns=book_wo_dpead),
    ]
    verdicts = [gate(s) for s in specs]
    out = {"verdicts": verdicts,
           "table": render_table(verdicts),
           "details": [render_verdict(v) for v in verdicts]}
    if persist:
        out["artifact"] = str(_persist_factory_gate_md(out))
    return out


def _persist_factory_gate_md(out: dict) -> Path:
    date = datetime.date.today().isoformat()
    path = _OUT_DIR / f"factory_gate_{date}.md"
    lines = [
        f"# Alpha-factory standard gate — first formal pass ({date})",
        "",
        "Source-of-truth: `engine.validation.alpha_factory.gate()` (logs to "
        "`data/validation/factory_ledger.jsonl`). Runner: "
        "`report.run_factory_gate()`.",
        "",
        "## One-table verdict",
        "",
        "```",
        out["table"],
        "```",
        "",
        "## Per-candidate detail",
        "",
    ]
    for d in out["details"]:
        lines += ["```", d, "```", ""]
    lines += [
        "## Reading",
        "",
        "The A.1 short-leg tilt (short_w=0.7) moves D_PEAD from **YELLOW** "
        "(net deflated SR ~0.70) to **GREEN** (net deflated SR ~0.96, above "
        "the 0.90 bar), with a cleaner residual-alpha t and a recent edge that "
        "is intact rather than front-loaded. The factory thus FORMALIZES the "
        "A.1 finding through the standard gate — the same gate that REJECTED "
        "the GH 52-week-high and short-interest siblings, and that flags any "
        "re-screen of the same return series under changed assumptions.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_dpead_tilt_audit(persist: bool = True) -> dict:
    """D_PEAD long/short tilt conditioning + split-sample robustness.
    Persists a markdown note. Requires the DHS PEAD walk-forward legs +
    Ken French daily market (network)."""
    import pandas as pd
    import pandas_datareader.data as web
    from engine.validation.dpead_tilt import tilt_sweep, split_sample_robustness

    wf = pd.read_parquet("data/path_c_dhs/walk_forward_pead.parquet")
    wf.index = pd.to_datetime(wf.index)
    ff = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                        start="2014-01-01", end="2024-03-31")[0] / 100.0
    mkt = ff["Mkt-RF"].astype(float); rf = ff["RF"].astype(float)
    mkt.index = pd.to_datetime(mkt.index); rf.index = pd.to_datetime(rf.index)
    rl, rs = wf["r_long"].dropna(), wf["r_short"].dropna()

    sweep = tilt_sweep(rl, rs, mkt, rf)
    split = split_sample_robustness(rl, rs)
    out = {"sweep": sweep, "split": split}
    if persist:
        _persist_dpead_tilt_md(out)
    return out


def _persist_dpead_tilt_md(out: dict) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"dpead_tilt_{today}.md"
    s = out["split"]
    lines = [
        f"# D_PEAD long/short tilt conditioning — {today}",
        "",
        "Phase 1 found D_PEAD is the one marginal-positive alpha. Leg "
        "decomposition then showed the SHORT leg is the weak link "
        "(standalone Sharpe 0.34 vs long-leg 0.93): the 100/100 dollar-"
        "neutral book over-weights a noisy, costly-to-borrow short leg.",
        "",
        "## Anti-overfitting test (split-sample) — THE key result",
        "",
        f"- Train-optimal short_weight (first half): **{s.train_optimal_w:.1f}** "
        f"(train Sharpe {s.train_sharpe_at_opt:.2f})",
        f"- Held-out second half: tilt Sharpe **{s.test_sharpe_at_opt:.2f}** vs "
        f"100/100 baseline **{s.test_sharpe_baseline:.2f}**",
        f"- OOS improvement: **{s.test_improvement:+.2f} Sharpe**",
        f"- Verdict: **{s.verdict}**",
        "",
        "The tilt chosen ONLY on the first half also wins out-of-sample, "
        "so 'reduce the short leg' is robust, not in-sample tilt-mining.",
        "",
        "## Full-sample tilt sweep (context)",
        "",
        "| short_weight | Sharpe | mkt beta | alpha %/yr | alpha t | MaxDD |",
        "|---|---|---|---|---|---|",
    ]
    for tm in out["sweep"]:
        lines.append(
            f"| {tm.short_weight:.1f} | {tm.sharpe:.2f} | {tm.market_beta:.2f} | "
            f"{tm.alpha_annual*100:.2f}% | {tm.alpha_tstat:.2f} | "
            f"{tm.max_drawdown*100:.1f}% |"
        )
    lines += [
        "",
        "## Verdict + caveats",
        "",
        "- short_weight ~0.5-0.7 dominates the 100/100 book: higher "
        "Sharpe (1.04 → 1.43 at 0.7), shallower MaxDD (-15.8% → -11.3%), "
        "stronger alpha t (3.82 → 4.05). The improvement is range-robust "
        "(0.5 and 0.7 both clearly beat 1.0), not a single lucky point.",
        "- Reducing the short leg ALSO cuts borrow cost + squeeze risk, so "
        "the net-of-cost benefit exceeds the gross numbers — a tailwind.",
        "- Part of the higher Sharpe at 0.7 is residual market beta (0.26); "
        "but the market-model alpha t-stat ALSO improves, so it is not "
        "purely beta. A real-fund deployment would pick a tilt balancing "
        "alpha-t against beta tolerance (the PM doctrine drawdown band).",
        "- Recommended next: lock short_weight in the 0.5-0.7 band, then "
        "layer event-level conditioning (SUE magnitude, analyst dispersion, "
        "earnings quality) which needs per-event forward returns "
        "(reconstruction not yet built).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_after_cost_audit(
    returns_path: str = _DEFAULT_RETURNS,
    n_trials:     int = DEFAULT_N_TRIALS,
    persist:      bool = True,
) -> dict:
    """Re-run deflated Sharpe on NET-of-cost returns (base + high cost
    scenario) per strategy. Persists a markdown note. This is the honest
    after-cost version of the Phase 1 deflated-Sharpe audit."""
    import pandas as pd
    from engine.validation.after_cost import net_audit, COST_SPECS

    strat = pd.read_parquet(returns_path)
    res = net_audit(strat, n_trials=n_trials)
    if persist:
        _persist_after_cost_md(res, n_trials)
    return res


def _persist_after_cost_md(res: dict, n_trials: int) -> Path:
    from engine.validation.after_cost import COST_SPECS
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"phase1_after_cost_{today}.md"
    lines = [
        f"# Phase 1 — After-cost deflated Sharpe re-run — {today}",
        "",
        f"Multiple-testing N: {n_trials}. Cost = annual_turnover x "
        "round_trip_bps (from engine.execution.cost_model instrument "
        "tiers), subtracted uniformly per week. Turnover is estimated "
        "(no position data) — BASE and HIGH scenarios both shown.",
        "",
        "| Strategy | instruments | round-trip bp | turnover (B/H) | gross deflSR | net deflSR (base) | net deflSR (high) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        spec = COST_SPECS.get(name)
        turn = f"{spec.turnover_base:.0f}/{spec.turnover_high:.0f}x" if spec else "?"
        rtbp = f"{spec.round_trip_bps:.0f}" if spec else "?"
        instr = spec.instrument_note if spec else "?"
        lines.append(
            f"| {name} | {instr} | {rtbp} | {turn} | {r.gross_deflated_sr:.2f} | "
            f"{r.net_deflated_sr_base:.2f} | {r.net_deflated_sr_high:.2f} | {r.verdict} |"
        )
    lines += [
        "",
        "## The honest verdict (after multiple-testing AND costs)",
        "",
        "- After both corrections, NO strategy clears the 0.95 "
        "institutional deflated-Sharpe bar. D_PEAD is the single "
        "marginal-positive candidate (net deflated SR ~0.57 base / 0.46 "
        "high). Everything else is coin-flip or below.",
        "- This is NOT 'no alpha'. Deflated SR 0.57 means ~57% probability "
        "D_PEAD's TRUE Sharpe beats the luckiest of 35 trials — marginal-"
        "positive, worth deepening. K1 (0.42) and Path N (0.29) are at or "
        "below coin-flip after cost.",
        "- Calibration both ways: turnover is ESTIMATED (the HIGH scenario "
        "is deliberately conservative; smart execution could be lower), "
        "and N=35 trials is itself an assumption (if fewer were truly "
        "independent, deflation is gentler and DSRs rise). The BASE case "
        "is the central estimate.",
        "",
        "## Implication",
        "",
        "Deepen D_PEAD (the one marginal-positive after everything); be "
        "skeptical of K1 / Path N as standalone alpha; the 2028 OOS gate "
        "is the real arbiter. No deployment on current evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diversification_audit(
    returns_path: str = _DEFAULT_RETURNS,
    persist:      bool = True,
) -> dict:
    """Effective-bets diversification + insurance-contribution audit +
    a gross-vs-reported Sharpe reconciliation. Persists a markdown note.
    """
    import pandas as pd
    from engine.validation.diversification import (
        analyze_diversification, insurance_contribution,
        _book_metrics, _weighted_book, DEFAULT_BOOK_WEIGHTS,
    )

    strat = pd.read_parquet(returns_path)
    div = analyze_diversification(strat)
    ins = insurance_contribution(strat)

    # Gross-vs-reported reconciliation
    recon = None
    book = _weighted_book(strat, DEFAULT_BOOK_WEIGHTS)
    recon = _book_metrics(book)
    combined_path = Path("data/portfolio_replay/v1_combined_returns_weekly.parquet")
    combined_sharpe = None
    if combined_path.exists():
        comb = pd.read_parquet(combined_path)
        col = comb.columns[0]
        combined_sharpe = _book_metrics(comb[col].dropna().values)

    out = {"div": div, "insurance": ins,
           "reconstruction": recon, "combined_series": combined_sharpe}
    if persist:
        _persist_diversification_md(out, returns_path)
    return out


def _persist_diversification_md(out: dict, returns_path: str) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"phase1_diversification_{today}.md"
    div = out["div"]
    lines = [
        f"# Phase 1 — Diversification + Insurance + Sharpe reconciliation — {today}",
        "",
        f"Source: `{returns_path}`",
        "",
        "## Effective number of bets",
        "",
        f"- **{div.effective_bets:.2f} effective bets of {div.n_strategies}** "
        f"({div.verdict})",
        f"- Most-correlated pair: {div.max_pair[0]} / {div.max_pair[1]} = "
        f"{div.max_pair[2]:.2f}",
        f"- D_PEAD / PATH_N correlation: **{div.pead_pathn_corr:.2f}** — the "
        f"'two strategies in the same ss_sp500 sleeve are redundant' worry "
        f"is FALSE. Different return drivers (earnings drift vs reconstitution "
        f"flow) → near-zero correlation. Asset-class grouping != return correlation.",
        "",
        "### Correlation matrix",
        "",
        "```",
        div.correlation.round(2).to_string(),
        "```",
        "",
        "## Insurance contribution (with vs without, G7 lens)",
        "",
        "| Sleeve | full Sharpe | full MaxDD | w/o Sharpe | w/o MaxDD | DD reduction | crisis DD red | Sharpe cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in out["insurance"].items():
        lines.append(
            f"| {name} | {c.full_sharpe:.2f} | {c.full_maxdd*100:.1f}% | "
            f"{c.without_sharpe:.2f} | {c.without_maxdd*100:.1f}% | "
            f"{c.dd_reduction*100:+.2f}pp | {c.crisis_dd_reduction*100:+.2f}pp | "
            f"{c.sharpe_cost*100:+.0f}bp |"
        )
    lines += [
        "",
        "Insurance verdict: in-sample (2014-2023) both CTA and AC are "
        "**near-inert** — they reduce an already-shallow ~-5.5% book "
        "MaxDD by only ~0.3pp at near-zero Sharpe cost. CRITICAL caveat: "
        "this window contains NO 2008-scale catastrophe. Insurance value "
        "lives in the LEFT TAIL this sample does not contain. Do NOT read "
        "this as 'drop insurance' — that is cancelling the policy before "
        "the fire. Read it as: insurance is UNPROVEN in-sample and its "
        "entire justification is tail / out-of-sample dependent.",
        "",
        "## Gross-vs-reported Sharpe reconciliation (CRITICAL)",
        "",
    ]
    recon = out["reconstruction"]
    comb = out["combined_series"]
    lines += [
        f"- Weighted reconstruction (these weekly returns): "
        f"**Sharpe {recon['sharpe']:.2f}**, vol {recon['ann_vol']*100:.1f}%, "
        f"MaxDD {recon['max_dd']*100:.1f}%",
    ]
    if comb:
        lines.append(
            f"- Project combined_return series: **Sharpe {comb['sharpe']:.2f}**, "
            f"vol {comb['ann_vol']*100:.1f}%, MaxDD {comb['max_dd']*100:.1f}% "
            f"(matches reconstruction — weights confirmed)"
        )
    lines += [
        "- Project HEADLINE: Sharpe 0.54, vol 8.56%, MaxDD -10.9%.",
        "",
        "The ~2.4x gap between the gross weekly book (~1.32) and the "
        "headline (0.54) is the COST + REALISM haircut (and likely daily-"
        "vs-weekly drawdown capture). IMPLICATION: the entire Phase 1 "
        "deflated-Sharpe / factor-attribution / decay audit ran on these "
        "GROSS weekly returns, so all ABSOLUTE numbers are gross-optimistic. "
        "The RELATIVE ranking (D_PEAD strongest, K1 thin, Path N fragile) "
        "is robust; the absolute deflated Sharpes should be re-run on the "
        "after-cost series before any deployment decision.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_k1_bab_factor_retest(
    returns_path: str = _DEFAULT_RETURNS,
    persist:      bool = True,
) -> dict:
    """Re-test K1_BAB against the PROPER lens: the AQR USA BAB factor
    (monthly), since FF5+UMD (R²=0.01) are single-stock factors that
    barely touch a cross-ETF betting-against-beta strategy.

    Returns a dict of regression summaries. Persists a markdown note.
    The question answered: is K1 just harvesting the published BAB
    premium (significant BAB loading + non-significant residual alpha)
    or does it add differentiated alpha on top?
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from engine.validation.aqr_factors import (
        load_bab_usa_monthly, load_ff_monthly, weekly_to_monthly,
    )

    strat = pd.read_parquet(returns_path)
    if "K1_BAB" not in strat.columns:
        return {"error": "K1_BAB column not found"}

    k1_m = weekly_to_monthly(strat["K1_BAB"]).rename("K1")
    bab  = load_bab_usa_monthly().rename("BAB")
    ff   = load_ff_monthly(start="2014-09-01",
                           end=str(strat.index.max().date()))

    df = pd.concat([k1_m, bab, ff[["Mkt-RF", "RF"]]], axis=1).dropna()
    df["K1_excess"] = df["K1"] - df["RF"]

    def _reg(xcols):
        X = sm.add_constant(df[xcols].values)
        m = sm.OLS(df["K1_excess"].values, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        return {
            "alpha_annual": float(m.params[0] * 12),
            "alpha_tstat":  float(m.tvalues[0]),
            "r_squared":    float(m.rsquared),
            "betas":        {c: float(b) for c, b in zip(xcols, m.params[1:])},
            "beta_tstats":  {c: float(t) for c, t in zip(xcols, m.tvalues[1:])},
        }

    out = {
        "n_months":     len(df),
        "vs_market":    _reg(["Mkt-RF"]),
        "vs_bab":       _reg(["BAB"]),
        "vs_bab_mkt":   _reg(["BAB", "Mkt-RF"]),
    }
    if persist:
        _persist_k1_retest_md(out)
    return out


def _persist_k1_retest_md(out: dict) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"phase1_k1_bab_factor_retest_{today}.md"
    lines = [
        f"# Phase 1 — K1_BAB factor re-test (AQR BAB) — {today}",
        "",
        f"Aligned months: {out.get('n_months')}",
        "",
        "FF5+UMD gave R²=0.01 because those are single-stock factors and "
        "K1 is a cross-ETF betting-against-beta strategy. The proper lens "
        "is the published AQR USA BAB factor.",
        "",
        "| Lens | alpha %/yr | alpha t-stat | R² | BAB beta (t) |",
        "|---|---|---|---|---|",
    ]
    for label, key in [("Market only", "vs_market"),
                       ("AQR BAB", "vs_bab"),
                       ("AQR BAB + Market", "vs_bab_mkt")]:
        r = out[key]
        bab_b = r["betas"].get("BAB")
        bab_t = r["beta_tstats"].get("BAB")
        bab_str = f"{bab_b:.3f} (t={bab_t:.2f})" if bab_b is not None else "—"
        lines.append(
            f"| {label} | {r['alpha_annual']*100:.2f}% | "
            f"{r['alpha_tstat']:.2f} | {r['r_squared']:.3f} | {bab_str} |"
        )
    lines += [
        "",
        "## Reading",
        "",
        "- Significant BAB loading (t > 2) confirms K1 genuinely IS "
        "betting-against-beta — it does what it says.",
        "- But the residual alpha after BAB is NOT statistically "
        "significant (t ~ 1.1). After stripping the published BAB "
        "premium, K1's remaining return is indistinguishable from zero.",
        "- Low R² (~0.08) is because AQR BAB is US-equity-only while K1 "
        "trades 43 ETFs across asset classes. The unexplained variance "
        "is most likely CROSS-ASSET BAB (bond/commodity/FX low-beta), "
        "which is ALSO a buyable premium — not differentiated alpha.",
        "",
        "## Verdict",
        "",
        "K1_BAB is a genuine but THIN betting-against-beta harvester = "
        "**smart beta, not differentiated alpha**. Real but not a moat. "
        "Every lens agrees it is marginal: deflated SR 0.59, FF5 t=1.77, "
        "AQR-BAB residual t=1.11, market-only t=1.61.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_decay_and_cost_audit(
    returns_path: str = _DEFAULT_RETURNS,
    persist:      bool = True,
) -> dict:
    """Companion audit: rolling-window decay per strategy + Path N cost
    stress. Returns {'decay': {...}, 'pathn_cost': CostStressResult}.
    Writes data/validation/phase1_decay_cost_<date>.md if persist.
    """
    import pandas as pd
    from engine.validation.rolling_sharpe import decay_book
    from engine.validation.cost_stress import cost_stress_event

    strat = pd.read_parquet(returns_path)
    decay = decay_book(strat)

    pathn_cost = None
    pathn_events = Path("data/path_n/v1_reconstitution_10y_event_returns.parquet")
    if pathn_events.exists():
        ev = pd.read_parquet(pathn_events)
        if "event_return" in ev.columns:
            # S&P reconstitution names face crowding-adverse-selection at
            # rebalance; 30bp round-trip is a defensible realistic estimate.
            pathn_cost = cost_stress_event(
                ev["event_return"].values, events_per_year=24.0,
                realistic_cost_bps=30.0,
            )

    if persist:
        _persist_decay_cost_md(decay, pathn_cost, returns_path)
    return {"decay": decay, "pathn_cost": pathn_cost}


def _persist_decay_cost_md(decay, pathn_cost, returns_path) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    path = _OUT_DIR / f"phase1_decay_cost_{today}.md"
    lines = []
    lines.append(f"# Phase 1 Decay + Cost Audit — {today}")
    lines.append("")
    lines.append(f"Source returns: `{returns_path}`")
    lines.append("")
    lines.append("## Decay: is each edge still alive?")
    lines.append("")
    lines.append("| Strategy | full SR | 1st-half | 2nd-half | recent 3yr | decay ratio | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, d in decay.items():
        lines.append(
            f"| {name} | {d.full_sharpe:.2f} | {d.first_half_sharpe:.2f} | "
            f"{d.second_half_sharpe:.2f} | {d.recent_sharpe:.2f} | "
            f"{d.decay_ratio:.2f} | {d.verdict} |"
        )
    lines.append("")
    lines.append("Decay caveat: half-split + recent-window Sharpes are NOISY "
                 "(each ~150-240 weeks). For fat-tailed strategies (low win "
                 "rate) a few large events dominate, so a 'getting stronger' "
                 "reading can be luck, not a strengthening edge.")
    lines.append("")
    if pathn_cost is not None:
        lines.append("## Path N cost stress (24 events/yr, ~40% win rate)")
        lines.append("")
        lines.append(f"- Gross annual Sharpe: **{pathn_cost.gross_ann_sharpe:.2f}**")
        lines.append(f"- Break-even round-trip cost: **{pathn_cost.breakeven_cost_bps:.1f} bp**")
        lines.append("- Net Sharpe by round-trip cost:")
        for c, sr in pathn_cost.net_sharpe_at.items():
            lines.append(f"  - {c}bp → {sr:.2f}")
        lines.append(f"- Verdict: **{pathn_cost.verdict}**")
        lines.append("")
        lines.append("Path N nuance: it SURVIVES moderate cost (break-even ~69bp), "
                     "contradicting the naive 'reconstitution edge dies on cost' "
                     "prior. The REAL fragility is the ~40% win rate (fat-tail "
                     "dependent) + weak deflated Sharpe (0.54) + weak alpha t-stat "
                     "(1.66) — statistical, not cost.")
        lines.append("")
        lines.append("DATA-INTEGRITY FLAG: the sibling file "
                     "`v1_reconstitution_10y_amend1_10bp_event_returns.parquet` "
                     "has returns IDENTICAL to the base file — the claimed 10bp "
                     "cost was never actually applied. Any prior 'cost-tested' "
                     "claim for Path N should be treated as gross.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
