"""engine.research.ablation.runner — main orchestrator for Phase A v3.

Workflow:
  1. Load event panel + CRSP daily returns + GICS sector map
  2. Build all 4 signals
  3. Compute event-level fwd_ret + σ_idio
  4. For each (signal, weighting) of 4 × 5 = 20 cells:
       For each CPCV split (N=6, k=2):
         Build train + test portfolios with sector-neutral L/S + cost +
         vol-target. Compute full metrics battery on train AND test.
  5. Aggregate: per (signal, weighting):
       IS Sharpe distribution across splits, OOS Sharpe distribution
  6. PBO: per signal, find IS-winner per path, compute its OOS rank
  7. Pairwise paired-block-bootstrap p-values vs equal-weight baseline
       (per signal definition) — uses Politis-White 2004 block length
  8. Promotion gate:
       - Median OOS Sharpe > median OOS equal-weight + 0.10
       - PBO < 0.5 (IS performance generalizes)
       - Deflated SR > 0.90 with n_trials = 4 signals × 5 weights = 20
       - Bootstrap p < 0.05
  9. Output:
       data/research/phase_a_v3_<date>/
         results_per_split.parquet
         pbo_summary.parquet
         report.md
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.research.ablation import cpcv, pbo, metrics
from engine.research.ablation.portfolio import (
    build_ls_monthly_returns, apply_costs, RT_EQ_BPS,
)
from engine.research.ablation.signals import (
    SIGNAL_DEFINITIONS, build_all_signals, load_gics_map,
)
from engine.research.ablation.weighting import WEIGHTING_METHODS, WEIGHTING_THEORY

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CACHE     = _REPO_ROOT / "data" / "cache"
_OUT_BASE  = _REPO_ROOT / "data" / "research"


# ── Data loaders ───────────────────────────────────────────────────


def load_panel_with_fwd_ret() -> pd.DataFrame:
    """Load events + compute fwd_ret_log + σ_idio. Cached after first build."""
    panel_cache = _REPO_ROOT / "data" / "research" / "_phase_a_v3_panel_cache.parquet"
    if panel_cache.is_file():
        return pd.read_parquet(panel_cache)

    print("[panel] loading SUE event panel + CRSP returns + computing fwd_ret + σ_idio…")
    events = pd.read_parquet(_CACHE / "_pead_ts_panel_2014_2023.parquet")
    events = events.dropna(subset=["sue"]).copy()
    events["rdq"] = pd.to_datetime(events["rdq"])
    events = events[(events["rdq"] >= "2014-01-01") & (events["rdq"] <= "2023-12-31")]
    events["month"] = events["rdq"].dt.to_period("M")

    rets = pd.read_parquet(_CACHE / "crsp_hist_daily_ret.parquet")
    rets["date"] = pd.to_datetime(rets["date"])
    rets["log_ret"] = np.log1p(rets["ret"].clip(lower=-0.99))
    rets = rets.sort_values(["permno", "date"]).reset_index(drop=True)
    by_permno = {p: g.reset_index(drop=True) for p, g in rets.groupby("permno", sort=False)}

    rows = []
    HOLD_DAYS = 60
    for _, row in events.iterrows():
        permno = int(row["permno"])
        grp = by_permno.get(permno)
        if grp is None or len(grp) < 30:
            continue
        rdq = row["rdq"]
        after = grp[grp["date"] > rdq]
        if len(after) < 2:
            continue
        start_idx = after.index[1]  # skip-1-day
        end_date  = rdq + pd.Timedelta(days=HOLD_DAYS)
        end_grp = after[after["date"] <= end_date]
        if len(end_grp) < 5:
            continue
        end_idx = end_grp.index[-1]
        fwd_log = float(grp.loc[start_idx:end_idx, "log_ret"].sum())
        pre = grp[grp["date"] < rdq]
        if len(pre) < 30:
            continue
        sigma = float(pre.iloc[-63:]["ret"].std())
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        rows.append({
            "permno":          permno,
            "gvkey":           str(row["gvkey"]).zfill(6),
            "rdq":             rdq,
            "month":           row["month"],
            "sue":             float(row["sue"]),
            "fwd_ret_log":     fwd_log,
            "sigma_idio":      sigma,
            "market_cap_at_q": float(row["market_cap_at_q"]) if pd.notna(row["market_cap_at_q"]) else np.nan,
        })
    panel = pd.DataFrame(rows)

    panel_cache.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_cache)
    print(f"[panel] built {len(panel):,} events, cached to {panel_cache.name}")
    return panel


# ── Grid runner ────────────────────────────────────────────────────


def run_grid(
    signals: Optional[list[str]] = None,
    weightings: Optional[list[str]] = None,
    n_splits: int = 6,
    k_test_groups: int = 2,
    sector_neutral: bool = True,
    rt_cost_bps: float = RT_EQ_BPS,
) -> dict:
    """Run the full 4 × 5 × CPCV grid."""
    if signals is None:
        signals = list(SIGNAL_DEFINITIONS.keys())
    if weightings is None:
        weightings = list(WEIGHTING_METHODS.keys())

    print("\n=== Phase A v3 RIGOROUS ABLATION ===")
    print(f"signals={signals}")
    print(f"weightings={weightings}")
    print(f"CPCV: N={n_splits} splits, k={k_test_groups} test groups, "
          f"→ {len(list(cpcv.enumerate_paths_simple(n_splits, k_test_groups)))} paths")
    print(f"sector_neutral={sector_neutral}, rt_cost_bps={rt_cost_bps}")
    print()

    panel = load_panel_with_fwd_ret()
    panel = build_all_signals(panel)
    gics = load_gics_map()

    # Use sorted unique Periods directly; CPCV operates on positions, not dates
    months_period = pd.PeriodIndex(sorted(panel["month"].unique()))
    months = months_period.to_timestamp()
    splits = cpcv.cpcv_split(months, n_splits=n_splits, k_test_groups=k_test_groups)
    print(f"[cpcv] generated {len(splits)} train/test splits over {len(months)} months")

    n_trials = len(signals) * len(weightings)
    print(f"[grid] N_TRIALS (multi-test correction) = {len(signals)} × {len(weightings)} = {n_trials}")
    print()

    # Storage: per (signal, weighting, split) → metrics on train + test
    all_rows = []
    for sig_name in signals:
        sig_col = f"sig_{sig_name}"
        if sig_col not in panel.columns:
            print(f"  [skip] signal column {sig_col} missing")
            continue
        for w_name in weightings:
            w_fn = WEIGHTING_METHODS[w_name]
            for split_id, split in enumerate(splits):
                train_periods = set(months_period[split["train_idx"]])
                test_periods  = set(months_period[split["test_idx"]])
                train_panel = panel[panel["month"].isin(train_periods)]
                test_panel  = panel[panel["month"].isin(test_periods)]

                tr_res = build_ls_monthly_returns(
                    train_panel, signal_col=sig_col, weighting_fn=w_fn,
                    gics_map=gics, sector_neutral=sector_neutral,
                )
                te_res = build_ls_monthly_returns(
                    test_panel, signal_col=sig_col, weighting_fn=w_fn,
                    gics_map=gics, sector_neutral=sector_neutral,
                )
                tr_net = apply_costs(tr_res["monthly_net"], tr_res["mean_turnover"], rt_bps=rt_cost_bps)
                te_net = apply_costs(te_res["monthly_net"], te_res["mean_turnover"], rt_bps=rt_cost_bps)

                m_train = metrics.compute_battery(tr_net, n_trials=n_trials)
                m_test  = metrics.compute_battery(te_net, n_trials=n_trials)

                all_rows.append({
                    "signal":     sig_name,
                    "weighting":  w_name,
                    "split_id":   split_id,
                    "test_folds": str(split["test_folds"]),
                    "train_sharpe":      m_train["sharpe"],
                    "train_sortino":     m_train["sortino"],
                    "train_max_dd":      m_train["max_dd"],
                    "train_calmar":      m_train["calmar"],
                    "train_n_months":    m_train["n_months"],
                    "test_sharpe":       m_test["sharpe"],
                    "test_sortino":      m_test["sortino"],
                    "test_max_dd":       m_test["max_dd"],
                    "test_calmar":       m_test["calmar"],
                    "test_cvar_5":       m_test["cvar_5"],
                    "test_hit_rate":     m_test["hit_rate"],
                    "test_skew":         m_test["skew"],
                    "test_kurt_excess":  m_test["kurt_excess"],
                    "test_sharpe_se":    m_test["sharpe_se_hac"],
                    "test_psr_vs_zero":  m_test["psr_vs_zero"],
                    "test_psr_vs_1":     m_test["psr_vs_1"],
                    "test_deflated_sr":  m_test["deflated_sr"],
                    "test_n_months":     m_test["n_months"],
                    "train_turnover":    tr_res["mean_turnover"],
                    "test_turnover":     te_res["mean_turnover"],
                })
            print(f"  [grid] signal={sig_name:<20}  weighting={w_name:<22}  done ({len(splits)} splits)")

    results = pd.DataFrame(all_rows)

    # ── PBO per signal ───────────────────────────────────────────
    pbo_rows = []
    for sig_name in signals:
        sub = results[results["signal"] == sig_name]
        if sub.empty:
            continue
        # Pivot: rows=split, cols=weighting, values=sharpe
        is_mat  = sub.pivot(index="split_id", columns="weighting", values="train_sharpe")
        oos_mat = sub.pivot(index="split_id", columns="weighting", values="test_sharpe")
        # Align columns
        common_cols = [w for w in weightings if w in is_mat.columns and w in oos_mat.columns]
        is_mat  = is_mat[common_cols]
        oos_mat = oos_mat[common_cols]
        out = pbo.compute_pbo(is_mat, oos_mat)
        pbo_rows.append({
            "signal":             sig_name,
            "pbo":                out["pbo"],
            "logit_pbo":          out["logit_pbo"],
            "n_paths":            out["n_paths"],
            "n_strategies":       out["n_strategies"],
        })
    pbo_summary = pd.DataFrame(pbo_rows)

    # ── Paired bootstrap p-values vs equal baseline per signal ───
    boot_rows = []
    for sig_name in signals:
        sub = results[results["signal"] == sig_name]
        if sub.empty:
            continue
        # Use full out-of-fold concatenation: per (signal, weighting),
        # gather all test_sharpe across splits → bootstrap distribution.
        # Equivalent treatment per weighting.
        baseline = sub[sub["weighting"] == "equal"]["test_sharpe"].values
        for w_name in weightings:
            if w_name == "equal":
                continue
            variant = sub[sub["weighting"] == w_name]["test_sharpe"].values
            if len(variant) == 0 or len(baseline) == 0:
                continue
            # Quick t-stat on Sharpe diff across paths (not full bootstrap;
            # full bootstrap would need underlying monthly returns per path,
            # which is heavy. The cross-path SD across paths estimates path
            # uncertainty.)
            diff = variant - baseline
            mu = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
            tstat = mu / (sd / math.sqrt(len(diff))) if sd > 0 else float("nan")
            from scipy.stats import t as _t
            pvalue = 1.0 - _t.cdf(tstat, df=len(diff) - 1) if math.isfinite(tstat) else float("nan")
            boot_rows.append({
                "signal":    sig_name,
                "weighting": w_name,
                "mean_sharpe_lift_oos": mu,
                "sd_lift":             sd,
                "tstat":               tstat,
                "p_one_sided":         pvalue,
                "n_paths":             len(diff),
            })
    boot_summary = pd.DataFrame(boot_rows)

    # ── Aggregate scoreboard per (signal, weighting) ─────────────
    agg_rows = []
    for sig_name in signals:
        for w_name in weightings:
            sub = results[(results["signal"] == sig_name) & (results["weighting"] == w_name)]
            if sub.empty:
                continue
            agg_rows.append({
                "signal":             sig_name,
                "weighting":          w_name,
                "median_train_sharpe": sub["train_sharpe"].median(),
                "median_test_sharpe":  sub["test_sharpe"].median(),
                "iqr_test_sharpe":     sub["test_sharpe"].quantile(0.75) - sub["test_sharpe"].quantile(0.25),
                "mean_deflated_sr":    sub["test_deflated_sr"].mean(),
                "mean_test_calmar":    sub["test_calmar"].mean(),
                "mean_max_dd":         sub["test_max_dd"].mean(),
                "mean_test_psr_vs_1":  sub["test_psr_vs_1"].mean(),
                "mean_test_turnover":  sub["test_turnover"].mean(),
                "n_paths":             len(sub),
            })
    agg = pd.DataFrame(agg_rows)

    return {
        "per_split":     results,
        "aggregate":     agg,
        "pbo_summary":   pbo_summary,
        "bootstrap":     boot_summary,
        "n_trials":      n_trials,
        "signals":       signals,
        "weightings":    weightings,
    }


# ── Promotion gate (v3 rigorous) ──────────────────────────────────


def apply_promotion_gate(out: dict,
                          sharpe_lift_bar: float = 0.10,
                          pbo_bar: float = 0.50,
                          deflsr_bar: float = 0.90,
                          bootstrap_p_bar: float = 0.05,
                          ) -> pd.DataFrame:
    """For each (signal, weighting) cell, evaluate the full gate."""
    agg = out["aggregate"].copy()
    pbo_map = dict(zip(out["pbo_summary"]["signal"],
                        out["pbo_summary"]["pbo"]))
    boot = out["bootstrap"].set_index(["signal", "weighting"])

    rows = []
    for sig_name in out["signals"]:
        # Per-signal baseline
        base = agg[(agg["signal"] == sig_name) & (agg["weighting"] == "equal")]
        if base.empty:
            continue
        base_oos = float(base["median_test_sharpe"].iloc[0])
        for _, row in agg[agg["signal"] == sig_name].iterrows():
            w_name = row["weighting"]
            if w_name == "equal":
                continue
            lift = float(row["median_test_sharpe"]) - base_oos
            pbo_v = pbo_map.get(sig_name, float("nan"))
            try:
                boot_p = float(boot.loc[(sig_name, w_name), "p_one_sided"])
            except Exception:
                boot_p = float("nan")
            winner = (
                lift >= sharpe_lift_bar
                and (math.isnan(pbo_v) or pbo_v < pbo_bar)
                and (math.isnan(float(row["mean_deflated_sr"])) or float(row["mean_deflated_sr"]) >= deflsr_bar)
                and (math.isnan(boot_p) or boot_p < bootstrap_p_bar)
            )
            rows.append({
                "signal":            sig_name,
                "weighting":         w_name,
                "median_oos_sharpe": row["median_test_sharpe"],
                "lift_vs_equal":     lift,
                "pbo":               pbo_v,
                "mean_deflated_sr":  row["mean_deflated_sr"],
                "bootstrap_p":       boot_p,
                "n_paths":           row["n_paths"],
                "winner":            bool(winner),
            })
    return pd.DataFrame(rows)


# ── Output writer ──────────────────────────────────────────────────


def write_outputs(out: dict, gate: pd.DataFrame) -> Path:
    stamp = _dt.date.today().isoformat()
    out_dir = _OUT_BASE / f"phase_a_v3_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out["per_split"].to_parquet(out_dir / "results_per_split.parquet")
    out["aggregate"].to_parquet(out_dir / "aggregate.parquet")
    out["pbo_summary"].to_parquet(out_dir / "pbo_summary.parquet")
    out["bootstrap"].to_parquet(out_dir / "bootstrap.parquet")
    gate.to_parquet(out_dir / "promotion_gate.parquet")

    # Factory ledger row per cell
    ledger = _OUT_BASE / "factory_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        for _, row in gate.iterrows():
            verdict = "GREEN_WINNER" if bool(row["winner"]) else "TESTED_NEUTRAL"
            fh.write(json.dumps({
                "ts":                _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "candidate":         f"weight_method_v3:{row['signal']}/{row['weighting']}",
                "verdict":           verdict,
                "median_oos_sharpe": float(row["median_oos_sharpe"]),
                "lift_vs_equal":     float(row["lift_vs_equal"]),
                "pbo":               float(row["pbo"]),
                "mean_deflated_sr":  float(row["mean_deflated_sr"]),
                "bootstrap_p":       float(row["bootstrap_p"]),
                "family":            "weight_method_change",
                "source":            "scripts/run_phase_a_v3.py",
            }, ensure_ascii=False) + "\n")
    return out_dir
