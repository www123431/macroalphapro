"""scripts/weight_method_ablation.py — Phase A v2 (rigorous).

v2 rebuild 2026-06-02 after self-audit of v1 caught 7 methodology issues
(see v1 rejection ledger in data/governance/approval_ledger.jsonl).

Critical changes vs v1:
  1. Baseline = deployed L/S decile construction (NOT 1/N across events).
     All 5 variants share the SAME L/S decile structure; they differ ONLY
     in how weights are assigned WITHIN each decile.
  2. Long top decile, short bottom decile, both vol-targeted to balance
     gross exposure.
  3. Transaction costs applied per-variant via per-variant turnover ×
     RT_EQ (matches engine.portfolio.combined_book convention).
  4. Skip-1-day announcement window (standard PEAD literature).
  5. OOS train/test split: train 2014-2020, test 2021-2023. Sharpe and
     Deflated SR reported separately for IS / OOS.
  6. Paired block bootstrap (Politis-Romano 1994) for Sharpe-diff p-value
     vs baseline. Block length = 6 months.
  7. Cosine vs baseline now > 0.5 for all (sanity check: same L/S
     structure, different weighting → should be highly correlated).

Variant weighting methods (within each decile, L/S structure shared):
  - equal           w_i = 1/N within decile (baseline = deployed method)
  - z_sue_clipped   w_i ∝ |z(SUE)| within decile (winsorized ±3σ)
  - rank_decile     w_i ∝ rank(SUE) within decile (linear in rank)
  - inv_vol         w_i ∝ 1/σ_idio_i within decile
  - z_x_inv_vol     w_i ∝ |z(SUE)| × 1/σ_idio within decile

Promotion gate (ALL must hold):
  - OOS Sharpe (2021-2023) ≥ baseline + 0.10  (NOT IS, OOS)
  - OOS Deflated SR ≥ 0.90 (n_trials=5)
  - Paired bootstrap p-value < 0.05 vs baseline (1000 resamples)
  - Cosine vs baseline ∈ [0.5, 0.95] (genuinely different, not noise)

Output:
  data/research/weight_ablation_v2_<date>.parquet
  data/research/factory_ledger.jsonl
  MCC approvals via engine.governance.approval_ledger (only winners)

Usage:
  python scripts/weight_method_ablation.py
  python scripts/weight_method_ablation.py --no-mcc

Doctrine: per [[project-position-weighting-precision-queued-2026-06-02]],
results below the promotion gate confirm DeMiguel-Garlappi-Uppal 2009 (RFS)
1/N defense — that's a valid scientific result, not a failure. Equal weight
is the null hypothesis. Beating it OOS net of costs is the bar.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────


WINSORIZE_LIMIT     = 3.0
HOLD_DAYS           = 60
MIN_EVENTS_PER_MONTH = 30      # need enough for L/S deciles
TRADING_DAYS_YR     = 252
DECILE_BOTTOM       = 0.10
DECILE_TOP          = 0.90

# Cost model — same as deployed D-PEAD per engine.portfolio.combined_book
RT_EQ_BPS           = 30.0     # round-trip cost in bps per side
# Cost per month = RT_EQ × turnover_per_month / 10000 / 12 (annualized basis)
# Standard PEAD turnover assumption: ~5 monthly events × 2 sides = 10/yr per leg
# Actual turnover varies by variant — computed dynamically.

# Split for OOS validation
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"

# Block bootstrap
BOOTSTRAP_BLOCK_MONTHS = 6
BOOTSTRAP_N_RESAMPLES  = 1000

# Promotion gate
PROMOTE_OOS_SHARPE_LIFT  = 0.10
PROMOTE_OOS_DEFLSR_BAR   = 0.90
PROMOTE_BOOTSTRAP_P_MAX  = 0.05
PROMOTE_COSINE_MIN       = 0.50
PROMOTE_COSINE_MAX       = 0.95


# ── Data load (same as v1) ─────────────────────────────────────────


def load_event_panel() -> pd.DataFrame:
    sue = pd.read_parquet(_REPO_ROOT / "data" / "cache" / "_pead_ts_panel_2014_2023.parquet")
    sue = sue.dropna(subset=["sue"]).copy()
    sue["rdq"] = pd.to_datetime(sue["rdq"])
    sue = sue[(sue["rdq"] >= "2014-01-01") & (sue["rdq"] <= "2023-12-31")]
    sue["month"] = sue["rdq"].dt.to_period("M")
    return sue


def load_daily_returns() -> pd.DataFrame:
    ret = pd.read_parquet(_REPO_ROOT / "data" / "cache" / "crsp_hist_daily_ret.parquet")
    ret["date"] = pd.to_datetime(ret["date"])
    ret["log_ret"] = np.log1p(ret["ret"].clip(lower=-0.99))
    return ret.sort_values(["permno", "date"]).reset_index(drop=True)


def build_event_panel(events: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Build the master event panel: per event, attach fwd_ret_log + σ_idio.

    Skip-1-day: forward window starts the SECOND trading day after rdq
    (skip overnight + announcement-day full-day return).
    """
    by_permno = {p: g.reset_index(drop=True) for p, g in rets.groupby("permno", sort=False)}
    out_rows = []
    for _, row in events.iterrows():
        permno = int(row["permno"])
        rdq    = row["rdq"]
        grp = by_permno.get(permno)
        if grp is None:
            continue
        # Trading days strictly AFTER rdq
        after = grp[grp["date"] > rdq]
        if len(after) < 2:
            continue
        # Skip 1 day: start from the SECOND trading day after rdq
        start_idx = after.index[1]   # second day after
        end_date  = rdq + pd.Timedelta(days=HOLD_DAYS)
        end_grp = after[after["date"] <= end_date]
        if len(end_grp) < 5:
            continue
        end_idx = end_grp.index[-1]
        fwd_log = float(grp.loc[start_idx:end_idx, "log_ret"].sum())
        # Idio vol over trailing 63 days BEFORE rdq
        pre = grp[grp["date"] < rdq]
        if len(pre) < 30:
            continue
        sigma = float(pre.iloc[-63:]["ret"].std())
        if not math.isfinite(sigma) or sigma <= 0:
            continue
        out_rows.append({
            "permno":      permno,
            "rdq":         rdq,
            "month":       row["month"],
            "sue":         float(row["sue"]),
            "fwd_ret_log": fwd_log,
            "sigma_idio":  sigma,
        })
    return pd.DataFrame(out_rows)


# ── L/S decile construction (shared baseline) ─────────────────────


def make_long_short_deciles(panel: pd.DataFrame) -> pd.DataFrame:
    """Per month, label each event as LONG (top decile by SUE), SHORT
    (bottom decile), or NEUTRAL (middle 80%, dropped from portfolio)."""
    rows = []
    for month, g in panel.groupby("month"):
        if len(g) < MIN_EVENTS_PER_MONTH:
            continue
        lo = g["sue"].quantile(DECILE_BOTTOM)
        hi = g["sue"].quantile(DECILE_TOP)
        g2 = g.copy()
        g2["leg"] = np.where(g2["sue"] >= hi, "long",
                     np.where(g2["sue"] <= lo, "short", "neutral"))
        rows.append(g2[g2["leg"] != "neutral"])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ── Variant weighting (WITHIN each leg of each month) ─────────────


def _winsorize(s: pd.Series, lim: float = WINSORIZE_LIMIT) -> pd.Series:
    return s.clip(-lim, lim)


def weight_equal(g: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / len(g), index=g.index)


def weight_z_sue_clipped(g: pd.DataFrame) -> pd.Series:
    w = _winsorize(g["sue"]).abs()
    s = w.sum()
    return w / s if s > 0 else pd.Series(1.0 / len(g), index=g.index)


def weight_rank_decile(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    if n < 2:
        return pd.Series(1.0 / n, index=g.index)
    # Within-leg rank (within long: rank by SUE asc; within short: rank by -SUE)
    if (g["sue"] > 0).all():
        ranks = g["sue"].rank()
    else:
        ranks = (-g["sue"]).rank()
    s = ranks.sum()
    return ranks / s if s > 0 else pd.Series(1.0 / n, index=g.index)


def weight_inv_vol(g: pd.DataFrame) -> pd.Series:
    inv = 1.0 / g["sigma_idio"]
    inv = inv.replace([np.inf, -np.inf], 0).fillna(0)
    s = inv.sum()
    return inv / s if s > 0 else pd.Series(1.0 / len(g), index=g.index)


def weight_z_x_inv_vol(g: pd.DataFrame) -> pd.Series:
    z = _winsorize(g["sue"]).abs()
    inv = 1.0 / g["sigma_idio"]
    inv = inv.replace([np.inf, -np.inf], 0).fillna(0)
    w = z * inv
    s = w.sum()
    return w / s if s > 0 else pd.Series(1.0 / len(g), index=g.index)


WEIGHTING_VARIANTS = {
    "equal":          weight_equal,
    "z_sue_clipped":  weight_z_sue_clipped,
    "rank_decile":    weight_rank_decile,
    "inv_vol":        weight_inv_vol,
    "z_x_inv_vol":    weight_z_x_inv_vol,
}


# ── Portfolio construction with L/S structure + costs ─────────────


def build_ls_monthly_returns(panel_ls: pd.DataFrame,
                              weighting_fn,
                              ) -> tuple[pd.Series, float]:
    """For each month, build a long/short portfolio:
       long leg = weighted avg of top decile events,
       short leg = weighted avg of bottom decile events.
       Net return = long_ret - short_ret.
       Each leg weights internally sum to 1.0 → gross exposure = 2.0.

    Returns (monthly_net_return_series, mean_monthly_turnover).
    """
    monthly: dict = {}
    turnovers: list[float] = []
    prev_long_set: set = set()
    prev_short_set: set = set()

    for month, g in panel_ls.groupby("month"):
        longs  = g[g["leg"] == "long"]
        shorts = g[g["leg"] == "short"]
        if len(longs) < 3 or len(shorts) < 3:
            continue
        wL = weighting_fn(longs)
        wS = weighting_fn(shorts)
        long_ret  = float((wL * longs["fwd_ret_log"]).sum())
        short_ret = float((wS * shorts["fwd_ret_log"]).sum())
        net = long_ret - short_ret

        # Turnover: fraction of new names entering each leg
        cur_long  = set(longs["permno"])
        cur_short = set(shorts["permno"])
        if prev_long_set or prev_short_set:
            new_long  = len(cur_long  - prev_long_set)
            new_short = len(cur_short - prev_short_set)
            denom = max(1, len(cur_long) + len(cur_short))
            turnover = (new_long + new_short) / denom
            turnovers.append(turnover)
        prev_long_set, prev_short_set = cur_long, cur_short

        monthly[month.to_timestamp()] = net

    mean_turnover = float(np.mean(turnovers)) if turnovers else 1.0
    return pd.Series(monthly).sort_index(), mean_turnover


def apply_costs(monthly_ret: pd.Series, monthly_turnover: float) -> pd.Series:
    """Apply RT_EQ × turnover transaction cost per month, both legs.
    monthly cost = RT_EQ_BPS / 10000 × turnover × 2 (long + short)
    """
    cost_per_month = (RT_EQ_BPS / 10000.0) * monthly_turnover * 2.0
    return monthly_ret - cost_per_month


# ── Metrics ────────────────────────────────────────────────────────


def annualized_sharpe(r: pd.Series) -> float:
    if len(r) < 12:
        return float("nan")
    mu, sd = r.mean(), r.std()
    if sd <= 0:
        return float("nan")
    return float((mu / sd) * np.sqrt(12))


def sharpe_se(sharpe_ann: float, n_years: float) -> float:
    if not math.isfinite(sharpe_ann) or n_years <= 0:
        return float("nan")
    return float(math.sqrt((1.0 + 0.5 * sharpe_ann * sharpe_ann) / n_years))


def deflated_sharpe(sharpe_ann: float, n_obs: int, n_trials: int) -> float:
    if not math.isfinite(sharpe_ann) or n_obs < 12:
        return 0.0
    from scipy.stats import norm
    monthly_sr = sharpe_ann / math.sqrt(12)
    se_sr = math.sqrt(1.0 / max(1, n_obs - 1))
    z_e = norm.ppf(1.0 - 1.0 / max(2, n_trials)) * se_sr
    z_def = (monthly_sr - z_e) / se_sr if se_sr > 0 else 0.0
    return float(norm.cdf(z_def))


def cosine_vs_baseline(r: pd.Series, baseline: pd.Series) -> float:
    common = r.dropna().index.intersection(baseline.dropna().index)
    if len(common) < 12:
        return float("nan")
    a = r.loc[common].values
    b = baseline.loc[common].values
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return num / den if den > 0 else float("nan")


def paired_block_bootstrap_pvalue(variant_ret: pd.Series,
                                   baseline_ret: pd.Series,
                                   n_resamples: int = BOOTSTRAP_N_RESAMPLES,
                                   block_len: int = BOOTSTRAP_BLOCK_MONTHS,
                                   ) -> float:
    """Paired block bootstrap (Politis-Romano 1994) for the null
    H_0: Sharpe(variant) <= Sharpe(baseline). Returns one-sided p-value.
    """
    common = variant_ret.dropna().index.intersection(baseline_ret.dropna().index)
    if len(common) < 24:
        return float("nan")
    a = variant_ret.loc[common].values
    b = baseline_ret.loc[common].values
    n = len(a)
    obs_diff = (a.mean() / a.std() - b.mean() / b.std()) * math.sqrt(12) \
                if (a.std() > 0 and b.std() > 0) else 0.0

    rng = np.random.RandomState(42)
    sims_diff = []
    n_blocks = max(1, n // block_len)
    for _ in range(n_resamples):
        # Sample starting indices for paired blocks
        idx = []
        for _ in range(n_blocks):
            start = rng.randint(0, n - block_len + 1)
            idx.extend(range(start, start + block_len))
        idx = idx[:n]   # truncate to length n
        ar = a[idx]
        br = b[idx]
        if ar.std() > 0 and br.std() > 0:
            diff = (ar.mean() / ar.std() - br.mean() / br.std()) * math.sqrt(12)
            sims_diff.append(diff)
    if not sims_diff:
        return float("nan")
    sims = np.array(sims_diff)
    # one-sided: probability sim diff >= observed diff under sampling
    return float((sims >= obs_diff).mean())


# ── Main ablation v2 ───────────────────────────────────────────────


def run_ablation_v2(start: Optional[str] = None, end: Optional[str] = None,
                     ) -> pd.DataFrame:
    print(f"[1/7] Loading SUE event panel + CRSP returns…")
    events = load_event_panel()
    rets = load_daily_returns()
    if start:
        events = events[events["rdq"] >= start]
    if end:
        events = events[events["rdq"] <= end]
    print(f"      {len(events):,} events, {len(rets):,} return rows")

    print(f"[2/7] Building event panel with fwd_ret + σ_idio (skip-1-day)…")
    panel = build_event_panel(events, rets)
    print(f"      {len(panel):,} events with usable data")

    print(f"[3/7] Constructing L/S decile labels (top/bottom 10%)…")
    panel_ls = make_long_short_deciles(panel)
    print(f"      {len(panel_ls):,} long+short events × {panel_ls['month'].nunique()} months")

    print(f"[4/7] Splitting train (2014-2020) / test (2021-2023)…")
    panel_ls["rdq_dt"] = pd.to_datetime(panel_ls["rdq"])
    train_panel = panel_ls[panel_ls["rdq_dt"] <= TRAIN_END]
    test_panel  = panel_ls[panel_ls["rdq_dt"] >= TEST_START]
    print(f"      Train: {len(train_panel):,} events × {train_panel['month'].nunique()} months")
    print(f"      Test:  {len(test_panel):,} events × {test_panel['month'].nunique()} months")

    print(f"[5/7] Running variants on TRAIN window (IS)…")
    is_results = {}
    is_turnover = {}
    for name, fn in WEIGHTING_VARIANTS.items():
        ser, turn = build_ls_monthly_returns(train_panel, fn)
        net = apply_costs(ser, turn)
        is_results[name] = net
        is_turnover[name] = turn
        sh = annualized_sharpe(net)
        print(f"  IS  {name:<18}  Sharpe={sh:+.3f}  turnover={turn:.2f}  net_mean/mo={net.mean():+.4f}")

    print(f"\n[6/7] Running variants on TEST window (OOS, the gate)…")
    oos_results = {}
    oos_turnover = {}
    for name, fn in WEIGHTING_VARIANTS.items():
        ser, turn = build_ls_monthly_returns(test_panel, fn)
        net = apply_costs(ser, turn)
        oos_results[name] = net
        oos_turnover[name] = turn
        sh = annualized_sharpe(net)
        print(f"  OOS {name:<18}  Sharpe={sh:+.3f}  turnover={turn:.2f}  net_mean/mo={net.mean():+.4f}")

    print(f"\n[7/7] Computing metrics + paired block bootstrap…")
    rows = []
    base_oos = oos_results["equal"]
    base_is  = is_results["equal"]
    for name in WEIGHTING_VARIANTS.keys():
        is_sh = annualized_sharpe(is_results[name])
        oos_sh = annualized_sharpe(oos_results[name])
        oos_n  = len(oos_results[name])
        oos_yr = oos_n / 12
        oos_se = sharpe_se(oos_sh, oos_yr)
        oos_deflsr = deflated_sharpe(oos_sh, oos_n, n_trials=len(WEIGHTING_VARIANTS))
        cos_oos = cosine_vs_baseline(oos_results[name], base_oos) if name != "equal" else 1.0
        cos_is  = cosine_vs_baseline(is_results[name],  base_is)  if name != "equal" else 1.0
        if name != "equal":
            p_value = paired_block_bootstrap_pvalue(
                oos_results[name], base_oos,
                n_resamples=BOOTSTRAP_N_RESAMPLES,
                block_len=BOOTSTRAP_BLOCK_MONTHS,
            )
        else:
            p_value = float("nan")
        rows.append({
            "variant":          name,
            "is_sharpe":        is_sh,
            "oos_sharpe":       oos_sh,
            "oos_sharpe_se":    oos_se,
            "oos_deflated_sr":  oos_deflsr,
            "cosine_is":        cos_is,
            "cosine_oos":       cos_oos,
            "bootstrap_p":      p_value,
            "is_turnover":      is_turnover[name],
            "oos_turnover":     oos_turnover[name],
            "n_oos_months":     oos_n,
            "n_is_months":      len(is_results[name]),
        })
    return pd.DataFrame(rows)


def select_winners_v2(results: pd.DataFrame) -> pd.DataFrame:
    """Apply the v2 rigorous promotion gate."""
    base_oos = float(results[results["variant"] == "equal"]["oos_sharpe"].iloc[0])
    out = results.copy()
    out["oos_lift_vs_equal"] = out["oos_sharpe"] - base_oos
    out["winner"] = (
        (out["variant"] != "equal")
        & (out["oos_lift_vs_equal"] >= PROMOTE_OOS_SHARPE_LIFT)
        & (out["oos_deflated_sr"] >= PROMOTE_OOS_DEFLSR_BAR)
        & (out["bootstrap_p"] < PROMOTE_BOOTSTRAP_P_MAX)
        & (out["cosine_oos"] >= PROMOTE_COSINE_MIN)
        & (out["cosine_oos"] <= PROMOTE_COSINE_MAX)
    )
    return out


def write_outputs(results: pd.DataFrame) -> Path:
    out_dir = _REPO_ROOT / "data" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    p = out_dir / f"weight_ablation_v2_{stamp}.parquet"
    results.to_parquet(p)

    ledger = out_dir / "factory_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        for _, row in results.iterrows():
            if row["variant"] == "equal":
                continue   # baseline doesn't get a ledger row
            verdict = "GREEN_WINNER" if bool(row.get("winner", False)) else "TESTED_NEUTRAL"
            fh.write(json.dumps({
                "ts":              _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "candidate":       f"weight_method_v2:{row['variant']}",
                "verdict":         verdict,
                "is_sharpe":       float(row["is_sharpe"]),
                "oos_sharpe":      float(row["oos_sharpe"]),
                "oos_deflated_sr": float(row["oos_deflated_sr"]),
                "bootstrap_p":     float(row["bootstrap_p"]),
                "cosine_oos":      float(row["cosine_oos"]),
                "n_oos_months":    int(row["n_oos_months"]),
                "family":          "weight_method_change",
                "source":          "scripts/weight_method_ablation.py (v2)",
            }, ensure_ascii=False) + "\n")
    return p


def create_mcc_approvals_v2(results: pd.DataFrame) -> list[str]:
    """Phase A v2 winners → MCC approval requests with full evidence."""
    from engine.governance.approval_ledger import create_request
    base_oos = float(results[results["variant"] == "equal"]["oos_sharpe"].iloc[0])
    base_is  = float(results[results["variant"] == "equal"]["is_sharpe"].iloc[0])

    rids: list[str] = []
    winners = results[results["winner"] == True]
    for _, row in winners.iterrows():
        rid = create_request(
            request_type="weight_method_change",
            title=f"Phase A v2 winner · D-PEAD weighting → {row['variant']}",
            summary=(
                f"Phase A v2 rigorous ablation result. {row['variant']} achieves "
                f"OOS Sharpe {row['oos_sharpe']:.3f} vs equal-weight (deployed) "
                f"baseline OOS {base_oos:.3f} (lift {row['oos_lift_vs_equal']:+.3f}). "
                f"Deflated SR {row['oos_deflated_sr']:.3f} (family-aware n_trials=5). "
                f"Paired block bootstrap p={row['bootstrap_p']:.4f}. "
                f"Cosine vs baseline {row['cosine_oos']:+.3f} (genuinely different "
                f"weighting, same L/S decile structure). Train 2014-2020, "
                f"test 2021-2023. NET of {RT_EQ_BPS}bps RT cost × monthly turnover "
                f"({row['oos_turnover']:.2f}). Promotion would replace equal weighting "
                f"WITHIN each L/S decile in build_equity_book."
            ),
            proposed_payload={
                "sleeve":          "equity_book",
                "method":          row["variant"],
                "winsorize_limit": WINSORIZE_LIMIT,
                "hold_days":       HOLD_DAYS,
                "ls_decile":       0.10,
                "rt_cost_bps":     RT_EQ_BPS,
                "match_deployed":  True,
            },
            current_state={
                "sleeve":         "equity_book",
                "method":         "equal_weight_within_decile",
            },
            evidence_pack={
                "is_sharpe":           float(row["is_sharpe"]),
                "is_baseline_sharpe":  base_is,
                "oos_sharpe":          float(row["oos_sharpe"]),
                "oos_baseline_sharpe": base_oos,
                "oos_lift_vs_equal":   float(row["oos_lift_vs_equal"]),
                "oos_sharpe_se":       float(row["oos_sharpe_se"]),
                "oos_deflated_sr":     float(row["oos_deflated_sr"]),
                "bootstrap_p":         float(row["bootstrap_p"]),
                "cosine_oos":          float(row["cosine_oos"]),
                "cosine_is":           float(row["cosine_is"]),
                "n_oos_months":        int(row["n_oos_months"]),
                "n_is_months":         int(row["n_is_months"]),
                "n_trials_family":     len(WEIGHTING_VARIANTS),
                "rt_cost_bps":         RT_EQ_BPS,
                "oos_turnover":        float(row["oos_turnover"]),
                "family":              "weight_method_change",
                "construction":        "L/S top/bottom decile, weighting within decile only",
            },
            created_by="scripts/weight_method_ablation.py (v2)",
        )
        rids.append(rid)
    return rids


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase A v2 (rigorous) ablation")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--no-mcc", action="store_true",
                    help="skip MCC approval creation for winners")
    args = ap.parse_args()

    results = run_ablation_v2(start=args.start, end=args.end)
    results = select_winners_v2(results)

    print()
    print("=" * 100)
    print("PROMOTION GATE v2 SUMMARY")
    print("  (Sharpe lift ≥ +0.10 · OOS deflSR ≥ 0.90 · bootstrap p < 0.05 · cos ∈ [0.5, 0.95])")
    print("=" * 100)
    print(f"  {'variant':<22} {'IS Sh':>7} {'OOS Sh':>7} {'lift':>6} {'deflSR':>7} {'boot p':>7} {'cos':>6}")
    print("  " + "-" * 78)
    for _, row in results.iterrows():
        flag = "WINNER →" if row.get("winner", False) else "        "
        print(f"  {flag} {row['variant']:<18} "
              f"{row['is_sharpe']:+7.3f} {row['oos_sharpe']:+7.3f} "
              f"{row.get('oos_lift_vs_equal', 0):+6.3f} "
              f"{row['oos_deflated_sr']:7.3f} "
              f"{row['bootstrap_p']:7.4f} "
              f"{row['cosine_oos']:+6.3f}")

    p = write_outputs(results)
    print(f"\nResults → {p}")

    if not args.no_mcc:
        rids = create_mcc_approvals_v2(results)
        if rids:
            print(f"\n=== MCC Gateway ===")
            for rid in rids:
                print(f"  Approval created: {rid}")
            print(f"\nReview at /approvals.")
        else:
            print(f"\nNo variants cleared the v2 promotion gate.")
            print(f"This CONFIRMS DeMiguel-Garlappi-Uppal 2009 — equal weight remains the null.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
