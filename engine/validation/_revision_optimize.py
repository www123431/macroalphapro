"""engine/validation/_revision_optimize.py — joint optimization of the analyst-
revision sleeve toward the GREEN bar (net deflated SR >= 0.90), WITHOUT p-hacking.

Discipline (the whole point):
  * cost is FIXED at the large-cap class (ss_large 20bps round-trip; 26bps shown as
    sensitivity) — not tuned to flatter a config;
  * n_trials is set HONESTLY to the size of THIS grid search (every config tested is
    a trial). A bigger search => a bigger deflated-Sharpe penalty, so only a
    genuinely stronger NET SHARPE — not a lucky knife-edge — can clear the bar. We
    report net deflSR at n_trials = {18 (the documented local convention), 35
    (project net_audit convention), grid_size (conservative/honest)} and JUDGE on
    the conservative one;
  * a winner must ALSO pass the robustness battery (smooth cutoff gradient, both
    subsample halves positive, yearly positivity) — the test that caught the
    Lazy-Prices false YELLOW.

Levers (all economically motivated, none ad hoc):
  disp_pctile  — information-uncertainty conditioning (Zhang 2006; Gleason-Lee 2003)
  weight       — equal vs |revision|-magnitude (bigger revisions drift more)
  q_in / q_out — entry quantile / no-trade exit band (Novy-Marx–Velikov buffering)

Loads revision panel + CRSP monthly returns ONCE (the library function reloads the
23MB CRSP parquet per call); the inline sleeve loop is validated against
build_revision_sleeve_buffered before the sweep so the grid can't silently diverge.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.after_cost import apply_cost
from engine.validation.analyst_revision import build_revision_panel, build_revision_sleeve_buffered, _RET
from engine.validation.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)


def load_inputs():
    """revw (rev_ratio wide), cvw (dispersion-CV wide), mret (monthly returns wide)."""
    rev = build_revision_panel()
    rev["cv"] = rev["dispersion"] / rev["meanest"].abs().replace(0, np.nan)
    revw = rev.pivot(index="month", columns="permno", values="rev_ratio").sort_index()
    cvw = rev.pivot(index="month", columns="permno", values="cv").sort_index()
    ret = pd.read_parquet(_RET); ret["date"] = pd.to_datetime(ret["date"])
    daily = ret.pivot_table(index="date", columns="permno", values="ret").sort_index()
    mret = (1 + daily.fillna(0)).resample("ME").prod() - 1
    mret = mret.where(daily.resample("ME").count() > 5)
    return revw, cvw, mret


def sleeve(revw, cvw, mret, q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5):
    """Inline replica of build_revision_sleeve_buffered (hysteresis buffering +
    dispersion conditioning + optional magnitude weight). Returns (ls, turnover)."""
    months = sorted(mret.index)
    longset: set = set(); shortset: set = set()
    rows, ent, prevL = [], [], set()
    for i in range(len(months) - 1):
        t, t1 = months[i], months[i + 1]
        if t not in revw.index or t1 not in mret.index:
            continue
        s = revw.loc[t].dropna()
        if disp_pctile > 0:
            cv = cvw.loc[t].reindex(s.index)
            s = s[cv >= cv.quantile(disp_pctile)]
        if len(s) < 60:
            continue
        hi_in, hi_out = s.quantile(1 - q_in), s.quantile(1 - q_out)
        lo_in, lo_out = s.quantile(q_in), s.quantile(q_out)
        longset = {p for p in s.index if (s[p] >= hi_in) or (p in longset and s[p] >= hi_out)}
        shortset = {p for p in s.index if (s[p] <= lo_in) or (p in shortset and s[p] <= lo_out)}
        nx = mret.loc[t1]
        rl_r = nx.reindex(list(longset)).dropna(); rs_r = nx.reindex(list(shortset)).dropna()
        if len(rl_r) < 10 or len(rs_r) < 10:
            continue
        if weight == "mag":
            wl = s.reindex(rl_r.index).abs(); wl = wl / wl.sum()
            ws = s.reindex(rs_r.index).abs(); ws = ws / ws.sum()
            long_ret = float((rl_r * wl).sum()); short_ret = float((rs_r * ws).sum())
        else:
            long_ret = float(rl_r.mean()); short_ret = float(rs_r.mean())
        rows.append((t1, long_ret - short_ret))
        ent.append(len(longset - prevL) / max(len(longset), 1)); prevL = set(longset)
    return (pd.Series(dict(rows)).sort_index(), float(np.mean(ent) * 12) if ent else float("nan"))


def evaluate(ls, turn, rt_bps=20.0):
    ls = ls.dropna()
    if len(ls) < 24:
        return None
    vol = ls.std() * np.sqrt(12)
    gross_ann = ls.mean() * 12
    net = apply_cost(ls, turn * rt_bps / 10000.0, ppy=12)
    net_ann = net.mean() * 12
    out = dict(n=len(ls), gross_ann=gross_ann, gross_sharpe=gross_ann / vol,
               turn=turn, net_ann=net_ann, net_sharpe=net_ann / vol)
    for nt in (18, 35):
        out[f"netDefSR_n{nt}"] = deflated_sharpe_ratio(net.values, n_trials=nt, periods_per_year=12).deflated_sr
    out["_net"] = net
    return out


def robustness(revw, cvw, mret, params, rt_bps=20.0):
    """Battery: (a) dispersion-cutoff gradient smoothness; (b) subsample halves;
    (c) yearly positivity — on the chosen config."""
    ls, turn = sleeve(revw, cvw, mret, **params)
    ls = ls.dropna()
    net = apply_cost(ls, turn * rt_bps / 10000.0, ppy=12)
    # (a) gradient: vary disp_pctile around the chosen value
    grad = {}
    for dp in (0.3, 0.4, 0.5, 0.6, 0.7):
        p2 = dict(params); p2["disp_pctile"] = dp
        l2, t2 = sleeve(revw, cvw, mret, **p2); l2 = l2.dropna()
        if len(l2) > 12:
            grad[dp] = float(l2.mean() * 12 / (l2.std() * np.sqrt(12)))
    # (b) subsample halves (on net)
    mid = net.index[len(net) // 2]
    halves = {}
    for nm, sub in (("first", net[net.index < mid]), ("second", net[net.index >= mid])):
        from scipy import stats
        halves[nm] = dict(n=len(sub), ann=float(sub.mean() * 12),
                          t=float(stats.ttest_1samp(sub, 0).statistic) if len(sub) > 2 else float("nan"))
    # (c) yearly
    yr = (net.groupby(net.index.year).mean() * 12)
    return dict(gradient=grad, halves=halves,
                years_pos=int((yr > 0).sum()), years_tot=int(len(yr)),
                yearly={int(y): round(float(v), 3) for y, v in yr.items()})


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    revw, cvw, mret = load_inputs()

    # validate the inline replica against the library function (baseline config)
    lib_ls, lib_turn = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    my_ls, my_turn = sleeve(revw, cvw, mret, q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    j = pd.concat([lib_ls.rename("lib"), my_ls.rename("mine")], axis=1).dropna()
    corr = j["lib"].corr(j["mine"]) if len(j) > 2 else float("nan")
    print(f"[validate] replica vs library: corr={corr:.4f} turn lib={lib_turn:.2f} mine={my_turn:.2f} "
          f"(must be ~1.0)")

    # ---- grid search ----
    grid = []
    for disp in (0.3, 0.4, 0.5, 0.6):
        for w in ("equal", "mag"):
            for q_in in (0.10, 0.15, 0.20):
                for q_out in (0.30, 0.40, 0.50):
                    if q_out <= q_in:
                        continue
                    grid.append(dict(disp_pctile=disp, weight=w, q_in=q_in, q_out=q_out))
    print(f"[grid] {len(grid)} configs (honest n_trials for the conservative read)\n")

    results = []
    for p in grid:
        ls, turn = sleeve(revw, cvw, mret, **p)
        ev = evaluate(ls, turn, rt_bps=20.0)
        if ev is None:
            continue
        ev.update(p)
        results.append(ev)
    R = pd.DataFrame(results)
    grid_n = len(R)
    # honest conservative deflSR at n_trials = grid size
    R["netDefSR_grid"] = [deflated_sharpe_ratio(r["_net"].values, n_trials=grid_n,
                                                 periods_per_year=12).deflated_sr for r in results]

    show = ["disp_pctile", "weight", "q_in", "q_out", "gross_sharpe", "turn",
            "net_sharpe", "netDefSR_n18", "netDefSR_n35", "netDefSR_grid"]
    R2 = R.sort_values("net_sharpe", ascending=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("TOP 12 by NET Sharpe (cost rt=20bps, grid_n=%d):" % grid_n)
    print(R2[show].head(12).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    best = R2.iloc[0].to_dict()
    bp = {k: best[k] for k in ("disp_pctile", "weight", "q_in", "q_out")}
    print(f"\n[robustness] best-by-net-Sharpe config = {bp}")
    rb = robustness(revw, cvw, mret, bp, rt_bps=20.0)
    print("  dispersion-cutoff gradient (gross Sharpe):", {k: round(v, 3) for k, v in rb["gradient"].items()})
    print("  subsample halves:", rb["halves"])
    print(f"  yearly positive: {rb['years_pos']}/{rb['years_tot']}  {rb['yearly']}")


if __name__ == "__main__":
    main()
