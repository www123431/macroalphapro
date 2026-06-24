"""engine/line_c/gate.py — two-lens incremental-over-SUE test for transcript text.

Lens A (statistical power): Fama-MacBeth cross-sectional regressions per quarter,
    fwd_ret ~ z(text_feature) + z(controls), coefficient time series aggregated
    with Newey-West HAC t (the powered statement; a tight null here IS a result).
Lens B (economic significance): quarterly decile L/S on the text score, annualized
    Sharpe + Deflated Sharpe (Bailey-Lopez de Prado) with HONEST n_trials.

Audit battery: subperiod split, raw vs sector-neutral, horizon 21 vs 63, and the
incremental coefficient (text residualized on controls) vs univariate.

CONTROLS isolate "soft information the number misses" (Tetlock 2008; Cohen-Malloy
2020): SUE + announcement CAR(-1,+1) + momentum + size. A text effect that
survives these is genuinely incremental over PEAD.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from engine.validation.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)

CONTROLS = ["sue", "car_3d", "mom_12_1", "log_size"]


def _zscore_within(df: pd.DataFrame, cols, group="quarter") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        g = out.groupby(group)[c]
        out[c + "_z"] = (out[c] - g.transform("mean")) / (g.transform("std") + 1e-9)
    return out


def fama_macbeth(panel: pd.DataFrame, feat: str, target: str,
                 controls=CONTROLS, nw_lags: int = 4) -> dict:
    """FM regression of target on z(feat) [+ z(controls)]; NW-HAC t on the mean coef.

    Returns dict for BOTH univariate and +controls specs.
    """
    cols = [feat] + list(controls) + [target, "quarter"]
    d = panel[cols].replace([np.inf, -np.inf], np.nan).dropna()
    d = _zscore_within(d, [feat] + list(controls))
    res = {}
    for spec, regs in [("uni", [feat]), ("ctrl", [feat] + list(controls))]:
        coefs = []
        for q, g in d.groupby("quarter"):
            if len(g) < 30:
                continue
            X = sm.add_constant(g[[r + "_z" for r in regs]].values)
            y = g[target].values
            try:
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                coefs.append(b[1])   # coefficient on feat (first regressor after const)
            except Exception:
                continue
        coefs = np.asarray(coefs)
        if len(coefs) < 8:
            res[spec] = {"n_q": len(coefs), "coef": np.nan, "t": np.nan}
            continue
        m = sm.OLS(coefs, np.ones(len(coefs))).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
        res[spec] = {"n_q": int(len(coefs)), "coef": float(m.params[0]), "t": float(m.tvalues[0]),
                     "mean_bps": float(coefs.mean() * 1e4)}
    return res


def add_residual(panel: pd.DataFrame, feat: str, controls=CONTROLS, group="quarter") -> pd.DataFrame:
    """Add `{feat}_resid`: feat orthogonalized to controls cross-sectionally per
    quarter (the INCREMENTAL text score for Lens B — what survives SUE+CAR+styles)."""
    out = panel.copy()
    d = _zscore_within(out[[feat] + list(controls) + [group]].replace([np.inf, -np.inf], np.nan).dropna(),
                       [feat] + list(controls))
    resid = pd.Series(np.nan, index=out.index)
    for q, g in d.groupby(group):
        if len(g) < 30:
            continue
        X = sm.add_constant(g[[c + "_z" for c in controls]].values)
        y = g[feat + "_z"].values
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid.loc[g.index] = y - X @ b
    out[feat + "_resid"] = resid
    return out


def decile_ls(panel: pd.DataFrame, score: str, target: str, n_bins: int = 10) -> pd.Series:
    """Quarterly long-top / short-bottom decile spread on `score`; returns quarterly series."""
    d = panel[[score, target, "quarter"]].replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for q, g in d.groupby("quarter"):
        if len(g) < n_bins * 5:
            continue
        try:
            bins = pd.qcut(g[score].rank(method="first"), n_bins, labels=False)
        except Exception:
            continue
        top = g.loc[bins == n_bins - 1, target].mean()
        bot = g.loc[bins == 0, target].mean()
        rows.append((q, top - bot))
    s = pd.Series(dict(rows)).sort_index()
    s.index = pd.PeriodIndex(s.index, freq="Q")
    return s.rename(score)


def sharpe_block(ls: pd.Series, n_trials: int, ppy: int = 4) -> dict:
    ls = ls.dropna()
    if len(ls) < 8:
        return {"n": len(ls), "sharpe_ann": np.nan, "deflated_sr": np.nan}
    mean, sd = ls.mean(), ls.std(ddof=1)
    sharpe_ann = (mean / sd) * np.sqrt(ppy) if sd > 0 else np.nan
    dsr = deflated_sharpe_ratio(ls.values, n_trials=n_trials, periods_per_year=ppy)
    return {"n": int(len(ls)), "mean_q_bps": float(mean * 1e4), "sharpe_ann": float(sharpe_ann),
            "deflated_sr": float(dsr.deflated_sr), "psr0": float(dsr.psr_vs_zero)}


def run_gate(panel: pd.DataFrame, primary_feat="finbert_tone_sn", target="fwd_ret_63",
             n_trials: int = 24) -> dict:
    """Primary spec + audit battery. n_trials = honest grid size for deflated SR."""
    out = {"primary_feat": primary_feat, "target": target, "n_events": int(len(panel))}

    # ---- Lens A: Fama-MacBeth incremental ----
    print("\n===== LENS A — Fama-MacBeth (NW-HAC t) =====")
    print(f"{'feature':26s} {'spec':5s} {'n_q':>4s} {'coef_bps':>10s} {'t':>7s}")
    fm_feats = [c for c in ["finbert_tone", "finbert_tone_sn", "lm_net_tone", "lm_net_tone_sn",
                            "lm_uncertainty_prop", "d_finbert_tone", "d_lm_net_tone",
                            "prior_call_cosine", "numeric_density"] if c in panel.columns]
    out["lens_A"] = {}
    for f in fm_feats:
        r = fama_macbeth(panel, f, target)
        out["lens_A"][f] = r
        for spec in ("uni", "ctrl"):
            rr = r.get(spec, {})
            print(f"{f:26s} {spec:5s} {rr.get('n_q',0):>4d} "
                  f"{rr.get('mean_bps', float('nan')):>10.2f} {rr.get('t', float('nan')):>7.2f}")

    # ---- Lens B: decile L/S + deflated Sharpe ----
    # residualize the primary feature on controls -> the INCREMENTAL economic score
    if primary_feat in panel.columns:
        panel = add_residual(panel, primary_feat)
    print("\n===== LENS B — decile L/S + Deflated Sharpe (n_trials=%d) =====" % n_trials)
    print("  ('*_resid' = primary feature orthogonalized to SUE+CAR+mom+size = incremental)")
    print(f"{'score':28s} {'target':11s} {'n':>3s} {'meanQ_bps':>10s} {'SR_ann':>7s} {'deflSR':>7s}")
    ls_feats = list(dict.fromkeys(
        [c for c in [primary_feat, primary_feat + "_resid", "finbert_tone", "lm_net_tone_sn"]
         if c in panel.columns]))
    out["lens_B"] = {}
    for f in ls_feats:
        for tgt in [target, "fwd_ret_21"]:
            ls = decile_ls(panel, f, tgt)
            sb = sharpe_block(ls, n_trials=n_trials)
            out["lens_B"][f"{f}|{tgt}"] = sb
            print(f"{f:28s} {tgt:11s} {sb['n']:>3d} {sb.get('mean_q_bps', float('nan')):>10.2f} "
                  f"{sb.get('sharpe_ann', float('nan')):>7.2f} {sb.get('deflated_sr', float('nan')):>7.3f}")

    # ---- Audit: subperiod + raw-vs-SN on primary ----
    print("\n===== AUDIT BATTERY (primary=%s, target=%s) =====" % (primary_feat, target))
    rdq = pd.to_datetime(panel["rdq"])
    for label, mask in [("2011-2017", rdq.dt.year <= 2017), ("2018-2024", rdq.dt.year >= 2018)]:
        sub = panel[mask]
        r = fama_macbeth(sub, primary_feat, target)
        ls = decile_ls(sub, primary_feat, target)
        sb = sharpe_block(ls, n_trials=n_trials)
        print(f"  {label}: FM ctrl t={r.get('ctrl',{}).get('t', float('nan')):+.2f} "
              f"| L/S SR={sb.get('sharpe_ann', float('nan')):+.2f} deflSR={sb.get('deflated_sr', float('nan')):.3f} (n={sb['n']})")
        out[f"audit_{label}"] = {"fm_ctrl_t": r.get("ctrl", {}).get("t"), **sb}
    return out


if __name__ == "__main__":
    import sys, json, warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    panel_path = sys.argv[1] if len(sys.argv) > 1 else str(__import__("engine.line_c.feature_panel", fromlist=["PANEL_OUT"]).PANEL_OUT)
    panel = pd.read_parquet(panel_path)
    feat = sys.argv[2] if len(sys.argv) > 2 else "finbert_tone_sn"
    res = run_gate(panel, primary_feat=feat)
    print("\n=== JSON ===")
    print(json.dumps(res, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x))
