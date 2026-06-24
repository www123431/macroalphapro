"""scripts/run_path_ak_verdict.py — Path AK commodity momentum v3 alpha verdict on 5-sleeve baseline."""
from __future__ import annotations
import datetime, json, math, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from scripts.run_path_v_verdict import WEEKLY_RFR, G1_SHARPE_THRESHOLD, G2_T_THRESHOLD, G3_RHO_THRESHOLD, NW_LAG, CRISIS_WINDOWS, annualized_sharpe, newey_west_t
from scripts.run_path_aa_verdict import peak_to_trough_dd, crisis_window_dd, load_spy_weekly
from scripts.run_v3_retro_analysis import G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD


SLEEVE_5_WEIGHTS = {"K1_BAB": 0.324, "D_PEAD": 0.243, "PATH_N": 0.243, "CTA_PQTIX": 0.090, "AC_proxy_AB_2014_23": 0.100}
ALPHA_WEIGHT_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20]
SPEC_WINDOW_START = "2014-09-12"
SPEC_WINDOW_END   = "2023-12-29"


def main() -> None:
    print("=" * 78); print("Path AK Commodity Momentum — v3 ALPHA on 5-sleeve baseline"); print("=" * 78); print()

    pv = pd.read_parquet(REPO_ROOT / "data/portfolio_replay/v1_path_ak_commodity_mom_weekly.parquet")
    pv.index = pd.to_datetime(pv.index)
    pv = pv.loc[SPEC_WINDOW_START:SPEC_WINDOW_END]
    pv_net = pv["net"].dropna()
    print(f"Path AK (filtered to spec window): n={len(pv_net)} weeks")

    ex = pd.read_parquet(REPO_ROOT / "data/portfolio_replay/v2_per_strategy_returns_5sleeve_weekly.parquet").astype("float64").fillna(0.0)
    ex.index = pd.to_datetime(ex.index)
    baseline = pd.Series(0.0, index=ex.index)
    for col, w in SLEEVE_5_WEIGHTS.items():
        if col in ex.columns: baseline += w * ex[col]

    spy_weekly = load_spy_weekly()
    print()

    sharpe_net = annualized_sharpe(pv_net)
    g1_pass = sharpe_net >= G1_SHARPE_THRESHOLD
    print(f"G1     Sharpe net ann.  = {sharpe_net:+.4f}   -> {'PASS' if g1_pass else 'FAIL'}")
    excess = (pv_net - WEEKLY_RFR).to_numpy()
    nw_t = newey_west_t(excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2     Newey-West t     = {nw_t:+.4f}   -> {'PASS' if g2_pass else 'FAIL'}")

    common = pv_net.index.intersection(ex.index)
    pv_a = pv_net.loc[common]; ex_a = ex.loc[common]
    rho_vec = {col: float(pv_a.corr(ex_a[col])) for col in ex_a.columns}
    max_abs_rho = max(abs(v) for v in rho_vec.values())
    g3_pass = max_abs_rho <= G3_RHO_THRESHOLD
    print(f"G3     max |rho|        = {max_abs_rho:+.4f}   -> {'PASS' if g3_pass else 'FAIL'}")
    for col, r in rho_vec.items():
        print(f"         rho(AK, {col:<22}) = {r:+.4f}")

    print()
    print("G5-v2  Crisis DD attenuation vs SPY:")
    g5_wins = 0; g5_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        spy_dd = crisis_window_dd(spy_weekly, start, end)
        cand_dd = crisis_window_dd(pv_net, start, end)
        win = cand_dd >= spy_dd
        if win: g5_wins += 1
        g5_details[label] = {"spy_dd": spy_dd, "cand_dd": cand_dd, "win": win}
        marker = "WIN " if win else "LOSE"
        print(f"         [{marker}] {label}: AK DD = {cand_dd*100:+.2f}%, SPY DD = {spy_dd*100:+.2f}%")
    g5_pass = g5_wins >= 2
    print(f"       result: {g5_wins}/3   -> {'PASS' if g5_pass else 'FAIL'}")

    print()
    print("G6     Portfolio Sharpe lift on 5-sleeve:")
    common2 = pv_net.index.intersection(baseline.index)
    base_sharpe = annualized_sharpe(baseline.loc[common2])
    print(f"         base 5-sleeve Sharpe = {base_sharpe:+.4f}")
    best_w, best_blend_sh, g6_detail = None, -1e9, {}
    for w in ALPHA_WEIGHT_SWEEP:
        blend = (1-w) * baseline.loc[common2] + w * pv_net.loc[common2]
        bsh = annualized_sharpe(blend); g6_detail[w] = float(bsh)
        print(f"         w={int(w*100):2d}%: blend Sharpe = {bsh:+.4f}")
        if bsh > best_blend_sh: best_blend_sh = bsh; best_w = w
    lift_pct = (best_blend_sh - base_sharpe) / abs(base_sharpe) if abs(base_sharpe) > 1e-6 else 0
    g6_pass = lift_pct >= G6_ALPHA_SHARPE_IMPROVEMENT_THRESHOLD
    print(f"       Best lift = {lift_pct*100:+.2f}% at w={int(best_w*100)}%   -> {'PASS' if g6_pass else 'FAIL'}")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g5_pass, g6_pass])
    verdict = "PASS" if n_pass == 5 else "MARGINAL" if n_pass == 4 else "FAIL"
    print("=" * 78); print(f"  Path AK: {n_pass}/5 v3 alpha gates PASS  ->  VERDICT: {verdict}"); print("=" * 78)

    today = datetime.date.today()
    payload = {
        "spec_id": 83, "spec_hash": "daf5f9a62d1562b744575ca224622ac71e6adde2",
        "spec_name": "Commodity Momentum on 5 DB ETFs (Erb-Harvey 2006)",
        "framework": "v3 alpha", "category": "alpha", "run_date": today.isoformat(),
        "verdict": verdict, "n_pass": int(n_pass),
        "gates": {
            "G1_sharpe": {"value": float(sharpe_net), "threshold": G1_SHARPE_THRESHOLD, "pass": bool(g1_pass)},
            "G2_nw_t": {"value": float(nw_t), "threshold": G2_T_THRESHOLD, "pass": bool(g2_pass)},
            "G3_max_abs_rho": {"value": float(max_abs_rho), "threshold": G3_RHO_THRESHOLD, "pass": bool(g3_pass), "rho_by_sleeve": rho_vec},
            "G5_v2_dd_atten": {"wins": int(g5_wins), "threshold": 2, "pass": bool(g5_pass), "detail": g5_details},
            "G6_portfolio_sharpe": {"base_sharpe": float(base_sharpe), "best_blend_sharpe": float(best_blend_sh),
                                     "best_w": float(best_w), "sharpe_improvement_pct": float(lift_pct),
                                     "pass": bool(g6_pass), "sharpe_by_w": g6_detail},
        },
        "summary_stats": {
            "weekly_mean_net": float(pv_net.mean()), "weekly_std_net": float(pv_net.std()),
            "annualized_vol": float(pv_net.std() * math.sqrt(52)),
            "annual_tc_drag": float(pv["tc"].sum() / (len(pv) / 52.0)),
            "max_dd_overall": peak_to_trough_dd(pv_net),
        },
    }
    out = REPO_ROOT / "data/portfolio_replay" / f"path_ak_verdict_{today.isoformat()}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Verdict saved: {out}")


if __name__ == "__main__":
    main()
