"""scripts/run_path_aj_verdict.py — Path AJ v3 overlay class verdict."""
from __future__ import annotations
import datetime, json, math, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import pandas as pd
from scripts.run_path_v_verdict import WEEKLY_RFR, CRISIS_WINDOWS, annualized_sharpe
from scripts.run_path_aa_verdict import peak_to_trough_dd, crisis_window_dd


# Locked v3 overlay-specific gate thresholds per spec §2.5
G1_OVERLAY_SHARPE_LIFT_MIN: float = 0.05   # ≥ +5%
G2_OVERLAY_VOL_INC_MAX:     float = 0.10   # ≤ +10%
G3_OVERLAY_MAXDD_RATIO_MAX: float = 1.0    # ≤ 1.0 of baseline (no worse)
G4_OVERLAY_CRISIS_MIN:      int   = 2      # ≥ 2/3 crisis windows


def main() -> None:
    print("=" * 78)
    print("Path AJ Vol-Target Overlay on 5-Sleeve — v3 OVERLAY class verdict")
    print("=" * 78)
    print()

    pv = pd.read_parquet(REPO_ROOT / "data/portfolio_replay/v1_path_aj_voltarget_5sleeve_weekly.parquet")
    pv.index = pd.to_datetime(pv.index)
    base   = pv["base_return"].dropna()
    aj_net = pv["aj_net"].dropna()
    print(f"AJ net returns: n={len(aj_net)} weeks")
    print(f"Avg scale (post-warmup): {pv['scale'].iloc[22:].mean():.3f}")
    raw_scale_hi = (pv['scale'] >= 1.499).sum() / len(pv) * 100
    raw_scale_lo = (pv['scale'] <= 0.501).sum() / len(pv) * 100
    print(f"Clamped LO ({0.5}x): {raw_scale_lo:.1f}% · HI ({1.5}x): {raw_scale_hi:.1f}%")
    print()

    # G1-overlay: Sharpe lift
    base_sharpe = annualized_sharpe(base)
    aj_sharpe   = annualized_sharpe(aj_net)
    if abs(base_sharpe) > 1e-6:
        sharpe_lift = (aj_sharpe - base_sharpe) / abs(base_sharpe)
    else:
        sharpe_lift = 0.0
    g1_pass = sharpe_lift >= G1_OVERLAY_SHARPE_LIFT_MIN
    print(f"G1-o  Sharpe lift     = {sharpe_lift*100:+.2f}%   "
          f"(baseline {base_sharpe:+.4f} -> AJ {aj_sharpe:+.4f})   "
          f"(>= +{G1_OVERLAY_SHARPE_LIFT_MIN*100:.0f}%)   -> {'PASS' if g1_pass else 'FAIL'}")

    # G2-overlay: vol increase
    base_vol = base.std() * math.sqrt(52)
    aj_vol   = aj_net.std() * math.sqrt(52)
    vol_change = (aj_vol - base_vol) / base_vol
    g2_pass = vol_change <= G2_OVERLAY_VOL_INC_MAX
    print(f"G2-o  Vol change      = {vol_change*100:+.2f}%   "
          f"(baseline {base_vol*100:.2f}% -> AJ {aj_vol*100:.2f}%)   "
          f"(<= +{G2_OVERLAY_VOL_INC_MAX*100:.0f}%)   -> {'PASS' if g2_pass else 'FAIL'}")

    # G3-overlay: Max DD no worse
    base_maxdd = peak_to_trough_dd(base)
    aj_maxdd   = peak_to_trough_dd(aj_net)
    maxdd_ratio = abs(aj_maxdd) / abs(base_maxdd) if abs(base_maxdd) > 1e-6 else float('inf')
    g3_pass = maxdd_ratio <= G3_OVERLAY_MAXDD_RATIO_MAX
    print(f"G3-o  Max DD ratio    = {maxdd_ratio:.3f}   "
          f"(baseline {base_maxdd*100:+.2f}% -> AJ {aj_maxdd*100:+.2f}%)   "
          f"(<= {G3_OVERLAY_MAXDD_RATIO_MAX})   -> {'PASS' if g3_pass else 'FAIL'}")

    # G4-overlay: crisis attenuation
    print()
    print("G4-o  Crisis DD attenuation:")
    g4_wins = 0
    g4_details = {}
    for label, (start, end) in CRISIS_WINDOWS.items():
        base_dd = crisis_window_dd(base, start, end)
        aj_dd = crisis_window_dd(aj_net, start, end)
        win = aj_dd >= base_dd
        if win: g4_wins += 1
        g4_details[label] = {"base_dd": base_dd, "aj_dd": aj_dd, "win": win}
        marker = "WIN " if win else "LOSE"
        print(f"        [{marker}] {label}: AJ DD = {aj_dd*100:+.2f}%, base DD = {base_dd*100:+.2f}%")
    g4_pass = g4_wins >= G4_OVERLAY_CRISIS_MIN
    print(f"      result: {g4_wins}/3 (>= {G4_OVERLAY_CRISIS_MIN})   -> {'PASS' if g4_pass else 'FAIL'}")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass])
    verdict = "PASS" if n_pass == 4 else "MARGINAL" if n_pass == 3 else "FAIL"
    print("=" * 78)
    print(f"  Path AJ: {n_pass}/4 v3 overlay gates PASS  ->  VERDICT: {verdict}")
    print("=" * 78)

    today = datetime.date.today()
    payload = {
        "spec_id": 82, "spec_hash": "568541f0e514acda9ea710244a88ca59da100af3",
        "spec_name": "Vol-Target Overlay on 5-Sleeve (Moreira-Muir 2017)",
        "framework": "v3 overlay class", "category": "overlay",
        "run_date": today.isoformat(),
        "baseline": "5-sleeve post-AC at PAPER_TRADE_SLEEVE_ALLOCATION",
        "params": {"sigma_target_ann": 0.08, "sigma_window_w": 22, "clamp": [0.5, 1.5]},
        "verdict": verdict, "n_pass": int(n_pass),
        "gates": {
            "G1_overlay_sharpe_lift":   {"value": float(sharpe_lift), "threshold": G1_OVERLAY_SHARPE_LIFT_MIN,
                                          "base_sharpe": float(base_sharpe), "aj_sharpe": float(aj_sharpe),
                                          "pass": bool(g1_pass)},
            "G2_overlay_vol_change":    {"value": float(vol_change), "threshold": G2_OVERLAY_VOL_INC_MAX,
                                          "base_vol": float(base_vol), "aj_vol": float(aj_vol),
                                          "pass": bool(g2_pass)},
            "G3_overlay_maxdd_ratio":   {"value": float(maxdd_ratio), "threshold": G3_OVERLAY_MAXDD_RATIO_MAX,
                                          "base_maxdd": float(base_maxdd), "aj_maxdd": float(aj_maxdd),
                                          "pass": bool(g3_pass)},
            "G4_overlay_crisis_atten":  {"wins": int(g4_wins), "threshold": G4_OVERLAY_CRISIS_MIN,
                                          "detail": g4_details, "pass": bool(g4_pass)},
        },
        "summary_stats": {
            "n_weeks": int(len(aj_net)),
            "weekly_mean_aj_net": float(aj_net.mean()),
            "weekly_std_aj_net": float(aj_net.std()),
            "annualized_tc_drag": float(pv["tc"].sum() / (len(pv) / 52.0)),
            "avg_scale_post_warmup": float(pv["scale"].iloc[22:].mean()),
            "pct_clamped_low": float(raw_scale_lo),
            "pct_clamped_high": float(raw_scale_hi),
        },
    }
    out = REPO_ROOT / "data/portfolio_replay" / f"path_aj_verdict_{today.isoformat()}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Verdict saved: {out}")


if __name__ == "__main__":
    main()
