"""engine/validation/_cn_pead_run.py — run the China A-share PEAD audit battery
and print a structured verdict report. Loads the US D_PEAD book daily series and
resamples to monthly for the diversification correlation.

Run: python -u -m engine.validation._cn_pead_run
"""
from __future__ import annotations

import logging

import pandas as pd

from engine.validation import cn_pead_data as cn


def _dpead_monthly() -> "pd.Series | None":
    try:
        d = pd.read_parquet("data/cache/_dpead_recon_base.parquet")
        s = d.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return ((1 + s).resample("ME").prod() - 1).rename("dpead")
    except Exception as exc:  # corr is secondary; never block the audit on it
        logging.getLogger(__name__).warning("dpead monthly load failed: %s", exc)
        return None


def _fmt(d: dict) -> str:
    if "note" in d and "gross_ann" not in d:
        return f"{d['name']:7s} n={d.get('n','?')}  ({d['note']})"
    return ("{name:7s} n={n:3d} | gross ann {gross_ann:+6.1%} sharpe {gross_sharpe:5.2f} "
            "defSR {gross_defSR:5.3f} | turn {ann_turnover:4.1f}x cost {ann_cost:5.2%} | "
            "NET ann {net_ann:+6.1%} sharpe {net_sharpe:5.2f} defSR {net_defSR:5.3f}").format(**d)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = cn.cn_pead_audit(dpead_monthly=_dpead_monthly())

    print("\n" + "=" * 78)
    print("CHINA A-SHARE PEAD — full audit battery")
    print("=" * 78)
    print(f"window {res['window'][0][:10]}..{res['window'][1][:10]}  months={res['n_months']}")

    pe = res["per_event"]
    if pe:
        print("\n[1] PER-EVENT 60d CAR (context — NOT tradeable):")
        print(f"    n={pe['n']}  hi-lo spread {pe['spread']:+.2%}  long {pe['long']:+.2%} "
              f"short {pe['short']:+.2%}  t={pe['t']:.2f}")
        print(f"    NB: {pe['note']}")

    print("\n[2/3/4] CALENDAR-TIME, after A-share cost, deflated SR (ppy=12):")
    print("        legs: lo_*=LONG-ONLY (deployable, short 融券 restricted) | ls_*=L/S (academic)")
    print("        weight: *_ew=equal | *_vw=ADV liquidity-weight (bid-ask-bounce guard)")
    for k in ("lo_vw", "lo_ew", "ls_vw", "ls_ew"):
        if k in res["constructions"]:
            print("    " + _fmt(res["constructions"][k]))

    rg = res["regimes"]
    print(f"\n[5] REGIME (deployable={res['deployable']}):")
    for nm in ("first_half", "second_half"):
        if nm in rg:
            print(f"    {nm:12s} n={rg[nm]['n']:3d}  ann {rg[nm]['ann']:+.1%}  t={rg[nm]['t']:.2f}")
    print(f"    yearly positive: {rg['years_positive']}/{rg['years_total']}  "
          f"-> {rg['yearly_ann']}")

    if res["corr_dpead"]:
        c = res["corr_dpead"]
        print(f"\n[7] CORR with US D_PEAD book: {c['corr']:+.2f} (n={c['n']} months)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
