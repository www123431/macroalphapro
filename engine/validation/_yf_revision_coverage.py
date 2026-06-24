"""Feasibility gate #2: are the analyst-revision fields (up/down revision counts,
#analysts, estimate dispersion) obtainable for the live universe from free
yfinance — the last data hurdle for the strong-YELLOW 2nd alpha? Throwaway."""
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SAMPLE = "data/cache/_yf_feas_sample.csv"


def main():
    import yfinance as yf
    samp = pd.read_csv(SAMPLE, header=None)[0].tolist()
    rows = []
    for tk in samp:
        rec = {"ticker": tk, "rev_ok": False, "est_ok": False,
               "numest": np.nan, "rev_ratio": np.nan, "disp_cv": np.nan}
        t = yf.Ticker(tk)
        for _ in range(2):
            try:
                rv = t.get_eps_revisions()
                if rv is not None and "0q" in rv.index:
                    up = rv.loc["0q", "upLast30days"]; dn = rv.loc["0q", "downLast30days"]
                    if pd.notna(up) and pd.notna(dn):
                        rec["rev_ok"] = True; rec["_up"] = up; rec["_dn"] = dn
                break
            except Exception:
                time.sleep(2)
        for _ in range(2):
            try:
                es = t.get_earnings_estimate()
                if es is not None and "0q" in es.index:
                    n = es.loc["0q", "numberOfAnalysts"]; avg = es.loc["0q", "avg"]
                    lo = es.loc["0q", "low"]; hi = es.loc["0q", "high"]
                    if pd.notna(n) and n > 0 and pd.notna(avg) and avg != 0:
                        rec["est_ok"] = True; rec["numest"] = n
                        rec["disp_cv"] = abs((hi - lo) / avg) if pd.notna(hi) and pd.notna(lo) else np.nan
                        if rec["rev_ok"]:
                            rec["rev_ratio"] = (rec.pop("_up") - rec.pop("_dn")) / n
                break
            except Exception:
                time.sleep(2)
        rows.append(rec)
        time.sleep(0.4)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"sample: {n} universe tickers")
    print(f"get_eps_revisions populated:   {df.rev_ok.sum()}/{n} = {df.rev_ok.mean()*100:.0f}%")
    print(f"get_earnings_estimate populated:{df.est_ok.sum()}/{n} = {df.est_ok.mean()*100:.0f}%")
    both = (df.rev_ok & df.est_ok)
    print(f"BOTH (full signal computable):  {both.sum()}/{n} = {both.mean()*100:.0f}%")
    d = df[both]
    print(f"\nfor the {both.sum()} fully-covered names:")
    print(f"  #analysts: median={d.numest.median():.0f}, min={d.numest.min():.0f}")
    print(f"  rev_ratio spread: [{d.rev_ratio.min():.2f}, {d.rev_ratio.max():.2f}], "
          f"std={d.rev_ratio.std():.2f}  (need cross-sectional spread to rank)")
    print(f"  dispersion CV proxy: median={d.disp_cv.median():.3f}, "
          f"[{d.disp_cv.min():.3f}, {d.disp_cv.max():.3f}]")
    miss = df[~both]["ticker"].tolist()
    print(f"\nuncovered: {miss}")


if __name__ == "__main__":
    main()
