"""Live-data feasibility gate: can D_PEAD's inputs be sourced from a free,
non-WRDS vendor (yfinance / Yahoo) and reproduce the WRDS-validated signal?
Throwaway validation harness (underscore-prefixed), not part of the engine."""
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SAMPLE = "data/cache/_yf_feas_sample.csv"
CACHE = "data/cache/_yf_earnings_feas.parquet"
PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"


def pull(samp):
    import yfinance as yf
    rows = []
    for tk in samp:
        try:
            ed = yf.Ticker(tk).get_earnings_dates(limit=40)
            if ed is not None and len(ed):
                ed = ed.reset_index()
                ed["ticker"] = tk
                rows.append(ed)
        except Exception:
            pass
        time.sleep(0.3)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df.to_parquet(CACHE, index=False)
    return df


def main():
    samp = pd.read_csv(SAMPLE, header=None)[0].tolist()
    allyf = pd.read_parquet(CACHE) if os.path.exists(CACHE) else pull(samp)
    allyf.columns = [c.lower().replace(" ", "_").replace("(%)", "_pct") for c in allyf.columns]
    dc = [c for c in allyf.columns if "date" in c][0]
    allyf["ed"] = pd.to_datetime(allyf[dc], utc=True).dt.tz_localize(None)
    allyf = allyf.dropna(subset=["reported_eps"])
    print(f"yfinance coverage: {allyf.ticker.nunique()}/{len(samp)} tickers; "
          f"{len(allyf)} past-earnings rows; {allyf.ed.min().date()}..{allyf.ed.max().date()}")

    panel = pd.read_parquet(PANEL)
    panel["rdq"] = pd.to_datetime(panel["rdq"])
    pr = panel[(panel.rdq >= "2020-01-01") & (panel.ticker.isin(allyf.ticker.unique()))].dropna(subset=["sue", "rdq"])

    matched = []
    for tk, g in pr.groupby("ticker"):
        yg = allyf[allyf.ticker == tk]
        for _, row in g.iterrows():
            d = (yg["ed"] - row["rdq"]).abs()
            if len(d) and d.min() <= pd.Timedelta(days=7):
                j = d.idxmin()
                matched.append({"datediff": (yg.loc[j, "ed"] - row["rdq"]).days,
                                "sue": row["sue"], "surprise_pct": yg.loc[j, "surprise_pct"]})
    m = pd.DataFrame(matched).dropna(subset=["surprise_pct"])
    sp = m.sue.corr(m.surprise_pct, method="spearman")
    sign_agree = (np.sign(m.sue) == np.sign(m.surprise_pct)).mean()
    print(f"panel quarters (2020+, covered tickers): {len(pr)}")
    print(f"MATCHED within +/-7d: {len(m)} = {len(m) / max(len(pr), 1) * 100:.0f}% match rate")
    print(f"date accuracy: median |diff|={m.datediff.abs().median():.1f}d, "
          f"within 2d={(m.datediff.abs() <= 2).mean() * 100:.0f}%, "
          f"within 4d={(m.datediff.abs() <= 4).mean() * 100:.0f}%")
    print(f"SIGNAL FIDELITY: sign(WRDS SUE)==sign(yf surprise%) = {sign_agree * 100:.0f}%")
    print(f"rank corr(SUE, yf surprise) = {sp:.3f}")


if __name__ == "__main__":
    main()
