"""engine/validation/edgar_8k.py — 2nd-alpha search: 8-K material-event items.

Last free-data direction. Uses SEC EDGAR submissions API, which returns each
8-K's ITEM numbers directly — so the signal is DETERMINISTIC (item-type events),
deliberately avoiding LLM text-sentiment, which carries a memory-contamination
validation pitfall (pre-cutoff events: an LLM can recall the outcome) and cost.

8-K item taxonomy (high-signal subset):
  2.02 results of operations (earnings — OVERLAPS PEAD, excluded from novel signal)
  1.03 bankruptcy ; 2.05 exit/restructuring costs ; 3.01 delisting notice
  4.02 non-reliance on prior financials (restatement) ; 5.02 exec/director departure
  1.01 material agreement ; 8.01 other events
The novel (non-earnings) test: do ADVERSE-event 8-Ks (distress/restatement/
forced departure) predict negative drift, and benign/agreement 8-Ks positive?

Screened through alpha_factory.gate(); GREEN-only deploys.

VERDICT (2026-05-20, CONFIRMED on the FULL pull — 812k filings / 6822 CIKs):
RED. (First read on a 42% partial gave composite -3.62%/t=-0.45; full sample
-4.04%/t=-0.58/Sharpe -0.18 — identical conclusion.) Item-type next-month excess returns: only 2.05 restructuring is
directionally meaningful (-13.3%/yr, t=-1.94) but thin/non-robust; 4.02
restatement -13.6% (n=24, too thin); others noisy (3.01 +12% t=0.9, 5.02
mixed). The COMPOSITE adverse-event short (1.03/2.05/3.01/4.02) is DEAD:
-3.62%/yr, t=-0.45, Sharpe -0.14, corr 0.03 w/ D_PEAD. Completing the pull was
NOT pursued — more CIKs won't make a t=-0.45 signal significant. The
deterministic 8-K item-type approach yields no tradeable edge; the stronger
disclosure signal is "Lazy Prices" 10-K text-change (Cohen-Malloy-Nguyen 2020),
a separate Tier-3 build. Partial cache kept at _edgar_8k_items_partial.parquet.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

import pandas as pd

logger = logging.getLogger(__name__)

_CIK_LIST = "data/cache/_8k_cik_list.parquet"
_8K_CACHE = "data/cache/_edgar_8k_items.parquet"
_UA = "research ${USER_EMAIL}"


_8K_PARTIAL = "data/cache/_edgar_8k_items_partial.parquet"


def fetch_8k_items(force: bool = False, sleep: float = 0.15) -> pd.DataFrame:
    """Pull recent submissions per CIK from data.sec.gov, extract 8-K filings
    (cik, filing_date, items). RESUMABLE: checkpoints to a partial cache every
    1000 CIKs and skips already-fetched CIKs on re-run, with 429 backoff.
    Rate-limited under SEC's 10 req/s."""
    import os
    if os.path.exists(_8K_CACHE) and not force:
        return pd.read_parquet(_8K_CACHE)
    ciks = pd.read_parquet(_CIK_LIST)["cik"].astype(int).tolist()

    rows = []
    done = set()
    if os.path.exists(_8K_PARTIAL):
        prev = pd.read_parquet(_8K_PARTIAL)
        rows = list(prev.itertuples(index=False, name=None))
        done = set(prev["cik"].unique())
        logger.info("resuming 8-K fetch: %d CIKs already done", len(done))

    todo = [c for c in ciks if c not in done]
    for i, cik in enumerate(todo):
        url = "https://data.sec.gov/submissions/CIK%010d.json" % cik
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                d = json.loads(urllib.request.urlopen(req, timeout=30).read())
                rec = d["filings"]["recent"]
                for form, fdate, items in zip(rec["form"], rec["filingDate"], rec["items"]):
                    if form == "8-K" and items:
                        rows.append((cik, fdate, items))
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                logger.debug("8-K fetch failed for CIK %d: %s", cik, exc)
                break
        if i % 1000 == 0 and rows:
            pd.DataFrame(rows, columns=["cik", "filing_date", "items"]).to_parquet(_8K_PARTIAL, index=False)
            logger.info("8-K fetch %d/%d (%d filings, checkpointed)", i, len(todo), len(rows))
        time.sleep(sleep)

    df = pd.DataFrame(rows, columns=["cik", "filing_date", "items"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df = df.drop_duplicates()
    df.to_parquet(_8K_CACHE, index=False)
    logger.info("8-K items: %d filings, %d CIKs", len(df), df["cik"].nunique())
    return df
