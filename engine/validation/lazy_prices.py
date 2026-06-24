"""engine/validation/lazy_prices.py — Tier-3 2nd-alpha: "Lazy Prices" 10-K text change.

Cohen-Malloy-Nguyen 2020 (JF): firms that CHANGE their 10-K language year-over-
year (LOW similarity vs the prior year's filing) subsequently UNDERPERFORM;
firms that keep disclosures the SAME (high similarity) outperform — changes bury
bad news. Signal = cosine similarity of consecutive annual 10-Ks. Long
high-similarity, short low-similarity.

Why this fits the agenda (Tier 3 — the AI/NLP edge):
  - DETERMINISTIC (cosine on bag-of-words/TF-IDF) — NO LLM judgment, so it is
    immune to the LLM-memory-contamination validation pitfall.
  - different trigger than D_PEAD (disclosure-text change vs earnings surprise)
    -> candidate low-correlation diversifier.
  - free data (SEC EDGAR 10-Ks) + returns we already cache. NO WRDS.

Heavy data-engineering: 10-Ks are large (1-10MB HTML). Scoped to a subset for a
first proof; resumable + throttled (SEC <10 req/s; lesson from the WRDS lockout
=> be gentle, checkpoint, never retry-storm).

Pipeline:
  fetch_10k_list   -> per-CIK 10-K filings (year, accession, primaryDocument)
  fetch_10k_texts  -> download + strip HTML -> cleaned token text, cached
  compute_similarity -> YoY cosine similarity per firm-year
  build_sleeve     -> monthly L/S long high-sim / short low-sim, gate-ready

VERDICT (2026-05-20): Jaccard YoY-similarity tertile L/S on 413 top-1500 firms
= **YELLOW** — the strongest + most-deployable 2nd-alpha lead of the search.
Gate: net deflated SR 0.724 (clears the 0.70 ok-bar, below the 0.90 GREEN bar,
n_trials=8 multiple-testing included), residual alpha vs FF5+UMD 4.36%/yr
**t=2.01 (real alpha, NOT size/value beta)**, corr -0.07 w/ D_PEAD, turnover
0.7x/yr (cost ~0.2%/yr), LARGE-CAP (net-tradeable, no micro-cap wall). Caveat:
"front-loaded" (early period stronger -> watch for decay) + small 413-firm
sample. Iterate toward GREEN (richer text measure / bigger universe), not
deployed. (TF-IDF/raw-TF variants were weaker; Jaccard set-overlap best.)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_UA = "research ${USER_EMAIL}"
_LIST_CACHE = "data/cache/_lazy_10k_list.parquet"
_TEXT_DIR = "data/cache/lazy_10k_text"          # one cleaned .txt per accession
_SIM_CACHE = "data/cache/_lazy_10k_similarity.parquet"
_MAP = "data/cache/_cik_gvkey_permno_map.parquet"   # top-1500 universe


def fetch_10k_list(ciks: list[int], sleep: float = 0.15, force: bool = False) -> pd.DataFrame:
    """Per-CIK 10-K filings (cik, filing_date, accession, primary_doc) from the
    EDGAR submissions API. Resumable + throttled."""
    if os.path.exists(_LIST_CACHE) and not force:
        prev = pd.read_parquet(_LIST_CACHE)
        if set(ciks).issubset(set(prev["cik"].unique())):
            return prev
    rows = []
    done = set()
    _cols = ["cik", "filing_date", "accession", "primary_doc"]
    if os.path.exists(_LIST_CACHE):
        prev = pd.read_parquet(_LIST_CACHE)
        prev["filing_date"] = prev["filing_date"].astype(str).str.slice(0, 10)
        rows = list(prev[_cols].itertuples(index=False, name=None))
        done = set(prev["cik"].unique())
    todo = [c for c in ciks if c not in done]
    for i, cik in enumerate(todo):
        url = "https://data.sec.gov/submissions/CIK%010d.json" % cik
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": _UA})
                d = json.loads(urllib.request.urlopen(req, timeout=30).read())
                rec = d["filings"]["recent"]
                for form, fdate, acc, doc in zip(rec["form"], rec["filingDate"],
                                                 rec["accessionNumber"], rec["primaryDocument"]):
                    if form == "10-K":
                        rows.append((cik, fdate, acc, doc))
                break
            except Exception as exc:
                if "429" in str(exc) and attempt < 3:
                    time.sleep(2.0 * (attempt + 1)); continue
                logger.debug("10-K list fail CIK %d: %s", cik, exc); break
        if i % 500 == 0 and rows:
            pd.DataFrame(rows, columns=["cik", "filing_date", "accession", "primary_doc"]).to_parquet(_LIST_CACHE, index=False)
            logger.info("10-K list %d/%d (%d filings)", i, len(todo), len(rows))
        time.sleep(sleep)
    df = pd.DataFrame(rows, columns=["cik", "filing_date", "accession", "primary_doc"]).drop_duplicates()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["year"] = df["filing_date"].dt.year
    df.to_parquet(_LIST_CACHE, index=False)
    logger.info("10-K list: %d filings, %d CIKs", len(df), df["cik"].nunique())
    return df


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_NONWORD = re.compile(r"[^a-z\s]")


def _clean_html(raw: bytes) -> str:
    txt = raw.decode("utf-8", errors="ignore")
    txt = re.sub(r"<script.*?</script>", " ", txt, flags=re.S | re.I)
    txt = re.sub(r"<style.*?</style>", " ", txt, flags=re.S | re.I)
    txt = _TAG.sub(" ", txt)
    txt = txt.lower()
    txt = _NONWORD.sub(" ", txt)
    return _WS.sub(" ", txt).strip()


def fetch_10k_texts(filings: pd.DataFrame, sleep: float = 0.15,
                    max_docs: int | None = None) -> int:
    """Download + clean each 10-K primary document to one .txt per accession.
    Resumable (skips existing). Returns count fetched this run. Throttled."""
    os.makedirs(_TEXT_DIR, exist_ok=True)
    n = 0
    for _, r in filings.iterrows():
        acc_nodash = r["accession"].replace("-", "")
        out = os.path.join(_TEXT_DIR, "%s.txt" % r["accession"])
        if os.path.exists(out):
            continue
        url = "https://www.sec.gov/Archives/edgar/data/%d/%s/%s" % (
            int(r["cik"]), acc_nodash, r["primary_doc"])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            raw = urllib.request.urlopen(req, timeout=60).read()
            txt = _clean_html(raw)
            with open(out, "w", encoding="utf-8") as f:
                f.write(txt)
            n += 1
        except Exception as exc:
            logger.debug("10-K text fail %s: %s", r["accession"], exc)
        if n % 200 == 0 and n:
            logger.info("10-K texts fetched %d", n)
        time.sleep(sleep)
        if max_docs and n >= max_docs:
            break
    return n


def compute_similarity(filings: pd.DataFrame) -> pd.DataFrame:
    """YoY cosine similarity of consecutive 10-Ks per CIK (TF bag-of-words).
    Returns (cik, year, filing_date, similarity)."""
    from collections import Counter
    texts = {}
    for _, r in filings.iterrows():
        p = os.path.join(_TEXT_DIR, "%s.txt" % r["accession"])
        if os.path.exists(p):
            texts[r["accession"]] = p

    def vec(path):
        with open(path, encoding="utf-8") as f:
            toks = f.read().split()
        return Counter(t for t in toks if len(t) > 2)

    def cosine(a, b):
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        na = np.sqrt(sum(v * v for v in a.values()))
        nb = np.sqrt(sum(v * v for v in b.values()))
        return float(dot / (na * nb)) if na and nb else 0.0

    rows = []
    for cik, g in filings.sort_values("filing_date").groupby("cik"):
        g = g[g["accession"].isin(texts)]
        prev_v, prev_y = None, None
        for _, r in g.iterrows():
            v = vec(texts[r["accession"]])
            if prev_v is not None and (r["year"] - prev_y) in (1, 2):
                rows.append((cik, r["year"], r["filing_date"], cosine(prev_v, v)))
            prev_v, prev_y = v, r["year"]
    return pd.DataFrame(rows, columns=["cik", "year", "filing_date", "similarity"])
