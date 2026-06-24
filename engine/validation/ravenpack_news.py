"""engine/validation/ravenpack_news.py — alt-data: news sentiment (RavenPack).

A GENUINELY orthogonal mechanism (news attention/sentiment, not earnings
underreaction). RavenPack DJ+PR equities (rpna.rpa_djpr_equities_YYYY) gives a
per-story Event Sentiment Score (ESS) computed AT THE TIME (deterministic, no LLM
→ no memory-contamination, point-in-time). Aggregated server-side to entity-day
(mean ESS, story count) restricted to high-relevance (>=90) US news + our liquid
universe entities, so the pull is small (raw is ~10.5M rows/yr).

Link: rp_entity_id -> cusip (wrds_rpa_company_mappings) -> permno (cached CRSP
stocknames ncusip). Test: daily news-sentiment signal -> short-horizon L/S.
Prior: news sentiment is a FAST signal (Tetlock 2007: predicts days, reverses
weeks) → likely turnover-walled, but orthogonal + untried → an honest fresh shot.

VERDICT (2026-05-21, US, 1.49M entity-days, 1598 permnos): no deployable edge.
  - STANDALONE monthly L/S (long high-ESS / short low): RED, gross -2%/yr, t=-0.90,
    net deflSR 0 — high-sentiment names slightly UNDERperform next month (Tetlock's
    monthly-horizon overreaction reversal). corr 0.44 w/ D_PEAD.
  - PEAD × NEWS fusion (does announcement-window news CONFIRMING the SUE sharpen
    the 60d drift?): NULL. baseline decile spread 2.38% t=5.68 → news-confirmed
    2.72% t=3.47 (spread barely wider, t DROPS = sample-narrowing not signal, same
    as the A.3 price-reaction NULL). corr(SUE, news-ESS)=0.26 → news around earnings
    is largely REDUNDANT with the SUE number, not orthogonal predictive power.
  => RavenPack news adds neither a standalone alpha nor a clean PEAD enhancement.
  (Caveat: coarse daily/monthly ESS aggregation; a faster intraday news signal
  might differ but would be turnover-walled.)

VERDICT 2 (2026-05-26, ATTENTION + CHANGE dimensions, not just ESS level):
re-mined the SAME cached aggregate along the two dimensions the first pass did NOT
test — news ATTENTION (n_stories abnormal-coverage vs own trailing-12m baseline)
and sentiment CHANGE (d_ESS vs own trailing-6m baseline) — monthly decile L/S, EW,
t->t+1, in ALL / small / large size buckets, both directions (24-cell pre-specified
grid; see engine/validation/_news_attention_explore.py). RESULT = RED.
  - ATTENTION shock: dead flat everywhere (abn_att primary small-cap reversal t=0.42;
    best raw-coverage cell t=1.52 = the old neglected-firm/illiquidity premium, sub-sig).
    Gate: residual-α t=-0.30, deflated-SR 0.13 → RED (logged graveyard).
  - sentiment CHANGE reversal = the ONLY flicker (high sentiment-surprise names
    underperform next month; consistent across size buckets t≈1.94-2.12). But best-of-24
    → fails ALL corrected bars: residual-α vs FF5+UMD+PEAD t≤2.14 (<3.0), deflated-SR
    0.48-0.55 (<<0.90), and OOS Sharpe COLLAPSES 0.67->0.11 (effect lived 2014-2019,
    decayed post-2019 = classic sentiment-anomaly arbitrage decay). small=RED, ALL=YELLOW;
    neither deployable (logged graveyard: news_ess_change_reversal_small).
  => News NLP on the cached RPNA aggregate is now a WELL-POWERED NULL across all three
  dimensions (level / attention / change). The change-reversal flicker is most likely
  a multiple-testing artifact + first-half luck. A deeper WRDS follow-up (per-story
  dispersion / topic / intraday / pre-2014) is NOT warranted — the cheap local first
  pass produced no above-bar flicker to justify the cost.
"""
from __future__ import annotations

import logging
import os
import socket
import time

import pandas as pd

logger = logging.getLogger(__name__)

_STOCKNAMES = "data/cache/_stocknames_ncusip.parquet"   # cusip(ncusip)+permno
_MAP_CACHE = "data/cache/_rpna_entity_map.parquet"
_SENT_CACHE = "data/cache/_rpna_daily_sentiment.parquet"


def _pg_engine():
    from sqlalchemy import create_engine
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    host, port, dbn, user, pw = open(pg).read().strip().splitlines()[0].split(":")
    return create_engine("postgresql+psycopg2://%s:%s@%s:%s/%s" % (user, pw, host, port, dbn),
                         connect_args={"sslmode": "require"})


def fetch_sentiment(y0: int = 2014, y1: int = 2024, force: bool = False) -> pd.DataFrame:
    """ONE WRDS connection: map universe permnos->entities, then pull per-year
    entity-day aggregated ESS for those entities. Cached."""
    if os.path.exists(_SENT_CACHE) and not force:
        return pd.read_parquet(_SENT_CACHE)
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    from sqlalchemy import text
    sn = pd.read_parquet(_STOCKNAMES).rename(columns={"ncusip": "cusip"})
    sn["cusip8"] = sn["cusip"].astype(str).str.slice(0, 8)
    eng = _pg_engine()
    try:
        m = pd.read_sql(text("select rp_entity_id, cusip from rpna.wrds_rpa_company_mappings "
                             "where cusip is not null"), eng)
        m["cusip8"] = m["cusip"].astype(str).str.slice(0, 8)
        link = m.merge(sn[["cusip8", "permno"]].drop_duplicates(), on="cusip8", how="inner")
        link = link.drop_duplicates(["rp_entity_id", "permno"])
        link.to_parquet(_MAP_CACHE, index=False)
        ents = sorted(link["rp_entity_id"].dropna().unique())
        ent_in = ",".join("'%s'" % e for e in ents)
        parts = []
        for y in range(y0, y1 + 1):
            ypath = f"data/cache/_rpna_sent_{y}.parquet"   # resumable per year
            if os.path.exists(ypath):
                parts.append(pd.read_parquet(ypath)); continue
            q = ("select rp_entity_id, rpa_date_utc as d, "
                 "avg(event_sentiment_score) as ess, count(*) as n_stories "
                 "from rpna.rpa_djpr_equities_%d "
                 "where relevance>=90 and event_sentiment_score is not null "
                 "and rp_entity_id in (%s) group by rp_entity_id, rpa_date_utc" % (y, ent_in))
            part = pd.read_sql(text(q), eng)
            part.to_parquet(ypath, index=False)
            parts.append(part)
            logger.info("rpna %d: %d entity-days", y, len(part))
        sent = pd.concat(parts, ignore_index=True)
    finally:
        eng.dispose()
    sent = sent.merge(link[["rp_entity_id", "permno"]].drop_duplicates("rp_entity_id"),
                      on="rp_entity_id", how="inner")
    sent["d"] = pd.to_datetime(sent["d"])
    sent.to_parquet(_SENT_CACHE, index=False)
    logger.info("rpna sentiment: %d entity-days, %d permnos", len(sent), sent["permno"].nunique())
    return sent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = fetch_sentiment()
    logger.info("DONE %s; ESS range %.1f..%.1f", s.shape, s["ess"].min(), s["ess"].max())
