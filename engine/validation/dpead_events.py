"""engine/validation/dpead_events.py — event-level PEAD conditioning.

A.2 of the deepen-D_PEAD work. The tilt finding (A.1) improved the
PORTFOLIO construction; this asks whether the SIGNAL itself can be
conditioned — is the post-earnings drift concentrated in particular
SUE magnitudes / market-cap tiers? If the drift lives mostly in
extreme-SUE or smaller-cap events, trading only those raises signal-to-
noise.

Reconstruction: the signal panel (data/path_c_dhs/_pead_ts_signal_panel)
has per-event SUE + market_cap_at_q[millions] + rdq (report date) +
permno, but NO forward return. We compute the forward CAR from the
weekly CRSP top-1500 PRICE panel:
  - enter the first week STRICTLY AFTER rdq (no look-ahead)
  - forward raw return over K weeks (drift window ~ D_PEAD 60d horizon)
  - abnormal = raw − equal-weight universe mean over the same window
    (controls for market moves during the drift)

Survivorship note: events whose permno leaves the panel within the
window (delist) are dropped. This is the SAME filter across all SUE /
cap buckets, so the RELATIVE comparison (which bucket drifts more) is
fair even though absolute CARs carry mild survivorship.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_PANEL = "data/factor_ensemble_singlename/_crsp_dsf_top1500_panel.parquet"
_SIGNAL_PANEL = "data/path_c_dhs/_pead_ts_signal_panel.parquet"


def _load_price_panel(path: str = _PRICE_PANEL) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    # columns are permno (ints, possibly as object); normalize to int
    df.columns = [int(c) for c in df.columns]
    return df.sort_index()


def compute_event_cars(
    drift_weeks: int = 8,
    signal_path: str = _SIGNAL_PANEL,
    price_path:  str = _PRICE_PANEL,
) -> pd.DataFrame:
    """Return a per-event frame: permno, rdq, sue, market_cap_m, fwd_raw,
    mkt_fwd, car (abnormal forward return over drift_weeks)."""
    sig = pd.read_parquet(signal_path)
    sig = sig.dropna(subset=["permno", "rdq", "sue"]).copy()
    sig["permno"] = sig["permno"].astype(int)
    sig["rdq"] = pd.to_datetime(sig["rdq"])

    px = _load_price_panel(price_path)
    weeks = px.index
    # Universe equal-weight weekly return (market proxy) from the price panel
    uni_ret = px.pct_change()
    uni_mean = uni_ret.mean(axis=1)   # equal-weight mean across permnos

    rows = []
    panel_permnos = set(px.columns)
    for _, ev in sig.iterrows():
        pn, rdq, sue = ev["permno"], ev["rdq"], float(ev["sue"])
        if pn not in panel_permnos:
            continue
        # entry week = first week strictly after rdq
        after = weeks[weeks > rdq]
        if len(after) < drift_weeks + 1:
            continue
        w0 = after[0]
        wK = after[drift_weeks]
        p0 = px.at[w0, pn] if w0 in px.index else np.nan
        pK = px.at[wK, pn] if wK in px.index else np.nan
        if not (np.isfinite(p0) and np.isfinite(pK)) or p0 <= 0:
            continue
        fwd_raw = float(pK / p0 - 1.0)
        # market forward over the same window (compound equal-weight mean)
        seg = uni_mean[(uni_mean.index > w0) & (uni_mean.index <= wK)]
        mkt_fwd = float((1.0 + seg.dropna()).prod() - 1.0) if len(seg) else 0.0
        rows.append({
            "permno": pn, "rdq": rdq, "sue": sue,
            "market_cap_m": float(ev.get("market_cap_at_q", np.nan)),
            "fwd_raw": fwd_raw, "mkt_fwd": mkt_fwd,
            "car": fwd_raw - mkt_fwd,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ReconstructionCheck:
    """Self-validation of the event-CAR reconstruction against a KNOWN
    fact: the production D_PEAD goes long high-SUE / short low-SUE and is
    profitable. If our reconstructed long-minus-short CAR is not clearly
    positive, the reconstruction is broken (data quality) and its
    conditioning output MUST NOT be trusted."""
    coverage_frac:        float
    long_minus_short_car: float
    sign_matches_production: bool
    reliable:             bool
    note:                 str


def validate_reconstruction(
    events:        pd.DataFrame,
    n_total_events: int,
    q:             float = 0.2,
) -> ReconstructionCheck:
    """Guard: only trust event conditioning if (a) coverage is adequate
    AND (b) the reconstructed long-high-SUE-minus-short-low-SUE CAR is
    positive (matching the known-profitable production strategy)."""
    ls = long_short_extreme(events, q=q)
    lms = ls["long_minus_short_car"]
    coverage = len(events) / n_total_events if n_total_events else 0.0
    sign_ok = bool(np.isfinite(lms) and lms > 0)
    reliable = bool(sign_ok and coverage >= 0.5)
    if reliable:
        note = "reconstruction passes sanity check — conditioning trustworthy"
    elif not sign_ok:
        note = (f"BROKEN — long-minus-short CAR {lms*100:.2f}% is not positive; "
                f"contradicts the known-profitable production strategy. Likely "
                f"causes: price-panel split artifacts + {(1-coverage)*100:.0f}% "
                f"event loss + survivorship. DO NOT use for conditioning.")
    else:
        note = (f"LOW COVERAGE — only {coverage*100:.0f}% of events reconstructed; "
                f"conditioning estimates unreliable.")
    return ReconstructionCheck(
        coverage_frac=coverage, long_minus_short_car=lms,
        sign_matches_production=sign_ok, reliable=reliable, note=note,
    )


@dataclass(frozen=True)
class BucketStat:
    label:   str
    n:       int
    mean_car: float
    t_stat:  float


def _bucket_stats(df: pd.DataFrame, group_col: str, labels: list) -> list[BucketStat]:
    out = []
    for lab in labels:
        seg = df[df[group_col] == lab]["car"].dropna()
        if len(seg) < 5:
            out.append(BucketStat(str(lab), len(seg), float("nan"), float("nan")))
            continue
        m = seg.mean()
        t = m / (seg.std(ddof=1) / np.sqrt(len(seg))) if seg.std(ddof=1) > 0 else float("nan")
        out.append(BucketStat(str(lab), len(seg), float(m), float(t)))
    return out


def condition_by_sue(events: pd.DataFrame, n_buckets: int = 5) -> list[BucketStat]:
    """Mean CAR by SUE quantile bucket. Monotone-increasing CAR in SUE ⇒
    the drift tracks the surprise; concentration in the extreme buckets
    ⇒ trade only high-|SUE| events."""
    df = events.copy()
    df["sue_bucket"] = pd.qcut(df["sue"], n_buckets,
                               labels=[f"Q{i+1}" for i in range(n_buckets)],
                               duplicates="drop")
    labels = list(df["sue_bucket"].cat.categories)
    return _bucket_stats(df, "sue_bucket", labels)


def condition_by_cap(events: pd.DataFrame, n_buckets: int = 3) -> list[BucketStat]:
    """Mean CAR by market-cap tertile. Stronger CAR in smaller caps ⇒
    classic PEAD liquidity effect (but fights capacity)."""
    df = events.dropna(subset=["market_cap_m"]).copy()
    df = df[df["market_cap_m"] > 0]
    df["cap_bucket"] = pd.qcut(df["market_cap_m"], n_buckets,
                               labels=["small", "mid", "large"][:n_buckets],
                               duplicates="drop")
    labels = list(df["cap_bucket"].cat.categories)
    return _bucket_stats(df, "cap_bucket", labels)


def long_short_extreme(events: pd.DataFrame, q: float = 0.2) -> dict:
    """Compare the strategy's effective bet: long top-q SUE, short
    bottom-q SUE. Reports each leg's mean CAR + t, confirming (or not)
    the A.1 finding that the long (high-SUE) leg is the cleaner one at
    the EVENT level."""
    df = events.dropna(subset=["sue", "car"])
    hi = df[df["sue"] >= df["sue"].quantile(1 - q)]["car"]
    lo = df[df["sue"] <= df["sue"].quantile(q)]["car"]

    def _stat(s):
        if len(s) < 5 or s.std(ddof=1) == 0:
            return (len(s), float("nan"), float("nan"))
        return (len(s), float(s.mean()),
                float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))))

    n_hi, m_hi, t_hi = _stat(hi)
    n_lo, m_lo, t_lo = _stat(lo)
    return {
        "long_high_sue":  {"n": n_hi, "mean_car": m_hi, "t": t_hi},
        "short_low_sue":  {"n": n_lo, "mean_car": m_lo, "t": t_lo},
        "long_minus_short_car": (m_hi - m_lo) if np.isfinite(m_hi) and np.isfinite(m_lo) else float("nan"),
    }
