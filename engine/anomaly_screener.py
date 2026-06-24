"""
engine/anomaly_screener.py — S6 Anomaly Screener (D4.2 — D4.6)

Pre-registration: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md
Architecture invariants: docs/decisions/llm_3layer_architecture_2026-05-05.md

Three detectors run in parallel (D1 Layer 1 — generation):
  • rule_baseline_a  — 5 if-then rules (price + concentration only)
  • rule_baseline_b  — baseline_a + macro_research forecast as additional rule
  • llm              — Gemini 2.5 Flash with thinking (D4.4 — separate module)

Output: AnomalyFlag rows (engine.memory.AnomalyFlag).

D4.5 forward verification engine + D4.6 daily cron + queue integration are
in companion modules. This module is the rule-based detector engine only.

Confidence is Likert 1-5 (Tetlock 2015 anchored scale; Tian-Ye-Bowman 2023):
  1 = single weak rule match     (e.g. abs return ~2σ but barely)
  2 = single solid rule match
  3 = two rules fire             OR  one strong + macro alignment
  4 = three rules fire
  5 = four or more rules fire
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from engine.memory import (
    AnomalyFlag,
    SessionFactory,
    SimulatedPosition,
)

logger = logging.getLogger(__name__)

# ── Pre-registered constants (centralized in engine/config.py 2026-05-06) ────
SPEC_HASH_PLACEHOLDER = "pending_d4_8_register"

from engine.config import (
    PRICE_SIGMA_THRESHOLD,
    VOLUME_SPIKE_MULTIPLIER,
    CONCENTRATION_THRESHOLD,
    DRAWDOWN_THRESHOLD,
    CROSS_ASSET_BOND_TICKER,
    CROSS_ASSET_EQUITY_TICKER,
    MACRO_REGIME_LOOKBACK_DAYS,
    DEFAULT_HORIZON_DAYS,
    ROLLING_VOL_WINDOW,
    VOLUME_MEDIAN_WINDOW,
    DRAWDOWN_LOOKBACK_DAYS,
)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RuleHit:
    """One fired rule, with its quantitative match."""
    rule_id: str               # "price_spike" / "volume_spike" / etc
    strength: str              # "weak" / "solid" / "strong"
    detail: dict               # numeric details (sigmas, ratios, etc)


@dataclass
class FlagCandidate:
    """A flag candidate produced by a detector before persistence."""
    detector: str              # "rule_baseline_a" / "rule_baseline_b"
    scan_date: datetime.date
    sector: str
    ticker: str
    event_class: str
    confidence_likert: int
    horizon_days: int = DEFAULT_HORIZON_DAYS
    evidence_summary: str = ""
    triggering_rules: list[RuleHit] = field(default_factory=list)


# ── Likert scoring ───────────────────────────────────────────────────────────

def compute_confidence_likert(rule_hits: list[RuleHit]) -> int:
    """
    Likert 1-5 anchored on rule count + strength (spec §3 LLM detector
    Likert table also anchors rule baseline). Tetlock 2015 anchored scale.
    """
    n = len(rule_hits)
    has_strong = any(h.strength == "strong" for h in rule_hits)
    has_solid  = any(h.strength == "solid"  for h in rule_hits)
    if n >= 4:
        return 5
    if n == 3:
        return 4
    if n == 2:
        return 3
    if n == 1:
        if has_strong:
            return 3
        if has_solid:
            return 2
        return 1
    return 1  # n == 0 should not produce a flag; defensive


def _classify_event(rule_hits: list[RuleHit]) -> str:
    """Pick the dominant event class from a list of rule hits."""
    rule_to_class = {
        "price_spike":     "price_spike",
        "volume_spike":    "volume_spike",
        "concentration":   "concentration",
        "drawdown":        "drawdown",
        "cross_asset":     "cross_asset",
        "macro_regime_shift": "news_driven",  # macro shift treated as news-driven
    }
    if not rule_hits:
        return "price_spike"
    # Pick the strongest rule's class; fallback alphabetic by rule_id
    rh_sorted = sorted(rule_hits, key=lambda h: ({"strong": 0, "solid": 1, "weak": 2}[h.strength], h.rule_id))
    return rule_to_class.get(rh_sorted[0].rule_id, "price_spike")


def _evidence_summary(rule_hits: list[RuleHit]) -> str:
    """Compose a short factual summary (≤ 200 chars) of all rule hits."""
    parts: list[str] = []
    for h in rule_hits:
        d = h.detail
        if h.rule_id == "price_spike":
            parts.append(f"|ret| {d.get('abs_ret', 0):.2%} = {d.get('sigma', 0):.1f}σ")
        elif h.rule_id == "volume_spike":
            parts.append(f"vol {d.get('mult', 0):.1f}× median")
        elif h.rule_id == "concentration":
            parts.append(f"sector {d.get('sector_weight', 0):.0%}")
        elif h.rule_id == "drawdown":
            parts.append(f"30d DD {d.get('max_dd', 0):.1%}")
        elif h.rule_id == "cross_asset":
            parts.append(f"SPY/TLT sign flip {d.get('spy_ret', 0):+.2%}/{d.get('bond_ret', 0):+.2%}")
        elif h.rule_id == "macro_regime_shift":
            parts.append(f"macro {d.get('from_regime', '?')}→{d.get('to_regime', '?')}")
    s = "; ".join(parts)
    return s[:200]


# ── Data fetching helpers ────────────────────────────────────────────────────

def _fetch_price_history(ticker: str, end_date: datetime.date, days: int = 90) -> pd.DataFrame:
    """
    Fetch OHLCV history for `ticker` ending at `end_date` covering `days` calendar days.
    Returns DataFrame indexed by date with columns Close, Volume.
    Returns empty DataFrame on failure.
    """
    try:
        import yfinance as yf
        start = end_date - datetime.timedelta(days=days + 14)  # buffer for non-trading days
        end_excl = end_date + datetime.timedelta(days=1)
        df = yf.download(
            ticker, start=str(start), end=str(end_excl),
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).date
        return df[["Close", "Volume"]] if "Volume" in df.columns else df[["Close"]]
    except Exception as exc:
        logger.debug("anomaly_screener: price fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _get_current_holdings(scan_date: datetime.date) -> dict[str, dict]:
    """
    Return {ticker: {sector, weight}} for non-zero current positions on scan_date.
    Uses the latest SimulatedPosition snapshot ≤ scan_date.
    """
    out: dict[str, dict] = {}
    with SessionFactory() as session:
        latest_date = (
            session.query(SimulatedPosition.snapshot_date)
            .filter(SimulatedPosition.snapshot_date <= scan_date)
            .order_by(SimulatedPosition.snapshot_date.desc())
            .limit(1)
            .scalar()
        )
        if latest_date is None:
            return out
        rows = (
            session.query(SimulatedPosition)
            .filter(SimulatedPosition.snapshot_date == latest_date)
            .all()
        )
        for r in rows:
            w = r.actual_weight if r.actual_weight is not None else (r.target_weight or 0.0)
            if abs(w) < 1e-6:
                continue
            ticker = r.ticker or ""
            if not ticker:
                continue
            out[ticker] = {"sector": r.sector or "—", "weight": float(w)}
    return out


# ── Rule implementations ─────────────────────────────────────────────────────

def _check_price_spike(prices: pd.DataFrame, scan_date: datetime.date) -> RuleHit | None:
    """Rule 1: |daily return| > 2σ_60d on the most recent trading day ≤ scan_date."""
    if prices.empty or len(prices) < ROLLING_VOL_WINDOW + 2:
        return None
    closes = prices["Close"].dropna()
    if len(closes) < ROLLING_VOL_WINDOW + 2:
        return None
    rets = closes.pct_change().dropna()
    last_date = max(d for d in rets.index if d <= scan_date) if any(d <= scan_date for d in rets.index) else None
    if last_date is None:
        return None
    last_ret = float(rets.loc[last_date])
    sigma_60 = float(rets.loc[:last_date].iloc[-ROLLING_VOL_WINDOW:].std())
    if sigma_60 <= 1e-9:
        return None
    z = abs(last_ret) / sigma_60
    if z < PRICE_SIGMA_THRESHOLD:
        return None
    strength = "strong" if z >= 3.0 else ("solid" if z >= 2.5 else "weak")
    return RuleHit("price_spike", strength,
                   {"abs_ret": abs(last_ret), "sigma": z, "ret": last_ret})


def _check_volume_spike(prices: pd.DataFrame, scan_date: datetime.date) -> RuleHit | None:
    """Rule 2: today's volume > 3× 30-day median."""
    if "Volume" not in prices.columns or prices.empty:
        return None
    vol = prices["Volume"].dropna()
    if len(vol) < VOLUME_MEDIAN_WINDOW + 1:
        return None
    last_date = max(d for d in vol.index if d <= scan_date) if any(d <= scan_date for d in vol.index) else None
    if last_date is None:
        return None
    today_vol = float(vol.loc[last_date])
    median_30 = float(vol.loc[:last_date].iloc[-VOLUME_MEDIAN_WINDOW - 1:-1].median())
    if median_30 <= 0:
        return None
    mult = today_vol / median_30
    if mult < VOLUME_SPIKE_MULTIPLIER:
        return None
    strength = "strong" if mult >= 5.0 else ("solid" if mult >= 4.0 else "weak")
    return RuleHit("volume_spike", strength, {"mult": mult, "today_vol": today_vol})


def _check_concentration(holdings: dict[str, dict], ticker: str) -> RuleHit | None:
    """Rule 3: ticker's sector weight > 30%."""
    sector = holdings.get(ticker, {}).get("sector", "—")
    if sector == "—":
        return None
    sector_w = sum(abs(v["weight"]) for k, v in holdings.items() if v.get("sector") == sector)
    if sector_w < CONCENTRATION_THRESHOLD:
        return None
    strength = "strong" if sector_w >= 0.40 else ("solid" if sector_w >= 0.35 else "weak")
    return RuleHit("concentration", strength,
                   {"sector": sector, "sector_weight": sector_w})


def _check_drawdown(prices: pd.DataFrame, scan_date: datetime.date) -> RuleHit | None:
    """Rule 4: 30-day max-drawdown on the holding > 10%."""
    if prices.empty:
        return None
    closes = prices["Close"].dropna()
    closes = closes.loc[closes.index <= scan_date]
    if len(closes) < DRAWDOWN_LOOKBACK_DAYS:
        return None
    window = closes.iloc[-DRAWDOWN_LOOKBACK_DAYS:]
    peak = window.cummax()
    dd = (window - peak) / peak
    max_dd = float(abs(dd.min()))
    if max_dd < DRAWDOWN_THRESHOLD:
        return None
    strength = "strong" if max_dd >= 0.20 else ("solid" if max_dd >= 0.15 else "weak")
    return RuleHit("drawdown", strength, {"max_dd": max_dd})


def _check_cross_asset(scan_date: datetime.date) -> RuleHit | None:
    """Rule 5: SPY return × TLT return change of sign in same day (stress signal)."""
    spy = _fetch_price_history(CROSS_ASSET_EQUITY_TICKER, scan_date, days=10)
    tlt = _fetch_price_history(CROSS_ASSET_BOND_TICKER, scan_date, days=10)
    if spy.empty or tlt.empty:
        return None
    spy_ret = spy["Close"].pct_change().dropna()
    tlt_ret = tlt["Close"].pct_change().dropna()
    common_dates = sorted(set(spy_ret.index) & set(tlt_ret.index) & {d for d in spy_ret.index if d <= scan_date})
    if len(common_dates) < 2:
        return None
    last = common_dates[-1]
    s = float(spy_ret.loc[last])
    t = float(tlt_ret.loc[last])
    # Positive correlation flip: same day SPY down + bond down (or both up) instead of usual hedge
    # We flag when SPY return < -1.5% and TLT return < 0 simultaneously (stress regime)
    if s < -0.015 and t < 0:
        return RuleHit("cross_asset", "solid",
                       {"spy_ret": s, "bond_ret": t, "stress_type": "equity_bond_codecline"})
    if s > 0.015 and t < -0.01:
        return RuleHit("cross_asset", "weak",
                       {"spy_ret": s, "bond_ret": t, "stress_type": "decoupled_bond_sell"})
    return None


def _check_macro_regime_shift(scan_date: datetime.date) -> RuleHit | None:
    """
    Rule 6 (baseline B only): macro_research_agent regime_assessment shifted
    category in the last 7 days.

    NOTE: Baseline B is allowed to read macro_research output by spec design.
    The LLM detector (D4.4) is forbidden from reading it (B-6 isolation).
    """
    try:
        from engine.macro_verification import get_recent_macro_briefs
        cutoff = scan_date - datetime.timedelta(days=MACRO_REGIME_LOOKBACK_DAYS)
        briefs = get_recent_macro_briefs(lookback_days=MACRO_REGIME_LOOKBACK_DAYS) or []
    except Exception as exc:
        logger.debug("anomaly_screener: macro brief fetch failed: %s", exc)
        return None
    if len(briefs) < 2:
        return None
    # briefs are sorted desc by created_at typically; we just look for two distinct regime values
    regimes = [b.get("regime_assessment") for b in briefs if b.get("regime_assessment")]
    distinct = list(dict.fromkeys(regimes))   # preserve order of first appearance
    if len(distinct) < 2:
        return None
    # detected a shift
    return RuleHit("macro_regime_shift", "solid",
                   {"from_regime": distinct[1], "to_regime": distinct[0]})


# ── Detector orchestration ───────────────────────────────────────────────────

def detect_rule_baseline(
    scan_date: datetime.date,
    *,
    include_macro: bool = False,
) -> list[FlagCandidate]:
    """
    Run rule-based detector on scan_date over current portfolio holdings.

    include_macro=False  → baseline_a (rule-only, 5 rules)
    include_macro=True   → baseline_b (baseline_a + Rule 6 macro_regime_shift)
    """
    holdings = _get_current_holdings(scan_date)
    if not holdings:
        logger.info("anomaly_screener: no holdings on %s; baseline skipped", scan_date)
        return []

    candidates: list[FlagCandidate] = []
    detector_label = "rule_baseline_b" if include_macro else "rule_baseline_a"

    # Rule 5 (cross-asset) is portfolio-wide; compute once
    cross_asset_hit = _check_cross_asset(scan_date)
    macro_hit = _check_macro_regime_shift(scan_date) if include_macro else None

    for ticker, info in holdings.items():
        prices = _fetch_price_history(ticker, scan_date, days=90)

        rule_hits: list[RuleHit] = []
        h = _check_price_spike(prices, scan_date);     h and rule_hits.append(h)
        h = _check_volume_spike(prices, scan_date);    h and rule_hits.append(h)
        h = _check_concentration(holdings, ticker);    h and rule_hits.append(h)
        h = _check_drawdown(prices, scan_date);        h and rule_hits.append(h)
        if cross_asset_hit:
            rule_hits.append(cross_asset_hit)
        if macro_hit:
            rule_hits.append(macro_hit)

        if not rule_hits:
            continue

        candidates.append(FlagCandidate(
            detector=detector_label,
            scan_date=scan_date,
            sector=info["sector"],
            ticker=ticker,
            event_class=_classify_event(rule_hits),
            confidence_likert=compute_confidence_likert(rule_hits),
            horizon_days=DEFAULT_HORIZON_DAYS,
            evidence_summary=_evidence_summary(rule_hits),
            triggering_rules=rule_hits,
        ))

    logger.info("anomaly_screener: %s produced %d candidates on %s",
                detector_label, len(candidates), scan_date)
    return candidates


def persist_flag_candidates(
    candidates: Iterable[FlagCandidate],
    *,
    spec_hash: str | None = None,
) -> list[int]:
    """
    Persist a batch of FlagCandidate to engine.memory.AnomalyFlag.
    De-duplicates within a (detector, scan_date, ticker) triple.

    Returns the inserted row ids.
    """
    inserted: list[int] = []
    h = spec_hash or SPEC_HASH_PLACEHOLDER
    with SessionFactory() as session:
        for c in candidates:
            existing = (
                session.query(AnomalyFlag)
                .filter(
                    AnomalyFlag.detector == c.detector,
                    AnomalyFlag.scan_date == c.scan_date,
                    AnomalyFlag.ticker == c.ticker,
                ).first()
            )
            if existing:
                continue
            row = AnomalyFlag(
                detector            = c.detector,
                scan_date           = c.scan_date,
                sector              = c.sector,
                ticker              = c.ticker,
                event_class         = c.event_class,
                confidence_likert   = c.confidence_likert,
                horizon_days        = c.horizon_days,
                evidence_summary    = c.evidence_summary,
                triggering_rules    = json.dumps([
                    {"rule_id": r.rule_id, "strength": r.strength, "detail": r.detail}
                    for r in c.triggering_rules
                ]) if c.triggering_rules else None,
                spec_hash           = h,
            )
            session.add(row)
            session.flush()
            inserted.append(row.id)
        session.commit()
    return inserted


def run_baseline_scan_for_date(scan_date: datetime.date) -> dict:
    """
    Convenience wrapper that runs baseline A + B for a given date and persists.
    Returns counts. Cron entry point (D4.6 will call this daily).
    """
    cands_a = detect_rule_baseline(scan_date, include_macro=False)
    cands_b = detect_rule_baseline(scan_date, include_macro=True)
    ids_a = persist_flag_candidates(cands_a)
    ids_b = persist_flag_candidates(cands_b)
    return {
        "scan_date":        str(scan_date),
        "baseline_a_count": len(ids_a),
        "baseline_b_count": len(ids_b),
        "baseline_a_ids":   ids_a,
        "baseline_b_ids":   ids_b,
    }
