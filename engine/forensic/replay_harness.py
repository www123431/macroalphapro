"""
engine/forensic/replay_harness.py — Forensic replay harness (Gap #3 SHIPPED 2026-05-15).

Validates that forensic agents (devils_advocate + factor decomposition) produce
well-formed output on three historical crisis anchor events:
  - Lehman 2008-09-15
  - COVID 2020-03-16
  - Christmas Eve 2018-12-24

Methodology (A+B hybrid, B-only after A unavailable):
  A = GDELT 2.0 historical news corpus — originally planned PRIMARY input
      UNAVAILABLE in execution environment (api.gdeltproject.org ConnectTimeout).
      Activation path preserved at engine/forensic/gdelt_historical_news.py
      for future use when GDELT becomes reachable.
  B = pre-registered anchor headlines (data/forensic_replay_anchors/*.json)
      9 hand-cited headlines (3 per event × 3 events) with Wikipedia + Reuters
      citation URLs, committed to git BEFORE replay execution.
      Pre-registration eliminates post-hoc selection bias / HARKing risk.
      Used in this v1 as the sole news_context input via AV cache injection.

Scope (honest disclosure):
  - devils_advocate replay: full dual-LLM (Gemini PRIMARY + DeepSeek DEVIL),
    real LLM cost recorded to llm_cost_ledger
  - residual_attribution: validated at _fetch_realized_factor_returns level
    only (skips decompose_strategy_day which requires PaperTradeStrategyLog
    DB row; seeding historical fake paper trades would violate doctrine).
    FF5 proxy ETFs QUAL launched 2013-07, USMV launched 2011-10 — so
    Lehman 2008 FF5 decomp will fail (honestly disclosed)
  - news_context not separately replayed; it is exercised transitively
    through devils_advocate which calls investigate_trade internally

Doctrine compliance:
  - 0-LLM-in-DECISION: this layer is pure measurement, no decision feedback
  - LLM-risk-side: validates forensic agents which are already risk-side
  - 7-agent ceiling: unchanged (no new agents)
  - Pre-registered ground truth = HARKing-immune

References:
  - LÓPEZ de Prado 2018 (PBO + DSR; "kill your own ideas before publishing")
  - Bailey & López de Prado 2014 ("Pseudo-Mathematics and Financial Charlatanism")
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_ANCHORS_DIR    = _REPO_ROOT / "data" / "forensic_replay_anchors"
_AV_CACHE_PATH  = _REPO_ROOT / "data" / "forensic" / "av_news_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class ReplayContext:
    """Replay input context for one (event, ticker) combination."""
    event_slug:                    str
    event_name:                    str
    event_date:                    datetime.date
    ticker:                        str
    realized_return_horizon_days:  int
    realized_return:               Optional[float]
    weight:                        float
    signal_value:                  Optional[float]
    n_anchors_injected:            int = 0


@dataclasses.dataclass
class ReplayResult:
    """Output of one forensic agent replay on one ReplayContext."""
    agent_name:      str
    success:         bool
    output_summary:  dict
    raw_output:      Any = None     # full result object (not serialized)
    notes:           str = ""


@dataclasses.dataclass
class ReplayReport:
    """Complete replay report for one (event, ticker) combination,
    aggregating all agent replay results."""
    context:           ReplayContext
    results:           list[ReplayResult]
    started_at_iso:    str
    completed_at_iso:  str

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (drops raw_output)."""
        return {
            "context": dataclasses.asdict(self.context, dict_factory=_date_safe_dict_factory),
            "results": [
                {
                    "agent_name":     r.agent_name,
                    "success":        r.success,
                    "output_summary": r.output_summary,
                    "notes":          r.notes,
                }
                for r in self.results
            ],
            "started_at_iso":   self.started_at_iso,
            "completed_at_iso": self.completed_at_iso,
        }


def _date_safe_dict_factory(items: list[tuple]) -> dict:
    """Custom dict_factory that serializes datetime.date to ISO string."""
    out = {}
    for k, v in items:
        if isinstance(v, datetime.date):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# AV cache injection (pre-registered anchors → news_context input path)
# ─────────────────────────────────────────────────────────────────────────────
def inject_anchors_to_av_cache(
    event_slug:    str,
    ticker:        str,
    event_date:    datetime.date,
    window_days:   int = 5,
    av_cache_path: Optional[Path] = None,
    anchors_dir:   Optional[Path] = None,
) -> int:
    """Read pre-registered anchor headlines from anchor JSON file and write
    them into the AV cache structure that news_context.investigate_trade reads.

    Returns number of headlines actually injected (after window filter).

    The injection populates the SAME cache key format used by
    news_context._av_cache_key, so news_context will pick up these headlines
    transparently when investigate_trade is called for (ticker, date).

    Headlines have neutral sentiment_label="Unknown" + sentiment_score=0.0 —
    NO pre-assigned sentiment (force the LLM to extract sentiment from title
    text, preventing label leak).
    """
    if anchors_dir is None:
        anchors_dir = _ANCHORS_DIR
    if av_cache_path is None:
        av_cache_path = _AV_CACHE_PATH

    anchor_path = anchors_dir / f"{event_slug}.json"
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor file not found: {anchor_path}")
    anchor_data = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchors = anchor_data.get("must_include_anchors", []) or []

    headlines = []
    for a in anchors:
        date_iso = a.get("date_iso", "")
        try:
            d = datetime.date.fromisoformat(date_iso)
            if abs((d - event_date).days) > window_days:
                continue
            published = f"{date_iso} 12:00 UTC"
        except Exception:
            published = "n/a"
        headlines.append({
            "title":           a.get("title", ""),
            "source":          a.get("source", "anchor-registered"),
            "published":       published,
            "sentiment_label": "Unknown",
            "sentiment_score": 0.0,
            "_provenance":     f"pre-registered-anchor:{event_slug}",
            "_citation_url":   a.get("citation_url", ""),
        })

    # Cache key MUST match news_context._av_cache_key() format
    window_start = event_date - datetime.timedelta(days=window_days)
    window_end   = event_date + datetime.timedelta(days=window_days)
    time_from = window_start.strftime("%Y%m%dT0000")
    time_to   = window_end.strftime("%Y%m%dT2359")
    cache_key = f"{ticker.upper()}|{time_from}|{time_to}"

    av_cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if av_cache_path.exists():
        try:
            cache = json.loads(av_cache_path.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("AV cache corrupt; starting fresh")
            cache = {}

    cache[cache_key] = {
        "headlines":     headlines,
        "cached_at_iso": datetime.datetime.utcnow().isoformat(),
        "n_articles":    len(headlines),
        "_provenance":   f"pre-registered-anchor:{event_slug}",
    }
    av_cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return len(headlines)


# ─────────────────────────────────────────────────────────────────────────────
# Replay context preparation
# ─────────────────────────────────────────────────────────────────────────────
def prepare_replay_context(
    event_slug:               str,
    ticker:                   str,
    realized_horizon_days:    int = 22,
    weight:                   float = 0.10,
) -> ReplayContext:
    """Fetch yfinance historical data and compute realized return over horizon.

    realized_return = (price[event_date + horizon] / price[event_date]) - 1.0
    using trading-day-snapping (entry/exit dates may shift to nearest trading day).
    """
    import yfinance as yf
    import pandas as pd

    anchor_path = _ANCHORS_DIR / f"{event_slug}.json"
    if not anchor_path.exists():
        raise FileNotFoundError(f"anchor file not found: {anchor_path}")
    anchor_data = json.loads(anchor_path.read_text(encoding="utf-8"))
    event_date = datetime.date.fromisoformat(anchor_data["event_date_center"])
    event_name = anchor_data["event_name"]

    start = event_date - datetime.timedelta(days=30)
    end   = event_date + datetime.timedelta(days=realized_horizon_days + 14)

    realized_return: Optional[float] = None
    try:
        df = yf.download(
            ticker, start=start.isoformat(), end=end.isoformat(),
            auto_adjust=True, progress=False,
        )
        if df is not None and not df.empty:
            close = df["Close"] if "Close" in df.columns else df
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).date
            on_or_after_entry = sorted(d for d in close.index if d >= event_date)
            target_exit_date = event_date + datetime.timedelta(days=realized_horizon_days)
            on_or_after_exit = sorted(d for d in close.index if d >= target_exit_date)
            if on_or_after_entry and on_or_after_exit:
                entry_price = float(close.at[on_or_after_entry[0]])
                exit_price  = float(close.at[on_or_after_exit[0]])
                if entry_price > 0:
                    realized_return = (exit_price / entry_price) - 1.0
    except Exception as exc:
        logger.warning("yfinance fetch failed for %s @ %s: %s",
                       ticker, event_date.isoformat(), exc)

    return ReplayContext(
        event_slug=event_slug,
        event_name=event_name,
        event_date=event_date,
        ticker=ticker,
        realized_return_horizon_days=realized_horizon_days,
        realized_return=realized_return,
        weight=weight,
        signal_value=None,    # signal-agnostic replay — tests forensic verdict on price+news
        n_anchors_injected=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-agent replay
# ─────────────────────────────────────────────────────────────────────────────
def replay_devils_advocate(
    context:         ReplayContext,
    strategy_name:   str = "ANCHOR_REPLAY",
    window_days:     int = 5,
) -> ReplayResult:
    """Replay devils_advocate on this context.

    Injects pre-registered anchor headlines into AV cache → calls
    investigate_with_devils_advocate which internally calls
    news_context.investigate_trade (which reads the cache we just populated)
    → Gemini PRIMARY verdict + DeepSeek DEVIL verdict + consistency score.
    """
    n_injected = inject_anchors_to_av_cache(
        event_slug=context.event_slug,
        ticker=context.ticker,
        event_date=context.event_date,
        window_days=window_days,
    )
    context.n_anchors_injected = n_injected

    if n_injected == 0:
        return ReplayResult(
            agent_name="devils_advocate",
            success=False,
            output_summary={"reason": "no anchors within window"},
            raw_output=None,
            notes=f"event_slug={context.event_slug} ticker={context.ticker} "
                  f"window_days={window_days} — pre-registered anchors did not "
                  f"fall within window",
        )

    try:
        from engine.forensic.devils_advocate import investigate_with_devils_advocate
        result = investigate_with_devils_advocate(
            date=context.event_date,
            ticker=context.ticker,
            signal_value=context.signal_value,
            weight=context.weight,
            realized_return=context.realized_return,
            strategy_name=strategy_name,
            expected_horizon_days=context.realized_return_horizon_days,
            window_days=window_days,
            skip_devil=False,
        )
    except Exception as exc:
        return ReplayResult(
            agent_name="devils_advocate",
            success=False,
            output_summary={"error_type": type(exc).__name__,
                            "error": str(exc)[:200]},
            raw_output=None,
            notes=f"investigate_with_devils_advocate raised exception",
        )

    # Extract structured summary from DualLLMForensicResult
    # (real schema: primary_summary = ForensicNewsSummary; devil_verdict = DevilsAdvocateVerdict;
    #  consistency_* + total_cost_usd at root)
    primary = getattr(result, "primary_summary", None)
    devil = getattr(result, "devil_verdict", None)
    summary: dict = {
        "consistency_score":   getattr(result, "consistency_score", None),
        "consistency_label":   getattr(result, "consistency_label", None),
        "verdict_agreement":   getattr(result, "verdict_agreement", None),
        "direction_agreement": getattr(result, "direction_agreement", None),
        "event_overlap":       getattr(result, "event_overlap", None),
        "total_cost_usd":      getattr(result, "total_cost_usd", None),
    }
    for prefix, v in [("primary", primary), ("devil", devil)]:
        if v is None:
            continue
        summary[f"{prefix}_forensic_verdict"]    = getattr(v, "forensic_verdict", None)
        summary[f"{prefix}_material_events_n"]   = len(getattr(v, "material_events", []) or [])
        # Use the actual attribute name (sentiment_assessment on both ForensicNewsSummary
        # and DevilsAdvocateVerdict)
        sent  = getattr(v, "sentiment_assessment", "") or ""
        macro = getattr(v, "macro_context", "") or ""
        sigal = getattr(v, "signal_alignment", "") or ""
        summary[f"{prefix}_sentiment_assessment"] = sent[:160]
        summary[f"{prefix}_macro_context"]        = macro[:160]
        summary[f"{prefix}_signal_alignment"]     = sigal[:160]

    return ReplayResult(
        agent_name="devils_advocate",
        success=True,
        output_summary=summary,
        raw_output=result,
        notes=f"{n_injected} pre-registered anchor headlines injected into AV "
              f"cache before invocation",
    )


def replay_factor_decomposition(context: ReplayContext) -> ReplayResult:
    """Validate FF5 factor return retrieval for historical date.

    Calls residual_attribution._fetch_realized_factor_returns directly to
    avoid decompose_strategy_day's DB dependency (seeding fake historical
    PaperTradeStrategyLog rows would violate doctrine).

    FF5 proxy ETFs:
      Mkt = SPY (1993+) · SMB = IWM-IWB (2000+) · HML = IWD-IWF (2000+)
      RMW = QUAL-SPY (2013-07+) · CMA = USMV-SPY (2011-10+)

    Lehman 2008-09 will FAIL because QUAL + USMV did not exist.
    Christmas Eve 2018-12 and COVID 2020-03 should both PASS.
    """
    from engine.forensic.residual_attribution import _fetch_realized_factor_returns

    factor_returns = _fetch_realized_factor_returns(context.event_date)
    if factor_returns is None:
        # Determine whether failure is due to proxy ETF launch date
        qual_launch = datetime.date(2013, 7, 18)
        usmv_launch = datetime.date(2011, 10, 18)
        reason_parts = []
        if context.event_date < qual_launch:
            reason_parts.append(f"QUAL launched {qual_launch.isoformat()}")
        if context.event_date < usmv_launch:
            reason_parts.append(f"USMV launched {usmv_launch.isoformat()}")
        reason = "; ".join(reason_parts) if reason_parts else \
                 "yfinance data unavailable or non-trading day"
        return ReplayResult(
            agent_name="residual_attribution_factor_returns",
            success=False,
            output_summary={"reason": "FF5 proxy ETF pre-launch", "detail": reason},
            raw_output=None,
            notes=f"event_date={context.event_date.isoformat()} predates one "
                  f"or more FF5 proxy ETFs; FF5 decomp unavailable",
        )

    return ReplayResult(
        agent_name="residual_attribution_factor_returns",
        success=True,
        output_summary={
            "Mkt": round(factor_returns["Mkt"], 4),
            "SMB": round(factor_returns["SMB"], 4),
            "HML": round(factor_returns["HML"], 4),
            "RMW": round(factor_returns["RMW"], 4),
            "CMA": round(factor_returns["CMA"], 4),
        },
        raw_output=factor_returns,
        notes="FF5 factor returns via proxy-ETF method (SPY / IWM-IWB / IWD-IWF / "
              "QUAL-SPY / USMV-SPY)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Anchor event orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_anchor_event_replay(
    event_slug:             str,
    tickers:                list[str],
    realized_horizon_days:  int = 22,
    weight:                 float = 0.10,
    include_devils_advocate: bool = True,
    include_factor_decomp:   bool = True,
) -> list[ReplayReport]:
    """Replay all forensic agents for one anchor event across N tickers.

    Returns one ReplayReport per ticker.
    """
    reports: list[ReplayReport] = []
    for ticker in tickers:
        started_at = datetime.datetime.utcnow().isoformat()
        context = prepare_replay_context(
            event_slug=event_slug,
            ticker=ticker,
            realized_horizon_days=realized_horizon_days,
            weight=weight,
        )

        results: list[ReplayResult] = []
        if include_devils_advocate:
            results.append(replay_devils_advocate(context))
        if include_factor_decomp:
            results.append(replay_factor_decomposition(context))

        completed_at = datetime.datetime.utcnow().isoformat()
        reports.append(ReplayReport(
            context=context,
            results=results,
            started_at_iso=started_at,
            completed_at_iso=completed_at,
        ))
    return reports
