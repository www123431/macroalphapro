"""
engine/anomaly_llm_detector.py — D4.4 of S6 anomaly_screener (2026-05-05)

Pre-registration: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md
  • Model:         gemini-2.5-flash       (locked 90d)
  • Temperature:   0                      (deterministic)
  • Thinking cap:  5000 tokens / call     (cost control)
  • Output schema: enforced JSON          (anti-fragile parse)
  • Repro:         prompt + response SHA-256 cached on disk

Inputs allowed (Layer 1 generation, D1 Invariant 1):
  • Today's portfolio (ticker, sector, weight)
  • Price summary (1d/5d/30d returns)
  • Filtered news flow (≤ 50 items, post-dedupe)
  • Concentration metrics

Inputs EXCLUDED (B-6 isolation, anti-leakage):
  • macro_research_agent output       ← strict
  • Other LLM agent reflections       ← strict
  • Future-dated data (post-cutoff)   ← strict (news cutoff filtered upstream)

Output (LLM only flags + cites evidence; narrative is deterministic templating):
  {
    "flags": [
      {
        "ticker": "XLE",
        "sector": "Energy",
        "event_class": "price_spike|news_driven|concentration|cross_asset|volume_spike|drawdown",
        "evidence_summary": "≤ 200 chars factual citation",
        "confidence_likert": 3,
        "horizon_days": 5
      }
    ]
  }
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# ── LLM constants — were centralized in engine/config.py (2026-05-06), then removed in a
# later refactor (the module-level import had been raising = this module unimportable).
# Restored locally 2026-05-22 (config-drift fix). PROVENANCE:
#   model / temperature / thinking / cost-in / cost-out  -> copied from live sibling
#     modules engine.etf_holdings_risk_monitor + engine.fomc_surprise_override (exact).
#   max_output_tokens=500  -> from engine.llm_extractor (exact).
#   horizon_days=5  -> this module's own AnomalyFlag schema default + dd_investigation (5).
#   max_news_items=8  -> the sector-news fetch cap (blueprint_p2 fetch_sector_news max_total=8).
# The last two are evidence-based reconstructions (originals not in git source) — FLAG for
# review if the intended values differ.
LLM_MODEL_VERSION: str = "gemini-2.5-flash"
LLM_TEMPERATURE: float = 0.0
LLM_THINKING_BUDGET: int = 1500
LLM_HORIZON_DAYS: int = 5            # FLAG: evidence-based (schema default), confirm if needed
LLM_MAX_NEWS_ITEMS: int = 8          # FLAG: evidence-based (news-fetch cap), confirm if needed
LLM_MAX_OUTPUT_TOKENS: int = 500
COST_PER_1M_INPUT_TOKENS: float = 0.30
COST_PER_1M_OUTPUT_TOKENS: float = 2.50
# 2026-05-08: budget moved from constant import to runtime SystemConfig-backed
# helper so supervisor can adjust without code edits. Default $250/yr unchanged.
from engine.llm_budget import get_s6_anomaly_budget_usd_per_year

# Reproducibility cache directory
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".streamlit" / "llm_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_COST_TRACKER_FILE = _CACHE_DIR / "anomaly_llm_cost_tracker.json"

# ── JSON output schema ───────────────────────────────────────────────────────

LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ticker":            {"type": "string"},
                    "sector":            {"type": "string"},
                    "event_class": {
                        "type": "string",
                        "enum": [
                            "price_spike", "news_driven", "concentration",
                            "cross_asset", "volume_spike", "drawdown",
                        ],
                    },
                    "evidence_summary":  {"type": "string"},
                    "confidence_likert": {
                        "type": "integer",
                        "minimum": 1, "maximum": 5,
                    },
                    "horizon_days":      {
                        "type": "integer",
                        "minimum": 1, "maximum": 30,
                    },
                    "news_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "ticker", "sector", "event_class",
                    "evidence_summary", "confidence_likert", "horizon_days",
                ],
            },
        },
    },
    "required": ["flags"],
}


# ── Prompt construction ──────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = """You are an anomaly screener for a quantitative ETF portfolio. Your job is
to flag tickers that show unusual conditions in the next 5 trading days.

YOU ARE A DETECTOR, NOT A NARRATOR.
- Output a structured list of flags + concise factual evidence.
- DO NOT write free-form analysis or causal explanations.
- DO NOT speculate beyond what the provided data shows.
- Confidence on a 1-5 Likert scale (NOT 0-1 free-form):
    1 = weak / single weak signal
    2 = single solid signal
    3 = two converging signals
    4 = three converging signals
    5 = four or more converging signals OR one definitive signal
- A "flag" requires AT LEAST ONE concrete data point (price move, news quote,
  concentration ratio). Speculation alone is not a flag.

EVENT CLASSES:
- price_spike    : daily return > 2σ rolling
- news_driven    : material news (earnings, M&A, regulation, accident)
- concentration  : portfolio sector / single-name overweight
- cross_asset    : equity-bond / dollar-yield decoupling
- volume_spike   : trading volume > 3× recent median
- drawdown       : multi-day decline > 8-10%

OUTPUT JSON ONLY. No prose."""


def _format_holdings(holdings: dict[str, dict]) -> str:
    rows = ["TICKER  SECTOR              WEIGHT"]
    for t, info in sorted(holdings.items(), key=lambda x: -abs(x[1].get("weight", 0))):
        w = info.get("weight", 0.0)
        sec = info.get("sector", "—")
        rows.append(f"{t:6s}  {sec:20s} {w:+.1%}")
    return "\n".join(rows) if len(rows) > 1 else "(no positions)"


def _format_price_summary(price_summary: dict[str, dict]) -> str:
    """price_summary[ticker] = {ret_1d, ret_5d, ret_30d, sigma_60d}"""
    rows = ["TICKER  RET_1D    RET_5D    RET_30D   σ_60d"]
    for t, p in sorted(price_summary.items()):
        rows.append(
            f"{t:6s}  {p.get('ret_1d', 0):+.2%}   {p.get('ret_5d', 0):+.2%}   "
            f"{p.get('ret_30d', 0):+.2%}   {p.get('sigma_60d', 0):.2%}"
        )
    return "\n".join(rows) if len(rows) > 1 else "(no price data)"


def _format_concentration(holdings: dict[str, dict]) -> str:
    by_sector: dict[str, float] = {}
    for info in holdings.values():
        s = info.get("sector", "—")
        by_sector[s] = by_sector.get(s, 0.0) + abs(info.get("weight", 0.0))
    rows = ["SECTOR              ABS_WEIGHT"]
    for s, w in sorted(by_sector.items(), key=lambda kv: -kv[1]):
        rows.append(f"{s:20s} {w:.1%}")
    return "\n".join(rows) if len(rows) > 1 else "(no concentration data)"


def _format_news(news_items: Iterable[dict]) -> str:
    rows = []
    for i, n in enumerate(list(news_items)[:LLM_MAX_NEWS_ITEMS], 1):
        sentiment = n.get("sentiment")
        sent_str = f" sent={sentiment:+.2f}" if isinstance(sentiment, (int, float)) else ""
        tickers = n.get("tickers", [])
        ticker_tag = f" [{','.join(tickers)}]" if tickers else ""
        rows.append(
            f"{i:2d}. ({n.get('source', '?'):14s}) {n.get('publish_date')}"
            f"{ticker_tag}{sent_str}: {(n.get('title') or '')[:120]}"
        )
    return "\n".join(rows) if rows else "(no relevant news in window)"


def build_prompt(
    scan_date: datetime.date,
    holdings: dict[str, dict],
    price_summary: dict[str, dict],
    news_items: Iterable[dict],
) -> str:
    """Compose the deterministic prompt sent to Gemini. Same inputs → same prompt."""
    news_blob = _format_news(news_items)
    return f"""SCAN DATE: {scan_date}

PORTFOLIO HOLDINGS
{_format_holdings(holdings)}

CONCENTRATION (by sector, abs weight)
{_format_concentration(holdings)}

PRICE SUMMARY (recent returns + 60d annualized vol)
{_format_price_summary(price_summary)}

RECENT NEWS ({sum(1 for _ in news_items) if not isinstance(news_items, list) else len(news_items)} items, ≤ {LLM_MAX_NEWS_ITEMS} cap)
{news_blob}

INSTRUCTIONS
{_SYSTEM_INSTRUCTIONS}

Return a JSON object matching the schema. Empty flags list is valid if no
anomalies are present. Never invent tickers absent from the portfolio."""


# ── SHA-256 caching ──────────────────────────────────────────────────────────

def _hash_bytes(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _cache_get(prompt_hash: str) -> dict | None:
    f = _CACHE_DIR / f"{prompt_hash}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_put(prompt: str, response_text: str, *,
                input_tokens: int, output_tokens: int, cost_usd: float) -> tuple[str, str]:
    p_hash = _hash_bytes(prompt)
    r_hash = _hash_bytes(response_text)
    payload = {
        "prompt_hash":   p_hash,
        "response_hash": r_hash,
        "prompt":        prompt,
        "response":      response_text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      cost_usd,
        "timestamp":     datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "model_version": LLM_MODEL_VERSION,
    }
    (_CACHE_DIR / f"{p_hash}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return p_hash, r_hash


# ── Cost tracking ────────────────────────────────────────────────────────────

def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * COST_PER_1M_INPUT_TOKENS +
            output_tokens * COST_PER_1M_OUTPUT_TOKENS) / 1_000_000.0


def _load_cost_tracker() -> dict:
    if not _COST_TRACKER_FILE.exists():
        return {"total_usd": 0.0, "calls": 0, "by_date": {}}
    try:
        return json.loads(_COST_TRACKER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"total_usd": 0.0, "calls": 0, "by_date": {}}


def _save_cost_tracker(state: dict) -> None:
    _COST_TRACKER_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def record_call_cost(scan_date: datetime.date, cost_usd: float) -> dict:
    """Append a cost event to the tracker and return updated state."""
    state = _load_cost_tracker()
    state["total_usd"] = round(state.get("total_usd", 0.0) + cost_usd, 6)
    state["calls"]     = state.get("calls", 0) + 1
    by_date = state.setdefault("by_date", {})
    key = str(scan_date)
    by_date[key] = round(by_date.get(key, 0.0) + cost_usd, 6)
    _save_cost_tracker(state)
    return state


def get_cost_status() -> dict:
    """Return current cost vs budget for the dashboard.

    Reads runtime budget from engine.llm_budget (SystemConfig-backed, falls
    back to engine.config.S6_COST_BUDGET_USD default if no override set).
    """
    state = _load_cost_tracker()
    total = state.get("total_usd", 0.0)
    budget = get_s6_anomaly_budget_usd_per_year()
    return {
        "total_usd":   total,
        "budget_usd":  budget,
        "fraction":    total / budget if budget > 0 else 0,
        "calls":       state.get("calls", 0),
        "alert_50pct": total >= 0.50 * budget,
        "alert_75pct": total >= 0.75 * budget,
        "alert_90pct": total >= 0.90 * budget,
    }


# ── LLM call (with cache + cost) ─────────────────────────────────────────────

def _call_llm(prompt: str) -> dict:
    """
    Call Gemini 2.5 Flash with locked config. Returns dict with parsed JSON
    + token usage + cost + cache hit indicator. Raises on hard error.
    """
    p_hash = _hash_bytes(prompt)
    cached = _cache_get(p_hash)
    if cached:
        return {
            "parsed":        json.loads(cached["response"]),
            "response_text": cached["response"],
            "prompt_hash":   p_hash,
            "response_hash": cached.get("response_hash", _hash_bytes(cached["response"])),
            "input_tokens":  cached.get("input_tokens", 0),
            "output_tokens": cached.get("output_tokens", 0),
            "cost_usd":      cached.get("cost_usd", 0.0),
            "cache_hit":     True,
        }

    from engine.key_pool import get_pool
    pool = get_pool()
    model = pool.get_model(
        model_name=LLM_MODEL_VERSION,
        response_schema=LLM_OUTPUT_SCHEMA,
        temperature=LLM_TEMPERATURE,
        thinking_budget=LLM_THINKING_BUDGET,
    )
    resp = model.generate_content(prompt)
    pool.report_success(has_content=True)

    text = getattr(resp, "text", None) or str(resp)
    usage = getattr(resp, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = (getattr(usage, "candidates_token_count", 0) or 0) + \
              (getattr(usage, "thoughts_token_count", 0) or 0)
    cost = _compute_cost(in_tok, out_tok)

    p_hash, r_hash = _cache_put(prompt, text,
                                 input_tokens=in_tok,
                                 output_tokens=out_tok,
                                 cost_usd=cost)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("anomaly_llm: JSON parse failed; first 200 chars: %s", text[:200])
        raise

    return {
        "parsed":        parsed,
        "response_text": text,
        "prompt_hash":   p_hash,
        "response_hash": r_hash,
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
        "cost_usd":      cost,
        "cache_hit":     False,
    }


# ── Main detector entry point ────────────────────────────────────────────────

def detect_llm(scan_date: datetime.date) -> list[dict]:
    """
    Run the LLM detector for scan_date. Returns list of flag dicts ready for
    persistence. Uses cached prompt/response if available (reproducibility).

    On API failure / quota exhaust, returns empty list and logs warning;
    never raises (resilience for daily cron).
    """
    from engine.anomaly_screener import (
        _get_current_holdings, _fetch_price_history,
        DEFAULT_HORIZON_DAYS,
    )

    holdings = _get_current_holdings(scan_date)
    if not holdings:
        logger.info("anomaly_llm: no holdings on %s; skip", scan_date)
        return []

    # Build price summary deterministically
    price_summary: dict[str, dict] = {}
    for ticker in holdings.keys():
        prices = _fetch_price_history(ticker, scan_date, days=90)
        if prices.empty:
            continue
        closes = prices["Close"].dropna()
        if len(closes) < 60:
            continue
        rets = closes.pct_change().dropna()
        try:
            ret_1d  = float(rets.iloc[-1])
            ret_5d  = float(closes.iloc[-1] / closes.iloc[-6] - 1) if len(closes) >= 6 else None
            ret_30d = float(closes.iloc[-1] / closes.iloc[-31] - 1) if len(closes) >= 31 else None
            sigma_60 = float(rets.iloc[-60:].std())
        except Exception:
            continue
        price_summary[ticker] = {
            "ret_1d":    ret_1d,
            "ret_5d":    ret_5d if ret_5d is not None else 0.0,
            "ret_30d":   ret_30d if ret_30d is not None else 0.0,
            "sigma_60d": sigma_60,
        }

    # Pull news (D4.3) — strictly NO macro_research input (B-6 isolation)
    try:
        from engine.news_fetchers import fetch_all_for_portfolio
        news_items_obj = fetch_all_for_portfolio(
            tickers=list(holdings.keys()),
            sectors=list({info.get("sector", "—") for info in holdings.values()}),
            days_back=2,
            max_items=LLM_MAX_NEWS_ITEMS,
        )
        news_items = [n.to_dict() for n in news_items_obj]
    except Exception as exc:
        logger.warning("anomaly_llm: news fetch failed (continuing with empty news): %s", exc)
        news_items = []

    prompt = build_prompt(scan_date, holdings, price_summary, news_items)

    try:
        out = _call_llm(prompt)
    except Exception as exc:
        logger.warning("anomaly_llm: API call failed: %s", exc)
        return []

    # Cost tracking
    if not out.get("cache_hit"):
        record_call_cost(scan_date, out["cost_usd"])
        status = get_cost_status()
        if status["alert_75pct"] and not status["alert_90pct"]:
            logger.warning("anomaly_llm: cost burn 75%% — total $%.2f / budget $%.2f",
                           status["total_usd"], status["budget_usd"])
        elif status["alert_90pct"]:
            logger.error("anomaly_llm: cost burn 90%% — total $%.2f / budget $%.2f",
                         status["total_usd"], status["budget_usd"])

    # Validate and shape
    parsed = out.get("parsed") or {}
    flags = parsed.get("flags") or []
    valid = []
    holdings_set = set(holdings.keys())
    for f in flags:
        # Schema enforced ticker presence; defensive skip
        if not isinstance(f, dict):
            continue
        ticker = f.get("ticker")
        if ticker not in holdings_set:
            logger.debug("anomaly_llm: dropping flag for non-held ticker %s", ticker)
            continue
        f["_meta"] = {
            "prompt_hash":   out["prompt_hash"],
            "response_hash": out["response_hash"],
            "input_tokens":  out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "cost_usd":      out["cost_usd"],
            "model_version": LLM_MODEL_VERSION,
            "news_refs":     [n.get("url") for n in news_items[:5]],  # top 5 audit trail
        }
        valid.append(f)

    logger.info("anomaly_llm: %d valid flags on %s (cache_hit=%s, cost=$%.4f)",
                len(valid), scan_date, out.get("cache_hit"), out.get("cost_usd", 0))
    return valid


# ── Persistence ──────────────────────────────────────────────────────────────

def persist_llm_flags(
    flags: list[dict],
    scan_date: datetime.date,
    *,
    spec_hash: str | None = None,
) -> list[int]:
    """Persist LLM-generated flags to engine.memory.AnomalyFlag with detector='llm'."""
    from engine.anomaly_screener import SPEC_HASH_PLACEHOLDER
    from engine.memory import SessionFactory, AnomalyFlag
    inserted: list[int] = []
    h = spec_hash or SPEC_HASH_PLACEHOLDER
    with SessionFactory() as session:
        for f in flags:
            ticker = f.get("ticker")
            existing = (
                session.query(AnomalyFlag)
                .filter(
                    AnomalyFlag.detector == "llm",
                    AnomalyFlag.scan_date == scan_date,
                    AnomalyFlag.ticker == ticker,
                ).first()
            )
            if existing:
                continue
            meta = f.get("_meta") or {}
            row = AnomalyFlag(
                detector            = "llm",
                scan_date           = scan_date,
                sector              = f.get("sector") or "—",
                ticker              = ticker,
                event_class         = f.get("event_class") or "price_spike",
                confidence_likert   = int(f.get("confidence_likert") or 1),
                horizon_days        = int(f.get("horizon_days") or LLM_HORIZON_DAYS),
                evidence_summary    = (f.get("evidence_summary") or "")[:200],
                triggering_rules    = None,
                news_refs           = json.dumps(meta.get("news_refs") or [], ensure_ascii=False),
                llm_model_version   = meta.get("model_version") or LLM_MODEL_VERSION,
                llm_prompt_hash     = meta.get("prompt_hash"),
                llm_response_hash   = meta.get("response_hash"),
                llm_cost_usd        = float(meta.get("cost_usd") or 0.0),
                llm_input_tokens    = int(meta.get("input_tokens") or 0),
                llm_output_tokens   = int(meta.get("output_tokens") or 0),
                spec_hash           = h,
            )
            session.add(row)
            session.flush()
            inserted.append(row.id)
        session.commit()
    return inserted


def run_llm_scan_for_date(scan_date: datetime.date) -> dict:
    """Cron entry point — detect + persist."""
    flags = detect_llm(scan_date)
    ids = persist_llm_flags(flags, scan_date)
    cost = get_cost_status()
    return {
        "scan_date":  str(scan_date),
        "n_flags":    len(ids),
        "flag_ids":   ids,
        "cost_total": cost["total_usd"],
        "cost_pct":   cost["fraction"],
    }
