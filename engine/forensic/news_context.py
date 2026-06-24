"""
engine/forensic/news_context.py — Sprint H follow-up LLM news summarizer.

Given a (date, ticker, signal_value, weight, realized_return) tuple from a
Sprint H trade log row, fetches news headlines around the trade date and
uses Gemini 2.5 Flash to produce a structured forensic summary classifying
the move as:
  case_a — signal was wrong / over-fit
  case_b — signal correct but horizon incomplete
  case_c — exogenous shock (specific news event)

DOCTRINE: This module is in the FORENSIC layer (engine.forensic.*), NOT in
the decision layer. It uses LLM, but its output never feeds back into
strategy / portfolio / backtest decisions. 0-LLM-in-DECISION preserved.

Reuses:
  - engine.news.NewsPerceiver (AV + GNews + Yahoo RSS, free tier)
  - Vertex ADC REST pattern from engine.d_pead_plus.llm_extractor_rest
  - engine.llm_cost_ledger.record_call (agent_id='forensic_news_context')

Spec: docs/spec_forensic_news_context_v1.md
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

from engine.agents.observability import track_agent_invocation

logger = logging.getLogger(__name__)

# ── LLM config (Vertex ADC REST, same model as Sprint I llm_extractor_rest) ──
LLM_MODEL:             str   = "gemini-2.5-flash"
LLM_TEMPERATURE:       float = 0.1   # slight diversity for narrative variety; not 0 because not statistical
LLM_TOP_P:             float = 0.95
LLM_MAX_OUTPUT_TOKENS: int   = 800
VERTEX_LOCATION:       str   = "us-central1"
HTTP_TIMEOUT_S:        float = 120.0

VERTEX_API_URL_TEMPLATE: str = (
    "https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
    "/locations/{location}/publishers/google/models/{model}:generateContent"
)

# Response schema (Vertex REST format)
_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "material_events":      {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "macro_context":        {"type": "string"},
        "sentiment_assessment": {"type": "string"},
        "signal_alignment":     {"type": "string"},
        "key_quotes":           {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "forensic_verdict":     {"type": "string", "enum": ["case_a", "case_b", "case_c"]},
    },
    "required": [
        "material_events", "macro_context", "sentiment_assessment",
        "signal_alignment", "key_quotes", "forensic_verdict",
    ],
}

_SYSTEM_PROMPT: str = (
    "You are a senior quant analyst doing forensic analysis of a paper-trade "
    "drawdown. You receive (a) trade context: strategy, signal value, weight, "
    "realized return, expected horizon, and (b) a list of news headlines from "
    "the 10-day window around the trade. Your job: classify the move into one "
    "of three cases and provide a structured forensic verdict. Be concise, "
    "evidence-based, and quote sources verbatim when possible. "
    "DO NOT speculate beyond the headlines."
)

_USER_TEMPLATE: str = (
    "TRADE CONTEXT\n"
    "  Date:              {date}\n"
    "  Ticker:            {ticker}\n"
    "  Strategy:          {strategy_name}\n"
    "  Signal value:      {signal_value}\n"
    "  Weight:            {weight:+.4f}\n"
    "  Realized return:   {realized_return}\n"
    "  Expected horizon:  {expected_horizon_days} days\n\n"
    "NEWS HEADLINES (date window {date_start} to {date_end}, {n_articles} articles)\n"
    "{headlines_text}\n\n"
    "TASK: Classify into case_a (signal wrong), case_b (horizon incomplete), "
    "or case_c (exogenous shock). Output structured JSON per schema."
)


@dataclasses.dataclass(frozen=True)
class ForensicNewsSummary:
    """Output of investigate_trade(). Forensic narrative for DD investigation."""
    # Echoed context
    date:                  datetime.date
    ticker:                str
    strategy_name:         str
    signal_value:          Optional[float]
    weight:                float
    realized_return:       Optional[float]
    expected_horizon_days: int

    # Fetch metadata
    date_window_start:     datetime.date
    date_window_end:       datetime.date
    n_articles:            int
    n_sources:             int

    # LLM outputs
    material_events:       tuple[str, ...]
    macro_context:         str
    sentiment_assessment:  str
    signal_alignment:      str
    key_quotes:            tuple[str, ...]
    forensic_verdict:      str          # 'case_a' / 'case_b' / 'case_c'

    # Audit
    cost_usd:              float
    llm_model:             str
    llm_latency_ms:        int
    extracted_at_utc:      datetime.datetime

    def to_markdown(self) -> str:
        """Human-readable forensic report."""
        verdict_label = {
            "case_a": "Signal Wrong / Over-fit",
            "case_b": "Horizon Incomplete (wait)",
            "case_c": "Exogenous Shock",
        }.get(self.forensic_verdict, self.forensic_verdict)
        return (
            f"# Forensic Investigation — {self.ticker} on {self.date}\n\n"
            f"**Strategy:** {self.strategy_name}  |  "
            f"**Signal:** {self.signal_value}  |  "
            f"**Weight:** {self.weight:+.4f}\n"
            f"**Realized return:** {self.realized_return}  |  "
            f"**Expected horizon:** {self.expected_horizon_days}d\n\n"
            f"## Verdict: **{self.forensic_verdict}** ({verdict_label})\n\n"
            f"### Material Events\n"
            + "\n".join(f"- {e}" for e in self.material_events)
            + "\n\n"
            f"### Macro Context\n{self.macro_context}\n\n"
            f"### Sentiment Assessment\n{self.sentiment_assessment}\n\n"
            f"### Signal Alignment Analysis\n{self.signal_alignment}\n\n"
            f"### Key Quotes\n"
            + "\n".join(f"> {q}" for q in self.key_quotes)
            + f"\n\n---\n"
            f"_Sources: {self.n_articles} articles from {self.n_sources} feeds "
            f"({self.date_window_start} → {self.date_window_end}). "
            f"LLM cost: ${self.cost_usd:.4f}, latency: {self.llm_latency_ms}ms. "
            f"Extracted: {self.extracted_at_utc.isoformat()}._"
        )

    def to_json(self) -> str:
        d = dataclasses.asdict(self)
        d["date"]                = self.date.isoformat()
        d["date_window_start"]   = self.date_window_start.isoformat()
        d["date_window_end"]     = self.date_window_end.isoformat()
        d["extracted_at_utc"]    = self.extracted_at_utc.isoformat()
        return json.dumps(d, ensure_ascii=False, indent=2)


def _get_auth_state(force_refresh: bool = False, _cache: list = []):
    """Vertex ADC token cache (same pattern as engine.d_pead_plus.llm_extractor_rest)."""
    from google.auth import default as default_auth
    from google.auth.transport.requests import Request as AuthRequest

    now = datetime.datetime.utcnow()
    if _cache and not force_refresh:
        state = _cache[0]
        if (state["expires_at"] - now).total_seconds() > 300:
            return state

    creds, project_id = default_auth(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(AuthRequest())
    state = {
        "token":      creds.token,
        "project_id": project_id,
        "expires_at": creds.expiry or (now + datetime.timedelta(minutes=55)),
    }
    _cache.clear()
    _cache.append(state)
    return state


# ── AV cache (light JSON file) ──────────────────────────────────────────────
_AV_CACHE_PATH:        Path = Path("data/forensic/av_news_cache.json")
_AV_CACHE_TTL_DAYS:    int  = 30        # news doesn't change after publication
_AV_RETRY_DELAY_S:     float = 5.0      # wait then retry once on 429 / Note


def _av_cache_load() -> dict:
    """Load AV cache from disk. Returns {} if missing/corrupt."""
    if not _AV_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(_AV_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("AV cache corrupt; starting fresh")
        return {}


def _av_cache_save(cache: dict) -> None:
    """Save AV cache to disk (atomic-ish via temp file)."""
    _AV_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _AV_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_AV_CACHE_PATH)


def _av_cache_key(ticker: str, time_from: str, time_to: str) -> str:
    return f"{ticker.upper()}|{time_from}|{time_to}"


def _av_cache_get(ticker: str, time_from: str, time_to: str) -> Optional[list[dict]]:
    """Return cached headlines if entry exists and not expired."""
    cache = _av_cache_load()
    entry = cache.get(_av_cache_key(ticker, time_from, time_to))
    if entry is None:
        return None
    cached_at = entry.get("cached_at_iso")
    if cached_at:
        try:
            age_days = (datetime.datetime.utcnow() - datetime.datetime.fromisoformat(cached_at)).days
            if age_days > _AV_CACHE_TTL_DAYS:
                return None
        except Exception:
            return None
    return entry.get("headlines", [])


def _av_cache_set(ticker: str, time_from: str, time_to: str, headlines: list[dict]) -> None:
    """Write headlines to cache."""
    cache = _av_cache_load()
    cache[_av_cache_key(ticker, time_from, time_to)] = {
        "headlines":     headlines,
        "cached_at_iso": datetime.datetime.utcnow().isoformat(),
        "n_articles":    len(headlines),
    }
    _av_cache_save(cache)


def _fetch_alpha_vantage_historical(
    ticker:       str,
    window_start: datetime.date,
    window_end:   datetime.date,
    av_key:       str,
    n_max:        int = 50,
    use_cache:    bool = True,
) -> list[dict]:
    """Direct AV NEWS_SENTIMENT call with time_from/time_to (NOT the 48hr-only
    helper in engine.news.NewsPerceiver).

    AV API supports time_from / time_to up to ~2 years history on free tier.
    Format: YYYYMMDDTHHMM (UTC).

    Cache: by (ticker, time_from, time_to) in data/forensic/av_news_cache.json,
    TTL 30 days. Honors `use_cache=False` for forced refresh.
    Retry: once on 429 / rate-limit Note, with 5s backoff.

    Returns list of headline dicts; empty if no key / API failure / no results.
    """
    if not av_key:
        return []

    time_from = window_start.strftime("%Y%m%dT0000")
    time_to   = window_end.strftime("%Y%m%dT2359")

    if use_cache:
        cached = _av_cache_get(ticker, time_from, time_to)
        if cached is not None:
            logger.info("AV cache HIT: %s [%s-%s] %d articles",
                        ticker, time_from, time_to, len(cached))
            return cached

    import requests
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}"
        f"&time_from={time_from}&time_to={time_to}"
        f"&sort=LATEST&limit={n_max}&apikey={av_key}"
    )

    def _call_once():
        resp = requests.get(url, timeout=15)
        return resp.json()

    try:
        data = _call_once()
        # Retry once if AV returns a rate-limit Note instead of feed
        if not data.get("feed") and (data.get("Note") or data.get("Information")):
            note = (data.get("Note") or data.get("Information"))[:160]
            logger.warning("AV rate-limit / info note: %s — retrying in %.1fs",
                           note, _AV_RETRY_DELAY_S)
            time.sleep(_AV_RETRY_DELAY_S)
            data = _call_once()
    except Exception as exc:
        logger.warning("AV historical fetch failed for %s [%s-%s]: %s",
                       ticker, time_from, time_to, exc)
        return []

    feed = data.get("feed", [])
    if not feed:
        # AV may return informational note when no data (e.g., rate-limited)
        note = data.get("Note") or data.get("Information") or ""
        if note:
            logger.info("AV historical note for %s: %s", ticker, note[:120])
        return []

    headlines: list[dict] = []
    for item in feed:
        try:
            t = item.get("time_published", "")
            published = datetime.datetime.strptime(t, "%Y%m%dT%H%M%S")
            date_str = published.strftime("%Y-%m-%d %H:%M UTC")
            # Defense-in-depth: explicitly drop entries outside window even if
            # AV ignores our params (free tier sometimes returns broader range)
            if published.date() < window_start or published.date() > window_end:
                continue
        except Exception:
            date_str = "n/a"

        headlines.append({
            "title":           item.get("title", "").strip(),
            "source":          item.get("source", "Alpha Vantage"),
            "published":       date_str,
            "sentiment_label": item.get("overall_sentiment_label", ""),
            "sentiment_score": float(item.get("overall_sentiment_score", 0)),
        })

    # Write to cache (even if empty list — caches the "no results" verdict for TTL)
    if use_cache:
        try:
            _av_cache_set(ticker, time_from, time_to, headlines)
        except Exception:
            logger.exception("AV cache write failed (non-fatal)")

    return headlines


def fetch_news_window(
    ticker:     str,
    date:       datetime.date,
    window_days: int = 5,
) -> tuple[list[dict], datetime.date, datetime.date, int]:
    """Fetch news headlines from (date - window_days) to (date + window_days).

    Primary: AlphaVantage NEWS_SENTIMENT with time_from/time_to (historical).
    Fallback: Yahoo Finance RSS (live, last ~30d only — not useful for historical).

    Returns (headlines, window_start, window_end, n_sources).
    """
    import os

    av_key = os.environ.get("AV_KEY", "") or ""
    if not av_key:
        # Fallback to streamlit secrets (matches app.py:248 pattern)
        try:
            import streamlit as st
            av_key = st.secrets.get("AV_KEY", "") or ""
        except Exception:
            pass

    window_start = date - datetime.timedelta(days=window_days)
    window_end   = date + datetime.timedelta(days=window_days)

    # Primary: AV historical query honoring date window
    av_headlines = _fetch_alpha_vantage_historical(
        ticker, window_start, window_end, av_key, n_max=30,
    )

    # Fallback: RSS only if AV returned nothing AND date is recent (< 30d old)
    rss_headlines: list[dict] = []
    today = datetime.date.today()
    if not av_headlines and (today - window_end).days <= 30:
        try:
            from engine.news import NewsPerceiver
            perceiver = NewsPerceiver(av_key=av_key, gnews_key="")
            rss_headlines = perceiver._fetch_rss(ticker, sector_name="general", n=10)
        except Exception as exc:
            logger.warning("RSS fallback failed for %s: %s", ticker, exc)

    all_h: list[dict] = []
    sources_set: set[str] = set()
    seen_titles: set[str] = set()

    for h in (av_headlines + rss_headlines):
        title = h.get("title", "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)
        all_h.append(h)
        sources_set.add(h.get("source", "unknown"))

    return all_h, window_start, window_end, len(sources_set)


def _build_headlines_text(headlines: list[dict]) -> str:
    if not headlines:
        return "(No news headlines found in the window.)"
    lines = []
    for i, h in enumerate(headlines[:25], 1):
        title    = h.get("title", "").strip()
        source   = h.get("source", "unknown")
        pub      = h.get("published", "n/a")
        sent     = h.get("sentiment_label", "")
        sent_s   = f" [{sent}]" if sent else ""
        lines.append(f"  {i}. ({source}, {pub}{sent_s}) {title}")
    return "\n".join(lines)


def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Gemini 2.5 Flash on VERTEX AI pricing per million tokens.
    Vertex tier (used by this module's Vertex ADC REST endpoint):
      input $0.30 / output $2.50 per 1M tokens.
    NOT Google AI Studio direct ($0.075 / $0.30 — 4-8x cheaper but different API).
    """
    INPUT_PER_M  = 0.30
    OUTPUT_PER_M = 2.50
    return (input_tokens * INPUT_PER_M + output_tokens * OUTPUT_PER_M) / 1_000_000.0


def _news_context_schema_validator(result: ForensicNewsSummary) -> bool:
    """Schema check: required fields present + verdict in valid set."""
    return (
        isinstance(result, ForensicNewsSummary)
        and result.forensic_verdict in ("case_a", "case_b", "case_c")
        and isinstance(result.material_events, tuple)
        and isinstance(result.signal_alignment, str)
        and result.cost_usd >= 0
    )


def _news_context_quality_extractor(result: ForensicNewsSummary) -> dict:
    """LCS-style grounding proxy: how much of LLM summary's keywords appear
    in source headlines? High overlap = LLM stayed grounded; low = potential
    hallucination."""
    # Build keyword set from LLM material_events
    llm_keywords: set[str] = set()
    for ev in result.material_events:
        for word in str(ev).lower().split():
            cleaned = "".join(c for c in word if c.isalnum())
            if len(cleaned) > 3:    # filter stopwords-ish
                llm_keywords.add(cleaned)

    # Need source headlines — fetched via cache. Best-effort retrieval:
    source_keywords: set[str] = set()
    try:
        cache_key = _av_cache_key(
            result.ticker,
            result.date_window_start.isoformat(),
            result.date_window_end.isoformat(),
        )
        cached_headlines = _av_cache_get(
            result.ticker,
            result.date_window_start.isoformat(),
            result.date_window_end.isoformat(),
        ) or []
        for h in cached_headlines:
            title = str(h.get("title") or "")
            for word in title.lower().split():
                cleaned = "".join(c for c in word if c.isalnum())
                if len(cleaned) > 3:
                    source_keywords.add(cleaned)
    except Exception:
        pass

    if not llm_keywords:
        overlap_ratio = None
    elif not source_keywords:
        overlap_ratio = None    # source data not available, can't measure
    else:
        overlap_ratio = len(llm_keywords & source_keywords) / len(llm_keywords)

    return {
        "source_keyword_overlap":  overlap_ratio,
        "n_llm_keywords":          len(llm_keywords),
        "n_source_keywords":       len(source_keywords),
        "n_material_events":       len(result.material_events),
        "n_key_quotes":            len(result.key_quotes),
        "forensic_verdict":        result.forensic_verdict,
        "n_articles_fetched":      result.n_articles,
        "n_sources_fetched":       result.n_sources,
    }


def _news_context_extra(result: ForensicNewsSummary) -> dict:
    return {
        "ticker":          result.ticker,
        "strategy_name":   result.strategy_name,
        "n_tool_calls":    1,    # 1 LLM call (Vertex Gemini)
        "llm_model":       result.llm_model,
    }


@track_agent_invocation(
    agent_id="forensic_news_context",
    schema_validator=_news_context_schema_validator,
    extract_extra=_news_context_extra,
    quality_extractor=_news_context_quality_extractor,
)
def investigate_trade(
    date:                  datetime.date,
    ticker:                str,
    signal_value:          Optional[float],
    weight:                float,
    realized_return:       Optional[float],
    strategy_name:         str,
    expected_horizon_days: int,
    *,
    window_days:           int = 5,
) -> ForensicNewsSummary:
    """Run end-to-end forensic news investigation for one trade.

    Convenience entry point. Typical use:
        from engine.forensic.news_context import investigate_trade
        summary = investigate_trade(
            date=datetime.date(2026,6,15), ticker='NVDA',
            signal_value=2.31, weight=0.04, realized_return=-0.154,
            strategy_name='D_PEAD', expected_horizon_days=60,
        )
        print(summary.to_markdown())
    """
    # Step 1-2: fetch + filter
    headlines, win_start, win_end, n_sources = fetch_news_window(
        ticker, date, window_days=window_days,
    )

    # Step 3: build prompt + LLM call
    auth = _get_auth_state()
    url = VERTEX_API_URL_TEMPLATE.format(
        location=VERTEX_LOCATION, project=auth["project_id"], model=LLM_MODEL,
    )
    user_prompt = _USER_TEMPLATE.format(
        date=date.isoformat(), ticker=ticker, strategy_name=strategy_name,
        signal_value=signal_value, weight=weight,
        realized_return=("n/a" if realized_return is None else f"{realized_return:+.4f}"),
        expected_horizon_days=expected_horizon_days,
        date_start=win_start.isoformat(), date_end=win_end.isoformat(),
        n_articles=len(headlines),
        headlines_text=_build_headlines_text(headlines),
    )
    body = {
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents":           [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature":       LLM_TEMPERATURE,
            "topP":               LLM_TOP_P,
            "maxOutputTokens":    LLM_MAX_OUTPUT_TOKENS,
            "responseMimeType":   "application/json",
            "responseSchema":     _RESPONSE_SCHEMA,
            "thinkingConfig":     {"thinkingBudget": 0},
        },
    }
    headers = {
        "Authorization": f"Bearer {auth['token']}",
        "Content-Type":  "application/json",
    }

    t0 = time.time()
    with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
        resp = client.post(url, json=body, headers=headers)
        if resp.status_code == 401:
            auth = _get_auth_state(force_refresh=True)
            headers["Authorization"] = f"Bearer {auth['token']}"
            resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    latency_ms = int((time.time() - t0) * 1000)

    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)

    usage   = payload.get("usageMetadata", {})
    in_tok  = int(usage.get("promptTokenCount", 0) or 0)
    out_tok = int(usage.get("candidatesTokenCount", 0) or 0) + int(usage.get("thoughtsTokenCount", 0) or 0)
    cost    = _compute_cost(in_tok, out_tok)

    # Cost ledger
    try:
        from engine.llm_cost_ledger import record_call
        record_call(
            agent_id          = "forensic_news_context",
            provider          = "gemini",
            model             = LLM_MODEL,
            prompt_tokens     = in_tok,
            completion_tokens = out_tok,
            cost_usd          = cost,
            latency_ms        = latency_ms,
            scope             = f"investigate_trade:{strategy_name}:{ticker}:{date.isoformat()}",
        )
    except Exception:
        logger.exception("forensic_news_context: cost ledger record failed (non-fatal)")

    summary = ForensicNewsSummary(
        date                  = date,
        ticker                = ticker,
        strategy_name         = strategy_name,
        signal_value          = signal_value,
        weight                = weight,
        realized_return       = realized_return,
        expected_horizon_days = expected_horizon_days,
        date_window_start     = win_start,
        date_window_end       = win_end,
        n_articles            = len(headlines),
        n_sources             = n_sources,
        material_events       = tuple(parsed.get("material_events", [])),
        macro_context         = parsed.get("macro_context", ""),
        sentiment_assessment  = parsed.get("sentiment_assessment", ""),
        signal_alignment      = parsed.get("signal_alignment", ""),
        key_quotes            = tuple(parsed.get("key_quotes", [])),
        forensic_verdict      = parsed.get("forensic_verdict", "case_a"),
        cost_usd              = cost,
        llm_model             = LLM_MODEL,
        llm_latency_ms        = latency_ms,
        extracted_at_utc      = datetime.datetime.utcnow(),
    )
    logger.info("forensic_news_context: %s on %s → %s (%d articles, $%.4f, %dms)",
                ticker, date, summary.forensic_verdict, summary.n_articles, cost, latency_ms)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# DD report synthesis pass (Phase 1.5 — single LLM consolidation call)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class DDSynthesis:
    """Output of synthesize_dd_report — narrative consolidation only."""
    tl_dr:                 str       # 2-3 sentence executive summary
    cross_trade_pattern:   str       # observation across worst trades
    action_priority:       tuple[str, ...]  # rule-mapped action list, ordered
    cost_usd:              float
    llm_latency_ms:        int


_SYNTHESIS_SYSTEM_PROMPT: str = (
    "You are a senior quant doing FORENSIC NARRATIVE CONSOLIDATION ONLY. "
    "You will receive: (a) deterministic strategy contribution data, (b) Brinson "
    "+ FF5 factor decomposition outputs, (c) 3-5 per-trade LLM forensic verdicts "
    "already classified as case_a/case_b/case_c. "
    "Your job: write a CONSOLIDATED summary. Constraints:\n"
    "- Use ONLY the data provided. NEVER invent historical analogies or pattern "
    "  comparisons not present in the data.\n"
    "- NEVER recommend strategy retirement / position-size changes / new actions. "
    "  Action recommendations MUST come from this fixed rule book:\n"
    "    case_a (signal wrong) → 'no new adds same direction; flag for Forward IC validation'\n"
    "    case_b (horizon incomplete) → 'hold position; reassess at horizon end'\n"
    "    case_c (exogenous shock) → 'hold position; assess cluster risk in Brinson decomp'\n"
    "- When FF5 decomp shows market_component >> idio_component, EXPLICITLY note "
    "  the DD was systematic, not signal-driven.\n"
    "- Cross-trade pattern: identify if multiple worst trades share strategy, "
    "  sector cluster, or factor exposure — based on input data ONLY.\n"
    "- TL;DR: 2-3 sentences. Action priority: ordered list of 1-3 items mapped "
    "  from the rule book."
)


_SYNTHESIS_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "tl_dr":               {"type": "string"},
        "cross_trade_pattern": {"type": "string"},
        "action_priority":     {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["tl_dr", "cross_trade_pattern", "action_priority"],
}


def _build_synthesis_user_prompt(
    as_of:                datetime.date,
    strategy_contribs:    list[dict],
    forensic_summaries:   list[ForensicNewsSummary],
    brinson_result:       Optional[dict],
    factor_decomp_result: Optional[dict],
) -> str:
    lines = [f"DD INVESTIGATION DATE: {as_of.isoformat()}\n"]

    lines.append("STRATEGY CONTRIBUTION (deterministic):")
    for c in strategy_contribs:
        lines.append(f"  - {c['strategy_name']}: {c['contribution']:+.4%} "
                     f"({c['n_trades']} trades, "
                     f"{c.get('n_with_returns','?')} with realized returns)")
    lines.append("")

    if brinson_result and brinson_result.get("status") == "OK":
        lines.append("BRINSON ATTRIBUTION:")
        lines.append(f"  Portfolio total: {brinson_result['portfolio_total']:+.4%}")
        for sleeve, d in brinson_result.get("by_sleeve", {}).items():
            lines.append(f"  Sleeve {sleeve}: {d['contribution']:+.4%} "
                         f"({d['n_trades']} trades)")
        lines.append("")

    if factor_decomp_result and factor_decomp_result.get("status") == "OK":
        lines.append("FF5 FACTOR DECOMP:")
        lines.append(f"  avg_trade_return:          {factor_decomp_result['avg_trade_ret']:+.4%}")
        lines.append(f"  approx_market_component:   {factor_decomp_result['approx_market_component']:+.4%}")
        lines.append(f"  approx_idio_component:     {factor_decomp_result['approx_idio_component']:+.4%}")
        lines.append("")

    lines.append("PER-TRADE FORENSIC VERDICTS (LLM-classified):")
    for i, s in enumerate(forensic_summaries, 1):
        rr = f"{s.realized_return:+.2%}" if s.realized_return is not None else "n/a"
        lines.append(f"  Trade #{i}: {s.ticker} ({s.strategy_name})")
        lines.append(f"    Signal: {s.signal_value}, Weight: {s.weight:+.4f}, Realized: {rr}")
        lines.append(f"    Verdict: {s.forensic_verdict}")
        if s.material_events:
            lines.append(f"    Top event: {s.material_events[0]}")
    lines.append("")

    lines.append("TASK: Produce structured JSON synthesis. Stay within the rule "
                 "book for action_priority. NEVER invent new actions or analogies.")
    return "\n".join(lines)


def synthesize_dd_report(
    as_of:                datetime.date,
    strategy_contribs:    list[dict],
    forensic_summaries:   list[ForensicNewsSummary],
    brinson_result:       Optional[dict] = None,
    factor_decomp_result: Optional[dict] = None,
) -> Optional[DDSynthesis]:
    """Single LLM consolidation call — produces TL;DR + cross-trade pattern +
    rule-book-mapped action priority. Returns None if no forensic verdicts.

    DOCTRINE: pure narrative consolidation; action recommendations come from
    a fixed rule book embedded in the system prompt. LLM cannot retire or
    resize strategies.
    """
    if not forensic_summaries:
        return None

    auth = _get_auth_state()
    url = VERTEX_API_URL_TEMPLATE.format(
        location=VERTEX_LOCATION, project=auth["project_id"], model=LLM_MODEL,
    )
    user_prompt = _build_synthesis_user_prompt(
        as_of, strategy_contribs, forensic_summaries, brinson_result, factor_decomp_result,
    )
    body = {
        "systemInstruction": {"parts": [{"text": _SYNTHESIS_SYSTEM_PROMPT}]},
        "contents":           [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature":      0.0,                            # synthesis = deterministic
            "topP":              1.0,
            "maxOutputTokens":   600,
            "responseMimeType":  "application/json",
            "responseSchema":    _SYNTHESIS_RESPONSE_SCHEMA,
            "thinkingConfig":    {"thinkingBudget": 0},
        },
    }
    headers = {
        "Authorization": f"Bearer {auth['token']}",
        "Content-Type":  "application/json",
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
            resp = client.post(url, json=body, headers=headers)
            if resp.status_code == 401:
                auth = _get_auth_state(force_refresh=True)
                headers["Authorization"] = f"Bearer {auth['token']}"
                resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning("synthesize_dd_report LLM call failed: %s", exc)
        return None
    latency_ms = int((time.time() - t0) * 1000)

    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)

    usage   = payload.get("usageMetadata", {})
    in_tok  = int(usage.get("promptTokenCount", 0) or 0)
    out_tok = int(usage.get("candidatesTokenCount", 0) or 0) + int(usage.get("thoughtsTokenCount", 0) or 0)
    cost    = _compute_cost(in_tok, out_tok)

    # Cost ledger
    try:
        from engine.llm_cost_ledger import record_call
        record_call(
            agent_id          = "forensic_news_context",
            provider          = "gemini",
            model             = LLM_MODEL,
            prompt_tokens     = in_tok,
            completion_tokens = out_tok,
            cost_usd          = cost,
            latency_ms        = latency_ms,
            scope             = f"synthesize_dd_report:{as_of.isoformat()}",
        )
    except Exception:
        logger.exception("synthesize cost ledger record failed (non-fatal)")

    return DDSynthesis(
        tl_dr               = parsed.get("tl_dr", ""),
        cross_trade_pattern = parsed.get("cross_trade_pattern", ""),
        action_priority     = tuple(parsed.get("action_priority", [])),
        cost_usd            = cost,
        llm_latency_ms      = latency_ms,
    )
