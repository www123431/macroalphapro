"""
Trading Schema — Structured data contracts for the Trading Agent system.

All agent outputs must conform to these schemas before persisting or passing
to downstream agents. This is the single source of truth for inter-agent
communication in the trading pipeline.

Design principles
-----------------
- No free text in decision-critical fields — everything typed
- Every recommendation traces back to a DecisionLog id (research provenance)
- Invalidation and entry conditions are typed dicts, serializable to JSON
- Schema is agent-agnostic: QuantAgent, ResearchAgent, RiskAgent all produce
  TradeRecommendation; attribution is tracked via source_agent field
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional

# ── Type aliases ───────────────────────────────────────────────────────────────

PositionRank  = Literal["core", "satellite", "tactical"]
Direction     = Literal["long", "short", "neutral"]
GateStatus    = Literal["open", "blocked"]
RegimeLabel   = Literal["risk-on", "risk-off", "transition"]

# ── Weight limits: position_rank × regime ─────────────────────────────────────
# Caps tighten as regime deteriorates; risk-off is binding for daily patrol.
# tactical is fully blocked in risk-off (0.00 forces exit via patrol).

WEIGHT_LIMITS: dict[PositionRank, dict[RegimeLabel, float]] = {
    "core":      {"risk-on": 0.20, "risk-off": 0.15, "transition": 0.17},
    "satellite": {"risk-on": 0.10, "risk-off": 0.06, "transition": 0.08},
    "tactical":  {"risk-on": 0.05, "risk-off": 0.00, "transition": 0.03},
}

# ── Composite score gate ───────────────────────────────────────────────────────
COMPOSITE_GATE_MIN = 35   # below this → gate_status = "blocked"


# ── Invalidation condition ─────────────────────────────────────────────────────

@dataclass
class InvalidationCondition:
    """
    Typed condition that, if met, transitions a WatchlistEntry to 'invalidated'.

    Quant conditions are evaluated automatically each day by DailyBatchJob.
    Descriptive conditions require manual confirmation via UI.

    Rules
    -----
    tsmom_flipped   : current TSMOM sign ≠ entry_value recorded at pool creation
    price_below_sma : T-day close < SMA(sma_period)
    descriptive     : free-text event (e.g. "Fed unexpected hike") — manual only
    """
    type: Literal["quant", "descriptive"]
    rule: Optional[Literal["tsmom_flipped", "price_below_sma"]] = None
    entry_value: Optional[int]   = None   # tsmom_flipped: original signal (+1 / -1)
    sma_period:  Optional[int]   = None   # price_below_sma: SMA window
    description: Optional[str]  = None   # descriptive conditions only


# ── Entry condition ────────────────────────────────────────────────────────────

@dataclass
class EntryCondition:
    """
    Typed price-based entry condition evaluated daily against T-day close.

    immediate       : enter at next available close (no condition)
    price_breakout  : T-day close > highest close over past n_days
    volume_confirm  : T-day volume > volume_multiple × 20d avg AND close up
    ma_crossover    : T-day close crosses above ma_period-day SMA
    """
    type: Literal["immediate", "price_breakout", "volume_confirm", "ma_crossover"]
    n_days:          Optional[int]   = None   # price_breakout lookback
    volume_multiple: Optional[float] = None   # volume_confirm multiplier
    ma_period:       Optional[int]   = None   # ma_crossover period


# ── Risk Condition ────────────────────────────────────────────────────────────

@dataclass
class RiskCondition:
    """
    Per-position risk constraint evaluated daily by _patrol_positions.

    If triggered → generates a PendingApproval with suggested_weight reduced.

    Rules
    -----
    vol_spike   : ann_vol > threshold → compress weight to vol_spike_cap
    drawdown    : position return < -threshold → generate exit approval
    regime_cap  : actual_weight > WEIGHT_LIMITS[rank][regime] → compress
    """
    type: Literal["vol_spike", "drawdown", "regime_cap"]
    threshold:     Optional[float] = None   # vol_spike: vol level; drawdown: loss level
    vol_spike_cap: Optional[float] = None   # vol_spike: target weight after compression
    description:   Optional[str]  = None   # human-readable label


# ── Trade Recommendation ───────────────────────────────────────────────────────

@dataclass
class TradeRecommendation:
    """
    Canonical structured output from any Trading Agent.

    This is the single data contract flowing from agent outputs into
    WatchlistEntry creation. All downstream state-machine logic (entry checks,
    invalidation checks, risk patrol) reads from the WatchlistEntry columns
    that are populated from this schema.

    Fields
    ------
    source_agent        : which agent generated this recommendation
    decision_log_id     : FK → DecisionLog (None for pure quant recommendations)
    quant_baseline_weight : QuantAgent's vol-parity weight before LLM adjustment
    llm_adjustment_pct  : delta applied by ResearchAgent (-0.10 to +0.10 as fraction)
    suggested_weight    : quant_baseline_weight + llm_adjustment_pct (clipped to cap)
    """
    # Identity
    sector: str
    ticker: str

    # Direction and sizing
    direction:              Direction
    position_rank:          PositionRank
    quant_baseline_weight:  float
    llm_adjustment_pct:     float
    suggested_weight:       float

    # Quant signal state at recommendation time
    regime_label:    RegimeLabel
    tsmom_signal:    int    # +1 / 0 / -1
    csmom_rank:      int    # 1 = strongest in universe
    composite_score: int    # 0-100
    ann_vol:         float
    gate_status:     GateStatus

    # Decision provenance
    source_agent:    str
    confidence:      int    # 0-100
    decision_log_id: Optional[int] = None

    # State machine config
    entry_condition:          EntryCondition              = field(default_factory=lambda: EntryCondition(type="immediate"))
    invalidation_conditions:  list[InvalidationCondition] = field(default_factory=list)
    risk_conditions:          list[RiskCondition]         = field(default_factory=list)

    # Metadata
    as_of_date: Optional[date] = None

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_watchlist_dict(self) -> dict:
        """Produce flat dict for WatchlistEntry ORM creation."""
        return {
            "sector":                  self.sector,
            "ticker":                  self.ticker,
            "direction":               self.direction,
            "position_rank":           self.position_rank,
            "quant_baseline_weight":   self.quant_baseline_weight,
            "llm_adjustment_pct":      self.llm_adjustment_pct,
            "suggested_weight":        self.suggested_weight,
            "regime_label":            self.regime_label,
            "tsmom_signal":            self.tsmom_signal,
            "csmom_rank":              self.csmom_rank,
            "composite_score":         self.composite_score,
            "ann_vol":                 self.ann_vol,
            "gate_status":             self.gate_status,
            "source_agent":            self.source_agent,
            "confidence":              self.confidence,
            "decision_log_id":         self.decision_log_id,
            "entry_condition_json":    json.dumps(dataclasses.asdict(self.entry_condition)),
            "invalidation_json":       json.dumps([dataclasses.asdict(c) for c in self.invalidation_conditions]),
            "risk_conditions_json":    json.dumps([dataclasses.asdict(c) for c in self.risk_conditions]),
        }

    @classmethod
    def from_watchlist_row(cls, row) -> "TradeRecommendation":
        """Reconstruct from a WatchlistEntry ORM row (partial — no state fields)."""
        ec_raw = json.loads(row.entry_condition_json or '{"type":"immediate"}')
        inv_raw = json.loads(row.invalidation_json or "[]")
        return cls(
            sector=row.sector, ticker=row.ticker,
            direction=row.direction, position_rank=row.position_rank,
            quant_baseline_weight=row.quant_baseline_weight,
            llm_adjustment_pct=row.llm_adjustment_pct,
            suggested_weight=row.suggested_weight,
            regime_label=row.regime_label,
            tsmom_signal=row.entry_tsmom_signal or 0,
            csmom_rank=row.entry_csmom_rank or 9,
            composite_score=row.entry_composite_score or 0,
            ann_vol=row.entry_ann_vol or 0.2,
            gate_status="open",
            source_agent=row.source_agent or "quant_agent",
            confidence=row.confidence or 50,
            decision_log_id=row.decision_log_id,
            entry_condition=EntryCondition(**ec_raw),
            invalidation_conditions=[InvalidationCondition(**c) for c in inv_raw],
        )


# ── Quant Assessment ───────────────────────────────────────────────────────────

@dataclass
class QuantAssessment:
    """
    Structured output from QuantAgent for a single sector on a given date.

    Injected as context into ResearchAgent prompts and stored alongside
    TradeRecommendation for downstream attribution.
    """
    sector:          str
    ticker:          str
    as_of_date:      date

    # Signal metrics
    tsmom_signal:    int    # +1 / 0 / -1
    tsmom_raw_return: float
    csmom_rank:      int
    ann_vol:         float
    composite_score: int
    gate_status:     GateStatus

    # Regime
    regime_label:    str
    p_risk_on:       float

    # Sizing
    vol_parity_weight: float
    regime_weight_cap: float

    # Risk metrics
    atr_14:           float   # ATR(21) — field name kept for backward compat
    atr_63:           float = 0.0  # ATR(63) ≈ quarterly; preferred for stop-loss on 3-6mo positions
    price_vs_sma_200: float = 0.0

    def to_prompt_context(self) -> str:
        """Compact string for injection into LLM prompts."""
        sign = "+1" if self.tsmom_signal > 0 else ("-1" if self.tsmom_signal < 0 else "0")
        return (
            f"[QUANT] {self.sector} ({self.ticker}) | {self.as_of_date}\n"
            f"  TSMOM: {sign} (raw: {self.tsmom_raw_return:.1%}) | "
            f"CSMOM rank: {self.csmom_rank} | Ann.Vol: {self.ann_vol:.1%} | "
            f"Composite: {self.composite_score}/100 | Gate: {self.gate_status}\n"
            f"  Regime: {self.regime_label} (p_risk_on={self.p_risk_on:.2f}) | "
            f"Vol-parity wt: {self.vol_parity_weight:.1%} | Cap: {self.regime_weight_cap:.1%}\n"
            f"  ATR(21): {self.atr_14:.2f} | ATR(63): {self.atr_63:.2f} | "
            f"vs SMA200: {self.price_vs_sma_200:+.1%}"
        )

    def to_prompt_context_raw(self) -> str:
        """Raw numeric context for LLM injection — directional conclusions excluded.

        Injection rule (P0-4 hard constraint):
          ALLOWED : tsmom_raw_return, ann_vol, atr_14, atr_63,
                    price_vs_sma_200, p_risk_on, csmom_rank
          FORBIDDEN: tsmom_signal (+1/-1/0), gate_status, composite_score
        Rationale: directional conclusions anchor LLM toward quant verdict,
        destroying dual-track independence.
        """
        return (
            f"[QUANT-RAW] {self.sector} ({self.ticker}) | {self.as_of_date}\n"
            f"  12M动量原始收益: {self.tsmom_raw_return:+.2%} | "
            f"截面排名: {self.csmom_rank}/18 | 年化波动率: {self.ann_vol:.2%}\n"
            f"  制度滤波概率(p_risk_on): {self.p_risk_on:.3f} | "
            f"ATR(21): {self.atr_14:.4f} | ATR(63): {self.atr_63:.4f} | "
            f"vs SMA200: {self.price_vs_sma_200:+.2%}"
        )


# ── P3-5: Structured LLM output schemas ───────────────────────────────────────

@dataclass
class StructuredTradeOutput:
    """Canonical structured output for sector trade recommendations.
    Produced by debate.py Blue/Arbitration nodes; stored in DecisionLog."""
    direction:         str        # 超配 / 标配 / 低配 / 拦截
    confidence:        int        # 0-100
    horizon:           str        # 季度(3个月) / 半年(6个月)
    key_thesis:        str        # core investment thesis, ≤200 chars
    primary_risk:      str        # main downside risk, ≤100 chars
    macro_regime_view: str        # risk-on / neutral / risk-off
    contradicts_quant: bool = False   # True when direction opposes TSMOM signal

    @classmethod
    def from_analysis_json(cls, data: dict, tsmom_signal: int = 0) -> "StructuredTradeOutput":
        direction  = data.get("recommendation", data.get("direction", "标配"))
        confidence = int(data.get("overall_confidence", data.get("confidence", 50)))
        horizon    = data.get("horizon", "季度(3个月)")
        key_thesis = (data.get("key_thesis") or data.get("synthesis") or
                      data.get("recommendation_rationale", ""))[:200]
        primary_risk = (data.get("primary_risk") or
                        data.get("invalidation_conditions", ""))[:100]
        mrv = data.get("macro_regime_view", "neutral")
        dir_lower = direction.lower()
        _contradicts = (
            tsmom_signal != 0 and (
                (tsmom_signal > 0 and direction == "低配") or
                (tsmom_signal < 0 and direction == "超配")
            )
        )
        return cls(
            direction=direction,
            confidence=confidence,
            horizon=horizon,
            key_thesis=key_thesis,
            primary_risk=primary_risk,
            macro_regime_view=mrv,
            contradicts_quant=_contradicts,
        )


STRUCTURED_TRADE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "recommendation":           {"type": "string", "enum": ["超配", "标配", "低配", "拦截"]},
        "confidence":               {"type": "integer"},
        "horizon":                  {"type": "string", "enum": ["季度(3个月)", "半年(6个月)"]},
        "key_thesis":               {"type": "string"},
        "primary_risk":             {"type": "string"},
        "macro_regime_view":        {"type": "string", "enum": ["risk-on", "neutral", "risk-off"]},
    },
    "required": ["recommendation", "confidence", "horizon", "key_thesis", "primary_risk", "macro_regime_view"],
}


@dataclass
class StructuredMacroBrief:
    """Structured output for the daily macro brief LLM call in daily_batch.py."""
    regime_assessment: str    # risk-on / neutral / risk-off
    key_driver:        str    # main macro driver, ≤150 chars
    tail_risk:         str    # main tail risk, ≤150 chars
    brief_text:        str    # 2-3 sentence Chinese summary
    confidence:        float = 0.7  # 0-1

    @classmethod
    def from_json(cls, data: dict) -> "StructuredMacroBrief":
        return cls(
            regime_assessment=data.get("regime_assessment", "neutral"),
            key_driver=data.get("key_driver", "")[:150],
            tail_risk=data.get("tail_risk", "")[:150],
            brief_text=data.get("brief_text", ""),
            confidence=float(data.get("confidence", 0.7)),
        )


STRUCTURED_MACRO_BRIEF_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "regime_assessment": {"type": "string", "enum": ["risk-on", "neutral", "risk-off"]},
        "key_driver":        {"type": "string"},
        "tail_risk":         {"type": "string"},
        "brief_text":        {"type": "string"},
        "confidence":        {"type": "number"},
    },
    "required": ["regime_assessment", "key_driver", "tail_risk", "brief_text"],
}
