"""
Alpha Memory — Decision logging, performance verification, and self-learning.

Storage:  SQLite (dev) / PostgreSQL (prod) via SQLAlchemy
Flow:
  1. save_decision()            → log every AI analysis with structured output
  2. verify_pending()           → 20 days later, fetch yfinance returns, score accuracy
  3. get_historical_ctx()       → inject past performance + learned patterns into prompts
  4. run_meta_agent_analysis()  → identify systematic biases, update learning log
  5. update_news_routing_weight()→ adjust which news categories matter per sector × regime
  6. get_stats()                → dashboard metrics
"""
import datetime
import json
import logging
import math
import os
from pathlib import Path

import pandas as pd

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint,
    create_engine, func, text,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# ── Database setup ──────────────────────────────────────────────────────────────

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "macro_alpha_memory.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{_DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ── Re-exports from engine.db_models (2026-05-14 restore) ────────────────────
# Pages historically import these from engine.memory for one-stop convenience.
# Defined in engine.db_models (which has its own Base). Re-exported here so
# existing `from engine.memory import PortfolioNavSnapshot, ...` paths work.
# The actual table creation runs against engine.db_models.Base in conftest
# fixtures + scripts (Phase A integration test pattern).
try:
    from engine.db_models import (
        PortfolioNavSnapshot,
        SpecRegistry,
        PaperTradingRun,
        PAPER_TRADING_ARMS,
        HARKingFlag,
        MemoryCuratorReport,
        CashFlow,
        AnomalyFlag,
        AnomalyUniverseEvent,
        AgentReflection,
        PaperTradeStrategyLog,
        PaperTradeTradeLog,
        PendingApproval as _DB_PendingApproval,
    )
    # PendingApproval is also defined directly in this file; prefer the
    # in-memory version below (it's the original); db_models version is
    # imported above for AuditFinding/etc. cross-table references.
except ImportError as _re_exp_err:
    logger.warning("Cannot re-export engine.db_models classes: %s", _re_exp_err)


# ── ORM Models ─────────────────────────────────────────────────────────────────

class DecisionLog(Base):
    __tablename__ = "decision_logs"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    tab_type       = Column(String(20),  nullable=False)   # macro/sector/audit/scanner
    created_at     = Column(DateTime,    default=datetime.datetime.utcnow)
    vix_level      = Column(Float,       nullable=True)
    sector_name    = Column(String(100), nullable=True)
    ticker         = Column(String(20),  nullable=True)
    news_summary   = Column(Text,        nullable=True)
    ai_conclusion  = Column(Text,        nullable=False)
    direction      = Column(String(20),  nullable=True)    # 超配/标配/低配/拦截/通过/中性

    # ── Structured output fields (iterative learning architecture) ────────────
    confidence_score        = Column(Integer, nullable=True)   # 0-100
    horizon                 = Column(String(50), nullable=True) # e.g. "中期约1个月"
    invalidation_conditions = Column(Text, nullable=True)       # plain text
    economic_logic          = Column(Text, nullable=True)       # rationale before data
    macro_regime            = Column(String(50), nullable=True) # e.g. "加息周期"
    news_categories_used    = Column(Text, nullable=True)       # JSON list
    quant_metrics           = Column(Text, nullable=True)       # JSON
    is_backtest             = Column(Boolean, default=False)    # historical replay
    decision_date           = Column(Date,    nullable=True)    # the calendar date the decision REPRESENTS
                                                                # (= created_at.date() for live; = historical T for backtest)

    # ── Performance verification ───────────────────────────────────────────────
    verified          = Column(Boolean, default=False)
    verified_at       = Column(DateTime, nullable=True)
    actual_return_5d  = Column(Float, nullable=True)
    actual_return_10d = Column(Float, nullable=True)
    actual_return_20d = Column(Float, nullable=True)   # primary metric (~1 month)
    accuracy_score    = Column(Float, nullable=True)   # 0.0 – 1.0

    # ── XAI: signal attribution & confidence decomposition ───────────────────
    macro_confidence    = Column(Integer, nullable=True)  # 0-100
    news_confidence     = Column(Integer, nullable=True)  # 0-100
    technical_confidence= Column(Integer, nullable=True)  # 0-100
    signal_attribution  = Column(Text, nullable=True)     # JSON: {macro, news, technical weights + drivers}
    sensitivity_flag    = Column(String(10), nullable=True) # LOW / MEDIUM / HIGH
    debate_transcript   = Column(Text, nullable=True)      # JSON: full debate history + arbitration notes

    # ── Self-reflection ────────────────────────────────────────────────────────
    reflection        = Column(Text, nullable=True)
    meta_verdict      = Column(String(20), nullable=True)   # HIGH / REVIEW / WEAK
    reflection_chain  = Column(Text, nullable=True)   # full reflection node reasoning (blocked audits)

    # ── Human-in-the-loop review ───────────────────────────────────────────────
    needs_review = Column(Boolean, default=False)       # auto-flagged by verify step
    human_label  = Column(String(20), nullable=True)   # "black_swan" / "analysis_error"
    needs_retry  = Column(Boolean, default=False)       # quota failure during backtest — awaiting retry
    # P3-5: Structured trade output fields (from StructuredTradeOutput)
    key_thesis         = Column(String(300), nullable=True)  # core thesis ≤200 chars
    primary_risk       = Column(String(200), nullable=True)  # main downside risk ≤100 chars
    macro_regime_view  = Column(String(20),  nullable=True)  # risk-on / neutral / risk-off

    # ── LCS quality gate (Logical Consistency Score) ───────────────────────────
    # Populated by run_full_lcs_audit() called inside verify_pending_decisions().
    # lcs_passed=False blocks write-back to learning tables (LearningLog,
    # QuantPatternLog, NewsRoutingWeight, SkillLibrary), regardless of accuracy.
    # NULL = LCS not yet run (model unavailable during verification).
    lcs_score          = Column(Float,   nullable=True)  # weighted 0.0–1.0
    lcs_passed         = Column(Boolean, nullable=True)  # lcs_score >= 0.70
    lcs_mirror_passed  = Column(Boolean, nullable=True)  # mirror test component
    lcs_noise_passed   = Column(Boolean, nullable=True)  # noise injection component
    lcs_cross_passed   = Column(Boolean, nullable=True)  # cross-cycle anchoring component
    lcs_notes          = Column(Text,    nullable=True)  # diagnostic summary

    # ── Triple-Barrier Method diagnostics ─────────────────────────────────────
    # Populated by _compute_triple_barrier_score() inside verify_pending_decisions().
    # barrier_hit: which barrier was touched first — "tp" / "sl" / "time"
    #   tp   = take-profit (1σ move in correct direction)  → strong correct
    #   sl   = stop-loss   (0.7σ adverse move)             → clear wrong
    #   time = neither TP nor SL hit within 2× half-life   → time-decayed signal
    # hist_vol_ann: annualised vol used to scale TP/SL (from 252-day lookback)
    barrier_hit      = Column(String(8), nullable=True)
    barrier_days     = Column(Integer,   nullable=True)   # calendar days to barrier hit
    barrier_return   = Column(Float,     nullable=True)   # cumulative return at hit
    hist_vol_ann     = Column(Float,     nullable=True)   # annualised σ used for barriers

    # ── Failure Mode Classification ────────────────────────────────────────────
    # Populated by _classify_failure_mode() inside verify_pending_decisions(),
    # only when accuracy_score < 0.5 (clear prediction failure).
    # NULL = prediction passed, or classification not yet run.
    # FM-A  Logic degradation    — LCS mirror test failed; conclusion is input-independent
    # FM-B  Overconfidence       — confidence_score ≥ 85 but accuracy_score < 0.5
    # FM-C  Signal contamination — short-term technical signal (RSI/Bollinger) in drivers
    # FM-D  Regime misclassif.  — same sector × regime produced 3+ consecutive failures
    failure_mode     = Column(String(8), nullable=True)

    # ── Dynamic revision tracking ──────────────────────────────────────────────
    # When a decision is superseded by a revised decision:
    #   superseded=True on the old record, parent_decision_id on the new record.
    # This allows full revision chain reconstruction and prevents superseded records
    # from polluting performance statistics and verification queues.
    superseded          = Column(Boolean, default=False)          # this record replaced by a revision
    parent_decision_id  = Column(Integer, nullable=True)          # FK → DecisionLog.id of original
    revision_reason     = Column(Text,    nullable=True)          # what triggered the revision

    # ── Regime drift detection ─────────────────────────────────────────────────
    # Populated by verify_pending_decisions() at barrier-hit time.
    # regime_at_verify: VIX-inferred regime label at the moment the barrier was hit.
    # regime_drifted:   True when macro_regime (at decision) ≠ regime_at_verify.
    #                   Separates "wrong thesis" failures from "regime changed" failures.
    #                   NULL = verification predates this field or VIX fetch failed.
    regime_at_verify    = Column(String(50), nullable=True)
    regime_drifted      = Column(Boolean,    nullable=True)

    # ── Human-AI collaboration provenance ──────────────────────────────────────
    # Records the degree of human involvement in this decision.
    # "ai_drafted"       — AI generated; human confirmed without substantive edits
    # "human_edited"     — AI draft was materially modified by human before saving
    # "human_initiated"  — Human wrote the core thesis; AI provided structural assist
    # NULL               — Pre-provenance records (created before this field existed)
    decision_source     = Column(String(20), nullable=True)

    # ── Quant Audit metrics at decision time ───────────────────────────────────
    # Snapshot of key quantitative metrics computed when the Sector Analysis ran.
    # Stored alongside the decision to enable future empirical analysis:
    # e.g., "do decisions made with p_noise < 0.1 outperform on Triple-Barrier?"
    # NULL = decision predates quant integration or quant data was unavailable.
    quant_p_noise       = Column(Float,   nullable=True)   # optimizer-curse noise probability
    quant_val_r2        = Column(Float,   nullable=True)   # in-sample R² (temporal-split Lasso)
    quant_test_r2       = Column(Float,   nullable=True)   # out-of-sample R² (hold-out)
    quant_active        = Column(Integer, nullable=True)   # number of non-zero Lasso features
    # Soft override: LLM-proposed weight adjustment relative to quant baseline (-20 to +20pp).
    # Enables future empirical analysis: did LLM discretion add or destroy value?
    weight_adjustment_pct = Column(Float,   nullable=True)   # NULL = pre-soft-override records
    adjustment_reason     = Column(Text,    nullable=True)   # LLM's stated justification

    # ── Human-edit magnitude ───────────────────────────────────────────────────
    # Levenshtein-based ratio of how much the human changed the AI draft.
    # 0.0 = no change (ai_drafted); 1.0 = complete rewrite.
    # NULL = pre-provenance records or decision_source is not "human_edited".
    # Classified as: none (<0.05) | minor (0.05-0.25) | moderate (0.25-0.60) | substantial (>0.60)
    edit_ratio          = Column(Float,   nullable=True)

    # ── Structured failure attribution (待实现-A) ──────────────────────────────
    # Populated manually via Admin UI when accuracy_score < 0.5 (clear failure).
    # 6-category taxonomy separates "wrong thesis" from "bad data" from "regime drift" etc.
    # failure_type values:
    #   hypothesis  — research direction itself was wrong (α-generating logic invalid)
    #   data        — FRED/yfinance quality, PIT bias, coverage gap caused bad signal
    #   regime_drift— macro regime changed materially during holding period
    #   robustness  — signal fit backtest but failed out-of-sample (overfitting)
    #   evaluation  — Triple-Barrier params mis-calibrated (TP/SL thresholds too tight/loose)
    #   execution   — timing/implementation gap (e.g. signal generated but not actionable)
    # NULL = not yet attributed or prediction did not clearly fail
    failure_type        = Column(String(20), nullable=True)
    failure_note        = Column(Text,       nullable=True)   # free-text analyst commentary

    # ── Payoff quality (normalized return at barrier hit) ─────────────────────
    # payoff_quality = barrier_return / (hist_vol_ann × sqrt(holding_days / 365))
    # Captures BOTH direction and magnitude on a risk-normalized scale.
    # > 1.0 = strong win (≥1 period-σ in correct direction)
    # 0–1.0 = weak win (hit TP but barely)
    # < 0   = loss (proportional to severity relative to position risk)
    # This is the primary learning signal for SkillLibrary — payoff_quality
    # replaces directional accuracy as the optimization target.
    payoff_quality = Column(Float, nullable=True)

    # ── Signal invalidation risk (LLM self-assessment) ─────────────────────────
    # LLM's forward estimate (0-100) of the probability that the current TSMOM/CSMOM
    # signal gets invalidated within 60 days (regime flip, narrative reversal, etc.).
    # Used to gate WatchlistEntry creation and calibrate entry conviction.
    # NULL = pre-field records or LLM fallback mode (defaulted to 50 in debate.py).
    signal_invalidation_risk = Column(Integer, nullable=True)

    # ── P2-8 Prompt 版本管理 ──────────────────────────────────────────────────────
    model_version  = Column(String(50),  nullable=True)   # e.g. "claude-sonnet-4-6"
    prompt_version = Column(String(16),  nullable=True)   # SHA-256[:8] of prompt text

    # ── P2-10 实验日志哈希链 ───────────────────────────────────────────────────
    # chain_hash = SHA-256(id | created_at | ai_conclusion | prev_chain_hash)
    # NULL for records created before P2-10, or on hash error.
    chain_hash     = Column(String(64),  nullable=True)

    # ── P1-E LLM Alpha Attribution ─────────────────────────────────────────────
    # llm_weight_alpha = actual_return × (main_weight - quant_weight)
    # Measures incremental return contributed by LLM weight adjustment over the
    # pure TSMOM quant baseline. Positive = LLM added value; negative = LLM detracted.
    # NULL = quant track unavailable (pre-P0-3 decisions) or no position found.
    llm_weight_alpha = Column(Float, nullable=True)

    # ── ORM/DB schema-drift restore (2026-05-14) ─────────────────────────────
    # These 10 columns exist in the live DB schema (added via prior migrations)
    # but were missing from this ORM class — causing AttributeError when pages
    # like approval_analytics / reflection / decision_context referenced them.
    # Re-added as nullable to match DB. No migration needed (already present).
    active_return              = Column(Float,       nullable=True)
    exit_reason                = Column(String(30),  nullable=True)
    mae                        = Column(Float,       nullable=True)
    mfe                        = Column(Float,       nullable=True)
    reflections_injected_count = Column(Integer,     nullable=True)
    reflections_injected_ids   = Column(Text,        nullable=True)
    sleeve_id                  = Column(String(20),  nullable=True)
    spec_hash                  = Column(String(64),  nullable=True)
    weight_after               = Column(Float,       nullable=True)
    weight_before              = Column(Float,       nullable=True)


class LearningLog(Base):
    """
    Meta-Agent findings: systematic biases and prompt improvement suggestions.
    Each row represents one discovered pattern from analysing failure cases.
    """
    __tablename__ = "learning_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)
    macro_regime = Column(String(50), nullable=True)   # which regime triggered this
    sector_name  = Column(String(100), nullable=True)  # None = applies to all sectors
    pattern_type = Column(String(50), nullable=False)  # bias / prompt_fix / routing
    description  = Column(Text, nullable=False)        # human-readable finding
    sample_count = Column(Integer, default=0)          # how many cases support this
    accuracy_impact = Column(Float, nullable=True)     # estimated accuracy delta
    dormant      = Column(Boolean, default=False)      # regime unseen for 180+ days → silenced but kept
    applied      = Column(Boolean, default=False)      # resolved by new contradicting evidence
    applied_at   = Column(DateTime, nullable=True)


class NewsRoutingWeight(Base):
    """
    Learned weights for which news categories to prioritise
    per (sector × macro_regime) combination.
    Updated by Meta-Agent after each learning cycle.
    """
    __tablename__ = "news_routing_weights"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    sector_name  = Column(String(100), nullable=False)
    macro_regime = Column(String(50),  nullable=False)
    news_category = Column(String(100), nullable=False)  # e.g. "央行声明", "供应链"
    weight       = Column(Float, default=0.5)            # 0.0 – 1.0
    sample_count = Column(Integer, default=0)
    updated_at   = Column(DateTime, default=datetime.datetime.utcnow)


class SpilloverWeight(Base):
    """
    Learned cross-sector transmission coefficients.
    Computed from Pearson correlation of accuracy_scores between sector pairs
    within the same macro_regime. Only activated when sample_count >= threshold.
    Falls back to SPILLOVER_MAP priors when data is insufficient.
    """
    __tablename__ = "spillover_weights"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    source_sector  = Column(String(100), nullable=False)
    target_sector  = Column(String(100), nullable=False)
    macro_regime   = Column(String(50),  nullable=False)
    correlation    = Column(Float, nullable=False)   # Pearson r, -1.0 to 1.0
    conflicts_prior = Column(Boolean, default=False) # True when sign contradicts SPILLOVER_MAP prior
    sample_count   = Column(Integer, default=0)
    updated_at     = Column(DateTime, default=datetime.datetime.utcnow)


class BacktestSession(Base):
    """
    Tracks an in-progress or paused backtest run.
    Persisted so the resume banner survives app restarts.
    status: 'running' | 'paused' | 'completed'
    """
    __tablename__ = "backtest_sessions"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    start_date  = Column(String(10), nullable=False)
    end_date    = Column(String(10), nullable=False)
    sectors     = Column(Text, nullable=False)   # JSON list
    freq        = Column(String(4), default="QS")
    total_pairs = Column(Integer, default=0)
    done_pairs  = Column(Integer, default=0)
    status      = Column(String(12), default="running")  # running/paused/quota_hit/completed
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.datetime.utcnow)


class SkillLibrary(Base):
    """
    Compressed behavioral instructions distilled from accumulated experience.
    Each row is a sector × macro_regime skill — a short, actionable rule the
    agent should follow, generated by LLM compression of LearningLog + benchmarks.

    Replaces verbose raw-history injection with dense, token-efficient guidance.
    Updated whenever new verified decisions push sample_count past a threshold.
    """
    __tablename__ = "skill_library"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    sector_name  = Column(String(100), nullable=False)
    macro_regime = Column(String(50),  nullable=False)
    skill_text   = Column(Text, nullable=False)      # compressed behavioral instruction
    version      = Column(Integer, default=1)        # increments on each recompression
    sample_count = Column(Integer, default=0)        # evidence base size
    avg_accuracy = Column(Float, nullable=True)      # directional accuracy (legacy, kept for compat)
    avg_payoff_quality = Column(Float, nullable=True)  # mean payoff_quality across evidence base
    created_at   = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.datetime.utcnow)


class StressTestLog(Base):
    """
    Lightweight record of each stress-test scenario run.
    Results are ephemeral in session_state; this table persists
    the key parameters and AI direction for retrospective review.
    Does NOT store the full AI output — only a short summary.
    """
    __tablename__ = "stress_test_log"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    run_at            = Column(DateTime, default=datetime.datetime.utcnow)
    scenario_id       = Column(String(50),  nullable=False)
    scenario_name     = Column(String(200), nullable=False)
    scenario_category = Column(String(100), nullable=True)
    sector            = Column(String(100), nullable=False)
    effective_vix     = Column(Float,       nullable=True)
    fed_funds_delta   = Column(Integer,     nullable=True)   # bps
    oil_delta_pct     = Column(Float,       nullable=True)
    usd_delta_pct     = Column(Float,       nullable=True)
    ai_direction      = Column(String(20),  nullable=True)   # 超配/标配/低配
    ai_summary        = Column(Text,        nullable=True)   # ≤ 300 chars
    custom_note       = Column(Text,        nullable=True)


class SystemConfig(Base):
    """
    Generic key-value store for persistent system-level settings.
    Keys are strings (e.g. "risk.vol_target_ann"), values stored as TEXT.
    """
    __tablename__ = "system_config"

    key        = Column(String(100), primary_key=True)
    value      = Column(Text,        nullable=False)
    updated_at = Column(DateTime,    default=datetime.datetime.utcnow,
                        onupdate=datetime.datetime.utcnow)


class QuantPatternLog(Base):
    """
    Conditional hit-rate table: tracks AI decision accuracy
    broken down by quantitative state vector × macro regime × direction.

    Serves as an empirical prior that is injected into future prompts:
    "在 动量:下降|RSI:超买|波动:高 × 高波动收缩 的历史记录中，
     超配判断胜率 31%（n=13）— 建议降低置信度。"

    Populated by: _update_quant_pattern() called after each verify_pending_decisions()
    Queried by:   get_quant_pattern_context() called in backtest/live analysis prompts

    Data leakage protection:
      - Only records with verified_at < cutoff_date are counted (same mechanism as
        get_historical_context). Future data never contaminates historical backtests.
      - Raw decision IDs tracked in decision_ids for auditability.
    """
    __tablename__ = "quant_pattern_log"

    id                = Column(Integer,     primary_key=True, autoincrement=True)
    state_fingerprint = Column(String(200), nullable=False)
    macro_regime      = Column(String(80),  nullable=False)
    direction         = Column(String(20),  nullable=False)   # 超配/标配/低配
    total_count       = Column(Integer,     default=0)
    correct_count     = Column(Integer,     default=0)        # accuracy_score >= 0.75
    partial_count     = Column(Integer,     default=0)        # accuracy_score == 0.5
    accuracy_rate     = Column(Float,       default=0.0)      # correct_count / total_count
    avg_accuracy      = Column(Float,       default=0.0)      # mean of all accuracy_scores
    last_updated      = Column(DateTime,    default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("state_fingerprint", "macro_regime", "direction",
                         name="uq_quant_pattern"),
    )


# ── Macro Watchlist ───────────────────────────────────────────────────────────

class MacroWatchItem(Base):
    """
    Persistent forward-monitoring item extracted from the macro analysis §6 output.

    Each macro analysis generates a "future 5-day monitoring list". Instead of
    letting that list disappear with the session, we parse it into rows here.
    The next macro analysis query loads unresolved items and injects them as
    prior context — creating a residual connection across analysis cycles.

    outcome values:
      "matched"    — actual outcome broadly aligned with expectation
      "surprised"  — material deviation from expectation (worth recording as experience)
      "expired"    — check_by date passed, not manually resolved
      "n/a"        — no verifiable expected value (qualitative signal)
    """
    __tablename__ = "macro_watch_items"

    id             = Column(Integer,   primary_key=True, autoincrement=True)
    created_at     = Column(DateTime,  default=datetime.datetime.utcnow)
    analysis_date  = Column(Date,      nullable=False)   # date of generating analysis
    check_by       = Column(Date,      nullable=False)   # expected check date (~5 trading days)
    item_text      = Column(Text,      nullable=False)   # raw watch item text
    category       = Column(String(30), nullable=True)   # data_release / key_level / market_signal
    expected_value = Column(String(200), nullable=True)  # e.g. "CPI 预期 3.2%"
    actual_value   = Column(String(200), nullable=True)  # filled in on resolution
    resolved       = Column(Boolean,   default=False)
    resolved_at    = Column(DateTime,  nullable=True)
    outcome        = Column(String(20), nullable=True)   # matched / surprised / expired / n/a
    macro_regime   = Column(String(50), nullable=True)
    notes          = Column(Text,      nullable=True)    # analyst commentary on resolution


# ── Simulated Execution Layer ──────────────────────────────────────────────────

class SimulatedPosition(Base):
    """
    Each month-end rebalancing snapshot after target weights have been applied.
    This is the authoritative state for the NEXT rebalancing cycle — the
    difference between this and the new target weights drives trade generation.

    Paper-trading fields (shares_held, cost_basis, position_value) are
    populated by _auto_link_position() when a sector analysis is saved.
    They allow the Live Dashboard to show absolute P&L vs. pure weight %.
    """
    __tablename__ = "simulated_positions"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date,        nullable=False)    # rebalance execution date
    sector        = Column(String(50),  nullable=False)
    ticker        = Column(String(20),  nullable=False)
    target_weight = Column(Float,       nullable=False)    # signal-suggested weight
    actual_weight = Column(Float,       nullable=True)     # weight after threshold filter
    entry_price   = Column(Float,       nullable=True)     # closing price at rebalance
    regime_label  = Column(String(20),  nullable=True)
    signal_tsmom  = Column(Integer,     nullable=True)     # +1 / -1 / 0
    notes         = Column(Text,        nullable=True)
    # Paper-trading absolute fields (NULL = weight-only mode, no NAV set)
    shares_held    = Column(Float,       nullable=True)     # simulated share count
    cost_basis     = Column(Float,       nullable=True)     # total cost in base currency
    position_value = Column(Float,       nullable=True)     # market value at entry
    direction      = Column(String(20),  nullable=True)     # 超配 / 标配 / 低配
    # Trailing stop support: updated daily to max(trailing_high, current_close)
    # Stop price = trailing_high - 2 × ATR(21)
    trailing_high  = Column(Float,       nullable=True)     # highest close since entry
    # P0-3 attribution: "main" = LLM-adjusted track; "quant" = pure signal baseline
    track          = Column(String(10),  nullable=False, default="main", server_default="main")

    __table_args__ = (
        UniqueConstraint("snapshot_date", "sector", "track", name="uq_pos_date_sector_track"),
    )


class SimulatedTrade(Base):
    """
    Individual trade instruction generated by each rebalancing cycle.
    delta = target_weight − current_weight → BUY / SELL / HOLD
    trigger_reason priority: signal_flip > regime_change > rebalance > threshold
    """
    __tablename__ = "simulated_trades"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    trade_date     = Column(Date,        nullable=False)
    sector         = Column(String(50),  nullable=False)
    ticker         = Column(String(20),  nullable=False)
    action         = Column(String(10),  nullable=False)   # BUY / SELL / HOLD
    weight_before  = Column(Float,       nullable=False)
    weight_after   = Column(Float,       nullable=False)
    weight_delta   = Column(Float,       nullable=False)   # signed
    cost_bps       = Column(Float,       nullable=True)    # estimated one-way cost
    trigger_reason = Column(String(50),  nullable=True)    # signal_flip / regime_change / rebalance / threshold
    # Share-level execution fields (populated when NAV and price are known)
    shares         = Column(Float,       nullable=True)    # number of shares traded (abs)
    fill_price     = Column(Float,       nullable=True)    # execution price per share
    notional       = Column(Float,       nullable=True)    # shares × fill_price in base currency


class SimulatedMonthlyReturn(Base):
    """
    Position-level return attribution for each month.
    Populated by record_monthly_return() at the start of the following month.
    Used for forward-test vs historical-backtest comparison.
    """
    __tablename__ = "simulated_monthly_returns"

    id           = Column(Integer,  primary_key=True, autoincrement=True)
    return_month = Column(Date,     nullable=False)    # first day of the measured month
    sector       = Column(String(50), nullable=False)
    weight_held  = Column(Float,    nullable=False)    # actual_weight from SimulatedPosition
    sector_return = Column(Float,   nullable=True)     # ETF price return that month
    contribution = Column(Float,    nullable=True)     # weight_held × sector_return
    regime_label = Column(String(20), nullable=True)
    is_profitable = Column(Boolean, nullable=True)     # contribution > 0

    __table_args__ = (
        UniqueConstraint("return_month", "sector", name="uq_ret_month_sector"),
    )


# ── Snapshot cache tables ─────────────────────────────────────────────────────

class RegimeSnapshot(Base):
    """
    Cached regime computation result for a given date.
    Avoids re-fetching FRED data and re-fitting MSM on every page load.
    Walk-forward integrity: train_end is stored so callers can verify
    the cache was produced with the correct data cutoff.
    """
    __tablename__ = "regime_snapshots"

    id           = Column(Integer,     primary_key=True, autoincrement=True)
    as_of_date   = Column(Date,        nullable=False)
    train_end    = Column(Date,        nullable=False)
    regime       = Column(String(20),  nullable=False)   # risk-on / risk-off / transition
    p_risk_on    = Column(Float,       nullable=False)
    p_risk_off   = Column(Float,       nullable=False)
    method       = Column(String(30),  nullable=True)    # msm / rule-based / msm-fallback
    n_obs        = Column(Integer,     nullable=True)
    yield_spread = Column(Float,       nullable=True)
    vix          = Column(Float,       nullable=True)
    warning      = Column(Text,        nullable=True)
    computed_at  = Column(DateTime,    default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("as_of_date", "train_end", name="uq_regime_snap"),
    )


class SignalSnapshot(Base):
    """
    Cached signal computation result for a given date + parameter set.
    Stores the full 18-sector signal DataFrame as JSON so downstream
    consumers (portfolio, dashboard) avoid redundant yfinance fetches.
    """
    __tablename__ = "signal_snapshots"

    id              = Column(Integer,  primary_key=True, autoincrement=True)
    as_of_date      = Column(Date,     nullable=False)
    lookback_months = Column(Integer,  nullable=False, default=12)
    skip_months     = Column(Integer,  nullable=False, default=1)
    signals_json    = Column(Text,     nullable=False)  # DataFrame.to_json(orient="split")
    sector_count    = Column(Integer,  nullable=True)
    computed_at     = Column(DateTime, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("as_of_date", "lookback_months", "skip_months",
                         name="uq_signal_snap"),
    )


# ── P6: Per-ticker signal audit tables ────────────────────────────────────────

class SignalRecord(Base):
    """
    Per-ticker per-day signal log.  Unlike SignalSnapshot (JSON blob cache),
    each row = one ticker × one date, enabling flip detection and decay patrol.
    Written by _write_signal_record() after signal computation completes.
    """
    __tablename__ = "signal_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    date            = Column(Date,    nullable=False, index=True)
    ticker          = Column(String(20), nullable=False)
    sector          = Column(String(50), nullable=False)
    tsmom_signal    = Column(Integer, nullable=True)        # +1 / 0 / -1
    tsmom_raw       = Column(Float,   nullable=True)        # raw_return (continuous)
    csmom_rank      = Column(Float,   nullable=True)        # within-class percentile 0-1
    carry_norm      = Column(Float,   nullable=True)        # normalised carry; 0 for undefined classes
    reversal_norm   = Column(Float,   nullable=True)        # normalised reversal; 0 outside transition
    factormad_score = Column(Float,   nullable=True)        # FactorMAD composite score
    composite_score = Column(Float,   nullable=True)        # final composite 0-100
    gate_status     = Column(String(10), nullable=True)     # "passed" / "blocked"
    regime_at_calc  = Column(String(20), nullable=True)     # risk-on / transition / risk-off

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_signal_record_date_ticker"),
    )


class SignalFlipLog(Base):
    """
    Recorded whenever a ticker's TSMOM signal changes direction.
    Written by _detect_signal_flips() by comparing today vs. previous SignalRecord.
    Used in Daily Brief to surface notable momentum reversals.
    """
    __tablename__ = "signal_flip_log"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    date            = Column(Date,    nullable=False, index=True)
    ticker          = Column(String(20), nullable=False)
    sector          = Column(String(50), nullable=False)
    prev_signal     = Column(Integer, nullable=True)
    new_signal      = Column(Integer, nullable=True)
    tsmom_raw_prev  = Column(Float,   nullable=True)
    tsmom_raw_new   = Column(Float,   nullable=True)
    regime_at_flip  = Column(String(20), nullable=True)


class DataQualityLog(Base):
    """
    Data freshness / quality check results per trading day.
    Written by _step1_data_quality() at the start of each daily batch.
    Feeds the data-quality indicator in the Daily Brief engineering panel.
    """
    __tablename__ = "data_quality_log"

    id          = Column(Integer,  primary_key=True, autoincrement=True)
    date        = Column(Date,     nullable=False, index=True)
    check_type  = Column(String(50), nullable=False)   # ohlcv_freshness / fred_delay / overall
    status      = Column(String(10), nullable=False)   # ok / warning / light
    detail      = Column(Text,     nullable=True)
    checked_at  = Column(DateTime, default=datetime.datetime.utcnow)


class CircuitBreakerLog(Base):
    """
    Persistent audit log for all Circuit Breaker events.
    Complements engine/state/circuit_breaker.json (which only holds current state).
    Allows historical review of how often and why the CB fired.
    """
    __tablename__ = "circuit_breaker_log"

    id            = Column(Integer,  primary_key=True, autoincrement=True)
    triggered_at  = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    level         = Column(String(10), nullable=False)   # light / medium / severe
    reason        = Column(Text,     nullable=True)
    auto_resolved = Column(Boolean,  default=False)
    resolved_at   = Column(DateTime, nullable=True)
    resolved_by   = Column(String(100), nullable=True)   # "auto" / username
    notes         = Column(Text,     nullable=True)


class AlphaMemory(Base):
    """
    Combined record for Track B decisions and ERA (External Reality Audit) results.

    Track B writes sector_delta + logic_chain + confidence at decision time.
    ERA fills era_verdict + era_score + macro_data_snapshot at verification time
    (same trigger as Triple-Barrier, decoupled from decision model).

    era_verdict:
      "logic_correct" — thesis predicted macro direction and macro confirmed it
      "lucky_guess"   — outcome correct but specific mechanism was wrong
      "logic_wrong"   — thesis direction contradicted by actual macro data
    """
    __tablename__ = "alpha_memory"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    decision_date       = Column(Date,    nullable=False)
    sector              = Column(String(50), nullable=False)
    source              = Column(String(20), nullable=False, default="track_b")
    # Track B fields
    quant_weight        = Column(Float,   nullable=True)
    llm_delta           = Column(Float,   nullable=True)   # signed weight adjustment
    adjusted_weight     = Column(Float,   nullable=True)
    logic_chain         = Column(Text,    nullable=True)   # LLM's stated thesis
    confidence          = Column(Integer, nullable=True)   # 0-100
    # ERA verification fields (filled later)
    era_verdict         = Column(String(20), nullable=True)
    era_score           = Column(Float,   nullable=True)   # 0-1
    era_reasoning       = Column(Text,    nullable=True)
    macro_data_snapshot = Column(Text,    nullable=True)   # JSON: FRED values during verification
    verified_at         = Column(DateTime, nullable=True)
    created_at          = Column(DateTime, default=datetime.datetime.utcnow)


class QuantOnlySnapshot(Base):
    """
    Daily NAV snapshot of the pure-quant portfolio (no Track B adjustments).
    Used to measure Track B's incremental alpha: alpha = primary_NAV - quant_only_NAV.
    Display shows cumulative excess return only; IR requires n≥30 to be meaningful.
    """
    __tablename__ = "quant_only_snapshots"

    date          = Column(Date,  primary_key=True)
    nav           = Column(Float, nullable=False)
    daily_return  = Column(Float, nullable=True)
    weights_json  = Column(Text,  nullable=True)   # JSON: {sector: weight}


# ── P2-18 TradingCycleOrchestrator state table ────────────────────────────────

class CycleState(Base):
    """
    Persists one record per orchestrated cycle run.

    cycle_type : daily | weekly | monthly | verification
    status     : running | completed | failed | pending_gate | approved | rejected
    gate       : which human gate is awaiting approval (NULL = none)
    """
    __tablename__ = "cycle_states"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    cycle_type     = Column(String(20),  nullable=False)
    as_of_date     = Column(Date,        nullable=False)
    status         = Column(String(20),  nullable=False, default="running")
    gate           = Column(String(30),  nullable=True)   # human gate label
    started_at     = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)
    finished_at    = Column(DateTime,    nullable=True)
    elapsed_s      = Column(Float,       nullable=True)
    error_log      = Column(Text,        nullable=True)
    result_summary = Column(Text,        nullable=True)   # JSON-serialised ChainResult summary


# ── Trading Agent tables ──────────────────────────────────────────────────────

class WatchlistEntry(Base):
    """
    State machine for a single sector position recommendation.

    Lifecycle: watching → triggered → active → exited
                        ↘ invalidated (at any pre-active stage)

    watching  : analysis created, waiting for entry condition
    triggered : entry condition met by DailyBatchJob (awaiting human approval)
    active    : human approved in Trading Desk → position is live
    exited    : position closed
    invalidated: invalidation condition fired before becoming active

    Created by QuantAgent or ResearchAgent via TradeRecommendation.to_watchlist_dict().
    Transitions driven by DailyBatchJob each trading day.
    """
    __tablename__ = "watchlist_entries"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    created_date    = Column(Date,        nullable=False)
    status          = Column(String(20),  nullable=False, default="watching")
    # watching / triggered / active / invalidated / exited / corr_blocked

    # Identity
    sector          = Column(String(50),  nullable=False)
    ticker          = Column(String(20),  nullable=False)
    direction       = Column(String(10),  nullable=False)    # long / short / neutral
    position_rank   = Column(String(15),  nullable=False)    # core / satellite / tactical

    # Sizing
    quant_baseline_weight = Column(Float, nullable=False)
    llm_adjustment_pct    = Column(Float, nullable=False, default=0.0)
    suggested_weight      = Column(Float, nullable=False)

    # Signal state at creation
    regime_label           = Column(String(20),  nullable=True)
    entry_tsmom_signal     = Column(Integer,      nullable=True)   # +1 / -1 / 0
    entry_csmom_rank       = Column(Integer,      nullable=True)
    entry_composite_score  = Column(Integer,      nullable=True)
    entry_ann_vol          = Column(Float,        nullable=True)

    # Provenance
    source_agent      = Column(String(30),  nullable=True)    # quant_agent / research_agent
    decision_log_id   = Column(Integer,     nullable=True)    # FK → DecisionLog (nullable)
    confidence        = Column(Integer,     nullable=True)

    # State machine config (stored as JSON)
    entry_condition_json  = Column(Text, nullable=True)   # EntryCondition dict
    invalidation_json     = Column(Text, nullable=True)   # list[InvalidationCondition]
    risk_conditions_json  = Column(Text, nullable=True)   # list[RiskCondition]

    # Transition timestamps and context
    triggered_date    = Column(Date,        nullable=True)    # entry condition met
    triggered_price   = Column(Float,       nullable=True)    # close price at trigger
    invalidated_date  = Column(Date,        nullable=True)
    invalidated_reason = Column(Text,       nullable=True)
    exited_date       = Column(Date,        nullable=True)
    exit_price        = Column(Float,       nullable=True)

    # Human override log
    human_override_note = Column(Text,      nullable=True)

    __table_args__ = (
        UniqueConstraint("sector", "created_date", "source_agent",
                         name="uq_watchlist_sector_date_agent"),
    )


class PendingApproval(Base):
    """
    Human-in-the-loop approval queue.

    Generated by DailyBatchJob when:
      - A WatchlistEntry entry_condition is triggered  (type=entry)
      - A risk patrol check fires                      (type=risk_control)
      - Month-end rebalance drift exceeds threshold    (type=rebalance)

    Approval transitions the linked WatchlistEntry to 'active'.
    Rejection on entry reverts to 'watching'.
    Rejection on risk_control logs a RiskOverrideLog note.
    """
    __tablename__ = "pending_approvals"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    created_at       = Column(DateTime, default=datetime.datetime.utcnow)
    approval_type    = Column(String(20), nullable=False)   # entry / risk_control / rebalance
    priority         = Column(String(10), nullable=False, default="normal")  # critical / normal

    # Linked watchlist entry (nullable for rebalance / risk_control without pool entry)
    watchlist_entry_id = Column(Integer, nullable=True)     # FK → WatchlistEntry.id

    # Trigger context
    sector             = Column(String(50),  nullable=False)
    ticker             = Column(String(20),  nullable=False)
    triggered_condition = Column(Text,       nullable=True)  # human-readable reason
    triggered_date     = Column(Date,        nullable=False)
    triggered_price    = Column(Float,       nullable=True)
    suggested_weight   = Column(Float,       nullable=True)
    position_rank      = Column(String(15),  nullable=True)

    # Approval deadline: entry approvals expire after 3 trading days
    approval_deadline  = Column(Date,        nullable=True)

    # Resolution
    status             = Column(String(20),  nullable=False, default="pending")
    # pending / approved / rejected / expired
    resolved_at        = Column(DateTime,    nullable=True)
    resolved_by        = Column(String(30),  nullable=True)  # "human" / "auto"
    rejection_reason   = Column(Text,        nullable=True)

    # For risk_control rejections: log the override reason
    risk_override_note = Column(Text,        nullable=True)

    # P3-12: LLM/Quant disagreement flag (True when LLM direction contradicts TSMOM signal)
    contradicts_quant  = Column(Boolean,     nullable=True, default=False)
    llm_confidence     = Column(Integer,     nullable=True)   # 0-100, copied from WatchlistEntry

    # ── P-AUDIT v1 audit columns (2026-05-04, restored 2026-05-14) ───────────
    # These exist in the live DB (migration applied) and on the canonical
    # engine.db_models.PendingApproval. Mirroring here lets pages/UIs that
    # historically `from engine.memory import PendingApproval` access the full
    # column set. Schema reference: docs/spec_supervisor_approval_panel_v1.md.
    review_rationale          = Column(Text,        nullable=True)
    review_category           = Column(String(32),  nullable=True)
    review_narrative_snapshot = Column(Text,        nullable=True)
    review_narrative_hash     = Column(String(64),  nullable=True)
    prev_narrative_hash       = Column(String(64),  nullable=True)
    approval_class            = Column(String(16),  nullable=False,
                                       default="governance",
                                       server_default=text("'governance'"))
    approval_latency_seconds  = Column(Integer,     nullable=True)
    rejection_category        = Column(String(32),  nullable=True)
    post_hoc_note             = Column(Text,        nullable=True)
    condition_signature       = Column(String(120), nullable=True)
    last_seen_at              = Column(DateTime,    nullable=True)
    consecutive_days_seen     = Column(Integer,     nullable=True, default=1,
                                       server_default=text("1"))


# ── DB init + migration ────────────────────────────────────────────────────────

# Idempotency guard (Phase 1 perf 2026-05-15 evening): init_db() previously ran
# Base.metadata.create_all + _migrate_db on EVERY call. Across 7 page files +
# app.py = up to 8 redundant runs per cold start. The work is mostly no-op
# (CREATE IF NOT EXISTS) but still does metadata reflection + ALTER TABLE
# probing in _migrate_db. Guard makes repeat calls cheap (<1µs return).
# Force re-init: init_db(force=True) for test fixtures.
_INIT_DB_DONE = False


def init_db(force: bool = False) -> None:
    """Initialize all DB tables and run schema migrations.

    Idempotent: repeat calls within same Python process are O(1) no-op
    unless force=True is passed (e.g., from test fixtures with fresh DB).

    Creates tables from BOTH engine.memory.Base AND engine.db_models.Base
    (two separate declarative_base instances; both must run create_all).
    """
    global _INIT_DB_DONE
    if _INIT_DB_DONE and not force:
        return

    # Create tables from engine.memory.Base (this module's declarative base).
    Base.metadata.create_all(engine)
    # ALSO create tables from engine.db_models.Base — separate declarative_base
    # instance; auto_audit_models / db_models tables not covered by memory.Base.
    # Fix 2026-05-15: cover both bases here so test teardown and prod bootstrap
    # both work without callers needing to know about the split.
    try:
        from engine.db_models import Base as _DBMBase
        _DBMBase.metadata.create_all(engine)
    except Exception as _exc:
        # Best-effort: log but don't fail init_db on partial schema
        logger.warning("db_models.Base.metadata.create_all failed: %s", _exc)
    _migrate_db()
    _INIT_DB_DONE = True


def _migrate_db() -> None:
    """
    Add any columns / tables introduced after the initial schema.
    Safe to run on every startup — skips columns that already exist.
    """
    new_columns = [
        ("confidence_score",        "INTEGER"),
        ("horizon",                 "VARCHAR(50)"),
        ("invalidation_conditions", "TEXT"),
        ("economic_logic",          "TEXT"),
        ("macro_regime",            "VARCHAR(50)"),
        ("news_categories_used",    "TEXT"),
        ("is_backtest",             "BOOLEAN DEFAULT 0"),
        ("actual_return_20d",       "FLOAT"),
        ("meta_verdict",            "VARCHAR(20)"),
        ("macro_confidence",        "INTEGER"),
        ("news_confidence",         "INTEGER"),
        ("technical_confidence",    "INTEGER"),
        ("signal_attribution",      "TEXT"),
        ("sensitivity_flag",        "VARCHAR(10)"),
        ("debate_transcript",       "TEXT"),
        ("decision_date",           "DATE"),
        ("needs_review",            "BOOLEAN DEFAULT 0"),
        ("human_label",             "VARCHAR(20)"),
    ]
    with engine.connect() as conn:
        # Fetch existing columns in decision_logs
        existing = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(decision_logs)")).fetchall()
        }
        for col_name, col_type in new_columns:
            if col_name not in existing:
                conn.execute(
                    text(f"ALTER TABLE decision_logs ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column decision_logs.%s", col_name)
        conn.commit()

    # Ensure new tables exist (SpilloverWeight, etc.)
    Base.metadata.create_all(engine)

    # Migrate learning_log: add columns introduced after initial schema
    ll_new_columns = [
        ("dormant", "BOOLEAN DEFAULT 0"),
    ]

    # Migrate decision_logs: add reflection_chain + LCS + Triple-Barrier columns
    dl_extra_columns = [
        ("reflection_chain",   "TEXT"),
        ("needs_retry",        "BOOLEAN DEFAULT 0"),
        ("lcs_score",          "FLOAT"),
        ("lcs_passed",         "BOOLEAN"),
        ("lcs_mirror_passed",  "BOOLEAN"),
        ("lcs_noise_passed",   "BOOLEAN"),
        ("lcs_cross_passed",   "BOOLEAN"),
        ("lcs_notes",          "TEXT"),
        ("barrier_hit",        "VARCHAR(8)"),
        ("barrier_days",       "INTEGER"),
        ("barrier_return",     "FLOAT"),
        ("hist_vol_ann",       "FLOAT"),
        ("failure_mode",       "VARCHAR(8)"),
        ("superseded",         "BOOLEAN DEFAULT 0"),
        ("parent_decision_id", "INTEGER"),
        ("revision_reason",    "TEXT"),
        ("regime_at_verify",   "VARCHAR(50)"),
        ("regime_drifted",     "BOOLEAN"),
        ("decision_source",    "VARCHAR(20)"),
        ("quant_p_noise",      "FLOAT"),
        ("quant_val_r2",       "FLOAT"),
        ("quant_test_r2",      "FLOAT"),
        ("quant_active",             "INTEGER"),
        ("weight_adjustment_pct",    "FLOAT"),
        ("adjustment_reason",        "TEXT"),
        ("edit_ratio",               "FLOAT"),
    ]
    # Migrate simulated_positions: paper-trading columns + P0-3 attribution track
    sp_extra_columns = [
        ("shares_held",    "FLOAT"),
        ("cost_basis",     "FLOAT"),
        ("position_value", "FLOAT"),
        ("direction",      "VARCHAR(20)"),
        ("trailing_high",  "FLOAT"),
        ("track",          "VARCHAR(10) NOT NULL DEFAULT 'main'"),
    ]
    with engine.connect() as conn:
        existing_sp = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(simulated_positions)")).fetchall()
        }
        for col_name, col_type in sp_extra_columns:
            if col_name not in existing_sp:
                conn.execute(
                    text(f"ALTER TABLE simulated_positions ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column simulated_positions.%s", col_name)
        # Drop old unique constraint index and recreate with track included.
        # SQLite stores named constraints as indexes; old name = uq_pos_date_sector.
        existing_indexes = {
            row[1]
            for row in conn.execute(text("PRAGMA index_list(simulated_positions)")).fetchall()
        }
        if "uq_pos_date_sector" in existing_indexes:
            conn.execute(text("DROP INDEX IF EXISTS uq_pos_date_sector"))
            logger.info("Migration: dropped old index uq_pos_date_sector")
        if "uq_pos_date_sector_track" not in existing_indexes:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_pos_date_sector_track "
                "ON simulated_positions (snapshot_date, sector, track)"
            ))
            logger.info("Migration: created index uq_pos_date_sector_track")
        conn.commit()
    dl_extra_columns = [
        ("failure_type",            "VARCHAR(20)"),
        ("failure_note",            "TEXT"),
        ("payoff_quality",          "FLOAT"),
        ("signal_invalidation_risk","INTEGER"),
        ("llm_weight_alpha",        "FLOAT"),
        ("model_version",           "VARCHAR(50)"),
        ("prompt_version",          "VARCHAR(16)"),
        ("chain_hash",              "VARCHAR(64)"),
    ]
    with engine.connect() as conn:
        existing_dl = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(decision_logs)")).fetchall()
        }
        for col_name, col_type in dl_extra_columns:
            if col_name not in existing_dl:
                conn.execute(
                    text(f"ALTER TABLE decision_logs ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column decision_logs.%s", col_name)
        conn.commit()
    with engine.connect() as conn:
        existing_ll = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(learning_log)")).fetchall()
        }
        for col_name, col_type in ll_new_columns:
            if col_name not in existing_ll:
                conn.execute(
                    text(f"ALTER TABLE learning_log ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column learning_log.%s", col_name)
        conn.commit()

    # Migrate skill_library: add avg_payoff_quality
    sl_new_columns = [
        ("avg_payoff_quality", "FLOAT"),
    ]
    with engine.connect() as conn:
        existing_sl = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(skill_library)")).fetchall()
        }
        for col_name, col_type in sl_new_columns:
            if col_name not in existing_sl:
                conn.execute(
                    text(f"ALTER TABLE skill_library ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column skill_library.%s", col_name)
        conn.commit()

    # Migrate simulated_trades: share-level execution fields
    st_new_columns = [
        ("shares",      "FLOAT"),
        ("fill_price",  "FLOAT"),
        ("notional",    "FLOAT"),
    ]
    with engine.connect() as conn:
        existing_st = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(simulated_trades)")).fetchall()
        }
        for col_name, col_type in st_new_columns:
            if col_name not in existing_st:
                conn.execute(
                    text(f"ALTER TABLE simulated_trades ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column simulated_trades.%s", col_name)
        conn.commit()

    # Migrate watchlist_entries: P1-3 risk_conditions
    with engine.connect() as conn:
        existing_wl = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(watchlist_entries)")).fetchall()
        }
        if "risk_conditions_json" not in existing_wl:
            conn.execute(text("ALTER TABLE watchlist_entries ADD COLUMN risk_conditions_json TEXT"))
            logger.info("Migration: added column watchlist_entries.risk_conditions_json")
        conn.commit()

    # P2-18: Create cycle_states table if not present
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cycle_states (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_type     VARCHAR(20) NOT NULL,
                as_of_date     DATE        NOT NULL,
                status         VARCHAR(20) NOT NULL DEFAULT 'running',
                gate           VARCHAR(30),
                started_at     DATETIME    NOT NULL,
                finished_at    DATETIME,
                elapsed_s      FLOAT,
                error_log      TEXT,
                result_summary TEXT
            )
        """))
        conn.commit()
        logger.info("Migration: ensured cycle_states table exists")

    # Daily Brief snapshots table (narrative + automation guards)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS daily_brief_snapshots (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of_date       DATE    UNIQUE NOT NULL,
                created_at       DATETIME,
                updated_at       DATETIME,
                regime           VARCHAR(50),
                regime_prev      VARCHAR(50),
                p_risk_on        FLOAT,
                regime_changed   BOOLEAN DEFAULT 0,
                n_long           INTEGER DEFAULT 0,
                n_short          INTEGER DEFAULT 0,
                signal_flips_json TEXT,
                risk_alerts_json  TEXT,
                n_entries         INTEGER DEFAULT 0,
                n_invalidations   INTEGER DEFAULT 0,
                n_rebalance       INTEGER DEFAULT 0,
                n_verified_today  INTEGER DEFAULT 0,
                verify_ran        BOOLEAN DEFAULT 0,
                icir_month        VARCHAR(7),
                narrative         TEXT,
                macro_brief_llm   TEXT
            )
        """))
        conn.commit()
        logger.info("Migration: ensured daily_brief_snapshots table exists")

    # Add macro_brief_llm column to existing daily_brief_snapshots (idempotent)
    with engine.connect() as conn:
        try:
            existing_dbs = {
                row[1]
                for row in conn.execute(
                    text("PRAGMA table_info(daily_brief_snapshots)")
                ).fetchall()
            }
            if "macro_brief_llm" not in existing_dbs:
                conn.execute(text(
                    "ALTER TABLE daily_brief_snapshots ADD COLUMN macro_brief_llm TEXT"
                ))
                conn.commit()
                logger.info("Migration: added column daily_brief_snapshots.macro_brief_llm")
            # P4-6: tactical patrol result columns
            for _col, _typ in [
                ("tactical_entries_json", "TEXT"),
                ("tactical_reduces_json", "TEXT"),
                ("regime_jump_today",     "BOOLEAN"),
            ]:
                if _col not in existing_dbs:
                    conn.execute(text(
                        f"ALTER TABLE daily_brief_snapshots ADD COLUMN {_col} {_typ}"
                    ))
                    logger.info("Migration: added column daily_brief_snapshots.%s", _col)
            conn.commit()
        except Exception:
            pass

    # P3-5: decision_logs — structured trade output fields
    _dl_extra = [
        ("key_thesis",        "VARCHAR(300)"),
        ("primary_risk",      "VARCHAR(200)"),
        ("macro_regime_view", "VARCHAR(20)"),
    ]
    with engine.connect() as conn:
        existing_dl = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(decision_logs)")).fetchall()
        }
        for col_name, col_type in _dl_extra:
            if col_name not in existing_dl:
                conn.execute(text(f"ALTER TABLE decision_logs ADD COLUMN {col_name} {col_type}"))
                logger.info("Migration: added column decision_logs.%s", col_name)
        conn.commit()

    # P3-12: pending_approvals — LLM/Quant disagreement columns
    pa_extra_columns = [
        ("contradicts_quant", "BOOLEAN DEFAULT 0"),
        ("llm_confidence",    "INTEGER"),
    ]
    with engine.connect() as conn:
        existing_pa = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(pending_approvals)")).fetchall()
        }
        for col_name, col_type in pa_extra_columns:
            if col_name not in existing_pa:
                conn.execute(
                    text(f"ALTER TABLE pending_approvals ADD COLUMN {col_name} {col_type}")
                )
                logger.info("Migration: added column pending_approvals.%s", col_name)
        conn.commit()

    # P6: regime_snapshots — add credit_spread column
    with engine.connect() as conn:
        existing_rs = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(regime_snapshots)")).fetchall()
        }
        if "credit_spread" not in existing_rs:
            conn.execute(text("ALTER TABLE regime_snapshots ADD COLUMN credit_spread FLOAT"))
            logger.info("Migration: added column regime_snapshots.credit_spread")
        conn.commit()

    # P6: new audit tables — CREATE IF NOT EXISTS (idempotent)
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS signal_records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            DATE    NOT NULL,
                ticker          VARCHAR(20) NOT NULL,
                sector          VARCHAR(50) NOT NULL,
                tsmom_signal    INTEGER,
                tsmom_raw       FLOAT,
                csmom_rank      FLOAT,
                carry_norm      FLOAT,
                reversal_norm   FLOAT,
                factormad_score FLOAT,
                composite_score FLOAT,
                gate_status     VARCHAR(10),
                regime_at_calc  VARCHAR(20),
                UNIQUE (date, ticker)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS signal_flip_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            DATE NOT NULL,
                ticker          VARCHAR(20) NOT NULL,
                sector          VARCHAR(50) NOT NULL,
                prev_signal     INTEGER,
                new_signal      INTEGER,
                tsmom_raw_prev  FLOAT,
                tsmom_raw_new   FLOAT,
                regime_at_flip  VARCHAR(20)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS data_quality_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        DATE NOT NULL,
                check_type  VARCHAR(50) NOT NULL,
                status      VARCHAR(10) NOT NULL,
                detail      TEXT,
                checked_at  DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                triggered_at  DATETIME NOT NULL,
                level         VARCHAR(10) NOT NULL,
                reason        TEXT,
                auto_resolved BOOLEAN DEFAULT 0,
                resolved_at   DATETIME,
                resolved_by   VARCHAR(100),
                notes         TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS alpha_memory (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_date       DATE NOT NULL,
                sector              VARCHAR(50) NOT NULL,
                source              VARCHAR(20) NOT NULL DEFAULT 'track_b',
                quant_weight        FLOAT,
                llm_delta           FLOAT,
                adjusted_weight     FLOAT,
                logic_chain         TEXT,
                confidence          INTEGER,
                era_verdict         VARCHAR(20),
                era_score           FLOAT,
                era_reasoning       TEXT,
                macro_data_snapshot TEXT,
                verified_at         DATETIME,
                created_at          DATETIME
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS quant_only_snapshots (
                date         DATE PRIMARY KEY,
                nav          FLOAT NOT NULL,
                daily_return FLOAT,
                weights_json TEXT
            )
        """))
        conn.commit()
        logger.info("Migration: ensured P6 audit tables exist")


# ── P2-8 Prompt version hash ──────────────────────────────────────────────────

def compute_prompt_version(prompt_text: str) -> str:
    """SHA-256[:8] of prompt template text. Use on the prompt *before* variable substitution."""
    import hashlib
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8]


# ── Direction extraction ───────────────────────────────────────────────────────

def extract_direction(text_: str) -> str:
    """
    Extract investment direction from LLM output text.

    Strategy (two-pass):
      Pass 1 — conclusion-section priority:
        Search within 150 chars after any conclusion marker
        (最终结论 / 建议方向 / 仲裁结论 / 投资建议 / 综合判断).
        LLM arbitration outputs always end with a structured conclusion block;
        extracting from there prevents false positives from quoted counter-arguments.

      Pass 2 — last-occurrence fallback:
        If no conclusion marker found, use rfind() to locate the LAST occurrence
        of each direction keyword across the full text. The final judgement
        always appears latest in the text, so the highest-index keyword wins.
        This correctly handles transitions like "从低配调整为标配".

    Special directions (拦截 / 通过) are for LCS/audit contexts; they are checked
    last and only when no investment direction is found.
    """
    _CONCLUSION_MARKERS = ["最终结论", "建议方向", "仲裁结论", "投资建议", "综合判断", "Final"]
    _DIRECTION_GROUPS: list[tuple[str, list[str]]] = [
        ("超配", ["超配", "看多", "强烈买入", "做多", "overweight"]),
        ("低配", ["低配", "看空", "减仓", "做空", "underweight"]),
        ("标配", ["标配", "中性", "观望", "持有", "neutral"]),
    ]
    _SPECIAL_GROUPS: list[tuple[str, list[str]]] = [
        ("拦截", ["🚨", "拦截", "否决", "blocked"]),
        ("通过", ["通过", "稳健", "PASS", "pass"]),
    ]

    # Pass 1: search inside conclusion section
    for marker in _CONCLUSION_MARKERS:
        idx = text_.rfind(marker)
        if idx == -1:
            continue
        snippet = text_[idx: idx + 150]
        for direction, keywords in _DIRECTION_GROUPS:
            if any(kw in snippet for kw in keywords):
                return direction

    # Pass 2: last-occurrence across full text
    last_pos  = -1
    last_dir  = "中性"
    for direction, keywords in _DIRECTION_GROUPS:
        for kw in keywords:
            pos = text_.rfind(kw)
            if pos > last_pos:
                last_pos = pos
                last_dir = direction

    if last_pos >= 0:
        return last_dir

    # Special audit directions (only when no investment direction found)
    for direction, keywords in _SPECIAL_GROUPS:
        if any(kw in text_ for kw in keywords):
            return direction

    return "中性"


# ── Accuracy scoring ───────────────────────────────────────────────────────────

BASELINE_HIT_RATE = 0.50
MIN_ACCEPTABLE    = 0.55
GOOD_THRESHOLD    = 0.65
EXCELLENT         = 0.75

# ── Train / test split boundary ────────────────────────────────────────────────
# Decisions with decision_date >= TRAIN_TEST_CUTOFF belong to the LOCKED TEST SET.
#
# Rule: test-period verified decisions may receive accuracy_score (price verification
# is always allowed), but MUST NOT write back into any learning table:
#   - LearningLog      (run_meta_agent_analysis, _check_horizon_mismatch)
#   - QuantPatternLog  (_update_quant_pattern)
#   - NewsRoutingWeight(_feedback_routing_weights)
#   - SkillLibrary     (maybe_update_skill)
#
# This date is SET ONCE and never moved forward. Moving it forward would constitute
# leaking test-set outcomes into the training pipeline.
TRAIN_TEST_CUTOFF: datetime.date = datetime.date(2023, 1, 1)

# ── Clean Zone boundary ────────────────────────────────────────────────────────
# Decisions with decision_date >= CLEAN_ZONE_START are in the "Clean Zone":
# data created after this date lies beyond the LLM's training-set knowledge
# cutoff, meaning the model has ZERO foreknowledge of outcomes for these dates.
#
# CLEAN_ZONE_START vs TRAIN_TEST_CUTOFF — two different purposes:
#   TRAIN_TEST_CUTOFF (2023-01-01) — learning-loop integrity gate:
#       prevents test-set outcomes from contaminating training tables.
#   CLEAN_ZONE_START  (2025-04-01) — foreknowledge isolation gate:
#       Clean Zone decisions are the only records that are BOTH test-set AND
#       free of LLM historical-narrative contamination. This is where genuine
#       out-of-sample alpha evidence would appear.
#
# All decisions in [TRAIN_TEST_CUTOFF, CLEAN_ZONE_START) are "Test Set A":
#   verified without write-back, but the LLM may still recall narratives
#   about outcomes in that period from its training data.
# Decisions >= CLEAN_ZONE_START are "Test Set B / Clean Zone":
#   true out-of-sample + foreknowledge-free.
CLEAN_ZONE_START: datetime.date = datetime.date(2025, 4, 1)


def score_accuracy(direction: str, return_pct: float, horizon: str = "季度(3个月)") -> float:
    """
    Score a past decision against actual price return.

    Horizons (aligned to actual verification windows — no proxy):
      季度(3个月) : strong=±8%,  partial=±4%   — one earnings cycle, macro regime check
      半年(6个月) : strong=±12%, partial=±6%   — two earnings cycles, policy transmission

    Key: first character of horizon label → "季" or "半".
    Legacy labels ("中", "长", "短") are mapped to the nearest new horizon for
    backward compatibility with records created before the rename.

    Rubric (0.0 – 1.0):
      超配 → rewards positive return
      低配 → rewards negative return
      标配 → rewards near-zero return (within partial band)
      拦截 → rewards negative return (correct block)
      通过 → rewards non-negative return
    """
    # Horizon-appropriate thresholds: (strong_move, partial_move)
    # Calibrated to 1σ of typical sector ETF vol over the respective window.
    _THRESHOLDS: dict[str, tuple[float, float]] = {
        "季": (0.08, 0.04),   # 季度(3个月): ±8% strong, ±4% partial
        "半": (0.12, 0.06),   # 半年(6个月): ±12% strong, ±6% partial
    }
    # Legacy key mapping for backward compatibility
    _LEGACY_MAP: dict[str, str] = {
        "中": "季",   # 中期(3-6m) → 季度
        "长": "半",   # 长期(1y+)  → 半年 (nearest, conservative)
        "短": "季",   # 短期 should not exist; clamp to quarterly
    }
    raw_key = (horizon or "季度(3个月)").strip()[:1]
    key = _LEGACY_MAP.get(raw_key, raw_key)   # map legacy, pass through new
    strong, partial = _THRESHOLDS.get(key, _THRESHOLDS["季"])

    r = return_pct
    if direction == "超配":
        if r >  strong:  return 1.00
        if r >  0.00:    return 0.75
        if r > -partial: return 0.50
        return 0.00
    if direction == "低配":
        if r < -strong:  return 1.00
        if r <  0.00:    return 0.75
        if r <  partial: return 0.50
        return 0.00
    if direction == "标配":
        if abs(r) <= partial:        return 1.00
        if abs(r) <= partial * 2.5:  return 0.50
        return 0.00
    if direction == "拦截":
        if r < -partial: return 1.00
        if r >  strong:  return 0.00
        return 0.50
    if direction == "通过":
        if r > -partial:       return 1.00
        if r < -(partial * 3): return 0.00
        return 0.50
    return 0.50


def _regime_from_vix(vix: float) -> str:
    """Map VIX level to macro regime label (mirrors _infer_macro_regime in ui/tabs.py)."""
    if vix >= 30: return "高波动/危机"
    if vix >= 20: return "震荡期"
    if vix >= 15: return "温和波动"
    return "低波动/牛市"


def _meta_verdict(score: float) -> str:
    if score >= EXCELLENT:       return "HIGH"
    if score >= MIN_ACCEPTABLE:  return "REVIEW"
    return "WEAK"


# ── CRUD ───────────────────────────────────────────────────────────────────────

def _to_serializable(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON."""
    try:
        import numpy as _np
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def _today_window() -> tuple[datetime.datetime, datetime.datetime]:
    start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + datetime.timedelta(days=1)


def save_decision(
    tab_type: str,
    ai_conclusion: str,
    vix_level: float = 0.0,
    sector_name: str = "",
    ticker: str = "",
    news_summary: str = "",
    quant_metrics: dict | None = None,
    overwrite: bool = False,
    # ── new structured fields ──────────────────────────────────────────
    confidence_score: int | None = None,
    horizon: str = "",
    invalidation_conditions: str = "",
    economic_logic: str = "",
    macro_regime: str = "",
    news_categories_used: list[str] | None = None,
    is_backtest: bool = False,
    macro_confidence: int | None = None,
    news_confidence: int | None = None,
    technical_confidence: int | None = None,
    signal_attribution: dict | None = None,
    sensitivity_flag: str = "",
    debate_transcript: dict | None = None,
    decision_date: datetime.date | None = None,  # historical date for backtest records
    reflection_chain: str = "",
    needs_retry: bool = False,
    parent_decision_id: int | None = None,
    revision_reason: str = "",
    decision_source: str = "ai_drafted",  # "ai_drafted" | "human_edited" | "human_initiated"
    quant_p_noise: float | None = None,
    quant_val_r2: float | None = None,
    quant_test_r2: float | None = None,
    quant_active: int | None = None,
    weight_adjustment_pct: float | None = None,
    adjustment_reason: str | None = None,
    edit_ratio: float | None = None,    # Levenshtein-based edit magnitude [0,1]; None if ai_drafted
    signal_invalidation_risk: int | None = None,
    model_version: str = "",          # P2-8: e.g. "claude-sonnet-4-6"
    prompt_version: str = "",         # P2-8: SHA-256[:8] of prompt template
) -> int:
    """
    Persist one AI decision. Returns the row id (existing or new).

    decision_source records the degree of human involvement:
      "ai_drafted"      — AI generated; human confirmed without substantive edits
      "human_edited"    — AI draft was materially modified before saving
      "human_initiated" — Human wrote the core thesis; AI provided structural assist

    Deduplication: same tab_type + sector_name logged today → return existing id.
    If overwrite=True (Refresh path), delete old record before inserting.
    """
    direction = extract_direction(ai_conclusion)
    today_start, today_end = _today_window()

    with SessionFactory() as session:
        if not is_backtest:
            existing = (
                session.query(DecisionLog)
                .filter(
                    DecisionLog.tab_type == tab_type,
                    DecisionLog.sector_name == (sector_name or None),
                    DecisionLog.created_at >= today_start,
                    DecisionLog.created_at < today_end,
                )
                .first()
            )
            if existing:
                if not overwrite:
                    logger.info(
                        "Skipping duplicate decision tab=%s sector=%s (id=%d)",
                        tab_type, sector_name, existing.id,
                    )
                    return existing.id
                session.delete(existing)
                session.flush()
        else:
            # Backtest deduplication: same tab_type × sector × decision_date with real
            # content must never be saved twice — DB-level safety net behind _done_pairs.
            # tab_type is included so simple backtest and walk-forward records for the
            # same sector/date are treated as distinct and never incorrectly deduplicated.
            if not needs_retry and decision_date is not None:
                existing_bt = (
                    session.query(DecisionLog)
                    .filter(
                        DecisionLog.tab_type == tab_type,
                        DecisionLog.is_backtest == True,
                        DecisionLog.sector_name == (sector_name or None),
                        DecisionLog.decision_date == decision_date,
                        DecisionLog.needs_retry == False,
                        DecisionLog.ai_conclusion.isnot(None),
                    )
                    .first()
                )
                if existing_bt:
                    logger.warning(
                        "Blocked duplicate backtest record tab=%s sector=%s date=%s (id=%d) — "
                        "_done_pairs check should have caught this",
                        tab_type, sector_name, decision_date, existing_bt.id,
                    )
                    return existing_bt.id

        _decision_date = decision_date or datetime.datetime.utcnow().date()

        # Macro decisions have no ticker and cannot be price-verified via Triple-Barrier.
        # Mark them verified=True immediately so they don't accumulate as phantom "pending"
        # records in the monitor. They still provide historical context but are excluded
        # from Clean Zone accuracy statistics (accuracy_score stays None).
        _macro_no_verify = (tab_type == "macro" and not ticker)

        log = DecisionLog(
            tab_type=tab_type,
            vix_level=vix_level,
            sector_name=sector_name or None,
            ticker=ticker or None,
            news_summary=(news_summary or None),
            ai_conclusion=ai_conclusion,
            direction=direction,
            quant_metrics=json.dumps(_to_serializable(quant_metrics)) if quant_metrics else None,
            confidence_score=confidence_score,
            horizon=horizon or None,
            invalidation_conditions=invalidation_conditions or None,
            economic_logic=economic_logic or None,
            macro_regime=macro_regime or None,
            news_categories_used=(
                json.dumps(news_categories_used) if news_categories_used else None
            ),
            is_backtest=is_backtest,
            decision_date=_decision_date,
            macro_confidence=macro_confidence,
            news_confidence=news_confidence,
            technical_confidence=technical_confidence,
            signal_attribution=(
                json.dumps(signal_attribution) if signal_attribution else None
            ),
            sensitivity_flag=sensitivity_flag or None,
            debate_transcript=(
                json.dumps(debate_transcript, ensure_ascii=False) if debate_transcript else None
            ),
            reflection_chain=reflection_chain or None,
            needs_retry=needs_retry,
            parent_decision_id=parent_decision_id,
            revision_reason=revision_reason or None,
            verified=_macro_no_verify,
            verified_at=datetime.datetime.utcnow() if _macro_no_verify else None,
            decision_source=decision_source or "ai_drafted",
            quant_p_noise=quant_p_noise,
            quant_val_r2=quant_val_r2,
            quant_test_r2=quant_test_r2,
            quant_active=quant_active,
            weight_adjustment_pct=weight_adjustment_pct,
            adjustment_reason=adjustment_reason or None,
            edit_ratio=edit_ratio,
            signal_invalidation_risk=signal_invalidation_risk,
            model_version=model_version or None,
            prompt_version=prompt_version or None,
        )
        session.add(log)
        session.flush()   # get log.id before commit

        # P2-10: chain hash — links this record to the previous one
        try:
            import hashlib as _hl
            _prev = (
                session.query(DecisionLog.chain_hash)
                .filter(DecisionLog.id < log.id, DecisionLog.chain_hash.isnot(None))
                .order_by(DecisionLog.id.desc())
                .first()
            )
            _prev_hash = _prev[0] if _prev else "genesis"
            _payload   = f"{log.id}|{log.created_at}|{log.ai_conclusion[:100]}|{_prev_hash}"
            log.chain_hash = _hl.sha256(_payload.encode("utf-8")).hexdigest()
        except Exception:
            pass  # non-critical

        session.commit()
        logger.info(
            "Saved decision id=%d tab=%s direction=%s regime=%s backtest=%s",
            log.id, tab_type, direction, macro_regime, is_backtest,
        )

        # P1.2 Auto-linkage: sector analysis → SimulatedPosition upsert
        if tab_type == "sector" and sector_name and direction and not is_backtest:
            _auto_link_position(
                session=session,
                sector=sector_name,
                ticker=ticker or "",
                direction=direction,
                regime_label=macro_regime or "",
                decision_date=_decision_date,
                weight_adjustment_pct=weight_adjustment_pct,
            )

        # P1.3 Auto-linkage: sector analysis → WatchlistEntry (watching status)
        # Only for high-conviction overweight decisions; sets up entry condition buffer.
        if tab_type == "sector" and sector_name and not is_backtest:
            _auto_link_watchlist(
                session=session,
                decision_id=log.id,
                sector=sector_name,
                ticker=ticker or "",
                direction=direction or "",
                confidence_score=confidence_score or 0,
                regime_label=macro_regime or "",
                invalidation_conditions_text=invalidation_conditions or "",
                weight_adjustment_pct=weight_adjustment_pct,
                decision_date=_decision_date,
                signal_invalidation_risk=signal_invalidation_risk,
            )

        return log.id


def _auto_link_position(
    session,
    sector:               str,
    ticker:               str,
    direction:            str,
    regime_label:         str,
    decision_date:        "datetime.date",
    weight_adjustment_pct: float | None = None,
) -> None:
    """
    Upsert a SimulatedPosition row when a sector decision is saved.
    Maps LLM recommendation → TSMOM signal → baseline target_weight.
    Soft override (weight_adjustment_pct) is applied as a delta.

    Paper-trading: if portfolio NAV is set (SystemConfig "paper_trading_nav"),
    also computes shares_held and cost_basis using the ETF's last close price.

    Baseline weight mapping:
        超配 → +0.08   (8% long allocation)
        标配 →  0.0   (neutral / flat)
        低配 → -0.04   (4% underweight / light short)
    """
    _SIGNAL_MAP = {"超配": 1, "标配": 0, "低配": -1,
                   "long": 1, "short": -1, "neutral": 0,
                   "overweight": 1, "underweight": -1}
    _BASE_WEIGHT = {"超配": 0.08, "标配": 0.0, "低配": -0.04,
                    "long": 0.08, "short": -0.04, "neutral": 0.0,
                    "overweight": 0.08, "underweight": -0.04}

    direction_lower = direction.lower().strip()
    tsmom  = _SIGNAL_MAP.get(direction_lower, _SIGNAL_MAP.get(direction, 0))
    base_w = _BASE_WEIGHT.get(direction_lower, _BASE_WEIGHT.get(direction, 0.0))

    adj    = float(weight_adjustment_pct or 0.0) / 100.0
    target = max(-0.20, min(0.20, base_w + adj))

    # Paper-trading: compute shares from NAV if configured
    entry_price   = None
    shares_held   = None
    cost_basis    = None
    position_value = None
    try:
        nav_str = get_system_config("paper_trading_nav", "")
        if nav_str and ticker:
            nav = float(nav_str)
            import yfinance as _yf
            _hist = _yf.download(ticker, period="2d", auto_adjust=True,
                                 progress=False, multi_level_index=False)
            if not _hist.empty:
                entry_price    = float(_hist["Close"].dropna().iloc[-1])
                alloc_amount   = nav * abs(target)          # $ amount for this position
                shares_held    = alloc_amount / entry_price if entry_price > 0 else 0.0
                cost_basis     = alloc_amount
                position_value = alloc_amount
    except Exception as e:
        logger.debug("Paper-trading price fetch failed for %s: %s", ticker, e)

    # ── Fetch pure quant TSMOM signal for attribution baseline ───────────────
    quant_tsmom: int | None = None
    quant_target: float | None = None
    try:
        from engine.signal import get_signal_dataframe
        _sig_df = get_signal_dataframe(as_of=decision_date)
        if ticker and ticker in _sig_df.index:
            quant_tsmom = int(_sig_df.loc[ticker, "tsmom"])
            _q_base = {1: 0.08, 0: 0.0, -1: -0.04}.get(quant_tsmom, 0.0)
            quant_target = max(-0.20, min(0.20, _q_base))
    except Exception as _e:
        logger.debug("Quant signal fetch skipped for %s: %s", ticker, _e)

    def _upsert(track: str, t_weight: float, s_tsmom: int | None,
                note_dir: str) -> None:
        existing = (
            session.query(SimulatedPosition)
            .filter_by(snapshot_date=decision_date, sector=sector, track=track)
            .first()
        )
        if existing:
            existing.target_weight = round(t_weight, 4)
            existing.actual_weight = round(t_weight, 4)
            existing.signal_tsmom  = s_tsmom
            existing.regime_label  = regime_label or existing.regime_label
            existing.ticker        = ticker or existing.ticker
            existing.direction     = note_dir
            existing.notes         = f"auto-linked ({note_dir}) [{track}]"
            if track == "main" and entry_price is not None:
                existing.entry_price    = entry_price
                existing.shares_held    = shares_held
                existing.cost_basis     = cost_basis
                existing.position_value = position_value
        else:
            session.add(SimulatedPosition(
                snapshot_date   = decision_date,
                sector          = sector,
                ticker          = ticker,
                target_weight   = round(t_weight, 4),
                actual_weight   = round(t_weight, 4),
                signal_tsmom    = s_tsmom,
                regime_label    = regime_label or "",
                direction       = note_dir,
                track           = track,
                entry_price     = entry_price if track == "main" else None,
                shares_held     = shares_held if track == "main" else None,
                cost_basis      = cost_basis  if track == "main" else None,
                position_value  = position_value if track == "main" else None,
                notes           = f"auto-linked ({note_dir}) [{track}]",
            ))

    try:
        _upsert("main", target, tsmom, direction)
        if quant_target is not None:
            _upsert("quant", quant_target, quant_tsmom, f"tsmom={quant_tsmom}")
        session.commit()
        logger.info(
            "Auto-linked position: sector=%s main_w=%.4f quant_w=%s entry_px=%s",
            sector, target,
            f"{quant_target:.4f}" if quant_target is not None else "n/a",
            entry_price,
        )
    except Exception as e:
        logger.warning("Auto-link position failed for %s: %s", sector, e)
        session.rollback()


def _auto_link_watchlist(
    session,
    decision_id:                int,
    sector:                     str,
    ticker:                     str,
    direction:                  str,
    confidence_score:           int,
    regime_label:               str,
    invalidation_conditions_text: str,
    decision_date:              "datetime.date",
    weight_adjustment_pct:      float | None = None,
    signal_invalidation_risk:   int | None = None,
) -> None:
    """
    Create a WatchlistEntry(status='watching') when a high-conviction overweight
    decision is saved.  Only fires for 超配/overweight with sufficient effective
    conviction, where:

        effective_conviction = confidence × (1 - 0.6 × invalidation_risk / 100)

    Gate threshold: effective_conviction ≥ 55.  A high invalidation_risk (e.g. 70)
    requires much higher LLM confidence to overcome.  Examples:
      conf=60, risk=0  → effective=60  ✓
      conf=60, risk=50 → effective=42  ✗  (signal too fragile)
      conf=80, risk=50 → effective=56  ✓  (high conviction overcomes moderate risk)
      conf=80, risk=70 → effective=46  ✗
    """
    _DIRECTION_MAP = {
        "超配": "long", "long": "long", "overweight": "long",
        "低配": "short", "short": "short", "underweight": "short",
        "标配": "neutral", "neutral": "neutral",
    }
    _BASE_WEIGHT = {
        "long": 0.08, "short": 0.04, "neutral": 0.0,
    }

    direction_norm = _DIRECTION_MAP.get(direction.lower().strip(), direction.lower())

    # Gate: direction must be long; effective conviction must clear threshold.
    _inv_risk = signal_invalidation_risk if signal_invalidation_risk is not None else 50
    effective_conviction = confidence_score * (1.0 - 0.6 * _inv_risk / 100.0)
    if direction_norm != "long" or effective_conviction < 55:
        return

    try:
        existing = (
            session.query(WatchlistEntry)
            .filter_by(sector=sector, created_date=decision_date, source_agent="research_agent")
            .first()
        )
        if existing:
            # Update confidence and decision_log_id if re-analysed same day
            existing.confidence     = confidence_score
            existing.decision_log_id = decision_id
            session.commit()
            logger.info("WatchlistEntry already exists for sector=%s date=%s, updated", sector, decision_date)
            return

        base_w = _BASE_WEIGHT[direction_norm]
        adj    = float(weight_adjustment_pct or 0.0) / 100.0
        suggested = round(max(0.02, min(0.20, base_w + adj)), 4)

        entry_condition = {"type": "price_breakout", "n_days": 20}
        invalidation: list[dict] = []
        if invalidation_conditions_text.strip():
            invalidation.append({
                "type": "descriptive",
                "description": invalidation_conditions_text.strip()[:500],
            })

        session.add(WatchlistEntry(
            created_date          = decision_date,
            status                = "watching",
            sector                = sector,
            ticker                = ticker,
            direction             = direction_norm,
            position_rank         = "satellite",
            quant_baseline_weight = base_w,
            llm_adjustment_pct    = float(weight_adjustment_pct or 0.0),
            suggested_weight      = suggested,
            regime_label          = regime_label,
            source_agent          = "research_agent",
            decision_log_id       = decision_id,
            confidence            = confidence_score,
            entry_condition_json  = json.dumps(entry_condition),
            invalidation_json     = json.dumps(invalidation) if invalidation else None,
        ))
        session.commit()
        logger.info(
            "Auto-linked WatchlistEntry: sector=%s ticker=%s confidence=%d suggested_w=%.4f",
            sector, ticker, confidence_score, suggested,
        )
    except Exception as e:
        logger.warning("Auto-link watchlist failed for %s: %s", sector, e)
        session.rollback()


def supersede_decision(decision_id: int, reason: str) -> None:
    """
    Mark a decision as superseded by a revision.
    The record is kept for audit purposes but excluded from active queues
    and performance statistics.
    """
    with SessionFactory() as session:
        dec = session.query(DecisionLog).filter(DecisionLog.id == decision_id).first()
        if dec:
            dec.superseded     = True
            dec.revision_reason = reason or None
            session.commit()
            logger.info("Decision id=%d marked superseded. Reason: %s", decision_id, reason)


def get_last_revision_time(sector: str) -> datetime.datetime | None:
    """
    Return the created_at timestamp of the most recent revision decision for
    a given sector (i.e. any DecisionLog row with parent_decision_id set).
    Used to enforce the 24-hour per-sector auto-revision cooldown in the DB
    instead of session_state so it survives page refreshes.
    """
    with SessionFactory() as session:
        row = (
            session.query(DecisionLog.created_at)
            .filter(
                DecisionLog.sector_name == sector,
                DecisionLog.parent_decision_id.isnot(None),
            )
            .order_by(DecisionLog.created_at.desc())
            .first()
        )
        return row[0] if row else None


def get_decision_by_id(decision_id: int) -> dict | None:
    """Return a lightweight dict of key fields for a single DecisionLog row."""
    with SessionFactory() as session:
        dec = session.query(DecisionLog).filter(DecisionLog.id == decision_id).first()
        if not dec:
            return None
        return {
            "id":                       dec.id,
            "sector_name":              dec.sector_name or "",
            "direction":                dec.direction or "—",
            "created_at":               dec.created_at.strftime("%Y-%m-%d"),
            "horizon":                  dec.horizon or "季度(3个月)",
            "economic_logic":           dec.economic_logic or "",
            "invalidation_conditions":  dec.invalidation_conditions or "",
            "macro_regime":             dec.macro_regime or "",
            "confidence_score":         dec.confidence_score,
        }


def get_revision_chains() -> list[dict]:
    """
    Return all revision pairs: original decision → revised decision.
    Each entry contains key fields from both sides for display in Alpha Memory.
    Only returns chains where the parent still exists in the DB.
    """
    with SessionFactory() as session:
        revisions = (
            session.query(DecisionLog)
            .filter(DecisionLog.parent_decision_id.isnot(None))
            .order_by(DecisionLog.created_at.desc())
            .limit(50)
            .all()
        )
        chains = []
        for rev in revisions:
            parent = session.query(DecisionLog).filter(
                DecisionLog.id == rev.parent_decision_id
            ).first()
            chains.append({
                "revision_id":       rev.id,
                "revision_date":     rev.created_at.strftime("%Y-%m-%d %H:%M"),
                "revision_direction":rev.direction or "—",
                "revision_reason":   rev.revision_reason or "—",
                "sector":            rev.sector_name or "—",
                "parent_id":         rev.parent_decision_id,
                "parent_date":       parent.created_at.strftime("%Y-%m-%d") if parent else "—",
                "parent_direction":  parent.direction if parent else "—",
                "parent_regime":     parent.macro_regime if parent else "—",
                "direction_changed": (
                    parent is not None and
                    rev.direction != parent.direction
                ),
            })
        return chains


def save_stress_test_log(
    scenario_id:       str,
    scenario_name:     str,
    scenario_category: str,
    sector:            str,
    effective_vix:     float,
    ai_output:         str,
    fed_funds_delta:   int | None = None,
    oil_delta_pct:     float | None = None,
    usd_delta_pct:     float | None = None,
    custom_note:       str = "",
) -> int:
    """
    Persist a lightweight stress-test record for retrospective review.
    Extracts direction from AI output and stores a short summary (≤300 chars).
    Returns the new row id.
    """
    direction = extract_direction(ai_output)

    # Extract "综合判断" line as summary; fall back to first 300 chars
    summary = ""
    for line in ai_output.splitlines():
        if "综合判断" in line or "→" in line:
            summary = line.strip().lstrip("→ ").strip()[:300]
            break
    if not summary:
        summary = ai_output.strip()[:300]

    with SessionFactory() as session:
        row = StressTestLog(
            scenario_id       = scenario_id,
            scenario_name     = scenario_name,
            scenario_category = scenario_category,
            sector            = sector,
            effective_vix     = effective_vix,
            fed_funds_delta   = fed_funds_delta,
            oil_delta_pct     = oil_delta_pct,
            usd_delta_pct     = usd_delta_pct,
            ai_direction      = direction,
            ai_summary        = summary,
            custom_note       = custom_note or None,
        )
        session.add(row)
        session.commit()
        return row.id


def get_stress_test_history(limit: int = 50) -> list[dict]:
    """Return recent stress test records for the retrospective review panel."""
    with SessionFactory() as session:
        rows = (
            session.query(StressTestLog)
            .order_by(StressTestLog.run_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":               r.id,
                "run_at":           r.run_at.strftime("%Y-%m-%d %H:%M") if r.run_at else "—",
                "scenario_name":    r.scenario_name,
                "scenario_category":r.scenario_category or "—",
                "sector":           r.sector,
                "effective_vix":    r.effective_vix,
                "fed_funds_delta":  r.fed_funds_delta,
                "oil_delta_pct":    r.oil_delta_pct,
                "usd_delta_pct":    r.usd_delta_pct,
                "ai_direction":     r.ai_direction or "—",
                "ai_summary":       r.ai_summary or "—",
                "custom_note":      r.custom_note or "",
            }
            for r in rows
        ]


def get_today_report(tab_type: str, sector_name: str = "") -> str | None:
    today_start, today_end = _today_window()
    with SessionFactory() as session:
        record = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type == tab_type,
                DecisionLog.sector_name == (sector_name or None),
                DecisionLog.created_at >= today_start,
                DecisionLog.created_at < today_end,
            )
            .order_by(DecisionLog.created_at.desc())
            .first()
        )
        return record.ai_conclusion if record else None


def get_all_today_sector_reports() -> dict[str, str]:
    today_start, today_end = _today_window()
    with SessionFactory() as session:
        records = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type == "sector",
                DecisionLog.created_at >= today_start,
                DecisionLog.created_at < today_end,
            )
            .order_by(DecisionLog.created_at.asc())
            .all()
        )
        return {r.sector_name: r.ai_conclusion for r in records if r.sector_name}


def get_all_today_sector_full() -> list[dict]:
    """
    Return full sector records for today including debate transcript and XAI data.
    Used by restore_today_from_db to repopulate session_state after page reload.
    """
    today_start, today_end = _today_window()
    with SessionFactory() as session:
        records = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type == "sector",
                DecisionLog.created_at >= today_start,
                DecisionLog.created_at < today_end,
                DecisionLog.is_backtest == False,
            )
            .order_by(DecisionLog.created_at.asc())
            .all()
        )
        results = []
        for r in records:
            if not r.sector_name:
                continue
            debate = {}
            if r.debate_transcript:
                try:
                    debate = json.loads(r.debate_transcript)
                except Exception:
                    pass
            attribution = {}
            if r.signal_attribution:
                try:
                    attribution = json.loads(r.signal_attribution)
                except Exception:
                    pass
            results.append({
                "sector_name":    r.sector_name,
                "ai_conclusion":  r.ai_conclusion or "",
                "created_at":     r.created_at,
                "debate_history": debate.get("history", []),
                "arb_notes":      debate.get("arbitration", ""),
                "blue_output":    debate.get("blue_output", ""),
                "xai": {
                    "overall_confidence":   r.confidence_score,
                    "macro_confidence":     r.macro_confidence,
                    "news_confidence":      r.news_confidence,
                    "technical_confidence": r.technical_confidence,
                    "signal_drivers":       attribution.get("drivers", ""),
                    "invalidation_conditions": r.invalidation_conditions or "",
                    "horizon":              r.horizon or "季度(3个月)",
                },
            })
        return results


def get_all_today_audit_records() -> list[dict]:
    """
    Return today's audit records for repopulating Tab4 audit_cache after reload.
    """
    today_start, today_end = _today_window()
    with SessionFactory() as session:
        records = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type == "audit",
                DecisionLog.created_at >= today_start,
                DecisionLog.created_at < today_end,
            )
            .order_by(DecisionLog.created_at.asc())
            .all()
        )
        results = []
        for r in records:
            qm = {}
            if r.quant_metrics:
                try:
                    qm = json.loads(r.quant_metrics)
                except Exception:
                    pass
            # ai_conclusion was stored as red_team_critique + "\n" + audit_memo
            conclusion = r.ai_conclusion or ""
            split_idx  = conclusion.find("\n")
            red_critique = conclusion[:split_idx] if split_idx != -1 else ""
            audit_memo   = conclusion[split_idx + 1:] if split_idx != -1 else conclusion
            # Prefer memo fields stored inside quant_metrics if present
            stored_critique = qm.get("red_team_critique", "")
            stored_memo     = qm.get("audit_memo", "")
            results.append({
                "target_assets":      r.sector_name or "",
                "vix_level":          r.vix_level or 20.0,
                "created_at":         r.created_at,
                "is_robust":          qm.get("is_robust", True),
                "red_team_critique":  stored_critique or red_critique,
                "audit_memo":         stored_memo or audit_memo,
                "technical_report":   qm.get("technical_report", ""),
                "news_context":       r.news_summary or "",
                "alternative_suggestion": "",
                "reflection_chain":   r.reflection_chain or "",
                "quant_results": {
                    "d_var":            qm.get("d_var"),
                    "var_ci":           qm.get("var_ci"),
                    "var_cf":           qm.get("var_cf"),
                    "confidence_score": qm.get("confidence"),
                    "is_robust":        qm.get("is_robust", True),
                    "sharpe":           qm.get("sharpe"),
                    "sharpe_ci":        qm.get("sharpe_ci"),
                    "skewness":         qm.get("skewness"),
                    "excess_kurt":      qm.get("excess_kurt"),
                    "vol":              qm.get("vol"),
                    "a_ret":            qm.get("a_ret"),
                    "a_vol":            qm.get("a_vol"),
                    "p_noise":          qm.get("p_noise"),
                    "active":           qm.get("active"),
                    "sparsity":         qm.get("sparsity"),
                    "mom_1m":           qm.get("mom_1m"),
                    "mom_3m":           qm.get("mom_3m"),
                    "mom_6m":           qm.get("mom_6m"),
                    "market_fit":       qm.get("market_fit"),
                    "fund_flow":        qm.get("fund_flow"),
                    "momentum":         qm.get("momentum"),
                },
            })
        return results


# ── Verification (5d + 20d) ────────────────────────────────────────────────────

# Sector ETF map for decisions that don't have a ticker.
# Lazily loaded from universe_manager (P2-11); static dict is the fallback.
_SECTOR_ETF_MAP_STATIC = {
    "AI算力/半导体":  "SMH",
    "科技成长(纳指)": "QQQ",
    "生物科技":       "XBI",
    "金融":           "XLF",
    "全球能源":       "XLE",
    "工业/基建":      "XLI",
    "医疗健康":       "XLV",
    "防御消费":       "XLP",
    "消费科技":       "XLY",
    "美国REITs":      "VNQ",
    "黄金":           "GLD",
    "美国长债":       "TLT",
    "清洁能源":       "ICLN",
    "沪深300":        "ASHR",
    "中国科技":       "KWEB",
    "新加坡蓝筹":     "EWS",
    "通讯传媒":       "XLC",
    "高收益债":       "HYG",
}


def _get_sector_etf_map() -> dict[str, str]:
    try:
        from engine.history import get_active_sector_etf
        active = get_active_sector_etf()
        if active:
            return active
    except Exception:
        pass
    return _SECTOR_ETF_MAP_STATIC


_SECTOR_ETF_MAP = _SECTOR_ETF_MAP_STATIC  # backward-compat alias


# ── Triple-Barrier Method ─────────────────────────────────────────────────────
#
# Parameters are FIXED by economic logic — never tuned on data:
#   TP multiplier = 1.0  (take-profit at 1σ of horizon volatility)
#   SL multiplier = 0.7  (stop-loss at 0.7σ — asymmetric: tighter than TP,
#                         reflects that adverse moves cut off faster in practice)
#   Time barrier  = 2×   signal half-life (economic estimate of information decay):
#       短期 half-life ≈  7 cal days → time barrier = 14 cal days
#       中期 half-life ≈ 45 cal days → time barrier = 90 cal days
#       长期 half-life ≈ 90 cal days → time barrier = 180 cal days
#
# Horizon σ = annualised_vol × √(trading_days / 252)
# where trading_days ≈ time_barrier_cal × (252/365)
#
# TP/SL levels are expressed as cumulative returns from entry:
#   超配: TP at +tp_mult×σ, SL at −sl_mult×σ
#   低配: TP at −tp_mult×σ (price fell = profit), SL at +sl_mult×σ (price rose = loss)
#   标配: TP/SL both signal a large move → fall back to original score_accuracy
#   拦截: treated as 低配 (blocking was correct if price fell)
#   通过: treated as 超配 (passing was correct if price held up)
#
# Scoring rule:
#   barrier = "tp"   → score = 1.0   (signal fully vindicated before time decays it)
#   barrier = "sl"   → score = 0.0   (signal clearly wrong; position stopped out)
#   barrier = "time" → score = score_accuracy(direction, cum_return_at_T, horizon_key)
#                       (fallback: same as old fixed-window method, but at 2×half-life T)
#
# This replaces the arbitrary fixed windows (5/40/90 days) with a path-based mechanism.
# The same signal that "looks good at day 40 but gives it back by day 60" now gets the
# high score at barrier hit, not the degraded score at the fixed point.

_TB_TP_MULT: float = 1.0   # economically anchored — DO NOT tune
_TB_SL_MULT: float = 0.7   # economically anchored — DO NOT tune

# Time barriers (calendar days) — aligned to declared horizon upper bounds.
# 季度(3个月): full 90-day window; no proxy, no truncation.
# 半年(6个月): full 180-day window; no proxy, no truncation.
# Legacy keys ("中", "长") mapped for backward compat with old DB records.
_TB_TIME_BARRIERS: dict[str, int] = {
    "季": 90,    # 季度(3个月) — 1 earnings cycle
    "半": 180,   # 半年(6个月) — 2 earnings cycles
    # Legacy backward-compat
    "中": 90,
    "长": 180,
}


def _compute_triple_barrier_score(
    direction:     str,
    post_prices:   "pd.Series",     # close prices FROM entry (index=dates, iloc[0]=entry)
    horizon_key:   str,             # "短" / "中" / "长" (first char of horizon label)
    annualized_vol: float,          # realised σ from 252-day lookback before entry
) -> tuple[float, str, float, int]:
    """
    Score a decision using the Triple-Barrier Method.

    Parameters
    ----------
    direction      : e.g. "超配" / "低配" / "标配" / "拦截" / "通过"
    post_prices    : close price series starting at entry date (DatetimeIndex)
    horizon_key    : first character of horizon label ("短" / "中" / "长")
    annualized_vol : annualised historical volatility (e.g. 0.22 for 22%)

    Returns
    -------
    score          : float 0.0 – 1.0
    barrier_type   : "tp" / "sl" / "time"
    cum_return     : float — cumulative return at the barrier hit point
    cal_days       : int   — calendar days from entry to barrier hit
    """
    time_barrier_cal = _TB_TIME_BARRIERS.get(horizon_key, _TB_TIME_BARRIERS["中"])

    # Convert calendar days → approximate trading days for vol scaling
    trading_days = max(5, int(time_barrier_cal * 252 / 365))
    horizon_sigma = annualized_vol * math.sqrt(trading_days / 252)

    # Up/down barrier levels as cumulative return from p0
    up_barrier   =  _TB_TP_MULT * horizon_sigma   # e.g. +12.5%
    down_barrier = -_TB_SL_MULT * horizon_sigma   # e.g. -8.75%

    p0 = float(post_prices.iloc[0])
    entry_date = post_prices.index[0]

    for dt_idx, px in post_prices.items():
        cum_ret = float(px) / p0 - 1.0

        # Calendar days elapsed from entry
        try:
            cal_days = (dt_idx.date() - entry_date.date()).days
        except AttributeError:
            cal_days = int((dt_idx - entry_date).days)

        # Time barrier reached → fall back to original scoring
        if cal_days >= time_barrier_cal:
            return score_accuracy(direction, cum_ret, horizon_key), "time", cum_ret, cal_days

        # ── Direction-specific barrier checks ─────────────────────────────
        if direction == "超配":
            if cum_ret >= up_barrier:
                return 1.0, "tp", cum_ret, cal_days
            if cum_ret <= down_barrier:
                return 0.0, "sl", cum_ret, cal_days

        elif direction == "低配":
            # Short position: profit when price falls (past down_barrier)
            if cum_ret <= down_barrier:
                return 1.0, "tp", cum_ret, cal_days   # "tp" = profit target hit for short
            if cum_ret >= up_barrier:
                return 0.0, "sl", cum_ret, cal_days   # "sl" = stopped out

        elif direction == "拦截":
            # Blocking was correct if price fell significantly
            if cum_ret <= down_barrier:
                return 1.0, "tp", cum_ret, cal_days
            if cum_ret >= up_barrier:
                return 0.0, "sl", cum_ret, cal_days

        elif direction == "通过":
            # Letting through was correct if price held up
            if cum_ret >= up_barrier:
                return 1.0, "tp", cum_ret, cal_days
            if cum_ret <= down_barrier:
                return 0.0, "sl", cum_ret, cal_days

        elif direction in ("标配", "中性"):
            # Neutral: large moves in either direction are undesirable
            # Both barriers trigger a fallback to original scoring at that point
            if cum_ret >= up_barrier or cum_ret <= down_barrier:
                return score_accuracy(direction, cum_ret, horizon_key), "time", cum_ret, cal_days
            # else: continue until time barrier

    # Exhausted all available data — treat as time barrier at last available point
    cum_ret  = float(post_prices.iloc[-1]) / p0 - 1.0
    try:
        cal_days = (post_prices.index[-1].date() - entry_date.date()).days
    except AttributeError:
        cal_days = len(post_prices)
    return score_accuracy(direction, cum_ret, horizon_key), "time", cum_ret, cal_days


def verify_pending_decisions(model=None) -> list[dict]:
    """
    Fetch actual price returns for all unverified decisions and score accuracy
    using the Triple-Barrier Method.

    Triple-Barrier replaces the old fixed-window approach (5/40/90 days) with a
    path-based mechanism: whichever of the three barriers is touched FIRST determines
    the score, not the price at an arbitrary future point.

    Barriers per horizon (parameters economically anchored, not tuned):
      季度(3个月): TP=+1σ, SL=-0.7σ (σ scaled to 90-day horizon),  time=90  cal days
      半年(6个月): TP=+1σ, SL=-0.7σ (σ scaled to 180-day horizon), time=180 cal days

    σ is estimated from the 252-day realised vol BEFORE the decision date (no future data).

    Minimum wait before attempting verification (prevents premature checks):
      季度 → 45 cal days · 半年 → 90 cal days

    Reference returns (5d / 20d) are still stored for backward-compat diagnostics,
    but primary accuracy_score now comes from the triple-barrier scorer.
    """
    import yfinance as yf

    # Minimum calendar days before verify attempt (unchanged from previous logic)
    # Prevents triggering yfinance downloads before any signal has had time to play out.
    _HORIZON_MIN_WAIT: dict[str, int] = {
        "季": 45,   # 季度(3个月) — start checking after 1.5 months
        "半": 90,   # 半年(6个月) — start checking after 3 months
        # Legacy backward-compat
        "中": 45,
        "长": 90,
    }
    _DEFAULT_MIN_WAIT = 45

    # Vol lookback: trading days of history before entry used to estimate σ
    _VOL_LOOKBACK_DAYS = 252

    now = datetime.datetime.utcnow()
    results: list[dict] = []

    with SessionFactory() as session:
        pending = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == False,
                (DecisionLog.superseded == False) | (DecisionLog.superseded == None),
            )
            .all()
        )

        # ── Pre-fetch equal-weight universe baseline (once per call) ──────────
        # Used to convert absolute returns to excess returns before scoring.
        # Excess return = ETF return − equal-weight universe return.
        # This removes market beta from the accuracy signal; a bull-market
        # winner that still underperforms its benchmark scores as incorrect.
        _universe_prices: "pd.DataFrame" = pd.DataFrame()
        try:
            _univ_tickers = list(_get_sector_etf_map().values())
            _dec_dates = [
                d.created_at.date() for d in pending if d.created_at
            ]
            if _dec_dates:
                _univ_start = min(_dec_dates) - datetime.timedelta(days=5)
                _univ_end   = max(_dec_dates) + datetime.timedelta(days=200)
                _univ_dl = yf.download(
                    _univ_tickers,
                    start=str(_univ_start),
                    end=str(_univ_end),
                    progress=False,
                    auto_adjust=True,
                )
                if not _univ_dl.empty:
                    _uc = _univ_dl["Close"] if isinstance(_univ_dl.columns, pd.MultiIndex) else _univ_dl
                    if isinstance(_uc, pd.Series):
                        _uc = _uc.to_frame()
                    _uc.index = pd.to_datetime(_uc.index).normalize()
                    _universe_prices = _uc
        except Exception:
            pass  # fall back to absolute returns if universe download fails

        for dec in pending:
            horizon_key = (dec.horizon or "季度(3个月)").strip()[:1]  # "季" / "半" (legacy: "中" / "长")
            min_wait    = _HORIZON_MIN_WAIT.get(horizon_key, _DEFAULT_MIN_WAIT)
            time_barrier_cal = _TB_TIME_BARRIERS.get(horizon_key, _TB_TIME_BARRIERS["季"])

            # Skip if minimum wait has not elapsed
            if (now - dec.created_at).days < min_wait:
                continue

            # Resolve ticker
            ticker = dec.ticker
            if not ticker and dec.sector_name:
                ticker = _get_sector_etf_map().get(dec.sector_name)
            if not ticker:
                continue

            try:
                entry_date  = dec.created_at.date()
                lookback_start = entry_date - datetime.timedelta(days=_VOL_LOOKBACK_DAYS + 30)
                forward_end    = entry_date + datetime.timedelta(days=time_barrier_cal + 15)

                # Single download covering [lookback_start, forward_end]
                full_data = yf.download(
                    ticker,
                    start=str(lookback_start),
                    end=str(forward_end),
                    progress=False,
                    auto_adjust=True,
                )
                if full_data.empty:
                    continue

                # Flatten MultiIndex columns produced by newer yfinance versions
                if isinstance(full_data.columns, pd.MultiIndex):
                    full_data.columns = [c[0] for c in full_data.columns]

                close_all = full_data["Close"].dropna()
                if len(close_all) < 10:
                    continue

                # ── Split pre / post entry ─────────────────────────────────
                pre_mask  = close_all.index.normalize() <  pd.Timestamp(entry_date)
                post_mask = close_all.index.normalize() >= pd.Timestamp(entry_date)

                pre_prices  = close_all[pre_mask]
                post_prices = close_all[post_mask]

                if len(post_prices) < 2:
                    continue

                # ── Compute historical volatility from pre-entry data ──────
                if len(pre_prices) >= 20:
                    log_rets    = pre_prices.pct_change().dropna()
                    daily_std   = float(log_rets.std())
                    ann_vol     = daily_std * math.sqrt(252)
                else:
                    # Insufficient pre-entry data — use conservative fallback
                    ann_vol = 0.20   # 20% annualised as sector-neutral prior
                    logger.debug(
                        "id=%d: insufficient pre-entry data (%d rows), using fallback vol=20%%",
                        dec.id, len(pre_prices),
                    )

                # Clamp vol to sane range [5%, 80%] — prevents degenerate barriers
                ann_vol = max(0.05, min(0.80, ann_vol))
                dec.hist_vol_ann = round(ann_vol, 4)

                p0 = float(post_prices.iloc[0])

                # ── Build excess-return price path (vs equal-weight universe) ──
                # Converts absolute ETF price path to an excess-return pseudo-price
                # so that triple-barrier barriers measure relative performance, not
                # raw bull/bear market returns.  Falls back to absolute if universe
                # data is unavailable.
                post_prices_scored = post_prices   # default
                if not _universe_prices.empty:
                    try:
                        _ew_slice = _universe_prices[
                            _universe_prices.index >= pd.Timestamp(entry_date)
                        ].dropna(how="all")
                        if len(_ew_slice) >= 2:
                            _ew_rets = _ew_slice.pct_change().fillna(0)
                            _ew_cum  = (_ew_rets + 1).cumprod().mean(axis=1) - 1
                            _ew_aligned = _ew_cum.reindex(
                                post_prices.index, method="ffill"
                            ).fillna(0)
                            _etf_cum = post_prices / p0 - 1
                            _excess_cum = _etf_cum - _ew_aligned
                            post_prices_scored = pd.Series(
                                p0 * (1 + _excess_cum.values),
                                index=post_prices.index,
                            )
                    except Exception:
                        pass  # keep absolute fallback

                # ── Reference returns (stored as excess vs universe) ──────────
                r5  = None
                r20 = None
                idx5  = min(4,  len(post_prices_scored) - 1)
                p5    = float(post_prices_scored.iloc[idx5])
                r5    = (p5 - p0) / p0
                dec.actual_return_5d = r5

                if len(post_prices_scored) >= 10:
                    p10 = float(post_prices_scored.iloc[min(9, len(post_prices_scored) - 1)])
                    dec.actual_return_10d = (p10 - p0) / p0

                if len(post_prices_scored) >= 20:
                    p20 = float(post_prices_scored.iloc[min(19, len(post_prices_scored) - 1)])
                    r20 = (p20 - p0) / p0
                    dec.actual_return_20d = r20

                # ── Triple-Barrier primary scoring ─────────────────────────
                # Requires the time barrier worth of data to be available.
                # Minimum number of post-entry trading days that must exist before
                # we consider the decision ready for triple-barrier scoring:
                #   14-day barrier  → ~10 trading days
                #   90-day barrier  → ~63 trading days
                #   180-day barrier → ~125 trading days
                min_trading_days = max(5, int(time_barrier_cal * 252 / 365) - 5)

                if len(post_prices_scored) < min_trading_days:
                    logger.debug(
                        "id=%d: only %d post-entry trading days, need ~%d — deferring",
                        dec.id, len(post_prices_scored), min_trading_days,
                    )
                    continue

                tb_score, barrier_type, barrier_ret, barrier_cal_days = (
                    _compute_triple_barrier_score(
                        direction=dec.direction or "中性",
                        post_prices=post_prices_scored,
                        horizon_key=horizon_key,
                        annualized_vol=ann_vol,
                    )
                )

                # Store barrier metadata for diagnostics
                dec.barrier_hit    = barrier_type
                dec.barrier_days   = barrier_cal_days
                dec.barrier_return = round(barrier_ret, 6)

                # Payoff quality: risk-normalized return at barrier hit.
                # Uses the same ann_vol used for barrier sizing — internally consistent.
                # A direction-adjusted sign: positive = correct direction.
                _holding_years = max(barrier_cal_days, 1) / 365.0
                _vol_scaled    = ann_vol * math.sqrt(_holding_years)
                if _vol_scaled > 0:
                    # Sign-adjust: barrier_ret is already direction-adjusted by
                    # _compute_triple_barrier_score (negative for wrong direction).
                    dec.payoff_quality = round(barrier_ret / _vol_scaled, 4)
                else:
                    dec.payoff_quality = None

                # Use triple-barrier score as the canonical accuracy score
                primary_return     = barrier_ret
                dec.accuracy_score = tb_score
                dec.meta_verdict   = _meta_verdict(tb_score)
                dec.verified       = True
                dec.verified_at    = now

                # ── Regime drift detection ─────────────────────────────────
                # Fetch VIX at the barrier-hit date to determine the regime
                # at the time the prediction was resolved.
                try:
                    _hit_date = entry_date + datetime.timedelta(days=max(0, barrier_cal_days - 1))
                    _vix_dl = yf.download(
                        "^VIX",
                        start=str(_hit_date - datetime.timedelta(days=5)),
                        end=str(_hit_date + datetime.timedelta(days=5)),
                        progress=False, auto_adjust=True,
                    )
                    if not _vix_dl.empty:
                        _vc = _vix_dl["Close"]
                        if isinstance(_vc, pd.DataFrame):
                            _vc = _vc.iloc[:, 0]
                        _vix_val = float(_vc.dropna().iloc[-1])
                        dec.regime_at_verify = _regime_from_vix(_vix_val)
                        dec.regime_drifted = (
                            dec.macro_regime is not None
                            and dec.macro_regime != dec.regime_at_verify
                        )
                except Exception:
                    pass  # non-critical — fields stay NULL

                # ── P1-E LLM Alpha Attribution ─────────────────────────────
                # Compare main (LLM-adjusted) vs quant (pure TSMOM) weights
                # for the same sector on the nearest snapshot_date ≤ decision.
                # llm_weight_alpha = actual_return × (main_weight - quant_weight)
                if dec.sector_name and r20 is not None:
                    try:
                        _dec_date = dec.decision_date or entry_date
                        _main_pos = (
                            session.query(SimulatedPosition)
                            .filter(
                                SimulatedPosition.sector == dec.sector_name,
                                SimulatedPosition.track  == "main",
                                SimulatedPosition.snapshot_date <= _dec_date,
                            )
                            .order_by(SimulatedPosition.snapshot_date.desc())
                            .first()
                        )
                        _quant_pos = (
                            session.query(SimulatedPosition)
                            .filter(
                                SimulatedPosition.sector == dec.sector_name,
                                SimulatedPosition.track  == "quant",
                                SimulatedPosition.snapshot_date <= _dec_date,
                            )
                            .order_by(SimulatedPosition.snapshot_date.desc())
                            .first()
                        )
                        if _main_pos is not None and _quant_pos is not None:
                            _main_w  = _main_pos.actual_weight  or _main_pos.target_weight or 0.0
                            _quant_w = _quant_pos.actual_weight or _quant_pos.target_weight or 0.0
                            dec.llm_weight_alpha = round(r20 * (_main_w - _quant_w), 6)
                            logger.debug(
                                "P1-E alpha: id=%d sector=%s main_w=%.3f quant_w=%.3f "
                                "r20=%.2f%% alpha=%.4f",
                                dec.id, dec.sector_name, _main_w, _quant_w,
                                r20 * 100, dec.llm_weight_alpha,
                            )
                    except Exception:
                        pass  # non-critical — field stays NULL

                logger.info(
                    "Triple-barrier: id=%d sector=%s direction=%s barrier=%s "
                    "days=%d ret=%.2f%% score=%.2f vol=%.1f%%",
                    dec.id, dec.sector_name, dec.direction, barrier_type,
                    barrier_cal_days, barrier_ret * 100, tb_score, ann_vol * 100,
                )

                # Auto-flag anomalies for human review:
                # High-confidence call badly wrong → may be analysis error or black swan
                if (
                    tb_score <= 0.25
                    and (dec.confidence_score or 0) >= 65
                    and not dec.human_label
                ):
                    dec.needs_review = True

                # Check if invalidation conditions were triggered
                if dec.invalidation_conditions and model:
                    inv_check = _check_invalidation(model, dec, primary_return)
                    if inv_check:
                        dec.accuracy_score = 0.5
                        dec.meta_verdict   = "REVIEW"
                        dec.reflection     = f"[失效条件触发] {inv_check}"

                # Only generate LLM reflection for non-trivial outcomes:
                # - skip backtest decisions (Phase-0 batch: hundreds of calls, low marginal value)
                # - skip clearly passing decisions (accuracy >= EXCELLENT) to save API cost;
                #   passing reflections have minimal diagnostic value vs. failures/borderline
                _needs_reflection = (
                    not dec.reflection
                    and not dec.is_backtest
                    and (dec.accuracy_score is None or dec.accuracy_score < EXCELLENT)
                )
                if model and _needs_reflection:
                    dec.reflection = _ai_reflection(model, dec)

                # ── LCS quality audit ─────────────────────────────────────
                if model and dec.direction not in ("中性",):
                    _run_lcs_on_decision(model, dec)

                # ── Failure mode classification ────────────────────────────
                # Runs after LCS so FM-A can read lcs_mirror_passed.
                dec.failure_mode = _classify_failure_mode(dec, session)
                if dec.failure_mode:
                    logger.info(
                        "FailureMode: id=%d sector=%s regime=%s → %s",
                        dec.id, dec.sector_name, dec.macro_regime, dec.failure_mode,
                    )

                # Reverse-validate: feed accuracy signal back into NewsRoutingWeight
                _feedback_routing_weights(dec, session)

                # Update quantitative pattern hit-rate table (Layer 1-2 learning)
                _update_quant_pattern(dec, session)

                # Cross-window horizon mismatch detection
                # Pass r5/r20 as reference; r60 = barrier_ret when barrier=time (long-term proxy)
                r60_ref = barrier_ret if (horizon_key in ("半", "长") and barrier_type == "time") else None
                _check_horizon_mismatch(dec, r5, r20, r60_ref, session)

                results.append({
                    "id":               dec.id,
                    "tab_type":         dec.tab_type,
                    "sector_name":      dec.sector_name,
                    "macro_regime":     dec.macro_regime or "",
                    "ticker":           ticker,
                    "direction":        dec.direction,
                    "horizon":          dec.horizon or "季度(3个月)",
                    "barrier":          barrier_type,
                    "barrier_days":     barrier_cal_days,
                    "return_5d":        r5,
                    "return_20d":       r20,
                    "barrier_return":   barrier_ret,
                    "payoff_quality":   dec.payoff_quality,
                    "llm_weight_alpha": dec.llm_weight_alpha,
                    "accuracy":         tb_score,
                    "verdict":          dec.meta_verdict,
                    "ann_vol":          ann_vol,
                    "created_at":       dec.created_at.strftime("%Y-%m-%d"),
                })
            except Exception as exc:
                logger.warning("Could not verify decision id=%s: %s", dec.id, exc)
                continue

        session.commit()

    # After all verifications, trigger skill recompression for affected cells
    if model:
        updated_cells = {(r["sector_name"], r.get("macro_regime", "")) for r in results}
        for sector, regime in updated_cells:
            if sector and regime:
                maybe_update_skill(model, sector, regime)

    # Auto-trigger Meta-Agent systematic bias analysis.
    # run_meta_agent_analysis() is pure statistics (no LLM calls); its output
    # populates LearningLog which get_historical_context() already consumes.
    # min_samples=25 gate prevents premature pattern discovery on sparse data.
    if results:
        meta_patterns = run_meta_agent_analysis(min_samples=25)
        if meta_patterns:
            logger.info(
                "MetaAgent auto-run: %d new pattern(s) written to LearningLog",
                len(meta_patterns),
            )

    return results


_SHORT_TERM_SIGNAL_MARKERS = ("RSI", "Bollinger", "MACD", "KDJ", "超买", "超卖", "均线", "布林")


def _classify_failure_mode(dec: DecisionLog, session) -> str:
    """
    Assign a structured failure mode code to a verified prediction that clearly
    failed (accuracy_score < 0.5).  Returns "" for passing decisions.

    Priority order (first match wins):
      FM-A  Logic degradation   — LCS mirror test failed; conclusion was input-independent
      FM-B  Overconfidence      — confidence ≥ 85 but accuracy < 0.5
      FM-C  Signal contamination — short-term technical signal in attribution drivers
      FM-D  Regime misclassif. — same sector × regime produced 3 consecutive failures

    Inspired by BrainAlpha's Failure Mode Taxonomy.
    """
    if dec.accuracy_score is None or dec.accuracy_score >= 0.5:
        return ""

    # FM-A: LCS mirror test explicitly failed → conclusion is input-independent
    if dec.lcs_mirror_passed is False:
        return "FM-A"

    # FM-B: Stated confidence ≥ 85 but prediction clearly wrong
    if (dec.confidence_score or 0) >= 85:
        return "FM-B"

    # FM-C: Short-term technical signals leaked into attribution drivers
    _drivers_text = ""
    if dec.signal_attribution:
        try:
            _attr = (
                json.loads(dec.signal_attribution)
                if isinstance(dec.signal_attribution, str)
                else dec.signal_attribution
            )
            _drivers_text = str(_attr.get("drivers", ""))
        except Exception:
            _drivers_text = str(dec.signal_attribution)
    if any(sig in _drivers_text for sig in _SHORT_TERM_SIGNAL_MARKERS):
        return "FM-C"

    # FM-D: 2 most recent prior verified decisions in same sector × regime also failed
    if dec.sector_name and dec.macro_regime:
        prior = (
            session.query(DecisionLog.accuracy_score)
            .filter(
                DecisionLog.sector_name  == dec.sector_name,
                DecisionLog.macro_regime == dec.macro_regime,
                DecisionLog.verified     == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.id           != dec.id,
            )
            .order_by(DecisionLog.verified_at.desc())
            .limit(2)
            .all()
        )
        if len(prior) >= 2 and all(r[0] < 0.5 for r in prior):
            return "FM-D"

    return ""

def _run_lcs_on_decision(model, dec: DecisionLog) -> None:
    """
    Run the full LCS audit on a verified decision and persist results.

    Called inside verify_pending_decisions() once accuracy_score is set.
    Writes lcs_score, lcs_passed, and component flags to the record in-place.
    The session.commit() in the calling loop persists these fields.

    Silently skips when:
      - direction is neutral / unknown (no meaningful mirror to construct)
      - quant_metrics JSON is missing or malformed
      - any model call fails (LCS columns remain NULL; gates treat NULL as passing)
    """
    try:
        from engine.lcs import run_full_lcs_audit
    except ImportError:
        logger.warning("engine.lcs not found; LCS audit skipped")
        return

    if not dec.direction or dec.direction in ("中性",):
        return

    # Reconstruct quant_metrics from stored JSON
    qm: dict = {}
    if dec.quant_metrics:
        try:
            qm = json.loads(dec.quant_metrics)
        except Exception:
            pass

    vix    = dec.vix_level or 20.0
    sector = dec.sector_name or (dec.ticker or "未知板块")
    regime = dec.macro_regime or "未知制度"
    conc   = (dec.ai_conclusion or "")[:300]

    try:
        result = run_full_lcs_audit(
            model=model,
            sector=sector,
            original_direction=dec.direction,
            vix=vix,
            macro_regime=regime,
            quant_metrics=qm,
            conclusion_text=conc,
            run_cross_cycle=bool(conc),
        )
        dec.lcs_score         = result.lcs_score
        dec.lcs_passed        = result.lcs_passed
        dec.lcs_mirror_passed = result.mirror_passed
        dec.lcs_noise_passed  = result.noise_passed
        dec.lcs_cross_passed  = result.cross_cycle_passed
        dec.lcs_notes         = result.notes[:500] if result.notes else None
        logger.info(
            "LCS stored: id=%d sector=%s score=%.2f passed=%s",
            dec.id, sector, result.lcs_score, result.lcs_passed,
        )
    except Exception as exc:
        logger.warning("LCS audit failed for decision id=%d: %s", dec.id, exc)


def _check_horizon_mismatch(
    dec: DecisionLog,
    r5: float | None,
    r20: float | None,
    r60: float | None,
    session,
) -> None:
    """
    Compare returns across all available windows to detect if the AI's declared
    horizon matches where value actually materialised.

    Logic:
      - For each available window (5d / 20d / 60d), compute a direction-alignment
        score: positive if return matches declared direction, negative if it doesn't.
      - If the best-scoring window differs from the declared horizon, write a
        'horizon_mismatch' LearningLog so Meta-Agent can surface the pattern.

    No-op when:
      - Direction is neutral / unknown
      - Fewer than 2 windows have data (can't compare)
      - The declared horizon already matches the best window (no mismatch)
    """
    if not dec.direction or dec.direction in ("中性", "标配"):
        return
    # Test-set gate
    _dec_date = dec.decision_date or (dec.created_at.date() if dec.created_at else None)
    if _dec_date is not None and _dec_date >= TRAIN_TEST_CUTOFF:
        return
    # LCS quality gate: if LCS explicitly failed, block write-back
    if dec.lcs_passed is False:
        logger.debug(
            "LearningLog horizon write blocked (LCS failed): sector=%s score=%.2f",
            dec.sector_name, dec.lcs_score or 0,
        )
        return

    expected_up = dec.direction == "超配"

    def alignment(ret: float) -> float:
        """Signed alignment: positive = return agrees with direction."""
        return ret if expected_up else -ret

    windows: dict[str, float] = {}
    if r20 is not None: windows["季度(3个月)"] = r20
    if r60 is not None: windows["半年(6个月)"] = r60

    if len(windows) < 2:
        return

    best_window = max(windows, key=lambda k: alignment(windows[k]))

    # Normalise declared horizon to one of the window keys
    raw = (dec.horizon or "季度(3个月)").strip()
    if "半" in raw or "长" in raw:
        declared_key = "半年(6个月)"
    else:
        declared_key = "季度(3个月)"

    if best_window == declared_key:
        return   # no mismatch — horizon was correct

    # Build a readable summary of returns
    parts = [f"{k}:{v:+.1%}" for k, v in windows.items()]
    ret_summary = "  ".join(parts)

    description = (
        f"horizon_mismatch · {dec.sector_name or dec.tab_type} · {dec.macro_regime or '未知制度'}\n"
        f"声明: {declared_key}  实际表现最佳窗口: {best_window}  方向: {dec.direction}\n"
        f"收益对比: {ret_summary}\n"
        f"建议: 在【{dec.macro_regime or '当前制度'}】下，{dec.sector_name or '该板块'}"
        f"的配置逻辑倾向于在{best_window}维度兑现，"
        f"下次分析请优先声明 horizon={best_window}"
    )

    # Check if a similar pattern already exists (avoid duplicate rows)
    existing = (
        session.query(LearningLog)
        .filter(
            LearningLog.pattern_type == "horizon_mismatch",
            LearningLog.sector_name  == dec.sector_name,
            LearningLog.macro_regime == dec.macro_regime,
            LearningLog.applied      == False,
        )
        .first()
    )

    if existing:
        # Update sample count and refresh description if direction is consistent
        existing.sample_count += 1
        existing.description   = description   # overwrite with latest evidence
    else:
        session.add(LearningLog(
            macro_regime   = dec.macro_regime,
            sector_name    = dec.sector_name,
            pattern_type   = "horizon_mismatch",
            description    = description,
            sample_count   = 1,
            accuracy_impact= alignment(windows[best_window]) - alignment(windows[declared_key]),
        ))

    logger.debug(
        "Horizon mismatch: sector=%s declared=%s best=%s",
        dec.sector_name, declared_key, best_window,
    )


def _feedback_routing_weights(dec: DecisionLog, session) -> None:
    """
    Reverse-validate news routing weights after a decision is verified.

    Logic:
      - accuracy >= EXCELLENT (0.75): news categories used were predictive → nudge weight +0.05
      - accuracy < 0.5 (wrong direction): categories misled the agent → nudge weight -0.05
      - 0.5 <= accuracy < EXCELLENT: ambiguous signal → no update

    Only fires when:
      - news_categories_used is populated (XAI block was parsed)
      - sector_name and macro_regime are known (needed as weight key)

    Train/test isolation: test-set decisions must not adjust routing weights.

    ── Horizon gate ──────────────────────────────────────────────────────────────
    News has a signal half-life of ~3-14 days. Reinforcing news routing weights
    based on 90- or 180-day triple-barrier outcomes is temporally incoherent:
    the news that was current at decision time bears no stable causal relationship
    to a price move verified 3-6 months later. Doing so would create spurious
    feedback where news categories that co-occurred with (macro-driven) correct
    calls get systematically up-weighted — polluting the routing weights with
    foreknowledge-era correlations from the 2015-2023 backtest dataset.

    Since all current horizons are 季度(90d) or 半年(180d), this gate effectively
    disables news routing weight updates until a short-term horizon is reintroduced.
    Static default weights are preferable to spuriously-trained dynamic weights.
    """
    if not dec.news_categories_used:
        return
    if not dec.sector_name or not dec.macro_regime:
        return
    if dec.accuracy_score is None:
        return

    # Horizon gate: news signal half-life (~14 days) is far shorter than
    # 季度/半年 verification windows — reinforcing categories here is spurious.
    _h_raw = (dec.horizon or "季度(3个月)").strip()[:1]
    if _h_raw in ("季", "半", "中", "长"):
        logger.debug(
            "NewsRoutingWeight write suppressed (horizon mismatch): "
            "sector=%s horizon=%s — news half-life << 90d verification window",
            dec.sector_name, dec.horizon or "季度",
        )
        return

    # Test-set gate
    _dec_date = dec.decision_date or (dec.created_at.date() if dec.created_at else None)
    if _dec_date is not None and _dec_date >= TRAIN_TEST_CUTOFF:
        logger.debug(
            "NewsRoutingWeight write blocked (test set): sector=%s date=%s",
            dec.sector_name, _dec_date,
        )
        return
    # LCS quality gate
    if dec.lcs_passed is False:
        logger.debug(
            "NewsRoutingWeight write blocked (LCS failed): sector=%s score=%.2f",
            dec.sector_name, dec.lcs_score or 0,
        )
        return

    try:
        categories: list[str] = json.loads(dec.news_categories_used)
    except Exception:
        return

    if not categories:
        return

    # Determine nudge direction
    if dec.accuracy_score >= EXCELLENT:
        nudge = +0.05   # these categories helped → reward
    elif dec.accuracy_score < 0.5:
        nudge = -0.05   # these categories misled → penalise
    else:
        return          # inconclusive — no update

    for category in categories:
        row = (
            session.query(NewsRoutingWeight)
            .filter(
                NewsRoutingWeight.sector_name   == dec.sector_name,
                NewsRoutingWeight.macro_regime  == dec.macro_regime,
                NewsRoutingWeight.news_category == category,
            )
            .first()
        )
        if row:
            row.weight       = round(max(0.1, min(1.0, row.weight + nudge)), 4)
            row.sample_count += 1
            row.updated_at   = datetime.datetime.utcnow()
        else:
            # Initialise from default, then apply nudge
            default = _DEFAULT_WEIGHTS.get(dec.sector_name, {}).get(category, 0.5)
            session.add(NewsRoutingWeight(
                sector_name  = dec.sector_name,
                macro_regime = dec.macro_regime,
                news_category= category,
                weight       = round(max(0.1, min(1.0, default + nudge)), 4),
                sample_count = 1,
            ))

    logger.debug(
        "Routing weight feedback: sector=%s regime=%s categories=%s nudge=%+.2f acc=%.2f",
        dec.sector_name, dec.macro_regime, categories, nudge, dec.accuracy_score,
    )


def _check_invalidation(model, dec: DecisionLog, actual_return: float) -> str:
    """
    Ask the AI whether the stored invalidation conditions were likely triggered
    given the observed return over the holding period.
    Returns a short explanation string if triggered, empty string if not.
    """
    if not dec.invalidation_conditions:
        return ""
    prompt = (
        f"以下是一个历史板块配置决策的失效条件评估：\n\n"
        f"原始建议: {dec.direction}  板块: {dec.sector_name or dec.ticker}\n"
        f"日期: {dec.created_at.strftime('%Y-%m-%d')}  VIX: {dec.vix_level}\n"
        f"失效条件: {dec.invalidation_conditions}\n"
        f"持仓期间实际收益: {actual_return:+.2%}\n\n"
        "问题：根据持仓期间的实际表现，上述失效条件是否有可能被触发？\n"
        "如果是，请用一句话说明理由。如果否，直接回答'未触发'。"
    )
    try:
        resp = model.generate_content(prompt).text.strip()
        if "未触发" in resp or "no" in resp.lower():
            return ""
        return resp[:150]
    except Exception:
        return ""


def _ai_reflection(model, dec: DecisionLog) -> str:
    flag = "✅" if dec.accuracy_score >= EXCELLENT else ("⚠️" if dec.accuracy_score >= 0.5 else "❌")

    # Determine proxy window label and caveat based on declared horizon
    horizon_raw = (dec.horizon or "").strip()
    if "半" in horizon_raw or "长" in horizon_raw:
        proxy_label = "90日"
        proxy_caveat = (
            f"⚠️ 注意：声明持仓周期为【{horizon_raw}】，当前评分基于三重障碍法（时间障碍180天），"
            "若障碍为时间触发，方向可能仍在兑现途中。请区分「时机偏早」与「判断错误」，"
            "避免因中期噪声错误否定半年结构性逻辑。"
        )
    else:
        proxy_label = "45日"
        proxy_caveat = (
            f"⚠️ 注意：声明持仓周期为【{horizon_raw}】，当前评分基于三重障碍法（时间障碍90天），"
            "若障碍为时间触发，请区分「催化剂尚未触发」与「基本面判断失误」，"
            "不应以短于季度的代理窗口全面否定季度逻辑。"
        )

    ret_str = ""
    if dec.actual_return_20d is not None:
        ret_str = f"{proxy_label}代理收益 {dec.actual_return_20d:+.2%}"
    elif dec.actual_return_5d is not None:
        ret_str = f"5日收益 {dec.actual_return_5d:+.2%}"

    proxy_section = f"\n\n{proxy_caveat}" if proxy_caveat else ""

    prompt = (
        f"你是一名量化策略反思分析师。请对以下历史决策进行简短反思（不超过120字）：\n\n"
        f"日期: {dec.created_at.strftime('%Y-%m-%d')}  "
        f"板块: {dec.sector_name or dec.ticker}  "
        f"宏观制度: {dec.macro_regime or '未记录'}\n"
        f"配置方向: {dec.direction}  持仓周期: {horizon_raw or '未记录'}  "
        f"置信度: {dec.confidence_score or '未记录'}  VIX: {dec.vix_level}\n"
        f"经济逻辑: {(dec.economic_logic or '')[:150]}\n\n"
        f"实际结果: {ret_str} {flag}  准确度: {dec.accuracy_score:.2f}/1.0"
        f"{proxy_section}\n\n"
        "请分析：① 判断对/错的关键因素（区分时机问题与逻辑问题）"
        " ② 下次类似情景应注意什么。机构语气，不超过120字。"
    )
    try:
        return model.generate_content(prompt).text.strip()
    except Exception:
        return ""


# ── Human-in-the-loop review ──────────────────────────────────────────────────

def get_records_needing_review() -> list[dict]:
    """
    Return verified decisions flagged for human review (unlabelled anomalies only).
    Criteria: accuracy_score ≤ 0.25 AND confidence_score ≥ 65 AND no human_label yet.
    """
    with SessionFactory() as session:
        records = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.needs_review == True,
                DecisionLog.human_label.is_(None),
            )
            .order_by(DecisionLog.created_at.desc())
            .all()
        )
        return [
            {
                "id":               r.id,
                "sector_name":      r.sector_name or r.ticker or "—",
                "direction":        r.direction or "—",
                "horizon":          r.horizon or "—",
                "confidence_score": r.confidence_score,
                "accuracy_score":   r.accuracy_score,
                "vix_level":        r.vix_level,
                "macro_regime":     r.macro_regime or "—",
                "created_at":       r.created_at,
                "reflection":       r.reflection or "",
            }
            for r in records
        ]


def set_human_label(record_id: int, label: str) -> bool:
    """
    Apply human label to a decision record.

    Post-verification labels (anomaly resolution):
      "black_swan"     — outcome was unforeseeable; excluded from meta-agent analysis
      "analysis_error" — clear reasoning fault identified in retrospect

    Pre-verification labels (human pre-scoring before outcome is known):
      "pre_strong"     — user judges thesis as logically sound
      "pre_uncertain"  — user has reservations about the thesis
      "pre_poor"       — user identifies a clear flaw in the reasoning

    Pre-verification labels enable independent correlation analysis:
    does human pre-scoring correlate with Triple-Barrier outcomes better
    or worse than LCS? This breaks the AI circular validation loop.

    Returns True on success.
    """
    _valid = {"black_swan", "analysis_error", "pre_strong", "pre_uncertain", "pre_poor"}
    if label not in _valid:
        return False
    with SessionFactory() as session:
        rec = session.get(DecisionLog, record_id)
        if not rec:
            return False
        rec.human_label  = label
        # Post-verification labels resolve the review flag; pre-verification labels do not.
        if label in ("black_swan", "analysis_error"):
            rec.needs_review = False
        session.commit()
    return True


# ── Meta-Agent: systematic bias analysis ──────────────────────────────────────

def run_meta_agent_analysis(min_samples: int = 25) -> list[dict]:
    """
    Analyse verified decisions to find systematic biases.
    Returns a list of bias patterns, and saves them to LearningLog.

    Checks:
      - Per (sector × macro_regime): is accuracy below MIN_ACCEPTABLE?
      - Confidence calibration: are high-confidence calls actually more accurate?
      - News category effectiveness: which categories correlate with accuracy?

    Train/test isolation: only training-set decisions (decision_date < TRAIN_TEST_CUTOFF)
    are used. Test-set outcomes must not influence pattern discovery.

    LCS gate: decisions with lcs_passed=False are excluded from meta-analysis.
    LCS is a gate-only mechanism throughout the system — it determines whether a
    decision enters the learning tables, but NEVER acts as a weighting factor.
    All learning weights (accuracy_rate, avg_accuracy, etc.) derive exclusively
    from empirical Triple-Barrier outcomes.
    """
    with SessionFactory() as session:
        verified = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                (DecisionLog.human_label != "black_swan") | DecisionLog.human_label.is_(None),
                # Backtest contamination gate: LLM has foreknowledge of all historical
                # dates in its training data — exclude to prevent bias amplification.
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                # Test-set gate: only training data drives meta-agent learning
                func.coalesce(
                    DecisionLog.decision_date,
                    func.date(DecisionLog.created_at),
                ) < TRAIN_TEST_CUTOFF,
                # LCS gate: logically inconsistent decisions must not feed bias patterns.
                # NULL = LCS not yet run → allowed through (pre-LCS era records).
                (DecisionLog.lcs_passed != False) | DecisionLog.lcs_passed.is_(None),
            )
            .all()
        )

        if len(verified) < min_samples:
            return []

        patterns: list[dict] = []

        # ── 1. Sector × Regime accuracy ───────────────────────────────────────
        cell: dict[tuple, list[float]] = {}
        for d in verified:
            key = (d.sector_name or "ALL", d.macro_regime or "UNKNOWN")
            cell.setdefault(key, []).append(d.accuracy_score)

        for (sector, regime), scores in cell.items():
            if len(scores) < 5:
                continue
            avg = sum(scores) / len(scores)
            se = math.sqrt(avg * (1 - avg) / len(scores))
            ci_lower = avg - 1.96 * se   # 95% CI lower bound
            if ci_lower < MIN_ACCEPTABLE:
                desc = (
                    f"在 [{regime}] 环境下对 [{sector}] 的判断准确率偏低 "
                    f"({avg:.0%}, 95%CI下界={ci_lower:.0%}, 样本={len(scores)})。"
                    f"建议：提高不确定性描述，降低置信度阈值。"
                )
                patterns.append({
                    "sector": sector, "regime": regime,
                    "type": "bias", "description": desc,
                    "samples": len(scores), "impact": MIN_ACCEPTABLE - ci_lower,
                })

        # ── 2. Confidence calibration ─────────────────────────────────────────
        high_conf = [d for d in verified if d.confidence_score and d.confidence_score >= 75]
        low_conf  = [d for d in verified if d.confidence_score and d.confidence_score < 60]
        if len(high_conf) >= 3 and len(low_conf) >= 3:
            hc_avg = sum(d.accuracy_score for d in high_conf) / len(high_conf)
            lc_avg = sum(d.accuracy_score for d in low_conf)  / len(low_conf)
            if hc_avg < lc_avg + 0.05:
                desc = (
                    f"置信度虚高：高置信度决策 ({hc_avg:.0%}) 并未显著优于低置信度 ({lc_avg:.0%})。"
                    f"建议：在 prompt 中要求 agent 更严格区分置信度等级。"
                )
                patterns.append({
                    "sector": None, "regime": None,
                    "type": "calibration", "description": desc,
                    "samples": len(high_conf) + len(low_conf),
                    "impact": lc_avg - hc_avg,
                })

        # ── 3. Signal attribution bias ────────────────────────────────────────
        # Check if over-reliance on a single signal correlates with poor accuracy
        for sig_key, sig_label in [
            ("macro_confidence", "宏观信号"),
            ("news_confidence",  "新闻信号"),
            ("technical_confidence", "技术信号"),
        ]:
            high_sig = [
                d for d in verified
                if getattr(d, sig_key) and getattr(d, sig_key) >= 80
            ]
            if len(high_sig) >= 3:
                avg = sum(d.accuracy_score for d in high_sig) / len(high_sig)
                if avg < MIN_ACCEPTABLE:
                    desc = (
                        f"过度依赖{sig_label}：当{sig_label}置信度≥80时，"
                        f"整体准确率仅 {avg:.0%}（样本={len(high_sig)}）。"
                        f"建议：要求 agent 在{sig_label}高权重时主动寻求其他信号交叉验证。"
                    )
                    patterns.append({
                        "sector": None, "regime": None,
                        "type": "signal_bias", "description": desc,
                        "samples": len(high_sig), "impact": MIN_ACCEPTABLE - avg,
                    })

        # ── 3. High-performing cells (strengths to reinforce) ─────────────────
        for (sector, regime), scores in cell.items():
            if len(scores) < 5:
                continue
            avg = sum(scores) / len(scores)
            se = math.sqrt(avg * (1 - avg) / len(scores))
            ci_lower = avg - 1.96 * se   # require lower bound also above threshold
            if ci_lower >= EXCELLENT:
                desc = (
                    f"在 [{regime}] 环境下对 [{sector}] 的判断准确率良好 "
                    f"({avg:.0%}, 95%CI下界={ci_lower:.0%}, 样本={len(scores)})。"
                    f"建议：维持当前分析框架，本情景下逻辑结构有效。"
                )
                patterns.append({
                    "sector": sector, "regime": regime,
                    "type": "strength", "description": desc,
                    "samples": len(scores), "impact": ci_lower - EXCELLENT,
                })

        # ── 4. Counter-evidence gate: inject adversarial prompt when data is rich ─
        # Activated only after ≥10 verified decisions to avoid premature skepticism
        _COUNTER_EVIDENCE_THRESHOLD = 10
        if len(verified) >= _COUNTER_EVIDENCE_THRESHOLD:
            _counter_prompt = (
                "\n⚠ 反向证据提示：请在应用此规律前，主动寻找反例——"
                "是否存在同等条件下预测准确的案例？"
                "该规律是否可能由短期市场异常驱动而非结构性因素？"
            )
            for p in patterns:
                p["description"] += _counter_prompt

        now = datetime.datetime.utcnow()

        # ── 5a. Evidence-based invalidation ───────────────────────────────────
        # For each active bias/strength pattern, recheck accuracy with latest data.
        # If a bias pattern's cell has recovered above MIN_ACCEPTABLE → resolve it.
        # If a strength pattern's cell has dropped below EXCELLENT → resolve it.
        active_stored = (
            session.query(LearningLog)
            .filter(LearningLog.applied == False)
            .all()
        )
        for lp in active_stored:
            if lp.pattern_type not in ("bias", "strength"):
                continue
            if not lp.sector_name or not lp.macro_regime:
                continue
            cell_key = (lp.sector_name, lp.macro_regime)
            cell_scores = cell.get(cell_key)
            if not cell_scores or len(cell_scores) < 5:
                continue
            cell_avg = sum(cell_scores) / len(cell_scores)
            cell_se  = math.sqrt(cell_avg * (1 - cell_avg) / len(cell_scores))
            cell_ci_lower = cell_avg - 1.96 * cell_se
            if lp.pattern_type == "bias" and cell_ci_lower >= MIN_ACCEPTABLE:
                # Bias resolved: accuracy has genuinely improved
                lp.applied    = True
                lp.applied_at = now
                lp.description = (
                    f"[已解决] {lp.description} "
                    f"→ 最新数据显示准确率已回升至 {cell_avg:.0%} "
                    f"(CI下界={cell_ci_lower:.0%}, 样本={len(cell_scores)})，偏差已修正。"
                )
            elif lp.pattern_type == "strength" and cell_ci_lower < EXCELLENT:
                # Strength no longer holds
                lp.applied    = True
                lp.applied_at = now
                lp.description = (
                    f"[已失效] {lp.description} "
                    f"→ 最新数据显示准确率已下降至 {cell_avg:.0%} "
                    f"(CI下界={cell_ci_lower:.0%}, 样本={len(cell_scores)})，优势不再显著。"
                )

        # ── 5b. Regime dormancy detection ─────────────────────────────────────
        # If a regime hasn't appeared in verified decisions for 180+ days,
        # mark its patterns dormant (silenced but preserved). Revive if it returns.
        _DORMANCY_DAYS = 180
        regime_last_seen: dict[str, datetime.datetime] = {}
        for d in verified:
            r = d.macro_regime or "UNKNOWN"
            ts = d.verified_at or d.created_at
            if ts and (r not in regime_last_seen or ts > regime_last_seen[r]):
                regime_last_seen[r] = ts

        for lp in active_stored:
            if lp.applied or not lp.macro_regime:
                continue
            last = regime_last_seen.get(lp.macro_regime)
            if last is None:
                continue
            age_days = (now - last).days
            if age_days > _DORMANCY_DAYS and not lp.dormant:
                lp.dormant = True   # silence but keep
            elif age_days <= _DORMANCY_DAYS and lp.dormant:
                lp.dormant = False  # regime returned → revive

        # ── 6. Save new patterns to LearningLog ───────────────────────────────
        for p in patterns:
            existing = (
                session.query(LearningLog)
                .filter(
                    LearningLog.sector_name == p["sector"],
                    LearningLog.macro_regime == p["regime"],
                    LearningLog.pattern_type == p["type"],
                    LearningLog.applied == False,
                )
                .first()
            )
            if existing:
                existing.description     = p["description"]
                existing.sample_count    = p["samples"]
                existing.accuracy_impact = p.get("impact")
                existing.dormant         = False  # refresh reactivates dormant pattern
            else:
                session.add(LearningLog(
                    macro_regime=p["regime"],
                    sector_name=p["sector"],
                    pattern_type=p["type"],
                    description=p["description"],
                    sample_count=p["samples"],
                    accuracy_impact=p.get("impact"),
                ))

        session.commit()

    # Update spillover weights alongside bias patterns (same data pass)
    update_spillover_weights()

    return patterns


def get_permutation_report(n_permutations: int = 10_000) -> list[dict]:
    """
    Run block-bootstrap permutation tests for every (sector × regime) cell
    in the training set, returning structured results per cell.

    Three distinct outcome states per cell:
      "insufficient_data" — n < 50; test not run; do NOT read as "not significant"
      "not_significant"   — test ran, p >= adjusted threshold
      "significant"       — test ran, p < adjusted threshold

    The adjusted threshold is computed via Romano-Wolf (16 sectors, α = 0.05).
    Cells with n < _MIN_N_PERMUTATION show progress toward the sample threshold,
    so the user can track how far away each cell is from a meaningful test.

    Only training-set decisions (< TRAIN_TEST_CUTOFF) that pass the LCS gate
    are included, for the same reasons as run_meta_agent_analysis().

    n_permutations: 10_000 recommended for stable p near 0.003 threshold.
                    Use 1_000 for faster dev-mode checks.
    """
    from engine.lcs import (
        compute_permutation_p_value,
        bonferroni_adjusted_threshold,
        _MIN_N_PERMUTATION,
    )

    with SessionFactory() as session:
        records = (
            session.query(
                DecisionLog.sector_name,
                DecisionLog.macro_regime,
                DecisionLog.accuracy_score,
                DecisionLog.created_at,
            )
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                (DecisionLog.human_label != "black_swan") | DecisionLog.human_label.is_(None),
                func.coalesce(
                    DecisionLog.decision_date,
                    func.date(DecisionLog.created_at),
                ) < TRAIN_TEST_CUTOFF,
                (DecisionLog.lcs_passed != False) | DecisionLog.lcs_passed.is_(None),
            )
            .order_by(DecisionLog.created_at.asc())   # chronological — required for block bootstrap
            .all()
        )

    # Group by (sector × regime), preserving chronological order
    cells: dict[tuple[str, str], list[float]] = {}
    for r in records:
        key = (r.sector_name or "ALL", r.macro_regime or "UNKNOWN")
        cells.setdefault(key, []).append(r.accuracy_score)

    if not cells:
        return []

    # Use actual cell count for threshold (more precise than hard-coded 16)
    n_cells   = len(cells)
    threshold = bonferroni_adjusted_threshold(n_tests=max(n_cells, 1))

    results = []
    for (sector, regime), scores in sorted(cells.items()):
        res = compute_permutation_p_value(
            accuracy_scores    = scores,
            n_permutations     = n_permutations,
            adjusted_threshold = threshold,
            sector             = sector,
            regime             = regime,
        )
        results.append({
            "sector":             sector,
            "regime":             regime,
            "n":                  res.n_samples,
            "n_needed":           res.n_needed,
            "progress_pct":       res.progress_pct,
            "observed_accuracy":  res.observed_accuracy,
            "p_value":            res.p_value,
            "status":             res.status,
            "threshold":          res.adjusted_threshold,
            "passed":             res.passed,
        })

    return results


def get_failure_mode_stats() -> dict:
    """
    Return a breakdown of failure modes across all verified decisions.

    Returns a dict with:
      total_verified   — total verified decisions
      total_failed     — decisions with accuracy_score < 0.5
      by_mode          — {FM code: count} for labelled failures
      unlabelled       — failures with no FM code assigned (accuracy < 0.5 but no FM match)
      by_sector        — {sector: {FM code: count}} for heatmap display
    """
    _FM_LABELS = {
        "FM-A": "逻辑退化",
        "FM-B": "过度自信",
        "FM-C": "信号污染",
        "FM-D": "制度误判",
    }
    with SessionFactory() as session:
        rows = (
            session.query(
                DecisionLog.sector_name,
                DecisionLog.failure_mode,
                DecisionLog.accuracy_score,
            )
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
            )
            .all()
        )

    total_verified = len(rows)
    total_failed   = sum(1 for r in rows if r.accuracy_score < 0.5)
    by_mode: dict[str, int] = {}
    by_sector: dict[str, dict[str, int]] = {}
    unlabelled = 0

    for r in rows:
        if r.accuracy_score >= 0.5:
            continue
        fm = r.failure_mode or ""
        sector = r.sector_name or "未知"
        if fm:
            by_mode[fm]  = by_mode.get(fm, 0) + 1
            if sector not in by_sector:
                by_sector[sector] = {}
            by_sector[sector][fm] = by_sector[sector].get(fm, 0) + 1
        else:
            unlabelled += 1

    return {
        "total_verified": total_verified,
        "total_failed":   total_failed,
        "by_mode":        by_mode,
        "fm_labels":      _FM_LABELS,
        "unlabelled":     unlabelled,
        "by_sector":      by_sector,
    }


def get_learning_patterns(
    sector_name: str = "",
    macro_regime: str = "",
    applied: bool = False,
    cutoff_date=None,
    exclude_backtest: bool = True,
) -> list[dict]:
    """Return unapplied (or applied) learning patterns for prompt injection.

    cutoff_date: if set, only patterns created before this date are returned
                 (temporal isolation for walk-forward backtest).
    """
    with SessionFactory() as session:
        q = session.query(LearningLog).filter(
            LearningLog.applied == applied,
            # dormant=NULL (legacy rows) treated as not dormant
            (LearningLog.dormant == False) | (LearningLog.dormant == None),
        )
        if sector_name:
            q = q.filter(
                (LearningLog.sector_name == sector_name) |
                (LearningLog.sector_name == None)
            )
        if macro_regime:
            q = q.filter(
                (LearningLog.macro_regime == macro_regime) |
                (LearningLog.macro_regime == None)
            )
        if cutoff_date is not None:
            q = q.filter(LearningLog.created_at < datetime.datetime.combine(
                cutoff_date, datetime.time.min
            ))
        rows = q.order_by(LearningLog.created_at.desc()).limit(10).all()
        return [
            {
                "id":          r.id,
                "sector":      r.sector_name,
                "regime":      r.macro_regime,
                "type":        r.pattern_type,
                "description": r.description,
                "samples":     r.sample_count,
            }
            for r in rows
        ]


def get_learning_log_raw(limit: int = 200) -> list[dict]:
    """
    Return ALL LearningLog entries (active + applied + dormant) with full metadata,
    sorted newest-first. Used by the Admin semantic drift monitor to let operators
    scan rule text for vagueness or contradiction over time.
    """
    with SessionFactory() as session:
        rows = (
            session.query(LearningLog)
            .order_by(LearningLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":          r.id,
                "created_at":  r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—",
                "sector":      r.sector_name or "全部板块",
                "regime":      r.macro_regime or "全部周期",
                "type":        r.pattern_type,
                "description": r.description,
                "samples":     r.sample_count or 0,
                "applied":     bool(r.applied),
                "dormant":     bool(r.dormant),
            }
            for r in rows
        ]


_STOP_FLAG = Path(__file__).parent.parent / "backtest_stop.flag"


def create_backtest_session(
    start_date: str, end_date: str,
    sectors: list[str], freq: str, total_pairs: int,
) -> int:
    """Create a new backtest session record and return its id."""
    with SessionFactory() as session:
        row = BacktestSession(
            start_date=start_date, end_date=end_date,
            sectors=json.dumps(sectors, ensure_ascii=False),
            freq=freq, total_pairs=total_pairs,
            done_pairs=0, status="running",
        )
        session.add(row)
        session.commit()
        return row.id


def update_backtest_session(session_id: int, done_pairs: int, status: str) -> None:
    with SessionFactory() as session:
        row = session.query(BacktestSession).filter(BacktestSession.id == session_id).first()
        if row:
            row.done_pairs  = done_pairs
            row.status      = status
            row.updated_at  = datetime.datetime.utcnow()
            session.commit()


def get_active_backtest_session() -> dict | None:
    """Return the most recent interrupted/paused session, or None if all complete."""
    with SessionFactory() as session:
        row = (
            session.query(BacktestSession)
            .filter(BacktestSession.status.in_(["paused", "quota_hit", "running"]))
            .order_by(BacktestSession.updated_at.desc())
            .first()
        )
        if not row:
            return None
        return {
            "id":          row.id,
            "start_date":  row.start_date,
            "end_date":    row.end_date,
            "sectors":     json.loads(row.sectors),
            "freq":        row.freq,
            "total_pairs": row.total_pairs,
            "done_pairs":  row.done_pairs,
            "remaining":   row.total_pairs - row.done_pairs,
            "status":      row.status,
        }


def get_system_config(key: str, default: str = "") -> str:
    """Read a system-level config value by key. Returns default if not set."""
    with SessionFactory() as session:
        row = session.get(SystemConfig, key)
        return row.value if row else default


def set_system_config(key: str, value: str) -> None:
    """Upsert a system-level config value."""
    with SessionFactory() as session:
        row = session.get(SystemConfig, key)
        if row:
            row.value = value
            row.updated_at = datetime.datetime.utcnow()
        else:
            session.add(SystemConfig(key=key, value=value))
        session.commit()


# Composite gate for λ unlock:
#   _LASSO_GATE_MIN_N       — minimum Clean Zone verified decisions
#   _LASSO_GATE_MIN_REGIMES — minimum distinct macro regimes in those decisions
# Rationale: n alone is insufficient if all samples come from one regime.
# A single-regime sample produces parameters that cannot generalise across
# market conditions. Both conditions must be met simultaneously.
_LASSO_GATE_MIN_N:       int = 20
_LASSO_GATE_MIN_REGIMES: int = 2

# Legacy alias — kept for any callers that imported the old name
_LASSO_GATE_THRESHOLD = _LASSO_GATE_MIN_N


def get_lasso_lambda() -> tuple[float, str]:
    """
    Return (lambda_value, source) where source is one of:
      "prior"     — composite gate not yet met; using structural prior
      "stability" — both gate conditions met (stub; falls back to prior until wired up)

    Composite gate (both required):
      1. Clean Zone verified decisions ≥ 20   (quantity floor)
      2. Distinct macro regimes in sample ≥ 2 (diversity floor)

    Rationale: parameters calibrated on a single-regime sample have no
    cross-regime generalisability. The diversity condition forces the system
    to wait for market cycle coverage before any parameter adjustment.
    """
    prior = float(get_system_config("risk.lasso_lambda_prior", "0.10"))

    cz_stats = get_clean_zone_stats()
    cz_n = cz_stats.get("clean_b", {}).get("n", 0)

    # Count distinct regimes in Clean Zone verified decisions
    with SessionFactory() as _s:
        _regime_rows = (
            _s.query(DecisionLog.macro_regime)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                DecisionLog.macro_regime.isnot(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .distinct()
            .all()
        )
    cz_regimes = len(_regime_rows)

    if cz_n < _LASSO_GATE_MIN_N or cz_regimes < _LASSO_GATE_MIN_REGIMES:
        return prior, "prior"

    # Both gate conditions met — stability criterion should run here.
    logger.warning(
        "get_lasso_lambda: gate open (n=%d≥%d, regimes=%d≥%d) "
        "but stability criterion not yet implemented; returning prior λ=%.4f",
        cz_n, _LASSO_GATE_MIN_N, cz_regimes, _LASSO_GATE_MIN_REGIMES, prior,
    )
    return prior, "stability"


def set_backtest_stop(value: bool) -> None:
    """Set or clear the stop flag for the running backtest."""
    if value:
        _STOP_FLAG.touch()
    else:
        _STOP_FLAG.unlink(missing_ok=True)


def get_backtest_stop() -> bool:
    """Return True if a pause has been requested."""
    return _STOP_FLAG.exists()


def get_backtest_retry_stubs() -> list[dict]:
    """Return backtest records that failed due to quota and need retry."""
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.needs_retry == True,
                DecisionLog.is_backtest == True,
            )
            .order_by(DecisionLog.decision_date.asc())
            .all()
        )
        return [
            {
                "id":           r.id,
                "sector":       r.sector_name,
                "date":         str(r.decision_date),
                "macro_regime": r.macro_regime,
                "vix":          r.vix_level,
            }
            for r in rows
        ]


def clear_retry_stub(record_id: int) -> None:
    """Mark a retry stub as no longer needing retry (after successful completion)."""
    with SessionFactory() as session:
        row = session.query(DecisionLog).filter(DecisionLog.id == record_id).first()
        if row:
            row.needs_retry = False
            session.commit()


def get_dormant_pattern_count() -> int:
    """Count patterns silenced due to regime dormancy (for admin dashboard display)."""
    with SessionFactory() as session:
        return session.query(func.count(LearningLog.id)).filter(
            LearningLog.applied == False,
            LearningLog.dormant == True,
        ).scalar() or 0


def _state_fingerprint(state_vector: dict, horizon_key: str = "季") -> str:
    """
    Canonical string key from a state vector dict, tagged with prediction horizon.

    Including horizon_key in the fingerprint ensures QuantPatternLog rows are
    segregated by horizon: RSI/momentum accuracy at 90-day horizon may differ
    from 180-day horizon, and mixing them produces unreliable priors.

    Legacy rows written before this change (without |H: suffix) will no longer
    be matched by new fingerprint lookups — they become dormant dead data.
    This is intentional: horizon-mixed patterns are not reliable priors.
    """
    return (
        f"动量:{state_vector.get('momentum_regime','?')}|"
        f"RSI:{state_vector.get('rsi_zone','?')}|"
        f"波动:{state_vector.get('vol_regime','?')}|"
        f"H:{horizon_key}"
    )


def _update_quant_pattern(dec, session) -> None:
    """
    Upsert one row in QuantPatternLog for a freshly-verified decision.
    Called inside verify_pending_decisions() immediately after accuracy_score is set.

    Strict leakage protection: this function only runs AFTER verification is
    complete (actual price return already fetched), so no future data is used.
    The verified_at timestamp on the record ensures cutoff_date queries exclude
    unverified data automatically.

    Train/test isolation: decisions on or after TRAIN_TEST_CUTOFF belong to the
    locked test set — their outcomes must not feed back into the pattern table.
    """
    if not dec.debate_transcript or not dec.direction or dec.accuracy_score is None:
        return
    # Backtest contamination gate: LLM has implicit foreknowledge of all historical
    # dates it was trained on — backtest decisions must never feed learning tables.
    if dec.is_backtest:
        logger.debug(
            "QuantPatternLog write blocked (backtest contamination): sector=%s date=%s",
            dec.sector_name, dec.decision_date,
        )
        return
    # Test-set gate: skip write-back for locked test period
    _dec_date = dec.decision_date or (dec.created_at.date() if dec.created_at else None)
    if _dec_date is not None and _dec_date >= TRAIN_TEST_CUTOFF:
        logger.debug(
            "QuantPatternLog write blocked (test set): sector=%s date=%s",
            dec.sector_name, _dec_date,
        )
        return
    # LCS quality gate
    if dec.lcs_passed is False:
        logger.debug(
            "QuantPatternLog write blocked (LCS failed): sector=%s score=%.2f",
            dec.sector_name, dec.lcs_score or 0,
        )
        return
    try:
        dt = json.loads(dec.debate_transcript)
    except (ValueError, TypeError):
        return

    sv = dt.get("state_vector")
    if not sv or all(v == "unknown" for v in sv.values()):
        return   # no usable state vector stored (pre-Layer-2 records)

    # Derive horizon key from declared horizon (legacy "中"/"长" mapped forward)
    _h_raw      = (dec.horizon or "季度(3个月)").strip()[:1]
    _h_key      = "半" if _h_raw in ("半", "长") else "季"
    fingerprint = _state_fingerprint(sv, _h_key)
    regime      = dec.macro_regime or "未知"
    direction   = dec.direction
    is_correct  = dec.accuracy_score >= 0.75
    is_partial  = dec.accuracy_score == 0.5

    row = session.query(QuantPatternLog).filter_by(
        state_fingerprint=fingerprint,
        macro_regime=regime,
        direction=direction,
    ).first()

    if row is None:
        row = QuantPatternLog(
            state_fingerprint=fingerprint,
            macro_regime=regime,
            direction=direction,
            total_count=0, correct_count=0, partial_count=0,
            accuracy_rate=0.0, avg_accuracy=0.0,
        )
        session.add(row)

    row.total_count   += 1
    row.correct_count += int(is_correct)
    row.partial_count += int(is_partial)
    row.accuracy_rate  = row.correct_count / row.total_count
    # Rolling mean of raw accuracy scores
    row.avg_accuracy   = (
        (row.avg_accuracy * (row.total_count - 1) + dec.accuracy_score)
        / row.total_count
    )
    row.last_updated   = datetime.datetime.utcnow()

    # Soft guard: warn when bucket sample is too small for reliable priors.
    # Prior to Phase-0 historical batch training (n ≥ 100 baseline), individual
    # updates carry high noise.  This does NOT block the write — the gate is
    # informational only.  Suppress once baseline is established.
    _BUCKET_NOISE_THRESHOLD = 20
    if row.total_count < _BUCKET_NOISE_THRESHOLD:
        logger.warning(
            "QuantPatternLog low-n write: fingerprint=%s regime=%s direction=%s "
            "n=%d (<%d) — prior is noisy; run Phase-0 historical batch training "
            "to establish a reliable baseline before trusting this bucket.",
            fingerprint, regime, direction,
            row.total_count, _BUCKET_NOISE_THRESHOLD,
        )


def get_quant_pattern_context(
    state_vector: dict,
    macro_regime: str,
    cutoff_date=None,
    min_n: int = 5,
) -> str:
    """
    Return a formatted string summarising historical accuracy for the given
    state vector × macro regime, for injection into AI analysis prompts.

    Queries both 季度 and 半年 horizon fingerprints separately and labels each
    block. This lets the AI see how the same market state (momentum/RSI/vol)
    translates to different accuracy rates at different horizons — directly
    informing its horizon choice and calibrating its confidence.

    Leakage protection: only rows whose last_updated <= cutoff_date are used.
    Returns empty string when insufficient data (n < min_n per direction
    across all horizons).
    """
    if cutoff_date is not None:
        cutoff_dt = datetime.datetime.combine(
            cutoff_date if isinstance(cutoff_date, datetime.date)
            else datetime.date.fromisoformat(str(cutoff_date)),
            datetime.time.max,
        )
    else:
        cutoff_dt = None

    _HORIZON_LABELS = {"季": "季度(3个月)", "半": "半年(6个月)"}
    all_blocks: list[str] = []

    with SessionFactory() as session:
        for hk, hlabel in _HORIZON_LABELS.items():
            fp = _state_fingerprint(state_vector, hk)
            q  = session.query(QuantPatternLog).filter_by(
                state_fingerprint=fp,
                macro_regime=macro_regime,
            )
            if cutoff_dt is not None:
                q = q.filter(QuantPatternLog.last_updated <= cutoff_dt)
            rows = q.all()

            if not rows:
                continue

            parts: list[str] = []
            total_n = sum(r.total_count for r in rows)
            for r in sorted(rows, key=lambda x: -x.total_count):
                if r.total_count < min_n:
                    continue
                parts.append(
                    f"{r.direction} {r.accuracy_rate:.0%}"
                    f"（n={r.total_count}, 均分={r.avg_accuracy:.2f}）"
                )

            if not parts:
                continue

            block_lines = [
                f"[QUANT_PATTERN_PRIOR · {hlabel}]",
                f"历史状态：{fp.rsplit('|H:', 1)[0]} × {macro_regime}  样本={total_n}",
                "各方向历史准确率：" + " / ".join(parts),
            ]
            # Direction-level nudges
            for r in rows:
                if r.total_count >= min_n and r.direction in ("超配", "低配"):
                    if r.accuracy_rate < 0.40:
                        block_lines.append(
                            f"  ⚠ {hlabel}下 {r.direction} 历史胜率偏低（{r.accuracy_rate:.0%}），"
                            "建议降低置信度或回避该方向。"
                        )
                    elif r.accuracy_rate >= 0.70:
                        block_lines.append(
                            f"  ✓ {hlabel}下 {r.direction} 历史胜率良好（{r.accuracy_rate:.0%}），"
                            "量化信号与配置方向一致。"
                        )
            block_lines.append(f"[/QUANT_PATTERN_PRIOR · {hlabel}]")
            all_blocks.append("\n".join(block_lines))

    return "\n\n".join(all_blocks)


def get_verified_decision_count() -> int:
    """
    Return number of decisions that have been verified with actual price returns.
    Used as the Layer 3 trigger: when count >= 300, Purged Walk-Forward mode activates.
    """
    with SessionFactory() as session:
        return session.query(func.count(DecisionLog.id)).filter(
            DecisionLog.actual_return_5d.isnot(None),
        ).scalar() or 0


def get_backtest_records(
    sector: str | None = None,
    regime: str | None = None,
    phase: str | None = None,
    limit: int = 500,
) -> list[dict]:
    """
    Return stored backtest records for the review panel.
    Excludes needs_retry stubs (no ai_conclusion).
    """
    with SessionFactory() as session:
        q = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.is_backtest == True,
                DecisionLog.needs_retry == False,
                DecisionLog.ai_conclusion.isnot(None),
                DecisionLog.decision_date.isnot(None),
            )
        )
        if sector:
            q = q.filter(DecisionLog.sector_name == sector)
        if regime:
            q = q.filter(DecisionLog.macro_regime == regime)
        if phase:
            if phase == "simple":
                q = q.filter(DecisionLog.tab_type.in_(["sector", "sector_backtest",
                                                        "sector_backtest_ms", "sector_backtest_qs"]))
            else:
                q = q.filter(DecisionLog.tab_type.contains(phase))
        q = q.order_by(DecisionLog.created_at.desc()).limit(limit)
        rows = q.all()
    return [
        {
            "id":            r.id,
            "date":          str(r.decision_date),       # 历史场景日期
            "run_date":      r.created_at.strftime("%Y-%m-%d") if r.created_at else "—",  # 实际运行日期
            "sector":        r.sector_name or "—",
            "regime":        r.macro_regime or "—",
            "direction":     r.direction or "—",
            "vix":           r.vix_level,
            "confidence":    r.confidence_score,
            "phase":         (r.tab_type or "")
                             .replace("sector_backtest_qs", "simple(季度)")
                             .replace("sector_backtest_ms", "simple(月度)")
                             .replace("sector_backtest", "simple")
                             .replace("walk_forward_", ""),
            "ai_conclusion": r.ai_conclusion or "",
            "horizon":       r.horizon or "—",
            "lcs_score":     r.lcs_score,
            "lcs_passed":    r.lcs_passed,
            "barrier_hit":   r.barrier_hit,
            "barrier_days":  r.barrier_days,
            "barrier_return": r.barrier_return,
            "hist_vol_ann":  r.hist_vol_ann,
        }
        for r in rows
    ]


def delete_backtest_record_by_ids(record_ids: list[int]) -> int:
    """Delete specific backtest records by their DB ids. Returns deleted count."""
    if not record_ids:
        return 0
    with SessionFactory() as session:
        count = (
            session.query(DecisionLog)
            .filter(DecisionLog.id.in_(record_ids), DecisionLog.is_backtest == True)
            .delete(synchronize_session=False)
        )
        session.commit()
    return count


def get_training_coverage(
    sectors: list[str],
    freq: str,
    start_date: str,
    end_date: str,
) -> dict:
    """
    Return training coverage stats for the given sectors × date window.

    Returns:
        {
          "total":     int,          # total (sector × date) pairs in window
          "completed": int,          # pairs with a real AI record in DB
          "pct":       float,        # completed / total * 100
          "by_sector": {
              sector: {
                "total": int, "completed": int, "remaining": int, "pct": float
              }, ...
          },
          "regime_dist": {regime: count, ...},   # completed records by macro regime
        }
    """
    import pandas as pd_cov

    dates = pd_cov.date_range(start=start_date, end=end_date, freq=freq).date.tolist()
    total_dates = len(dates)
    date_strs   = {str(d) for d in dates}

    # Fetch all completed backtest records for these sectors in the window
    # (tab_type covers all freq variants + legacy)
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog.sector_name, DecisionLog.decision_date,
                          DecisionLog.macro_regime)
            .filter(
                DecisionLog.is_backtest == True,
                DecisionLog.needs_retry == False,
                DecisionLog.ai_conclusion.isnot(None),
                DecisionLog.sector_name.in_(sectors),
                DecisionLog.decision_date.isnot(None),
            )
            .all()
        )

    # Index completed pairs
    completed_pairs: set[tuple] = set()
    regime_dist:     dict[str, int] = {}
    for r in rows:
        if str(r.decision_date) in date_strs:
            completed_pairs.add((r.sector_name, str(r.decision_date)))
            reg = r.macro_regime or "未知"
            regime_dist[reg] = regime_dist.get(reg, 0) + 1

    # Aggregate by sector
    by_sector: dict[str, dict] = {}
    for sec in sectors:
        done = sum(1 for d in date_strs if (sec, d) in completed_pairs)
        by_sector[sec] = {
            "total":     total_dates,
            "completed": done,
            "remaining": total_dates - done,
            "pct":       round(done / total_dates * 100, 1) if total_dates else 0.0,
        }

    total     = total_dates * len(sectors)
    completed = len(completed_pairs)
    return {
        "total":       total,
        "completed":   completed,
        "pct":         round(completed / total * 100, 1) if total else 0.0,
        "by_sector":   by_sector,
        "regime_dist": regime_dist,
    }


def delete_backtest_records(
    sector: str | None = None,
    phase: str | None = None,
) -> int:
    """
    Delete backtest records from DecisionLog.
    Returns the number of rows deleted.
    scope: sector=None + phase=None → delete ALL backtest records.
    """
    with SessionFactory() as session:
        q = session.query(DecisionLog).filter(DecisionLog.is_backtest == True)
        if sector:
            q = q.filter(DecisionLog.sector_name == sector)
        if phase and phase != "simple":
            q = q.filter(DecisionLog.tab_type.contains(phase))
        elif phase == "simple":
            q = q.filter(DecisionLog.tab_type.in_(["sector", "sector_backtest",
                                                    "sector_backtest_ms", "sector_backtest_qs"]))
        count = q.count()
        q.delete(synchronize_session=False)
        session.commit()
    return count


def mark_pattern_applied(pattern_id: int) -> None:
    with SessionFactory() as session:
        row = session.query(LearningLog).filter(LearningLog.id == pattern_id).first()
        if row:
            row.applied    = True
            row.applied_at = datetime.datetime.utcnow()
            session.commit()


# ── News routing weights ───────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "科技":   {"央行声明": 0.8, "科技监管": 0.7, "供应链": 0.6, "地缘政治": 0.4},
    "能源":   {"央行声明": 0.8, "OPEC动态": 0.9, "地缘政治": 0.7, "供应链": 0.5},
    "金融":   {"央行声明": 0.9, "信贷数据": 0.8, "监管政策": 0.7, "地缘政治": 0.4},
    "医疗":   {"FDA动态": 0.8, "政策医改": 0.7, "央行声明": 0.3, "地缘政治": 0.3},
    "工业":   {"PMI数据": 0.8, "地缘政治": 0.7, "供应链": 0.7, "央行声明": 0.5},
    "消费":   {"零售数据": 0.8, "就业数据": 0.7, "央行声明": 0.6, "地缘政治": 0.3},
    "必需消费":{"CPI数据": 0.8, "就业数据": 0.6, "央行声明": 0.5, "地缘政治": 0.3},
    "公用事业":{"央行声明": 0.9, "能源政策": 0.7, "监管政策": 0.6, "地缘政治": 0.3},
    "房地产": {"央行声明": 0.9, "信贷数据": 0.8, "监管政策": 0.7, "就业数据": 0.5},
    "材料":   {"供应链": 0.8, "地缘政治": 0.7, "PMI数据": 0.7, "央行声明": 0.5},
    "通信":   {"科技监管": 0.7, "央行声明": 0.5, "地缘政治": 0.4, "供应链": 0.4},
    "通讯传媒":{"科技监管": 0.7, "监管政策": 0.6, "央行声明": 0.4, "地缘政治": 0.3},
    "高收益债":{"央行声明": 0.9, "信贷数据": 0.8, "就业数据": 0.6, "地缘政治": 0.5},
}


def get_news_routing_weights(
    sector_name: str,
    macro_regime: str = "",
) -> dict[str, float]:
    """
    Return news category weights for a given sector × regime.
    Falls back to default weights if no learned data exists yet.
    """
    with SessionFactory() as session:
        q = session.query(NewsRoutingWeight).filter(
            NewsRoutingWeight.sector_name == sector_name
        )
        if macro_regime:
            q = q.filter(NewsRoutingWeight.macro_regime == macro_regime)
        rows = q.all()

    if rows:
        return {r.news_category: r.weight for r in rows}

    # Fall back to defaults
    return _DEFAULT_WEIGHTS.get(sector_name, {"宏观经济": 0.7, "央行声明": 0.7})


def update_news_routing_weight(
    sector_name: str,
    macro_regime: str,
    news_category: str,
    accuracy_when_used: float,
    accuracy_when_not_used: float,
) -> None:
    """
    Adjust the weight of a news category based on its observed impact on accuracy.
    Called by Meta-Agent after each learning cycle.
    """
    delta = accuracy_when_used - accuracy_when_not_used
    with SessionFactory() as session:
        row = (
            session.query(NewsRoutingWeight)
            .filter(
                NewsRoutingWeight.sector_name  == sector_name,
                NewsRoutingWeight.macro_regime == macro_regime,
                NewsRoutingWeight.news_category == news_category,
            )
            .first()
        )
        if row:
            # Exponential moving average: blend old weight with signal
            row.weight       = max(0.1, min(1.0, row.weight + 0.1 * delta))
            row.sample_count += 1
            row.updated_at   = datetime.datetime.utcnow()
        else:
            default = _DEFAULT_WEIGHTS.get(sector_name, {}).get(news_category, 0.5)
            session.add(NewsRoutingWeight(
                sector_name=sector_name,
                macro_regime=macro_regime,
                news_category=news_category,
                weight=max(0.1, min(1.0, default + 0.1 * delta)),
                sample_count=1,
            ))
        session.commit()


# ── Regime-conditional benchmarks ─────────────────────────────────────────────

def get_regime_benchmarks(
    sector_name: str,
    macro_regime: str,
    min_samples: int = 3,
    cutoff_date=None,
    exclude_backtest: bool = True,
) -> str:
    """
    Query Alpha Memory for verified decisions with the same sector × macro_regime.
    Compute regime-conditional statistics and return a formatted string for
    prompt injection — the system's empirical baseline for this exact situation.

    cutoff_date / exclude_backtest: same isolation semantics as get_historical_context().
    Returns empty string if not enough data.
    """
    with SessionFactory() as session:
        q = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.sector_name == sector_name,
                DecisionLog.macro_regime == macro_regime,
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
            )
        )
        if exclude_backtest:
            q = q.filter(
                (DecisionLog.is_backtest == False) | (DecisionLog.is_backtest == None)
            )
        if cutoff_date is not None:
            q = q.filter(
                func.coalesce(DecisionLog.decision_date, func.date(DecisionLog.created_at))
                < cutoff_date
            )
        records = (
            q
            .order_by(
                func.coalesce(DecisionLog.decision_date, func.date(DecisionLog.created_at)).desc()
            )
            .limit(30)
            .all()
        )

    if len(records) < min_samples:
        return ""

    total       = len(records)
    hit_rate    = sum(1 for r in records if r.accuracy_score >= EXCELLENT) / total
    avg_score   = sum(r.accuracy_score for r in records) / total

    # Direction breakdown
    direction_counts: dict[str, int] = {}
    direction_accuracy: dict[str, list[float]] = {}
    for r in records:
        d = r.direction or "中性"
        direction_counts[d] = direction_counts.get(d, 0) + 1
        direction_accuracy.setdefault(d, []).append(r.accuracy_score)

    # Return stats when 超配
    overweight_rets = [
        r.actual_return_20d for r in records
        if r.direction == "超配" and r.actual_return_20d is not None
    ]
    underweight_rets = [
        r.actual_return_20d for r in records
        if r.direction == "低配" and r.actual_return_20d is not None
    ]

    # Quant metric averages (from stored quant_metrics JSON)
    sharpe_vals, rel_mom_vals = [], []
    for r in records:
        if r.quant_metrics:
            try:
                qm = json.loads(r.quant_metrics)
                if qm.get("rolling_sharpe_20d") is not None:
                    sharpe_vals.append(float(qm["rolling_sharpe_20d"]))
                if qm.get("relative_momentum") is not None:
                    rel_mom_vals.append(float(qm["relative_momentum"]))
            except Exception:
                pass

    # Payoff quality aggregates for the benchmarks
    _bm_pq_vals = [r.payoff_quality for r in records if r.payoff_quality is not None]
    _bm_avg_pq  = sum(_bm_pq_vals) / len(_bm_pq_vals) if _bm_pq_vals else None

    lines = [
        f"【制度条件基准 · {sector_name} × {macro_regime} · 共{total}条历史记录】",
        f"历史命中率: {hit_rate:.0%}  均分: {avg_score:.2f}/1.0"
        + (f"  平均盈亏质量: {_bm_avg_pq:.2f}" if _bm_avg_pq is not None else ""),
    ]

    # Direction accuracy breakdown
    dir_parts = []
    for d, scores in direction_accuracy.items():
        avg = sum(scores) / len(scores)
        dir_parts.append(f"{d}({len(scores)}次, 准确率{avg:.0%})")
    if dir_parts:
        lines.append("方向准确率: " + " · ".join(dir_parts))

    # Average returns
    if overweight_rets:
        avg_ow = sum(overweight_rets) / len(overweight_rets)
        lines.append(f"超配时平均20日收益: {avg_ow:+.2%}  (n={len(overweight_rets)})")
    if underweight_rets:
        avg_uw = sum(underweight_rets) / len(underweight_rets)
        lines.append(f"低配时平均20日收益: {avg_uw:+.2%}  (n={len(underweight_rets)})")

    # Quant metric baselines
    if sharpe_vals:
        lines.append(f"该情景下历史平均Sharpe: {sum(sharpe_vals)/len(sharpe_vals):.2f}")
    if rel_mom_vals:
        lines.append(f"该情景下历史平均相对动量: {sum(rel_mom_vals)/len(rel_mom_vals):+.2f}%")

    # Performance nudge
    if hit_rate >= EXCELLENT:
        lines.append(f"→ 历史上在此制度下判断准确，当前框架有效。")
    elif hit_rate < MIN_ACCEPTABLE and total >= 5:
        lines.append(f"→ 历史上在此制度下判断偏差较大，请提高谨慎度。")

    return "\n".join(lines)


# ── Skill Library: compress experience into behavioral instructions ────────────

_SKILL_COMPRESS_THRESHOLD = 5   # minimum verified decisions to trigger compression
_SKILL_REFRESH_EVERY      = 3   # recompress after this many new decisions


def _compress_skill(
    model,
    sector_name: str,
    macro_regime: str,
    benchmarks: str,
    patterns: list[dict],
    sample_count: int,
    avg_accuracy: float,
    avg_payoff_quality: float | None = None,
    high_pq_contexts: str = "",
    low_pq_contexts: str = "",
) -> str:
    """
    Call LLM to compress accumulated evidence into a short behavioral instruction.
    The primary learning signal is payoff_quality (risk-normalized return), not
    directional accuracy — optimizing for asymmetric payoff, not win rate.
    Returns the skill text (≤ 120 tokens target).
    """
    pattern_text = "\n".join(
        f"- [{p['type']}] {p['description'][:150]}" for p in patterns[:6]
    ) or "暂无识别模式"

    pq_section = ""
    if avg_payoff_quality is not None:
        pq_label = (
            "强（每单位风险回报优秀）" if avg_payoff_quality >= 1.0
            else ("中等" if avg_payoff_quality >= 0.3 else "弱（回报未补偿风险）")
        )
        pq_section = (
            f"\n=== 盈亏质量分析（核心学习信号）===\n"
            f"平均盈亏质量(payoff_quality): {avg_payoff_quality:.2f}  [{pq_label}]\n"
            f"解释：payoff_quality = 实际回报 / 持仓期波动率，>1.0=强赢，0~1=弱赢，<0=亏损\n"
        )
        if high_pq_contexts:
            pq_section += f"高盈亏质量(>1.0)决策共同特征：{high_pq_contexts}\n"
        if low_pq_contexts:
            pq_section += f"低盈亏质量(<0)决策共同特征：{low_pq_contexts}\n"

    prompt = (
        f"你是一个量化投资系统的经验压缩器。\n"
        f"以下是系统在【{sector_name}】板块、【{macro_regime}】制度下，"
        f"基于{sample_count}条已验证决策（方向准确率{avg_accuracy:.0%}）积累的原始经验：\n\n"
        f"=== 制度条件基准 ===\n{benchmarks}\n"
        f"{pq_section}\n"
        f"=== 识别到的模式 ===\n{pattern_text}\n\n"
        "请将以上内容压缩成一段【行为指令】，格式要求：\n"
        "1. 总长度不超过120字\n"
        "2. 直接写给下一个分析Agent看，告诉它在此板块+制度下应该怎么做\n"
        "3. 重点：哪些情境下盈亏质量高（值得重仓）？哪些情境下即使方向对也回报不足？\n"
        "4. 包含：触发条件 → 建议方向 → 最优持仓窗口 → 需要回避的陷阱\n"
        "5. 机构语气，无废话，无解释性文字\n"
        "6. 【严禁】将RSI、Bollinger Band、短期技术反弹、超买/超卖等信号半衰期<30天的技术指标"
        "作为季度/半年行为规则的触发条件——这类信号的预测窗口与验证窗口不匹配，"
        "写入规则会污染未来决策的信号归因。触发条件只能基于宏观制度、动量趋势、基本面变化。\n\n"
        "只输出行为指令本身，不加任何前缀或标签。"
    )
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        logger.warning("Skill compression failed: %s", e)
        return ""


def maybe_update_skill(model, sector_name: str, macro_regime: str) -> bool:
    """
    Check if enough new verified decisions have accumulated to warrant
    (re)compressing the skill for this sector × regime combination.

    Returns True if a skill was created or updated.
    """
    if not model or not sector_name or not macro_regime:
        return False

    with SessionFactory() as session:
        # Count verified decisions for this cell — training set only, LCS-passed only
        records = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.sector_name  == sector_name,
                DecisionLog.macro_regime == macro_regime,
                DecisionLog.verified     == True,
                DecisionLog.accuracy_score.isnot(None),
                # Backtest contamination gate: LLM has foreknowledge of historical dates
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                # Test-set gate: skill compression must not use test outcomes
                func.coalesce(
                    DecisionLog.decision_date,
                    func.date(DecisionLog.created_at),
                ) < TRAIN_TEST_CUTOFF,
                # LCS quality gate: only logically consistent decisions feed skills
                # (NULL = LCS not yet run → allowed through; False = explicitly failed)
                (DecisionLog.lcs_passed != False) | DecisionLog.lcs_passed.is_(None),
            )
            .order_by(DecisionLog.created_at.desc())
            .limit(30)
            .all()
        )

        n = len(records)
        if n < _SKILL_COMPRESS_THRESHOLD:
            return False

        avg_acc = sum(r.accuracy_score for r in records) / n

        # Compute payoff quality statistics — the primary learning signal.
        pq_records  = [r for r in records if r.payoff_quality is not None]
        avg_pq      = (sum(r.payoff_quality for r in pq_records) / len(pq_records)
                       if pq_records else None)

        # Summarize high-payoff and low-payoff decision contexts for the compression prompt.
        # "Context" = direction + barrier type + a snippet of the signal drivers.
        def _ctx_snippet(r) -> str:
            drivers = ""
            if r.signal_attribution:
                try:
                    _a = json.loads(r.signal_attribution) if isinstance(r.signal_attribution, str) else r.signal_attribution
                    drivers = str(_a.get("drivers", ""))[:60]
                except Exception:
                    pass
            return f"{r.direction}·{r.barrier_hit or '?'}·{drivers}"

        high_pq_recs = sorted(
            [r for r in pq_records if r.payoff_quality >= 1.0],
            key=lambda r: r.payoff_quality, reverse=True
        )[:3]
        low_pq_recs  = sorted(
            [r for r in pq_records if r.payoff_quality < 0],
            key=lambda r: r.payoff_quality
        )[:3]
        high_pq_ctx = "；".join(_ctx_snippet(r) for r in high_pq_recs)
        low_pq_ctx  = "；".join(_ctx_snippet(r) for r in low_pq_recs)

        # Check existing skill
        existing = (
            session.query(SkillLibrary)
            .filter(
                SkillLibrary.sector_name  == sector_name,
                SkillLibrary.macro_regime == macro_regime,
            )
            .first()
        )

        # Skip if existing skill is still fresh (not enough new data since last compression)
        if existing and (n - existing.sample_count) < _SKILL_REFRESH_EVERY:
            return False

        # Gather inputs for compression
        benchmarks = get_regime_benchmarks(sector_name, macro_regime)
        patterns   = get_learning_patterns(sector_name=sector_name, macro_regime=macro_regime)

        skill_text = _compress_skill(
            model, sector_name, macro_regime,
            benchmarks, patterns, n, avg_acc,
            avg_payoff_quality=avg_pq,
            high_pq_contexts=high_pq_ctx,
            low_pq_contexts=low_pq_ctx,
        )
        if not skill_text:
            return False

        if existing:
            existing.skill_text         = skill_text
            existing.version           += 1
            existing.sample_count       = n
            existing.avg_accuracy       = avg_acc
            existing.avg_payoff_quality = avg_pq
            existing.updated_at         = datetime.datetime.utcnow()
            logger.info(
                "Skill updated: sector=%s regime=%s v%d (n=%d acc=%.2f pq=%.2f)",
                sector_name, macro_regime, existing.version, n, avg_acc, avg_pq or 0,
            )
        else:
            session.add(SkillLibrary(
                sector_name         = sector_name,
                macro_regime        = macro_regime,
                skill_text          = skill_text,
                version             = 1,
                sample_count        = n,
                avg_accuracy        = avg_acc,
                avg_payoff_quality  = avg_pq,
            ))
            logger.info(
                "Skill created: sector=%s regime=%s (n=%d acc=%.2f pq=%.2f)",
                sector_name, macro_regime, n, avg_acc, avg_pq or 0,
            )

        session.commit()
        return True


def get_skill(
    sector_name: str,
    macro_regime: str,
    cutoff_date=None,
    exclude_backtest: bool = True,
) -> str:
    """
    Retrieve the compressed skill for a sector × regime combination.
    Returns empty string if no skill exists yet (cold start).

    cutoff_date: if set, only skills last updated before this date are returned.
    """
    with SessionFactory() as session:
        q = (
            session.query(SkillLibrary)
            .filter(
                SkillLibrary.sector_name  == sector_name,
                SkillLibrary.macro_regime == macro_regime,
            )
        )
        if cutoff_date is not None:
            q = q.filter(SkillLibrary.updated_at < datetime.datetime.combine(
                cutoff_date, datetime.time.min
            ))
        row = q.first()
        if not row:
            return ""
        pq_str = (
            f" · 平均盈亏质量{row.avg_payoff_quality:.2f}"
            if row.avg_payoff_quality is not None else ""
        )
        return (
            f"[SKILL v{row.version} · {row.sample_count}条经验 · "
            f"准确率{row.avg_accuracy:.0%}{pq_str}]\n{row.skill_text}"
        )


# ── Historical context builder ─────────────────────────────────────────────────

def get_historical_context(
    tab_type: str = "",
    sector_name: str = "",
    macro_regime: str = "",
    n: int = 5,
    cutoff_date=None,          # walk-forward isolation: only see records before this date
    exclude_backtest: bool = True,  # exclude is_backtest=True from live context
) -> str:
    """
    Return a formatted string for prompt injection covering:
      - Recent verified performance
      - Active learning patterns (systematic biases)
      - Regime-conditional benchmarks (sector × macro_regime empirical baseline)

    cutoff_date: if provided (datetime.date), only records created before this date
                 are visible — enforces temporal isolation in walk-forward backtests.
    exclude_backtest: True by default — prevents synthetic backtest decisions from
                      contaminating live analysis context.
    Empty string if no history exists yet.
    """
    with SessionFactory() as session:
        q = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
            )
        )
        # tab_type="" or None → cross-tab query (e.g. from portfolio audit agent)
        if tab_type:
            q = q.filter(DecisionLog.tab_type == tab_type)
        if sector_name:
            q = q.filter(DecisionLog.sector_name == sector_name)
        if macro_regime:
            q = q.filter(DecisionLog.macro_regime == macro_regime)
        if exclude_backtest:
            q = q.filter(
                (DecisionLog.is_backtest == False) | (DecisionLog.is_backtest == None)
            )
        if cutoff_date is not None:
            # Use decision_date (the historical date the record REPRESENTS) for isolation.
            # Fall back to created_at.date() for old rows that predate this column.
            q = q.filter(
                func.coalesce(DecisionLog.decision_date, func.date(DecisionLog.created_at))
                < cutoff_date
            )

        _pool_size = max(n * 4, 20)
        _candidates = q.order_by(
            func.coalesce(DecisionLog.decision_date, func.date(DecisionLog.created_at)).desc()
        ).limit(_pool_size).all()

    if not _candidates:
        return ""

    # P2-1: Time-decay weighting — λ = ln(2)/90 (half-life 90 days)
    import math as _math
    _lambda = _math.log(2) / 90.0
    _today  = datetime.date.today() if cutoff_date is None else cutoff_date

    def _decay_weight(rec) -> float:
        _ref_date = rec.decision_date or rec.created_at.date()
        _days_old = max((_today - _ref_date).days, 0)
        _quality  = rec.accuracy_score if rec.accuracy_score is not None else 0.5
        return _math.exp(-_lambda * _days_old) * _quality

    _candidates.sort(key=_decay_weight, reverse=True)
    recent = _candidates[:n]

    scores   = [d.accuracy_score for d in recent]
    hit_rate = sum(1 for s in scores if s >= EXCELLENT) / len(scores)
    avg      = sum(scores) / len(scores)
    label    = sector_name or tab_type

    # Payoff quality stats — the primary learning signal
    pq_vals  = [d.payoff_quality for d in recent if d.payoff_quality is not None]
    avg_pq   = sum(pq_vals) / len(pq_vals) if pq_vals else None
    wins_pq  = [v for v in pq_vals if v >= 1.0]
    loss_pq  = [v for v in pq_vals if v < 0]
    avg_win_pq  = sum(wins_pq)  / len(wins_pq)  if wins_pq  else None
    avg_loss_pq = sum(loss_pq)  / len(loss_pq)  if loss_pq  else None

    lines = [
        f"【历史决策绩效参考 · {label} · 时间衰减加权 Top{len(recent)}（候选池 {len(_candidates)} 条）】",
        f"方向命中率: {hit_rate:.0%}  均分: {avg:.2f}/1.0",
    ]

    if avg_pq is not None:
        pq_label = "强" if avg_pq >= 1.0 else ("中" if avg_pq >= 0.3 else "弱")
        lines.append(
            f"平均盈亏质量: {avg_pq:.2f} [{pq_label}]"
            + (f"  | 赢时均质量: {avg_win_pq:.2f}" if avg_win_pq is not None else "")
            + (f"  | 输时均质量: {avg_loss_pq:.2f}" if avg_loss_pq is not None else "")
        )
        lines.append(
            "⚠ 盈亏质量是主要优化目标：识别「方向对且赢得充分」的情境，"
            "而非追求高胜率。payoff_quality > 1.0 = 值得下重注的机会。"
        )
    lines.append("")

    for d in recent[:3]:
        pq_str  = f" PQ={d.payoff_quality:.2f}" if d.payoff_quality is not None else ""
        flag    = "✅" if d.accuracy_score >= EXCELLENT else ("⚠️" if d.accuracy_score >= 0.5 else "❌")
        ret_str = (
            f"{d.actual_return_20d:+.1%}(20d)" if d.actual_return_20d is not None
            else (f"{d.actual_return_5d:+.1%}(5d)" if d.actual_return_5d is not None else "—")
        )
        display_date = d.decision_date or d.created_at.date()
        lines.append(
            f"{flag} [{display_date.strftime('%m-%d')}] "
            f"VIX={d.vix_level:.1f} 方向={d.direction} 实际={ret_str} "
            f"评分={d.accuracy_score:.2f}{pq_str}"
        )
        if d.reflection:
            lines.append(f"   └ 反思: {d.reflection[:90]}")

    # ── Skill injection (preferred) vs verbose fallback ──────────────────────
    # cutoff_date and exclude_backtest are propagated to all sub-queries so that
    # no future data (patterns, benchmarks, skills derived from T ≥ cutoff) can
    # bleed into the context for decision date T.
    skill = (
        get_skill(sector_name, macro_regime, cutoff_date=cutoff_date, exclude_backtest=exclude_backtest)
        if (sector_name and macro_regime) else ""
    )

    if skill:
        # Compressed behavioral instruction replaces verbose pattern + benchmark blocks
        lines.append("")
        lines.append("【行为指令 · 经验压缩】")
        lines.append(skill)
    else:
        # Cold-start fallback: inject raw patterns + regime benchmarks verbosely
        patterns = get_learning_patterns(
            sector_name=sector_name, macro_regime=macro_regime,
            cutoff_date=cutoff_date, exclude_backtest=exclude_backtest,
        )
        if patterns:
            biases             = [p for p in patterns if p["type"] in ("bias", "calibration")]
            strengths          = [p for p in patterns if p["type"] == "strength"]
            horizon_mismatches = [p for p in patterns if p["type"] == "horizon_mismatch"]
            if strengths:
                lines.append("")
                lines.append("【有效框架 · 请维持】")
                for p in strengths[:2]:
                    lines.append(f"✅ {p['description'][:100]}")
            if biases:
                lines.append("")
                lines.append("【已识别偏差 · 请注意】")
                for p in biases[:2]:
                    lines.append(f"⚠ {p['description'][:100]}")
            if horizon_mismatches:
                lines.append("")
                lines.append("【投资期限校准 · 请调整horizon声明】")
                for p in horizon_mismatches[:2]:
                    lines.append(f"🕐 {p['description'][:120]}")

        # Regime-conditional benchmarks (sector × macro_regime empirical baseline)
        if sector_name and macro_regime:
            regime_bench = get_regime_benchmarks(
                sector_name, macro_regime,
                cutoff_date=cutoff_date,
                exclude_backtest=exclude_backtest,
            )
            if regime_bench:
                lines.append("")
                lines.append(regime_bench)

    # ── Known failures injection ─────────────────────────────────────────────
    # Query attributed failures for this sector × regime and inject as warnings.
    # Uses all verified data (not restricted to training set) — failure patterns
    # are informational guardrails, not statistical estimates that require
    # train/test separation.
    if sector_name or macro_regime:
        with SessionFactory() as _kf_sess:
            _kf_q = (
                _kf_sess.query(DecisionLog)
                .filter(
                    DecisionLog.verified     == True,
                    DecisionLog.accuracy_score < 0.5,
                    DecisionLog.failure_type.isnot(None),
                    (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                )
            )
            if sector_name:
                _kf_q = _kf_q.filter(DecisionLog.sector_name == sector_name)
            if macro_regime:
                _kf_q = _kf_q.filter(DecisionLog.macro_regime == macro_regime)
            if cutoff_date is not None:
                _kf_q = _kf_q.filter(
                    func.coalesce(DecisionLog.decision_date, func.date(DecisionLog.created_at))
                    < cutoff_date
                )
            _kf_records = _kf_q.order_by(DecisionLog.created_at.desc()).limit(5).all()

        if _kf_records:
            # Tally failure types for this cell
            from collections import Counter as _Counter
            _ft_counts = _Counter(r.failure_type for r in _kf_records)
            _ft_top    = _ft_counts.most_common(3)

            _ft_labels = {
                "hypothesis":   "假设方向错误",
                "data":         "数据质量问题",
                "regime_drift": "宏观制度漂移",
                "robustness":   "样本外稳健性差",
                "evaluation":   "验证参数问题",
                "execution":    "执行偏差",
            }
            lines.append("")
            lines.append(
                f"【已知失败模式 · {sector_name or '全局'} × {macro_regime or '全制度'} · "
                f"n={len(_kf_records)} 条归因失败】"
            )
            for _ft, _cnt in _ft_top:
                _ft_label = _ft_labels.get(_ft, _ft)
                # Find most recent note for this failure type
                _note = next(
                    (r.failure_note for r in _kf_records
                     if r.failure_type == _ft and r.failure_note),
                    None,
                )
                _note_str = f": {_note[:80]}" if _note else ""
                lines.append(f"❌ {_ft_label}（{_cnt}次）{_note_str}")
            lines.append(
                "→ 本次分析请显式检查上述失败模式是否仍然适用，并在结论中说明规避理由。"
            )

    # Strategy nudge
    if hit_rate < MIN_ACCEPTABLE and len(recent) >= 3:
        lines.append("")
        lines.append("⚠ 近期命中率低于合格线，本次分析请提高谨慎度，增加不确定性描述。")
    elif hit_rate >= EXCELLENT and len(recent) >= 3:
        lines.append("")
        lines.append("✅ 近期判断准确率良好，当前分析框架有效，维持现有逻辑结构。")

    return "\n".join(lines)


# ── Cross-sector spillover learning ───────────────────────────────────────────

# Minimum co-occurring verified decisions per (source, target, regime) pair
# before a learned coefficient is trusted and activates.
_SPILLOVER_MIN_SAMPLES = 10


def update_spillover_weights() -> int:
    """
    Compute Pearson correlation of accuracy_scores between all sector pairs
    that share the same macro_regime. Persists results to SpilloverWeight.
    Flags rows where the sign conflicts with the SPILLOVER_MAP prior.
    Returns the number of pairs updated.
    """
    from engine.news import SPILLOVER_MAP

    # Build prior sign map: (source, target) → expected sign
    # Positive prior means source news is expected to matter to target (no directional sign)
    # We use +1 as a placeholder for all priors (prior only says "transmission exists")
    prior_pairs: set[tuple[str, str]] = set()
    for target, sources in SPILLOVER_MAP.items():
        for source_sector, _, _ in sources:
            prior_pairs.add((source_sector, target))

    updated = 0
    with SessionFactory() as session:
        verified = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                (DecisionLog.human_label != "black_swan") | DecisionLog.human_label.is_(None),
            )
            .all()
        )

        # Group by (sector_name, macro_regime) → list of accuracy scores with date key
        # Use decision date as alignment key for co-occurrence
        from collections import defaultdict
        cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for d in verified:
            sector  = d.sector_name or "ALL"
            regime  = d.macro_regime or "UNKNOWN"
            date_key = (d.verified_at or d.created_at).strftime("%Y-%m-%d") if (d.verified_at or d.created_at) else "unknown"
            # Keep latest score per date per cell
            cell[(sector, regime)][date_key] = d.accuracy_score

        # For each (source, target) pair within the same regime, compute correlation
        regimes = set(k[1] for k in cell.keys())
        for regime in regimes:
            sectors_in_regime = [s for (s, r) in cell.keys() if r == regime]
            for source in sectors_in_regime:
                for target in sectors_in_regime:
                    if source == target:
                        continue
                    src_scores = cell[(source, regime)]
                    tgt_scores = cell[(target, regime)]
                    # Find co-occurring dates
                    common_dates = sorted(set(src_scores) & set(tgt_scores))
                    if len(common_dates) < _SPILLOVER_MIN_SAMPLES:
                        continue

                    x = [src_scores[dt] for dt in common_dates]
                    y = [tgt_scores[dt] for dt in common_dates]

                    # Pearson r
                    n = len(x)
                    mx, my = sum(x) / n, sum(y) / n
                    num   = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
                    denom = math.sqrt(
                        sum((xi - mx) ** 2 for xi in x) *
                        sum((yi - my) ** 2 for yi in y)
                    )
                    if denom == 0:
                        continue
                    r = num / denom

                    # Conflict detection: prior says transmission exists (any sign);
                    # we flag when |r| < 0.1 (near-zero, prior overstated) or
                    # when this is a known prior pair and r < -0.3 (strongly reversed)
                    is_prior = (source, target) in prior_pairs
                    conflicts = is_prior and r < -0.30

                    existing = (
                        session.query(SpilloverWeight)
                        .filter(
                            SpilloverWeight.source_sector == source,
                            SpilloverWeight.target_sector == target,
                            SpilloverWeight.macro_regime  == regime,
                        )
                        .first()
                    )
                    if existing:
                        existing.correlation     = r
                        existing.sample_count    = n
                        existing.conflicts_prior = conflicts
                        existing.updated_at      = datetime.datetime.utcnow()
                    else:
                        session.add(SpilloverWeight(
                            source_sector  = source,
                            target_sector  = target,
                            macro_regime   = regime,
                            correlation    = r,
                            sample_count   = n,
                            conflicts_prior = conflicts,
                        ))
                    updated += 1

        session.commit()
    return updated


def get_spillover_weights(
    target_sector: str, macro_regime: str
) -> list[dict]:
    """
    Return learned spillover weights for a target sector in the given regime.
    Only returns rows where sample_count >= _SPILLOVER_MIN_SAMPLES (activated).
    Each dict: {source_sector, correlation, sample_count, conflicts_prior}
    Returns empty list when data is insufficient → caller falls back to prior.
    """
    with SessionFactory() as session:
        rows = (
            session.query(SpilloverWeight)
            .filter(
                SpilloverWeight.target_sector == target_sector,
                SpilloverWeight.macro_regime  == macro_regime,
                SpilloverWeight.sample_count  >= _SPILLOVER_MIN_SAMPLES,
            )
            .order_by(SpilloverWeight.correlation.desc())
            .all()
        )
        return [
            {
                "source_sector":   r.source_sector,
                "correlation":     r.correlation,
                "sample_count":    r.sample_count,
                "conflicts_prior": r.conflicts_prior,
            }
            for r in rows
        ]


# ── P2-10 哈希链完整性验证 ────────────────────────────────────────────────────

def verify_chain_integrity() -> tuple[int, int, int]:
    """
    Walk all DecisionLog rows that have a chain_hash, in id order.
    Recompute each hash and compare. Returns (ok_count, broken_count, total).
    """
    import hashlib as _hl
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog)
            .filter(DecisionLog.chain_hash.isnot(None))
            .order_by(DecisionLog.id.asc())
            .all()
        )
    if not rows:
        return 0, 0, 0

    ok = broken = 0
    prev_hash = "genesis"
    for row in rows:
        payload   = f"{row.id}|{row.created_at}|{(row.ai_conclusion or '')[:100]}|{prev_hash}"
        expected  = _hl.sha256(payload.encode("utf-8")).hexdigest()
        if row.chain_hash == expected:
            ok += 1
        else:
            broken += 1
        prev_hash = row.chain_hash   # use stored hash to propagate chain

    return ok, broken, len(rows)


# ── P2-9 Learning Stage State Machine ─────────────────────────────────────────

from dataclasses import dataclass as _dataclass

@_dataclass
class LearningStageInfo:
    stage:           str    # cold_start | memory_active | parameter_adaptive | structural_adaptive
    n_verified:      int    # total verified decisions (live only)
    next_threshold:  int | None   # n needed to advance; None at final stage
    progress_frac:   float  # 0.0–1.0 within current stage band
    label:           str    # human-readable stage name (Chinese)
    description:     str    # one-line explanation of what the system can do at this stage


_STAGE_BANDS = [
    (0,  10, "cold_start",          "冷启动",       "决策记录不足，学习功能受限"),
    (10, 30, "memory_active",       "记忆激活",     "历史上下文注入生效，识别基本偏差"),
    (30, 50, "parameter_adaptive",  "参数自适应",   "γ/λ 参数开始自动迭代，模式识别稳定"),
    (50, None, "structural_adaptive", "结构自适应", "跨板块溢出学习、FactorMAD 全面激活"),
]


def get_learning_stage(exclude_backtest: bool = True) -> LearningStageInfo:
    """
    Derive the current learning stage from verified live decision count.
    Real-time computation — not persisted. Used for gating Defer-tier features.
    """
    with SessionFactory() as session:
        q = session.query(func.count(DecisionLog.id)).filter(
            DecisionLog.verified == True,
            DecisionLog.accuracy_score.isnot(None),
        )
        if exclude_backtest:
            q = q.filter(
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None)
            )
        n = q.scalar() or 0

    for lo, hi, stage_id, label, desc in _STAGE_BANDS:
        if hi is None or n < hi:
            if hi is None:
                progress = 1.0
                nxt = None
            else:
                progress = (n - lo) / (hi - lo)
                nxt = hi
            return LearningStageInfo(
                stage=stage_id,
                n_verified=n,
                next_threshold=nxt,
                progress_frac=min(max(progress, 0.0), 1.0),
                label=label,
                description=desc,
            )
    # Fallback — should never be reached
    return LearningStageInfo("structural_adaptive", n, None, 1.0, "结构自适应", "")


# ── Aggregate stats for dashboard ─────────────────────────────────────────────

def get_stats() -> dict:
    with SessionFactory() as session:
        verified = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
            )
            .all()
        )

        total = len(verified)
        # Count decisions that have passed their horizon-specific minimum wait window.
        # Uses the shortest threshold (7d for 短期) as a DB-level pre-filter,
        # then applies per-record horizon logic to get an accurate ready count.
        _now = datetime.datetime.utcnow()
        _horizon_min = {"短": 7, "中": 45, "长": 90}
        _unverified = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == False,
                DecisionLog.created_at <= _now - datetime.timedelta(days=7),
            )
            .all()
        )
        pending_count = sum(
            1 for d in _unverified
            if (_now - d.created_at).days >= _horizon_min.get(
                (d.horizon or "中")[:1], 45
            )
        )
        logged_count = session.query(func.count(DecisionLog.id)).scalar()
        pattern_count = session.query(func.count(LearningLog.id)).filter(
            LearningLog.applied == False
        ).scalar()

        if total == 0:
            return {
                "total_verified":       0,
                "total_logged":         logged_count,
                "pending_verification": pending_count,
                "unapplied_patterns":   pattern_count,
            }

        by_tab: dict[str, list[float]] = {}
        for d in verified:
            by_tab.setdefault(d.tab_type, []).append(d.accuracy_score)

        sector_rows = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type.in_(["sector", "scanner"]),
                DecisionLog.verified == True,
            )
            .order_by(DecisionLog.created_at.desc())
            .limit(20)
            .all()
        )

        return {
            "total_verified":       total,
            "total_logged":         logged_count,
            "pending_verification": pending_count,
            "unapplied_patterns":   pattern_count,
            "overall_hit_rate":     sum(1 for d in verified if d.accuracy_score >= EXCELLENT) / total,
            "overall_avg_score":    sum(d.accuracy_score for d in verified) / total,
            "by_tab": {
                tab: {
                    "count":    len(sc),
                    "hit_rate": sum(1 for s in sc if s >= EXCELLENT) / len(sc),
                    "avg":      sum(sc) / len(sc),
                }
                for tab, sc in by_tab.items()
            },
            "history": [
                {
                    "date":       d.created_at.strftime("%Y-%m-%d"),
                    "tab":        d.tab_type,
                    "sector":     d.sector_name or d.ticker or "—",
                    "direction":  d.direction,
                    "return_5d":  d.actual_return_5d,
                    "return_20d": d.actual_return_20d,
                    "score":      d.accuracy_score,
                    "verdict":    d.meta_verdict or "—",
                    "reflection": d.reflection or "",
                    "source":     d.decision_source or "ai_drafted",
                    "edit_ratio": d.edit_ratio,
                }
                for d in sector_rows
            ],
        }


def get_clean_zone_stats() -> dict:
    """
    Return performance statistics split across three temporal zones:

    ┌──────────────────────────────┬───────────────────────────────────────────┐
    │ Zone                         │ Date range                                │
    ├──────────────────────────────┼───────────────────────────────────────────┤
    │ Training Set                 │ decision_date < TRAIN_TEST_CUTOFF         │
    │ Test Set A  (contaminated)   │ TRAIN_TEST_CUTOFF ≤ date < CLEAN_ZONE_START│
    │ Test Set B  (Clean Zone)     │ decision_date ≥ CLEAN_ZONE_START          │
    └──────────────────────────────┴───────────────────────────────────────────┘

    "Contaminated" means the LLM may have seen narratives about outcomes in
    that period during its pre-training.  Clean Zone decisions are the only
    truly foreknowledge-free evidence.

    Also returns LCS quality gate statistics across all verified decisions.
    """
    with SessionFactory() as session:
        verified = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.decision_date.isnot(None),
                (DecisionLog.superseded == False) | (DecisionLog.superseded == None),
                # Backtest contamination gate: only live decisions reflect real
                # out-of-sample performance. Backtest decisions are excluded because
                # the LLM has implicit foreknowledge of all historical dates it was
                # trained on — their verified accuracy is not meaningful evidence.
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .all()
        )

    def _zone_stats(records: list, run_binomial: bool = False) -> dict:
        if not records:
            return {
                "n": 0, "avg_accuracy": None, "hit_rate": None,
                "lcs_pass_rate": None, "brier_score": None,
                "binom_pvalue": None, "binom_ci_lo": None, "binom_ci_hi": None,
            }
        n          = len(records)
        avg_acc    = sum(r.accuracy_score for r in records) / n
        wins       = sum(1 for r in records if r.accuracy_score >= EXCELLENT)
        hit_rate   = wins / n
        lcs_run    = [r for r in records if r.lcs_passed is not None]
        lcs_pass   = sum(1 for r in lcs_run if r.lcs_passed) / len(lcs_run) if lcs_run else None

        # Brier Score: mean((predicted_prob - actual_outcome)^2)
        # Only for records where confidence_score was captured.
        conf_recs  = [r for r in records if r.confidence_score is not None]
        brier      = None
        if conf_recs:
            brier = sum(
                (r.confidence_score / 100.0 - r.accuracy_score) ** 2
                for r in conf_recs
            ) / len(conf_recs)

        # Binomial test: H0 win_rate = 0.5 (random baseline), one-sided greater.
        # Only meaningful when n >= 30.
        binom_p = binom_ci_lo = binom_ci_hi = None
        if run_binomial and n >= 30:
            try:
                from scipy.stats import binomtest as _binomtest
                _bt = _binomtest(wins, n, p=0.5, alternative="greater")
                binom_p    = round(_bt.pvalue, 4)
                _ci        = _bt.proportion_ci(confidence_level=0.95, method="exact")
                binom_ci_lo = round(_ci.low,  4)
                binom_ci_hi = round(_ci.high, 4)
            except Exception:
                pass

        return {
            "n":             n,
            "avg_accuracy":  round(avg_acc,  4),
            "hit_rate":      round(hit_rate, 4),
            "lcs_pass_rate": round(lcs_pass, 4) if lcs_pass is not None else None,
            "brier_score":   round(brier,    4) if brier    is not None else None,
            "binom_pvalue":  binom_p,
            "binom_ci_lo":   binom_ci_lo,
            "binom_ci_hi":   binom_ci_hi,
        }

    train_recs  = [r for r in verified if r.decision_date < TRAIN_TEST_CUTOFF]
    testa_recs  = [
        r for r in verified
        if TRAIN_TEST_CUTOFF <= r.decision_date < CLEAN_ZONE_START
    ]
    cleanb_recs = [r for r in verified if r.decision_date >= CLEAN_ZONE_START]

    # LCS-filtered training accuracy (only logically consistent decisions)
    lcs_passed_train = [
        r for r in train_recs if r.lcs_passed is True
    ]

    # Overall LCS stats
    all_lcs_run  = [r for r in verified if r.lcs_passed is not None]
    lcs_fail_rate = (
        sum(1 for r in all_lcs_run if not r.lcs_passed) / len(all_lcs_run)
        if all_lcs_run else None
    )

    return {
        "training": _zone_stats(train_recs),
        "test_a":   _zone_stats(testa_recs),
        "clean_b":  _zone_stats(cleanb_recs, run_binomial=True),
        "lcs_filtered_training": _zone_stats(lcs_passed_train),
        "lcs_overall": {
            "total_audited": len(all_lcs_run),
            "fail_rate":     round(lcs_fail_rate, 4) if lcs_fail_rate is not None else None,
        },
        "boundaries": {
            "train_test_cutoff": str(TRAIN_TEST_CUTOFF),
            "clean_zone_start":  str(CLEAN_ZONE_START),
        },
    }


def get_clean_zone_time_series() -> list[dict]:
    """
    Return monthly win-rate time series for Clean Zone verified decisions.
    Each dict: {month: "2025-04", n: int, wins: int, win_rate: float,
                avg_confidence: float|None, avg_accuracy: float}
    Sorted ascending by month.
    """
    with SessionFactory() as session:
        recs = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .order_by(DecisionLog.decision_date)
            .all()
        )

    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for r in recs:
        key = r.decision_date.strftime("%Y-%m")
        buckets[key].append(r)

    result = []
    for month in sorted(buckets):
        rs    = buckets[month]
        n     = len(rs)
        wins  = sum(1 for r in rs if r.accuracy_score >= EXCELLENT)
        confs = [r.confidence_score for r in rs if r.confidence_score is not None]
        result.append({
            "month":          month,
            "n":              n,
            "wins":           wins,
            "win_rate":       round(wins / n, 4),
            "avg_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "avg_accuracy":   round(sum(r.accuracy_score for r in rs) / n, 4),
        })
    return result


def get_pending_decisions_for_monitor() -> list[dict]:
    """
    Return all unverified live Clean Zone decisions with invalidation metadata.
    Used by the Invalidation Monitor tab.

    Each dict includes:
      id, tab_type, sector_name, ticker, direction, decision_date,
      horizon, invalidation_conditions, confidence_score, macro_regime,
      days_elapsed, horizon_days, deadline_date, days_to_deadline, urgency
        urgency: "normal" | "approaching" (≤14d) | "overdue" (past deadline)
    """
    _HORIZON_DAYS = {"季": 90, "半": 180, "中": 90, "长": 180}
    _DEFAULT_DAYS = 90

    with SessionFactory() as session:
        recs = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == False,
                DecisionLog.decision_date >= CLEAN_ZONE_START,
                DecisionLog.tab_type != "macro",   # macro decisions are not price-verifiable
                DecisionLog.ticker.isnot(None),    # must have a ticker for Triple-Barrier
                (DecisionLog.superseded == False) | DecisionLog.superseded.is_(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
                (DecisionLog.needs_retry == False) | DecisionLog.needs_retry.is_(None),
            )
            .order_by(DecisionLog.decision_date.desc())
            .all()
        )

    today  = datetime.date.today()
    result = []
    for r in recs:
        h_key      = (r.horizon or "季度(3个月)").strip()[:1]
        h_days     = _HORIZON_DAYS.get(h_key, _DEFAULT_DAYS)
        dec_date   = r.decision_date or (r.created_at.date() if r.created_at else today)
        deadline   = dec_date + datetime.timedelta(days=h_days)
        days_left  = (deadline - today).days
        elapsed    = (today - dec_date).days

        if days_left < 0:
            urgency = "overdue"
        elif days_left <= 14:
            urgency = "approaching"
        else:
            urgency = "normal"

        result.append({
            "id":                   r.id,
            "tab_type":             r.tab_type or "",
            "sector_name":          r.sector_name or "—",
            "ticker":               r.ticker or "—",
            "direction":            r.direction or "—",
            "decision_date":        str(dec_date),
            "horizon":              r.horizon or "季度(3个月)",
            "horizon_days":         h_days,
            "invalidation_conditions": r.invalidation_conditions or "",
            "confidence_score":     r.confidence_score,
            "macro_regime":         r.macro_regime or "—",
            "days_elapsed":         elapsed,
            "deadline_date":        str(deadline),
            "days_to_deadline":     days_left,
            "urgency":              urgency,
            "regime_drifted":       r.regime_drifted,
            "regime_at_verify":     r.regime_at_verify or "",
            "human_label":          r.human_label or "",
        })
    return result


# ── Approval queue helpers ─────────────────────────────────────────────────────

# Priority rank for sorting (lower = more urgent)
_APPROVAL_PRIORITY_RANK = {
    "urgent": 0,   # regime switch — affects all open positions
    "critical": 0, # legacy alias
    "high":   1,   # signal flip — existing position direction wrong
    "normal": 2,   # new entry trigger (default)
    "low":    3,   # rebalance drift / covariance override
}


def expire_stale_approvals() -> int:
    """
    Mark PendingApproval records whose approval_deadline has passed as 'expired'.
    Returns the number of records expired.
    Called automatically at the start of each daily chain run.
    """
    today = datetime.date.today()
    with SessionFactory() as session:
        expired = (
            session.query(PendingApproval)
            .filter(
                PendingApproval.status == "pending",
                PendingApproval.approval_deadline.isnot(None),
                PendingApproval.approval_deadline < today,
            )
            .all()
        )
        for pa in expired:
            pa.status      = "expired"
            pa.resolved_at = datetime.datetime.utcnow()
            pa.resolved_by = "auto"
        session.commit()
        n = len(expired)
    if n:
        logger.info("expire_stale_approvals: %d records expired", n)
    return n


def get_recent_verifications(n: int = 6) -> list[dict]:
    """Return the n most recently verified decisions for the Daily Brief feed."""
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.verified == True,
                DecisionLog.accuracy_score.isnot(None),
                (DecisionLog.is_backtest == False) | DecisionLog.is_backtest.is_(None),
            )
            .order_by(DecisionLog.verified_at.desc())
            .limit(n)
            .all()
        )
        return [
            {
                "id":            r.id,
                "sector_name":   r.sector_name or "—",
                "ticker":        r.ticker or "—",
                "direction":     r.direction or "—",
                "horizon":       r.horizon or "季度(3个月)",
                "accuracy":      r.accuracy_score,
                "verdict":       r.meta_verdict or "—",
                "barrier":       r.barrier_hit or "—",
                "lcs_passed":    r.lcs_passed,
                "needs_review":  r.needs_review,
                "verified_at":   r.verified_at.strftime("%m-%d") if r.verified_at else "—",
                "macro_regime":  r.macro_regime or "—",
            }
            for r in rows
        ]


def get_pending_approvals_by_priority() -> list[dict]:
    """
    Return all pending PendingApproval records sorted by priority then creation time.
    Priority order: urgent → high → normal → low
    """
    with SessionFactory() as session:
        rows = (
            session.query(PendingApproval)
            .filter(PendingApproval.status == "pending")
            .order_by(PendingApproval.created_at.asc())
            .all()
        )
        result = []
        for r in rows:
            result.append({
                "id":                   r.id,
                "approval_type":        r.approval_type,
                "priority":             r.priority or "normal",
                "sector":               r.sector,
                "ticker":               r.ticker,
                "triggered_condition":  r.triggered_condition or "—",
                "triggered_date":       str(r.triggered_date),
                "suggested_weight":     r.suggested_weight,
                "approval_deadline":    str(r.approval_deadline) if r.approval_deadline else "—",
                "created_at":           r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—",
                "contradicts_quant":    bool(r.contradicts_quant),
                "llm_confidence":       r.llm_confidence,
                "_rank":                _APPROVAL_PRIORITY_RANK.get(r.priority or "normal", 2),
            })

    result.sort(key=lambda x: (x["_rank"], x["created_at"]))
    return result


def resolve_pending_approval(
    approval_id: int,
    approved: bool,
    resolved_by: str = "human",
    rejection_reason: str = "",
    review_rationale: str = "",
    review_category: str = "",
) -> dict:
    """
    Layer 3 executor: resolve a PendingApproval and, on approval, execute
    the underlying position action.

    P-AUDIT v1 audit fields (review_rationale / review_category) are
    persisted on the row when supplied (CFA GIPS §III.A.18 — every
    supervisor decision carries an independent typed justification). These
    params were dropped in a refactor and restored 2026-05-23 to unbreak
    bulk_resolve_pending_approvals → /api/approvals/resolve → the UI
    Approvals inbox (and the L2 propose→approve loop). The narrative-snapshot
    hash chain is a separate feature and intentionally not handled here.

    approval_type == 'entry':
        approved  → set WatchlistEntry.status = 'active', actual_weight = suggested_weight
        rejected  → set WatchlistEntry.status = 'watching'
    approval_type == 'risk_control':
        approved  → zero SimulatedPosition.actual_weight (stop executed)
        rejected  → append risk_override_note, leave position unchanged
    approval_type == 'rebalance':
        approved  → execute_rebalance() (full portfolio rebalance)
        rejected  → log note, no action

    Returns {'ok': bool, 'message': str, 'exec_detail': dict}
    """
    with SessionFactory() as session:
        pa = session.get(PendingApproval, approval_id)
        if pa is None:
            return {"ok": False, "message": f"Approval {approval_id} not found", "exec_detail": {}}
        if pa.status != "pending":
            return {"ok": False, "message": f"Approval {approval_id} already {pa.status}", "exec_detail": {}}

        pa.status        = "approved" if approved else "rejected"
        pa.resolved_at   = datetime.datetime.utcnow()
        pa.resolved_by   = resolved_by
        if not approved and rejection_reason:
            pa.rejection_reason = rejection_reason
        # P-AUDIT v1 audit trail — persist the typed justification + category on
        # both approve and reject (the UI / bulk resolver always supplies them).
        if review_rationale:
            pa.review_rationale = review_rationale
        if review_category:
            pa.review_category = review_category

        exec_detail: dict = {}

        # Helper: find the latest snapshot position for a ticker
        _latest_snap = (
            session.query(SimulatedPosition.snapshot_date)
            .order_by(SimulatedPosition.snapshot_date.desc())
            .limit(1)
            .scalar_subquery()
        )

        if pa.approval_type == "entry":
            if pa.watchlist_entry_id:
                entry = session.get(WatchlistEntry, pa.watchlist_entry_id)
                if entry:
                    # ── P3-12: LLM/Quant disagreement arbitration ─────────────
                    # Gate fires when LLM direction contradicts TSMOM signal.
                    # Low confidence contradictions are auto-rejected even if the
                    # human clicks Approve — the system overrides to prevent
                    # inadvertent approval of low-conviction contrarian trades.
                    _contradicts = bool(pa.contradicts_quant)
                    _conf        = pa.llm_confidence or (entry.confidence or 0)
                    try:
                        _cfg_row = (
                            session.query(SystemConfig)
                            .filter_by(key="contrarian_min_confidence")
                            .first()
                        )
                        _min_conf = int(_cfg_row.value) if _cfg_row else 75
                    except Exception:
                        _min_conf = 75

                    if _contradicts and approved and _conf < _min_conf:
                        pa.status           = "rejected"
                        pa.resolved_at      = datetime.datetime.utcnow()
                        pa.resolved_by      = "auto_arbitration"
                        pa.rejection_reason = (
                            f"P3-12 自动仲裁驳回：LLM方向与TSMOM相反，"
                            f"置信度 {_conf} < 阈值 {_min_conf}。"
                            f"如需强制执行请联系 Supervisor 手动覆盖。"
                        )
                        session.commit()
                        return {
                            "ok":          False,
                            "message":     f"自动仲裁驳回：contradicts_quant=True，置信度 {_conf}<{_min_conf}",
                            "exec_detail": {"auto_rejected": True, "confidence": _conf, "threshold": _min_conf},
                        }

                    if approved:
                        entry.status = "active"
                        if _contradicts:
                            # High-confidence contrarian — log the override
                            pa.risk_override_note = (
                                f"P3-12 高置信度反向信号已批准：contradicts_quant=True，"
                                f"置信度 {_conf} ≥ {_min_conf}。Supervisor 明确批准。"
                            )
                        pos = (
                            session.query(SimulatedPosition)
                            .filter(
                                SimulatedPosition.ticker == pa.ticker,
                                SimulatedPosition.snapshot_date == _latest_snap,
                            )
                            .first()
                        )
                        if pos and pa.suggested_weight is not None:
                            pos.actual_weight = pa.suggested_weight
                            exec_detail["weight_set"] = pa.suggested_weight
                    else:
                        entry.status = "watching"
                    exec_detail["watchlist_entry_id"] = pa.watchlist_entry_id

        elif pa.approval_type == "risk_control":
            if approved:
                pos = (
                    session.query(SimulatedPosition)
                    .filter(
                        SimulatedPosition.ticker == pa.ticker,
                        SimulatedPosition.snapshot_date == _latest_snap,
                    )
                    .first()
                )
                if pos:
                    w_before = pos.actual_weight or 0.0
                    pos.actual_weight = 0.0
                    pos.notes = (pos.notes or "") + f" | human_stop {datetime.date.today()}: approved by {resolved_by}"
                    session.add(SimulatedTrade(
                        trade_date   = datetime.date.today(),
                        sector       = pos.sector,
                        ticker       = pa.ticker,
                        action       = "SELL",
                        weight_before = round(w_before, 6),
                        weight_after  = 0.0,
                        weight_delta  = round(-w_before, 6),
                        cost_bps      = round(abs(w_before) * 10, 2),
                        trigger_reason = f"human_stop: {(pa.triggered_condition or '')[:80]}",
                    ))
                    exec_detail["weight_zeroed"] = w_before
            else:
                if rejection_reason:
                    pa.risk_override_note = rejection_reason

        elif pa.approval_type == "rebalance":
            if approved:
                try:
                    from engine.portfolio_tracker import execute_rebalance
                    result = execute_rebalance(rebalance_date=pa.triggered_date, dry_run=False) or {}
                    exec_detail = {
                        "turnover":       result.get("turnover", 0),
                        "total_cost_bps": result.get("total_cost_bps", 0),
                        "n_trades":       len(result.get("trades", [])),
                    }
                except Exception as exc:
                    session.rollback()
                    return {"ok": False, "message": str(exc), "exec_detail": {}}

        elif pa.approval_type == "overlay":
            # L2 operator-overlay execution (2026-05-24): a human-originated
            # discretionary position. On approve, the deterministic overlay executor
            # validates against the sleeve risk budget and writes the position to the
            # ISOLATED file-backed overlay store (never the systematic book). The LLM
            # only emitted the intent; this path is pure code behind a human approval.
            if approved:
                from engine.overlay_executor import apply_overlay
                res = apply_overlay(
                    ticker=pa.ticker,
                    target_weight=pa.suggested_weight,
                    approval_id=pa.id,
                    rationale=(pa.review_rationale or pa.triggered_condition or ""),
                    resolved_by=resolved_by,
                )
                if not res.get("ok"):
                    session.rollback()  # reverts pa.status → stays pending (cap breach / bad intent)
                    return {"ok": False, "message": res.get("message", "overlay rejected"), "exec_detail": {}}
                exec_detail = res.get("exec_detail", {})

        session.commit()

    return {
        "ok":          True,
        "message":     f"{'Approved' if approved else 'Rejected'}: {pa.approval_type} for {pa.ticker}",
        "exec_detail": exec_detail,
    }


# ── Structured Backtest Persistence ───────────────────────────────────────────

class StructuredBacktestRun(Base):
    """Metadata + metrics for one structured-signal backtest run."""
    __tablename__ = "structured_backtest_runs"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)
    start_date      = Column(String(10), nullable=False)
    end_date        = Column(String(10), nullable=False)
    lookback_months = Column(Integer,  default=12)
    skip_months     = Column(Integer,  default=1)
    regime_scale    = Column(Float,    default=0.3)
    n_months        = Column(Integer,  nullable=True)
    metrics_json    = Column(Text,     nullable=True)   # JSON: {tsmom, tsmom_regime, benchmark}
    warnings_json   = Column(Text,     nullable=True)   # JSON list


class StructuredBacktestReturn(Base):
    """Monthly return series for a structured backtest run."""
    __tablename__ = "structured_backtest_returns"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_id       = Column(Integer, nullable=False)      # FK → StructuredBacktestRun.id
    date         = Column(Date,    nullable=False)
    tsmom        = Column(Float,   nullable=True)
    tsmom_regime = Column(Float,   nullable=True)
    benchmark    = Column(Float,   nullable=True)
    regime_label = Column(String(20), nullable=True)
    p_risk_on    = Column(Float,   nullable=True)
    yield_spread = Column(Float,   nullable=True)


# ── P2-13 FactorMAD ────────────────────────────────────────────────────────────

class FactorDefinition(Base):
    """注册的 alpha 因子（已部署/生产中）。每个因子对应 FACTOR_REGISTRY 中的一个函数。"""
    __tablename__ = "factor_definitions"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    factor_id     = Column(String(50),  nullable=False, unique=True)
    description   = Column(String(200), nullable=True)
    asset_class   = Column(String(30),  nullable=False, default="equity_sector")
    active        = Column(Boolean,     default=True)
    created_at    = Column(DateTime,    default=datetime.datetime.utcnow)


class FactorICIR(Base):
    """每月滚动计算的 IC / ICIR 记录。IC = Spearman(因子值, 下月截面收益)。"""
    __tablename__ = "factor_icir"

    id            = Column(Integer,     primary_key=True, autoincrement=True)
    factor_id     = Column(String(50),  nullable=False)
    calc_date     = Column(Date,        nullable=False)
    ic_value      = Column(Float,       nullable=True)
    icir_12m      = Column(Float,       nullable=True)
    n_assets      = Column(Integer,     nullable=True)
    asset_class   = Column(String(30),  nullable=False, default="equity_sector")

    __table_args__ = (
        UniqueConstraint("factor_id", "calc_date", "asset_class",
                         name="uq_factor_icir_date_class"),
    )


class DiscoveredFactor(Base):
    """候选因子（通过 FactorMAD 辩论流程发现，等待/已完成人工审批）。"""
    __tablename__ = "discovered_factors"

    id                        = Column(Integer,     primary_key=True, autoincrement=True)
    name                      = Column(String(100), nullable=False)
    description               = Column(Text,        nullable=True)
    code_snippet              = Column(Text,        nullable=True)
    debate_log                = Column(Text,        nullable=True)
    ic_train                  = Column(Float,       nullable=True)
    icir_train                = Column(Float,       nullable=True)
    ic_test                   = Column(Float,       nullable=True)
    icir_test                 = Column(Float,       nullable=True)
    correlation_with_existing = Column(Float,       nullable=True)
    mi_ratio                  = Column(Float,       nullable=True)
    audit_signal_type         = Column(String(10),  nullable=True)
    audit_report              = Column(Text,        nullable=True)
    # pending / active / rejected / pending_further_review
    status                    = Column(String(30),  nullable=False, default="pending")
    rejection_reason          = Column(Text,        nullable=True)
    inactivation_reason       = Column(Text,        nullable=True)
    discovered_at             = Column(DateTime,    default=datetime.datetime.utcnow)
    activated_at              = Column(DateTime,    nullable=True)
    weight_cap                = Column(Float,       default=0.10)


class DailyBriefSnapshot(Base):
    """
    每日自动生成的 Morning Brief 快照。
    由 ensure_daily_batch_completed() 写入，驱动 Daily Brief Section B 的叙述层。
    每个交易日一条记录（as_of_date 唯一），幂等更新。
    """
    __tablename__ = "daily_brief_snapshots"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    as_of_date     = Column(Date,    unique=True, nullable=False)
    created_at     = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at     = Column(DateTime, onupdate=datetime.datetime.utcnow)

    # 制度状态
    regime         = Column(String(50), nullable=True)
    regime_prev    = Column(String(50), nullable=True)
    p_risk_on      = Column(Float,      nullable=True)
    regime_changed = Column(Boolean,    default=False)

    # 信号摘要
    n_long              = Column(Integer, default=0)
    n_short             = Column(Integer, default=0)
    signal_flips_json   = Column(Text,    nullable=True)   # JSON list[str]
    risk_alerts_json    = Column(Text,    nullable=True)   # JSON list[str]
    n_entries           = Column(Integer, default=0)
    n_invalidations     = Column(Integer, default=0)
    n_rebalance         = Column(Integer, default=0)

    # 验证摘要
    n_verified_today    = Column(Integer, default=0)

    # 自动化任务执行标记（幂等守卫）
    verify_ran          = Column(Boolean, default=False)
    icir_month          = Column(String(7), nullable=True)  # "YYYY-MM"

    # 叙述层（rule-based 自动生成，2-3 句专业中文摘要）
    narrative           = Column(Text, nullable=True)
    # LLM 宏观简报（每日首次 or 制度切换时生成，优先于 rule-based narrative 展示）
    macro_brief_llm     = Column(Text, nullable=True)

    # P4 战术巡逻结果
    tactical_entries_json  = Column(Text,    nullable=True)  # JSON list[str] — 新战术入场标的
    tactical_reduces_json  = Column(Text,    nullable=True)  # JSON list[str] — 战术减仓标的
    regime_jump_today      = Column(Boolean, default=False)  # 今日是否检测到制度跃变


def get_daily_brief_snapshot(as_of_date: "datetime.date" = None) -> "DailyBriefSnapshot | None":
    """Return today's DailyBriefSnapshot if it exists, else None."""
    import datetime as _dt
    if as_of_date is None:
        as_of_date = _dt.date.today()
    with SessionFactory() as session:
        return session.query(DailyBriefSnapshot).filter_by(as_of_date=as_of_date).first()


def upsert_daily_brief_snapshot(as_of_date: "datetime.date", **kwargs) -> None:
    """Create-or-update today's DailyBriefSnapshot. Thread-safe for SQLite (single writer)."""
    with SessionFactory() as session:
        obj = session.query(DailyBriefSnapshot).filter_by(as_of_date=as_of_date).first()
        if obj is None:
            obj = DailyBriefSnapshot(as_of_date=as_of_date)
            session.add(obj)
        for k, v in kwargs.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        obj.updated_at = datetime.datetime.utcnow()
        session.commit()


def save_structured_backtest(result) -> int:
    """
    Persist a BacktestResult to the database.
    Returns the new run_id.

    Args:
        result: engine.backtest.BacktestResult
    """
    from dataclasses import asdict
    import json as _json

    def _metrics_dict(m):
        return {
            "label":           m.label,
            "ann_return":      m.ann_return,
            "ann_vol":         m.ann_vol,
            "sharpe":          m.sharpe,
            "dsr":             m.dsr if m.dsr == m.dsr else None,  # NaN → None
            "max_drawdown":    m.max_drawdown,
            "calmar":          m.calmar,
            "win_rate_vs_bm":  m.win_rate_vs_bm,
            "ir_vs_bm":        m.ir_vs_bm,
            "sharpe_risk_on":  m.sharpe_risk_on,
            "sharpe_risk_off": m.sharpe_risk_off,
            "n_months":        m.n_months,
        }

    metrics_payload = {
        "tsmom":        _metrics_dict(result.metrics_tsmom),
        "tsmom_regime": _metrics_dict(result.metrics_regime),
        "benchmark":    _metrics_dict(result.metrics_bm),
    }

    # Infer parameter range from returns index
    df = result.returns
    start_str = str(df.index.min().date()) if not df.empty else ""
    end_str   = str(df.index.max().date()) if not df.empty else ""

    with SessionFactory() as session:
        run = StructuredBacktestRun(
            start_date      = start_str,
            end_date        = end_str,
            n_months        = result.metrics_tsmom.n_months,
            metrics_json    = _json.dumps(metrics_payload),
            warnings_json   = _json.dumps(result.warnings),
        )
        session.add(run)
        session.flush()
        run_id = run.id

        for ts, row in df.iterrows():
            session.add(StructuredBacktestReturn(
                run_id       = run_id,
                date         = ts.date(),
                tsmom        = float(row["tsmom"])        if "tsmom"        in row else None,
                tsmom_regime = float(row["tsmom_regime"]) if "tsmom_regime" in row else None,
                benchmark    = float(row["benchmark"])    if "benchmark"    in row else None,
                regime_label = str(row["regime_label"])   if "regime_label" in row else None,
                p_risk_on    = float(row["p_risk_on"])    if "p_risk_on"    in row else None,
                yield_spread = float(row["yield_spread"]) if "yield_spread" in row and row["yield_spread"] == row["yield_spread"] else None,
            ))

        session.commit()
        logger.info("Saved structured backtest run_id=%d (%s → %s, %d months)",
                    run_id, start_str, end_str, result.metrics_tsmom.n_months)
        return run_id


def load_structured_backtest(run_id: int | None = None) -> dict | None:
    """
    Load a structured backtest result from the database.

    Args:
        run_id: specific run to load; if None, loads the most recent run.

    Returns dict with keys:
        run_id, created_at, start_date, end_date, n_months,
        metrics (dict), warnings (list), returns (pd.DataFrame)
    """
    import json as _json
    import pandas as pd

    with SessionFactory() as session:
        if run_id is None:
            run = (session.query(StructuredBacktestRun)
                   .order_by(StructuredBacktestRun.id.desc())
                   .first())
        else:
            run = session.query(StructuredBacktestRun).get(run_id)

        if run is None:
            return None

        rows = (session.query(StructuredBacktestReturn)
                .filter(StructuredBacktestReturn.run_id == run.id)
                .order_by(StructuredBacktestReturn.date)
                .all())

        records = [
            {
                "date":         r.date,
                "tsmom":        r.tsmom,
                "tsmom_regime": r.tsmom_regime,
                "benchmark":    r.benchmark,
                "regime_label": r.regime_label,
                "p_risk_on":    r.p_risk_on,
                "yield_spread": r.yield_spread,
            }
            for r in rows
        ]
        df = pd.DataFrame(records)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return {
            "run_id":     run.id,
            "created_at": run.created_at,
            "start_date": run.start_date,
            "end_date":   run.end_date,
            "n_months":   run.n_months,
            "metrics":    _json.loads(run.metrics_json) if run.metrics_json else {},
            "warnings":   _json.loads(run.warnings_json) if run.warnings_json else [],
            "returns":    df,
        }


def list_structured_backtests() -> list[dict]:
    """Return summary list of all stored backtest runs (newest first)."""
    import json as _json
    with SessionFactory() as session:
        runs = (session.query(StructuredBacktestRun)
                .order_by(StructuredBacktestRun.id.desc())
                .all())
        return [
            {
                "run_id":     r.id,
                "created_at": str(r.created_at)[:16],
                "start_date": r.start_date,
                "end_date":   r.end_date,
                "n_months":   r.n_months,
                "sharpe_regime": (_json.loads(r.metrics_json) or {})
                                 .get("tsmom_regime", {}).get("sharpe"),
            }
            for r in runs
        ]


def backfill_macro_verified() -> int:
    """
    One-time migration: mark all existing unverified macro decisions (no ticker)
    as verified=True so they no longer clog verify_pending_decisions().

    These records remain in DB for historical context injection; they are excluded
    from Clean Zone accuracy stats because accuracy_score stays NULL.

    Returns the number of rows updated.
    """
    now = datetime.datetime.utcnow()
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.tab_type == "macro",
                DecisionLog.ticker.is_(None),
                DecisionLog.verified == False,
            )
            .all()
        )
        for r in rows:
            r.verified    = True
            r.verified_at = now
        session.commit()
        return len(rows)


# ── Macro Watchlist helpers ────────────────────────────────────────────────────

def save_watch_items(
    items: list[dict],
    analysis_date: datetime.date,
    macro_regime: str = "",
    check_days: int = 7,
) -> int:
    """
    Persist a list of parsed watchlist items extracted from §6 of macro analysis.

    Args:
        items        : list of dicts with keys: item_text, category, expected_value
        analysis_date: the date of the macro analysis that generated these items
        macro_regime : regime label at analysis time
        check_days   : calendar days until check_by date (default 7 ~ 5 trading days)

    Returns number of rows inserted.
    """
    if not items:
        return 0
    check_by = analysis_date + datetime.timedelta(days=check_days)
    with SessionFactory() as session:
        count = 0
        for item in items:
            text = (item.get("item_text") or "").strip()
            if not text:
                continue
            session.add(MacroWatchItem(
                analysis_date=analysis_date,
                check_by=check_by,
                item_text=text,
                category=item.get("category"),
                expected_value=item.get("expected_value"),
                macro_regime=macro_regime,
            ))
            count += 1
        session.commit()
    return count


def get_pending_watch_items(as_of: datetime.date | None = None) -> list[dict]:
    """
    Return all unresolved watch items, optionally filtered to those whose
    check_by date is on or before as_of.

    Pass as_of=None to get all pending items regardless of check_by date.
    """
    with SessionFactory() as session:
        q = session.query(MacroWatchItem).filter(MacroWatchItem.resolved == False)
        if as_of is not None:
            q = q.filter(MacroWatchItem.check_by <= as_of)
        rows = q.order_by(MacroWatchItem.analysis_date.desc()).all()
    return [
        {
            "id":             r.id,
            "analysis_date":  r.analysis_date,
            "check_by":       r.check_by,
            "item_text":      r.item_text,
            "category":       r.category,
            "expected_value": r.expected_value,
            "actual_value":   r.actual_value,
            "macro_regime":   r.macro_regime,
            "notes":          r.notes,
        }
        for r in rows
    ]


def resolve_watch_item(
    item_id: int,
    actual_value: str = "",
    outcome: str = "matched",
    notes: str = "",
) -> bool:
    """
    Mark a watch item as resolved with actual outcome.

    outcome: "matched" | "surprised" | "expired" | "n/a"
    Returns True if the item was found and updated.
    """
    with SessionFactory() as session:
        item = session.query(MacroWatchItem).filter(MacroWatchItem.id == item_id).first()
        if item is None:
            return False
        item.resolved    = True
        item.resolved_at = datetime.datetime.utcnow()
        item.actual_value = actual_value
        item.outcome     = outcome
        item.notes       = notes
        session.commit()
    return True


def expire_overdue_watch_items() -> int:
    """
    Auto-expire watch items whose check_by date has passed and are still unresolved.
    Called at the start of each macro analysis run.
    Returns number of items expired.
    """
    today = datetime.date.today()
    with SessionFactory() as session:
        rows = (
            session.query(MacroWatchItem)
            .filter(
                MacroWatchItem.resolved == False,
                MacroWatchItem.check_by < today,
            )
            .all()
        )
        for r in rows:
            r.resolved    = True
            r.resolved_at = datetime.datetime.utcnow()
            r.outcome     = "expired"
        session.commit()
        return len(rows)


def parse_watch_items_from_memo(memo_text: str) -> list[dict]:
    """
    Extract structured watch items from §6 of the macro analysis output.

    Looks for the section "### 6. 未来" (or "## 6.") and extracts bullet points.
    Each bullet is tagged with a category heuristic:
      - contains time/date words → data_release
      - contains price/level/支撑/阻力 → key_level
      - otherwise → market_signal

    Returns list of dicts: {item_text, category, expected_value}
    """
    import re as _re

    if not memo_text:
        return []

    # Find §6 block — everything between "### 6." and the next "###" or end
    pattern = _re.compile(
        r"###?\s*6[.、．]\s*[^\n]*\n(.*?)(?=###?\s*\d|$)",
        _re.DOTALL | _re.IGNORECASE,
    )
    match = pattern.search(memo_text)
    if not match:
        return []

    block = match.group(1)

    # Extract bullet points (-, *, •, ·, 1. 2. etc.)
    bullet_re = _re.compile(r"^\s*[-*•·]|\s*\d+[.)]\s", _re.MULTILINE)
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    items = []
    for line in lines:
        # Strip leading bullet markers
        clean = _re.sub(r"^[-*•·\d.)]+\s*", "", line).strip()
        if len(clean) < 8:   # skip empty / header lines
            continue

        # Category heuristic
        lower = clean.lower()
        if any(kw in lower for kw in [
            "发布", "公布", "数据", "报告", "会议", "讲话", "决议",
            "cpi", "pce", "gdp", "pmi", "nfp", "fomc", "预期值", "前值",
        ]):
            category = "data_release"
        elif any(kw in lower for kw in [
            "支撑", "阻力", "价位", "关口", "水平", "level", "support", "resistance",
            "%", "点位", "均线",
        ]):
            category = "key_level"
        else:
            category = "market_signal"

        # Try to extract an expected value (number + unit pattern)
        val_match = _re.search(r"(\d[\d.,]*\s*%|\$[\d.,]+|\d[\d.,]*\s*[亿万bpsBPS]+)", clean)
        expected_value = val_match.group(0) if val_match else None

        items.append({
            "item_text":      clean,
            "category":       category,
            "expected_value": expected_value,
        })

    return items[:10]   # cap at 10 items per analysis — keep it focused


# ── Failure attribution helpers (待实现-A) ────────────────────────────────────

_FAILURE_TYPES = [
    "hypothesis",    # research direction itself was wrong
    "data",          # data quality / PIT bias / coverage gap
    "regime_drift",  # macro regime changed materially during holding
    "robustness",    # signal overfit to backtest, failed OOS
    "evaluation",    # Triple-Barrier params mis-calibrated
    "execution",     # timing / implementation gap
]

_FAILURE_TYPE_LABELS = {
    "hypothesis":   "假设失效 — 研究方向本身无效",
    "data":         "数据问题 — FRED/yfinance 质量/PIT 偏差",
    "regime_drift": "制度漂移 — 持仓期间宏观制度发生切换",
    "robustness":   "稳健性 — 信号回测有效但样本外失效",
    "evaluation":   "评估问题 — Triple-Barrier 参数误校准",
    "execution":    "执行偏差 — 信号生成但无法在当时落实",
}


def get_unattributed_failures(min_age_days: int = 20) -> list[dict]:
    """
    Return verified failure records (accuracy_score < 0.5) that have not yet
    been attributed a failure_type. Only returns records at least min_age_days
    old (gives time for the barrier to resolve before annotation).
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=min_age_days)
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog)
            .filter(
                DecisionLog.accuracy_score < 0.5,
                DecisionLog.verified == True,
                DecisionLog.failure_type.is_(None),
                DecisionLog.superseded == False,
                DecisionLog.created_at <= cutoff,
            )
            .order_by(DecisionLog.created_at.desc())
            .all()
        )
    return [
        {
            "id":             r.id,
            "created_at":     r.created_at,
            "sector_name":    r.sector_name,
            "direction":      r.direction,
            "confidence_score": r.confidence_score,
            "accuracy_score": r.accuracy_score,
            "macro_regime":   r.macro_regime,
            "regime_drifted": r.regime_drifted,
            "failure_mode":   r.failure_mode,
            "failure_type":   r.failure_type,
            "failure_note":   r.failure_note,
            "economic_logic": r.economic_logic,
            "invalidation_conditions": r.invalidation_conditions,
        }
        for r in rows
    ]


def set_failure_attribution(
    decision_id: int,
    failure_type: str,
    failure_note: str = "",
) -> bool:
    """
    Set failure_type and failure_note on a DecisionLog record.
    Returns True if the record was found and updated.
    """
    if failure_type not in _FAILURE_TYPES:
        raise ValueError(f"Invalid failure_type: {failure_type}. Must be one of {_FAILURE_TYPES}")
    with SessionFactory() as session:
        row = session.query(DecisionLog).filter(DecisionLog.id == decision_id).first()
        if row is None:
            return False
        row.failure_type = failure_type
        row.failure_note = failure_note
        session.commit()
    return True


def get_failure_attribution_stats() -> dict:
    """
    Return breakdown of attributed failures by type.
    Used for Admin dashboard summary.
    """
    with SessionFactory() as session:
        rows = (
            session.query(DecisionLog.failure_type, func.count(DecisionLog.id))
            .filter(
                DecisionLog.accuracy_score < 0.5,
                DecisionLog.verified == True,
                DecisionLog.failure_type.isnot(None),
            )
            .group_by(DecisionLog.failure_type)
            .all()
        )
        unattributed = (
            session.query(func.count(DecisionLog.id))
            .filter(
                DecisionLog.accuracy_score < 0.5,
                DecisionLog.verified == True,
                DecisionLog.failure_type.is_(None),
            )
            .scalar()
        )
    return {
        "by_type":       {ft: cnt for ft, cnt in rows},
        "unattributed":  unattributed or 0,
        "total_failures": (unattributed or 0) + sum(cnt for _, cnt in rows),
    }


# ── Snapshot cache API ─────────────────────────────────────────────────────────

def save_regime_snapshot(regime_result) -> None:
    """
    Persist a RegimeResult to the cache table.
    Idempotent: upserts on (as_of_date, train_end).
    regime_result must be a RegimeResult dataclass from engine.regime.
    """
    with SessionFactory() as session:
        existing = (
            session.query(RegimeSnapshot)
            .filter_by(as_of_date=regime_result.date, train_end=regime_result.date)
            .first()
        )
        if existing:
            existing.regime       = regime_result.regime
            existing.p_risk_on    = regime_result.p_risk_on
            existing.p_risk_off   = regime_result.p_risk_off
            existing.method       = regime_result.method
            existing.n_obs        = regime_result.n_obs
            existing.yield_spread = regime_result.yield_spread
            existing.vix          = regime_result.vix
            existing.warning      = regime_result.warning or ""
            existing.computed_at  = datetime.datetime.utcnow()
        else:
            session.add(RegimeSnapshot(
                as_of_date   = regime_result.date,
                train_end    = regime_result.date,
                regime       = regime_result.regime,
                p_risk_on    = regime_result.p_risk_on,
                p_risk_off   = regime_result.p_risk_off,
                method       = regime_result.method,
                n_obs        = regime_result.n_obs,
                yield_spread = regime_result.yield_spread,
                vix          = regime_result.vix,
                warning      = regime_result.warning or "",
            ))
        session.commit()


def get_regime_snapshot(as_of_date: datetime.date):
    """
    Load a cached RegimeResult for the given date.
    Returns a RegimeResult-compatible dict on hit, None on miss.
    Only returns results computed within the last 24 hours to avoid stale data.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    with SessionFactory() as session:
        row = (
            session.query(RegimeSnapshot)
            .filter(
                RegimeSnapshot.as_of_date == as_of_date,
                RegimeSnapshot.computed_at >= cutoff,
            )
            .first()
        )
        if row is None:
            return None
        return {
            "date":          row.as_of_date,
            "regime":        row.regime,
            "p_risk_on":     row.p_risk_on,
            "p_risk_off":    row.p_risk_off,
            "method":        row.method or "cached",
            "n_obs":         row.n_obs or 0,
            "yield_spread":  row.yield_spread,
            "vix":            row.vix,
            "warning":       row.warning or "",
        }


def save_signal_snapshot(
    as_of_date:      datetime.date,
    lookback_months: int,
    skip_months:     int,
    signals_df,
) -> None:
    """
    Persist a signal DataFrame to the cache table.
    Idempotent: upserts on (as_of_date, lookback_months, skip_months).
    signals_df must be a pandas DataFrame (output of get_signal_dataframe).
    """
    import pandas as pd
    if not isinstance(signals_df, pd.DataFrame) or signals_df.empty:
        return
    signals_json = signals_df.to_json(orient="split")
    with SessionFactory() as session:
        existing = (
            session.query(SignalSnapshot)
            .filter_by(
                as_of_date=as_of_date,
                lookback_months=lookback_months,
                skip_months=skip_months,
            )
            .first()
        )
        if existing:
            existing.signals_json = signals_json
            existing.sector_count = len(signals_df)
            existing.computed_at  = datetime.datetime.utcnow()
        else:
            session.add(SignalSnapshot(
                as_of_date      = as_of_date,
                lookback_months = lookback_months,
                skip_months     = skip_months,
                signals_json    = signals_json,
                sector_count    = len(signals_df),
            ))
        session.commit()


def get_signal_snapshot(
    as_of_date:      datetime.date,
    lookback_months: int,
    skip_months:     int,
    max_age_hours:   int = 24,
):
    """
    Load a cached signal DataFrame for the given parameters.
    Returns a pandas DataFrame on hit, None on miss or if cache is stale.
    """
    import pandas as pd
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=max_age_hours)
    with SessionFactory() as session:
        row = (
            session.query(SignalSnapshot)
            .filter(
                SignalSnapshot.as_of_date      == as_of_date,
                SignalSnapshot.lookback_months == lookback_months,
                SignalSnapshot.skip_months     == skip_months,
                SignalSnapshot.computed_at     >= cutoff,
            )
            .first()
        )
        if row is None:
            return None
        try:
            return pd.read_json(row.signals_json, orient="split")
        except Exception:
            return None


# ── HITL slim refactor 2026-05-05 helpers (recovered 2026-05-14) ─────────────
# These two helpers are imported by engine/daily_batch.py (Layer 2 tactical
# auto-entry + Layer 4 month-end rebalance) and were referenced in commit
# aab5f47 but their definitions were lost. Restored here as minimal-viable
# routine_review trace writers to unblock app bootstrap.
#
# Spec: docs/decisions/hitl_architecture_audit_2026-05-05.md
#   approval_class = "routine_review" partitions auto-executed audit rows
#   from governance queue. status="approved" + resolved_by="auto_*" at write
#   time (the action already happened; this is an audit trail entry).


def create_routine_review_trace(
    *,
    approval_type:        str,
    sector:               str,
    ticker:               str,
    triggered_condition:  str,
    triggered_date:       "datetime.date",
    triggered_price:      float | None       = None,
    suggested_weight:     float | None       = None,
    watchlist_entry_id:   int  | None        = None,
    position_rank:        str  | None        = None,
    contradicts_quant:    bool | None        = None,
    llm_confidence:       int  | None        = None,
    spec_reference:       str  | None        = None,
    resolved_by:          str                = "auto",
) -> "PendingApproval":
    """Write a routine_review PendingApproval audit trace (HITL slim, 2026-05-05).

    The actual execution (entry / rebalance) has already happened upstream in
    daily_batch — this row exists purely for the Operations Routine Timeline.
    """
    now = datetime.datetime.utcnow()
    with SessionFactory() as session:
        pa = PendingApproval(
            approval_type       = approval_type,
            priority            = "normal",
            watchlist_entry_id  = watchlist_entry_id,
            sector              = sector,
            ticker              = ticker,
            triggered_condition = triggered_condition,
            triggered_date      = triggered_date,
            triggered_price     = triggered_price,
            suggested_weight    = suggested_weight,
            position_rank       = position_rank,
            status              = "approved",
            resolved_at         = now,
            resolved_by         = resolved_by,
            contradicts_quant   = bool(contradicts_quant) if contradicts_quant is not None else False,
            llm_confidence      = llm_confidence,
            approval_class      = "routine_review",
        )
        if spec_reference is not None:
            try:
                pa.review_rationale = f"spec_reference={spec_reference}"
            except Exception:
                pass
        session.add(pa)
        session.commit()
        session.refresh(pa)
        session.expunge(pa)
        return pa


def write_routine_review_audit_row(
    *,
    approval_type:        str,
    sector:               str,
    ticker:               str,
    triggered_condition:  str,
    triggered_date:       "datetime.date",
    triggered_price:      float | None       = None,
    suggested_weight:     float | None       = None,
    resolved_by:          str                = "auto_companion",
    spec_reference:       str  | None        = None,
) -> "PendingApproval":
    """Companion audit row variant of `create_routine_review_trace`.

    Used by daily_batch when multiple sectors share a single rebalance event:
    primary sector uses create_routine_review_trace; subsequent sectors use
    this helper. Same routine_review approval_class, distinct resolved_by.
    """
    return create_routine_review_trace(
        approval_type       = approval_type,
        sector              = sector,
        ticker              = ticker,
        triggered_condition = triggered_condition,
        triggered_date      = triggered_date,
        triggered_price     = triggered_price,
        suggested_weight    = suggested_weight,
        spec_reference      = spec_reference,
        resolved_by         = resolved_by,
    )
