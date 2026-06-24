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

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, Index, Integer, LargeBinary,
    String, Text, UniqueConstraint, create_engine, func, text,
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

    # ── DL-P0 attribution fields (master_backlog 第七节, 2026-05-03) ───────────
    # Populated by save_decision (weight_before/after/exit_reason) at decision time
    # and by verify_pending_decisions (active_return / mae / mfe) at verification.
    #
    # weight_before     : portfolio target weight for this sector BEFORE LLM debate
    #                     adjustment was applied (= quant baseline weight)
    # weight_after      : portfolio target weight AFTER LLM debate adjustment
    #                     (= weight_before + scaled_adj, clamped to MAX_WEIGHT)
    # active_return     : ETF return − SPY return at decision horizon (≈ 20-day default).
    #                     Captures *active* alpha vs market-cap benchmark, not raw return.
    # mae               : Maximum Adverse Excursion — most-negative cumulative return
    #                     during holding period (peak-to-trough drawdown × direction).
    #                     Always ≤ 0; large negative value = position spent time deep underwater.
    # mfe               : Maximum Favorable Excursion — most-positive cumulative return
    #                     during holding period. Always ≥ 0; high MFE + low active_return =
    #                     "gave back gains, exit timing matters".
    # exit_reason       : why position closed; one of:
    #                     "signal_flip"         — TSMOM sign changed
    #                     "regime_change"       — risk-off compression triggered
    #                     "threshold"           — weight fell below MIN_WEIGHT cap
    #                     "expired"             — held past invalidation_horizon
    #                     "barrier_tp"          — Triple-Barrier take-profit hit
    #                     "barrier_sl"          — Triple-Barrier stop-loss hit
    #                     "barrier_time"        — Triple-Barrier time-out
    #                     "manual"              — human override
    weight_before  = Column(Float,        nullable=True)
    weight_after   = Column(Float,        nullable=True)
    active_return  = Column(Float,        nullable=True)
    mae            = Column(Float,        nullable=True)
    mfe            = Column(Float,        nullable=True)
    exit_reason    = Column(String(30),   nullable=True)

    # ── S2 Reflection-memory audit (spec §5.3, 2026-05-04) ─────────────────────
    # Records which past reflections were RAG-retrieved and injected into the
    # debate prompt for this decision. Lets us back-trace decision quality
    # against the reflections it consumed (capability metric, not alpha).
    # NULL = decision predates S2 hook or retrieval failed (graceful no-op).
    reflections_injected_count = Column(Integer, nullable=True)
    reflections_injected_ids   = Column(Text,    nullable=True)  # JSON list[int]

    # ── S3 Pre-Registration Enforcement (spec §3 Sprint 2, 2026-05-04) ────────
    # git-blob hash of the spec governing this decision (e.g. the sector
    # pipeline unification spec). NULL = decision predates S3 or its source
    # had no registered spec. Used by HARKing R3 (unannounced trial) to flag
    # decisions whose spec hash is not in the SpecRegistry.
    spec_hash = Column(String(64), nullable=True)

    # ── MS-1 Multi-Sleeve Commit (spec_factor_lab.md §6 + project_final_vision_hybrid_2026-05-10) ─
    # Identifies which portfolio sleeve this decision belongs to:
    #   'etf_l1'    = ETF tier 1/2 production (QL01 BAB + Multivariate v3)
    #   'ss_sp500'  = single-stock S&P 500 (Wave B post-WRDS activation)
    # Existing rows backfill to 'etf_l1' (only ETF universe was live at MS-1 ship).
    # Cross-sleeve attribution: per-sleeve TWR/MWR/HPR added in MS-3 P-FUND.
    sleeve_id = Column(String(20), nullable=False, default="etf_l1", server_default="etf_l1")


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
    # Wave 7 (2026-05-07): cap-fix restart era marker.
    # 'live'              = post-2026-05-04 cap-enforced data
    # 'pre_cap_fix_legacy'= pre-2026-05-04, written under buggy cap path
    era            = Column(String(32),  nullable=True, default="live", server_default="live")
    # MS-1 (2026-05-10): multi-sleeve commit per project_final_vision_hybrid.
    # 'etf_l1'    = ETF tier 1/2 production (QL01 BAB + Multivariate v3 overlay)
    # 'ss_sp500'  = single-stock S&P 500 sleeve (Wave B post-WRDS activation)
    # Existing rows backfill to 'etf_l1' (only ETF universe was live at MS-1 ship).
    sleeve_id      = Column(String(20),  nullable=False, default="etf_l1", server_default="etf_l1")

    __table_args__ = (
        UniqueConstraint("snapshot_date", "sector", "track", "sleeve_id",
                         name="uq_pos_date_sector_track_sleeve"),
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
    # TL-ATTR-2: approval lag attribution fields
    trigger_price      = Column(Float,   nullable=True)    # price at the time the approval was triggered
    execution_lag_days = Column(Integer, nullable=True)    # calendar days from trigger to execution
    # Wave 7 (2026-05-07): cap-fix restart era marker (mirror SimulatedPosition.era).
    era                = Column(String(32), nullable=True, default="live", server_default="live")
    # MS-1 (2026-05-10): multi-sleeve commit (mirrors SimulatedPosition.sleeve_id).
    sleeve_id          = Column(String(20), nullable=False, default="etf_l1", server_default="etf_l1")


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
    # MS-1 (2026-05-10): multi-sleeve commit (mirrors SimulatedPosition.sleeve_id).
    sleeve_id    = Column(String(20), nullable=False, default="etf_l1", server_default="etf_l1")

    __table_args__ = (
        UniqueConstraint("return_month", "sector", "sleeve_id",
                         name="uq_ret_month_sector_sleeve"),
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
    # P3b (2026-05-07): tail-risk shadow features. CAPTURED but NOT yet
    # consumed by regime label / p_risk_on logic — see engine/regime.py
    # _fetch_vvix / _fetch_skew. Future regime model amendment may flip
    # them on as MSM endogenous inputs after pre-reg + power analysis.
    vvix         = Column(Float,       nullable=True)   # CBOE volatility-of-VIX
    skew         = Column(Float,       nullable=True)   # CBOE SKEW index (tail-risk premium)
    warning      = Column(Text,        nullable=True)
    # Code-version stamp (regime._REGIME_CODE_VERSION at write time).
    # Reads filter on the current version so MSM/label/window changes in
    # regime.py automatically invalidate stale rows. Without this, a
    # backtest re-run after a code fix returns the previous version's
    # cached regime for any date that was already computed in the last 24h.
    code_version = Column(String(50),  nullable=True)
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


# ── V4.0 Agent ORM ───────────────────────────────────────────────────────────

class RiskNarrativeLog(Base):
    """
    风控官 Agent 每日叙事风险输出（≤3条/日）。
    confidence_weight 快照记录生成时的置信权重，供 ERA 回填。
    """
    __tablename__ = "risk_narrative_logs"

    id                = Column(Integer,  primary_key=True, autoincrement=True)
    generated_at      = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    as_of_date        = Column(Date,     nullable=False)
    risk_type         = Column(String(50))   # geopolitical / policy / liquidity / contagion / other
    affected_tickers  = Column(Text)         # JSON list
    narrative         = Column(Text)         # 叙事描述（≤300字）
    suggested_action  = Column(String(100))  # 减仓 / 观察 / 无需行动
    severity          = Column(String(20))   # low / medium / high
    confidence_weight = Column(Float)        # 生成时置信权重快照
    approval_id       = Column(Integer, nullable=True)   # FK → pending_approvals.id
    supervisor_verdict = Column(String(20), nullable=True)  # accepted / dismissed / noted
    era_verdict       = Column(String(20), nullable=True)   # ERA 验证后回填


class MemoryCuratorReport(Base):
    """
    记忆管理 Agent 月度报告。
    patterns_found: JSON list of PatternCandidate dicts.
    bh_correction_passed: JSON list of pattern_ids that passed BH correction.
    injected_to_skill_library: True if ≥1 confirmed pattern was injected.
    """
    __tablename__ = "memory_curator_reports"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    report_month            = Column(String(7), unique=True)   # "2026-04"
    generated_at            = Column(DateTime, default=datetime.datetime.utcnow)
    n_decisions_scanned     = Column(Integer, default=0)
    patterns_found          = Column(Text)    # JSON list of PatternCandidate
    bh_correction_passed    = Column(Text)    # JSON list of confirmed pattern_ids
    injected_to_skill_library = Column(Boolean, default=False)
    report_summary          = Column(Text)    # LLM 生成的月度摘要


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


# ── E-pivot: Paper Trading Three-Arm Forward Ablation ────────────────────────
#
# Spec: docs/spec_paper_trading_three_arm_e.md (2026-05-03).
# Forward-only ablation to validate sector_pipeline LLM debate alpha contribution.
#
# Three arms snapshot at every month-end:
#   A baseline    : TSMOM + vol-targeting, NO LLM debate
#   B production  : current sector_pipeline LLM debate (with confidence-scaled adj)
#   C placebo     : random N(0, σ_real_debate) adjustment
#
# SimulatedPosition.track field (already String(10)) accepts new values:
#   "paper_A" / "paper_B" / "paper_C"  — 7 chars, fits existing schema unchanged.
# Use module constants below as the canonical source for valid track values.
PAPER_TRADING_ARMS: tuple[str, ...] = ("A", "B", "C")
PAPER_TRADING_TRACKS: tuple[str, ...] = tuple(f"paper_{a}" for a in PAPER_TRADING_ARMS)


class PaperTradingRun(Base):
    """
    Per-month per-arm snapshot of forward paper trading.

    Insertion: each month-end orchestrator cycle persists 3 rows (one per arm).
    Backfill: `next_month_return` and `cum_nav` filled at t+1 month-end.

    UniqueConstraint on (as_of_date, arm) prevents duplicate monthly snapshots.
    """
    __tablename__ = "paper_trading_runs"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    as_of_date           = Column(Date,       nullable=False)        # month-end snapshot date
    arm                  = Column(String(10), nullable=False)        # "A" / "B" / "C"
    weights_json         = Column(Text,       nullable=False)        # JSON: {sector: weight}
    sector_debate_output = Column(Text,       nullable=True)         # JSON, only arm B
    placebo_seed         = Column(Integer,    nullable=True)         # only arm C
    placebo_adjustments  = Column(Text,       nullable=True)         # JSON {sector: adj}, only arm C (Path 1 redesign 2026-05-03)
    next_month_return    = Column(Float,      nullable=True)         # backfilled at t+1
    cum_nav              = Column(Float,      nullable=True)         # cumulative NAV (start = 1.0)
    notes                = Column(Text,       nullable=True)
    created_at           = Column(DateTime,   nullable=False, default=datetime.datetime.utcnow)

    # B-PLUS-PROD migration 2026-05-05: signal baseline tag.
    # Pre-2026-05-05 rows used TSMOM(12,1) baseline; from 2026-05-05 onwards
    # rows use QL01 BAB (Frazzini-Pedersen 2014). Verdict computation MUST
    # filter on a single signal_baseline (typically the most recent / production)
    # because mixing baselines invalidates the LLM ablation comparison.
    # Spec: docs/decisions/b_plus_prod_migration_2026-05-05.md.
    signal_baseline      = Column(String(20), nullable=True)            # "tsmom" / "ql01_bab"

    __table_args__ = (
        UniqueConstraint("as_of_date", "arm", name="uq_paper_trading_date_arm"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# S2 Agent Reflection Memory (2026-05-04)
# Spec: docs/spec_agent_reflection_memory.md v1.0
# ─────────────────────────────────────────────────────────────────────────────

class AgentReflection(Base):
    """
    Per-decision reflection memo with retrievable embedding.

    Each agent decision (e.g., sector_pipeline LLM debate) is followed by a
    structured reflection memo (4 sections: CONTEXT/DECISION/OUTCOME/LESSON)
    once realized outcome backfills. The memo + sentence-transformer embedding
    are persisted here so future decisions can RAG-retrieve relevant past
    reflections as prompt context.

    Spec: docs/spec_agent_reflection_memory.md §2 (frozen v1.0).
    """
    __tablename__ = "agent_reflections"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    agent_id        = Column(String(50),  nullable=False, index=True)
    # "sector_pipeline" / "macro_research" / future agent identifiers

    decision_ref_id = Column(Integer, nullable=True)
    # FK to decision_logs.id (NULL for paper-trading-only or audit-only sources)

    decision_date   = Column(Date,    nullable=False, index=True)

    decision_summary = Column(Text, nullable=False)
    # JSON: {sector, direction, confidence, rationale_excerpt, ...}

    realized_outcome = Column(Float, nullable=True)
    # e.g., next-month return for the predicted sector. NULL until backfill.

    hit_flag        = Column(String(10), nullable=True)
    # "hit" / "miss" / "partial" / "neutral" / "pending" — rule-based, zero LLM

    factor_context  = Column(Text, nullable=True)
    # JSON snapshot of B++ factor IC / ICIR / β-decomp at decision_date

    reflection_text = Column(Text, nullable=False)
    # LLM-generated 4-section narrative (200-400 chars)

    embedding       = Column(Text, nullable=True)
    # JSON-serialized list[float], 384-dim sentence-transformer

    embedding_model = Column(String(80), nullable=True,
                             default="sentence-transformers/all-MiniLM-L6-v2")

    created_at      = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    updated_at      = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_reflection_agent_date", "agent_id", "decision_date"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# S3 Pre-Registration Enforcement (2026-05-04)
# Spec: docs/spec_pre_registration_enforcement.md  (spec_hash 292fdd6039f90d05)
# ─────────────────────────────────────────────────────────────────────────────

class SpecRegistry(Base):
    """
    Registry of pre-registered specifications. Each row records when a
    `docs/spec_*.md` file was first locked, its git-blob-style content hash,
    and an append-only ledger of subsequent amendments. Used to compute
    pre-registration contributions to EFFECTIVE_N_TRIALS and to surface
    HARKing-style integrity violations (silent edits, threshold drift, etc.).

    Spec §2.2 (frozen 2026-05-04).
    """
    __tablename__ = "spec_registry"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    spec_path       = Column(String(255), nullable=False, unique=True)
    git_blob_hash   = Column(String(64), nullable=False)  # initial hash at register
    current_hash    = Column(String(64), nullable=False)  # latest observed hash
    registered_at   = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    amendment_log   = Column(Text, nullable=False, default="[]")
    # JSON: [{"at": ISO8601, "kind": str, "reason": str, "new_hash": str, "n_trials_added": int}, ...]
    status          = Column(String(16), nullable=False, default="active")
    # active / superseded / archived
    retro_registered = Column(Boolean, nullable=False, default=False)
    # True = registered after the spec already existed; does NOT count toward
    # forward-integrity n_trials contribution
    first_referenced_at      = Column(DateTime, nullable=True)
    n_trials_contributed     = Column(Integer, nullable=False, default=1)
    last_validated_at        = Column(DateTime, nullable=True)

    # ── P-LAB Factor Lab columns (2026-05-08) ────────────────────────────────
    # Spec: docs/spec_factor_lab.md §2.3
    # Legacy spec rows (pre-LAB v1-v8 hypothesis tests) keep these as NULL —
    # rendered as "pre-LAB legacy" by the UI, not part of the state machine.
    lab_state    = Column(String(32),  nullable=True)
    # FactorState enum value: DRAFT / PROPOSED / BLOCKED_UNDERPOWERED /
    # REGISTERED / TESTING / PASS / MARGINAL / FAIL / FAIL_UNDERPOWERED.
    factor_kind  = Column(String(32),  nullable=True)
    # 'production_swap' | 'overlay' | 'shadow' | 'infrastructure_spec' | NULL


# ─────────────────────────────────────────────────────────────────────────────
# P-FUND Investor-Grade Performance Reporting (2026-05-04)
# Spec: docs/spec_performance_reporting_v1.md (sha256[:16]=f1c9b693f7a6a6df)
# ─────────────────────────────────────────────────────────────────────────────

class CashFlow(Base):
    """
    Supervisor-controlled and portfolio-internal cash flows.

    External flows (deposit / withdraw / fee) drive Modified Dietz sub-period
    splits and XIRR computation. Internal flows (dividend / coupon / interest)
    affect NAV but do NOT split sub-periods (per GIPS 2020 §III.5.A.20).

    Sign convention: amount_usd > 0 = into portfolio.

    Spec §3.1 (frozen 2026-05-04).
    """
    __tablename__ = "cash_flows"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    flow_date     = Column(Date,    nullable=False, index=True)
    flow_type     = Column(String(16), nullable=False)
    # External: deposit / withdraw / fee
    # Internal: dividend / coupon / interest
    amount_usd    = Column(Float,   nullable=False)
    is_external   = Column(Boolean, nullable=False)
    # External = supervisor-controlled, splits TWR sub-periods + drives MWR
    # Internal = portfolio-internal, MTM-equivalent

    status        = Column(String(16), nullable=False, default="pending")
    # pending / applied / cancelled — only `applied` rows enter NAV rollup

    supervisor_id = Column(String(80), nullable=True)
    approval_id   = Column(Integer,    nullable=True)
    # FK → PendingApproval.id when require_approval=True; NULL for direct
    # internal flows (dividend / fee) that bypass the gate.

    notes         = Column(Text,    nullable=True)
    created_at    = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)
    applied_at    = Column(DateTime, nullable=True)
    # Stamped when status flips pending → applied


class PortfolioNavSnapshot(Base):
    """
    Daily NAV roll with external-flow normalization.

    Three NAV states per day:
      nav_open       = NAV at start of day, before any external flow
      nav_after_flow = nav_open + external_flow (deposit/withdraw applied)
      nav_close      = end-of-day NAV after MTM

    daily_modified_dietz pre-computed for fast TWR aggregation.

    Spec §3.2 (frozen 2026-05-04).
    """
    __tablename__ = "portfolio_nav_snapshots"

    snapshot_date         = Column(Date, primary_key=True)
    nav_open              = Column(Float, nullable=False)
    external_flow         = Column(Float, nullable=False, default=0.0)
    nav_after_flow        = Column(Float, nullable=False)
    nav_close             = Column(Float, nullable=False)
    gross_pnl             = Column(Float, nullable=False)
    benchmark_close       = Column(Float, nullable=True)
    daily_modified_dietz  = Column(Float, nullable=True)
    notes                 = Column(Text,  nullable=True)
    created_at            = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)


class HARKingFlag(Base):
    """
    Flags raised by `engine.preregistration.detect_harking()` against the
    spec registry. Each flag references one rule (R1-R4) and one spec_path.

    Spec §2.4 (frozen 2026-05-04).
    """
    __tablename__ = "harking_flags"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    rule          = Column(String(8),  nullable=False)   # R1 / R2 / R3 / R4
    spec_path     = Column(String(255), nullable=False)
    severity      = Column(String(16), nullable=False)   # CRITICAL / HIGH / MEDIUM
    detected_at   = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)
    resolved_at   = Column(DateTime,    nullable=True)
    notes         = Column(Text,        nullable=True)


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

    # P-AUDIT v1 (2026-05-04, spec docs/spec_supervisor_approval_panel_v1.md):
    # Mandatory fields when supervisor approves/rejects via the audit panel.
    # NULL on legacy rows (44 historical) and on programmatic auto-resolutions.
    review_rationale   = Column(Text,        nullable=True)
    # Free-text supervisor justification, ≥10 chars enforced at the UI gate.
    review_category    = Column(String(32),  nullable=True)
    # Enum: signal_confirmed / regime_driven / supervisor_discretion /
    #       risk_override / cash_flow_routine / other

    # P-AUDIT v1 amendment NARRATIVE-C v2 (2026-05-04, spec § F-pre.2 Block C):
    # Frozen narrative snapshot + SHA-256 hash chain. CFA GIPS 2020 §III.A.18 +
    # SEC 17a-4(b) reference + López de Prado 2018 §10 hash chain. Append-only;
    # written by _apply_decision before resolve. NULL on legacy / auto rows.
    review_narrative_snapshot = Column(Text,        nullable=True)
    review_narrative_hash     = Column(String(64),  nullable=True)
    prev_narrative_hash       = Column(String(64),  nullable=True)

    # 2026-05-05 HITL slim refactor (D3 of B-pragmatic-v2 sprint).
    # Spec: docs/decisions/hitl_architecture_audit_2026-05-05.md
    #
    # approval_class partitions the queue into Governance (requires supervisor
    # decision) vs routine_review (auto-executed audit trail) vs llm_output
    # (S6 anomaly_screener post-D4). Supervisor confirmed 5 governance + 1
    # llm_output + 1 routine_review on 2026-05-05.
    approval_class = Column(String(16), nullable=False,
                             default="governance",
                             server_default=text("'governance'"))
    # Values: governance | routine_review | llm_output

    # Approval ergonomics — populated at resolve time. Used by Operations
    # Ergonomics tab to detect rubber-stamping (approvals < 30 s = warning).
    approval_latency_seconds = Column(Integer, nullable=True)

    # Categorical rejection reason; complements free-text rejection_reason.
    # Enum: insufficient_evidence | contradicts_quant | risk_breach |
    #       harking_flag | cash_compliance | arm_lock | other
    rejection_category = Column(String(32), nullable=True)

    # Post-hoc supervisor note on routine_review entries (auto_executed). Not
    # part of hash chain because added after resolve_at; provides annotation
    # capability without breaking SHA-256 chain integrity (P-AUDIT v1 NARRATIVE-C
    # v2 hash includes only fields frozen at resolve time).
    post_hoc_note = Column(Text, nullable=True)

    # 2026-05-07 dedup (Wave 7 follow-up): same persistent risk condition was
    # creating a fresh PA every day (e.g. XLK weight > soft_cap fired 3 days
    # in a row → PA #14 / #17 / #19, all "the same problem").  Added:
    #   condition_signature : stable fingerprint = approval_type|sector|ticker
    #                         |normalized_kind (numbers stripped from reason)
    #   last_seen_at        : updated each time the condition re-fires
    #   consecutive_days_seen: ≥2 surfaces "持续 N 天" badge to supervisor
    # _add_risk_approval upserts on signature instead of date.
    condition_signature   = Column(String(120), nullable=True)
    last_seen_at          = Column(DateTime,    nullable=True)
    consecutive_days_seen = Column(Integer,     nullable=False, default=1,
                                    server_default=text("1"))


class AnomalyFlag(Base):
    """
    S6 anomaly_screener flag (D4.1 of B-pragmatic-v2 sprint, 2026-05-05).
    Spec: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md.

    One row per (detector, scan_date, ticker). Three detectors run in parallel:
      - rule_baseline_a  (rule-only, ≤ 6 if-then; price + concentration)
      - rule_baseline_b  (baseline_a + macro_research forecast as feature)
      - llm              (Gemini 2.5 Flash with thinking; news + portfolio)

    M1 forward event verification (precision+recall+F1) is denormalized into
    this row at T+K (K = horizon_days, typically 5). Recall is computed by
    joining against AnomalyUniverseEvent.
    """
    __tablename__ = "anomaly_flags"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    created_at          = Column(DateTime, default=datetime.datetime.utcnow)

    detector            = Column(String(20), nullable=False)
    # rule_baseline_a | rule_baseline_b | llm
    scan_date           = Column(Date, nullable=False)
    sector              = Column(String(50), nullable=False)
    ticker              = Column(String(20), nullable=False)

    # Classification
    event_class         = Column(String(32), nullable=False)
    # price_spike | news_driven | concentration | cross_asset | volume_spike | drawdown
    confidence_likert   = Column(Integer, nullable=False)   # 1-5 anchored
    horizon_days        = Column(Integer, nullable=False, default=5)

    # Evidence
    evidence_summary    = Column(Text, nullable=True)        # ≤ 200 chars factual
    triggering_rules    = Column(Text, nullable=True)        # JSON list (rule-based) or null (LLM)
    news_refs           = Column(Text, nullable=True)        # JSON list of news URLs (LLM only)

    # LLM reproducibility (D1 Invariant 1; spec §4)
    llm_model_version   = Column(String(50), nullable=True)
    llm_prompt_hash     = Column(String(64), nullable=True)
    llm_response_hash   = Column(String(64), nullable=True)
    llm_cost_usd        = Column(Float,      nullable=True, default=0.0)
    llm_input_tokens    = Column(Integer,    nullable=True)
    llm_output_tokens   = Column(Integer,    nullable=True)

    # Spec hash (registered in D4.8)
    spec_hash           = Column(String(64), nullable=True)

    # Linkage to PendingApproval queue (when promoted to llm_output category)
    pending_approval_id = Column(Integer,    nullable=True)

    # Supervisor M2 labeling
    supervisor_useful   = Column(Boolean,    nullable=True)
    # True (useful) | False (noise) | None (not yet labeled)
    supervisor_acted    = Column(Boolean,    nullable=True)
    # True if portfolio changed within 7 days of accept; for M3 case study
    supervisor_label_at = Column(DateTime,   nullable=True)

    # M1 forward verification — denormalized at T+K
    event_occurred      = Column(Boolean,    nullable=True)
    # null = not yet verified (window not closed); True/False = verified
    verified_at         = Column(DateTime,   nullable=True)
    event_date          = Column(Date,       nullable=True)
    event_return        = Column(Float,      nullable=True)   # daily return on event_date
    event_sigma         = Column(Float,      nullable=True)   # # of σ_60d

Index("ix_anomaly_flags_detector_scan", AnomalyFlag.detector, AnomalyFlag.scan_date)
Index("ix_anomaly_flags_ticker_scan",   AnomalyFlag.ticker,   AnomalyFlag.scan_date)
Index("ix_anomaly_flags_supervisor_label_at", AnomalyFlag.supervisor_label_at)
Index("ix_pending_approvals_status_triggered_date", PendingApproval.status, PendingApproval.triggered_date)


class AnomalyUniverseEvent(Base):
    """
    All M1-meaningful events in scope universe, regardless of whether any
    detector flagged them. Populated by daily universe sweep (D4.5).

    Used for recall computation:
       recall(detector) = (events flagged by detector) / (total universe events)

    Event definition (spec §2): abs daily return > 2σ_60d on a ticker in
    portfolio holdings ∪ flagged tickers, on or after 2026-01-15 (post LLM cutoff).
    """
    __tablename__ = "anomaly_universe_events"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    event_date      = Column(Date, nullable=False)
    sector          = Column(String(50), nullable=False)
    ticker          = Column(String(20), nullable=False)

    event_return    = Column(Float,  nullable=False)
    event_sigma     = Column(Float,  nullable=False)
    rolling_vol_60d = Column(Float,  nullable=True)

    # Detection coverage flags — set TRUE if any flag from that detector
    # had scan_date ≤ event_date ≤ scan_date + horizon AND ticker matches.
    detected_by_baseline_a = Column(Boolean, default=False, nullable=False)
    detected_by_baseline_b = Column(Boolean, default=False, nullable=False)
    detected_by_llm        = Column(Boolean, default=False, nullable=False)

    # JSON list of matching anomaly_flags.id (for traceability)
    matched_flag_ids       = Column(Text,    nullable=True)

    discovered_at   = Column(DateTime, default=datetime.datetime.utcnow)

Index("ix_anomaly_universe_events_date", AnomalyUniverseEvent.event_date)
Index("ix_anomaly_universe_events_ticker", AnomalyUniverseEvent.ticker, AnomalyUniverseEvent.event_date,
      unique=True)


# ── CFTC Commitments of Traders (P3a, 2026-05-07) ────────────────────────────

class CftcCotWeekly(Base):
    """
    One row per (futures contract × Tuesday-of-record × report-type).

    Two CFTC archives are unified into this single table via the
    ``report_type`` discriminator:

        report_type='disagg_fut'  ←  Disaggregated futures-only
            URL: fut_disagg_xls_<year>.zip  (commodities; ~13k rows/year)
            Trader categories: prod_merc / swap / m_money / other_rept / non_rept

        report_type='tff_fut'     ←  Traders in Financial Futures (TFF)
            URL: fut_fin_xls_<year>.zip     (equity / rates / VIX / FX; ~3k rows/year)
            Trader categories: dealer / asset_mgr / lev_money / other_rept / non_rept

    Both schemas share open_interest + other_rept_long/short + non_rept_long/short;
    the report-specific columns are NULL when not applicable. Use
    ``report_type`` to filter to the correct subset for query-time analysis.

    Used downstream by P3c (COT-conditional BAB extension test): equity-index
    asset_mgr vs lev_money positioning + treasury / VIX positioning as a
    regime / sentiment overlay on the BAB factor.
    """
    __tablename__ = "cftc_cot_weekly"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    contract_market_code  = Column(String(16), nullable=False)
    report_date           = Column(DateTime,   nullable=False)
    report_type           = Column(String(16), nullable=False, default="disagg_fut")

    market_name           = Column(String(120), nullable=True)
    market_code           = Column(String(8),   nullable=True)
    commodity_code        = Column(String(8),   nullable=True)

    open_interest         = Column(Integer, nullable=False, default=0)

    # ── Disaggregated (commodity) trader categories ──────────────────────
    prod_merc_long        = Column(Integer, nullable=True)   # commercial hedger
    prod_merc_short       = Column(Integer, nullable=True)
    swap_long             = Column(Integer, nullable=True)   # swap dealer
    swap_short            = Column(Integer, nullable=True)
    swap_spread           = Column(Integer, nullable=True)
    m_money_long          = Column(Integer, nullable=True)   # managed money
    m_money_short         = Column(Integer, nullable=True)
    m_money_spread        = Column(Integer, nullable=True)

    # ── TFF (financial) trader categories ────────────────────────────────
    dealer_long           = Column(Integer, nullable=True)   # dealer / intermediary
    dealer_short          = Column(Integer, nullable=True)
    dealer_spread         = Column(Integer, nullable=True)
    asset_mgr_long        = Column(Integer, nullable=True)   # asset manager / institutional
    asset_mgr_short       = Column(Integer, nullable=True)
    asset_mgr_spread      = Column(Integer, nullable=True)
    lev_money_long        = Column(Integer, nullable=True)   # leveraged funds (hedge funds)
    lev_money_short       = Column(Integer, nullable=True)
    lev_money_spread      = Column(Integer, nullable=True)

    # ── Shared (both report types) ───────────────────────────────────────
    other_rept_long       = Column(Integer, nullable=False, default=0)
    other_rept_short      = Column(Integer, nullable=False, default=0)
    other_rept_spread     = Column(Integer, nullable=True)
    non_rept_long         = Column(Integer, nullable=False, default=0)
    non_rept_short        = Column(Integer, nullable=False, default=0)

    fetched_at            = Column(DateTime, nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("contract_market_code", "report_date", "report_type",
                         name="uq_cftc_cot_weekly_code_date_type"),
    )

Index("ix_cftc_cot_weekly_report_date",
      CftcCotWeekly.report_date)
Index("ix_cftc_cot_weekly_market_name_date",
      CftcCotWeekly.market_name, CftcCotWeekly.report_date)
Index("ix_cftc_cot_weekly_report_type",
      CftcCotWeekly.report_type, CftcCotWeekly.report_date)


# ── Sprint B (2026-05-13): Paper-trade combined per-strategy daily log ────────
#
# Stores per-strategy daily snapshot for the 4-component paper-trade
# orchestrator (engine.portfolio.paper_trade_combined). Combined portfolio
# state is persisted to existing SimulatedPosition / SimulatedTrade /
# SimulatedMonthlyReturn / PortfolioNavSnapshot tables with
# track='paper_trade_combined' (separate from 'main'/'quant' tracks).
#
# This table is the strategy-level audit trail (per-strategy attribution,
# signal metadata, rebalance flags) that the existing schema does not capture.

class PaperTradeStrategyLog(Base):
    """
    Per-strategy daily snapshot for paper-trade orchestrator (Sprint B+).

    One row per (date, strategy_name). Composite PK ensures idempotency:
    re-running orchestrator for same date overwrites prior row.

    Strategies tracked:
      - K1_BAB    (etf_l1 sleeve)
      - D_PEAD    (ss_sp500 sleeve)
      - PATH_N    (ss_sp500 sleeve)
      - CTA_PQTIX (cta_defensive sleeve)

    Spec references:
      - K1: docs/spec_path_k1_size_expanded_b_plus_v1.md (id=61)
      - D-PEAD: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62)
      - Path N: docs/spec_path_n_index_reconstitution_drift_v1.md (id=70, amend 1)
      - CTA: docs/spec_path_o_cta_defensive_overlay_v1.md (id=73)
    """
    __tablename__ = "paper_trade_strategy_log"

    date              = Column(Date,         primary_key=True)
    strategy_name     = Column(String(20),   primary_key=True)  # K1_BAB / D_PEAD / PATH_N / CTA_PQTIX
    sleeve_id         = Column(String(20),   nullable=False)    # etf_l1 / ss_sp500 / cta_defensive
    status            = Column(String(16),   nullable=False)    # OK / NO_SIGNAL / ERROR / STUB
    is_rebalance_day  = Column(Boolean,      nullable=False, default=False)
    n_positions       = Column(Integer,      nullable=False, default=0)
    intra_sleeve_weight = Column(Float,      nullable=False)    # this strategy's weight within sleeve

    # Per-strategy daily return components (filled when return data available)
    daily_gross_return  = Column(Float,      nullable=True)     # before TC
    daily_net_return    = Column(Float,      nullable=True)     # after TC drag (if rebalance fires)
    tc_drag_today       = Column(Float,      nullable=True)     # 0 on hold days, >0 on rebalance days

    # Position / signal detail (JSON blobs)
    positions_json      = Column(Text,       nullable=True)     # {ticker: weight} for this strategy
    signal_metadata_json = Column(Text,      nullable=True)     # signal values, top-decile names, etc.

    notes               = Column(Text,       nullable=True)
    created_at          = Column(DateTime,   nullable=False, default=datetime.datetime.utcnow)


Index("ix_paper_trade_strategy_log_date",
      PaperTradeStrategyLog.date)
Index("ix_paper_trade_strategy_log_strategy",
      PaperTradeStrategyLog.strategy_name, PaperTradeStrategyLog.date)
Index("ix_paper_trade_strategy_log_sleeve",
      PaperTradeStrategyLog.sleeve_id, PaperTradeStrategyLog.date)


# ── Sprint H (2026-05-13): Per-strategy attribution logger ─────────────────────
#
# Trade-level forensic log for DD investigation. Records the DECISION layer's
# state at trade time: which strategy / spec / signal_value / event_trigger /
# expected_horizon → enables post-hoc root cause analysis without re-building
# context from raw data sources.
#
# Spec: docs/spec_per_strategy_attribution_logger_v1.md
#
# DOCTRINE: this table is WRITE-only from the decision layer; reads happen in
# FORENSIC layer (sql queries + on-demand LLM news summarizer). LLM never feeds
# back into decision. 0-LLM-in-DECISION preserved.

class PaperTradeTradeLog(Base):
    """
    Trade-level forensic log for paper-trade orchestrator (Sprint H v1.0).

    One row per (date, trade_id) — composite PK ensures idempotency:
    re-running orchestrator for same date with same (strategy, ticker) generates
    same deterministic trade_id and UPSERTs the row.

    Reads consumed by:
      - DD investigator (SQL queries + numpy contribution decomposition)
      - engine.forensic.news_context (on-demand LLM news summary)
      - Sprint E forward IC validation (corr(signal_value, realized_return))
      - Sprint C UI per-strategy drill-down

    Strategies tracked: K1_BAB / D_PEAD / PATH_N / CTA_PQTIX
    """
    __tablename__ = "paper_trade_trade_log"

    # Composite PK
    date                  = Column(Date,        primary_key=True)
    trade_id              = Column(String(36),  primary_key=True)  # deterministic UUID5

    # Strategy identity
    strategy_name         = Column(String(20),  nullable=False)    # K1_BAB / D_PEAD / PATH_N / CTA_PQTIX
    spec_id               = Column(Integer,     nullable=False)
    spec_hash_short       = Column(String(16),  nullable=False)
    sleeve_id             = Column(String(20),  nullable=False)

    # Trade detail
    ticker                = Column(String(16),  nullable=False)
    side                  = Column(String(8),   nullable=False)    # 'long' / 'short'
    weight                = Column(Float,       nullable=False)    # signed: + long / - short

    # Forensic context
    signal_value          = Column(Float,       nullable=True)     # None for non-signal strategies (CTA)
    event_trigger         = Column(String(64),  nullable=True)     # ISO date or descriptor
    expected_horizon_days = Column(Integer,     nullable=False)
    is_rebalance_day      = Column(Boolean,     nullable=False)    # fresh open vs hold day

    # Per-strategy free-form
    notes_json            = Column(Text,        nullable=False, default="{}")

    created_at            = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)


Index("ix_paper_trade_trade_log_strategy_date",
      PaperTradeTradeLog.strategy_name, PaperTradeTradeLog.date)
Index("ix_paper_trade_trade_log_ticker_date",
      PaperTradeTradeLog.ticker, PaperTradeTradeLog.date)
Index("ix_paper_trade_trade_log_spec_id",
      PaperTradeTradeLog.spec_id)


# ── Sprint D-1 (2026-05-13): Real-time S&P 500 reconstitution announcements ──
#
# Stores S&P 500 add/delete events detected from public free sources
# (Wikipedia + SEC EDGAR 8-K). Used by engine.portfolio.paper_trade_combined
# Path N signal in 'live' mode (replaces CRSP msp500list backtest-time approx).
#
# Sources reconciled by engine.data_sources.sp500_announcements.reconciler:
#   - 'wikipedia'                   — Wikipedia "Selected changes" table only
#   - 'wikipedia+edgar_8k_<acc>'    — refined announcement_date via EDGAR 8-K
#
# Path N alpha mechanism (Chen-Noronha-Singal 2004): pre-effective drift
# captured by long entry T-5 to T-1 of effective_date. announcement_date
# matters for forward execution timing.

class SP500AnnouncementEvent(Base):
    """
    One detected S&P 500 reconstitution add/delete event.

    Unique on (ticker, effective_date, action) — re-runs of reconciler
    update in place (announcement_date refinement / source upgrade).
    """
    __tablename__ = "sp500_announcement_events"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    ticker             = Column(String(20), nullable=False)
    effective_date     = Column(Date,        nullable=False)
    announcement_date  = Column(Date,        nullable=True)   # heuristic OR EDGAR-refined
    action             = Column(String(10),  nullable=False)  # ADD / REMOVE
    company_name       = Column(String(200), nullable=True)
    reason             = Column(Text,        nullable=True)
    source             = Column(String(80),  nullable=False)  # 'wikipedia' / 'wikipedia+edgar_8k_<acc>'
    detected_at        = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)
    updated_at         = Column(DateTime,    nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "effective_date", "action",
                         name="uq_sp500_announcement_ticker_date_action"),
    )


Index("ix_sp500_announcement_eff_date",
      SP500AnnouncementEvent.effective_date)
Index("ix_sp500_announcement_ticker",
      SP500AnnouncementEvent.ticker)
Index("ix_sp500_announcement_action_date",
      SP500AnnouncementEvent.action, SP500AnnouncementEvent.effective_date)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Manager Agent — Phase 4 of spec id=69 (current hash in SpecRegistry)
# ─────────────────────────────────────────────────────────────────────────────
#
# RiskManagerAlert is the persistent record of every breach detected by
# the 12 deterministic gates in engine.agents.risk_manager.gates. Written
# by engine.agents.risk_manager.persist after each pre-trade and post-trade
# check in run_paper_trade_day (Phase 6).
#
# Composite PK (date, alert_id) ensures idempotency: re-running orchestrator
# for the same date with the same gate output produces the same alert_id
# (deterministic uuid5 over (date, mode_id, affected_canonical)), so the
# row UPSERTs cleanly.
#
# Reads consumed by:
#   - Risk Console UI (4 tabs Exposures/Stress/Correlations/Limits)
#   - narrator.py (Phase 7) — generates plain-English prose per alert
#   - Audit Recorder (DD investigation workflow Pattern 6)
#   - DD investigator script (forensic replay)

class RiskManagerAlert(Base):
    """One breach detected by one Risk Manager gate in one orchestrator cycle.

    Mirrors AuditFinding pattern (engine.auto_audit_models) — one row per
    breach, never updated retroactively. Append-only audit trail.

    Severity scheme matches engine.circuit_breaker:
      NONE / LIGHT / MEDIUM / SEVERE
    so Phase 5 absorption of circuit_breaker is byte-compatible.
    """
    __tablename__ = "risk_manager_alerts"

    # ── Composite PK (deterministic alert_id derived from breach identity) ──
    date              = Column(Date,        primary_key=True)
    alert_id          = Column(String(36),  primary_key=True)   # uuid5(date, mode, affected)

    # ── Breach identity ──
    mode_id           = Column(String(8),   nullable=False)     # "1" / "6b" / "10" etc.
    severity          = Column(String(12),  nullable=False)     # HARD_HALT / SOFT_WARN
    cb_severity       = Column(String(12),  nullable=False)     # NONE / LIGHT / MEDIUM / SEVERE
    halt_decision     = Column(Boolean,     nullable=False, default=False)
    phase             = Column(String(12),  nullable=False)     # 'pre_trade' / 'post_trade'

    # ── Detector output ──
    rule_description  = Column(Text,        nullable=False)     # 1-line human-readable rule
    observed_value    = Column(Float,       nullable=True)      # what we measured
    threshold         = Column(Float,       nullable=True)      # what the rule says
    affected_json     = Column(Text,        nullable=False)     # JSON list[str] of ticker/sleeve/strategy names
    extra_json        = Column(Text,        nullable=False, default="{}")  # mode-specific context

    # ── Narrator output (populated Phase 7; NULL until narrator runs) ──
    narrative_text    = Column(Text,        nullable=True)      # one-paragraph BlackRock-Slack-tone prose
    narrative_cost_usd = Column(Float,      nullable=True)      # LLM cost for this narrative

    # ── Spec lineage ──
    spec_anchor       = Column(String(80),  nullable=False)     # "spec id=69 §2.1 Mode 5"

    # ── Audit ──
    generated_at_utc  = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)


Index("ix_risk_manager_alerts_date",
      RiskManagerAlert.date)
Index("ix_risk_manager_alerts_mode_id",
      RiskManagerAlert.mode_id)
Index("ix_risk_manager_alerts_severity_date",
      RiskManagerAlert.cb_severity, RiskManagerAlert.date)
Index("ix_risk_manager_alerts_halt",
      RiskManagerAlert.halt_decision, RiskManagerAlert.date)


# ─────────────────────────────────────────────────────────────────────────────
# DQ Inspector Agent — Phase 5 of spec id=70 (hash 31b5ad97)
# ─────────────────────────────────────────────────────────────────────────────
#
# DataQualityAlert mirrors RiskManagerAlert schema (composite PK +
# narrator-fill columns + audit lineage) plus a source_id column unique
# to DQ (the originating data source: 'fred:DGS10' / 'yfinance:bab_cache'
# / 'pead_panel' / 'universe:k1' / etc.) so dashboards can group by
# source.
#
# Phase 5 of spec id=70 §2.2 — engine/agents/dq_inspector/persist.py
# writes rows; query helpers consumed by Phase 9 Risk Console UI panel.

class DataQualityAlert(Base):
    """One DQ Inspector detector breach.

    Composite PK (date, alert_id) — alert_id is deterministic uuid5 over
    (date, mode_id, source_id, affected_canonical) so re-running orchestrator
    for the same date produces the same alert_id → idempotent UPSERT.
    """
    __tablename__ = "data_quality_alerts"

    # ── Composite PK ──
    date              = Column(Date,        primary_key=True)
    alert_id          = Column(String(36),  primary_key=True)

    # ── Breach identity ──
    mode_id           = Column(String(8),   nullable=False)     # "1" / "10a" / "10b" etc.
    severity          = Column(String(12),  nullable=False)     # HARD_HALT / SOFT_WARN
    cb_severity       = Column(String(12),  nullable=False)     # NONE / LIGHT / MEDIUM / SEVERE
    halt_decision     = Column(Boolean,     nullable=False, default=False)
    phase             = Column(String(12),  nullable=False)     # 'pre_batch' / 'post_feed' / 'post_batch'
    source_id         = Column(String(80),  nullable=False)     # 'fred:DGS10' / 'yfinance:bab_cache' / etc.

    # ── Detector output ──
    rule_description  = Column(Text,        nullable=False)
    observed_value    = Column(Float,       nullable=True)
    threshold         = Column(Float,       nullable=True)
    affected_json     = Column(Text,        nullable=False)
    extra_json        = Column(Text,        nullable=False, default="{}")

    # ── Narrator output (Phase 7) ──
    narrative_text    = Column(Text,        nullable=True)
    narrative_cost_usd = Column(Float,      nullable=True)

    # ── Spec lineage ──
    spec_anchor       = Column(String(80),  nullable=False)
    generated_at_utc  = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)


Index("ix_dq_alerts_date",          DataQualityAlert.date)
Index("ix_dq_alerts_mode_id",       DataQualityAlert.mode_id)
Index("ix_dq_alerts_source_id",     DataQualityAlert.source_id)
Index("ix_dq_alerts_severity_date", DataQualityAlert.cb_severity, DataQualityAlert.date)
Index("ix_dq_alerts_halt",          DataQualityAlert.halt_decision, DataQualityAlert.date)


# ─────────────────────────────────────────────────────────────────────────────
# Persona chat session memory (Phase A.3 — ε session memory persistence).
#
# Holds one row per (agent_id) — a local-single-user Streamlit deployment
# does not need a user_id dimension, and the chat UI sidebar shows one
# session per agent. Stored fields:
#   - history_json   : JSON-serialized Anthropic-format message list
#   - tool_log_json  : JSON list-of-lists (one per turn) of tool-call dicts
#   - cost_usd       : running session cost
#   - latency_ms     : running session latency
# A single agent_id row is REPLACED on every save (no audit churn here —
# DecisionLog / RiskManagerAlert handle audit history; this table is
# only the cross-process scratch pad so the Streamlit chat survives a
# browser refresh or restart).
#
# Pattern 5 ban (autonomous-debate ban) compatibility: this table is
# keyed only by agent_id, never by "conversation between A and B". One
# agent_id ↔ one chat session. Two agents cannot share a session row.
# ─────────────────────────────────────────────────────────────────────────────
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    # Phase A.7 Wave 4.1 (2026-05-19): multi-session support. The row is
    # now keyed by (agent_id, session_id) so the user can open multiple
    # independent conversations with the same agent (e.g. one for daily
    # risk Q&A, one for a focused investigation). Existing rows were
    # migrated to session_id='default'; new sessions get a generated
    # short id ('s_<hex>').
    id              = Column(Integer,    primary_key=True, autoincrement=True)
    agent_id        = Column(String(64), nullable=False)
    session_id      = Column(String(64), nullable=False, default="default")
    history_json    = Column(Text,       nullable=False, default="[]")
    tool_log_json   = Column(Text,       nullable=False, default="[]")
    cost_usd        = Column(Float,      nullable=False, default=0.0)
    latency_ms      = Column(Integer,    nullable=False, default=0)
    updated_at_utc  = Column(DateTime,   nullable=False,
                             default=datetime.datetime.utcnow,
                             onupdate=datetime.datetime.utcnow)
    # Phase A.4: visit tracking for "N new alerts since last visit" badge.
    last_visited_at = Column(DateTime,   nullable=True)
    # Phase A.7 Wave 3.2: human-readable session title.
    title           = Column(String(120), nullable=True)

    __table_args__ = (
        UniqueConstraint("agent_id", "session_id", name="uq_chat_session"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase A.7 Wave 4.3 (2026-05-19): Tier 2.5 cross-session semantic memory.
#
# Each user→assistant turn pair is embedded with sentence-transformers
# (all-MiniLM-L6-v2, 384-dim) and persisted here. recall_past_turns()
# (in engine.agents.persona.turn_memory) does cosine-similarity retrieval
# against this table so agents can answer "what did we discuss yesterday
# about X" without re-storing the answer in Tier 3 project memory
# (which is human-curated only).
#
# Pattern 5 ban compatibility: the table is indexed by agent_id so an
# agent CAN ONLY retrieve its own past turns. CoS can retrieve across
# agents (it has its own row plus can lookup by agent_id), which mirrors
# its supervisor role. There is no cross-agent autonomous flow.
#
# HARKing defense: retrieved turns are CLAIMS to be re-verified, not
# ground truth. Each persona's system prompt already enforces this via
# the "memory and evidence boundary" block. The retrieval tool returns
# both the historical assistant_text AND a created_at timestamp so the
# agent can age-discount its trust in the recall.
# ─────────────────────────────────────────────────────────────────────────────
class ChatTurnEmbedding(Base):
    """One semantic-search row per (agent_id, turn_idx) pair.

    embedding is stored as raw float32 bytes (np.float32 → 384*4 = 1536 B
    per row). Solo-scale projection: 50 turns/day × 365 days = ~28 MB
    after a year. SQLite handles this trivially; no separate vector store.
    """
    __tablename__ = "chat_turn_embeddings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    agent_id        = Column(String(64),  nullable=False, index=True)
    # Phase A.7 Wave 4.1: session_id scopes embeddings to a single
    # conversation thread, mirroring ChatSession. recall_past_turns
    # can ignore session_id (cross-session lookup) or filter to one.
    session_id      = Column(String(64),  nullable=False, default="default")
    turn_idx        = Column(Integer,     nullable=False)  # position in chat history
    user_text       = Column(Text,        nullable=False)
    assistant_text  = Column(Text,        nullable=False)
    embedding       = Column(LargeBinary, nullable=False)  # np.float32.tobytes()
    created_at      = Column(DateTime,    nullable=False, default=datetime.datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("agent_id", "session_id", "turn_idx",
                         name="uq_chat_turn_emb"),
    )


Index("ix_chat_turn_emb_agent_created",
      ChatTurnEmbedding.agent_id, ChatTurnEmbedding.created_at.desc())
