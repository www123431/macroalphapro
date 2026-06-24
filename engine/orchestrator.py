"""
Orchestration Layer — TradingCycleOrchestrator
===============================================
Formalises the four standard trading cycles and four human approval gates.

Cycles
------
  daily        Signals + regime + risk patrol (盘后，每交易日)
  weekly       Sector debate for signal-flip sectors (周末)
  monthly      Full rebalancing: signals → portfolio → execute (月末)
  verification Verify pending decisions with actual returns (异步触发)

Human gates (approval required before proceeding)
--------------------------------------------------
  analysis_draft      Draft analysis → Supervisor review before saving
  risk_approval       Risk recommendation → sign-off required
  monthly_rebalance   Rebalance trades → final human approval
  covariance_override LW covariance model change → approval required

Each cycle run is persisted to the `cycle_states` DB table for audit trail.

Usage:
    from engine.orchestrator import TradingCycleOrchestrator, run_daily_chain
    orch = TradingCycleOrchestrator()
    result = orch.run_daily(as_of=date.today())
    # Legacy compat:
    result = run_daily_chain(as_of=date.today(), dry_run=True)
"""
import datetime
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Human gate constants ───────────────────────────────────────────────────────
GATE_ANALYSIS_DRAFT      = "analysis_draft"
GATE_RISK_APPROVAL       = "risk_approval"
GATE_MONTHLY_REBALANCE   = "monthly_rebalance"
GATE_COVARIANCE_OVERRIDE = "covariance_override"


@dataclass
class ChainStep:
    name:       str
    status:     str        # "ok" | "failed" | "skipped"
    elapsed_s:  float = 0.0
    detail:     str   = ""


@dataclass
class ChainResult:
    as_of_date:    datetime.date
    started_at:    datetime.datetime
    finished_at:   datetime.datetime | None = None
    steps:         list[ChainStep]  = field(default_factory=list)
    regime:        str  = ""
    p_risk_on:     float = 0.0
    n_long:        int   = 0
    n_short:       int   = 0
    signal_flips:  list[str] = field(default_factory=list)
    portfolio_turnover: float = 0.0
    errors:        list[str] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        if self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    @property
    def ok(self) -> bool:
        return all(s.status != "failed" for s in self.steps)


def _timed(name: str, fn, *args, **kwargs) -> tuple[ChainStep, any]:
    t0 = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.monotonic() - t0
        return ChainStep(name=name, status="ok", elapsed_s=round(elapsed, 2)), result
    except Exception as e:
        elapsed = time.monotonic() - t0
        logger.error("Chain step '%s' failed: %s", name, e, exc_info=True)
        return ChainStep(name=name, status="failed", elapsed_s=round(elapsed, 2),
                         detail=str(e)), None


def run_daily_chain(
    as_of:           datetime.date | None = None,
    lookback_months: int  = 12,
    skip_months:     int  = 1,
    dry_run:         bool = True,
    run_sectors:     bool = False,  # sector LLM analysis is slow; opt-in
    run_verify:      bool = True,   # verify pending decisions + trigger LEARN write-back
    model                = None,    # required if run_sectors=True or for LCS/LEARN
) -> ChainResult:
    """
    Run the full daily computation chain.

    Steps:
      1. Compute TSMOM/CSMOM signal DataFrame for all sectors
      2. Detect macro regime (MSM or rule-based fallback)
      3. Detect signal flips vs previous month
      4. Construct portfolio target weights
      5. (opt-in) Run sector LLM debate for all sectors
      6. (opt-in) Execute or simulate rebalance
      7. (opt-in) Expire stale approvals + verify pending decisions
      8. (auto, when model provided) LEARN — LCS quality gate → SkillLibrary write-back
         Note: Step 8 is embedded inside verify_pending_decisions() when model is passed.

    Args:
        as_of:           Date for computation (default: today)
        lookback_months: TSMOM formation period
        skip_months:     Skip most-recent months
        dry_run:         If False, persists portfolio positions
        run_sectors:     If True, run LLM sector analysis for signal-flip sectors
        run_verify:      If True, run VERIFY+LEARN tail (default True)
        model:           LLM model instance (required if run_sectors=True or for LCS/LEARN)

    Returns:
        ChainResult with per-step timing, errors, and summary metrics.
    """
    as_of = as_of or datetime.date.today()
    result = ChainResult(as_of_date=as_of, started_at=datetime.datetime.utcnow())

    # ── P2-16: Circuit Breaker pre-flight check ────────────────────────────────
    from engine.circuit_breaker import evaluate as _cb_evaluate, LEVEL_SEVERE, LEVEL_MEDIUM
    _cb = _cb_evaluate(as_of)
    if _cb.level == LEVEL_SEVERE:
        result.errors.append(f"🔴 CIRCUIT BREAKER SEVERE — 链中止: {_cb.reason}")
        result.steps.append(ChainStep(name="circuit_breaker", status="failed",
                                      detail=_cb.reason))
        result.finished_at = datetime.datetime.utcnow()
        return result
    if _cb.level == LEVEL_MEDIUM:
        result.errors.append(f"🟡 CIRCUIT BREAKER MEDIUM — 非核心 LLM 已暂停: {_cb.reason}")
        run_sectors = False   # Override: protect remaining quota for risk patrol

    # ── Step 1: Signals ────────────────────────────────────────────────────────
    from engine.signal import get_signal_dataframe
    step, signal_df = _timed("signals", get_signal_dataframe,
                              as_of, lookback_months, skip_months)
    result.steps.append(step)
    if signal_df is None or (hasattr(signal_df, "empty") and signal_df.empty):
        result.errors.append("Signal computation failed — aborting chain")
        result.finished_at = datetime.datetime.utcnow()
        return result

    result.n_long  = int((signal_df["tsmom"] == 1).sum())
    result.n_short = int((signal_df["tsmom"] == -1).sum())

    # ── Step 2: Regime ─────────────────────────────────────────────────────────
    from engine.regime import get_regime_on
    step, regime_result = _timed("regime", get_regime_on,
                                  as_of=as_of, train_end=as_of)
    result.steps.append(step)
    if regime_result is not None:
        result.regime    = regime_result.regime
        result.p_risk_on = regime_result.p_risk_on
    else:
        result.errors.append("Regime detection failed — using neutral fallback")
        result.regime = "transition"

    # ── Step 3: Signal flips vs previous month ─────────────────────────────────
    prev_date = (datetime.date(as_of.year, as_of.month, 1) - datetime.timedelta(days=1))
    step, prev_signal_df = _timed("prev_signals", get_signal_dataframe,
                                   prev_date, lookback_months, skip_months)
    result.steps.append(step)
    if prev_signal_df is not None and not prev_signal_df.empty:
        for sector in signal_df.index:
            if sector in prev_signal_df.index:
                cur  = int(signal_df.loc[sector, "tsmom"])
                prev = int(prev_signal_df.loc[sector, "tsmom"])
                if cur != prev:
                    result.signal_flips.append(
                        f"{sector}: {prev:+d} → {cur:+d}"
                    )

    # ── Step 4: Portfolio construction (with tactical overlay) ───────────────
    from engine.portfolio import construct_portfolio, compute_tactical_overlay
    _overlay = compute_tactical_overlay(regime_result)
    step, portfolio = _timed("portfolio", construct_portfolio,
                              signal_df, regime=regime_result, overlay=_overlay)
    result.steps.append(step)
    result.errors.append(f"[Tactical] {_overlay.note}") if _overlay.entry_throttle else None

    # ── Step 5 (opt-in): Sector LLM analysis for signal-flip sectors ──────────
    if run_sectors and model and result.signal_flips:
        flip_sectors = [s.split(":")[0].strip() for s in result.signal_flips]
        for sector in flip_sectors:
            from engine.debate import run_sector_debate
            step, debate = _timed(
                f"sector_analysis:{sector}",
                run_sector_debate,
                model=model,
                sector_name=sector,
                vix=regime_result.vix or 20.0 if regime_result else 20.0,
            )
            result.steps.append(step)

    # ── Step 6 (opt-in): Rebalance ────────────────────────────────────────────
    if not dry_run:
        from engine.portfolio_tracker import execute_rebalance
        step, rebal = _timed("rebalance", execute_rebalance,
                              rebalance_date=as_of,
                              dry_run=False,
                              lookback_months=lookback_months,
                              skip_months=skip_months)
        result.steps.append(step)
        if rebal:
            result.portfolio_turnover = rebal.get("turnover", 0.0)
    else:
        result.steps.append(ChainStep(name="rebalance", status="skipped",
                                      detail="dry_run=True"))

    # ── Step 7 (opt-in): VERIFY + LEARN ──────────────────────────────────────
    # Expire stale approval records, then run Triple-Barrier verification on all
    # pending decisions. When model is provided, the LEARN phase also runs inside
    # verify_pending_decisions(): LCS audit → SkillLibrary write-back → MetaAgent.
    if run_verify:
        from engine.memory import expire_stale_approvals, verify_pending_decisions

        # 7a: expire stale approvals
        try:
            n_expired = expire_stale_approvals()
            step_expire = ChainStep(
                name="expire_approvals", status="ok",
                detail=f"{n_expired} 条审批已自动过期" if n_expired else "无过期审批",
            )
        except Exception as _e:
            step_expire = ChainStep(name="expire_approvals", status="failed",
                                    detail=str(_e))
        result.steps.append(step_expire)

        # 7b: verify pending decisions (LEARN embedded when model is passed)
        step_v, verify_results = _timed("verify_learn", verify_pending_decisions,
                                        model=model)
        n_verified = len(verify_results) if verify_results else 0
        step_v.detail = f"{n_verified} 条决策已验证"
        if n_verified:
            result.errors.append(f"[VERIFY] {n_verified} 条决策完成验证")
        result.steps.append(step_v)
    else:
        result.steps.append(ChainStep(name="verify_learn", status="skipped",
                                      detail="run_verify=False"))

    result.finished_at = datetime.datetime.utcnow()
    logger.info(
        "Daily chain complete: as_of=%s regime=%s n_long=%d n_short=%d "
        "flips=%d elapsed=%.1fs",
        as_of, result.regime, result.n_long, result.n_short,
        len(result.signal_flips), result.elapsed_s,
    )
    return result


# ── P2-18 TradingCycleOrchestrator ────────────────────────────────────────────

class TradingCycleOrchestrator:
    """
    Formal orchestrator for the four standard trading cycles.
    Persists each run to `cycle_states` for audit and monitoring.
    """

    # ── Internal DB helpers ───────────────────────────────────────────────────

    def _start_cycle(self, cycle_type: str, as_of: datetime.date) -> int:
        """Insert a running CycleState row, return its id."""
        from engine.memory import SessionFactory, CycleState
        with SessionFactory() as session:
            cs = CycleState(
                cycle_type=cycle_type,
                as_of_date=as_of,
                status="running",
                started_at=datetime.datetime.utcnow(),
            )
            session.add(cs)
            session.commit()
            return cs.id

    def _finish_cycle(
        self,
        cycle_id: int,
        status: str,
        result=None,
        error: str = "",
        gate: str = "",
    ) -> None:
        from engine.memory import SessionFactory, CycleState
        _now = datetime.datetime.utcnow()
        with SessionFactory() as session:
            cs = session.get(CycleState, cycle_id)
            if cs is None:
                return
            cs.status      = status
            cs.finished_at = _now
            cs.elapsed_s   = round((_now - cs.started_at).total_seconds(), 2)
            cs.gate        = gate or None
            if error:
                cs.error_log = error
            if result is not None:
                try:
                    cs.result_summary = json.dumps({
                        "regime":       getattr(result, "regime", ""),
                        "n_long":       getattr(result, "n_long", 0),
                        "n_short":      getattr(result, "n_short", 0),
                        "signal_flips": getattr(result, "signal_flips", []),
                        "errors":       getattr(result, "errors", []),
                    }, ensure_ascii=False)
                except Exception:
                    pass
            session.commit()

    # ── Cycle 1: Daily ────────────────────────────────────────────────────────

    def run_daily(
        self,
        as_of: datetime.date | None = None,
        lookback_months: int = 12,
        skip_months: int = 1,
        dry_run: bool = True,
        run_sectors: bool = False,
        model=None,
    ) -> ChainResult:
        """
        Daily cycle (post-market): signals → regime → risk patrol.
        Corresponds to tactical layer in TiMi time-scale hierarchy.
        """
        as_of = as_of or datetime.date.today()
        cycle_id = self._start_cycle("daily", as_of)
        try:
            result = run_daily_chain(
                as_of=as_of,
                lookback_months=lookback_months,
                skip_months=skip_months,
                dry_run=dry_run,
                run_sectors=run_sectors,
                model=model,
            )
            status = "completed" if result.ok else "failed"
            self._finish_cycle(cycle_id, status, result=result,
                               error="; ".join(result.errors))
            return result
        except Exception as exc:
            self._finish_cycle(cycle_id, "failed", error=str(exc))
            raise

    # ── Cycle 2: Weekly ───────────────────────────────────────────────────────

    def run_weekly(
        self,
        as_of: datetime.date | None = None,
        model=None,
        dry_run: bool = True,
    ) -> ChainResult:
        """
        Weekly cycle (weekend): sector debate for all signal-flip sectors.
        Corresponds to tactical-to-strategic bridge in TiMi hierarchy.
        Gate: analysis_draft — debate outputs await Supervisor review.
        """
        as_of = as_of or datetime.date.today()
        cycle_id = self._start_cycle("weekly", as_of)
        try:
            result = run_daily_chain(
                as_of=as_of,
                dry_run=dry_run,
                run_sectors=True,   # full sector debate
                model=model,
            )
            # Weekly cycle halts at analysis_draft gate for human review
            gate = GATE_ANALYSIS_DRAFT if not dry_run else ""
            status = "pending_gate" if gate else ("completed" if result.ok else "failed")
            self._finish_cycle(cycle_id, status, result=result,
                               gate=gate, error="; ".join(result.errors))
            return result
        except Exception as exc:
            self._finish_cycle(cycle_id, "failed", error=str(exc))
            raise

    # ── Cycle 3: Monthly ──────────────────────────────────────────────────────

    def run_monthly(
        self,
        as_of: datetime.date | None = None,
        model=None,
        require_approval: bool = True,
    ) -> ChainResult:
        """
        Monthly cycle (month-end): full rebalancing with human approval gate.
        Gate: monthly_rebalance — trades are not executed until approved.
        Corresponds to strategic layer in TiMi hierarchy.
        """
        as_of = as_of or datetime.date.today()
        cycle_id = self._start_cycle("monthly", as_of)
        try:
            # Compute signals + portfolio (dry_run until gate is cleared)
            result = run_daily_chain(
                as_of=as_of,
                dry_run=True,   # always dry_run first; execution requires gate approval
                run_sectors=True,
                model=model,
            )
            gate = GATE_MONTHLY_REBALANCE if require_approval else ""
            status = "pending_gate" if gate else ("completed" if result.ok else "failed")
            self._finish_cycle(cycle_id, status, result=result,
                               gate=gate, error="; ".join(result.errors))
            return result
        except Exception as exc:
            self._finish_cycle(cycle_id, "failed", error=str(exc))
            raise

    def approve_gate(self, cycle_id: int, approved: bool, note: str = "") -> dict:
        """
        Human approval action for a pending_gate cycle.
        If approved=True and gate=monthly_rebalance → execute rebalance.
        Returns exec_result dict (empty if not applicable or rejected).
        """
        import json
        from engine.memory import SessionFactory, CycleState
        gate = None
        as_of_date = None
        with SessionFactory() as session:
            cs = session.get(CycleState, cycle_id)
            if cs is None or cs.status != "pending_gate":
                return {}
            gate = cs.gate
            as_of_date = cs.as_of_date
            cs.status = "approved" if approved else "rejected"
            cs.error_log = (cs.error_log or "") + f"\nGate decision: {'approved' if approved else 'rejected'}. {note}"
            session.commit()

        exec_result: dict = {}
        if approved and gate == GATE_MONTHLY_REBALANCE:
            try:
                from engine.portfolio_tracker import execute_rebalance
                exec_result = execute_rebalance(rebalance_date=as_of_date, dry_run=False) or {}
                logger.info("Monthly rebalance executed after gate approval: cycle_id=%d", cycle_id)
                summary = {
                    "turnover":       exec_result.get("turnover", 0),
                    "total_cost_bps": exec_result.get("total_cost_bps", 0),
                    "n_long":         exec_result.get("n_long", 0),
                    "n_short":        exec_result.get("n_short", 0),
                    "n_trades":       len(exec_result.get("trades", [])),
                }
                with SessionFactory() as s2:
                    cs2 = s2.get(CycleState, cycle_id)
                    if cs2:
                        cs2.result_summary = json.dumps(summary, ensure_ascii=False)
                        s2.commit()
            except Exception as exc:
                logger.error("Monthly rebalance execution failed: %s", exc)
                exec_result = {"error": str(exc)}

        return exec_result

    # ── Cycle 4: Verification ─────────────────────────────────────────────────

    def run_verification(self, model=None) -> list[dict]:
        """
        Verification cycle (async): verify pending decisions with actual returns.
        When model is provided, the LEARN phase runs inside verify_pending_decisions():
        LCS audit → SkillLibrary write-back → MetaAgent bias analysis.
        Corresponds to learning layer — runs independently of market hours.
        """
        as_of = datetime.date.today()
        cycle_id = self._start_cycle("verification", as_of)
        try:
            from engine.memory import expire_stale_approvals, verify_pending_decisions
            expire_stale_approvals()
            results = verify_pending_decisions(model=model)
            n_verified = len(results) if results else 0
            self._finish_cycle(
                cycle_id, "completed",
                error="" if n_verified >= 0 else "verification returned no results",
            )
            return results
        except Exception as exc:
            self._finish_cycle(cycle_id, "failed", error=str(exc))
            raise

    # ── Query helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def get_recent_cycles(n: int = 20) -> list[dict]:
        """Return last n cycle runs for Admin UI display."""
        from engine.memory import SessionFactory, CycleState
        with SessionFactory() as session:
            rows = (
                session.query(CycleState)
                .order_by(CycleState.id.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "id":             r.id,
                    "cycle_type":     r.cycle_type,
                    "as_of_date":     str(r.as_of_date),
                    "status":         r.status,
                    "gate":           r.gate,
                    "started_at":     str(r.started_at)[:16],
                    "elapsed_s":      r.elapsed_s,
                    "error_log":      r.error_log,
                }
                for r in rows
            ]

    @staticmethod
    def get_pending_gates() -> list[dict]:
        """Return cycles awaiting human gate approval."""
        from engine.memory import SessionFactory, CycleState
        with SessionFactory() as session:
            rows = (
                session.query(CycleState)
                .filter(CycleState.status == "pending_gate")
                .order_by(CycleState.id.desc())
                .all()
            )
            return [
                {
                    "id":         r.id,
                    "cycle_type": r.cycle_type,
                    "as_of_date": str(r.as_of_date),
                    "gate":       r.gate,
                    "started_at": str(r.started_at)[:16],
                }
                for r in rows
            ]
