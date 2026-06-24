"""
engine/llm_budget.py — Runtime-tunable LLM budget governance (2026-05-08).

SystemConfig-backed read/write API for LLM cost budgets so supervisor
can adjust caps without code edits + restart. Defaults (project anchor
values) come from engine.config; SystemConfig overrides them at runtime
when set.

Four independent budget scopes (separate cost trackers, per spec):
  - **R-audit**       : engine/auto_audit_proposer.py LLM budget (default $50/yr)
  - **S6-anomaly**    : engine/anomaly_screener.py LLM detector budget ($250/yr)
  - **RAG-synthesis** : engine/agents/history_rag/synthesize.py daily budget ($0.05/day)
  - **DeepSeek**      : engine/deepseek_client.py 2nd LLM provider (default $50/yr)

Defense in depth — these are caps, not guarantees:
  - Pricing constants (COST_PER_1M_*_TOKENS) remain hardcoded in
    engine/config.py — they reflect Gemini API pricing, not user knobs.
  - Budget changes are written to SystemConfig with a `_history` key
    appended (audit trail; complements amendment_log on code files).

Boundary invariant: zero LLM imports — pure deterministic config IO.
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── SystemConfig key namespace ───────────────────────────────────────────────
_KEY_R_AUDIT_USD_PER_YEAR    = "llm_budget.audit_proposer.usd_per_year"
_KEY_S6_ANOMALY_USD_PER_YEAR = "llm_budget.s6_anomaly.usd_per_year"
_KEY_RAG_SYNTH_USD_PER_DAY   = "llm_budget.rag_synthesis.usd_per_day"
_KEY_DEEPSEEK_USD_PER_YEAR   = "llm_budget.deepseek.usd_per_year"
_KEY_HISTORY                  = "llm_budget._history"

# DeepSeek default annual cap (USD). DeepSeek V4-flash is ~9x cheaper than
# Gemini 2.5 Flash on output, so $50/yr is generous for demo-stage usage
# (Tool 2 judge agent + Tool 4 outcome reasoner future scopes).
_DEFAULT_DEEPSEEK_USD_PER_YEAR = 50.0

# R-tier / S6 annual default budgets. Previously read from engine.config
# (R_COST_BUDGET_USD / S6_COST_BUDGET_USD), but those constants were dropped in a config
# refactor (never present in tracked history) — the getters' lazy import raised on every
# call. Values restored from the getters' own documented defaults ($50/yr R-audit,
# $250/yr S6-anomaly); overridable at runtime via SystemConfig (get_*_budget checks that
# first). NOT a guessed value — this is the documented intent. (2026-05-22 config-drift fix.)
_DEFAULT_R_AUDIT_USD_PER_YEAR = 50.0
_DEFAULT_S6_ANOMALY_USD_PER_YEAR = 250.0

# Validation bounds: cents to a few thousand dollars per year.
# Below 0.01 = degenerate; above 10000 = signaling typo not deliberate increase.
_MIN_USD_PER_YEAR = 0.01
_MAX_USD_PER_YEAR = 10_000.0

# Daily budgets: tighter range. Below 0.001 = under one cheap call;
# above 100/day = typo or misunderstanding (≈$36k/yr).
_MIN_USD_PER_DAY = 0.001
_MAX_USD_PER_DAY = 100.0


def _default_r_audit() -> float:
    return _DEFAULT_R_AUDIT_USD_PER_YEAR


def _default_s6_anomaly() -> float:
    return _DEFAULT_S6_ANOMALY_USD_PER_YEAR


def _default_rag_synth_daily() -> float:
    """Default RAG synthesis daily budget — sourced from history_rag config.
    Wrapped in a function (not module-level import) to avoid circular import
    risk with engine.agents.history_rag if other code imports this module
    early in startup."""
    try:
        from engine.agents.history_rag.config import SYNTHESIS_DAILY_BUDGET
        return float(SYNTHESIS_DAILY_BUDGET)
    except Exception:
        return 0.05  # fallback constant if config import fails


def get_r_audit_budget_usd_per_year() -> float:
    """R-tier (auto_audit_proposer) annual LLM budget. Default $50/yr."""
    from engine.memory import get_system_config
    val = get_system_config(_KEY_R_AUDIT_USD_PER_YEAR, None)
    if val is None or val == "":
        return _default_r_audit()
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning(
            "llm_budget: invalid SystemConfig value for %s: %r — using default",
            _KEY_R_AUDIT_USD_PER_YEAR, val,
        )
        return _default_r_audit()


def get_s6_anomaly_budget_usd_per_year() -> float:
    """S6 (anomaly_screener LLM detector) annual LLM budget. Default $250/yr."""
    from engine.memory import get_system_config
    val = get_system_config(_KEY_S6_ANOMALY_USD_PER_YEAR, None)
    if val is None or val == "":
        return _default_s6_anomaly()
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning(
            "llm_budget: invalid SystemConfig value for %s: %r — using default",
            _KEY_S6_ANOMALY_USD_PER_YEAR, val,
        )
        return _default_s6_anomaly()


def set_r_audit_budget_usd_per_year(amount: float, *, actor: str = "user") -> None:
    """Write new R-tier budget cap + history audit entry."""
    _set_with_history(
        scope     = "r_audit",
        key       = _KEY_R_AUDIT_USD_PER_YEAR,
        amount    = amount,
        actor     = actor,
        prior     = get_r_audit_budget_usd_per_year(),
    )


def set_s6_anomaly_budget_usd_per_year(amount: float, *, actor: str = "user") -> None:
    """Write new S6 anomaly budget cap + history audit entry."""
    _set_with_history(
        scope     = "s6_anomaly",
        key       = _KEY_S6_ANOMALY_USD_PER_YEAR,
        amount    = amount,
        actor     = actor,
        prior     = get_s6_anomaly_budget_usd_per_year(),
    )


def get_rag_synthesis_daily_budget_usd() -> float:
    """RAG synthesis (engine.agents.history_rag.synthesize) daily LLM budget.

    Default $0.05/day from engine.agents.history_rag.config.SYNTHESIS_DAILY_BUDGET.
    SystemConfig key llm_budget.rag_synthesis.usd_per_day overrides at runtime.
    """
    from engine.memory import get_system_config
    val = get_system_config(_KEY_RAG_SYNTH_USD_PER_DAY, None)
    if val is None or val == "":
        return _default_rag_synth_daily()
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning(
            "llm_budget: invalid SystemConfig value for %s: %r — using default",
            _KEY_RAG_SYNTH_USD_PER_DAY, val,
        )
        return _default_rag_synth_daily()


def set_rag_synthesis_daily_budget_usd(amount: float, *, actor: str = "user") -> None:
    """Write new RAG synthesis daily budget cap + history audit entry."""
    _set_with_history_daily(
        scope     = "rag_synthesis_daily",
        key       = _KEY_RAG_SYNTH_USD_PER_DAY,
        amount    = amount,
        actor     = actor,
        prior     = get_rag_synthesis_daily_budget_usd(),
    )


def get_deepseek_budget_usd_per_year() -> float:
    """DeepSeek V4-flash annual LLM budget. Default $50/yr (9x cheaper
    than Gemini → $50/yr ≈ $450/yr Gemini-equivalent output coverage)."""
    from engine.memory import get_system_config
    val = get_system_config(_KEY_DEEPSEEK_USD_PER_YEAR, None)
    if val is None or val == "":
        return _DEFAULT_DEEPSEEK_USD_PER_YEAR
    try:
        return float(val)
    except (TypeError, ValueError):
        logger.warning(
            "llm_budget: invalid SystemConfig value for %s: %r — using default",
            _KEY_DEEPSEEK_USD_PER_YEAR, val,
        )
        return _DEFAULT_DEEPSEEK_USD_PER_YEAR


def set_deepseek_budget_usd_per_year(amount: float, *, actor: str = "user") -> None:
    """Write new DeepSeek budget cap + history audit entry."""
    _set_with_history(
        scope     = "deepseek",
        key       = _KEY_DEEPSEEK_USD_PER_YEAR,
        amount    = amount,
        actor     = actor,
        prior     = get_deepseek_budget_usd_per_year(),
    )


def get_budget_status() -> dict[str, Any]:
    """Snapshot of all budget scopes for UI display.

    Returns:
        {
            'r_audit': {
                'current_usd_per_year': float,
                'default_usd_per_year': float,
                'spent_usd':            float,
                'remaining_usd':        float,
                'fraction_used':        float,
            },
            's6_anomaly': { ... same shape ... },
            'history': [ recent entries ],
        }
    """
    # R-audit spent state
    try:
        from engine.auto_audit_proposer import get_cost_status as _r_status
        r_state = _r_status()
        r_spent = float(r_state.get("total_usd", 0.0))
    except Exception:
        r_spent = 0.0

    # S6 spent state
    try:
        from engine.anomaly_llm_detector import get_cost_status as _s6_status
        s6_state = _s6_status()
        s6_spent = float(s6_state.get("total_usd", 0.0))
    except Exception:
        s6_spent = 0.0

    # RAG synthesis daily spend (today only — per-day tracker)
    try:
        from engine.agents.history_rag.synthesize import get_synthesis_cost_status
        rag_state = get_synthesis_cost_status()
        rag_today_spent = float(rag_state.get("today_usd", 0.0))
    except Exception:
        rag_today_spent = 0.0

    # DeepSeek YTD spend (per-year tracker; resets on year rollover in client ledger)
    try:
        from engine.deepseek_client import get_deepseek_cumulative_cost
        ds_state = get_deepseek_cumulative_cost()
        ds_spent = float(ds_state.get("ytd_spend_usd", 0.0))
    except Exception:
        ds_spent = 0.0

    r_cur   = get_r_audit_budget_usd_per_year()
    s6_cur  = get_s6_anomaly_budget_usd_per_year()
    rag_cur = get_rag_synthesis_daily_budget_usd()
    ds_cur  = get_deepseek_budget_usd_per_year()

    return {
        "r_audit": {
            "current_usd_per_year": r_cur,
            "default_usd_per_year": _default_r_audit(),
            "spent_usd":            r_spent,
            "remaining_usd":        max(r_cur - r_spent, 0.0),
            "fraction_used":        (r_spent / r_cur) if r_cur > 0 else 0.0,
        },
        "s6_anomaly": {
            "current_usd_per_year": s6_cur,
            "default_usd_per_year": _default_s6_anomaly(),
            "spent_usd":            s6_spent,
            "remaining_usd":        max(s6_cur - s6_spent, 0.0),
            "fraction_used":        (s6_spent / s6_cur) if s6_cur > 0 else 0.0,
        },
        "rag_synthesis_daily": {
            "current_usd_per_day":  rag_cur,
            "default_usd_per_day":  _default_rag_synth_daily(),
            "spent_usd_today":      rag_today_spent,
            "remaining_usd_today":  max(rag_cur - rag_today_spent, 0.0),
            "fraction_used_today":  (rag_today_spent / rag_cur) if rag_cur > 0 else 0.0,
        },
        "deepseek": {
            "current_usd_per_year": ds_cur,
            "default_usd_per_year": _DEFAULT_DEEPSEEK_USD_PER_YEAR,
            "spent_usd":            ds_spent,
            "remaining_usd":        max(ds_cur - ds_spent, 0.0),
            "fraction_used":        (ds_spent / ds_cur) if ds_cur > 0 else 0.0,
        },
        "history": _read_history(),
    }


# ── Internals ────────────────────────────────────────────────────────────────

def _set_with_history(
    *,
    scope:  str,
    key:    str,
    amount: float,
    actor:  str,
    prior:  float,
) -> None:
    """Validate + persist + append history entry (annual scope)."""
    from engine.memory import set_system_config

    if not isinstance(amount, (int, float)):
        raise ValueError(f"amount must be numeric; got {type(amount).__name__}")
    amt = float(amount)
    if not (_MIN_USD_PER_YEAR <= amt <= _MAX_USD_PER_YEAR):
        raise ValueError(
            f"amount {amt} outside allowed range "
            f"[{_MIN_USD_PER_YEAR}, {_MAX_USD_PER_YEAR}] USD/yr. "
            f"Below floor = degenerate; above ceiling = likely typo."
        )

    set_system_config(key, str(amt))

    # Append to history (kept compact; recent N entries)
    history = _read_history()
    history.append({
        "at":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scope":  scope,
        "actor":  actor,
        "prior":  prior,
        "new":    amt,
    })
    _MAX_HISTORY = 50
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]
    set_system_config(_KEY_HISTORY, json.dumps(history, ensure_ascii=False))
    logger.info(
        "llm_budget: %s changed %s → %s by actor=%s",
        scope, prior, amt, actor,
    )


def _set_with_history_daily(
    *,
    scope:  str,
    key:    str,
    amount: float,
    actor:  str,
    prior:  float,
) -> None:
    """Validate + persist + append history entry (daily scope)."""
    from engine.memory import set_system_config

    if not isinstance(amount, (int, float)):
        raise ValueError(f"amount must be numeric; got {type(amount).__name__}")
    amt = float(amount)
    if not (_MIN_USD_PER_DAY <= amt <= _MAX_USD_PER_DAY):
        raise ValueError(
            f"amount {amt} outside allowed range "
            f"[{_MIN_USD_PER_DAY}, {_MAX_USD_PER_DAY}] USD/day. "
            f"Below floor = under one cheap call; above ceiling = "
            f"likely typo (~$36k/yr equivalent)."
        )

    set_system_config(key, str(amt))

    # Append to history (kept compact; recent N entries)
    history = _read_history()
    history.append({
        "at":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scope":  scope,
        "actor":  actor,
        "prior":  prior,
        "new":    amt,
    })
    _MAX_HISTORY = 50
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]
    set_system_config(_KEY_HISTORY, json.dumps(history, ensure_ascii=False))
    logger.info(
        "llm_budget: %s changed %s → %s by actor=%s",
        scope, prior, amt, actor,
    )
    return  # don't fall through to annual-scope history append below

    # Append to history (kept compact; recent N entries)
    history = _read_history()
    history.append({
        "at":     datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "scope":  scope,
        "actor":  actor,
        "prior":  prior,
        "new":    amt,
    })
    _MAX_HISTORY = 50
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]
    set_system_config(_KEY_HISTORY, json.dumps(history, ensure_ascii=False))
    logger.info(
        "llm_budget: %s changed %s → %s by actor=%s",
        scope, prior, amt, actor,
    )


def _read_history() -> list[dict[str, Any]]:
    from engine.memory import get_system_config
    raw = get_system_config(_KEY_HISTORY, "")
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []
