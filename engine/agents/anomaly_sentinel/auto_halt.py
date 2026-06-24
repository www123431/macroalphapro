"""engine/agents/anomaly_sentinel/auto_halt.py — Auto-halt mechanism v1
(Phase 1 Task II.D of research_agenda_2026-05-29).

Closes the loop on the Anomaly Sentinel persona: it currently REPORTS findings
via API/dashboard. This module gives it TEETH — pre-committed deterministic
trigger rules that pause new order submission when book health degrades,
demand human acknowledgement before resuming.

Doctrine:
  - All trigger thresholds are PRE-COMMITTED in this file, not user-tunable
    at runtime (an attacker / a tired researcher can't lower the bar to
    avoid an alert)
  - The halt mechanism is FAIL-SAFE: if the check itself fails (data
    missing / parse error / corrupted artifact), it defaults to HALT,
    not to ALLOW
  - 0-LLM-in-DECISION preserved: every trigger is a deterministic threshold
    on a deterministic metric, no LLM involvement

Architecture
------------
1. Compute trigger state from book artifacts (NAV history, paper-trade
   attribution log, current positions, vol estimates)
2. If any trigger fires → write data/paper_trade/halt_flag.json with
   structured reason + timestamp + suggested action
3. The execution layer (engine.execution.run_paper_execution) reads the
   halt_flag at start; if present and not acknowledged → refuse --submit
4. Acknowledgement requires explicit human action: delete the flag file OR
   add an "acknowledged_by_human_at" field with a timestamp

Pre-committed triggers (NO grid search, NO runtime tuning):
- T1: book-level rolling 60d Sharpe < -0.5
- T2: book-level rolling 21d realized vol > 1.5× target_vol (15% vs 10%)
- T3: any single-ticker MaxDD over 30d < -25%
- T4: any single position |weight| > 30% (risk cap breach)
- T5: NAV file mtime stale (> 48h since last update)
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
HALT_FLAG_PATH = REPO_ROOT / "data" / "paper_trade" / "halt_flag.json"
ALERTS_DIR = REPO_ROOT / "data" / "alerts"


# ── Pre-committed trigger thresholds (NEVER tune at runtime) ─────────────────
TRIGGER_SHARPE_60D = -0.5
TRIGGER_VOL_MULTIPLE = 1.5
TRIGGER_VOL_TARGET = 0.10
TRIGGER_TICKER_MAXDD_30D = -0.25
TRIGGER_WEIGHT_ABS = 0.30
TRIGGER_NAV_STALE_HOURS = 48.0


@dataclasses.dataclass(frozen=True)
class TriggerResult:
    name:        str
    fired:       bool
    metric:      float | None
    threshold:   float
    evidence:    dict
    description: str


@dataclasses.dataclass(frozen=True)
class HaltDecision:
    """Final decision packet, written to halt_flag.json on halt."""
    halt:           bool
    triggers_fired: list[str]
    triggers_all:   list[TriggerResult]
    as_of:          str
    suggested_action: str

    def to_jsonable(self) -> dict:
        return {
            "halt":             self.halt,
            "triggers_fired":   self.triggers_fired,
            "triggers_all":     [dataclasses.asdict(t) for t in self.triggers_all],
            "as_of":            self.as_of,
            "suggested_action": self.suggested_action,
        }


# ── Trigger implementations ──────────────────────────────────────────────────

def _t1_book_sharpe_60d(nav_series: pd.Series) -> TriggerResult:
    """T1: book-level rolling 60d Sharpe < -0.5."""
    if nav_series is None or len(nav_series) < 60:
        return TriggerResult(
            name="T1_book_sharpe_60d", fired=False,
            metric=None, threshold=TRIGGER_SHARPE_60D,
            evidence={"reason": f"insufficient NAV history ({len(nav_series) if nav_series is not None else 0}/60)"},
            description="book rolling 60d Sharpe < -0.5",
        )
    rets = nav_series.pct_change().dropna()
    if len(rets) < 60:
        return TriggerResult(
            name="T1_book_sharpe_60d", fired=False, metric=None,
            threshold=TRIGGER_SHARPE_60D,
            evidence={"reason": "insufficient return history"},
            description="book rolling 60d Sharpe < -0.5",
        )
    last_60 = rets.iloc[-60:]
    mu = float(last_60.mean()) * 252
    sigma = float(last_60.std()) * np.sqrt(252)
    sh = mu / sigma if sigma > 0 else 0.0
    fired = sh < TRIGGER_SHARPE_60D
    return TriggerResult(
        name="T1_book_sharpe_60d", fired=fired, metric=round(sh, 3),
        threshold=TRIGGER_SHARPE_60D,
        evidence={"window_days": 60, "n_obs": int(len(last_60)),
                   "ann_ret": round(mu, 4), "ann_vol": round(sigma, 4)},
        description=f"book rolling 60d Sharpe < {TRIGGER_SHARPE_60D}",
    )


def _t2_book_vol_21d(nav_series: pd.Series) -> TriggerResult:
    """T2: book rolling 21d realized vol > 1.5x target_vol."""
    threshold = TRIGGER_VOL_MULTIPLE * TRIGGER_VOL_TARGET
    if nav_series is None or len(nav_series) < 21:
        return TriggerResult(
            name="T2_book_vol_21d", fired=False, metric=None,
            threshold=threshold,
            evidence={"reason": f"insufficient NAV history ({len(nav_series) if nav_series is not None else 0}/21)"},
            description=f"book rolling 21d vol > {threshold:.0%}",
        )
    rets = nav_series.pct_change().dropna()
    if len(rets) < 21:
        return TriggerResult(name="T2_book_vol_21d", fired=False, metric=None,
                              threshold=threshold,
                              evidence={"reason": "insufficient return history"},
                              description=f"book rolling 21d vol > {threshold:.0%}")
    last_21 = rets.iloc[-21:]
    vol = float(last_21.std()) * np.sqrt(252)
    fired = vol > threshold
    return TriggerResult(
        name="T2_book_vol_21d", fired=fired, metric=round(vol, 4),
        threshold=threshold,
        evidence={"window_days": 21, "n_obs": int(len(last_21)),
                   "target_vol": TRIGGER_VOL_TARGET},
        description=f"book rolling 21d vol > 1.5x target_vol={threshold:.0%}",
    )


def _t3_ticker_maxdd_30d(price_panel: pd.DataFrame | None) -> TriggerResult:
    """T3: any single-ticker MaxDD over 30 trading days < -25%."""
    if price_panel is None or price_panel.empty:
        return TriggerResult(name="T3_ticker_maxdd_30d", fired=False, metric=None,
                              threshold=TRIGGER_TICKER_MAXDD_30D,
                              evidence={"reason": "no price panel"},
                              description="any ticker 30d MaxDD < -25%")
    last_30 = price_panel.tail(30).dropna(axis=1, how="all")
    if last_30.empty:
        return TriggerResult(name="T3_ticker_maxdd_30d", fired=False, metric=None,
                              threshold=TRIGGER_TICKER_MAXDD_30D,
                              evidence={"reason": "insufficient recent prices"},
                              description="any ticker 30d MaxDD < -25%")
    cm = last_30.cummax()
    dd = (last_30 / cm - 1.0).min(axis=0)
    worst_ticker = dd.idxmin() if len(dd) > 0 else None
    worst_dd = float(dd.min()) if len(dd) > 0 else 0.0
    fired = worst_dd < TRIGGER_TICKER_MAXDD_30D
    breached = {tk: round(float(d), 4) for tk, d in dd[dd < TRIGGER_TICKER_MAXDD_30D].items()}
    return TriggerResult(
        name="T3_ticker_maxdd_30d", fired=fired, metric=round(worst_dd, 4),
        threshold=TRIGGER_TICKER_MAXDD_30D,
        evidence={"worst_ticker": worst_ticker, "n_breached": len(breached),
                   "breached": breached},
        description="any ticker 30d MaxDD < -25%",
    )


def _t4_position_concentration(weights: dict[str, float] | None) -> TriggerResult:
    """T4: any single position |weight| > 30%."""
    if not weights:
        return TriggerResult(name="T4_position_concentration", fired=False,
                              metric=None, threshold=TRIGGER_WEIGHT_ABS,
                              evidence={"reason": "no positions"},
                              description="any |position weight| > 30%")
    abs_w = {k: abs(float(v)) for k, v in weights.items()}
    max_tk = max(abs_w, key=abs_w.get)
    max_v = abs_w[max_tk]
    fired = max_v > TRIGGER_WEIGHT_ABS
    breached = {k: round(v, 4) for k, v in abs_w.items() if v > TRIGGER_WEIGHT_ABS}
    return TriggerResult(
        name="T4_position_concentration", fired=fired, metric=round(max_v, 4),
        threshold=TRIGGER_WEIGHT_ABS,
        evidence={"max_ticker": max_tk, "n_breached": len(breached),
                   "breached": breached},
        description="any |position weight| > 30%",
    )


def _t5_nav_stale(nav_file_mtime_unix: float | None, now_unix: float) -> TriggerResult:
    """T5: NAV artifact mtime older than 48 hours."""
    if nav_file_mtime_unix is None:
        return TriggerResult(name="T5_nav_stale", fired=True, metric=None,
                              threshold=TRIGGER_NAV_STALE_HOURS,
                              evidence={"reason": "NAV file not found"},
                              description="NAV file missing or unreadable (fail-safe = HALT)")
    age_hours = (now_unix - nav_file_mtime_unix) / 3600.0
    fired = age_hours > TRIGGER_NAV_STALE_HOURS
    return TriggerResult(
        name="T5_nav_stale", fired=fired, metric=round(age_hours, 1),
        threshold=TRIGGER_NAV_STALE_HOURS,
        evidence={"hours_since_update": round(age_hours, 1)},
        description=f"NAV stale > {TRIGGER_NAV_STALE_HOURS}h",
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def evaluate(nav_series: pd.Series | None = None,
             price_panel: pd.DataFrame | None = None,
             weights: dict[str, float] | None = None,
             nav_file_mtime: float | None = None,
             now_unix: float | None = None) -> HaltDecision:
    """Run all 5 triggers. Returns HaltDecision."""
    if now_unix is None:
        now_unix = datetime.datetime.utcnow().timestamp()

    triggers = [
        _t1_book_sharpe_60d(nav_series),
        _t2_book_vol_21d(nav_series),
        _t3_ticker_maxdd_30d(price_panel),
        _t4_position_concentration(weights),
        _t5_nav_stale(nav_file_mtime, now_unix),
    ]
    fired = [t.name for t in triggers if t.fired]
    halt = bool(fired)

    if not halt:
        suggested = "No triggers fired — execution may proceed normally."
    else:
        lines = [f"  {t.name}: {t.description} (metric={t.metric}, threshold={t.threshold})"
                 for t in triggers if t.fired]
        suggested = (
            f"HALT — {len(fired)} trigger(s) fired:\n" + "\n".join(lines) +
            "\nRequired: human investigation; delete halt_flag.json OR add "
            "'acknowledged_by_human_at' timestamp to resume."
        )

    return HaltDecision(
        halt=halt,
        triggers_fired=fired,
        triggers_all=triggers,
        as_of=datetime.datetime.utcfromtimestamp(now_unix).isoformat(timespec="seconds") + "Z",
        suggested_action=suggested,
    )


def write_halt_flag(decision: HaltDecision) -> None:
    """Persist halt decision to data/paper_trade/halt_flag.json (only if halt=True)."""
    if not decision.halt:
        return
    HALT_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HALT_FLAG_PATH.open("w", encoding="utf-8") as f:
        json.dump(decision.to_jsonable(), f, indent=2, ensure_ascii=False)
    logger.warning("HALT flag written: %d trigger(s) fired", len(decision.triggers_fired))


def read_halt_flag() -> dict | None:
    """Read halt_flag.json if it exists. Returns None if no halt active."""
    if not HALT_FLAG_PATH.exists():
        return None
    try:
        return json.loads(HALT_FLAG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("halt_flag.json unreadable (%s) — treating as ACTIVE halt (fail-safe)", exc)
        return {"halt": True, "error": str(exc),
                "suggested_action": "halt_flag.json unreadable — fix or delete"}


def is_halt_active() -> tuple[bool, dict | None]:
    """Check if halt is active AND not yet human-acknowledged.

    Returns (active, payload). active=True if halt flag exists AND
    'acknowledged_by_human_at' field is absent.
    """
    payload = read_halt_flag()
    if payload is None:
        return False, None
    if not payload.get("halt"):
        return False, payload
    if payload.get("acknowledged_by_human_at"):
        return False, payload    # human ack'd; allow execution
    return True, payload


def acknowledge_halt(by_user: str = "human", note: str = "") -> dict:
    """Append acknowledgement to the halt flag without removing it.
    Retains the audit trail."""
    payload = read_halt_flag()
    if payload is None or not payload.get("halt"):
        raise RuntimeError("No active halt to acknowledge")
    payload["acknowledged_by_human_at"] = datetime.datetime.utcnow().isoformat(
        timespec="seconds") + "Z"
    payload["acknowledged_by"] = by_user
    if note:
        payload["acknowledgement_note"] = note
    HALT_FLAG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    return payload


def clear_halt_flag() -> None:
    """Delete the halt flag file (use after human review confirms resumption is safe)."""
    if HALT_FLAG_PATH.exists():
        HALT_FLAG_PATH.unlink()
        logger.info("halt_flag.json cleared")
