"""Operator Console — per-session LLM cost ledger (D4).

Two responsibilities:
    1. Track running cost per session_id (additive ledger)
    2. Enforce per-session hard cap (refuse trigger if exceeded)

Storage: data/operator_console/session_cost_ledger.jsonl, append-only.
Read: compute(session_id) sums the per-session deltas.

This is SEPARATE from data/llm_cost_ledger.jsonl (the project-wide
absolute spend tracker). Both are written in parallel — this one
gates triggers, the project-wide one tracks total burn.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "data" / "operator_console" / "session_cost_ledger.jsonl"


# Per-session cap policy:
#   - default $1.00 per session
#   - user may raise to MAX $5.00 at session-create time
#   - hard ceiling never crossed even on explicit override
DEFAULT_CAP_USD = 1.00
HARD_CEILING_USD = 5.00

# Tolerance applied to mid-execution halt per R6:
#   halt if actual > cap * (1 + COST_OVERRUN_TOLERANCE)
COST_OVERRUN_TOLERANCE = 0.20


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ensure() -> None:
    _LEDGER.parent.mkdir(parents=True, exist_ok=True)


def record_charge(
    *,
    session_id: str,
    actor_id: str,
    station_id: str | None,
    job_id: str | None,
    amount_usd: float,
    note: str = "",
) -> None:
    """Append a cost row. Positive amount = charge. Negative would be
    refund (cancellation refunds for tokens not yet sent), but we
    don't refund LLM tokens already sent — those are sunk."""
    if amount_usd == 0:
        return
    _ensure()
    row = {
        "ts":         _utc_iso(),
        "session_id": session_id,
        "actor_id":   actor_id,
        "station_id": station_id,
        "job_id":     job_id,
        "amount_usd": float(amount_usd),
        "note":       note[:200],
    }
    with _LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_session_spend(session_id: str) -> float:
    """Sum of all charges for this session.

    D4 load-bearing: this number gates triggers. If the ledger has
    corrupt lines we log loudly — a corrupted charge silently dropped
    means the user effectively has *more* budget than the declared
    cap, breaking the D4 invariant. Operator should investigate any
    warning here before continuing to trigger expensive stations."""
    if not _LEDGER.is_file():
        return 0.0
    total = 0.0
    corrupt_lines = 0
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except json.JSONDecodeError:
            corrupt_lines += 1
            continue
        if row.get("session_id") == session_id:
            total += float(row.get("amount_usd", 0.0))
    if corrupt_lines:
        logger.warning(
            "compute_session_spend(%s): D4 ALERT — %d corrupt line(s) in %s "
            "silently skipped. Actual session spend may exceed reported "
            "$%.4f. Investigate before next trigger.",
            session_id, corrupt_lines, _LEDGER, total)
    return round(total, 4)


def can_trigger(*, session_id: str, session_cap_usd: float,
                estimated_charge_usd: float) -> tuple[bool, str]:
    """Gate check before station trigger. Returns (allowed, reason).

    Uses WORST-CASE estimate: if (current_spend + estimate) > cap,
    reject. Better to over-reject than to surprise the user with
    mid-execution halts."""
    cap = min(session_cap_usd, HARD_CEILING_USD)
    spent = compute_session_spend(session_id)
    projected = spent + estimated_charge_usd
    if projected > cap:
        return False, (
            f"Cost cap would be exceeded: "
            f"spent ${spent:.4f} + estimated ${estimated_charge_usd:.4f} "
            f"= ${projected:.4f} > cap ${cap:.4f}"
        )
    return True, "ok"


def must_halt_mid_execution(*, session_id: str, session_cap_usd: float) -> tuple[bool, str]:
    """Stage-boundary check during execution. Returns (must_halt, reason).

    Per R6: halt if actual cost exceeds cap by more than the
    tolerance. Stations call this between stages."""
    cap = min(session_cap_usd, HARD_CEILING_USD)
    spent = compute_session_spend(session_id)
    threshold = cap * (1 + COST_OVERRUN_TOLERANCE)
    if spent > threshold:
        return True, (
            f"Cost overrun: actual ${spent:.4f} > "
            f"cap ${cap:.4f} * (1 + {COST_OVERRUN_TOLERANCE:.0%}) = ${threshold:.4f}"
        )
    return False, "within tolerance"
