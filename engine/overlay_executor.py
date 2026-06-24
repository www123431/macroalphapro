"""engine/overlay_executor.py — deterministic operator-overlay executor (2026-05-24).

The operator overlay is a discretionary sleeve of HUMAN-ORIGINATED positions that
sits ON TOP of the mechanical systematic book — the institutional "discretionary
overlay" pattern. It is the L2 execution target: a chat directive becomes a typed
proposal, the human approves it in the inbox, and THIS deterministic code (no LLM)
validates it against the sleeve's risk budget and writes the position.

0-LLM-in-DECISION preserved: the LLM only emitted a structured intent (ticker +
target weight). Nothing in this module calls a model — validation and execution are
pure deterministic code, gated behind a human approval.

ISOLATED file-backed store (data/overlay/) — deliberately NOT the simulated_positions
table, so the overlay can NEVER contaminate the systematic book's readers (live
dashboard, risk calcs) which do not all filter by track.

Per the human-on-the-loop doctrine, the discretionary sleeve is measured SEPARATELY
(its own positions + trade log) so its value-add is auditable — if its realized IC is
≤ 0 long-run, the data retires it. See [[feedback-human-on-the-loop-inbox-charter]].
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from engine.agents.risk_manager.thresholds import BOOK_SINGLE_TICKER_ABS_CAP

OVERLAY_DIR = Path("data/overlay")
POSITIONS_PATH = OVERLAY_DIR / "positions.json"
TRADES_PATH = OVERLAY_DIR / "trades.jsonl"

# ── Overlay sleeve risk budget ────────────────────────────────────────────────
# NEW policy specific to the discretionary overlay sleeve — NOT a Tier-3-locked
# spec value. Deliberately TIGHTER than the book single-name ceiling: a discretionary
# overlay is a small tilt, not a second book. The locked book ceiling still binds as
# a hard backstop (we take the min of the two).
OVERLAY_SINGLE_NAME_CAP: float = 0.10   # |weight| per overlay name ≤ 10% of book
OVERLAY_GROSS_CAP: float = 0.25         # Σ|weight| across the overlay sleeve ≤ 25% of book
_HARD_SINGLE_CEILING: float = float(BOOK_SINGLE_TICKER_ABS_CAP)  # 0.25 locked backstop


def _now_iso() -> str:
    return _dt.datetime.utcnow().isoformat(timespec="seconds")


def _load_raw() -> dict:
    if POSITIONS_PATH.exists():
        try:
            return json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"as_of": None, "positions": {}}


def read_overlay() -> dict:
    """Current overlay sleeve: {as_of, positions:[{ticker,weight,...}], gross, net, n}."""
    raw = _load_raw()
    pos = raw.get("positions", {})
    items = [{"ticker": k, **v} for k, v in pos.items() if abs((v or {}).get("weight", 0.0)) > 1e-9]
    items.sort(key=lambda x: -abs(x["weight"]))
    gross = sum(abs(i["weight"]) for i in items)
    net = sum(i["weight"] for i in items)
    return {
        "as_of": raw.get("as_of"),
        "positions": items,
        "gross": round(gross, 6),
        "net": round(net, 6),
        "n": len(items),
        "single_name_cap": min(OVERLAY_SINGLE_NAME_CAP, _HARD_SINGLE_CEILING),
        "gross_cap": OVERLAY_GROSS_CAP,
    }


def read_overlay_trades(limit: int = 50) -> list[dict]:
    if not TRADES_PATH.exists():
        return []
    lines = TRADES_PATH.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for ln in lines[-limit:][::-1]:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def validate_overlay_intent(ticker, target_weight, raw: dict | None = None) -> tuple[bool, str]:
    """Deterministic schema + risk-budget gate. Returns (ok, reason)."""
    t = (ticker or "").strip().upper()
    if not t or not t.replace(".", "").replace("-", "").isalnum():
        return False, f"invalid ticker {ticker!r}"
    try:
        w = float(target_weight)
    except (TypeError, ValueError):
        return False, f"target_weight not numeric: {target_weight!r}"
    cap = min(OVERLAY_SINGLE_NAME_CAP, _HARD_SINGLE_CEILING)
    if abs(w) > cap + 1e-9:
        return False, f"|{w:.1%}| exceeds overlay single-name cap {cap:.0%} for {t}"
    raw = raw if raw is not None else _load_raw()
    positions = dict(raw.get("positions", {}))
    if abs(w) < 1e-9:
        positions.pop(t, None)
    else:
        positions[t] = {"weight": w}
    gross = sum(abs((p or {}).get("weight", 0.0)) for p in positions.values())
    if gross > OVERLAY_GROSS_CAP + 1e-9:
        return False, f"resulting overlay gross {gross:.1%} exceeds cap {OVERLAY_GROSS_CAP:.0%}"
    return True, "ok"


def apply_overlay(ticker, target_weight, approval_id=None, rationale: str = "",
                  resolved_by: str = "human") -> dict:
    """Set an overlay position to target_weight (0 → exit). Validates first; on success
    writes positions.json + appends a trade audit row. Pure deterministic; no LLM.

    Returns {'ok': bool, 'message': str, 'exec_detail': dict}."""
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    raw = _load_raw()
    ok, reason = validate_overlay_intent(ticker, target_weight, raw)
    if not ok:
        return {"ok": False, "message": reason, "exec_detail": {}}

    t = ticker.strip().upper()
    w = float(target_weight)
    positions = dict(raw.get("positions", {}))
    before = (positions.get(t) or {}).get("weight", 0.0)

    if abs(w) < 1e-9:
        positions.pop(t, None)
        action = "EXIT"
    else:
        positions[t] = {
            "weight": w,
            "rationale": (rationale or "")[:500],
            "approval_id": approval_id,
            "set_at": _now_iso(),
        }
        action = "SET"

    raw["positions"] = positions
    raw["as_of"] = _dt.date.today().isoformat()
    POSITIONS_PATH.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    trade = {
        "ts": _now_iso(), "date": raw["as_of"], "ticker": t, "action": action,
        "weight_before": round(before, 6), "weight_after": round(w, 6),
        "weight_delta": round(w - before, 6), "approval_id": approval_id,
        "resolved_by": resolved_by, "rationale": (rationale or "")[:500],
    }
    with TRADES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trade, ensure_ascii=False) + "\n")

    return {
        "ok": True,
        "message": f"overlay {action} {t} {before:+.1%}→{w:+.1%}",
        "exec_detail": {"ticker": t, "weight_before": before, "weight_after": w, "action": action},
    }
