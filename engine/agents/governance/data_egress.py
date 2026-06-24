"""engine/agents/governance/data_egress.py — LLM data-egress / residency governance.

THREAT-MODEL control (blueprint spec id=78 §6/§8): a quant fund must not send position-
level or PII data to a provider whose residency / terms disallow it. DeepSeek is a CN
provider; position data egressing there is a real red line. This module:

  - declares provider RESIDENCY + a per-provider EGRESS_POLICY (max sensitivity allowed),
  - classifies a payload's sensitivity DETERMINISTICALLY (anchored to the actual
    position-bearing tool outputs: top_holdings / n_positions / per-name weights),
  - guards an outbound call: warn (log) or enforce (raise) — default warn (non-breaking).

The classifier is a DETECTOR (necessary-not-sufficient): it catches the structural
signature of position/PII data we actually produce; it is conservative to avoid blocking
benign aggregate stats. 0-LLM — pure regex/keyword logic.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Sensitivity lattice (low -> high).
SENSITIVITY_ORDER = ("PUBLIC", "AGGREGATE", "POSITION", "PII")
_RANK = {s: i for i, s in enumerate(SENSITIVITY_ORDER)}

# Provider residency (where the data goes).
PROVIDER_RESIDENCY = {"anthropic": "US", "gemini": "US", "deepseek": "CN"}

# Max sensitivity each provider may RECEIVE. CN provider gets aggregate at most; PII never
# egresses anywhere by default (must be redacted upstream).
EGRESS_POLICY = {
    "anthropic": "POSITION",   # US, enterprise terms — position-level OK, PII not
    "gemini":    "POSITION",
    "deepseek":  "AGGREGATE",  # CN residency — NO position-level / PII
}
_DEFAULT_MAX = "PUBLIC"        # unknown provider -> most restrictive

AUDIT_PATH = Path("data/governance/egress_audit.jsonl")

# ── deterministic sensitivity detectors (anchored to our real tool outputs) ──
_PII_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b"                          # SSN
                     r"|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")  # email
# position-bearing structural signatures (read_today_book_state / lookup_strategy_status
# emit top_holdings = [{"ticker":..,"weight":..}], n_positions, per-strategy weights).
# STRONG structural keys — these appear only in position-bearing tool outputs
# (read_today_book_state / lookup_strategy_status), so any one is sufficient.
_POSITION_KEYS = ("top_holdings", "n_positions", '"ticker"', '"weight"', "position_weight")
_TICKER_WEIGHT = re.compile(r"\b[A-Z]{1,5}\b[^A-Za-z0-9]{0,8}[-+]?\d{0,3}\.\d+")
_AGG_KEYS = ("sharpe", "var", "cvar", "drawdown", "maxdd", "nav", "vol", "rolling",
             "verdict", "alert", "correlation", "ic ", "signal-ic")


def classify_sensitivity(payload: str) -> str:
    """Highest sensitivity class detected in the payload text."""
    p = payload or ""
    low = p.lower()
    if _PII_RE.search(p):
        return "PII"
    key_hits = sum(1 for k in _POSITION_KEYS if k in low)
    tw_hits = len(_TICKER_WEIGHT.findall(p))
    # position data = any explicit holdings key (strong, structural), or several
    # ticker:weight adjacencies (catches free-text holdings even without the JSON keys)
    if key_hits >= 1 or tw_hits >= 3:
        return "POSITION"
    if any(k in low for k in _AGG_KEYS):
        return "AGGREGATE"
    return "PUBLIC"


@dataclasses.dataclass
class EgressDecision:
    provider: str
    residency: str
    sensitivity: str
    max_allowed: str
    allowed: bool
    reason: str


def evaluate_egress(provider: str, payload: str) -> EgressDecision:
    sens = classify_sensitivity(payload)
    max_allowed = EGRESS_POLICY.get(provider, _DEFAULT_MAX)
    residency = PROVIDER_RESIDENCY.get(provider, "UNKNOWN")
    allowed = _RANK[sens] <= _RANK[max_allowed]
    reason = (f"{sens} data -> {provider} ({residency}); max allowed {max_allowed}"
              + ("" if allowed else " — VIOLATION (residency/terms red line)"))
    return EgressDecision(provider, residency, sens, max_allowed, allowed, reason)


def _audit(decision: EgressDecision, workload: str, scope: str) -> None:
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.datetime.utcnow().isoformat() + "Z", "workload": workload,
                "scope": scope, **dataclasses.asdict(decision)}, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("egress audit write failed (non-fatal)")


class EgressViolation(RuntimeError):
    pass


def guard_egress(provider: str, payload: str, *, workload: str = "", scope: str = "",
                 mode: "str | None" = None) -> EgressDecision:
    """Guard an outbound LLM call. mode: off | warn | enforce (default from
    AGENT_EGRESS_MODE env, else 'warn'). warn logs + audits a violation but does not block;
    enforce raises EgressViolation. Never blocks an ALLOWED call."""
    mode = (mode or os.environ.get("AGENT_EGRESS_MODE") or "warn").lower()
    d = evaluate_egress(provider, payload)
    if mode == "off":
        return d
    if not d.allowed:
        _audit(d, workload, scope)
        logger.warning("DATA-EGRESS %s: %s", "BLOCK" if mode == "enforce" else "WARN", d.reason)
        if mode == "enforce":
            raise EgressViolation(d.reason)
    return d


def egress_matrix() -> dict:
    """The documented policy matrix (for review / UI / docs)."""
    return {p: {"residency": PROVIDER_RESIDENCY.get(p, "UNKNOWN"),
                "max_sensitivity": EGRESS_POLICY.get(p, _DEFAULT_MAX)}
            for p in sorted(set(PROVIDER_RESIDENCY) | set(EGRESS_POLICY))}
