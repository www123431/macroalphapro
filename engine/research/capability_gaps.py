"""engine.research.capability_gaps — Commit 3 of the flexibility chain.

Refusal routing slips + the capability-gap demand ledger. Implements
[[feedback-dead-wall-monitoring-standing-2026-06-10]]: every refusal
surface must carry guidance (tier + live data_check + next_action +
single-sourced effort estimate), never a dead wall — and every
refusal is a DEMAND SIGNAL worth logging, because "which template
should I build next" should be answered by measured demand, not
guesswork.

THE THREE LOCKED QUALIFICATIONS (from the 2026-06-10 discussion):
  1. Statistical gates' guidance PRESERVES FRICTION — explains why
     the gate exists + legitimate alternatives; override always
     requires a written reason. Guidance quality ≠ guidance ease.
  2. Effort estimates single-sourced here (GAP_EFFORT) and clearly
     marked estimates; data_check facts are LIVE-PROBED.
  3. The demand ledger dedups by (hypothesis_id, gap_signature) —
     an extractor loop re-refusing the same paper inflates nothing.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
GAPS_LEDGER = _REPO_ROOT / "data" / "research" / "capability_gaps.jsonl"


# Gap classes — the tier taxonomy from the flexibility discussion.
GAP_TIER_2_SIGNAL    = "TIER_2_SIGNAL_FORMULA"     # data cached, formula missing
GAP_TIER_2_APPROVAL  = "TIER_2_PENDING_APPROVAL"   # registered, awaiting card approval
GAP_TIER_3_TEMPLATE  = "TIER_3_TEMPLATE"           # signal_kind×universe untempleted
GAP_TIER_4_DATA      = "TIER_4_DATA_DOMAIN"        # data itself missing
GAP_STAT_GATE        = "STATISTICAL_GATE"          # by-design friction, not a gap

# Single-sourced effort estimates (qualification 2). These are
# ESTIMATES — the live facts come from data_check probes.
GAP_EFFORT: dict[str, str] = {
    GAP_TIER_2_SIGNAL:   "~30 min: one SignalDefinition entry + card review",
    GAP_TIER_2_APPROVAL: "~5 min: review verification card + approve_signal()",
    GAP_TIER_3_TEMPLATE: "~1-3 days: template + contract + tests (C-2f precedent)",
    GAP_TIER_4_DATA:     "~1 week+: data acquisition chain first (LRV precedent: 5 commits)",
    GAP_STAT_GATE:       "n/a — gate is by design; see alternatives in next_action",
}


# ────────────────────────────────────────────────────────────────────
# Live data probes (qualification 2: facts are probed, not asserted)
# ────────────────────────────────────────────────────────────────────
def _probe_cross_sec_data() -> dict:
    """Are the CRSP + Compustat caches present for Tier-2 signal work?"""
    crsp = (_REPO_ROOT / "data" / "cache"
              / "_crsp_msf_long_history.parquet").is_file()
    funda_pit = (_REPO_ROOT / "data" / "cache"
                   / "_compustat_funda_pit.parquet").is_file()
    funda_legacy = (_REPO_ROOT / "data" / "cache"
                      / "_compustat_funda_long_history.parquet").is_file()
    return {
        "crsp_msf_cached":      crsp,
        "compustat_pit_cached": funda_pit,
        "compustat_any_cached": funda_pit or funda_legacy,
    }


_UNIVERSE_DATA_PROBES: dict[str, Path] = {
    "fx_g10": _REPO_ROOT / "data" / "anchor_library"
                / "fx_spot_g10_monthly.parquet",
    "us_equities_top_3000": _REPO_ROOT / "data" / "cache"
                / "_crsp_msf_long_history.parquet",
    "commodity_futures_27": _REPO_ROOT / "data" / "cache"
                / "_cmdty_settle.parquet",
    "us_treasury_curve": _REPO_ROOT / "data" / "cache"
                / "_rates_settle.parquet",
    # bt-flex-4.1 (2026-06-11): SPY monthly + IEF daily are the data
    # foundations for the portfolio_overlay template. Probe checks SPY
    # monthly cache; IEF is daily ETF cache that exists if any
    # template can run.
    "us_balanced_60_40": _REPO_ROOT / "data" / "multivariate_msm_v4"
                / "spy_monthly.parquet",
    # bt-flex-4.2 (2026-06-11): Ken French FF5+Mom weekly source.
    "ken_french_ff5_mom": _REPO_ROOT / "data" / "cache"
                / "ken_french_ff5_mom_weekly.parquet",
    # W6-rigor-A-validate-loop-closed (2026-06-22): MOVE + TLT daily
    # cache for the vrp_treasury Bond-VRP MVP template.
    "us_treasury_options": _REPO_ROOT / "data" / "cache"
                / "_move_tlt_daily.parquet",
}


def _probe_universe_data(universe: str) -> Optional[bool]:
    """True/False when we know how to probe this universe's data;
    None when no probe is registered (unknown domain)."""
    p = _UNIVERSE_DATA_PROBES.get(universe)
    if p is None:
        return None
    return p.is_file()


# ────────────────────────────────────────────────────────────────────
# Guidance builders — one per refusal surface
# ────────────────────────────────────────────────────────────────────
def guidance_unsupported_signal(signal_inputs: tuple) -> dict:
    probe = _probe_cross_sec_data()
    data_ok = probe["crsp_msf_cached"] and probe["compustat_any_cached"]
    return {
        "gap_class":  GAP_TIER_2_SIGNAL if data_ok else GAP_TIER_4_DATA,
        "data_check": probe,
        "next_action": (
            "Add a SignalDefinition entry in engine/research/"
            "signal_registry.py (fields from FIELD_CATALOG; extend the "
            "catalog if a new Compustat column is needed — it is in "
            "the cached parquet for most funda columns). Then "
            "generate_verification_card + approve_signal."
            if data_ok else
            "Underlying CRSP/Compustat caches missing — run the "
            "fetch scripts first (see scripts/extend_compustat_*)."
        ),
        "effort": GAP_EFFORT[GAP_TIER_2_SIGNAL if data_ok
                               else GAP_TIER_4_DATA],
        "requested": list(signal_inputs),
    }


def guidance_unsupported_universe(
    signal_kind: str, universe: str,
) -> dict:
    data_present = _probe_universe_data(universe)
    if data_present is True:
        gap = GAP_TIER_3_TEMPLATE
        action = (f"Data for {universe!r} is cached. Build a template "
                    f"for ({signal_kind!r}, {universe!r}): template fn + "
                    "TemplateContract + dispatcher registry wiring. "
                    "Precedent: carry_g10_fx (C-2f, single commit).")
    elif data_present is False:
        gap = GAP_TIER_4_DATA
        action = (f"No cached data for {universe!r}. Acquisition chain "
                    "first (LRV precedent: fetcher → panel → template).")
    else:
        gap = GAP_TIER_4_DATA
        action = (f"Unknown data domain {universe!r} — no probe "
                    "registered. Assess data availability first; add a "
                    "probe to capability_gaps._UNIVERSE_DATA_PROBES "
                    "when the domain lands.")
    return {
        "gap_class":  gap,
        "data_check": {"universe": universe, "data_cached": data_present},
        "next_action": action,
        "effort":     GAP_EFFORT[gap],
        "requested":  {"signal_kind": signal_kind, "universe": universe},
    }


def guidance_statistical_gate(
    gate: str, detail: str,
) -> dict:
    """Qualification 1: friction PRESERVED. The guidance explains why
    the gate exists and what the legitimate paths are — it never
    makes override one click."""
    alternatives = {
        "N_TRIALS_HARD": (
            "This gate is Bailey-LdP multiple-testing protection — at "
            "this family trial count, 'significant' results are mostly "
            "selection bias. Legitimate paths: (1) review the family's "
            "existing verdicts — the answer may already be on file; "
            "(2) abandon the family — repeated mining of one mechanism "
            "is the factor-zoo failure mode; (3) ONLY with a genuinely "
            "new mechanism hypothesis, pass human_override='<written "
            "reason ≥10 chars>' which is permanently recorded in the "
            "dispatch audit log."
        ),
        "WEEKLY_CAP": (
            "This gate is runaway-automation protection, not a "
            "statistical bar. If this is a deliberate human-initiated "
            "session, pass human_override='<written reason ≥10 chars>' "
            "(audited). If the cap keeps binding on genuine research "
            "cadence, the right fix is a retrospective on dispatch "
            "quality, not a raised cap."
        ),
    }
    return {
        "gap_class":  GAP_STAT_GATE,
        "data_check": {"gate": gate, "detail": detail},
        "next_action": alternatives.get(
            gate, "By-design gate; consult _safety_constants doctrine."),
        "effort": GAP_EFFORT[GAP_STAT_GATE],
    }


# ────────────────────────────────────────────────────────────────────
# Demand ledger
# ────────────────────────────────────────────────────────────────────
def _gap_signature(guidance: dict) -> str:
    """Stable signature for dedup + aggregation."""
    req = guidance.get("requested")
    if isinstance(req, dict):
        req_s = f"{req.get('signal_kind')}|{req.get('universe')}"
    else:
        req_s = ",".join(sorted(map(str, req or [])))
    return f"{guidance['gap_class']}::{req_s}"


def log_gap(
    *,
    hypothesis_id: str,
    guidance: dict,
) -> Optional[str]:
    """Append a demand row — deduped by (hypothesis_id, signature)
    so extractor retries don't inflate counts (qualification 3).
    Returns the signature, or None when deduped/failed."""
    sig = _gap_signature(guidance)
    try:
        if GAPS_LEDGER.is_file():
            for line in GAPS_LEDGER.read_text(
                    encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if (row.get("hypothesis_id") == hypothesis_id
                        and row.get("signature") == sig):
                    return None   # dedup
        GAPS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with GAPS_LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts":            _dt.datetime.utcnow().isoformat(
                                     timespec="seconds") + "Z",
                "hypothesis_id": hypothesis_id,
                "signature":     sig,
                "gap_class":     guidance["gap_class"],
                "requested":     guidance.get("requested"),
                "next_action":   guidance.get("next_action", "")[:300],
                "effort":        guidance.get("effort", ""),
            }) + "\n")
        return sig
    except Exception:
        logger.exception("capability_gaps: ledger write failed")
        return None


# ────────────────────────────────────────────────────────────────────
# Refusal guidance registry (flex-7, 2026-06-10)
# ────────────────────────────────────────────────────────────────────
# Declarative provider registry: reason_code -> callable(context) -> guidance.
# Replaces flex-6's per-site inline guidance attachment (caught by the
# user as a regression to the anchor-as-module disease: adding a new
# refusal class shouldn't require editing every refusal site).
#
# A provider receives the raw refusal `context` dict (whatever the
# refusal site collected) and returns a guidance dict shaped exactly
# like the guidance_* builders above. NOTE: provider names use the
# same reason_code strings DispatchRefusal already declares; nothing
# magic.

REFUSAL_GUIDANCE: dict[str, callable] = {
    "UNSUPPORTED_SIGNAL":      lambda ctx: guidance_unsupported_signal(
                                  tuple(ctx.get("signal_inputs") or ())),
    # 2026-06-13: SIGNAL_INPUT_UNKNOWN is the dispatcher gate #8 reason_code
    # (PIT whitelist failure). Same guidance shape as UNSUPPORTED_SIGNAL —
    # both demand the data path / signal definition. Listing as a
    # separate key (instead of renaming) keeps the dispatcher's existing
    # reason_code stable for log readers.
    "SIGNAL_INPUT_UNKNOWN":    lambda ctx: guidance_unsupported_signal(
                                  tuple(ctx.get("signal_inputs") or ())),
    "TEMPLATE_NOT_CERTIFIED":  lambda ctx: guidance_unsupported_universe(
                                  ctx.get("signal_kind"),
                                  ctx.get("universe")),
    "TEMPLATE_CERT_STALE":     lambda ctx: {
        "gap_class":  GAP_STAT_GATE,   # internal maintenance, not user demand
        "data_check": {"template": ctx.get("template"),
                         "audit_date": ctx.get("audit_date")},
        "next_action": ("Bump pit_audit_date in engine/agents/strengthener/"
                          "templates/_template_contract.py after a fresh PIT "
                          "review (365d freshness window)."),
        "effort":     "~30 min: PIT re-audit + date bump",
    },
    "WEEKLY_CAP":              lambda ctx: guidance_statistical_gate(
                                  "WEEKLY_CAP",
                                  f"{ctx.get('week_count')}/{ctx.get('cap')}"),
    "N_TRIALS_HARD":           lambda ctx: guidance_statistical_gate(
                                  "N_TRIALS_HARD",
                                  (f"family={ctx.get('family')} "
                                   f"n={ctx.get('n_trials')}/"
                                   f"{ctx.get('threshold')}")),
}


def build_refusal(
    reason_code: str,
    *,
    detail:        str,
    metrics:       dict,
    hypothesis_id: Optional[str] = None,
    context:       Optional[dict] = None,
):
    """Single-source refusal factory. Looks up the guidance provider
    for `reason_code`, attaches the guidance to detail + metrics,
    and logs the demand UNLESS the gap is a statistical/maintenance
    gate (which by design isn't unmet user demand).

    Returns a DispatchRefusal. Callers declare only reason_code +
    context — no inline imports, no per-site try/except, no
    duplication.
    """
    from engine.agents.strengthener.factor_dispatcher import DispatchRefusal

    guidance: dict = {}
    provider = REFUSAL_GUIDANCE.get(reason_code)
    if provider is not None:
        try:
            guidance = provider(context or {})
        except Exception:
            logger.exception("build_refusal: guidance provider for %s "
                                "raised", reason_code)
            guidance = {}

    final_detail = detail
    if guidance:
        next_action = guidance.get("next_action", "") or ""
        if next_action and next_action[:60] not in detail:
            final_detail = (detail.rstrip() + " " + next_action[:240]).strip()
        if hypothesis_id and guidance.get("gap_class") != GAP_STAT_GATE:
            try:
                log_gap(hypothesis_id=hypothesis_id, guidance=guidance)
            except Exception:
                logger.exception("build_refusal: log_gap failed")

    enriched_metrics = dict(metrics or {})
    if guidance:
        enriched_metrics["guidance"] = guidance

    return DispatchRefusal(
        reason_code = reason_code,
        detail      = final_detail,
        metrics     = enriched_metrics,
    )


def aggregate_gaps(days_back: int = 30) -> list[dict]:
    """Demand summary: one row per signature with distinct-hypothesis
    count, newest first. Statistical-gate rows excluded (they're
    friction working as designed, not unmet demand)."""
    if not GAPS_LEDGER.is_file():
        return []
    cutoff = (_dt.datetime.utcnow()
                - _dt.timedelta(days=days_back)).isoformat(
                    timespec="seconds") + "Z"
    buckets: dict[str, dict] = {}
    for line in GAPS_LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if (row.get("ts") or "") < cutoff:
            continue
        if row.get("gap_class") == GAP_STAT_GATE:
            continue
        sig = row.get("signature")
        b = buckets.setdefault(sig, {
            "signature":   sig,
            "gap_class":   row.get("gap_class"),
            "requested":   row.get("requested"),
            "next_action": row.get("next_action"),
            "effort":      row.get("effort"),
            "hypotheses":  set(),
            "latest_ts":   "",
        })
        b["hypotheses"].add(row.get("hypothesis_id"))
        b["latest_ts"] = max(b["latest_ts"], row.get("ts") or "")
    out = []
    for b in buckets.values():
        b["demand_count"] = len(b.pop("hypotheses"))
        out.append(b)
    out.sort(key=lambda x: (-x["demand_count"], x["latest_ts"]),
               reverse=False)
    return out
