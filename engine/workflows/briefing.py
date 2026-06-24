"""engine/workflows/briefing.py — Morning Briefing generator (Wave A).

One page, ≤8 items, deterministic. The PM-facing "what happened
overnight, what needs your attention now" surface that turns the
collection of agents + alert tables into a single workflow loop.

Architecture (per Wave A spec discussion 2026-05-19):
  Section gatherers each return list[BriefingItem]. Curation caps the
  union at MAX_ITEMS using a severity-then-recency priority. Markdown
  renderer is the persistent artifact (data/briefings/YYYY-MM-DD.md).
  Streamlit page (pages/morning_briefing.py) consumes the same in-
  memory Briefing object so the page and the persisted MD never drift.

Doctrine:
  - DETERMINISTIC ONLY. No LLM in this module. The optional Haiku
    narrative summary is a future cherry, not the MVP.
  - READ-ONLY. Briefing generation never writes to any source table.
  - PATTERN 5 BAN COMPATIBILITY. Sections pull from independent
    sources; they don't ask agents anything. The "suggested_followup"
    field is a STRING the user copy-pastes into Chief of Staff — no
    autonomous routing.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


MAX_ITEMS_PER_BRIEFING = 8
BRIEFINGS_DIR          = Path("data/briefings")

_SEVERITY_RANK = {"URGENT": 3, "NOTABLE": 2, "INFO": 1, "": 0}


@dataclass(frozen=True)
class BriefingItem:
    """One curated item the PM should see on the morning briefing.

    Fields:
      section:            'risk' / 'data' / 'attribution' / 'anomaly' / 'governance'
      title:              one-line headline (≤80 chars)
      body:               2-4 sentence detail
      severity:           'URGENT' (act today) / 'NOTABLE' (worth looking) / 'INFO'
      source_table:       table the item was derived from (audit anchor)
      source_ids:         row IDs in that table (lineage)
      suggested_followup: STRING the user can paste into Chief of Staff
                          (e.g. "Ask Risk Manager: 'detail Mode 6 breach'")
      observed_at:        when the underlying event happened (NOT generation time)
    """
    section:            str
    title:              str
    body:               str
    severity:           str
    source_table:       str
    source_ids:         tuple[int, ...]
    suggested_followup: Optional[str] = None
    observed_at:        Optional[datetime.datetime] = None


@dataclass(frozen=True)
class Briefing:
    """One full daily briefing. Items are already curated + ranked."""
    as_of:        datetime.date
    generated_at: datetime.datetime
    items:        tuple[BriefingItem, ...]
    headline:     str   # one-line scan-summary (deterministic, not LLM)
    n_total:      int   # before curation cap (so user knows things were trimmed)


# ──────────────────────────────────────────────────────────────────────────────
# Section gatherers
# ──────────────────────────────────────────────────────────────────────────────
def _section_risk(as_of: datetime.date) -> list[BriefingItem]:
    """Risk Manager alerts in the last 7 calendar days, ranked HARD HALT > SOFT WARN."""
    out: list[BriefingItem] = []
    try:
        from engine.db_models import RiskManagerAlert, SessionFactory
        cutoff = as_of - datetime.timedelta(days=7)
        with SessionFactory() as s:
            rows = (s.query(RiskManagerAlert)
                      .filter(RiskManagerAlert.date >= cutoff)
                      .order_by(RiskManagerAlert.generated_at_utc.desc())
                      .limit(20)
                      .all())
        # Group by mode_id to avoid 5 nearly-duplicate rows for the same mode
        seen_modes: dict[str, list] = {}
        for r in rows:
            seen_modes.setdefault(r.mode_id, []).append(r)

        for mode_id, group in seen_modes.items():
            r0 = group[0]
            sev = "URGENT" if r0.halt_decision else (
                "NOTABLE" if r0.severity == "HARD_HALT" else "INFO"
            )
            affected = []
            try:
                affected = json.loads(r0.affected_json or "[]")
            except Exception:
                pass
            affected_str = ", ".join(affected[:5]) if affected else "—"
            out.append(BriefingItem(
                section            = "risk",
                title              = f"RM Mode {mode_id}: "
                                     f"{r0.rule_description[:80]}",
                body               = (
                    f"{len(group)} breach{'es' if len(group) > 1 else ''} "
                    f"in past 7d (latest: {r0.date}). "
                    f"Severity: {r0.severity}, halt_decision={r0.halt_decision}. "
                    f"Affected: {affected_str}"
                ),
                severity           = sev,
                source_table       = "risk_manager_alerts",
                source_ids         = tuple(g.alert_id for g in group)[:5],
                suggested_followup = (
                    f"Ask Risk Manager: 'Detail Mode {mode_id} breaches in the last week + remediation status.'"
                ),
                observed_at        = r0.generated_at_utc,
            ))
    except Exception as exc:
        logger.warning("briefing._section_risk failed: %s", exc)
    return out


def _section_data(as_of: datetime.date) -> list[BriefingItem]:
    """DQ Inspector — live pre-batch gate result + recent DataQualityAlert."""
    out: list[BriefingItem] = []

    # Live Mode 1-4 pre-batch check
    try:
        from engine.agents.dq_inspector.gates import (
            classify_severity, evaluate_pre_batch,
        )
        breaches = evaluate_pre_batch(as_of)
        severity = classify_severity(breaches)
        if breaches:
            halt_modes = sorted({b.mode_id for b in breaches if b.halt_decision})
            sev = "URGENT" if halt_modes else "NOTABLE"
            out.append(BriefingItem(
                section            = "data",
                title              = f"DQ pre-batch gate: {severity} "
                                     f"({len(breaches)} breach{'es' if len(breaches) > 1 else ''})",
                body               = (
                    "Live Mode 1-4 verdict against today's data: "
                    + "; ".join(f"Mode {b.mode_id} ({b.severity})"
                                for b in breaches[:4])
                ),
                severity           = sev,
                source_table       = "live_gate_evaluate_pre_batch",
                source_ids         = (),
                suggested_followup = (
                    "Ask DQ Inspector: 'Run run_dq_pre_batch_check and explain each breach.'"
                ),
                observed_at        = datetime.datetime.utcnow(),
            ))
    except Exception as exc:
        logger.warning("briefing._section_data live_gate failed: %s", exc)

    # Recent DQ alerts (independent of live gate, captures post-feed / post-batch too)
    try:
        from engine.db_models import DataQualityAlert, SessionFactory
        cutoff = as_of - datetime.timedelta(days=3)
        with SessionFactory() as s:
            rows = (s.query(DataQualityAlert)
                      .filter(DataQualityAlert.date >= cutoff)
                      .order_by(DataQualityAlert.generated_at_utc.desc())
                      .limit(15)
                      .all())
        by_source: dict[str, list] = {}
        for r in rows:
            by_source.setdefault(r.source_id, []).append(r)

        for source_id, group in by_source.items():
            r0 = group[0]
            sev = "URGENT" if r0.halt_decision else (
                "NOTABLE" if r0.severity == "HARD_HALT" else "INFO"
            )
            out.append(BriefingItem(
                section            = "data",
                title              = f"DQ source {source_id}: "
                                     f"{r0.rule_description[:80]}",
                body               = (
                    f"{len(group)} alert{'s' if len(group) > 1 else ''} in past 3d "
                    f"(latest: {r0.date}, phase={r0.phase}, "
                    f"severity={r0.severity}). Mode {r0.mode_id}."
                ),
                severity           = sev,
                source_table       = "data_quality_alerts",
                source_ids         = (),
                suggested_followup = (
                    f"Ask DQ Inspector: 'Why is {source_id} flagging?'"
                ),
                observed_at        = r0.generated_at_utc,
            ))
    except Exception as exc:
        logger.warning("briefing._section_data alerts failed: %s", exc)
    return out


def _section_attribution(as_of: datetime.date) -> list[BriefingItem]:
    """NAV path — last 7 day total return + max single-day move."""
    out: list[BriefingItem] = []
    try:
        from engine.db_models import PortfolioNavSnapshot, SessionFactory
        cutoff = as_of - datetime.timedelta(days=8)
        with SessionFactory() as s:
            rows = (s.query(PortfolioNavSnapshot)
                      .filter(PortfolioNavSnapshot.snapshot_date >= cutoff)
                      .filter(PortfolioNavSnapshot.snapshot_date <= as_of)
                      .order_by(PortfolioNavSnapshot.snapshot_date.asc())
                      .all())
        if len(rows) < 2:
            return out

        nav_first = float(rows[0].nav_close or 0.0)
        nav_last  = float(rows[-1].nav_close or 0.0)
        total_ret = (nav_last / nav_first - 1.0) if nav_first > 0 else 0.0

        # Max single-day Modified Dietz return
        daily = [
            (r.snapshot_date, float(r.daily_modified_dietz))
            for r in rows
            if r.daily_modified_dietz is not None
        ]
        max_up = max(daily, key=lambda t: t[1], default=(None, 0.0))
        max_dn = min(daily, key=lambda t: t[1], default=(None, 0.0))

        # Severity heuristic: |total return| > 1% in a week is NOTABLE,
        # > 3% URGENT; max single-day |move| > 1.5% always NOTABLE.
        abs_ret = abs(total_ret)
        sev = "INFO"
        if abs_ret > 0.03 or abs(max_dn[1]) > 0.015 or abs(max_up[1]) > 0.015:
            sev = "NOTABLE"
        if total_ret < -0.05:
            sev = "URGENT"

        out.append(BriefingItem(
            section            = "attribution",
            title              = (
                f"NAV {nav_first:.4f} → {nav_last:.4f} "
                f"({total_ret*100:+.2f}% over {len(rows)}d)"
            ),
            body               = (
                f"Max +day: {max_up[0]} {max_up[1]*100:+.2f}%. "
                f"Max -day: {max_dn[0]} {max_dn[1]*100:+.2f}%. "
                f"External flow effects ignored in this summary."
            ),
            severity           = sev,
            source_table       = "portfolio_nav_snapshots",
            source_ids         = (),
            suggested_followup = (
                "Ask Attribution Analyst: 'Decompose last week's return by sleeve.'"
            ),
            observed_at        = datetime.datetime.combine(
                rows[-1].snapshot_date, datetime.time(),
            ),
        ))
    except Exception as exc:
        logger.warning("briefing._section_attribution failed: %s", exc)
    return out


def _section_anomaly(as_of: datetime.date) -> list[BriefingItem]:
    """AnomalyFlag rows with confidence_likert >= 3 in past 7 days."""
    out: list[BriefingItem] = []
    try:
        from engine.db_models import AnomalyFlag, SessionFactory
        cutoff = as_of - datetime.timedelta(days=7)
        with SessionFactory() as s:
            rows = (s.query(AnomalyFlag)
                      .filter(AnomalyFlag.scan_date >= cutoff)
                      .filter(AnomalyFlag.confidence_likert >= 3)
                      .order_by(AnomalyFlag.confidence_likert.desc(),
                                AnomalyFlag.scan_date.desc())
                      .limit(10)
                      .all())
        by_ticker: dict[str, list] = {}
        for r in rows:
            by_ticker.setdefault(r.ticker, []).append(r)
        for ticker, group in by_ticker.items():
            r0 = group[0]
            sev = "URGENT" if r0.confidence_likert >= 5 else (
                "NOTABLE" if r0.confidence_likert >= 4 else "INFO"
            )
            out.append(BriefingItem(
                section            = "anomaly",
                title              = (
                    f"Anomaly: {ticker} ({r0.event_class}, "
                    f"confidence {r0.confidence_likert}/5)"
                ),
                body               = (
                    f"{len(group)} flag{'s' if len(group) > 1 else ''} in past 7d "
                    f"(latest: {r0.scan_date}, detector={r0.detector}). "
                    f"Evidence: {(r0.evidence_summary or '')[:160]}"
                ),
                severity           = sev,
                source_table       = "anomaly_flags",
                source_ids         = tuple(g.id for g in group)[:5],
                suggested_followup = (
                    f"Ask Anomaly Sentinel: 'Run forensic_ticker_check on {ticker} '"
                    f"'and pull recent flag history.'"
                ),
                observed_at        = datetime.datetime.combine(
                    r0.scan_date, datetime.time(),
                ),
            ))
    except Exception as exc:
        logger.warning("briefing._section_anomaly failed: %s", exc)
    return out


def _section_governance(as_of: datetime.date) -> list[BriefingItem]:
    """Recent SpecRegistry amendments + open HIGH/MID AuditFinding."""
    out: list[BriefingItem] = []

    # Recent amendments (past 7 days)
    try:
        from engine.preregistration import list_specs
        cutoff_dt = datetime.datetime.combine(
            as_of - datetime.timedelta(days=7), datetime.time(),
        )
        recent_amends: list[tuple] = []   # (spec_id, path, count, latest_reason)
        for r in list_specs():
            log = r.get("amendment_log") or []
            recent = []
            for entry in log:
                at = entry.get("at")
                if not at:
                    continue
                try:
                    at_dt = datetime.datetime.fromisoformat(at.rstrip("Z"))
                except Exception:
                    continue
                if at_dt >= cutoff_dt:
                    recent.append(entry)
            if recent:
                recent_amends.append((
                    int(r["id"]),
                    r.get("spec_path"),
                    len(recent),
                    recent[-1].get("reason", "")[:140],
                ))
        for spec_id, path, n, reason in recent_amends[:3]:
            out.append(BriefingItem(
                section            = "governance",
                title              = f"Spec id={spec_id} amended {n}× this week",
                body               = (
                    f"Path: {path}. Latest reason: {reason}"
                ),
                severity           = "INFO",
                source_table       = "spec_registry",
                source_ids         = (spec_id,),
                suggested_followup = (
                    f"Ask Audit Recorder: 'Show amendment_log entries for spec {spec_id} "
                    f"since {(as_of - datetime.timedelta(days=7)).isoformat()}.'"
                ),
            ))
    except Exception as exc:
        logger.warning("briefing._section_governance amends failed: %s", exc)

    # Open HIGH/MID audit findings
    try:
        from engine.auto_audit_models import AuditFinding
        from engine.memory import SessionFactory
        cutoff = as_of - datetime.timedelta(days=14)
        cutoff_dt = datetime.datetime.combine(cutoff, datetime.time())
        with SessionFactory() as s:
            rows = (s.query(AuditFinding)
                      .filter(AuditFinding.detected_at >= cutoff_dt)
                      .filter(AuditFinding.status == "OPEN")
                      .filter(AuditFinding.severity.in_(["HIGH", "MID"]))
                      .order_by(AuditFinding.severity.desc(),
                                AuditFinding.detected_at.desc())
                      .limit(5)
                      .all())
        if rows:
            sev = "URGENT" if any(r.severity == "HIGH" for r in rows) else "NOTABLE"
            top = rows[0]
            out.append(BriefingItem(
                section            = "governance",
                title              = (
                    f"{len(rows)} OPEN audit finding"
                    f"{'s' if len(rows) > 1 else ''} (severity HIGH/MID)"
                ),
                body               = (
                    f"Most recent: {top.rule_name} ({top.severity}) "
                    f"detected {top.detected_at}. Total open: "
                    f"HIGH={sum(1 for r in rows if r.severity == 'HIGH')}, "
                    f"MID={sum(1 for r in rows if r.severity == 'MID')}."
                ),
                severity           = sev,
                source_table       = "auto_audit_findings",
                source_ids         = tuple(r.id for r in rows),
                suggested_followup = (
                    "Ask Audit Recorder: 'List open HIGH/MID findings "
                    "with rule names and detection dates.'"
                ),
                observed_at        = top.detected_at,
            ))
    except Exception as exc:
        logger.warning("briefing._section_governance findings failed: %s", exc)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Curation: sort by severity then recency, cap to MAX_ITEMS
# ──────────────────────────────────────────────────────────────────────────────
def _curate(
    items:     list[BriefingItem],
    max_k:     int = MAX_ITEMS_PER_BRIEFING,
) -> list[BriefingItem]:
    def _key(it: BriefingItem):
        rank = _SEVERITY_RANK.get(it.severity, 0)
        # observed_at None → very old (sorts last). Compare datetimes
        # directly via isoformat string so we don't trip Windows' epoch
        # underflow on naive datetime.timestamp() near 1970.
        ts = it.observed_at or datetime.datetime(1970, 1, 1)
        return (-rank, -ts.toordinal(), -(ts.hour * 60 + ts.minute))
    sorted_items = sorted(items, key=_key)
    return sorted_items[:max_k]


def _headline(items: list[BriefingItem]) -> str:
    """One-line deterministic scan-summary. Counts only, no LLM."""
    if not items:
        return "All quiet — no items in last 7d above the briefing threshold."
    by_sev = {"URGENT": 0, "NOTABLE": 0, "INFO": 0}
    for it in items:
        by_sev[it.severity] = by_sev.get(it.severity, 0) + 1
    bits = []
    if by_sev["URGENT"]:
        bits.append(f"{by_sev['URGENT']} URGENT")
    if by_sev["NOTABLE"]:
        bits.append(f"{by_sev['NOTABLE']} NOTABLE")
    if by_sev["INFO"]:
        bits.append(f"{by_sev['INFO']} INFO")
    return " · ".join(bits) if bits else "All quiet."


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def generate_briefing(as_of: Optional[datetime.date] = None) -> Briefing:
    """Build the briefing for ``as_of`` (default today). Pure function;
    safe to call from cron or Streamlit page render.

    Each section gatherer is wrapped in a defensive try/except so a
    single bad table or missing model class cannot sink the whole
    briefing — the user still sees what the other sections found.
    """
    if as_of is None:
        as_of = datetime.date.today()

    items: list[BriefingItem] = []
    for fn in (_section_risk, _section_data, _section_attribution,
               _section_anomaly, _section_governance):
        try:
            items += fn(as_of)
        except Exception as exc:
            logger.warning("briefing: section %s raised, skipping: %s",
                           fn.__name__, exc)

    n_total = len(items)
    curated = _curate(items, MAX_ITEMS_PER_BRIEFING)

    return Briefing(
        as_of        = as_of,
        generated_at = datetime.datetime.utcnow(),
        items        = tuple(curated),
        headline     = _headline(curated),
        n_total      = n_total,
    )


def render_as_markdown(briefing: Briefing) -> str:
    """Persistent Markdown form for archive (data/briefings/YYYY-MM-DD.md).

    Format matches what the Streamlit page renders so the on-disk
    artifact and the in-app view never drift in content (only style).
    """
    lines: list[str] = []
    lines.append(f"# Morning Briefing — {briefing.as_of.isoformat()}")
    lines.append("")
    lines.append(f"_Generated {briefing.generated_at.isoformat(timespec='seconds')}Z._")
    lines.append("")
    lines.append(f"**Headline:** {briefing.headline}")
    if briefing.n_total > len(briefing.items):
        lines.append(
            f"_(Curated to top {len(briefing.items)} of {briefing.n_total} "
            f"items; lower-severity / older items omitted.)_"
        )
    lines.append("")
    if not briefing.items:
        lines.append("Nothing to report. Enjoy your morning.")
        return "\n".join(lines) + "\n"

    by_section: dict[str, list[BriefingItem]] = {}
    for it in briefing.items:
        by_section.setdefault(it.section, []).append(it)
    section_order = ["risk", "data", "anomaly", "attribution", "governance"]
    for sec in section_order:
        if sec not in by_section:
            continue
        lines.append(f"## {sec.title()}")
        lines.append("")
        for it in by_section[sec]:
            sev_tag = f"`{it.severity}`"
            lines.append(f"### {sev_tag}  {it.title}")
            lines.append("")
            lines.append(it.body)
            lines.append("")
            if it.suggested_followup:
                lines.append(f"> Follow-up: {it.suggested_followup}")
                lines.append("")
            if it.source_ids:
                ids_preview = ", ".join(str(x) for x in it.source_ids[:5])
                lines.append(f"_(source: `{it.source_table}` ids: {ids_preview})_")
                lines.append("")
            else:
                lines.append(f"_(source: `{it.source_table}`)_")
                lines.append("")
    return "\n".join(lines) + "\n"


def persist_briefing(
    briefing: Briefing,
    out_dir:  Optional[Path] = None,
) -> Path:
    """Write the Markdown briefing to data/briefings/YYYY-MM-DD.md (or
    ``out_dir`` override). Idempotent — overwrites same-date file."""
    out_dir = out_dir or BRIEFINGS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{briefing.as_of.isoformat()}.md"
    path.write_text(render_as_markdown(briefing), encoding="utf-8")
    return path
