"""engine.inbox.composer — RESEARCH OPS inbox (NOT a market news feed).

Doctrine (2026-06-02): the system trades by deterministic formulas, NOT
by narrative. Therefore the inbox MUST NOT carry trade-intel like
"FOMC this week" or "this 8-K is positive for our carry leg" — that's
useless content for a 0-LLM-in-DECISION shop. Instead the inbox carries
RESEARCH-PROCESS intel:

  - Engine self-reports (decay diff, capital ramp checkpoints, SLM
    gate progress, OOS countdown, risk forecast delta, DQ flags)
  - New research direction candidates (PFH suggestions, Council
    REJECT verdicts — those signal where NOT to go)
  - Methodology updates (memory entries with new doctrines, capability
    evidence files)
  - Graveyard reinforcement (papers/memory showing X mechanism died)

Each source contributes 0..N items in a stable shape:

    {
      "id":         str   # stable identifier (so unread tracking works)
      "ts":         str   # iso-8601 UTC
      "lane":       "engine" | "direction" | "methodology" | "graveyard"
      "source":     str   # "decay" | "pfh" | "council" | "memory" | ...
      "title":      str
      "summary":    str   # 1-2 sentence summary
      "tone":       "ok" | "warn" | "alert" | "muted" | "info"  (UI color hint)
      "href":       str | None   # drill link
      "metadata":   dict          # source-specific raw fields, hidden by default
    }

Sources are cheap-read (jsonl tails, file mtime checks); we don't compute
anything new — just curate what already exists, and project the doctrine.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent
_MEMORY_DIR     = Path(os.path.expanduser(
    r"~\.claude\projects\c--Users-${USER}-Desktop-intern\memory"
))

_LANE_ENGINE       = "engine"       # System Self-Report (engine telling us about itself)
_LANE_DIRECTION    = "direction"    # Research Direction (new candidates / where to look next)
_LANE_METHODOLOGY  = "methodology"  # Methodology (process-level: doctrines / methods / capability evidence)
_LANE_GRAVEYARD    = "graveyard"    # Reinforcement (past judgments still hold; failed strategies elsewhere)


# Public doctrine — surfaced in /api/research_ops/inbox response so the
# UI can render it inline as a header banner.
DOCTRINE = (
    "Research Ops, NOT a market news feed. "
    "If a piece of content tempts you to override a specific position, "
    "it doesn't belong here — acting on it means you don't trust the "
    "system; not trusting what you yourself researched is incoherent. "
    "Trade-intel (news, macro releases, position-specific filings) is "
    "deliberately excluded. What you find here: engine self-reports, "
    "research direction candidates, methodology updates, and "
    "reinforcement of past judgments."
)


def _utc_iso(d: _dt.datetime | None = None) -> str:
    d = d or _dt.datetime.utcnow()
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=4).hexdigest()


def _stable_id(source: str, payload: str) -> str:
    """Stable id so the same item across two refreshes has the same id —
    that's what enables 'unread' tracking via the user's last-visit ts."""
    return f"ix_{source}_{_short_hash(payload)}"


def _tail_jsonl(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    """Read the last `limit` lines of a jsonl, parse, skip malformed."""
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        # Reasonable for our ledger sizes (< 10MB); read whole file then tail.
        with path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        logger.exception("inbox: tail_jsonl failed for %s", path)
    return out


# ── Per-source probes ─────────────────────────────────────────────


def source_deploy_age() -> list[dict[str, Any]]:
    """Engine self-report: current deploy age + days-to-Tier-3 OOS gate.

    The deployed config has an institutional Tier-3 governance gate at 24
    months OOS. This source surfaces the countdown so the user always sees
    the gate as an active checkpoint, not abstract future."""
    try:
        from engine.portfolio.deployed_registry import load_active
        cfg = load_active()
        days_live = cfg.days_since_deploy
        days_to_tier3 = max(0, 730 - days_live)
        # Lane: engine. Only surface when meaningful (every check, low-noise).
        progress_pct = (days_live / 730.0) * 100.0 if days_live > 0 else 0
        if progress_pct >= 95:
            tone = "ok"
            title = f"Tier-3 OOS gate near · {days_live}d / 730d ({progress_pct:.0f}%)"
        elif progress_pct >= 50:
            tone = "info"
            title = f"Tier-3 OOS gate progress · {days_live}d / 730d ({progress_pct:.0f}%)"
        else:
            tone = "muted"
            title = f"Tier-3 OOS gate · {days_to_tier3}d remaining"
        return [{
            "id":       _stable_id("deploy_age", cfg.id),
            "ts":       _utc_iso(),
            "lane":     _LANE_ENGINE,
            "source":   "deploy_age",
            "title":    title,
            "summary":  (
                f"Config {cfg.id} live since {cfg.deploy_date}. "
                f"Real-capital deployment requires 24-month OOS forward "
                f"validation; checkpoint review at 25%, 50%, 75%, 95% "
                f"progress."
            ),
            "tone":     tone,
            "href":     "/ops",
            "metadata": {
                "config_id":     cfg.id,
                "deploy_date":   cfg.deploy_date,
                "days_live":     days_live,
                "days_remaining": days_to_tier3,
                "progress_pct":  round(progress_pct, 1),
            },
        }]
    except Exception:
        logger.exception("inbox.source_deploy_age failed")
        return []


def source_mcc_pending() -> list[dict[str, Any]]:
    """Engine self-report: Model Change Control queue has pending decisions
    awaiting your two-eye approval."""
    try:
        from engine.governance.approval_ledger import list_requests
        pending = list_requests(status="pending", limit=20)
        if not pending:
            return []
        out = []
        for r in pending[:5]:
            out.append({
                "id":       _stable_id("mcc", r["id"]),
                "ts":       r["created_at"],
                "lane":     _LANE_ENGINE,
                "source":   "mcc",
                "title":    f"MCC pending · {r['title'][:80]}",
                "summary":  r.get("summary", "")[:280],
                "tone":     "warn",   # always demands attention
                "href":     "/approvals",
                "metadata": {"request_id": r["id"], "request_type": r["request_type"]},
            })
        return out
    except Exception:
        logger.exception("inbox.source_mcc_pending failed")
        return []


def source_code_drift() -> list[dict[str, Any]]:
    """Engine self-report: Python constants disagree with active_deployment.yaml.

    Surfaces the silent-drift mode that bit us on 2026-06-02 (1.03 Sharpe
    showing despite config C live). If drift detected, it's a hard alert."""
    try:
        from engine.portfolio.deployed_registry import assert_constants_match
        from engine.portfolio.combined_book import (
            DEFAULT_CARRY_RISK_WEIGHT, DEFAULT_TSMOM_RISK_WEIGHT,
            DEFAULT_CRISIS_HEDGE_RISK_WEIGHT, DEFAULT_MOM_HEDGE_RISK_WEIGHT,
            DEFAULT_BOOK_VOL_TARGET,
        )
        issues = assert_constants_match(
            carry_risk_weight    = DEFAULT_CARRY_RISK_WEIGHT,
            tsmom_risk_weight    = DEFAULT_TSMOM_RISK_WEIGHT,
            crisis_risk_weight   = DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
            mom_hedge_risk_weight= DEFAULT_MOM_HEDGE_RISK_WEIGHT,
            book_vol_target      = DEFAULT_BOOK_VOL_TARGET,
        )
        if not issues:
            return []
        return [{
            "id":       _stable_id("drift", "|".join(issues)),
            "ts":       _utc_iso(),
            "lane":     _LANE_ENGINE,
            "source":   "code_drift",
            "title":    f"Code drift detected · {len(issues)} mismatch(es)",
            "summary":  "; ".join(issues)[:280],
            "tone":     "alert",
            "href":     "/ops",
            "metadata": {"issues": issues},
        }]
    except Exception:
        logger.exception("inbox.source_code_drift failed")
        return []


def source_decay_alerts(limit: int = 5) -> list[dict[str, Any]]:
    """Recent decay sentinel alerts (degradation flags / regime shifts)."""
    path = _REPO_ROOT / "data" / "research" / "decay_alerts.jsonl"
    rows = _tail_jsonl(path, limit=20)
    if not rows:
        return []
    out = []
    for r in rows[-limit:]:
        sleeve  = r.get("sleeve") or r.get("mechanism") or "?"
        level   = (r.get("level") or r.get("severity") or "info").lower()
        message = r.get("message") or r.get("rule") or ""
        ts      = r.get("ts") or r.get("as_of") or _utc_iso()
        out.append({
            "id":       _stable_id("decay", f"{ts}:{sleeve}:{level}"),
            "ts":       ts,
            "lane":     _LANE_ENGINE if level in ("alert", "warn") else _LANE_DIRECTION,
            "source":   "decay",
            "title":    f"Decay {level.upper()} · {sleeve}",
            "summary":  message[:280],
            "tone":     "alert" if level == "alert" else "warn" if level == "warn" else "muted",
            "href":     f"/lab/decay/detail?sleeve={sleeve}",
            "metadata": {"sleeve": sleeve, "level": level, "raw": r},
        })
    return out


def source_decay_alerts_canonical(
    limit: int = 10,
    *,
    show_acked: bool = False,
) -> list[dict[str, Any]]:
    """G.1 (2026-06-09): canonical decay_alert events from research_store.

    Routes events emitted by engine.research.decay_watch_trigger into the
    inbox. Coexists with legacy `source_decay_alerts` (which reads
    data/research/decay_alerts.jsonl from the older per-mechanism decay
    history system) — both surface in the same inbox lane.

    Per [[feedback-research-auto-capital-human-2026-06-05]] the event
    carries SUGGESTION not command; tone reflects severity:
      RED severity      → tone "alert" (red)
      MARGINAL severity → tone "warn"  (amber)
      NEUTRAL / else    → tone "info"  (blue — surfaced but no urgency)

    I (2026-06-09): acked alerts are HIDDEN from inbox by default.
    The Inbox v3 surface is for ACTION ITEMS; once the principal has
    acked, the alert moves to /lab/decay/detail history. Pass
    show_acked=True to also surface acked items (e.g. for audit views).
    """
    try:
        from engine.research_store import store
    except Exception:
        return []
    try:
        # Fetch a broader window than the visible limit so we can
        # find ack events that reference older originals.
        events = store.filter_events(event_type="decay_alert",
                                          limit=max(limit * 5, 200))
    except Exception:
        logger.exception("inbox: filter_events for decay_alert failed")
        return []

    # Compute current ack state per ORIGINAL event so we can skip
    # acked alerts (I (2026-06-09))
    canonical_events = [
        e for e in events if "decay_watch" in (e.tags or ())
    ]
    try:
        from api.main import _decay_ack_chain    # reuse the helper
        ack_state = _decay_ack_chain(canonical_events)
    except Exception:
        # Defensive fallback — if helper unavailable, don't break inbox
        ack_state = {}

    out: list[dict[str, Any]] = []
    for e in events:
        verdict = (e.verdict.value if hasattr(e.verdict, "value")
                     else str(e.verdict)).upper()
        # SLM legacy emits the same event_type with different verdict
        # semantics; only the C trigger writes RED/MARGINAL with the
        # `decay_watch` tag. Filter to those so we don't double-surface
        # legacy decay sentinel rows. Tag check keeps this inbox slice
        # SPECIFICALLY the new C trigger.
        tags = e.tags or ()
        is_c_trigger = "decay_watch" in tags
        if not is_c_trigger:
            continue
        # I: skip admin (ack/unack) events themselves — they're not
        # action items, they're records of action.
        if ("acknowledged" in tags) or ("unacknowledged" in tags):
            continue
        # I: skip originals whose latest state is `acknowledged`
        # (unless show_acked override).
        if not show_acked:
            cur = ack_state.get(e.event_id)
            if cur and cur.get("is_acknowledged"):
                continue
        if len(out) >= limit:
            break
        metrics = e.metrics or {}
        sleeve = e.subject_id
        triggers = metrics.get("triggers_hit") or []
        severity = metrics.get("severity") or verdict
        wbr = metrics.get("worst_best_sharpe_ratio")
        if severity == "RED":
            tone = "alert"
        elif severity == "MARGINAL":
            tone = "warn"
        else:
            tone = "info"
        # Tight title: "Decay RED · cross_asset_carry (A,B,C)"
        title = (f"Decay {severity} · {sleeve}"
                   + (f" ({','.join(triggers)})" if triggers else ""))
        # 1-line preview: strip the "SUGGESTION" suffix for the inbox
        # preview (it's redundant when the action button says "Review")
        summary = e.summary
        if "SUGGESTION" in summary:
            summary = summary.split(" — SUGGESTION:")[0]
        if wbr is not None:
            summary = (f"worst/best={wbr:.2f}, "
                         f"triggers={','.join(triggers) or 'none'}. {summary}")
        out.append({
            "id":       _stable_id("decay_canonical",
                                       f"{e.event_id}"),
            "ts":       e.ts,
            "lane":     _LANE_ENGINE,   # engine self-report
            "source":   "decay_watch",
            "title":    title[:120],
            "summary":  summary[:280],
            "tone":     tone,
            "href":     f"/lab/decay/{sleeve}",
            "metadata": {
                "sleeve":         sleeve,
                "severity":       severity,
                "triggers_hit":   triggers,
                "n_triggers":     metrics.get("n_triggers"),
                "worst_best_sharpe_ratio": wbr,
                "monotone_decay": metrics.get("monotone_decay"),
                "event_id":       e.event_id,
                "verdict_event_id": e.event_id,
            },
        })
    return out


def source_specification_robustness_overfit(
    limit: int = 10,
) -> list[dict[str, Any]]:
    """G.2 (2026-06-09): surfaces B-lens (specification_robustness)
    verdicts of LIKELY_OVERFIT / MARGINAL_OVERFIT into the inbox.

    The lens output is embedded inside `factor_verdict_filed` event
    metrics (under metrics.specification_robustness). We scan recent
    factor verdict events and surface only the overfit cases — ROBUST
    is the happy path, no notification needed.
    """
    try:
        from engine.research_store import store
    except Exception:
        return []
    try:
        events = store.filter_events(
            event_type="factor_verdict_filed", limit=200,
        )
    except Exception:
        logger.exception("inbox: filter_events for factor_verdict failed")
        return []

    out: list[dict[str, Any]] = []
    for e in events:
        sr = (e.metrics or {}).get("specification_robustness") or {}
        if not sr:
            continue
        v = sr.get("verdict")
        if v not in ("LIKELY_OVERFIT", "MARGINAL_OVERFIT"):
            continue
        score = sr.get("stability_score")
        score_str = (f"{score:.2f}" if isinstance(score, (int, float))
                      else "N/A")
        # LIKELY → alert; MARGINAL → warn
        tone = "alert" if v == "LIKELY_OVERFIT" else "warn"
        # Tight title: "Overfit risk · subject_id (stability=0.17)"
        title = (f"{'Likely overfit' if v == 'LIKELY_OVERFIT' else 'Marginal overfit'}"
                   f" · {e.subject_id} (stability={score_str})")
        base_sharpe = sr.get("base_sharpe")
        median = sr.get("sharpe_median")
        summary_parts = []
        if base_sharpe is not None:
            summary_parts.append(f"base Sharpe={base_sharpe:+.2f}")
        if median is not None:
            summary_parts.append(f"neighborhood median={median:+.2f}")
        ns = sr.get("neighborhood_size") or 0
        summary_parts.append(f"{ns}-cell ablation")
        summary = "; ".join(summary_parts) + ". " + (
            "Likely cherry-picked parameters."
            if v == "LIKELY_OVERFIT"
            else "Mild parameter sensitivity."
        )
        out.append({
            "id":       _stable_id("spec_robust", f"{e.event_id}:{v}"),
            "ts":       e.ts,
            "lane":     _LANE_DIRECTION,   # affects how we read research candidates
            "source":   "spec_robust",
            "title":    title[:120],
            "summary":  summary[:280],
            "tone":     tone,
            "href":     f"/research/verdict?event_id={e.event_id}",
            "metadata": {
                "subject_id":      e.subject_id,
                "verdict":         v,
                "stability_score": score,
                "base_sharpe":     base_sharpe,
                "sharpe_median":   median,
                "neighborhood_size": ns,
                "event_id":        e.event_id,
            },
        })
        if len(out) >= limit:
            break
    return out


def source_anchor_spanned_factors(
    limit: int = 10,
    *,
    headline_t_floor:    float = 1.65,   # MARGINAL or better
    residual_t_ceiling:  float = 1.65,   # but residual fails MARGINAL
    headline_minus_residual_min: float = 1.5,   # gap matters
) -> list[dict[str, Any]]:
    """G.3 (2026-06-09): surfaces factors whose headline t passed
    significance but whose anchor-orthogonal residual α did NOT —
    "your factor is a known-risk-premium restatement". GP/A is the
    canonical example: headline t=3.57, residual t=0.8 → not novel α.

    Reads anchor_orthogonality (FF5+MOM equity, or LRV HML_FX for FX)
    from recent factor_verdict_filed events.
    """
    try:
        from engine.research_store import store
    except Exception:
        return []
    try:
        events = store.filter_events(
            event_type="factor_verdict_filed", limit=200,
        )
    except Exception:
        logger.exception(
            "inbox: filter_events for anchor-spanned scan failed")
        return []

    out: list[dict[str, Any]] = []
    for e in events:
        m = e.metrics or {}
        ao = m.get("anchor_orthogonality") or {}
        if not ao:
            continue
        headline_t = m.get("nw_t_stat")
        residual_t = ao.get("alpha_nw_t")
        if headline_t is None or residual_t is None:
            continue
        try:
            headline_abs  = abs(float(headline_t))
            residual_abs  = abs(float(residual_t))
        except (TypeError, ValueError):
            continue
        # Filter: was the factor "significant" at headline level but
        # NOT at residual level? + a meaningful gap (not just sampling
        # noise around the threshold).
        gap = headline_abs - residual_abs
        if not (headline_abs >= headline_t_floor
                  and residual_abs <  residual_t_ceiling
                  and gap         >= headline_minus_residual_min):
            continue
        anchor_lib = ao.get("anchor_library", "anchors")
        betas = ao.get("betas") or {}
        # Pick the largest-|β| anchor as the smoking gun
        if betas:
            dominant = max(betas.items(), key=lambda kv:
                              abs(kv[1]) if kv[1] is not None else 0)
            dom_name, dom_beta = dominant
            dom_str = f"{dom_name} β={dom_beta:+.2f}"
        else:
            dom_str = ""
        title = (f"Anchor-spanned · {e.subject_id} "
                   f"(t {headline_abs:.2f}→{residual_abs:.2f})")
        summary = (
            f"Headline t={headline_abs:.2f} but residual α t={residual_abs:.2f} "
            f"after stripping {anchor_lib}"
            + (f" (dominant: {dom_str})" if dom_str else "")
            + ". Likely textbook restatement, not novel alpha."
        )
        out.append({
            "id":       _stable_id("anchor_spanned",
                                       f"{e.event_id}"),
            "ts":       e.ts,
            "lane":     _LANE_DIRECTION,
            "source":   "anchor_spanned",
            "title":    title[:120],
            "summary":  summary[:280],
            "tone":     "warn",
            "href":     f"/research/verdict?event_id={e.event_id}",
            "metadata": {
                "subject_id":      e.subject_id,
                "headline_t":      headline_abs,
                "residual_alpha_t": residual_abs,
                "anchor_library":  anchor_lib,
                "dominant_beta":   dom_str,
                "event_id":        e.event_id,
            },
        })
        if len(out) >= limit:
            break
    return out


def source_capability_gaps_digest(
    days_back: int = 30, limit: int = 5,
) -> list[dict[str, Any]]:
    """flex-3 (2026-06-10): capability-gap demand digest. One info
    row per gap signature with distinct-hypothesis demand count —
    'which template/signal should I build next' answered by measured
    demand, per [[feedback-dead-wall-monitoring-standing-2026-06-10]].
    Statistical-gate refusals excluded (friction working as designed)."""
    try:
        from engine.research.capability_gaps import aggregate_gaps
        gaps = aggregate_gaps(days_back=days_back)
    except Exception:
        logger.exception("inbox: capability gaps aggregate failed")
        return []
    out: list[dict[str, Any]] = []
    for g in gaps[:limit]:
        req = g.get("requested")
        if isinstance(req, dict):
            req_label = f"{req.get('signal_kind')}×{req.get('universe')}"
        else:
            req_label = ", ".join(map(str, (req or [])))[:60]
        n = g.get("demand_count", 1)
        out.append({
            "id":       _stable_id("capgap", g["signature"]),
            "ts":       g.get("latest_ts") or _utc_iso(),
            "lane":     _LANE_DIRECTION,
            "source":   "capability_gap",
            "title":    (f"Capability demand ×{n}: {req_label} "
                           f"[{g.get('gap_class')}]"),
            "summary":  (f"{g.get('next_action', '')[:200]} "
                           f"(effort: {g.get('effort', '?')})"),
            "tone":     "info" if n < 3 else "warn",
            "href":     None,
            "metadata": dict(g),
        })
    return out


def _safe_relpath(p: Path) -> str:
    """Best-effort path-relative-to-repo. Falls back to absolute string
    when the path is outside the repo (typical in tests with tmp_path)."""
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def source_external_audit_digest(
    *,
    days_back: int = 7,
) -> list[dict[str, Any]]:
    """external_audit (2026-06-13): adversarial LLM review digest. ONE row
    summarizing recent verdicts' independent audits. Alert tone if ANY
    critical/concern severity surfaced in the window.

    When EXTERNAL_AUDIT_PROVIDER is 'stub' (default), all severities are
    'skipped' and this surface emits nothing — no noise until a real
    provider is wired.
    """
    try:
        from engine.research.external_audit import (
            recent_audits, severity_breakdown,
        )
    except Exception:
        logger.exception("inbox: external_audit import failed")
        return []

    audits = recent_audits(days_back=days_back)
    if not audits:
        return []
    breakdown = severity_breakdown(audits)
    total_real = sum(breakdown.values())   # excludes skipped
    if total_real == 0:
        return []   # provider was stub for all; nothing to report

    critical = breakdown["critical"]
    concern  = breakdown["concern"]

    if critical > 0:
        tone = "alert"
    elif concern > 0:
        tone = "warn"
    else:
        tone = "info"

    # Collect flagged categories with counts
    cat_counts: dict[str, int] = {}
    for a in audits:
        for c in (a.get("flagged_categories") or []):
            cat_counts[c] = cat_counts.get(c, 0) + 1
    cat_str = ", ".join(
        f"{c}({n})" for c, n in sorted(cat_counts.items(), key=lambda kv: -kv[1])[:5]
    ) or "none"

    cost_total = sum(float(a.get("cost_estimate_usd", 0.0) or 0.0) for a in audits)

    return [{
        "id":       _stable_id("ext_audit", str(total_real)),
        "ts":       _utc_iso(),
        "lane":     _LANE_ENGINE,
        "source":   "external_audit",
        "title":    (
            f"External audit: {total_real} verdict(s) reviewed, "
            f"{critical} critical / {concern} concern"
        ),
        "summary":  (
            f"flagged: {cat_str} | budget ~${cost_total:.2f} this week"
        )[:240],
        "tone":     tone,
        "href":     None,
        "metadata": {
            "n_audits":        total_real,
            "breakdown":       breakdown,
            "cat_counts":      cat_counts,
            "cost_total_usd":  cost_total,
        },
    }]


def source_belief_autopsy_digest(
    *,
    days_back: int = 7,
    min_for_summary: int = 3,
) -> list[dict[str, Any]]:
    """belief-2 (2026-06-12): system-self-calibration digest. Reads
    autopsies (predictions vs verdicts) + pattern detection output,
    surfaces ONE row summarizing:

      - how many autopsies accumulated in the last 7 days
      - overall mean Brier (lower = better calibrated)
      - per-family Brier hotspot (worst-calibrated family this week)
      - active pattern flags (GREEN_OVERCONFIDENCE etc.)
      - direction breakdown counts

    The principal reads this nightly to see "is the system over/under
    predicting? which family is most miscalibrated?" without opening
    /lab/system_epistemics (belief-3 dashboard, not yet shipped).

    Lane: _LANE_ENGINE (engine telling us about itself).

    Tone:
      - alert  : any pattern flag fired (GREEN_OVERCONFIDENCE etc.)
      - warn   : mean Brier > 0.50
      - info   : default
    """
    try:
        from engine.research.belief_autopsy import (
            detect_patterns, AUTOPSIES_PATH,
        )
    except Exception:
        logger.exception("inbox: belief_autopsy import failed")
        return []

    if not AUTOPSIES_PATH.is_file():
        return []

    stats = detect_patterns(autopsies_path=AUTOPSIES_PATH)
    n = stats.get("n_autopsies", 0)
    if n < min_for_summary:
        return []

    flags = stats.get("patterns", []) or []
    flag_names = [p["pattern"] for p in flags]
    mean_brier = float(stats.get("mean_brier", 0.0))
    direction_counts = stats.get("direction_counts", {}) or {}
    family_brier = stats.get("family_brier", {}) or {}

    # Identify hotspot — family with highest Brier (worst calibrated)
    hotspot = ""
    hotspot_brier = 0.0
    if family_brier:
        sorted_fam = sorted(family_brier.items(), key=lambda kv: -kv[1])
        hotspot, hotspot_brier = sorted_fam[0]

    if flags:
        tone = "alert"
    elif mean_brier > 0.50:
        tone = "warn"
    else:
        tone = "info"

    direction_str = ", ".join(
        f"{k}={v}" for k, v in sorted(direction_counts.items()) if v > 0
    ) or "no surprises"

    title = (
        f"Belief autopsy: {n} run(s), mean Brier {mean_brier:.2f}"
        + (f" — FLAGS: {','.join(flag_names)}" if flag_names else "")
    )

    summary_parts = [direction_str]
    if hotspot:
        summary_parts.append(
            f"hotspot {hotspot} Brier {hotspot_brier:.2f}"
        )
    if flags:
        for f in flags[:1]:
            advice = (f.get("advice") or "")[:200]
            if advice:
                summary_parts.append(f"advice: {advice}")
    summary = " | ".join(summary_parts)

    return [{
        "id":       _stable_id("belief_autopsy", str(n)),
        "ts":       _utc_iso(),
        "lane":     _LANE_ENGINE,
        "source":   "belief_autopsy",
        "title":    title,
        "summary":  summary[:240],
        "tone":     tone,
        "href":     None,
        "metadata": {
            "n_autopsies":      n,
            "mean_brier":       mean_brier,
            "family_brier":     family_brier,
            "direction_counts": direction_counts,
            "flags":            flags,
        },
    }]


def source_daily_ingest_digest(*, days_back: int = 2) -> list[dict[str, Any]]:
    """2026-06-14: papers_curator daily ingest summary. Reads recent
    papers_curator_synthesis_run events from the research_store and
    renders one Inbox row per day with crawl/filter/summarize/synthesize
    counts. Companion to source_burndown_digest — together they form
    the daily research heartbeat.

    Why event-sourced (not file-scraped): synthesis_runner emits the
    event with snapshot + candidates_summary already structured. The
    Inbox just needs to fetch + render.
    """
    try:
        from engine.research_store.store import filter_events
        from engine.research_store.schema import EventType
    except Exception:
        return []

    try:
        events = filter_events(event_type=EventType.papers_curator_synthesis_run)
    except Exception:
        logger.warning("daily_ingest_digest: filter_events failed", exc_info=True)
        return []

    if not events:
        return []

    cutoff_iso = (_dt.datetime.utcnow() - _dt.timedelta(days=days_back)).strftime("%Y-%m-%d")
    recent = [e for e in events if (getattr(e, "ts", None) or "")[:10] >= cutoff_iso]
    if not recent:
        return []

    # Sort newest first
    recent.sort(key=lambda e: getattr(e, "ts", "") or "", reverse=True)

    out: list[dict[str, Any]] = []
    for ev in recent[:10]:   # cap rendering
        ts = getattr(ev, "ts", "") or ""
        m = getattr(ev, "metrics", None) or {}
        n_cand = m.get("n_candidates") or 0
        n_written = m.get("n_written") or 0
        snap = m.get("snapshot") or {}
        recent_summ = snap.get("recent_summaries") or 0
        cs = m.get("candidates_summary") or []
        fams = sorted({(c or {}).get("mechanism_family") for c in cs if c}) if cs else []

        tone = "ok" if n_written > 0 else "info"

        title = (
            f"Daily ingest {ts[:10]}: {n_written} hyp added "
            f"(from {n_cand} candidate(s), {recent_summ} summary window)"
        )
        summary_parts = [
            f"papers_curator synthesis: candidates={n_cand}, written={n_written}.",
        ]
        if fams:
            summary_parts.append(f"families: {','.join(f for f in fams if f)[:120]}.")
        errors = m.get("errors") or []
        if errors:
            summary_parts.append(f"errors: {len(errors)}.")
        summary = " ".join(summary_parts)

        out.append({
            "id":       _stable_id("ingest", getattr(ev, "event_id", "") or ""),
            "ts":       ts,
            "lane":     _LANE_ENGINE,
            "source":   "papers_curator_daily",
            "title":    title,
            "summary":  summary[:240],
            "tone":     tone,
            "href":     None,
            "metadata": {
                "event_id":     getattr(ev, "event_id", None),
                "n_candidates": n_cand,
                "n_written":    n_written,
                "families":     fams,
            },
        })
    return out


def _load_audits_by_event_id(event_ids: set) -> dict[str, dict]:
    """Phase 4 (2026-06-13): load external_audits.jsonl rows whose
    subject_ref matches any of the provided dispatch event_ids.
    Returns {event_id: audit_row}."""
    if not event_ids:
        return {}
    path = _REPO_ROOT / "data" / "research" / "external_audits.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ref = row.get("subject_ref")
            if ref in event_ids:
                out[ref] = row
    except Exception:
        logger.warning("digest: load audits failed", exc_info=True)
    return out


def _load_rigor_by_event_id(event_ids: set) -> dict[str, dict]:
    """Phase 4 (2026-06-13): load post_green_rigor.jsonl rows whose
    verdict_event_id matches any of the provided dispatch event_ids.
    Returns {event_id: rigor_report}."""
    if not event_ids:
        return {}
    path = _REPO_ROOT / "data" / "research" / "post_green_rigor.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            ref = row.get("verdict_event_id")
            if ref in event_ids:
                out[ref] = row
    except Exception:
        logger.warning("digest: load rigor failed", exc_info=True)
    return out


def source_burndown_digest(
    *,
    days_back: int = 2,
    plan_dir: Optional[Path] = None,
    outcome_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """burn-2 (2026-06-11): daily cron burndown digest. One row per recent
    plan run with verdict / refusal / extraction summary; pulls from the
    plan + outcome JSONs that burn-1a / burn-1b write to
    data/cron_burndown/plans/ and data/cron_burndown/outcomes/.

    Why an Inbox row instead of a new page: principal doesn't need a
    burndown UI — just one nightly summary they can scan in 30 seconds:
    "what cron ran, what verdict, what was skipped, what cap is left".
    If they want detail, the row links to the plan JSON on disk via
    metadata['plan_path'].

    Each row pairs (plan, outcome): if outcome is missing (dry-run only
    or execution didn't enable), surfaces as 'info' tone with planned-
    candidate preview. If outcome is present, surfaces verdict
    distribution + cap-usage summary."""
    plans_root = plan_dir or (_REPO_ROOT / "data" / "cron_burndown" / "plans")
    outcomes_root = outcome_dir or (_REPO_ROOT / "data" / "cron_burndown" / "outcomes")
    if not plans_root.is_dir():
        return []

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days_back)
    cutoff_iso = cutoff.strftime("%Y-%m-%d")

    plan_files = sorted(
        [p for p in plans_root.glob("*.json") if p.stem >= cutoff_iso],
        reverse=True,
    )

    out: list[dict[str, Any]] = []
    for pf in plan_files:
        try:
            plan_obj = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("burndown_digest: cannot parse %s", pf.name)
            continue

        plan_id = plan_obj.get("plan_id", "")
        ts      = plan_obj.get("ts", "")
        target_k = plan_obj.get("target_k", 0)
        actual_k = plan_obj.get("actual_k", 0)
        candidates = plan_obj.get("candidates", []) or []
        dry_run  = plan_obj.get("dry_run", True)
        usage_summary = plan_obj.get("usage_summary", "")

        # Find matching outcomes (same date prefix + plan_id[:8])
        outcome_glob = f"{pf.stem[:10]}_{plan_id[:8]}.json"
        outcome_path = outcomes_root / outcome_glob if outcomes_root.is_dir() else None
        outcomes_data = None
        if outcome_path and outcome_path.is_file():
            try:
                outcomes_data = json.loads(outcome_path.read_text(encoding="utf-8"))
            except Exception:
                outcomes_data = None

        if outcomes_data is not None:
            # Execution actually ran — summarize outcomes
            outs = outcomes_data.get("outcomes", []) or []
            green = sum(1 for o in outs if (o.get("verdict") or "") == "GREEN")
            margin = sum(1 for o in outs if (o.get("verdict") or "") == "MARGINAL")
            red = sum(1 for o in outs if (o.get("verdict") or "") == "RED")
            err = sum(1 for o in outs
                       if (o.get("verdict") or "") in ("EXECUTION_ERROR",))
            refused = sum(1 for o in outs if o.get("refusal_reason"))
            extract_fail = sum(1 for o in outs if not o.get("extraction_ok"))
            decay_severe = sum(
                1 for o in outs
                if (o.get("decay_severity") or "") in ("severe", "broken")
            )
            # Phase 4 (2026-06-13): cross-ref audit + rigor ledgers by
            # dispatch_event_id so digest surfaces flags the cron pipeline
            # produced. Critical flags bump tone to critical.
            event_ids = {o.get("dispatch_event_id") for o in outs
                            if o.get("dispatch_event_id")}
            audit_by_ev = _load_audits_by_event_id(event_ids)
            rigor_by_ev = _load_rigor_by_event_id(event_ids)
            audit_concern  = sum(1 for a in audit_by_ev.values()
                                    if a.get("severity") == "concern")
            audit_critical = sum(1 for a in audit_by_ev.values()
                                    if a.get("severity") == "critical")
            # Flag tallies across rigor reports
            critical_flag_set = {"DEAD_POST_PUB", "SUBSUMED_BY_FF5_MOM",
                                   "DEAD_UNDER_BORROW_COST"}
            concern_flag_set  = {"DEGRADED_POST_PUB",
                                   "MARGINAL_UNDER_BORROW_COST"}
            rigor_critical = sum(
                1 for r in rigor_by_ev.values()
                if any(f in critical_flag_set for f in (r.get("flags") or []))
            )
            rigor_concern = sum(
                1 for r in rigor_by_ev.values()
                if any(f in concern_flag_set for f in (r.get("flags") or []))
            )

            tone = "info"
            if (red > 0 or decay_severe > 0 or err > 0
                    or audit_critical > 0 or rigor_critical > 0):
                tone = "warn"
            title = (
                f"Burndown {pf.stem[:10]}: {actual_k} run "
                f"(G{green}/M{margin}/R{red}, refused {refused}, "
                f"extract_fail {extract_fail})"
            )
            summary_parts = [usage_summary]
            if decay_severe:
                summary_parts.append(
                    f"{decay_severe} verdict(s) carry severe/broken decay "
                    f"per bt-flex-1 OOS triple."
                )
            if extract_fail:
                summary_parts.append(
                    f"{extract_fail} extraction failure(s) (LLM ineligibility "
                    f"or tool-call miss) — hypotheses remain queued."
                )
            # NEW: audit + rigor summaries
            if audit_by_ev:
                summary_parts.append(
                    f"External audit: {len(audit_by_ev)} run "
                    f"(critical={audit_critical}, concern={audit_concern})."
                )
            if rigor_by_ev:
                rigor_msg = (f"Post-GREEN rigor: {len(rigor_by_ev)} run "
                              f"(critical_flags={rigor_critical}, "
                              f"concerns={rigor_concern}).")
                # Add specific critical-flag callouts
                callouts = []
                for r in rigor_by_ev.values():
                    for f in (r.get("flags") or []):
                        if f in critical_flag_set:
                            callouts.append(f"{r.get('hypothesis_id','?')[:8]}:{f}")
                if callouts:
                    rigor_msg += " " + ",".join(callouts[:3])
                summary_parts.append(rigor_msg)
            summary = " ".join(summary_parts)
            href = None
            metadata = {
                "plan_path":     _safe_relpath(pf),
                "outcome_path":  _safe_relpath(outcome_path),
                "plan_id":       plan_id,
                "actual_k":      actual_k,
                "verdicts":      {"GREEN": green, "MARGINAL": margin, "RED": red},
                "refused":       refused,
                "extract_fail":  extract_fail,
                "decay_severe":  decay_severe,
                "audit_runs":    len(audit_by_ev),
                "audit_critical": audit_critical,
                "audit_concern":  audit_concern,
                "rigor_runs":     len(rigor_by_ev),
                "rigor_critical": rigor_critical,
                "rigor_concern":  rigor_concern,
            }
        else:
            # No outcomes file → dry-run / planner-only artifact
            top_families = sorted({c.get("family") for c in candidates if c.get("family")})
            title = (
                f"Burndown plan {pf.stem[:10]} (DRY-RUN, {actual_k}/{target_k}): "
                f"{', '.join(top_families[:4]) if top_families else 'no candidates'}"
            )
            summary = (
                f"{usage_summary} "
                f"Touch data/cron_burndown/_enabled to allow execution; "
                f"re-run scripts/burndown_run.py to actually dispatch."
            )
            tone = "info"
            href = None
            metadata = {
                "plan_path": _safe_relpath(pf),
                "plan_id":   plan_id,
                "actual_k":  actual_k,
                "dry_run":   bool(dry_run),
            }

        out.append({
            "id":       _stable_id("burndown", plan_id),
            "ts":       ts,
            "lane":     _LANE_ENGINE,
            "source":   "burndown_cron",
            "title":    title,
            "summary":  summary[:240],
            "tone":     tone,
            "href":     href,
            "metadata": metadata,
        })
    return out


def source_council_recent(limit: int = 5) -> list[dict[str, Any]]:
    """Council critique runs — REJECT / FAIL verdicts are the most
    inspiration-rich (they hint at directions the system already vetoed)."""
    path = _REPO_ROOT / "data" / "research" / "council_runs.jsonl"
    rows = _tail_jsonl(path, limit=50)
    if not rows:
        return []
    # Filter: most recent N with verdict != "PASS" (those are research-relevant)
    candidates = [r for r in rows if (r.get("consensus") or "").upper() in
                  ("REJECT", "FAIL", "REWORK", "BORDERLINE", "WEAK")]
    candidates = candidates[-limit:] if candidates else rows[-limit:]
    out = []
    for r in candidates:
        run_id   = r.get("run_id") or r.get("id") or "?"
        consensus = (r.get("consensus") or "?").upper()
        candidate = r.get("candidate") or r.get("hypothesis_id") or "?"
        ts = r.get("ts") or r.get("created_at") or _utc_iso()
        out.append({
            "id":       _stable_id("council", f"{run_id}"),
            "ts":       ts,
            "lane":     _LANE_DIRECTION,
            "source":   "council",
            "title":    f"Council {consensus} · {candidate}",
            "summary":  (r.get("summary") or r.get("verdict_text") or "")[:280],
            "tone":     "alert" if consensus in ("REJECT", "FAIL") else "warn",
            "href":     f"/lab/council/detail?run_id={run_id}",
            "metadata": {"run_id": run_id, "consensus": consensus, "candidate": candidate},
        })
    return out


def source_pfh_suggestions(limit: int = 5) -> list[dict[str, Any]]:
    """Recent PFH (proposed factor hypothesis) suggestions — research candidates."""
    path = _REPO_ROOT / "data" / "research" / "pfh_suggestions.jsonl"
    rows = _tail_jsonl(path, limit=30)
    if not rows:
        return []
    out = []
    for r in rows[-limit:]:
        ts = r.get("ts") or r.get("created_at") or _utc_iso()
        sid = r.get("spec_id") or r.get("id") or r.get("name") or "?"
        out.append({
            "id":       _stable_id("pfh", f"{ts}:{sid}"),
            "ts":       ts,
            "lane":     _LANE_DIRECTION,
            "source":   "pfh",
            "title":    f"PFH candidate · {sid}",
            "summary":  (r.get("rationale") or r.get("title") or r.get("description") or "")[:280],
            "tone":     "info",
            "href":     "/lab/factor-lab",
            "metadata": {"spec_id": sid, "raw": r},
        })
    return out


def source_memory_recent(limit: int = 6) -> list[dict[str, Any]]:
    """Memory entries written recently — lessons locked in this session.

    Reads file mtimes from the memory dir; the newest 6 with mtime in
    the last 24h are surfaced (longer-history memories are the BACKGROUND,
    not the inbox)."""
    if not _MEMORY_DIR.is_dir():
        return []
    cutoff = _dt.datetime.now() - _dt.timedelta(days=2)
    candidates: list[tuple[float, Path]] = []
    try:
        for p in _MEMORY_DIR.glob("*.md"):
            if p.name == "MEMORY.md":
                continue
            try:
                m = _dt.datetime.fromtimestamp(p.stat().st_mtime)
                if m >= cutoff:
                    candidates.append((p.stat().st_mtime, p))
            except Exception:
                continue
    except Exception:
        logger.exception("inbox.source_memory_recent failed listing memory dir")
        return []

    candidates.sort(reverse=True)
    out = []
    for mtime, p in candidates[:limit]:
        # Pull title from frontmatter if available
        title = p.stem.replace("_", " ")
        description = ""
        try:
            text = p.read_text(encoding="utf-8")[:2000]
            # tiny frontmatter parse — grab description: line
            for line in text.splitlines()[:20]:
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
        ts = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append({
            "id":       _stable_id("memory", p.name),
            "ts":       ts,
            "lane":     _LANE_METHODOLOGY,
            "source":   "memory",
            "title":    f"Memory · {title[:80]}",
            "summary":  description[:280] or "(no description)",
            "tone":     "info",
            "href":     None,
            "metadata": {"path": str(p), "mtime": ts},
        })
    return out


def source_capability_evidence(limit: int = 5) -> list[dict[str, Any]]:
    """Recent docs/capability_evidence/*.md files — verified findings."""
    ev_dir = _REPO_ROOT / "docs" / "capability_evidence"
    if not ev_dir.is_dir():
        return []
    cutoff = _dt.datetime.now() - _dt.timedelta(days=14)
    cands: list[tuple[float, Path]] = []
    for p in ev_dir.glob("*.md"):
        try:
            m = p.stat().st_mtime
            if _dt.datetime.fromtimestamp(m) >= cutoff:
                cands.append((m, p))
        except Exception:
            continue
    cands.sort(reverse=True)
    out = []
    for mtime, p in cands[:limit]:
        ts = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
        name = p.stem
        # PASS / RED hint from filename
        is_pass = "pass" in name.lower() or "green" in name.lower()
        is_red  = "red" in name.lower() or "fail" in name.lower()
        out.append({
            "id":       _stable_id("cap_ev", p.name),
            "ts":       ts,
            "lane":     _LANE_DIRECTION,
            "source":   "capability_evidence",
            "title":    f"Evidence · {name.replace('_', ' ')[:80]}",
            "summary":  ("PASS-class finding" if is_pass else
                          "RED-class finding" if is_red else
                          "verified evidence document"),
            "tone":     "ok" if is_pass else "alert" if is_red else "info",
            "href":     None,
            "metadata": {"path": str(p)},
        })
    return out


def source_dq_inspector() -> list[dict[str, Any]]:
    """Data-quality flags from the live DQ Inspector — show only when
    HALT or WARN. No noise on CLEAN."""
    try:
        from engine.agents.dq_inspector.gates import evaluate_pre_batch
        breaches = evaluate_pre_batch(_dt.date.today())
        if not breaches:
            return []
        hard = [b for b in breaches if getattr(b, "severity", "") == "HARD_HALT"]
        warn = [b for b in breaches if getattr(b, "severity", "") == "SOFT_WARN"]
        out = []
        for b in (hard + warn)[:5]:
            sev = getattr(b, "severity", "?")
            mode_id = getattr(b, "mode_id", "?")
            rule    = getattr(b, "rule_description", "")
            out.append({
                "id":       _stable_id("dq", f"{mode_id}:{rule[:40]}"),
                "ts":       _utc_iso(),
                "lane":     _LANE_ENGINE,
                "source":   "dq_inspector",
                "title":    f"DQ {sev} · mode {mode_id}",
                "summary":  rule[:280],
                "tone":     "alert" if sev == "HARD_HALT" else "warn",
                "href":     "/ops",
                "metadata": {"mode_id": mode_id, "severity": sev},
            })
        return out
    except Exception:
        logger.exception("inbox.source_dq_inspector failed")
        return []


# ── Public API ────────────────────────────────────────────────────


def source_papers_scored(limit: int = 10) -> list[dict[str, Any]]:
    """L2 scored papers from data/research_ops/papers_scored.jsonl.

    Filters: drop killed papers; surface non-killed papers from last 14 days.
    Routes by `lane_hint` in the LLM score (direction / methodology / graveyard).
    """
    path = _REPO_ROOT / "data" / "research_ops" / "papers_scored.jsonl"
    rows = _tail_jsonl(path, limit=200)
    if not rows:
        return []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=14)
    survivors = []
    for r in rows:
        sc = r.get("score") or {}
        if sc.get("kill"):
            continue
        try:
            ts = _dt.datetime.strptime(r.get("scored_ts", ""), "%Y-%m-%dT%H:%M:%SZ")
            if ts < cutoff:
                continue
        except Exception:
            pass
        survivors.append(r)
    # rank by score desc, prefer relevant_to_deployed
    survivors.sort(key=lambda r: (
        1 if r.get("score", {}).get("relevant_to_deployed") else 0,
        r.get("score", {}).get("score", 0),
    ), reverse=True)
    out = []
    for r in survivors[:limit]:
        sc = r.get("score") or {}
        lane_hint = sc.get("lane_hint", "direction")
        lane_map = {
            "direction":   _LANE_DIRECTION,
            "methodology": _LANE_METHODOLOGY,
            "graveyard":   _LANE_GRAVEYARD,
        }
        lane = lane_map.get(lane_hint, _LANE_DIRECTION)
        tone = "ok" if (sc.get("score", 0) >= 8) else "info" if (sc.get("score", 0) >= 5) else "muted"
        families = ", ".join({f["family"] for f in r.get("family_match", [])})
        out.append({
            "id":       r.get("id") or _stable_id("paper", r.get("title", "")),
            "ts":       r.get("scored_ts") or r.get("fetched_ts") or _utc_iso(),
            "lane":     lane,
            "source":   "paper",
            "title":    r.get("title", "")[:200],
            "summary":  sc.get("summary_one_line") or r.get("abstract", "")[:280],
            "tone":     tone,
            "href":     r.get("link") or None,
            "metadata": {
                "link":           r.get("link"),
                "abstract":       (r.get("abstract") or "")[:600],
                "score":          sc.get("score"),
                "novelty":        sc.get("novelty"),
                "relevant_to_deployed": sc.get("relevant_to_deployed"),
                "lane_hint":      lane_hint,
                "family_match":   families,
                "rss_source":     r.get("source"),
            },
        })
    return out


def source_new_literature_summary(window_days: int = 7) -> list[dict[str, Any]]:
    """Aggregated 'N new papers' notification → /lab/literature.

    2026-06-04: rewired to T7 paper registry (created_ts within window)
    after retiring the legacy engine.inbox.paper_fetcher Haiku flow.
    A "new paper" now = a row added to papers_registry.jsonl in the
    last window_days. Score thresholds map to T7 shelf assignment via
    the same table as source_papers_from_t7.
    """
    registry_path = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"
    if not registry_path.exists():
        return []

    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=window_days)
    survivors: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            continue
        ts_str = r.get("created_ts") or r.get("ingested_ts") or ""
        if not ts_str: continue
        try:
            ts = _dt.datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff: continue
        survivors.append(r)
    if not survivors:
        return []

    _SHELF_HIGH = {"doctrine_method", "green_motivation"}

    newest_ts = max(
        (r.get("created_ts") or r.get("ingested_ts") or "") for r in survivors
    )
    n_total           = len(survivors)
    n_high_relevance  = sum(1 for r in survivors
                            if any(sh in _SHELF_HIGH for sh in (r.get("shelves") or [])))
    n_relevant_deploy = n_high_relevance  # same predicate under T7 doctrine

    tone = "ok" if n_high_relevance > 0 else "info"

    parts = [f"{n_total} new paper{'s' if n_total != 1 else ''}"]
    if n_high_relevance > 0:
        parts.append(f"{n_high_relevance} doctrine/green-motivation")
    title = " · ".join(parts)

    # Highest-relevance survivor headlines the summary
    survivors.sort(key=lambda r: (
        1 if any(sh in _SHELF_HIGH for sh in (r.get("shelves") or [])) else 0,
        r.get("year") or 0,
    ), reverse=True)
    top = survivors[0]
    top_title = (top.get("title") or "").strip()[:140]
    summary = f"Latest: \"{top_title}\". Open /lab/literature to triage."

    date_bucket = newest_ts[:10] if newest_ts else _utc_iso()[:10]
    return [{
        "id":       _stable_id("literature_new", date_bucket),
        "ts":       newest_ts or _utc_iso(),
        "lane":     _LANE_DIRECTION,
        "source":   "paper",
        "title":    title,
        "summary":  summary,
        "tone":     tone,
        "href":     "/lab/literature",
        "metadata": {
            "n_total":            n_total,
            "n_high_relevance":   n_high_relevance,
            "n_relevant_deploy":  n_relevant_deploy,
            "window_days":        window_days,
            "newest_ts":          newest_ts,
        },
    }]


def source_weekly_digest() -> list[dict[str, Any]]:
    """The most recent weekly digest (one item in Methodology lane)."""
    path = _REPO_ROOT / "data" / "research_ops" / "weekly_digest.jsonl"
    rows = _tail_jsonl(path, limit=5)
    if not rows:
        return []
    latest = rows[-1]
    digest = latest.get("digest") or {}
    return [{
        "id":       latest["id"],
        "ts":       latest.get("ts") or _utc_iso(),
        "lane":     _LANE_METHODOLOGY,
        "source":   "weekly_digest",
        "title":    f"Weekly digest · {digest.get('headline','(no headline)')[:120]}",
        "summary":  (digest.get("narrative") or "")[:280],
        "tone":     "info",
        "href":     None,
        "metadata": {
            "n_papers":              latest.get("n_top_papers"),
            "paper_ids":             latest.get("paper_ids", []),
            "improvement_dirs_hit":  digest.get("improvement_directions_hit", []),
            "methodology_advances":  digest.get("methodology_advances", []),
            "graveyard_reinforcement": digest.get("graveyard_reinforcement", []),
            "full_narrative":        digest.get("narrative"),
        },
    }]


def source_graveyard_reinforcement(limit: int = 5) -> list[dict[str, Any]]:
    """Reinforcement of past RED / killed judgments.

    Reads data/research/gate_runs.jsonl + data/validation/factory_ledger.jsonl
    for recent RED verdicts. Surfaces them as 'these stayed dead' — the value
    is anti-temptation (you don't re-attempt a graveyard mechanism)."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    for fname in ("gate_runs.jsonl", "factory_ledger.jsonl"):
        p = _REPO_ROOT / "data" / "research" / fname
        if not p.is_file():
            p = _REPO_ROOT / "data" / "validation" / fname
        if not p.is_file():
            continue
        rows = _tail_jsonl(p, limit=80)
        for r in rows:
            verdict = (r.get("verdict") or r.get("status") or r.get("result") or "").upper()
            if verdict in ("RED", "FAIL", "REJECT"):
                candidates.append((fname, r))
    if not candidates:
        return []
    # Newest first by ts/created_at
    def _ts(r): return r.get("ts") or r.get("created_at") or r.get("as_of") or ""
    candidates.sort(key=lambda c: _ts(c[1]), reverse=True)
    out = []
    for fname, r in candidates[:limit]:
        ts = _ts(r) or _utc_iso()
        sid = r.get("spec_id") or r.get("candidate") or r.get("name") or "?"
        out.append({
            "id":       _stable_id("graveyard", f"{fname}:{sid}:{ts}"),
            "ts":       ts,
            "lane":     _LANE_GRAVEYARD,
            "source":   "graveyard",
            "title":    f"RED holds · {sid}",
            "summary":  (r.get("reason") or r.get("narrative") or r.get("summary") or
                          "Past RED judgment still applies — don't re-attempt this mechanism.")[:280],
            "tone":     "muted",   # anti-temptation, not alarm
            "href":     "/research",
            "metadata": {"spec_id": sid, "source_file": fname},
        })
    return out


def compose_inbox(since_iso: Optional[str] = None) -> dict[str, Any]:
    """Aggregate all sources into a single inbox payload.

    Args:
        since_iso: if provided, items with ts < since are tagged "unread"=False

    Returns:
        {
          "as_of":         iso-8601 UTC
          "doctrine":      str (the public framing)
          "n_total":       int
          "n_unread":      int (items with ts >= since_iso)
          "by_lane":       {lane: n} count map
          "items":         [...]   sorted newest first
        }
    """
    items: list[dict[str, Any]] = []
    # 2026-06-02 — inbox trimmed to PURE NOTIFICATIONS per user critique.
    # Sources that surface STATUS (deploy_age — on /ops), reference
    # material (capability_evidence — on /research, memory_recent — grep
    # target), or duplicate existing pages (pfh — on /lab/factor-lab,
    # graveyard — on /research) are NO LONGER routed through inbox.
    # Papers + weekly_digest moved to /lab/literature reading queue
    # (see compose_literature() below).
    #
    # Inclusion criteria for an inbox source:
    #   1. Discrete event (timestamped), not perpetual status
    #   2. Action-required OR time-critical info
    #   3. No better home for the user to find it
    items.extend(source_mcc_pending())             # pending two-eye decision
    items.extend(source_code_drift())              # alert: constants vs manifest
    items.extend(source_dq_inspector())            # alert: data quality
    items.extend(source_decay_alerts())            # warn: sleeve degradation
    # G.1 (2026-06-09): canonical decay_alert events from Tier C
    # decay_watch_trigger. Coexists with legacy `source_decay_alerts`
    # which reads the older decay_alerts.jsonl.
    items.extend(source_decay_alerts_canonical())
    # G.2 (2026-06-09): B-lens specification_robustness verdicts of
    # LIKELY_OVERFIT / MARGINAL_OVERFIT — overfit candidates surface
    # here so they get human review before promote.
    items.extend(source_specification_robustness_overfit())
    # G.3 (2026-06-09): anchor-spanned factors (residual α t-stat ≪
    # headline t-stat) — "textbook restatement, not novel alpha".
    items.extend(source_anchor_spanned_factors())
    # flex-3 (2026-06-10): capability-gap demand digest — refusals
    # as build-priority signal.
    items.extend(source_capability_gaps_digest())
    # burn-2 (2026-06-11): daily backlog burndown cron digest. One row
    # per recent plan run (dry-run OR executed) with verdict summary.
    items.extend(source_burndown_digest())
    # 2026-06-14 cron architecture reset: daily papers_curator ingest
    # heartbeat. Sibling to source_burndown_digest — ingest grows the
    # substrate, burndown consumes it. Together they form the daily
    # research pulse the principal scans in 30 seconds.
    items.extend(source_daily_ingest_digest())
    # belief-2 (2026-06-12): system self-calibration summary. One row
    # showing autopsy count + mean Brier + per-family hotspot + active
    # pattern flags (GREEN_OVERCONFIDENCE etc.).
    items.extend(source_belief_autopsy_digest())
    # external_audit (2026-06-13): adversarial LLM review of verdicts.
    # ONE row aggregating critical/concern flags from independent
    # non-Anthropic LLM. Silent when stub provider (default) — no noise.
    items.extend(source_external_audit_digest())
    items.extend(source_council_recent())          # research verdict events
    # Aggregated "N new papers" → /lab/literature reminder. ONE row,
    # not N. The reading queue lives at /lab/literature; this just
    # nudges the user to go look when fresh papers arrive.
    items.extend(source_new_literature_summary())

    # Tag read/unread + sort
    n_unread = 0
    for it in items:
        is_unread = True
        if since_iso:
            try:
                is_unread = it["ts"] > since_iso
            except Exception:
                pass
        it["unread"] = is_unread
        if is_unread:
            n_unread += 1

    # Dedupe by id (same source may surface the same item via two probes)
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in items:
        if it["id"] in seen_ids:
            continue
        seen_ids.add(it["id"])
        deduped.append(it)
    items = deduped

    items.sort(key=lambda x: x.get("ts", ""), reverse=True)

    by_lane: dict[str, int] = {}
    for it in items:
        by_lane[it["lane"]] = by_lane.get(it["lane"], 0) + 1

    return {
        "as_of":    _utc_iso(),
        "doctrine": DOCTRINE,
        "n_total":  len(items),
        "n_unread": n_unread,
        "by_lane":  by_lane,
        "items":    items,
    }


# ── compose_literature: papers + weekly digest reading queue ───


LITERATURE_DOCTRINE = (
    "Literature reading queue. Curated academic papers from ArXiv q-fin "
    "+ NBER WP, keyword-pre-filtered against active_deployment.yaml "
    "sleeves, then LLM-scored for relevance to our deployed mechanisms. "
    "Plus the weekly cross-paper digest. "
    "NOT trade-intel — papers that score as 'predicts X will outperform' "
    "are killed at L2.2. What survives is methodology improvement, "
    "research direction, and graveyard reinforcement."
)


def source_papers_from_t7(limit: int = 100) -> list[dict[str, Any]]:
    """Reading queue sourced from the T7 paper registry — the canonical
    PAPER → HYPOTHESIS → TEST → VERDICT chain locked 2026-06-04.

    Replaces the deprecated engine.inbox.paper_fetcher / papers_scored.jsonl
    flow. The "score" displayed in /lab/literature is derived from
    shelf assignment (doctrine_method = 10, green_motivation = 9,
    yellow_motivation = 7, red = 5, dormant_revisit = 6, other = 4),
    which is itself the human-curated relevance signal — the legacy
    Haiku numeric score was a proxy for this.

    Hypothesis density is surfaced too: a paper with 12 hypotheses
    (2 tested) is a richer reading target than one with 3.

    Latest version per DOI (older versions hidden, but parent_paper_id
    lineage stays queryable via the detail page).
    """
    registry_path   = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"
    hypotheses_path = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
    lessons_path    = _REPO_ROOT / "data" / "research_store" / "red_lessons.jsonl"

    if not registry_path.exists():
        return []

    # Read full files (small — 57 / 210 / ~50 rows today; will scale via
    # SQLite migration in a follow-up PR, see PR-A+B audit).
    rows: list[dict[str, Any]] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s: continue
        try: rows.append(json.loads(s))
        except json.JSONDecodeError: pass

    # Latest version per DOI (fall back to paper_id when DOI missing)
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        pid = r.get("paper_id")
        if not pid: continue
        key = r.get("doi") or f"_no_doi::{pid}"
        cur = latest.get(key)
        if cur is None or (r.get("version", 1) > cur.get("version", 1)):
            latest[key] = r

    # Index hypotheses by source_paper_id + tested status
    hyp_count: dict[str, int]    = {}
    hyp_tested: dict[str, int]   = {}
    tested_hyp_ids: set[str]     = set()
    if lessons_path.exists():
        for line in lessons_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s: continue
            try:
                lr = json.loads(s)
                for hid in (lr.get("tested_hypothesis_ids") or []):
                    tested_hyp_ids.add(hid)
            except json.JSONDecodeError: pass

    if hypotheses_path.exists():
        for line in hypotheses_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s: continue
            try:
                h = json.loads(s)
                pid = h.get("source_paper_id")
                if not pid: continue
                hyp_count[pid]  = hyp_count.get(pid, 0) + 1
                if h.get("hypothesis_id") in tested_hyp_ids:
                    hyp_tested[pid] = hyp_tested.get(pid, 0) + 1
            except json.JSONDecodeError: pass

    _SHELF_SCORE = {
        "doctrine_method":    10,
        "green_motivation":    9,
        "green_critique":      8,
        "yellow_motivation":   7,
        "dormant_revisit":     6,
        "red_critique":        5,
        "red_motivation":      5,
        "other":               4,
    }
    _SHELF_NOVELTY = {
        "doctrine_method":    "methodology",
        "green_motivation":   "extension",
        "green_critique":     "extension",
        "yellow_motivation":  "extension",
        "dormant_revisit":    "extension",
        "red_critique":       "refutation",
        "red_motivation":     "refutation",
        "other":              "irrelevant",
    }

    out: list[dict[str, Any]] = []
    for r in latest.values():
        pid     = r["paper_id"]
        shelves = r.get("shelves") or ["other"]
        # primary shelf = highest-scoring of the assigned shelves
        primary = max(shelves, key=lambda s: _SHELF_SCORE.get(s, 0))
        score   = _SHELF_SCORE.get(primary, 4)
        novelty = _SHELF_NOVELTY.get(primary, "irrelevant")
        nh      = hyp_count.get(pid, 0)
        nt      = hyp_tested.get(pid, 0)
        relevant_to_deployed = primary in ("doctrine_method", "green_motivation")

        # Lane routing matches the legacy contract
        lane = (_LANE_METHODOLOGY if primary == "doctrine_method"
                else _LANE_GRAVEYARD if "red" in primary
                else _LANE_DIRECTION)

        tone = ("ok" if score >= 8 else "info" if score >= 5 else "muted")

        authors = r.get("authors") or []
        author_str = (
            ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
        ) if authors else ""

        summary = (
            f"{author_str} ({r.get('year', '?')}) · {nh} hypotheses"
            + (f" ({nt} tested)" if nt > 0 else "")
            + ("" if not r.get("shelf_notes", {}).get(primary)
               else f" — {r['shelf_notes'][primary][:120]}")
        )

        # Prefer the structured detail page (T7 chain trace) over the
        # raw PDF — the whole point of consolidating /lab/literature
        # into the chain is that "Open original" stops being the only
        # action. PDF stays accessible via metadata.pdf_link.
        href = f"/research/papers/{pid}"

        out.append({
            "id":       f"t7_paper::{pid}",
            "ts":       r.get("updated_ts") or r.get("created_ts") or r.get("ingested_ts") or _utc_iso(),
            "lane":     lane,
            "source":   "paper",
            "title":    (r.get("title") or "")[:200],
            "summary":  summary,
            "tone":     tone,
            "href":     href,
            "metadata": {
                "paper_id":             pid,
                "score":                score,
                "novelty":              novelty,
                "relevant_to_deployed": relevant_to_deployed,
                "family_match":         primary,
                "shelves":              shelves,
                "shelf_notes":          r.get("shelf_notes") or {},
                "n_hypotheses":         nh,
                "n_hypotheses_tested":  nt,
                "abstract":             (r.get("abstract") or "")[:1200],
                "link":                 f"/research/papers/{pid}",
                "pdf_link":             r.get("pdf_source_url"),
                "rss_source":           "t7_registry",
                "year":                 r.get("year"),
                "venue":                r.get("venue"),
                "fulltext_status":      r.get("fulltext_status"),
                "n_chunks":             r.get("n_chunks", 0),
                "referenced_by_lessons": r.get("referenced_by_lessons") or [],
                "referenced_by_sleeves": r.get("referenced_by_sleeves") or [],
            },
        })

    # Sort: relevant_to_deployed first, then score desc, then year desc
    out.sort(key=lambda x: (
        1 if x["metadata"]["relevant_to_deployed"] else 0,
        x["metadata"]["score"],
        x["metadata"].get("year") or 0,
    ), reverse=True)
    return out[:limit]


def compose_literature(since_iso: Optional[str] = None) -> dict[str, Any]:
    """Academic reading queue: T7 paper registry + weekly digest.

    Data source updated 2026-06-04: replaces papers_scored.jsonl (the
    deprecated engine.inbox.paper_fetcher Haiku-scoring flow) with the
    locked T7 PAPER → HYPOTHESIS → TEST → VERDICT registry. See
    source_papers_from_t7 for the score-derivation rules.

    Separate from /inbox per the 2026-06-02 split: inbox = pure
    notifications, literature = reading queue for academic content.
    """
    items: list[dict[str, Any]] = []
    items.extend(source_weekly_digest())
    items.extend(source_papers_from_t7(limit=200))

    # Dedupe + sort newest first
    seen_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for it in items:
        if it["id"] in seen_ids:
            continue
        seen_ids.add(it["id"])
        deduped.append(it)

    n_unread = 0
    for it in deduped:
        is_unread = True
        if since_iso:
            try: is_unread = it["ts"] > since_iso
            except Exception: pass
        it["unread"] = is_unread
        if is_unread: n_unread += 1

    deduped.sort(key=lambda x: x.get("ts", ""), reverse=True)

    by_family: dict[str, int] = {}
    for it in deduped:
        fam = (it.get("metadata") or {}).get("family_match", "")
        if fam:
            by_family[fam] = by_family.get(fam, 0) + 1

    return {
        "as_of":     _utc_iso(),
        "doctrine":  LITERATURE_DOCTRINE,
        "n_total":   len(deduped),
        "n_unread":  n_unread,
        "by_family": by_family,
        "items":     deduped,
    }
