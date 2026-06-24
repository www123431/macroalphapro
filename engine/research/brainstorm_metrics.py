"""engine/research/brainstorm_metrics.py — measurement substrate (2026-06-15).

End-to-end lineage analyzer + KPI computer for the brainstorm system.
Read-time JOIN over:
  brainstorm_drafts.jsonl       → ideas (with pack / provider / lessons)
  brainstorm_decisions.jsonl    → PM promote / reject
  hypotheses.jsonl              → promoted hypotheses (extraction_method=
                                    LLM_BRAINSTORM_<pack>)
  events.jsonl (research_store) → factor_verdict_filed (final GREEN/RED)
  red_attributions.jsonl        → RED failure categorization

Computes:
  - Per-pack funnel: drafts → promoted → green
  - LLM calibration: novelty_self_score bucket → actual GREEN rate
  - Failure mode histogram (last N days)
  - Total cost spent on brainstorm (aggregate cost_usd from drafts)

Without this read-time join, the 12-commit brainstorm system has no
verification of its own output quality — Ioannidis 2005 pre-registration
discipline applied to LLM output.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

DRAFTS_PATH    = _REPO_ROOT / "data" / "research" / "brainstorm_drafts.jsonl"
DECISIONS_PATH = _REPO_ROOT / "data" / "research" / "brainstorm_decisions.jsonl"
HYP_PATH       = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
EVENTS_PATH    = _REPO_ROOT / "data" / "research_store" / "events.jsonl"


def _iter_jsonl(p: Path):
    if not p.is_file():
        return
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def _verdict_from_event(e: dict) -> str:
    """Normalize verdict string."""
    v = str(e.get("verdict") or "")
    if "." in v:
        v = v.split(".")[-1]
    return v.upper()


def _is_brainstorm_extraction(method: str) -> bool:
    return (method or "").upper().startswith("LLM_BRAINSTORM")


def _pack_from_extraction(method: str) -> Optional[str]:
    """LLM_BRAINSTORM_PHYSICS_ANALOGIES → physics_analogies"""
    if not _is_brainstorm_extraction(method):
        return None
    return method[len("LLM_BRAINSTORM_"):].lower()


def compute_metrics() -> dict:
    """Build the full measurement payload. Single read pass per file."""
    # ── Step 1: drafts indexed by idea_id + session_id ──────────────
    drafts_by_id: dict[str, dict] = {}
    pack_session_ids: dict[str, set] = defaultdict(set)
    pack_idea_count: dict[str, int] = defaultdict(int)
    pack_cost_total: dict[str, float] = defaultdict(float)
    novelty_bucket_drafts: dict[str, int] = defaultdict(int)
    novelty_bucket_promoted: dict[str, int] = defaultdict(int)
    novelty_bucket_green: dict[str, int] = defaultdict(int)
    total_drafts = 0
    total_cost = 0.0
    no_idea_count = 0
    for d in _iter_jsonl(DRAFTS_PATH):
        idx = d.get("idea_id")
        if not idx:
            continue
        drafts_by_id[idx] = d
        pack = d.get("source_pack") or "unknown"
        if d.get("is_no_idea"):
            no_idea_count += 1
            continue
        total_drafts += 1
        pack_idea_count[pack] += 1
        try:
            pack_cost_total[pack] += float(d.get("cost_usd") or 0)
            total_cost += float(d.get("cost_usd") or 0)
        except Exception:
            pass
        try:
            ns = float(d.get("novelty_self_score") or 0)
            bkt = (_novelty_bucket(ns))
            novelty_bucket_drafts[bkt] += 1
        except Exception:
            pass
        if d.get("session_id"):
            pack_session_ids[pack].add(d["session_id"])

    # ── Step 2: decisions → idea_id → decision ──────────────────────
    decision_for_idea: dict[str, dict] = {}
    for dec in _iter_jsonl(DECISIONS_PATH):
        idx = dec.get("idea_id")
        if not idx:
            continue
        prev = decision_for_idea.get(idx)
        if prev is None or (dec.get("decided_ts") or "") > (prev.get("decided_ts") or ""):
            decision_for_idea[idx] = dec

    pack_promoted: dict[str, int] = defaultdict(int)
    pack_rejected: dict[str, int] = defaultdict(int)
    promoted_to_idea: dict[str, str] = {}   # new_hypothesis_id → idea_id (reverse map)
    for idx, dec in decision_for_idea.items():
        idea = drafts_by_id.get(idx)
        if idea is None:
            continue
        pack = idea.get("source_pack") or "unknown"
        if dec.get("decision") == "promote":
            pack_promoted[pack] += 1
            try:
                ns = float(idea.get("novelty_self_score") or 0)
                novelty_bucket_promoted[_novelty_bucket(ns)] += 1
            except Exception:
                pass
            new_hyp = dec.get("new_hypothesis_id")
            if new_hyp:
                promoted_to_idea[new_hyp] = idx
        elif dec.get("decision") == "reject":
            pack_rejected[pack] += 1

    # ── Step 3: hypotheses → tag whether brainstorm-origin ──────────
    hyp_brainstorm_pack: dict[str, str] = {}
    for h in _iter_jsonl(HYP_PATH):
        hid = h.get("hypothesis_id")
        method = h.get("extraction_method") or ""
        pack = _pack_from_extraction(method)
        if hid and pack:
            hyp_brainstorm_pack[hid] = pack

    # ── Step 4: verdicts → bucket by pack via hypothesis lineage ───
    pack_green: dict[str, int] = defaultdict(int)
    pack_marginal: dict[str, int] = defaultdict(int)
    pack_red: dict[str, int] = defaultdict(int)
    pack_verdict_events: dict[str, list[str]] = defaultdict(list)
    for e in _iter_jsonl(EVENTS_PATH):
        if e.get("event_type") != "factor_verdict_filed":
            continue
        m = e.get("metrics") or {}
        # Try multiple lineage paths
        hyp_id = (m.get("source_hypothesis_id")
                  or m.get("hypothesis_id"))
        pack = hyp_brainstorm_pack.get(hyp_id or "")
        if not pack:
            continue
        v = _verdict_from_event(e)
        if v == "GREEN":
            pack_green[pack] += 1
            # Calibration: which novelty bucket did this green come from?
            idea_id = None
            if hyp_id in promoted_to_idea:
                idea_id = promoted_to_idea.get(hyp_id)
            if idea_id and idea_id in drafts_by_id:
                try:
                    ns = float(drafts_by_id[idea_id].get("novelty_self_score") or 0)
                    novelty_bucket_green[_novelty_bucket(ns)] += 1
                except Exception:
                    pass
        elif v == "MARGINAL":
            pack_marginal[pack] += 1
        elif v == "RED":
            pack_red[pack] += 1
        if e.get("event_id"):
            pack_verdict_events[pack].append(e["event_id"])

    # ── Step 5: red attribution histogram ───────────────────────────
    try:
        from engine.research.red_attribution import category_counts
        # Last 90 days
        since = (_dt.datetime.utcnow() -
                 _dt.timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%SZ")
        red_categories = category_counts(since_iso=since)
    except Exception:
        red_categories = {}

    # ── Build per-pack rows ─────────────────────────────────────────
    all_packs = set(pack_idea_count) | set(pack_promoted) | set(pack_green) | set(pack_red)
    per_pack = []
    for pack in sorted(all_packs):
        n_idea  = pack_idea_count.get(pack, 0)
        n_prom  = pack_promoted.get(pack, 0)
        n_rej   = pack_rejected.get(pack, 0)
        n_grn   = pack_green.get(pack, 0)
        n_mar   = pack_marginal.get(pack, 0)
        n_red   = pack_red.get(pack, 0)
        n_decided = n_grn + n_mar + n_red
        per_pack.append({
            "pack":              pack,
            "n_sessions":        len(pack_session_ids.get(pack) or set()),
            "n_drafts":          n_idea,
            "n_promoted":        n_prom,
            "n_rejected":        n_rej,
            "promote_rate":      round(n_prom / n_idea, 3) if n_idea else None,
            "n_verdicts":        n_decided,
            "n_green":           n_grn,
            "n_marginal":        n_mar,
            "n_red":             n_red,
            "green_rate":        round(n_grn / n_decided, 3) if n_decided else None,
            "cost_total_usd":    round(pack_cost_total.get(pack, 0), 4),
            "cost_per_promote":  round(pack_cost_total.get(pack, 0) / n_prom, 4) if n_prom else None,
        })

    # ── Calibration table ───────────────────────────────────────────
    calibration = []
    for bkt in ("0.00-0.30", "0.30-0.50", "0.50-0.70", "0.70-1.00"):
        n_draft = novelty_bucket_drafts.get(bkt, 0)
        n_prom  = novelty_bucket_promoted.get(bkt, 0)
        n_grn   = novelty_bucket_green.get(bkt, 0)
        calibration.append({
            "novelty_bucket": bkt,
            "n_drafts":        n_draft,
            "n_promoted":      n_prom,
            "n_green":         n_grn,
            "promote_rate":    round(n_prom / n_draft, 3) if n_draft else None,
            # GREEN rate AMONG promoted (most actionable)
            "green_rate_given_promoted": round(n_grn / n_prom, 3) if n_prom else None,
        })

    return {
        "summary": {
            "total_drafts":     total_drafts,
            "total_promoted":   sum(pack_promoted.values()),
            "total_rejected":   sum(pack_rejected.values()),
            "total_no_idea":    no_idea_count,
            "total_verdicts":   sum(pack_green.values()) + sum(pack_marginal.values()) + sum(pack_red.values()),
            "total_green":      sum(pack_green.values()),
            "total_marginal":   sum(pack_marginal.values()),
            "total_red":        sum(pack_red.values()),
            "total_cost_usd":   round(total_cost, 4),
        },
        "per_pack":           per_pack,
        "calibration":        calibration,
        "red_categories_90d": red_categories,
    }


def _novelty_bucket(score: float) -> str:
    if score < 0.30: return "0.00-0.30"
    if score < 0.50: return "0.30-0.50"
    if score < 0.70: return "0.50-0.70"
    return "0.70-1.00"
