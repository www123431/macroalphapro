"""engine/research/brainstorm/divergent_generator.py — Layer 3 of the
brainstorm architecture (Phase 2 MVP, 2026-06-14).

Single Sonnet call per (seed_pack, context) combo. Output: 3-5
structured BrainstormIdea items, each with MANDATORY Popper-
falsifier per [[project-brainstorm-architecture-2026-06-14]] audit
P1 item.

Context fed to the model:
  - 1 seed pack (principles + example_applications + thinking_templates +
    caveats; rendered from YAML)
  - Top-N lessons from lesson_distiller (Layer 1, deterministic)
  - Current book state (deployed sleeves count, recent verdicts)
  - Demand-trigger context (decay alert / empty family / RED cluster /
    weekly default) — when called from cron

Output schema (BrainstormIdea):
  - claim_one_line:     1-sentence testable claim
  - target_asset_class: e.g. "us_equity_top_3000", "fx_g10", "spx_options"
  - expected_mechanism: 1-2 sentence economic intuition
  - data_required:      list[str] — must be deployable
  - novelty_self_score: 0-1 LLM self-rating
  - falsifier:          MANDATORY — "would be falsified by observing X"
  - precedent_paper:    optional cite if any close paper
  - lessons_invoked:    list[lesson_id] explicitly used

Persisted to data/research/brainstorm_drafts.jsonl. PM reviews via
/research/brainstorm UI (Phase 4) and Promote → hypotheses.jsonl.

PHASE 2 MVP SCOPE: single provider (Sonnet). Phase 3 adds DeepSeek
+ dedup. Phase 4 adds UI. Phase 5 adds demand-driven cron.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

SEED_PACK_DIR    = _REPO_ROOT / "data" / "research" / "brainstorm_seeds"
DRAFTS_PATH      = _REPO_ROOT / "data" / "research" / "brainstorm_drafts.jsonl"


VALID_PACKS = [
    "physics_analogies", "network_theory", "behavioral_inverse",
    "alternative_data", "macro_regime_shifts",
    "cross_section_anomaly_inversion", "time_horizon_arbitrage",
]


@_dc.dataclass(frozen=True)
class BrainstormIdea:
    idea_id:               str
    session_id:            str        # group multiple ideas from one session
    source_pack:           str
    source_provider:       str        # "sonnet" / "deepseek" / etc
    claim_one_line:        str
    target_asset_class:    str
    expected_mechanism:    str
    data_required:         tuple[str, ...]
    novelty_self_score:    float      # 0-1
    falsifier:             str        # MANDATORY
    precedent_paper:       str        # optional cite (empty ok)
    lessons_invoked:       tuple[str, ...]
    generated_ts:          str
    model:                 str
    cost_usd:              float
    # P0-2 (2026-06-15) — cost-aware fields. LLM-estimated at idea time;
    # if it can't estimate, idea is rejected as not deployment-ready.
    estimated_turnover_pct_annual:    float = -1.0    # -1 = not supplied
    estimated_capacity_usd_millions:  float = -1.0
    estimated_tc_bp_per_round_trip:   float = -1.0
    # P0-1 (2026-06-15) — multi-stage revision audit trail. Survived
    # attacker round + defender response + data critic check.
    revision_log:          tuple[str, ...] = ()    # 1 line per stage with verdict


# ─── Seed pack loader ────────────────────────────────────────────────


def _load_pack(pack_name: str) -> dict:
    """Read YAML for a seed pack. Raises on missing / malformed."""
    import yaml as _pyyaml
    p = SEED_PACK_DIR / f"{pack_name}.yaml"
    if not p.is_file():
        raise FileNotFoundError(
            f"seed pack '{pack_name}' not found at {p}. "
            f"Available: {VALID_PACKS}"
        )
    d = _pyyaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(d, dict):
        raise ValueError(f"seed pack '{pack_name}' is not a valid YAML dict")
    return d


def _render_pack_block(pack: dict) -> str:
    """Render the seed pack into a prompt section."""
    lines = [
        f"SEED PACK: {pack.get('name', '?')}",
        "=" * 60,
        f"Domain: {pack.get('domain', '?')}",
        f"Goal:   {pack.get('short_description', '?').strip()}",
        "",
        "PRINCIPLES (apply these patterns to financial markets):",
    ]
    for p in pack.get("principles", []) or []:
        lines.append(f"  - [{p.get('id', '?')}] {p.get('label', '?')}")
        lines.append(f"      {p.get('description', '').strip()}")
    lines.append("")
    lines.append("EXAMPLE APPLICATIONS (real papers that did this):")
    for ex in pack.get("example_applications", []) or []:
        lines.append(f"  - {ex.get('paper', '?')}")
        lines.append(f"      angle: {ex.get('angle', '?')}")
    lines.append("")
    lines.append("THINKING TEMPLATES (use ≥1 as inspiration per idea):")
    for t in (pack.get("thinking_templates", []) or [])[:4]:
        # Compact template
        compact = " ".join(t.split())
        lines.append(f"  - {compact}")
    lines.append("")
    lines.append("PACK CAVEATS (respect these when generating):")
    for c in pack.get("caveats", []) or []:
        compact = " ".join(c.split())
        lines.append(f"  - {compact}")
    lines.append("")
    return "\n".join(lines)


# ─── Lesson + state context ──────────────────────────────────────────


def _render_lessons_block() -> str:
    """Pull top lessons from Layer 1 and render for prompt."""
    try:
        from engine.research.brainstorm.lesson_distiller import (
            load_lessons, render_for_prompt,
        )
        lessons = load_lessons()
        return render_for_prompt(lessons, limit=12)
    except Exception:
        logger.warning("lesson load failed", exc_info=True)
        return "(no lessons available — Layer 1 may not have run yet)"


def _render_regime_context_block() -> str:
    """P0-3 (2026-06-15) — current regime + recent stress as context.
    Without this, brainstorm is regime-agnostic; with this, ideas
    are timing-aware. Pulls from existing live endpoints (dq /
    decay / brief / autopsies tail). All best-effort — partial data
    is fine, we want what we have."""
    parts = ["CURRENT MARKET / BOOK REGIME (timing context for ideas)",
             "=" * 56]
    # Daily brief — regime + p_risk_on
    try:
        from api.main import _daily_brief_payload
        b = _daily_brief_payload()
        if b.get("regime"):
            parts.append(
                f"  regime: {b['regime']}  (p_risk_on={b.get('p_risk_on')}, "
                f"as_of={b.get('regime_as_of')}, {b.get('regime_days_stale')}d stale)")
    except Exception:
        pass
    # Decay sentinel — overall state + alarming sleeves count
    try:
        from engine.agents.persona.tools import read_decay_sentinel_report
        rep = json.loads(read_decay_sentinel_report())
        overall = rep.get("overall")
        n_alarms = len([a for a in (rep.get("alarms") or [])
                        if a.get("level") in ("ALERT", "WARN")])
        if overall:
            parts.append(f"  decay sentinel overall: {overall} "
                          f"({n_alarms} alarming sleeves)")
    except Exception:
        pass
    # Recent autopsy heat — were there any RED autopsies in last 30d?
    try:
        cutoff_iso = (_dt.datetime.utcnow() -
                      _dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        autopsy_path = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"
        if autopsy_path.is_file():
            n_recent_red = 0
            for ln in autopsy_path.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if (r.get("ts") or "") >= cutoff_iso and \
                   r.get("actual_verdict") == "RED" and \
                   not r.get("superseded_by"):
                    n_recent_red += 1
            parts.append(f"  RED autopsies last 30d: {n_recent_red}")
    except Exception:
        pass
    # Recent CONFIRMED_DECAY
    try:
        cutoff_iso = (_dt.datetime.utcnow() -
                      _dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        rp = _REPO_ROOT / "data" / "research" / "decay_retest_results.jsonl"
        if rp.is_file():
            n_conf = 0
            for ln in rp.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if (r.get("triggered_at") or "") >= cutoff_iso and \
                   r.get("verdict") == "CONFIRMED_DECAY":
                    n_conf += 1
            parts.append(f"  CONFIRMED_DECAY (Phase 9) last 30d: {n_conf}")
    except Exception:
        pass
    parts.append("")
    parts.append("  Use this regime context to bias ideas: e.g. high VIX "
                 "→ favor crisis-payoff hedges; sleeves in WATCH/ACTION "
                 "→ propose replacement; calm regime → propose "
                 "carry/value variants. Idea must be PLAUSIBLE in current "
                 "regime, not just textbook average regime.")
    parts.append("")
    return "\n".join(parts)


def _render_book_state_block() -> str:
    """Compact summary of current deployed book + verdict mix."""
    try:
        import yaml as _pyyaml
        lib = _REPO_ROOT / "data" / "research" / "mechanism_library"
        deployed = []
        for fp in lib.glob("*.yaml"):
            if fp.name.startswith("_"):
                continue
            try:
                d = _pyyaml.safe_load(fp.read_text(encoding="utf-8"))
                if isinstance(d, dict) and d.get("id"):
                    fam = d.get("family") or d.get("parent_family") or "?"
                    deployed.append(f"{d['id']} ({fam})")
            except Exception:
                continue
        return (
            f"CURRENT DEPLOYED BOOK ({len(deployed)} sleeves)\n"
            f"=================================\n"
            + "\n".join(f"  - {s}" for s in deployed)
            + "\n"
        )
    except Exception:
        return "(book state unavailable)"


# ─── System prompt ───────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a senior quant researcher running an INTERNAL FOUR-STAGE
multi-perspective revision in your own head before emitting the
final structured output. Single agent, role-switching internally —
NOT a debate.

THE FOUR INTERNAL STAGES (do all four before emitting)
=======================================================
Stage 1 — DRAFT
  Generate 5 raw ideas using the seed pack + lessons + regime.
Stage 2 — ATTACK (channel Asness's adversarial review style)
  For each draft idea, ask: "what's the most embarrassing way this
  fails on backtest? what HXZ-2020-style replication evidence
  would kill it? does the data carry survivorship/look-ahead?"
Stage 3 — DEFEND or REFINE (channel Fama's economic-theory rigor)
  For each idea that survived attack: refine the mechanism statement
  to be ECONOMICALLY tight (limit-to-arb / behavioral / risk-premium
  / friction). Drop ideas without a defensible mechanism.
Stage 4 — DATA CRITIC (channel Pearson's data-integrity reflex)
  Reject any idea whose data_required has known issues (survivorship,
  look-ahead, regime-incomplete coverage). Verify capacity is real.

OUTPUT
======
Emit 2-4 ideas that SURVIVED all four stages. Also acceptable: emit
0 ideas with `no_idea_rationale` if the seed pack + current regime
+ lesson constraints genuinely don't admit a novel idea right now.
Saying "no" is a SENIOR behavior — interns always have ideas.

YOU HAVE TWO INPUT CHANNELS
============================
1. A SEED PACK — principles + example papers from a non-finance
   domain (physics / network theory / behavioral / alt-data / macro /
   anomaly-inversion / time-horizon). Use ≥ 1 principle per idea.
2. LESSONS FROM OUR TEAM'S OWN EXPERIENCE — deterministic distillation
   of which families have died, which are robust, which failure modes
   recur, where we have capability gaps. RESPECT lessons explicitly:
     - DEAD_FAMILY  → do NOT propose more variants there
     - ROBUST_FAMILY → ok to propose adjacent sub-mechanisms / transfers
     - ANCHOR_SPANNING → if you propose in that family, MUST claim LOW
       residual r² vs FF5+MOM, otherwise you're proposing factor exposure
       not alpha
     - RECURRING_FAILURE_MODE → idea must address this category, e.g.
       if OVERFITTING is recurring, prefer simple signals over multi-knob
     - CAPABILITY_GAP → bonus points for filling gap families
     - PUB_YEAR_DECAY → be skeptical of post-2010 published anomalies
     - N_TRIALS_NEAR_CAP → AVOID adding variants in saturated families
     - HIGH_POTENTIAL_TRANSFER → can extend β's transfer proposals

ABSOLUTE REQUIREMENTS PER SURVIVING IDEA
=========================================
1. claim_one_line: 1 testable sentence. Not "X works"; instead
   "stocks with characteristic A and B, ranked Q5-Q1, earn α > 0
   over horizon H".
2. expected_mechanism: 1-2 sentence economic-intuition. NOT
   "because markets are inefficient" — name the SPECIFIC source
   (limit to arbitrage / behavioral bias / institutional friction /
   information asymmetry / risk premium / regime shift).
3. data_required: list of data sources. Must be ACCESSIBLE — if you
   need tick-level / Reddit-WSB / satellite, flag it explicitly.
   Default-available: CRSP, Compustat, IBES, OptionMetrics (PIT),
   Ken French factors, FRED macro, paper text.
4. **FALSIFIER (MANDATORY)**: "This hypothesis would be FALSIFIED
   by observing X." Specific: "α-t < 1.0 in last 36 months" /
   "Sharpe < 0 in 2 of 3 sub-periods" / "FF5+MOM residual r² > 0.9".
5. novelty_self_score: 0-1 LLM rating vs seed pack examples.
6. lessons_invoked: list[lesson_id] you explicitly used.
7. **COST-AWARE (P0-2 2026-06-15) — MANDATORY for senior-level**:
     - estimated_turnover_pct_annual: % portfolio turnover/yr
       (e.g. monthly rebal value sleeve ~150-300%; momentum ~600%+;
       options short-dated 1000%+)
     - estimated_capacity_usd_millions: AUM at which alpha degrades
       50%; honest 5-50M for small-cap signals, 500M+ for liquid
       large-cap, 5000M+ for index-level
     - estimated_tc_bp_per_round_trip: realistic per-round-trip TC
       (5-15bp large-cap US; 20-50bp mid-cap; 80-200bp small/EM;
        300bp+ illiquid microcap)
   These reflect the fact that a senior NEVER proposes alpha without
   thinking about capacity + costs. If you can't honestly estimate,
   DROP the idea.
8. revision_log: 1 line per stage (DRAFT / ATTACK / DEFEND / DATA-
   CRITIC) noting what specifically you concluded for THIS idea. This
   is the audit trail of multi-stage revision.

AVOID
=====
- "Long quality short junk" / "long momentum short reversal" — these
  are textbook and already in our deployed sleeves
- Anomaly stacking ("combine X, Y, Z factors") without specific
  interaction theory
- Proposing in DEAD or N_TRIALS_NEAR_CAP families (per lessons)
- Vague mechanisms ("post-earnings drift" without naming WHICH
  earnings, WHICH stocks, WHICH horizon)
- Strategy that needs data we don't have (be honest in data_required)
"""


_TOOL_SCHEMA = {
    "name": "emit_brainstorm",
    "description": "Emit structured brainstorm ideas (post 4-stage revision).",
    "input_schema": {
        "type": "object",
        "required": ["ideas", "session_rationale"],
        "properties": {
            "ideas": {
                "type": "array",
                "minItems": 0,                       # P0-2: 0 ideas allowed
                "maxItems": 4,                       # P0-1: post-revision cap
                "items": {
                    "type": "object",
                    "required": ["claim_one_line", "target_asset_class",
                                 "expected_mechanism", "data_required",
                                 "novelty_self_score", "falsifier",
                                 "lessons_invoked",
                                 # P0-2 cost-aware fields
                                 "estimated_turnover_pct_annual",
                                 "estimated_capacity_usd_millions",
                                 "estimated_tc_bp_per_round_trip",
                                 # P0-1 multi-stage audit trail
                                 "revision_log"],
                    "properties": {
                        "claim_one_line":     {"type": "string", "maxLength": 400},
                        "target_asset_class": {"type": "string", "maxLength": 100},
                        "expected_mechanism": {"type": "string", "maxLength": 500},
                        "data_required":      {"type": "array",
                                                "items": {"type": "string",
                                                          "maxLength": 100},
                                                "maxItems": 8},
                        "novelty_self_score": {"type": "number",
                                                "minimum": 0.0, "maximum": 1.0},
                        "falsifier":          {"type": "string", "maxLength": 400},
                        "precedent_paper":    {"type": "string", "maxLength": 300},
                        "lessons_invoked":    {"type": "array",
                                                "items": {"type": "string",
                                                          "maxLength": 50},
                                                "maxItems": 8},
                        # P0-2
                        "estimated_turnover_pct_annual":
                            {"type": "number", "minimum": 0, "maximum": 5000},
                        "estimated_capacity_usd_millions":
                            {"type": "number", "minimum": 0, "maximum": 100000},
                        "estimated_tc_bp_per_round_trip":
                            {"type": "number", "minimum": 0, "maximum": 1000},
                        # P0-1
                        "revision_log":       {"type": "array",
                                                "items": {"type": "string",
                                                          "maxLength": 200},
                                                "maxItems": 4},
                    },
                },
            },
            "session_rationale":   {"type": "string", "maxLength": 800},
            # P0-2: senior-style "no idea this round" path
            "no_idea_rationale":   {"type": "string", "maxLength": 600},
        },
    },
}


# ─── Main entry ──────────────────────────────────────────────────────


def brainstorm_session(
    pack_name: str,
    *,
    trigger: str = "manual",
    trigger_context: str = "",
    persist: bool = True,
) -> Optional[tuple[BrainstormIdea, ...]]:
    """Run one brainstorm session — single Sonnet call.

    Args:
      pack_name: seed pack identifier (see VALID_PACKS)
      trigger: "manual" / "decay_replacement" / "empty_family_seed" /
               "red_cluster_pivot" / "weekly_default"
      trigger_context: free-text describing why this session fired
      persist: write ideas to brainstorm_drafts.jsonl (default true)

    Returns tuple of BrainstormIdea or None on failure.
    """
    if pack_name not in VALID_PACKS:
        raise ValueError(f"unknown pack {pack_name!r}; valid: {VALID_PACKS}")

    pack = _load_pack(pack_name)
    pack_block    = _render_pack_block(pack)
    lessons_block = _render_lessons_block()
    regime_block  = _render_regime_context_block()       # P0-3
    book_block    = _render_book_state_block()

    user_msg = "\n".join([
        pack_block,
        lessons_block,
        regime_block,
        book_block,
        f"TRIGGER: {trigger}",
        f"TRIGGER CONTEXT: {trigger_context}" if trigger_context else "",
        "",
        "Now run the internal 4-stage revision (DRAFT → ATTACK → DEFEND "
        "→ DATA CRITIC) and emit 0-4 ideas via the emit_brainstorm tool. "
        "Each surviving idea MUST include: falsifier, mechanism, lessons, "
        "AND cost-aware fields (turnover/capacity/TC). If after revision "
        "no idea is genuinely worth proposing (regime + lessons constrain "
        "you), emit 0 ideas + no_idea_rationale — that's a senior call.",
    ])

    try:
        result = llm_call(
            workload   = "brainstorm_divergent",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "brainstorm_divergent",
            tools      = [_TOOL_SCHEMA],
            # P0-1: multi-stage revision needs more tokens (~3x)
            max_tokens = 6000,
            scope      = f"brainstorm/{pack_name}/{trigger}",
        )
    except Exception as exc:
        logger.warning("brainstorm: llm_call failed for pack %s: %s",
                        pack_name, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_brainstorm":
            payload = tc.input
            break
    if payload is None:
        logger.warning("brainstorm: pack %s did not call emit_brainstorm",
                        pack_name)
        return None

    session_id = str(uuid.uuid4())
    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_ideas = payload.get("ideas") or []
    # P0-2: senior may legitimately emit 0 ideas with rationale
    if not raw_ideas:
        no_idea_text = (payload.get("no_idea_rationale") or "").strip()
        logger.info("brainstorm: 0 ideas emitted (no_idea path). "
                     "Rationale: %s", no_idea_text[:200])
        # Persist a sentinel row for visibility — PM sees the "no idea
        # this round" call alongside other drafts
        if persist and no_idea_text:
            try:
                DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                with DRAFTS_PATH.open("a", encoding="utf-8") as f:
                    sentinel = {
                        "idea_id":            str(uuid.uuid4()),
                        "session_id":         session_id,
                        "source_pack":        pack_name,
                        "source_provider":    "sonnet",
                        "claim_one_line":     f"[NO IDEA THIS ROUND] {no_idea_text[:300]}",
                        "target_asset_class": "n/a",
                        "expected_mechanism": no_idea_text[:500],
                        "data_required":      [],
                        "novelty_self_score": 0.0,
                        "falsifier":          "n/a — no proposal made",
                        "precedent_paper":    "",
                        "lessons_invoked":    [],
                        "generated_ts":       now_iso,
                        "model":              result.model,
                        "cost_usd":           float(result.cost_usd),
                        "is_no_idea":         True,
                        "estimated_turnover_pct_annual":   -1.0,
                        "estimated_capacity_usd_millions": -1.0,
                        "estimated_tc_bp_per_round_trip":  -1.0,
                        "revision_log":       [],
                    }
                    f.write(json.dumps(sentinel, ensure_ascii=False) + "\n")
            except Exception:
                logger.warning("brainstorm: no-idea sentinel persist failed",
                                exc_info=True)
        return ()

    ideas: list[BrainstormIdea] = []
    per_idea_cost = float(result.cost_usd) / max(1, len(raw_ideas))
    for raw in raw_ideas:
        try:
            score = float(raw.get("novelty_self_score"))
            if not (0.0 <= score <= 1.0):
                score = 0.5
            # Mandatory falsifier check
            falsifier = str(raw.get("falsifier") or "").strip()
            if len(falsifier) < 20:
                logger.warning("brainstorm: idea dropped — falsifier missing/short")
                continue
            # P0-2: cost-aware fields mandatory (-1 means LLM omitted)
            try:
                turn = float(raw.get("estimated_turnover_pct_annual", -1.0))
                cap  = float(raw.get("estimated_capacity_usd_millions", -1.0))
                tc   = float(raw.get("estimated_tc_bp_per_round_trip", -1.0))
            except (TypeError, ValueError):
                turn = cap = tc = -1.0
            if turn < 0 or cap < 0 or tc < 0:
                logger.warning("brainstorm: idea dropped — cost-aware "
                                "fields missing (turn=%s cap=%s tc=%s)",
                                turn, cap, tc)
                continue
            data_req_raw = raw.get("data_required") or []
            data_req = tuple(str(x)[:100] for x in data_req_raw if x)
            lessons_inv_raw = raw.get("lessons_invoked") or []
            lessons_inv = tuple(str(x)[:50] for x in lessons_inv_raw if x)
            # P0-1: capture revision audit trail (4 stages)
            rev_log_raw = raw.get("revision_log") or []
            rev_log = tuple(str(x)[:200] for x in rev_log_raw if x)
            ideas.append(BrainstormIdea(
                idea_id            = str(uuid.uuid4()),
                session_id         = session_id,
                source_pack        = pack_name,
                source_provider    = "sonnet",
                claim_one_line     = str(raw.get("claim_one_line"))[:400],
                target_asset_class = str(raw.get("target_asset_class"))[:100],
                expected_mechanism = str(raw.get("expected_mechanism"))[:500],
                data_required      = data_req,
                novelty_self_score = round(score, 3),
                falsifier          = falsifier[:400],
                precedent_paper    = str(raw.get("precedent_paper") or "")[:300],
                lessons_invoked    = lessons_inv,
                generated_ts       = now_iso,
                model              = result.model,
                cost_usd           = per_idea_cost,
                estimated_turnover_pct_annual   = round(turn, 1),
                estimated_capacity_usd_millions = round(cap, 1),
                estimated_tc_bp_per_round_trip  = round(tc, 1),
                revision_log       = rev_log,
            ))
        except Exception:
            logger.warning("brainstorm: skipped malformed idea", exc_info=True)
            continue

    if persist and ideas:
        try:
            DRAFTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DRAFTS_PATH.open("a", encoding="utf-8") as f:
                for idea in ideas:
                    row = _dc.asdict(idea)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("brainstorm: persist failed", exc_info=True)

    return tuple(ideas)


# ─── Read API ────────────────────────────────────────────────────────


def list_drafts(*, limit: int = 50,
                pack: Optional[str] = None) -> list[dict]:
    if not DRAFTS_PATH.is_file():
        return []
    rows: list[dict] = []
    for ln in DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if pack and r.get("source_pack") != pack:
            continue
        rows.append(r)
    rows.sort(key=lambda r: r.get("generated_ts", ""), reverse=True)
    return rows[:limit]


# ─── CLI ─────────────────────────────────────────────────────────────


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, choices=VALID_PACKS)
    ap.add_argument("--trigger", default="manual")
    ap.add_argument("--context", default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ideas = brainstorm_session(args.pack,
                                trigger=args.trigger,
                                trigger_context=args.context)
    if ideas is None:
        print(f"[brainstorm] FAIL — no ideas generated for pack {args.pack}")
        return 1
    print(f"[brainstorm] {len(ideas)} ideas from pack {args.pack} "
          f"(session={ideas[0].session_id[:8] if ideas else '-'})")
    for i, idea in enumerate(ideas, 1):
        print()
        print(f"--- Idea {i} ---")
        print(f"  claim:        {idea.claim_one_line}")
        print(f"  target:       {idea.target_asset_class}")
        print(f"  mechanism:    {idea.expected_mechanism[:200]}")
        print(f"  data:         {list(idea.data_required)[:4]}")
        print(f"  novelty:      {idea.novelty_self_score}")
        print(f"  falsifier:    {idea.falsifier[:200]}")
        print(f"  precedent:    {idea.precedent_paper[:120] or '(none)'}")
        print(f"  lessons used: {list(idea.lessons_invoked)[:4]}")
        # P0-2 cost-aware (senior reflex)
        print(f"  COST:         turnover={idea.estimated_turnover_pct_annual}% / "
              f"capacity=${idea.estimated_capacity_usd_millions:.0f}M / "
              f"TC={idea.estimated_tc_bp_per_round_trip}bp")
        # P0-1 multi-stage revision trail
        if idea.revision_log:
            print(f"  REVISION:")
            for line in idea.revision_log:
                print(f"     · {line[:160]}")
        print(f"  cost (alloc): ${idea.cost_usd:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
