"""engine.agents.chief_of_staff.memo — Phase 2.0 step 14b.

ONE single Sonnet 4.6 call per weekly session that produces the
principal's 30-second-scan memo:

  - 1-line headline ("the week in 12 words")
  - 3-7 bullets (D + A + B + queue + delta vs last week)
  - 1-2 sentence "what's next" focus

NOT multi-agent. Just one model call with a strict JSON-schema tool
that emits the memo as a typed payload. Stays Pattern 5-safe.

Inputs the model sees:
  - This session's SessionResult (counts + verdicts + errors)
  - Up to last 3 prior memos (continuity / delta context)
  - Pending /approvals queue depth (B work the principal hasn't decided)

Output:
  - WeeklyMemo dataclass persisted to data/chief_of_staff/weekly_memos.jsonl
  - Referenced as artifact["memo_doc"] on the chief_of_staff_session_run
    event (so audit queries find it from the event)

Cost: ≤ $0.05/call → ~$2.60/yr at weekly cadence.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

# Top-level import so monkeypatch in tests works (same pattern as
# synthesis.py / review.py)
from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_MEMOS_PATH = _REPO_ROOT / "data" / "chief_of_staff" / "weekly_memos.jsonl"


# ────────────────────────────────────────────────────────────────────
# Output shape
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class WeeklyMemo:
    session_id:    str
    headline:      str                  # ≤ 120 chars, ONE line
    bullets:       tuple[str, ...]      # 3-7 bullets; each ≤ 240 chars
    whats_next:    str                  # 1-2 sentence forward focus
    generated_ts:  str
    model:         str

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "headline":     self.headline,
            "bullets":      list(self.bullets),
            "whats_next":   self.whats_next,
            "generated_ts": self.generated_ts,
            "model":        self.model,
        }


# ────────────────────────────────────────────────────────────────────
# System prompt — the brief for the model
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the chief_of_staff for a solo-quant AI research workbench. ONCE
per week you write the principal's "30-second scan" memo summarizing
what D (book monitor) / A (papers curator) / B (strengthener) did.

Audience: ONE busy quant who wants to absorb the week in 30 seconds.

TONE rules:
  - Institutional, terse, factual. NO marketing language. NO "exciting",
    NO "promising", NO "leveraging".
  - Cite SPECIFIC counts / IDs / families. "PROFITABILITY family had 4
    REDs" is good; "some families had RED activity" is bad.
  - If nothing happened, SAY so. Do NOT inflate. "Quiet week —
    substrate sparse, A returned empty, B queue unchanged at 0" is a
    valid memo.
  - Never recommend specific trades or sleeve sizing. The principal
    owns capital decisions.

CONTENT shape (call emit_weekly_memo with):

  headline (≤ 120 chars, ONE line):
    The single most important sentence about this week. If A persisted
    a candidate worth principal attention, lead with that. Otherwise
    lead with the most informative thing (D cluster severity, B queue
    state, substrate condition).

  bullets (3-7 entries, each ≤ 240 chars):
    Mix across D + A + B + delta vs last week (if memos in context).
    Each bullet should be self-contained — the principal may skim only
    some of them. Order by importance: actionable items first, then
    informational.

  whats_next (1-2 sentences):
    What the principal should consider doing this week. Honest about
    next steps the system is blocked on (e.g. "substrate enrichment is
    the binding constraint — consider INGESTing 2-3 papers manually").
    Do NOT manufacture work just to suggest something.

Hard rules:
  - Empty bullets array is INVALID — minimum 3.
  - If session had errors, surface them in ONE bullet (don't bury).
  - If 'last memos' shows the same headline 3 weeks in a row, call out
    the stagnation in whats_next.

Call emit_weekly_memo with your output. ALWAYS call it.
"""


# ────────────────────────────────────────────────────────────────────
# Tool schema
# ────────────────────────────────────────────────────────────────────
_TOOL_DEFINITION = {
    "name": "emit_weekly_memo",
    "description": (
        "Emit this week's chief_of_staff memo. Exactly one call. "
        "Bullets array must have 3-7 entries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline":   {"type": "string", "maxLength": 120},
            "bullets":    {
                "type":     "array",
                "items":    {"type": "string", "maxLength": 240},
                "minItems": 3,
                "maxItems": 7,
            },
            "whats_next": {"type": "string", "maxLength": 400},
        },
        "required": ["headline", "bullets", "whats_next"],
        "additionalProperties": False,
    },
}


# ────────────────────────────────────────────────────────────────────
# Input formatting
# ────────────────────────────────────────────────────────────────────
def _summarize_a(a: dict) -> str:
    snap = a.get("snapshot") or {}
    cands = a.get("candidates") or []
    parts = [
        f"snapshot: {snap.get('recent_summaries', 0)} papers, "
        f"{snap.get('deployed_sleeves', 0)} sleeves, "
        f"{snap.get('recent_events', 0)} events, "
        f"{snap.get('doctrine_snippets', 0)} doctrine",
        f"n_candidates: {a.get('n_candidates', 0)} (written: {a.get('n_written', 0)})",
    ]
    if cands:
        parts.append("candidates:")
        for c in cands[:5]:
            parts.append(
                f"  - {c.get('mechanism_family')}/{c.get('mechanism_subtype')} "
                f"({c.get('predicted_direction')}, {c.get('cochrane_frame')}): "
                f"{(c.get('claim') or '')[:120]}"
            )
    if a.get("errors"):
        parts.append(f"A errors: {a['errors'][:3]}")
    return "\n".join(parts)


def _summarize_d(d: dict) -> str:
    hits = d.get("hits") or []
    parts = [
        f"events scanned: {d.get('n_events_scanned', 0)}",
        f"hits total: {d.get('n_hits_total', 0)}, fresh: {d.get('n_hits_fresh', 0)}, "
        f"emitted: {d.get('n_emitted', 0)}",
    ]
    if hits:
        parts.append("hits (top 6):")
        for h in hits[:6]:
            tag = "FRESH" if h.get("is_fresh") else "dedup"
            m = h.get("metrics") or {}
            parts.append(
                f"  [{tag}] {h.get('rule_name')} · {h.get('family')} · "
                f"severity={h.get('severity')} · n_red={m.get('red_count', '?')}"
            )
    if d.get("errors"):
        parts.append(f"D errors: {d['errors'][:3]}")
    return "\n".join(parts)


def _summarize_b(b: dict, pending: int) -> str:
    verds = b.get("verdicts") or []
    by_type: dict[str, int] = {}
    for v in verds:
        by_type[v.get("verdict_type", "?")] = by_type.get(v.get("verdict_type", "?"), 0) + 1
    parts = [
        f"candidates queued: {b.get('n_candidates', 0)}",
        f"reviewed: {b.get('n_reviewed', 0)}, persisted: {b.get('n_persisted', 0)}",
        f"this run verdict mix: {by_type or '(none)'}",
        f"pending /approvals (cumulative): {pending}",
    ]
    if b.get("errors"):
        parts.append(f"B errors: {b['errors'][:3]}")
    return "\n".join(parts)


def _summarize_last_memos(last_memos: list[dict]) -> str:
    if not last_memos:
        return "(no prior memos yet — this is the first)"
    parts = ["last memos (newest first):"]
    for m in last_memos[:3]:
        parts.append(f"  [{m.get('session_id')}] {m.get('headline','')}")
        for b in (m.get("bullets") or [])[:3]:
            parts.append(f"     · {b}")
    return "\n".join(parts)


def _format_input(*, session_id: str, session_result: dict,
                    last_memos: list[dict], pending_b: int) -> str:
    return "\n\n".join([
        f"SESSION: {session_id}",
        f"--- D (book monitor) ---\n{_summarize_d(session_result.get('d_result') or {})}",
        f"--- A (synthesis) ---\n{_summarize_a(session_result.get('a_result') or {})}",
        f"--- B (strengthener) ---\n{_summarize_b(session_result.get('b_result') or {}, pending_b)}",
        f"--- prior context ---\n{_summarize_last_memos(last_memos)}",
        "Call emit_weekly_memo per the tool schema.",
    ])


# ────────────────────────────────────────────────────────────────────
# Memo persistence
# ────────────────────────────────────────────────────────────────────
def _load_last_memos(path: Path, n: int = 3) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # newest first
    rows.sort(key=lambda r: r.get("generated_ts", ""), reverse=True)
    return rows[:n]


def _persist_memo(memo: WeeklyMemo, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(memo.to_dict(), ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────────
# Validate the LLM-emitted dict
# ────────────────────────────────────────────────────────────────────
def _parse_memo(
    *,
    session_id: str,
    raw: dict,
    model: str,
    generated_ts: str,
) -> Optional[WeeklyMemo]:
    headline = (raw.get("headline") or "").strip()
    if not headline:
        logger.warning("memo: headline missing — dropping memo")
        return None
    # Fix (2026-06-08): previously dropped the WHOLE memo when LLM
    # emitted a headline > 120 chars. Schema declares maxLength=120
    # but the model still goes slightly over (125-140 chars typical)
    # — truncating with ellipsis is the right move; losing the whole
    # memo over a 5-char overshoot was wasting a Sonnet call.
    if len(headline) > 120:
        logger.warning("memo: headline %d chars > 120, truncating",
                        len(headline))
        headline = headline[:117].rstrip() + "…"
    bullets = raw.get("bullets") or []
    if not isinstance(bullets, list):
        logger.warning("memo: bullets not a list")
        return None
    clean_bullets = [str(b).strip() for b in bullets if str(b).strip()]
    # Also lenient on bullet count: prefer truncating to 7 over dropping
    if len(clean_bullets) > 7:
        logger.warning("memo: %d bullets > 7 cap, truncating",
                        len(clean_bullets))
        clean_bullets = clean_bullets[:7]
    if len(clean_bullets) < 3:
        logger.warning("memo: %d bullets < 3 floor — dropping memo",
                        len(clean_bullets))
        return None
    whats_next = (raw.get("whats_next") or "").strip()
    if not whats_next:
        logger.warning("memo: whats_next required — dropping memo")
        return None
    return WeeklyMemo(
        session_id   = session_id,
        headline     = headline,
        bullets      = tuple(b[:240] for b in clean_bullets),
        whats_next   = whats_next[:400],
        generated_ts = generated_ts,
        model        = model,
    )


# ────────────────────────────────────────────────────────────────────
# Public entry
# ────────────────────────────────────────────────────────────────────
def generate_memo(
    *,
    session_id:     str,
    session_result: dict,
    pending_b:      int,
    memos_path:     Optional[Path] = None,
) -> Optional[WeeklyMemo]:
    """ONE LLM call → WeeklyMemo. Persists to weekly_memos.jsonl.
    Returns None on any unrecoverable failure (caller treats as
    'memo skipped this week' — not a session-killer)."""
    memos_path   = memos_path or _DEFAULT_MEMOS_PATH
    generated_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    last_memos = _load_last_memos(memos_path, n=3)
    user_msg   = _format_input(
        session_id     = session_id,
        session_result = session_result,
        last_memos     = last_memos,
        pending_b      = pending_b,
    )

    try:
        result = llm_call(
            workload   = "chief_of_staff_memo",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "chief_of_staff_memo",
            tools      = [_TOOL_DEFINITION],
            max_tokens = 2000,
        )
    except Exception as exc:
        logger.exception("memo: llm_call raised: %s", exc)
        return None

    tc = None
    for cand in (result.tool_calls or ()):
        if cand.name == "emit_weekly_memo":
            tc = cand
            break
    if tc is None:
        logger.warning("memo: model did not call emit_weekly_memo")
        return None

    raw = tc.input
    if not isinstance(raw, dict):
        try:
            raw = json.loads(raw)
        except Exception:
            logger.warning("memo: tool input not parseable")
            return None

    memo = _parse_memo(
        session_id   = session_id,
        raw          = raw,
        model        = result.model,
        generated_ts = generated_ts,
    )
    if memo is None:
        return None

    try:
        _persist_memo(memo, path=memos_path)
    except Exception as exc:
        logger.exception("memo: persist failed: %s", exc)
        # Return memo anyway — caller can still display / log it
        return memo

    return memo
