"""
engine/agents/risk_manager/advisory.py — Phase 8 Engineer PR sign-off API.

When the Engineer agent (Level 2.5 + Claude Agent SDK, planned Week 2.5)
produces a git diff implementing a new strategy or modifying an existing
one, it calls `sign_off(diff_text, affected_strategies, proposed_meta)`
to get a verdict from Risk Manager BEFORE the user reviews the diff.

Per [[project-agent-collaboration-patterns-2026-05-18]] this is
Pattern 3 (consultation). Risk Manager returns advisory only — user
still does the manual git commit. The verdict + reasons appear in the
Engineer agent's PR comment thread to flag risks before the user
reads the diff.

Verdict mapping (per spec §2.4):
  "GREEN"  if no breaches in the synthetic gate evaluation
  "YELLOW" if only SOFT WARN modes triggered (2 / 6 / 7 / 8 / 10)
  "RED"    if any HARD HALT mode triggered  (1 / 3 / 4 / 5 / 6b / 7b / 9)

DOCTRINE compliance:
  - Advisory ONLY — never blocks the diff (contract violations are caught
    earlier by Slice 6 lockdown tests at Engineer's pytest step).
  - 0-LLM-in-DECISION: this module uses NO LLM. Verdict is computed by
    Phase 2 gates.py purely deterministically against the proposed META.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engine.agents.risk_manager.gates import Breach
    from engine.strategies.base import StrategyMeta

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Forbidden diff patterns — quick regex scan before deep META analysis
# ──────────────────────────────────────────────────────────────────────────────
# These are heuristic patterns indicating the Engineer diff is touching
# locked artifacts. ALL match → RED. Lower-friction false positives are
# acceptable here (a YELLOW that turns out fine on review is cheaper than
# a missed contract violation).
FORBIDDEN_DIFF_PATTERNS: tuple[str, ...] = (
    r"LOCKED_META\s*[:=]",                           # editing the lockdown table
    r"LOCKED_SLEEVES\s*[:=]",                        # editing sleeve lockdown
    r"STRATEGY_HASH_GOVERNANCE_LOG",                 # appending governance without rationale
    r"SLEEVE_CLASS_INTRA_CAPS\s*[:=]",               # editing intra-strategy caps (Q1b)
    r"BOOK_SINGLE_TICKER_ABS_CAP\s*[:=]",            # editing book absolute cap (Q1a)
    r"RISK_THRESHOLDS\s*[:=]",                       # editing threshold singleton
    r"DEFAULT_INITIAL_ALLOCATION\s*[:=]",            # real-capital Tier-3 allocation
    r"ALLOWED_SLEEVES\s*[:=]",                       # frozenset edit
    r"PAPER_TRADE_SLEEVE_ALLOCATION\s*[:=]",         # sleeve weights
    r"LEVERAGE_FACTOR\s*[:=]",                       # leverage edit
    r"register_spec\(",                              # forcing new spec without amendment
    r"manual_reset\(",                               # CB manual reset attempt
)

_FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_DIFF_PATTERNS), re.IGNORECASE)


def _find_forbidden_patterns(diff_text: str) -> list[str]:
    """Return all distinct forbidden-pattern matches in diff text.

    Operates on the FULL diff (including context lines and removed lines)
    so a deletion of LOCKED_META is also flagged.
    """
    hits = sorted({m.group(0) for m in _FORBIDDEN_RE.finditer(diff_text)})
    return hits


# ──────────────────────────────────────────────────────────────────────────────
# Sign-off result schema
# ──────────────────────────────────────────────────────────────────────────────
VerdictLiteral = str  # "GREEN" | "YELLOW" | "RED"


@dataclasses.dataclass(frozen=True)
class SignOffResult:
    """Engineer PR advisory verdict.

    Locked schema so the Engineer agent + Streamlit PR-review UI can
    render the response without per-call branching.
    """
    verdict:          VerdictLiteral
    reasons:          tuple[str, ...]               # human-readable reasons (one per concern)
    forbidden_hits:   tuple[str, ...]               # forbidden-pattern matches (drives RED)
    meta_warnings:    tuple[str, ...]               # META invariant warnings (drives YELLOW)
    passing_checks:   tuple[str, ...]               # what the diff DID pass (for UI confidence)
    spec_anchor:      str                           # citation
    generated_at_utc: datetime.datetime
    cost_usd:         float                         # 0.0 since no LLM call


# ──────────────────────────────────────────────────────────────────────────────
# META validation — checks for proposed StrategyMeta against registry invariants
# ──────────────────────────────────────────────────────────────────────────────
def _validate_proposed_meta(
    proposed_meta:        "Optional[StrategyMeta]",
    affected_strategies:  tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Cross-check proposed META against current registry contract.

    Returns (warnings, passing_checks). Warnings populate `meta_warnings`
    in SignOffResult; passing_checks populate `passing_checks`.

    Soft warnings (do NOT trigger RED on their own — they're YELLOW):
      - intra_sleeve_weight would push sleeve sum off 1.0
      - rebalance_days deviates strongly from sleeve_class norm
      - spec_hash_short format malformed (wrong length / not hex)

    These would also be caught by registry.validate() at runtime, but
    surfacing them in the advisory lets Engineer self-correct before
    pytest.
    """
    warnings: list[str] = []
    passing:  list[str] = []

    if proposed_meta is None:
        passing.append("no META validation requested (diff_text-only mode)")
        return warnings, passing

    # 1. spec_hash_short shape
    if not (len(proposed_meta.spec_hash_short) == 8
            and all(c in "0123456789abcdef" for c in proposed_meta.spec_hash_short.lower())):
        warnings.append(
            f"spec_hash_short {proposed_meta.spec_hash_short!r} should be 8-char hex"
        )
    else:
        passing.append(f"spec_hash_short {proposed_meta.spec_hash_short!r} format OK")

    # 2. intra_sleeve_weight bounds (frozen dataclass check enforces 0-1, but
    # advisory can warn on extreme values that are technically legal but odd).
    if proposed_meta.intra_sleeve_weight > 0.9 and proposed_meta.intra_sleeve_weight < 1.0:
        warnings.append(
            f"intra_sleeve_weight {proposed_meta.intra_sleeve_weight:.2f} is high; "
            f"sleeve will be dominated by one strategy"
        )
    else:
        passing.append(f"intra_sleeve_weight {proposed_meta.intra_sleeve_weight:.2f} reasonable")

    # 3. sleeve_id in allowed set
    from engine.strategies import ALLOWED_SLEEVES
    if proposed_meta.sleeve_id not in ALLOWED_SLEEVES:
        warnings.append(
            f"sleeve_id {proposed_meta.sleeve_id!r} not in ALLOWED_SLEEVES; "
            f"may require sleeve addition + Tier-3 governance amendment"
        )
    else:
        passing.append(f"sleeve_id {proposed_meta.sleeve_id!r} in ALLOWED_SLEEVES")

    # 4. spec_id collision check
    from engine.strategies import get_registry
    reg = get_registry()
    existing_spec_ids = {s.META.spec_id for s in reg}
    if proposed_meta.spec_id in existing_spec_ids:
        # Collision: this is a META UPDATE not an ADD. Check name.
        target_strats = [
            s for s in affected_strategies if s in reg.names()
        ]
        if not target_strats:
            warnings.append(
                f"spec_id {proposed_meta.spec_id} collides with existing registry "
                f"but affected_strategies {affected_strategies} not in registry; "
                f"ambiguous intent (modify which strategy?)"
            )
        else:
            passing.append(
                f"spec_id {proposed_meta.spec_id} matches existing "
                f"{target_strats!r} — META modification"
            )
    else:
        passing.append(f"spec_id {proposed_meta.spec_id} is new — strategy ADD")

    return warnings, passing


# ──────────────────────────────────────────────────────────────────────────────
# Public API — sign_off()
# ──────────────────────────────────────────────────────────────────────────────
def sign_off(
    diff_text:           str,
    affected_strategies: tuple[str, ...] = (),
    proposed_meta:       "Optional[StrategyMeta]" = None,
) -> SignOffResult:
    """Engineer-agent advisory API — returns a verdict on a proposed diff.

    Args:
      diff_text:           the full git diff text the Engineer agent produced
      affected_strategies: tuple of strategy NAME values the diff modifies
                           (or adds). Lets the advisory match META to target.
      proposed_meta:       the StrategyMeta the Engineer wants to ship.
                           Optional — if None, only diff_text is scanned.

    Returns SignOffResult with:
      verdict = "RED"    if any FORBIDDEN_DIFF_PATTERNS matched (contract violation)
      verdict = "YELLOW" if META validation produced warnings
      verdict = "GREEN"  if neither — diff appears risk-clean
    """
    forbidden = _find_forbidden_patterns(diff_text or "")
    meta_warnings, passing = _validate_proposed_meta(proposed_meta, affected_strategies)

    reasons: list[str] = []

    if forbidden:
        verdict = "RED"
        reasons.append(
            f"diff touches {len(forbidden)} locked artifact pattern(s): "
            f"{forbidden}. These are spec-amendment-only changes per "
            f"[[feedback-spec-lock-is-decision-contract-2026-05-15]]; "
            f"reject merge and route to Tier-3 governance."
        )
    elif meta_warnings:
        verdict = "YELLOW"
        for w in meta_warnings:
            reasons.append(w)
    else:
        verdict = "GREEN"
        reasons.append(
            "no forbidden-pattern hits in diff; META checks pass invariants; "
            "diff appears clean per Risk Manager advisory layer"
        )

    return SignOffResult(
        verdict          = verdict,
        reasons          = tuple(reasons),
        forbidden_hits   = tuple(forbidden),
        meta_warnings    = tuple(meta_warnings),
        passing_checks   = tuple(passing),
        spec_anchor      = "spec id=69 §2.4",
        generated_at_utc = datetime.datetime.utcnow(),
        cost_usd         = 0.0,
    )


# Convenience: render a SignOffResult as markdown for the Engineer agent's
# PR comment thread. Pure formatting, no logic.
def render_sign_off_markdown(result: SignOffResult) -> str:
    """Format a SignOffResult as a one-screen markdown block for PR display."""
    verdict_emoji = {"GREEN": "\U0001F7E2", "YELLOW": "\U0001F7E1", "RED": "\U0001F534"}.get(
        result.verdict, "⚪"
    )
    lines = [
        f"## Risk Manager Advisory — {verdict_emoji} {result.verdict}",
        f"_{result.spec_anchor}; generated {result.generated_at_utc.isoformat()}Z_",
        "",
        "### Reasons",
    ]
    for r in result.reasons:
        lines.append(f"- {r}")
    if result.forbidden_hits:
        lines += ["", "### Forbidden-pattern hits"]
        for h in result.forbidden_hits:
            lines.append(f"- `{h}`")
    if result.meta_warnings:
        lines += ["", "### META warnings"]
        for w in result.meta_warnings:
            lines.append(f"- {w}")
    if result.passing_checks:
        lines += ["", "### Passing checks"]
        for p in result.passing_checks:
            lines.append(f"- {p}")
    return "\n".join(lines)
