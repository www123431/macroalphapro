"""engine.research.belief_synthesis_context — Phase B (2026-06-14).

Build belief-layer summary for the papers_curator synthesis prompt.
Maps the autopsy ledger into a compact per-family verdict-distribution
view that Sonnet can use to:
  - Avoid mining families that have shown nothing (e.g. CROSS_SEC_UNKNOWN
    13/13 RED → "avoid standard cross-sec equity factor proposals")
  - Explore neighborhoods of robust families (e.g. VRP 8/8 GREEN →
    "propose VRP variants but avoid same-paper dups")
  - Lean into known orthogonal-alpha factors (e.g. SPANNING_MOM 4G/1M
    → "MOM has independent alpha vs FF5; propose MOM × X combos")
  - Skip dead orthogonality claims (e.g. SPANNING_RMW 0/0/5 → "RMW
    subsumes other profitability; don't propose RMW-replacement")

Public API
==========
build_belief_summary(min_obs_per_family: int = 3) -> tuple[FamilyBelief, ...]
  Returns sorted summary tuples (most-observed first), filtered to
  families with at least min_obs_per_family non-superseded autopsies.
"""
from __future__ import annotations

import dataclasses as _dc
import json
from collections import Counter
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUTOPSIES_PATH = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"


@_dc.dataclass(frozen=True)
class FamilyBelief:
    """One family's verdict distribution + directional hint."""
    family:        str
    n_obs:         int
    n_green:       int
    n_marginal:    int
    n_red:         int
    direction_hint: str   # "explore" / "marginal-only" / "avoid" / "thin"


def _classify_direction(n_green: int, n_marginal: int, n_red: int) -> str:
    """Map verdict distribution → 1-line directional hint for synthesis."""
    n = n_green + n_marginal + n_red
    if n < 3:
        return "thin (insufficient evidence)"
    p_green = n_green / n
    p_red = n_red / n
    if p_green >= 0.5:
        return "EXPLORE neighborhood (robust GREEN signal)"
    if p_red >= 0.8:
        return "AVOID (mostly RED — post-pub decayed or never worked)"
    if (n_marginal / n) >= 0.5:
        return "MARGINAL-ONLY (orthogonal alpha exists but weakened)"
    if p_green >= 0.2:
        return "MIXED (some GREEN; investigate specific subset)"
    return "WEAK (mostly RED with rare GREEN)"


def build_belief_summary(
    *,
    min_obs_per_family: int = 3,
    autopsies_path: Optional[Path] = None,
) -> tuple[FamilyBelief, ...]:
    """Read autopsies.jsonl, aggregate per family (non-superseded),
    return tuple of FamilyBelief sorted by n_obs descending.
    """
    p = autopsies_path or _AUTOPSIES_PATH
    if not p.is_file():
        return ()
    counts: dict[str, dict[str, int]] = {}
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            if row.get("superseded_by"):
                continue
            fam = row.get("strategy_family") or "UNKNOWN"
            v = row.get("actual_verdict") or "?"
            d = counts.setdefault(fam, {"GREEN": 0, "MARGINAL": 0, "RED": 0})
            if v in d:
                d[v] += 1
    except Exception:
        return ()

    out: list[FamilyBelief] = []
    for fam, d in counts.items():
        n = d["GREEN"] + d["MARGINAL"] + d["RED"]
        if n < min_obs_per_family:
            continue
        out.append(FamilyBelief(
            family        = fam,
            n_obs         = n,
            n_green       = d["GREEN"],
            n_marginal    = d["MARGINAL"],
            n_red         = d["RED"],
            direction_hint = _classify_direction(d["GREEN"], d["MARGINAL"], d["RED"]),
        ))
    out.sort(key=lambda fb: -fb.n_obs)
    return tuple(out)


def render_for_prompt(beliefs: tuple[FamilyBelief, ...]) -> list[str]:
    """Render belief summary as prompt-ready lines.
    Returns a list of strings (caller joins with newline)."""
    if not beliefs:
        return []
    lines = [
        "SYSTEM VERDICT HISTORY (belief layer, from autopsy ledger)",
        "-" * 40,
        "Each family below is what the system has TRIED so far. Use this to",
        "AVOID re-proposing in DEAD families and to EXPLORE neighborhoods",
        "of families with robust GREEN signal. Sub-period dups inflate n_obs;",
        "interpret with that caveat.",
        "",
    ]
    for b in beliefs:
        lines.append(
            f"  {b.family:30s}  n={b.n_obs:3d}  "
            f"G={b.n_green} M={b.n_marginal} R={b.n_red}  "
            f"→ {b.direction_hint}"
        )
    lines.append("")
    return lines
