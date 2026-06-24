"""engine.research.deployment_demand_emitter — close the deployment→research loop.

Why this exists
===============
The factor factory was demand-blind. burndown_ranker reads
`capability_gaps.jsonl` to boost candidates whose mechanism_family is
in the demand ledger (×1.5 multiplier on the final score). But the
ledger only accumulated TIER_3_TEMPLATE refusals — it had no awareness
of `data/portfolio/active_deployment.yaml` and its per-sleeve
`improvement_directions`.

Effect: hypotheses that would FILL a deployed sleeve's documented
improvement gap ranked the same as a random new test. Principal had
to manually scan the queue + match-to-gap each session
(empirical cost: 30+ hrs/session in the 2026-06-17 GP/A audit).

This module closes the loop. For each sleeve in the active deploy
config, parse `improvement_directions` + `research_keywords`, map to
MechanismFamily enum, emit a `capability_gaps` row tagged
`source:deployment_improvement` so the ranker auto-boosts matching
hypotheses. Re-run is idempotent (signature-keyed).

Wire-in
=======
No change to burndown_ranker required — the ledger row carries a
`family` field, and `load_demand_families()` already reads that field.
The `source` tag is for audit-trail differentiation (TIER_3_TEMPLATE
refusal vs deployment-driven demand) but the ranker treats both the
same.

Academic anchor
===============
Cochrane 2011 presidential address: the factor zoo crisis is not lack
of factors, it's lack of mechanism for filtering toward what the
portfolio actually needs. AQR-style institutional practice: deployed
state drives research priority, not vice versa.

Usage
=====
    python -m engine.research.deployment_demand_emitter --write
    python -m engine.research.deployment_demand_emitter --dry-run   # default

The script is idempotent. Re-running with --write will not duplicate
rows; rows are keyed by (sleeve, family, direction_hash) signature.
"""
from __future__ import annotations

import argparse
import dataclasses as _dc
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import yaml

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT_YAML = _REPO_ROOT / "data" / "portfolio" / "active_deployment.yaml"
DEFAULT_GAPS_PATH       = _REPO_ROOT / "data" / "research" / "capability_gaps.jsonl"


# ── Direction text → MechanismFamily rules ────────────────────────
#
# Order matters: more-specific patterns first. Each pattern is a
# regex against lowercase concatenated (direction + keywords) string.
# Multiple families per sleeve are allowed (sleeve may have multiple
# improvement axes).
#
# Adding a new family: append to the list AND update mechanism_families.py
# if the family isn't in the controlled vocab yet.
_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    # CARRY family — cross-asset carry mechanisms
    (r"\bcarry trade\b|\bfx carry\b|\bcommodity carry\b|\bg10 rate carry\b|"
     r"\bcurrency carry\b|\bconvenience yield\b|\bfutures basis\b|"
     r"\bbond carry\b|\bequity carry\b|\bkoijen carry\b|\bcross-asset carry\b", "CARRY"),
    # CROSS_ASSET_MOMENTUM family — TSMOM
    (r"\btime-series momentum\b|\btsmom\b|\btrend following\b|"
     r"\b12-1 momentum\b|\bmoskowitz ooi pedersen\b|\bcross-asset trend\b|"
     r"\bfutures momentum\b|\bfast vs slow trend\b|\btrend strength\b", "CROSS_ASSET_MOMENTUM"),
    # MOMENTUM family — equity cross-sectional momentum (specific to equity sleeves)
    (r"\bcross-sectional momentum\b|\bumd\b|\bequity momentum\b|"
     r"\bjt momentum\b|\bjegadeesh.titman\b", "MOMENTUM"),
    # LOW_VOL family — BAB / IVOL anomaly
    (r"\bbetting against beta\b|\bbab\b|\blow.vol\b|\blow.volatility\b|"
     r"\bidiosyncratic vol\b|\bivol\b|\bmin.vol\b|\bminimum variance\b", "LOW_VOL"),
    # VOL_RISK_PREMIUM family — variance swap / VIX premium / dispersion
    (r"\bvariance risk premium\b|\bvrp\b|\bvariance swap\b|"
     r"\bdispersion trade\b|\bvix premium\b", "VOL_RISK_PREMIUM"),
    # EARNINGS_DRIFT family — PEAD
    (r"\bpost-earnings announcement drift\b|\bpead\b|\bearnings surprise\b|"
     r"\bsue\b|\bearnings announcement drift\b|\bearnings momentum\b|"
     r"\bsector-neutralization\b|\babnormal sue\b", "EARNINGS_DRIFT"),
    # ANALYST_REVISION family
    (r"\banalyst revision\b|\bi/b/e/s revision\b|\bibes revision\b|"
     r"\bguidance.*revision\b|\brevision.*combination\b", "ANALYST_REVISION"),
    # PROFITABILITY family
    # NB: bare \bquality\b is a false-positive trap — it matches phrases
    # like "flight to quality" that have nothing to do with the
    # profitability factor. Require quality with a factor-y qualifier.
    (r"\bgross profitability\b|\boperating profitability\b|\brmw\b|"
     r"\bquality factor\b|\bquality minus junk\b|\bqmj\b|\bnovy.?marx\b",
     "PROFITABILITY"),
    # VALUE family
    (r"\bbook.to.market\b|\bhml\b|\bvalue.factor\b|\bfama.french value\b",
     "VALUE"),
)


@_dc.dataclass(frozen=True)
class DeploymentDemand:
    """One unit of demand emitted for an active sleeve gap."""
    sleeve_name:       str
    sleeve_role:       str
    direction_text:    str
    family:            str          # MechanismFamily.value
    matched_pattern:   str          # regex that matched (for audit)
    signature:         str          # idempotency key

    def to_capability_gap_row(self, ts_iso: str, config_id: str) -> dict:
        return {
            "ts":          ts_iso,
            "signature":   self.signature,
            "gap_class":   "DEPLOYMENT_DEMAND",
            "family":      self.family,
            "source":      "deployment_improvement",
            "sleeve":      self.sleeve_name,
            "role":        self.sleeve_role,
            "direction":   self.direction_text,
            "config_id":   config_id,
            "next_action": (f"Boost demand_score for any {self.family} "
                              f"hypothesis matching '{self.sleeve_name}' improvement "
                              f"direction: {self.direction_text}"),
            "effort":      "ranker auto-boost (no manual effort)",
        }


def _signature(sleeve: str, family: str, direction: str) -> str:
    """Stable hash for idempotency. Re-running emitter never duplicates."""
    blob = f"DEPLOYMENT_DEMAND::{sleeve}::{family}::{direction.lower().strip()}"
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"DEPLOYMENT_DEMAND::{sleeve}::{family}::{digest}"


def _direction_to_families(direction: str, sleeve_keywords: Iterable[str]
                              ) -> list[tuple[str, str]]:
    """Map a single improvement_direction text → list of (family, pattern).

    `sleeve_keywords` provides ambient context (the sleeve's research_keywords
    list). We concat them so 'G10 rate carry expansion' matches CARRY rule
    via 'carry' in the direction itself OR via 'FX carry' in the keywords.
    """
    haystack = (direction + " " + " ".join(sleeve_keywords)).lower()
    matches: list[tuple[str, str]] = []
    for pattern, fam in _FAMILY_RULES:
        if re.search(pattern, haystack):
            matches.append((fam, pattern))
    return matches


def parse_active_deployment(yaml_path: Optional[Path] = None
                              ) -> tuple[dict, list[DeploymentDemand]]:
    """Read deployment config + walk sleeves → list of DeploymentDemand."""
    path = yaml_path or DEFAULT_DEPLOYMENT_YAML
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    active_id = cfg["active_config_id"]
    active = next(c for c in cfg["configs"] if c["id"] == active_id)

    demands: list[DeploymentDemand] = []
    seen_signatures: set[str] = set()
    for sleeve in active.get("sleeves", []):
        name      = sleeve.get("name", "?")
        role      = sleeve.get("role", "?")
        directions = sleeve.get("improvement_directions") or []
        keywords  = sleeve.get("research_keywords") or []
        for direction in directions:
            for fam, pattern in _direction_to_families(direction, keywords):
                sig = _signature(name, fam, direction)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                demands.append(DeploymentDemand(
                    sleeve_name     = name,
                    sleeve_role     = role,
                    direction_text  = direction,
                    family          = fam,
                    matched_pattern = pattern[:80],
                    signature       = sig,
                ))
    return {"active_config_id": active_id}, demands


def emit_deployment_demand(
    *,
    yaml_path: Optional[Path] = None,
    gaps_path: Optional[Path] = None,
    dry_run:   bool = True,
    now:       Optional[_dt.datetime] = None,
) -> dict:
    """Emit DEPLOYMENT_DEMAND rows into capability_gaps.jsonl, idempotently.

    Returns summary dict with counts: {parsed, already_present, written}.
    """
    if now is None:
        now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    cfg_info, demands = parse_active_deployment(yaml_path)

    out_path = gaps_path or DEFAULT_GAPS_PATH
    existing_signatures: set[str] = set()
    if out_path.is_file():
        with out_path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                sig = row.get("signature")
                if sig and sig.startswith("DEPLOYMENT_DEMAND::"):
                    existing_signatures.add(sig)

    new_rows = []
    for d in demands:
        if d.signature in existing_signatures:
            continue
        new_rows.append(d.to_capability_gap_row(ts_iso, cfg_info["active_config_id"]))

    if not dry_run and new_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "active_config_id": cfg_info["active_config_id"],
        "parsed":           len(demands),
        "already_present":  len(demands) - len(new_rows),
        "written":          len(new_rows) if not dry_run else 0,
        "dry_run":          dry_run,
        "new_rows":         new_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                     help="append new DEPLOYMENT_DEMAND rows to capability_gaps")
    ap.add_argument("--dry-run", action="store_true", default=False,
                     help="print but don't write (default if neither flag given)")
    args = ap.parse_args()
    dry = not args.write

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    result = emit_deployment_demand(dry_run=dry)
    print(f"active_config_id: {result['active_config_id']}")
    print(f"parsed demands:   {result['parsed']}")
    print(f"already present:  {result['already_present']}")
    print(f"written:          {result['written']}")
    if dry:
        print(f"\n(--dry-run) New rows that WOULD be written:")
    for row in result["new_rows"]:
        print(f"  [{row['family']:<22}] {row['sleeve']:<20} → {row['direction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
