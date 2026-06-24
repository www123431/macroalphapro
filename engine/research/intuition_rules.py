"""engine/research/intuition_rules.py — load + query the Intuition
Rules Base.

Replaces fine-tuning a quant-intuition model. Each rule is an explicit,
debate-able, removable YAML entry capturing a senior-quant pattern.

Public API (designed to be a thin shim over the YAML so MCP can expose
this as a tool to L4's strategy_architect agent):

  load_rules() -> list[IntuitionRule]
  query_rules(category=..., severity=..., context_text=...) -> list[IntuitionRule]
  validate_rules_file() -> ValidationReport

Usage in L4 flow:
  Before architect proposes a candidate, MCP wraps query_rules() with
  the candidate context (proposed family, role, market, sample window)
  and surfaces matched rules as "considerations". Architect must
  acknowledge each FATAL_BLOCK rule in its proposal rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_PATH = REPO_ROOT / "data" / "research" / "intuition_rules.yaml"

Severity = Literal["FATAL_BLOCK", "HARD_WARN", "SOFT_INFO"]
Category = Literal[
    "statistical", "structural", "data_quality", "regime",
    "decay", "cross_market", "role_interpretation", "process", "evidence",
]


@dataclass(frozen=True)
class IntuitionRule:
    """One codified senior-quant pattern."""

    id: str
    category: str
    severity: str
    when: str
    then: str
    evidence_source: str
    added_date: str
    added_by: str
    removable: bool = False


@dataclass
class ValidationReport:
    """Output of validate_rules_file()."""

    n_rules: int
    n_by_category: dict[str, int]
    n_by_severity: dict[str, int]
    schema_violations: list[str] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    is_valid: bool = True


_REQUIRED_FIELDS = {
    "id", "category", "severity", "when", "then",
    "evidence_source", "added_date", "added_by",
}
_VALID_CATEGORIES = {
    "statistical", "structural", "data_quality", "regime",
    "decay", "cross_market", "role_interpretation", "process", "evidence",
}
_VALID_SEVERITIES = {"FATAL_BLOCK", "HARD_WARN", "SOFT_INFO"}


def load_rules() -> list[IntuitionRule]:
    """Load all rules from YAML. Skips invalid entries with a warning."""
    if not RULES_PATH.exists():
        logger.warning("intuition_rules.yaml not found at %s", RULES_PATH)
        return []
    raw = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8")) or {}
    rules_raw = raw.get("rules") or []
    out: list[IntuitionRule] = []
    for r in rules_raw:
        if not isinstance(r, dict):
            continue
        missing = _REQUIRED_FIELDS - set(r.keys())
        if missing:
            logger.warning("rule missing fields %s: %s", missing, r.get("id"))
            continue
        out.append(IntuitionRule(
            id=str(r["id"]),
            category=str(r["category"]),
            severity=str(r["severity"]),
            when=str(r["when"]).strip(),
            then=str(r["then"]).strip(),
            evidence_source=str(r["evidence_source"]),
            added_date=str(r["added_date"]),
            added_by=str(r["added_by"]),
            removable=bool(r.get("removable", False)),
        ))
    return out


def query_rules(
    *,
    category: Optional[str] = None,
    severity: Optional[str] = None,
    context_text: Optional[str] = None,
    rule_id: Optional[str] = None,
) -> list[IntuitionRule]:
    """Filter rules by category / severity / id, OR by free-text context.

    The context_text filter is a SIMPLE substring match — sufficient for
    L4's architect agent to surface relevant rules ("if my proposal mentions
    cosine, surface rules with 'cosine' in when/then"). Use LLM-side
    embedding search for richer matching if needed (Phase 2).
    """
    rules = load_rules()
    out = rules
    if rule_id:
        return [r for r in rules if r.id == rule_id]
    if category:
        out = [r for r in out if r.category == category]
    if severity:
        out = [r for r in out if r.severity == severity]
    if context_text:
        needle = context_text.lower()
        out = [r for r in out
               if needle in r.when.lower()
               or needle in r.then.lower()
               or needle in r.id.lower()]
    return out


def list_rule_ids() -> list[str]:
    return [r.id for r in load_rules()]


def validate_rules_file() -> ValidationReport:
    """Validate schema + check for duplicate ids."""
    rules = load_rules()
    n_cat: dict[str, int] = {}
    n_sev: dict[str, int] = {}
    seen_ids: set[str] = set()
    duplicates: list[str] = []
    violations: list[str] = []

    for r in rules:
        n_cat[r.category] = n_cat.get(r.category, 0) + 1
        n_sev[r.severity] = n_sev.get(r.severity, 0) + 1
        if r.category not in _VALID_CATEGORIES:
            violations.append(
                f"rule {r.id!r} has invalid category {r.category!r}"
            )
        if r.severity not in _VALID_SEVERITIES:
            violations.append(
                f"rule {r.id!r} has invalid severity {r.severity!r}"
            )
        if r.id in seen_ids:
            duplicates.append(r.id)
        seen_ids.add(r.id)
        # Sanity: when + then must be non-empty
        if not r.when.strip():
            violations.append(f"rule {r.id!r} has empty when")
        if not r.then.strip():
            violations.append(f"rule {r.id!r} has empty then")

    return ValidationReport(
        n_rules=len(rules),
        n_by_category=n_cat,
        n_by_severity=n_sev,
        schema_violations=violations,
        duplicate_ids=duplicates,
        is_valid=(not violations and not duplicates),
    )


def main():
    """CLI: validate + print summary. Used as pre-commit + manual audit."""
    import json
    report = validate_rules_file()
    print(json.dumps({
        "n_rules":        report.n_rules,
        "n_by_category":  report.n_by_category,
        "n_by_severity":  report.n_by_severity,
        "violations":     report.schema_violations,
        "duplicates":     report.duplicate_ids,
        "is_valid":       report.is_valid,
    }, indent=2))


if __name__ == "__main__":
    main()
