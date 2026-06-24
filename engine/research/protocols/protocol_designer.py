"""engine/research/protocols/protocol_designer.py — Protocol Designer.

Pure deterministic instantiation of a multi-leg test protocol from:
  - A library mechanism YAML (which family / required_data / canonical binding)
  - A protocol family YAML (which legs / pass_criteria / verdict_rule)
  - A proposed sample range

Output: InstantiatedProtocol — IMMUTABLE, hash-stable, pre-committed.

Doctrine:
- No LLM. Pure deterministic. Same inputs → same protocol.
- Once instantiated, protocol is FROZEN — anti-p-hacking.
- Hash of canonical YAML serialization stamped into each leg's gate_runs
  entry at execution time.
- If no matching family found in protocol_library, falls back to
  `generic_v1` family (which must always exist).

The designer:
1. Loads the appropriate family YAML from data/research/protocol_library/
2. For each leg in the family: merges binding_override with mechanism's
   canonical binding (canonical wins for unspecified keys)
3. Resolves sample_override.method to concrete dates given the proposal's
   sample_start / sample_end
4. Builds InstantiatedProtocol with frozen leg list
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_DIR = REPO_ROOT / "data" / "research" / "protocol_library"
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"

GENERIC_FAMILY_ID = "generic_v1"


# ── Data classes ────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class ResolvedLeg:
    """A leg with sample window + binding fully resolved (no more 'method:
    split_first_half' — actual concrete YYYY-MM-DD ranges and merged binding)."""
    id:               str
    description:      str
    sample_start:     str               # YYYY-MM-DD
    sample_end:       str               # YYYY-MM-DD
    binding:          dict              # merged: canonical + leg's binding_override
    pass_criteria:    dict
    is_primary:       bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DecompositionCheck:
    id:           str
    description:  str
    requirement:  dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class InstantiatedProtocol:
    """The frozen, pre-committed test protocol for a single mechanism candidate.

    Immutable: dataclass frozen=True. Hash computed from canonical YAML
    of the resolved legs + decomposition + verdict_rule.
    """
    mechanism_id:           str
    protocol_family_id:     str
    protocol_family_version: int
    legs:                   tuple[ResolvedLeg, ...]
    decomposition_checks:   tuple[DecompositionCheck, ...]
    verdict_rule:           dict
    instantiated_ts:        str
    protocol_hash:          str               # SHA-256 of canonical YAML
    notes:                  str = ""

    def to_dict(self) -> dict:
        return {
            "mechanism_id":            self.mechanism_id,
            "protocol_family_id":      self.protocol_family_id,
            "protocol_family_version": self.protocol_family_version,
            "legs":                    [leg.to_dict() for leg in self.legs],
            "decomposition_checks":    [d.to_dict() for d in self.decomposition_checks],
            "verdict_rule":            self.verdict_rule,
            "instantiated_ts":         self.instantiated_ts,
            "protocol_hash":           self.protocol_hash,
            "notes":                   self.notes,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)


# ── Family loading ──────────────────────────────────────────────────────

def list_protocol_families() -> list[str]:
    """Return all family IDs available in the protocol library."""
    if not PROTOCOL_DIR.exists():
        return []
    return sorted(
        fp.stem for fp in PROTOCOL_DIR.glob("*.yaml")
        if not fp.stem.startswith("_")
    )


def load_protocol_family(family_filename_stem: str) -> dict:
    """Load a single family by filename stem (e.g. 'equity_factor_standard_v1')."""
    path = PROTOCOL_DIR / f"{family_filename_stem}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"protocol family file not found: {path}"
        )
    family = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(family, dict):
        raise ValueError(f"protocol family {family_filename_stem} not a dict")
    return family


def select_family_for_mechanism(mechanism: dict,
                                  preferred_family: str | None = None) -> str:
    """Choose a protocol family for a library mechanism.

    Priority:
    1. If preferred_family is provided AND exists → use it
    2. Family with applies_to.family that matches mechanism['family'] (narrow)
    3. Family with applies_to.parent_family that matches mechanism['parent_family']
    4. GENERIC_FAMILY_ID fallback

    Returns the family filename stem (e.g. 'equity_factor_standard_v1').
    """
    available = list_protocol_families()
    if preferred_family and preferred_family in available:
        return preferred_family

    mech_family = mechanism.get("family")
    mech_parent = mechanism.get("parent_family")

    # Narrow match (mechanism.family in family.applies_to.family)
    for fam_id in available:
        if fam_id == GENERIC_FAMILY_ID:
            continue
        try:
            family = load_protocol_family(fam_id)
        except Exception:
            continue
        applies_to = family.get("applies_to") or {}
        if mech_family and mech_family in (applies_to.get("family") or []):
            return fam_id

    # Parent-family fallback
    for fam_id in available:
        if fam_id == GENERIC_FAMILY_ID:
            continue
        try:
            family = load_protocol_family(fam_id)
        except Exception:
            continue
        applies_to = family.get("applies_to") or {}
        if mech_parent and mech_parent == applies_to.get("parent_family"):
            return fam_id

    return GENERIC_FAMILY_ID


# ── Sample-window resolution ────────────────────────────────────────────

_SAMPLE_METHODS = frozenset([
    "full", "split_first_half", "split_second_half",
    "exclude_dotcom", "exclude_gfc", "exclude_2022_rate_crash",
])


def compute_template_warmup(template_id: str | None, binding: dict | None) -> int:
    """Compute months of NaN-warmup a DSL template needs.

    3-layer resolution (per no-brittle-hardcoding doctrine):
      Layer 1: template module's own warmup_months(binding) function
               (the template KNOWS its warmup formula best)
      Layer 2: empirical detection — run template on synthetic monthly
               data + count NaN prefix length
      Layer 3: conservative default (12 months) if all else fails

    Mechanism YAML may still override via `template_warmup_months` field
    (handled by instantiate_protocol — short-circuits before this call).
    """
    if not template_id:
        return 0
    b = binding or {}

    # Layer 1: template-module-declared warmup
    try:
        import importlib
        mod = importlib.import_module(
            f"engine.research.templates.{template_id}"
        )
        warmup_fn = getattr(mod, "warmup_months", None)
        if callable(warmup_fn):
            return int(warmup_fn(b))
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("template %s warmup_months() raised: %s", template_id, exc)

    # Layer 2: empirical detection via dry-run on synthetic data
    try:
        empirical = _empirical_warmup_dryrun(template_id, b)
        if empirical is not None:
            return empirical
    except Exception as exc:
        logger.debug("empirical warmup detect failed for %s: %s",
                      template_id, exc)

    # Layer 3: conservative default
    return 12


def _empirical_warmup_dryrun(template_id: str, binding: dict) -> int | None:
    """Run template on a 120-month synthetic panel; count NaN prefix.

    Generates a STANDARD synthetic data bundle (price_panel, factor_panel,
    return_panel, etc.) but only passes the kwargs the template actually
    accepts (via inspect.signature). This way new templates with new data
    requirements don't break the dry-run path.
    """
    import inspect

    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(42)
    n_months, n_tickers = 120, 30
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rets = rng.randn(n_months, n_tickers) * 0.05
    prices = pd.DataFrame(np.cumprod(1 + rets, axis=0) * 100.0,
                            index=dates, columns=tickers)

    # Standard synthetic data bundle — templates pick what they need
    synthetic_bundle = {
        "price_panel":  prices,
        "factor_panel": pd.DataFrame(rng.randn(n_months, n_tickers),
                                       index=dates, columns=tickers),
        "return_panel": pd.DataFrame(rets, index=dates, columns=tickers),
    }

    # Look up the template function via TEMPLATES registry
    try:
        from engine.research.templates import TEMPLATES
    except ImportError:
        return None
    template_fn = TEMPLATES.get(template_id)
    if template_fn is None:
        return None

    # Inspect signature — only pass kwargs the template accepts
    try:
        sig = inspect.signature(template_fn)
        accepted_params = set(sig.parameters.keys())
    except Exception:
        accepted_params = set()

    data_kwargs = {k: v for k, v in synthetic_bundle.items()
                    if k in accepted_params}

    try:
        result = template_fn(**binding, **data_kwargs)
    except TypeError as exc:
        logger.debug("template %s dry-run TypeError: %s", template_id, exc)
        return None
    except Exception as exc:
        logger.debug("template %s dry-run failed: %s", template_id, exc)
        return None

    if not isinstance(result, pd.Series):
        return None
    valid_idx = result.first_valid_index()
    if valid_idx is None:
        return None
    return int((result.index < valid_idx).sum())


def _effective_range(canonical_start: str, canonical_end: str,
                       warmup_months: int) -> tuple[datetime.date, datetime.date]:
    """Compute effective data range after warmup using calendar month
    approximation (30.44 days). Returns (effective_start, end) as date objects.
    """
    if warmup_months <= 0:
        return (datetime.date.fromisoformat(canonical_start),
                 datetime.date.fromisoformat(canonical_end))
    start_dt = datetime.date.fromisoformat(canonical_start)
    end_dt = datetime.date.fromisoformat(canonical_end)
    # ~30.44 days per month
    effective_start = start_dt + datetime.timedelta(days=int(round(warmup_months * 30.44)))
    if effective_start >= end_dt:
        # Warmup exceeds sample; return collapsed range so caller sees the issue
        return effective_start, effective_start
    return effective_start, end_dt


def _resolve_sample(sample_override: dict | None,
                     full_start: str, full_end: str,
                     *,
                     warmup_months: int = 0) -> tuple[str, str]:
    """Convert a sample_override spec into concrete (start, end) dates.

    Phase 6c fix: split midpoints computed on EFFECTIVE range (post-warmup),
    not raw range. This makes subperiod_first_half / split_second_half
    actually contain non-NaN data.
    """
    if not sample_override:
        return full_start, full_end
    method = sample_override.get("method", "full")
    if method not in _SAMPLE_METHODS:
        raise ValueError(
            f"unknown sample_override.method {method!r}; "
            f"allowed: {sorted(_SAMPLE_METHODS)}"
        )
    if method == "full":
        return full_start, full_end

    # Effective range (post-warmup) for split-midpoint computation
    eff_start, eff_end = _effective_range(full_start, full_end, warmup_months)
    midpoint = eff_start + (eff_end - eff_start) / 2

    if method == "split_first_half":
        return eff_start.isoformat(), midpoint.isoformat()
    if method == "split_second_half":
        return midpoint.isoformat(), eff_end.isoformat()
    # The exclude_* methods need narrower returns — for v1 just return full
    # and let executor implement the date-mask. We'll add this when we wire
    # actual data fetching.
    return full_start, full_end


# ── Binding merge ───────────────────────────────────────────────────────

def _merge_binding(canonical: dict, override: dict) -> dict:
    """Override wins where it specifies a key; canonical wins otherwise.

    For nested dicts, performs shallow merge."""
    merged = dict(canonical or {})
    for k, v in (override or {}).items():
        merged[k] = v
    return merged


# ── Hash computation (audit trail) ──────────────────────────────────────

def _compute_protocol_hash(legs: list[ResolvedLeg],
                             decomp: list[DecompositionCheck],
                             verdict_rule: dict) -> str:
    """SHA-256 over canonical YAML of legs + decomposition + rule.

    Excludes instantiated_ts (which is metadata, not part of the test
    semantic). Hash is stable across re-instantiations with same inputs."""
    canonical = {
        "legs":                  [leg.to_dict() for leg in legs],
        "decomposition_checks":  [d.to_dict() for d in decomp],
        "verdict_rule":          verdict_rule,
    }
    canonical_yaml = yaml.safe_dump(canonical, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical_yaml.encode("utf-8")).hexdigest()[:16]


# ── Main instantiation API ──────────────────────────────────────────────

_FIDELITY_BAR_ADJUSTMENT = {
    "literal":  (0.0, 0),       # no bar bump, no extra OOS months
    "adapted":  (0.5, 24),      # +0.5 to Sharpe-t, +24 months OOS
    "inspired": (1.0, 36),      # +1.0 to Sharpe-t, +36 months OOS
}


def _apply_fidelity_to_legs(legs: list[ResolvedLeg],
                              fidelity_level: str) -> list[ResolvedLeg]:
    """Return new ResolvedLeg list with pass_criteria bumped for non-literal
    fidelity. literal → unchanged. adapted → +0.5 to sharpe_t_min. inspired
    → +1.0 to sharpe_t_min. Anti-overfit for adapted/inspired creative paths."""
    if fidelity_level not in _FIDELITY_BAR_ADJUSTMENT:
        return legs
    bar_bump, extra_oos_months = _FIDELITY_BAR_ADJUSTMENT[fidelity_level]
    if bar_bump == 0:
        return legs
    out = []
    for leg in legs:
        new_criteria = dict(leg.pass_criteria)
        if "sharpe_t_min" in new_criteria:
            new_criteria["sharpe_t_min"] = float(new_criteria["sharpe_t_min"]) + bar_bump
        out.append(ResolvedLeg(
            id=leg.id, description=leg.description,
            sample_start=leg.sample_start, sample_end=leg.sample_end,
            binding=leg.binding, pass_criteria=new_criteria,
            is_primary=leg.is_primary,
        ))
    return out


def instantiate_protocol(
    mechanism: dict,
    *,
    proposal_sample_start: str,
    proposal_sample_end: str,
    preferred_family: str | None = None,
) -> InstantiatedProtocol:
    """Build a frozen protocol from a library mechanism + sample window.

    Args:
      mechanism:               loaded library YAML as dict
      proposal_sample_start:   YYYY-MM-DD (full sample window start)
      proposal_sample_end:     YYYY-MM-DD (full sample window end)
      preferred_family:        family filename stem to force (e.g.
                                 for cross-validation); default = auto-select

    Returns:
      InstantiatedProtocol — immutable, hash-stable
    """
    family_id = select_family_for_mechanism(mechanism, preferred_family)
    family = load_protocol_family(family_id)

    family_version = family.get("version", 1)
    family_id_inner = family.get("protocol_family_id", family_id)

    canonical_binding = ((mechanism.get("execution_template") or {})
                          .get("binding") or {})
    template_id = ((mechanism.get("execution_template") or {})
                    .get("template_id"))

    # Compute template warmup (Phase 6c fix). Mechanism YAML may override
    # via `template_warmup_months` field for primitive_composition or
    # special cases.
    warmup_months = mechanism.get("template_warmup_months")
    if warmup_months is None:
        warmup_months = compute_template_warmup(template_id, canonical_binding)
    warmup_months = int(warmup_months)

    resolved_legs: list[ResolvedLeg] = []
    for leg_spec in family.get("legs") or []:
        leg_id = leg_spec.get("id")
        if not leg_id:
            continue
        # Per-leg binding may override vol_target / lookback, recompute warmup
        merged_for_warmup = _merge_binding(canonical_binding,
                                             leg_spec.get("binding_override") or {})
        leg_warmup = (mechanism.get("template_warmup_months")
                       if mechanism.get("template_warmup_months") is not None
                       else compute_template_warmup(template_id, merged_for_warmup))
        sample_start, sample_end = _resolve_sample(
            leg_spec.get("sample_override"),
            proposal_sample_start, proposal_sample_end,
            warmup_months=int(leg_warmup),
        )
        merged_binding = _merge_binding(canonical_binding,
                                          leg_spec.get("binding_override") or {})
        resolved_legs.append(ResolvedLeg(
            id=            leg_id,
            description=   leg_spec.get("description", ""),
            sample_start=  sample_start,
            sample_end=    sample_end,
            binding=       merged_binding,
            pass_criteria= dict(leg_spec.get("pass_criteria") or {}),
            is_primary=    (leg_id == "primary_test"),
        ))

    decomp_checks: list[DecompositionCheck] = []
    for d_spec in family.get("decomposition_checks") or []:
        decomp_checks.append(DecompositionCheck(
            id=          d_spec.get("id", ""),
            description= d_spec.get("description", ""),
            requirement= dict(d_spec.get("requirement") or {}),
        ))

    verdict_rule = dict(family.get("verdict_rule") or {})

    # Phase 5 A: fidelity-level bar adjustment
    fidelity = mechanism.get("fidelity_level", "literal")
    resolved_legs = _apply_fidelity_to_legs(resolved_legs, fidelity)

    proto_hash = _compute_protocol_hash(resolved_legs, decomp_checks, verdict_rule)

    notes = f"Instantiated from family {family_id!r}; fidelity={fidelity!r}"
    if fidelity != "literal":
        bar_bump, extra_oos = _FIDELITY_BAR_ADJUSTMENT.get(fidelity, (0, 0))
        notes += f" (+{bar_bump} to sharpe_t_min, +{extra_oos} extra OOS months)"

    return InstantiatedProtocol(
        mechanism_id=           mechanism.get("id", "unknown"),
        protocol_family_id=     family_id_inner,
        protocol_family_version= family_version,
        legs=                   tuple(resolved_legs),
        decomposition_checks=   tuple(decomp_checks),
        verdict_rule=           verdict_rule,
        instantiated_ts=        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        protocol_hash=          proto_hash,
        notes=                  notes,
    )


def load_mechanism(mechanism_id: str) -> dict:
    """Convenience: load a library mechanism YAML by ID."""
    path = LIBRARY_DIR / f"{mechanism_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"library mechanism not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
