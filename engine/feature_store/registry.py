"""engine/feature_store/registry.py — Spec discovery + loading.

A "spec" is a YAML file in data/feature_store/_specs/ describing how
to materialize one deployed factor. The schema is intentionally small
(see SPEC_SCHEMA_DOC below); richer scientific context lives in the
sibling mechanism_library YAMLs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPECS_DIR = REPO_ROOT / "data" / "feature_store" / "_specs"
COMPUTED_DIR = REPO_ROOT / "data" / "feature_store" / "_computed"


def _safe_relpath(p: Path) -> str:
    """Tolerates monkeypatched paths outside REPO_ROOT (test fixtures)."""
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


SPEC_SCHEMA_DOC = """\
Required top-level fields:
  spec_id:             unique snake_case, also used as output filename root
  version:             int, bump when materialize semantics change
  description:         1-2 sentence summary

  materialize:
    module:            Python module path (e.g. "engine.portfolio.carry_sleeve")
    function:          callable name in that module
    args:              optional dict of kwargs

  output:
    kind:              "monthly_returns" | "daily_returns" | "signal_panel"
    expected_date_range:
      start:           ISO date, output must START at or before this
      end_min:         ISO date, output must EXTEND past this
    expected_shape:
      n_rows:          [min, max] inclusive range
    sanity:
      no_nan_after_first_observation: bool
      annualized_vol_range:    [min, max]    # required for *_returns kinds
      annualized_sharpe_range: [min, max]    # required for *_returns kinds

  inputs:                list of {cache_path: "data/cache/foo.parquet"} entries
                          used for hash invalidation. Order does not matter.
  source_module_files:   list of source file paths (e.g. "engine/portfolio/carry_sleeve.py")
                          used for hash invalidation.

Optional fields:
  mechanism_library_id:  links to data/research/mechanism_library/<id>.yaml
  audit:
    added_date:          ISO date
    added_by:            username
    status:              "pending_review" | "approved"
"""


@dataclass
class FeatureSpec:
    """Parsed YAML spec — the runtime object the materializer consumes."""
    spec_id:               str
    version:               int
    description:           str
    materialize_module:    str
    materialize_function:  str
    materialize_args:      dict
    output_kind:           str
    output_start:          str
    output_end_min:        str
    output_n_rows_range:   list[int]
    sanity:                dict
    inputs:                list[str]            # cache paths
    source_module_files:   list[str]
    mechanism_library_id:  Optional[str] = None
    audit:                 dict = field(default_factory=dict)
    spec_path:             Optional[Path] = None

    @property
    def output_path(self) -> Path:
        """Materialized output WITHOUT the hash suffix (computed at materialize time)."""
        return COMPUTED_DIR / f"{self.spec_id}.v{self.version}.parquet"


def _validate_spec(raw: dict, spec_path: Path) -> None:
    """Validate schema. Raises ValueError with a pointed message on violations."""
    required_top = {"spec_id", "version", "description",
                    "materialize", "output", "inputs", "source_module_files"}
    missing = required_top - set(raw.keys())
    if missing:
        raise ValueError(
            f"spec {spec_path.name}: missing top-level fields {sorted(missing)}"
        )

    mat = raw["materialize"]
    for k in ("module", "function"):
        if k not in mat:
            raise ValueError(
                f"spec {spec_path.name}: materialize.{k} required"
            )

    out = raw["output"]
    for k in ("kind", "expected_date_range", "expected_shape", "sanity"):
        if k not in out:
            raise ValueError(
                f"spec {spec_path.name}: output.{k} required"
            )

    valid_kinds = {"monthly_returns", "daily_returns", "signal_panel"}
    if out["kind"] not in valid_kinds:
        raise ValueError(
            f"spec {spec_path.name}: output.kind must be one of "
            f"{sorted(valid_kinds)}; got {out['kind']!r}"
        )

    rng = out["expected_date_range"]
    for k in ("start", "end_min"):
        if k not in rng:
            raise ValueError(
                f"spec {spec_path.name}: output.expected_date_range.{k} required"
            )

    shape = out["expected_shape"]
    if "n_rows" not in shape or not isinstance(shape["n_rows"], list) \
            or len(shape["n_rows"]) != 2:
        raise ValueError(
            f"spec {spec_path.name}: output.expected_shape.n_rows must be [min,max]"
        )

    sanity = out["sanity"]
    if out["kind"] in ("monthly_returns", "daily_returns"):
        for k in ("annualized_vol_range", "annualized_sharpe_range"):
            if k not in sanity:
                raise ValueError(
                    f"spec {spec_path.name}: returns kinds require sanity.{k}"
                )

    if not isinstance(raw["inputs"], list):
        raise ValueError(
            f"spec {spec_path.name}: inputs must be a list"
        )


def load_spec(spec_path: Path | str) -> FeatureSpec:
    """Load + validate one spec by path. Raises on schema violation."""
    p = Path(spec_path) if not isinstance(spec_path, Path) else spec_path
    if not p.is_file():
        raise FileNotFoundError(f"spec not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    _validate_spec(raw, p)

    out = raw["output"]
    return FeatureSpec(
        spec_id=str(raw["spec_id"]),
        version=int(raw["version"]),
        description=str(raw["description"]),
        materialize_module=str(raw["materialize"]["module"]),
        materialize_function=str(raw["materialize"]["function"]),
        materialize_args=dict(raw["materialize"].get("args") or {}),
        output_kind=str(out["kind"]),
        output_start=str(out["expected_date_range"]["start"]),
        output_end_min=str(out["expected_date_range"]["end_min"]),
        output_n_rows_range=list(out["expected_shape"]["n_rows"]),
        sanity=dict(out["sanity"]),
        inputs=[str(i["cache_path"]) for i in raw["inputs"]],
        source_module_files=list(raw["source_module_files"]),
        mechanism_library_id=raw.get("mechanism_library_id"),
        audit=dict(raw.get("audit") or {}),
        spec_path=p,
    )


def list_specs() -> list[dict]:
    """Discover all specs in SPECS_DIR. Returns summary dicts (not full objects)
    suitable for a listing UI / CLI.

    Auto-routes between v1 (function-wrapper) and v2 (compose) spec formats
    by detecting the `compose:` top-level key.
    """
    if not SPECS_DIR.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(SPECS_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            is_compose = "compose" in raw
            if is_compose:
                from engine.feature_store.composer import load_compose_spec
                spec = load_compose_spec(p)
                summary = {
                    "spec_id":               spec.spec_id,
                    "version":               spec.version,
                    "spec_kind":             "compose",
                    "description":           spec.description,
                    "output_kind":           spec.output_kind,
                    "mechanism_library_id":  spec.mechanism_library_id,
                    "audit_status":          spec.audit.get("status", "unknown"),
                    "spec_path":             _safe_relpath(p),
                    "axes": {
                        "universe":  spec.universe.name,
                        "signal":    spec.signal.name,
                        "weighting": spec.weighting.name,
                        "rebalance": spec.rebalance,
                    },
                }
            else:
                spec = load_spec(p)
                summary = {
                    "spec_id":               spec.spec_id,
                    "version":               spec.version,
                    "spec_kind":             "function",
                    "description":           spec.description,
                    "output_kind":           spec.output_kind,
                    "mechanism_library_id":  spec.mechanism_library_id,
                    "audit_status":          spec.audit.get("status", "unknown"),
                    "spec_path":             _safe_relpath(p),
                }
            out.append(summary)
        except Exception as exc:
            logger.warning("spec %s failed to load: %s", p.name, exc)
            out.append({
                "spec_id":   p.stem,
                "error":     str(exc)[:200],
                "spec_path": _safe_relpath(p),
            })
    return out
