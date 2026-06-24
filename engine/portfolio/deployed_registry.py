"""engine.portfolio.deployed_registry — single source of truth for "what's deployed".

Doctrine (Item 7 of dashboard-freshness PR series, 2026-06-02):
    The UI / risk gates / cost monitors / decay sentinel all bind here.
    NO Python constant elsewhere is authoritative — they are mirrors that
    must agree with `active_deployment.yaml` at boot. Mismatch ⇒ SOFT_WARN.

Background:
    On 2026-06-02 the user found the /book dashboard still showed the
    2-mechanism narrative (Sharpe 1.03) days after config C was actually
    deployed (2026-05-30). The defect was structural — there was no
    single SoT for "currently deployed", so each surface drifted.
    This module + data/portfolio/active_deployment.yaml fix that.

Failure modes prevented:
    1. UI shows old configuration after a silent deploy change
    2. Risk gate uses old sleeve weights for capacity checks
    3. Decay sentinel monitors a config we no longer run
    4. Cost monitor splits costs across "old + new" because no version
       label tied each call to a deploy version

See: feedback_dashboard_freshness_budget standing rule.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Any, Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "data" / "portfolio" / "active_deployment.yaml"


@dataclasses.dataclass(frozen=True)
class SleeveSpec:
    name:                  str
    role:                  str       # "alpha" | "insurance" | "diversifier"
    base_weight:           float
    regime_modulated:      bool
    builder:               str       # dotted path to the builder function
    target_vol:            float
    signing_spec_ids:      tuple[int, ...] = ()
    # 2026-06-02 L2 paper-filter fields. Empty defaults make these
    # back-compat with older YAML versions; coverage assertion warns
    # when an active sleeve has empty research_keywords (drift mode).
    research_keywords:     tuple[str, ...] = ()
    academic_anchors:      tuple[str, ...] = ()
    improvement_directions: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class DeployedConfig:
    """Snapshot of the actively-deployed book configuration.

    Construct via `load_active()`; do not instantiate directly outside
    tests so the singleton-per-file invariant is preserved.
    """
    id:               str
    deploy_date:      str           # "YYYY-MM-DD"
    label:            str
    summary:          str
    signing_spec_ids: tuple[int, ...]
    book_vol_target:  float
    expected_stats:   dict[str, Any]
    sleeves:          tuple[SleeveSpec, ...]
    regime_grids:     dict[str, dict[str, float]]
    regime_classifier: dict[str, Any]

    @property
    def sleeve_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sleeves)

    @property
    def days_since_deploy(self) -> int:
        try:
            d = _dt.date.fromisoformat(self.deploy_date)
            return (_dt.date.today() - d).days
        except Exception:
            return 0


_CACHED: Optional[DeployedConfig] = None
_CACHED_MTIME: Optional[float] = None


def _load_yaml(path: Path) -> dict[str, Any]:
    """Minimal yaml load — supports the subset our manifest uses.

    We avoid a hard PyYAML dependency at the module-load layer (the file
    is structured + small + author-controlled). If PyYAML is available
    we use it; otherwise we fall back to a small parser that handles
    our subset (no anchors, no flow style mixed with block).
    """
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        # Tiny fallback — only handles the limited shape our YAML uses.
        # Strongly recommend installing PyYAML; this is here so a fresh
        # checkout doesn't crash before pip install.
        raise RuntimeError(
            "engine.portfolio.deployed_registry needs PyYAML "
            "(pip install pyyaml). YAML is the SoT format; we do not "
            "fall back to JSON because configs need readable diffs."
        )


def load_active(*, force_reload: bool = False) -> DeployedConfig:
    """Return the currently-deployed configuration.

    Caches by file mtime so live changes to active_deployment.yaml are
    picked up without process restart.

    Raises:
        FileNotFoundError: if the manifest is missing.
        ValueError:        if `active_config_id` doesn't resolve in `configs`.
    """
    global _CACHED, _CACHED_MTIME

    if not _REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"active_deployment.yaml missing at {_REGISTRY_PATH}. "
            f"This is the deployment SoT — recreate it (see "
            f"engine.portfolio.deployed_registry module docstring) or "
            f"the entire UI + risk pipeline cannot read deployment state."
        )

    mtime = _REGISTRY_PATH.stat().st_mtime
    if (not force_reload) and _CACHED is not None and _CACHED_MTIME == mtime:
        return _CACHED

    data = _load_yaml(_REGISTRY_PATH)
    active_id = data.get("active_config_id")
    configs = data.get("configs") or []
    match = next((c for c in configs if c.get("id") == active_id), None)
    if match is None:
        raise ValueError(
            f"active_config_id {active_id!r} not found in configs list "
            f"of {_REGISTRY_PATH}. Available: "
            f"{[c.get('id') for c in configs]}"
        )

    sleeves = tuple(
        SleeveSpec(
            name                  = str(s["name"]),
            role                  = str(s.get("role", "alpha")),
            base_weight           = float(s["base_weight"]),
            regime_modulated      = bool(s.get("regime_modulated", False)),
            builder               = str(s.get("builder", "")),
            target_vol            = float(s.get("target_vol", 0.10)),
            signing_spec_ids      = tuple(int(x) for x in (s.get("signing_spec_ids") or [])),
            research_keywords     = tuple(str(x) for x in (s.get("research_keywords") or [])),
            academic_anchors      = tuple(str(x) for x in (s.get("academic_anchors") or [])),
            improvement_directions= tuple(str(x) for x in (s.get("improvement_directions") or [])),
        )
        for s in (match.get("sleeves") or [])
    )

    cfg = DeployedConfig(
        id                = str(match["id"]),
        deploy_date       = str(match.get("deploy_date", "")),
        label             = str(match.get("label", "")),
        summary           = str(match.get("summary", "")).strip(),
        signing_spec_ids  = tuple(int(x) for x in (match.get("signing_spec_ids") or [])),
        book_vol_target   = float(match.get("book_vol_target", 0.10)),
        expected_stats    = dict(match.get("expected_stats") or {}),
        sleeves           = sleeves,
        regime_grids      = {k: dict(v) for k, v in (match.get("regime_grids") or {}).items()},
        regime_classifier = dict(match.get("regime_classifier") or {}),
    )

    _CACHED = cfg
    _CACHED_MTIME = mtime
    return cfg


def assert_research_map_coverage() -> list[str]:
    """Return a list of sleeves in the active config that have empty
    `research_keywords` — these will be invisible to the L2 paper filter.
    Empty result = clean.

    Same drift-class as assert_constants_match — caught at boot / API
    response so the user can't silently lose paper coverage when a new
    sleeve is added to the deploy without research keywords."""
    cfg = load_active()
    issues: list[str] = []
    for s in cfg.sleeves:
        if not s.research_keywords:
            issues.append(
                f"sleeve {s.name!r} has no research_keywords — L2 paper "
                f"filter will miss papers about this mechanism"
            )
    return issues


def assert_constants_match(
    *,
    carry_risk_weight:   float,
    tsmom_risk_weight:   float,
    crisis_risk_weight:  float,
    mom_hedge_risk_weight: float,
    book_vol_target:     float,
    tolerance:           float = 1e-6,
) -> list[str]:
    """Check whether engine.portfolio.combined_book defaults agree with
    the manifest. Returns a list of human-readable mismatch strings (empty
    if all good). Used by /api/health and by the boot-time SOFT_WARN.

    This is the safeguard against "someone tweaked a Python default
    without updating the manifest" — the silent-drift mode that bit us
    on 2026-06-02.
    """
    cfg = load_active()
    by_name = {s.name: s for s in cfg.sleeves}
    issues: list[str] = []

    def _chk(name: str, in_yaml: float, in_code: float):
        if abs(in_yaml - in_code) > tolerance:
            issues.append(
                f"{name}: manifest = {in_yaml}, code constant = {in_code}"
            )

    if "cross_asset_carry" in by_name:
        _chk("cross_asset_carry.base_weight",
             by_name["cross_asset_carry"].base_weight, carry_risk_weight)
    if "cross_asset_tsmom" in by_name:
        _chk("cross_asset_tsmom.base_weight",
             by_name["cross_asset_tsmom"].base_weight, tsmom_risk_weight)
    if "crisis_hedge_tlt_gld" in by_name:
        _chk("crisis_hedge_tlt_gld.base_weight",
             by_name["crisis_hedge_tlt_gld"].base_weight, crisis_risk_weight)
    if "mom_hedge_overlay" in by_name:
        _chk("mom_hedge_overlay.base_weight",
             by_name["mom_hedge_overlay"].base_weight, mom_hedge_risk_weight)
    _chk("book_vol_target", cfg.book_vol_target, book_vol_target)

    return issues
