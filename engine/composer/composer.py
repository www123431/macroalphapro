"""composer.composer — spec → atomic components → returns series.

This is the load-bearing assembly logic. ~200 lines, NEVER touched
per-family. New family = new components; composer unchanged.

Pipeline
--------
  1. Resolve every spec field to its registered Component (raise if any
     missing — NEVER substitute)
  2. UNIVERSE.build(spec, {})                 → membership panel
  3. SIGNAL.build(spec, {universe})           → signal panel (per leg)
  4. WEIGHTING.build(spec, {signals, universe}) → weights panel
  5. (REBALANCE.build) → rebalance dates (used as index filter on weights)
  6. Apply weights to per-period returns → raw returns Series
  7. RISK_FILTER.build(spec, {pre_filter_returns}) → multiplier Series
  8. Final = raw_returns × multiplier
  9. Append provenance row
  10. Cache by spec_hash
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.composer.contract import (
    Component, ComponentRole, ComponentResult, get_component,
    ComponentNotFound, is_spec_covered,
)
from engine.hypothesis_spec.enums import ClaimType
from engine.hypothesis_spec.schema import HypothesisSpec
from engine.hypothesis_spec.hash import spec_hash


class NotAFactorHypothesisError(ValueError):
    """B.2-A3: raised when compose() receives a spec whose claim_type
    is not FACTOR_HYPOTHESIS. Non-factor claims (METHODOLOGY,
    MICROSTRUCTURE, etc.) are research evidence but have no returns
    series to build — calling compose() on them is a programmer error,
    not a data error, so we fail loud rather than silently producing
    a meaningless series."""
    def __init__(self, spec_hash_: str, claim_type: ClaimType):
        super().__init__(
            f"compose() refused: spec {spec_hash_} has claim_type="
            f"{claim_type.value} (not FACTOR_HYPOTHESIS). Non-factor "
            f"claims (methodology / microstructure / capacity / decay / "
            f"factor-structure / domain-fact) don't have a returns series. "
            f"Filter on claim_type == FACTOR_HYPOTHESIS before composing."
        )
        self.spec_hash = spec_hash_
        self.claim_type = claim_type

logger = logging.getLogger(__name__)

_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR       = _REPO_ROOT / "data" / "composer_cache"
_PROVENANCE_PATH = _REPO_ROOT / "data" / "composer" / "provenance.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_path(s_hash: str) -> Path:
    return _CACHE_DIR / f"{s_hash}.parquet"


def cached_for(s_hash: str) -> Optional[Path]:
    p = cache_path(s_hash)
    return p if p.is_file() else None


def _append_provenance(row: dict) -> None:
    try:
        _PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PROVENANCE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("composer: failed to write provenance")


def _resolve_component(role: ComponentRole, key: str) -> type[Component]:
    cls = get_component(role, key)
    if cls is None:
        raise ComponentNotFound(role, key)
    return cls


def _returns_panel_for(spec: HypothesisSpec) -> pd.DataFrame:
    """Lookup the underlying per-asset returns panel matching the spec's
    universe. Used to convert weights → returns."""
    ac = spec.universe.asset_class.value
    if ac == "FX":
        from engine.validation.crossasset_carry import build_fx_carry
        _cw, rw, _ls = build_fx_carry()
    elif ac == "RATES":
        from engine.validation.crossasset_carry import build_rates_xc_carry
        _cw, rw, _ls = build_rates_xc_carry()
    elif ac == "COMMODITY":
        from engine.portfolio.carry_sleeve import build_carry_contract_panels
        _cwide, rw = build_carry_contract_panels()
    elif ac == "EQUITY":
        # C1 substrate (2026-06-05): EQUITY backed by cached CRSP universe.
        from engine.composer.components.equity_data import crsp_returns_wide
        rw = crsp_returns_wide()
    else:
        raise ValueError(f"composer has no returns panel for asset_class={ac}")
    return rw


# ── Main entrypoint ─────────────────────────────────────


def compose(spec: HypothesisSpec, *, force: bool = False) -> dict:
    """Build the returns series for one spec. Returns a dict:
      {ok, spec_hash, path, n_obs, elapsed_s, from_cache, components_used,
       data_vintages, error?}

    Raises ComponentNotFound (caller catches) if any required component
    is missing — LdP §2 / SR-11-7: NEVER silently substitute.
    """
    s_hash = spec_hash(spec)
    t0 = time.perf_counter()
    out = {
        "ok":               False,
        "spec_hash":        s_hash,
        "path":             None,
        "n_obs":            0,
        "elapsed_s":        0.0,
        "from_cache":       False,
        "components_used":  [],
        "data_vintages":    {},
        "error":            None,
        "ts":               _utc_iso(),
    }

    # B.2-A3: claim_type gate. Only FACTOR_HYPOTHESIS specs have a
    # returns series to build. UNKNOWN is allowed for backward-compat
    # with v1 specs (extractor_v=v1) so they don't break in-flight
    # work; the audit_verifier C0 check is the longer-term home for
    # surfacing them. Anything else is a hard reject.
    if spec.claim_type not in (ClaimType.FACTOR_HYPOTHESIS, ClaimType.UNKNOWN):
        out["error"] = f"non_factor_claim_type:{spec.claim_type.value}"
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        _append_provenance(out)
        raise NotAFactorHypothesisError(s_hash, spec.claim_type)

    # Cache hit
    cached = cached_for(s_hash) if not force else None
    if cached is not None:
        try:
            s = pd.read_parquet(cached).iloc[:, 0]
            out.update(
                ok        = True,
                path      = str(cached),
                n_obs     = int(len(s.dropna())),
                from_cache= True,
                elapsed_s = round(time.perf_counter() - t0, 2),
            )
            _append_provenance(out)
            return out
        except Exception as exc:
            logger.warning("cached series unreadable, rebuilding: %s", exc)

    # Coverage check — fail FAST + LOUD
    covered, gaps = is_spec_covered(spec)
    if not covered:
        gap_strs = [f"{g.role.value}/{g.expected_key}" for g in gaps]
        out["error"]     = f"coverage_gaps:{gap_strs}"
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        _append_provenance(out)
        # NOT a silent failure — raise so callers know
        raise ComponentNotFound(gaps[0].role, gaps[0].expected_key)

    # ── 1. UNIVERSE ──
    u_cls = _resolve_component(
        ComponentRole.UNIVERSE,
        f"{spec.universe.asset_class.value}__{spec.universe.subset.value}",
    )
    if u_cls is None:
        u_cls = _resolve_component(
            ComponentRole.UNIVERSE,
            f"{spec.universe.asset_class.value}__ALL",
        )
    universe_result = u_cls().build(spec, {})
    out["components_used"].append({"role": "UNIVERSE", "cls": u_cls.__name__})
    out["data_vintages"]["UNIVERSE"] = {
        "start": universe_result.metadata.get("date_start"),
        "end":   universe_result.metadata.get("date_end"),
    }

    # ── 2. SIGNALS (one per leg) ──
    signal_results: list[ComponentResult] = []
    for leg_i, leg in enumerate(spec.legs):
        s_cls = _resolve_component(ComponentRole.SIGNAL, leg.signal_type.value)
        ctx = {"universe": universe_result, "leg": leg}
        r = s_cls().build(spec, ctx)
        # Attach the leg role so the weighting can find primary
        r = ComponentResult(
            data=r.data,
            metadata={**r.metadata, "role": leg.role, "leg_index": leg_i},
        )
        signal_results.append(r)
        out["components_used"].append({
            "role": "SIGNAL", "cls": s_cls.__name__, "leg": leg_i,
        })

    # ── 3. WEIGHTING ──
    w_cls = _resolve_component(ComponentRole.WEIGHTING, spec.construction.weighting.value)
    weights_result = w_cls().build(spec, {
        "signals":  signal_results,
        "universe": universe_result,
    })
    out["components_used"].append({"role": "WEIGHTING", "cls": w_cls.__name__})

    # ── 4. REBALANCE ──
    rb_cls = _resolve_component(ComponentRole.REBALANCE, spec.construction.rebalance.value)
    rebalance_result = rb_cls().build(spec, {"weights": weights_result})
    out["components_used"].append({"role": "REBALANCE", "cls": rb_cls.__name__})

    # ── 5. Apply weights to returns ──
    rw = _returns_panel_for(spec)
    weights = weights_result.data.reindex_like(rw).fillna(0.0)
    # Lag weights by 1 to avoid look-ahead — weights set at end of t-1
    # earn returns of t.
    weights_lagged = weights.shift(1).fillna(0.0)
    raw_returns = (weights_lagged * rw).sum(axis=1)
    # Restrict to rebalance dates
    raw_returns = raw_returns.loc[raw_returns.index.intersection(rebalance_result.data)]

    # ── 6. RISK_FILTER (optional vol-target) ──
    if spec.risk.vol_target_annual is not None:
        rf_cls = _resolve_component(ComponentRole.RISK_FILTER, "VOL_TARGET")
        rf_result = rf_cls().build(spec, {"pre_filter_returns": raw_returns})
        final = (raw_returns * rf_result.data.reindex(raw_returns.index).fillna(0.0))
        out["components_used"].append({"role": "RISK_FILTER", "cls": rf_cls.__name__})
    else:
        final = raw_returns

    # ── 7. Cache + return ──
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(s_hash)
    final.to_frame(name="return").to_parquet(p)

    out.update(
        ok        = True,
        path      = str(p),
        n_obs     = int(len(final.dropna())),
        elapsed_s = round(time.perf_counter() - t0, 2),
    )
    _append_provenance(out)
    return out
