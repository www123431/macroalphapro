"""engine.series_factory — generic family + subtype builder registry.

User feedback (verbatim 2026-06-05):
  "你要保证通用性诶 ... 不然你 carry 解决了又会有别的问题冒出来"

The architecture is GENERIC by design:
  - core = registry + resolve + cache + audit  (this file, never touched)
  - families/<name>.py = subtype builders for that family (additive only)
  - adding a new family = drop a new file in families/, no core change

Registry contract
-----------------
@register_subtype(family, subtype, *, deployed_replay=False)
def my_builder(hypothesis_id: str, params: dict) -> pd.Series:
    ...

  - family + subtype identifies the builder; (family, subtype) is unique
  - hypothesis_id is the cache key
  - params is hypothesis-specific config (passed through from the UI)
  - returns a pandas Series of returns (monthly or daily; coherent within family)
  - deployed_replay=True marks builders that just rebuild the existing
    deployed sleeve — meaningful for sanity / decay but NOT a new test

Caller contract
---------------
build(family=..., subtype=..., hypothesis_id=..., params=...)
  → {ok, path, family, subtype, hypothesis_id, n_obs, from_cache,
     elapsed_s, deployed_replay, error?}

NO SILENT FALLBACK. Unknown subtype → ok=False, error='unknown_subtype'.
Frontend then surfaces "this subtype not yet covered; use Claude handoff".

This is the LdP §2 epistemic discipline: never test the wrong thing
silently. A categorical refusal is louder than an inadvertent default.
"""
from __future__ import annotations

import json
import logging
import time
import datetime as _dt
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


logger = logging.getLogger(__name__)

_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR  = _REPO_ROOT / "data" / "cache"
_LOG_PATH   = _REPO_ROOT / "data" / "series_factory" / "build_log.jsonl"


# ── Generic registry ────────────────────────────────────────


BuilderFn = Callable[[str, dict], pd.Series]


class _Registration:
    __slots__ = ("fn", "deployed_replay", "doc")
    def __init__(self, fn: BuilderFn, deployed_replay: bool, doc: str):
        self.fn = fn
        self.deployed_replay = deployed_replay
        self.doc = doc


# Key: (FAMILY_UPPER, subtype_lower) → Registration
_REGISTRY: dict[tuple[str, str], _Registration] = {}


def register_subtype(
    family:           str,
    subtype:          str,
    *,
    deployed_replay:  bool = False,
):
    """Decorator. Maps (family, subtype) → builder. Raises on duplicate.

    Keys are case-folded for tolerance: ('CARRY', 'fx_carry_g10') matches
    incoming ('Carry', 'FX_Carry_G10') in the resolver.
    """
    fkey = family.upper().strip()
    skey = subtype.lower().strip()
    if not fkey or not skey:
        raise ValueError(f"family + subtype both required, got {family!r} / {subtype!r}")
    def deco(fn: BuilderFn) -> BuilderFn:
        key = (fkey, skey)
        if key in _REGISTRY:
            raise ValueError(f"duplicate series_factory registration: {family}/{subtype}")
        _REGISTRY[key] = _Registration(
            fn              = fn,
            deployed_replay = deployed_replay,
            doc             = (fn.__doc__ or "").strip().split("\n")[0][:200],
        )
        return fn
    return deco


def _ensure_families_imported() -> None:
    """Side-effect import: triggers every families/<name>.py to register."""
    try:
        from engine.series_factory import families  # noqa: F401
    except Exception as exc:
        logger.exception("series_factory.families failed to import: %s", exc)


def list_families() -> list[str]:
    _ensure_families_imported()
    return sorted({f for f, _ in _REGISTRY})


def list_subtypes(family: str) -> list[dict]:
    """Return every registered subtype for a family with its metadata.
    Used by the UI to show "X/Y subtypes covered" badges."""
    _ensure_families_imported()
    fkey = family.upper().strip()
    out: list[dict] = []
    for (f, s), reg in sorted(_REGISTRY.items()):
        if f != fkey:
            continue
        out.append({
            "subtype":          s,
            "doc":              reg.doc,
            "deployed_replay":  reg.deployed_replay,
        })
    return out


def is_subtype_covered(family: str, subtype: str) -> bool:
    _ensure_families_imported()
    if not family or not subtype:
        return False
    return (family.upper().strip(), subtype.lower().strip()) in _REGISTRY


# F6.4 (2026-06-05): curated subtype alias map.
#
# The extractor / forward_vector generator emit free-text human-readable
# subtypes like "cross_asset_carry_long_short" while builders are
# registered under technical keys like "cross_asset_carry_4leg". This
# gap is the SAME shape as the BAB <-> "Betting-Against-Beta" gap in
# graveyard_collision (T4.5/T4.6): semantically identical, string
# different. Until the extractor is constrained to output only
# registered subtype enums (next-session B.2 follow-up "C path"),
# this hand-maintained map closes the gap.
#
# Format: (FAMILY_UPPER, fv_subtype_lower) → registered_subtype.
# CARRY: 32 forward_vectors observed in corpus; ~11 of them describe
# strategies semantically equivalent to a registered builder. The
# rest are METHODOLOGY / DOMAIN_FACT claims that shouldn't be testable
# at all (they got mis-extracted as FACTOR_HYPOTHESIS) — those stay
# unmapped and return unknown_subtype as before.
_SUBTYPE_ALIASES: dict[tuple[str, str], str] = {
    # Cross-asset carry — all variants of "long high-carry / short
    # low-carry across asset classes" map to the deployed 4-leg
    ("CARRY", "cross_asset_carry_long_short"):       "cross_asset_carry_4leg",
    ("CARRY", "cross_asset_carry_within_asset_class"): "cross_asset_carry_4leg",
    ("CARRY", "global_diversified_carry_factor"):    "cross_asset_carry_4leg",
    ("CARRY", "global_multi_asset_carry"):           "cross_asset_carry_4leg",
    ("CARRY", "asset_class_specific_carry"):         "cross_asset_carry_4leg",

    # Carry timing — these are TSMOM-style overlays on carry
    ("CARRY", "carry_timing_time_series"):                "carry_tsmom_filtered",
    ("CARRY", "carry_timing_time_series_predictability"): "carry_tsmom_filtered",

    # Bond carry → rates carry (forward-spot spread IS rates carry)
    ("CARRY", "bond_carry_forward_spot_spread"):     "rates_carry_xc",

    # Commodity basis → commodity carry XS
    ("CARRY", "commodity_carry_basis"):              "commodity_carry_xs",

    # UIP failure cross-section IS the FX carry premium
    ("CARRY", "uip_failure_cross_section"):                       "fx_carry_cross_sectional",
    ("CARRY", "fx_carry_return_predictability_forward_discount"): "fx_carry_cross_sectional",
}


def _resolve(family: str, subtype: str) -> Optional[_Registration]:
    """F6.4: resolve (family, subtype) to a registered builder. Tries:
      1. literal match in _REGISTRY
      2. alias lookup in _SUBTYPE_ALIASES → registered builder
    Returns None if neither hits. Always ensures families/* imported
    so the registry is populated before lookup."""
    _ensure_families_imported()
    fkey = family.upper().strip()
    skey = subtype.lower().strip()
    direct = _REGISTRY.get((fkey, skey))
    if direct is not None:
        return direct
    aliased = _SUBTYPE_ALIASES.get((fkey, skey))
    if aliased is not None:
        return _REGISTRY.get((fkey, aliased))
    return None


def resolved_subtype(family: str, subtype: str) -> Optional[str]:
    """Return the EFFECTIVE registered subtype that build() would use
    for (family, subtype), with alias indirection applied. None if no
    builder is reachable. Used by callers (API, audit) for provenance:
    'you asked for X, we ran Y because alias map'."""
    _ensure_families_imported()
    fkey = family.upper().strip()
    skey = subtype.lower().strip()
    if (fkey, skey) in _REGISTRY:
        return skey
    return _SUBTYPE_ALIASES.get((fkey, skey))


# ── Cache + audit ───────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_path(hypothesis_id: str) -> Path:
    """Deterministic per-hypothesis cache path. Same hypothesis_id always
    produces the same parquet location → re-runs hit cache → reproducible
    per the LdP §2 discipline."""
    safe = "".join(c for c in (hypothesis_id or "") if c.isalnum() or c in "-_")[:60]
    if not safe:
        safe = "unknown"
    return _CACHE_DIR / f"_series_factory_{safe}.parquet"


def cached_path_for(hypothesis_id: str) -> Optional[Path]:
    p = cache_path(hypothesis_id)
    return p if p.is_file() else None


def _append_log(row: dict) -> None:
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("series_factory: failed to write build_log")


# ── Public build() entrypoint ───────────────────────────────


def build(
    *,
    family:        str,
    subtype:       str,
    hypothesis_id: str,
    params:        Optional[dict] = None,
    force:         bool = False,
) -> dict:
    """Build (or read cached) returns series for a hypothesis-grounded
    (family, subtype). NEVER falls back to a default — unknown subtype
    returns ok=False so the UI surfaces honest coverage."""
    _ensure_families_imported()
    params = dict(params or {})
    t0 = time.perf_counter()

    out = {
        "ok":              False,
        "path":            None,
        "family":          family,
        "subtype":         subtype,
        "resolved_subtype": None,         # F6.4: which registered key actually ran
        "via_alias":        False,        # F6.4: True if alias map mediated
        "hypothesis_id":   hypothesis_id,
        "params":          params,
        "n_obs":           0,
        "from_cache":      False,
        "elapsed_s":       0.0,
        "deployed_replay": False,
        "doc":             None,
        "error":           None,
        "ts":              _utc_iso(),
    }

    # F6.4: capture alias-vs-direct provenance before the build runs
    _effective = resolved_subtype(family, subtype)
    if _effective is not None:
        out["resolved_subtype"] = _effective
        out["via_alias"] = (_effective != subtype.lower().strip())

    # Cache hit (idempotent per LdP §2)
    cached = cached_path_for(hypothesis_id) if not force else None
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
            # Mark cache hits with subtype/replay metadata we can re-derive
            reg = _resolve(family, subtype)
            if reg is not None:
                out["deployed_replay"] = reg.deployed_replay
                out["doc"] = reg.doc
            _append_log(out)
            return out
        except Exception as exc:
            logger.warning("cached parquet unreadable, rebuilding: %s", exc)

    reg = _resolve(family, subtype)
    if reg is None:
        out["error"]     = "unknown_subtype"
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        _append_log(out)
        return out

    try:
        series = reg.fn(hypothesis_id, params)
        if not isinstance(series, pd.Series):
            raise TypeError(
                f"builder for {family}/{subtype} returned "
                f"{type(series).__name__}, expected pd.Series")
        n_clean = int(len(series.dropna()))
        if n_clean < 24:
            raise ValueError(
                f"series too short ({n_clean} obs); need >= 24 for any "
                f"meaningful deflated-Sharpe estimate")

        p = cache_path(hypothesis_id)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        series.to_frame(name="return").to_parquet(p)

        out.update(
            ok              = True,
            path            = str(p),
            n_obs           = n_clean,
            from_cache      = False,
            elapsed_s       = round(time.perf_counter() - t0, 2),
            deployed_replay = reg.deployed_replay,
            doc             = reg.doc,
        )
        _append_log(out)
        return out
    except Exception as exc:
        logger.exception("series_factory build failed for %s/%s", family, subtype)
        out["error"]     = str(exc)[:300]
        out["elapsed_s"] = round(time.perf_counter() - t0, 2)
        _append_log(out)
        return out


# ── Backward-compat shim ────────────────────────────────────
# Old code (early Phase) called `register_family` + family-only `build`.
# Keep the old call working but warn — eventually all callers should
# pass an explicit subtype.


def register_family(family: str):
    """DEPRECATED. Use register_subtype(family, subtype). This shim
    registers under family/'(legacy)' subtype so old callers don't
    crash, but the build path now requires explicit subtype."""
    return register_subtype(family, "_legacy_no_subtype", deployed_replay=True)


def has_builder(family: str) -> bool:
    """DEPRECATED. Use is_subtype_covered. This returns True if ANY
    subtype for the family is registered."""
    _ensure_families_imported()
    fkey = family.upper().strip()
    return any(f == fkey for f, _ in _REGISTRY)
