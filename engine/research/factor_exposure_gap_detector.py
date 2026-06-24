"""engine.research.factor_exposure_gap_detector — Phase 1 MVP.

Closes the YAML-curated-knowledge blindspot identified in the 2026-06-17
session: the `active_deployment.yaml` improvement_directions list only
contains gaps the writer KNEW about. Nobody listed "low-vol BAB" because
nobody who wrote the yaml realized equity_book has zero BAB exposure.

This module regresses deployed-sleeve PnL on a canonical factor matrix
and identifies factors where the loading is statistically indistinguishable
from zero. Those factors are EMPIRICAL gaps — directions the deployed
sleeve has NO exposure to. Proposed improvement_directions get emitted as
deployment-demand-style rows for principal review.

Academic anchors
================
  - Sharpe 1992 "Asset Allocation: Management Style and Performance
    Measurement" — original return-based style analysis
  - Fung-Hsieh 1997, 2004 — extension to hedge funds
  - Bender et al. (MSCI Barra) — institutional factor exposure
    decomposition standard
  - Cochrane 2011 AFA presidential — "Discount Rates": the factor-zoo
    response is attribution, not addition

Phase 1 MVP scope
=================
  ✓ Canonical factor matrix (FF5 + MOM + BAB + XA_CARRY + XA_TSMOM + VRP)
  ✓ HAC-OLS sleeve PnL regression
  ✓ Gap identification (|t| < 1.65)
  ✓ Mapping gap-factor → MechanismFamily + direction text
  ✓ Smoke test on real equity_book PnL

Deferred to Phase 2
===================
  - QMJ factor builder (requires fundamentals + market caps + 6-10h
    self-build per `engine.risk.barra_lite.build_qmj_factor`).
    RMW partially covers as proxy.
  - /approvals UI integration (write to data/research/proposed_improvement_directions.jsonl)
  - Integration with deployment_demand_emitter (auto-feed proposals
    as separate-tagged demand rows)

Honest substrate limits
=======================
- Deployed-sleeve windows are short (equity_book ~111mo, carry ~316mo).
  Factor t-stat power is bounded; threshold deliberately conservative
  at 1.65 not 1.96 to flag MARGINAL gaps for human review.
- XA_CARRY and XA_TSMOM use OUR deployed sleeve PnL as the factor
  series — circular if we regress a deployed sleeve on itself.
  Skip self-regression at usage site.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KF_DAILY  = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_daily.parquet"
_AQR_BAB   = _REPO_ROOT / "data" / "cache" / "aqr_bab_usa_monthly.parquet"
_VRP_PNL   = _REPO_ROOT / "data" / "research_store" / "tier_c_pnl" / "afb46008e68c3625_GREEN.parquet"

# Gap threshold: |t-stat| < this → factor judged absent from sleeve.
# 1.65 chosen (not 1.96) to flag MARGINAL absence for human review on
# short windows where power is bounded.
GAP_T_THRESHOLD = 1.65


@_dc.dataclass(frozen=True)
class FactorExposure:
    """Single-factor regression result."""
    factor:   str
    beta:     float
    t_stat:   float
    p_value:  float

    def is_gap(self, t_thresh: float = GAP_T_THRESHOLD) -> bool:
        return (not math.isfinite(self.t_stat)) or abs(self.t_stat) < t_thresh


@_dc.dataclass(frozen=True)
class ExposureReport:
    """Output of deployed_factor_exposure."""
    sleeve_id:        str
    n_obs:            int
    window_start:     str
    window_end:       str
    r_squared:        float
    exposures:        tuple[FactorExposure, ...]
    gap_factors:      tuple[str, ...]

    def to_dict(self) -> dict:
        d = _dc.asdict(self)
        d["exposures"] = [{k: v for k, v in _dc.asdict(e).items()}
                            for e in self.exposures]
        return d


@_dc.dataclass(frozen=True)
class ProposedDirection:
    """Improvement-direction proposal for a detected gap factor."""
    sleeve_id:       str
    gap_factor:      str
    direction_text:  str
    mechanism_family: str
    rationale:       str


# ── Canonical factor matrix builder ────────────────────────────────


def _load_kf_monthly() -> pd.DataFrame:
    df = pd.read_parquet(_KF_DAILY)
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def _load_aqr_bab_monthly() -> pd.Series:
    df = pd.read_parquet(_AQR_BAB)
    if "BAB" in df.columns:
        s = df["BAB"]
    else:
        s = df.iloc[:, 0]
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s.dropna().rename("BAB")


def _load_vrp_monthly() -> pd.Series:
    df = pd.read_parquet(_VRP_PNL)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    s = df["pnl_net_13bp"].dropna()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
    return s.rename("VRP")


def _load_xa_carry_monthly() -> Optional[pd.Series]:
    """Use deployed carry sleeve PnL as XA_CARRY factor proxy.

    Returns None if the carry sleeve infra is unavailable (test mode).
    """
    try:
        from engine.portfolio.combined_book import build_carry_book
        s = build_carry_book().dropna()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        return s.rename("XA_CARRY")
    except Exception as exc:
        logger.warning("XA_CARRY proxy unavailable: %s", exc)
        return None


def _load_xa_tsmom_monthly() -> Optional[pd.Series]:
    """Use deployed TSMOM sleeve PnL as XA_TSMOM factor proxy."""
    try:
        from engine.portfolio.combined_book import build_tsmom_book
        s = build_tsmom_book().dropna()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp("M")
        return s.rename("XA_TSMOM")
    except Exception as exc:
        logger.warning("XA_TSMOM proxy unavailable: %s", exc)
        return None


def build_canonical_factor_matrix() -> pd.DataFrame:
    """Build monthly factor matrix for sleeve exposure regression.

    Factors included (Phase 1):
      MKT_RF SMB HML RMW CMA  — Fama-French 5-factor
      MOM                     — Carhart momentum / Fama-French addendum
      BAB                     — Frazzini-Pedersen 2014 (AQR data library)
      XA_CARRY                — our deployed cross-asset carry sleeve
      XA_TSMOM                — our deployed cross-asset TSMOM sleeve
      VRP                     — Carr 2009 SPX variance risk premium

    QMJ deferred to Phase 2 — RMW partially covers (literature: corr ~0.6).
    """
    parts: list[pd.Series | pd.DataFrame] = []

    kf = _load_kf_monthly()
    kf.index = pd.to_datetime(kf.index).to_period("M").to_timestamp("M")
    parts.append(kf[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]])

    parts.append(_load_aqr_bab_monthly())
    parts.append(_load_vrp_monthly())

    xa_carry = _load_xa_carry_monthly()
    if xa_carry is not None:
        parts.append(xa_carry)
    xa_tsmom = _load_xa_tsmom_monthly()
    if xa_tsmom is not None:
        parts.append(xa_tsmom)

    matrix = pd.concat(parts, axis=1)
    matrix.index = pd.to_datetime(matrix.index).to_period("M").to_timestamp("M")
    return matrix


# ── Regression core ──────────────────────────────────────────────


def _hac_regression(y: pd.Series, X: pd.DataFrame, lag: int = 6
                      ) -> tuple[float, dict[str, FactorExposure], int]:
    """HAC-OLS regression of y on X. Returns (r_squared, factor_dict, n_obs).

    Drops rows with any NaN. Newey-West lag default 6 (consistent with
    repo-wide convention used in cross_sec_us_equities template).
    """
    df = pd.concat({"y": y, **{c: X[c] for c in X.columns}}, axis=1).dropna()
    n = len(df)
    if n < 24:
        return float("nan"), {}, n

    import statsmodels.api as sm
    X_const = sm.add_constant(df[list(X.columns)].values)
    ols = sm.OLS(df["y"].values, X_const).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag},
    )
    factor_dict = {}
    for i, c in enumerate(X.columns):
        # Skip the constant (index 0); factor i in X.columns is at OLS index i+1
        beta = float(ols.params[i + 1])
        t    = float(ols.tvalues[i + 1])
        p    = float(ols.pvalues[i + 1])
        factor_dict[c] = FactorExposure(
            factor   = c,
            beta     = beta,
            t_stat   = t,
            p_value  = p,
        )
    return float(ols.rsquared), factor_dict, n


def deployed_factor_exposure(
    sleeve_pnl:        pd.Series,
    *,
    sleeve_id:         str,
    factor_matrix:     Optional[pd.DataFrame] = None,
    exclude_factors:   tuple[str, ...] = (),
    t_threshold:       float = GAP_T_THRESHOLD,
) -> ExposureReport:
    """Regress sleeve_pnl on canonical factor matrix; flag gap factors.

    exclude_factors: skip these from the regression entirely. Use this
    when the sleeve itself IS one of the proxy factors (e.g. regressing
    cross_asset_carry on a matrix that includes XA_CARRY = self).
    """
    if factor_matrix is None:
        factor_matrix = build_canonical_factor_matrix()
    sleeve_pnl = sleeve_pnl.dropna().copy()
    sleeve_pnl.index = pd.to_datetime(sleeve_pnl.index).to_period("M").to_timestamp("M")

    X_cols = [c for c in factor_matrix.columns if c not in exclude_factors]
    X = factor_matrix[X_cols]

    r2, factors, n_obs = _hac_regression(sleeve_pnl, X)

    # Determine the actual overlapping window after NaN-drop
    aligned = pd.concat({"y": sleeve_pnl, "_x": X.iloc[:, 0]}, axis=1).dropna()
    window_start = str(aligned.index.min().date()) if not aligned.empty else ""
    window_end   = str(aligned.index.max().date()) if not aligned.empty else ""

    exposures = tuple(factors[c] for c in X_cols if c in factors)
    gap_factors = tuple(e.factor for e in exposures
                         if e.is_gap(t_threshold))

    return ExposureReport(
        sleeve_id     = sleeve_id,
        n_obs         = n_obs,
        window_start  = window_start,
        window_end    = window_end,
        r_squared     = r2,
        exposures     = exposures,
        gap_factors   = gap_factors,
    )


# ── Direction proposal ──────────────────────────────────────────


# Map canonical-factor name → (canonical_direction_text, MechanismFamily.value)
# Keep concise and aligned with deployment_demand_emitter's regex rules
# so emitted proposals route through the same family-boost path.
# Phase 2 — sleeve-mechanism relevance filter
# Each sleeve's mechanism implies what factor families are RELEVANT
# research directions. A gap factor outside the relevant set is technically
# a gap but semantically noise (e.g. MOM gap in carry sleeve — carry isn't
# supposed to have momentum exposure; MKT_RF gap in market-neutral L/S
# equity_book — market neutrality is intentional, not a gap to fill).
#
# Mapping: sleeve_id → set of MechanismFamily values that are valid
# research directions for that sleeve. Empty/None = no filter (emit all).
SLEEVE_FAMILY_RELEVANCE: dict[str, frozenset[str]] = {
    "equity_book": frozenset({
        "VALUE", "PROFITABILITY", "SIZE", "INVESTMENT", "LOW_VOL",
        "VOL_RISK_PREMIUM", "EARNINGS_DRIFT", "ANALYST_REVISION",
        "MOMENTUM", "REVERSAL",
        # excluded: MKT_RF/OTHER (market neutrality intentional),
        # CARRY (book-level, not sleeve-level)
    }),
    "cross_asset_carry": frozenset({
        "CARRY", "TERM_STRUCTURE", "MACRO_SURPRISE",
        # excluded: equity factors (HML/RMW/CMA/MOM) — carry isn't supposed
        # to load on equity factors
    }),
    "cross_asset_tsmom": frozenset({
        "CROSS_ASSET_MOMENTUM", "MOMENTUM",
        # excluded: equity factors and carry — TSMOM is pure trend
    }),
}


_FACTOR_TO_DIRECTION: dict[str, tuple[str, str]] = {
    "BAB":      ("low-vol BAB variants",            "LOW_VOL"),
    "XA_CARRY": ("cross-asset carry exposure",      "CARRY"),
    "XA_TSMOM": ("time-series momentum / TSMOM exposure", "CROSS_ASSET_MOMENTUM"),
    "VRP":      ("variance risk premium exposure",  "VOL_RISK_PREMIUM"),
    "RMW":      ("quality factor QMJ / profitability variants", "PROFITABILITY"),
    "HML":      ("value factor HML variants",       "VALUE"),
    "MOM":      ("cross-sectional momentum variants", "MOMENTUM"),
    "SMB":      ("size factor SMB variants",        "SIZE"),
    "CMA":      ("investment factor CMA variants",  "INVESTMENT"),
    "MKT_RF":   ("market beta / equity exposure",   "OTHER"),
}


# ── Stage 2: pre-enhance filter ──────────────────────────────────


# Recommendation tiers — Phase 3 5-tier (calibrated 2026-06-17)
#
# Phase 3 (2026-06-17): added WEAK_PROCEED zone after the BAB book-overlay
# audit showed t=+1.648 → PROCEED → NOISE. Hard PROCEED/WARN boundary at
# 1.65 was unreliable in the 1.5-1.85 fuzzy zone; book-level induced
# exposure from constituent sleeves can be high enough to absorb a
# nominal "gap" candidate.
#
# Tier semantics:
#   |t| < 1.0           → STRONG_PROCEED  (very clean gap)
#   1.0 ≤ |t| < 1.5    → PROCEED         (likely gap)
#   1.5 ≤ |t| < 1.85   → WEAK_PROCEED    (boundary; expect possible NOISE)
#   1.85 ≤ |t| < 2.5   → WARN            (marginal loading present)
#   |t| ≥ 2.5          → SKIP            (strongly loaded; enhance NOISE)
#
# Legacy aliases (deprecated, kept for backward compat):
STRONG_PROCEED_T = 1.0    # below this → very clean gap
PROCEED_T_THRESHOLD = 1.5   # below this → likely gap (was 1.65 pre-Phase-3)
WEAK_PROCEED_T = 1.85   # below this → boundary
SKIP_T_THRESHOLD    = 2.5    # above this → strongly loaded


@_dc.dataclass(frozen=True)
class EnhanceFilterDecision:
    """Output of pre_enhance_check.

    recommendation: PROCEED / WARN / SKIP — the filter's verdict.
    Caller decides whether to honor (filter is ADVISORY by design).
    """
    recommendation:       str          # PROCEED / WARN / SKIP
    reason:               str
    sleeve_id:            str
    candidate_family:     str
    matched_gap_factor:   Optional[str]  # the canonical factor we matched
    factor_t_stat:        Optional[float]
    factor_beta:          Optional[float]

    def to_dict(self) -> dict:
        return _dc.asdict(self)


def pre_enhance_check(
    sleeve_id:                str,
    candidate_mechanism_family: str,
    sleeve_pnl:               "pd.Series",
    *,
    factor_matrix:            Optional["pd.DataFrame"] = None,
    exclude_factors:          tuple[str, ...] = (),
    proceed_t:                float = PROCEED_T_THRESHOLD,
    skip_t:                   float = SKIP_T_THRESHOLD,
) -> EnhanceFilterDecision:
    """Pre-check whether an enhance candidate's mechanism fills a real gap.

    Logic
    -----
    1. Find the canonical factor in factor_matrix that maps to
       candidate_mechanism_family (e.g. PROFITABILITY → RMW, LOW_VOL → BAB).
    2. Run FEGD regression on the target sleeve, read the |t-stat| of
       that factor.
    3. Classify:
       - |t| < proceed_t (1.0)  → PROCEED — true gap, enhance likely productive
       - proceed_t ≤ |t| < skip_t (1.0-2.0) → WARN — marginal, may be NOISE
       - |t| ≥ skip_t (2.0)     → SKIP — already loaded, enhance likely NOISE

    Why these thresholds
    --------------------
    PROCEED at |t| < 1.65 matches the FEGD GAP definition (consistency).
    SKIP at |t| ≥ 2.5 catches strongly-loaded factors where enhance is
    almost certainly NOISE. WARN tier captures ambiguity.

    Empirical from this session:
      RMW canonical vs combined_book — book has modest RMW loading
        → PROCEED — audit produced session's only marginal positive
        (t=+1.72 / p=0.04, NOISE marginal).
      TSMOM-speed-blend vs cross_asset_tsmom (CROSS_ASSET_MOMENTUM family)
        — TSMOM sleeve β_XA_TSMOM excluded (self), but family already
        strongly loaded via MOM (t=5.88)
        → SKIP would have prevented audit; audit result was all NOISE.

    Caveat: GP/A case (t=1.37 < 1.65) → PROCEED here, but audit was NOISE.
    The pre-check is a CHEAP SIEVE; multi-layer audit (FF5 spanning,
    paired enhance) catches subtle redundancy the filter misses.

    Returns an EnhanceFilterDecision; CALLER decides whether to honor
    the recommendation (Phase 1 is advisory, not blocking).
    """
    # 1. Find canonical factors for the candidate's family
    target_family = (candidate_mechanism_family or "").upper().strip()
    relevant_factors = [
        fac for fac, (_dir_text, fam) in _FACTOR_TO_DIRECTION.items()
        if fam == target_family
    ]
    if not relevant_factors:
        return EnhanceFilterDecision(
            recommendation     = "PROCEED",
            reason             = (f"No canonical factor maps to family "
                                    f"{target_family!r} — cannot pre-check, "
                                    f"defaulting to PROCEED."),
            sleeve_id          = sleeve_id,
            candidate_family   = target_family,
            matched_gap_factor = None,
            factor_t_stat      = None,
            factor_beta        = None,
        )

    # 2. Run FEGD regression
    report = deployed_factor_exposure(
        sleeve_pnl, sleeve_id=sleeve_id,
        factor_matrix=factor_matrix, exclude_factors=exclude_factors,
    )
    relevant_exposures = [e for e in report.exposures
                           if e.factor in relevant_factors]
    if not relevant_exposures:
        return EnhanceFilterDecision(
            recommendation     = "PROCEED",
            reason             = (f"Family {target_family!r} maps to factors "
                                    f"{relevant_factors} but none present in "
                                    f"factor matrix or all excluded; defaulting "
                                    f"to PROCEED."),
            sleeve_id          = sleeve_id,
            candidate_family   = target_family,
            matched_gap_factor = None,
            factor_t_stat      = None,
            factor_beta        = None,
        )

    # 3. Take the strongest-loaded factor in the family (max |t|)
    strongest = max(relevant_exposures, key=lambda e: abs(e.t_stat))
    abs_t = abs(strongest.t_stat)

    # 4. Classify — Phase 3 5-tier
    if abs_t < STRONG_PROCEED_T:           # < 1.0
        rec = "STRONG_PROCEED"
        reason = (f"Sleeve {sleeve_id} has near-zero {target_family} loading: "
                    f"|t({strongest.factor})|={abs_t:.2f} < {STRONG_PROCEED_T}. "
                    f"Very clean gap — enhance highly productive.")
    elif abs_t < proceed_t:                 # 1.0-1.5
        rec = "PROCEED"
        reason = (f"Sleeve {sleeve_id} has no significant {target_family} "
                    f"loading: |t({strongest.factor})|={abs_t:.2f} < "
                    f"{proceed_t}. Likely gap — enhance likely productive.")
    elif abs_t < WEAK_PROCEED_T:            # 1.5-1.85
        rec = "WEAK_PROCEED"
        reason = (f"Sleeve {sleeve_id} has BOUNDARY {target_family} loading: "
                    f"|t({strongest.factor})|={abs_t:.2f} in [{proceed_t}, "
                    f"{WEAK_PROCEED_T}). Below significance but inducedhood "
                    f"exposure may absorb enhance — proceed cautiously, expect "
                    f"possible NOISE (BAB book-overlay at t=1.65 had ΔSharpe "
                    f"~0.003 NOISE for this reason).")
    elif abs_t < skip_t:                    # 1.85-2.5
        rec = "WARN"
        reason = (f"Sleeve {sleeve_id} has marginal {target_family} loading: "
                    f"|t({strongest.factor})|={abs_t:.2f} in [{WEAK_PROCEED_T}, "
                    f"{skip_t}). Enhance likely NOISE due to partial redundancy "
                    f"with existing exposure.")
    else:                                    # >= 2.5
        rec = "SKIP"
        reason = (f"Sleeve {sleeve_id} already strongly loads on "
                    f"{strongest.factor} (t={strongest.t_stat:+.2f}, "
                    f"β={strongest.beta:+.3f}). Adding more {target_family} "
                    f"exposure highly likely NOISE — skip enhance test or "
                    f"expect failure.")

    return EnhanceFilterDecision(
        recommendation     = rec,
        reason             = reason,
        sleeve_id          = sleeve_id,
        candidate_family   = target_family,
        matched_gap_factor = strongest.factor,
        factor_t_stat      = float(strongest.t_stat),
        factor_beta        = float(strongest.beta),
    )


def _fegd_signature(sleeve_id: str, family: str, gap_factor: str) -> str:
    """Stable signature for the FEGD demand row. Idempotent across re-runs;
    parallel to deployment_demand_emitter._signature but tagged FEGD_DEMAND
    so the two sources are distinguishable in audit."""
    import hashlib
    blob = f"FEGD_DEMAND::{sleeve_id}::{family}::{gap_factor.lower().strip()}"
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    return f"FEGD_DEMAND::{sleeve_id}::{family}::{digest}"


def emit_fegd_demand(
    sleeve_id:        str,
    sleeve_pnl:       "pd.Series",
    *,
    factor_matrix:    Optional["pd.DataFrame"] = None,
    exclude_factors:  tuple[str, ...] = (),
    gaps_path:        Optional[Path] = None,
    dry_run:          bool = True,
    now:              Optional["_dt.datetime"] = None,
) -> dict:
    """Run FEGD on a sleeve → emit capability_gaps rows for detected gaps.

    Closes the FEGD Stage 1 loop: detected factor-exposure gaps become
    demand-ledger rows that burndown_ranker auto-boosts at the family
    level (×1.5 multiplier on demand_score).

    Idempotent: re-runs skip already-present signatures. Tagged
    `source:fegd_factor_gap` to distinguish from deployment_demand_emitter
    (`source:deployment_improvement`) at audit time. The ranker doesn't
    care about source — it reads the `family` field.

    Args:
      sleeve_id:       canonical sleeve name (matches subjects registry)
      sleeve_pnl:      monthly returns Series for the sleeve
      factor_matrix:   override the canonical factor matrix (test mode)
      exclude_factors: skip these from regression (e.g. self-regression)
      gaps_path:       override capability_gaps.jsonl path (test mode)
      dry_run:         if True, return what WOULD be written without writing
      now:             override timestamp (test mode)

    Returns dict with: parsed, already_present, written, new_rows.
    """
    import datetime as _dt
    import json
    if now is None:
        now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    ts_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    report = deployed_factor_exposure(
        sleeve_pnl, sleeve_id=sleeve_id,
        factor_matrix=factor_matrix, exclude_factors=exclude_factors,
    )
    proposals = propose_improvement_directions(report)

    # Phase 2 — apply sleeve-mechanism relevance filter if a mapping exists
    # for this sleeve. Drops semantic-noise gaps (e.g. MOM in carry sleeve).
    relevant = SLEEVE_FAMILY_RELEVANCE.get(sleeve_id)
    if relevant is not None:
        before = len(proposals)
        proposals = tuple(p for p in proposals if p.mechanism_family in relevant)
        if before != len(proposals):
            logger.info(
                "emit_fegd_demand: sleeve-mechanism filter dropped %d/%d gaps "
                "as not-relevant for sleeve %r",
                before - len(proposals), before, sleeve_id,
            )

    # Determine target path
    from engine.research.deployment_demand_emitter import (
        DEFAULT_GAPS_PATH as _DEFAULT_GAPS_PATH,
    )
    out_path = gaps_path or _DEFAULT_GAPS_PATH

    # Read existing FEGD signatures from the ledger to dedup
    existing: set = set()
    if out_path.is_file():
        for ln in out_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            sig = row.get("signature")
            if sig and sig.startswith("FEGD_DEMAND::"):
                existing.add(sig)

    # Build candidate rows
    new_rows: list[dict] = []
    for p in proposals:
        sig = _fegd_signature(sleeve_id, p.mechanism_family, p.gap_factor)
        if sig in existing:
            continue
        # Pull exposure record for audit detail
        exp = next((e for e in report.exposures if e.factor == p.gap_factor), None)
        row = {
            "ts":            ts_iso,
            "signature":     sig,
            "gap_class":     "FEGD_GAP",
            "family":        p.mechanism_family,
            "source":        "fegd_factor_gap",
            "sleeve":        sleeve_id,
            "gap_factor":    p.gap_factor,
            "direction":     p.direction_text,
            "beta":          exp.beta   if exp else None,
            "t_stat":        exp.t_stat if exp else None,
            "p_value":       exp.p_value if exp else None,
            "window_start":  report.window_start,
            "window_end":    report.window_end,
            "n_obs":         report.n_obs,
            "next_action":   (f"Boost demand_score for {p.mechanism_family} hyps to "
                                f"fill {sleeve_id}'s {p.gap_factor} exposure gap "
                                f"(observed loading t={exp.t_stat:+.2f} below "
                                f"|t|<{GAP_T_THRESHOLD} threshold)"),
            "effort":        "ranker auto-boost (no manual effort)",
        }
        new_rows.append(row)

    if not dry_run and new_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "sleeve_id":       sleeve_id,
        "parsed":          len(proposals),
        "already_present": len(proposals) - len(new_rows),
        "written":         len(new_rows) if not dry_run else 0,
        "dry_run":         dry_run,
        "new_rows":        new_rows,
    }


def propose_improvement_directions(
    report:    ExposureReport,
) -> tuple[ProposedDirection, ...]:
    """Map gap factors → canonical improvement_direction proposals.

    Empty tuple = no gaps detected (sleeve covers all canonical
    factors at |t| ≥ threshold).
    """
    proposals: list[ProposedDirection] = []
    for fac in report.gap_factors:
        if fac not in _FACTOR_TO_DIRECTION:
            continue
        direction, family = _FACTOR_TO_DIRECTION[fac]
        # Find the exposure record for the rationale text
        exp = next((e for e in report.exposures if e.factor == fac), None)
        if exp is None:
            continue
        rationale = (f"Sleeve {report.sleeve_id} loading on {fac} is β={exp.beta:+.3f} "
                       f"with t={exp.t_stat:+.3f} (|t| < {GAP_T_THRESHOLD}). "
                       f"Adding {fac}-related sleeve / overlay may add orthogonal alpha.")
        proposals.append(ProposedDirection(
            sleeve_id         = report.sleeve_id,
            gap_factor        = fac,
            direction_text    = direction,
            mechanism_family  = family,
            rationale         = rationale,
        ))
    return tuple(proposals)
