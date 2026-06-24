"""
engine/agents/risk_manager/gates.py — 12 deterministic risk gates.

Phase 2 of Risk Manager Agent v1.0 (spec id=69; current hash lives in
SpecRegistry — call engine.agents.persona.tools.lookup_spec(69) or
engine.preregistration.list_specs() for the canonical state). Pure
Python detectors — no LLM, no network, no I/O. Each returns a list of
Breach dataclasses; the orchestrator aggregates and decides halt/warn.

Senior-review upgrades applied during Phase 2 build (per
[[feedback-iterative-self-correction]] proactive build-time audit):

  1. Multi-sleeve ticker conservatism (Mode 1)
     Spec pseudocode used single-sleeve lookup. Real future case: same
     ticker in two sleeves (e.g. TLT in rms_crisis_hedge + a future
     etf_l1 risk-overlay strategy). Institutional Basel-III-style answer:
     when in doubt, apply the MOST RESTRICTIVE cap. ticker_to_sleeves
     is therefore a set; cap = min across all sleeves the ticker is in.

  2. HHI uses abs-normalized weights (Mode 5)
     Spec said (combined**2).sum() but long-short net weights squared
     misses the institutional Markowitz / Sharpe HHI definition. Switched
     to `(w_abs / w_abs.sum()).pow(2).sum()` matching
     engine.risk_metrics.compute_concentration's convention.

  3. Sleeve drift consumes orchestrator's pre-computed attribution
     (Mode 2)
     Avoid re-computing sleeve_eff inside this module — accept
     `sleeve_attribution: dict[str, float]` from the orchestrator step 4
     output (single source of truth).

  4. Locked Breach schema (8 fields, frozen)
     All 12 modes return the same shape so downstream
     persist/narrator/dashboard code doesn't need mode-specific branches.

DOCTRINE INVARIANTS (verified by Phase 9 tests):
  - All thresholds come from thresholds.RISK_THRESHOLDS /
    BOOK_SINGLE_TICKER_ABS_CAP / SLEEVE_CLASS_INTRA_CAPS (no magic
    numbers in this file)
  - Each gate is pure functional — same inputs → same outputs
  - Gates never call LLM, network, or DB
  - Gates never mutate inputs
"""
from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import pandas as pd
    from engine.portfolio.paper_trade_combined import StrategySignal

from engine.agents.risk_manager.thresholds import (
    BOOK_SINGLE_TICKER_ABS_CAP,
    RISK_THRESHOLDS,
    SLEEVE_CLASS_INTRA_CAPS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Breach dataclass — locked 8-field schema (Upgrade #4)
# ──────────────────────────────────────────────────────────────────────────────
SeverityLiteral = Literal["HARD_HALT", "SOFT_WARN"]


@dataclasses.dataclass(frozen=True)
class Breach:
    """One detected rule breach.

    Single locked schema across all 12 detectors. Downstream consumers
    (persist.py, narrator.py, advisory.py, dashboard) get a uniform
    shape regardless of which mode fired.
    """
    mode_id:           str                       # "1" / "2" / ... / "6b" / "7" / "7b" / "8" / "9" / "10"
    severity:          str                       # "HARD_HALT" or "SOFT_WARN"
    rule_description:  str                       # one-line human-readable rule
    observed_value:    float                     # what we measured (nan if not numeric)
    threshold:         float                     # what the rule says (nan if not numeric)
    affected:          tuple[str, ...]           # ticker / sleeve / strategy names
    extra:             dict                      # mode-specific context
    spec_anchor:       str                       # spec §X.Y citation


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _build_ticker_to_sleeves(signals: list["StrategySignal"]) -> dict[str, set[str]]:
    """Build ticker → {sleeve_id, ...} from per-strategy signals.

    Upgrade #1: returns a SET of sleeves (not a single sleeve_id) so the
    conservative-cap logic in Mode 1 handles future cross-sleeve overlap.
    """
    out: dict[str, set[str]] = {}
    for sig in signals:
        if sig.status != "OK":
            continue
        for ticker in sig.weights.index:
            out.setdefault(str(ticker), set()).add(sig.sleeve_id)
    return out


def _compute_hhi_book_level(combined: "pd.Series") -> float:
    """Markowitz / Sharpe HHI on abs-normalized weights.

    Upgrade #2: spec said (combined**2).sum() which is wrong for long-short
    portfolios (negative weights flip positive when squared, conflating
    short concentration with long concentration). The institutional
    convention used in engine.risk_metrics.compute_concentration is
    abs-normalize first, square, sum:
        w_abs = |w|;  w_norm = w_abs / Σw_abs;  hhi = Σw_norm²
    """
    import pandas as pd  # noqa: F401  (TYPE_CHECKING-guarded above)
    w_abs = combined.abs()
    total = float(w_abs.sum())
    if total <= 0.0:
        return 0.0
    w_norm = w_abs / total
    return float((w_norm ** 2).sum())


# ──────────────────────────────────────────────────────────────────────────────
# Gate functions — 13 modes (Mode 1 split into 1a + 1b per 2026-05-19 amend)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_1a_book_abs_cap(combined: "pd.Series") -> list[Breach]:
    """Mode 1a — book-level absolute single-ticker cap (operational risk).

    Defends against issuer-specific blowups: a single ETF delisting,
    failing tracking, or counterparty default. Cap is UNIFORM across all
    sleeves and ticker types (Aladdin "single-name exposure limit"
    standard). Combined book weight of any one ticker must not exceed
    BOOK_SINGLE_TICKER_ABS_CAP (default 25%).

    Per spec §2.1a Q1a. Distinct from Mode 1b which defends a different
    risk (intra-strategy concentration); both gates run independently
    without coordination — clean separation of concerns.
    """
    breaches: list[Breach] = []
    cap = BOOK_SINGLE_TICKER_ABS_CAP
    for ticker, w in combined.items():
        if abs(float(w)) > cap:
            breaches.append(Breach(
                mode_id          = "1a",
                severity         = "HARD_HALT",
                rule_description = (
                    f"book-level single-ticker weight |{w:+.4f}| exceeds "
                    f"{cap:.2%} operational-risk cap (uniform across sleeves)"
                ),
                observed_value   = abs(float(w)),
                threshold        = cap,
                affected         = (str(ticker),),
                extra            = {
                    "signed_weight":  float(w),
                    "risk_layer":     "operational",
                },
                spec_anchor      = "spec id=69 §2.1a Q1a",
            ))
    return breaches


def gate_mode_1b_intra_sleeve_cap(
    signals:   list["StrategySignal"],
    registry,
) -> list[Breach]:
    """Mode 1b — per-strategy intra-strategy ticker cap by sleeve_class.

    Defends against any single strategy over-leaning on one ticker
    within its own gross. Cap is per-sleeve-class because strategies
    have different universe sizes (BAB ~45 ETFs / D-PEAD ~1500 stocks /
    AC 2 tickers); a uniform intra cap would either over-restrict
    insurance designs or under-restrict equity factors.

    Evaluates against `signal.weights` (the strategy's intra-strategy
    weights, summing to ~1.0 absolute), NOT the post-orchestration book
    weights. A ticker held by N strategies is checked N times — once per
    strategy. Cross-strategy aggregation is Mode 1a's job, not Mode 1b's.

    Per spec §2.1a Q1b.
    """
    # Float-epsilon tolerance: AC TLT/GLD's deterministic 0.5/0.5 weights
    # and CTA PQTIX's 1.0 weight sit EXACTLY at their sleeve_class caps
    # (50% / 100%). Without an epsilon, any future code path that introduces
    # IEEE-754 rounding noise (e.g. weight = sum(parts)/total) could push
    # 0.5 to 0.5000000001 and trigger a false-positive HALT. 1e-6 tolerance
    # absorbs that without weakening the cap semantics (real concentration
    # excesses are always > 1e-6 above cap).
    _FP_EPSILON = 1e-6
    breaches: list[Breach] = []
    for sig in signals:
        if sig.status != "OK":
            continue
        try:
            sleeve = registry.get_sleeve(sig.sleeve_id)
        except KeyError:
            continue
        cap = SLEEVE_CLASS_INTRA_CAPS.get(sleeve.sleeve_class)
        if cap is None:
            continue
        for ticker, w in sig.weights.items():
            if abs(float(w)) > cap + _FP_EPSILON:
                breaches.append(Breach(
                    mode_id          = "1b",
                    severity         = "HARD_HALT",
                    rule_description = (
                        f"strategy {sig.strategy_name!r} intra-strategy weight "
                        f"|{float(w):+.4f}| on {ticker} exceeds {cap:.2%} cap "
                        f"for sleeve_class {sleeve.sleeve_class.value!r}"
                    ),
                    observed_value   = abs(float(w)),
                    threshold        = cap,
                    affected         = (str(ticker),),
                    extra            = {
                        "signed_weight":   float(w),
                        "strategy":        sig.strategy_name,
                        "sleeve_id":       sig.sleeve_id,
                        "sleeve_class":    sleeve.sleeve_class.value,
                        "risk_layer":      "strategy_concentration",
                    },
                    spec_anchor      = "spec id=69 §2.1a Q1b",
                ))
    return breaches


def gate_mode_2_sleeve_drift(
    sleeve_attribution: dict[str, float],
    sleeve_target:      dict[str, float],
    signals:            Optional[list["StrategySignal"]] = None,
) -> list[Breach]:
    """Mode 2 — relative sleeve drift > 10% of target (Q5 resolution).

    Upgrade #3: consume orchestrator's pre-computed sleeve_attribution
    instead of re-deriving. Single source of truth.

    2026-05-19 status-aware refinement: when a sleeve's strategies are
    event-driven (Path N reconstitution, D-PEAD post-earnings) the
    "no-event day" return is `status='NO_SIGNAL'` and the sleeve's
    effective weight legitimately collapses to ~0. Comparing that 0
    against the static `sleeve_target` produces a permanent false-
    positive SOFT_WARN on every non-event day. Fix: rescale the drift
    baseline to the share of intra-sleeve weight contributed by
    strategies that actually emitted positions today (status=='OK').

    Semantics:
      expected_weight = sleeve_target * sum(intra_sleeve_weight for OK strategies in sleeve)
      drift           = |effective - expected| / expected      (when expected > 0)
      sleeve skipped  when expected == 0                       (structurally inactive)

    The `signals` arg is optional — when None, this reverts to the
    pre-refinement behavior (compare effective vs full sleeve_target),
    preserving back-compat for callers that haven't been wired with
    the signals list (tests / legacy invocations).

    Zero-target sleeves are skipped (cannot compute relative drift).
    """
    breaches: list[Breach] = []
    th = RISK_THRESHOLDS.sleeve_drift_relative_max

    # Pre-compute per-sleeve OK-strategy intra_sleeve_weight share.
    ok_share_by_sleeve: dict[str, float] = {}
    if signals is not None:
        for sig in signals:
            if sig.status != "OK":
                continue
            ok_share_by_sleeve[sig.sleeve_id] = (
                ok_share_by_sleeve.get(sig.sleeve_id, 0.0)
                + float(sig.intra_sleeve_weight)
            )

    for sid, target in sleeve_target.items():
        if target == 0:
            continue
        eff = float(sleeve_attribution.get(sid, 0.0))

        if signals is not None:
            ok_share = ok_share_by_sleeve.get(sid, 0.0)
            if ok_share <= 1e-9:
                # No OK strategy in sleeve today → drift is structural,
                # not anomalous. Mode 9 handles "too few OK overall".
                continue
            expected = target * ok_share
        else:
            expected = target

        if expected <= 1e-9:
            continue
        rel = abs(eff - expected) / expected
        if rel <= th:
            continue

        breaches.append(Breach(
            mode_id          = "2",
            severity         = "SOFT_WARN",
            rule_description = (
                f"sleeve {sid!r} drift {rel:.1%} relative to "
                f"{'expected' if signals is not None else 'target'} "
                f"{expected:.1%} (eff {eff:.1%}); exceeds {th:.0%} threshold"
            ),
            observed_value   = rel,
            threshold        = th,
            affected         = (sid,),
            extra            = {
                "target_weight":     target,
                "expected_weight":   expected,
                "effective_weight":  eff,
                "absolute_diff":     eff - expected,
                "ok_share":          ok_share_by_sleeve.get(sid)
                                     if signals is not None else None,
            },
            spec_anchor      = "spec id=69 §3.1",
        ))
    return breaches


def gate_mode_3_gross_leverage(combined: "pd.Series") -> list[Breach]:
    """Mode 3 — gross leverage cap (Tier-3 1.5× nominal + 10pp band)."""
    th = RISK_THRESHOLDS.gross_leverage_max
    gross = float(combined.abs().sum())
    if gross <= th:
        return []
    return [Breach(
        mode_id          = "3",
        severity         = "HARD_HALT",
        rule_description = f"gross leverage {gross:.2f}× exceeds {th:.2f}× cap",
        observed_value   = gross,
        threshold        = th,
        affected         = (),
        extra            = {"n_tickers": int((combined.abs() > 1e-9).sum())},
        spec_anchor      = "spec id=69 §2.1 Mode 3",
    )]


def gate_mode_4_net_exposure(combined: "pd.Series") -> list[Breach]:
    """Mode 4 — net exposure outside [net_exposure_min, net_exposure_max]."""
    th_min = RISK_THRESHOLDS.net_exposure_min
    th_max = RISK_THRESHOLDS.net_exposure_max
    net = float(combined.sum())
    if th_min <= net <= th_max:
        return []
    binding = th_max if net > th_max else th_min
    return [Breach(
        mode_id          = "4",
        severity         = "HARD_HALT",
        rule_description = (
            f"net exposure {net:+.2f} outside [{th_min:+.2f}, {th_max:+.2f}] band"
        ),
        observed_value   = net,
        threshold        = binding,
        affected         = (),
        extra            = {
            "net_above_max": net > th_max,
            "net_below_min": net < th_min,
        },
        spec_anchor      = "spec id=69 §2.1 Mode 4",
    )]


def gate_mode_5_hhi(combined: "pd.Series") -> list[Breach]:
    """Mode 5 — Herfindahl-Hirschman concentration cap on abs-normalized weights.

    Upgrade #2: uses abs-normalized convention (Markowitz/Sharpe), not
    naive signed-weight squared sum.
    """
    th = RISK_THRESHOLDS.hhi_max
    hhi = _compute_hhi_book_level(combined)
    if hhi <= th:
        return []
    return [Breach(
        mode_id          = "5",
        severity         = "HARD_HALT",
        rule_description = (
            f"HHI {hhi:.3f} (abs-normalized) exceeds {th:.2f} cap; "
            f"book over-concentrated"
        ),
        observed_value   = hhi,
        threshold        = th,
        affected         = (),
        extra            = {
            "n_nonzero":    int((combined.abs() > 1e-9).sum()),
            "top1_abs":     float(combined.abs().max()) if len(combined) else 0.0,
        },
        spec_anchor      = "spec id=69 §2.1 Mode 5",
    )]


def gate_mode_6_var_95(var_95_historical: Optional[float]) -> list[Breach]:
    """Mode 6 — 1-day VaR-95 < soft-warn threshold (-3% NAV)."""
    th = RISK_THRESHOLDS.var_95_soft_warn
    if var_95_historical is None or math.isnan(var_95_historical):
        return []
    if var_95_historical >= th:
        return []
    return [Breach(
        mode_id          = "6",
        severity         = "SOFT_WARN",
        rule_description = (
            f"1-day VaR-95 (historical) {var_95_historical:.2%} < "
            f"{th:.2%} soft-warn threshold"
        ),
        observed_value   = var_95_historical,
        threshold        = th,
        affected         = (),
        extra            = {},
        spec_anchor      = "spec id=69 §2.1 Mode 6",
    )]


def gate_mode_6b_var_95_model_integrity(
    var_95_historical: Optional[float],
) -> list[Breach]:
    """Mode 6b — VaR-95 < -9% NAV (3× threshold; model-integrity HARD HALT)."""
    th = RISK_THRESHOLDS.var_95_hard_halt
    if var_95_historical is None or math.isnan(var_95_historical):
        return []
    if var_95_historical >= th:
        return []
    return [Breach(
        mode_id          = "6b",
        severity         = "HARD_HALT",
        rule_description = (
            f"VaR-95 (historical) {var_95_historical:.2%} < {th:.2%} "
            f"hard-halt threshold (3× soft-warn); model integrity breach"
        ),
        observed_value   = var_95_historical,
        threshold        = th,
        affected         = (),
        extra            = {"reason": "model_integrity_breach"},
        spec_anchor      = "spec id=69 §2.1 Mode 6b (Q4)",
    )]


def gate_mode_7_es_95(es_95_historical: Optional[float]) -> list[Breach]:
    """Mode 7 — 1-day ES-95 < soft-warn threshold (-5% NAV)."""
    th = RISK_THRESHOLDS.es_95_soft_warn
    if es_95_historical is None or math.isnan(es_95_historical):
        return []
    if es_95_historical >= th:
        return []
    return [Breach(
        mode_id          = "7",
        severity         = "SOFT_WARN",
        rule_description = (
            f"1-day ES-95 (historical) {es_95_historical:.2%} < "
            f"{th:.2%} soft-warn threshold"
        ),
        observed_value   = es_95_historical,
        threshold        = th,
        affected         = (),
        extra            = {},
        spec_anchor      = "spec id=69 §2.1 Mode 7",
    )]


def gate_mode_7b_es_95_model_integrity(
    es_95_historical: Optional[float],
) -> list[Breach]:
    """Mode 7b — ES-95 < -15% NAV (3× threshold; model-integrity HARD HALT)."""
    th = RISK_THRESHOLDS.es_95_hard_halt
    if es_95_historical is None or math.isnan(es_95_historical):
        return []
    if es_95_historical >= th:
        return []
    return [Breach(
        mode_id          = "7b",
        severity         = "HARD_HALT",
        rule_description = (
            f"ES-95 (historical) {es_95_historical:.2%} < {th:.2%} "
            f"hard-halt threshold (3× soft-warn); model integrity breach"
        ),
        observed_value   = es_95_historical,
        threshold        = th,
        affected         = (),
        extra            = {"reason": "model_integrity_breach"},
        spec_anchor      = "spec id=69 §2.1 Mode 7b (Q4)",
    )]


def gate_mode_8_short_side_ratio(combined: "pd.Series") -> list[Breach]:
    """Mode 8 — short-side aggregate > 50% of gross."""
    th = RISK_THRESHOLDS.short_side_max_of_gross
    gross = float(combined.abs().sum())
    if gross <= 0.0:
        return []
    short_total = float(combined[combined < 0].abs().sum())
    ratio = short_total / gross
    if ratio <= th:
        return []
    return [Breach(
        mode_id          = "8",
        severity         = "SOFT_WARN",
        rule_description = (
            f"short-side {short_total:.2%} of NAV is {ratio:.1%} of gross "
            f"({gross:.2%}); exceeds {th:.0%} cap"
        ),
        observed_value   = ratio,
        threshold        = th,
        affected         = (),
        extra            = {
            "short_total_abs":  short_total,
            "gross":            gross,
            "n_short_tickers":  int((combined < -1e-9).sum()),
        },
        spec_anchor      = "spec id=69 §2.1 Mode 8",
    )]


def gate_mode_9_min_ok_strategies(
    signals:  list["StrategySignal"],
    registry,
) -> list[Breach]:
    """Mode 9 — number of OK strategies < min_ok_strategies (default 3).

    HARD HALT with cb-cascade semantics: if a majority of strategies failed
    today, the book is in degraded state and should not be persisted.
    """
    th = RISK_THRESHOLDS.min_ok_strategies
    n_ok = sum(1 for s in signals if s.status == "OK")
    n_total = len(registry)
    if n_ok >= th:
        return []
    not_ok = tuple(sorted(s.strategy_name for s in signals if s.status != "OK"))
    return [Breach(
        mode_id          = "9",
        severity         = "HARD_HALT",
        rule_description = (
            f"only {n_ok} of {n_total} strategies are OK today "
            f"(threshold {th}); cb-cascade"
        ),
        observed_value   = float(n_ok),
        threshold        = float(th),
        affected         = not_ok,
        extra            = {
            "n_strategies_total":   n_total,
            "non_ok_strategy_set":  list(not_ok),
        },
        spec_anchor      = "spec id=69 §2.1 Mode 9",
    )]


def gate_mode_10_cross_cancel(
    signals:  list["StrategySignal"],
) -> list[Breach]:
    """Mode 10 — # tickers appearing both long and short across strategies > 5.

    "Cross-cancel" = inefficient hedging: two strategies expressing
    opposite signals on the same name. Some cross-cancel is healthy
    (intentional diversification) but > 5 tickers signals overlap that
    erodes capital efficiency.
    """
    th = RISK_THRESHOLDS.cross_cancel_ticker_max
    # ticker → {long_strats, short_strats}
    direction: dict[str, dict[str, set[str]]] = {}
    for sig in signals:
        if sig.status != "OK":
            continue
        for ticker, w in sig.weights.items():
            if abs(float(w)) < 1e-9:
                continue
            slot = direction.setdefault(str(ticker), {"long": set(), "short": set()})
            (slot["long"] if w > 0 else slot["short"]).add(sig.strategy_name)
    cross_tickers = [
        t for t, d in direction.items()
        if d["long"] and d["short"]
    ]
    if len(cross_tickers) <= th:
        return []
    return [Breach(
        mode_id          = "10",
        severity         = "SOFT_WARN",
        rule_description = (
            f"{len(cross_tickers)} tickers held both long and short "
            f"across strategies; exceeds {th} cap"
        ),
        observed_value   = float(len(cross_tickers)),
        threshold        = float(th),
        affected         = tuple(sorted(cross_tickers)),
        extra            = {
            "n_cross_tickers":  len(cross_tickers),
            "sample_tickers":   sorted(cross_tickers)[:10],
        },
        spec_anchor      = "spec id=69 §2.1 Mode 10",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Top-level evaluator — entry point used by RiskManagerAgent.check()
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_all_modes(
    *,
    combined:           "pd.Series",
    signals:            list["StrategySignal"],
    sleeve_attribution: dict[str, float],
    sleeve_target:      dict[str, float],
    registry,
    var_95_historical:  Optional[float] = None,
    es_95_historical:   Optional[float] = None,
) -> list[Breach]:
    """Run all 12 detectors against the given book state.

    VaR/ES are optional: when None (insufficient history or skip), the
    VaR/ES modes are no-ops. Mode 1/2/3/4/5/8/9/10 always run.

    Returns a flat list of all Breaches in mode-numeric order (1, 2, 3,
    4, 5, 6, 6b, 7, 7b, 8, 9, 10) — downstream consumers can rely on
    sort.
    """
    all_breaches: list[Breach] = []
    all_breaches += gate_mode_1a_book_abs_cap(combined)
    all_breaches += gate_mode_1b_intra_sleeve_cap(signals, registry)
    all_breaches += gate_mode_2_sleeve_drift(sleeve_attribution, sleeve_target, signals)
    all_breaches += gate_mode_3_gross_leverage(combined)
    all_breaches += gate_mode_4_net_exposure(combined)
    all_breaches += gate_mode_5_hhi(combined)
    all_breaches += gate_mode_6_var_95(var_95_historical)
    all_breaches += gate_mode_6b_var_95_model_integrity(var_95_historical)
    all_breaches += gate_mode_7_es_95(es_95_historical)
    all_breaches += gate_mode_7b_es_95_model_integrity(es_95_historical)
    all_breaches += gate_mode_8_short_side_ratio(combined)
    all_breaches += gate_mode_9_min_ok_strategies(signals, registry)
    all_breaches += gate_mode_10_cross_cancel(signals)
    return all_breaches


def classify_severity(breaches: list[Breach]) -> str:
    """Map a list of breaches to a circuit-breaker-compatible severity.

    Reuses the 4-level severity scheme from engine.circuit_breaker so
    Phase 5 absorption is byte-identical (G4 verdict gate).
    Severity rules:
      - any HARD_HALT     → SEVERE
      - >=2 SOFT_WARN     → MEDIUM
      - exactly 1 SOFT_WARN → LIGHT
      - empty             → NONE
    """
    if any(b.severity == "HARD_HALT" for b in breaches):
        return "SEVERE"
    n_warn = sum(1 for b in breaches if b.severity == "SOFT_WARN")
    if n_warn >= 2:
        return "MEDIUM"
    if n_warn == 1:
        return "LIGHT"
    return "NONE"


def any_hard_halt(breaches: list[Breach]) -> bool:
    """Convenience: True iff any breach is HARD_HALT severity."""
    return any(b.severity == "HARD_HALT" for b in breaches)
