"""engine.research_store.red_lessons.backfill_heuristics — RED record → REDLesson.

Heuristic classifier that maps the existing 23 + 44 historical RED records
(data/research/gate_runs.jsonl + data/validation/factory_ledger.jsonl) into
the new REDLesson schema.

Design philosophy:
  - Heuristic FIRST pass produces `review_state=claude_drafted` lessons.
  - Anything ambiguous gets the most-likely failure mode + a `tag` marking
    it as low-confidence, NOT skipped.
  - Human review (later) advances claude_drafted → human_reviewed.
  - LLM-assist is a P1.5 optional layer — first run is pure rules.

Heuristic rules — failure mode inference from stats:

  F8_OVERFIT_INDUCED  ← deflated_sr < 0.9 (our codebase HLZ-equivalent bar)
  F9_RESIDUAL_NULL    ← |alpha_t_ff5umd| < 2 AND standalone_sharpe > 0.5
  F3_SUBSUMED_BY_EXISTING ← corr_with_book > 0.5
  F7_POWER_INSUFFICIENT ← n_months < 60 OR (n_trials high but n_obs small)
  F1/F2/F4/F5/F6 ← CANNOT infer from stats alone; flag with tag and
                  leave for human/LLM review pass.

Mechanism family inference uses a keyword table + paper-title heuristic.
Records that don't match any family go to MechanismFamily.OTHER with a
`needs_review` tag.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from pathlib import Path
from typing import Any

from engine.research_store.red_lessons.failure_modes import FailureMode
from engine.research_store.red_lessons.mechanism_families import MechanismFamily
from engine.research_store.red_lessons.schema import (
    LessonStrength,
    REDLesson,
    ReviewState,
)

logger = logging.getLogger(__name__)


# ────────────────────── mechanism family mapping ──────────────────────

# Keyword → (family, subtype) tuples. First match wins; ordered by specificity.
# Each entry: (regex pattern on lowercased combined name+mechanism, family,
# subtype label)
MECHANISM_KEYWORDS: list[tuple[str, MechanismFamily, str]] = [
    # ─── event-driven ───
    # Lookaround on letters only — underscores / digits don't block boundary.
    (r"(?<![a-z])pead(?![a-z])|post.{0,3}earnings.{0,3}drift|(?<![a-z])dpead", MechanismFamily.EARNINGS_DRIFT, "post_earnings_drift"),
    (r"analyst[_ ]?rev|\banalyst[_ ]?rev\b|sue|ibes|earnings[_ ]?revision", MechanismFamily.ANALYST_REVISION, "analyst_eps_revision"),
    (r"guidance",                                               MechanismFamily.GUIDANCE,           "management_guidance"),
    (r"insider",                                                MechanismFamily.INSIDER_TRADING,    "form4_insider_buy"),
    (r"short[_ ]?interest|borrow[_ ]?fee",                      MechanismFamily.SHORT_INTEREST,     "short_interest_predict"),
    # ─── text / attention / news ───
    (r"news[_ ]?attention|google[_ ]?trends|wikipedia[_ ]?view|attention.{0,5}shock", MechanismFamily.ATTENTION,  "investor_attention"),
    (r"sentiment|finbert|lm[_ ]?dict|tone",                     MechanismFamily.SENTIMENT,          "text_sentiment"),
    (r"news[_ ]?shock|8.?k[_ ]?body|edgar.{0,5}body|merger[_ ]?arb", MechanismFamily.NEWS_SHOCK,    "news_event_shock"),
    # ─── carry / cross-asset ───
    # Use lookaround on letters only — underscore is a word char in \b semantics
    # so `\bcarry\b` doesn't match inside `cross_asset_carry_3leg`.
    (r"carry[_ ]?(equity|div|x_|aug|book)|carry.{0,15}sleeve",  MechanismFamily.CARRY,              "cross_asset_carry"),
    (r"(?<![a-z])carry(?![a-z])",                               MechanismFamily.CARRY,              "carry_generic"),
    (r"credit[_ ]?(spread|risk)|hyg.{0,5}ief|credit[_ ]?premium", MechanismFamily.CARRY,            "credit_risk_premium"),
    (r"bond[_ ]?carry|sovereign[_ ]?bond|bond[_ ]?term",        MechanismFamily.CARRY,              "bond_carry"),
    (r"tsmom|cta|trend|moskowitz[_ ]?ooi[_ ]?pedersen|cross[_ ]?asset[_ ]?momentum", MechanismFamily.CROSS_ASSET_MOMENTUM, "futures_tsmom"),
    (r"term[_ ]?structure|yield[_ ]?curve|cochrane.{0,8}piazzesi", MechanismFamily.TERM_STRUCTURE,  "term_structure"),
    (r"macro|fomc|nfp|cpi.{0,15}surprise",                      MechanismFamily.MACRO_SURPRISE,     "macro_surprise"),
    # ─── options / vol ───
    (r"vrp|vix.{0,15}(carry|term|structure)|variance[_ ]?risk", MechanismFamily.VOL_RISK_PREMIUM,   "vix_term_carry"),
    (r"skew|iv[_ ]?rank|put[_ ]?call|implied",                  MechanismFamily.OPTIONS_IMPLIED,    "options_implied"),
    # ─── cross-section equity classics ───
    (r"value|book[_ ]?to[_ ]?market|ep_ratio",                  MechanismFamily.VALUE,              "value_anomaly"),
    (r"profitability|gross[_ ]?profit|novy[_ ]?marx",           MechanismFamily.PROFITABILITY,      "gross_profitability"),
    (r"investment|asset[_ ]?growth|capex|r&d|\brd\b|patents?",  MechanismFamily.INVESTMENT,         "investment_anomaly"),
    (r"\bbab\b|betting[_ ]?against[_ ]?beta|low[_ ]?vol|min[_ ]?vol|k1_bab", MechanismFamily.LOW_VOL, "bab_low_vol"),
    (r"momentum|umd|jegadeesh|blitz.{0,5}huij.{0,5}martens|residual[_ ]?momentum", MechanismFamily.MOMENTUM, "cross_sectional_momentum"),
    (r"reversal|short[_ ]?term[_ ]?reversal|\bstr\b|lazy[_ ]?prices?|cohen[_ ]?malloy[_ ]?nguyen", MechanismFamily.REVERSAL, "short_term_reversal"),
    (r"\bsize\b|small[_ ]?cap[_ ]?premium|banz",                MechanismFamily.SIZE,               "size_premium"),
    # ─── holdings / supply chain ───
    (r"13f|holdings|fund[_ ]?flows|smart[_ ]?money",            MechanismFamily.HOLDINGS_BASED,     "13f_holdings"),
    (r"supply[_ ]?chain|customer.{0,5}supplier|cohen[_ ]?frazzini", MechanismFamily.SUPPLY_CHAIN,   "supply_chain"),
    (r"sector[_ ]?lead[_ ]?lag|hong.{0,5}lim.{0,5}stein",       MechanismFamily.SUPPLY_CHAIN,       "sector_lead_lag"),
    # ─── tactical / overlay / crisis hedge (engineering pattern, not a clean family) ───
    (r"crisis[_ ]?hedge|tail[_ ]?hedge|mom[_ ]?hedge|put[_ ]?spread", MechanismFamily.OTHER,        "crisis_hedge_overlay"),
    (r"vol[_ ]?target|tactical[_ ]?risk[_ ]?weight|regime[_ ]?conditional|covariance[_ ]?aware", MechanismFamily.OTHER, "portfolio_construction"),
]


# Some heuristic markers for failure modes from the `reasons` field
REASON_MARKERS: list[tuple[str, FailureMode]] = [
    (r"deflated[_ ]?sr.{0,15}below|fail.{0,15}deflated|multiple[_ ]?testing", FailureMode.F8_OVERFIT_INDUCED),
    (r"residual.{0,15}alpha.{0,15}(null|zero|insig)",                        FailureMode.F9_RESIDUAL_NULL),
    (r"subsumed|spanned[_ ]?by|correlat|orthogonal",                          FailureMode.F3_SUBSUMED_BY_EXISTING),
    (r"insufficient[_ ]?power|too[_ ]?few[_ ]?(events|months|obs)",          FailureMode.F7_POWER_INSUFFICIENT),
    (r"cost[_ ]?fragile|tc.{0,5}swamp|net[_ ]?of[_ ]?cost",                  FailureMode.F4_IMPLEMENTATION_COST),
    (r"regime|crisis.{0,5}fail",                                              FailureMode.F5_REGIME_DEPENDENT),
    (r"published|already[_ ]?arbitraged|post[_ ]?2000",                      FailureMode.F1_PUBLICATION_DECAY),
    (r"mismatch|wrong[_ ]?market|china.{0,15}retail",                         FailureMode.F2_MECHANISM_MISMATCH),
    (r"survivorship|look[_ ]?ahead|delisting|data[_ ]?quality",              FailureMode.F6_DATA_QUALITY),
]


def classify_mechanism(name: str, mechanism: str | None) -> tuple[MechanismFamily, str]:
    """Map a (name, mechanism) pair to (MechanismFamily, mechanism_subtype).

    Falls back to OTHER + raw mechanism string if no keyword matches.
    """
    haystack = " ".join(filter(None, [name or "", mechanism or ""])).lower()
    for pattern, fam, subtype in MECHANISM_KEYWORDS:
        if re.search(pattern, haystack):
            return fam, subtype
    # Fallback
    raw_subtype = (mechanism or name or "unknown").strip().lower()[:80]
    return MechanismFamily.OTHER, raw_subtype


def classify_failure_modes(record: dict[str, Any]) -> tuple[
    tuple[FailureMode, ...], dict[str, str]
]:
    """Heuristic failure-mode classification + per-mode evidence string.

    Looks at the statistical fields (deflated_sr, alpha_t_ff5umd,
    corr_with_book, n_months) AND the `reasons` field if present.

    Returns:
        modes:    tuple of FailureMode codes (1-3, ordered by importance)
        evidence: dict {failure_mode_value: 1-sentence evidence}
    """
    modes: list[FailureMode] = []
    evidence: dict[str, str] = {}

    # Coerce missing stats to None
    def _get(*keys, default=None):
        for k in keys:
            v = record.get(k)
            if v is not None:
                return v
        return default

    dsr_raw        = _get("deflated_sr")          # pre-cost / raw DSR
    dsr_net        = _get("net_deflated_sr")      # post-cost DSR (factory_ledger)
    alpha_t        = _get("alpha_t_ff5umd", "residual_alpha_t", "alpha_t")
    corr_book      = _get("corr_with_book")
    n_months       = _get("n_months", "n_obs")
    standalone_sr  = _get("standalone_sharpe", "annualized_sharpe")
    reasons        = " ".join(_get("reasons", default=[]) or []) if isinstance(_get("reasons"), list) else str(_get("reasons", "") or "")

    # F4 detection: raw DSR clears but cost-net DSR fails → cost-fragile, not overfit
    f4_triggered = (
        isinstance(dsr_raw, (int, float)) and isinstance(dsr_net, (int, float))
        and dsr_raw >= 0.9 and dsr_net < 0.9
    )
    # F8 detection: raw DSR fails outright → genuine overfit / multiple-testing fail
    f8_triggered_by_dsr = isinstance(dsr_raw, (int, float)) and dsr_raw < 0.9

    # Rule: high corr with book → F3 subsumption (strongest signal — check first)
    if isinstance(corr_book, (int, float)) and abs(corr_book) > 0.5:
        modes.append(FailureMode.F3_SUBSUMED_BY_EXISTING)
        evidence[FailureMode.F3_SUBSUMED_BY_EXISTING.value] = (
            f"corr_with_book = {corr_book:.2f} (> 0.5 spanning-test threshold)"
        )

    # Rule: alpha_t null vs FF5+UMD AND signal IS solo-significant → F9
    if (isinstance(alpha_t, (int, float)) and abs(alpha_t) < 2.0
        and isinstance(standalone_sr, (int, float)) and standalone_sr > 0.5):
        modes.append(FailureMode.F9_RESIDUAL_NULL)
        evidence[FailureMode.F9_RESIDUAL_NULL.value] = (
            f"standalone_sharpe={standalone_sr:.2f} but alpha_t_ff5umd={alpha_t:.2f} "
            f"(<2; sub-HLZ residual)"
        )

    # Rule: low sample → F7
    if isinstance(n_months, (int, float)) and n_months < 60:
        modes.append(FailureMode.F7_POWER_INSUFFICIENT)
        evidence[FailureMode.F7_POWER_INSUFFICIENT.value] = (
            f"n_months={n_months} (< 60 → insufficient power for HLZ |t|>=3 bar)"
        )

    # Rule: cost cuts the strategy below the bar → F4 implementation cost
    if f4_triggered:
        modes.append(FailureMode.F4_IMPLEMENTATION_COST)
        evidence[FailureMode.F4_IMPLEMENTATION_COST.value] = (
            f"deflated_sr={dsr_raw:.3f} (raw clears) but "
            f"net_deflated_sr={dsr_net:.3f} (< 0.9 post-cost) → cost-fragile"
        )

    # Rule: raw deflated SR fail → F8 overfit-induced
    if f8_triggered_by_dsr and not f4_triggered:
        modes.append(FailureMode.F8_OVERFIT_INDUCED)
        evidence[FailureMode.F8_OVERFIT_INDUCED.value] = (
            f"deflated_sr={dsr_raw:.3f} (< 0.9 HLZ-equivalent bar; raw fail)"
        )

    # Rule: `reasons` text signals — additive
    for pattern, fm in REASON_MARKERS:
        if re.search(pattern, reasons, re.IGNORECASE) and fm not in modes:
            modes.append(fm)
            evidence[fm.value] = f"reasons field mentions: matched '{pattern}' in: '{reasons[:120]}'"

    # Truncate to top 3
    modes = modes[:3]

    # Backstop: if nothing matched but verdict is RED, classify as F8 with note
    if not modes:
        modes.append(FailureMode.F8_OVERFIT_INDUCED)
        evidence[FailureMode.F8_OVERFIT_INDUCED.value] = (
            "FALLBACK: no specific stat triggered classification; "
            "marking F8 by default for RED verdict. NEEDS HUMAN REVIEW."
        )

    return tuple(modes), evidence


def lesson_from_gate_run(record: dict[str, Any], session_id: str) -> REDLesson | None:
    """Convert one gate_runs.jsonl record into a REDLesson.

    Skips records whose verdict isn't RED or YELLOW (GREEN / null / GREEN-
    with-conditions don't become lessons).
    """
    raw_verdict = (record.get("verdict") or "").strip().upper()
    # gate_runs verdicts have trailing notes like "GREEN — 4/4 strict bars..."
    if raw_verdict.startswith("GREEN"):
        return None
    if raw_verdict not in ("RED", "YELLOW"):
        return None  # NULL / in-progress

    name = record.get("name", "unknown")
    mechanism = record.get("mechanism")
    family, subtype = classify_mechanism(name, mechanism)
    modes, evidence = classify_failure_modes(record)

    stat_evidence = {k: record.get(k) for k in (
        "standalone_sharpe", "alpha_t_ff5umd", "alpha_t_ff5umd_pead",
        "deflated_sr", "oos_sharpe", "corr_with_book",
        "n_months", "n_trials",
    ) if record.get(k) is not None}

    # subsumed_by inference — if F3, try to extract from corr_with_book + book composition
    subsumed_by: tuple[str, ...] = ()
    if FailureMode.F3_SUBSUMED_BY_EXISTING in modes:
        # We don't know WHICH deployed factor; mark as TBD for human
        subsumed_by = ("TBD_human_review",)

    summary = (
        f"{name} → {raw_verdict}: family={family.value}/{subtype}, "
        f"failure_modes={[m.value for m in modes]}. "
        f"Backfilled 2026-06-03 from gate_runs.jsonl."
    )[:400]

    return REDLesson(
        lesson_id          = REDLesson.new_id(),
        candidate_name     = name,
        version            = 1,
        parent_lesson_id   = None,
        source_event_ids   = (),  # gate_runs predates event store
        verdict            = raw_verdict,
        stat_evidence      = stat_evidence,
        mechanism_family   = family,
        mechanism_subtype  = subtype,
        failure_modes      = modes,
        failure_evidence   = evidence,
        paper_motivation   = None,  # P2 will fill via OpenAlex
        paper_critiques    = (),
        subsumed_by        = subsumed_by,
        related_lesson_ids = (),
        forward_directions = (),  # P5 will populate
        do_not_retry       = (),
        dormant_revisits   = (),
        review_state       = ReviewState.claude_drafted,
        strength           = LessonStrength.weak,
        created_ts         = "2026-06-03T00:00:00Z",
        updated_ts         = "2026-06-03T00:00:00Z",
        created_by         = "engine.backfill_heuristics",
        summary            = summary,
        tags               = ("backfill_p1", "needs_paper_anchor",
                              "needs_human_review",
                              f"src:gate_runs",
                              *(("needs_subsumed_by_resolution",) if subsumed_by == ("TBD_human_review",) else ())),
    )


def lesson_from_factory_ledger(record: dict[str, Any], session_id: str) -> REDLesson | None:
    """Convert one factory_ledger.jsonl record into a REDLesson."""
    light = (record.get("light") or "").strip().upper()
    if light == "GREEN":
        return None
    if light not in ("RED", "YELLOW"):
        return None

    name = record.get("name", "unknown")
    family, subtype = classify_mechanism(name, None)
    modes, evidence = classify_failure_modes(record)

    stat_evidence = {k: record.get(k) for k in (
        "deflated_sr", "net_deflated_sr", "residual_alpha_t",
        "residual_alpha_ann", "effective_bets_delta",
        "n_trials", "n_obs", "annual_turnover",
    ) if record.get(k) is not None}

    subsumed_by: tuple[str, ...] = ()
    if FailureMode.F3_SUBSUMED_BY_EXISTING in modes:
        subsumed_by = ("TBD_human_review",)

    summary = (
        f"{name} → {light}: family={family.value}/{subtype}, "
        f"failure_modes={[m.value for m in modes]}. "
        f"Backfilled 2026-06-03 from factory_ledger.jsonl."
    )[:400]

    return REDLesson(
        lesson_id          = REDLesson.new_id(),
        candidate_name     = name,
        version            = 1,
        parent_lesson_id   = None,
        source_event_ids   = (),
        verdict            = light,
        stat_evidence      = stat_evidence,
        mechanism_family   = family,
        mechanism_subtype  = subtype,
        failure_modes      = modes,
        failure_evidence   = evidence,
        paper_motivation   = None,
        paper_critiques    = (),
        subsumed_by        = subsumed_by,
        related_lesson_ids = (),
        forward_directions = (),
        do_not_retry       = (),
        dormant_revisits   = (),
        review_state       = ReviewState.claude_drafted,
        strength           = LessonStrength.weak,
        created_ts         = record.get("ts", "2026-06-03T00:00:00Z"),
        updated_ts         = "2026-06-03T00:00:00Z",
        created_by         = "engine.backfill_heuristics",
        summary            = summary,
        tags               = ("backfill_p1", "needs_paper_anchor",
                              "needs_human_review",
                              f"src:factory_ledger",
                              *(("needs_subsumed_by_resolution",) if subsumed_by == ("TBD_human_review",) else ())),
    )
