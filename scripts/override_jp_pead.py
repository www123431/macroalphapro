"""scripts/override_jp_pead.py — request override for JP PEAD via the
structured 5-step workflow (commit f04e261 fix + this commit).

If granted: write entry to override_ledger.jsonl and PERMIT downstream
JP PEAD build work. If denied: log errors + abort.

Per senior re-design 2026-05-31: process-rigor not evidence-rigor.
Our OWN judgment + structured discipline is the bar, not requiring
proprietary-paper citations that don't exist publicly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.research.graveyard import (
    CandidateInfo, build_graveyard, check_against_graveyard,
)
from engine.research.override_workflow import (
    CousinRebuttal, ExplorationBudget, FalsificationCriterion,
    OverrideRequest, empirical_override_success_rate, grant_override,
)


def main() -> int:
    # ── Step 0: get graveyard signal for JP PEAD ────────────────────
    candidate = CandidateInfo(
        title="Japan PEAD (TOPIX universe)",
        family="forward-earnings information",
        parent_family="equity_factor",
        required_data=["SUE_panel", "quarterly_eps", "ann_dates", "ret_60d"],
        economics_text=(
            "Post-earnings announcement drift in Japanese equities. "
            "Institutional-dominated market with high analyst coverage."
        ),
    )
    match = check_against_graveyard(candidate)
    graveyard_signal = match.to_dict()
    print(f"  graveyard signal: {graveyard_signal['recommendation']} "
          f"(cousins={graveyard_signal['cousin_count_in_family']})")

    # Pull matched cousins for explicit rebuttal
    matched_cousins = graveyard_signal.get("matched_entries", [])
    # Convert to plain dicts (might be GraveyardEntry objects from to_dict)
    cousin_names = []
    for c in matched_cousins:
        if isinstance(c, dict):
            cousin_names.append((c.get("name"), c.get("failure_reason", "")[:200]))
        else:
            cousin_names.append((getattr(c, "name", "?"),
                                  getattr(c, "failure_reason", "")[:200]))

    print(f"  graveyard reports {len(cousin_names)} matched cousins to rebut")

    # ── Step 1: structured cousin analysis ──────────────────────────
    # NB: must address EACH cousin, not just the headliner.
    cousins_analysis = [
        CousinRebuttal(
            cousin_id="china_a_share_pead",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=[
                "retail_share_pct (CN ~80% vs JP ~25%)",
                "price_limits (CN ±10% vs JP none)",
                "settlement (CN T+1+short-restrictions vs JP T+2+free-short)",
                "analyst_coverage (CN ~3 vs JP ~12 per firm)",
            ],
            why_might_not_apply=(
                "China PEAD RED was diagnosed as retail-overreaction / "
                "reversal-killed-by-turnover. Japan's institutional dominance "
                "+ no price limits + dense analyst coverage means the "
                "underreaction mechanism (Hong-Stein 1999) could apply where "
                "the reversal mechanism does not."
            ),
        ),
        CousinRebuttal(
            cousin_id="management_guidance_drift",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=["signal_source (guidance vs EPS surprise)"],
            why_might_not_apply=(
                "Different signal source — management guidance vs actual EPS "
                "surprise. Guidance drift was RED for low Sharpe + 0.48 corr "
                "with analyst-revision (same-family redundancy). EPS surprise "
                "is the original Bernard-Thomas 1989 signal, distinct from "
                "guidance."
            ),
        ),
        CousinRebuttal(
            cousin_id="restatement_drift",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=["signal_source (restatement event vs EPS)"],
            why_might_not_apply=(
                "Different signal — restatement events. Post-SOX decay does "
                "not apply to ongoing earnings surprises."
            ),
        ),
        CousinRebuttal(
            cousin_id="d_pead_plus_overlay",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=["scope (overlay add-on vs standalone)"],
            why_might_not_apply=(
                "RED was for redundancy with parent D_PEAD — applies to US "
                "context, not relevant to building a fresh sleeve in a new "
                "geography."
            ),
        ),
        CousinRebuttal(
            cousin_id="pre_fomc_drift",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=["mechanism (macro event vs micro EPS)"],
            why_might_not_apply=(
                "Pre-FOMC is macro-context not earnings-surprise — different "
                "mechanism family despite graveyard family co-tagging."
            ),
        ),
        CousinRebuttal(
            cousin_id="labor_signal_drift",
            cousin_verdict="RED",
            same_or_different="different",
            structural_diff_dimensions=["signal_source (labor vs EPS)"],
            why_might_not_apply=(
                "Labor signal drift RED was sample-power. Different signal "
                "(labor data vs EPS), different power consideration."
            ),
        ),
    ]

    # ── Step 2: pre-committed falsification ─────────────────────────
    falsification = [
        FalsificationCriterion(
            label="F1", metric="per_event_t_stat",
            operator=">=", threshold=2.0,
            rationale="HLZ-Beng 2.0 threshold (relaxed from 3.0 since this is "
                      "single-mechanism not multi-trial)",
        ),
        FalsificationCriterion(
            label="F2", metric="monthly_long_short_decile_sharpe_net",
            operator=">=", threshold=0.5,
            rationale="HLZ floor — anything below is below noise",
        ),
        FalsificationCriterion(
            label="F3", metric="deflated_sharpe_with_n_trials_8",
            operator=">=", threshold=0.6,
            rationale="Bailey-LdP DeflSR floor; n_trials=8 = family count",
        ),
        FalsificationCriterion(
            label="F4", metric="cosine_with_us_pit_sn",
            operator="<", threshold=0.4,
            rationale="must be ORTHOGONAL to US PIT SN (cross-country "
                      "diversification value)",
        ),
        FalsificationCriterion(
            label="F5", metric="per_event_sample_size",
            operator=">=", threshold=5000.0,
            rationale="China PEAD RED was 8717 events; need comparable power",
        ),
    ]

    # ── Step 3: exploration budget ──────────────────────────────────
    budget = ExplorationBudget(
        max_person_hours=8.0,
        checkpoint_at_hours=4.0,
        abandon_if_budget_exhausted=True,
    )

    # ── Build override request ──────────────────────────────────────
    request = OverrideRequest(
        candidate_id="jp_pead",
        candidate_family="forward-earnings information",
        candidate_title="Japan PEAD (TOPIX universe)",
        graveyard_signal=graveyard_signal,
        cousins_analysis=cousins_analysis,
        falsification=falsification,
        exploration_budget=budget,
        da_review_status="deferred",  # Phase 2 DA-on-override build
        override_author="zhangxizhe",
        override_reason_summary=(
            "Cross-country PEAD test on JP TOPIX. Graveyard cousin China "
            "A-share PEAD was retail-microstructure-killed; Japan is "
            "structurally opposite (institutional-dominated, no price "
            "limits, free shorts, dense analyst coverage). Hong-Stein 1999 "
            "info-diffusion model PREDICTS strongest PEAD where coverage "
            "high + retail low. Pre-committed F1-F5 abandon criteria + "
            "8h budget cap."
        ),
        cited_evidence=[
            "Hou-Karolyi-Kho 2011 RFS — PEAD in 41 countries incl. Japan",
            "Hong-Stein 1999 JF — information diffusion model",
            "Bernard-Thomas 1989 JAR — original PEAD on US EPS surprise",
            "OWN: graveyard CN PEAD failure mode = retail microstructure "
            "(non-applicable to JP)",
        ],
    )

    # ── Show empirical history (will be empty first time) ───────────
    print(f"\n  empirical override track record so far:")
    hist = empirical_override_success_rate()
    print(f"    {hist}")

    # ── Try grant ───────────────────────────────────────────────────
    print(f"\n  attempting grant...")
    granted = grant_override(request)
    if granted:
        print(f"  → GRANTED. Logged to data/research/override_ledger.jsonl")
        print(f"  Next: build JP PEAD via SUE panel + PIT SN method.")
        print(f"  Budget: 8h max. Falsify on ANY of F1-F5.")
        return 0
    else:
        print(f"  → DENIED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
