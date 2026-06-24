"""scripts/test_loop_graveyard_on_jp_pead.py — test whether
graveyard system catches cross-country PEAD candidates as cousins of
the already-RED China A-share PEAD entry.

Per user: "PEAD 我们在这几个市场上都做过了, 看看 loop 能不能发现"

Three candidates submitted to graveyard:
  1. Japan PEAD (cross-country PEAD; family=forward-earnings)
  2. China PEAD (re-proposal of EXACT existing RED — must catch)
  3. India PEAD (cross-country PEAD; family=forward-earnings)

Expected:
  - China PEAD: SHOULD trigger block (exact match)
  - JP/IN PEAD: SHOULD trigger warn or review (family/economics cousin)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.research.graveyard import (
    CandidateInfo, build_graveyard, check_against_graveyard,
)


def main() -> int:
    print("=" * 90)
    print(" GRAVEYARD COUSIN-DETECTION TEST: cross-country PEAD candidates")
    print("=" * 90)

    candidates = [
        CandidateInfo(
            title="Japan PEAD (TOPIX universe)",
            family="forward-earnings information",
            parent_family="equity_factor",
            required_data=["SUE_panel", "quarterly_eps", "ann_dates", "ret_60d"],
            economics_text=(
                "Post-earnings announcement drift in Japanese equities. "
                "Investors underreact to earnings surprises; drift persists "
                "60-90 days after announcement. Cross-country PEAD per "
                "Hou-Karolyi-Kho 2011 documented in 41 countries including "
                "Japan."
            ),
        ),
        CandidateInfo(
            title="China A-share PEAD",
            family="forward-earnings information",
            parent_family="equity_factor",
            required_data=["SUE_panel", "quarterly_eps", "ann_dates", "ret_60d"],
            economics_text=(
                "Post-earnings announcement drift in Chinese A-share equities. "
                "Standard PEAD with quarterly EPS surprise."
            ),
        ),
        CandidateInfo(
            title="India PEAD (Nifty 500)",
            family="forward-earnings information",
            parent_family="equity_factor",
            required_data=["SUE_panel", "quarterly_eps", "ann_dates", "ret_60d"],
            economics_text=(
                "Post-earnings announcement drift in Indian equity market. "
                "EM PEAD typically stronger than developed-market PEAD "
                "due to less institutional arbitrage capital."
            ),
        ),
    ]

    entries = build_graveyard()
    print(f"\nGraveyard has {len(entries)} entries\n")

    for candidate in candidates:
        print("=" * 90)
        print(f" CANDIDATE: {candidate.title}")
        print(f"   family: {candidate.family}")
        print("=" * 90)

        match = check_against_graveyard(candidate)
        d = match.to_dict()
        print(f"\n  matched:        {d['matched']}")
        print(f"  recommendation: {d['recommendation']}")
        print(f"  confidence:     {d.get('overall_confidence', '?')}")
        print(f"  signals:        {d.get('signals_matched', [])}")
        print(f"  cousin_count_in_family: {d.get('cousin_count_in_family', '?')}")
        print(f"  elevated:       {d.get('elevated', '?')}")
        print(f"  explanation:")
        for line in (d.get('explanation', '') or '').splitlines()[:8]:
            print(f"    {line}")
        if d.get('matched_entries'):
            print(f"  matched_entries:")
            for e in d['matched_entries']:
                if isinstance(e, dict):
                    print(f"    - {e.get('name')}  (verdict {e.get('verdict')})")
                else:
                    print(f"    - {e.name}  (verdict {e.verdict})")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
