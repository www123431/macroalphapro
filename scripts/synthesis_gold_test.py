"""scripts/synthesis_gold_test.py — Phase 2.0 calibration tool.

Hand-crafted gold-standard substrate test for Employee A's synthesis.
Question this answers: "Is Sonnet 4.6 actually thinking, or is it
defaulting to empty regardless of substrate?"

Design:
  - Real deployed_sleeves (loaded from library yamls — accurate)
  - Real recent_events (loaded from events.jsonl — has the 6 D signals)
  - HAND-CRAFTED recent_summaries: 3 papers all converging on
    VOL_RISK_PREMIUM, a family that is:
      * NOT in any deployed sleeve (clean slot)
      * NOT in any doctrine_signal RED cluster (no graveyard block)
      * Backed by deep academic literature (Bakshi-Kapadia 2003,
        Carr-Wu 2009, Bollerslev-Tauchen 2009)
      * Has a clear discount-rate story (insurance against vol
        spikes — risk + behavioral)

Any honest reviewer SHOULD propose at least one VRP candidate
here. If Sonnet still returns empty → the production prompt is
over-calibrated to "prefer empty" and we need to dial it back.
If Sonnet proposes → the empty results on real-world sparse
substrate are GENUINE clinic discipline, not stuckness.

Outputs:
  - Sonnet's RAW pre-tool-use text reasoning (full transcript)
  - The tool call payload (candidates emitted)
  - Cost + latency for forensic comparison vs production runs

This is NOT a production endpoint — it's a one-off diagnostic.
Cost: 1 LLM call (~$0.03). Run with:

  python scripts/synthesis_gold_test.py [--json]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ────────────────────────────────────────────────────────────────────
# Hand-crafted gold-standard paper summaries
# ────────────────────────────────────────────────────────────────────
def _gold_papers():
    from engine.agents.papers_curator.synthesis import PaperSummaryRef
    return (
        PaperSummaryRef(
            paper_id            = "arxiv/2606.GOLD1",
            title               = "Variance Risk Premium in SPX: Post-2010 Stability and Costs-Aware Sharpe",
            authors_short       = "Bollerslev, Tauchen, Zhou (refresh)",
            thesis              = "Selling 1-month SPX variance and rolling delta-hedged short straddles "
                                  "captures a positive risk premium of ~4-6% annualized post-2010. The "
                                  "premium survives transaction costs (bid-ask, slippage) at retail size "
                                  "and is robust to crisis windows when sized via vol-targeting.",
            testable_hypothesis = "Short 1m ATM SPX straddle, rebalance monthly, vol-target 10%. "
                                  "Sharpe > 0.8 post-cost over 2010-2024.",
            why_matters_for_us  = "We have no VRP sleeve deployed. Clean orthogonal source vs the 5 "
                                  "deployed sleeves (carry, TSMOM, PEAD, crisis hedge, mom hedge).",
            risk_flags_short    = ("requires options data", "crisis tail exposure", "execution complexity"),
            recommended_action  = "INGEST",
        ),
        PaperSummaryRef(
            paper_id            = "arxiv/2606.GOLD2",
            title               = "Asymmetric Variance Risk Premium: Put Insurance Carry vs Naked Short Vol",
            authors_short       = "Andersen, Fusari, Todorov (extension)",
            thesis              = "Decomposing VRP into the up-side (call) and down-side (put) shows "
                                  "the put-side premium is 3× larger. A put-spread overlay (short 25-delta "
                                  "put, long 10-delta put) isolates the carry while bounding tail loss to "
                                  "~5% per month. Long-run Sharpe 0.7-0.9 post-cost.",
            testable_hypothesis = "Roll short 25d put / long 10d put on SPX monthly. Vol-target 8%. "
                                  "Sharpe > 0.6 post-cost over 2010-2024. Max monthly loss capped at -5%.",
            why_matters_for_us  = "Bounded-loss VRP structure addresses the principal's 'don't blow up' "
                                  "constraint AND is orthogonal to our existing put-spread crisis hedge "
                                  "(which is long protection, not short).",
            risk_flags_short    = ("requires SPX skew surface", "complex sizing"),
            recommended_action  = "INGEST",
        ),
        PaperSummaryRef(
            paper_id            = "arxiv/2606.GOLD3",
            title               = "Variance Risk Premium Across Asset Classes: 2010-2024 Walk-Forward",
            authors_short       = "Bekaert, Hoerova, Engle (replication)",
            thesis              = "VRP exists and is harvestable across SPX, EURO STOXX, NIKKEI, TY "
                                  "futures, with cross-asset diversification lifting portfolio Sharpe "
                                  "to 1.2-1.4 post-cost in walk-forward OOS 2015-2024. Premium is "
                                  "stable across regimes after excluding 1 month around vol spikes.",
            testable_hypothesis = "Equal-risk portfolio of VRP harvesters across 4 indices. Sharpe > 1.0 "
                                  "post-cost in walk-forward; max DD < 15%.",
            why_matters_for_us  = "Multi-asset breadth (which we explicitly seek per "
                                  "project-cross-asset-breadth-focus-2026-05-28). VRP cluster is novel "
                                  "vs all 5 deployed sleeves.",
            risk_flags_short    = ("data costs across markets", "execution coordination"),
            recommended_action  = "INGEST",
        ),
    )


# ────────────────────────────────────────────────────────────────────
# Build the gold input
# ────────────────────────────────────────────────────────────────────
def _build_gold_input():
    """Real sleeves + real events (with D signals) + GOLD papers."""
    from engine.agents.papers_curator.synthesis import SynthesisInput
    from engine.agents.papers_curator.synthesis_context import (
        _load_deployed_sleeves, _load_recent_events,
    )
    import datetime as _dt
    return SynthesisInput(
        recent_summaries  = _gold_papers(),
        deployed_sleeves  = _load_deployed_sleeves(),
        recent_events     = _load_recent_events(days=30),
        doctrine_snippets = (),       # stub, same as production
        snapshot_ts       = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ────────────────────────────────────────────────────────────────────
# Direct llm_call so we capture text + tool call together
# ────────────────────────────────────────────────────────────────────
def _run_diagnostic(verbose: bool = False) -> dict:
    """Use the EXACT production prompt + tool schema + format_input,
    but call llm_call directly so we get result.text (the pre-tool
    reasoning) which production run_synthesis discards."""
    from engine.agents.papers_curator.synthesis import (
        _SYSTEM_PROMPT, _TOOL_DEFINITION, _format_input,
    )
    from engine.llm.call import call as llm_call

    si = _build_gold_input()
    user_msg = _format_input(si)

    if verbose:
        print("=" * 70)
        print("USER MESSAGE (truncated)")
        print("=" * 70)
        print(user_msg[:3000])
        print("..." if len(user_msg) > 3000 else "")
        print()

    result = llm_call(
        workload   = "papers_curator_synthesis",
        system     = _SYSTEM_PROMPT,
        user       = user_msg,
        agent_id   = "papers_curator_synthesis",
        tools      = [_TOOL_DEFINITION],
        max_tokens = 4000,
        scope      = "gold_diagnostic",
    )

    candidates = []
    tool_call_input = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_synthesis":
            tool_call_input = tc.input
            cands = tc.input.get("candidates", []) if isinstance(tc.input, dict) else []
            candidates = cands
            break

    return {
        "snapshot": {
            "recent_summaries":  len(si.recent_summaries),
            "deployed_sleeves":  len(si.deployed_sleeves),
            "recent_events":     len(si.recent_events),
        },
        "model":             result.model,
        "cost_usd":          result.cost_usd,
        "latency_ms":        result.latency_ms,
        "input_tokens":      None,   # not on LLMCallResult; cost_ledger has it
        "raw_text":          result.text,
        "tool_call_input":   tool_call_input,
        "n_candidates":      len(candidates),
        "candidates":        candidates,
        "stop_reason":       result.stop_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                     help="Emit JSON instead of human-readable.")
    ap.add_argument("--show-prompt", action="store_true",
                     help="Print the user message Sonnet sees (truncated).")
    ap.add_argument("--quiet-llm", action="store_true",
                     help="Suppress LLM client log lines.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet_llm else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    print("Running gold-standard substrate test against Sonnet 4.6...")
    print(f"  (real deployed sleeves + real events with D signals +"
          f" 3 hand-crafted strong VRP papers)")
    print()

    result = _run_diagnostic(verbose=args.show_prompt)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    print("=" * 70)
    print(f"VERDICT: Sonnet proposed {result['n_candidates']} candidate(s)")
    print("=" * 70)
    print(f"  cost            : ${result['cost_usd']:.4f}")
    print(f"  latency         : {result['latency_ms']}ms")
    print(f"  stop_reason     : {result['stop_reason']}")
    print()

    print("─" * 70)
    print("SONNET'S RAW TEXT (pre-tool reasoning)")
    print("─" * 70)
    if result["raw_text"]:
        print(result["raw_text"])
    else:
        print("(no text — model went straight to tool call)")
    print()

    if result["n_candidates"] > 0:
        print("─" * 70)
        print(f"CANDIDATES EMITTED ({result['n_candidates']})")
        print("─" * 70)
        for i, c in enumerate(result["candidates"], 1):
            print(f"  [{i}] {c.get('claim', '')[:200]}")
            print(f"      family         : {c.get('mechanism_family')} / "
                  f"{c.get('mechanism_subtype')}")
            print(f"      direction/magn : {c.get('predicted_direction')} / "
                  f"{c.get('predicted_magnitude')}")
            print(f"      cochrane       : {c.get('cochrane_frame')}")
            print(f"      prior          : {c.get('expected_outcome_prior')}")
            print(f"      novelty        : {c.get('novelty_vs_known')}")
            paps = c.get("synthesizes_paper_ids", [])
            print(f"      cites papers   : {paps}")
            gc = c.get("graveyard_conflicts") or []
            dc = c.get("doctrine_conflicts") or []
            if gc: print(f"      graveyard_conf : {gc}")
            if dc: print(f"      doctrine_conf  : {dc}")
            print()
    else:
        print("─" * 70)
        print("NO CANDIDATES PROPOSED — VERDICT INTERPRETATION")
        print("─" * 70)
        print("Sonnet stayed empty even on the gold substrate. This means")
        print("one of the following:")
        print()
        print("  (a) The production prompt is OVER-calibrated to 'prefer")
        print("      empty over weak' — needs softening.")
        print("  (b) The doctrine_signal events for OTHER/macro etc. are")
        print("      bleeding over and spooking Sonnet on UNRELATED")
        print("      families (VRP shouldn't be affected, but model may")
        print("      over-generalize).")
        print("  (c) The 3 gold papers aren't crafted strongly enough.")
        print()
        print("Look at SONNET'S RAW TEXT above to see WHY it didn't")
        print("propose. If it doesn't even mention VRP, the prompt is")
        print("the issue. If it mentions VRP but rejects it, examine")
        print("the rejection reasoning.")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
