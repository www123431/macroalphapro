"""scripts/seed_tier_c_idle.py — Tier C L3-2 "engine idle" seed.

Dispatches a handful of FactorSpecs end-to-end through the full
Tier C pipeline:

  C-2a gates  →  C-2b template  →  L3-2 self_doubt  →  C-2c emit

Purpose: populate data/research_store/events.jsonl with real
tier_c_auto-tagged factor_verdict_filed events so the new
/api/research/tier_c_verdicts endpoint has data to return,
and so we exercise the L3-2 module on live backtests (not just
the literal fixture used in scripts/verify_l3_2_self_doubt_live.py).

NOT idempotent — re-running produces additional events. Use the
--limit flag to cap how many SPECs to dispatch this run.

Cost: ~$0.04/spec for L3-2 Sonnet call. Template backtest is free.
Wall time: ~2-3 min per spec (full monthly backtest 1992-2024).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_seed_specs():
    """Three SPECs spanning GREEN/MARGINAL/RED expectations:
      1. GP/A vanilla — known REPLICATED GREEN
      2. reversal_1m — known short-term mean-reversion signal
      3. mktcap — pure size; expected RED in cross-sec L/S
    All run on the cross_sec_us_equities template (no new template
    dependencies; smoke-tested via parity suite)."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    base = dict(
        universe                = "us_equities_top_3000",
        date_range              = "1992-01:2024-12",
        rebal                   = "monthly",
        weighting               = "quintile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months          = 120,
        pit_audits              = ("lookahead",),
        cost_model              = "basic",
        extracted_ts            = "2026-06-08T00:00:00Z",
        model                   = "claude-sonnet-4-6",
        paper_original_window   = None,
        paper_reported_t        = None,
        universe_size           = None,
        n_buckets               = None,
        signal_lookback_m       = None,
        signal_skip_m           = None,
        vol_target_annual       = None,
        weighting_scheme_alt    = None,
    )
    return [
        ("PROFITABILITY", FactorSpec(
            hypothesis_id = "seed_gpa_2026_06_08",
            signal_kind   = "cross_sectional_rank",
            signal_inputs = ("compustat.funda.gp_at",),
            rationale     = ("Seed dispatch: Novy-Marx 2013 gross "
                              "profitability premium — known REPLICATED."),
            paper_original_window = "1963:2010",
            paper_reported_t      = 3.0,
            **{k: v for k, v in base.items()
               if k not in {"paper_original_window", "paper_reported_t"}},
        )),
        ("BEHAVIORAL", FactorSpec(
            hypothesis_id = "seed_reversal1m_2026_06_08",
            signal_kind   = "cross_sectional_rank",
            signal_inputs = ("crsp.msf.reversal_1m",),
            rationale     = ("Seed dispatch: Jegadeesh 1990 short-term "
                              "reversal — known anomaly, cost-sensitive."),
            **base,
        )),
        ("SIZE", FactorSpec(
            hypothesis_id = "seed_size_2026_06_08",
            signal_kind   = "cross_sectional_rank",
            signal_inputs = ("crsp.msf.mktcap",),
            rationale     = ("Seed dispatch: SMB / size factor — expected "
                              "RED in modern post-1990 sample."),
            **base,
        )),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3,
                          help="Max specs to dispatch (default 3)")
    parser.add_argument("--dry-run", action="store_true",
                          help="Show what would dispatch without "
                                "running templates or emitting")
    args = parser.parse_args()

    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )

    seeds = _build_seed_specs()[: args.limit]
    print(f"Seeding {len(seeds)} Tier C dispatch(es)"
            f"{' [DRY RUN]' if args.dry_run else ''}\n")

    results = []
    for i, (family, spec) in enumerate(seeds, 1):
        print(f"[{i}/{len(seeds)}] Dispatching "
                f"{spec.hypothesis_id} family={family} "
                f"signal={spec.signal_inputs[0]}...")
        try:
            out = dispatch_factor_spec(
                spec,
                family_hint   = family,
                spec_approved = True,
                dry_run       = args.dry_run,
            )
            tr = out.get("template_result") or {}
            verdict = tr.get("verdict", "?")
            refusal = out.get("refusal")
            disp_eid = out.get("dispatch_event_id") or "(no dispatch log)"
            verdict_eid = out.get("verdict_event_id") or "(no emit)"
            sd = out.get("self_doubt") or {}
            sd_summary = (f"conf={sd.get('confidence'):.2f} "
                            f"caveats={len(sd.get('caveats') or ())}") if sd else "(none)"
            if refusal:
                print(f"   REFUSED: {refusal.get('reason_code')} - {refusal.get('detail')}\n")
            else:
                print(f"   verdict={verdict}  dispatch_eid={disp_eid}  "
                        f"verdict_eid={verdict_eid}  self_doubt={sd_summary}\n")
            results.append({"hid": spec.hypothesis_id, "verdict": verdict,
                              "verdict_eid": verdict_eid,
                              "refusal": refusal})
        except Exception as exc:
            print(f"   [FAIL] {exc!r}\n")
            results.append({"hid": spec.hypothesis_id, "error": str(exc)})

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
