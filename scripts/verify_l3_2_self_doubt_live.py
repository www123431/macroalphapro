"""scripts/verify_l3_2_self_doubt_live.py — Tier C L3-2 live sanity.

Calls the real Sonnet endpoint on a synthetic GP/A-shaped verdict
(no actual backtest run; just FactorSpec + TemplateResult literals
that match the locked baseline). Confirms:
  - LLM successfully invokes emit_self_doubt
  - Confidence + caveats land in the documented range
  - Returned dataclass is well-formed

Costs ~$0.04. Run manually after touching self_doubt prompt / schema.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import sys
from pathlib import Path

# Allow running from anywhere in repo
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _build_gpa_fixture():
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    spec = FactorSpec(
        hypothesis_id    = "hid_gpa_live",
        signal_kind      = "cross_sectional_rank",
        universe         = "us_equities_top_3000",
        date_range       = "1992-01:2024-12",
        signal_inputs    = ("compustat.funda.gp_at",),
        rebal            = "monthly",
        weighting        = "quintile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months   = 120,
        pit_audits       = ("lookahead",),
        cost_model       = "basic",
        rationale        = ("Novy-Marx 2013 profitability premium: gross "
                              "profits / total assets predicts cross-section "
                              "of returns out-of-sample."),
        extracted_ts     = "2026-06-08T00:00:00Z",
        model            = "claude-sonnet-4-6",
        # B-class
        paper_original_window = "1963:2010",
        paper_reported_t      = 3.0,
        universe_size         = None,
        n_buckets             = None,
        signal_lookback_m     = None,
        signal_skip_m         = None,
        vol_target_annual     = None,
        weighting_scheme_alt  = None,
    )

    metrics = {
        "sharpe":      0.67,
        "nw_t_stat":   3.57,
        "ann_return":  0.069,
        "ann_vol":     0.103,
        "n_months":    395,
        "avg_turnover": 0.17,
        "naive_verdict":       "GREEN",
        "cost_robust_verdict": "GREEN",
        "cost_stress": {
            "0bp":  {"sharpe": 0.70, "nw_t_stat": 3.69, "verdict": "GREEN"},
            "30bp": {"sharpe": 0.63, "nw_t_stat": 3.41, "verdict": "GREEN"},
            "60bp": {"sharpe": 0.57, "nw_t_stat": 3.14, "verdict": "GREEN"},
            "80bp": {"sharpe": 0.53, "nw_t_stat": 2.95, "verdict": "GREEN"},
        },
        "drawdown_naive": {
            "max_drawdown_pct":       -0.205,
            "max_underwater_months":  42,
            "calmar_ratio":           0.32,
        },
        "replication": {
            "status":           "REPLICATED",
            "our_t":            3.04,
            "paper_reported_t": 3.0,
            "t_gap":            0.044,
            "window":           "1992:2010",
        },
    }
    tr = TemplateResult(
        verdict          = "GREEN",
        summary          = ("GP/A (Novy-Marx 2013) 1992-2024 monthly long-"
                              "short top/bottom quintile. Sharpe 0.67, NW-t "
                              "3.57, REPLICATED vs paper (gap 0.044)."),
        metrics          = metrics,
        artifacts        = {},
        template_version = "v1.1_2026-06-08",
    )
    return spec, tr


def main():
    from engine.agents.strengthener.self_doubt import assess_self_doubt
    spec, tr = _build_gpa_fixture()
    print("Calling Sonnet for self-doubt assessment on GP/A GREEN...")
    sd = assess_self_doubt(
        spec, tr,
        family_hint     = "PROFITABILITY",
        n_trials_family = 3,
    )
    if sd is None:
        print("FAIL: assess_self_doubt returned None")
        sys.exit(1)
    print()
    print("RESULT")
    print("======")
    payload = _dc.asdict(sd)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()
    # Soft assertions — calibration not exact, but should land in band
    assert 0.30 <= sd.confidence <= 0.85, \
        f"confidence {sd.confidence} outside expected 0.30-0.85 band"
    assert 2 <= len(sd.caveats) <= 5, \
        f"caveats len {len(sd.caveats)} outside 2-5"
    print(f"PASS: confidence={sd.confidence:.2f} caveats={len(sd.caveats)} "
            f"methodological_concerns={len(sd.methodological_concerns)} "
            f"suspicious_metrics={len(sd.suspicious_metrics)}")


if __name__ == "__main__":
    main()
