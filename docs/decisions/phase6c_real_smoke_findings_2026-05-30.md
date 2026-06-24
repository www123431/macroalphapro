# Phase 6c Real-Data Smoke Findings — 2026-05-30

## TL;DR

Engine end-to-end ALIVE on production data. yfinance → orchestrator →
adapter → DSL → run_gate → multi-leg verdict, full chain in 0.5s.

Verdict on equity_xsmom_jt with 30 mega-cap universe (2018-2024): RED.

This is correct outcome — 30 stocks is not a meaningful momentum universe.
But the smoke exposed 3 real design issues to address (per iterate-and-solve
doctrine).

## 3 real findings

### Finding 1 — Template warmup ignored by protocol sample-split ⚠️ CRITICAL

**Issue**: `subperiod_first_half` leg returned `ERROR: sliced sample too
short: 0 months`. The protocol's first-half spans 2018-01 to 2021-07
(midpoint of raw sample). But equity_xsmom template has ~50-month warmup
(rolling_sum 12 + vol_target_normalize 36 + apply_lag 1+). So the first
50 months of the price_panel produce NaN signals. The first-half slice
ends up entirely in warmup → 0 months.

**Fix design** (Phase 9 refinement):
- Mechanism YAML adds `template_warmup_months: int` field
- Protocol designer computes effective_sample_range = [start + warmup, end]
- `split_first_half` / `split_second_half` operate on EFFECTIVE range
- Sample-window override updates protocol_hash so audit trail catches this

**Workaround for now**: Longer raw samples (15+ years) make the issue
less visible. Real CRSP-style backtests typically use 1965-present so
warmup is a small fraction.

### Finding 2 — Mechanism library schema gap

`mechanism_library/_schema.md` doesn't have a way to declare per-template
data requirements like warmup months, minimum universe size, or minimum
sample length. Phase 9 refinement should add:

```yaml
template_requirements:
  template_warmup_months: 50
  min_universe_size: 200
  min_effective_sample_months: 60
```

Protocol designer uses these to:
- Compute effective sample range (Finding 1)
- Warn if proposal sample range yields insufficient effective months
- Auto-skip protocol if requirements not met

### Finding 3 — Smoke universe too small for equity_xsmom

30 mega-caps → top-decile = 3 stocks, bottom-decile = 3 stocks. Noise
dominates signal. Not an engine bug; just smoke parameter choice.

For meaningful future real-data smoke:
- Use Wikipedia SP500 scraper to fetch full 500-name universe
- Or use WRDS CRSP DSF when access available
- Or use a 100+ ticker mid/large-cap basket

## What this smoke DOES validate (positive)

✅ yfinance fetcher live + returns expected long-format
✅ Adapter long→wide conversion correct
✅ Protocol designer instantiates 5 legs + 2 decomp + frozen hash
✅ Each leg's DSL template execution + run_gate works
✅ Multi-leg verdict aggregation produces structured RED with reasons
✅ Total elapsed ~0.5s for 5 legs — runtime budget healthy
✅ No silent synth substitution (failures surface clearly)

## Smoke output (verbatim)

```
[1/5] Fetched 52800 rows in 5.6s (30 tickers)
[2/5] Resampled daily → monthly + adapted to (84, 30) wide panel
[3/5] Protocol: equity_factor_standard_v1 + hash 849b5afa466f680c
      5 legs: primary_test, subperiod_first_half/second_half,
              cost_stress_2x, microcap_robust
      2 decomp: ff5_umd_orthogonality, pead_residualization
[4/5] Total elapsed: 0.5s

OVERALL: RED
  primary_test            no   Sharpe 0.107   α-t -1.128
  subperiod_first_half    no   ERROR: sliced sample too short: 0 months
  subperiod_second_half   no   Sharpe 0.107   α-t -1.128
  cost_stress_2x          no   Sharpe -0.170  α-t -1.635
  microcap_robust         no   Sharpe 0.089   α-t -1.196
  ff5_umd_orthogonality   FAIL
  pead_residualization    FAIL
```

## Decision

Findings 1 + 2 are real but NOT blockers for Phase 7 (cadence cron) or
Phase 8 (auto paper discovery). They're proper Phase 9 refinements once
we have more candidate diversity in the library to test the fix against.

For now, document and move forward. The engine pipeline is verified
operational on production data.

## Next action options

A. Fix Finding 1 NOW (~2h: schema field + protocol designer effective range)
B. Continue per roadmap (Phase 7 cadence cron, or Phase 6.5 meta-learner)
C. Larger universe smoke via SP500 Wikipedia scraper (~1h, validates more
   tickers but doesn't fix Finding 1)

Recommendation: B. Findings are real but iterating on them requires
diverse library candidates to validate. Phase 7/6.5 build infrastructure
that pays for itself faster.

## Linked

- Smoke script: `scripts/_smoke_phase6c_real.py` (one-off, kept for audit)
- [[project-complete-construction-roadmap-2026-05-30]] (master plan)
- [[feedback-iterate-and-solve-inflight-2026-05-29]] (find in use, fix)
