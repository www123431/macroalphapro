# MacroAlphaPro — Quantitative Factor Discovery Engine (MVP)

> An evidence-grounded factor discovery engine with audit-first multi-agent governance.
> Single-researcher scale, institutional-quality engineering. Pre-PhD portfolio.

**Quick links**
- 📄 [Technical paper draft](papers/engine_design_2026_06_01.md)
- 🧪 [Live closed-loop demo script](../scripts/demo_closed_loop.py)
- 🔬 [PFH Bayesian methodology](../engine/research/pfh/__init__.py)
- 🏛 [Audit ledgers](../data/research/) — every council run / tool call / PFH suggestion is git-tracked
- 🧰 [Source by layer](#source-organization)

---

## 1. The 90-second pitch

Most "AI for quant finance" GitHub projects are either:
- (a) **toy generators**: "GPT, write me an alpha" → produces plausible-looking output with no grounding
- (b) **wrapper backtests**: hooks a backtest engine to an LLM that decides which factor to test next

This project takes a third position:

> **The hard problem in factor research is not generating ideas. It is knowing when to stop.**

We accumulated a structured dataset of 35 labeled mechanism outcomes (6 GREEN
deployed sleeves + 2 YELLOW + 28 RED graveyard entries with documented failure
mechanisms) over ~6 months of single-researcher quant work. We then built an
engine that uses this dataset as informative Bayesian prior to rank candidate
factor proposals, with a multi-agent critique council that has per-critic
marginal-information-gain calibration.

**Headline differentiation:** every step of suggestion → council → pipeline →
outcome is recorded in append-only JSONL ledgers designed for SR 11-7 / OCC
Heightened Standards model-governance audit. Most "AI quant" demos cannot
pass model risk management review. This one is designed to.

## 2. What works (validated)

### 2.1 4-axis factor composition algebra

`factor = Compose(universe, signal_recipe, weighting, rebalance)`

Each axis is an independently-loadable YAML component. Any combination is
materialize-able into a return series via deterministic composition + sha256-
addressed caching.

**Cross-asset coverage** (Path B extension, 2026-06-01):

| Asset class | Universe | Coverage |
|---|---|---|
| US equity (full) | `equity_us_crsp_monthly` | 2249 stocks × 129 months |
| US equity (high-vol) | `equity_us_high_vol_monthly` | 742 stocks × 129 months |
| US equity (post-2018) | `equity_us_post2018_monthly` | 2249 stocks × 78 months |
| Cross-asset futures (17) | `futures_cross_asset_17_monthly` | FX + energy + metals + equity index + rates + grains × 228 months |
| FX futures (G3) | `futures_fx_g3_monthly` | GBP / EUR / JPY × 228 months |
| Commodity futures | `futures_commodity_8_monthly` | Energy + metals + grains × 228 months |
| Synthetic | `synthetic_equity_demo` | Test fixture |

Sample materialize outputs on real data (PFH-suggested, no human Python):

```
Spec                                                          Sharpe   AnnVol
─────────────────────────────────────────────────────────────────────────────
equity_us_high_vol × momentum_8_1 × rank_weighted_full         0.763    0.112
equity_us_crsp × momentum_12_1_vol_scaled × decile_ls_10       0.601    0.134
equity_us_crsp × momentum_8_1 × decile_ls_10                   0.573    0.242
equity_us_crsp × momentum_12_1 × decile_ls_10                  0.415    0.184
equity_us_post2018 × reversal_1m × decile_ls_10                0.301    0.208
futures_cross_asset_17 × momentum_12_1 × sign_then_vol_target  0.175    0.045  ← cross-asset TSMOM
equity_us_crsp × zscore_36mo × decile_ls_10                   -0.339    0.131  ← engine correctly assigns negative
futures_commodity × momentum_8_1 × sign_then_vol_target       -0.269    0.052  ← commodity momentum 2006-2024 was weak
```

**Honest scoping note**: the compose-spec cross-asset TSMOM (Sharpe 0.04-0.18)
scores lower than our DEPLOYED 5-leg TSMOM function-wrapper sleeve (Sharpe 0.62
net). The deployed sleeve has leg-structured risk-parity combine across
commodity / FX / rates / rates_xc / equity-index sub-strategies — sophistication
that the simple 4-axis DSL doesn't capture. Complex multi-leg combining stays
in function-wrapper specs by design; this is the right DSL boundary.

### 2.2 PFH — Probabilistic Factor Hypothesizer

Bayesian-prior factor ranking using the labeled mechanism dataset:

- **Beta-Binomial** conjugate model with empirical-Bayes hyperprior centered
  on overall base rate (18.6% vs literature's publication-biased ~65%).
- **Family alias resolution** bridges source-specific naming conventions —
  e.g. graveyard "forward-earnings information" ↔ library "earnings_underreaction".
  Worked example: posterior_mean shifted 0.47 → 0.39 after fix.
- **Constrained mode**: enumerates the Cartesian product of EXISTING axes
  minus already-tested combinations. Output is immediately materialize-able
  compose-spec YAMLs.
- **Open mode**: allows PLACEHOLDER axis refs for proposals that need human
  follow-up on axis definitions.
- **Diversity cap** prevents single-family dominance in top-K output.
- **Cousin penalty** (multiplicative, 0.85 per graveyard match) downweights
  proposals structurally similar to known REDs.

### 2.3 Per-critic marginal information gain

`ΔI(c) = accuracy(consensus_full) - accuracy(consensus_without_c)`

For the 3-agent council (architect + behavioral_theorist + empirical_devils_advocate),
each critic's marginal contribution to council-vs-pipeline alignment is computed
via counterfactual ablation, using the deterministic pipeline_v2 verdict as
ground truth. ΔI(c) ≥ 0.05 = "keep"; |ΔI(c)| ≤ 0.02 = "redundant"; ΔI(c) ≤ -0.02
= "hurting accuracy, review prompt".

To our knowledge this is the first application of marginal-information-gain
measurement to a financial multi-agent system. The DSPy-style multi-agent
literature treats agent outputs as ground-truth-unknown; finance has a
ground-truth available (pipeline_v2), so the measurement is tractable here
in a way it is not in general-purpose reasoning.

### 2.4 Audit-first design

Every step is in an append-only JSONL ledger:

| Ledger | Records |
|---|---|
| `ui_tool_calls.jsonl` | Every research-tool dispatch + args + result_hash + latency + caller |
| `council_runs.jsonl` | Every council critique + per-critic verdict + reflection action |
| `l4_iterations.jsonl` | Every L4 workflow + proposal + consensus + outcome + alignment |
| `critic_calibration.jsonl` | Per-(iteration, critic) rows for marginal-info-gain analysis |
| `pfh_suggestions.jsonl` | Every PFH run + candidates + scores + evidence chain |
| `chain_runs.jsonl` | Declarative chain executions |
| `proposed_intuition_rules.jsonl` | Auto-synthesized rule proposals + human review status |

## 3. What is infrastructure-ready (awaits validation)

Honest distinction. The following modules are architected, tested with mocked
dependencies, and ship in this repo — but they have not yet accumulated live
production data, so their effectiveness is not yet measured:

- **Council reflection round** (Pattern 6 structured review): built, tested
  with mocked Anthropic; awaits live (with-reflection, without-reflection)
  outcome pairs to validate that reflection improves verdicts.
- **L4 daily cron continuous discovery**: built on Temporal Schedule API,
  tested with mocked client; awaits live production runs.
- **Calibration feedback loop**: synthesizes proposed intuition rules from
  council_wrong clusters; min_cluster_size hyperparameter awaits ground-truth
  tuning.

This distinction is deliberately surfaced. "Working code" and "validated
behavior" are different claims; conflating them is the most common mistake
in AI-for-finance portfolios.

## 4. What this does NOT claim

- **Industrial-scale feature store**. We ship ~10 axis components.
  Chen-Zimmermann 2021 has 319 factors with infrastructure. Scaling to 300+
  axes is standard quant engineering investment, not a research question.
- **Causal identification**. Statistical association only; no IV / RD /
  Mendelian-analog instruments.
- **Alternative data depth**. WRDS-standard sources only (CRSP, Refinitiv,
  FRED). No Revelio / M Science / Yipit / Earnest integration.
- **Capacity modeling**. Cost model present; Bouchaud-Donier impact /
  Almgren-Chriss execution not addressed.
- **A "satiated" engine**. Full institutional discovery engines require ~50
  person-years; this is an MVP at single-researcher scale.

We make these statements because honest scoping is the load-bearing portfolio
signal in a market crowded with overclaims.

## 5. Live demo

The closed loop runs in one command:

```bash
python scripts/demo_closed_loop.py
```

What this does:
1. PFH reads 35 labeled mechanisms + ~10 axis components from disk.
2. Enumerates 58 untested (universe × signal × weighting) tuples.
3. Computes Beta-Binomial posterior for each with cousin-penalty + diversity.
4. Writes top-6 as compose-spec YAML stubs.
5. Materializes each against real CRSP data.
6. Prints a results table with Sharpe + posterior CI for each.

Expected output (representative, deterministic up to non-determinism in
materialize timestamps):

```
PFH base rate (used as prior centering): 0.186
Enumerated 58 untested combinations.

Top 6 after family-diversity cap:
  spec_id                                                    Sharpe    Post.CI
  reversal_1m × decile_ls_10                                  0.415    [0.05, 0.61]
  momentum_12_1 × rank_weighted                               0.314    [0.05, 0.61]
  reversal_1m × rank_weighted                                 0.254    [0.05, 0.61]
  momentum_12_1 × sign_then_vol_target_12pct                  0.145    [0.05, 0.61]
  zscore_36mo_residual × decile_ls_10                        -0.339    [0.05, 0.61]
  zscore_36mo_residual × rank_weighted_full                  -0.431    [0.05, 0.61]
```

**Interpretation:**
- Lehmann (1990) short-term reversal scores highest — well-documented in literature.
- Cross-sectional momentum × time-series weighting (signal/weighting mismatch)
  correctly scores low.
- Time-series z-score with no economic motivation correctly scores negative.

The engine penalizes mismatched and unmotivated combinations — it is not
producing random numbers.

## 6. Source organization

```
engine/
├── feature_store/        # L0: 4-axis composition + materializer + hash-cache
│   ├── composer.py       # Universe × Signal × Weighting → return series
│   ├── primitives.py     # rolling_return / xs_rank / vol_scale / etc.
│   └── registry.py       # spec discovery + schema validation
├── research/
│   ├── pfh/              # L1: Probabilistic Factor Hypothesizer
│   │   ├── catalog.py            # Load labeled mechanisms (lib + graveyard)
│   │   ├── bayesian.py           # Beta-Binomial posterior + credible intervals
│   │   ├── generator.py          # Open-mode candidate enumeration
│   │   ├── constrained_generator.py  # Closed-loop catalog-bounded enumeration
│   │   └── proposer.py           # Score + diversify + emit compose-spec YAMLs
│   ├── rbg/              # Research Brief Generator (markdown + LLM prose)
│   ├── agent_council.py          # 3-critic fan-out + reflection round
│   ├── critic_calibration.py     # Per-critic marginal-information-gain
│   ├── calibration_feedback.py   # Auto-propose intuition rules from council_wrong
│   ├── l4_workflow.py            # Temporal L4 discovery workflow
│   ├── l4_cron.py                # Continuous-background cron schedule
│   ├── research_chain.py         # Declarative DAG runner
│   ├── outcome_ledger.py         # Iteration outcome persistence
│   └── intuition_rules.py        # Senior-quant rule base
data/
├── feature_store/
│   ├── _universes/       # 4 universes (real CRSP + synthetic + subsets)
│   ├── _signal_recipes/  # 5 signal recipes
│   ├── _weightings/      # 3 weightings
│   ├── _specs/           # compose-specs (function + compose modes)
│   └── _computed/        # hash-addressed materialized outputs
├── research/
│   ├── mechanism_library/        # 13 deployed-sleeve YAMLs (6 GREEN)
│   ├── graveyard.json            # 24 RED with mechanism documentation
│   ├── intuition_rules.yaml      # 12 codified senior-quant rules
│   └── *.jsonl                   # 7 append-only audit ledgers
tests/                            # ~100 deterministic tests
docs/papers/                      # Technical paper draft
scripts/demo_closed_loop.py       # End-to-end live demo
```

## 7. Key numbers

| Metric | Value |
|---|---|
| Labeled mechanism dataset | 35 (6 GREEN + 1 YELLOW + 28 RED) |
| Axis components in catalog | 7 universes × 6 signals × 3 weightings = 126 possible |
| Universe asset classes covered | US equity (3 variants) + cross-asset futures (3 subsets: 17-instrument / FX / commodity) + synthetic |
| Already-tested compose specs | 2 (synthetic + real CRSP) |
| PFH-enumerable untested combinations | 124 |
| Real-data materialize time per spec | ~0.3s (cached: <50ms) |
| Determinstic tests | ~105 |
| Pre-commit drift checks | 66 + 4 audit hooks |
| Append-only audit ledger types | 7 |
| Intuition rules codified | 12 |
| Anthropic API calls in tests | 0 (all mocked) |

## 8. References

The paper draft cites all sources; quick highlights:

- Harvey-Liu-Zhu (2016), Hou-Xue-Zhang (2020) — factor zoo / multi-testing problem
- Chen-Zimmermann (2021) — Open Source Cross-Sectional Asset Pricing
- McLean-Pontiff (2016) — post-publication decay
- Gelman et al. (2013) — empirical Bayes methodology
- Khattab et al. (2024) DSPy, Yao et al. (2023) ReAct — multi-agent calibration

---

*This is a single-author MVP, Q2-2026. Built in ~50h on top of ~6 months of
prior quant research that produced the labeled mechanism dataset. The
distinguishing artifact is the dataset and its use as informative Bayesian
prior — not the engineering count.*
