# An Evidence-Grounded Factor Discovery Engine with Audit-First Agent Governance

**Author:** Xizhe Zhang
**Date:** 2026-06-01
**Status:** Draft, not peer-reviewed
**Code:** github.com/zhangxizhe/macroalphapro (private)

---

## Abstract

We describe a prototype quantitative factor discovery engine that combines three
elements not usually deployed together in open-source quant tooling:
(1) a declarative 4-axis factor composition algebra
(universe × signal recipe × weighting × rebalance) materialized via reproducible
YAML specs with sha256-addressed caching;
(2) a Probabilistic Factor Hypothesizer (PFH) that uses the researcher's own
labeled mechanism history — 35 hand-curated outcomes including 28 RED graveyard
entries with mechanism-level failure documentation — as informative
Beta-Binomial prior to rank candidate factor proposals;
(3) a multi-agent critique council with per-critic marginal-information-gain
calibration, structured reflection round, and human-gated rule synthesis from
council disagreements.

The system is positioned as an MVP demonstrating institutional-quality
engineering on a single-researcher scale rather than as a competitor to
production hedge-fund infrastructure. Our principal contributions are
methodological:

  - An **anti-publication-bias factor labelset** that, in information-theoretic
    terms, complements the Chen-Zimmermann 2021 Open Source Asset Pricing
    library by providing class-balanced GREEN/RED labels with mechanism-level
    failure annotation — addressing the Harvey-Liu-Zhu 2016 multi-testing
    concern at the dataset level rather than the test level.
  - A **self-prior Bayesian hypothesis generator** that, to our knowledge,
    is the first open-source implementation of meta-learning over a single
    researcher's structured RED/GREEN history.
  - A **per-critic marginal-information-gain measure** for multi-agent
    financial reasoning systems, using deterministic pipeline outcomes as
    ground truth — connecting the DSPy-style multi-agent calibration
    literature (Khattab et al. 2024) to financial domain ground-truth
    availability.
  - **Audit-first agent governance**: every step of the suggestion → council →
    pipeline → outcome loop is recorded in append-only JSONL ledgers,
    designed for SR 11-7 / OCC Heightened Standards compatibility.

## 1. Motivation

The empirical factor-zoo literature suffers severe publication bias
(Harvey-Liu-Zhu 2016; Hou-Xue-Zhang 2020 show ~65% non-replication of
published anomalies). Bayesian inference over the published factor set
therefore produces systematically optimistic posteriors for "what makes
an anomaly real."

Practitioners working on a portfolio of factor research accumulate a
private dataset of negative outcomes that, in principle, could correct
this bias. In practice, this dataset is unstructured — Slack threads,
memory files, dead notebooks — and decays with researcher turnover.

This work asks: what would a single-researcher quantitative research
environment look like if all negative outcomes were as structured as
positive ones? And if such a labelset existed, could a factor
discovery engine use it as informative Bayesian prior to rank new
hypotheses?

## 2. System architecture

The system is a 7-layer factor discovery loop:

  L0. **Data primitives** — cached parquet files (CRSP, WRDS futures
      / FX / rates / commodities settlements, etc.)
  L0b. **4-axis composition algebra** — declarative YAML axes
      (universes, signal recipes, weightings, rebalance) with a sha256-
      addressed materializer that turns any 4-tuple into a return series.
      Implemented in `engine/feature_store/`.
  L1. **Hypothesis generation** — Probabilistic Factor Hypothesizer
      (`engine/research/pfh/`). Generates and ranks candidates either in
      `open` mode (allowing PLACEHOLDER axis references for human follow-up)
      or `constrained` mode (Cartesian product of EXISTING axes minus
      already-tested combinations).
  L2. **Pre-flight filtering** — graveyard query + intuition rules +
      cousin-warning surfacing. Each PFH candidate carries an explicit
      evidence chain (`derived_from`, `cousin_warnings`, etc.).
  L3. **Validation pipeline** — `candidate_pipeline_v2` (20 deterministic
      nodes including deflated Sharpe, paired bootstrap, cosine vs book,
      cost-aware filter, role-aware acceptance).
  L4. **Council critique** — 3-agent fan-out (architect / behavioral
      theorist / empirical devil's advocate) with optional bounded
      reflection round (Pattern 6 structured review, not Pattern 5
      autonomous debate).
  L5. **Deployment + monitoring** — SLM 3-layer validator + capital ramp;
      decay sentinel + Ledoit-Wolf shrinkage risk forecast.
  L6. **Knowledge accumulation** — append-only JSONL ledgers for tool
      calls (`ui_tool_calls.jsonl`), council runs (`council_runs.jsonl`),
      L4 iterations (`l4_iterations.jsonl`), critic calibration
      (`critic_calibration.jsonl`), PFH suggestions
      (`pfh_suggestions.jsonl`), chain runs (`chain_runs.jsonl`).

## 3. The anti-publication-bias labelset

### 3.1 Construction

Our labeled mechanism dataset comprises 35 observations:

  - **6 GREEN** (deployed sleeves from `data/research/mechanism_library/`):
    post_earnings_drift, post_earnings_drift_pit_sn, cross_asset_carry,
    time_series_momentum, crisis_hedge_tlt_gld, tail_hedge_put_spread.
  - **1 YELLOW** (deployed but broken, currently being replaced):
    mom_hedge_overlay.
  - **28 RED** (24 from `graveyard.json` + 4 library cousin_anchor
    markers): includes China A-share PEAD, management-guidance drift,
    cross-sectional momentum (post-decay), idiosyncratic volatility,
    transcript NLP signals, multi-frequency TSMOM, and 22 others, each
    with a documented failure mechanism (turnover-killed reversal,
    post-publication arbitrage, regime hostility, look-ahead leakage,
    etc.).

Plus a continuously growing stream of (council_verdict, pipeline_outcome)
pairs from the L4 daily cron workflow.

### 3.2 Comparison to Chen-Zimmermann (2021) Open Source Asset Pricing

Chen-Zimmermann 2021 catalogs 319 published equity factors. While larger
in count, the CZ dataset is positive-biased by construction: a factor
typically enters the catalog only after being published, and Hou-Xue-Zhang
2020 demonstrate ~65% non-replication of published factors.

For Bayesian inference about P(success | family, role, market), the
publication-bias-corrected information content of CZ's 319 factors is
substantially lower than its raw count suggests. Negative outcomes —
factors that failed empirical testing without being published — are
absent from the literature dataset.

Our 35-label set is smaller in raw count but:

  - **Class-balanced**: ~5:1 RED:GREEN, versus literature's heavily
    GREEN-biased distribution.
  - **Mechanism-level failure annotation**: each RED carries a
    documented failure cause ("turnover-killed reversal in A-shares",
    "post-SOX decay", "regime hostility", etc.) rather than the binary
    replicates/doesn't of CZ.
  - **Single-framework**: all labels generated by a consistent research
    framework, reducing the heterogeneity noise that affects multi-paper
    meta-analyses.
  - **Live**: the cron L4 loop appends new (council, outcome) pairs
    daily, with a clear human-vs-PFH provenance bit so calibration
    feedback does not contaminate the prior.

We do not claim our dataset is universally superior. We claim it is
informationally COMPLEMENTARY to CZ — useful for the "what makes a
factor fail" question that publication-biased literature cannot answer.

## 4. Self-prior Bayesian hypothesis generation

The Probabilistic Factor Hypothesizer (PFH) applies a Beta-Binomial
conjugate model with empirical-Bayes hyperprior centered on the overall
base rate:

```
p ~ Beta(α₀, β₀)
α₀ = 1 + base_rate × w
β₀ = 1 + (1 - base_rate) × w
```

with prior strength w = 4 pseudo-observations. Per-family posterior:

```
p | data ~ Beta(α₀ + n_green + 0.5·n_yellow,
                 β₀ + n_red + 0.5·n_yellow)
```

YELLOW outcomes split evenly into both directions on the grounds that
"deployed but broken" carries information in both — what worked enough
to deploy and what broke enough to retire.

A multiplicative cousin penalty (0.85 per RED warning, floor 0.05)
reduces the final ranking score for candidates structurally similar to
known graveyarded mechanisms.

Crucially, the model uses N=35 labels (after alias resolution). At this
scale, logistic regression or ML classifiers would overfit
catastrophically; the Gelman-style Bayesian-with-weak-informative-prior
approach is the only honest treatment. All posteriors are reported
with [5th, 95th] credible intervals; point estimates alone are
intentionally suppressed.

### 4.1 Family alias resolution

A critical engineering subtlety: the source schemas use different
naming conventions ("forward-earnings information" in graveyard vs
"earnings_underreaction" in library). A naive snake-case normalization
leaves these in separate Bayesian cells, producing dramatically
different (and incorrect) posteriors.

We hand-curate a `_FAMILY_ALIASES` table that bridges semantically-
identical families. After alias resolution, the earnings-information
family contains 2 GREEN + 6 RED → posterior_mean 0.39 (CI [0.13, 0.69]),
versus the pre-alias spurious 0.47 (CI [0.20, 0.75]). This 8-point
shift is load-bearing for downstream council decisions.

### 4.2 Closed-loop materialization

PFH's `constrained` mode enumerates the Cartesian product of EXISTING
universe × signal × weighting axes minus already-tested combinations,
producing immediately materialize-able compose-spec YAMLs. The
composer's hash-addressed caching ensures byte-identical reruns 5 years
from now if the source code and input data are unchanged.

### 4.3 Sample run

A live PFH run against the system's current state (catalog: 4 universes
× 5 signals × 3 weightings = 60 possible factors; 2 already tested,
58 untested):

  - 58 candidates enumerated.
  - Top 6 (after family-diversity cap = 2 per family):
    * reversal_1m × decile_ls_10: Sharpe 0.415 ✓
      (matches Lehmann 1990 / Khang-Garcia 2024 range)
    * momentum_12_1 × rank_weighted: Sharpe 0.314
    * reversal_1m × rank_weighted: Sharpe 0.254
    * momentum_12_1 × sign_then_vol_target: Sharpe 0.145
      (mismatched: cross-sectional signal × time-series weighting
       — engine correctly penalizes)
    * zscore_36mo_residual × decile_ls: Sharpe -0.339
      (no economic motivation — engine correctly assigns negative)
    * zscore_36mo_residual × rank_weighted: Sharpe -0.431

The economic interpretation matters more than the numerical ranking:
mismatched and unmotivated combinations score appropriately low, while
well-motivated combinations land in literature-expected ranges. This
distinguishes the engine from generative AI tools that produce
plausible-looking but unmoored output.

## 5. Per-critic marginal information gain

For the council, we compute per-critic marginal information gain via
counterfactual ablation: for each iteration, we rebuild consensus using
N-1 critics (excluding each critic in turn) and measure the change in
council-vs-pipeline alignment.

Specifically, for critic c:

```
ΔI(c) = accuracy(consensus_full) - accuracy(consensus_without_c)
```

where accuracy is computed against the deterministic pipeline_v2
verdict as ground truth. ΔI(c) ≥ 0.05 indicates the critic adds material
information; |ΔI(c)| ≤ 0.02 indicates redundancy with other critics;
ΔI(c) ≤ -0.02 indicates the critic is actively hurting calibration.

To our knowledge, this is the first application of marginal-information-
gain measurement to a financial multi-agent system. The multi-agent
calibration literature (Khattab et al. 2024 "DSPy"; Yao et al. 2023
"ReAct") generally treats agent outputs as ground-truth-unknown, since
in their domains there is no deterministic outcome to compare against.
Our domain has one: the pipeline_v2 verdict. This makes per-critic
calibration tractable in finance in a way that it is not in general-
purpose multi-agent reasoning.

The companion pairwise-critic-agreement measurement flags excessive
redundancy: agreement > 85% indicates the ensemble is paying 2× LLM
cost for ~1× information.

## 6. Audit-first design

Every step in the loop emits an append-only JSONL row:

  - `data/research/ui_tool_calls.jsonl` — every research-tool dispatch
    with args + result_hash + latency + caller identity.
  - `data/research/council_runs.jsonl` — every council critique with
    per-critic verdict + reflection action + tool call summary.
  - `data/research/l4_iterations.jsonl` — every L4 workflow with
    proposal + council consensus + (optional) pipeline outcome +
    verdict_alignment classification.
  - `data/research/critic_calibration.jsonl` — per-(iteration, critic)
    rows for per-critic accuracy and marginal-information-gain
    computation.
  - `data/research/pfh_suggestions.jsonl` — every PFH run with
    candidates, scores, evidence chain.
  - `data/research/chain_runs.jsonl` — declarative chain executions.
  - `data/research/proposed_intuition_rules.jsonl` — calibration-
    feedback-loop-generated rule proposals with human review status.

The intent is SR 11-7 / OCC Heightened Standards compatibility: an
auditor 5 years hence can ask "why did this candidate get a REJECT
verdict on 2026-06-01" and receive a full reconstruction from the
ledgers.

## 7. Limitations

We are explicit about what this work does and does not establish:

### What is validated:

  - The 4-axis composition algebra works on real data
    (CRSP 2249 stocks × 129 months, materialized Sharpe 0.41 for
    standard 12-1 momentum, falling in McLean-Pontiff 2016 post-decay
    range).
  - PFH constrained mode produces 58 untested compose-spec proposals
    on the current catalog, all immediately materialize-able with no
    human Python authoring.
  - 100+ deterministic tests across the agent layer + feature store +
    PFH.

### What is infrastructure-ready but awaits live-run validation:

  - Council reflection round — built and tested with mocked LLM, but
    no production accumulation of (with-reflection, without-reflection)
    outcome pairs yet.
  - L4 daily cron continuous discovery — built and tested with mocked
    Temporal client, but no live production cron runs yet.
  - Calibration feedback loop — synthesizes proposed intuition rules
    from council_wrong clusters, but min_cluster_size threshold is
    a hyperparameter not yet validated against ground-truth.

### What this work does NOT address:

  - **Industrial-scale feature store**. We ship ~10 axis components;
    Chen-Zimmermann 2021 has 319 factors with industrial replication
    infrastructure. Scaling to 300+ axes is standard quant engineering
    investment, not a research question.
  - **Causal identification**. Our pipeline is statistical association;
    causal inference (instrumental variables, regression discontinuity,
    Mendelian randomization analogues) is not addressed.
  - **Alternative data depth**. We use WRDS-standard sources (CRSP,
    Refinitiv futures, FRED). Institutional alt-data sources (Revelio
    Labs, M Science, Yipit, Earnest) are not integrated.
  - **Capacity modeling**. We have a cost model but not the
    Bouchaud-Donier impact model needed for capacity-adjusted alpha.
  - **A "satiated" engine**. A full institutional factor discovery
    engine requires ~50 person-years of investment. This work is an
    MVP demonstrating engine-level thinking, not a production system.

## 8. Related work

  - **Factor zoo and multi-testing**: Harvey-Liu-Zhu (2016) "...and the
    Cross Section of Expected Returns"; Hou-Xue-Zhang (2020)
    "Replicating Anomalies".
  - **Reproducible factor research**: Chen-Zimmermann (2021)
    "Open Source Cross-Sectional Asset Pricing".
  - **Multi-agent LLM systems**: Khattab et al. (2024) "DSPy";
    Yao et al. (2023) "ReAct". Our per-critic marginal-information-gain
    measure is, to our knowledge, the first financial-domain
    application using deterministic pipeline output as ground truth.
  - **Calibration**: Gelman et al. (2013) "Bayesian Data Analysis"
    Chapter 5 (empirical Bayes hyperprior).
  - **Decay and arbitrage**: McLean-Pontiff (2016) "Does Academic
    Research Destroy Stock Return Predictability".

## 9. Conclusion

We have described an evidence-grounded factor discovery engine prototype
with four specific contributions: a 4-axis composition algebra, a
self-prior Bayesian hypothesis generator, per-critic marginal-information-
gain calibration, and audit-first agent governance. The engine
demonstrably produces economically interpretable output on real CRSP
data and is positioned as an MVP for institutional-quality engineering
at single-researcher scale.

The system's principal limitation is the small (N=35) labeled mechanism
dataset; this is also its principal contribution, since the dataset is
class-balanced and mechanism-level-annotated in ways that publication-
biased literature datasets are not.

We do not claim this is a "satiated" factor discovery engine. We do
claim it is the first open-source implementation we are aware of that
treats the negative-result documentation problem as a first-class data
engineering concern, and uses the resulting dataset as informative
Bayesian prior for hypothesis generation.

---

*Code: see `engine/feature_store/`, `engine/research/pfh/`,
`engine/research/rbg/`, `engine/research/agent_council.py`,
`engine/research/critic_calibration.py`, and `tests/` in the repo.
Total: ~100 deterministic tests; ~50h engineering Q2-2026.*
