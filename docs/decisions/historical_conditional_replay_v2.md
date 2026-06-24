# Historical Conditional Replay v2

**Status**: in progress 2026-05-04
**Type**: capability extension to P-AUDIT v1 L5 panel (clarification, no new hypothesis)
**Triggered by**: supervisor screenshot showing L5 "n=0 < 5, insufficient data" — rationally true (paper trading early) but UX-empty. Question raised: "能不能用以前发生的事情来给出 hit rate?"

## Problem

L5 conditional history (P-AUDIT decision panel) reads from project-internal `DecisionLog` rows that are verified=True. In paper-trading early period, almost every (sector × direction × regime) combination has n=0–5. The panel section displays "insufficient data" and the supervisor gets no historical signal at all.

User asked whether external / pre-existing market data could supply hit rate. The senior-quant answer is **yes** — Asness-Moskowitz-Pedersen 2013 *Value and Momentum Everywhere* and Moskowitz-Ooi-Pedersen 2012 *Time Series Momentum* both publish exactly this style of conditional empirical study. yfinance has 15+ years of sector ETF + SPY benchmark + VIX coverage.

**But** displaying historical hit rate in supervisor's decision panel introduces *behavioral anchoring risk* (Ariely-Loewenstein-Prelec 2003) — exactly the vector that caused `narrative_overlay` rejection 2026-05-02. We must wire it carefully.

## Two-step design (v2 = ex-ante MSM)

### Backend: `engine/historical_replay.py`

```python
def get_historical_conditional_hit_rate(
    ticker:           str,
    direction:        str,            # "long" / "short"
    target_regime:    str,            # "risk-on" / "neutral" / "risk-off"
    horizon_days:     int = 21,
    lookback_years:   int = 15,
    regime_proxy:     str = "msm_walk_forward",   # or "vix_simple"
    benchmark_ticker: str = "SPY",
) -> dict
```

Pipeline:
1. yfinance daily closes for `ticker` + `benchmark_ticker` over `lookback_years` (cached locally)
2. compute monthly TSMOM signal: `sign(12-month return − 1-month return)` at each month end
3. compute regime label series:
   - **`vix_simple`**: VIX > 30 → risk-off / VIX < 15 → risk-on / else neutral. Fast but ex-post smoothed.
   - **`msm_walk_forward`** ⭐ v2: reuse `engine.regime.get_regime_series(dates)` which calls `get_regime_on(as_of=t, train_end=t)` per t — proper walk-forward MSM with no look-ahead.
4. find condition-match dates: `signal_t == direction AND regime_t == target_regime`
5. for each match `t`, compute active return `(ticker_{t+H}/ticker_t − 1) − (SPY_{t+H}/SPY_t − 1)`
6. aggregate: `n_obs / hit_rate / mean / median / std / 5%-95% percentile`
7. return dict + caveats list

### UI: dual-source render in L5 tab

```
HISTORY tab content:

[PROJECT EX-ANTE]                                     ← Always rendered first
  same sector × direction × regime: n=0 < 5
  insufficient — wait for data accumulation

[HISTORICAL REPLAY · ex-post bias caveat applied]    ← Only render if v1 says insufficient
  ticker=USO, regime proxy=msm_walk_forward, lookback=15y
  n_obs=42 · hit_rate=58% · mean active +0.85% · 5-95: -3.2/+5.1%
  ⚠ for context only — NOT a signal layer
  ⚠ strategy params pre-registered; cannot be retro-tuned from this view

[DISAGREEMENT FLAG]                                   ← When project ex-ante and historical disagree
  none yet (project n=0)
```

## Anti-anchoring guards (4 hard rules)

Per [Ariely-Loewenstein-Prelec 2003](https://academic.oup.com/qje/article/118/1/73/1917002) on coherent arbitrariness, displaying a number anchors decisions even with caveats. To minimize this:

1. **No sort by hit rate** — sectors/positions render in pre-existing order
2. **Project ex-ante always rendered first** — ex-post historical only as fallback when project n < 5
3. **Disagreement flag prominent** — when historical hit rate trend contradicts project trend (or current quant signal direction), surface "DISAGREEMENT" badge instead of hiding
4. **Caveat language explicit and consistent**: every render of historical replay must include
   > "for context only — not a signal; strategy params are pre-registered and cannot be retro-tuned from this view"

## Hard invariant (added to project)

> **Historical replay data is pre-decision context only. It MUST NOT trigger:**
> 1. Strategy parameter changes outside the S3 SpecRegistry amendment workflow
> 2. Selective approval / rejection patterns biased by displayed hit rate
> 3. Backtest re-running with different conditional filters
>
> Each unique conditional query counts toward EFFECTIVE_N_TRIALS multiple-testing budget per `feedback_spec_power_analysis.md`. The query log is auditable via the agent_runs table (input_params hash).

## Academic anchors

- **Asness, Moskowitz, Pedersen (2013)** *Value and Momentum Everywhere*, JF — conditional analysis methodology
- **Moskowitz, Ooi, Pedersen (2012)** *Time Series Momentum*, JFE — TSMOM hit rates by sector
- **Hamilton (1989)** *A New Approach to the Economic Analysis of Nonstationary Time Series*, Econometrica — regime switching
- **Diebold, Lee, Weinbach (1994)** — walk-forward filtered prob (the ex-ante caveat)
- **Ariely, Loewenstein, Prelec (2003)** *Coherent Arbitrariness*, QJE — anchoring even with caveats
- **López de Prado (2018)** *Advances in Financial Machine Learning* §11 — conditional event study methodology
- **Cochrane (2011)** *Discount Rates*, JF — conditional posterior framework

## Out of scope

- **Backtest replay from project's own BacktestRun**: deferred. Project's TSMOM baseline was self-falsified by S1 multi-window analysis (2026-05-03); using it for hit rate would re-import that noise.
- **External data sources beyond yfinance** (Bloomberg, FactSet, Refinitiv): no subscription, premature.
- **Cross-asset / global universe replay** (FX, bonds, commodities): future work tier 2.
- **LLM-summarized historical narrative**: explicitly excluded — Layer-1 LLM not allowed to write retrospective interpretation that supervisor sees.

## Verify (v2.3 facets)

- backend returns expected shape on real ticker (USO, 15y)
- vix_simple and msm_walk_forward both produce non-empty distributions
- conditional filter actually excludes wrong-regime dates
- AppTest cold + seeded panel renders dual-source section without exception
- every rendered HISTORICAL REPLAY block contains caveat text (string check)
- 0 LLM call in backend (model param does not exist in signature)
