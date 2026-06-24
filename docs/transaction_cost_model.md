# Transaction Cost Model

| Field | Value |
|---|---|
| Status | Documented 2026-05-07 (P1.1) — unification shipped 2026-05-07 (P1.2) |
| Audience | Supervisor / interviewer asking "how do you model transaction costs?" |

> **Honest framing**: project has 3 cost contexts (backtest / live / research baseline), not 1 unified model. This doc explains why each is appropriate in its context, plus the upcoming P1.2 unification work.

---

## 1. The 3 contexts

| Context | Code location | Model | Justification |
|---|---|---|---|
| **Backtest (production)** | `engine/backtest.py:_atr_transaction_cost` | **ATR-based dynamic**: `cost_i = \|Δw_i\| × max(3bp, std_14d_i × 0.15)` | Vol-aware spread approximation; floor at 3bp for liquid US ETFs; tested in `_TC_FLOOR_BPS / _TC_VOL_SCALE / _TC_ATR_WINDOW` constants |
| **Live trade generation** | `engine/portfolio_tracker.py:417`, `daily_batch.py:731`, `memory.py:5942` (all delegate to `engine/cost_model.py`) | **flat 10bp** (vol-aware optional): `cost_bps = compute_cost_bps(\|Δw\|)` → `\|Δw\| × 10` when no return window provided | Diagnostic only — applied at trade-write time for visibility; **not used in NAV computation** (NAV uses real daily price returns). Unified P1.2 (2026-05-07). |
| **B++ Mass FDR research** | `engine/b_plus_search.py:44` | **flat 13bp round-trip**: `8bp slippage + 5bp spread` | Conservative single-number baseline used for the 40-spec systematic search; documented in `b_plus_prod_migration_2026-05-05` decision |

## 2. Why 3 contexts not 1

**Backtest** needs vol-conditioned cost because high-vol regimes have wider spreads. The ATR-based model captures this empirically (15% of daily vol, floor 3bp) — calibrated against historical ETF spread data (typical liquid US ETFs run 2-8bp spread, 30-50bp during stress).

**Live** trades currently use flat 10bp because:
1. cost_bps is a **diagnostic field** in the trade ledger, not P&L input
2. Real P&L = NAV(t+1) - NAV(t) - flows; uses actual market prices, not modeled cost
3. So cost_bps is a "what we'd budget" estimate, not actual realized cost

**B++ research** uses 13bp flat because the 40-spec FDR test needs a single-number cost assumption to compare strategies fairly. ATR-based would let high-vol strategies appear cheaper; flat keeps comparison clean.

## 3. Empirical reference (why these numbers)

| Source | Typical ETF spread | Notes |
|---|---|---|
| ICI 2024 ETF Liquidity Study | 2-8 bp average liquid; 20-40 bp small-cap | matches our 3bp floor + dynamic model |
| Frazzini-Pedersen 2014 BAB paper | Used 0-30bp depending on size | their universe similar to ours |
| AQR practitioner notes | 5-15 bp typical for monthly rebal ETF strategies | our 10bp flat is mid-range |

So both 10bp flat and ATR-based dynamic are within literature range.

## 4. P1.2 unification — shipped 2026-05-07

**Goal achieved**: one shared `engine.cost_model.compute_cost_bps()` is now the canonical entry point for the 3 live sites. Backtest path retains its sector-aware multi-leg `_atr_transaction_cost` (different signature — operates on dict-of-weight-deltas with a multi-sector return window — and is intentionally kept separate; both modules share the same constants `TC_FLOOR_BPS / TC_VOL_SCALE / TC_ATR_WINDOW`).

**Implementation** (engine/cost_model.py):
```python
def compute_cost_bps(
    weight_delta: float,
    daily_ret_window: pd.Series | None = None,
    floor_bps: float = TC_FLOOR_BPS,         # 3.0
    vol_scale: float = TC_VOL_SCALE,         # 0.15
    flat_fallback_bps: float = LIVE_FLAT_COST_BPS,  # 10.0
) -> float:
    abs_delta = abs(weight_delta)
    if daily_ret_window is None or len(daily_ret_window) < 5:
        return abs_delta * flat_fallback_bps
    vol_daily = float(daily_ret_window.dropna().tail(TC_ATR_WINDOW).std())
    if vol_daily != vol_daily or vol_daily <= 0:    # NaN / non-positive guard
        return abs_delta * flat_fallback_bps
    half_spread_bps = max(floor_bps, vol_daily * vol_scale * 10_000.0)
    return abs_delta * half_spread_bps
```

**Refactored sites** — all delegate to `compute_cost_bps()`:
- `engine/portfolio_tracker.py:417` — `round(compute_cost_bps(delta), 2)`
- `engine/daily_batch.py:731` — `round(compute_cost_bps(target_weight - w_before), 2)`
- `engine/memory.py:5942` — `round(compute_cost_bps(-w_before), 2)` (inline import)

**Numerical contract preserved**: with no `daily_ret_window` argument provided (current behavior), output exactly equals `abs(weight_delta) * 10.0` — verified against 4 representative deltas; 109 pytest pass; Tier R 11 rules / 0 findings post-refactor.

**Future enabling**: any site can opt into vol-aware costing later by passing a 14-day return series — the function signature is forward-compatible.

**Site outside P1.2 scope** (intentional): `engine/decision_context.py:880` uses a separate `estimated_trade_cost_bps = 7.0` for the supervisor approval-preview UI. This is a *roundtrip* estimate (vs. one-way at the trade-ledger sites) used purely for displaying expected USD cost in the approval card. Different semantics → not unified. Documented here for transparency.

## 5. Pre-emptive interview answers

**"What transaction cost model do you use?"**
> "3 contexts. Backtest uses ATR-based dynamic — half-spread = max(3bp floor, 15% × 14-day return std). Live trades use flat 10bp as diagnostic at trade-write time; actual P&L comes from realized prices not modeled cost. B++ Mass FDR research uses 13bp round-trip flat for fair cross-strategy comparison. The split is intentional — each context has different needs. Unification is upcoming P1.2."

**"Is 10bp realistic for ETF rebal?"**
> "Yes, mid-literature range. ICI 2024 reports 2-8bp average for liquid US ETFs; AQR practitioner notes use 5-15bp for monthly rebal. Our universe is 50 ETFs all liquid; 10bp captures spread+commission+impact at our $1m scale. ATR model in backtest scales to 30-50bp during high-vol regimes which matches observed spread widening."

**"How much does cost actually eat into your Sharpe?"**
> "B++ Mass FDR test ran with 13bp round-trip applied to all 40 strategies — QL01 BAB still shipped at Sharpe 0.985. So cost-aware Sharpe shipping rule is satisfied. Backtest ATR model shows similar — high-turnover variants get penalized appropriately."

## 6. Code references

- `engine/cost_model.py` — **canonical live-cost utility** (P1.2, 2026-05-07): `compute_cost_bps()` + constants `LIVE_FLAT_COST_BPS / TC_FLOOR_BPS / TC_VOL_SCALE / TC_ATR_WINDOW`
- `engine/backtest.py:457-467` — ATR cost model constants (mirrors cost_model.py constants)
- `engine/backtest.py:470-540` — `_atr_transaction_cost` (multi-sector batch path; intentionally separate signature)
- `engine/backtest.py:825-840` — backtest application
- `engine/portfolio_tracker.py:51,417` — `from engine.cost_model import compute_cost_bps` + live trade write
- `engine/daily_batch.py:53,731` — `from engine.cost_model import compute_cost_bps` + auto_stop trade write
- `engine/memory.py:5930,5942` — inline import + human_stop trade write
- `engine/b_plus_search.py:44` — 13bp research baseline (intentionally NOT unified — research-context constant for cross-strategy fair comparison)
- `engine/decision_context.py:880` — 7bp roundtrip estimate (UI cost preview, separate semantics; not unified)
