# Verdict — Multivariate MSM v3 DESCRIPTIVE_POSITIVE (2026-05-08)

**Spec**: `docs/spec_multivariate_msm_v3.md` (registered id=46, status=active post-verdict)
**Run script**: `scripts/run_multivariate_msm_v3_d6.py`
**Output cache**: `data/multivariate_msm_v3/`
**Run date**: 2026-05-08

---

## 1. Verdict

**DESCRIPTIVE_POSITIVE** per spec_v3 §3.2 framework.

**This is the project's first ship-suggesting verdict** after 8 prior hypothesis tests (8 falsifications + 1 marginal). It does NOT contradict the falsification chain — it confirms that pre-registration discipline produces honest verdicts in BOTH directions when applied rigorously to architecturally sound experiments.

---

## 2. Verdict Numbers (from `data/multivariate_msm_v3/d6_v3_verdict.txt`)

| Field | Value |
|---|---|
| OOS window | 2019-01 to 2024-12 |
| OOS months captured | **72 / 72** (after BME→ME calendar fix) |
| Sharpe(multivariate v3 overlay) | +0.481 |
| Sharpe(univariate baseline overlay) | −0.845 |
| **ΔŜ (annualized)** | **+1.326** |
| Bootstrap 95% CI for ΔŜ | **[+0.514, +2.535]** |
| CI lower > 0 ? | **TRUE** |
| CI lower ≥ +0.05 ? | **TRUE** (ship-suggesting heuristic) |
| Politis-White block size | 2 |
| Memmel Z (descriptive only) | +2.427 |
| Paired ρ̂ | +0.356 |
| Achieved power at observed ρ̂ | 5.4% |
| Multivariate fallback rate | 0.0% (NORMAL tier) |
| **Decision label** | **DESCRIPTIVE_POSITIVE** |

---

## 3. Two-Run Transparency (BME→ME Bug Discovery)

This verdict reflects the **second** run after a calendar-alignment bug fix.

### Run 1 (BME-labeled): 2026-05-08 ~15:40-16:04
- script `freq="BME"` (business month-end) for rebalance dates
- SPY index `freq="ME"` (calendar month-end)
- Mismatch on weekend-ending months → `intersection()` dropped 21 of 72 OOS months
- **Captured 51 / 72 OOS months**
- Verdict: ΔŜ = +0.885, CI = [−0.078, +2.319] → **DESCRIPTIVE_INSUFFICIENT** (CI lower bound −0.078, just barely below 0)

### Bug fix: BME → ME alignment (commit 2026-05-08)
- `scripts/run_multivariate_msm_v3_d6.py` line: `freq="BME"` → `freq="ME"`
- Rationale: same underlying data (last trading day's close); only label differs. Calendar-end labels match SPY's `.resample("ME")` index. **Implementation bug, not hypothesis change** — fix permitted under spec §6 forbidden mods (calendar conventions not enumerated).

### Run 2 (ME-aligned): 2026-05-08 ~16:08-16:??
- Both rebalance and SPY index on calendar month-end labels
- **Captured 72 / 72 OOS months** ✓
- Verdict: ΔŜ = +1.326, CI = [+0.514, +2.535] → **DESCRIPTIVE_POSITIVE**

**Both runs are documented** to preempt any "you cherry-picked" challenge. The bug was real, the fix was orthogonal to the verdict question, and the second run's higher CI tightening (1.19× narrower) is purely from including the missing 21 months of unbiased data.

---

## 4. Architectural Journey (v1 → v2 → v3)

This verdict is the result of a **3-iteration architectural refinement**, each iteration superseded for an honest reason:

| Spec | Status | Reason |
|---|---|---|
| `spec_multivariate_msm_v1.md` | superseded | D1-D5 architectural defects identified during W1 D6 (regime label permutation / K=2 misspec / binary 2p-1 overlay / unvalidated proxy / underpowered framing) |
| `spec_multivariate_msm_v2.md` | superseded | D4 gate empirically invalidated HYG-LQD ETF return-spread proxy: Pearson r = −0.03 vs FRED OAS-diff (duration mismatch) — **discipline working as intended: caught wrong proxy BEFORE verdict** |
| `spec_multivariate_msm_v3.md` | **active** | 2-feature MSM (yield_spread + VIX), VIX-anchored regime ID, ternary [0.45, 0.55] hysteresis overlay, descriptive verdict framework. **DESCRIPTIVE_POSITIVE on architecturally sound experiment.** |

**Pre-registration discipline did its job**: each iteration's defect was caught explicitly (D1-D5 in v1, D4 in v2). v3 was the first architecturally sound design, and verdict came back positive with reasonable effect size + bootstrap CI clearly excluding zero.

---

## 5. Honest Caveats (per spec §3.3 D5 framework)

1. **Achieved power 5.4%**: even with this POSITIVE verdict, 6yr OOS at observed paired ρ̂ ≈ 0.36 is severely underpowered for δ=0.10 detection (~1003 calendar yrs to 80% power). The fact that we observed ΔŜ = +1.326 with CI clearly excluding zero is genuine signal, but the spec framework explicitly anticipated that "even when alpha is real, we'd miss it 95% of times". Conversely: this 5% time we caught it could in principle be the lucky 5% under H₀. We cannot resolve this with current OOS sample.

2. **Long-bias coincidence**: 2019-2024 was a strong bull market (SPY +118% cumulative). The multivariate path was risk-on more often than the univariate path; in a bull market, more long-time = more return. Some of the +1.326 ΔŜ is "structural" rather than "tactical alpha". A bear market OOS would test whether multi's correct risk-off calls (e.g., 2020-Q1 COVID, 2022 inflation) outweigh structural long-bias.

3. **Implementation lag**: Multi correctly flipped to risk-off in 2020-03 (COVID) and 2021-Q4 (inflation/Russia start), and back to risk-on in 2021-Q3 and 2023-Q2. **However**, multi got stuck at risk-off for 2020-Q2 to 2021-Q2 (9 months) during COVID recovery — missing significant rebound. This is "high-VIX regime over-fit" symptom (a covariance asymmetry artifact) — not severe enough to flip verdict but a real model limitation.

4. **VIX 2024-08 brief spike (yen carry unwind)**: VIX hit 38 briefly in early August 2024 but settled within 2 weeks. Multi at month-end remained risk-on (didn't catch the brief stress). Monthly rebalance frequency is too coarse for sub-month spikes.

These are the project's HONEST caveats — disclosed pre-supervisor-decision. Supervisor (= user) chose **c = 0.6** with full knowledge of these caveats.

---

## 6. Production Swap Decision

Per spec_v3 §3.2 supervisor framework: DESCRIPTIVE_POSITIVE → "supervisor MAY PendingApproval(production_signal_swap)".

**Supervisor decision (2026-05-08)**: **PROCEED WITH SWAP**.

| Choice | Selection | Rationale |
|---|---|---|
| `c` (REGIME_SCALE) | **0.6** | Within spec_v1 §3.6 procedural bounds [0.3, 0.7]. Slightly above 0.5 midpoint reflecting confidence in directional signal (CI lower bound +0.51 well above 0); below 0.7 upper bound preserving conservative cap given 5.4% achieved power caveat. |
| Production code change | `_USE_MULTIVARIATE_REGIME = True` in engine/regime.py | Production `get_regime_on()` now tries multivariate v3 path first |
| Production overlay engine | `_get_regime_multivariate_v3` | (was `_get_regime_multivariate` v1; v3 is the verified path) |
| `engine/config.py::REGIME_SCALE` | `0.6` (was `1.0` = disabled) | Activates portfolio.py overlay live branch |

### Production Effect

After swap (effective 2026-05-08):
- `get_regime_on()` consults multivariate v3 classifier (yield_spread + VIX, VIX-anchored, K=2)
- Falls back to univariate yield_spread MSM on ConvergenceError / InsufficientData / MissingFeatureData
- Falls back to rule-based on second failure
- Returned regime label feeds `portfolio.py::construct_portfolio` Step 5 regime overlay
- When regime = "risk-off": long positions scaled by **0.6**
- When regime = "transition": long positions scaled by `0.6 + 0.4 × p_risk_on`
- When regime = "risk-on": no scaling (multiplier = 1.0)

### Counter-factual

If supervisor had chosen NOT to swap (DESCRIPTIVE_INSUFFICIENT-equivalent action):
- Production stays REGIME_SCALE = 1.0 (overlay disabled)
- Multivariate v3 evidence on file but not deployed
- Spec v3 stays active for forward demonstration

User (= supervisor) explicitly chose to swap, on the grounds that the symmetric Bayesian framing favors action when point estimate + CI both clearly positive: "我也可以 challenge 为什么你就觉得 uni 在未来会表现得更好" (paraphrased: status quo bias deserves no default).

---

## 7. Project Tally Update

| Metric | Pre-2026-05-08 | Post-2026-05-08 |
|---|---|---|
| Hypothesis tests | 9 (8 falsifications + 1 marginal + 0 PASS-equivalent) | 9 (8 falsifications + 1 marginal + **1 ship-suggesting POSITIVE**) |
| Falsification chain length | 8 | 8 (multivariate_msm_v3 NOT a falsification — it ships) |
| Production signal | `ql01_bab` no overlay | `ql01_bab` × multivariate v3 overlay (c=0.6) |
| EFFECTIVE_N_TRIALS | 11 | 11 (this is verdict on pre-registered spec, not new trial) |

This is **not** a "we found alpha" claim — it's a **"pre-registration framework produced an honest POSITIVE verdict on a sound architecture"** demonstration. The framework's value is shown by its 8 prior FALSE-EQUIVALENT verdicts AND this 1 TRUE-EQUIVALENT verdict.

---

## 8. Reproducibility

```bash
git hash-object docs/spec_multivariate_msm_v3.md docs/decisions/multivariate_msm_v3_2026-05-08_DESCRIPTIVE_POSITIVE.md
python scripts/run_multivariate_msm_v3_d6.py    # ME-aligned; ~10 min walk-forward
# Output: data/multivariate_msm_v3/d6_v3_verdict.txt
```

Cached probs: `data/multivariate_msm_v3/walk_forward_probs_v3.parquet` (180 monthly observations 2010-2024).
Cached SPY: `data/multivariate_msm_v3/spy_monthly.parquet`.

---

## 9. Disposition

- Spec status: **active** (POSITIVE → spec stays alive; production swap implements it)
- Verdict file: this document
- Production constants: `engine/config.py::REGIME_SCALE = 0.6` and `engine/regime.py::_USE_MULTIVARIATE_REGIME = True`
- Production overlay path: `_get_regime_multivariate_v3` (replaces v1 in soft-rollout branch)
- Updated tests: `tests/test_regime_multivariate.py::test_use_multivariate_regime_flag_state` + `tests/test_signal_portfolio.py::test_regime_scale_within_spec_bounds`
- Updated alignment surface: `engine/auto_audit_rules.py::ALIGNMENT_SURFACE["REGIME_SCALE"] = 0.6`
- Memory entry: `project_multivariate_v3_first_positive_2026-05-08.md` (new)
- MEMORY.md index entry: added

---

**Verdict locked. Swap executed. First ship-suggesting verdict in project history.**
