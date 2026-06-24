# Three-Piece Strategy Uplift (E + F + A) — REJECTED (2026-05-03)

**Spec**：[../spec_strategy_uplift_2026-05-03.md](../spec_strategy_uplift_2026-05-03.md)
**Memory**：
- `project_cleanup_2026-05-03.md` (cleanup + bug fix → baseline 0.236)
- `feedback_alpha_hard_polish_easy_drift.md` (polish drift 警告)
- `feedback_spec_power_analysis.md` (pre-registration 纪律)

---

## TL;DR

按 spec v1.0 pre-registered 标准，三件套 uplift（E universe expansion + F asset-class-conditional short cap + A TSMOM ensemble）**整体 FAIL**：5-year window 2021-05 → 2026-05 上 Sharpe **-0.174 / NW HAC t = -0.419**，显著低于 baseline 0.236 / NW t = 0.75 / 0 alpha confidence interval。

按 pre-registration 纪律，**接受 FAIL，不调参，不拆分诊断后挑选**。production defaults 已 revert 至 pre-uplift baseline。代码作为 research scaffolding 保留。

---

## 1. Pre-registered 设计与学术依据

每件套基于已发表学术 prior（spec §1）：

| 组件 | 改动 | 学术 prior | Pre-registered ΔSharpe expectation |
|---|---|---|---|
| **E** | Universe 35 → 45 ETF（+10 cross-asset） | Moskowitz-Ooi-Pedersen (2012) | +0.10 - 0.20 |
| **F** | Asset-class-conditional short cap (equity 6 / non-equity 12) | Asness-Frazzini-Pedersen (2014) | +0.05 - 0.15 |
| **A** | TSMOM single 12-1 → ensemble {3, 6, 12, 24} | Hurst-Ooi-Pedersen (2017) | +0.10 - 0.20 |
| **Joint** | Three-piece simultaneous | conservative additive | +0.20 - 0.35 |

预期 post-uplift Sharpe：**0.43 - 0.59 区间**。

---

## 2. 实证 Backtest Verdict

**配置**：5-year 2021-05-03 → 2026-05-03，monthly rebalance, regime_scale=1.0, NW HAC + bootstrap inference, n=59 paired observations.

```
=== POST-UPLIFT METRICS (metrics_tsmom production) ===
Sharpe (annualised)       : -0.174
Annualised return          : -0.530%
Annualised vol             : 3.07%
Max drawdown               : -1.96% (similar to baseline)

=== NW HAC inference ===
Sharpe                     : -0.174
SE (annualised)            : 0.416
NW HAC t-stat              : -0.419
95% CI                     : (-0.988, +0.640)  ← crosses 0 in both directions
n                          : 59 paired months
```

### 2.1 Pre-registered Verdict (spec §2.1)

| Tier | Threshold | Actual | Outcome |
|---|---|---|---|
| **PASS** | Sharpe ≥ 0.40 AND t ≥ 1.65 | -0.174 / -0.419 | ❌ |
| **PARTIAL** | Sharpe ≥ 0.30 AND t ≥ 1.0 | — | ❌ |
| **FAIL** | Sharpe < 0.30 OR t < 0.5 | -0.174 < 0.30; t < 0.5 | ✅ **TRIGGERED** |

**Verdict**：**FAIL**。

### 2.2 Baseline Comparison

| State | Sharpe | NW t | Note |
|---|---|---|---|
| Pre-uplift baseline | +0.236 | +0.747 | After cleanup + `_month_offset` bug fix |
| **Post-uplift** | **-0.174** | **-0.419** | After E + F + A simultaneous |
| **Δ** | **-0.410** | **-1.166** | Active harm, not noise |

uplift 不仅没改善，反而把 Sharpe 拉到负。-0.410 跨度远超 noise 范围。

---

## 3. Hypothesized Failure Mechanisms

按 spec §5.3 限制，**不允许拆三件套做事后诊断 + 挑选**。以下仅为**事前列出的 risk factor 复盘**，不用于 spec 调整：

### 3.1 A (TSMOM ensemble) — Prime suspect

- 3-month lookback 在月频 rebalance 上是高 noise 信号
- 2024-2025 是 trend-reversal 频繁期（Fed pivot anticipation, AI rotation 多次）
- 短 lookback whipsaw 损失叠加
- 学术依据局限：Hurst (2017) "Century of Evidence" 用 daily/weekly TSMOM data，monthly rebalance 上 ensemble 优势 less robust

### 3.2 F (relaxed short cap) — Likely contributor

- 2024-2025 risk-on 复苏期：bonds rallied (rate cuts), TLT/IEF short 持续亏
- non-equity short cap 12 让损失放大到 equity 6 之外的 sector
- F 的合理性 contingent on 大量 risk-off 月份；2021-2026 主要是 risk-on/transition

### 3.3 E (universe expansion) — Likely neutral or mild

- FXE/FXY/FXC 长期窄区间 mean-reverting → TSMOM 信号弱
- SVXY 与 VXX 反向冗余（已有 vol exposure）
- BWX 长期与 TLT 高相关 → diversification gain 有限
- 估计 -0.05 to +0.05

### 3.4 三件套交互项

最 plausible 解释：**A 在新 45-ETF universe 上的 noise 放大，叠加 F 在 risk-on 时期的 short losses，超过 E 的边际改善**。

但**事后机制分析不能用于改 spec**——只能 inform 未来在 fresh OOS window 上的新 pre-registered design。

---

## 4. Pre-Registration 守住的边界

按 `feedback_spec_power_analysis.md` 与 spec §5.3 严格执行：

| 禁止行为 | 是否触犯 | 备注 |
|---|---|---|
| 看到 verdict 后调参 | ❌ 未触犯 | TSMOM_LOOKBACKS / short cap 数字未调 |
| 拆三件套挑选保留组件 | ❌ 未触犯 | E / F / A 同时 reject，未事后挑 |
| 在同窗口 retest 改良版 | ❌ 未触犯 | 2021-2026 数据"已用尽"，不重测 |
| 扩 universe 救场 | ❌ 未触犯 | 撤回 batch E |
| 加 lookback 救场 | ❌ 未触犯 | 撤回 ensemble |

**学术守纪。**

---

## 5. Production Default Revert

接受 FAIL → revert defaults 到 pre-uplift baseline（spec §5.3 "接受 0.236 baseline"）：

| 文件 | 改动 |
|---|---|
| `engine/signal.py` | `TSMOM_LOOKBACKS = (12,)`, `TSMOM_WEIGHTS = (1.0,)` (single 12-1) |
| `engine/portfolio.py` | `MAX_SHORT_NONEQUITY = 6` (parity with equity); `_REGIME_POSITION_LIMITS` 三 tuple 第三位 = 第二位 |
| `app.py` | 删除 `seed_batch_e()` 调用 |
| `macro_alpha_memory.db` | DELETE FROM universe_etfs WHERE batch=5 (10 rows removed) |
| `signal_snapshots` cache | Cleared (59 rows) — force recomputation under reverted defaults |

**保留作 research scaffolding**：
- `_BATCH_E` tuple list + `seed_batch_e()` function 保留
- `compute_arm_C_weights` ensemble loop 保留
- Asset-class-conditional short cap 逻辑保留（cap 数字调成 parity）
- `get_short_cap_group()` helper 保留

任何这些 capability 都可以在新 pre-registered spec 上重启用，但**不在本 spec 框架下**。

---

## 6. 学术 + 工程教训

1. **三件套 simultaneous 是反 PBO trap 的 pre-registration 设计**——单件套挑选会触发 multiple-testing inflation；同时 implement + 同时 verdict 才是合规
2. **Hurst (2017) ensemble 优势不是 monthly rebalance 通用真理**——daily/weekly 上才稳健
3. **2021-2026 单一 5-year window 的 evidence value 有限**——一个 macro regime（COVID 复苏 + 加息 + AI rally）不足以判定 strategy generalization
4. **学术 prior 的方向性 +/- 不等于 magnitude 可叠加**——"E +0.10, F +0.10, A +0.10 → joint +0.30" 是错的；交互项可以让 joint < 0
5. **Pre-registration 纪律的真实代价**——FAIL 后不能 cherry-pick "E alone might work"；必须 fresh window 重做
6. **验证 baseline 0.236 在 pre-uplift code 上是稳定的**——不是数据 noise，不是 bug 残留；strategy 本身在该 universe 上 alpha 上限即此

---

## 7. Future Work (未来不在本项目 horizon 内)

如果未来 retry：
- 必须用**fresh OOS window**：2010-2020（之前 backtest 跑过但未做 EFA 决策）
- 必须**事前 pre-register** 新 spec（不能照搬本 spec 改一两个数）
- 必须**降低三件套并行度**：单件套测试 + BHY FDR 校正
- 应**先做 power analysis**：2021-2026 上 EFA 整体 power 估算约 50-70%，不够 robust verdict

但这是 hypothetical future work，不是 master's project 9-12 month horizon 内的事。

---

## 8. 与其他 falsification 的关系

EFA 是项目第 5 个 documented LLM-as-alpha / strategy reject（前 4 个：D1, D1.1, Phase 0, FactorMAD Q1）：

| # | Hypothesis | Layer | Verdict | Doc |
|---|---|---|---|---|
| 1 | D1 narrative risk gate | LLM macro signal | SOFT REJECT | [d1_soft_rejected](narrative_risk_gate_d1_soft_rejected.md) |
| 2 | D1.1 narrative retry | LLM macro signal | HARD REJECT | [d1_1_rejected](narrative_risk_gate_d1_1_rejected.md) |
| 3 | Phase 0 narrative overlay | LLM cross-sectional | REJECT | [phase0_rejected](narrative_overlay_phase0_rejected.md) |
| 4 | FactorMAD Q1 mining | LLM factor mining | REJECT | [factor_mad_reject](factor_mad_reject.md) |
| **5** | **EFA strategy uplift** | **Quant strategy upgrade** | **REJECT** | **本 doc** |

注：1-4 是 LLM-as-alpha reject；5 是 pure quant strategy reject。增加这一笔表明项目的 falsification 纪律不只针对 LLM，也针对自家 quant 改良。

---

## 9. Honest disclosure

按 `feedback_quant_perspective.md` 主动指出：

1. **Single-window verdict 局限**：5-year n=59 月对 0.10-0.20 effect size 检测 power ~50-70%；实证 -0.174 远超 noise floor，但仍是单窗口
2. **不重做 sub-uplift 不等于 sub-uplift 一定无效**：A 可能在不同 ETF universe / 不同 lookback 配置 / 不同 horizon 上有效——本 spec 框架内无法判定
3. **Production default revert 不是"放弃 capability"**：基础设施保留，未来 fresh-window pre-registration 可调用
4. **0.236 baseline 也是 marginal alpha**：NW t=0.75 不显著；但作为 honest 起点 acceptable，不为追求"更高数字"再做事后调整
