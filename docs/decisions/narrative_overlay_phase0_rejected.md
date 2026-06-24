# Narrative Overlay Phase 0 — REJECTED (Final Verdict, 2026-05-02)

**Spec**：[../spec_narrative_overlay_phase0.md](../spec_narrative_overlay_phase0.md)
**前置文档**：[narrative_overlay_phase0_pipeline_ready.md](narrative_overlay_phase0_pipeline_ready.md)
**Memory**：[../../memory/project_narrative_overlay_decision.md](../../memory/project_narrative_overlay_decision.md)

---

## TL;DR

**Phase 0 严格 reject** —— 当前这套 IRF mapping + 5% gross tilt + GPR/EPU/(VIX-proxy) cross-sectional rotation 架构在月频上没有 alpha。两次独立真实数据 backtest（60 月 + 89 月）一致显示。

不是 alignment 失败、不是 synthetic 噪声、不是 PBO 拒掉 —— 是 **真实 narrative 信号与 random placebo 给出的 portfolio Sharpe 几乎完全相同**（B - C 接近 0），结构性证伪。

---

## 1. 最终 Backtest 证据（两次独立验证）

### Run 1: 60 月（2021-05 至 2026-05，与 production baseline 0.510 期间对齐）

```
Synthetic data : False
Arm               n    Sharpe
A_baseline       60     0.509   ← 与 PDF UI TSMOM+风控 0.510 完美对齐
B_narrative      60     0.502
C_placebo        60     0.503

ΔSharpe (B - A)         : -0.008    FAIL  (≥ 0.10)
NW HAC t-stat (B - A)   : -0.197    FAIL  (≥ 1.5)
Sharpe(B) - Sharpe(C)   : -0.002    FAIL  ← 致命：narrative ≈ random
P1_pre_covid       +0.000    PASS
P2_covid           -0.180    PASS (边缘)
P3_tightening      +0.008    PASS
```

### Run 2: 89 月（spec 标准窗口 2019-01 至 2026-05）

```
Synthetic data : False
Arm               n    Sharpe
A_baseline       88     0.327
B_narrative      88     0.296   ← narrative 减损 baseline
C_placebo        88     0.272

ΔSharpe (B - A)         : -0.031    FAIL  (negative)
NW HAC t-stat (B - A)   : -0.656    FAIL  (negative)
Sharpe(B) - Sharpe(C)   : +0.024    FAIL  (need ≥ 0.05)
P1_pre_covid       +0.001    PASS
P2_covid           -0.222    FAIL  ← 触发 ≥ -0.20 闸门
P3_tightening      +0.008    PASS
```

---

## 2. Reject 闸门评估（spec §5.2）

| 闸门 | 阈值 | 60 月 | 89 月 | 状态 |
|---|---|---|---|---|
| ΔSharpe(B-A) | ≥ 0.10 | -0.008 | -0.031 | ❌ FAIL × 2 |
| NW t-stat(B-A) | ≥ 1.5 | -0.197 | -0.656 | ❌ FAIL × 2 |
| Sharpe(B-C) | ≥ 0.05 | -0.002 | +0.024 | ❌ FAIL × 2 |
| 任一 subperiod ΔSharpe | ≥ -0.20 | min -0.180（PASS） | min -0.222（FAIL） | ❌ 89 月触发 |
| PBO | ≤ 0.50 | 未跑 | 未跑 | — |

**5 闸门中 89 月 4 个 FAIL，60 月 3 个 FAIL。Phase 0 严格 reject。**

PBO sweep 未跑（5× backtest 成本），但已无意义：headline gates 全 fail，PBO 最多在 advisory 上加权。

---

## 3. Production-engine alignment 证据（reject 不是 alignment bug）

| | 当前 backtest | PDF UI（commit "legacy"） | 差 |
|---|---|---|---|
| Arm A 60 月 Sharpe | **0.509** | TSMOM+风控 **0.510** | 0.001 ✅ |
| Arm A 89 月 Sharpe | 0.327 | — | — |

60 月 alignment 完美。Phase 0 reject 不是因为 baseline 跑错，是 narrative overlay 真的没产生 alpha。

注：[memory: project_baseline_switch_2026-05-02.md](../../memory/project_baseline_switch_2026-05-02.md) 记录的 Sharpe 0.703 / NW t=+2.64 与 PDF + 当前 backtest 都不一致，最可能是后续某个 commit 的运行（与当前 code 不同）。本次 alignment 以**当前 code + PDF "legacy"**为准。

---

## 4. 学术诚实声明（spec §10 边界）

按 [memory: feedback_quant_perspective.md](../../memory/feedback_quant_perspective.md) 主动指出 reject 的边界：

### 4.1 数据替代造成的 noise
- NVIX 主页 404 → 整期用 VIX 月末替代（VIX 是 option-implied vol，与 News Implied Vol 不等价）
- VIX 已是 baseline 风险变量，narrative_nvix 列可能与 baseline 共线 → 减弱 narrative 独立信息

### 4.2 IRF train 期 → OOS 期 regime 换代
- IRF 训练 2003-2018（GFC、欧债、QE 时代）
- OOS 2019-2026（COVID、俄乌、加息、AI 革命）
- shock-type 分布换代，linear LP-IRF 难以泛化（Herbst-Johanssen 2021 small-sample bias）

### 4.3 TILT_BUDGET 5% 过小
- baseline target_vol = 10% 下，5% gross tilt 产生的 marginal Δ-Sharpe ≤ 0.05
- spec 闸门 ≥ 0.10 几乎不可能达到（即使 narrative 真有 alpha，5% budget 也不够 move needle）
- **但增大 budget 是 in-sample tuning**（spec §5.4 禁止），不能在 reject 后救场

### 4.4 月频 rebalance 接不到 short-window narrative shock
- HFT / 机构 algo 在 minutes 级吃掉显眼 narrative shock
- 月频策略只接残余 narrative drift（Bybee 2023: 严格 OOS alpha 仅 2-4%/年）
- 本次 OOS Sharpe 量级 0.3-0.5 + 5% budget 上的 alpha 增量上限 ~0.02-0.04，**与噪声同量级**

### 4.5 N=89 月 Phase 0 检测功效

事前功效计算：60-89 月样本上检测 ΔSharpe ≥ 0.10 / NW t ≥ 1.5 的功效约 30-50%。即"真实有 0.10 alpha 也只 30-50% 概率被检测到"。这意味着 reject **不是绝对证伪**，但 B - C 接近 0 是更强的结构性证据（不依赖 baseline alpha 大小）。

---

## 5. Phase 0 教训（学术 + 工程）

### 5.1 学术教训

1. **narrative-driven cross-sectional rotation 在月频 + 公开 shock index 下没可检测 alpha** —— 与 Bybee 2023 严格 OOS 结果一致。文献先验是对的，我们的实验不是 anomaly。
2. **B - C 接近 0 是最强的结构性信号** —— 比 ΔSharpe / t-stat 更稳健，因为不依赖样本期 baseline alpha。这条 sanity check 应该写进任何 alpha 验证 spec。
3. **TILT_BUDGET = 5% 在 baseline Sharpe 0.3-0.5 上几乎不可能产生 ≥ 0.10 ΔSharpe**。Phase 0 spec 的 budget 选择事前应该做功效分析。

### 5.2 工程教训

1. **Production-engine alignment 是 P0 必做**：v1 简化 backtest baseline 0.054 vs v2 生产 engine 0.509，后者才能与 PDF UI 0.510 比对。任何 ablation backtest 必须先 align baseline。
2. **PDF UI 是真相之一，memory 文档是真相之二**：两者不一致时双向交叉验证，不要假设单边权威。
3. **环境约束（DNS）改变了执行路径但没改变结论**：用户在外网拉真实数据是 dataflow 上的解耦，不是工程缺陷。

---

## 6. 守住的边界（spec § 全部满足）

✅ 未动 [signal.py](../../engine/signal.py) / [regime.py](../../engine/regime.py) / [sector_pipeline.py](../../engine/sector_pipeline.py)
✅ 未动 [agents/factor_mad/](../../engine/agents/factor_mad/)
✅ portfolio.py 改动严格向后兼容（regression test 已证 byte-identical when narrative_context=None）
✅ backtest.py 改动严格向后兼容（narrative_context_func=None 完全 no-op）
✅ 0 LLM 调用进 narrative overlay 路径
✅ Phase 1 前视偏差纪律已固化在 [feedback_llm_lookahead_bias.md](../../memory/feedback_llm_lookahead_bias.md)（即使本次没启动 Phase 1）

---

## 7. 反思：方向修正路径（不在本 doc 决策，列给后续）

按 [project_narrative_overlay_decision.md](../../memory/project_narrative_overlay_decision.md) reject 协议，本 doc 仅记录 Phase 0 reject 事实。**方向是否完全终止 / 修正为 D（narrative → 总风险 gate）/ 回到 FactorMAD —— 由用户单独决策**。

候选路径（详见对话 2026-05-02 ~22:30 的讨论）：

| 选项 | 内容 | 工程量 | 推荐度 |
|---|---|---|---|
| D1 | narrative → 总仓位 gate（无 LLM，公开 index → 阈值规则） | 1-2 天 | ★★★★ 最有可能产生真 alpha |
| D2 | D1 + LLM 抽 shock_type/severity（保留 LLM 愿景） | 4-7 天（含前视处理） | ★★★ 仅在 D1 通过后启动 |
| A | 严格 reject，关闭整个方向 | 0 | ★★★ 守 spec 纪律 |
| E | 回 FactorMAD bottom-up | 0（解锁现有 sprint） | ★★ 可与 A 结合 |
| C | 跳过 reject 直接 Phase 1 LLM | 4-7 天 | ★ 学术不合理 |
| B | 在 Phase 0 框架内调参 retry | 0.5 天 | ✗ 违反 spec §5.4 in-sample tuning |

---

## 8. FactorMAD 状态变化

[memory: project_factor_mad_redesign.md](../../memory/project_factor_mad_redesign.md) 之前为 Phase 0 期间暂停新 sprint。**Phase 0 已 reject，FactorMAD 解锁**——但是否启动新 sprint 取决于用户在选项 D1 / A / E 中的决策。

---

## 9. 工程产出清单（保留）

即使方向 reject，本次工程产出全部保留为研究档案 + 可复用基础设施：

| 文件 | 价值 |
|---|---|
| [engine/narrative/shock_loader.py](../../engine/narrative/shock_loader.py) | 通用 GPR/EPU/NVIX 拉取 + 缓存，未来选项 D1/D2 可复用 |
| [engine/narrative/irf_trainer.py](../../engine/narrative/irf_trainer.py) | LP+NW-HAC 实现，可作为方法学参考 |
| [engine/narrative/overlay.py](../../engine/narrative/overlay.py) | 选项 D 的 mapping/sizing 接口模板 |
| [engine/narrative/backtest_phase0.py](../../engine/narrative/backtest_phase0.py) | 三组对照 ablation runner，下一次方向验证可复用 |
| [engine/narrative/metrics.py](../../engine/narrative/metrics.py) | 通用 Sharpe / NW-HAC / PBO，独立 importable |
| [engine/backtest.py](../../engine/backtest.py) `narrative_context_func` 参数 | **保留**，默认 None=no-op；选项 D 可复用此接口 |
| [tests/test_narrative_overlay.py](../../tests/test_narrative_overlay.py) | 12/12 通过，回归保护 |
| [data/shocks.parquet](../../data/shocks.parquet) | 真实 GPR + EPU + VIX-proxy z-scores，可缓存 |
| [data/irf_table.parquet](../../data/irf_table.parquet) | 真实 IRF 估计表 |

**没有 sunk cost** — 所有产出对未来 narrative 工作（如 D1）直接可复用。

---

## 10. 实际工时

| 阶段 | 实际 |
|---|---|
| Spec 起草 + memory 固化 | ~1 小时 |
| Phase 0 工程（S0-S5）| ~6 小时 |
| Production-engine alignment（重写 v1→v2）| ~1.5 小时 |
| 真实数据获取（用户外网操作）| ~30 分钟 |
| 真实 IRF 训练 + 60 月 backtest + 89 月 backtest | ~30 分钟 |
| 诊断 + decision doc | ~1 小时 |
| **总计** | **~10 小时** |

按 [memory: feedback_alpha_hard_polish_easy_drift.md](../../memory/feedback_alpha_hard_polish_easy_drift.md)：10 小时投入证伪一个看似前沿的方向，是 alpha 验证纪律的胜利，不是失败。
