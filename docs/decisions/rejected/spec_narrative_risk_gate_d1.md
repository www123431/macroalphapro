# Spec — Narrative Risk Gate D1

**版本**：v0.1（2026-05-02 ~22:50 起草）
**状态**：⏳ 等待用户审阅；spec 通过后开工
**前置**：[narrative_overlay_phase0_rejected.md](decisions/narrative_overlay_phase0_rejected.md)
**Memory**：[project_narrative_overlay_decision.md](../memory/project_narrative_overlay_decision.md)

---

## 0. 决策与方向

Phase 0 真实数据 REJECTED（B - C ≈ 0 是结构性证据）。架构 pivot 到 **D1：narrative → 总仓位 gate**——
不做 sector tilt，让 shock magnitude 调整 target_vol 标量。无 LLM。

**Why D1**：
- 维度坍缩：参数空间 312 → 3，bias-variance tradeoff 大幅改善
- 与 TSMOM baseline 正交：baseline 是 directional / gate 是 magnitude
- 文献支持：Caldara-Iacoviello 2022 主线证据是 GPR → aggregate stock returns 负向（不是 cross-sectional）
- 复用 Phase 0 90% 基础设施

---

## 1. 设计（事前定，spec 锁定）

```
shocks.parquet (gpr_z, epu_z, nvix_z) ─┐
                                        │
        max_z = max(|gpr_z|, |epu_z|, |nvix_z|)   ← 取最严重 shock 维度
                                        │
        ┌───────────────────────────────┘
        ↓
  if max_z ≥ 2.0  → vol_multiplier = 0.50    (强减仓: 2σ shock)
  elif max_z ≥ 1.0 → vol_multiplier = 0.75    (中等减仓: 1σ shock)
  else            → vol_multiplier = 1.00    (维持基线)
        ↓
construct_portfolio(target_vol = base_target_vol × vol_multiplier)
        ↓
现有 portfolio.py Step 1-6 既有流程
```

**阈值选择 rationale（事前 + 文献依据）**：
- 2.0 σ ≈ 5% tail event（normal approx）—— 大约 88 月里触发 4-5 次（COVID 2020-03、俄乌 2022-02 等）
- 1.0 σ ≈ 32% time → 中间档 → 让 gate 不至于"全有全无"
- vol_multiplier {0.5, 0.75, 1.0} 是常见 risk parity 通用挡位（Bridgewater All Weather "vol scaling" 同套档）
- **不调参** —— 一次锁定 spec，跑完后无论 reject/pass 都不允许 retroactive 修改

---

## 2. 实施（仅修改 4 处，无新依赖）

### 新增（1 文件）
```
engine/narrative/risk_gate.py  (~120 行)
  RiskGateContext dataclass: shock_state, thresholds, multipliers
  compute_vol_multiplier(shock_state, thresholds) -> float
  make_gate_factory(shocks_z, arm) -> Callable[[date], RiskGateContext | None]
```

### 修改（3 处，最小侵入）
- [engine/portfolio.py](../engine/portfolio.py) `construct_portfolio`：
  - 现有 `target_vol` 参数已存在，**不需要改 signature**
  - 在 `narrative_context` 处理处加分支：if ctx is RiskGateContext → 改 target_vol；if NarrativeContext → 现有 Step 5f tilt（保留 backward compat）
- [engine/backtest.py](../engine/backtest.py) `run_backtest`：
  - 现有 `narrative_context_func` 参数已存在，**不需要改 signature**
  - 不修改：context_func 返回 RiskGateContext 时 portfolio.py 内部分支处理
- [engine/narrative/backtest_phase0.py](../engine/narrative/backtest_phase0.py) → fork 为 `backtest_d1.py`：
  - 三组对照: A baseline / B gate / C placebo（同 Phase 0）
  - 同 5 reject gates（不放宽，否则 in-sample tuning）

### 复用（不动）
- [engine/narrative/shock_loader.py](../engine/narrative/shock_loader.py) ✅
- [engine/narrative/metrics.py](../engine/narrative/metrics.py) ✅
- [data/shocks.parquet](../data/shocks.parquet) ✅
- [tests/test_narrative_overlay.py](../tests/test_narrative_overlay.py) ✅（加 4 个 D1 单元测试）

---

## 3. Backtest 协议（与 Phase 0 一致）

| 项 | D1 设定 | 与 Phase 0 比 |
|---|---|---|
| OOS 窗口 | 2019-01 至 2026-05（89 月） | 同 |
| Train 窗口 | **N/A**（D1 无需 train，阈值事前定） | 不同 |
| 三组对照 | A baseline / B gate / C placebo（random N(0,1) shocks） | 同 |
| Reject 闸门 | ΔSharpe ≥ 0.10 / NW t ≥ 1.5 / B-C ≥ 0.05 / subperiod ≥ -0.20 / PBO ≤ 50% | **完全同 Phase 0**（不放宽） |
| Subperiods | P1/P2/P3 三段 | 同 |
| Production engine | 用 [engine.backtest.run_backtest](../engine/backtest.py) | 同 |

**关键纪律**：reject 闸门**不放宽**。即使 D1 是 architectural pivot，spec 闸门保持与 Phase 0 相同——
否则就是 in-sample tuning（明知 Phase 0 没过，故意放宽 D1 闸门让它过）。

---

## 4. PBO Sweep — D1 必须跑

D1 阈值 {1.0, 2.0} + multiplier {0.5, 0.75, 1.0} 都是 hyperparameter。Phase 0 跳过 PBO sweep（无谓——其他 4 闸门已 fail）。**D1 必须跑 PBO sweep**：

阈值组合（事前定，5 个，对应 spec budget sweep 类比）：
- (low=1.0, high=2.0)（spec primary）
- (0.75, 2.0)
- (1.25, 2.0)
- (1.0, 1.75)
- (1.0, 2.25)

PBO ≤ 50% 是闸门。Phase 0 PBO=98%（production 自己 overfit）警告我们必须严格。

---

## 5. Sprint 拆分（1-2 天）

| Sprint | 工作量 | 产出 |
|---|---|---|
| S1 risk_gate 模块 | 0.5 天 | `engine/narrative/risk_gate.py` + 4 单元测试 |
| S2 portfolio.py 接入 | 0.5 天 | 加 RiskGateContext 分支 + regression test 证 None 仍 no-op |
| S3 backtest_d1 runner | 0.5 天 | `engine/narrative/backtest_d1.py` 三组对照 + 5 budget PBO sweep |
| S4 Ablation + decision | 0.5 天 | `docs/decisions/narrative_risk_gate_d1_*.md` |

**总工时**：~2 天（spec 估），实际预期 4-6 小时（Phase 0 经验：估算保守）。

---

## 6. 学术诚实声明（事前）

按 [memory: feedback_quant_perspective.md](../memory/feedback_quant_perspective.md) 主动指出 D1 的边界：

1. **NVIX 仍是 VIX-proxy**（Phase 0 已注明）—— 这影响 nvix_z 列的真实信息含量，但 max(...) 操作让单一 column 失效不致命
2. **阈值 {1.0, 2.0} 是事前选定，不基于优化** —— Bridgewater 通用档位，不是 GPR/EPU 历史分布拟合的最优值
3. **OOS 检测功效**：n=89, 闸门 ≥ 0.10 ΔSharpe，功效约 80%（vs Phase 0 cross-sectional 30-50%）—— D1 维度坍缩使统计力大幅改善，但不是无穷
4. **若 D1 通过 ≠ 真实可投产 alpha**：spec 通过只是必要条件，不是充分条件。Clean Zone calendar-bound 验证仍需 Phase 2 paper trading 6-12 月累积
5. **若 D1 reject** → 严格证伪 narrative-driven trading 在月频 + 5 年 OOS 框架下的可行性。整个方向终止，FactorMAD 接管 P0

---

## 7. 用户审阅 checklist

**只需 ≤5 分钟阅读**。任意一项 NO → spec 修订，不开工：

- [ ] D1 设计（max-z scalar → 3 档 vol multiplier）合理
- [ ] 阈值 {1.0, 2.0} + multipliers {0.5, 0.75, 1.0} 事前选定可接受
- [ ] PBO sweep 5 个阈值组合 + ≤ 50% 闸门可接受
- [ ] Reject 闸门**不放宽**纪律可接受（与 Phase 0 完全一致）
- [ ] 不动 [signal.py](../engine/signal.py) / [regime.py](../engine/regime.py) / [sector_pipeline.py](../engine/sector_pipeline.py) / FactorMAD 边界
- [ ] 工程量 1-2 天可接受
- [ ] D1 reject 后整个方向终止（不再有 D2 LLM 版）的纪律可接受
- [ ] D1 pass 后下一步是写 D2 spec（含前视偏差协议），不是直接上生产

---

**审阅人签字位**：（spec 通过后我立即开工 S1）
