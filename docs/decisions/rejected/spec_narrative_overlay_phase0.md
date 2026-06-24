# Spec — Narrative Overlay Phase 0

**版本**：v0.1（2026-05-02 起草）
**状态**：⏳ 等待用户审阅；spec 通过后开工
**作者**：项目对话 2026-05-02
**适用范围**：仅 Phase 0（Mapping Layer Ablation）。Phase 1 / 2 暂不展开。

---

## 0. TL;DR

把 LLM 从"sector 综合判断 + sizing"重新定位到"宏观 narrative shock 抽取"——这是真正的项目方向修正。
**Phase 0 不引入任何 LLM**，先用三个公开 narrative shock index（GPR / EPU / NVIX）训练 IRF
表，叠加到现有 TSMOM baseline 之上，**OOS 验证 mapping layer 本身有没有 alpha**。

如果 Phase 0 的纯规则映射都做不出 ΔSharpe ≥ 0.10 / NW t ≥ 1.5 / PBO ≤ 50%，那 Phase 1
让 LLM 抽更细粒度 shock 也不会救——整个 narrative overlay 方向 reject，止损在 1-2 周。

---

## 1. 方向决策与三阶段框架

### 1.1 方向决策（2026-05-02 用户授权）

> "用 LLM 判断什么时候选择什么策略——看到特朗普要打仗，可能就去买防御板块。"

这是 **narrative-driven asset rotation**，对应学术名字：event-conditional regime detection +
narrative-induced sector rotation。

### 1.2 严格分层（与 [memory: feedback_no_llm_as_judge.md](../memory/feedback_no_llm_as_judge.md) 对齐）

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 1（生成层 — LLM 是 alpha 来源，必须保留）                     │
│   原始宏观新闻 / FOMC minutes / 财报 transcript                    │
│       ↓ LLM 结构化抽取                                             │
│   shock_schema = {                                                 │
│       shock_type:   categorical (war / election / pandemic / ...)  │
│       severity:     1-5                                            │
│       horizon_days: short / mid / long                             │
│       affected_regions: [...]                                       │
│       confidence:   0-100                                          │
│   }                                                                │
│   Phase 0 用 GPR / EPU / NVIX 公开 index 替代；Phase 1 才接 LLM     │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ Layer 2（映射层 — 严禁 LLM，确定性）                                │
│   shock_vector × IRF_table → tilt_vector ∈ R^n_sectors             │
│   IRF_table 用 Jordà (2005) Local Projections 训练                 │
└────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ Layer 3（仓位层 — 严禁 LLM，确定性）                                │
│   tilt_vector ⊕ baseline_weights                                   │
│   → vol targeting → position cap → final_weights                   │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 三阶段框架（仅 Phase 0 在本 spec 范围内）

| 阶段 | 内容 | 状态 | reject 闸门 |
|---|---|---|---|
| **Phase 0** | Layer 2 + Layer 3 用 GPR/EPU/NVIX 公开 index 训练 IRF + 接入 portfolio overlay + OOS 验证 | 本 spec | ΔSharpe < 0.10 / NW t < 1.5 / PBO > 50% → reject 整个方向 |
| Phase 1 | 加 Layer 1 LLM shock 抽取，对比 LLM-shock vs 公开 index 增量 | 仅 Phase 0 通过后开工 | LLM-shock 不显著优于公开 index → 不接 LLM，只用公开 index |
| Phase 2 | 接入 [orchestrator.py](../engine/orchestrator.py) daily/weekly cycle、UI、approval gate | 仅 Phase 1 通过后开工 | — |

---

## 2. 学术框架 + 文献 attribution

按 [memory: feedback_attribute_borrowed_designs.md](../memory/feedback_attribute_borrowed_designs.md)
显式标注每个借鉴源 + 项目自身扩展。

### 2.1 Shock index 来源

| Index | 文献 | 数据源 URL | 频率 | 历史起点 |
|---|---|---|---|---|
| **GPR** (Geopolitical Risk) | Caldara & Iacoviello (2022, AER) "Measuring Geopolitical Risk" | `https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls` (.xls 不是 csv，用 xlrd 读) | 月度 + 日度 | 1900 (Historical 'GPRH') / 1985 (Recent 'GPR') |
| **EPU** (Economic Policy Uncertainty) | Baker, Bloom, Davis (2016, QJE) "Measuring Economic Policy Uncertainty" | `https://www.policyuncertainty.com/us_monthly.html` | 月度 + 日度 | 1985 |
| **NVIX** (News Implied Vol) | Manela & Moreira (2017, JFE) "News Implied Volatility and Disaster Concerns" | `https://asafmanela.github.io/papers/` 附录 | 月度 | 1890 |

**Origin**：三个 index 全部来自外部。
**Project extension**：把三者**联合**作为 shock_vector 输入到 IRF 训练（单 index 学术上有人做过，
三者联合 + cross-sector ETF response 这个组合到目前没在已发表文献中见到——但**这不是真正的创新点**，
真正的 alpha 论证靠 OOS Sharpe 不靠组合新颖性）。

### 2.2 IRF 估计方法

**Jordà (2005, AER) Local Projections**——选这个不选 VAR 的理由：

- VAR 在多 shock 类型 + 多 sector 下参数爆炸（k_shock × k_sector × n_lag），n=192 月度样本
  会严重过拟合
- LP 对每个 (shock, sector, horizon) 单独跑一条回归，自由度可控
- LP 不依赖识别假设的 ordering（VAR Cholesky 顺序是争议来源）
- LP 标准误用 Newey-West HAC（与项目 [engine/factor_mad_methodology.py](../engine/factor_mad_methodology.py)
  现有 NW-HAC 体系一致）

**模型形式**（每个 sector i × shock_index k × horizon h）：

```
r_{i, t→t+h} = α_{i,k,h} + β_{i,k,h} · shock_{k,t} + γ' · controls_t + ε_{i,k,h,t}
```

其中：
- `r_{i, t→t+h}` = sector i 从 t 到 t+h 的累计超额收益（相对 SPY）
- `shock_{k,t}` = shock index k 在 t 月的标准化值（z-score over rolling 60-month window）
- `controls_t` = [VIX_t, yield_spread_t, lagged_return_{i,t-1}]（避免遗漏变量）
- IRF 估计量 = `β_{i,k,h}`，h ∈ {1, 2, 3, 6} 月

### 2.3 应用层

**Origin**：vol targeting + regime overlay 来自 Moreira-Muir (2017) + Ang-Bekaert (2004)，已在
[engine/portfolio.py](../engine/portfolio.py) 落地。
**Project extension**：在现有 `construct_portfolio` Step 5（regime overlay）之后、Step 6
（position cap）之前，新增 Step 5b：narrative tilt。

---

## 3. 数据规范

### 3.1 时间窗（用户已拍板：选项 C）

- **Train**：2003-01 至 2018-12（192 个月）
- **OOS**：2019-01 至 2026-05（89 个月）
- **OOS 包含 COVID（2020）+ 通胀冲击（2021-2022）+ 加息周期**——真正的 stress test
- **训练样本说明**：每个 (sector, shock, horizon) 192 obs，控制变量 3 个 → 自由度足够；
  Phase 0 不做 jackknife / CPCV 因为 LP 单方程结构没有 multiple-testing 放大

### 3.2 ETF universe

直接复用 [engine/history.py](../engine/history.py) 的 `SECTOR_ETF` 映射（26 个 sector ETF），
**但 2003-2010 之前不存在的 ETF 用 mutual fund proxy 替代**：

- ETF 上市日期早于 2003-01 → 直接用 ETF 价格
- ETF 上市日期晚于 2003-01 → 用 mutual fund / index proxy 回填到 2003-01；标注 `proxy_used=True`
- 每个 ETF 的 first-trade-date 写入 `data/eft_inception_dates.json`，spec 验收时人工核对

**重要**：proxy 用于 IRF 训练，不用于 OOS 评估。OOS 期（2019-2026）所有 ETF 都已上市。

### 3.3 Shock index 处理

- 月度对齐到 ETF 月末
- z-score 标准化：rolling 60-month window，避免前视偏差
- 缺失月份用 forward-fill（小于 1 个月）；超过 1 个月报错
- **Phase 0 不做 LLM shock 抽取**——三个 index 直接作为三维 shock_vector 喂给 IRF

---

## 4. 实现规格

### 4.1 新建文件清单

| 文件 | 行数估算 | 职责 |
|---|---|---|
| `engine/narrative/__init__.py` | 5 | package marker |
| `engine/narrative/shock_loader.py` | 150-200 | 拉取 GPR/EPU/NVIX，标准化、对齐、缓存到 `data/shocks.parquet` |
| `engine/narrative/irf_trainer.py` | 250-300 | LP 训练；输出 IRF 表 → `data/irf_table.parquet` |
| `engine/narrative/overlay.py` | 150-200 | tilt_vector 构造 + clip + 接入 portfolio.py |
| `engine/narrative/backtest_phase0.py` | 200-250 | Train/OOS split、ablation runner、reject 闸门评估 |
| `tests/test_narrative_overlay.py` | 150-200 | smoke test + 数值正确性 |
| `data/shocks.parquet` | — | 3 个 shock index 月度数据（约 280 月 × 3 列） |
| `data/irf_table.parquet` | — | IRF 估计表（n_sector × n_shock × n_horizon × {β, se, p}） |
| `data/etf_inception_dates.json` | — | proxy 启用判定 |

### 4.2 修改现有文件

| 文件 | 修改点 | 风险 |
|---|---|---|
| `engine/portfolio.py` | `construct_portfolio` Step 5 后插入 Step 5b（narrative_tilt overlay）；signature 加 `shock_state: dict \| None = None` 默认 None 保持向后兼容 | 低 — 默认 None 时行为不变 |
| `engine/backtest.py` | `run_backtest` 加 `enable_narrative_overlay: bool = False` 参数；ablation 用 | 低 — 默认 False 时行为不变 |

**不改动**：
- [engine/signal.py](../engine/signal.py) — 信号层完全不动
- [engine/regime.py](../engine/regime.py) — yield_spread MSM 完全不动
- [engine/sector_pipeline.py](../engine/sector_pipeline.py) — sector debate 完全不动（Phase 0 不动 LLM 任何路径）
- [engine/agents/factor_mad/](../engine/agents/factor_mad/) — pause 状态，季度运行不变

### 4.3 Layer 2 + Layer 3 算法

```python
# Layer 2: shock_vector → tilt_vector
def compute_narrative_tilt(
    shock_state: dict[str, float],  # {"gpr": z-score, "epu": z-score, "nvix": z-score}
    sectors:     list[str],          # 26 sector tickers
    irf_table:   pd.DataFrame,       # cached IRF estimates
    horizon:     int = 3,            # months ahead
) -> pd.Series:
    """
    For each sector i:
      tilt_i = Σ_k irf_table.loc[(i, k, horizon), 'beta'] · shock_state[k]
    Then z-score normalise tilt_i across sectors → unit gross exposure.
    """
    raw_tilt = pd.Series(0.0, index=sectors)
    for i in sectors:
        for k, z_val in shock_state.items():
            beta = irf_table.loc[(i, k, horizon), 'beta']
            raw_tilt[i] += beta * z_val
    # Cross-sectional z-score
    tilt = (raw_tilt - raw_tilt.mean()) / raw_tilt.std()
    return tilt

# Layer 3: tilt → final weights (in portfolio.py Step 5b)
TILT_BUDGET = 0.05   # 5% gross exposure budget for narrative tilt (conservative)
def apply_narrative_tilt(
    weights: pd.Series,
    tilt:    pd.Series,
    budget:  float = TILT_BUDGET,
) -> pd.Series:
    """
    Add additive tilt scaled to budget, then renormalize to preserve gross exposure.
    """
    # Scale tilt to gross budget
    tilt_scaled = tilt * (budget / tilt.abs().sum())
    new_weights = weights + tilt_scaled
    # Re-clip to MAX_WEIGHT (handled by existing Step 6)
    return new_weights
```

**关键约束**：
- `TILT_BUDGET = 0.05` 是事前定的 5% gross exposure 预算（不是事后调出来的）
- IRF 表训练后 freeze 到 parquet——OOS 期间不再更新；Phase 0 不做 walk-forward IRF refit
  （这是 Phase 0 的简化；如果通过，Phase 1 时讨论是否加 expanding window refit）

---

## 5. Backtest 与 Ablation 协议

### 5.1 三组对照

| Run | enable_narrative_overlay | shock_source | 含义 |
|---|---|---|---|
| **A — Baseline** | False | — | 当前生产 baseline（TSMOM + inverse-vol + REGIME_SCALE=1.0） |
| **B — Narrative** | True | GPR + EPU + NVIX (公开 index) | Phase 0 实验组 |
| **C — Random** | True | shocks shuffled within columns | 安慰剂检验：如果 B vs C 不显著，说明 IRF 是 spurious |

### 5.2 评估指标

| 指标 | 公式 / 来源 | 阈值（Phase 0 通过） |
|---|---|---|
| **ΔSharpe** | `Sharpe(B) - Sharpe(A)`，OOS 89 月 | ≥ 0.10 |
| **NW t-stat 增量** | Newey-West HAC t on `excess_return(B) - excess_return(A)`；lag = 4 (= 12^(1/3) 约 2.3 取整) | ≥ 1.5 |
| **PBO** | López de Prado (2014, 2016)；用 Phase 0 的 5 个超参组合（TILT_BUDGET ∈ {0.03, 0.04, 0.05, 0.06, 0.07}）做 CSCV | ≤ 50% |
| **DSR** | Bailey-LdP (2014) Deflated Sharpe Ratio | > 0（advisory，不是闸门） |
| **B vs C 差异** | `Sharpe(B) - Sharpe(C)` | ≥ 0.05（防 spurious） |

**reject 决策**：5 个指标里 **任意一个**未达标 → 整个 narrative overlay 方向 reject，
不进 Phase 1。

### 5.3 Subperiod 一致性检查

把 OOS 89 月切成三段：
- 2019-2020 上半（pre-COVID）
- 2020 下半-2021（COVID + 复苏）
- 2022-2026 上半（通胀 + 加息）

**事前定的合理性约束**：每段 ΔSharpe 都要 ≥ 0（不要求全部显著，但不能有段是负向）。
任一段为负 → flag 但不直接 reject（因为单段样本太小，统计力不足）。

### 5.4 Calendar bound 声明

[memory: project_clean_zone_calendar_bound.md](../memory/project_clean_zone_calendar_bound.md):
OOS 89 月 = 2019-01 ~ 2026-05 是**回测式 OOS**，不是 paper trading 意义的 Clean Zone。
Phase 0 的判定**仅基于回测 OOS**，但写入文档时必须诚实标注"非 Clean Zone 验证"。
真正的 Clean Zone 验证在 Phase 2 上线后才能开始累积。

---

## 6. 实施 Sprint 拆分（Phase 0 内部）

| Sprint | 工作量 | 产出 | 验收 |
|---|---|---|---|
| **S0 数据层** | 0.5 天 | `shock_loader.py` + `shocks.parquet` + `etf_inception_dates.json` | 三个 index 历史 1985-2026 拉全；ETF inception 全部覆盖 |
| **S1 IRF 训练** | 1.5 天 | `irf_trainer.py` + `irf_table.parquet` + 训练日志 | LP 收敛；NW-HAC SE 数值正常；至少 30% sector × shock × h=3 cell 在 train 上 t > 1.96 |
| **S2 Overlay 接入** | 1 天 | `overlay.py` + `portfolio.py` Step 5b | 单元测试通过；signature 默认参数下 baseline 数值不变（regression test）|
| **S3 Backtest 框架** | 1 天 | `backtest_phase0.py` + 三组对照的 runner | A/B/C 三组都能跑；输出标准化 metrics dict |
| **S4 Ablation 评估** | 0.5 天 | `notebooks/phase0_results.md`（结果报告） | 5 个指标全部计算并写入报告；reject/pass 决策落地 |
| **S5 Smoke + Tests** | 0.5 天 | `tests/test_narrative_overlay.py` | pytest 通过；端到端跑通 |

**总工时**：5 天工程时间，外加 IRF 训练 + backtest 计算时间（机器跑，不算工时）。

---

## 7. 测试协议

### 7.1 单元测试（`tests/test_narrative_overlay.py`）

- `test_shock_loader_alignment`：拉数据后月度对齐到 ETF 月末
- `test_irf_trainer_known_dgp`：用合成数据（已知 IRF）训练，估计 β 应在 ±10% 区间
- `test_overlay_zero_shock`：所有 shock = 0 时 tilt = 0，weights 不变（regression）
- `test_overlay_budget_respected`：tilt gross sum ≈ TILT_BUDGET ± ε
- `test_portfolio_backward_compat`：`construct_portfolio(...)` 不传 `shock_state` 时输出与改动前 byte-identical

### 7.2 数值正确性

- IRF 估计的 NW-HAC SE 用 statsmodels 的 `cov_type='HAC'` 与手算对比，差 < 1e-8
- z-score 与 pandas `.rolling(60).apply(zscore)` 数值一致

### 7.3 端到端 smoke

```bash
python -m engine.narrative.backtest_phase0 --runs A,B,C --report
```

应在 < 10 分钟内完成 OOS 89 月回测，输出标准化 metrics 报告。

---

## 8. Phase 0 Reject 闸门（再次明文）

**5 个 reject 条件，任意一个触发即整个 narrative overlay 方向终止**：

1. ΔSharpe(B - A) < 0.10
2. NW t-stat (B - A) < 1.5
3. PBO > 50%
4. B vs C (random shock) Sharpe 差 < 0.05
5. 任一 subperiod ΔSharpe < -0.20（极端负向）

如果触发 reject：
- 写入 `docs/decisions/narrative_overlay_phase0_rejected.md` 记录 metrics + 数据 + 复现命令
- memory 更新决策（不抹掉 spec，作为"考察过 + 否决"的研究档案）
- FactorMAD 解锁恢复新 sprint

如果通过：
- 写入 `docs/decisions/narrative_overlay_phase0_passed.md`
- 启动 Phase 1 spec 编写

---

## 9. 与项目其他 backlog 的关系

### 9.1 优先级冲突

[docs/master_backlog.md](master_backlog.md) 当前已完成 P0-P6 + V4.0 + 第十节 Spread Consensus。
后续待办（Defer-12 / Defer-13 / 第七节 DL-P0 等）**全部延后**，narrative overlay Phase 0 提到 P0
最高优先级。

### 9.2 FactorMAD 状态

按 [memory: project_factor_mad_redesign.md](../memory/project_factor_mad_redesign.md) S0-S4
全部完成，**但本 spec 期间不启动新 sprint**：

- 已实现 Stages 1-5 继续按季度跑（[orchestrator.run_quarterly](../engine/orchestrator.py#L519)）
- 不新增 Sprint 6+
- 6 个月后审查：BH FDR 是否 promote 出 ≥ 1 个因子。0 个 → 重审 FactorMAD 整体方向

### 9.3 LLM ablation 测试（之前提议的另一项 P0）

**包含在 Phase 1 内**——Phase 0 通过后，Phase 1 同时回答两个问题：
1. LLM-shock 是否优于公开 index？
2. 当前 sector debate 的 LLM 层是否产生独立 alpha？

合并节省工程，避免重复跑 backtest infrastructure。

---

## 10. 边界与诚实声明

按 [memory: feedback_quant_perspective.md](../memory/feedback_quant_perspective.md) 必须主动指出局限：

1. **回测 OOS ≠ Clean Zone 验证**——Phase 0 通过只是必要条件，不是充分条件。真正的 alpha
   归属还要看 Phase 2 上线后的真实 paper trading 表现
2. **公开 shock index 在 academic literature 上的可交易 alpha modest**——Bybee 2023 OOS
   年化 2-4%，不要预期 5%+
3. **IRF 表 freeze 在 train 期，OOS 不 refit**——这是简化假设，可能 Phase 1 需要 expanding
   window；但 Phase 0 不做以避免 multiple-testing
4. **Local Projections 的 small-sample bias**——Herbst-Johanssen (2021) 显示 LP 在 h > 1 时
   IRF 估计偏 attenuated；Phase 0 接受这个偏差，h=3 是事前选定（不调到事后看好的 horizon）
5. **三个 shock index 之间相关性**——GPR / EPU / NVIX 在 1985-2026 期间历史相关性约 0.4-0.6，
   不正交。LP 单方程不要求正交，但联合解释力会被高估。Phase 0 的 IRF 表保留每个 shock 的独立
   β——不做正交化（PCA 处理破坏经济解释）
6. **TILT_BUDGET = 0.05 是经验值**——不基于优化。如果 Phase 0 通过，Phase 1 会做 sensitivity 分析
7. **Phase 0 不做执行成本建模**——backtest 假设 1 bps slippage（与现有 [backtest.py](../engine/backtest.py)
   一致）；narrative tilt 增加 turnover 但 5% 预算下边际成本可忽略。如果 Phase 1 把预算加到
   10%+ 才需要重新测算
8. **Constructed shock indices 的 backfill bias**——GPR / EPU / NVIX 都是用事后定义的
   keyword 列表 / SVR 模型回填到历史新闻得到的 index：
   - GPR keyword set 由 Caldara-Iacoviello 2016+ 年定，回填到 1985-2003 train 早段
   - EPU keyword set 由 Baker-Bloom-Davis 2010 年定，回填到 1985-2010
   - NVIX SVR 训练集 1996-2009，应用到 1890-1996 是 OOS prediction
   这是 narrative-finance 领域**已知 + 学术接受**的 trade-off（Bybee 2023 同样有这个问题），
   不是 Phase 0 设计的偏差。但用户必须知道：Phase 0 通过不能宣称"严格无前视"，只能宣称
   "在文献接受的 backfill bias 下显著"。
9. **Phase 0 不触发 LLM 前视偏差** —— 因为完全不调 LLM。但 Phase 1 一旦引入 LLM 抽 shock，
   必须按 [memory: feedback_llm_lookahead_bias.md](../memory/feedback_llm_lookahead_bias.md)
   的 4 类前视偏差处理协议设计 spec。**Phase 0 通过后的 Phase 1 spec 不许跳过这一步**。

---

## 11. 用户审阅 checklist（spec 通过条件）

请逐项确认。任意一项 NO → spec 修订，不开工：

- [ ] 三阶段框架（Phase 0 / 1 / 2）合理，Phase 0 reject 闸门清晰
- [ ] 时间窗选项 C（2003-2018 train / 2019-2026 OOS）已确认
- [ ] 三个 shock index（GPR / EPU / NVIX）选择合理；Phase 1 才接 LLM
- [ ] IRF 用 Local Projections 而非 VAR 的理由可接受
- [ ] TILT_BUDGET = 0.05 的事前选择可接受
- [ ] reject 闸门 5 个指标 + 阈值合理
- [ ] FactorMAD pause 决策（不启动新 sprint，已实现部分继续季度跑）可接受
- [ ] 文件清单 + 修改现有文件清单可接受；不动 signal.py / regime.py / sector_pipeline.py 的边界清晰
- [ ] 5 天工程时间预算可接受
- [ ] 学术诚实声明（第 10 节）合理；Phase 0 通过 ≠ Clean Zone 验证

---

**审阅人签字位**：（spec 通过后我开工 Sprint S0）
