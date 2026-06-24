# FactorMAD 重构规格 — Institutional-Grade Alpha Mining Agent

> **作者视角**：顶级量化金融分析师 + 严谨学术标准
> **创建日期**：2026-05-02
> **定位**：将 FactorMAD 从"统计审计员"升级为机构级 alpha mining agent，并接入 operations 架构
> **基调**：模拟盘优先，关键节点人工审批；所有方法学问题修复优先于功能扩展

---

## 零、决策清单（已锁定）

### Q1 — 是否同意"当前模块不是 agent"？
**结论**：是。当前 `factor_mad.py` 只有验证工具（Layer 1/2/3）+ ICIR 监控 + BH 修正，
没有任何候选因子生成机制。`run_quarterly_factor_mining()` 在 master_backlog 被引用
但代码中不存在。需要从 hypothesis generation 开始重建。

### Q2 — DSL 形式
**结论**：(c) **pandas 表达式 + AST 静态检查**。原因：
- 与 `prices: pd.DataFrame → pd.Series` 已有签名兼容
- 可序列化、可哈希（每个因子有唯一 AST fingerprint）
- AST walker 能静态检测 t+1 数据引用，无需运行
- 自研轻量 DSL 学习曲线陡；纯 gplearn 演化局限于符号回归

### Q3 — LLM Proposer 触发节奏
**结论**：(a) **季度跑** + (c) **事件触发**。
- 季度：每季度首个交易日生成 ~20 个候选
- 事件：制度切换 / 现有因子 ICIR 衰减 / Universe 扩容 / Supervisor 手动
- 月度跑成本过高且质量提升边际递减

### Q4 — GP Search 是否保留
**结论**：**Defer**。当前 432 样本对 SR 是噪音生成器，等 Universe ≥ 50 ETF 再激活。
保留 `audit_factor_structure` 作为 Layer 3 旁证报告，不投入 GP search 开发。

### Q5 — 失败因子记忆
**结论**：**长期存储所有被拒因子的 DSL + 拒绝原因 + 验证指标**。
作为下次 LLM Proposer 的 negative example 注入 prompt，并对其他 agent 可见。

### Q6 — Agent 基础设施改造范围
**结论**：(a) **大改**。建 `engine/agents/` 框架，先重构 FactorMAD，后续逐步迁移
ERA / UniverseReview / FailureAttribution / SignalDecayPatrol。
理由：已有 ≥4 个 agent，不统一框架的耦合成本只会越来越高。

### Q7 — 季度作业放进 orchestrator 还是保留 daemon thread
**结论**：**放进 `orchestrator.run_quarterly()`**。
daemon thread 在 Streamlit 重启后丢状态、无重试、无审计。

### Q8 — 事件总线持久化
**结论**：(b) **SQLite 表**。与 `agent_runs` 一致，可重放、可审计。

### Q9 — FactorMAD 失败因子记忆对其他 agent 可见
**结论**：是。FailureAttribution 归因后能查 FactorMAD 因子族谱，找到曾在该 sector 失败的因子。
跨 agent 共享认知是 operations 架构的核心含义。

---

## 一、方法学硬伤（必修，否则统计结论不可信）

### C1. Walk-Forward 70/30 切分污染 → **替换为 Purged CPCV**
当前 `run_layer2_audit:1007` 的 `ic_train, ic_test = ic_vals[:split], ic_vals[split:]` 不是真正 OOS：
候选因子在被研究者设计时已经看到全部数据，存在 dataset snooping (Lopez de Prado 2018 第7章)。

**实施**：
- K=10 块切分 + 22 日 embargo
- 共 C(10,2)=45 个 OOS 路径
- 输出：OOS Sharpe **分布** + PBO（Probability of Backtest Overfitting）

### C2. IC 没控制 Beta → **截面回归剔除风险因子**
18 个 sector ETF 高度共动；当前 Spearman IC 大部分是 Beta 暴露，不是 alpha。

**实施**（最低成本起步）：
```
r_i,t = α_i + β_i · MKT_t + ε_i,t   # 截面去市场
IC_t = ρ(factor_i,t-fwd_days, ε_i,t)
```
后续可扩展到 FF5 残差化（见三·A5）。

### C3. MI 样本窗口重叠 → **不重叠 step + Block Bootstrap null**
当前 `compute_factor_mi` step=22d，因子值（如 mom_3m 用 65 天）窗口重叠。

**实施**：
- step 拉到 `max(64, fwd_days)` 让因子窗口不重叠
- 用随机置换的因子值跑 200 次得到 null MI 分布
- 阈值改为 null 分布的 95% 分位数（替换魔数 `2.0`）

### S1. Harvey-Liu t-stat 用了原始 n → **Newey-West HAC 标准误**
月度 IC 有 1-2 阶自相关；当前 `t = ICIR × √n` 虚高 1.3-2 倍。

**实施**：用 `statsmodels.regression.linear_model.OLS` 拟合 IC 序列对常数项，
取 `HAC` 协方差（lag=6）的 t-stat。

### S2. BH FDR 用固定 N_TEST=24 → **存真实 n_test 进 DB**
`DiscoveredFactor` 表加 `n_test_obs` 字段，BH 时用真实 n。

### S3. 负 ICIR 因子被错误丢弃 → **取绝对值 + 方向**
```python
# 当前
if icir_val < 0.05: continue
w = icir_val
# 修复
if abs(icir_val) < 0.05: continue
w = abs(icir_val)
contrib = math.copysign(1, icir_val) * ranked * w
```

### S4. `_classify_audit` 关键词重叠是噪音 → **bootstrap CI on R²**
R²=0.31 在 432 样本上 95%CI 包含 0；改为 bootstrap residuals 200 次，
CI 下界 > 0.1 才报 positive。

---

## 一·B 功效分析与最小可检出 ICIR（学术诚实化补充，2026-05-02 修订）

> **修订动机**：原 spec §7 用"指示性而非决定性"模糊处理样本量约束。
> 顶级量化标准要求把 minimum detectable effect (MDE) 作为开工承诺写明。

### 1B.1 当前架构的功效曲线

设月度 IC 真值 ICIR\* = 0.20（参考 Asness-Moskowitz-Pedersen 2013 industry portfolio 均值）：

- 月度 IC 一阶自相关 ρ̂ ≈ 0.2-0.3（经验，需上线后实测覆盖）
- HAC 调整后有效样本 n_eff ≈ N × (1−ρ̂)/(1+ρ̂)
- N=120 月（10 年）→ n_eff ≈ 60-70
- SE(ICIR) ≈ 1/√n_eff ≈ 0.13
- α=0.05 双尾 + power=0.80 下，**MDE ≈ (z_{α/2} + z_β) × SE ≈ 0.36**

**核心结论**：在 10 年 monthly IC 样本下，真值 ICIR\* = 0.20 的因子有 ~60% 概率被错误判为不显著（Type II 错误）。

### 1B.2 自动通过门槛的修订

当前 spec §1 C1 隐含的门槛 `icir_oos_median ≥ 0.30 → auto-approve` 在统计上**过严**（详见上表）。
基于功效分析修订为分级门槛：

```
verdict gates（icir_oos_median）：
  n_obs < 24                                  → reject (insufficient data)
  median < 0.10                                → hard reject, no LLM
  0.10 ≤ median < 0.20                         → reject unless p05 > 0
  0.20 ≤ median < 0.30 AND p05 > 0             → LLM Critic
  median ≥ 0.30 AND p05 > 0.10                 → auto-approve
  median ≥ 0.30 AND p05 ≤ 0.10                 → LLM Critic（CPCV 不稳定）
```

`p05` = 5th percentile of CPCV path ICIRs（已在 Layer2Result 输出）。
新加 `p05 > 0` / `p05 > 0.10` 条件防止"中位数高但分布尾部为负"的因子直接通过。

### 1B.3 MDE 必须随结果输出

每次 Layer 2 audit 必须把 **MDE 写进 Layer2Result.mde**。Supervisor 看到的不是
"ICIR=0.18，拒绝"，而是"ICIR=0.18，n=120 下 MDE=0.36，统计上无法区分 0.18 vs 0"。

**实施位置**：
- `engine/factor_mad_methodology.py` 加 `compute_minimum_detectable_icir(n_obs, alpha=0.05, power=0.80, ar1_rho=0.25)` 函数
- `Layer2Result` 加 `mde: float | None` 字段
- Admin UI 因子卡片：在 ICIR 数值下方显示 MDE 与 power state

### 1B.4 文献置信带（McLean-Pontiff 2016 校正后）

| 论文 | 资产 | 报告 IR | 净 IR (post-publication 校正) |
|---|---|---|---|
| Moskowitz-Grinblatt 1999 | 20 industry MOM | 1.10 | 0.55-0.70 |
| Asness-Moskowitz-Pedersen 2013 | industry MOM+VAL | 0.93 | ~0.50 |
| Stivers-Sun 2010 | 30 industry, regime cond. | 0.7-1.1 | 0.40-0.60 |
| Bali-Cakici-Whitelaw 2011 | industry MAX | 0.6-0.8 | 0.30-0.45 |
| Conrad-Kaul 1998 | weekly industry rev. | 0.4-0.6 | 0.15-0.30 |

**结论**：sector ETF 上经得起 OOS 的 IR 上限约 **0.50**，对应 ICIR 真值上限 ≈ **0.20-0.25**。
任何 in-sample ICIR > 0.40 的候选都应**默认怀疑过拟合**，要求 sealed OOS 验证（A4，仍 defer）。

---

## 二、Agent 架构重建（机构级核心）

### 2.1 Agent 基础设施（新建 `engine/agents/`）

```
engine/agents/
├── __init__.py
├── base.py              # Agent 基类
├── factor_mad/          # FactorMAD agent 拆分
│   ├── __init__.py
│   ├── agent.py         # FactorMADAgent 主类
│   ├── proposer.py      # LLM Proposer
│   ├── critic.py        # 多轮 Critic 循环
│   ├── dsl.py           # Factor DSL 定义 + AST walker
│   ├── search.py        # 搜索空间管理
│   ├── lifecycle.py     # 退役 + 自学习
│   └── memory.py        # FactorFailureMemory I/O
└── event_bus.py         # 跨 agent 事件总线
```

### 2.2 Agent 基类合约

```python
# engine/agents/base.py
@dataclass
class Trigger:
    type: str              # "scheduled" | "event" | "manual"
    source: str            # "quarterly_tick" | "regime.switch" | "supervisor:zhang"
    payload: dict

@dataclass
class AgentResult:
    run_id: str
    agent_id: str
    status: str            # "succeeded" | "failed" | "interrupted"
    started_at: datetime
    finished_at: datetime
    summary: dict
    events_emitted: list[str]
    error: str | None = None

class Agent(ABC):
    AGENT_ID: str
    @abstractmethod
    def run(self, trigger: Trigger, as_of: date) -> AgentResult: ...
    def get_health(self, as_of: date) -> dict: ...
    def _persist_run(self, run: AgentResult) -> None: ...
    def _emit_event(self, event: AgentEvent) -> None: ...
    def _claim_lock(self, key: str, ttl_seconds: int) -> bool: ...
```

### 2.3 数据库新表

```sql
-- agent_runs (所有 agent 共用)
CREATE TABLE agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    agent_id TEXT NOT NULL,
    triggered_by TEXT NOT NULL,
    status TEXT NOT NULL,
    state TEXT,                       -- 当前 stage（如 "validating"）
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    input_params TEXT,                -- JSON
    output_summary TEXT,              -- JSON
    error TEXT,
    parent_run_id TEXT,               -- agent 因果链
    INDEX(agent_id, started_at)
);

-- agent_events (事件总线)
CREATE TABLE agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,         -- "regime.switch" | "factor.decay" | ...
    source_agent TEXT,
    payload TEXT NOT NULL,            -- JSON
    occurred_at TIMESTAMP NOT NULL,
    consumed_by TEXT,                 -- JSON list of agent_ids that processed
    INDEX(event_type, occurred_at)
);

-- factor_failure_memory (FactorMAD 自学习)
CREATE TABLE factor_failure_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dsl_expression TEXT NOT NULL,
    dsl_ast_hash TEXT UNIQUE,         -- 防重复
    rejection_stage TEXT,             -- "pre_filter" | "validation" | "fdr" | "supervisor"
    rejection_reason TEXT,
    icir_train REAL,
    icir_test REAL,
    pbo REAL,
    dsr REAL,
    regime_breakdown TEXT,            -- JSON
    economic_hypothesis TEXT,         -- LLM 生成时的逻辑陈述
    lessons_learned TEXT,             -- LLM 复盘
    rejected_at TIMESTAMP,
    proposer_run_id TEXT
);

-- DiscoveredFactor 加字段
ALTER TABLE discovered_factors ADD COLUMN n_test_obs INTEGER;
ALTER TABLE discovered_factors ADD COLUMN dsl_expression TEXT;
ALTER TABLE discovered_factors ADD COLUMN dsl_ast_hash TEXT;
ALTER TABLE discovered_factors ADD COLUMN dsr REAL;            -- Deflated Sharpe Ratio
ALTER TABLE discovered_factors ADD COLUMN pbo REAL;            -- Prob of Backtest Overfitting
ALTER TABLE discovered_factors ADD COLUMN n_test_obs INTEGER;
ALTER TABLE discovered_factors ADD COLUMN harvey_liu_t REAL;
ALTER TABLE discovered_factors ADD COLUMN refinement_rounds INTEGER DEFAULT 0;
```

### 2.4 Orchestrator 集成

```python
# engine/orchestrator.py 新增
def run_quarterly(self, as_of: date) -> ChainResult:
    """季度作业，替换 daily_batch 里的 daemon thread。"""
    steps = []
    # 1. ERA 季度审计
    steps.append(("era_audit", lambda: ERAAgent().run(...)))
    # 2. Universe Review
    steps.append(("universe_review", lambda: UniverseReviewAgent().run(...)))
    # 3. FactorMAD mining loop
    steps.append(("factor_mad_mine", lambda: FactorMADAgent().run(
        Trigger(type="scheduled", source="quarterly_tick", payload={}),
        as_of=as_of
    )))
    # 4. BH FDR correction（已存在）
    steps.append(("factor_mad_bh", lambda: run_quarterly_bh_correction(as_of)))
    # 同步执行，失败重试 1 次，状态写 cycle_states
    return self._run_chain(steps, cycle_type="quarterly", as_of=as_of)
```

`daily_batch.py` 中原有的 daemon thread 删除，改为：
```python
if _is_first_trading_day_of_quarter(t_day):
    from engine.orchestrator import TradingCycleOrchestrator
    TradingCycleOrchestrator().run_quarterly(as_of=t_day)
```

### 2.5 EventBus 设计

```python
# engine/agents/event_bus.py
class EventBus:
    """SQLite-backed event bus. Synchronous dispatch with persistence."""

    def publish(self, event_type: str, payload: dict, source_agent: str) -> str:
        """Insert into agent_events; return event_id."""

    def subscribe(self, event_type: str, handler: Callable[[AgentEvent], None]) -> None: ...

    def replay_unconsumed(self, agent_id: str, since: datetime) -> list[AgentEvent]:
        """重启后消费未处理事件。"""
```

**订阅注册**（启动时）：
```python
event_bus.subscribe("regime.switch",     FactorMADAgent.on_regime_switch)
event_bus.subscribe("factor.decay",      FactorMADAgent.on_factor_decay)
event_bus.subscribe("universe.expanded", FactorMADAgent.on_universe_expanded)

# Supervisor 决策事件回流给 FactorMAD
event_bus.subscribe("approval.factor_approved", FactorMADAgent.on_proposal_approved)
event_bus.subscribe("approval.factor_rejected", FactorMADAgent.on_proposal_rejected)

# Failure attribution 反馈
event_bus.subscribe("attribution.factor_blamed", FactorMADAgent.on_factor_blamed)
```

### 2.6 FactorMAD Agent 7-Stage Pipeline

```
Stage 1: HYPOTHESIS GENERATION（mutation-based，2026-05-02 修订）
  - LLM Proposer 不做 free generation；从 active 因子池采样 1-2 个
    parent 因子，要求 LLM 提出 mutation：
      * 参数扰动（window 22 → 21/26/33）
      * 单算子替换（TS_Mean → Decay_Linear；TS_Rank → Zscore）
      * regime 包裹（baseline 外加 When/RegimeMix）
      * cross-sector 包裹（baseline 外加 CSDemean/PairSpread）
  - 距离约束：每个 mutation 与 parent 的 ast_hash 必须不同；
    与 parent 因子值的 |Spearman ρ| ≤ 0.7（在 Stage 2 验证）
  - 每次 LLM 调用产 5-10 mutations（quantity → quality）
  - 一季度多次调用，总候选规模仍 ~30-50，但每个携带更高研究信号密度
  - 文献依据：Schölkopf et al. 2021 论证小样本下 prior-anchored mutation
    在统计上优于 free generation；McLean-Pontiff 2016 显示 published anomaly
    OOS Sharpe 下降 ~50%，意味着 LLM 直接复现论文因子的预期收益已被市场吸收

Stage 2: PRE-FILTER（廉价过滤，淘汰 95%）
  - DSL AST 静态检查（t+1 引用、未来变量）
  - MI 污染扫描（修复 C3 后）
  - 与已存因子相关性（>0.7 淘汰）
  - 存活：5-15 个

Stage 3: RIGOROUS VALIDATION
  - Purged CPCV（修复 C1）
  - Beta-neutralized IC（修复 C2）
  - HAC t-stat（修复 S1）
  - DSR / PBO 计算
  - Net ICIR（扣除交易成本）
  - 存活：1-5 个

Stage 4: AGENTIC REFINEMENT（多轮辩论）
  - 复用 LangGraph（engine/debate.py 框架）
  - Round 1: Critic 指出弱点 + 给出可执行改进
  - Round 2: Proposer 修改 DSL，重新 validate
  - 早停：3 轮无改善 / DSR > 0.5 / 5 轮上限

Stage 5: MULTIPLE-TESTING CORRECTION
  - BH FDR α=0.10（用真实 n_test，修复 S2）
  - 配额检查：active factormad 因子 ≤ 8

Stage 6: HUMAN GATE
  - 写 PendingApproval(GATE_FACTORMAD_CANDIDATE)
  - 包含：DSL、经济学假设、CPCV 报告、DSR、PBO、Critic 辩论日志

Stage 7: LIFECYCLE & SELF-LEARNING
  - 上线后持续监控 ICIR + DSR + 边际贡献
  - 衰减检测：half-life 估计 + 连续 N 月 ICIR < 0.05 → archive
  - 失败因子写 factor_failure_memory，下次 Proposer 注入
  - emit "factor.deployed" / "factor.retired" 事件
```

### 2.7 Factor DSL 设计

```python
# engine/agents/factor_mad/dsl.py
"""
Factor DSL: pandas-expression based, AST-validated.

每个因子是一个表达式树，叶子是数据原语，内部节点是算子。
LLM Proposer 输出表达式字符串，AST walker 验证后编译为可调用函数。
"""

# ── 数据原语 ─────────────────────────────────────────
class Close: ...        # prices
class Volume: ...
class High: ...
class Low: ...
class Returns(window: int): ...
class LogReturns(window: int): ...

# ── 算子 ────────────────────────────────────────────
class TS_Rank(expr, window): ...        # rolling rank
class CS_Rank(expr): ...                # cross-sectional rank
class Decay_Linear(expr, decay): ...    # linear decay weight
class Zscore(expr, window): ...
class Winsorize(expr, q_low=0.01, q_high=0.99): ...
class Ind_Neutralize(expr, group): ...  # group-demean (sector neutral)
class TS_Mean(expr, window): ...
class TS_Std(expr, window): ...
class TS_Max(expr, window): ...
class TS_Min(expr, window): ...

# ── ETF-native 算子（2026-05-02 修订加入）──────────
# 项目最大架构优势：regime.py + macro_fetcher 已接入，但原 spec DSL 没有
# regime / cross-sector 接口，浪费宏观数据投资。文献：Stivers-Sun 2010
# 显示 regime-conditional sector momentum 把 unconditional Sharpe 0.7
# 提升到 1.1；Hong-Lim-Stein 2000 显示 cross-sector lead-lag 信号在
# 小样本下显著。
class When(expr, condition): ...           # condition 真则保留 expr，假则 NaN
class RegimeMix(expr_on, expr_off, regime_id): ...
                                            # 按 ctx['regime_probs'][regime_id] 加权
class MacroSignal(name): ...                # 读 ctx['macro_panel'][name] 并广播
                                            # name ∈ {'vix','yield_curve','credit','dollar'}
class CSDemean(expr): ...                   # 截面减中位 — relative-strength 原语
class PairSpread(a, b): ...                 # log(prices[a]/prices[b])，广播到全部 sector

# ── 算术辅助（2026-05-02 修订加入，用于 baseline 移植）──
class Const(value): ...
class Neg(expr): ...
class Add(a,b); Sub(a,b); Mul(a,b); Div(a,b): ...
class Lag(expr, k): ...                     # k≥0 强制（k<0 视为前视错误）

# ── AST Walker（静态检查）──────────────────────────
def validate_no_lookahead(ast: FactorAST) -> ValidationResult:
    """检查是否引用 t+1 数据；所有时序算子的 window 必须 > 0；
    Lag.k 必须 ≥ 0；Volume/High/Low 阻塞（数据层 stub）。"""

def compute_ast_hash(ast: FactorAST) -> str:
    """SHA256 of canonical form; 用于去重。"""

def compile_to_callable(ast: FactorAST, ctx: dict | None = None
                        ) -> Callable[[pd.DataFrame], pd.Series]:
    """编译为 (prices: pd.DataFrame) -> pd.Series 函数。
    ctx 携带 groups / regime_probs / macro_panel 等共享上下文。"""
```

### 2.8 LLM Proposer Prompt 模板

```python
# engine/agents/factor_mad/proposer.py
PROPOSER_PROMPT = """
You are a quantitative researcher discovering alpha factors for sector ETFs.

CURRENT CONTEXT:
- Universe: {n_etfs} sector ETFs
- Active regime: {regime} (P_risk_on={p_risk_on:.2f})
- Currently active factors: {active_factor_summaries}
- Available DSL operators: {dsl_operators}

RECENT FAILURES (avoid these patterns):
{failure_examples}

CONSTRAINTS:
1. Output {n_proposals} factor expressions in the DSL (n_proposals=5-10).
2. Each must be a MUTATION of one of {parent_factors} — not a from-scratch design.
   Mutation types: param-perturb / op-swap / regime-wrap / cross-sector-wrap.
3. Each must include economic_hypothesis (≥3 sentences, 中文).
4. No t+1 references (validated by AST walker).
5. Prefer factors with |Spearman ρ| ≤ 0.7 vs the parent factor (verified Stage 2).
6. State which regime each factor should perform best in.

OUTPUT JSON SCHEMA (2026-05-02 升级 — 增加可证伪性 + 衰减预期 + 文献定位):
{{
  "proposals": [
    {{
      "name": "...",
      "parent_factor_id": "mom_3m",                    # which active factor it mutates
      "mutation_type": "regime-wrap",                  # one of {{param-perturb, op-swap, regime-wrap, cross-sector-wrap}}
      "dsl_expression": "When(Lag(Close,k=64)/Lag(Close,k=21)-1, MacroSignal(name='vix'))",
      "economic_hypothesis": "...",                    # ≥3 句中文，说明为什么应该有效
      "falsifiability_test": "...",                    # 怎样的 OOS 数据能证伪这个假设？
      "expected_regime": "risk-on" | "risk-off" | "all",
      "expected_decay_months": 12,
      "nearest_published_factor": "Asness 1997 industry MOM",  # 与哪篇文献最相关
      "differentiator": "..."                          # 与该论文因子的关键区别
    }}
  ]
}}
"""
```

**Prompt 升级理由（学术）**：

1. `falsifiability_test` 把 LLM 从"采样器"升级为"研究助手"。Popper 标准：
   假设若不可证伪即非科学。每个候选必须带证伪条件，否则 Stage 4 Critic
   就只能在 ICIR 数值上扯皮，无法触及经济假设本身。

2. `nearest_published_factor` + `differentiator` 防止 LLM 重新发明 Carhart momentum
   再包装成"novel discovery"。McLean-Pontiff 2016 显示 published anomaly OOS
   Sharpe 下降 ~50%；强制定位文献迫使 LLM 直面 publication bias。

3. `parent_factor_id` + `mutation_type` 锁定 mutation-based search（spec §2.6 Stage 1）。
   小样本下 prior-anchored 搜索的统计效率比 free generation 高 3-5×
   （Schölkopf et al. 2021）。

---

## 三、学术深度补充建议（前面未充分展开）

> 顶级学术标准下，仅修复 C1-C3/S1-S4 仍不够。以下是机构 alpha mining 实证标准的补充项。
> 标记为 ★ 的优先级更高（投入产出比好）。

### A1. ★ Survivorship Bias — Universe 层面校正
**问题**：18 个 SPDR sector ETF 全是 still-listed 的"幸存者"。回测从未包含退市样本。
对 sector ETF 而言这个 bias 较小（GICS 分类稳定，退市极少），但**应记入数据局限报告**。

**实施**：
- 在 `DiscoveredFactor` 加 `survivorship_adjusted: bool` 字段
- 当扩展到个股时（Phase 3+）必须使用 CRSP 含退市的全样本

**优先级**：低（sector ETF 影响有限），但**模块化数据层时必须预留接口**。

### A2. ★★ Block Bootstrap CI on ICIR — 替代 iid 假设
**问题**：当前 ICIR 报告是点估计，没有置信区间。月度 IC 有自相关，
直接 `mean/std` 假设 iid 错误。

**实施**：
```python
def block_bootstrap_icir(ic_series, n_iter=1000, block_size=6) -> tuple[float, float]:
    """Politis-Romano stationary bootstrap on IC time series.
    Returns (icir_lower_95, icir_upper_95)."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block_size, ic_series)
    icir_dist = []
    for data in bs.bootstrap(n_iter):
        ic = data[0][0]
        icir_dist.append(ic.mean() / ic.std() if ic.std() > 1e-9 else 0)
    return np.percentile(icir_dist, [2.5, 97.5])
```

放进 Layer 2 报告：候选因子若 95%CI 下界 < 0.1 → 直接拒绝，无需 LLM Critic。

**优先级**：高 — 让所有 ICIR 数值都有不确定性表达，是机构标准。

### A3. ★★ Permutation Test for Spurious IC — 给出 p-value 而非阈值
**问题**：阈值（如 ICIR≥0.30）是经验值；不同因子的 null 分布不同。
应该计算"在零假设下，纯噪音得到这个 ICIR 的概率"。

**实施**：
```python
def permutation_p_value(factor_fn, prices, observed_icir, n_perm=500):
    """随机置换时间标签，计算 null ICIR 分布。
    p-value = P(null_icir >= observed_icir)。"""
    null_icirs = []
    for _ in range(n_perm):
        shuffled_prices = prices.sample(frac=1).reset_index(drop=True)
        shuffled_prices.index = prices.index
        null_ic = compute_ic_series(factor_fn, shuffled_prices)
        null_icirs.append(null_ic.mean() / null_ic.std())
    return np.mean(np.array(null_icirs) >= observed_icir)
```

放进 Layer 2 报告：p < 0.05 才进入下游。

**优先级**：高 — 配合 A2 给出"这个因子真的非随机"的证据。

### A4. ★ Sealed True OOS Holdout — 防 multi-stage overfit
**问题**：CPCV 解决了"研究者在设计因子时已经看到全部数据"的部分问题，
但**因子超参数（如 window=20）的选择仍然在 CPCV 内部完成**，存在二次过拟合。

**实施**：
- 把最近 12 个月数据**密封**为 sealed_oos
- 所有 CPCV / Critic / Refinement **不允许访问** sealed_oos
- 仅在 Stage 6 写 PendingApproval 时计算一次 sealed_oos ICIR
- 若 sealed_oos ICIR < 0.5 × CPCV 中位数 → 自动拒绝

**优先级**：中 — 增加严谨性但减少有效训练样本。建议 Phase 3 启用。

### A5. ★★ Fama-French 5 因子残差化（C2 升级版）
**问题**：C2 只剔除了 MKT beta；学术标准是用 FF5（MKT + SMB + HML + RMW + CMA）残差。
但 sector ETF 没有 SMB/HML 概念，应改用 **sector-rotation 风险因子**：
- MKT (SPY)
- LongRate (TLT - SHY)
- HighYield (HYG - LQD)
- Dollar (UUP)
- Vol (VXX 倒数)

**实施**：在 C2 基础上扩展为 5 因子线性回归，取残差。

**优先级**：高 — 这才是真正的 alpha 定义。但需要数据接入这 5 个对照 ETF。

### A6. ★ Multi-Horizon Consistency Check
**问题**：好因子在 5d/10d/22d/63d 多个 forward window 都应该有相关 IC。
若只在某一个 window 有效，大概率是过拟合到该 window。

**实施**：
```python
def multi_horizon_consistency(factor_fn, prices, horizons=[5, 10, 22, 63]) -> dict:
    """每个 horizon 计算 IC 序列。返回各 horizon ICIR + pairwise correlation matrix。"""
    icirs = {h: compute_icir_at_horizon(factor_fn, prices, h) for h in horizons}
    consistency_score = mean of pairwise correlations of IC series
    return {"icirs": icirs, "consistency": consistency_score}
```

放进 Layer 2 报告：consistency < 0.3 → 标记为 `horizon_specific`，需 Critic 复审。

**优先级**：中 — 简单且有效的过拟合检测器。

### A7. ★★ HMM-Based Regime Conditioning（D3 升级版）
**问题**：当前 VIX 阈值（<18 / >25）是静态、单变量、人为阈值。
2024-2025 年 VIX 中位数在 14-16，"transition"区被吞掉。

**实施**：
- 用 `hmmlearn` 训练 2-state Gaussian HMM 在 [VIX, MKT_ret, credit_spread] 上
- 输出每月的 regime probability
- regime-conditional ICIR 用 P(risk_off) 加权而非硬阈值
- 若 hmmlearn 已被项目使用（regime.py），直接复用

**优先级**：高 — 真正的概率化 regime，与 `regime.py` 一致。

### A8. Causal Validation via Event Study（Phase 3+）
**问题**：相关性不等于因果。FOMC 决议、CPI surprise 等外生事件应该作为天然实验。

**实施**：
- 选定 ~50 个历史 FOMC 决议日
- 计算因子在事件前 5 日 / 事件后 5 日 IC 差异（DiD）
- 若因子声称是"宏观传导"，但事件前后 IC 无变化 → 经济学假设不成立

**优先级**：低 — 实施复杂，留 Phase 3+。

### A9. ★ Calendar / Seasonality Adjustment
**问题**：1 月效应、季末橱窗、月初首日效应可以制造虚假 IC。

**实施**：在 IC 计算前先对 forward returns 做日历回归去除：
```python
fwd_residual = fwd - (β_jan * is_january + β_eom * is_month_end + ...)
```

**优先级**：中 — 简单且消除已知混淆。

### A10. Capacity / Crowding（仅供 DSL 接口预留）
**问题**：sector ETF 流动性极佳，crowding 不是问题。但因子产出 score
应该乘以 `min(volume, ADV_30d) / ADV_30d` 的流动性权重，使框架对未来个股扩展兼容。

**优先级**：低（sector ETF 阶段），但 DSL 应原生支持 `Volume` 原语。

---

## 四、不在本规格范围内（明确 defer）

| 项目 | 理由 |
|---|---|
| Genetic Programming Search | 432 样本无意义，等 Universe ≥50 ETF |
| Bayesian Hierarchical Model | 跨资产 strength sharing，等多资产类成熟 |
| 替代数据接入（卫星/Google Trends） | API 成本高，与 sector alpha 收益比例不匹配 |
| Stochastic Discount Factor 框架 | 学术上正确但工程复杂度过高 |
| BNN / MC Dropout 不确定性 | DSR + PBO + Bootstrap CI 已足够 |
| 实时因子（intraday） | 与系统月度/季度节奏不符 |

---

## 五、施工 Sprint 规划

### S0｜Agent 基础设施（前置硬依赖，2-3 天）
**交付物**：
- `engine/agents/base.py` (Agent 基类 + Trigger + AgentResult)
- `engine/agents/event_bus.py` (SQLite-backed)
- `engine/memory.py` 加 `agent_runs`、`agent_events`、`factor_failure_memory` 表 + 迁移
- `engine/orchestrator.py` 加 `run_quarterly()`
- `engine/daily_batch.py` 删 daemon thread，改调 `run_quarterly()`

### S1｜方法学硬伤修复（1-2 天）
**交付物**：
- C1: Purged CPCV (`engine/factor_mad.py` `run_layer2_audit` 重写)
- C2: Beta-neutral IC (`compute_monthly_ic` 增加 `neutralize_beta=True` 参数)
- C3: MI 不重叠 step + null 分布
- S1: HAC t-stat (`compute_harvey_liu_t` 重写)
- S2: 真实 n_test 入库 (`DiscoveredFactor.n_test_obs`)
- S3: 负 ICIR 利用 (`get_factor_mad_scores`)
- S4: bootstrap CI on R² (`_classify_audit`)

### S2｜DSL + LLM Proposer（3-5 天）
**交付物**：
- `engine/agents/factor_mad/dsl.py` (数据原语 + 算子 + AST + 编译器)
- `engine/agents/factor_mad/proposer.py` (LLM Proposer)
- `engine/agents/factor_mad/search.py` (搜索循环)
- 把现有 `FACTOR_REGISTRY` 4 个内置因子改写为 DSL 表达式作为基准

### S3｜Proposer-Critic Refinement Loop + Failure Memory（2-3 天）
**交付物**：
- `engine/agents/factor_mad/critic.py` (LangGraph 多轮辩论)
- `engine/agents/factor_mad/memory.py` (FactorFailureMemory I/O)
- EventBus 接入：`approval.*` / `attribution.factor_blamed` 订阅
- `pages/_archive/admin.py` 重新激活后增加 Supervisor 决策回流

### S4｜DSR / PBO / Lifecycle / 学术深度补充（2-3 天） ✅ 完成 2026-05-02
**交付物**：
- ✅ DSR (Bailey-LdP 2014): `engine/factor_mad_methodology.py::compute_deflated_sharpe`
- ✅ PBO (Lopez de Prado 2014): `engine/factor_mad_methodology.py::compute_pbo_from_cpcv`
  - per-factor sign-flip rate；非 zoo PBO（Universe ≥ 50 ETF + GP search 启用后再升级）
- ✅ A2 Block Bootstrap CI on ICIR: `block_bootstrap_icir_ci`（arch.bootstrap.StationaryBootstrap）
- ✅ A3 Permutation p-value: `permutation_p_value_icir`
  - **方法学修订**：spec 原文写 block-shuffle，实测会保留 mean+std 导致 ICIR 退化为常数 → null 分布固定为 observed 值，p=1。已改为 sign-flip 一样本检验（Pitman 1937 标准方法），signal 与 noise 区分明显（signal p=0.0000，noise p=0.748）。
- ⏸ A6 Multi-Horizon Consistency: 推迟（与 §一·S4 已实现的 `bootstrap_r2_ci` 重叠度评估后再做）
- ✅ A7 HMM Regime Conditioning: `fit_hmm_regime_probabilities` + `regime_weighted_icir`
  - hmmlearn 0.3.3 安装于 2026-05-02；2-state Gaussian HMM；按列 0 mean 标定 risk_off
  - 与 `engine/regime.py` 互不依赖（两者都跑，下游各自消费）
- ✅ A9 Calendar Adjustment: `calendar_adjust_ic_series`
  - **位置修订**：spec 原文"对 forward returns 做日历回归"在单截面下退化为 no-op，改为对 IC 时序回归 Jan / month-end / quarter-end dummies。
- ✅ 因子 lifecycle: `engine/agents/factor_mad/lifecycle.py::run_lifecycle`
  - 单点 pulse-check + 自动 archive（current_icir < max(0.05, MDE) → status='retired'，emit `factor.retired`）
  - 50% decay warning（current_icir < 0.5 × promote_icir → emit `factor.decay`）
  - half-life ML 估计 deferred（需 DecayHistory 表 + 至少 4 季度历史）

**Layer-2 verdict 修订**：auto-approve 增加 `permutation_p < 0.10` 约束；DSR/PBO/BB CI 仅 advisory。
**Critic 早停**：`DSR ≥ 0.50` AND verdict=needs_review → promoted=True（stop_reason="dsr_promote_X.XXX"）。
**orchestrator 缝合**：`TradingCycleOrchestrator.__init__` 调 `subscribe_factor_mad_handlers()`；`run_quarterly` Step 3 调 `FactorMADAgent().run(...)`。

### 总工时：~10-15 天 / 实际 S0-S4 ≈ 10 天（2026-05-02 收口）

### 暂定不做（Defer 到 M5+）
- A1 Survivorship（接口预留）
- A4 Sealed OOS（Phase 3 启用）
- A5 FF5 Residualization（5 因子数据接入后）
- A8 Causal Validation
- A10 Crowding

---

## 六、文件改动 Map

### 新建
- `engine/agents/__init__.py`
- `engine/agents/base.py`
- `engine/agents/event_bus.py`
- `engine/agents/factor_mad/__init__.py`
- `engine/agents/factor_mad/agent.py`
- `engine/agents/factor_mad/proposer.py`
- `engine/agents/factor_mad/critic.py`
- `engine/agents/factor_mad/dsl.py`
- `engine/agents/factor_mad/search.py`
- `engine/agents/factor_mad/lifecycle.py`
- `engine/agents/factor_mad/memory.py`
- `engine/agents/factor_mad/validation.py`  (CPCV / DSR / PBO / Bootstrap / Permutation)

### 重写
- `engine/factor_mad.py` → 保留为薄 shim，仅 re-export 公共 API
  - 内部实现迁移到 `engine/agents/factor_mad/`
  - 保持向后兼容（signal.py / daily_batch.py 不需改 import）

### 修改
- `engine/memory.py` — 加 3 张新表 + 5 个字段 + 迁移函数
- `engine/orchestrator.py` — 加 `run_quarterly()`
- `engine/daily_batch.py` — 删 daemon thread，改调 orchestrator
- `pages/_archive/admin.py` 或新 admin 页 — Supervisor 决策回流 EventBus

### 不动
- `engine/signal.py` — 仍调用 `get_factor_mad_scores()`（API 兼容）
- `engine/regime.py` — A7 HMM 复用，不改
- `engine/debate.py` — Stage 4 Critic 复用 LangGraph 框架
- `engine/era.py` / `universe_review.py` / `failure_attribution_agent.py` — 后续逐步迁移到 agents/，本次不动

---

## 七、风险与局限（学术诚实声明）

1. **样本量根本约束**：Universe=18 ETF × 月度 IC，年新增观测仅 12 个。
   即使 CPCV + Bootstrap，统计功效仍有限。所有 ICIR 数值在 24-36 月窗口下
   都应视为"指示性"而非"决定性"。**不要把方法学严谨性当成数据充分性。**

2. **LLM Proposer 的 hindsight bias 不可完全消除**：LLM 训练数据
   包含 2024-2025 年市场知识，生成的"动量因子"自带后视。
   缓解：在 prompt 中**禁止**引用具体时段的市场表现；用历史 1990-2010 数据再做 sealed validation。

3. **多重检验问题在 LLM 时代加剧**：每季度 LLM 生成 50 个候选 → 一年 200 个。
   即使 BH α=0.10，期望仍有 20 个 false positive。配额 cap=8 提供一定保护。
   **真正的护城河是经济学假设的可证伪性，不是 p-value。**

4. **CPCV 计算成本**：45 个 OOS 路径 × 每路径全量 backtest，单因子可能 5-30 秒。
   50 个候选 × 30 秒 = 25 分钟，季度作业可接受。但**禁止**把 CPCV 用进日频流程。

5. **DSL 表达力 vs 安全性的 tradeoff**：
   AST 静态检查能挡住 90% 的 t+1 错误，但**复杂复合算子**（如自定义 EWMA 衰减）
   仍可能引入隐性前视。Layer 1 MI 扫描是最后兜底。

6. **Operations 架构的可观测性还不够**：本规格没有提到 Prometheus / Grafana 接入。
   当前仅靠 SQLite agent_runs 表 + Streamlit Admin UI。**生产级部署时
   应加 metrics 端点**，本规格视为 Phase 4+。

---

## 八、执行节奏建议

```
Week 1: S0 (基础设施) + S1 (方法学修复)
Week 2: S2 (DSL + Proposer) 第一阶段
Week 3: S2 收尾 + S3 (Critic loop)
Week 4: S4 (DSR/PBO + 学术补充)
Week 5: 集成测试 + Admin UI 调整 + Supervisor 培训
```

**每周末**：跑一次端到端 dry-run（不实际写入 PendingApproval），观察候选质量。
**每个 Sprint 结束**：写一份 `report_factor_mad_sprint_X.md` 记录方法学决策。

---

*本规格为 2026-05-02 v1。所有决策已锁定，开工后变更需走 Supervisor 审批流。*

---

## 九、施工状态汇总（2026-05-02 收口）

| Sprint | 状态 | 备注 |
|---|---|---|
| S0 Agent 基础设施 | ✅ | base.py / event_bus.py / agent_runs+events 表 / orchestrator.run_quarterly |
| S1 方法学硬伤修复 | ✅ | C1 CPCV / C2 Beta-neutral / C3 MI null / S1 HAC / S2 真实 n_test / S3 负 ICIR / S4 bootstrap R² |
| S2 DSL + LLM Proposer | ✅ | 28 operators DSL + Proposer + Stage 1+2 search |
| S3 Critic + Failure Memory | ✅ | LangGraph Stage 4 + EventBus 接入 + Approval 页 |
| S4 DSR / PBO / Lifecycle / 学术补充 | ✅ | 7 项 + verdict gate 修订 + Critic DSR 早停 + orchestrator 缝合 |

**S4 方法学修订（已记入相应小节）**：
1. A3 permutation 改 sign-flip（block-shuffle 退化）
2. A9 calendar adj 改对 IC 时序回归（forward returns 单截面无时序）
3. PBO 走 per-factor sign-flip rate（zoo PBO 等 Universe ≥ 50 ETF）
4. A6 Multi-Horizon Consistency 推迟（与 bootstrap_r2_ci 评估后再做）

**Defer 项保持**（接口预留，本次未实施）：A1 Survivorship / A4 Sealed OOS / A5 FF5 / A8 Causal / A10 Crowding / GP Search / Bayesian Hierarchical / SDF / BNN。
