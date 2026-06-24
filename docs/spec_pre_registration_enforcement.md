## Spec — Pre-Registration Enforcement Layer (PR-Enforce v1)

**起草日期**：2026-05-03
**状态**：spec 已定稿，等待用户确认开工
**优先级**：P0（agentic AI capability frontier 之一；与 Paper Trading E 的可发表性强耦合）
**前置触发**：用户在"自动推理自动迭代"方向讨论中授权这一项；其余两项（Adversarial OOS sampling / Auto-deflation 单独立项）已撤回或并入 backlog
**学术依据**：
- Camerer et al. *Science* 2018, "Evaluating the replicability of social science experiments"
- Olken *JEP* 2015, "Promises and Perils of Pre-Analysis Plans"
- Harvey & Liu *Critical Finance Review* 2020, "False (and Missed) Discoveries in Financial Economics"
- López de Prado 2018, *Advances in Financial Machine Learning* §11 Backtesting Pitfalls
**Origin / Extension 标注**：核心思想（spec hash + amendment ledger + HARKing 检测）是 Olken 2015 经济学预登记范式 + Harvey-Liu 多重检验框架的 **直接移植**；本项目原创 extension 是 (1) 把 hash 锚到 git blob，(2) 与既有 `EFFECTIVE_N_TRIALS` ([engine/backtest.py:84-104](engine/backtest.py#L84-L104)) 动态联动，(3) HARKing 检测规则 R1-R4 是针对本项目 spec 形态定制。

---

## 一、目的与边界

### 1.1 目的

把"科学纪律"从 **依赖人记忆** 升级为 **harness 自动强制执行**。具体做三件事：

1. **Spec 不可静默修改**：任何已注册 spec 的 hash 一旦在 backtest/paper trading 输出里被引用，后续修改必须走 amendment workflow，留可审计 trail。
2. **n_trials 自动累加**：每个 amendment 触发一次 `EFFECTIVE_N_TRIALS += k`（k 由修改性质决定），自动反馈到 DSR deflation。
3. **HARKing 自动检测**：定义 4 条规则（R1-R4），系统自动 flag 可疑模式（事后改假设、阈值微调、未声明 trial、predictions section 重写）。

### 1.2 这是什么、不是什么

| 是 | 不是 |
|---|---|
| 工程化的 scientific discipline 强制层 | 不是新 agent，全程零 LLM |
| 一张新表 + ~300 行代码 | 不是 LLM-as-judge（违反 [feedback_no_llm_as_judge](../memory/feedback_no_llm_as_judge.md)） |
| 与 EFFECTIVE_N_TRIALS / DSR 联动 | 不重新实现 DSR/PBO，那部分已在 [backtest.py](engine/backtest.py) |
| Paper Trading E 可发表性的硬前置 | 不阻塞 E 现有 arms（已注册的 spec retro-snapshot 即可） |
| 复用 git blob hash | 不引入额外加密原语 / blockchain 类玩具 |

### 1.3 严格边界（明确不做）

| 不做 | 理由 |
|---|---|
| 不强制 spec 必须用某种模板 | 现有 15 个 spec 形态各异，强模板化会阻塞工作；只要文件存在 + hash 稳定即可 |
| 不引入 IPFS / 区块链 / 第三方 timestamping | git 本身已是 cryptographic content-addressable store；外部 timestamping 除非投稿审稿要求否则是 vanity |
| 不阻塞 amendment | 系统**记录**而非**禁止**；纪律强制靠 transparency，不靠 lockout |
| 不回填历史 spec 的"原版"hash | 历史无法重构；retro-snapshot 即注册当下版本，并标 `retro_registered=True`，**不计入 forward integrity** |
| 不处理非 spec 文件的 HARKing | 配置文件 / config dict 的 hash 留 v2；v1 只覆盖 docs/spec_*.md |
| 不做 UI 重设计 | 复用现有 Decision Log / Backtest 页面追加 1 个 panel 即可 |

### 1.4 必要性论证（替代 power analysis）

按 [feedback_spec_power_analysis](../memory/feedback_spec_power_analysis.md) 要求，量化决策的 spec 必须事前算 power；本 spec 不直接产 alpha，故改用**必要性量化**：

- 现有 spec 文件：15 份（`Glob docs/spec_*.md`）
- 当前每份 spec 平均生命周期内修改次数（粗估，按 git log 假设）：3-5 次
- 每次 amendment 若不计入 n_trials，等价于 **1 次未声明的 hyperparameter 试验**
- 6 个月剩余 capability demonstration 周期，预计新增 spec 5-8 份 + 每份 2-4 次 amendment → **静默 trial 数估算 30-50**
- DSR 阈值在 EFFECTIVE_N_TRIALS 从当前值（[backtest.py:99](engine/backtest.py#L99) 给出 raw=1800、effective=待审）增加到 1800 + 50 时变化：
  - `t_threshold_DSR ≈ Φ⁻¹(1 - α/n_eff)` 单调上升
  - 在 α=0.05 / 现有 baseline NW t=2.64 附近，新增 50 trial 把 deflated p-value 上推 ~5-10%（视相关性结构）
- **结论**：若不做 pre-registration，6 个月后 Paper Trading E 即使阳性也无法在严肃学术场合给出 honest deflated p-value；若做，多余成本 ~300 行 + 每次 amendment 1 行 commit message 纪律。**净增益强正**。

---

## 二、架构

### 2.1 组件图

```
┌──────────────────────────────────────────────────────────────┐
│ 1. SpecRegistry (新表, engine/memory.py)                      │
│   spec_path · git_blob_hash · registered_at                   │
│   amendment_log (JSONB) · status · retro_registered           │
│   first_referenced_at · n_trials_contributed                  │
└──────────────────────────────────────────────────────────────┘
                       ▲                       ▲
                       │ register_spec(path)   │ amend_spec(path, reason)
                       │                       │
┌──────────────────────────────────────────────┴──────────────┐
│ 2. engine/preregistration.py (新文件, 主逻辑 ~200 行)        │
│   - register_spec(path)        # 首次注册                    │
│   - amend_spec(path, reason, kind)  # 追加 amendment         │
│   - validate_reference(spec_path, run_id)  # backtest 引用前检 │
│   - detect_harking() -> List[Flag]  # 定时任务跑 R1-R4       │
│   - compute_n_trials_contribution(amendment_kind) -> int     │
└──────────────────────────────────────────────────────────────┘
                       │                       │
                       ▼                       ▼
┌────────────────────────┐   ┌───────────────────────────────┐
│ 3. backtest.py 改动     │   │ 4. DecisionLog 改动            │
│   _estimate_effective_  │   │   新增 spec_hash 字段          │
│   n_trials() 增加        │   │   save_decision() 自动填充     │
│   pre-reg 贡献          │   │                                │
└────────────────────────┘   └───────────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │ 5. UI panel (pages/decision_  │
                       │    log.py 或 backtest.py)     │
                       │   - 显示 spec 注册状态         │
                       │   - HARKing flag 红牌展示      │
                       │   - amendment timeline         │
                       └───────────────────────────────┘
```

### 2.2 数据 schema（SpecRegistry）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | Integer PK | autoincrement |
| `spec_path` | String(255) UNIQUE | 相对 repo root，e.g. `docs/spec_paper_trading_three_arm_e.md` |
| `git_blob_hash` | String(40) | SHA1 git blob hash（`git hash-object` 输出）；首次注册时的版本 |
| `registered_at` | DateTime | 首次 register_spec 调用时间 |
| `amendment_log` | Text(JSON) | `[{"at": ISO8601, "reason": str, "kind": Enum, "new_hash": str, "n_trials_added": int}, ...]` |
| `status` | String(16) | `active` / `superseded` / `archived` |
| `retro_registered` | Boolean | True = 历史 spec 注册当下版本，不算 forward integrity |
| `first_referenced_at` | DateTime nullable | 第一次被 backtest/paper trading run 引用的时刻；HARKing R1 用 |
| `n_trials_contributed` | Integer | 累计贡献给 EFFECTIVE_N_TRIALS 的次数（首次 1 + 每次 amendment k） |
| `current_hash` | String(40) | 当前文件 hash，用于检测 silent edit（未走 amendment 但 hash 已变） |
| `last_validated_at` | DateTime | 最近一次 validate_reference 检查时间 |

### 2.3 Amendment 分类与 n_trials 贡献表

| Kind | 触发条件（人工声明） | n_trials += | 备注 |
|---|---|---|---|
| `clarification` | 仅措辞 / 排版 / typo / 引文补充 | 0 | 必须 diff 不涉及任何数值/predictions/hypothesis |
| `scope_narrow` | 缩小适用范围（删除某 sector/regime） | 0 | 范围缩小不增加自由度 |
| `threshold_tweak` | 调任一数值阈值（如 NW t≥1.5 → ≥1.8） | 1 | 经典 p-hacking surface |
| `hypothesis_amend` | H1/H2 prediction 文本变化 > 20 字符 | 3 | 重视化为 ~3 个新 trial（Harvey-Liu 风格保守上界） |
| `endpoint_swap` | 主指标更换（Sharpe → Sortino 之类） | 5 | endpoint switching 是最严重 HARKing 之一 |
| `superseded` | 整 spec 被新 spec 替换 | 0（旧 spec 不再 contribute）+ 新 spec register | 旧 status → superseded |

**Amendment kind 由人工在 amend CLI 里显式声明，无 LLM 介入**。系统只校验 `kind` 与实际 diff 的最低一致性（如 `clarification` 不允许数字字段变化）。

### 2.4 HARKing 检测规则

| Rule | 名称 | 检测 | Severity |
|---|---|---|---|
| **R1** | Late silent edit | `current_hash != amendment_log[last].new_hash` 且 `first_referenced_at IS NOT NULL` | **CRITICAL** |
| **R2** | Threshold drift without amendment | spec 文件 grep 数值变化（正则匹配 `t≥X` / `Sharpe>Y` 等模式）但近 7 天无 amendment | HIGH |
| **R3** | Unannounced trial | DecisionLog/BacktestRun 新增记录引用 spec_hash 不在 SpecRegistry | HIGH |
| **R4** | Predictions section rewrite | amendment_log 累计 hypothesis_amend ≥ 2 次 | MEDIUM |

R1 是最关键的 — silent edit 是 HARKing 的纯净形式。R2 用简单正则即可（不依赖 LLM）。R3 防 spec 绕过。R4 是软警告，提醒 spec 已经被改太多次。

### 2.5 Workflow（人工纪律 + 自动校验）

**首次注册 spec**：
```
$ python -m engine.preregistration register docs/spec_xxx.md
[OK] Registered: docs/spec_xxx.md
     hash: a3f8e2c... (git blob)
     registered_at: 2026-05-03T14:23:11Z
     n_trials_contributed: 1
```

**修改 spec**：
```
$ python -m engine.preregistration amend docs/spec_xxx.md \
    --kind threshold_tweak \
    --reason "回测显示 t≥1.5 power 不足，提升到 t≥1.8"
[OK] Amendment recorded.
     new_hash: b7d2f91...
     amendments_total: 2
     n_trials_added: 1 (cumulative: 4)
     EFFECTIVE_N_TRIALS now: 1804
```

**Backtest 引用**（自动）：每次 `_compute_metrics()` 调用前，如果 backtest run 关联某 spec，自动调 `validate_reference(spec_path, run_id)`，hash mismatch 触发 R3 flag 并写入 BacktestRun.harking_flags。

**HARKing 检测**（定时）：daily cron 调 `detect_harking()`，结果写入新表 `HARKingFlag`，UI 红牌展示。

---

## 三、Sprint 规划（4 sprint，每 sprint ≤ 1.5 天 LOC）

### Sprint 1 — Schema + 注册 / 修改 CLI（最小可用）
- **新增** `engine/preregistration.py`：`register_spec`, `amend_spec`, `_compute_git_blob_hash`, `_load_registry`, CLI entrypoint
- **修改** `engine/memory.py`：新增 `class SpecRegistry(Base)`（按 §2.2 schema）
- **新增** alembic 风格 migration（或 `Base.metadata.create_all` 自带）
- **回填**：脚本一次性把 `docs/spec_*.md`（除本 spec）注册为 `retro_registered=True`
- **验收**：`python -m engine.preregistration list` 输出 14+1 行；`amend` 一次随便哪个 spec，amendment_log 正确累加

### Sprint 2 — backtest / DecisionLog 集成
- **修改** `engine/backtest.py`：
  - `_estimate_effective_n_trials()` 增加 `pre_registration_contribution` 项 = SUM(SpecRegistry.n_trials_contributed)（仅 retro_registered=False 的）
  - `_N_TRIALS_AUDIT` dict 新增 `pre_registration` key
  - 新增 spec_hash 参数到 `_compute_metrics`，未声明则不阻断但写入 warning
- **修改** `engine/memory.py` `class DecisionLog`：新增 `spec_hash = Column(String(40), nullable=True)`
- **修改** `engine/sector_pipeline.py` save_decision 调用点：自动从 active spec context 注入 spec_hash
- **验收**：跑一次 backtest，观察 DSR 阈值随 EFFECTIVE_N_TRIALS 变化；DecisionLog 新行 spec_hash 非空

### Sprint 3 — HARKing 检测 + UI 红牌
- **新增** `engine/preregistration.py::detect_harking()`：4 条规则
- **新增** `class HARKingFlag(Base)`（极小 schema：rule, spec_path, detected_at, resolved_at, severity, notes）
- **新增 cron**（复用 [agentic_orchestration_v1](spec_agentic_orchestration_v1.md) scheduler）：daily 23:00 跑 detect_harking
- **修改** UI（[pages/backtest.py](pages/backtest.py) 或新建 panel）：
  - SpecRegistry 表格 + amendment timeline
  - HARKingFlag 红/黄/橙牌
  - `EFFECTIVE_N_TRIALS` 当前值 + breakdown（含 pre_registration 贡献）
- **验收**：手动 silent-edit 一个 spec 文件 → 24 小时内 R1 红牌出现；amend 修复后 flag → resolved

### Sprint 4 — 文档 + 学术表述
- **新增** `docs/preregistration_methodology.md`：放给外部读者看的 methodology section（capability demo 叙事用）
- **修改** `docs/falsification_chain.md` 与 `docs/capability_evidence.md`：加入 pre-registration 作为 capability dimension
- **修改** `README.md` agentic capabilities 部分：加 "Self-Enforced Scientific Discipline" 一行
- **验收**：方法论文档可独立读懂；外部 reviewer 能复现 hash 校验

---

## 四、与既有系统的耦合点

| 系统 | 耦合方式 | 风险 |
|---|---|---|
| `EFFECTIVE_N_TRIALS` ([backtest.py:104](engine/backtest.py#L104)) | 直接相加 | 启动时初始化，新 amendment 当 cycle 生效；冷启动一次性 jump 已 documented |
| DecisionLog | 新增 nullable column | 历史行 spec_hash NULL（pre-feature），不影响 |
| Paper Trading E ([spec_paper_trading_three_arm_e.md](spec_paper_trading_three_arm_e.md)) | E 的 spec 自身被 retro-registered；后续若 amend E 的判定阈值，n_trials += k | 直接影响 E 的 deflated p-value；这正是目的 |
| Forecast Verification Phase 0 ([spec_forecast_verification_phase0.md](spec_forecast_verification_phase0.md)) | 同上 | 同上 |
| `agentic_orchestration_v1` cron | 复用 scheduler 跑 detect_harking | 0 |
| FactorMAD ([spec_factor_mad_redesign.md](spec_factor_mad_redesign.md)) | 暂停状态，spec retro-register 即可；恢复时若 amend 才贡献 trial | 0 |

---

## 五、退路与失败条件

| 情景 | 应对 |
|---|---|
| Sprint 1 后发现现有 spec 文件命名 / 路径不规范导致 hash 不稳定 | 先做 spec normalize pass（行尾、BOM、frontmatter timestamp 剥离），不阻断 |
| 用户嫌每次 amend 跑 CLI 麻烦 | 提供 git pre-commit hook 自动 detect spec edit 并 prompt amend；不强制 |
| HARKing 红牌频繁误报 | 调 R2 正则白名单；R4 阈值上调；保留人工 dismiss 机制 |
| 回测 n_trials 暴涨导致没有任何阳性 deflated 结果 | 这就是真相 — 接受它，调整 baseline 期望，**不要回退 pre-registration** |
| 6 个月后 0 次 amendment 触发任何 flag | 说明纪律已内化；工程价值仍在（capability evidence + 外部 reviewer 信任） |

---

## 六、与"自动推理自动迭代"主轴的定位

本 spec 是 [project_reframe_2026-05-03](../memory/project_reframe_2026-05-03.md) reframe 后 **agentic capability frontier 的实质性扩展**：

- 老叙事："agentic AI 能产更多假设" → 学术圈一眼看穿，无说服力
- 新叙事："agentic AI 能 **self-enforce scientific discipline**，把 HARKing / p-hacking / endpoint switching 的 attack surface 工程化堵住" → 这是 2024-2026 应用 ML in finance 真正在 push 的 frontier，与 López de Prado / Harvey / Bailey 所有近期 keynote 的方向一致

完成后，"capability_evidence.md" 可以新增一条 axis：**Methodological Integrity Automation**，与 Forecast Verification / Paper Trading / Falsification Chain 并列。这是项目唯一不依赖 alpha 实证就能 stand-alone 的 capability dimension，独立于 calendar-bound 的 Clean Zone 验证。

---

## 七、决策与签字

| 决策 | 内容 | 签字 |
|---|---|---|
| D1 | 用 git blob hash 而非自定义 hash | ✓（避免重复造 cryptographic 原语） |
| D2 | 不引入 LLM 任何参与 | ✓（[feedback_no_llm_as_judge](../memory/feedback_no_llm_as_judge.md)） |
| D3 | retro-register 历史 spec 而非阻塞 | ✓（保持 paper trading E 不中断） |
| D4 | n_trials 贡献按 amendment kind 分级 | ✓（Harvey-Liu 风格保守，不"一刀切 +1"也不"全免"） |
| D5 | 不阻塞 amendment，只 transparency | ✓（执行靠 reviewer 看红牌，不靠 lockout） |
| D6 | UI panel 复用现有页面 | ✓（不打开 [project_ui_redesign](../memory/project_ui_redesign.md) 子项） |

**待用户确认后开 Sprint 1。**
