# Operating Model v1.0 — Macro Alpha Pro

**起草**: 2026-05-19
**目的**: 整个项目的单一总图 (single map)。新人 / 未来的你 / 任何 agent 读这一份就能理解"系统长什么样、谁负责什么、什么时候做什么"。
**状态**: LIVING DOCUMENT — 自由演进,不走 spec 预注册流程 (这是 reference,不是 hypothesis)。
**维护**: 任何架构层变动后更新本文。本文是 RACI "Accountable" 格 + cadence 的书面合同。

---

## 一、一句话定位

> 一个单人运营的微型量化基金的**完整运营流程预演 (shadow-run)**。纸面交易阶段建立 operating model,2028-05 (24mo OOS gate) 真金白银时流程已 battle-tested 两年。

任何不服务于"2028 真钱时这套流程已经跑顺"的功能 = resume polish,该砍。

---

## 二、四层架构 (Four-Tier Model)

机构 LLM 只活在非赚钱的 4 层。本项目严格对齐。

| 层 | 用 LLM | 本项目组件 | Doctrine |
|---|---|---|---|
| **L1 赚钱核心** alpha/组合/执行 | **绝不** | 5 strategies: K1 BAB / D-PEAD / Path N / CTA PQTIX / AC TLT-GLD | 0-LLM-in-DECISION |
| **L2 风控/运维/监控** | LLM 当解读器 | RM / DQ / Anomaly Sentinel / Attribution agents + Watchdog cron | risk-side-not-alpha-side |
| **L3 研究/PM 工作流** | LLM 当 copilot | Devil's Advocate + Chief of Staff + Claude Code | 人保留决策权 |
| **L4 治理/合规** | LLM 当记录员 | Audit Recorder + SpecRegistry + Morning Briefing | 人 curate Tier 3 |

---

## 三、决策权模型 (RACI)

**核心铁律: 没有任何 agent 是 Accountable。** PM 永远担责。这是 0-LLM-in-DECISION 用组织语言重述。

| 角色 | 谁 | 说明 |
|---|---|---|
| **R**esponsible 干活 | 系统引擎 (确定性) + agents (advisory) | strategy cron 产 signal;agent 产 finding/critique |
| **A**ccountable 担责 | **PM (你),永远** | 所有 halt-or-proceed / deploy / 资本决策 |
| **C**onsulted 被咨询 | 7 个 agent | 通过 Chief of Staff 路由或直接 chat |
| **I**nformed 被告知 | audit trail | SpecRegistry / alert tables / briefing / case (待建) |

单人运营 = 你戴所有帽子。agent 的价值 = **不雇团队就模拟一个团队的专家评审 (role multiplication)**,不是自动化。

---

## 四、Agent 名册 (7 persona + 2 cron)

| Agent | 层 | 职责 | 工具 | 决策权 | spec/role |
|---|---|---|---|---|---|
| **Chief of Staff** | L3 | 单一对话入口,路由到 6 专家 | delegate_to_specialist + recall_past_turns + lookup_spec + read_project_memory | 只读·路由 | spec id=74 |
| Risk Manager | L2 | 12 模式 pre/post-trade 风控门 | query_recent_alerts / read_today_book_state / lookup_strategy_status / lookup_spec / read_project_memory | 只读·advisory | spec id=69 |
| DQ Inspector | L2 | 数据层 10 模式门 + 实时 pre-batch | run_dq_pre_batch_check + 5 共享工具 | 只读·advisory | spec id=70 |
| Anomaly Sentinel | L2 | 单 ticker forensic z-score | query_recent_anomalies / forensic_ticker_check + 3 共享 | 只读 | role_id=anomaly_sentinel_forensic |
| Attribution Analyst | L2 | sleeve/strategy P&L 拆解 | read_nav_history + 4 共享 (NO 因子回归 — 老实拒绝) | 只读 | role_id=attribution_analyst_forensic |
| Audit Recorder | L4 | 治理 trail (report not rule) | query_audit_findings / query_audit_runs + 3 共享 | 只读·只汇报不裁决 | role_id=audit_recorder_governance |
| Devil's Advocate | L3 | 反事实/p-hacking 批判 (V4 Pro, evidence-only) | NO tools (单轮) | 只读·只批判不验证 | role_id=devils_advocate_constrained_evidence |
| ~Watchdog~ | L2 | 日运维监控 (29 规则) | cron, 非 persona | 自动状态修复 (非代码) | 生产 06:10 SGT |
| ~ETF Holdings~ | L2 | 月度 LLM 风险筛查 | cron, 非 persona | 只读 | spec id=49, 06:30 SGT |

**Pattern 5 ban**: agent 之间无自主通信。专家只通过 CoS 串行调用,彼此 context 隔离。

---

## 五、运营节奏 (Cadence) — 一切按周期挂载

机构的 agent/workflow 不是随机调用,是挂在固定 cadence 上。下表是项目的心跳。✅=已建 ⏳=待建。

### 日内 (Daily)
| 时刻 | 动作 | 组件 | 状态 |
|---|---|---|---|
| 06:10 SGT | 运维监控 | Watchdog cron | ✅ |
| 06:30 开盘前 | 数据门 (能不能用) | DQ pre-batch (Mode 1-4) | ✅ |
| 06:30 开盘前 | 生成 signal (无 LLM) | 5 strategy | ✅ |
| 06:30 开盘前 | pre-trade 风控门 | RM pre-trade gate | ✅ |
| 收盘后 | NAV roll + anomaly scan + P&L | orchestrator + AnomalyFlag | ✅ |
| **次日晨** | **PM scan → 决策** | **Morning Briefing (Wave A)** | ✅ |

### 周度 (Weekly) — ⏳ 待建 (Wave B)
- strategy 表现 review · 风险限额 review · OPEN findings 清理 (case table)

### 月度 (Monthly) — ⏳ 待建
- attribution 深挖 · capacity/流动性 · 模型漂移检查

### 季度 (Quarterly) — ⏳ 待建
- strategy lifecycle (上线/扩容/退役) · 治理审计 · 资本再分配

### 事件触发 (Event-driven)
| 事件 | workflow | 状态 |
|---|---|---|
| 新 strategy 上线 | pre-deploy review (DA+RM+DQ+AR 集合评审) | ⏳ Wave C |
| drawdown 事件 | DD investigation workflow | ⏳ |
| 数据中断 | 降级模式协议 | ⏳ (部分: Watchdog 3-Layer mode_2) |
| spec 改动 | amend_spec + pre-commit drift hook | ✅ |

---

## 六、当前建造状态 (项目地图)

### ✅ 已完成
- **L1**: 5 strategy 全生产 (Sharpe 0.54 / MaxDD -10.9% / scale-invariant $10k-$1B)
- **L2**: RM v1.0 (spec 69) + DQ Phase 6c (spec 70) + Anomaly/Attribution persona
- **L3**: Chief of Staff supervisor (spec 74) + Devil's Advocate + 6 专家全 persona
- **L4**: Audit Recorder + SpecRegistry + drift hook (.githooks/pre-commit) + Morning Briefing (Wave A)
- **记忆**: Tier 2.5 跨会话语义召回 (ChatTurnEmbedding) + Tier 3b 项目记忆 + 多 session 支持
- **测试**: 307 passed / 9 skipped (持续)

### ⏳ 进行中 / 下一步
- Wave B: agent_cases 表 (open/investigating/resolved/ignored 生命周期)
- Wave C: strategy pre-deploy review workflow
- Wave D: case → amendment → commit lineage chain
- PM Doctrine (见 docs/pm_doctrine_*.md — 填完接进所有 agent system prompt)

### 🚫 拒绝 / 永久 defer
- Pattern 5 (agent 自主辩论) — 永久禁
- LLM 进决策路径 (下单/sizing/审批自己) — 永久禁
- Level 4 SDK 全自治 engineer — 拒绝
- Research Co-Pilot / Quant Engineer 独立 agent — SUPERSEDED → Claude Code

---

## 七、距离企业级还差什么 (诚实清单)

| 企业级要素 | 现状 | 差距 |
|---|---|---|
| 文档化 operating model | 本文 (刚建) | ✅ 补上了 |
| 清晰决策权 | RACI (本文 §三) | ✅ |
| Runbook (出事怎么办) | 部分 (Watchdog) | ⏳ DD workflow + 降级协议待建 |
| 可观测/审计 | alert tables + observability + briefing | 🟡 case lineage 待建 (Wave B/D) |
| 测试/版本/治理变更流程 | pytest + spec registry + drift hook | ✅ |
| PM 决策合同 | — | ⏳ PM Doctrine 待填 |
| 真钱前 shadow run | paper trade 进行中 | 🟡 持续到 2028-05 |

**结论**: 离企业级最近的 3 个动作 = (1) 本文 operating model ✅ (2) PM Doctrine (你填) (3) Wave B case table。做完这 3 个,"混乱感"基本消失,框架就立住了。

---

## 八、引用

- 四层 + RACI 推导: 本 session 2026-05-19 架构讨论
- doctrine 来源: [[feedback-llm-risk-side-not-alpha-side]] · [[project-agent-collaboration-patterns-2026-05-18]] (Pattern 5 ban) · [[feedback-spec-amendment-workflow-and-hash-discipline-2026-05-19]]
- agent 名册详情: [[project-persona-agent-architecture-2026-05-19]] + 各 agent spec
- cadence: scripts/run_paper_trade_daily.py + Watchdog cron
