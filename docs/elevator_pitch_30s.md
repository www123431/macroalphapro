# Elevator Pitch — 30 / 60 / 5-min versions

---

## 30-second (use in any first-touch)

> "我做了一个 production-grade applied AI-quant 系统 MVP——在 yfinance + FRED + CFTC COT 这种 free data 限制下，用 B++ Mass FDR 40-spec systematic search 筛出 1 个 shipped 策略（QL01 BAB / Frazzini-Pedersen 2014 / literature-conditional ship rule，BHY-aware）。架构 3 轴：production engineering（daily auto-run / Tier R audit / hash chain）+ methodological rigor（pre-registration + amendment ledger + HARKing detection）+ AI-era patterns（Tier R 3-layer 0-LLM-in-evaluation / Project History RAG / Reflexion-style memory / LLM-as-scientific-collaborator auto-spec drafting）。Forward NAV 跟踪从 2026-05-07 启动，每天累积。"

**Key tokens**（面试官一听就识别行家话）：
- B++ Mass FDR / BHY-aware
- literature-conditional ship rule
- 0-LLM-in-evaluation invariant
- amendment ledger / HARKing R1-R4
- Tier R 3-layer
- Reflexion / Sakana AI-style
- forward NAV tracking

---

## 60-second（recruiter / supervisor follow-up）

> "细节展开——这套系统每天 daily_batch 自动跑：信号层用 TSMOM 12-1 + BAB ranking，组合层用 vol-target + cap-enforced + regime overlay（MSM + VIX/VVIX/SKEW），风险层周期性 patrol 触发 PendingApproval 走 supervisor 审批。
>
> 治理上是 Tier R 3-layer：Layer 0 是 18 条 deterministic 规则（11 critical daily + 7 weekly slow-drift）；Layer 1 是 Gemini 2.5 Flash 用 response_schema 提案修复方案；Layer 2 是 11 条 V-rules（V0-V10）安全 gate。LLM 在 proposer 隔离层，evaluation 严格 0-LLM 红线（Zheng 2023 LLM-as-judge bias 学术依据）。
>
> 方法论上 pre-registration spec_hash + amendment ledger + HARKing R1-R4 detection，EFFECTIVE_N_TRIALS 跟着 amendment 动态变化；Backtest 走 NW HAC + Deflated Sharpe + BHY FDR + bootstrap CI。
>
> AI-era patterns 现状：Tier R 3-layer 0-LLM-in-eval 已运行；Reflexion-style memory infra 已就位（reflection 累积中）；Project History RAG（P2 in progress，~2 周）+ LLM auto-spec drafting（P3.5 planned）。
>
> Shipped 策略是 QL01 BAB，2026-05-05 ship 时 raw t=2.31 5% sig 但 BHY 不过——用 literature-conditional ship rule 豁免（FP 2014 cited 5000+ 给独立证据），决策走完整 spec amendment 流程留底。Forward record 从 2026-05-07 起。"

---

## 5-min（30-min interview opening）

按 **identity → architecture → 1 strategy → live evidence** 4 段讲：

### 段 1：Identity (60s)
- "production-grade applied AI-quant system MVP"
- 不是 fund / lab / startup — 是 applied AI-quant engineering
- target: senior tech / quant fund recruiter + MSBA committee

### 段 2：3-axis architecture (120s)
- **production engineering**: daily auto-run / Tier R / hash chain / 22 pages / 109 pytest
- **methodological rigor**: pre-reg + amendment ledger + HARKing + BHY FDR
- **AI-era**: Tier R 3-layer / RAG / Reflexion / auto-spec
- 现场 demo: Brief page → POSITIONS → audit panel → spec inspection

### 段 3：1 shipped strategy (60s)
- QL01 BAB (FP 2014)
- B++ Mass FDR 40-spec search → BAB raw 5% sig
- BHY fail → literature-conditional ship rule
- 决策完整 spec amendment 流程

### 段 4：Live evidence + roadmap (60s)
- forward NAV from 2026-05-07
- 7-test falsification chain
- 8-phase roadmap to production-readiness (see `docs/decisions/path_d_roadmap_2026-05-07.md`)
- 数据限制下未来扩展：CFTC COT / VVIX/SKEW / RAG

---

## 8 architectural distinctiveness × 30-second talk

每条单挑出来都能聊 5-10 分钟。背熟的 2 句话版本：

### 1. Tier R 3-layer Auto-Audit
> "我把 LLM 隔离在 proposer 层做 Layer 1，Layer 0 是 18 条 deterministic 规则（11 critical + 7 weekly），Layer 2 是 11 条 V-rules（V0-V10）安全 gate。LLM 升级直接吃 Layer 1，红线 Layer 0/2 不动——这是 production AI safety 模式，跟 Anthropic Constitutional AI (Bai 2022) 思路一致。"

### 2. 0-LLM-in-evaluation invariant
> "evaluation 层我严格不让 LLM 进——基于 Zheng 2023 LLM-as-judge bias 实证。LLM 只在 generation / proposal 层活动；scoring / verdict 走 deterministic 路径。这是 trustworthy AI 的硬要求。"

### 3. spec_hash + amendment ledger
> "我把 pre-registration 嵌进系统而不是依赖 OSF / AsPredicted 外部 commit。每个 spec 文件 SHA-256 hash 进 SpecRegistry，所有 amendment 走 amendment_ledger 留底，HARKing R1-R4（silent edit / threshold drift / unannounced trial / predictions rewrite）4 类自动检测。"

### 4. EFFECTIVE_N_TRIALS dynamic
> "BHY FDR 多重检验 penalty 不是静态——我让 EFFECTIVE_N_TRIALS 跟着 amendment ledger 动态算。注册新 spec / amendment 自动更新，防止 multiple-testing 被 silent dilution。"

### 5. Hash chain across DecisionLog + PA
> "DecisionLog 和 PendingApproval 的 chain_hash 字段 SHA-256 链接前一行，参考 SEC 17a-4 tamper-evidence 标准。任何 row 被改 chain 立刻断，audit 重建可证明完整性。"

### 6. Project History RAG (P2)
> "Project History RAG 用 chromadb local + sentence-transformers，index 整个项目的 decision_log + spec amendments + audit findings + macro briefs。Supervisor 自然语言查询「为什么 ship BAB」直接 retrieve 相关 spec + 时点 audit + ship 论据，可选 LLM synthesis 给整段答案。"

### 7. LLM Auto-Spec Drafting (P3.5)
> "Auto-Spec Drafting 是 Sakana AI Scientist 风格——supervisor 自然语言描述假设，LLM 起草完整 pre-reg spec（hypothesis / decision rule / N_TRIALS impact / risk profile），supervisor review + freeze hash。LLM 当 scientific collaborator 不当 decision maker。"

### 8. Era flag legacy data isolation
> "今天系统 cap-fix restart 后，老 buggy 期数据没删，加了 era 字段标 'pre_cap_fix_legacy'，新数据 'live'，UI 默认 filter live 但 supervisor 可 toggle 看 legacy。这样 hash chain 不断 / forensics evidence 全在 / supervisor 视图干净——同时满足 audit + UX 要求。"

---

## 面试官常见追问 + 答案

### Q: "Sharpe 0.985 raw 但 BHY fail 你为什么 ship？"
> "literature-conditional ship rule。BAB 的 academic anchor 是 Frazzini-Pedersen 2014 RFS，cited 5000+，full FF/Carhart 跨 30 年跨多市场重复验证。这种独立外部 evidence 让我 grant BHY exemption，但**整套决策走 spec amendment 留底**——决策可被 challenge / reverse / audit。"

### Q: "你的 BAB 跟 vanilla FP 2014 有什么区别？"
> "本质是同一个 ranking——但加了 (a) regime-conditional position cap (Ang-Bekaert 2004 启发)，(b) MSM 概率加权 vol target，(c) 15bp transaction cost penalty 真建模（P1 deliverable，旧 13bp 是 placeholder）。"

### Q: "你怎么避免 overfitting？"
> "三层防御：(1) pre-registration enforce — spec_hash + amendment ledger 防 silent edit，(2) HARKing R1-R4 自动检测 (3) BHY FDR + Deflated Sharpe + Newey-West HAC + bootstrap CI 多重检验 penalty。完整文档在 `docs/falsification_chain.md`——7 个 hypothesis test，6 reject + 1 marginal。"

### Q: "为什么不 paper trade 真钱？"
> "MSBA capstone 时间 + 资源限制，目前 simulated $1m，monthly rebal。Forward NAV tracking 从 2026-05-07 启动，每天累积 1 行。3-6 月后 forward record 累积量足够时可 lift to 真钱（要走 SEC compliance 才合法）。"

### Q: "你的 LLM 对策略真的没影响吗？"
> "是。production strategy 走 pure quant path：signal → construct_portfolio → trade。LLM 只在 (a) sector_pipeline debate（决策 audit log，不影响仓位）/ (b) Tier R proposer / (c) macro_brief_llm（supervisor narrative）/ (d) auto-spec drafting（research aid）。narrative_overlay 2026-05-03 实证 reject 后已物理删除——LLM 进决策的桥已被切断，0-LLM-in-evaluation 红线维持。"

### Q: "为什么用 yfinance 不是真 data feed？"
> "数据限制下的 deliberate choice——MSBA scale 不可能买 Bloomberg / Refinitiv。配 free data (yfinance / FRED / CFTC COT 业界 free 但 MSBA 少用)，**focus 在 methodology + architecture 而不是 data moat**。这反而让方法论严谨度可被 reviewer 直接复现。"

### Q: "Tier R Auto-Audit 真在跑吗？"
> "现场可 demo——`python scripts/run_auto_audit.py --scope critical` 半秒出 11 rules / 0 findings 的 JSON，hash chain INTACT。运行历史在 AuditRun 表里。"

---

## Recruiter-specific framing

### tech recruiter (e.g., Two Sigma engineering / FB ML infra)
- Lead with: production engineering + Tier R + hash chain + 109 pytest
- "I built a production-grade system with proper monitoring, audit, recovery"

### quant fund recruiter (Citadel / AQR / Bridgewater)
- Lead with: B++ Mass FDR + literature-conditional ship + falsification chain
- "I built a research lab with rigorous shipping discipline"

### MSBA committee
- Lead with: applied focus + 落地性 + 8-phase roadmap to production-readiness
- "我做的是可部署的工程系统不是论文研究"

---

## "Don't say" list

要避免的话：
- ❌ "research lab" / "academic project" — 已 reframe to applied
- ❌ "I made money" / "alpha generator" — 没 ship alpha 不假装有
- ❌ "agentic AI capability demonstration" — 太抽象
- ❌ "novel methodology" — 不 claim 学术原创
- ✅ "production-grade engineering MVP" — 准确
- ✅ "applied under free-data constraints" — 诚实
- ✅ "1 shipped strategy with literature support" — 具体
- ✅ "rigorous shipping discipline" — 不是吹牛是事实
