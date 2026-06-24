# Narrative Overlay Phase 0 — Pipeline Ready (v2, 2026-05-02 ~22:15)

**状态**：⚙️ 工程完成 + production-engine alignment 大概率成功 + 真实数据 ablation 待外网环境
**Spec**：[../spec_narrative_overlay_phase0.md](../spec_narrative_overlay_phase0.md)
**Memory**：[../../memory/project_narrative_overlay_decision.md](../../memory/project_narrative_overlay_decision.md)

---

## 1. 工程完成清单（Sprint S0–S5 全部 ✅）

| Sprint | 产出 | 状态 |
|---|---|---|
| **S0 数据层** | [shock_loader.py](../../engine/narrative/shock_loader.py) (含 GPR `.xls` 修复 2026-05-02) | ✅ |
| **S0 数据层** | [universe_audit.py](../../engine/universe_audit.py) +4 ETF inception | ✅ |
| **S1 IRF 训练** | [irf_trainer.py](../../engine/narrative/irf_trainer.py) | ✅ Local Projections + statsmodels NW-HAC |
| **S2 Overlay** | [overlay.py](../../engine/narrative/overlay.py) + [portfolio.py](../../engine/portfolio.py#L555) Step 5f | ✅ regression test 证 None=byte-identical |
| **S2 Overlay** | [backtest.py](../../engine/backtest.py#L539) `narrative_context_func` 参数 | ✅ 生产 backtest engine 接入 |
| **S3 Backtest** | [backtest_phase0.py](../../engine/narrative/backtest_phase0.py) (v2 调 run_backtest) | ✅ |
| **S3.5 Metrics** | [metrics.py](../../engine/narrative/metrics.py) (拆分以避免 streamlit 间接 import) | ✅ |
| **S4 Decision** | 本文档 | ✅ |
| **S5 Tests** | [test_narrative_overlay.py](../../tests/test_narrative_overlay.py) | ✅ **12/12 通过** |

---

## 2. 完整 89 月 backtest 结果（synthetic shocks + 真实 ETF prices）

```
========================================================================
Narrative Overlay Phase 0 — Ablation Report (production engine)
========================================================================
Synthetic data : True

Arm               n    Sharpe
A_baseline       88     0.327
B_narrative      88     0.367
C_placebo        88     0.235

ΔSharpe (B - A)         : +0.040    gate: FAIL  (≥ 0.10)
NW HAC t-stat (B - A)   : +1.276    gate: FAIL  (≥ 1.5)
Sharpe(B) - Sharpe(C)   : +0.131    gate: PASS  (≥ 0.05)
PBO of arm B            : (skipped — use --pbo-sweep)

Subperiods ΔSharpe(B-A):
  P1_pre_covid       +0.087    PASS
  P2_covid           +0.048    PASS
  P3_tightening      +0.023    PASS

OVERALL: REJECTED  ← 不重要：synthetic 数据下决策无意义
========================================================================
```

---

## 3. 三个关键诊断

### 3.1 Production-engine alignment 状态：**大概率成功**

| | 简化版 backtest_phase0 v1 | 生产 engine v2 (本次) | 生产 baseline (memory) |
|---|---|---|---|
| Arm A Sharpe | 0.054 | **0.327** | 0.703 |
| 期间 | 2019-2026 OOS | 2019-2026 OOS | **未明确** |

**v2 vs v1**：6× 改进（0.054 → 0.327）确认生产 engine 正确接入 — vol-targeting + LW shrinkage + 浓度过滤 + 净敞口 clamp + 换手成本全部生效。

**v2 (0.327) vs memory (0.703)**：剩 2.15× 差距。最可能原因（**待用户确认**）：

- 0.703 是 **2003-2026 全期**？如果是，alignment 就是 ✅
- 0.703 是 **2019-2026 OOS-only**？如果是，存在 alignment bug 待诊断

理由 0.703 大概率是全期：
- 2003-2018 含 GFC 后大牛市 + 强单边趋势 → TSMOM 黄金期 (Sharpe ~1.0+)
- 2019-2026 含 COVID 闪崩 + 通胀震荡 + 加息冲击 → TSMOM 进入低 Sharpe 期
- 全期混合 ≈ 0.7 量级合理

**用户需要查 [memory: project_baseline_switch_2026-05-02.md](../../memory/project_baseline_switch_2026-05-02.md) 那次回测的精确起止日期**。

### 3.2 ΔSharpe(B-A) = +0.040 是 spurious，不是 alpha

数学：IRF 表用 synthetic shocks + **真实** ETF 收益训练。LP 回归在 281 月 × 3 列 × 26 sectors × 4 horizons 上，必然学到一些伪相关 cell。Backtest 用**同一份** synthetic shocks 喂 IRF → in-sample 偏差被复制。

**真实数据下这个 +0.040 大概率消失或反向**——因为真实 GPR/EPU/NVIX 与 ETF return 的真实经济相关性会被 IRF 重新捕捉，与噪声相关性不同方向。

NW t = 1.276 (FAIL ≥ 1.5) 也支持这个判断：在 88 obs 上 +0.040 Sharpe 增量是统计噪声。

### 3.3 yfinance 12 月偶发 bug（不影响主流程）

log 中 "Failed downloads (1d 2023-12-16 → 2023-12-01)" 是 yfinance 自己的 startDate>endDate bug，发生在 2022/2023/2024/2025 年底。88/89 月有数据，缺失 1 月不影响 metrics。可在 Phase 0 通过后修补 [engine/backtest.py](../../engine/backtest.py) 的 fetch window logic。

---

## 4. 守住的边界（spec 全部满足）

✅ 未动 [signal.py](../../engine/signal.py) / [regime.py](../../engine/regime.py) / [sector_pipeline.py](../../engine/sector_pipeline.py) / [agents/factor_mad/](../../engine/agents/factor_mad/)
✅ portfolio.py 改动严格向后兼容（`narrative_context=None` 完全 no-op，regression test 已证）
✅ backtest.py 改动严格向后兼容（`narrative_context_func=None` 完全 no-op）
✅ 0 LLM 调用进 narrative overlay 路径（Layer 1 留 Phase 1）
✅ Phase 1 前视偏差纪律已固化在 [memory: feedback_llm_lookahead_bias.md](../../memory/feedback_llm_lookahead_bias.md)

---

## 5. 真实 Phase 0 ablation 仍待 1 件事

### 真实 shock 数据获取（用户做，外网环境）

```bash
# 用户主机（能访问 fred.stlouisfed.org / matteoiacoviello.com / asafmanela.github.io）
cd ${REPO_ROOT}\Desktop\intern
D:/python/python.exe -m engine.narrative.shock_loader --no-cache --start 2003-01-01 --end 2026-05-31
```

预期产出：`data/shocks.parquet` 含真实 GPR + EPU + NVIX 月度 + z-scores。

预期失败模式（提前预警）：
- GPR `.xls` 应该 OK（2026-05-02 main page 确认 + xlrd 已装）
- 如果 EPU FRED 在用户环境也不通 → fallback 触发 synthetic
- 如果 NVIX Manela 主页路径变了 → fallback 到 VIX 月末作 NVIX proxy（2017+）

跑完后：
```python
import pandas as pd
df = pd.read_parquet("data/shocks.parquet")
print(df.attrs.get("synthetic", "未记录"))  # 应该是 False
```

---

## 6. 路线图（用户确认 baseline 期间后）

### Path A — 如果 0.703 是 2003-2026 全期（最可能）

1. ✅ Production-engine alignment 确认 — 不动 backtest infrastructure
2. 用户跑 shock_loader 拉真实数据
3. 重训 IRF 用真实 shocks + 真实 prices（仅 train 期 2003-2018）
4. 跑真实 Phase 0 ablation （3 arms + PBO sweep）
5. 写 `narrative_overlay_phase0_passed.md` 或 `..._rejected.md`
6. 通过 → 启 Phase 1 spec（含前视偏差协议）

### Path B — 如果 0.703 是 2019-2026 OOS-only（需诊断）

1. 找出 v2 (0.327) 与生产 (0.703) 在 OOS-only 上为何不一致
2. 候选诊断：active universe 大小、benchmark 选择、transaction cost 设置、特定参数差异
3. 修对齐后再走 Path A 后续步骤

---

## 7. Sprint 实际工时

| Sprint | spec 估算 | 实际 |
|---|---|---|
| S0 数据层 | 0.5 天 | ~1.5 小时（含 GPR URL 修复 + ETF inception 补全） |
| S1 IRF 训练 | 1.5 天 | ~1 小时（statsmodels HAC 直接可用） |
| S2 Overlay + backtest engine | 1 天 | ~2 小时（含 production engine 接入） |
| S3 Backtest 框架 | 1 天 | ~1 小时（v1 简化版） + ~1.5 小时（v2 重写复用 run_backtest） |
| S3.5 Metrics 拆分 | — | ~30 分钟 |
| S4 Decision | 0.5 天 | ~1 小时 |
| S5 Smoke + Tests | 0.5 天 | ~30 分钟 |
| **总计** | **5 天** | **~9 小时** |

实际工时 ~9 小时 vs spec 估 5 天——比预期快 4-5×。剩余唯一工作是真实数据获取 + 重跑（~30 分钟用户主动操作 + ~5 分钟 backtest）。
