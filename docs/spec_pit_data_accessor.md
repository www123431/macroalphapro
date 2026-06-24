# Tier C L2-1: PIT Data Accessor 架构 spec (locked 2026-06-08)

**状态**: 设计锁定，准备实施
**总工时**: 30h（拆 5 个 phase）
**依赖**: L2-3 / L2-8 / L2-2 已 ship 且 E2E 测试绿（commit 05758be9）
**解锁**: 一次性消灭 B0/B1/B2/B5/B7 silent bugs；未来 template 自动继承 PIT 正确性

---

## 1. 问题陈述

当前 Tier C templates 实现"templates 直接读 parquet + 每 signal 自己处理滞后"：

```python
# 当前 cross_sec_us_equities.py 风格
msf = _load_crsp_msf()
funda = _load_compustat_funda()
mktcap_lagged = mktcap_panel.shift(1)   # ← 硬编码滞后
funda_with_lag = funda.public_date >= start - 120d   # ← 硬编码滞后
```

**fragility 验证**（已识别 silent bugs，按严重度）：

| Bug | 类别 | 当前状态 |
|---|---|---|
| **B0** | comp.funda 是 latest-restated 不是 PIT | **整个 fundamental signal 都污染** |
| B1 | Universe selection 用同期 mktcap | 已点状修（commit 328154dc，hardcoded shift） |
| B2 | 没有 survivorship-aware universe | 未修 |
| B5 | n_trials 是 family 级，应 mechanism-class 级 | 未修 |
| B7 | ROE 用年度 ni/ceq，HXZ 用季度 ROE-q | 未修（结构性低估） |

**架构层面问题**：每加新 template（C-2f carry, vrp, event_drift...）= 重新审一遍 PIT，10 个 template = 60+ 滞后判断点 = **必然漏掉一个**。

---

## 2. 设计目标

**用 type system + data structure 强制 PIT 正确性**，不靠人审。具体：

1. **数据层缓存 PIT-clean by construction** — pull `comp_pit.pithistdataus` 替代 `comp.funda`
2. **Simulation clock** 推进 — data access 按 clock_time 过滤
3. **Template contract** 字段显式声明 PIT 假设
4. **Audit gate** — 未审计 template 不能 dispatch

新增 template 时**只需写信号逻辑**，PIT 不可能写错（因为底层 access 拒绝）。

---

## 3. 4 层架构详细设计

```
┌─────────────────────────────────────────────────────────────┐
│  L1 PIT Data Warehouse                                       │
│   data/cache/_pit_*.parquet                                  │
│   每张表 PIT-clean by construction                            │
│   - _crsp_msf_pit.parquet (existing _crsp_msf_long_history)  │
│   - _compustat_funda_pit.parquet (NEW: 从 comp_pit 拉)       │
│   - _crsp_ccm_link.parquet (existing, 已 PIT 因 linkdt 字段) │
│   - _ff_factors_pit.parquet (NEW: Ken French monthly)        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  L2 Simulation Clock                                          │
│   engine/data/pit_warehouse/simulation_clock.py              │
│   class SimClock:                                             │
│     - now: pd.Timestamp                                       │
│     - advance(dt) → 新 clock                                  │
│     - knows_about(as_of: pd.Timestamp) → bool                 │
│   每次回测有自己的 clock                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  L3 PIT Data Accessor (data-layer × clock-layer 接合)        │
│   engine/data/pit_warehouse/accessor.py                      │
│   class PITDataAccessor:                                      │
│     def __init__(self, clock: SimClock)                      │
│     def mktcap_panel(window: tuple) → DataFrame              │
│     def returns_panel(window: tuple) → DataFrame             │
│     def funda(field: str, gvkey: str) → ScalarOrSeries       │
│     def funda_panel(field: str, window: tuple) → DataFrame   │
│   返回的所有数据自动按 clock 过滤                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  L4 Template Contract + Audit Gate                            │
│   engine/agents/strengthener/templates/_template_contract.py │
│   @dataclass class TemplateContract:                         │
│     name: str                                                 │
│     version: str                                              │
│     pit_audit_certified_by: str  # 人审签名                   │
│     pit_audit_date: str                                       │
│     supported_signal_kinds: tuple                            │
│     supported_universes: tuple                               │
│     supported_rebal_freqs: tuple                             │
│     declared_data_sources: tuple                             │
│   Dispatcher 加 audit gate：未 certified 拒绝 dispatch        │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. API 详细规格

### 4.1 SimulationClock

```python
class SimClock:
    """Backtest simulation clock. Each dispatch instantiates ONE
    clock; data access is filtered by clock.now."""

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self._start = start
        self._end = end
        self._now: pd.Timestamp = start

    @property
    def now(self) -> pd.Timestamp:
        return self._now

    def advance(self, dt: pd.Timedelta | str) -> "SimClock":
        """Advance clock; returns self for chaining."""
        ...

    def knows_about(self, as_of: pd.Timestamp) -> bool:
        """True iff as_of <= self._now. Used by data layer to
        filter PIT-restricted reads."""
        return as_of <= self._now

    def iter_rebal_dates(self, freq: str = "ME") -> Iterator[pd.Timestamp]:
        """Iterate rebal dates from start to end at given freq."""
        ...
```

### 4.2 PITDataAccessor

```python
class PITDataAccessor:
    """Sole interface between templates and cached data. ALL data
    access goes through here; templates cannot read parquet directly."""

    def __init__(self, clock: SimClock):
        self._clock = clock

    # ── CRSP price layer ───────────────────────────────────────
    def mktcap_panel(self, lagged: bool = True) -> pd.DataFrame:
        """Returns mktcap matrix. `lagged=True` returns prior-month
        values (PIT-correct for universe selection at month-end t).
        `lagged=False` returns current month (for ex-post return
        computation, where it's PIT-correct)."""
        ...

    def returns_panel(self) -> pd.DataFrame:
        """Returns matrix indexed by month_end. Auto-filtered to
        clock.now."""
        ...

    def delisting_returns(self) -> pd.DataFrame:
        """Delisting returns by (permno, month_end). PIT-clean by
        construction since dlret known on dlst_dt."""
        ...

    # ── Compustat fundamental layer ────────────────────────────
    def funda_pit(
        self,
        gvkeys: list[str] | None = None,
        fields: list[str] = None,
        as_of: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Returns Compustat funda values FIRST REPORTED (PIT).
        For each (gvkey, fiscal_year), returns the value as
        reported in the first earnings release covering that year
        — NOT the latest restated. Filtered to as_of <= clock.now.

        If as_of given, returns the snapshot known by as_of.
        Otherwise returns the snapshot known by clock.now.

        Implements B0 fix: latest-restated → PIT.
        """
        ...

    def funda_panel(
        self,
        field: str,
        window: tuple[pd.Timestamp, pd.Timestamp],
    ) -> pd.DataFrame:
        """Returns (month_end, permno) panel of a Compustat field,
        with CCM link applied + 120d public-availability lag baked in."""
        ...

    # ── Universe construction ──────────────────────────────────
    def universe_top_n_by_mktcap(
        self,
        n: int,
        as_of: pd.Timestamp,
    ) -> set[int]:
        """Returns set of permnos in top N by mktcap as of clock.now.
        Uses lagged mktcap automatically (PIT-correct).

        Implements B1 fix: same-month look-ahead.
        """
        ...

    def universe_sp500_constituents(
        self,
        as_of: pd.Timestamp,
    ) -> set[int]:
        """Returns set of permnos in S&P 500 as of as_of. Uses PIT
        constituents table (NEW data pull required).

        Implements B2 fix: survivorship-aware universe.
        """
        ...
```

### 4.3 TemplateContract

```python
@dataclass(frozen=True)
class TemplateContract:
    """Declarative contract for each template. Dispatcher refuses
    to call templates without a valid contract."""

    template_name: str
    template_version: str

    # PIT certification — human-reviewed
    pit_audit_certified_by: str    # "claude-2026-06-08" or "user-zxz"
    pit_audit_date: str            # "2026-06-08"
    pit_audit_notes: str           # ≤ 800 chars

    # Scope
    supported_signal_kinds: tuple[str, ...]
    supported_universes:    tuple[str, ...]
    supported_rebal_freqs:  tuple[str, ...]
    supported_signals:      tuple[str, ...]  # canonical signal_key
                                              # (e.g. "gp_at", "vol_12m")

    # Data sources declared (for accessor whitelist enforcement)
    declared_data_sources: tuple[str, ...]
    # e.g. ("crsp.msf", "crsp.dsedelist", "compustat.pit.funda")

    # Bibliography (for replication mode auto-population)
    canonical_paper_id: str | None = None
    canonical_paper_window: str | None = None     # "YYYY-MM:YYYY-MM"
    canonical_paper_t: float | None = None
```

### 4.4 Dispatcher audit gate

```python
# In dispatch_factor_spec:
def pre_dispatch_check(spec, ...):
    ...
    # NEW gate #9: template audit certification
    contract = get_template_contract(spec.signal_kind, spec.universe)
    if contract is None:
        return DispatchRefusal(
            reason_code="TEMPLATE_NOT_CERTIFIED",
            detail=f"no certified template for signal_kind="
                   f"{spec.signal_kind} + universe={spec.universe}",
            metrics={},
        )
    # Check PIT certification freshness (re-cert required every 90 days
    # OR after refactor)
    cert_age_days = (pd.Timestamp.now() -
                     pd.Timestamp(contract.pit_audit_date)).days
    if cert_age_days > 90:
        return DispatchRefusal(
            reason_code="TEMPLATE_CERT_STALE",
            detail=f"template {contract.template_name} cert is "
                   f"{cert_age_days}d old (>90d). Re-audit required.",
            metrics={"cert_age_days": cert_age_days},
        )
```

---

## 5. PIT 滞后字典（loadbearing reference）

每个（数据源, 字段）的 PIT 公开时刻：

| Source / Field | 公开时刻 | 用于 universe select 需滞后 |
|---|---|---|
| **CRSP** | | |
| `crsp.msf.prc` | T 收盘 | shift(1) 月 |
| `crsp.msf.ret` | T 收盘后 | 当 t+1 月的"forward return" 不滞后 |
| `crsp.msf.mktcap` | T 收盘 | shift(1) 月 |
| `crsp.msf.shrout` | T 收盘 | shift(1) 月 |
| `crsp.dsedelist.dlret` | dlstdt | 当月内可用 |
| **Compustat PIT** | | |
| `comp_pit.pithistdataus.atqh (qtrsback=earliest>=0)` | 季报公开日（rdq） | 自动通过 first-report 选择 |
| `comp_pit.pithistdataus.niqh (qtrsback=earliest>=0)` | 季报公开日 | 同上 |
| **CCM Link** | | |
| `crsp.ccm_link.linkdt` | linkdt（永久） | by construction |
| **Fama-French** | | |
| `ff.factors_monthly.mkt_rf` | T+1 月 5 日（French website） | 默认滞后 5 天 |
| **OptionMetrics** | | |
| `optionmetrics.standardized_options` | T 收盘 | shift(1) 日 |
| **FRED Macro** | | |
| `fred.vix` | T 收盘 | 当日可用 |
| `fred.gs10` | T 收盘 | 当日可用 |

---

## 6. 实施阶段（5 phase, 30h）

### Phase 1: 数据层 L1（10h）
- **1.1** (1h) Spec doc 完成（本 doc）
- **1.2** (3h) 写 `scripts/extend_compustat_funda_pit_history.py`：
  - SQL: `SELECT * FROM comp_pit.pithistdataus`
  - 筛选：`qtrsback = MIN(qtrsback) per (gvkey, datadate)` 取**首报值**
  - 输出 `data/cache/_compustat_funda_pit.parquet`
  - 预估 5-10M 行 × 主要字段 ~50MB
- **1.3** (3h) 跑 WRDS pull（后台 ~15min）+ validation script
- **1.4** (2h) 拉 PIT S&P 500 constituents (`crsp.dsp500list`) → `_sp500_constituents_pit.parquet`
- **1.5** (1h) Smoke compare：comp.funda 原始 vs comp.funda_pit 在 1995 年某些公司 `at` 值的偏差实测

### Phase 2: SimulationClock + Accessor（8h）
- **2.1** (2h) 写 `engine/data/pit_warehouse/__init__.py` + `simulation_clock.py`
- **2.2** (4h) 写 `engine/data/pit_warehouse/accessor.py` 全部 API（不集成 template）
- **2.3** (2h) Unit tests on synthetic data — accessor 严格按 clock 过滤验证

### Phase 3: 重构 cross_sec template（6h）
- **3.1** (3h) Refactor `cross_sec_us_equities.py` 全部数据 access 走 accessor
- **3.2** (1h) 加 `TemplateContract` 实例
- **3.3** (2h) 跑 E2E test（commit 05758be9 那个）+ 对照 pre-refactor verdict 数字
  - **关键**：GP/A naive t 必须从 3.34 → 显著下降（因为 PIT 数据 + survivorship 修复）
  - 预测：drop 到 ~2.5-2.8 范围（更接近 Novy-Marx 原文 t≈3.0 paper window 表现）

### Phase 4: 重构 tsmom template（3h）
- **4.1** (2h) Refactor `tsmom_sector_etf.py` 走 accessor
- **4.2** (1h) E2E test + pre-refactor verdict 对照

### Phase 5: Audit gate + 文档（3h）
- **5.1** (1h) Dispatcher 加 gate #9（audit cert check）
- **5.2** (1h) 写 dispatcher tests
- **5.3** (1h) 更新 CLAUDE.md 项目根 doctrine：
  - "新 template 必须有 TemplateContract"
  - "新 template 必须 pit_audit_certified_by 字段"
  - "未认证 template dispatch 报 TEMPLATE_NOT_CERTIFIED"

---

## 7. 重构期 parity 测试策略

**核心担忧**：30h 重构改了 data access 层，但 cross_sec verdict 数字不能"突然"变化——必须对照 pre-refactor 数字 + 解释每个变化的因果。

具体做法：

### 7.1 Pre-refactor baseline 锁定
跑 E2E test（commit 05758be9）记录基准数字：
```json
{
  "GP/A_1992-2024": {
    "verdict": "GREEN",
    "sharpe": 0.6391,
    "nw_t_stat": 3.3421,
    "replication_overlap_t": 2.665,
    "replication_t_gap": 0.335,
    "cost_stress_80bp_t": 2.72,
    "drawdown_naive_pct": -20.54
  },
  "Momentum_1992-2024": {
    "verdict": "RED",
    "sharpe": 0.27,
    "nw_t_stat": 1.43,
    "cost_stress_60bp_sharpe": -0.004,
    "drawdown_naive_pct": -75.45
  }
}
```

保存为 `tests/_baseline_l2_pre_pit_rebuild.json`，commit。

### 7.2 Post-refactor 对照
重构完每个 phase 都跑同一 E2E 然后 diff：

| Phase | 期待 verdict 变化 | 原因 |
|---|---|---|
| Phase 1 完（PIT data 拉好但 template 未改） | 无变化 | template 还在用旧 cache |
| Phase 3 完（cross_sec 改完） | **GP/A naive t: 3.34 → 2.5-2.8** | PIT comp.funda + survivorship 双重修复 |
| Phase 4 完（tsmom 改完） | TSMOM 数字基本不变 | TSMOM 只用 CRSP 价格，PIT 影响小 |
| Phase 5 完 | 无变化 | 只加 gate 不改 backtest |

**如果 Phase 3 后 t 变化不在预测区间** → 重构出 bug，必须 debug 到原因再 ship。

### 7.3 第三方 sanity check
用 Kenneth French 公开 data（mom + value factors）跑同样的 1992-2024 backtest（**通过 PIT accessor**），对照 French website 公开 monthly Sharpe / t-stat。差异 < 0.5 → accessor 实施正确。

---

## 8. 当前数据资产与新增数据需求

### 已有
- `_crsp_msf_long_history.parquet` (2M 行, 1990-2024) ✓ PIT-clean，重命名为 `_crsp_msf_pit.parquet`
- `_crsp_dsedelist.parquet` ✓ PIT-clean
- `_crsp_ccm_link.parquet` ✓ PIT-clean
- `_compustat_funda_long_history.parquet` (572K 行, 1962-2024) ✗ **latest-restated 污染** — Phase 1 替换

### 新拉
1. **`_compustat_funda_pit.parquet`** — 从 `comp_pit.pithistdataus` 取 first-report 值
2. **`_sp500_constituents_pit.parquet`** — 从 `crsp.dsp500list` 取 PIT constituents（survivor-bias-free universe）
3. **`_ff_factors_pit.parquet`** — Kenneth French 5-factor monthly（用于 future anchor orthogonality + parity test）

---

## 9. 风险 + 缓解

| 风险 | 缓解 |
|---|---|
| Phase 3 重构破坏 E2E test 但原因不明 | parity baseline + phase 间 diff，定位精确 |
| comp_pit 表 schema 跟我假设的不同 | Phase 1.2 第一件事就是探查 schema（已部分 probe） |
| accessor 实现引入性能 regression（cross_sec 现在 ~30s → 3min） | Phase 2.2 加 lru_cache + 提前 benchmark；如果 regression > 2x 则改用 lazy view |
| PIT comp.funda 比 latest-restated 缺失大量行（数据不完整） | Phase 1.5 smoke compare 时检测；如果 > 20% 缺失则 fallback 现有 cache + 文档警告 |
| 新增 audit gate 破坏 L1 现有 commits 的回归测试 | Phase 5 时给现有 cross_sec + tsmom templates 创建 grandfathered TemplateContract（pit_audit_certified_by = "grandfathered_2026-06-08"）|

---

## 10. 何时停 + 失败模式

### 提前停的触发条件
1. **Phase 1 后** comp_pit data 实测发现 > 30% 关键字段缺失 → 改方案，用 comp.funda 但应用估算 lag 滞后字段（不如 PIT 好但比当前好）
2. **Phase 3 后** GP/A t 不降反升 → 实施有 bug，不能 ship Phase 4+
3. **Phase 5 后** dispatcher gate 破坏 L1 cross_sec + tsmom 现有路径 → 加 grandfather logic

### 永远不做的
- ❌ 把 latest-restated comp.funda 当 PIT 用（B0 永久 ban）
- ❌ 让 template 直接读 parquet 绕过 accessor（架构核心）
- ❌ template 自己声明 PIT certification（必须人审）

---

## 11. 衔接其它 L2 项目

| 项目 | 衔接方式 |
|---|---|
| L2-2 Replication Mode（已 ship） | 重构后继续 work；accessor 内部用同一 PIT 数据，replication subsample 数字会更精确 |
| L2-3 Multi-Cost Stress（已 ship） | 重构后继续 work；不需要改 |
| L2-8 Drawdown Metrics（已 ship） | 重构后继续 work；不需要改 |
| L2-4 Anchor Orthogonality（未 ship） | **L2-1 必须先完成** — accessor 提供 anchor library factor returns 的接口 |
| L2-5 Subsample Analysis（未 ship） | 设计层依赖 SimClock 概念（rolling window） |
| L2-6 Attribution（未 ship） | 设计层依赖 accessor universe + sector mapping |
| L2-7 Multi-Dim Dashboard（未 ship） | UI 层；可以在 L2-1 之前或之后做 |

---

## 12. 下个 session 起手 checklist

进入 next session 第一件事：

1. **读本 doc 第 2-3 节**复盘 4 层架构 + API
2. **跑 E2E baseline** lock `tests/_baseline_l2_pre_pit_rebuild.json`
3. **开始 Phase 1.2** 写 comp_pit pull 脚本
4. 后台跑 WRDS 拉取，**同时**继续 Phase 1 其它子任务
5. **每个 phase 完成立即跑 parity diff** — 不能让 silent 数字漂移

---

**Spec locked 2026-06-08 by L2-1 design pass.**
**WRDS comp_pit schema probed and confirmed accessible.**
**Pre-refactor L2 E2E baseline available via commit 05758be9 test.**
**Estimated total work: 30h over 3-5 sessions.**

---

## APPENDIX A — Expanded scope (2026-06-08 senior re-critique)

After Phase 2.2 (accessor) ship, user-driven re-audit identified two
deeper hardcoding issues beyond the original PIT focus. **Spec
extended from 30h → 40h** to address. These additions are NOT
optional — they are the architectural prerequisite for Layer 3
(LLM critic loop / variant generation / self-doubt) to actually
function. Without them, Layer 3 LLM "reasoning" becomes theatrical
because the system cannot ACT on the LLM's suggestions.

### A.1 Two-layer hardcoding errors found

**Error 1 — application-layer PIT** (already covered above):
  `_FUNDA_PUBLIC_LAG_DAYS = 120` in accessor.py centralizes lag
  rules in application code rather than in data files. Industry
  standard (AQR / Two Sigma / Citadel) uses BITEMPORAL DATA
  MODELING: every data row carries its own `knowable_at` column
  computed at data-engineering time. Application layer just
  filters `knowable_at <= clock.now`.

**Error 2 — parameter-layer hardcoding** (NEW):
  Across cross_sec + tsmom templates, ~15 design-choice values
  are hardcoded at module-constant level:
    _UNIVERSE_TOP_N = 3000          (universe size)
    _N_QUINTILES = 5                (bucket count)
    _VOL_LOOKBACK_M = 12            (signal vol window)
    _MOM_12_LOOKBACK = 12           (momentum window)
    _MOM_6_LOOKBACK = 6
    _LOOKBACK_WEEKS = 52 (TSMOM)    (TSMOM specific)
    _SKIP_WEEKS = 4 (TSMOM)
    _VOL_TARGET = 0.10 (TSMOM)
    ... etc

  Result: LLM extractor's design space is ~0.1% of the actual
  research universe. LLM cannot propose:
    - "GP/A on top-500 with DECILE L/S"  (universe_size + n_buckets)
    - "TSMOM(6,1) on sector ETFs"        (lookback + skip)
    - "Momentum 12-2 (skip 2 months)"    (skip parameter)
  These are basic factor research variations.

### A.2 Four-class taxonomy of hardcoded values

Audit revealed every hardcoded value falls into one of 4 categories:

**A class — Safety boundaries (MUST stay hardcoded)**:
  N_TRIALS_HARD=15, _T_GREEN=1.96, _T_MARGINAL=1.65,
  _MIN_STOCKS_PER_BUCKET=30, MAX_AUTO_DISPATCHES_PER_WEEK=5.
  These are statistical / theoretical constants. Auto-relaxing
  them = enabling p-hacking. Lock to `_safety_constants.py`
  module with non-negotiable doctrine.

**B class — Design choices (MUST become parameters)**:
  universe_size, n_buckets, lookback windows, vol_target,
  weighting variants. These are research design freedom that
  hypotheses naturally vary. Add as Optional fields on FactorSpec
  with typed-range validation at dispatcher gate.

**C class — Controlled vocabularies (KEEP hardcoded + documented)**:
  SIGNAL_KINDS, UNIVERSES, REBAL_FREQS, WEIGHTINGS enums.
  PIT_CORRECT_SOURCES whitelist. These define safe operating
  scope; loosening = unsafe expansion of attack surface.

**D class — Implementation details (KEEP hardcoded)**:
  _PIT_FIELD_MAP, _REBAL_DOW="W-FRI". Choices that don't affect
  external behavior or backtest semantics.

### A.3 Expanded phase plan (40h total)

```
Phase 1.1: spec doc                                  SHIPPED
Phase 1.2: comp_pit pull script + baseline JSON     SHIPPED
Phase 1.4: SP500 PIT constituents                   SHIPPED
Phase 1.5: knowable_at column (NEW)                 +2h
           - JOIN fundq for rdq; COALESCE with datadate+120d
           - Rewrite parquet with knowable_at column
Phase 1.6: bitemporal-driven accessor (NEW)         +2h
           - Delete _FUNDA_PUBLIC_LAG_DAYS constant
           - funda_panel: filter by knowable_at <= clock.now
           - Remove application-layer lag arithmetic entirely
Phase 2.1: SimClock                                  SHIPPED
Phase 2.2: PITDataAccessor (current version)        SHIPPED
Phase 2.3: Accessor unit tests                      SHIPPED
Phase 2.5: safety_constants module (NEW)            +1h
           - Move A-class values to engine.agents.strengthener.
             _safety_constants.py
           - Add docstring per constant explaining why immutable
Phase 2.6: FactorSpec v2 + typed-range gate (NEW)   +4h
           - Add Optional B-class fields to FactorSpec:
             universe_size, n_buckets, signal_lookback_m,
             signal_skip_m, vol_target_annual,
             weighting_scheme_alt
           - Dispatcher pre_dispatch_check: B_CLASS_OUT_OF_RANGE
             refusal when LLM picks outside safe range
           - Update extractor prompt to teach LLM about these
             (mark optional, prefer None unless paper specifies)
Phase 3: cross_sec refactor                          +1h on top
         - Pre-existing scope: through accessor
         - NEW: read B-class params from spec, fallback to
           module defaults (no behavior change when None)
         - PARITY: when spec.universe_size=None,
           spec.n_buckets=None, etc., GP/A verdict must match
           pre-refactor baseline ± documented PIT tolerance
Phase 4: tsmom refactor (parameterize lookback/skip/vol_target)
                                                      no extra
Phase 5: audit gate                                  unchanged
```

Total: 30h (original) + 10h (expansion) = **40h**.

### A.4 Layer 3 ENABLEMENT — why this matters

L3-1 LLM Critic Loop CAN now suggest:
  "Suspicion: GP/A's high t is driven by small-cap exposure.
   Recommended follow-up: spec.universe_size=500 to test on
   large-caps only."
And the system can DISPATCH that follow-up. Without B-class
parameters, this suggestion is dead.

L3-3 Variant Generation CAN now produce:
  variant_1: GP/A on top-3000 (baseline)
  variant_2: GP/A on top-500 (large-cap)
  variant_3: GP/A with decile L/S
  variant_4: GP/A industry-demeaned
All as auto-dispatchable SPECs sharing parent_hypothesis_id.

L3-2 Self-Doubt CAN now report:
  "confidence 0.40 — tested only on top-3000 + quintile; pending
   variants on universe_size=500 + n_buckets=10."
This is honest skepticism because the variants ARE possible.

Without B-class parameterization, all three Layer 3 capabilities
are theatrical — they generate text but the system cannot act.

### A.5 Architectural principle (locked)

> "Freedom within safe rails. A-class lock down (statistical
>  truths); B-class parameterize with typed ranges; C-class
>  controlled vocabulary; D-class implementation choice. LLM
>  has maximum design space within the rails."
> — DE Shaw / AQR institutional pattern

This expanded L2-1 plan implements this principle as architecture,
not aspiration.

---

**Spec re-locked 2026-06-08 with expanded scope.**
**Estimated total work: 40h over 4-6 sessions (was 30h).**
**Layer 3 dependency: L2-1 expanded plan is THE prerequisite.**

