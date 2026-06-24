"""
Capability evidence MD writer for horse race specs (P/Q/S/T/U).

Writes `docs/capability_evidence/path_<X>_macro_<...>_<verdict>_<date>.md`
per spec post-backtest. Format matches existing project capability evidence
convention (e.g., `path_k1_size_expanded_b_plus_*` pattern).

Output includes:
  - Pre-registration anchor (spec hash · dataset hash · code commit)
  - All 4 gate results (PASS/FAIL + value + threshold)
  - Statistical augmentation (Newey-West t · bootstrap CI)
  - Crisis window breakdown
  - Decision rule application
  - Multiple-comparison caveat (per spec §2.6)
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

from engine.portfolio.macro_cta_research.gate_eval import GateResult
from engine.portfolio.macro_cta_research.crisis_windows import CRISIS_WINDOWS

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("docs/capability_evidence")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _verdict_class(verdict: str) -> str:
    return {
        "PASS":     "**PASS** ✓",
        "MARGINAL": "**MARGINAL** ◐",
        "FAIL":     "**FAIL** ✗",
    }.get(verdict, verdict)


def _gate_row(name: str, passed: bool, value: str, threshold: str) -> str:
    marker = "PASS" if passed else "FAIL"
    return f"| {name} | {marker} | {value} | {threshold} |"


def write_capability_evidence(
    *,
    spec_id:          int,
    spec_path:        str,
    spec_label:       str,
    spec_hash:        str,
    dataset_hash:     str,
    backtest_result,           # BacktestResult instance
    gate_result:      GateResult,
    n_backtest_weeks: int,
    code_commit_hash: Optional[str] = None,
    horse_race_context: Optional[str] = None,
) -> Path:
    """Write capability evidence MD for one spec post-backtest. Returns path."""
    today = datetime.date.today()
    fname = f"{spec_label.lower().replace(' ', '_')}_{gate_result.verdict.lower()}_{today.isoformat()}.md"
    out_path = _OUTPUT_DIR / fname

    crisis_table = "\n".join(
        f"| {key} | {start.isoformat()} → {end.isoformat()} | "
        f"{gate_result.crisis_returns.get(key, float('nan')):+.2%} | "
        f"{'positive' if gate_result.crisis_returns.get(key, 0) > 0 else 'non-positive'} |"
        for key, (start, end) in CRISIS_WINDOWS.items()
    )

    content = f"""# Capability Evidence — {spec_label}

**Spec**: id={spec_id} · hash `{spec_hash[:12]}` · `{spec_path}`
**Window**: 2014-09-12 → 2023-12-29 ({n_backtest_weeks} weeks effective)
**Run date**: {today.isoformat()}
**Reproducibility trinity**:
- Code commit: {code_commit_hash[:12] if code_commit_hash else 'pending'}
- Spec hash: `{spec_hash[:12]}`
- Dataset hash: `{dataset_hash[:12]}`

## Verdict: {_verdict_class(gate_result.verdict)} ({gate_result.n_gates_passed}/4 gates passed)

{horse_race_context or ''}

## Gate evaluation (head-to-head vs PQTIX baseline)

| Gate | Outcome | Spec value | Threshold |
|---|---|---|---|
{_gate_row('G1 Sharpe (net 10bp TC)', gate_result.g1_pass,
            f'{gate_result.spec_sharpe:.3f}',
            f'≥ PQTIX {gate_result.pqtix_sharpe:.3f}')}
{_gate_row('G2 Max drawdown', gate_result.g2_pass,
            f'{gate_result.spec_max_dd:+.2%}',
            f'≥ PQTIX × 1.1 = {gate_result.pqtix_max_dd * 1.10:+.2%}')}
{_gate_row('G3 ρ vs (K1+DPEAD+PathN)', gate_result.g3_pass,
            f'{gate_result.spec_corr_other:+.3f}',
            '|ρ| ≤ 0.15')}
{_gate_row('G4 Crisis-positive', gate_result.g4_pass,
            f'{gate_result.spec_crisis_pos} of 3',
            '≥ 2 of 3')}

## Crisis window breakdown

| Window | Date range | Cumulative return | Status |
|---|---|---|---|
{crisis_table}

## Statistical augmentation

- **Newey-West HAC t-stat** (lag-8) on excess return vs PQTIX: `{gate_result.sharpe_nw_t:.3f}`
- **Sharpe-difference 95% CI** (stationary bootstrap, 12-week blocks, n=1000):
  `[{gate_result.sharpe_ci_lo:+.3f}, {gate_result.sharpe_ci_hi:+.3f}]`

Spec Sharpe Δ vs PQTIX: `{gate_result.sharpe_delta:+.3f}`

## Backtest metrics summary

| Metric | Value |
|---|---|
| Annualized return | {backtest_result.ann_return*100:+.2f}% |
| Annualized vol | {backtest_result.ann_vol*100:.2f}% |
| Sharpe (RFR=4%) | {backtest_result.sharpe:.3f} |
| Max drawdown | {backtest_result.max_drawdown*100:+.2f}% |
| N rebalances | {backtest_result.n_rebalances} |
| N weeks | {backtest_result.n_weeks} |
| Avg turnover/rebal | {backtest_result.avg_turnover_per_rebalance:.4f} |
| Final NAV (start 1.0) | {backtest_result.nav.iloc[-1]:.4f} |

## Multiple-comparison caveat (per spec §2.6 disclosure)

This spec is one of **5 active candidates** in pre-registered horse race
(Path P/Q/S/T/U vs PQTIX baseline). Family-wise Type-I error: with 5
candidates × 4 gates = 20 tests at α=0.05 each, probability of at least
one false PASS = 1 − (1−0.05)^5 = 22.6%.

Per McLean-Pontiff 2016 / Harvey-Liu-Zhu 2016 academic standard, this
single-spec verdict should be interpreted in horse-race context. Bonferroni
NOT applied within-spec (would over-shrink); rather disclosed at family level.

Winner (if any single PASS) stands on its head-to-head merits vs PQTIX
baseline, not on "we tested 5 and 1 passed."

## Decision rule outcome

- 4/4 gates PASS → spec PASS · candidate for sleeve replacement
- 3/4 PASS      → MARGINAL · log entry · keep PQTIX
- ≤ 2/4 PASS    → FAIL · falsification chain entry · keep PQTIX

**This spec result: {gate_result.n_gates_passed}/4 → {gate_result.verdict}**
"""

    out_path.write_text(content, encoding="utf-8")
    logger.info("capability evidence written: %s", out_path)
    return out_path
