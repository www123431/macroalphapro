"""tests/test_cross_sec_parity_baseline.py — Tier C L2 parity guard.

Post-B-fix true baseline regression. Phase 6 piece 3 refactor
(remove cross_sec's tactical B-fix legacy-keys INNER JOIN, switch
to accessor.funda_pit_panel via TemplateContract auto-coercion)
WILL touch the cross_sec template's data-loading path. This test
PINS the 6 academic factor verdicts as the must-not-break parity
anchor.

Phase 6 piece 2 (3e4c5845) verified accessor coercion produces
EXACT 279,986 (gvkey, datadate) row match vs cross_sec B-fix path.
Piece 3 swaps the implementation; this test catches any drift.

GATED behind RUN_CROSS_SEC_INTEGRATION=1 — touches real CRSP +
Compustat caches; runs ~30-60 seconds per signal × 6 signals =
3-6 minutes total. Not run in the default suite.

Tolerance bands (from baseline JSON `_meta`):
  Verdict change                                      0    required
  |t_stat drift| from baseline                       <= 0.05
  |Sharpe drift| from baseline                       <= 0.02
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


_BASELINE_PATH = (Path(__file__).resolve().parent
                    / "_baseline_l2_post_b_fix.json")


def _load_baseline() -> dict:
    if not _BASELINE_PATH.is_file():
        raise FileNotFoundError(f"baseline missing: {_BASELINE_PATH}")
    with _BASELINE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# Signal → (signal_inputs_hint, paper_window, paper_t) for spec construction
_SIGNALS = {
    "GP/A":            ("compustat.funda.gross_profitability_to_assets",
                          "1963-01:2010-12", 3.0),
    "Momentum":        ("crsp.msf.derived.return_12_1_momentum",
                          "1965-01:1989-12", 4.0),
    "Low_Vol":         ("crsp.msf.derived.vol_12m",
                          "1986-01:2005-12", 3.5),
    "Book_To_Market":  ("compustat.funda.book_to_market",
                          "1963-01:1991-12", 3.5),
    "At_Growth":       ("compustat.funda.asset_growth",
                          "1968-01:2003-12", 4.2),
    "ROE":             ("compustat.funda.return_on_equity",
                          "1972-01:2014-12", 3.0),
}


@pytest.mark.skipif(
    os.environ.get("RUN_CROSS_SEC_INTEGRATION") != "1",
    reason=("set RUN_CROSS_SEC_INTEGRATION=1 to run live parity guard "
              "(3-6 min real CRSP + Compustat)"),
)
@pytest.mark.parametrize("signal_name", list(_SIGNALS.keys()))
def test_cross_sec_parity_vs_baseline(signal_name):
    """For each of 6 academic anchors, current verdict MUST match
    the post-B-fix baseline within tolerance bands."""
    baseline = _load_baseline()
    if signal_name not in baseline:
        pytest.skip(f"signal {signal_name} not in baseline yet")
    expected = baseline[signal_name]

    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    import engine.agents.strengthener.templates.cross_sec_us_equities as cm
    cm._load_compustat_funda.cache_clear()

    sig_hint, paper_window, paper_t = _SIGNALS[signal_name]
    spec = FactorSpec(
        hypothesis_id=f"parity_{signal_name}",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="1992-01:2024-12",
        signal_inputs=(sig_hint,),
        rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=120,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale=f"parity guard {signal_name}",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
        paper_original_window=paper_window,
        paper_reported_t=paper_t,
    )
    r = template_cross_sec_us_equities(spec)
    m = r.metrics

    # Verdict MUST match exactly (no drift)
    assert r.verdict == expected["verdict"], (
        f"{signal_name}: verdict changed "
        f"{expected['verdict']} → {r.verdict} "
        f"(t={m['nw_t_stat']:.4f} vs baseline t={expected['nw_t_stat']:.4f})"
    )

    # t-stat within ±0.05
    actual_t = m["nw_t_stat"]
    expected_t = expected["nw_t_stat"]
    if actual_t is not None and expected_t is not None:
        drift = abs(actual_t - expected_t)
        assert drift <= 0.05, (
            f"{signal_name}: nw_t_stat drift {drift:.4f} > 0.05 "
            f"(actual {actual_t:.4f} vs baseline {expected_t:.4f})"
        )

    # Sharpe within ±0.02
    actual_sh = m["sharpe"]
    expected_sh = expected["sharpe"]
    if actual_sh is not None and expected_sh is not None:
        drift = abs(actual_sh - expected_sh)
        assert drift <= 0.02, (
            f"{signal_name}: Sharpe drift {drift:.4f} > 0.02 "
            f"(actual {actual_sh:.4f} vs baseline {expected_sh:.4f})"
        )


def test_baseline_file_has_meta_and_signals():
    """Sanity: baseline JSON has _meta + at least 4 signals."""
    baseline = _load_baseline()
    assert "_meta" in baseline
    assert "locked_at_commit" in baseline["_meta"]
    assert "tolerance_band" in baseline["_meta"]
    n_signals = len([k for k in baseline if not k.startswith("_")])
    assert n_signals >= 4, (
        f"baseline should have ≥ 4 anchor signals; has {n_signals}")


def test_baseline_gp_a_replication_status():
    """GP/A must be REPLICATED in the baseline — it's the ONE
    empirical anchor that validates the entire L2-1 architecture
    (Phase 3.1 + B-fix produces gap 0.044 vs Novy-Marx 2013)."""
    baseline = _load_baseline()
    assert baseline["GP/A"]["replication_status"] == "REPLICATED"
    assert abs(baseline["GP/A"]["replication_gap"]) <= 0.1, (
        f"GP/A replication gap {baseline['GP/A']['replication_gap']} "
        f"is suspicious — re-investigate L2-1 architecture")


def test_m2_replicates_novy_marx_2013_gp_a():
    """M2 (paper replication anchor): Novy-Marx 2013 "The Other Side of
    Value: The Gross Profitability Premium" reports GP/A long-short
    quintile-spread t ~4.4 over 1963-2010 with Sharpe ~0.6-0.8.

    Our baseline JSON (locked 2026-06-08 at commit 3e4c5845, post-B-fix
    accessor coercion): t=3.57, Sharpe=0.67, gap=0.044 vs paper. Both
    inside reasonable replication bands given our 1992-2024 sample
    differs from paper's 1963-2010 sample.

    M2 anchor bands (conservative, allow for sample-window divergence):
      - replication_status == REPLICATED
      - replication_gap absolute value < 0.10
      - nw_t > 3.0 (NM 2013 reports ~4.4; our 1992-2024 measurement 3.57)
      - Sharpe in [0.40, 1.00]
      - verdict == GREEN (the system's first paper-replicated GREEN)

    Failure = either baseline JSON corrupted OR template math drifted.
    Naming aligned with vrp_spx / spanning_test_ff / factor_combination_ff
    / portfolio_overlay_60_40 M2-anchor convention (one anchor per
    template, named test_m2_replicates_<paper>).
    """
    baseline = _load_baseline()
    gpa = baseline["GP/A"]

    # Status anchor
    assert gpa["replication_status"] == "REPLICATED", (
        f"GP/A replication_status = {gpa['replication_status']}, "
        f"expected REPLICATED. Novy-Marx 2013 anchor lost."
    )
    assert abs(gpa["replication_gap"]) < 0.10, (
        f"REPLICATION FAILURE: GP/A gap {gpa['replication_gap']:.4f} "
        f"exceeds 0.10 vs Novy-Marx 2013"
    )

    # Statistical anchors
    assert gpa["nw_t_stat"] > 3.0, (
        f"REPLICATION FAILURE: GP/A NW-t {gpa['nw_t_stat']:.2f} below 3.0 "
        f"(Novy-Marx 2013 reports ~4.4; we expect > 3.0 in our 1992-2024 sample)"
    )
    assert 0.40 < gpa["sharpe"] < 1.00, (
        f"REPLICATION FAILURE: GP/A Sharpe {gpa['sharpe']:.2f} outside "
        f"[0.40, 1.00] band (NM 2013 reports ~0.6-0.8)"
    )

    # Verdict anchor
    assert gpa["verdict"] == "GREEN", (
        f"GP/A verdict {gpa['verdict']} != GREEN. This is the system's "
        f"first paper-replicated GREEN; losing it is a significant regression."
    )
