"""bt-flex-4.2 tests for factor_combination_ff template."""
from __future__ import annotations

import math
import pandas as pd
import pytest

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.agents.strengthener.templates import factor_combination_ff as fc


def _spec(**kw):
    base = dict(
        hypothesis_id="t-fc",
        signal_kind="factor_combination",
        universe="ken_french_ff5_mom",
        date_range="1972-01:2025-12",
        signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"),
        rebal="monthly",
        weighting="ew",
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("restatement",),
        cost_model="80bp_per_yr",
        rationale="test",
        extracted_ts="2026-06-11T11:30:00Z",
        model="test",
    )
    base.update(kw)
    return FactorSpec(**base)


# ── parse_weight ──────────────────────────────────────────────────


def test_parse_weight_decimal():
    s = _spec(weighting_scheme_alt="0.30")
    assert fc._parse_weight(s) == 0.30


def test_parse_weight_percent():
    s = _spec(weighting_scheme_alt="40")
    assert abs(fc._parse_weight(s) - 0.40) < 1e-9


def test_parse_weight_default_when_missing():
    assert fc._parse_weight(_spec()) == 0.50


def test_parse_weight_clamps_out_of_range():
    assert fc._parse_weight(_spec(weighting_scheme_alt="0.99")) == 0.50
    assert fc._parse_weight(_spec(weighting_scheme_alt="0.01")) == 0.50


# ── parse_factor_inputs ───────────────────────────────────────────


def test_parse_inputs_canonical_form():
    s = _spec(signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"))
    assert fc._parse_factor_inputs(s) == ("HML", "MOM")


def test_parse_inputs_short_prefix_accepted():
    s = _spec(signal_inputs=("ff.smb", "ff.cma"))
    assert fc._parse_factor_inputs(s) == ("SMB", "CMA")


def test_parse_inputs_rejects_wrong_count():
    s = _spec(signal_inputs=("ff.factors_weekly.hml",))
    assert fc._parse_factor_inputs(s) is None


def test_parse_inputs_rejects_duplicate_factor():
    s = _spec(signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.hml"))
    assert fc._parse_factor_inputs(s) is None


def test_parse_inputs_rejects_unknown_factor():
    s = _spec(signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.qrt"))
    assert fc._parse_factor_inputs(s) is None


# ── classify_verdict ──────────────────────────────────────────────


def test_verdict_green_when_strong_nw_t_and_alpha_t():
    v, _ = fc._classify_verdict(nw_t=3.0, alpha_t=2.0, nw_t_cost=2.5)
    assert v == "GREEN"


def test_verdict_marginal_when_t_in_marginal_band():
    v, _ = fc._classify_verdict(nw_t=1.80, alpha_t=1.50, nw_t_cost=1.70)
    assert v == "MARGINAL"


def test_verdict_red_when_t_below_marginal():
    v, _ = fc._classify_verdict(nw_t=1.20, alpha_t=0.80, nw_t_cost=1.10)
    assert v == "RED"


def test_verdict_cost_stress_downgrades_green_to_marginal():
    v, note = fc._classify_verdict(nw_t=2.5, alpha_t=2.0, nw_t_cost=1.75)
    assert v == "MARGINAL"
    assert "cost-stress" in note


# ── End-to-end with live data ─────────────────────────────────────


def test_live_smoke_amp2013_50_50_hml_mom():
    """Asness-Moskowitz-Pedersen 2013 50/50 HML+MOM canonical test.

    BUG-1 fix 2026-06-13: previously this expected verdict ∈ {GREEN,
    MARGINAL, RED} freely. With the FF-complement spanning fix, the
    verdict is constrained — combo of HML+MOM should NOT clear FF5+MOM
    complement spanning at high confidence (RMW+CMA absorb most of the
    Sharpe improvement). Expect MARGINAL or RED, NOT GREEN.
    """
    s = _spec(weighting_scheme_alt="0.50")
    result = fc.template_factor_combination_ff(s)
    assert result.verdict in {"MARGINAL", "RED",
                                "INSUFFICIENT_DATA",
                                "INSUFFICIENT_HISTORY"}, (
        f"BUG-1 regression — HML+MOM combo cleared FF-complement "
        f"spanning at GREEN ({result.metrics.get('ff_complement_alpha_t')})"
    )
    if result.verdict in {"MARGINAL", "RED"}:
        m = result.metrics
        # Sanity: factor names + weights echo correctly
        assert m["factor_a"] == "HML"
        assert m["factor_b"] == "MOM"
        assert abs(m["weight_a"] - 0.50) < 1e-9
        # Sample size: Ken French data is 1963-2026, ≥ 60 years monthly
        assert m["n_obs_months"] >= 600
        # Sharpe should be plausible (HML+MOM combo historical ~0.5-1.0)
        assert -0.5 < m["sharpe_gross"] < 1.5
        # Cost-stressed Sharpe < gross by ~0.10 (80bp/yr drag)
        assert m["sharpe_net_80bp"] < m["sharpe_gross"]


def test_bug1_ff_complement_anchor_excludes_combo_factors():
    """The complement of a combo must NOT include the combo's own factors.
    HML+MOM combo regressed against FF5+MOM \\ {HML, MOM} = MKT+SMB+RMW+CMA."""
    cols = fc._ff_complement_columns("HML", "MOM")
    assert "HML" not in cols
    assert "MOM" not in cols
    assert set(cols) == {"MKT_RF", "SMB", "RMW", "CMA"}


def test_bug1_ff_complement_anchor_for_rmw_cma():
    """RMW+CMA combo's complement = FF5+MOM \\ {RMW, CMA} = MKT+SMB+HML+MOM."""
    cols = fc._ff_complement_columns("RMW", "CMA")
    assert "RMW" not in cols
    assert "CMA" not in cols
    assert set(cols) == {"MKT_RF", "SMB", "HML", "MOM"}


def test_m2_replicates_amp2013_50_50_hml_mom_sharpe_within_band():
    """M2 (paper replication anchor): AMP-2013 reports 50/50 V+M
    combo Sharpe ~0.86 in their global universe. Our US-equity-only
    Ken French 1963-2026 monthly produces Sharpe ~0.70 — different
    universe, different period, different convention. Anchor: gross
    Sharpe in [0.50, 1.00] band and J-K vs each component t > 1.5.

    Failure = template math drifted; CI blocks merge.
    """
    s = _spec(weighting_scheme_alt="0.50")
    result = fc.template_factor_combination_ff(s)
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        m = result.metrics
        # Sharpe band: AMP-2013 paper number is 0.86; ours is 0.70 in
        # this sample. Both well inside [0.50, 1.00].
        assert 0.50 < m["sharpe_gross"] < 1.00, (
            f"REPLICATION FAILURE: AMP-2013 HML+MOM combo Sharpe "
            f"{m['sharpe_gross']:.2f} outside [0.50, 1.00] band"
        )
        # J-K paired t-stats vs each component should both be
        # at least marginally significant (combo strictly beats each)
        assert m["jk_vs_a_t"] > 1.5, (
            f"REPLICATION FAILURE: combo vs HML t={m['jk_vs_a_t']:.2f} < 1.5"
        )
        assert m["jk_vs_b_t"] > 1.5, (
            f"REPLICATION FAILURE: combo vs MOM t={m['jk_vs_b_t']:.2f} < 1.5"
        )


def test_bug1_live_hml_mom_combo_ff_complement_alpha_is_small():
    """BUG-1 regression test: after fix, HML+MOM 50/50 should show
    near-zero alpha vs FF complement (RMW+CMA absorb the alpha).

    Sanity: previous CAPM α-t was ~2.30 (which falsely gave GREEN).
    FF-complement α-t should be << 1.65 (the GREEN threshold).
    """
    s = _spec(weighting_scheme_alt="0.50")
    result = fc.template_factor_combination_ff(s)
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        m = result.metrics
        ff_t = m["ff_complement_alpha_t"]
        capm_t = m["capm_alpha_t"]
        # FF complement α-t MUST be much smaller than CAPM α-t for
        # the AMP-2013 case (proof BUG-1 fix is engaged)
        assert abs(ff_t) < abs(capm_t), (
            f"BUG-1 expects FF-complement α-t ({ff_t:.2f}) < "
            f"CAPM α-t ({capm_t:.2f}) — the spanning is now stricter"
        )
        # And the FF complement α should be < the GREEN threshold
        assert abs(ff_t) < 1.65, (
            f"FF-complement α-t {ff_t:.2f} should be below 1.65 "
            f"for HML+MOM combo — combo is explained by RMW+CMA"
        )


def test_live_smoke_signal_input_unknown():
    """Bad signal_inputs surface as SIGNAL_INPUT_UNKNOWN verdict."""
    s = _spec(signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.invalid"))
    result = fc.template_factor_combination_ff(s)
    assert result.verdict == "SIGNAL_INPUT_UNKNOWN"


def test_live_smoke_pnl_series_present():
    s = _spec()
    result = fc.template_factor_combination_ff(s)
    if result.verdict in {"GREEN", "MARGINAL", "RED"}:
        assert "pnl_series_df" in result.artifacts
        assert "pnl_default_col" in result.artifacts
        df = result.artifacts["pnl_series_df"]
        assert "pnl_gross" in df.columns
        assert "pnl_net_13bp" in df.columns
