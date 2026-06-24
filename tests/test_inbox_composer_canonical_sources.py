"""tests/test_inbox_composer_canonical_sources.py — G.1+G.2+G.3.

Tests the 3 new composer sources that route canonical research_store
events into the Inbox v3 UI:
  - source_decay_alerts_canonical (G.1)
  - source_specification_robustness_overfit (G.2)
  - source_anchor_spanned_factors (G.3)

Architectural intent: zero new UI code; events emitted by C trigger
and B lens flow into the existing inbox composer via the same
source_* contract every other inbox source uses.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_event(
    *,
    event_id="evt_test_1",
    event_type="decay_alert",
    subject_id="cross_asset_carry",
    verdict="MARGINAL",
    ts="2026-06-09T12:00:00Z",
    summary="Decay watch [MARGINAL] on cross_asset_carry: w/b=0.13 — SUGGESTION: review",
    metrics=None,
    tags=("decay_watch", "review_recommended"),
):
    """Minimal ResearchEvent stand-in matching the filter_events
    return shape."""
    class _V:
        def __init__(s, v): s.value = v
    return SimpleNamespace(
        event_id=event_id,
        event_type=event_type,
        subject_id=subject_id,
        verdict=_V(verdict),
        ts=ts,
        summary=summary,
        metrics=metrics or {},
        tags=tags,
        family=None,
    )


# ────────────────────────────────────────────────────────────────────
# G.1 — decay_alert canonical source
# ────────────────────────────────────────────────────────────────────
def test_g1_decay_alert_renders_when_decay_watch_tag_present():
    from engine.inbox import composer
    metrics = {
        "triggers_hit":            ["A", "B", "C"],
        "n_triggers":              3,
        "severity":                "RED",
        "worst_best_sharpe_ratio": 0.13,
        "monotone_decay":          True,
    }
    evt = _fake_event(
        verdict="RED", metrics=metrics,
        tags=("decay_watch", "review_recommended"),
    )
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_decay_alerts_canonical()
    assert len(items) == 1
    it = items[0]
    assert it["source"] == "decay_watch"
    assert it["tone"] == "alert"    # RED → alert
    assert "RED" in it["title"]
    assert "cross_asset_carry" in it["title"]
    # All 3 trigger letters render in the title
    for letter in ("A", "B", "C"):
        assert letter in it["title"]
    assert it["metadata"]["severity"] == "RED"
    assert it["metadata"]["worst_best_sharpe_ratio"] == 0.13


def test_g1_skips_events_without_decay_watch_tag():
    """Legacy SLM decay sentinel writes decay_alert event_type too,
    but without the `decay_watch` tag. G.1 source must filter to
    ONLY the C trigger events to avoid double-surface."""
    from engine.inbox import composer
    legacy = _fake_event(tags=("slm_legacy",))
    with patch("engine.research_store.store.filter_events",
                  return_value=[legacy]):
        items = composer.source_decay_alerts_canonical()
    assert items == []


def test_g1_tone_marginal_when_severity_marginal():
    from engine.inbox import composer
    metrics = {"triggers_hit": ["A", "B"], "severity": "MARGINAL",
                 "worst_best_sharpe_ratio": 0.15}
    evt = _fake_event(verdict="MARGINAL", metrics=metrics)
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_decay_alerts_canonical()
    assert items[0]["tone"] == "warn"


def test_g1_strips_suggestion_suffix_from_preview():
    """Per [[feedback-no-ugly-markdown-in-llm-render-2026-06-02]], the
    inbox 1-line preview must be clean. The "SUGGESTION:" suffix is
    redundant when the click target makes the action obvious."""
    from engine.inbox import composer
    evt = _fake_event(
        summary="Decay watch [RED] on X: w/b=0.07 — SUGGESTION: review capital allocation.",
        metrics={"severity": "RED", "worst_best_sharpe_ratio": 0.07,
                   "triggers_hit": ["A", "B", "C"]},
    )
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_decay_alerts_canonical()
    assert "SUGGESTION" not in items[0]["summary"]


# ────────────────────────────────────────────────────────────────────
# G.2 — specification_robustness LIKELY_OVERFIT
# ────────────────────────────────────────────────────────────────────
def test_g2_likely_overfit_surfaces():
    from engine.inbox import composer
    metrics = {
        "specification_robustness": {
            "verdict":         "LIKELY_OVERFIT",
            "stability_score": 0.17,
            "base_sharpe":     1.80,
            "sharpe_median":   0.30,
            "neighborhood_size": 8,
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed",
                          metrics=metrics, subject_id="auto_factor_X")
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_specification_robustness_overfit()
    assert len(items) == 1
    it = items[0]
    assert it["source"] == "spec_robust"
    assert it["tone"] == "alert"   # LIKELY → alert (red)
    assert "Likely overfit" in it["title"]
    assert "stability=0.17" in it["title"]
    assert "auto_factor_X" in it["title"]


def test_g2_marginal_overfit_uses_warn_tone():
    from engine.inbox import composer
    metrics = {
        "specification_robustness": {
            "verdict":         "MARGINAL_OVERFIT",
            "stability_score": 0.50,
            "base_sharpe":     1.00,
            "sharpe_median":   0.50,
            "neighborhood_size": 8,
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed", metrics=metrics)
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_specification_robustness_overfit()
    assert items[0]["tone"] == "warn"


def test_g2_skips_robust_verdicts():
    """ROBUST is the happy path — no notification needed."""
    from engine.inbox import composer
    metrics = {
        "specification_robustness": {
            "verdict": "ROBUST", "stability_score": 0.90,
            "base_sharpe": 1.0, "sharpe_median": 0.92,
            "neighborhood_size": 8,
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed", metrics=metrics)
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_specification_robustness_overfit()
    assert items == []


def test_g2_skips_events_without_spec_robustness_block():
    from engine.inbox import composer
    evt = _fake_event(event_type="factor_verdict_filed", metrics={})
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_specification_robustness_overfit()
    assert items == []


# ────────────────────────────────────────────────────────────────────
# G.3 — anchor-spanned factors (residual α << headline α)
# ────────────────────────────────────────────────────────────────────
def test_g3_surfaces_gp_a_style_anchor_spanned_factor():
    """GP/A canonical pattern: headline t=3.57, residual t=0.80 →
    factor is a textbook RMW restatement, not novel alpha."""
    from engine.inbox import composer
    metrics = {
        "nw_t_stat": 3.57,
        "anchor_orthogonality": {
            "alpha_nw_t": 0.80,
            "anchor_library": "ken_french_ff5_mom",
            "betas": {"RMW": 0.67, "HML": -0.35, "MKT_RF": 0.14},
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed",
                          metrics=metrics, subject_id="auto_gpa_like")
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_anchor_spanned_factors()
    assert len(items) == 1
    it = items[0]
    assert it["source"] == "anchor_spanned"
    assert it["tone"] == "warn"
    assert "auto_gpa_like" in it["title"]
    # Headline → residual t gap visible
    assert "3.57" in it["title"]
    assert "0.80" in it["title"]
    # Dominant β (RMW) surfaced in summary
    assert "RMW" in it["summary"]


def test_g3_skips_factor_with_genuine_residual_alpha():
    """A factor with residual t=2.5 has SURVIVED anchor stripping —
    not spanned. No notification needed (it's the happy path)."""
    from engine.inbox import composer
    metrics = {
        "nw_t_stat": 3.20,
        "anchor_orthogonality": {
            "alpha_nw_t": 2.5,
            "anchor_library": "ken_french_ff5_mom",
            "betas": {"RMW": 0.15, "HML": 0.05},
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed", metrics=metrics)
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_anchor_spanned_factors()
    assert items == []


def test_g3_skips_factor_without_meaningful_gap():
    """headline t=1.80, residual t=1.50 — both pretty close to the
    threshold, gap=0.30 is just sampling noise. Don't fire."""
    from engine.inbox import composer
    metrics = {
        "nw_t_stat": 1.80,
        "anchor_orthogonality": {
            "alpha_nw_t": 1.50,
            "anchor_library": "ken_french_ff5_mom",
            "betas": {"HML": 0.20},
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed", metrics=metrics)
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_anchor_spanned_factors()
    assert items == []


def test_g3_fx_lens_also_surfaces():
    """FX-carry anchor lens (B.1) produces the same shape — must also
    surface when an FX carry candidate is just a HML_FX restatement."""
    from engine.inbox import composer
    metrics = {
        "nw_t_stat": 2.94,
        "anchor_orthogonality": {
            "alpha_nw_t": 0.20,
            "anchor_library": "lrv_fx_carry",
            "betas": {"HML_FX": 1.0, "DOL": -0.01},
        }
    }
    evt = _fake_event(event_type="factor_verdict_filed",
                          metrics=metrics, subject_id="auto_carry_rephrase")
    with patch("engine.research_store.store.filter_events",
                  return_value=[evt]):
        items = composer.source_anchor_spanned_factors()
    assert len(items) == 1
    assert "lrv_fx_carry" in items[0]["summary"]
    assert "HML_FX" in items[0]["summary"]


# ────────────────────────────────────────────────────────────────────
# Integration — all 3 sources wired into compose_inbox
# ────────────────────────────────────────────────────────────────────
def test_compose_inbox_includes_new_sources():
    """All 3 G.* sources must appear in the standard inbox composition."""
    from engine.inbox import composer
    decay_evt = _fake_event(
        event_type="decay_alert",
        metrics={"triggers_hit":["A","B","C"], "severity":"RED",
                  "worst_best_sharpe_ratio":0.07, "monotone_decay":True},
        tags=("decay_watch","review_recommended"),
    )
    overfit_evt = _fake_event(
        event_id="evt_overfit", event_type="factor_verdict_filed",
        subject_id="auto_overfit",
        metrics={"specification_robustness":{
            "verdict":"LIKELY_OVERFIT", "stability_score":0.17,
            "base_sharpe":1.8, "sharpe_median":0.3,
            "neighborhood_size":8,
        }}, tags=(),
    )
    spanned_evt = _fake_event(
        event_id="evt_spanned", event_type="factor_verdict_filed",
        subject_id="auto_gpa_like",
        metrics={"nw_t_stat":3.57,
                  "anchor_orthogonality":{"alpha_nw_t":0.80,
                    "anchor_library":"ken_french_ff5_mom",
                    "betas":{"RMW":0.67}}},
        tags=(),
    )
    def _fake_filter(event_type=None, limit=None, **kw):
        if event_type == "decay_alert":
            return [decay_evt]
        if event_type == "factor_verdict_filed":
            return [overfit_evt, spanned_evt]
        return []
    with patch("engine.research_store.store.filter_events", _fake_filter):
        out = composer.compose_inbox()
    sources = {it["source"] for it in out["items"]}
    assert "decay_watch" in sources
    assert "spec_robust" in sources
    assert "anchor_spanned" in sources
