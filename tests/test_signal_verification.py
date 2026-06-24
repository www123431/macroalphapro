"""tests/test_signal_verification.py — Commit 2 of the flexibility chain.

Verification cards + redundancy gate + approval ledger.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Approval ledger
# ────────────────────────────────────────────────────────────────────
def _redirect_ledger(monkeypatch, tmp_path):
    import engine.research.signal_verification as sv
    ledger = tmp_path / "approvals.jsonl"
    monkeypatch.setattr(sv, "APPROVALS_LEDGER", ledger)
    return ledger


def test_approve_requires_reason(monkeypatch, tmp_path):
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        sv.approve_signal("gp_at", actor="me", reason="ok")


def test_approve_unknown_signal_raises(monkeypatch, tmp_path):
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    with pytest.raises(KeyError):
        sv.approve_signal("no_such_signal", actor="me",
                            reason="a perfectly valid reason here")


def test_approve_then_load_roundtrip(monkeypatch, tmp_path):
    from engine.research import signal_verification as sv
    ledger = _redirect_ledger(monkeypatch, tmp_path)
    sv.approve_signal("gp_at", actor="principal",
                        reason="card reviewed; spot checks verified")
    approvals = sv.load_approvals()
    assert "gp_at" in approvals
    assert approvals["gp_at"]["actor"] == "principal"
    # ledger is append-only jsonl
    rows = [json.loads(l) for l in
              ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1


def test_tombstone_revokes(monkeypatch, tmp_path):
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    sv.approve_signal("gp_at", actor="p",
                        reason="card reviewed and approved today")
    sv.revoke_signal("gp_at", actor="p",
                       reason="found a PIT bug in the formula, revoking")
    assert "gp_at" not in sv.load_approvals()


def test_ledger_approval_unlocks_dispatchable(monkeypatch, tmp_path):
    """proposed entry + ledger approval → appears in
    dispatchable_signals(). The full proposed→dispatchable arc."""
    import engine.research.signal_registry as sr
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    probe = sr.SignalDefinition(
        key="probe_arc", kind="crsp_panel", direction="long_high",
        family="TEST", required_fields=("crsp.msf.ret",),
        formula=lambda ctx: ctx.rets,
        aliases=(r"probe_arc_nomatch",),
        paper_citation="n/a", pit_notes="n/a", status="proposed",
    )
    monkeypatch.setitem(sr.SIGNAL_REGISTRY, "probe_arc", probe)
    assert "probe_arc" not in sr.dispatchable_signals()
    sv.approve_signal("probe_arc", actor="principal",
                        reason="verification card reviewed, approving")
    assert "probe_arc" in sr.dispatchable_signals()


# ────────────────────────────────────────────────────────────────────
# Redundancy math (synthetic — no data dependency)
# ────────────────────────────────────────────────────────────────────
def test_spearman_identical_signals_near_one():
    from engine.research.signal_verification import (
        _mean_cross_sectional_spearman,
    )
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    cols = range(200)
    a = pd.DataFrame(rng.normal(0, 1, (12, 200)), index=idx, columns=cols)
    rho = _mean_cross_sectional_spearman(a, a * 3.0 + 5.0)  # monotone map
    assert rho is not None and rho > 0.999


def test_spearman_independent_signals_near_zero():
    from engine.research.signal_verification import (
        _mean_cross_sectional_spearman,
    )
    rng = np.random.default_rng(11)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    cols = range(200)
    a = pd.DataFrame(rng.normal(0, 1, (12, 200)), index=idx, columns=cols)
    b = pd.DataFrame(rng.normal(0, 1, (12, 200)), index=idx, columns=cols)
    rho = _mean_cross_sectional_spearman(a, b)
    # 3σ band: SE per month ≈ 1/sqrt(199); averaged over 12 months
    assert abs(rho) < 3.0 / np.sqrt(199 * 12)


def test_spearman_none_when_no_overlap():
    from engine.research.signal_verification import (
        _mean_cross_sectional_spearman,
    )
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    a = pd.DataFrame(np.ones((6, 50)), index=idx, columns=range(50))
    b = pd.DataFrame(np.ones((6, 50)), index=idx,
                       columns=range(100, 150))   # disjoint permnos
    assert _mean_cross_sectional_spearman(a, b) is None


# ────────────────────────────────────────────────────────────────────
# Card generation (monkeypatched panels — fast, no data dependency)
# ────────────────────────────────────────────────────────────────────
def _fake_panels(monkeypatch, panels: dict):
    import engine.research.signal_verification as sv
    monkeypatch.setattr(sv, "_card_panel",
                          lambda key: panels.get(key))


def test_card_flags_strong_redundancy_and_suggests_family(
    monkeypatch, tmp_path,
):
    """A near-duplicate of at_growth must trigger family review with
    at_growth's INVESTMENT family suggested — the Bailey-LdP
    family-dodge defense."""
    import engine.research.signal_registry as sr
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)

    probe = sr.SignalDefinition(
        key="probe_dup", kind="crsp_panel", direction="long_low",
        family="SELF_DECLARED_NEW",         # the dodge attempt
        required_fields=("crsp.msf.ret",),
        formula=lambda ctx: ctx.rets,
        aliases=(r"probe_dup_nomatch",),
        paper_citation="n/a", pit_notes="n/a", status="proposed",
    )
    monkeypatch.setitem(sr.SIGNAL_REGISTRY, "probe_dup", probe)

    rng = np.random.default_rng(13)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    base = pd.DataFrame(rng.normal(0, 1, (12, 200)),
                          index=idx, columns=range(200))
    noisy_dup = base + rng.normal(0, 0.1, (12, 200))   # rank-corr ~0.99
    indep = pd.DataFrame(rng.normal(0, 1, (12, 200)),
                           index=idx, columns=range(200))
    _fake_panels(monkeypatch, {
        "probe_dup": noisy_dup, "at_growth": base, "gp_at": indep,
    })

    card = sv.generate_verification_card(
        "probe_dup",
        redundancy_against=("at_growth", "gp_at"),
        write_md=False,
    )
    assert card is not None
    assert card["family_review_required"] is True
    assert card["suggested_family"] == "INVESTMENT"
    assert abs(card["redundancy"]["at_growth"]) >= 0.7
    assert abs(card["redundancy"]["gp_at"]) < 0.4


def test_card_clean_when_independent(monkeypatch, tmp_path):
    import engine.research.signal_registry as sr
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    probe = sr.SignalDefinition(
        key="probe_indep", kind="crsp_panel", direction="long_high",
        family="GENUINELY_NEW", required_fields=("crsp.msf.ret",),
        formula=lambda ctx: ctx.rets,
        aliases=(r"probe_indep_nomatch",),
        paper_citation="n/a", pit_notes="n/a", status="proposed",
    )
    monkeypatch.setitem(sr.SIGNAL_REGISTRY, "probe_indep", probe)
    rng = np.random.default_rng(17)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    mk = lambda seed: pd.DataFrame(
        np.random.default_rng(seed).normal(0, 1, (12, 200)),
        index=idx, columns=range(200))
    _fake_panels(monkeypatch, {
        "probe_indep": mk(1), "at_growth": mk(2),
    })
    card = sv.generate_verification_card(
        "probe_indep", redundancy_against=("at_growth",),
        write_md=False)
    assert card["family_review_required"] is False
    assert card["suggested_family"] is None
    # Spot checks + distribution present
    assert len(card["spot_checks"]) > 0
    assert "p50" in card["distribution"]


def test_card_md_renders(monkeypatch, tmp_path):
    import engine.research.signal_registry as sr
    from engine.research import signal_verification as sv
    _redirect_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(sv, "CARDS_DIR", tmp_path / "cards")
    rng = np.random.default_rng(19)
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    p = pd.DataFrame(rng.normal(0, 1, (12, 150)),
                       index=idx, columns=range(150))
    _fake_panels(monkeypatch, {"gp_at": p})
    card = sv.generate_verification_card(
        "gp_at", redundancy_against=(), write_md=True)
    md = (tmp_path / "cards" / "gp_at_card.md").read_text(encoding="utf-8")
    assert "SIGNAL VERIFICATION CARD: gp_at" in md
    assert "Spot checks" in md
    assert "Novy-Marx" in md


# ────────────────────────────────────────────────────────────────────
# Template status gate (routing slip, not wall)
# ────────────────────────────────────────────────────────────────────
def test_template_blocks_proposed_signal_with_routing_slip(monkeypatch):
    import engine.research.signal_registry as sr
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        template_cross_sec_us_equities,
    )
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    probe = sr.SignalDefinition(
        key="probe_gate", kind="crsp_panel", direction="long_high",
        family="TEST", required_fields=("crsp.msf.ret",),
        formula=lambda ctx: ctx.rets,
        aliases=(r"probe_gate_signal",),
        paper_citation="n/a", pit_notes="n/a", status="proposed",
    )
    monkeypatch.setitem(sr.SIGNAL_REGISTRY, "probe_gate", probe)
    # rebuild compiled aliases so the probe matches
    monkeypatch.setattr(sr, "_COMPILED_ALIASES", [
        (key, __import__("re").compile(pat, __import__("re").I))
        for key, sdef in sr.SIGNAL_REGISTRY.items()
        for pat in sdef.aliases
    ])
    spec = FactorSpec(
        hypothesis_id="gate_probe", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2014-01:2024-12",
        signal_inputs=("probe_gate_signal",), rebal="monthly",
        weighting="decile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="t", extracted_ts="2026-06-10T00:00:00Z", model="c",
    )
    r = template_cross_sec_us_equities(spec)
    assert r.verdict == "SIGNAL_PENDING_APPROVAL"
    # Routing slip, not a wall: tells the user EXACTLY how to unblock
    assert "verification card" in r.summary
    assert "approve_signal" in r.summary
    assert r.metrics["gap_class"] == "TIER_2_PENDING_APPROVAL"
