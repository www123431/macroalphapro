"""tests/test_factor_dispatcher.py — Tier C-2a.

Tests the dispatcher's outer shell + gates + audit log. Template
implementations are stubs in C-2a (PENDING_TEMPLATE_BUILD verdict);
C-2b/e/f will add per-template test files when real templates land.

All tests offline + free — no LLM calls (extractor not invoked,
dispatcher reads FactorSpec dataclass directly).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────
def _spec(**overrides):
    """Build a FactorSpec with defaults that pass all gates."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id           = "hid_test",
        signal_kind             = "cross_sectional_rank",
        universe                = "us_equities_top_3000",
        date_range              = "2000-01:2024-12",
        signal_inputs           = ("crsp.msf.ret",
                                    "compustat.funda.gp_at"),
        rebal                   = "monthly",
        weighting               = "decile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months          = 120,
        pit_audits              = ("restatement", "lookahead",
                                    "survivorship"),
        cost_model              = "engine.execution.cost_model.basic",
        rationale               = "test spec",
        extracted_ts            = "2026-06-08T00:00:00Z",
        model                   = "claude-sonnet-4-6",
    )
    base.update(overrides)
    return FactorSpec(**base)


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    """Tmp dispatch log path + redirect family-n_trials ledger to
    an empty location so tests are deterministic."""
    log = tmp_path / "factor_dispatch_log.jsonl"
    # Point family-n_trials ledger at empty tmp (else CI vs local
    # diverge based on whatever lives in data/agents/...)
    fake_ledger_root = tmp_path / "data_root_fake"
    (fake_ledger_root / "data" / "agents"
       / "workflow_executor").mkdir(parents=True, exist_ok=True)
    from engine.agents.strengthener import factor_dispatcher as fd

    # Monkey-patch _family_n_trials_now to return 0 by default; per-test
    # overrides set it higher to exercise the gate
    monkeypatch.setattr(fd, "_family_n_trials_now",
                          lambda fam: 0)
    return log


# ────────────────────────────────────────────────────────────────────
# Spec hash — stable + sensitive
# ────────────────────────────────────────────────────────────────────
def test_spec_hash_stable_across_diagnostics_change():
    from engine.agents.strengthener.factor_dispatcher import _spec_hash
    s1 = _spec()
    s2 = _spec(extracted_ts="2099-12-31T23:59:59Z",
                model="claude-sonnet-9-99")
    assert _spec_hash(s1) == _spec_hash(s2), \
        "spec_hash must be stable across diagnostic fields"


def test_spec_hash_changes_on_signal_kind_change():
    from engine.agents.strengthener.factor_dispatcher import _spec_hash
    assert _spec_hash(_spec()) != \
        _spec_hash(_spec(signal_kind="vrp"))


def test_spec_hash_is_16_hex_chars():
    from engine.agents.strengthener.factor_dispatcher import _spec_hash
    h = _spec_hash(_spec())
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


# ────────────────────────────────────────────────────────────────────
# pre_dispatch_check — gates
# ────────────────────────────────────────────────────────────────────
def test_pre_dispatch_passes_happy_path(tmp_log):
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(_spec(), spec_approved=True,
                              family_hint="PROFITABILITY",
                              log_path=tmp_log)
    assert r is None


def test_pre_dispatch_refuses_not_approved(tmp_log):
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(_spec(), spec_approved=False,
                              family_hint="X", log_path=tmp_log)
    assert r is not None
    assert r.reason_code == "NOT_APPROVED"


def test_pre_dispatch_refuses_weekly_cap(tmp_log):
    """Pre-seed the log with 5 dispatches in the last week → 6th
    should be refused."""
    import datetime as _dt
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check, MAX_AUTO_DISPATCHES_PER_WEEK,
    )
    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    with tmp_log.open("w", encoding="utf-8") as f:
        for i in range(MAX_AUTO_DISPATCHES_PER_WEEK):
            f.write(json.dumps({"ts": now_iso,
                                 "dispatch_event_id": f"ev{i}"}) + "\n")
    r = pre_dispatch_check(_spec(), spec_approved=True,
                              family_hint="X", log_path=tmp_log)
    assert r is not None
    assert r.reason_code == "WEEKLY_CAP"
    assert r.metrics["week_count"] == MAX_AUTO_DISPATCHES_PER_WEEK


def test_pre_dispatch_old_dispatches_dont_count_to_weekly_cap(tmp_log):
    """Dispatches older than 7d don't count against the cap."""
    import datetime as _dt
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check, MAX_AUTO_DISPATCHES_PER_WEEK,
    )
    old_iso = (_dt.datetime.utcnow() - _dt.timedelta(days=30)
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp_log.parent.mkdir(parents=True, exist_ok=True)
    with tmp_log.open("w", encoding="utf-8") as f:
        for i in range(MAX_AUTO_DISPATCHES_PER_WEEK + 3):
            f.write(json.dumps({"ts": old_iso,
                                 "dispatch_event_id": f"ev{i}"}) + "\n")
    r = pre_dispatch_check(_spec(), spec_approved=True,
                              family_hint="X", log_path=tmp_log)
    assert r is None


def test_pre_dispatch_refuses_n_trials_hard(tmp_log, monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    monkeypatch.setattr(fd, "_family_n_trials_now",
                          lambda fam: fd.N_TRIALS_HARD)
    r = fd.pre_dispatch_check(_spec(), spec_approved=True,
                                  family_hint="PROFITABILITY",
                                  log_path=tmp_log)
    assert r is not None
    assert r.reason_code == "N_TRIALS_HARD"


def test_pre_dispatch_passes_n_trials_caution(tmp_log, monkeypatch):
    """CAUTION threshold (7) does NOT refuse — only logs a warning
    via the existing n_trials_family_counter inbox alert. Dispatcher
    only blocks at HARD (15)."""
    from engine.agents.strengthener import factor_dispatcher as fd
    monkeypatch.setattr(fd, "_family_n_trials_now",
                          lambda fam: fd.N_TRIALS_CAUTION)
    r = fd.pre_dispatch_check(_spec(), spec_approved=True,
                                  family_hint="X", log_path=tmp_log)
    assert r is None


# ────────────────────────────────────────────────────────────────────
# 2026-06-08: _family_n_trials_now rewrite — counts Tier C verdict
# events grouped by family + tier_c_auto tag (Bailey-LdP §3 within-
# family multi-testing accounting). Pre-rewrite the function read
# the unrelated workflow_executor n_trials_ledger.jsonl and always
# returned 0. L3-2 self_doubt caught it on GP/A seed dispatch.
# ────────────────────────────────────────────────────────────────────
def _ev(family, tags=("tier_c_auto",), verdict="GREEN"):
    from types import SimpleNamespace
    return SimpleNamespace(
        event_id="x", event_type="factor_verdict_filed",
        subject_id="auto_x", subject_type="factor",
        family=family, ts="2026-06-08T12:00:00Z",
        verdict=verdict, summary="", metrics={},
        parent_event_ids=(), artifacts=(), tags=tuple(tags),
        actor="t",
    )


def test_family_n_trials_now_counts_tier_c_auto_events_by_family(
    monkeypatch,
):
    """Real bug: PROFITABILITY count was 0 even with a GP/A verdict
    in the store. After rewrite, we count exactly the tier_c_auto-
    tagged factor_verdict_filed events whose family matches."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev("PROFITABILITY"),
        _ev("PROFITABILITY", verdict="RED"),
        _ev("BEHAVIORAL"),
        _ev("PROFITABILITY", tags=("strict_gate",)),       # excluded — no tier_c_auto
        _ev("CARRY"),
    ])
    assert fd._family_n_trials_now("PROFITABILITY") == 2
    assert fd._family_n_trials_now("BEHAVIORAL")    == 1
    assert fd._family_n_trials_now("CARRY")         == 1
    assert fd._family_n_trials_now("VALUE")         == 0


def test_family_n_trials_now_case_insensitive(monkeypatch):
    """Dispatcher passes MechanismFamily upper-case enum value but
    legacy events may have stored family in any case."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [
        _ev("profitability"),                 # legacy lowercase
        _ev("PROFITABILITY"),                 # new uppercase
        _ev("Profitability"),                 # title case
    ])
    assert fd._family_n_trials_now("PROFITABILITY") == 3
    assert fd._family_n_trials_now("profitability") == 3


def test_family_n_trials_now_fails_open_on_store_error(monkeypatch):
    """Research store missing / import broken → return 0 not crash.
    Gate fails open by design for brand-new system."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.research_store import store as st
    def _boom(**kw):
        raise RuntimeError("event store offline")
    monkeypatch.setattr(st, "filter_events", _boom)
    assert fd._family_n_trials_now("PROFITABILITY") == 0


def test_pre_dispatch_refuses_unknown_signal_input(tmp_log):
    """signal_inputs reference path outside PIT_CORRECT_SOURCES."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(_spec(
        signal_inputs=("crsp.msf.ret", "kaggle.scraped.weird")
    ), spec_approved=True, family_hint="X", log_path=tmp_log)
    assert r is not None
    assert r.reason_code == "SIGNAL_INPUT_UNKNOWN"
    assert "kaggle.scraped.weird" in r.metrics["violators"]


def test_pre_dispatch_escape_hatch_skips_signal_input_check(tmp_log):
    """requires_custom_code escape hatch passes the whitelist gate
    even with unwhitelisted inputs — human takes over anyway."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(_spec(
        signal_kind="requires_custom_code",
        signal_inputs=("custom.weird.data",),
    ), spec_approved=True, family_hint="X", log_path=tmp_log)
    assert r is None


def test_pre_dispatch_passes_b_class_within_range(tmp_log):
    """L2-1 Phase 2.6: FactorSpec v2 B-class params within safe
    range should pass dispatcher gate #9."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    s = _spec(universe_size=1000, n_buckets=10, signal_lookback_m=6,
                signal_skip_m=0, vol_target_annual=0.15,
                weighting_scheme_alt="vw")
    r = pre_dispatch_check(s, spec_approved=True, family_hint="X",
                              log_path=tmp_log)
    assert r is None


def test_pre_dispatch_refuses_universe_size_below_range(tmp_log):
    """L2-1 Phase 2.6: universe_size=50 outside [100, 5000]."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(universe_size=50), spec_approved=True,
        family_hint="X", log_path=tmp_log,
    )
    assert r is not None
    assert r.reason_code == "B_CLASS_OUT_OF_RANGE"
    assert r.metrics["field"] == "universe_size"


def test_pre_dispatch_refuses_n_buckets_above_range(tmp_log):
    """L2-1 Phase 2.6: n_buckets=20 outside [3, 10]
    (decile sorts are statistically thin beyond 10)."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(n_buckets=20), spec_approved=True,
        family_hint="X", log_path=tmp_log,
    )
    assert r is not None
    assert r.reason_code == "B_CLASS_OUT_OF_RANGE"


def test_pre_dispatch_refuses_vol_target_above_range(tmp_log):
    """L2-1 Phase 2.6: vol_target_annual=0.50 outside [0.03, 0.30]."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(vol_target_annual=0.50), spec_approved=True,
        family_hint="X", log_path=tmp_log,
    )
    assert r is not None
    assert r.reason_code == "B_CLASS_OUT_OF_RANGE"
    assert r.metrics["field"] == "vol_target_annual"


def test_pre_dispatch_refuses_bad_weighting_alt(tmp_log):
    """L2-1 Phase 2.6: weighting_scheme_alt must be in
    {ew, vw, rank, None}."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(weighting_scheme_alt="markowitz"), spec_approved=True,
        family_hint="X", log_path=tmp_log,
    )
    assert r is not None
    assert r.reason_code == "B_CLASS_OUT_OF_RANGE"


def test_pre_dispatch_b_class_all_none_uses_defaults(tmp_log):
    """L2-1 Phase 2.6: all B-class params None → no gate refusal
    (backward-compat with pre-v2 specs)."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    s = _spec()   # all defaults / None for v2 fields
    r = pre_dispatch_check(s, spec_approved=True, family_hint="X",
                              log_path=tmp_log)
    assert r is None


def test_pre_dispatch_refuses_unknown_signal_kind(tmp_log):
    """Defense in depth: if somehow a FactorSpec with a non-enum
    signal_kind reaches us, refuse rather than crash."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    # Build a spec then mutate signal_kind via __dict__ workaround
    # (FactorSpec is frozen; this simulates a tampered spec)
    import dataclasses as _dc
    s = _spec()
    # Use replace would only allow valid value; bypass for test:
    object.__setattr__(s, "signal_kind", "ml_hocus_pocus")
    r = pre_dispatch_check(s, spec_approved=True, family_hint="X",
                              log_path=tmp_log)
    assert r is not None
    assert r.reason_code == "UNKNOWN_SIGNAL_KIND"


# ────────────────────────────────────────────────────────────────────
# Template registry — C-2a stubs
# ────────────────────────────────────────────────────────────────────
def test_template_registry_all_signal_kinds_routable():
    """Every SIGNAL_KIND has a registered template (stub OK in
    C-2a). Guards against adding a signal_kind to extractor without
    a dispatcher route — would cause silent KeyError in prod."""
    from engine.agents.strengthener.factor_dispatcher import (
        TEMPLATE_REGISTRY,
    )
    from engine.agents.strengthener.factor_spec_extractor import (
        SIGNAL_KINDS,
    )
    for sk in SIGNAL_KINDS:
        assert sk in TEMPLATE_REGISTRY, f"signal_kind {sk} not routable"


def test_stub_template_returns_pending():
    from engine.agents.strengthener.factor_dispatcher import (
        _template_pending_build,
    )
    r = _template_pending_build(_spec())
    assert r.verdict == "PENDING_TEMPLATE_BUILD"
    assert r.template_version == "v0_stub"


def test_escape_hatch_template_returns_custom_code():
    from engine.agents.strengthener.factor_dispatcher import (
        _template_custom_code_escape,
    )
    r = _template_custom_code_escape(
        _spec(signal_kind="requires_custom_code"))
    assert r.verdict == "CUSTOM_CODE_REQUIRED"
    assert r.metrics.get("escape_hatch") is True


# ────────────────────────────────────────────────────────────────────
# dispatch_factor_spec — end-to-end happy path + refusal paths
# ────────────────────────────────────────────────────────────────────
def test_dispatch_writes_log_on_template_stub_path(tmp_log):
    """Use vrp (still stubbed in C-2e.1) to exercise the stub-template
    + audit-log path. cross_sectional_rank is no longer a stub."""
    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )
    out = dispatch_factor_spec(
        _spec(signal_kind="vrp",
              signal_inputs=("optionmetrics.standardized_options.vrp",)),
        family_hint="VOL_RISK_PREMIUM",
        spec_approved=True, log_path=tmp_log,
    )
    assert out["refusal"] is None
    assert out["template_result"]["verdict"] == "PENDING_TEMPLATE_BUILD"
    assert out["dispatch_event_id"]
    assert tmp_log.exists()
    # Confirm log row carries provenance tags
    rows = [json.loads(l) for l in
              tmp_log.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(rows) == 1
    r = rows[0]
    assert r["auto_test_spec_hash"] == out["spec_hash"]
    assert r["auto_test_llm_model"] == "claude-sonnet-4-6"
    assert r["extractor_workload"] == "strengthener_factor_spec"
    assert r["dispatcher_version"]


def test_dispatch_writes_log_on_refusal(tmp_log):
    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )
    out = dispatch_factor_spec(_spec(), family_hint="X",
                                  spec_approved=False,
                                  log_path=tmp_log)
    assert out["refusal"] is not None
    assert out["refusal"]["reason_code"] == "NOT_APPROVED"
    assert out["template_result"] is None
    # Refusal is ALSO logged (not silent)
    rows = [json.loads(l) for l in
              tmp_log.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(rows) == 1
    assert rows[0]["refusal"]["reason_code"] == "NOT_APPROVED"


def test_dispatch_dry_run_skips_log_write(tmp_log):
    """Use vrp (still stubbed in C-2e.1) so the test stays focused on
    dry_run behavior rather than picking a template-shipped signal_kind."""
    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )
    out = dispatch_factor_spec(_spec(signal_kind="vrp"),
                                  family_hint="X",
                                  spec_approved=True, dry_run=True,
                                  log_path=tmp_log)
    assert out["template_result"]["verdict"] == "PENDING_TEMPLATE_BUILD"
    assert out["dispatch_event_id"] is None
    assert not tmp_log.exists()


def test_dispatch_escape_hatch_path(tmp_log):
    from engine.agents.strengthener.factor_dispatcher import (
        dispatch_factor_spec,
    )
    out = dispatch_factor_spec(
        _spec(signal_kind="requires_custom_code",
                signal_inputs=("anything.goes",)),
        family_hint="MICROSTRUCTURE", spec_approved=True,
        log_path=tmp_log,
    )
    assert out["refusal"] is None
    assert out["template_result"]["verdict"] == "CUSTOM_CODE_REQUIRED"
    assert out["template_result"]["metrics"]["escape_hatch"] is True


def test_dispatch_template_exception_does_not_crash(tmp_log,
                                                       monkeypatch):
    """If a template raises, dispatcher catches it + emits
    EXECUTION_ERROR verdict (so the user sees what broke, instead
    of dispatcher dying silently in a cron run)."""
    from engine.agents.strengthener import factor_dispatcher as fd
    def _boom(spec):
        raise RuntimeError("data fetch failed")
    monkeypatch.setitem(fd.TEMPLATE_REGISTRY,
                          "cross_sectional_rank", _boom)
    out = fd.dispatch_factor_spec(_spec(), family_hint="X",
                                       spec_approved=True,
                                       log_path=tmp_log)
    assert out["template_result"]["verdict"] == "EXECUTION_ERROR"
    assert "data fetch failed" in \
        out["template_result"]["metrics"]["error"]


# ────────────────────────────────────────────────────────────────────
# PIT_CORRECT_SOURCES sanity
# ────────────────────────────────────────────────────────────────────
def test_pit_correct_sources_has_required_prefixes():
    """Smoke check on the whitelist — must include the data sources
    the 3 initial templates (tsmom on sector_etf, cross_sec on
    CRSP+Compustat, carry on FX G10) need. Catches accidental
    deletion or renaming."""
    from engine.agents.strengthener.factor_dispatcher import (
        PIT_CORRECT_SOURCES,
    )
    for required in ("crsp.msf.", "compustat.funda.", "etf.adj_close.",
                       "fx.spot."):
        assert required in PIT_CORRECT_SOURCES, \
            f"missing required prefix {required}"


# ────────────────────────────────────────────────────────────────────
# 2026-06-08 INTEGRATION regression: post-template wiring (L3-2
# self_doubt + verdict emit) must execute without NameError /
# AttributeError / TypeError on every emittable verdict.
#
# Bug 5c46c500 was a textbook example: self_doubt block referenced
# `fam_n` which only existed in pre_dispatch_check's local scope.
# Module-level unit tests for self_doubt module didn't catch it
# because they mocked at the assess_self_doubt boundary, not the
# dispatcher integration.
#
# These tests run the FULL dispatcher path (gates + template stub +
# self_doubt + emit) with a fake template that returns a fixed
# verdict, asserting NO Python-level errors propagate and the
# resulting `out` dict contains the expected keys.
# ────────────────────────────────────────────────────────────────────
def _wire_fake_template(monkeypatch, verdict):
    """Replace cross_sectional_rank template with a fast fake that
    returns a fixed TemplateResult — no DB, no parquet, no LLM."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    def _fake(spec):
        return TemplateResult(
            verdict          = verdict,
            summary          = f"fake {verdict} for integration regression",
            metrics          = {"sharpe": 0.5, "nw_t_stat": 2.1,
                                  "n_months": 240, "avg_turnover": 0.2,
                                  "naive_verdict": verdict,
                                  "cost_robust_verdict": verdict,
                                  "cost_stress": {}, "drawdown_naive": {},
                                  "replication": {}},
            artifacts        = {},
            template_version = "fake_v1",
        )
    monkeypatch.setitem(fd.TEMPLATE_REGISTRY,
                          "cross_sectional_rank", _fake)


@pytest.mark.parametrize("verdict", ["GREEN", "MARGINAL", "RED"])
def test_dispatch_integration_self_doubt_wiring_intact(
    tmp_log, monkeypatch, verdict,
):
    """L3-2 + emit wiring must execute without NameError /
    AttributeError for every emittable verdict. Regression for
    commit 5c46c500 (fam_n NameError).

    self_doubt module is stubbed (no real Sonnet call) — only the
    DISPATCHER integration is under test."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    _wire_fake_template(monkeypatch, verdict)

    # Stub self_doubt to return None — exercise the call path
    # WITHOUT spending Sonnet $0.04 per test. The bug was that the
    # CALL itself crashed before reaching this stub.
    called = {"n": 0}
    def _stub(spec, tr, *, family_hint, n_trials_family,
               anchor_orthogonality=None,
               subsample_stability=None,
               industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        called["n"] += 1
        # Validate the bug 5c46c500 fix: n_trials_family is an int,
        # not undefined. Will raise TypeError if dispatcher passed
        # None or NameError if scope was broken (the original bug).
        assert isinstance(n_trials_family, int)
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _stub)

    # Stub emit to avoid touching real research store from a unit
    # test. Returns a fake event_id.
    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    emit_calls = {"n": 0, "last_self_doubt": "NOT_SET"}
    def _stub_emit(spec, family_hint, tr, *,
                     dispatch_event_id=None,
                     parent_event_ids=(),
                     self_doubt=None,
                     anchor_orthogonality=None,
                     subsample_stability=None,
                     industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        emit_calls["n"] += 1
        emit_calls["last_self_doubt"] = self_doubt
        return "fake_event_id_integration"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _stub_emit)

    out = fd.dispatch_factor_spec(_spec(), family_hint="PROFITABILITY",
                                       spec_approved=True,
                                       log_path=tmp_log)

    # Wiring assertions — the actual regressions
    assert out["template_result"]["verdict"] == verdict
    assert called["n"] == 1, (
        "L3-2 assess_self_doubt was never called — wiring broken"
    )
    assert emit_calls["n"] == 1, (
        "emit_tier_c_verdict was never called — wiring broken"
    )
    # self_doubt was None (stubbed), so emit received None
    assert emit_calls["last_self_doubt"] is None
    # Dispatcher returned the emit's event_id
    assert out["verdict_event_id"] == "fake_event_id_integration"


def test_dispatch_integration_self_doubt_failure_does_not_block_emit(
    tmp_log, monkeypatch,
):
    """Graceful degradation contract: if L3-2 self_doubt raises,
    dispatcher MUST still emit the verdict (audit/research integrity
    > self_doubt observability)."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    _wire_fake_template(monkeypatch, "GREEN")

    def _crash(spec, tr, *, family_hint, n_trials_family,
                 anchor_orthogonality=None,
                 subsample_stability=None,
                 industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        raise RuntimeError("self_doubt LLM rate-limited")
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _crash)

    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    emit_calls = {"n": 0}
    def _stub_emit(*a, **kw):
        emit_calls["n"] += 1
        return "fake_eid"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _stub_emit)

    out = fd.dispatch_factor_spec(_spec(), family_hint="PROFITABILITY",
                                       spec_approved=True,
                                       log_path=tmp_log)
    assert out["template_result"]["verdict"] == "GREEN"
    assert emit_calls["n"] == 1, (
        "emit_tier_c_verdict MUST run even when self_doubt raises"
    )
    assert out["verdict_event_id"] == "fake_eid"


# ────────────────────────────────────────────────────────────────────
# L2-4 Commit 3: anchor_orthogonality wiring through dispatcher
# ────────────────────────────────────────────────────────────────────
def _wire_fake_template_with_pnl(monkeypatch, verdict):
    """Like _wire_fake_template but ALSO populates a pnl_series_df
    DataFrame in artifacts (60 months of synthetic data) so the
    anchor_orthogonality path can exercise."""
    import pandas as pd_l
    import numpy as np_l
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    rng = np_l.random.default_rng(7)
    n = 60
    idx = pd_l.date_range("2015-01-31", periods=n, freq="ME")
    pnl_df = pd_l.DataFrame({
        "pnl_gross":    rng.normal(0.005, 0.04, n),
        "pnl_net_13bp": rng.normal(0.005, 0.04, n),
        "pnl_net_80bp": rng.normal(0.003, 0.04, n),
        "turnover":     rng.uniform(0.3, 0.6, n),
    }, index=idx)

    def _fake(spec):
        return TemplateResult(
            verdict          = verdict,
            summary          = f"fake {verdict} with pnl",
            metrics          = {"sharpe": 0.5, "nw_t_stat": 2.1,
                                  "n_months": n, "naive_verdict": verdict,
                                  "cost_robust_verdict": verdict,
                                  "cost_stress": {}, "drawdown_naive": {},
                                  "replication": {}},
            artifacts        = {"pnl_series_df": pnl_df},
            template_version = "fake_with_pnl_v1",
        )
    monkeypatch.setitem(fd.TEMPLATE_REGISTRY,
                          "cross_sectional_rank", _fake)


def test_dispatch_integration_anchor_orthogonality_wired_to_self_doubt(
    tmp_log, monkeypatch,
):
    """L2-4 Commit 3: anchor_orthogonality is computed from
    template.artifacts.pnl_series_df and passed to assess_self_doubt
    as a kwarg. Verdict event metrics also include it.

    Stubs compute_for_tier_c_pnl_series so test is fast + offline."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")

    # Stub the anchor regression so tests don't need Ken French parquet
    fake_ao = {
        "alpha_monthly":  0.0026, "alpha_annual": 0.0316,
        "alpha_nw_t":     1.88,   "alpha_nw_se": 0.0014,
        "betas":          {"RMW": 0.67}, "beta_nw_t": {"RMW": 10.0},
        "r2":             0.25, "r2_adj": 0.24,
        "n_overlap":      60, "anchor_names": ["RMW"],
        "nw_lag_used":    3,  "window": "2015-01:2019-12",
        "anchor_library": "ken_french_ff5_mom",
    }
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
                          lambda series, **kw: fake_ao)

    # Stub self_doubt to spy on what it received
    spied = {}
    def _spy(spec, tr, *, family_hint, n_trials_family,
              anchor_orthogonality=None,
              subsample_stability=None,
              industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        spied["anchor_orthogonality"] = anchor_orthogonality
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    # Stub emit to spy on what it received
    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    emit_spied = {}
    def _stub_emit(spec, family_hint, tr, *,
                     dispatch_event_id=None, parent_event_ids=(),
                     self_doubt=None, anchor_orthogonality=None,
                     subsample_stability=None,
                     industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        emit_spied["anchor_orthogonality"] = anchor_orthogonality
        return "fake_eid_l2_4"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _stub_emit)

    out = fd.dispatch_factor_spec(
        _spec(), family_hint="PROFITABILITY",
        spec_approved=True, log_path=tmp_log,
    )

    # Anchor regression ran and produced the stubbed dict
    assert out.get("anchor_orthogonality") == fake_ao
    # self_doubt received it as a kwarg
    assert spied["anchor_orthogonality"] == fake_ao
    # emit received it as a kwarg
    assert emit_spied["anchor_orthogonality"] == fake_ao
    assert out["verdict_event_id"] == "fake_eid_l2_4"


def test_dispatch_integration_anchor_failure_does_not_block_self_doubt(
    tmp_log, monkeypatch,
):
    """Graceful degradation: if anchor_orthogonality raises (e.g.
    Ken French parquet missing), self_doubt still runs (with None)
    and emit still fires."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
        lambda series, **kw: (_ for _ in ()).throw(
            RuntimeError("anchor parquet missing")))

    spied = {"called": False, "anchor": "NOT_SET"}
    def _spy(spec, tr, *, family_hint, n_trials_family,
              anchor_orthogonality=None,
              subsample_stability=None,
              industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        spied["called"] = True
        spied["anchor"] = anchor_orthogonality
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict",
                          lambda *a, **kw: "eid_resilient")

    out = fd.dispatch_factor_spec(
        _spec(), family_hint="X", spec_approved=True, log_path=tmp_log,
    )
    assert spied["called"] is True
    assert spied["anchor"] is None
    assert out["verdict_event_id"] == "eid_resilient"


def test_dispatch_integration_subsample_stability_wired_to_self_doubt(
    tmp_log, monkeypatch,
):
    """L2-5 Commit 2: subsample_stability computed from
    template.artifacts.pnl_series_df and passed to assess_self_doubt
    + emit. Stubs subsample compute so test is fast + offline."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod
    from engine.research import subsample_stability as ss_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
                          lambda series, **kw: None)

    fake_ss = {
        "n_splits": 4, "n_total_months": 60,
        "windows": [{"start": "2015-01", "end": "2019-12",
                       "n_months": 60, "sharpe_ann": 0.5,
                       "nw_t_stat": 1.2, "ann_return": 0.05,
                       "ann_vol": 0.10}],
        "worst_best_sharpe_ratio": 0.85,
        "institutional_stable": True,
        "monotone_decay": False, "monotone_growth": False,
        "decay_slope_per_year": 0.0, "decay_slope_t": 0.0,
    }
    monkeypatch.setattr(ss_mod, "compute_for_tier_c_pnl_series",
                          lambda df, **kw: fake_ss)

    spied = {}
    def _spy(spec, tr, *, family_hint, n_trials_family,
              anchor_orthogonality=None, subsample_stability=None,
              industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        spied["subsample_stability"] = subsample_stability
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    emit_spied = {}
    def _stub_emit(spec, family_hint, tr, *,
                     dispatch_event_id=None, parent_event_ids=(),
                     self_doubt=None, anchor_orthogonality=None,
                     subsample_stability=None,
                     industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        emit_spied["subsample_stability"] = subsample_stability
        return "fake_eid_l2_5"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _stub_emit)

    out = fd.dispatch_factor_spec(
        _spec(), family_hint="PROFITABILITY",
        spec_approved=True, log_path=tmp_log,
    )

    assert out.get("subsample_stability") == fake_ss
    assert spied["subsample_stability"] == fake_ss
    assert emit_spied["subsample_stability"] == fake_ss


def test_dispatch_integration_subsample_failure_does_not_block_emit(
    tmp_log, monkeypatch,
):
    """Graceful degradation: subsample compute raises → emit still
    fires with subsample_stability=None."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod
    from engine.research import subsample_stability as ss_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: None)
    monkeypatch.setattr(ss_mod, "compute_for_tier_c_pnl_series",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("subsample boom")))

    spied = {"ss": "NOT_SET"}
    def _spy(spec, tr, *, family_hint, n_trials_family,
              anchor_orthogonality=None, subsample_stability=None,
              industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        spied["ss"] = subsample_stability
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict",
                          lambda *a, **kw: "eid_resilient_ss")

    out = fd.dispatch_factor_spec(
        _spec(), family_hint="X", spec_approved=True, log_path=tmp_log,
    )
    assert spied["ss"] is None
    assert out["verdict_event_id"] == "eid_resilient_ss"


def test_dispatch_integration_industry_extension_wired_to_self_doubt(
    tmp_log, monkeypatch,
):
    """L2-6 Commit 3: industry_extension computed from
    template.artifacts.pnl_series_df + stage1 anchor result,
    passed to assess_self_doubt + emit. Stubs compute functions
    so test is fast + offline."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod
    from engine.research import industry_attribution as ix_mod
    from engine.research import subsample_stability as ss_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")

    # Stub Stage 1 anchor to return a non-None dict (needed for
    # industry_extension wiring guard)
    fake_ao = {
        "alpha_monthly": 0.005, "alpha_annual": 0.06,
        "alpha_nw_t": 2.5, "alpha_nw_se": 0.002,
        "betas": {"RMW": 0.3}, "beta_nw_t": {"RMW": 4.0},
        "r2": 0.25, "r2_adj": 0.24, "n_overlap": 60,
        "anchor_names": ["RMW"], "nw_lag_used": 3,
        "window": "2015-01:2019-12",
        "anchor_library": "ken_french_ff5_mom",
    }
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: fake_ao)
    monkeypatch.setattr(ss_mod, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: None)

    fake_ix = {
        "alpha_full_monthly": 0.004, "alpha_full_annual": 0.048,
        "alpha_full_nw_t": 2.2, "alpha_full_nw_se": 0.0018,
        "alpha_ff5mom_only_nw_t": 2.5,
        "delta_alpha_monthly": 0.001, "delta_alpha_nw_t_approx": 0.3,
        "ff5mom_betas": {"RMW": 0.32}, "ff5mom_beta_nw_t": {"RMW": 4.2},
        "industry_betas": {"NoDur": 0.1, "BusEq": 0.15},
        "industry_beta_nw_t": {"NoDur": 1.5, "BusEq": 2.0},
        "r2_full": 0.45, "r2_adj_full": 0.43, "n_overlap": 60,
        "industry_names": ["NoDur", "BusEq"], "nw_lag_used": 3,
        "window": "2015-01:2019-12",
        "industry_joint_f_test": {"f_stat": 5.0, "f_pvalue": 0.001,
                                     "df_num": 12, "df_denom": 41},
        "industry_snapshot_sha": "abc123",
        "model_form": "joint_ff5mom_plus_12_industry",
    }
    monkeypatch.setattr(ix_mod, "compute_for_tier_c_with_stage1_residual",
                          lambda *a, **kw: fake_ix)

    spied = {}
    def _spy(spec, tr, *, family_hint, n_trials_family,
              anchor_orthogonality=None, subsample_stability=None,
              industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        spied["industry_extension"] = industry_extension
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    emit_spied = {}
    def _stub_emit(spec, family_hint, tr, *,
                     dispatch_event_id=None, parent_event_ids=(),
                     self_doubt=None, anchor_orthogonality=None,
                     subsample_stability=None,
                     industry_extension=None,
              cross_asset_extension=None,
              routing_decisions=None,
              **_kwargs):
        emit_spied["industry_extension"] = industry_extension
        return "fake_eid_l2_6"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _stub_emit)

    out = fd.dispatch_factor_spec(
        _spec(), family_hint="PROFITABILITY",
        spec_approved=True, log_path=tmp_log,
    )

    assert out.get("industry_extension") == fake_ix
    assert spied["industry_extension"] == fake_ix
    assert emit_spied["industry_extension"] == fake_ix


def test_dispatch_integration_industry_extension_skipped_when_stage1_missing(
    tmp_log, monkeypatch,
):
    """If Stage 1 anchor regression returned None (no anchor parquet
    cached, or rank-deficient), industry_extension wiring MUST be
    skipped — no point computing Δα when there's no baseline α."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    from engine.research import anchor_regression as ar_mod
    from engine.research import industry_attribution as ix_mod
    from engine.research import subsample_stability as ss_mod

    _wire_fake_template_with_pnl(monkeypatch, "GREEN")
    monkeypatch.setattr(ar_mod, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: None)   # Stage 1 missing
    monkeypatch.setattr(ss_mod, "compute_for_tier_c_pnl_series",
                          lambda *a, **kw: None)

    called = {"n": 0}
    def _ix_compute(*a, **kw):
        called["n"] += 1
        return None
    monkeypatch.setattr(ix_mod, "compute_for_tier_c_with_stage1_residual",
                          _ix_compute)

    monkeypatch.setattr(sd_mod, "assess_self_doubt",
                          lambda *a, **kw: None)
    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict",
                          lambda *a, **kw: "eid")

    fd.dispatch_factor_spec(
        _spec(), family_hint="X", spec_approved=True, log_path=tmp_log,
    )
    assert called["n"] == 0, (
        "industry_extension MUST NOT compute when Stage 1 is missing"
    )


def test_dispatch_integration_skips_self_doubt_on_internal_verdict(
    tmp_log, monkeypatch,
):
    """Non-emittable verdicts (EXECUTION_ERROR, INSUFFICIENT_HISTORY,
    PENDING_TEMPLATE_BUILD) MUST NOT call self_doubt — wastes Sonnet
    $0.04 + would emit_tier_c_verdict short-circuit anyway."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import self_doubt as sd_mod
    _wire_fake_template(monkeypatch, "EXECUTION_ERROR")

    called = {"n": 0}
    def _spy(*a, **kw):
        called["n"] += 1
        return None
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy)

    out = fd.dispatch_factor_spec(_spec(), family_hint="X",
                                       spec_approved=True,
                                       log_path=tmp_log)
    assert out["template_result"]["verdict"] == "EXECUTION_ERROR"
    assert called["n"] == 0, (
        "self_doubt MUST NOT run on non-emittable verdicts"
    )
    assert "anchor_orthogonality" not in out, (
        "anchor_orthogonality MUST NOT run on non-emittable verdicts"
    )
