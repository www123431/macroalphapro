"""tests/test_dq_narrator.py — Phase 7 DQ narrator tests.

Mirrors tests/test_risk_manager_narrator.py 1:1 with DQ-specific
mode IDs (1/2/3/4/5/6/7/8/9/10a/10b vs RM's 1/2/3/.../10).

Per spec id=70 §2.5 + Phase 7 build plan.
"""
from __future__ import annotations

import datetime
from unittest.mock import patch

import pytest

from engine.agents.dq_inspector.gates import Breach
from engine.agents.dq_inspector.narrator import (
    BANNED_PHRASES,
    DeterministicNarrator,
    GeminiFlashNarrator,
    NarrationResult,
    PersonaContext,
    contains_banned_phrase,
    narrate_breach,
    narrate_run_result,
    _select_backend,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — one Breach per mode
# ──────────────────────────────────────────────────────────────────────────────
def _make_breach(mode_id: str, **extra) -> Breach:
    """Synthesize a Breach for a specific mode with sensible defaults."""
    defaults = {
        "1":   ("HARD_HALT", "FRED 'CPIAUCSL' stale", 100.0, 30.0, ("fred:CPIAUCSL",),
                {"source_id": "fred:CPIAUCSL", "last_obs_date": "2026-04-01"}),
        "2":   ("HARD_HALT", "bab_compat cache stale", 5.0, 1.0, ("engine.factors.bab_compat",),
                {"source_id": "yfinance:bab_compat_cache", "mtime": "2026-05-14"}),
        "3":   ("SOFT_WARN", "D-PEAD panel stale", 75.0, 60.0, ("engine.path_c.dhs panel parquet",),
                {"source_id": "internal_parquet:pead_panel", "mtime": "2026-03-01"}),
        "4":   ("SOFT_WARN", "SP500 feed stale", 45.0, 30.0, ("engine.data_sources.sp500_announcements",),
                {"source_id": "wikipedia+edgar:sp500_feed", "last_detected_at": "2026-04-04"}),
        "5":   ("HARD_HALT", "K1 coverage degraded", 0.70, 0.90, ("k1_universe",),
                {"source_id": "universe:k1_universe", "expected_n": 43}),
        "6":   ("HARD_HALT", "D-PEAD coverage degraded", 0.75, 0.80, ("pead_universe",),
                {"source_id": "universe:pead_universe", "expected_n": 1500}),
        "7":   ("HARD_HALT", "Anomaly", 0.35, 0.30, ("SPY",),
                {"source_id": "price_anomaly:SPY", "signed_return": 0.35, "ticker_class": "etf"}),
        "8":   ("SOFT_WARN", "Volume dropoff", 0.05, 0.10, ("XYZ",),
                {"source_id": "volume:XYZ"}),
        "9":   ("HARD_HALT", "NaN burst", 0.12, 0.05, (),
                {"n_nan": 5, "n_universe": 42}),
        "10a": ("SOFT_WARN", "Row count drop moderate", 0.25, 0.20, ("PaperTradeStrategyLog",),
                {"today_rows": 3, "yesterday_rows": 4, "tier": "10a_moderate"}),
        "10b": ("HARD_HALT", "Row count catastrophic", 0.60, 0.50, ("PaperTradeStrategyLog",),
                {"today_rows": 2, "yesterday_rows": 5, "tier": "10b_catastrophic"}),
    }
    sev, rule, obs, thr, aff, ext = defaults[mode_id]
    ext = {**ext, **extra}
    return Breach(
        mode_id          = mode_id,
        severity         = sev,
        rule_description = rule,
        observed_value   = obs,
        threshold        = thr,
        affected         = aff,
        extra            = ext,
        spec_anchor      = f"spec id=70 §2.1 Mode {mode_id}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Banned-phrases enforcement
# ──────────────────────────────────────────────────────────────────────────────
class TestBannedPhrases:
    def test_clean_text_returns_none(self):
        assert contains_banned_phrase("Cache stale — refresh before re-submit.") is None

    @pytest.mark.parametrize("hedge", [
        "maybe", "perhaps", "could be", "might be", "probably",
        "possibly", "likely", "I think", "seems to",
    ])
    def test_hedging_detected(self, hedge):
        text = f"This {hedge} a problem."
        result = contains_banned_phrase(text)
        assert result is not None
        assert hedge.lower() in result.lower()

    def test_case_insensitive(self):
        assert contains_banned_phrase("MAYBE we should investigate") is not None


# ──────────────────────────────────────────────────────────────────────────────
# DeterministicNarrator template coverage
# ──────────────────────────────────────────────────────────────────────────────
class TestDeterministicNarrator:
    @pytest.mark.parametrize("mode_id", [
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10a", "10b",
    ])
    def test_every_dq_mode_has_template(self, mode_id):
        """Each of the 11 DQ mode IDs must produce a non-empty narration
        from the template path (not the fallback)."""
        breach = _make_breach(mode_id)
        result = DeterministicNarrator().generate(breach, PersonaContext())
        assert isinstance(result, NarrationResult)
        assert result.text
        assert result.backend == "deterministic"
        assert result.cost_usd == 0.0

    def test_no_template_uses_banned_phrase(self):
        """Every template's output for every mode must pass banned-phrase
        check. Authoring guard — catches hedging slipping into templates."""
        for mode_id in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10a", "10b"]:
            breach = _make_breach(mode_id)
            result = DeterministicNarrator().generate(breach, PersonaContext())
            bad = contains_banned_phrase(result.text)
            assert bad is None, (
                f"Mode {mode_id!r} template produced banned phrase {bad!r}: "
                f"{result.text!r}"
            )

    def test_unknown_mode_falls_back(self):
        """An unknown mode_id should produce the fallback template."""
        breach = Breach(
            mode_id="99_unknown", severity="HARD_HALT",
            rule_description="hypothetical future mode", observed_value=0.0,
            threshold=0.0, affected=(), extra={},
            spec_anchor="spec id=70 §future",
        )
        result = DeterministicNarrator().generate(breach, PersonaContext())
        assert "99_unknown" in result.text
        assert "hypothetical future mode" in result.text

    def test_mode_1_renders_observed_threshold(self):
        breach = _make_breach("1")
        result = DeterministicNarrator().generate(breach, PersonaContext())
        assert "100" in result.text     # observed days stale
        assert "30" in result.text       # threshold

    def test_mode_5_renders_coverage_percentages(self):
        breach = _make_breach("5")
        result = DeterministicNarrator().generate(breach, PersonaContext())
        assert "70" in result.text          # observed_value 0.70 as %
        assert "90" in result.text          # threshold 0.90 as %

    def test_mode_10b_distinguishes_from_10a(self):
        b_10a = _make_breach("10a")
        b_10b = _make_breach("10b")
        t_a = DeterministicNarrator().generate(b_10a, PersonaContext()).text
        t_b = DeterministicNarrator().generate(b_10b, PersonaContext()).text
        # Soft vs catastrophic phrasing should diverge
        assert "soft" in t_a.lower() or "warn" in t_a.lower() or "moderate" in t_a.lower()
        assert "catastrophic" in t_b.lower() or "halt" in t_b.lower() or "severe" in t_b.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────────
class TestBackendSelection:
    def test_default_is_deterministic(self, monkeypatch):
        monkeypatch.delenv("DQ_INSPECTOR_NARRATOR_BACKEND", raising=False)
        assert _select_backend().name == "deterministic"

    def test_env_var_switches(self, monkeypatch):
        monkeypatch.setenv("DQ_INSPECTOR_NARRATOR_BACKEND", "gemini_flash")
        backend = _select_backend()
        assert backend.name == "gemini_flash"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown DQ narrator backend"):
            _select_backend("xyz_not_a_backend")

    def test_gemini_flash_raises_not_implemented(self):
        breach = _make_breach("1")
        with pytest.raises(NotImplementedError, match="deferred"):
            GeminiFlashNarrator().generate(breach, PersonaContext())


# ──────────────────────────────────────────────────────────────────────────────
# narrate_breach / narrate_run_result façade
# ──────────────────────────────────────────────────────────────────────────────
class TestPublicAPI:
    def test_narrate_breach_returns_result(self):
        breach = _make_breach("2")
        result = narrate_breach(breach)
        assert isinstance(result, NarrationResult)
        assert "bab_compat" in result.text

    def test_narrate_run_result_empty_breaches(self):
        from engine.agents.dq_inspector.agent import DQInspectorRunResult
        empty_result = DQInspectorRunResult(
            started_at_iso  = "2026-05-19T06:00:00",
            finished_at_iso = "2026-05-19T06:00:01",
            today_iso       = "2026-05-19",
            phase           = "pre_batch",
            dry_run         = True,
            n_modes_evaluated = 4,
            breaches        = (),
            halt            = False,
            severity        = "NONE",
            narratives      = (),
            llm_cost_usd    = 0.0,
            audit_alert_ids = (),
        )
        assert narrate_run_result(empty_result) == []

    def test_narrate_run_result_with_alert_ids_no_db_when_flag_off(self):
        """When audit_alert_ids is populated, breaches are narrated;
        update_db=False prevents DB write. Mirrors RM narrator contract
        (narrate_run_result requires alert_ids to iterate; dry_run runs
        produce no alert_ids hence no narratives — caller should call
        narrate_breach directly for dry-run preview)."""
        from engine.agents.dq_inspector.agent import DQInspectorRunResult
        breach = _make_breach("1")
        run_result = DQInspectorRunResult(
            started_at_iso    = "2026-05-19T06:00:00",
            finished_at_iso   = "2026-05-19T06:00:01",
            today_iso         = "2026-05-19",
            phase             = "pre_batch",
            dry_run           = False,
            n_modes_evaluated = 4,
            breaches          = (breach,),
            halt              = True,
            severity          = "SEVERE",
            narratives        = (),
            llm_cost_usd      = 0.0,
            audit_alert_ids   = ("alert-id-1",),
        )
        results = narrate_run_result(run_result, update_db=False)
        assert len(results) == 1
        assert results[0].cost_usd == 0.0

    def test_narrate_run_result_no_alert_ids_skips_iteration(self):
        """dry_run with no persisted alerts → no narratives (matches RM
        contract; caller should use narrate_breach for preview)."""
        from engine.agents.dq_inspector.agent import DQInspectorRunResult
        breach = _make_breach("1")
        run_result = DQInspectorRunResult(
            started_at_iso    = "2026-05-19T06:00:00",
            finished_at_iso   = "2026-05-19T06:00:01",
            today_iso         = "2026-05-19",
            phase             = "pre_batch",
            dry_run           = True,
            n_modes_evaluated = 4,
            breaches          = (breach,),
            halt              = True,
            severity          = "SEVERE",
            narratives        = (),
            llm_cost_usd      = 0.0,
            audit_alert_ids   = (),
        )
        # Caller wanting dry-run preview should call narrate_breach directly:
        preview = narrate_breach(breach)
        assert preview.text
        # narrate_run_result without alert_ids produces no narrations
        results = narrate_run_result(run_result)
        assert results == []
