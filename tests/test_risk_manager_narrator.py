"""tests/test_risk_manager_narrator.py — Phase 9 narrator unit tests.

Covers:
  - 12 deterministic templates (one per mode, exhaustive)
  - Banned-phrase regex catches hedge language
  - Banned-phrase regex passes active-voice prose
  - PersonaContext forward-compatibility stub
  - Backend selection via env var + explicit param
  - GeminiFlashNarrator stub raises explicitly (no silent fallback)
  - DeterministicNarrator template output has no banned phrases
"""
from __future__ import annotations

import os
import pytest

from engine.agents.risk_manager.gates import Breach
from engine.agents.risk_manager.narrator import (
    BANNED_PHRASES,
    DeterministicNarrator,
    GeminiFlashNarrator,
    NarrationResult,
    PersonaContext,
    contains_banned_phrase,
    narrate_breach,
    _select_backend,
)


# ──────────────────────────────────────────────────────────────────────────────
# Banned-phrase regex coverage — must catch hedging vocabulary
# ──────────────────────────────────────────────────────────────────────────────
class TestBannedPhrases:
    @pytest.mark.parametrize("phrase", [
        "maybe", "perhaps", "could be", "might be", "probably", "possibly",
        "likely", "I think", "I feel", "seems to", "appears to",
        "just a thought", "you might want to", "consider",
    ])
    def test_banned_phrase_caught(self, phrase):
        text = f"This {phrase} should not slip through the narrator."
        assert contains_banned_phrase(text) is not None

    @pytest.mark.parametrize("text", [
        "Halt issued. Re-evaluate position sizing.",
        "Reduce position magnitudes before re-submission.",
        "Investigate the originating strategy's signal path.",
        "Book persisted. Monitor for further deterioration.",
        "Trim the largest weights or expand the active universe.",
        "Strategy availability degraded.",
    ])
    def test_active_voice_passes(self, text):
        assert contains_banned_phrase(text) is None

    def test_case_insensitive(self):
        # Banned phrases are case-insensitive
        assert contains_banned_phrase("MAYBE this") is not None
        assert contains_banned_phrase("I Think") is not None


# ──────────────────────────────────────────────────────────────────────────────
# DeterministicNarrator — 12 templates, all banned-phrase clean
# ──────────────────────────────────────────────────────────────────────────────
class TestDeterministicTemplates:
    @pytest.fixture
    def narrator(self):
        return DeterministicNarrator()

    @pytest.mark.parametrize("mode_id,severity,observed,threshold", [
        ("1",  "HARD_HALT", 0.07, 0.05),
        ("2",  "SOFT_WARN", 0.50, 0.10),
        ("3",  "HARD_HALT", 1.80, 1.60),
        ("4",  "HARD_HALT", 1.60, 1.50),
        ("5",  "HARD_HALT", 0.30, 0.25),
        ("6",  "SOFT_WARN", -0.04, -0.03),
        ("6b", "HARD_HALT", -0.12, -0.09),
        ("7",  "SOFT_WARN", -0.06, -0.05),
        ("7b", "HARD_HALT", -0.20, -0.15),
        ("8",  "SOFT_WARN", 0.55, 0.50),
        ("9",  "HARD_HALT", 1.0,  3.0),
        ("10", "SOFT_WARN", 6.0,  5.0),
    ])
    def test_template_renders_clean(self, narrator, mode_id, severity, observed, threshold):
        """Each of 12 modes produces banned-phrase-clean prose."""
        breach = Breach(
            mode_id=mode_id, severity=severity, rule_description="sample",
            observed_value=observed, threshold=threshold, affected=("AAPL",),
            extra={"n_strategies_total": 5}, spec_anchor="spec id=69 §2.1",
        )
        result = narrator.generate(breach, PersonaContext())
        # Tone enforcement
        assert contains_banned_phrase(result.text) is None, (
            f"mode {mode_id} template contains banned phrase: {result.text}"
        )
        # Zero cost
        assert result.cost_usd == 0.0
        # Sensible length
        assert 30 <= len(result.text) <= 500

    def test_fallback_template_for_unknown_mode(self, narrator):
        """Unknown mode_id falls back to a generic template (banned-clean)."""
        breach = Breach(
            mode_id="999", severity="HARD_HALT", rule_description="hypothetical",
            observed_value=1.0, threshold=0.0, affected=(),
            extra={}, spec_anchor="spec id=69 §future",
        )
        result = narrator.generate(breach, PersonaContext())
        assert contains_banned_phrase(result.text) is None
        assert "999" in result.text   # falls back to mode_id reference

    def test_backend_name(self, narrator):
        assert narrator.name == "deterministic"


# ──────────────────────────────────────────────────────────────────────────────
# GeminiFlashNarrator stub — must raise NotImplementedError explicitly
# ──────────────────────────────────────────────────────────────────────────────
class TestGeminiFlashStub:
    def test_raises_not_implemented(self):
        breach = Breach("1", "HARD_HALT", "", 0, 0, (), {}, "s")
        with pytest.raises(NotImplementedError):
            GeminiFlashNarrator().generate(breach, PersonaContext())


# ──────────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────────
class TestBackendSelection:
    def test_default_is_deterministic(self):
        # Clear env var
        os.environ.pop("RISK_MANAGER_NARRATOR_BACKEND", None)
        backend = _select_backend()
        assert backend.name == "deterministic"

    def test_env_var_override(self):
        os.environ["RISK_MANAGER_NARRATOR_BACKEND"] = "gemini_flash"
        try:
            backend = _select_backend()
            assert backend.name == "gemini_flash"
        finally:
            os.environ.pop("RISK_MANAGER_NARRATOR_BACKEND", None)

    def test_explicit_param_wins(self):
        backend = _select_backend("deterministic")
        assert backend.name == "deterministic"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown narrator backend"):
            _select_backend("nonsense_backend")


# ──────────────────────────────────────────────────────────────────────────────
# Public API — narrate_breach()
# ──────────────────────────────────────────────────────────────────────────────
class TestPublicAPI:
    def test_narrate_breach_returns_result(self):
        b = Breach("1", "HARD_HALT", "test", 0.06, 0.05, ("AAPL",), {}, "s")
        r = narrate_breach(b)
        assert isinstance(r, NarrationResult)
        assert r.backend == "deterministic"
        assert r.cost_usd == 0.0
        assert contains_banned_phrase(r.text) is None

    def test_persona_context_optional(self):
        # narrate_breach without explicit context — uses default PersonaContext
        b = Breach("1", "HARD_HALT", "", 0.06, 0.05, ("AAPL",), {}, "s")
        r1 = narrate_breach(b)
        r2 = narrate_breach(b, context=PersonaContext())
        # Default context produces identical output to explicit default
        assert r1.text == r2.text


# ──────────────────────────────────────────────────────────────────────────────
# PersonaContext forward-compatibility stub
# ──────────────────────────────────────────────────────────────────────────────
class TestPersonaContext:
    def test_frozen(self):
        ctx = PersonaContext()
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.role_id = "different"  # type: ignore[misc]

    def test_default_role(self):
        ctx = PersonaContext()
        assert ctx.role_id == "head_of_risk_blackrock_slack"

    def test_extension_fields_present_but_empty(self):
        """Phase 7 ships the fields; Persona Voice sprint populates them."""
        ctx = PersonaContext()
        assert ctx.voice_phrase_library is None
        assert ctx.cross_agent_references == ()
        assert ctx.episodic_memory_hits == ()


# ──────────────────────────────────────────────────────────────────────────────
# Banned-phrase list sanity
# ──────────────────────────────────────────────────────────────────────────────
class TestBannedPhrasesList:
    def test_minimum_coverage(self):
        """Must include the canonical hedging vocabulary."""
        joined = " ".join(BANNED_PHRASES).lower()
        for canonical in ["maybe", "perhaps", "might", "probably",
                          "i think", "i feel", "appears", "consider"]:
            assert canonical in joined, f"missing canonical banned phrase {canonical!r}"
