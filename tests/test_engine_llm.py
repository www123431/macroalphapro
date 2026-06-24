"""tests/test_engine_llm.py — engine.llm wrapper tests.

Mocks the underlying provider call so we don't burn live API quota in
CI. One opt-in live ping test (skip by default; runs only with the
ANTHROPIC_LIVE_TEST=1 env var) verifies the SDK wiring.
"""
from __future__ import annotations

import dataclasses
import os
from unittest.mock import MagicMock, patch

import pytest

from engine.llm.call import LLMCallResult, ToolCall, _WORKLOAD_ROUTING, call
from engine.llm.pricing import compute_cost, supported_models
from engine.llm.providers.anthropic_provider import _RawCallResult


# ──────────────────────────────────────────────────────────────────────────────
# Workload routing
# ──────────────────────────────────────────────────────────────────────────────
class TestWorkloadRouting:
    def test_narrator_routes_to_anthropic_haiku(self):
        assert _WORKLOAD_ROUTING["narrator"] == ("anthropic", "claude-haiku-4-5")

    def test_rm_agent_routes_to_anthropic_sonnet(self):
        assert _WORKLOAD_ROUTING["rm_agent"] == ("anthropic", "claude-sonnet-4-6")

    def test_devils_advocate_routes_to_deepseek(self):
        assert _WORKLOAD_ROUTING["devils_advocate"] == ("deepseek", "deepseek-v4-pro")

    def test_unknown_workload_raises(self):
        with pytest.raises(ValueError, match="unknown workload"):
            call(workload="xyz_not_a_workload", system="", user="", agent_id="risk_manager")


# ──────────────────────────────────────────────────────────────────────────────
# Pricing
# ──────────────────────────────────────────────────────────────────────────────
class TestPricing:
    def test_haiku_4_5_basic(self):
        # 1M in + 1M out = $1 + $5 = $6
        cost = compute_cost(model="claude-haiku-4-5",
                            input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 6.0) < 1e-9

    def test_sonnet_4_6_basic(self):
        # 1M in + 1M out = $3 + $15 = $18
        cost = compute_cost(model="claude-sonnet-4-6",
                            input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - 18.0) < 1e-9

    def test_cache_read_at_10pct_input(self):
        # 1M cache_read on Haiku = $0.10 (vs $1 uncached)
        cost = compute_cost(model="claude-haiku-4-5",
                            input_tokens=0, output_tokens=0,
                            cache_read_tokens=1_000_000)
        assert abs(cost - 0.10) < 1e-9

    def test_cache_write_at_125pct_input(self):
        cost = compute_cost(model="claude-haiku-4-5",
                            input_tokens=0, output_tokens=0,
                            cache_write_tokens=1_000_000)
        assert abs(cost - 1.25) < 1e-9

    def test_deepseek_v4_pro_promo(self):
        # Promo: $0.435 in / $0.87 out per 1M
        cost = compute_cost(model="deepseek-v4-pro",
                            input_tokens=1_000_000, output_tokens=1_000_000,
                            use_promo=True)
        assert abs(cost - (0.435 + 0.87)) < 1e-9

    def test_deepseek_v4_pro_list_rates(self):
        # Post-promo: $1.74 / $3.48
        cost = compute_cost(model="deepseek-v4-pro",
                            input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(cost - (1.74 + 3.48)) < 1e-9

    def test_unknown_model_returns_zero(self):
        cost = compute_cost(model="not-a-real-model",
                            input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_supported_models_includes_all_workload_targets(self):
        models = set(supported_models())
        for _, model in _WORKLOAD_ROUTING.values():
            assert model in models, f"{model!r} from workload routing missing in pricing"


# ──────────────────────────────────────────────────────────────────────────────
# Provider dispatch (mocked — no live API)
# ──────────────────────────────────────────────────────────────────────────────
class TestCallDispatch:
    def _fake_raw(self, **overrides) -> _RawCallResult:
        defaults = dict(
            text="ok", tool_calls=[], stop_reason="end_turn",
            model="claude-haiku-4-5-20251001",
            input_tokens=10, output_tokens=20,
            cache_read_tokens=0, cache_write_tokens=0,
            latency_ms=500,
            raw_usage={"input_tokens": 10, "output_tokens": 20},
        )
        defaults.update(overrides)
        return _RawCallResult(**defaults)

    def test_narrator_dispatches_to_anthropic_adapter(self):
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=self._fake_raw()) as mock_anth:
            result = call(workload="narrator", system="sys", user="hi",
                          agent_id="risk_manager", record_cost=False)
        mock_anth.assert_called_once()
        assert result.provider == "anthropic"
        assert result.text == "ok"

    def test_rm_agent_dispatches_to_anthropic_with_sonnet(self):
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=self._fake_raw(model="claude-sonnet-4-6-X")
                   ) as mock_anth:
            call(workload="rm_agent", system="sys", user="hi",
                 agent_id="risk_manager", record_cost=False)
        # Anthropic adapter called with claude-sonnet-4-6 (the routing model,
        # not the response model)
        call_kwargs = mock_anth.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_devils_advocate_dispatches_to_deepseek_v4_pro(self):
        """DeepSeek provider wired 2026-05-19 — dispatch must reach it."""
        from engine.llm.providers.anthropic_provider import _RawCallResult
        fake_raw = _RawCallResult(
            text="critique", tool_calls=[], stop_reason="end_turn",
            model="deepseek-v4-pro", input_tokens=10, output_tokens=20,
            cache_read_tokens=0, cache_write_tokens=0, latency_ms=500,
            raw_usage={},
        )
        with patch("engine.llm.providers.deepseek_provider.call_deepseek",
                   return_value=fake_raw) as mock_ds:
            result = call(workload="devils_advocate", system="s", user="u",
                          agent_id="devils_advocate", record_cost=False)
        mock_ds.assert_called_once()
        assert result.provider == "deepseek"
        call_kwargs = mock_ds.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-v4-pro"

    def test_tool_calls_propagate(self):
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=self._fake_raw(
                       tool_calls=[{"id": "toolu_1", "name": "query_alerts",
                                    "input": {"days": 7}}],
                       stop_reason="tool_use",
                   )):
            result = call(workload="rm_agent", system="s", user="u",
                          agent_id="risk_manager", record_cost=False)
        assert len(result.tool_calls) == 1
        assert isinstance(result.tool_calls[0], ToolCall)
        assert result.tool_calls[0].name == "query_alerts"
        assert result.tool_calls[0].input == {"days": 7}
        assert result.stop_reason == "tool_use"


# ──────────────────────────────────────────────────────────────────────────────
# Cost ledger integration
# ──────────────────────────────────────────────────────────────────────────────
class TestCostLedgerIntegration:
    def test_successful_call_records_to_ledger(self, monkeypatch, tmp_path):
        # Isolate ledger to tmp file
        ledger_file = tmp_path / "test_ledger.jsonl"
        lock_file = tmp_path / "test_ledger.jsonl.lock"
        monkeypatch.setattr("engine.llm_cost_ledger._LEDGER_PATH", ledger_file)
        monkeypatch.setattr("engine.llm_cost_ledger._LOCK_PATH", lock_file)

        fake_raw = _RawCallResult(
            text="x", tool_calls=[], stop_reason="end_turn",
            model="claude-haiku-4-5-X", input_tokens=100, output_tokens=50,
            cache_read_tokens=0, cache_write_tokens=0, latency_ms=300,
            raw_usage={},
        )
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=fake_raw):
            result = call(workload="narrator", system="s", user="u",
                          agent_id="risk_manager", record_cost=True)

        # Ledger should have 1 entry
        from engine.llm_cost_ledger import get_calls
        entries = list(get_calls())
        assert len(entries) == 1
        e = entries[0]
        assert e.agent_id == "risk_manager"
        assert e.provider == "anthropic"
        assert e.model == "claude-haiku-4-5"   # routing model, not response model
        assert e.cost_usd > 0
        assert e.extra["workload"] == "narrator"

    def test_ledger_failure_does_not_break_call(self, monkeypatch):
        # Force ledger write to fail
        def _boom(**kwargs):
            raise RuntimeError("ledger disk full")
        monkeypatch.setattr("engine.llm_cost_ledger.record_call", _boom)

        fake_raw = _RawCallResult(
            text="alive", tool_calls=[], stop_reason="end_turn",
            model="x", input_tokens=1, output_tokens=1,
            cache_read_tokens=0, cache_write_tokens=0, latency_ms=1,
            raw_usage={},
        )
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=fake_raw):
            result = call(workload="narrator", system="s", user="u",
                          agent_id="risk_manager", record_cost=True)
        # Call still returns successfully
        assert result.text == "alive"

    def test_invalid_agent_id_caught_at_ledger(self, monkeypatch, tmp_path):
        ledger_file = tmp_path / "ledger.jsonl"
        lock_file = tmp_path / "ledger.jsonl.lock"
        monkeypatch.setattr("engine.llm_cost_ledger._LEDGER_PATH", ledger_file)
        monkeypatch.setattr("engine.llm_cost_ledger._LOCK_PATH", lock_file)

        fake_raw = _RawCallResult(
            text="x", tool_calls=[], stop_reason="end_turn",
            model="x", input_tokens=1, output_tokens=1,
            cache_read_tokens=0, cache_write_tokens=0, latency_ms=1,
            raw_usage={},
        )
        with patch("engine.llm.providers.anthropic_provider.call_anthropic",
                   return_value=fake_raw):
            # call() should still succeed (ledger error is swallowed),
            # but the ledger would have rejected the entry
            result = call(workload="narrator", system="s", user="u",
                          agent_id="bogus_typo_agent", record_cost=True)
        assert result.text == "x"
        # Ledger file should NOT have been written (validation failed before write)
        if ledger_file.exists():
            assert ledger_file.read_text().strip() == ""


# ──────────────────────────────────────────────────────────────────────────────
# Live ping (opt-in only — burns real API quota)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    os.environ.get("ANTHROPIC_LIVE_TEST") != "1",
    reason="Set ANTHROPIC_LIVE_TEST=1 to run live API test",
)
class TestLivePing:
    def test_haiku_4_5_ping(self):
        result = call(
            workload="narrator", system="Reply only with: pong",
            user="ping", agent_id="risk_manager", max_tokens=30,
            record_cost=False,
        )
        assert "pong" in result.text.lower()
        assert result.provider == "anthropic"
        assert result.cost_usd > 0
