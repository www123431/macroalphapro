"""
tests/test_portfolio_sleeves.py — MS-2 cross-sleeve allocation tests.

Coverage:
  - SleeveCapitalConfig validation (sum-to-one, allowlist, type, range)
  - Default initial state (100% etf_l1 / 0% ss_sp500)
  - combine_sleeve_weights math correctness
  - Cross-sleeve ticker overlap handling
  - 0-allocation sleeve skipped
  - SystemConfig persistence round-trip (with monkeypatched store)
  - is_sleeve_active feature flag behavior
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.portfolio_sleeves import (
    ALLOWED_SLEEVES,
    DEFAULT_INITIAL_ALLOCATION,
    SleeveCapitalConfig,
    combine_sleeve_weights,
    get_active_config,
    is_sleeve_active,
    set_active_config,
)


# ── Locked allowlist ────────────────────────────────────────────────────────
def test_allowed_sleeves_match_db_models() -> None:
    """ALLOWED_SLEEVES must match the values used in db_models.py sleeve_id.

    2026-05-13 evening: crypto_btc_eth removed (specs 71/72 deprecated;
    self-deceiving ρ=0.32), cta_defensive added (Path O spec id=73 SAA_DEPLOYABLE).
    """
    assert ALLOWED_SLEEVES == {"etf_l1", "ss_sp500", "cta_defensive"}


def test_default_initial_allocation() -> None:
    """Post Path O SAA_DEPLOYABLE: 90% etf_l1 + 10% cta_defensive (institutional floor)."""
    assert DEFAULT_INITIAL_ALLOCATION == {
        "etf_l1":        0.90,
        "ss_sp500":      0.00,
        "cta_defensive": 0.10,
    }
    assert abs(sum(DEFAULT_INITIAL_ALLOCATION.values()) - 1.0) < 1e-9


# ── SleeveCapitalConfig validation ─────────────────────────────────────────
def test_initial_config_valid() -> None:
    cfg = SleeveCapitalConfig.initial()
    assert cfg.allocations == {
        "etf_l1":        0.90,
        "ss_sp500":      0.00,
        "cta_defensive": 0.10,
    }


def test_config_rejects_unknown_sleeve_id() -> None:
    with pytest.raises(ValueError, match="unknown sleeve_id"):
        SleeveCapitalConfig(allocations={"etf_l1": 0.5, "fake_sleeve": 0.5})


def test_config_rejects_non_unit_sum() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SleeveCapitalConfig(allocations={"etf_l1": 0.5, "ss_sp500": 0.4})


def test_config_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        SleeveCapitalConfig(allocations={"etf_l1": 1.5, "ss_sp500": -0.5})


def test_config_rejects_weight_above_one() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        SleeveCapitalConfig(allocations={"etf_l1": 1.5, "ss_sp500": 0.0})


def test_config_rejects_non_numeric_weight() -> None:
    with pytest.raises(TypeError, match="must be numeric"):
        SleeveCapitalConfig(allocations={"etf_l1": "1.0", "ss_sp500": 0.0})  # type: ignore


def test_config_accepts_balanced_50_50() -> None:
    cfg = SleeveCapitalConfig(allocations={"etf_l1": 0.5, "ss_sp500": 0.5})
    assert cfg.allocations["etf_l1"] == 0.5
    assert cfg.allocations["ss_sp500"] == 0.5


def test_config_accepts_partial_alloc() -> None:
    """Wave B PASS scenario: 70/30 split."""
    cfg = SleeveCapitalConfig(allocations={"etf_l1": 0.70, "ss_sp500": 0.30})
    assert abs(cfg.allocations["etf_l1"] - 0.70) < 1e-9


def test_config_frozen_dataclass() -> None:
    cfg = SleeveCapitalConfig.initial()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.allocations = {}  # type: ignore


# ── combine_sleeve_weights math ─────────────────────────────────────────────
def test_combine_initial_alloc_only_etf_contributes() -> None:
    """Initial 90% etf_l1 + 0% ss_sp500 (+ 10% cta_defensive separate sleeve, no
    per-ticker weights here) → ss weights ignored, etf weights scaled by 0.90."""
    etf_w = pd.Series({"XLK": 0.20, "XLF": 0.15, "XLE": -0.10})
    ss_w  = pd.Series({"AAPL": 0.04, "MSFT": 0.03})
    combined = combine_sleeve_weights({
        "etf_l1": etf_w,
        "ss_sp500": ss_w,
    })
    assert set(combined.index) == {"XLK", "XLF", "XLE"}
    assert abs(combined["XLK"] - 0.20 * 0.90) < 1e-9   # 0.18
    assert abs(combined["XLF"] - 0.15 * 0.90) < 1e-9   # 0.135
    assert abs(combined["XLE"] - (-0.10) * 0.90) < 1e-9  # -0.09


def test_combine_50_50_split_scales_each_sleeve() -> None:
    """50/50 alloc → each sleeve weight halved."""
    etf_w = pd.Series({"XLK": 0.20})
    ss_w  = pd.Series({"AAPL": 0.04})
    cfg = SleeveCapitalConfig(allocations={"etf_l1": 0.5, "ss_sp500": 0.5})
    combined = combine_sleeve_weights(
        {"etf_l1": etf_w, "ss_sp500": ss_w},
        config=cfg,
    )
    assert abs(combined["XLK"] - 0.10) < 1e-9   # 0.20 × 0.5
    assert abs(combined["AAPL"] - 0.02) < 1e-9  # 0.04 × 0.5


def test_combine_handles_overlapping_tickers() -> None:
    """If AAPL appears in both sleeves, contributions sum."""
    etf_w = pd.Series({"AAPL": 0.05, "MSFT": 0.04})
    ss_w  = pd.Series({"AAPL": 0.03})
    cfg = SleeveCapitalConfig(allocations={"etf_l1": 0.6, "ss_sp500": 0.4})
    combined = combine_sleeve_weights(
        {"etf_l1": etf_w, "ss_sp500": ss_w},
        config=cfg,
    )
    expected_aapl = 0.05 * 0.6 + 0.03 * 0.4   # = 0.030 + 0.012 = 0.042
    assert abs(combined["AAPL"] - expected_aapl) < 1e-9
    assert abs(combined["MSFT"] - 0.04 * 0.6) < 1e-9


def test_combine_empty_input_returns_empty() -> None:
    assert combine_sleeve_weights({}).empty


def test_combine_unknown_sleeve_id_skipped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging
    etf_w = pd.Series({"XLK": 0.20})
    bogus = pd.Series({"XYZ": 0.10})
    with caplog.at_level(logging.WARNING):
        combined = combine_sleeve_weights({
            "etf_l1": etf_w,
            "fake_sleeve": bogus,
        })
    assert "fake_sleeve" not in str(combined.index.tolist())
    assert any("not in ALLOWED_SLEEVES" in r.message for r in caplog.records)


def test_combine_zero_allocation_sleeve_skipped() -> None:
    etf_w = pd.Series({"XLK": 0.20})
    ss_w  = pd.Series({"AAPL": 0.04})
    # initial config: ss_sp500 = 0
    combined = combine_sleeve_weights({"etf_l1": etf_w, "ss_sp500": ss_w})
    assert "AAPL" not in combined.index   # ss_sp500 alloc = 0 → AAPL contribution = 0


def test_combine_drops_near_zero_entries() -> None:
    """Numeric noise (< 1e-12) should be dropped to keep output clean."""
    etf_w = pd.Series({"XLK": 0.20, "XLF": 1e-15})
    combined = combine_sleeve_weights({"etf_l1": etf_w})
    assert "XLF" not in combined.index
    assert "XLK" in combined.index


# ── SystemConfig round-trip ────────────────────────────────────────────────
def test_set_get_active_config_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist a non-default config; read it back."""
    store: dict[str, str] = {}
    def fake_set(k, v): store[k] = v
    def fake_get(k, default=""): return store.get(k, default)
    monkeypatch.setattr("engine.memory.set_system_config", fake_set)
    monkeypatch.setattr("engine.memory.get_system_config", fake_get)

    cfg_in = SleeveCapitalConfig(allocations={"etf_l1": 0.7, "ss_sp500": 0.3})
    set_active_config(cfg_in, actor="test")

    cfg_out = get_active_config()
    assert abs(cfg_out.allocations["etf_l1"] - 0.7) < 1e-9
    assert abs(cfg_out.allocations["ss_sp500"] - 0.3) < 1e-9


def test_get_active_config_falls_back_to_initial_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("engine.memory.get_system_config", lambda k, default="": "")
    cfg = get_active_config()
    assert cfg.allocations == DEFAULT_INITIAL_ALLOCATION


def test_get_active_config_falls_back_on_corrupt_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engine.memory.get_system_config",
        lambda k, default="": "{not valid json",
    )
    cfg = get_active_config()
    assert cfg.allocations == DEFAULT_INITIAL_ALLOCATION


# ── is_sleeve_active ────────────────────────────────────────────────────────
def test_is_sleeve_active_initial() -> None:
    cfg = SleeveCapitalConfig.initial()
    assert is_sleeve_active("etf_l1", cfg) is True
    assert is_sleeve_active("ss_sp500", cfg) is False


def test_is_sleeve_active_unknown_id() -> None:
    cfg = SleeveCapitalConfig.initial()
    assert is_sleeve_active("nonexistent", cfg) is False


def test_is_sleeve_active_with_alloc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = SleeveCapitalConfig(allocations={"etf_l1": 0.7, "ss_sp500": 0.3})
    assert is_sleeve_active("ss_sp500", cfg) is True


# ── 0-LLM-imports invariant ────────────────────────────────────────────────
def test_module_has_no_llm_imports() -> None:
    import engine.portfolio_sleeves as mod
    src = open(mod.__file__, encoding="utf-8").read()
    forbidden = ["google.generativeai", "google.genai",
                 "from engine.deepseek_client", "from engine.key_pool"]
    for pattern in forbidden:
        assert pattern not in src, (
            f"portfolio_sleeves violates 0-LLM-imports invariant: found {pattern!r}"
        )
