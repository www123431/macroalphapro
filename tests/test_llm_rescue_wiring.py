"""Tests for LLM-rescue wiring through discovery_pipeline → batch → runner."""
from __future__ import annotations

from unittest import mock

import pandas as pd
import pytest


@pytest.fixture
def low_conf_extraction():
    """Extraction matching the Bitcoin ETF case: low conf + known family."""
    from engine.research.discovery import paper_extractor
    return paper_extractor.PaperExtraction(
        arxiv_id="x", title="Implied ETF Carry Rates",
        mechanism_proposal="Exploit ETF carry-rate wedges",
        family_guess="carry", parent_family_guess="cross_asset",
        required_data_tokens=["fx_futures"],
        economic_intuition="Carry segmentation creates wedges.",
        decay_resilience_claim="not addressed",
        novelty_assessment="extension",
        confidence=0.10,    # below borderline floor 0.30
        cost_usd=0.0005, mode="llm",
    )


# ── process_paper rescue path ────────────────────────────────────────────

def test_process_paper_skips_rescue_by_default(monkeypatch, low_conf_extraction):
    """Without use_llm_rescue, low-conf paper skips (no LLM rescue called)."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor
    from engine.research.discovery import llm_feature_extractor
    import engine.research.hygiene_tools as ht

    monkeypatch.setattr(paper_extractor, "extract_from_paper",
                          lambda *a, **kw: low_conf_extraction)
    monkeypatch.setattr(ht, "DATA_INVENTORY", {"fx_futures": {}})
    # Spy on hybrid call — must NOT fire
    hybrid_spy = mock.MagicMock()
    monkeypatch.setattr(llm_feature_extractor, "compute_hybrid_confidence",
                          hybrid_spy)

    paper = {"arxiv_id": "x", "source_id": "x",
              "title": "Implied ETF Carry Rates",
              "abstract": "We study carry rates.",
              "submitted_date": "2024-05-01", "venue": "arxiv"}
    # Note: use_llm=False so paper_extractor mock returns regardless
    out = dp.process_paper(paper, use_llm=True, library_titles=set(),
                              library_families={}, use_llm_rescue=False)
    # Low-conf unknown family with bonus → still in borderline
    # (family=carry adds bonus → conf 0.30)
    assert out["verdict"] in ("borderline_review", "skip")
    hybrid_spy.assert_not_called()


def test_process_paper_rescue_calls_hybrid_when_low_conf(
    monkeypatch, low_conf_extraction,
):
    """With use_llm_rescue=True and conf < 0.30, hybrid IS called."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor
    from engine.research.discovery import llm_feature_extractor
    import engine.research.hygiene_tools as ht

    monkeypatch.setattr(paper_extractor, "extract_from_paper",
                          lambda *a, **kw: low_conf_extraction)
    monkeypatch.setattr(ht, "DATA_INVENTORY", {"fx_futures": {}})

    fake_hybrid = {
        "base_confidence":     0.10,
        "hybrid_confidence":   0.40,    # LLM rescue raised it to review tier
        "rescued_features":    [{"llm_feature": "specifies_long_short",
                                    "regex_feature": "return_prediction_claim",
                                    "weight": 0.20}],
        "llm_features":        {"specifies_long_short": True},
        "llm_extraction_ok":   True,
        "llm_cost_usd":         0.0008,
    }
    monkeypatch.setattr(llm_feature_extractor, "compute_hybrid_confidence",
                          lambda *a, **kw: fake_hybrid)

    paper = {"arxiv_id": "x", "source_id": "x",
              "title": "Implied ETF Carry Rates",
              "abstract": "We study carry rates.",
              "submitted_date": "2024-05-01", "venue": "arxiv"}
    out = dp.process_paper(paper, use_llm=True, library_titles=set(),
                              library_families={}, use_llm_rescue=True)
    assert "llm_rescue" in out
    assert out["llm_rescue"]["hybrid_confidence"] == 0.40
    assert out["llm_rescue"]["llm_cost_usd"] == 0.0008
    # 0.40 ≥ carry threshold 0.40 → review tier
    assert out["routing"]["routing"] == "review"


def test_process_paper_rescue_not_called_for_already_passing(
    monkeypatch,
):
    """When conf ≥ 0.30 already, rescue is NOT called (saves cost)."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor
    from engine.research.discovery import llm_feature_extractor
    import engine.research.hygiene_tools as ht

    high_conf_extraction = paper_extractor.PaperExtraction(
        arxiv_id="x", title="X", mechanism_proposal="X",
        family_guess="carry", parent_family_guess="cross_asset",
        required_data_tokens=["fx_futures"],
        economic_intuition="", decay_resilience_claim="",
        novelty_assessment="", confidence=0.55,     # already above floor
        cost_usd=0.0005, mode="llm",
    )
    monkeypatch.setattr(paper_extractor, "extract_from_paper",
                          lambda *a, **kw: high_conf_extraction)
    monkeypatch.setattr(ht, "DATA_INVENTORY", {"fx_futures": {}})

    hybrid_spy = mock.MagicMock()
    monkeypatch.setattr(llm_feature_extractor, "compute_hybrid_confidence",
                          hybrid_spy)

    paper = {"arxiv_id": "x", "source_id": "x", "title": "X",
              "abstract": "ab", "submitted_date": "2024-01-01", "venue": "arxiv"}
    dp.process_paper(paper, use_llm=True, library_titles=set(),
                        library_families={}, use_llm_rescue=True)
    hybrid_spy.assert_not_called()


def test_process_paper_rescue_handles_exception(
    monkeypatch, low_conf_extraction,
):
    """If hybrid raises, paper still processes (rescue is best-effort)."""
    from engine.research.discovery import discovery_pipeline as dp
    from engine.research.discovery import paper_extractor
    from engine.research.discovery import llm_feature_extractor
    import engine.research.hygiene_tools as ht

    monkeypatch.setattr(paper_extractor, "extract_from_paper",
                          lambda *a, **kw: low_conf_extraction)
    monkeypatch.setattr(ht, "DATA_INVENTORY", {"fx_futures": {}})
    monkeypatch.setattr(llm_feature_extractor, "compute_hybrid_confidence",
                          lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated")))

    paper = {"arxiv_id": "x", "source_id": "x", "title": "X",
              "abstract": "ab", "submitted_date": "2024-01-01", "venue": "arxiv"}
    out = dp.process_paper(paper, use_llm=True, library_titles=set(),
                              library_families={}, use_llm_rescue=True)
    # Should NOT crash; rescue error recorded
    assert "llm_rescue" in out
    assert "error" in out["llm_rescue"]


# ── runner CLI flag ──────────────────────────────────────────────────────

def test_runner_cli_passes_rescue_flag_through(monkeypatch, tmp_path):
    """--use-llm-rescue propagates through discover_new_flow → batch."""
    monkeypatch.chdir(tmp_path)
    import sys
    repo = "${REPO_ROOT}/Desktop/intern"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts import run_paper_discovery

    received = {}
    def _capture_batch(papers_df, **kw):
        received.update(kw)
        return {"total": 0, "queued": 0, "review_with_caveat": 0,
                "borderline": 0, "stage_counts": {}}
    monkeypatch.setattr(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        _capture_batch,
    )
    monkeypatch.setattr(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        lambda **kw: pd.DataFrame([{"source": "x", "source_id": "1",
                                          "title": "T", "abstract": "A"}]),
    )
    rc = run_paper_discovery.main(
        ["--new-flow", "--no-llm", "--use-llm-rescue", "--max-per-source", "1"],
    )
    assert rc == 0
    assert received.get("use_llm_rescue") is True


def test_runner_cli_default_is_no_rescue(monkeypatch, tmp_path):
    """Default (no flag) → use_llm_rescue=False."""
    monkeypatch.chdir(tmp_path)
    import sys
    repo = "${REPO_ROOT}/Desktop/intern"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from scripts import run_paper_discovery

    received = {}
    def _capture_batch(papers_df, **kw):
        received.update(kw)
        return {"total": 0, "queued": 0, "review_with_caveat": 0,
                "borderline": 0, "stage_counts": {}}
    monkeypatch.setattr(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        _capture_batch,
    )
    monkeypatch.setattr(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        lambda **kw: pd.DataFrame([{"source": "x", "source_id": "1",
                                          "title": "T", "abstract": "A"}]),
    )
    rc = run_paper_discovery.main(["--new-flow", "--no-llm"])
    assert rc == 0
    assert received.get("use_llm_rescue") is False
