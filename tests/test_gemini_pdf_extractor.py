"""Tests for engine.research.discovery.gemini_pdf_extractor."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.discovery import gemini_pdf_extractor as gpe


# ── _cache_key ───────────────────────────────────────────────────────────

def test_cache_key_stable_for_same_inputs():
    a = gpe._cache_key("10.1/x", "https://a.pdf")
    b = gpe._cache_key("10.1/x", "https://a.pdf")
    assert a == b


def test_cache_key_differs_for_different_inputs():
    a = gpe._cache_key("10.1/x", "https://a.pdf")
    b = gpe._cache_key("10.1/y", "https://a.pdf")
    assert a != b


def test_cache_key_handles_none():
    assert gpe._cache_key(None, None) != ""    # non-empty even with no input


# ── _parse_gemini_response ───────────────────────────────────────────────

def test_parse_valid_json():
    text = '{"reconstructed_abstract": "X", "estimates_sharpe_or_alpha": true}'
    parsed = gpe._parse_gemini_response(text)
    assert parsed is not None
    assert parsed["estimates_sharpe_or_alpha"] is True


def test_parse_json_embedded_in_text():
    text = "Here is the result:\n\n{\"x\": 1}\n\nDone."
    parsed = gpe._parse_gemini_response(text)
    assert parsed == {"x": 1}


def test_parse_invalid_json_returns_none():
    assert gpe._parse_gemini_response("not json at all") is None
    assert gpe._parse_gemini_response("{broken") is None


def test_parse_empty_returns_none():
    assert gpe._parse_gemini_response("") is None
    assert gpe._parse_gemini_response(None) is None


# ── _read_gemini_key ─────────────────────────────────────────────────────

def test_read_gemini_key_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_KEY", "test-key-123")
    assert gpe._read_gemini_key() == "test-key-123"


def test_read_gemini_key_fallback_to_api_key_env(monkeypatch):
    monkeypatch.delenv("GEMINI_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "alt-key-456")
    assert gpe._read_gemini_key() == "alt-key-456"


# ── Cache save / load ───────────────────────────────────────────────────

def test_cache_save_load_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(gpe, "PDF_CACHE_DIR", tmp_path)
    ext = gpe.GeminiPdfExtraction(
        ok=True,
        reconstructed_abstract="An abstract.",
        estimates_sharpe_or_alpha=True,
        specifies_long_short=True,
        family_guess="carry",
        cost_usd=0.003,
    )
    key = "test_key"
    gpe._save_cached(key, ext)
    loaded = gpe._load_cached(key)
    assert loaded is not None
    assert loaded.cached is True     # set on load
    assert loaded.reconstructed_abstract == "An abstract."
    assert loaded.family_guess == "carry"
    assert loaded.estimates_sharpe_or_alpha is True


def test_cache_load_missing_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(gpe, "PDF_CACHE_DIR", tmp_path)
    assert gpe._load_cached("nonexistent_key") is None


def test_cache_load_corrupted_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(gpe, "PDF_CACHE_DIR", tmp_path)
    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    assert gpe._load_cached("bad") is None


# ── _download_pdf ────────────────────────────────────────────────────────

def test_download_pdf_returns_none_for_404(monkeypatch):
    mock_response = mock.MagicMock()
    mock_response.status_code = 404
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    assert gpe._download_pdf("https://nope.pdf") is None


def test_download_pdf_returns_none_for_non_pdf_body(monkeypatch):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: iter([b"<html>not pdf</html>"])
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    assert gpe._download_pdf("https://x.pdf") is None


def test_download_pdf_returns_bytes_for_valid_pdf(monkeypatch):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content = lambda chunk_size: iter([
        b"%PDF-1.4 minimal pdf body",
    ])
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    body = gpe._download_pdf("https://x.pdf")
    assert body is not None
    assert body.startswith(b"%PDF")


def test_download_pdf_returns_none_when_exceeds_max_bytes(monkeypatch):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    # Stream a huge chunk
    mock_response.iter_content = lambda chunk_size: iter([
        b"%PDF" + b"x" * 100_000_000,
    ])
    monkeypatch.setattr("requests.get", lambda *a, **kw: mock_response)
    assert gpe._download_pdf("https://big.pdf", max_bytes=1_000_000) is None


# ── extract_from_pdf — main entry ─────────────────────────────────────────

def test_extract_no_client_returns_error(monkeypatch):
    """When both Vertex ADC and AI Studio key fail → returns clear error."""
    monkeypatch.setattr(gpe, "_build_genai_client", lambda: None)
    # Mock pdf download to succeed so we test the client-build failure path
    monkeypatch.setattr(gpe, "_download_pdf",
                          lambda *a, **kw: b"%PDF-1.4 mock pdf")
    result = gpe.extract_from_pdf("https://x.pdf", doi="10.1/x", use_cache=False)
    assert result.ok is False
    assert "Gemini auth" in (result.error or "")


def test_extract_pdf_download_fails(monkeypatch):
    """PDF download fails → returns clear error BEFORE trying client."""
    monkeypatch.setattr(gpe, "_download_pdf", lambda *a, **kw: None)
    result = gpe.extract_from_pdf("https://x.pdf", doi="10.1/x", use_cache=False)
    assert result.ok is False
    assert "pdf download" in (result.error or "")


def test_extract_returns_cached_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(gpe, "PDF_CACHE_DIR", tmp_path)
    # Pre-save a cache entry
    fake = gpe.GeminiPdfExtraction(ok=True, reconstructed_abstract="Cached!",
                                          family_guess="carry")
    cache_key = gpe._cache_key("10.1/x", "https://x.pdf")
    gpe._save_cached(cache_key, fake)
    # Should NOT call _download_pdf or _build_genai_client
    monkeypatch.setattr(gpe, "_download_pdf",
                          lambda *a, **kw: pytest.fail("should not be called"))
    monkeypatch.setattr(gpe, "_build_genai_client",
                          lambda: pytest.fail("should not be called"))
    result = gpe.extract_from_pdf("https://x.pdf", doi="10.1/x", use_cache=True)
    assert result.ok is True
    assert result.cached is True
    assert result.reconstructed_abstract == "Cached!"
