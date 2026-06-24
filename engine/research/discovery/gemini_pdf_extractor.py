"""engine/research/discovery/gemini_pdf_extractor.py — full-paper PDF
extraction via Gemini 2M-context for the "no abstract anywhere" case.

Senior 漏洞 8 per user 2026-05-30: papers like FF 2015 have NO public
abstract in Crossref / OpenAlex / Semantic Scholar. The current
venue-tier fallback works but loses the actual content signal.
Gemini 2.5 Pro / Flash with 2M-token context can ingest the full
PDF body and emit:
  - reconstructed abstract (~200 words)
  - 7-bool feature extraction (same as llm_feature_extractor)
  - mechanism / family / required_data fields

DESIGN PRINCIPLES:
  1. Only fires when abstract is MISSING (< 50 chars). Don't waste
     Gemini tokens when Crossref/OpenAlex already gave a usable
     abstract.
  2. PDF acquisition: try the paper's pdf_url first. If publisher-
     locked, try OpenAlex's best-known PDF mirror. Don't scrape
     publisher sites — respect their access controls.
  3. Cache: ~1 PDF download per nominate is acceptable; cache by
     DOI in data/cache/gemini_pdf_extract/ so re-nominate is free.
  4. Cost: Gemini 2.5 Flash input is ~$0.075/1M tokens. A 100-page
     PDF ≈ 50k tokens ≈ $0.004 input + ~$0.002 output ≈ $0.006 per
     extraction. Total budget for nominate flow: trivial.
  5. Graceful fallback: if PDF fetch fails / Gemini API down / key
     missing, return None — caller (review_ui.nominate) falls back
     to venue-tier scoring as before.

LIMITATIONS:
  - Won't work on paywalled-PDF-only papers (e.g. paywalled Elsevier
    where pdf_url returns 403)
  - Gemini hallucination risk on numeric extraction; we treat output
    as confidence FEATURES not as ground truth
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
PDF_CACHE_DIR = REPO_ROOT / "data" / "cache" / "gemini_pdf_extract"

GEMINI_MODEL = "gemini-2.5-flash"      # cheap; switch to 2.5-pro for accuracy

SYSTEM_PROMPT = """You are extracting structured information from a
finance research paper's full-text PDF. Return STRICT JSON with these
fields:
{
  "reconstructed_abstract": "...",
  "estimates_sharpe_or_alpha":  true|false,
  "reports_tstatistic":         true|false,
  "specifies_long_short":       true|false,
  "specifies_holding_period":   true|false,
  "specifies_universe":         true|false,
  "specifies_sample_window":    true|false,
  "proposes_tradable_mechanism": true|false,
  "family_guess":               "carry|momentum|value|...|unknown",
  "sample_period":              "1990-2020 or null",
  "key_numerics":               ["sharpe X", "t-stat Y", ...]
}

Be strict — only mark booleans true if the paper EXPLICITLY contains
the corresponding marker. Default to false on uncertainty."""


@dataclasses.dataclass
class GeminiPdfExtraction:
    ok:                        bool
    reconstructed_abstract:    str = ""
    estimates_sharpe_or_alpha: bool = False
    reports_tstatistic:        bool = False
    specifies_long_short:      bool = False
    specifies_holding_period:  bool = False
    specifies_universe:        bool = False
    specifies_sample_window:   bool = False
    proposes_tradable_mechanism: bool = False
    family_guess:              str = "unknown"
    sample_period:             str | None = None
    key_numerics:              list[str] = dataclasses.field(default_factory=list)
    cost_usd:                   float = 0.0
    cached:                    bool = False
    error:                     str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _read_vertex_config() -> dict | None:
    """Read [VERTEX] {project, location} from secrets.toml. Returns
    None if either is missing."""
    try:
        secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
        if not secrets_path.exists():
            return None
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                return None
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        vertex = data.get("VERTEX") or {}
        if isinstance(vertex, dict):
            project = vertex.get("project")
            location = vertex.get("location") or "us-central1"
            if project:
                return {"project": project, "location": location}
    except Exception:
        return None
    return None


def _build_genai_client():
    """Return a configured google.genai.Client, trying paths in order:
    1) Vertex AI via ADC (gcloud auth application-default login) + [VERTEX]
       project/location from secrets.toml — preferred (uses GCP credits).
    2) AI Studio direct API key (AIzaSy*) — fallback (free-tier quota).
    Returns None if neither configured."""
    try:
        from google import genai
    except ImportError:
        return None

    # Path 1: Vertex AI via ADC
    vcfg = _read_vertex_config()
    if vcfg:
        try:
            return genai.Client(
                vertexai=True,
                project=vcfg["project"],
                location=vcfg["location"],
            )
        except Exception as exc:
            logger.warning("Vertex ADC client init failed: %s", exc)

    # Path 2: AI Studio direct
    key = _read_gemini_key()
    if key and key.startswith("AIzaSy"):
        try:
            return genai.Client(api_key=key)
        except Exception as exc:
            logger.warning("AI Studio client init failed: %s", exc)

    return None


def _read_gemini_key() -> str | None:
    """env var → direct TOML parse of .streamlit/secrets.toml.

    Looks in TOML at top-level + nested locations the project actually
    uses: [VERTEX].GEMINI_KEY and [GEMINI_POOL] entries.
    """
    k = os.environ.get("GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    try:
        secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
        if not secrets_path.exists():
            return None
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Last-resort regex (also tries VERTEX.GEMINI_KEY heuristic)
                import re
                text = secrets_path.read_text(encoding="utf-8")
                for pat in (
                    r'^GEMINI_KEY\s*=\s*["\']([^"\']+)["\']',
                    r'^GEMINI_API_KEY\s*=\s*["\']([^"\']+)["\']',
                ):
                    m = re.search(pat, text, re.MULTILINE)
                    if m:
                        return m.group(1)
                return None
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        # Try top-level first
        for k_name in ("GEMINI_KEY", "GEMINI_API_KEY"):
            v = data.get(k_name)
            if v and isinstance(v, str):
                return v
        # Then nested VERTEX section
        vertex = data.get("VERTEX") or {}
        if isinstance(vertex, dict):
            v = vertex.get("GEMINI_KEY") or vertex.get("GEMINI_API_KEY")
            if v and isinstance(v, str):
                return v
        # Then GEMINI_POOL — pick first available
        pool = data.get("GEMINI_POOL") or {}
        if isinstance(pool, dict) and pool:
            for v in pool.values():
                if isinstance(v, str) and v:
                    return v
        return None
    except Exception:
        return None


def _cache_key(doi: str | None, pdf_url: str | None) -> str:
    """Stable hash for cache lookup."""
    raw = f"{doi or ''}|{pdf_url or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_cached(cache_key: str) -> GeminiPdfExtraction | None:
    cache_path = PDF_CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        data["cached"] = True
        return GeminiPdfExtraction(**data)
    except Exception:
        return None


def _save_cached(cache_key: str, ext: GeminiPdfExtraction) -> None:
    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = PDF_CACHE_DIR / f"{cache_key}.json"
    try:
        d = ext.to_dict()
        d["cached"] = False    # don't write cached=True; that's runtime-set
        cache_path.write_text(json.dumps(d, ensure_ascii=False),
                                  encoding="utf-8")
    except Exception as exc:
        logger.warning("cache save failed: %s", exc)


def _download_pdf(pdf_url: str, *, max_bytes: int = 30_000_000) -> bytes | None:
    """Best-effort PDF download. Capped at 30MB to avoid runaway."""
    if not pdf_url:
        return None
    try:
        import requests
        r = requests.get(
            pdf_url, timeout=30,
            headers={"User-Agent": "macro-alpha-research/1.0"},
            stream=True,
        )
        if r.status_code != 200:
            logger.info("pdf fetch HTTP %s: %s", r.status_code, pdf_url[:80])
            return None
        body = bytearray()
        for chunk in r.iter_content(chunk_size=65536):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > max_bytes:
                logger.warning("pdf exceeds %d bytes; aborting", max_bytes)
                return None
        if not body.startswith(b"%PDF"):
            logger.info("URL did not return a PDF: %s", pdf_url[:80])
            return None
        return bytes(body)
    except Exception as exc:
        logger.warning("pdf download failed: %s", exc)
        return None


def _parse_gemini_response(text: str) -> dict | None:
    """Extract first balanced {...} from response."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_from_pdf(
    pdf_url: str,
    *,
    doi: str | None = None,
    use_cache: bool = True,
) -> GeminiPdfExtraction:
    """Download PDF + extract structured features via Gemini.

    Returns ok=False with error msg on any failure. Caller falls back
    to venue-tier scoring etc.
    """
    cache_key = _cache_key(doi, pdf_url)
    if use_cache:
        cached = _load_cached(cache_key)
        if cached is not None:
            return cached

    pdf_bytes = _download_pdf(pdf_url)
    if pdf_bytes is None:
        return GeminiPdfExtraction(ok=False, error="pdf download failed")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return GeminiPdfExtraction(
            ok=False, error="google-genai SDK not installed",
        )

    client = _build_genai_client()
    if client is None:
        return GeminiPdfExtraction(
            ok=False,
            error=("no Gemini auth path works — neither Vertex (ADC + project) "
                   "nor AI Studio (AIzaSy* API key) configured"),
        )

    try:
        # Upload PDF as inline part
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=pdf_bytes, mime_type="application/pdf",
                ),
                SYSTEM_PROMPT,
                "Return strict JSON only.",
            ],
        )
        text = getattr(response, "text", "") or ""
        # Best-effort cost from usage_metadata
        usage = getattr(response, "usage_metadata", None)
        cost = 0.0
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            # Gemini 2.5 Flash: $0.075/M input, $0.30/M output (approx)
            cost = (input_tokens * 0.075 / 1_000_000
                      + output_tokens * 0.30 / 1_000_000)

        parsed = _parse_gemini_response(text)
        if not parsed:
            ext = GeminiPdfExtraction(
                ok=False, cost_usd=cost,
                error="gemini response not parseable as JSON",
            )
            return ext

        ext = GeminiPdfExtraction(
            ok=True,
            reconstructed_abstract=str(parsed.get("reconstructed_abstract", "")),
            estimates_sharpe_or_alpha=bool(parsed.get("estimates_sharpe_or_alpha", False)),
            reports_tstatistic=bool(parsed.get("reports_tstatistic", False)),
            specifies_long_short=bool(parsed.get("specifies_long_short", False)),
            specifies_holding_period=bool(parsed.get("specifies_holding_period", False)),
            specifies_universe=bool(parsed.get("specifies_universe", False)),
            specifies_sample_window=bool(parsed.get("specifies_sample_window", False)),
            proposes_tradable_mechanism=bool(parsed.get("proposes_tradable_mechanism", False)),
            family_guess=str(parsed.get("family_guess", "unknown")),
            sample_period=parsed.get("sample_period"),
            key_numerics=list(parsed.get("key_numerics") or []),
            cost_usd=cost,
        )
        if use_cache:
            _save_cached(cache_key, ext)
        return ext

    except Exception as exc:
        logger.warning("gemini extract failed: %s", exc)
        return GeminiPdfExtraction(ok=False, error=str(exc)[:300])
