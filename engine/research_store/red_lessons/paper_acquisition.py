"""engine.research_store.red_lessons.paper_acquisition — full-text PDF acquisition.

Tries multiple sources in priority order to find an open-access PDF for a
paper. Returns (pdf_bytes, source_url, source_kind) or None.

Legal sources (whitelist):
  - OpenAlex `best_oa_location.pdf_url`  (already-vetted open access)
  - OpenAlex `oa_url`                    (broader open-access link)
  - arXiv (search by title; abs/pdf URLs)
  - NBER (working paper number pattern)
  - SSRN — disabled by default (no public API, fragile scraping)

Forbidden:
  - JSTOR / Wiley / Elsevier / SAGE journal-version PDFs (paywall +
    distribution license)

The acquisition layer NEVER hits a paywalled URL. If only paywalled
URLs are known, returns None with kind=`paywalled_only`.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


USER_AGENT = "macroalphapro/0.1 (research; mailto:${USER_EMAIL})"

# Hostnames we trust as legally-distributable open-access
_OA_HOSTNAME_WHITELIST = {
    "arxiv.org",
    "www.nber.org", "nber.org",
    "openaccess.thecvf.com",
    "www.aeaweb.org",
    "papers.nips.cc",
    "papers.ssrn.com",      # SSRN preprint host — usually author-uploaded
    "ssrn.com",
    "www.ssrn.com",
}

# Explicit blacklist of commercial-publisher hostnames whose PDFs are
# paywalled regardless of URL extension. .pdf endings on these hosts
# typically redirect to a "purchase" page or 403; never download.
_PAYWALLED_PUBLISHER_HOSTS = {
    "academic.oup.com",     # Oxford University Press journals
    "onlinelibrary.wiley.com",
    "www.sciencedirect.com", "sciencedirect.com",
    "link.springer.com",
    "journals.sagepub.com",
    "www.jstor.org", "jstor.org",
    "www.tandfonline.com",  # Taylor & Francis
    "pubsonline.informs.org",
    "www.aeaweb.org/articles",  # AEA is mixed; landing pages paywalled
}


def _hostname_ok(url: str) -> bool:
    """Allow if hostname is whitelisted OR .edu/.gov. Explicit deny on
    commercial-publisher hosts even if URL ends in .pdf."""
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if host in _PAYWALLED_PUBLISHER_HOSTS:
        return False
    if any(host.endswith(b) for b in _PAYWALLED_PUBLISHER_HOSTS):
        return False
    if host in _OA_HOSTNAME_WHITELIST:
        return True
    if host.endswith(".edu") or host.endswith(".gov") or host.endswith(".ac.uk"):
        return True
    # Conservative: unknown hosts require explicit .pdf extension AND not
    # being a publisher. Personal academic pages on .org / .net /
    # subdomains are common but risk false positives; gate them on .pdf.
    if url.lower().endswith(".pdf") and "doi.org" not in host:
        return True
    return False


# ─────────────────────── source 1: OpenAlex OA links ──────────────────


def _ssrn_doi_to_pdf(doi: str) -> str | None:
    """Convert SSRN-prefixed DOI `10.2139/ssrn.NNNNN` to a direct PDF URL.
    SSRN DOIs reliably point to author-uploaded preprints — almost always
    legitimate to download per SSRN ToS."""
    if not doi:
        return None
    m = re.search(r"10\.2139/ssrn\.(\d+)", doi.lower())
    if not m:
        return None
    ssrn_id = m.group(1)
    # SSRN's "Delivery.cfm" returns the PDF directly with this query shape
    return (
        f"https://papers.ssrn.com/sol3/Delivery.cfm/"
        f"SSRN_ID{ssrn_id}_code.pdf?abstractid={ssrn_id}"
    )


def _openalex_oa_urls(work: dict[str, Any]) -> list[str]:
    """Extract candidate OA URLs from an OpenAlex work dict.

    Priority order:
      1. best_oa_location.pdf_url        (vetted PDF)
      2. open_access.oa_url              (general OA landing)
      3. locations[*].pdf_url            (other deposits)
      4. primary_location.pdf_url        (sometimes set even if best_oa is null)
    """
    urls: list[str] = []
    boa = work.get("best_oa_location") or {}
    if boa.get("pdf_url"):
        urls.append(boa["pdf_url"])

    oa = work.get("open_access") or {}
    if oa.get("oa_url") and oa["oa_url"] not in urls:
        urls.append(oa["oa_url"])

    for loc in (work.get("locations") or []):
        pdf = (loc or {}).get("pdf_url")
        if pdf and pdf not in urls:
            urls.append(pdf)

    plp = (work.get("primary_location") or {}).get("pdf_url")
    if plp and plp not in urls:
        urls.append(plp)

    return urls


# ─────────────────────── source 2: arXiv search ───────────────────────


def _arxiv_search_for_pdf(title: str, year: int | None,
                          polite_sleep_s: float = 3.5) -> str | None:
    """Query arXiv API for the title; return the first PDF URL or None.

    arXiv API: http://export.arxiv.org/api/query?search_query=ti:"..."
    Returns ATOM XML; we extract the <id> field of the first entry.
    """
    if not title:
        return None
    q = f'ti:"{title}"'
    params = {
        "search_query": q,
        "start":        "0",
        "max_results":  "3",
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    time.sleep(polite_sleep_s)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.info("arxiv search failed for %r: %s", title[:60], e)
        return None

    # Crude ATOM parse — find first <id>http://arxiv.org/abs/NNNN.NNNNN</id>
    m = re.search(r"<id>(http://arxiv\.org/abs/[^<]+)</id>", body)
    if not m:
        return None
    abs_url = m.group(1)
    # Convert /abs/ID to /pdf/ID.pdf
    pdf_url = abs_url.replace("/abs/", "/pdf/")
    if not pdf_url.endswith(".pdf"):
        pdf_url = pdf_url + ".pdf"
    return pdf_url


# ─────────────────────── source 3: NBER pattern (light) ───────────────

def _nber_url_from_work(work: dict[str, Any]) -> str | None:
    """If the OpenAlex work has an NBER working-paper number in the URL or
    metadata, construct the canonical NBER PDF URL."""
    for loc in (work.get("locations") or []):
        u = (loc or {}).get("landing_page_url") or ""
        m = re.search(r"nber\.org/papers/w(\d+)", u)
        if m:
            return f"https://www.nber.org/system/files/working_papers/w{m.group(1)}/w{m.group(1)}.pdf"
    return None


# ─────────────────────── PDF download ─────────────────────────────────


@dataclass(frozen=True)
class AcquisitionResult:
    pdf_bytes:   bytes | None
    source_url:  str
    source_kind: str          # "openalex_oa" | "arxiv" | "nber" | "none"
    note:        str = ""

    @property
    def ok(self) -> bool:
        return self.pdf_bytes is not None and len(self.pdf_bytes) > 1024


def _download(url: str, max_bytes: int = 50_000_000,
              timeout: int = 30) -> bytes | None:
    """Download PDF bytes. Refuse if hostname not OA-whitelisted, content
    not PDF, or size > max_bytes."""
    if not _hostname_ok(url):
        logger.info("refusing non-OA hostname: %s", url[:100])
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower() and not url.lower().endswith(".pdf"):
                # Not actually a PDF (probably a landing page); skip
                logger.info("not a PDF (%s) at %s", ctype, url[:100])
                return None
            data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            logger.info("PDF too large (>%d bytes): %s", max_bytes, url[:100])
            return None
        return data
    except Exception as e:
        logger.info("download failed (%s): %s", url[:100], e)
        return None


def acquire_pdf(work: dict[str, Any],
                title: str = "",
                year: int | None = None,
                try_arxiv: bool = False) -> AcquisitionResult:
    """Try to acquire a legally-distributable PDF for the given OpenAlex work.

    Strategy (priority order):
      1. OpenAlex OA URLs (best_oa_location → open_access → other locations)
      2. NBER working paper PDF URL if metadata mentions NBER
      3. (optional) arXiv search by title — finance papers rarely on arXiv,
         and arXiv aggressively rate-limits unkeyed traffic. Disabled by
         default; set try_arxiv=True to enable.
    """
    # 1. OpenAlex OA URLs
    for url in _openalex_oa_urls(work):
        if not _hostname_ok(url):
            continue
        data = _download(url)
        if data:
            return AcquisitionResult(data, url, "openalex_oa")

    # 1b. SSRN DOI fallback — OpenAlex OA field often points to publisher
    #     paywalled version even when an SSRN preprint exists. Try the
    #     SSRN PDF URL constructed from the SSRN DOI if present in
    #     work["doi"] or any location.
    candidate_dois = [work.get("doi") or ""]
    for loc in (work.get("locations") or []):
        candidate_dois.append((loc or {}).get("doi") or "")
    for d in candidate_dois:
        su = _ssrn_doi_to_pdf(d)
        if su:
            data = _download(su)
            if data:
                return AcquisitionResult(data, su, "ssrn")

    # 2. NBER
    nu = _nber_url_from_work(work)
    if nu:
        data = _download(nu)
        if data:
            return AcquisitionResult(data, nu, "nber")

    # 3. arXiv (optional)
    if try_arxiv:
        arxiv_url = _arxiv_search_for_pdf(title, year)
        if arxiv_url:
            data = _download(arxiv_url)
            if data:
                return AcquisitionResult(data, arxiv_url, "arxiv")

    return AcquisitionResult(None, "", "none",
                             note="no OA / preprint URL found via OpenAlex / NBER" +
                                  (" / arXiv" if try_arxiv else ""))


# ─────────────────────── PDF extraction (pymupdf) ─────────────────────


def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Extract concatenated full text from a PDF using pymupdf. Returns
    None on failure."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(parts)
    except Exception as e:
        logger.warning("pdf extraction failed: %s", e)
        return None
