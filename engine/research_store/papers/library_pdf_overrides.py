"""engine.research_store.papers.library_pdf_overrides — manual_pdf_url for non-doctrine.

Flat dict keyed by lowercased DOI. Each entry is a `PdfOverride` carrying:
  - manual_pdf_url     (verified author-hosted / NBER / SSRN preprint URL)
  - expected_authors   (used to validate the OpenAlex / registry metadata)
  - verified_first_page_quote   (string we copied from the PDF first page;
                                 future re-runs use this for additional
                                 sanity checks)

This file is the SOURCE OF TRUTH for Layer-1 lib/lesson paper PDFs. The
seed driver iterates over registry entries with status=METADATA_ONLY,
looks up their DOI here, and runs the existing acquire→validate→chunk→
ingest pipeline if a match is found.

Same legal whitelist as doctrine: arXiv / NBER / SSRN preprint / .edu /
.gov / .ac.uk. Hand-vetted URLs that DON'T match the whitelist get a
`bypass_whitelist=True` marker — used sparingly (e.g. CBS Research Portal,
edu redirect targets).

Adding entries:
  1. WebSearch for paper's preprint URL
  2. WebFetch the URL; verify first page via pymupdf:
       python -c "import fitz; print(fitz.open('...').get_text('text')[:400])"
  3. Add entry with verified_first_page_quote = first 200 chars
  4. Re-run scripts/seed_library_papers.py --write
"""
from __future__ import annotations

import dataclasses as _dc


@_dc.dataclass(frozen=True)
class PdfOverride:
    manual_pdf_url:               str
    expected_authors:             tuple[str, ...]
    verified_first_page_quote:    str
    note:                         str = ""
    bypass_whitelist:             bool = False


# DOI → PdfOverride. Lowercase DOI keys for case-insensitive lookup.
LIBRARY_PDF_OVERRIDES: dict[str, PdfOverride] = {

    # ─── Tetlock 2007 "Giving Content to Investor Sentiment" (JF) ─────
    "10.1111/j.1540-6261.2007.01232.x": PdfOverride(
        manual_pdf_url = "https://business.columbia.edu/sites/default/files-efs/pubfiles/3097/Tetlock_Media_Sentiment_JF.pdf",
        expected_authors = ("Tetlock",),
        verified_first_page_quote =
            "Giving Content to Investor Sentiment: The Role of Media in the Stock Market / Paul C. Tetlock",
        note = "Columbia Business School author-hosted, verified 2026-06-04.",
    ),

    # ─── Da-Engelberg-Gao 2011 "In Search of Attention" (JF) ──────────
    "10.1111/j.1540-6261.2011.01679.x": PdfOverride(
        manual_pdf_url = "https://academicweb.nd.edu/~zda/Google.pdf",
        expected_authors = ("Da", "Engelberg", "Gao"),
        verified_first_page_quote =
            "THE JOURNAL OF FINANCE / In Search of Attention / "
            "ZHI DA, JOSEPH ENGELBERG, and PENGJIE GAO",
        note = "Notre Dame Zhi Da homepage (academicweb.nd.edu after redirect "
               "from www3.nd.edu), verified 2026-06-04.",
    ),

    # ─── Cohen-Frazzini 2008 "Economic Links and Predictable Returns" (JF) ─
    "10.1111/j.1540-6261.2008.01379.x": PdfOverride(
        manual_pdf_url = "https://pages.stern.nyu.edu/~afrazzin/pdf/Economic%20Links%20and%20Predictable%20Returns%20-%20Cohen%20and%20Frazzini.pdf",
        expected_authors = ("Cohen", "Frazzini"),
        verified_first_page_quote =
            "THE JOURNAL OF FINANCE / Economic Links and Predictable Returns / "
            "LAUREN COHEN and ANDREA FRAZZINI",
        note = "NYU Stern Frazzini homepage, verified 2026-06-04.",
    ),

    # ─── Carr-Wu 2009 "Variance Risk Premiums" (RFS) ──────────────────
    "10.1093/rfs/hhn038": PdfOverride(
        manual_pdf_url = "https://engineering.nyu.edu/sites/default/files/2019-01/CarrReviewofFinStudiesMarch2009-a.pdf",
        expected_authors = ("Carr", "Wu"),
        verified_first_page_quote =
            "Variance Risk Premia / PETER CARR (Bloomberg) / LIUREN WU (Baruch)",
        note = "NYU Engineering hosted (Wu's faculty profile materials), "
               "verified 2026-06-04. Title 'Variance Risk Premia' (early draft) "
               "later renamed 'Variance Risk Premiums' for RFS publication.",
    ),

    # ─── Novy-Marx 2013 "The Other Side of Value" (JFE) ───────────────
    "10.1016/j.jfineco.2013.01.003": PdfOverride(
        manual_pdf_url = "https://mysimon.rochester.edu/novy-marx/research/OSoV.pdf",
        expected_authors = ("Novy-Marx",),
        verified_first_page_quote =
            "The Other Side of Value: The Gross Profitability Premium / Robert Novy-Marx",
        note = "Rochester Simon Business School author-hosted, verified 2026-06-04.",
    ),

    # ─── Hirshleifer-Lim-Teoh 2009 "Driven to Distraction" (JF) ───────
    "10.1111/j.1540-6261.2009.01501.x": PdfOverride(
        manual_pdf_url = "https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/c/362/files/2020/07/Driven-to-Distraction-Extraneous-Events-and-Underreaction-to-Earnings-News.pdf",
        expected_authors = ("Hirshleifer", "Lim", "Teoh"),
        verified_first_page_quote =
            "THE JOURNAL OF FINANCE / Driven to Distraction: Extraneous "
            "Events and Underreaction to Earnings News / DAVID HIRSHLEIFER, "
            "SONYA SEONGYEON LIM, and SIEW HONG TEOH",
        note = "UCI Hirshleifer faculty CDN (sites.uci.edu), verified 2026-06-04.",
    ),

    # ─── Lustig-Roussanov-Verdelhan 2011 "Common Risk Factors in Currency" (RFS) ─
    "10.1093/rfs/hhr068": PdfOverride(
        manual_pdf_url = "https://www.nber.org/system/files/working_papers/w14082/w14082.pdf",
        expected_authors = ("Lustig", "Roussanov", "Verdelhan"),
        verified_first_page_quote =
            "NBER WORKING PAPER SERIES / COMMON RISK FACTORS IN CURRENCY "
            "MARKETS / Hanno Lustig / Nikolai Roussanov / Adrien Verdelhan",
        note = "NBER WP 14082, verified 2026-06-04.",
    ),

    # ─── Asness-Frazzini-Pedersen 2019 "Quality Minus Junk" (RAS) ─────
    "10.1007/s11142-018-9470-2": PdfOverride(
        manual_pdf_url = "https://images.aqr.com/-/media/AQR/Documents/Insights/Working-Papers/Quality-Minus-Junk.pdf",
        expected_authors = ("Asness", "Frazzini", "Pedersen"),
        verified_first_page_quote =
            "Quality Minus Junk / Clifford S. Asness, Andrea Frazzini, "
            "and Lasse H. Pedersen / This draft: October 9, 2013",
        note = "AQR images.aqr.com (author-uploaded working paper). "
               "Non-OA-whitelisted host but AQR is canonical free distribution "
               "for the working-paper version. verified 2026-06-04.",
        bypass_whitelist = True,
    ),

    # ─── Hurst-Ooi-Pedersen 2017 "Century of Evidence on Trend-Following" (JPM) ─
    "10.3905/jpm.2017.44.1.015": PdfOverride(
        manual_pdf_url = "https://fairmodel.econ.yale.edu/ec439/hurst.pdf",
        expected_authors = ("Hurst", "Ooi", "Pedersen"),
        verified_first_page_quote =
            "A Century of Evidence on Trend-Following Investing / "
            "BRIAN HURST, YAO HUA OOI, AND LASSE HEJE PEDERSEN",
        note = "Yale Fairmodel (Ray Fair's course material hosting), "
               "verified 2026-06-04.",
    ),
}
