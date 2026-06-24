"""engine.research_store.papers.doctrine_seed — 14 framework papers.

Hand-curated list of methodology / framework papers that anchor our
judgment calls. Each gets shelf=DOCTRINE_METHOD by default; some carry
additional shelves (e.g. KMPV 2018 also = GREEN_MOTIVATION because
we deploy a carry sleeve).

These are the load-bearing references for the entire RAG briefing
system. When a candidate is critiqued, these are the papers whose
claims get checked first.

DESIGN — failure-surface aware:

  - `expected_authors`: hard-validated against OpenAlex result authors.
    Defense against wrong-domain matches on generic titles ("Carry" /
    "Trading Costs"). If OpenAlex returns a paper with NO surname match,
    we reject the result.

  - `manual_pdf_url`: optional override pointing at a KNOWN free source
    (NBER PDF / Notices-of-AMS / aqr.com / author homepage). When set,
    we try this URL FIRST before any OpenAlex-derived URL. This is the
    primary recovery from the "OpenAlex points at paywalled Elsevier"
    failure observed in P2.

  - `expected_doi_substring`: an optional sanity check on the matched
    DOI. Catches the case where the OpenAlex top result is a wrong paper
    that happens to share an author surname.

These 14 are the SEED — once cross-link Q-E populates inbound references,
more papers will get added from GREEN factor lookups (Q-B) + library
YAML refs (Q-C).
"""
from __future__ import annotations

import dataclasses as _dc
from engine.research_store.papers.shelves import Shelf


@_dc.dataclass(frozen=True)
class ManualMetadata:
    """Override metadata when OpenAlex can't find the paper.

    Used for papers with generic titles (e.g. "Carry" / "Trading Costs")
    that OpenAlex returns wrong-domain results for, OR for working papers
    that aren't in OpenAlex at all.
    """
    title:    str
    authors:  tuple[str, ...]
    year:     int
    doi:      str = ""
    venue:    str = ""
    abstract: str = ""


@_dc.dataclass(frozen=True)
class DoctrineSeedEntry:
    anchor_str:               str
    expected_authors:         tuple[str, ...]
    shelves:                  tuple[Shelf, ...]
    shelf_notes:              dict[str, str]
    note:                     str
    manual_pdf_url:           str = ""
    expected_doi_substring:   str = ""
    manual_metadata:          ManualMetadata | None = None


DOCTRINE_SEED: list[DoctrineSeedEntry] = [

    # ── Multiple-testing / overfit doctrine ─────────────────────────
    DoctrineSeedEntry(
        anchor_str = "Harvey & Liu & Zhu 2016, 'and the Cross-Section of Expected Returns', JF",
        expected_authors = ("Harvey", "Liu", "Zhu"),
        expected_doi_substring = "10.1093/rfs/",
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Canonical |t|>=3 bar for factor research. Most-cited "
            "doctrine reference in our REDLesson F8 classification."},
        # Verified 2026-06-04: NBER WP 20592 first page reads
        # "...AND THE CROSS-SECTION OF EXPECTED RETURNS / Campbell R.
        # Harvey / Yan Liu / Heqing Zhu / Working Paper 20592"
        manual_pdf_url = "https://www.nber.org/system/files/working_papers/w20592/w20592.pdf",
        note = "HLZ 2016. NBER WP 20592 verified-match.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Bailey & Lopez de Prado 2014, 'The Deflated Sharpe Ratio', JPM",
        expected_authors = ("Bailey", "Prado"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Deflated SR formula. Our DSR threshold (0.9) anchored "
            "to this. F8 OVERFIT_INDUCED relies on it."},
        # Verified 2026-06-04: author homepage davidhbailey.com
        # first page reads "THE DEFLATED SHARPE RATIO: CORRECTING
        # FOR SELECTION BIAS, BACKTEST OVERFITTING AND NON-NORMALITY
        # / David H. Bailey † / Marcos López de Prado ‡"
        manual_pdf_url = "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
        note = "Bailey-LdP 2014. davidhbailey.com author-hosted, verified-match.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Bailey & Borwein & Lopez de Prado & Zhu 2014, 'Pseudo-Mathematics and Financial Charlatanism', NAMS",
        expected_authors = ("Bailey", "Borwein", "Prado"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Foundational warning about backtest overfitting. "
            "Less-formal companion to Deflated SR."},
        # Notices of the AMS — freely available, OA from AMS itself
        manual_pdf_url = "https://www.ams.org/notices/201405/rnoti-p458.pdf",
        note = "Notices of AMS publication — freely available.",
    ),

    # ── Publication-decay doctrine ─────────────────────────────────
    DoctrineSeedEntry(
        anchor_str = "McLean & Pontiff 2016, 'Does Academic Research Destroy Stock Return Predictability', JF",
        expected_authors = ("McLean", "Pontiff"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "26-58% post-publication decay across 97 anomalies. "
            "Cited by F1 PUBLICATION_DECAY."},
        # Verified 2026-06-04: hec.ca/finance/Fichier/McLean.pdf
        # author-hosted, first page reads "Does Academic Research
        # Destroy Stock Return Predictability?* / R. David McLean /
        # Jeffrey Pontiff / October 23, 2012"
        # (earlier guess NBER w23048 was a DIFFERENT MP paper — bug from
        # 2026-06-03 caught by PDF-title validator.)
        manual_pdf_url = "https://www.hec.ca/finance/Fichier/McLean.pdf",
        note = "MP 2016. HEC author-hosted, verified-match.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Linnainmaa & Roberts 2018, 'The History of the Cross-Section of Stock Returns', RFS",
        expected_authors = ("Linnainmaa", "Roberts"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Pre-pub sample analysis showing ~70% decay post-2000. "
            "predict_forward_decay() uses lambda=0.30 from this."},
        manual_pdf_url = "https://www.nber.org/system/files/working_papers/w22894/w22894.pdf",
        note = "LR 2018. NBER WP 22894. Loaded by candidate_pipeline.",
    ),

    # ── Spanning / replication ─────────────────────────────────────
    DoctrineSeedEntry(
        anchor_str = "Hou & Xue & Zhang 2020, 'Replicating Anomalies', RFS",
        expected_authors = ("Hou", "Xue", "Zhang"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Systematic spanning replication of 447 anomalies. "
            "Catalog of F3 SUBSUMED_BY_EXISTING failures."},
        # Verified 2026-06-04: NBER WP 23394 first page reads
        # "NBER WORKING PAPER SERIES / REPLICATING ANOMALIES / Kewei Hou
        # / Chen Xue / Lu Zhang / Working Paper 23394 / May 2017"
        #
        # Earlier 2026-06-03 attempt failed because OpenAlex returned
        # the WRONG HXZ paper ("Augmented q-Factor Model") for the
        # anchor; title-validator correctly rejected the mismatch.
        # Use ManualMetadata to bypass OpenAlex's bad result.
        manual_pdf_url = "https://www.nber.org/system/files/working_papers/w23394/w23394.pdf",
        manual_metadata = ManualMetadata(
            title    = "Replicating Anomalies",
            authors  = ("Hou", "Xue", "Zhang"),
            year     = 2020,
            doi      = "10.1093/rfs/hhy131",
            venue    = "Review of Financial Studies",
        ),
        note = "HXZ 2020. NBER WP 23394 verified-match; OpenAlex returns "
               "wrong HXZ paper so manual_metadata overrides.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Fama & French 2015, 'A Five-Factor Asset Pricing Model', JFE",
        expected_authors = ("Fama", "French"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "FF5 factor definitions. Our regressions use FF5 + UMD baseline."},
        # Verified 2026-06-04: tevgeniou.github.io (INSEAD prof
        # Theodoros Evgeniou's bibliography page) first page reads
        # "A five-factor asset pricing model$ / Eugene F. Fama /
        # Kenneth R. French"
        manual_pdf_url = "https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf",
        note = "FF5 paper. Powers engine.factor_regression. INSEAD-hosted, verified-match.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Fama & French 2018, 'Choosing Factors', JFE",
        expected_authors = ("Fama", "French"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Empirical model-selection guidance. Reference for F9 RESIDUAL_NULL."},
        note = "FF 2018.",
    ),

    # ── TC doctrine ─────────────────────────────────────────────────
    # NOTE: OpenAlex fails to find this paper for the generic title
    # "Trading Costs". Hand-encoded metadata + AQR-hosted PDF URL.
    DoctrineSeedEntry(
        anchor_str = "Frazzini & Israel & Moskowitz 2018, 'Trading Costs', SSRN Working Paper",
        expected_authors = ("Frazzini", "Israel", "Moskowitz"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Realistic TC estimates per asset class. tc_ablation_v1 "
            "uses these as anchor."},
        note = "FIM 2018. NYU Stern Frazzini homepage, verified-match "
               "(2012 draft 'Trading Costs of Asset Pricing Anomalies' "
               "preceding the 2018 'Trading Costs' SSRN update).",
        manual_metadata = ManualMetadata(
            title    = "Trading Costs",
            authors  = ("Frazzini", "Israel", "Moskowitz"),
            year     = 2018,
            doi      = "10.2139/ssrn.3229719",
            venue    = "SSRN Working Paper",
        ),
        # Verified 2026-06-04: pages.stern.nyu.edu/~afrazzin/pdf/...
        # author-hosted, first page reads "Trading Costs of Asset
        # Pricing Anomalies / ANDREA FRAZZINI, RONEN ISRAEL, AND
        # TOBIAS J. MOSKOWITZ"
        manual_pdf_url = "https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini,%20Israel%20and%20Moskowitz.pdf",
    ),

    # ── Carry doctrine + GREEN motivation ──────────────────────────
    # NOTE: OpenAlex returns wrong-domain matches for the 5-char title
    # "Carry" (medical paper about plasmids "carrying" resistance genes).
    # Hand-encoded metadata + Elsevier DOI.
    DoctrineSeedEntry(
        anchor_str = "Koijen & Moskowitz & Pedersen & Vrugt 2018, 'Carry', JFE",
        expected_authors = ("Koijen", "Moskowitz", "Pedersen", "Vrugt"),
        shelves = (Shelf.DOCTRINE_METHOD, Shelf.GREEN_MOTIVATION),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Canonical cross-asset carry framework.",
            Shelf.GREEN_MOTIVATION.value:
            "Deployed carry sleeve (commodity+FX + A.1+A.2 rates "
            "extension) motivated by this."},
        note = "KMPV 2018. Doctrine + deployed-sleeve motivation. JFE "
               "127(2):197-225.",
        manual_metadata = ManualMetadata(
            title    = "Carry",
            authors  = ("Koijen", "Moskowitz", "Pedersen", "Vrugt"),
            year     = 2018,
            doi      = "10.1016/j.jfineco.2017.11.002",
            venue    = "Journal of Financial Economics",
        ),
        # Verified 2026-06-04: CBS Research Portal accepted-manuscript
        # version. First page reads "Carry / Koijen, Ralph S.J.;
        # Moskowitz, Tobias; Pedersen, Lasse Heje; Vrugt, Evert B."
        manual_pdf_url = "https://research-api.cbs.dk/ws/portalfiles/portal/57294842/lasse_heje_pedersen_et_al_carry_acceptedmanuscript.pdf",
    ),

    # ── Momentum doctrine ───────────────────────────────────────────
    DoctrineSeedEntry(
        anchor_str = "Asness & Moskowitz & Pedersen 2013, 'Value and Momentum Everywhere', JF",
        expected_authors = ("Asness", "Moskowitz", "Pedersen"),
        shelves = (Shelf.DOCTRINE_METHOD,),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Cross-asset value + momentum + regime-conditional "
            "correlation. Key reference for F5 REGIME_DEPENDENT."},
        # Verified 2026-06-04: NYU Stern Pedersen homepage. First page
        # reads "THE JOURNAL OF FINANCE / Value and Momentum Everywhere
        # / CLIFFORD S. ASNESS, TOBIAS J. MOSKOWITZ, and LASSE HEJE
        # PEDERSEN"
        manual_pdf_url = "https://w4.stern.nyu.edu/facdir/lpederse/papers/ValMomEverywhere.pdf",
        note = "AMP 2013. NYU Stern-hosted, verified-match.",
    ),

    DoctrineSeedEntry(
        anchor_str = "Moskowitz & Ooi & Pedersen 2012, 'Time Series Momentum', JFE",
        expected_authors = ("Moskowitz", "Ooi", "Pedersen"),
        shelves = (Shelf.DOCTRINE_METHOD, Shelf.GREEN_MOTIVATION),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "TSMOM definition.",
            Shelf.GREEN_MOTIVATION.value:
            "Deployed futures TSMOM sleeve (B-axis, commit 474bf32)."},
        # Verified 2026-06-04: NYU Stern Pedersen homepage. First page
        # reads "Time series momentum / Tobias J. Moskowitz / Yao Hua
        # Ooi / Lasse Heje Pedersen"
        manual_pdf_url = "https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf",
        note = "MOP 2012. NYU Stern-hosted, verified-match.",
    ),

    # ── Low-vol doctrine ────────────────────────────────────────────
    DoctrineSeedEntry(
        anchor_str = "Frazzini & Pedersen 2014, 'Betting Against Beta', JFE",
        expected_authors = ("Frazzini", "Pedersen"),
        shelves = (Shelf.DOCTRINE_METHOD, Shelf.GREEN_MOTIVATION),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "BAB framework. Foundational low-vol anomaly reference.",
            Shelf.GREEN_MOTIVATION.value:
            "Deployed K1_BAB sleeve."},
        # Verified 2026-06-04: NYU Stern Pedersen homepage. First page
        # reads "Betting Against Beta / Andrea Frazzini and Lasse Heje
        # Pedersen* / This draft: May 10, 2013"
        manual_pdf_url = "https://w4.stern.nyu.edu/facdir/lpederse/papers/BettingAgainstBeta.pdf",
        note = "FP 2014. NYU Stern-hosted, verified-match.",
    ),

    # ── PEAD doctrine ───────────────────────────────────────────────
    DoctrineSeedEntry(
        anchor_str = "Bernard & Thomas 1989, 'Post-Earnings-Announcement Drift', JAR",
        expected_authors = ("Bernard", "Thomas"),
        shelves = (Shelf.DOCTRINE_METHOD, Shelf.GREEN_MOTIVATION),
        shelf_notes = {
            Shelf.DOCTRINE_METHOD.value:
            "Original PEAD doc. Loaded into D_PEAD spec.",
            Shelf.GREEN_MOTIVATION.value:
            "D_PEAD motivation. F1 PUBLICATION_DECAY risk per LR 2018."},
        note = "BT 1989. Doctrine + D_PEAD motivation.",
    ),
]
