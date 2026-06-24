"""
engine/narrative_classifier.py — FOMC narrative scoring (W3 D2a + D2b, 2026-05-08).

Pre-registration: docs/spec_multivariate_msm_v4_narrative.md §2.7
Spec id (engine.preregistration.SpecRegistry): 47
Single-source amendment: 2026-05-08 (ECB dropped after 6-path URL recovery failure;
see spec §2.7.6 + rule-9 N12 + amend_log).

Scope:
  • Deterministic NLP score for FOMC press-statement text (hawkish-dovish).
  • Lexicon locked from peer-reviewed publications:
      Hansen & McMahon (2016) JFE 121(1)
      Apel, Blix Grimaldi & Hokkanen (2022) JMCB 54(8)
  • z-score normalized using in-sample 1994-2018 (μ, σ) — populated W3 D3.
  • Monthly aggregation = mean of meetings in [t-30, t]; forward-fill else.

Boundary invariant (per project rule "0-LLM-in-evaluation"):
  • Score path = pure deterministic NLP. Verdict path = 0 LLM.
  • LLM tag (`llm_narrative_tag`) is descriptive supplement only — NOT in score
    and NOT in verdict computation. Tag distribution is sanity-check context.

Implementation status:
  • D2a (2026-05-08): score + aggregation + lexicon lock + unit tests.
  • D2b (2026-05-08): FOMC HTML fetcher + parse + tests; ECB removed per amend_log.
  • D3 (pending) : populate _RAW_SCORE_INSAMPLE_MEAN / _STD; lock thereafter.

Forbidden modifications (HARKing R1-R4, per spec §6):
  • Lexicon edits (24 + 24 entries locked at register-time).
  • LLM in score path.
  • z-score (μ, σ) recomputation on OOS data.
  • Aggregation rule change (window = [t-30, t], forward-fill).
  • Re-add ECB / any alt source post-OOS verdict (per 2026-05-08 amendment lock).
"""
from __future__ import annotations

import datetime
import logging
import re
import ssl
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Locked lexicon (Hansen-McMahon 2016 + Apel-Blix-Grimaldi 2022) ───────────
# Spec §2.7.1; 24 + 24 entries. DO NOT MODIFY without amend_spec(superseded).

HAWKISH_WORDS_V4: tuple[str, ...] = (
    "tighten",
    "tightening",
    "inflationary",
    "inflation pressure",
    "overheating",
    "accelerating",
    "firming",
    "restrict",
    "restrictive",
    "vigilant",
    "concern",
    "risks to upside",
    "upward pressure",
    "wage pressure",
    "taper",
    "tapering",
    "reduce balance sheet",
    "policy normalization",
    "raise",
    "raising rate",
    "hike",
    "hiking",
    "further increase",
    "additional tightening",
)

DOVISH_WORDS_V4: tuple[str, ...] = (
    "accommodative",
    "accommodation",
    "easing",
    "loosen",
    "loosening",
    "support",
    "supportive",
    "subdued",
    "weak",
    "weakening",
    "decelerating",
    "slack",
    "unemployment",
    "disinflation",
    "deflation",
    "risks to downside",
    "downward pressure",
    "pause",
    "patient",
    "patience",
    "cut",
    "cutting rate",
    "lower",
    "further easing",
)

assert len(HAWKISH_WORDS_V4) == 24, "HAWKISH lexicon must be 24 entries (spec §2.7.1)"
assert len(DOVISH_WORDS_V4) == 24, "DOVISH lexicon must be 24 entries (spec §2.7.1)"
assert len(set(HAWKISH_WORDS_V4)) == 24, "HAWKISH lexicon must be unique"
assert len(set(DOVISH_WORDS_V4)) == 24, "DOVISH lexicon must be unique"


# ── In-sample z-norm parameters (W3 D3 will populate; locked thereafter) ─────
# Spec §2.7.3. Until populated, compute_narrative_score raises NotImplementedError.

_RAW_SCORE_INSAMPLE_MEAN: Optional[float] = -2.2246952285e-03  # locked W3 D3 2026-05-08, n=231
_RAW_SCORE_INSAMPLE_STD: Optional[float] = 7.1297395149e-03   # locked W3 D3 2026-05-08, n=231


# ── Aggregation window (spec §2.7.4) ─────────────────────────────────────────
_AGG_WINDOW_DAYS: int = 30


# ─────────────────────────────────────────────────────────────────────────────
# Pure deterministic functions (no LLM, no network)
# ─────────────────────────────────────────────────────────────────────────────


def compute_raw_score(text: str) -> float:
    """
    Spec §2.7.2:
        raw = (hawkish_count - dovish_count) / total_words

    Counting is case-insensitive substring count. Multi-word phrases (e.g.
    "inflation pressure") count substring occurrences in the lower-cased text.

    Empty / blank text → 0.0 (degenerate).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    lower = text.lower()
    total_words = len(lower.split())
    if total_words == 0:
        return 0.0
    hawkish_count = sum(lower.count(w) for w in HAWKISH_WORDS_V4)
    dovish_count = sum(lower.count(w) for w in DOVISH_WORDS_V4)
    return (hawkish_count - dovish_count) / total_words


def compute_narrative_score(text: str) -> float:
    """
    z-score normalized narrative score per spec §2.7.2.

        narrative = (raw - μ_in_sample) / σ_in_sample

    Raises:
        RuntimeError if (μ, σ) not yet locked (W3 D3 prerequisite).
    """
    if _RAW_SCORE_INSAMPLE_MEAN is None or _RAW_SCORE_INSAMPLE_STD is None:
        raise RuntimeError(
            "narrative_classifier: in-sample (μ, σ) not yet locked; "
            "run W3 D3 in-sample script before scoring OOS text."
        )
    if _RAW_SCORE_INSAMPLE_STD <= 0.0:
        raise RuntimeError(
            f"narrative_classifier: in-sample σ must be > 0, got {_RAW_SCORE_INSAMPLE_STD}"
        )
    raw = compute_raw_score(text)
    return (raw - _RAW_SCORE_INSAMPLE_MEAN) / _RAW_SCORE_INSAMPLE_STD


def aggregate_monthly(
    meeting_scores: pd.Series,
    end_date: datetime.date,
    *,
    prior_value: Optional[float] = None,
) -> float:
    """
    Spec §2.7.4 monthly aggregation:

        meetings_in_month = {scores ∈ [end_date - 30d, end_date]}
        if empty:  forward-fill prior_value (or NaN if no prior)
        else:      mean of in-window scores

    Args:
        meeting_scores: pd.Series indexed by datetime.date / DatetimeIndex,
                        values are per-meeting narrative_scores.
        end_date:       month-end date for which we want the monthly value.
        prior_value:    value from previous month-end (forward-fill source);
                        None on the very first month (returns NaN if no
                        meetings found in window).

    Returns:
        float monthly_narrative_score_t (NaN allowed only when prior_value
        is None AND no meetings in window — i.e. start of series).
    """
    if not isinstance(meeting_scores, pd.Series):
        raise TypeError("meeting_scores must be pd.Series")
    if len(meeting_scores) == 0:
        return float("nan") if prior_value is None else float(prior_value)

    idx = pd.to_datetime(meeting_scores.index)
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - pd.Timedelta(days=_AGG_WINDOW_DAYS)
    mask = (idx >= start_ts) & (idx <= end_ts)
    in_window = meeting_scores.loc[mask.tolist()]

    if len(in_window) == 0:
        return float("nan") if prior_value is None else float(prior_value)
    return float(np.mean(in_window.values))


def aggregate_monthly_series(
    meeting_scores: pd.Series,
    month_ends: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convenience wrapper applying aggregate_monthly across a DatetimeIndex of
    month-ends with forward-fill carried forward.

    Returns pd.Series indexed by month_ends.
    """
    if not isinstance(month_ends, pd.DatetimeIndex):
        month_ends = pd.DatetimeIndex(month_ends)

    out = []
    prior: Optional[float] = None
    for ts in month_ends:
        val = aggregate_monthly(meeting_scores, ts.date(), prior_value=prior)
        out.append(val)
        if not (isinstance(val, float) and np.isnan(val)):
            prior = val
    return pd.Series(out, index=month_ends, name="monthly_narrative_score")


# ─────────────────────────────────────────────────────────────────────────────
# FOMC fetcher (D2b, single-source per spec §2.7.6)
# ─────────────────────────────────────────────────────────────────────────────

#
# Fed migrated their FOMC statement URL convention multiple times. D2c probes
# (2026-05-08) confirmed 4 distinct era patterns by direct URL verification:
#
#   Era 1 (1994-1995): /fomc/{YMD}default.htm
#       verified: 19940204, 19940517, 19950201
#   Era 2 (1996-2002): /boarddocs/press/general/{YYYY}/{YMD}/default.htm
#       verified: 19990630
#   Era 3 (2003-2007): /boarddocs/press/monetary/{YYYY}/{YMD}/  (trailing slash)
#       verified: 20030625
#   Era 4 (2008-2024+): /newsevents/pressreleases/monetary{YMD}a.htm
#       verified: 20070918, 20100127, 20240131
#
# Order is most-recent-first because primary OOS window 2019-2024 lives in Era 4.
# Each candidate is tried in turn; first 200 with ≥50 word body wins.
#
_FOMC_URL_TEMPLATES: tuple[str, ...] = (
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary{ymd}a.htm",
    "https://www.federalreserve.gov/boarddocs/press/monetary/{yyyy}/{ymd}/",
    "https://www.federalreserve.gov/boarddocs/press/monetary/{yyyy}/{ymd}/default.htm",
    "https://www.federalreserve.gov/boarddocs/press/general/{yyyy}/{ymd}/default.htm",
    "https://www.federalreserve.gov/fomc/{ymd}default.htm",
)
_USER_AGENT = (
    "Mozilla/5.0 (compatible; MacroAlphaPro-research/0.1; "
    "+https://github.com/zhangxizhe; academic; ${USER_EMAIL})"
)
_DEFAULT_TIMEOUT_SEC = 30
_SSL_CTX = ssl.create_default_context()


def _strip_html(html: str) -> str:
    """
    Strip <script>/<style> blocks then all tags, decode common entities, and
    collapse whitespace. Pure-stdlib (no BeautifulSoup) because Fed statement
    HTML is regular and small (~80 KB).
    """
    if not isinstance(html, str):
        raise TypeError("html must be str")
    s = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    s = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", s)
    s = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    entity_map = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&rsquo;": "'",
        "&lsquo;": "'",
        "&rdquo;": '"',
        "&ldquo;": '"',
        "&mdash;": "—",
        "&ndash;": "–",
        "&hellip;": "...",
        "﻿": "",
    }
    for k, v in entity_map.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fetch_url(url: str, *, timeout: int = _DEFAULT_TIMEOUT_SEC) -> str:
    """Fetch raw HTML/text body. Raises urllib.error.HTTPError on non-2xx."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        if r.status != 200:
            raise urllib.error.HTTPError(
                url, r.status, f"unexpected status {r.status}", r.headers, None
            )
        data = r.read()
    return data.decode("utf-8", errors="ignore")


def fetch_fomc_press_statement(
    meeting_date: datetime.date,
    *,
    timeout: int = _DEFAULT_TIMEOUT_SEC,
    return_url: bool = False,
):
    """
    Fetch FOMC press statement plain-text for `meeting_date`.

    Tries 4 URL eras in cascading fallback (most-recent first); first 200 with
    ≥50 word body wins. See `_FOMC_URL_TEMPLATES` docstring for the era table.

    Args:
        meeting_date : date of FOMC meeting (last day for 2-day meetings).
        timeout      : per-request timeout in seconds.
        return_url   : if True, return (text, url) instead of just text.

    Returns:
        text (str) — or (text, url) when return_url=True.

    Raises:
        TypeError              : meeting_date not a date.
        urllib.error.HTTPError : every candidate URL returned non-200.
        urllib.error.URLError  : network failure on every candidate URL.
    """
    if not isinstance(meeting_date, datetime.date):
        raise TypeError("meeting_date must be datetime.date")

    ymd = meeting_date.strftime("%Y%m%d")
    yyyy = meeting_date.strftime("%Y")
    candidates = tuple(t.format(ymd=ymd, yyyy=yyyy) for t in _FOMC_URL_TEMPLATES)

    last_err: Optional[Exception] = None
    for url in candidates:
        try:
            html = _fetch_url(url, timeout=timeout)
            text = _strip_html(html)
            if len(text.split()) < 50:
                # Page rendered but body too short — likely an error page or
                # non-statement release stub; try next candidate.
                last_err = ValueError(
                    f"fetch_fomc_press_statement: body suspiciously short "
                    f"({len(text.split())} words) from {url}"
                )
                continue
            logger.info(
                "fetch_fomc_press_statement: %s (%d words) from %s",
                meeting_date, len(text.split()), url,
            )
            return (text, url) if return_url else text
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            last_err = e
            continue

    if last_err is None:
        last_err = RuntimeError("fetch_fomc_press_statement: unknown error")
    raise last_err


# ─────────────────────────────────────────────────────────────────────────────
# LLM tag wrapper — descriptive only, NOT in score/verdict path
# ─────────────────────────────────────────────────────────────────────────────


_LLM_TAG_ENUM: tuple[str, ...] = (
    "hawkish_pivot",
    "dovish_pivot",
    "patience",
    "balance_sheet",
    "data_dependent",
    "no_change",
    "other",
)


def llm_narrative_tag(text: str) -> str:
    """
    Gemini 2.5 Flash supplementary tag (spec §2.7.5, descriptive only).
    Output ∈ _LLM_TAG_ENUM ∪ {'lookup_failed'} when budget exhausted.

    LLM never enters score path; never enters verdict path.
    Cost cap $30/yr per spec §2.7.5 (engine.llm_budget LLM_NARRATIVE budget).

    D2a: stub (returns 'other' deterministically — placeholder for unit tests
    that exercise score path without LLM availability).
    D2b: implement with Gemini API + locked prompt + JSON schema + cost gate.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return "other"
