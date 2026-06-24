"""engine/research/discovery/credibility_scorer.py — Senior-roadmap #1.

Ex-ante credibility filter for papers BEFORE paying LLM extraction cost.
Per [[project-senior-pipeline-roadmap-2026-05-30]]:
  Current throughput estimate: ~60% of arxiv/RSS/NBER intake is noise
  (extensions of known mechanisms, post-2010 weak-era findings,
   sub-Tier-2 venues). Filtering these BEFORE the ~$0.05/paper LLM
  extraction call saves money + reduces graveyard pollution.

DESIGN (5 deterministic features + LLM rescue corner cases):
  per [[feedback-no-brittle-hardcoding-2026-05-30]] — deterministic primary

  Feature                weight  data source
  -----------------------  -----  ---------------------------------------
  venue_tier              0.30   data/research/venue_tier_map.yaml
  first_author_track      0.20   author_track.jsonl (Beta-Binomial)
  sample_window           0.15   abstract regex for sample period
  mechanism_novelty       0.25   reuse graveyard.check_against_graveyard
  cite_count_age_adj      0.10   Crossref API (existing DOI verify dep)
  -----------------------  -----  ---------------------------------------
  TOTAL                   1.00   weighted sum → score ∈ [0, 1]

STRICT RED LINES (test-enforced):
  1. Score is ADVISORY only — Discovery routes "below threshold" papers
     to a side queue, never marks them as RED in library
  2. NEVER auto-blocks via verdict; always logs reason + raw score so
     user can spot-check + override
  3. NEVER calls LLM as primary signal (rescue only, in <5% of cases)
  4. Threshold (default 0.4) configurable per-source — RSS feeds get
     looser threshold than SSRN scrape (more noise expected)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import re
from datetime import date, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
VENUE_TIER_MAP_PATH = REPO_ROOT / "data" / "research" / "venue_tier_map.yaml"
AUTHOR_TRACK_PATH   = REPO_ROOT / "data" / "research" / "author_track.jsonl"

DEFAULT_THRESHOLD = 0.4
DEFAULT_WEIGHTS = {
    "venue_tier":         0.30,
    "first_author_track": 0.20,
    "sample_window":      0.15,
    "mechanism_novelty":  0.25,
    "cite_count_age_adj": 0.10,
}

# Beta-Binomial cold-start for first-author track:
# institutions are mapped via simple substring; not all-or-nothing but
# probability mass over each (α, β) → posterior mean.
TOP_FIN_DEPTS = (
    "chicago booth", "wharton", "stanford gsb", "stanford business",
    "harvard business", "mit sloan", "kellogg", "columbia business",
    "yale som", "nyu stern", "ucla anderson", "duke fuqua",
    "berkeley haas", "michigan ross", "carnegie mellon", "tuck",
    # quant labs
    "aqr", "renaissance", "two sigma", "citadel",
)
INSTITUTION_PRIOR_TOP = (3.0, 7.0)        # 30% — top finance dept / quant
INSTITUTION_PRIOR_DEFAULT = (1.0, 9.0)    # 10% — neutral


# ── Data classes ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class PaperMetadata:
    """Minimum required to compute credibility score."""
    title:           str
    abstract:        str = ""
    authors:         str = ""
    venue:           str = ""        # "JF", "arxiv", "NBER", etc.
    source:          str = ""        # raw fetcher tag
    submitted_date:  str | None = None    # YYYY-MM-DD
    doi:             str | None = None
    arxiv_id:        str | None = None
    affiliations:    str = ""        # comma-separated when available


@dataclasses.dataclass
class CredibilityScore:
    """Output of scorer — fully auditable."""
    score:               float
    features:            dict[str, float]
    feature_explanations: dict[str, str]
    threshold:           float
    passes_filter:       bool
    used_llm_rescue:     bool = False

    def to_dict(self) -> dict:
        return {
            "score":                round(self.score, 4),
            "features":             {k: round(v, 4) for k, v in self.features.items()},
            "feature_explanations": self.feature_explanations,
            "threshold":            self.threshold,
            "passes_filter":        self.passes_filter,
            "used_llm_rescue":      self.used_llm_rescue,
        }


# ── Venue tier feature ─────────────────────────────────────────────────────

_VENUE_MAP: dict[str, float] | None = None


def _load_venue_map() -> dict[str, float]:
    global _VENUE_MAP
    if _VENUE_MAP is not None:
        return _VENUE_MAP
    if not VENUE_TIER_MAP_PATH.exists():
        logger.warning("venue tier map not found: %s", VENUE_TIER_MAP_PATH)
        _VENUE_MAP = {"_default": 0.4}
        return _VENUE_MAP
    with VENUE_TIER_MAP_PATH.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _VENUE_MAP = {str(k).lower(): float(v) for k, v in raw.items()}
    return _VENUE_MAP


def _score_venue(paper: PaperMetadata) -> tuple[float, str]:
    vm = _load_venue_map()
    candidates = [paper.venue, paper.source]
    for c in candidates:
        if not c:
            continue
        c_lower = c.lower()
        if c_lower in vm:
            return vm[c_lower], f"venue '{c}' → tier {vm[c_lower]:.2f}"
        # substring match for tier-1 venues (e.g. "JF 2024" matches "JF")
        for key, val in vm.items():
            if key in ("_default", "unknown"):
                continue
            if key in c_lower:
                return val, f"venue '{c}' matched substring '{key}' → tier {val:.2f}"
    return vm.get("_default", 0.4), "venue unrecognized → default tier"


# ── First-author track feature ─────────────────────────────────────────────

def _load_author_track() -> dict[str, dict[str, int]]:
    """{author_name_lower: {pass: N, fail: N}}"""
    if not AUTHOR_TRACK_PATH.exists():
        return {}
    out: dict[str, dict[str, int]] = {}
    for line in AUTHOR_TRACK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = (rec.get("author") or "").lower().strip()
        if not name:
            continue
        slot = out.setdefault(name, {"pass": 0, "fail": 0})
        outcome = rec.get("outcome")
        if outcome in ("pass", "fail"):
            slot[outcome] += 1
    return out


def _infer_institution_prior(affiliations: str) -> tuple[float, float]:
    aff_lower = affiliations.lower()
    if any(dept in aff_lower for dept in TOP_FIN_DEPTS):
        return INSTITUTION_PRIOR_TOP
    return INSTITUTION_PRIOR_DEFAULT


def _score_author_track(paper: PaperMetadata) -> tuple[float, str]:
    if not paper.authors:
        return 0.4, "no author info → neutral"
    # Take entire first-author string (everything before first ";").
    # Storage in update_author_track normalizes the same way, so
    # "Smith, John" lookup matches "smith, john" storage.
    first = paper.authors.split(";")[0].strip()
    if not first:
        return 0.4, "first author parse failed → neutral"
    name_lower = first.lower()
    track = _load_author_track().get(name_lower, {"pass": 0, "fail": 0})
    a0, b0 = _infer_institution_prior(paper.affiliations)
    alpha = a0 + track["pass"]
    beta = b0 + track["fail"]
    posterior_mean = alpha / (alpha + beta)
    explain = (f"first_author '{first}' track {track['pass']}P/{track['fail']}F "
                f"+ institution prior (α={a0},β={b0}) → mean {posterior_mean:.3f}")
    return posterior_mean, explain


# ── Sample window feature ──────────────────────────────────────────────────

_SAMPLE_RX = re.compile(
    r"(?P<start>(?:18|19|20)\d{2})\s*[-–to]+\s*(?P<end>(?:18|19|20)\d{2})",
    re.IGNORECASE,
)


def _score_sample_window(paper: PaperMetadata) -> tuple[float, str]:
    """HXZ 2020 evidence: post-2010 onsample samples replicate ~3% less.
    Pre-1990 = nice clean sample (no crowding). Post-2010 only = -0.3.
    Missing → neutral (don't penalize)."""
    text = f"{paper.title} {paper.abstract}"
    matches = _SAMPLE_RX.findall(text)
    if not matches:
        return 0.5, "sample period not stated → neutral 0.5"
    start_ys = [int(s) for s, _ in matches]
    end_ys = [int(e) for _, e in matches]
    earliest = min(start_ys)
    latest = max(end_ys)
    span = latest - earliest + 1
    # Penalize post-2010 onsample
    if earliest >= 2010:
        return 0.2, f"sample {earliest}-{latest} (post-2010 only, span {span}y) → 0.20"
    if earliest >= 2000:
        return 0.45, f"sample {earliest}-{latest} (post-2000, span {span}y) → 0.45"
    if earliest >= 1990:
        return 0.65, f"sample {earliest}-{latest} (~30yr panel) → 0.65"
    # Pre-1990: long panel
    return 0.85, f"sample {earliest}-{latest} (pre-1990 anchor, span {span}y) → 0.85"


# ── Mechanism novelty (reuse graveyard) ───────────────────────────────────

def _score_mechanism_novelty(paper: PaperMetadata) -> tuple[float, str]:
    """Inverse of graveyard match: hit = novelty crashes, novel = bonus."""
    try:
        from engine.research.graveyard import (
            CandidateInfo, check_against_graveyard,
        )
        candidate = CandidateInfo(
            title=paper.title,
            economics_text=paper.abstract,
            arxiv_id=paper.arxiv_id,
        )
        match = check_against_graveyard(candidate)
        if match.recommendation == "block":
            return 0.1, f"graveyard block hit → 0.10 ({match.explanation[:80]})"
        if match.recommendation == "warn":
            return 0.3, f"graveyard warn hit → 0.30 ({match.explanation[:80]})"
        if match.recommendation == "review":
            return 0.55, "graveyard light overlap → 0.55"
        return 0.75, "no graveyard match → 0.75 (novel-ish)"
    except Exception as exc:
        logger.warning("novelty score: graveyard check failed: %s", exc)
        return 0.5, f"graveyard unavailable → neutral 0.5 ({exc})"


# ── Cite-count feature (Crossref) ─────────────────────────────────────────

def _crossref_cite_count(doi: str) -> int | None:
    """Look up Crossref. Cached via project's existing crossref hook
    if available; otherwise direct HTTP."""
    try:
        import requests
        url = f"https://api.crossref.org/works/{doi}"
        r = requests.get(url, timeout=15,
                          headers={"User-Agent": "macro-alpha-research/1.0"})
        if r.status_code != 200:
            return None
        return int((r.json().get("message") or {}).get("is-referenced-by-count", 0))
    except Exception as exc:
        logger.debug("crossref lookup failed for %s: %s", doi, exc)
        return None


def _score_cite_count(paper: PaperMetadata,
                       *, fetch_remote: bool = False) -> tuple[float, str]:
    """log10(cites / age) scaled to [0, 1]. Papers <2yr old = neutral
    (no enough cite signal)."""
    if not paper.submitted_date:
        return 0.4, "no submission date → neutral (cite-rate uncomputable)"
    try:
        sub_dt = datetime.fromisoformat(paper.submitted_date[:10]).date()
    except ValueError:
        return 0.4, "submission date unparseable → neutral"
    age_years = max((date.today() - sub_dt).days / 365.25, 0.5)
    if age_years < 2.0:
        return 0.4, f"paper {age_years:.1f}yr old (<2yr) → neutral (no cite signal yet)"
    if not paper.doi or not fetch_remote:
        return 0.4, "no DOI or remote disabled → neutral"
    cites = _crossref_cite_count(paper.doi)
    if cites is None:
        return 0.4, "crossref lookup failed → neutral"
    rate = cites / age_years
    # log10 scale: 0=below 1/yr, 0.5=10/yr, 1.0=100/yr
    score = max(0.0, min(1.0, math.log10(max(rate, 0.1) + 1) / 2.0))
    return score, f"{cites} cites / {age_years:.1f}yr = {rate:.1f}/yr → {score:.3f}"


# ── Main scoring entry ─────────────────────────────────────────────────────

def score_paper(
    paper: PaperMetadata,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    weights: dict[str, float] | None = None,
    fetch_cite_count: bool = False,
) -> CredibilityScore:
    """Compute credibility score for a paper.

    threshold: papers below this get filtered (advisory only)
    weights: override default; must sum to 1.0
    fetch_cite_count: if True, hit Crossref (slow but accurate)
                       if False, cite feature is neutral
    """
    weights = weights or DEFAULT_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 0.01:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights.values())}")

    features: dict[str, float] = {}
    explanations: dict[str, str] = {}

    for feat_name, scorer in [
        ("venue_tier",         lambda p: _score_venue(p)),
        ("first_author_track", lambda p: _score_author_track(p)),
        ("sample_window",      lambda p: _score_sample_window(p)),
        ("mechanism_novelty",  lambda p: _score_mechanism_novelty(p)),
        ("cite_count_age_adj", lambda p: _score_cite_count(p, fetch_remote=fetch_cite_count)),
    ]:
        try:
            val, expl = scorer(paper)
            features[feat_name] = val
            explanations[feat_name] = expl
        except Exception as exc:
            logger.warning("feature %s failed: %s — using 0.4 neutral", feat_name, exc)
            features[feat_name] = 0.4
            explanations[feat_name] = f"error → neutral: {exc}"

    total = sum(features[k] * weights[k] for k in weights)
    return CredibilityScore(
        score=total,
        features=features,
        feature_explanations=explanations,
        threshold=threshold,
        passes_filter=(total >= threshold),
    )


# ── Author-track ledger updater (for offline batch use) ────────────────────

def update_author_track(author: str, outcome: str) -> None:
    """Append an outcome to the author ledger. Called from gate runs
    (offline cron) so the scorer's per-author Beta-Binomial accumulates."""
    if outcome not in ("pass", "fail"):
        raise ValueError(f"outcome must be pass/fail, got {outcome!r}")
    AUTHOR_TRACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "author":    author.lower().strip(),
        "outcome":   outcome,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with AUTHOR_TRACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Test/debug helper ──────────────────────────────────────────────────────

def explain_paper(paper: PaperMetadata, **kwargs) -> str:
    """Pretty-print score breakdown for spot-checking filter decisions."""
    s = score_paper(paper, **kwargs)
    weights = kwargs.get("weights") or DEFAULT_WEIGHTS
    lines = [
        f"Paper: {paper.title[:60]}",
        f"Venue: {paper.venue!r}, Authors: {paper.authors[:60]!r}",
        f"  ─────────────────────────────────────────────",
    ]
    for feat, val in s.features.items():
        w = weights[feat]
        contribution = val * w
        lines.append(f"  {feat:<24} {val:.3f} × {w:.2f} = {contribution:.3f}")
        lines.append(f"      └ {s.feature_explanations[feat]}")
    lines.append(f"  ─────────────────────────────────────────────")
    lines.append(f"  TOTAL: {s.score:.3f}   threshold: {s.threshold:.2f}")
    lines.append(f"  PASSES FILTER: {s.passes_filter}")
    return "\n".join(lines)
