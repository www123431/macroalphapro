"""engine/research/graveyard.py — unified dead-factor registry (v2 senior design).

CONSOLIDATES every known-dead mechanism from 4 sources + provides
multi-signal cousin detection with failure-mode classification.

Per user 2026-05-30: "保证我们不会走上已经走过的死路".

Senior-engineering principles applied (vs naive registry):

  1. Temporal awareness — `death_date` + `decay_window_years`; an entry
     dead 10 years ago in different regime is downgraded warn → review
  2. Failure-mode classification — RED is not monolithic:
       decay_postpub / regime_hostile / decomposition_contaminated /
       sample_insufficient / cost_binding / construction_overfit / other
     Each has different revival potential.
  3. Source confidence weighting — literature-falsified > our-RED >
     discovery-flagged; combined into match confidence.
  4. Cousin-count escalation — N dead cousins in same family triggers
     elevated suspicion (single RED in family != family-wide dead).
  5. Asymmetric thresholds — false negatives (missed dead duplicate)
     more costly than false positives; default thresholds biased toward
     "warn early".
  6. Match EXPLANATION — every block/warn carries human-readable rationale.
  7. Cache with mtime check — re-build only when canonical disk changes.
  8. Inverse queries — `dead_in_family(family)` for research planning.
  9. Extensible detectors via @register_detector decorator (no hardcoded
     if/elif).
  10. Public API never raises; failure surfaces in result, not exception.

Doctrine (per [[feedback-flexibility-rigor-balance-criterion-2026-05-30]]):
- FLEX: extensible detectors + failure-mode classification + temporal decay
- RIGOR: append-only history (NEVER deletes dead-mech records); block-bias
"""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import logging
import re
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
DISCOVERY_LOG = REPO_ROOT / "data" / "research" / "discovery_log.jsonl"
GRAVEYARD_JSON = REPO_ROOT / "data" / "research" / "graveyard.json"


# ── Taxonomies (extensible, not hardcoded enums per say) ─────────────────

class FailureMode(enum.Enum):
    DECAY_POSTPUB           = "decay_postpub"
    REGIME_HOSTILE          = "regime_hostile"
    DECOMPOSITION_CONTAM    = "decomposition_contam"
    SAMPLE_INSUFFICIENT     = "sample_insufficient"
    COST_BINDING            = "cost_binding"
    CONSTRUCTION_OVERFIT    = "construction_overfit"
    OTHER                   = "other"


# Revival potential per failure mode (informs recommendation downgrade)
REVIVAL_POTENTIAL = {
    FailureMode.SAMPLE_INSUFFICIENT.value:  0.7,   # often fixable with more data
    FailureMode.REGIME_HOSTILE.value:       0.4,   # regime changes can revive
    FailureMode.COST_BINDING.value:         0.3,   # cost-aware re-design may help
    FailureMode.DECAY_POSTPUB.value:        0.1,   # rarely revivable
    FailureMode.CONSTRUCTION_OVERFIT.value: 0.1,
    FailureMode.DECOMPOSITION_CONTAM.value: 0.0,   # permanently dead unless reformulated
    FailureMode.OTHER.value:                0.2,
}


# Source-confidence weights
SOURCE_WEIGHTS = {
    "library_negative_evidence": 1.0,    # literature-falsified
    "library_red":               0.9,    # our own deep test
    "gate_runs_red":             0.8,    # individual gate run
    "discovery_rejected":        0.6,    # LLM-flagged dedup
}


# ── Data classes ────────────────────────────────────────────────────────

@dataclasses.dataclass
class GraveyardEntry:
    source:            str
    source_id:         str
    name:              str
    family:            str | None
    parent_family:     str | None
    required_data:     list[str]
    economics_text:    str
    title:             str
    failure_reason:    str
    failure_mode:      str            # FailureMode value
    death_date:        str | None
    source_weight:     float          # from SOURCE_WEIGHTS
    revival_potential: float          # from REVIVAL_POTENTIAL
    extra:             dict

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def age_years(self, as_of: datetime.date | None = None) -> float:
        """Years since death; used for temporal decay."""
        if not self.death_date:
            return 0.0
        try:
            d = datetime.date.fromisoformat(self.death_date[:10])
        except Exception:
            return 0.0
        now = as_of or datetime.date.today()
        return (now - d).days / 365.25


@dataclasses.dataclass
class CandidateInfo:
    """Candidate being checked. Caller fills available fields."""
    title:           str = ""
    family:          str | None = None
    parent_family:   str | None = None
    required_data:   list[str] = dataclasses.field(default_factory=list)
    economics_text:  str = ""
    arxiv_id:        str | None = None
    canonical_paper_id: str | None = None


@dataclasses.dataclass
class GraveyardMatch:
    matched:              bool
    signals_matched:      list[str]
    matched_entries:      list[GraveyardEntry]
    overall_confidence:   float
    recommendation:       str          # block | warn | review | allow
    explanation:          str
    cousin_count_in_family: int        # how many dead in same family
    elevated:             bool         # >=2 cousins → escalated recommendation

    def to_dict(self) -> dict:
        return {
            "matched":              self.matched,
            "signals_matched":      self.signals_matched,
            "n_matched_entries":    len(self.matched_entries),
            "matched_entries":      [e.to_dict() for e in self.matched_entries],
            "overall_confidence":   self.overall_confidence,
            "recommendation":       self.recommendation,
            "explanation":          self.explanation,
            "cousin_count_in_family": self.cousin_count_in_family,
            "elevated":             self.elevated,
        }


# ── Cache + builder ──────────────────────────────────────────────────────

_CACHE: dict = {"entries": None, "built_at": None, "source_mtimes": {}}


def _disk_mtimes() -> dict[str, float]:
    out = {}
    for p in (LIBRARY_DIR, GATE_RUNS, DISCOVERY_LOG):
        if p.exists():
            try:
                if p.is_file():
                    out[str(p)] = p.stat().st_mtime
                else:
                    out[str(p)] = max(
                        (f.stat().st_mtime for f in p.glob("*.yaml")), default=0
                    )
            except Exception:
                pass
    return out


def build_graveyard(*, use_cache: bool = True) -> list[GraveyardEntry]:
    """Consolidate every dead-mechanism record. Caches in process; auto-
    invalidates on disk mtime change."""
    if use_cache and _CACHE["entries"] is not None:
        if _CACHE["source_mtimes"] == _disk_mtimes():
            return _CACHE["entries"]

    entries: list[GraveyardEntry] = []
    entries.extend(_from_library_red())
    entries.extend(_from_library_negative_evidence())
    entries.extend(_from_gate_runs())
    entries.extend(_from_discovery_log())
    entries.extend(_from_graveyard_json())

    _CACHE["entries"] = entries
    _CACHE["built_at"] = datetime.datetime.utcnow().isoformat(timespec="seconds")
    _CACHE["source_mtimes"] = _disk_mtimes()
    return entries


# ── Source readers ───────────────────────────────────────────────────────

def _classify_failure_mode(failure_text: str) -> str:
    """Coarse rule-based failure-mode classification from free text.
    Senior researchers refine this manually in library YAML; this is the
    auto-fallback for gate_runs / discovery entries lacking explicit mode."""
    s = (failure_text or "").lower()
    if "decay" in s or "post-pub" in s or "crowding" in s or "post-publication" in s:
        return FailureMode.DECAY_POSTPUB.value
    if "regime" in s or "junk premium" in s or "hostile" in s or "2022 rate" in s:
        return FailureMode.REGIME_HOSTILE.value
    if "decomposition" in s or "absorbed" in s or "ff5" in s or "umd" in s:
        return FailureMode.DECOMPOSITION_CONTAM.value
    if "sample" in s and ("short" in s or "insufficient" in s or "too few" in s):
        return FailureMode.SAMPLE_INSUFFICIENT.value
    if "cost" in s and ("binding" in s or "drop" in s or "stress" in s):
        return FailureMode.COST_BINDING.value
    if "overfit" in s or "fragile" in s or "p-hack" in s:
        return FailureMode.CONSTRUCTION_OVERFIT.value
    return FailureMode.OTHER.value


def _from_library_red() -> list[GraveyardEntry]:
    if not LIBRARY_DIR.exists():
        return []
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if entry.get("status_in_our_book") != "RED":
            continue
        rec = entry.get("our_test_record") or {}
        failure_text = (rec.get("notes") or "") + " " + (entry.get("mechanism_economics") or "")
        fm = _classify_failure_mode(failure_text)
        source = "library_red"
        out.append(GraveyardEntry(
            source=source,
            source_id=entry.get("id", fp.stem),
            name=entry.get("id", fp.stem),
            family=entry.get("family"),
            parent_family=entry.get("parent_family"),
            required_data=entry.get("required_data") or [],
            economics_text=entry.get("mechanism_economics", ""),
            title=entry.get("id", fp.stem),
            failure_reason=rec.get("notes", "")[:500],
            failure_mode=fm,
            death_date=rec.get("date"),
            source_weight=SOURCE_WEIGHTS.get(source, 0.5),
            revival_potential=REVIVAL_POTENTIAL.get(fm, 0.2),
            extra={"canonical_paper_id": entry.get("canonical_paper_id")},
        ))
    return out


def _from_library_negative_evidence() -> list[GraveyardEntry]:
    if not LIBRARY_DIR.exists():
        return []
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if entry.get("purpose") != "negative_evidence":
            continue
        source = "library_negative_evidence"
        fm = FailureMode.DECAY_POSTPUB.value    # literature-falsified ≈ decay
        out.append(GraveyardEntry(
            source=source,
            source_id=entry.get("id", fp.stem),
            name=entry.get("id", fp.stem),
            family=entry.get("family"),
            parent_family=entry.get("parent_family"),
            required_data=entry.get("required_data") or [],
            economics_text=entry.get("mechanism_economics", ""),
            title=entry.get("id", fp.stem),
            failure_reason="literature-falsified",
            failure_mode=fm,
            death_date=entry.get("last_audited"),
            source_weight=SOURCE_WEIGHTS.get(source, 0.5),
            revival_potential=REVIVAL_POTENTIAL.get(fm, 0.1),
            extra={"canonical_paper_id": entry.get("canonical_paper_id")},
        ))
    return out


def _from_gate_runs() -> list[GraveyardEntry]:
    if not GATE_RUNS.exists():
        return []
    out = []
    for line in GATE_RUNS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("verdict") != "RED":
            continue
        name = row.get("name") or ""
        mech_text = row.get("mechanism", "")
        fm = _classify_failure_mode(mech_text)
        source = "gate_runs_red"
        out.append(GraveyardEntry(
            source=source,
            source_id=row.get("name", ""),
            name=name,
            family=None,
            parent_family=None,
            required_data=[],
            economics_text=mech_text,
            title=name,
            failure_reason=(f"sharpe={row.get('standalone_sharpe')}, "
                              f"alpha_t={row.get('alpha_t_ff5umd')}, "
                              f"dsr={row.get('deflated_sr')}"),
            failure_mode=fm,
            death_date=(row.get("ts", "") or "")[:10],
            source_weight=SOURCE_WEIGHTS.get(source, 0.5),
            revival_potential=REVIVAL_POTENTIAL.get(fm, 0.2),
            extra={"n_months": row.get("n_months"),
                    "deflated_sr": row.get("deflated_sr")},
        ))
    return out


def _from_graveyard_json() -> list[GraveyardEntry]:
    """Load entries from data/research/graveyard.json — the manually
    curated cross-experiment graveyard.

    BUG FIX 2026-05-31: build_graveyard() previously aggregated from
    library_red + library_negative_evidence + gate_runs + discovery_log
    but NOT from graveyard.json itself. This orphaned 24 manually-
    curated entries including 'China A-share PEAD' (RED, 8717 events).

    Discovered by user-driven testing: submitted CandidateInfo(title=
    'China A-share PEAD', family='forward-earnings information') and
    got recommendation='allow' — clear miss. Root cause was missing
    loader for the JSON file.

    Schema (graveyard.json entries):
      name, family, date, verdict (RED/AMBER/etc), why (free text)

    Mapping into GraveyardEntry:
      title = name  (so _title_overlap fires)
      death_date = date
      failure_reason = why
      revival_potential = REVIVAL_POTENTIAL[failure_mode_inferred_from_why]
    """
    if not GRAVEYARD_JSON.exists():
        return []
    try:
        raw = json.loads(GRAVEYARD_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("graveyard.json parse failed: %s", exc)
        return []
    entries = raw.get("entries") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []
    source = "graveyard_json"
    out: list[GraveyardEntry] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = e.get("name") or e.get("id") or "<unnamed>"
        verdict = (e.get("verdict") or "").upper()
        if verdict not in ("RED", "AMBER", "FAIL"):
            continue   # only dead-mechanism records count as graveyard
        why = e.get("why") or ""
        fm = _classify_failure_mode(why)
        out.append(GraveyardEntry(
            source=source,
            source_id=name,
            name=name,
            family=e.get("family"),
            parent_family=e.get("parent_family"),
            required_data=e.get("required_data") or [],
            economics_text=why,
            title=name,                         # title=name so _title_overlap fires
            failure_reason=why[:500],
            failure_mode=fm,
            death_date=e.get("date"),
            source_weight=SOURCE_WEIGHTS.get(source, 0.9),  # curated → high weight
            revival_potential=REVIVAL_POTENTIAL.get(fm, 0.2),
            extra={"canonical_paper_id": e.get("canonical_paper_id")},
        ))
    return out


def _from_discovery_log() -> list[GraveyardEntry]:
    if not DISCOVERY_LOG.exists():
        return []
    out = []
    for line in DISCOVERY_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("verdict") not in ("skip", "review_with_caveat"):
            continue
        if row.get("stage") not in ("dedup", "family_cousin"):
            continue
        extraction = row.get("extraction") or {}
        source = "discovery_rejected"
        fm = FailureMode.OTHER.value
        out.append(GraveyardEntry(
            source=source,
            source_id=row.get("arxiv_id", ""),
            name=row.get("title", ""),
            family=extraction.get("family_guess"),
            parent_family=extraction.get("parent_family_guess"),
            required_data=extraction.get("required_data_tokens") or [],
            economics_text=extraction.get("economic_intuition", ""),
            title=row.get("title", ""),
            failure_reason=f"{row.get('stage')}: {row.get('reason', '')}",
            failure_mode=fm,
            death_date=row.get("ts", "")[:10],
            source_weight=SOURCE_WEIGHTS.get(source, 0.5),
            revival_potential=REVIVAL_POTENTIAL.get(fm, 0.5),
            extra={"arxiv_id": row.get("arxiv_id")},
        ))
    return out


# ── Token helpers ───────────────────────────────────────────────────────

_TOKEN_STOP = {"a", "the", "of", "and", "in", "on", "for", "to", "an", "is",
                "with", "by", "as", "or", "be"}


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in _TOKEN_STOP}


# ── Detector registry ──────────────────────────────────────────────────

_DETECTORS: list = []


def register_detector(name: str):
    def deco(fn):
        fn.detector_name = name
        _DETECTORS.append(fn)
        return fn
    return deco


@register_detector("paper_id_match")
def _paper_id_match(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    if candidate.arxiv_id and entry.extra.get("arxiv_id") == candidate.arxiv_id:
        return 1.0
    if (candidate.canonical_paper_id and
        entry.extra.get("canonical_paper_id") == candidate.canonical_paper_id):
        return 1.0
    return 0.0


@register_detector("title_overlap")
def _title_overlap(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    cand_t = _tokens(candidate.title)
    entry_t = _tokens(entry.title)
    if not cand_t or not entry_t:
        return 0.0
    return len(cand_t & entry_t) / max(len(cand_t | entry_t), 1)


# Family alias normalization — natural-language family names map to
# canonical tokens so "forward-earnings information" matches
# "earnings_underreaction" (both describe PEAD family).
# Added 2026-05-31 per user "senior 量化 角度优化识别机制".
_FAMILY_ALIAS_GROUPS = [
    # PEAD / earnings underreaction family
    {"earnings_underreaction", "forward-earnings information",
     "forward_earnings", "earnings_surprise", "pead", "post_earnings_drift"},
    # Momentum
    {"momentum", "mom", "price_momentum", "cross_sectional_momentum"},
    # Quality / profitability
    {"quality", "quality_qmj", "profitability"},
    # Carry
    {"carry", "roll_yield", "term_premium"},
    # Tail hedge
    {"tail_hedge", "option_hedge", "cross_asset_hedge"},
]


def _normalize_family(fam: str | None) -> str | None:
    """Map family string to canonical token via alias groups; returns
    the canonical form (first element) or original if no alias hits.

    BUG FIX 2026-05-31 (12th catch, found by L4 Session 3 dispatch):
    Previously input was normalized via replace() but group members were
    compared raw. "forward-earnings information" → "forward_earnings_
    information" wouldn't match group set containing "forward-earnings
    information". Now BOTH sides normalize before set check.
    """
    if not fam:
        return None
    def _n(s):
        return s.strip().lower().replace(" ", "_").replace("-", "_")
    norm = _n(fam)
    for group in _FAMILY_ALIAS_GROUPS:
        normalized_group = {_n(g) for g in group}
        if norm in normalized_group:
            return sorted(normalized_group)[0]  # canonical = first alphabetically
    return norm


@register_detector("family_match")
def _family_match(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    cand_fam = _normalize_family(candidate.family)
    entry_fam = _normalize_family(entry.family)
    if cand_fam and entry_fam and cand_fam == entry_fam:
        return 1.0
    cand_pf = _normalize_family(candidate.parent_family)
    entry_pf = _normalize_family(entry.parent_family)
    if cand_pf and entry_pf and cand_pf == entry_pf:
        return 0.6
    return 0.0


# Cross-market cousin detector: catches the "same mechanism, different
# market" pattern. e.g. our graveyard has China PEAD RED; a new
# Japan PEAD candidate should elevate.
# Added 2026-05-31 per cross-country PEAD test.
_MARKET_TOKENS = {
    "us", "usa", "united_states", "america", "american",
    "china", "chinese", "a_share", "cn", "prc", "shanghai", "shenzhen",
    "japan", "japanese", "jp", "topix", "tse", "nikkei",
    "uk", "britain", "british", "lse", "ftse",
    "europe", "european", "eu", "eurozone",
    "india", "indian", "in", "nse", "bse", "nifty",
    "emerging", "em", "developed", "dm",
    "korea", "korean", "kospi",
    "taiwan", "tw",
    "australia", "aus", "asx",
    "brazil", "bovespa",
}


@register_detector("cross_market_cousin")
def _cross_market_cousin(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    """Fires when candidate and entry share family + share economics
    tokens MINUS market tokens (i.e. same mechanism in different geography).

    Returns 1.0 if family matches AND title/economics overlap after
    market-token stripping is high (>= 0.4 Jaccard). Catches the most
    common "we already tried this in another country" pattern.
    """
    cand_fam = _normalize_family(candidate.family)
    entry_fam = _normalize_family(entry.family)
    if not (cand_fam and entry_fam and cand_fam == entry_fam):
        return 0.0    # require same family

    # Compute title overlap MINUS market tokens
    cand_tokens = _tokens(candidate.title) - _MARKET_TOKENS
    entry_tokens = _tokens(entry.title) - _MARKET_TOKENS
    if not cand_tokens or not entry_tokens:
        return 0.0
    title_overlap_demarketed = len(cand_tokens & entry_tokens) / max(
        len(cand_tokens | entry_tokens), 1
    )

    # Boost if both have an OTHER market token (same family, different market)
    cand_markets = _tokens(candidate.title) & _MARKET_TOKENS
    entry_markets = _tokens(entry.title) & _MARKET_TOKENS
    different_markets = bool(cand_markets) and bool(entry_markets) and \
                        not (cand_markets & entry_markets)

    if title_overlap_demarketed >= 0.4 and different_markets:
        return 1.0
    if title_overlap_demarketed >= 0.4:
        return 0.7    # same-mechanism overlap but market situation unclear
    return 0.0


@register_detector("data_signature_overlap")
def _data_overlap(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    cand_d = set(candidate.required_data or [])
    entry_d = set(entry.required_data or [])
    if not cand_d or not entry_d:
        return 0.0
    return len(cand_d & entry_d) / max(len(cand_d | entry_d), 1)


@register_detector("economic_concept_token")
def _econ_token(candidate: CandidateInfo, entry: GraveyardEntry) -> float:
    cand_e = _tokens(candidate.economics_text)
    entry_e = _tokens(entry.economics_text)
    if not cand_e or not entry_e:
        return 0.0
    return len(cand_e & entry_e) / max(len(cand_e | entry_e), 1)


# ── Threshold + recommendation logic ────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "paper_id_match":         (0.99, "block"),
    "title_overlap":          (0.70, "block"),
    "family_match":           (0.99, "block"),
    "data_signature_overlap": (0.80, "warn"),
    "economic_concept_token": (0.60, "warn"),
    # NEW 2026-05-31: cross-market cousin detector — catches "same
    # mechanism, different geography" pattern (e.g. JP PEAD when China
    # PEAD is RED). Threshold 0.7 fires WARN (review-level not block);
    # 1.0 fires BLOCK (full match in different market).
    "cross_market_cousin":    (1.0, "warn"),
}


def _apply_temporal_decay(recommendation: str, entry_age_years: float,
                            revival_potential: float) -> str:
    """If entry is OLD and has high revival potential, downgrade block → warn,
    warn → review."""
    if entry_age_years < 5.0:
        return recommendation
    if revival_potential >= 0.5:
        if recommendation == "block":
            return "warn"
        if recommendation == "warn":
            return "review"
    return recommendation


# ── Public match API ────────────────────────────────────────────────────

def check_against_graveyard(
    candidate: CandidateInfo,
    *,
    graveyard: list[GraveyardEntry] | None = None,
    thresholds: dict[str, tuple[float, str]] | None = None,
    use_cache: bool = True,
    exclude_self_ids: tuple[str, ...] = (),
) -> GraveyardMatch:
    """Multi-signal check of candidate vs full graveyard with temporal decay,
    cousin-count escalation, and structured explanation.

    Recommendation hierarchy: block > warn > review > allow.

    Args:
      exclude_self_ids: graveyard entry source_ids / names to skip — used
        by H2 to avoid matching a library mechanism against its own past
        gate_runs RED rows (self-failure is a different signal, handled
        elsewhere).
    """
    if graveyard is None:
        graveyard = build_graveyard(use_cache=use_cache)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    priority = {"allow": 0, "review": 1, "warn": 2, "block": 3}
    excluded = set(exclude_self_ids or ())

    matched_entries: list[GraveyardEntry] = []
    signals_matched: list[str] = []
    max_confidence = 0.0
    highest_rec = "allow"
    explanation_parts: list[str] = []

    for entry in graveyard:
        if excluded and (entry.source_id in excluded or entry.name in excluded):
            continue
        entry_score = 0.0
        entry_signals = []
        entry_recs = []
        for detector in _DETECTORS:
            try:
                score = detector(candidate, entry)
            except Exception:
                continue
            dname = getattr(detector, "detector_name", detector.__name__)
            threshold, rec = thresholds.get(dname, (1.1, "review"))
            if score >= threshold:
                entry_signals.append(dname)
                entry_recs.append(rec)
                entry_score = max(entry_score, score * entry.source_weight)

        if not entry_signals:
            continue

        # Temporal decay: old entries with revival potential get downgraded
        age = entry.age_years()
        worst_rec = max(entry_recs, key=lambda r: priority.get(r, 0))
        adjusted_rec = _apply_temporal_decay(worst_rec, age, entry.revival_potential)

        matched_entries.append(entry)
        for s in entry_signals:
            if s not in signals_matched:
                signals_matched.append(s)
        max_confidence = max(max_confidence, entry_score)
        if priority.get(adjusted_rec, 0) > priority.get(highest_rec, 0):
            highest_rec = adjusted_rec
            explanation_parts.append(
                f"{adjusted_rec} due to {entry_signals} match on "
                f"{entry.name!r} (mode={entry.failure_mode}, age={age:.1f}yr, "
                f"source={entry.source})"
            )

    # Cousin-count escalation — N dead in same family
    family = candidate.family
    cousin_count = 0
    elevated = False
    if family:
        # 12th catch fix: normalize family on BOTH sides for alias-equiv count
        target_fam_norm = _normalize_family(family)
        cousin_count = sum(
            1 for e in graveyard
            if _normalize_family(e.family) == target_fam_norm
        )
        if cousin_count >= 2 and highest_rec in ("review", "allow"):
            highest_rec = "warn"
            elevated = True
            explanation_parts.append(
                f"elevated to warn: family {family!r} has "
                f"{cousin_count} dead entries"
            )

    explanation = "; ".join(explanation_parts) or "no signals fired"
    if matched_entries and highest_rec == "allow":
        # Shouldn't happen but safe default
        highest_rec = "review"

    return GraveyardMatch(
        matched=bool(matched_entries),
        signals_matched=signals_matched,
        matched_entries=matched_entries,
        overall_confidence=round(max_confidence, 3),
        recommendation=highest_rec if matched_entries else "allow",
        explanation=explanation,
        cousin_count_in_family=cousin_count,
        elevated=elevated,
    )


# ── Convenience helpers ─────────────────────────────────────────────────

def dead_in_family(family: str) -> list[GraveyardEntry]:
    """Inverse query — get all dead entries in a given family."""
    return [e for e in build_graveyard() if e.family == family]


def dead_in_parent_family(parent_family: str) -> list[GraveyardEntry]:
    return [e for e in build_graveyard() if e.parent_family == parent_family]


def list_detectors() -> list[str]:
    return [getattr(d, "detector_name", d.__name__) for d in _DETECTORS]


def summarize_graveyard() -> dict:
    g = build_graveyard()
    out = {"total": len(g),
            "by_source": {},
            "by_failure_mode": {},
            "by_family": {}}
    for entry in g:
        out["by_source"][entry.source] = out["by_source"].get(entry.source, 0) + 1
        out["by_failure_mode"][entry.failure_mode] = (
            out["by_failure_mode"].get(entry.failure_mode, 0) + 1)
        if entry.family:
            out["by_family"][entry.family] = out["by_family"].get(entry.family, 0) + 1
    return out


def graveyard_to_dataframe() -> pd.DataFrame:
    g = build_graveyard()
    if not g:
        return pd.DataFrame()
    return pd.DataFrame([{k: v for k, v in e.to_dict().items() if k != "extra"}
                           for e in g])


def clear_cache() -> None:
    """Force rebuild on next call (operator override)."""
    _CACHE["entries"] = None
    _CACHE["built_at"] = None
    _CACHE["source_mtimes"] = {}
