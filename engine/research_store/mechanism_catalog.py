"""engine.research_store.mechanism_catalog — joined view per (family, signal_type).

F13.1 (2026-06-05): the substrate for A+B automation. Before any L4 cron
auto-tests a candidate, it consults this catalog to check whether the
(family, signal_type) cluster already carries a RED verdict, a deployed
sleeve, or other prior evidence that should gate / inform the run.

Designed as a JOIN-AT-VIEW-TIME aggregator over 6 sources of truth:

    library YAML        (data/research/mechanism_library/*.yaml)
      → deployed sleeves, status_in_our_book, parent_family

    papers_registry     (data/research_store/papers_registry.jsonl)
      → published evidence count per cluster

    hypotheses          (data/research_store/hypotheses.jsonl)
      → extracted testable proposals per cluster

    hypothesis_specs    (data/research_store/hypothesis_specs.jsonl)
      → typed specs, claim_type, composer_status

    research_store      (data/research_store/events.jsonl)
      → factor_verdict_filed: RED / GREEN / MARGINAL outcomes

    composer.contract   (in-memory registry)
      → is_spec_covered for buildability check

No new persistent state. Every read recomputes. This keeps the catalog
honest with respect to the upstream stores at the cost of ~100ms per
call (negligible vs the 30-60s LLM costs A+B will amortize against).

Granularity
-----------
Two views:
  - LEVEL 2 (default): per (family, signal_type) row. Captures the
    distinct mechanisms within a family (CARRY_FORWARD_DISCOUNT vs
    CARRY_ROLL_YIELD are economically different even though both CARRY).
  - LEVEL 1: per-family aggregate. For high-level UI / chat-ask /
    family-level white-space analysis.

Cross-source name normalization
-------------------------------
Library YAML uses lowercase ad-hoc family names ('carry',
'earnings_underreaction', 'tsmom'). HypothesisSpec uses FamilyV2 enum
('CARRY', 'EARNINGS_DRIFT', 'MOMENTUM'). _LIB_FAMILY_TO_TYPED maps
between them; _normalize_family is the lookup helper.

Designed to be CONSUMED by:
  - F13.2 reports (redundancy / whitespace / convergence)
  - F14a/b autopilot cron (pre-flight gate)
  - chat_ask RAG (semantic + structured join)
  - direction_proposer (enrich existing ranking with cluster context)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIBRARY_DIR = _REPO_ROOT / "data" / "research" / "mechanism_library"
_PAPERS_PATH = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"
_HYP_PATH = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"


# Library YAML family names vary in style. Map to FamilyV2 canonical UPPER.
# Add new entries here as new families appear in library/spec corpus.
_LIB_FAMILY_TO_TYPED: dict[str, str] = {
    "carry":                 "CARRY",
    "tsmom":                 "MOMENTUM",            # time-series momentum
    "momentum":              "MOMENTUM",
    "cross_asset_hedge":     "OTHER",
    "factor_hedge":          "OTHER",
    "quality":               "QUALITY",
    "earnings_underreaction": "EARNINGS_DRIFT",
    "residual_momentum":     "MOMENTUM",
}


def _normalize_family(name: str) -> str:
    """Best-effort: lower → typed UPPER via _LIB_FAMILY_TO_TYPED, else upper."""
    if not name:
        return "OTHER"
    low = name.lower().strip()
    if low in _LIB_FAMILY_TO_TYPED:
        return _LIB_FAMILY_TO_TYPED[low]
    # Already typed-style (CARRY, MOMENTUM, ...)
    return name.upper().strip()


@dataclass(frozen=True)
class MechanismRow:
    """One catalog row. Either a (family, signal_type) cluster row or a
    family-aggregate row (signal_type=None).

    A row is a JOIN, not a stored record — see module docstring for sources.
    """
    family:      str
    signal_type: Optional[str]      # None for family-level aggregate

    # Volumes
    n_papers:           int
    n_hypotheses:       int
    n_specs_typed:      int         # claim_type=FACTOR_HYPOTHESIS
    n_specs_ready:      int         # composer is_spec_covered=True
    n_red_verdicts:     int
    n_green_verdicts:   int
    n_deployed_sleeves: int

    # Subject lists (capped for log size; full lists via separate query)
    deployed_sleeve_ids:   tuple[str, ...]
    red_subject_ids:       tuple[str, ...]
    green_subject_ids:     tuple[str, ...]

    # Library status histogram
    library_status_distribution: dict[str, int]

    # Derived gating signals (F13.2 inputs)
    exploration_depth:    float     # n_tested / max(1, n_hypotheses), 0..1
    has_white_space:      bool      # papers/hyps proposed, nothing tested or deployed
    has_redundancy_risk:  bool      # ≥1 RED here AND ≥1 untested-hypothesis here
    is_actively_deployed: bool      # ≥1 DEPLOYED sleeve

    computed_ts: str


# ── Loaders ──────────────────────────────────────────────────────────


def _load_library_rows() -> list[dict]:
    """Return one dict per library YAML, normalized to {id, family, parent_family,
    status_in_our_book, purpose}."""
    if not _LIBRARY_DIR.is_dir():
        return []
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML missing — library YAML join disabled")
        return []
    out = []
    for p in sorted(_LIBRARY_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("library yaml %s parse failed: %s", p.name, exc)
            continue
        out.append({
            "id":                 d.get("id") or p.stem,
            "family":             _normalize_family(d.get("family", "")),
            "parent_family":      d.get("parent_family") or "",
            "status_in_our_book": d.get("status_in_our_book") or "UNKNOWN",
            "purpose":            d.get("purpose") or "",
            "yaml_path":          str(p.relative_to(_REPO_ROOT)),
        })
    return out


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _latest_specs_by_hyp() -> dict[str, dict]:
    """Per source_hypothesis_id, return the latest spec dict (by
    extraction.extracted_ts). Mirrors latest_for() but operates on raw
    dicts so we don't import HypothesisSpec dataclass overhead per row."""
    specs_path = _REPO_ROOT / "data" / "research_store" / "hypothesis_specs.jsonl"
    by_hyp: dict[str, dict] = {}
    latest_ts: dict[str, str] = {}
    for d in _load_jsonl(specs_path):
        hid = d.get("source_hypothesis_id", "")
        if not hid:
            continue
        ts = (d.get("extraction") or {}).get("extracted_ts", "") or d.get("created_ts", "")
        if hid not in latest_ts or ts > latest_ts[hid]:
            latest_ts[hid] = ts
            by_hyp[hid] = d
    return by_hyp


def _factor_verdicts() -> list[dict]:
    """RED / GREEN / MARGINAL factor_verdict_filed events from store."""
    try:
        from engine.research_store import store
        evs = store.filter_events(event_type="factor_verdict_filed", limit=2000)
    except Exception:
        return []
    out: list[dict] = []
    for ev in evs:
        out.append({
            "event_id":    ev.event_id,
            "subject_id":  ev.subject_id,
            "family":      _normalize_family(ev.family or ""),
            "verdict":     ev.verdict.value if hasattr(ev.verdict, "value") else str(ev.verdict),
            "ts":          ev.ts,
            "metrics":     ev.metrics or {},
            "summary":     ev.summary or "",
        })
    return out


def _composer_ready_for(spec_dict: dict) -> bool:
    """Best-effort coverage check on a spec dict. Returns False on any
    error (conservative)."""
    try:
        from engine.hypothesis_spec.schema import HypothesisSpec
        from engine.composer.contract import is_spec_covered
        spec = HypothesisSpec.from_dict(spec_dict)
        covered, _gaps = is_spec_covered(spec)
        return bool(covered)
    except Exception:
        return False


# ── Builder ──────────────────────────────────────────────────────────


def build_catalog() -> list[MechanismRow]:
    """Build the level-2 (family, signal_type) catalog. Returns one row per
    cluster that has ANY content (≥1 paper / hyp / spec / verdict / sleeve).
    Clusters with zero content across all sources are omitted — they're
    enumerated separately in white_space_cells()."""
    library_rows = _load_library_rows()
    paper_rows = _load_jsonl(_PAPERS_PATH)
    hyp_rows = _load_jsonl(_HYP_PATH)
    specs_by_hyp = _latest_specs_by_hyp()
    verdicts = _factor_verdicts()

    # Group keys
    Key = tuple   # (family, signal_type_or_None)

    # 1. Library: keyed by family ONLY (signal_type not in YAML)
    library_by_family: dict[str, list[dict]] = defaultdict(list)
    for L in library_rows:
        library_by_family[L["family"]].append(L)

    # 2. Hypotheses: keyed by mechanism_family (from typed extractor pipeline)
    hyp_by_family: dict[str, list[dict]] = defaultdict(list)
    for h in hyp_rows:
        fam = _normalize_family(h.get("mechanism_family", ""))
        hyp_by_family[fam].append(h)

    # 3. Specs: keyed by (family, signal_type). Walk latest per hyp.
    spec_by_key: dict[Key, list[dict]] = defaultdict(list)
    spec_ready_by_key: dict[Key, int] = defaultdict(int)
    spec_typed_by_key: dict[Key, int] = defaultdict(int)
    for hid, s in specs_by_hyp.items():
        fam = _normalize_family(s.get("family", ""))
        legs = s.get("legs") or []
        signal = (legs[0].get("signal_type") if legs else None) or "UNKNOWN"
        ct = s.get("claim_type", "")
        key = (fam, signal)
        spec_by_key[key].append(s)
        if ct == "FACTOR_HYPOTHESIS":
            spec_typed_by_key[key] += 1
            if _composer_ready_for(s):
                spec_ready_by_key[key] += 1

    # 4. Verdicts: keyed by family (signal_type usually not in event)
    verdict_by_family: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        verdict_by_family[v["family"]].append(v)

    # 5. Paper-to-family attribution is via paper_id → hypotheses' family.
    # A paper "touches" a family if ≥1 of its hypotheses lands there.
    paper_families: dict[str, set] = defaultdict(set)
    for h in hyp_rows:
        pid = h.get("source_paper_id", "")
        if pid:
            paper_families[pid].add(_normalize_family(h.get("mechanism_family", "")))

    n_papers_by_family: dict[str, int] = Counter()
    for pid, fams in paper_families.items():
        for fam in fams:
            n_papers_by_family[fam] += 1

    # ── Compose rows ──
    # spec_by_key already has all (family, signal_type) cells with a spec.
    # For families with hypotheses but no spec (or library entry but no
    # spec), add a (family, None) aggregate cell.
    all_keys: set[Key] = set(spec_by_key.keys())
    fams_with_specs = {k[0] for k in all_keys}
    fams_with_any = (set(library_by_family) | set(hyp_by_family)
                     | set(verdict_by_family) | set(n_papers_by_family))
    for fam in fams_with_any - fams_with_specs:
        all_keys.add((fam, None))

    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[MechanismRow] = []
    for (fam, sig) in sorted(all_keys, key=lambda k: (k[0], k[1] or "")):
        # Per-key counts
        n_specs_typed_k = spec_typed_by_key.get((fam, sig), 0)
        n_specs_ready_k = spec_ready_by_key.get((fam, sig), 0)
        n_papers_k = n_papers_by_family.get(fam, 0)
        n_hyps_k = len(hyp_by_family.get(fam, []))
        v_list = verdict_by_family.get(fam, [])
        n_red = sum(1 for v in v_list if v["verdict"] == "RED")
        n_green = sum(1 for v in v_list if v["verdict"] == "GREEN")
        red_ids = tuple(v["subject_id"] for v in v_list if v["verdict"] == "RED")[:20]
        green_ids = tuple(v["subject_id"] for v in v_list if v["verdict"] == "GREEN")[:20]

        lib_list = library_by_family.get(fam, [])
        deployed_ids = tuple(L["id"] for L in lib_list
                              if L["status_in_our_book"] == "DEPLOYED")
        status_dist = dict(Counter(L["status_in_our_book"] for L in lib_list))

        n_deployed = len(deployed_ids)
        is_deployed = n_deployed > 0
        exploration = (n_red + n_green) / max(1, n_hyps_k)
        white_space = (n_papers_k > 0 or n_hyps_k > 0) and (n_red + n_green + n_deployed) == 0
        redundancy = (n_red > 0) and (n_hyps_k > n_red + n_green)

        rows.append(MechanismRow(
            family            = fam,
            signal_type       = sig,
            n_papers          = n_papers_k,
            n_hypotheses      = n_hyps_k,
            n_specs_typed     = n_specs_typed_k,
            n_specs_ready     = n_specs_ready_k,
            n_red_verdicts    = n_red,
            n_green_verdicts  = n_green,
            n_deployed_sleeves= n_deployed,
            deployed_sleeve_ids = deployed_ids,
            red_subject_ids   = red_ids,
            green_subject_ids = green_ids,
            library_status_distribution = status_dist,
            exploration_depth = round(exploration, 3),
            has_white_space   = white_space,
            has_redundancy_risk = redundancy,
            is_actively_deployed = is_deployed,
            computed_ts       = now_iso,
        ))
    return rows


# ── F13.2 Reports ────────────────────────────────────────────────────


@dataclass(frozen=True)
class RedundancyMatch:
    """One catalog cluster a new spec matches that already carries REDs.

    Surfaces "you've already RED'd this in disguise" — the pre-flight
    gate for F14 cron and the alert overlay for the candidate UI.
    """
    family:             str
    signal_type:        Optional[str]
    n_red_in_cluster:   int
    red_subject_ids:    tuple[str, ...]
    n_deployed_in_cluster: int
    advice:             str        # human-readable next step


def find_redundancy_for_spec(spec_dict: dict,
                              catalog: Optional[list[MechanismRow]] = None
                              ) -> list[RedundancyMatch]:
    """For a NEW spec dict (extracted but not yet tested), find catalog
    clusters with prior RED verdicts that the spec falls into.

    Used by:
      - F14 cron pre-flight: skip auto-test if cluster has RED + spec
        doesn't materially differ
      - candidate dropdown UI: warning badge on row
      - PROMOTE decision page: "this candidate's cluster has N prior REDs"
    """
    fam = _normalize_family(spec_dict.get("family", ""))
    legs = spec_dict.get("legs") or []
    sig = (legs[0].get("signal_type") if legs else None) or "UNKNOWN"
    if catalog is None:
        catalog = build_catalog()
    matches: list[RedundancyMatch] = []
    for r in catalog:
        # Exact (family, signal_type) match first
        if r.family != fam:
            continue
        # Either signal match or family-level fallback (sig=None row)
        signal_match = (r.signal_type == sig) or (r.signal_type is None)
        if not signal_match:
            continue
        if r.n_red_verdicts == 0:
            continue
        if r.n_red_verdicts >= 3:
            advice = ("STRONG WARN — cluster has 3+ REDs; do NOT auto-test, "
                      "re-read REDs first")
        elif r.n_red_verdicts >= 1 and r.n_deployed_sleeves == 0:
            advice = ("WARN — cluster has REDs and no deployed sleeve; "
                      "likely already-killed mechanism")
        else:
            advice = "INFO — cluster has REDs alongside a deployed sleeve; verify novelty"
        matches.append(RedundancyMatch(
            family               = r.family,
            signal_type          = r.signal_type,
            n_red_in_cluster     = r.n_red_verdicts,
            red_subject_ids      = r.red_subject_ids,
            n_deployed_in_cluster= r.n_deployed_sleeves,
            advice               = advice,
        ))
    return matches


def white_space_cells(min_papers: int = 1,
                       catalog: Optional[list[MechanismRow]] = None
                       ) -> list[MechanismRow]:
    """Cells with extractor-surfaced interest (papers / hyps) but zero
    testing or deployment activity. The "engineering target list" — each
    is a cluster where building a Composer component or running a pipeline
    would close a research gap.

    Sorted by n_papers desc (papers most pointing at the gap come first).
    Filter min_papers cutoff to surface only the "real" gaps (not single-
    paper outliers).
    """
    if catalog is None:
        catalog = build_catalog()
    out = [r for r in catalog if r.has_white_space and r.n_papers >= min_papers]
    out.sort(key=lambda r: (-r.n_papers, -r.n_hypotheses, r.family))
    return out


def convergence_clusters(min_papers: int = 3,
                          catalog: Optional[list[MechanismRow]] = None
                          ) -> list[MechanismRow]:
    """Cells where multiple independent papers point at the same
    mechanism, there's no deployed sleeve, AND at least one spec is
    composer-ready. These are the highest-EV auto-test candidates —
    multiple-paper convergence is real-prior evidence the mechanism
    exists; no deployed sleeve means low book-correlation risk;
    composer-ready means F14 can actually test tonight.

    F14 cron should prioritize these. Sort by (n_papers, n_specs_ready) desc.
    """
    if catalog is None:
        catalog = build_catalog()
    out = [r for r in catalog
           if r.n_papers >= min_papers
           and r.n_deployed_sleeves == 0
           and r.n_specs_ready >= 1]
    out.sort(key=lambda r: (-r.n_papers, -r.n_specs_ready, r.family))
    return out


def build_catalog_family_aggregates() -> list[MechanismRow]:
    """Level-1 aggregates: one row per family across all signal_types.
    Useful for high-level UI cells, chat_ask family-level questions, and
    white-space-by-family analysis."""
    fine = build_catalog()
    by_family: dict[str, list[MechanismRow]] = defaultdict(list)
    for r in fine:
        by_family[r.family].append(r)
    now_iso = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[MechanismRow] = []
    for fam, rs in sorted(by_family.items()):
        # Use first row's "global per-family" fields (papers, verdicts, lib
        # status, deployed sleeves) — those are already family-level invariant.
        head = rs[0]
        n_specs_typed = sum(r.n_specs_typed for r in rs)
        n_specs_ready = sum(r.n_specs_ready for r in rs)
        # exploration as max across signal_types (best within family)
        explor = round(max(r.exploration_depth for r in rs), 3)
        white_space = all(r.has_white_space for r in rs) and head.n_red_verdicts == 0
        redundancy  = any(r.has_redundancy_risk for r in rs)
        out.append(MechanismRow(
            family            = fam,
            signal_type       = None,
            n_papers          = head.n_papers,
            n_hypotheses      = head.n_hypotheses,
            n_specs_typed     = n_specs_typed,
            n_specs_ready     = n_specs_ready,
            n_red_verdicts    = head.n_red_verdicts,
            n_green_verdicts  = head.n_green_verdicts,
            n_deployed_sleeves= head.n_deployed_sleeves,
            deployed_sleeve_ids = head.deployed_sleeve_ids,
            red_subject_ids   = head.red_subject_ids,
            green_subject_ids = head.green_subject_ids,
            library_status_distribution = head.library_status_distribution,
            exploration_depth = explor,
            has_white_space   = white_space,
            has_redundancy_risk = redundancy,
            is_actively_deployed = head.is_actively_deployed,
            computed_ts       = now_iso,
        ))
    return out
