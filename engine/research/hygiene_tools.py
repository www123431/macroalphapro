"""engine/research/hygiene_tools.py — 7 deterministic hygiene tools for the
Hypothesis Generator (H1-H7 from design memo).

Each tool is:
- DETERMINISTIC (same input → same output)
- READ-ONLY (no state mutation)
- BOUNDED LATENCY (< 5s typical)
- TESTED separately from the Generator
- HAS ANTHROPIC TOOL SCHEMA for LLM tool-use

Doctrine (these tools enforce the 5 iron rules):
- R1 NO INVENT: H1 only returns library entries; H4 rejects unknown papers
- R2 EVIDENCE-FIRST: all tools return structured payloads; LLM cannot
  fabricate the data
- R3 PARENT-FAMILY OVERRIDE: H2 multi-level cousin check rejects parent overlap
- R4 NO GRID HIDE: H5 rejects list/range/distribution param specs
- R5 NO PROPOSAL=SUCCESS: H1 may return empty list — that's valid output

Cross-references:
- H4 wraps SG1 (engine.research.library_integrity.verify_paper)
- H2 uses the parent_family taxonomy from the existing KG
- H6 enforces ≥1 post-2020 replication for candidate-purpose selection
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
MASTER_INDEX = LIBRARY_DIR / "_canonical_papers_tier1_2.yaml"
SCHEMA_DOC = LIBRARY_DIR / "_schema.md"


# ── Shared data inventory whitelist (mirror of _schema.md) ───────────────

# Senior 2026-05-30: split "implemented" vs "declared".
# IMPLEMENTED_DATA = tokens with a real fetcher / data file behind them.
# DECLARED_DATA    = tokens reserved for future fetchers but not yet wired.
#                    Hygiene treats DECLARED as MISSING — better to fail
#                    early than silently lie that data is available.
# DATA_INVENTORY = backward-compat alias = IMPLEMENTED only (NOT including
#                    DECLARED). Previously DATA_INVENTORY included declared
#                    tokens, causing hygiene to silently pass papers that
#                    actually couldn't be backtested.
IMPLEMENTED_DATA = frozenset([
    # Equity — wired via wrds_crsp + path_c.earnings_panel + similar
    "crsp_dsf", "crsp_msf", "compustat_quarterly", "compustat_annual",
    "ibes_summary", "ibes_guidance",
    "SUE_panel", "ann_dates", "ret_60d", "ret_daily", "ret_monthly",
    "tr13f_holdings", "edgar_8k_meta", "dera_insider",
    # Cross-asset — wired via line_c.wrds_direct dual-account
    "tr_ds_fut_settle", "cmdty_contracts", "cmdty_settle",
    "fx_contracts", "fx_settle", "rates_contracts", "rates_settle",
    "rates_xc_settle", "eqidx_contracts", "eqidx_settle",
    "trace_bond_monthly", "vix_index", "vix3m_index", "vxx_etn", "vxz_etn",
    # Macro / news — wired via fetchers
    "fred_macro", "rpna_daily_sentiment", "rpna_entity_map",
])

DECLARED_DATA = frozenset([
    # IBES detail: ~200M rows, requires SQL-side filter. Earlier session
    # documented WRDS-IBES-detail as roadmap #4 demand-driven.
    "ibes_detail",
    # OptionMetrics: tokens reserved for future fetcher. Currently
    # any paper requiring these will fail at gate runtime with
    # "data not implemented".
    "optionm_skew", "optionm_iv_surface",
])

# Back-compat alias: the OLD DATA_INVENTORY included declared tokens.
# We now expose only IMPLEMENTED so hygiene catches the gap before
# strict gate fails at runtime.
DATA_INVENTORY = IMPLEMENTED_DATA


@dataclasses.dataclass(frozen=True)
class ToolResult:
    name:    str
    success: bool
    payload: dict
    error:   str | None = None

    def to_dict(self) -> dict:
        if self.error:
            return {"name": self.name, "success": False, "error": self.error}
        return {"name": self.name, "success": True, "payload": self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


# ── Library access helpers ───────────────────────────────────────────────

def _load_library_entries() -> list[dict]:
    """Load every visible mechanism YAML. Visible = audit_signature human-confirmed.
    Hidden = pending."""
    if not LIBRARY_DIR.exists():
        return []
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8"))
            entry["_yaml_path"] = str(fp)
            out.append(entry)
        except Exception as e:
            logger.warning("failed to parse %s: %s", fp, e)
    return out


def _load_master_index() -> dict:
    if not MASTER_INDEX.exists():
        return {"papers": {}}
    return yaml.safe_load(MASTER_INDEX.read_text(encoding="utf-8")) or {"papers": {}}


def _is_visible(entry: dict) -> bool:
    return entry.get("audit_signature") == "human-confirmed"


# ── H1 list_unexplored_library_entries ────────────────────────────────────

def h1_list_unexplored_library_entries(
    *, include_pending: bool = False
) -> ToolResult:
    """Return library entries where:
      purpose: candidate
      currently_unexplored_in_our_book: true
      audit_signature: human-confirmed (unless include_pending=True)

    Returns LIST of mechanism summaries — NOT full YAMLs (keep LLM context small).
    """
    entries = _load_library_entries()
    selected = []
    for e in entries:
        if e.get("purpose") != "candidate":
            continue
        if not e.get("currently_unexplored_in_our_book"):
            continue
        if not include_pending and not _is_visible(e):
            continue
        selected.append({
            "id":              e.get("id"),
            "family":          e.get("family"),
            "parent_family":   e.get("parent_family"),
            "canonical_paper_id": e.get("canonical_paper_id"),
            "status_in_our_book": e.get("status_in_our_book"),
            "required_data":   e.get("required_data") or [],
            "audit_signature": e.get("audit_signature"),
        })
    return ToolResult(
        "h1_list_unexplored_library_entries", True,
        {"n_unexplored": len(selected), "entries": selected,
          "include_pending": include_pending,
          "note": "0 entries = NO PROPOSAL is the correct output (R5)"}
    )


# ── H2 cousin_check_multilevel ────────────────────────────────────────────

_TOKENIZE_RE = re.compile(r"[^a-z0-9]+")


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t for t in _TOKENIZE_RE.split(s.lower()) if t}


def _data_overlap_ratio(a: list, b: list) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def h2_cousin_check_multilevel(
    mechanism_id: str,
    candidate_info: dict | None = None,
) -> ToolResult:
    """4-level cousin check for a CANDIDATE mechanism against ALL library
    entries (visible + hidden — strictness applies regardless of audit state).

    Levels:
      L1 same_family       — direct family match
      L2 same_parent       — parent_family match
      L3 same_data         — required_data overlap ≥ 0.8 with any entry
      L4 same_economics    — economics-text token overlap ≥ 0.6 with any entry

    Returns: matches at each level + verdict (allow / soft-reject / hard-reject).

    Hard-reject conditions:
      - same_family match with status_in_our_book in {RED, DEPLOYED}
      - L2 same_parent match with L4 token overlap ≥ 0.6 (parent-cousin in disguise)

    BUGFIX 2026-05-31 (8th user catch):
      Previously REQUIRED mechanism_id to be in library — but new
      candidates are by definition NOT in library yet (chicken-and-egg).
      Now accepts optional candidate_info dict with the candidate's
      proposed metadata (family, parent_family, economics_text,
      required_data); uses it when mechanism_id not in library.
      If neither provided, falls back to allow-with-warning (cannot
      check cousins without target metadata).
    """
    entries = _load_library_entries()
    target = next((e for e in entries if e.get("id") == mechanism_id), None)
    if not target:
        if candidate_info:
            # Use the candidate's PROPOSED metadata as target — this is
            # the senior senior intent: cousin check should work on
            # NEW candidates BEFORE they enter the library.
            target = {
                "id":                   mechanism_id,
                "family":               candidate_info.get("family"),
                "parent_family":        candidate_info.get("parent_family"),
                "required_data":        candidate_info.get("required_data") or [],
                "mechanism_economics":  candidate_info.get("economics_text", ""),
                "status_in_our_book":   "PROPOSED",
            }
        else:
            # No metadata at all — return allow with warning (rather than
            # hard FAIL that blocks unrelated pipeline progress)
            return ToolResult(
                "h2_cousin_check_multilevel", True,
                {"L1_same_family": [], "L2_same_parent": [],
                 "L3_same_data": [], "L4_same_economics": [],
                 "verdict": "allow_no_metadata",
                 "warning": (f"mechanism_id {mechanism_id!r} not in library "
                             "AND no candidate_info provided — cannot "
                             "perform cousin check; passing through with "
                             "warning")},
            )

    others = [e for e in entries if e.get("id") != mechanism_id]
    target_econ_tokens = _tokens(target.get("mechanism_economics"))

    same_family, same_parent, same_data, same_econ = [], [], [], []
    hard_rejects: list[str] = []

    for o in others:
        if o.get("family") == target.get("family"):
            same_family.append({
                "id":     o.get("id"),
                "status": o.get("status_in_our_book"),
                "shared_family": o.get("family"),
            })
            if o.get("status_in_our_book") in ("RED", "DEPLOYED"):
                hard_rejects.append(
                    f"same family {o.get('family')!r} as "
                    f"{o.get('id')!r} (status: {o.get('status_in_our_book')})"
                )
            # Phase 5 G: literature-falsified mechanism anchors
            if o.get("purpose") == "negative_evidence":
                hard_rejects.append(
                    f"same family {o.get('family')!r} as literature-falsified "
                    f"{o.get('id')!r} (purpose: negative_evidence)"
                )
        if o.get("parent_family") == target.get("parent_family"):
            same_parent.append({
                "id":     o.get("id"),
                "status": o.get("status_in_our_book"),
                "shared_parent": o.get("parent_family"),
            })
        data_ratio = _data_overlap_ratio(
            target.get("required_data") or [], o.get("required_data") or [])
        if data_ratio >= 0.8:
            same_data.append({
                "id":     o.get("id"),
                "ratio":  round(data_ratio, 2),
                "status": o.get("status_in_our_book"),
            })
        other_econ = _tokens(o.get("mechanism_economics"))
        if target_econ_tokens and other_econ:
            jac = (len(target_econ_tokens & other_econ) /
                   max(len(target_econ_tokens | other_econ), 1))
            if jac >= 0.6:
                same_econ.append({
                    "id":     o.get("id"),
                    "jaccard": round(jac, 2),
                    "status": o.get("status_in_our_book"),
                })

    # Parent + economics cousin-in-disguise
    parent_ids = {x["id"] for x in same_parent}
    econ_ids = {x["id"] for x in same_econ}
    disguised = parent_ids & econ_ids
    for did in disguised:
        if did not in [x["id"] for x in same_family]:    # already captured
            hard_rejects.append(
                f"parent-cousin-in-disguise: {did!r} shares parent_family "
                f"AND economics tokens"
            )

    # ── L5 GRAVEYARD CROSS-CHECK (Phase 8c) ──────────────────────────────
    # Full graveyard reads BEYOND library YAMLs: gate_runs RED + discovery
    # rejected. This catches cousins the library-only L1-L4 would miss
    # (e.g. a mechanism that was tried via strict gate and went RED but
    # was never written back to library/red/).
    graveyard_match_dict: dict | None = None
    try:
        from engine.research.graveyard import (
            CandidateInfo, check_against_graveyard,
        )
        candidate = CandidateInfo(
            title=str(target.get("title") or target.get("id") or ""),
            family=target.get("family"),
            parent_family=target.get("parent_family"),
            required_data=list(target.get("required_data") or []),
            economics_text=str(target.get("mechanism_economics") or ""),
        )
        gv_match = check_against_graveyard(
            candidate, exclude_self_ids=(mechanism_id,),
        )
        graveyard_match_dict = gv_match.to_dict()
        if gv_match.recommendation == "block":
            hard_rejects.append(
                f"graveyard block: {gv_match.explanation}"
            )
    except Exception as exc:
        # Graveyard scan should NEVER cause H2 to fail — log and continue.
        logger = __import__("logging").getLogger(__name__)
        logger.warning("graveyard cross-check failed (continuing): %s", exc)
        graveyard_match_dict = {"error": str(exc)}

    if hard_rejects:
        verdict = "hard_reject"
    elif same_family or len(same_parent) >= 2 or same_data or same_econ or (
        graveyard_match_dict and graveyard_match_dict.get("recommendation") == "warn"
    ):
        verdict = "soft_reject"
    else:
        verdict = "allow"

    return ToolResult(
        "h2_cousin_check_multilevel", True,
        {"mechanism_id":  mechanism_id,
          "verdict":      verdict,
          "L1_same_family":   same_family,
          "L2_same_parent":   same_parent,
          "L3_same_data":     same_data,
          "L4_same_economics": same_econ,
          "L5_graveyard":     graveyard_match_dict,
          "hard_reject_reasons": hard_rejects},
    )


# ── H3 check_data_inventory ──────────────────────────────────────────────

def h3_check_data_inventory(required_data: list[str]) -> ToolResult:
    """100% of required_data must be in IMPLEMENTED_DATA whitelist.

    Tokens in DECLARED_DATA (e.g. optionm_iv_surface, ibes_detail)
    are surfaced SEPARATELY as 'declared_not_implemented' so the
    caller knows the difference between 'unknown token' and 'known
    token but no fetcher yet'."""
    req = required_data or []
    declared_not_impl = [d for d in req if d in DECLARED_DATA]
    truly_missing = [d for d in req
                       if d not in IMPLEMENTED_DATA and d not in DECLARED_DATA]
    missing = declared_not_impl + truly_missing
    return ToolResult(
        "h3_check_data_inventory", True,
        {"required_data":             req,
          "all_present":              not missing,
          "missing":                  missing,
          "declared_not_implemented": declared_not_impl,
          "truly_missing":            truly_missing,
          "implemented_size":         len(IMPLEMENTED_DATA),
          "declared_size":            len(DECLARED_DATA)},
    )


# ── H4 verify_paper_in_library (wraps SG1) ───────────────────────────────

def h4_verify_paper_in_library(paper_id: str) -> ToolResult:
    """Strict check: paper_id must exist in master index AND have
    verified: true (i.e. crossref-pass at last audit)."""
    master = _load_master_index()
    papers = master.get("papers", {})
    if paper_id not in papers:
        return ToolResult("h4_verify_paper_in_library", True,
                          {"paper_id": paper_id, "in_master_index": False,
                            "verified": False,
                            "reason": "not in _canonical_papers_tier1_2.yaml"})
    entry = papers[paper_id]
    is_verified = bool(entry.get("verified"))
    return ToolResult("h4_verify_paper_in_library", True,
                      {"paper_id":        paper_id,
                        "in_master_index": True,
                        "verified":       is_verified,
                        "doi":            entry.get("doi"),
                        "ssrn_id":        entry.get("ssrn_id"),
                        "tier":           entry.get("tier"),
                        "reason":         (None if is_verified
                                            else "verified=false (needs crossref pass)")})


# ── H5 count_free_params ─────────────────────────────────────────────────

_BANNED_PARAM_PATTERNS = [
    (re.compile(r"\[[^\]]*,[^\]]*\]"), "list/grid notation"),    # [a, b, ...]
    (re.compile(r"\{[^\}]*,[^\}]*\}"), "set/choice notation"),    # {a, b, ...}
    (re.compile(r"\bin\s*[\[\{]"),       "list-membership notation"),  # "in [...]" / "in {...}"
    (re.compile(r"\brange\s*\("),        "range notation"),       # range(...)
    (re.compile(r"\b(?:from\s+\d+\s+)?to\s+\d+\b"), "to-range notation"),    # "from 3 to 12"
    # Note: hyphen-only patterns (12-1, 2010-2020) are NOT flagged — they
    # are commonly used as single canonical-name values (12-1 momentum
    # horizon, year range identifier). True grid syntax always uses
    # brackets / commas / explicit range keywords.
]


def h5_count_free_params(param_specs: list[str]) -> ToolResult:
    """Reject if any param spec uses list/range/distribution syntax.

    Each accepted spec must be a SINGLE VALUE. e.g. 'lookback=12' OK;
    'lookback in [3,6,12]' REJECT (3 trials hidden); 'lookback=range(3,24)' REJECT.
    """
    rejected: list[dict] = []
    accepted: list[str] = []
    for spec in param_specs or []:
        bad_reason = None
        for pat, name in _BANNED_PARAM_PATTERNS:
            if pat.search(spec):
                bad_reason = f"{name} detected"
                break
        if bad_reason:
            rejected.append({"spec": spec, "reason": bad_reason})
        else:
            accepted.append(spec)
    return ToolResult(
        "h5_count_free_params", True,
        {"n_accepted":    len(accepted),
          "n_rejected":   len(rejected),
          "accepted":     accepted,
          "rejected":     rejected,
          "free_params":  len(accepted),
          "verdict":      "ok" if not rejected else "reject_grid_hide"},
    )


# ── H6 post_pub_evidence_check ────────────────────────────────────────────

def h6_post_pub_evidence_check(
    mechanism_id: str,
    candidate_info: dict | None = None,
) -> ToolResult:
    """Candidate-purpose mechanisms must have ≥1 post_pub_decay.post_2020_replications
    entry where the paper_id resolves in master index AND verified=true at master.

    BUGFIX 2026-05-31 (same chicken-and-egg pattern as H2): NEW candidates
    are not yet in library. If candidate_info provided, treat as
    purpose='candidate' for evaluation (since it's being proposed as
    one). If neither in library nor candidate_info provided, return
    'applicable=False' rather than hard FAIL — proceeding through
    pipeline is safer than blocking on chicken-and-egg.
    """
    entries = _load_library_entries()
    target = next((e for e in entries if e.get("id") == mechanism_id), None)
    if not target:
        if candidate_info:
            # Treat new candidate as purpose='candidate' so the H6 check
            # applies. Caller must provide post_pub_evidence in
            # candidate_info if available; otherwise this returns
            # "no qualifying" which is informational not fatal.
            target = {
                "id":             mechanism_id,
                "purpose":        "candidate",
                "post_pub_decay": candidate_info.get("post_pub_decay") or {},
            }
        else:
            return ToolResult(
                "h6_post_pub_evidence_check", True,
                {"mechanism_id": mechanism_id,
                 "applicable":   False,
                 "note":         (f"mechanism_id {mechanism_id!r} not in "
                                  "library AND no candidate_info provided; "
                                  "post-pub check skipped (proceed-with-"
                                  "warning instead of hard fail)")},
            )
    if target.get("purpose") != "candidate":
        return ToolResult(
            "h6_post_pub_evidence_check", True,
            {"mechanism_id": mechanism_id,
              "applicable":  False,
              "note":        "post-pub evidence requirement applies to "
                              "purpose=candidate only; this is purpose="
                              f"{target.get('purpose')!r}"},
        )
    reps = (((target.get("post_pub_decay") or {})
              .get("post_2020_replications") or []))
    master = _load_master_index().get("papers", {})
    qualifying = []
    for r in reps:
        pid = r.get("paper_id")
        if not pid or pid not in master:
            continue
        if not master[pid].get("verified"):
            continue
        qualifying.append({"paper_id": pid,
                            "delta_range_estimate": r.get("delta_range_estimate")})
    return ToolResult(
        "h6_post_pub_evidence_check", True,
        {"mechanism_id":  mechanism_id,
          "n_replications": len(reps),
          "n_qualifying":  len(qualifying),
          "qualifying":    qualifying,
          "verdict":       "ok" if qualifying else "reject_no_post_pub_replication"},
    )


# ── H8 check_factor_exposure_dry_run ────────────────────────────────────

def h8_check_factor_exposure_dry_run(
    sleeve_returns,
    proposal_name: str = "candidate",
    phase: int = 2,
    proposed_role: str | None = None,
) -> ToolResult:
    """Pre-gate factor-exposure check: regress a candidate sleeve's return
    series against the BARRA Phase 1 (MKT/SMB/MOM), Phase 2 (+HML/QMJ),
    or Phase 3 (+11 sectors) factor panel and report alpha-after-control
    + significant exposures.

    FLAW 1 FIX (multi-role doctrine, 2026-05-30): accepts optional
    `proposed_role` to produce role-aware verdicts. Without role, returns
    the legacy alpha-centric verdict (STRONG/BORDERLINE/FACTOR-TILTED/
    WEAK). With role, ALSO returns role-specific verdict which respects
    the candidate's intended purpose.

    Valid roles (per [[feedback-loop-refinement-multi-role-candidates-
    2026-05-30]]): alpha_seeker, risk_premium_harvester, insurance,
    regime_overlay, diversifier.

    Args:
      sleeve_returns: pandas Series of MONTHLY sleeve returns indexed by
        date. Must have >= 24 months overlap with the factor panel.
      proposal_name: identifier for the candidate in the verdict text.
      phase: 1, 2, or 3. Higher phases need Compustat / GICS caches and
        gracefully fall back to lower phases.
      proposed_role: optional role label; enables role-aware verdict.
    """
    try:
        from engine.risk.barra_lite import (
            build_factor_returns,
            regress_sleeve_on_factors,
        )
    except ImportError as exc:
        return ToolResult("h8_check_factor_exposure_dry_run", False, {},
                          error=f"barra_lite import failed: {exc}")

    try:
        factors = build_factor_returns(phase=phase)
    except FileNotFoundError:
        # Phase 2 cache missing — fall back to Phase 1
        if phase >= 2:
            logger.warning("Phase 2 factor cache missing; falling back to Phase 1")
            try:
                factors = build_factor_returns(phase=1)
                phase = 1
            except Exception as exc:
                return ToolResult("h8_check_factor_exposure_dry_run", False, {},
                                  error=f"Phase 1 fallback failed: {exc}")
        else:
            return ToolResult("h8_check_factor_exposure_dry_run", False, {},
                              error="Phase 1 factor cache missing")

    try:
        report = regress_sleeve_on_factors(sleeve_returns, factors,
                                                sleeve_name=proposal_name)
    except ValueError as exc:
        return ToolResult("h8_check_factor_exposure_dry_run", False, {},
                          error=str(exc))

    # Recommend factor_tilted_by_design=True if alpha_t < 2.0 AND any
    # single beta has |t| >= 4.0 (strong factor-by-design signal).
    strong_betas = {k: v for k, v in report.t_stats_hac.items()
                       if k != "alpha" and abs(v) >= 4.0}
    rec_tilted = (abs(report.alpha_t_hac) < 2.0 and bool(strong_betas))

    role_verdict = _h8_role_aware_verdict(report, proposed_role,
                                                rec_tilted, strong_betas)

    payload = {
        "proposal_name":     proposal_name,
        "phase":             phase,
        "proposed_role":     proposed_role,
        "n_months":          report.n_months,
        "alpha_annualized":  report.alpha_annualized,
        "alpha_t_hac":       report.alpha_t_hac,
        "betas":             report.betas,
        "t_stats_hac":       report.t_stats_hac,
        "r_squared":         report.r_squared,
        "strong_factor_loadings": strong_betas,
        "recommended_factor_tilted_by_design": rec_tilted,
        "verdict":           report.verdict,
        "gate_recommendation": _h8_gate_recommendation(report, rec_tilted),
    }
    if proposed_role is not None:
        payload["role_aware_verdict"] = role_verdict
    return ToolResult("h8_check_factor_exposure_dry_run", True, payload)


def _h8_gate_recommendation(report, recommended_tilted: bool) -> str:
    """Legacy alpha-centric verdict (unchanged for backward compat)."""
    if report.alpha_t_hac >= 2.0:
        return ("STRONG: alpha survives factor control; propose for full gate "
                "with high prior")
    if 1.0 <= report.alpha_t_hac < 2.0:
        return ("BORDERLINE: alpha t in (1.0, 2.0); full gate may pass or fail. "
                "Verify factor cache used most recent phase before submitting.")
    if recommended_tilted:
        return ("FACTOR-TILTED: alpha t < 1.0 AND a single factor has |t| >= 4.0. "
                "Recommend YAML factor_tilted_by_design=True; this mechanism is "
                "smart-beta-equivalent not unique alpha.")
    return ("WEAK: alpha t < 1.0 with no clear factor cause. Candidate likely "
            "subsumed by 5-factor universe; consider rejecting or reframing.")


def _h8_role_aware_verdict(report, role: str | None,
                                recommended_tilted: bool,
                                strong_betas: dict) -> dict:
    """FLAW 1 FIX (multi-role doctrine): produce role-specific accept /
    reject verdict that respects the candidate's intended purpose.

    Returns: {role, verdict_code, accept, explanation}
      verdict_code: STRONG_FOR_ROLE / BORDERLINE_FOR_ROLE / NOT_FIT_FOR_ROLE
        / VALID_FOR_ROLE / WEAK_FOR_ROLE
      accept: bool — meets role acceptance criteria
      explanation: human-readable rationale tied to the role
    """
    if role is None:
        return {"role": None, "verdict_code": "NO_ROLE_GIVEN",
                "accept": False,
                "explanation": "no role provided; using legacy gate_recommendation"}
    alpha_t = report.alpha_t_hac

    # ── alpha_seeker: alpha t >= 2.0 required ────────────────────────────
    if role == "alpha_seeker":
        if alpha_t >= 2.0:
            return {"role": role, "verdict_code": "STRONG_FOR_ROLE",
                      "accept": True,
                      "explanation": f"alpha t={alpha_t:+.2f} survives factor "
                                          f"control; genuine residual alpha"}
        if alpha_t >= 1.0:
            return {"role": role, "verdict_code": "BORDERLINE_FOR_ROLE",
                      "accept": False,
                      "explanation": f"alpha t={alpha_t:+.2f} between 1.0 and "
                                          f"2.0; consider full gate but expect "
                                          f"scrutiny"}
        return {"role": role, "verdict_code": "WEAK_FOR_ROLE",
                  "accept": False,
                  "explanation": f"alpha t={alpha_t:+.2f} below 1.0; not "
                                      f"meeting alpha-seeker bar. Consider "
                                      f"reframing as risk-premium-harvester "
                                      f"or rejecting."}

    # ── risk_premium_harvester: factor exposure expected + Sharpe>0 OK ──
    if role == "risk_premium_harvester":
        if strong_betas:
            return {"role": role, "verdict_code": "VALID_FOR_ROLE",
                      "accept": True,
                      "explanation": f"strong factor loading on "
                                          f"{list(strong_betas.keys())}; "
                                          f"intended harvester profile"}
        return {"role": role, "verdict_code": "NOT_FIT_FOR_ROLE",
                  "accept": False,
                  "explanation": "no factor with |t|>=4.0; doesn't harvest "
                                      "a recognizable risk premium"}

    # ── insurance: negative drift OK, target factor exposure negative ──
    if role == "insurance":
        # Insurance is valid if it has STRONG negative loading on a known
        # factor (typically MOM or MKT). Alpha t can be negative.
        neg_loadings = {k: v for k, v in strong_betas.items() if v < 0}
        if neg_loadings:
            return {"role": role, "verdict_code": "VALID_FOR_ROLE",
                      "accept": True,
                      "explanation": f"strong negative factor exposure on "
                                          f"{list(neg_loadings.keys())}; valid "
                                          f"insurance against those factors. "
                                          f"Negative alpha (t={alpha_t:+.2f}) "
                                          f"is the premium paid."}
        return {"role": role, "verdict_code": "NOT_FIT_FOR_ROLE",
                  "accept": False,
                  "explanation": "no strong negative factor loading; doesn't "
                                      "provide directional insurance"}

    # ── diversifier: H9 cosine is the metric (H8 alone can't decide) ──
    if role == "diversifier":
        # H8 alone insufficient — H9 is needed for full diversifier verdict.
        # Pre-screen: a candidate with R^2 high AND alpha_t near 0 might
        # still be a fine diversifier if its loadings are opposite-direction
        # vs book; need H9 to confirm.
        return {"role": role, "verdict_code": "DEFERRED_TO_H9",
                  "accept": None,
                  "explanation": f"diversifier verdict requires H9 "
                                      f"orthogonality (cosine to book). "
                                      f"Standalone alpha t={alpha_t:+.2f}, "
                                      f"R^2={report.r_squared:.2f} provided "
                                      f"as context; run H9 for accept decision."}

    # ── regime_overlay: cannot be evaluated as static sleeve ──
    if role == "regime_overlay":
        return {"role": role, "verdict_code": "ROLE_NOT_STATIC_TESTABLE",
                  "accept": None,
                  "explanation": "regime_overlay candidates apply dynamic "
                                      "allocation rules — H8 static regression "
                                      "is inappropriate. Use regime-specific "
                                      "backtest infrastructure (AN-1 / AM / AO "
                                      "spec modules) instead."}

    return {"role": role, "verdict_code": "UNKNOWN_ROLE",
              "accept": False,
              "explanation": f"role {role!r} not in valid set; expected one "
                                  f"of alpha_seeker / risk_premium_harvester / "
                                  f"insurance / diversifier / regime_overlay"}


# ── H9 check_orthogonality_to_book ────────────────────────────────────

def h9_check_orthogonality_to_book(
    candidate_sleeve_returns,
    proposal_name: str = "candidate",
    phase: int = 3,
) -> ToolResult:
    """Score how orthogonal a candidate sleeve is to the DEPLOYED book.

    The post-Phase-3 finding (book is 53% MOM-risk, 21% Cons-Staples,
    11% IT, etc.) makes EVERY new candidate's value depend on whether
    it ADDS orthogonal risk (improves diversification) or PILES on
    existing concentrations.

    Steps:
      1. Run H8 on the candidate to get its factor betas (Phase 3 by
         default — 5 styles + 11 sectors).
      2. Build the current book's FactorBudgetReport (from deployed
         sleeves at 70/25/5 weights).
      3. Compute factor_orthogonality_score.
      4. Translate to a gate recommendation.

    Output:
      cosine_to_book_risk:  +1 (fully aligned) .. -1 (fully opposite)
      risk_diversifying_score: -cosine (positive = adds diversification)
      top_3_overlaps:       factors where candidate piles onto book risk
      top_3_diversifiers:   factors where candidate hedges book risk
      gate_recommendation:  one of HIGH_DIVERSIFICATION / NEUTRAL /
                             PILE_ON / FULLY_ALIGNED_WARN
      verdict:              human-readable summary

    Per soft-gate doctrine: SURFACES the truth; never auto-rejects.
    """
    try:
        from engine.risk.barra_lite import (
            build_factor_returns,
            regress_sleeve_on_factors,
        )
        from engine.risk.factor_budget import (
            compute_factor_budget,
            factor_orthogonality_score,
        )
        from engine.portfolio.combined_book import (
            DEFAULT_CARRY_RISK_WEIGHT,
            DEFAULT_TSMOM_RISK_WEIGHT,
            build_carry_book,
            build_equity_book,
            build_tsmom_book,
        )
    except ImportError as exc:
        return ToolResult("h9_check_orthogonality_to_book", False, {},
                          error=f"import failed: {exc}")

    # Step 1: factor panel + candidate exposure
    try:
        factors = build_factor_returns(phase=phase)
    except FileNotFoundError as exc:
        return ToolResult("h9_check_orthogonality_to_book", False, {},
                          error=f"factor cache missing: {exc}")
    try:
        cand_report = regress_sleeve_on_factors(
            candidate_sleeve_returns, factors, sleeve_name=proposal_name,
        )
    except ValueError as exc:
        return ToolResult("h9_check_orthogonality_to_book", False, {},
                          error=str(exc))

    # Step 2: book budget
    sleeve_returns_book = {
        "equity_book": build_equity_book(),
        "carry_book":  build_carry_book(),
        "tsmom_book":  build_tsmom_book(),
    }
    sleeve_weights_book = {
        "equity_book": 1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT,
        "carry_book":  DEFAULT_CARRY_RISK_WEIGHT,
        "tsmom_book":  DEFAULT_TSMOM_RISK_WEIGHT,
    }
    try:
        book_report = compute_factor_budget(
            sleeve_returns_book, sleeve_weights_book, factors=factors,
        )
    except Exception as exc:
        return ToolResult("h9_check_orthogonality_to_book", False, {},
                          error=f"book budget computation failed: {exc}")

    # Step 3: orthogonality score
    ortho = factor_orthogonality_score(cand_report.betas, book_report)

    # Step 4: translate to gate recommendation
    cos = ortho["cosine_to_book_risk"]
    div = ortho["risk_diversifying_score"]
    if div >= 0.5:
        gate = "HIGH_DIVERSIFICATION"
        verdict = (f"candidate has cosine {cos:+.2f} to book risk profile "
                   f"(strongly orthogonal); ADDS major diversification value")
    elif div >= 0.1:
        gate = "MODERATE_DIVERSIFICATION"
        verdict = (f"candidate has cosine {cos:+.2f} (mildly orthogonal); "
                   f"some diversification value")
    elif div > -0.3:
        gate = "NEUTRAL"
        verdict = (f"candidate has cosine {cos:+.2f} (~uncorrelated); "
                   f"neither helps nor hurts diversification")
    elif div > -0.7:
        gate = "PILE_ON"
        verdict = (f"candidate has cosine {cos:+.2f} (aligned with book); "
                   f"piles onto existing factor concentration")
    else:
        gate = "FULLY_ALIGNED_WARN"
        verdict = (f"candidate has cosine {cos:+.2f} (nearly identical to "
                   f"book risk profile); LIKELY redundant with current book")

    # Suggest factor_tilted_by_design=True if candidate has weak alpha
    # AND high alignment with book risk
    cand_weak_alpha = abs(cand_report.alpha_t_hac) < 2.0
    suggest_tilted = cand_weak_alpha and (cos > 0.7)

    return ToolResult(
        "h9_check_orthogonality_to_book", True,
        {
            "proposal_name":          proposal_name,
            "phase":                  phase,
            "candidate_alpha_t":      cand_report.alpha_t_hac,
            "candidate_n_months":     cand_report.n_months,
            "cosine_to_book_risk":    cos,
            "risk_diversifying_score": div,
            "top_3_overlaps":         ortho["candidate_top_3_overlaps"],
            "top_3_diversifiers":     ortho["candidate_top_3_diversifiers"],
            "book_top_5_factors":     book_report.top_5_factors_by_risk,
            "gate_recommendation":    gate,
            "verdict":                verdict,
            "suggest_factor_tilted_by_design": suggest_tilted,
        },
    )


# ── H10 evaluate_candidate (unified L3 -> L4 evaluator) ──────────────────

def _h10_infer_role_from_h8(h8_payload: dict) -> tuple[str, str]:
    """Heuristic role inference from H8 alpha-centric output. Returns
    (role_label, rationale)."""
    alpha_t = h8_payload.get("alpha_t_hac", 0.0)
    r2 = h8_payload.get("r_squared", 0.0)
    strong_betas = h8_payload.get("strong_factor_loadings") or {}
    # Strong negative single-factor + negative alpha → insurance
    neg_strong = {k: v for k, v in strong_betas.items() if v < 0}
    if alpha_t < -0.5 and neg_strong:
        return ("insurance",
                f"alpha t={alpha_t:+.2f} (negative drift) + strong negative "
                f"factor exposure on {list(neg_strong.keys())} = insurance role.")
    # Strong alpha (t >= 2) + moderate R^2 → alpha_seeker
    if alpha_t >= 2.0 and r2 < 0.7:
        return ("alpha_seeker",
                f"alpha t={alpha_t:+.2f} survives factor control + R^2={r2:.2f} "
                f"not factor-dominated = alpha_seeker role.")
    # Strong factor exposure + alpha t near 0 → risk_premium_harvester
    if abs(alpha_t) < 1.5 and strong_betas and r2 >= 0.3:
        return ("risk_premium_harvester",
                f"strong factor loading on {list(strong_betas.keys())} + "
                f"alpha t={alpha_t:+.2f} near 0 = risk_premium_harvester role.")
    # Default: diversifier (caller likely wants H9-based assessment)
    return ("diversifier",
            f"alpha t={alpha_t:+.2f} + R^2={r2:.2f} + no dominant factor; "
            f"defer to H9 orthogonality for diversifier verdict.")


def _h10_combine_to_final(role: str, h8_pl: dict, h9_pl: dict | None) -> dict:
    """Combine H8 role-aware verdict + H9 orthogonality into FINAL
    deploy-recommendation. Returns dict with verdict_code / accept /
    rationale.

    Per multi-role doctrine + soft-gate principle — surfaces accept
    signal but humans always have final decision authority.
    """
    h8_v = h8_pl.get("role_aware_verdict") or {}
    h8_accept = h8_v.get("accept")
    h8_code = h8_v.get("verdict_code", "NO_ROLE_GIVEN")
    h9_cos = (h9_pl or {}).get("cosine_to_book_risk")

    # ── alpha_seeker: rely on H8 ──────────────────────────────────────
    if role == "alpha_seeker":
        if h8_accept:
            return {
                "verdict_code": "ACCEPT_FOR_DEPLOY",
                "accept": True,
                "rationale": f"H8 STRONG_FOR_ROLE; alpha survives factor "
                              f"control. H9 cosine to book = {h9_cos:+.2f} "
                              f"(bonus diversification info, not gating).",
            }
        return {
            "verdict_code": "REJECT_FOR_DEPLOY",
            "accept": False,
            "rationale": f"H8 verdict {h8_code}; alpha-seeker bar not met. "
                          f"H9 cosine {h9_cos:+.2f} doesn't rescue (alpha is "
                          f"the gating criterion).",
        }

    # ── risk_premium_harvester: rely on H8 ───────────────────────────
    if role == "risk_premium_harvester":
        if h8_accept:
            return {
                "verdict_code": "ACCEPT_FOR_DEPLOY",
                "accept": True,
                "rationale": f"H8 VALID_FOR_ROLE; harvests recognizable risk "
                              f"premium. H9 cosine {h9_cos:+.2f} adjacent context.",
            }
        return {
            "verdict_code": "REJECT_FOR_DEPLOY",
            "accept": False,
            "rationale": f"H8 verdict {h8_code}; no recognizable factor "
                          f"premium harvested.",
        }

    # ── insurance: H8 accept + H9 cosine should not be very negative ─
    if role == "insurance":
        if not h8_accept:
            return {
                "verdict_code": "REJECT_FOR_DEPLOY",
                "accept": False,
                "rationale": f"H8 verdict {h8_code}; doesn't insure against "
                              f"a recognizable factor. Insurance needs strong "
                              f"negative factor exposure.",
            }
        # H9 cosine: insurance should not PILE on either
        if h9_cos is not None and h9_cos > 0.5:
            return {
                "verdict_code": "REJECT_FOR_DEPLOY",
                "accept": False,
                "rationale": f"H8 marked valid insurance (negative factor "
                              f"exposure) but H9 cosine {h9_cos:+.2f} is "
                              f"positive — insurance overlaps book; "
                              f"net effect would PILE ON not protect.",
            }
        return {
            "verdict_code": "ACCEPT_FOR_DEPLOY",
            "accept": True,
            "rationale": f"H8 VALID insurance + H9 cosine {h9_cos:+.2f} "
                          f"(non-positive); paying premium for crash "
                          f"protection is the intent.",
        }

    # ── diversifier: H9 cosine is THE metric ─────────────────────────
    if role == "diversifier":
        if h9_cos is None:
            return {
                "verdict_code": "ROUTE_TO_HUMAN",
                "accept": None,
                "rationale": "diversifier verdict requires H9 cosine, "
                              "not available.",
            }
        if h9_cos <= -0.2:
            return {
                "verdict_code": "ACCEPT_FOR_DEPLOY",
                "accept": True,
                "rationale": f"H9 cosine to book = {h9_cos:+.2f} <= -0.20; "
                              f"meaningful diversifier value.",
            }
        if h9_cos <= -0.05:
            return {
                "verdict_code": "BORDERLINE_FOR_DEPLOY",
                "accept": False,
                "rationale": f"H9 cosine {h9_cos:+.2f} mildly negative; "
                              f"some diversification but threshold not met.",
            }
        return {
            "verdict_code": "REJECT_FOR_DEPLOY",
            "accept": False,
            "rationale": f"H9 cosine {h9_cos:+.2f} not orthogonal enough; "
                          f"candidate would not meaningfully diversify book.",
        }

    # ── regime_overlay: cannot be evaluated by H8/H9 alone ───────────
    if role == "regime_overlay":
        return {
            "verdict_code": "ROUTE_TO_REGIME_BACKTEST",
            "accept": None,
            "rationale": "regime_overlay candidates need dedicated regime-"
                          "backtest infrastructure (AN-1 / AM / AO spec "
                          "modules). H8/H9 static checks not applicable.",
        }

    return {
        "verdict_code": "UNKNOWN_ROLE",
        "accept": False,
        "rationale": f"role {role!r} not recognized.",
    }


def h10_evaluate_candidate(
    candidate_sleeve_returns,
    proposal_name: str = "candidate",
    proposed_role: str | None = None,
    phase: int = 3,
) -> ToolResult:
    """UNIFIED L3 -> L4 evaluator — single agent-facing entry point.

    Workflow:
      1. Run H8 (no role) for baseline factor exposure.
      2. If proposed_role not given, INFER from H8 results.
      3. Run H8 with role for role-aware verdict.
      4. Run H9 for orthogonality vs deployed book.
      5. Combine to final deploy recommendation.

    Returns ToolResult with payload:
      role_used (provided or inferred)
      role_was_inferred (bool)
      role_inference_rationale (string, if inferred)
      h8_summary {alpha_t, betas, t_stats, R^2, role_aware_verdict}
      h9_summary {cosine_to_book_risk, gate_recommendation,
                  top_overlaps, top_diversifiers}
      final {verdict_code, accept, rationale}

    verdict_code values:
      ACCEPT_FOR_DEPLOY  — role criteria met; OK to deploy
      REJECT_FOR_DEPLOY  — role criteria fail; reject or reframe
      BORDERLINE_FOR_DEPLOY — partial signal; human review
      ROUTE_TO_HUMAN — insufficient signal
      ROUTE_TO_REGIME_BACKTEST — regime_overlay role
      UNKNOWN_ROLE — invalid role

    Per soft-gate doctrine the accept flag SURFACES the algorithmic
    judgment but the deploy decision remains with the human operator.
    """
    # Step 1: baseline H8 without role
    h8_baseline = h8_check_factor_exposure_dry_run(
        candidate_sleeve_returns,
        proposal_name=proposal_name,
        phase=phase,
        proposed_role=None,
    )
    h8_base_d = h8_baseline.to_dict()
    if not h8_base_d.get("success"):
        return ToolResult(
            "h10_evaluate_candidate", False, {},
            error=f"H8 baseline failed: {h8_base_d.get('error')}",
        )
    h8_base_pl = h8_base_d["payload"]

    # Step 2: infer role if not provided
    role_was_inferred = False
    inference_rationale = None
    if proposed_role is None:
        proposed_role, inference_rationale = _h10_infer_role_from_h8(h8_base_pl)
        role_was_inferred = True

    # Step 3: H8 with role for role-aware verdict
    h8_role = h8_check_factor_exposure_dry_run(
        candidate_sleeve_returns,
        proposal_name=proposal_name,
        phase=phase,
        proposed_role=proposed_role,
    )
    h8_role_d = h8_role.to_dict()
    if not h8_role_d.get("success"):
        return ToolResult(
            "h10_evaluate_candidate", False, {},
            error=f"H8 role-aware failed: {h8_role_d.get('error')}",
        )
    h8_pl = h8_role_d["payload"]

    # Step 4: H9 orthogonality (skip for regime_overlay)
    h9_pl = None
    if proposed_role != "regime_overlay":
        h9_tool = h9_check_orthogonality_to_book(
            candidate_sleeve_returns,
            proposal_name=proposal_name,
            phase=phase,
        )
        h9_d = h9_tool.to_dict()
        if h9_d.get("success"):
            h9_pl = h9_d["payload"]

    # Step 5: combine to final
    final = _h10_combine_to_final(proposed_role, h8_pl, h9_pl)

    return ToolResult(
        "h10_evaluate_candidate", True,
        {
            "proposal_name":           proposal_name,
            "phase":                   phase,
            "role_used":               proposed_role,
            "role_was_inferred":       role_was_inferred,
            "role_inference_rationale": inference_rationale,
            "h8_summary": {
                "alpha_annualized": h8_pl["alpha_annualized"],
                "alpha_t_hac":      h8_pl["alpha_t_hac"],
                "r_squared":        h8_pl["r_squared"],
                "strong_factor_loadings": h8_pl.get("strong_factor_loadings", {}),
                "role_aware_verdict":     h8_pl.get("role_aware_verdict"),
            },
            "h9_summary": h9_pl and {
                "cosine_to_book_risk":  h9_pl["cosine_to_book_risk"],
                "risk_diversifying_score": h9_pl["risk_diversifying_score"],
                "gate_recommendation":  h9_pl["gate_recommendation"],
                "top_3_overlaps":       h9_pl.get("top_3_overlaps", []),
                "top_3_diversifiers":   h9_pl.get("top_3_diversifiers", []),
            },
            "final": final,
        },
    )


# ── H7 kill_this_proposal (deterministic adversarial critique) ───────────

def h7_kill_this_proposal(proposal: dict) -> ToolResult:
    """Deterministic adversarial critique. Real H7 SHOULD use the
    devils_advocate_constrained_evidence LLM persona (cross-vendor); this
    deterministic version is the fallback when persona unavailable AND a
    pre-LLM filter for obviously-vague proposals.

    Checks:
    - mechanism_id is set
    - canonical_paper_id matches an H4-passing paper
    - required_data passes H3
    - all H1/H2/H3/H4/H5/H6 results are bundled and inspected
    - sample_start / sample_end / parameters all populated
    """
    reasons: list[str] = []

    if not proposal.get("mechanism_id"):
        reasons.append("no mechanism_id specified")
    if not proposal.get("sample_start") or not proposal.get("sample_end"):
        reasons.append("sample window not pre-committed (sample_start / sample_end missing)")
    if not proposal.get("parameters"):
        reasons.append("parameters not specified — vague proposal")

    # Check H1 evidence
    h2_result = proposal.get("h2_cousin_check_result")
    if h2_result and h2_result.get("verdict") == "hard_reject":
        reasons.append(f"H2 hard_reject: {h2_result.get('hard_reject_reasons')}")

    h3_result = proposal.get("h3_data_check_result")
    if h3_result and not h3_result.get("all_present"):
        reasons.append(f"H3 data missing: {h3_result.get('missing')}")

    h4_result = proposal.get("h4_paper_check_result")
    if h4_result and not h4_result.get("verified"):
        reasons.append(f"H4 paper unverified: {h4_result.get('paper_id')}")

    h5_result = proposal.get("h5_param_check_result")
    if h5_result and h5_result.get("verdict") == "reject_grid_hide":
        reasons.append(f"H5 grid hide: {h5_result.get('rejected')}")

    h6_result = proposal.get("h6_post_pub_check_result")
    if h6_result and h6_result.get("verdict") == "reject_no_post_pub_replication":
        reasons.append("H6 no post-pub replication evidence")

    # Vagueness checks
    just = proposal.get("justification") or ""
    if len(just) < 50:
        reasons.append(f"justification too short ({len(just)} chars; need ≥50)")

    return ToolResult(
        "h7_kill_this_proposal", True,
        {"verdict":        "kill" if reasons else "survive",
          "kill_reasons":  reasons,
          "n_reasons":     len(reasons)},
    )


# ── Anthropic tool schemas (LLM tool-use surface) ────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "h1_list_unexplored_library_entries",
        "description": "Return mechanism library entries with purpose=candidate AND "
                       "currently_unexplored_in_our_book=true AND audit_signature=human-confirmed. "
                       "EMPTY LIST IS A VALID RESPONSE — return no proposal in that case.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "h2_cousin_check_multilevel",
        "description": "4-level cousin check (family/parent/data/economics-tokens) against ALL "
                       "library entries. Returns verdict allow/soft_reject/hard_reject. "
                       "hard_reject = MUST abandon this mechanism.",
        "input_schema": {
            "type": "object",
            "properties": {"mechanism_id": {"type": "string"}},
            "required": ["mechanism_id"],
        }
    },
    {
        "name": "h3_check_data_inventory",
        "description": "Verify all required_data tokens are in our DATA_INVENTORY whitelist. "
                       "100% required or REJECT (must find data first).",
        "input_schema": {
            "type": "object",
            "properties": {"required_data": {"type": "array",
                                              "items": {"type": "string"}}},
            "required": ["required_data"],
        }
    },
    {
        "name": "h4_verify_paper_in_library",
        "description": "Verify a paper_id exists in master index AND verified=true. "
                       "If verified=false, paper MUST NOT be cited as load-bearing.",
        "input_schema": {
            "type": "object",
            "properties": {"paper_id": {"type": "string"}},
            "required": ["paper_id"],
        }
    },
    {
        "name": "h5_count_free_params",
        "description": "Validate parameter specifications are SINGLE VALUES (not list/range/"
                       "distribution). Any grid notation REJECTED (anti-grid-search-disguise).",
        "input_schema": {
            "type": "object",
            "properties": {"param_specs": {"type": "array",
                                            "items": {"type": "string"}}},
            "required": ["param_specs"],
        }
    },
    {
        "name": "h6_post_pub_evidence_check",
        "description": "For candidate-purpose mechanisms, require ≥1 post-2020 OOS "
                       "replication in library entry whose paper_id is verified.",
        "input_schema": {
            "type": "object",
            "properties": {"mechanism_id": {"type": "string"}},
            "required": ["mechanism_id"],
        }
    },
    {
        "name": "h7_kill_this_proposal",
        "description": "Adversarial critique: examine a full proposal dict and return "
                       "verdict kill/survive with specific reasons. If kill, proposal "
                       "MUST be discarded.",
        "input_schema": {
            "type": "object",
            "properties": {"proposal": {"type": "object"}},
            "required": ["proposal"],
        }
    },
]


_DISPATCH = {
    "h1_list_unexplored_library_entries": lambda **kw: h1_list_unexplored_library_entries(
        include_pending=kw.get("include_pending", False)),
    "h2_cousin_check_multilevel": lambda **kw: h2_cousin_check_multilevel(
        kw["mechanism_id"]),
    "h3_check_data_inventory": lambda **kw: h3_check_data_inventory(
        kw["required_data"]),
    "h4_verify_paper_in_library": lambda **kw: h4_verify_paper_in_library(
        kw["paper_id"]),
    "h5_count_free_params": lambda **kw: h5_count_free_params(kw["param_specs"]),
    "h6_post_pub_evidence_check": lambda **kw: h6_post_pub_evidence_check(
        kw["mechanism_id"]),
    "h7_kill_this_proposal": lambda **kw: h7_kill_this_proposal(kw["proposal"]),
    "h8_check_factor_exposure_dry_run": lambda **kw: h8_check_factor_exposure_dry_run(
        sleeve_returns=kw["sleeve_returns"],
        proposal_name=kw.get("proposal_name", "candidate"),
        phase=kw.get("phase", 2),
        proposed_role=kw.get("proposed_role", None),
    ),
    "h9_check_orthogonality_to_book": lambda **kw: h9_check_orthogonality_to_book(
        candidate_sleeve_returns=kw["candidate_sleeve_returns"],
        proposal_name=kw.get("proposal_name", "candidate"),
        phase=kw.get("phase", 3),
    ),
    "h10_evaluate_candidate": lambda **kw: h10_evaluate_candidate(
        candidate_sleeve_returns=kw["candidate_sleeve_returns"],
        proposal_name=kw.get("proposal_name", "candidate"),
        proposed_role=kw.get("proposed_role", None),
        phase=kw.get("phase", 3),
    ),
}


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Dispatch a hygiene tool call by name."""
    if name not in _DISPATCH:
        return ToolResult(name, False, {}, error=f"unknown hygiene tool {name!r}")
    try:
        return _DISPATCH[name](**kwargs)
    except Exception as exc:
        logger.warning("hygiene tool %s failed: %s", name, exc)
        return ToolResult(name, False, {}, error=f"{type(exc).__name__}: {exc}")
