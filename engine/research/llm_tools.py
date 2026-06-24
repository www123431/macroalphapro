"""engine/research/llm_tools.py — LLM-callable tool layer for L4.

Each tool is a thin wrapper over our existing Python modules + a Pydantic
schema describing inputs/outputs. The registry exposes:

  - tool_specs_for_anthropic() — list of dicts in Anthropic Tool Use format
  - dispatch(tool_name, **kwargs) — execute by name

WHY THIS SHAPE (vs MCP server):
  MCP is a wire-protocol standard but the underlying need is "expose Python
  functions to LLM with schema + dispatch". Our `tool_specs_for_anthropic`
  format is directly consumable by anthropic SDK's tools= parameter. A
  later MCP wrapper can convert this same registry without re-implementing
  the tools.

The 9 tools (Session 3 minimum viable subset = first 4; rest stubbed for
Session 3.5 follow-up):

  KNOWLEDGE (4 — built):
    query_intuition_rules     surface senior-quant patterns
    query_graveyard           check if mechanism / family is already RED
    query_library             check what's deployed / pending / RED
    query_master_index        check if a paper is verified

  COMPUTE (3 — built):
    compute_cosine_with_book  geometry vs deployed sleeves
    estimate_sharpe_se        Bailey-LdP SE of Sharpe estimate
    family_n_trials_lookup    family-aware n_trials for DeflSR

  HISTORY (2 — built):
    query_outcome_ledger      past L4 propose-outcome pairs
    query_override_ledger     past graveyard overrides + outcomes
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 4f: trace every dispatch call. trace_log import is at module
# load so failures here surface immediately, not at first dispatch.
from engine.research.trace_log import add_attr as _trace_add_attr, span as _trace_span


# ── Tool input schemas (Pydantic, for both validation + Claude schema) ─


class QueryIntuitionRulesInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description=(
            "Filter rules by category. Valid: statistical, structural, "
            "data_quality, regime, decay, cross_market, "
            "role_interpretation, process, evidence."
        ),
    )
    severity: Optional[str] = Field(
        None,
        description=(
            "Filter rules by severity. Valid: FATAL_BLOCK, HARD_WARN, "
            "SOFT_INFO."
        ),
    )
    context_text: Optional[str] = Field(
        None,
        description=(
            "Substring matched against rule when/then. Use for relevance "
            "search e.g. 'cosine' or 'sample bias'."
        ),
    )


class QueryGraveyardInput(BaseModel):
    family: Optional[str] = Field(None,
        description="Mechanism family e.g. 'earnings_underreaction'.")
    candidate_title: Optional[str] = Field(None,
        description=("Free-text candidate name. Cross-market cousins are "
                      "auto-detected if family + title pattern match."))
    parent_family: Optional[str] = Field(None,
        description="Optional broader parent_family for cousin search.")


class QueryLibraryInput(BaseModel):
    status: Optional[str] = Field(None,
        description="DEPLOYED / PENDING_DEPLOY / RED / etc.")
    family: Optional[str] = Field(None,
        description="Mechanism family filter.")


class QueryMasterIndexInput(BaseModel):
    paper_id: str = Field(...,
        description="canonical paper_id e.g. 'bernard_thomas_1989_jar'.")


class ComputeCosineInput(BaseModel):
    candidate_returns_path: str = Field(...,
        description="Absolute or repo-relative path to parquet with single "
                    "return series column.")


class EstimateSharpeSEInput(BaseModel):
    sharpe_ann: float = Field(...,
        description="Annualized Sharpe ratio (point estimate).")
    n_years: float = Field(..., gt=0,
        description="Sample size in years.")


class FamilyNTrialsInput(BaseModel):
    family: str = Field(...,
        description=("Mechanism family. Returns within-family trial count "
                      "per Bailey-LdP §3 doctrine."))


class QueryOutcomeLedgerInput(BaseModel):
    candidate_id: Optional[str] = Field(None,
        description="Filter outcomes by candidate_id.")


class QueryOverrideLedgerInput(BaseModel):
    candidate_id: Optional[str] = Field(None,
        description="Filter override records by candidate_id.")


class GraveyardSummaryInput(BaseModel):
    """Aggregate stats — fills the gap query_graveyard leaves for
    dashboard / monitoring use cases that don't have a candidate to
    check against."""
    top_n_families: int = Field(
        10, ge=1, le=50,
        description="How many top families by death count to surface.",
    )


class GetSuggestionsInput(BaseModel):
    """L1 candidate seed recommender input — ranked seed pool +
    library-derived options, no LLM call."""
    limit: int = Field(
        10, ge=1, le=50,
        description="Max suggestions to return (sorted by score desc).",
    )


class ListToolsByCategoryInput(BaseModel):
    """Meta-discovery input for grouping the tool registry by category."""
    category: Optional[str] = Field(None,
        description="Filter by one of {knowledge, compute, history, "
                    "external_data, action}. None returns the full "
                    "grouped view.")


class ArxivSearchInput(BaseModel):
    """arxiv academic-paper search input."""
    query: str = Field(...,
        description="Free-text query against title/abstract/authors. "
                    "Use quant-finance terms e.g. 'cross-sectional "
                    "momentum', 'term structure carry'.")
    max_results: int = Field(5, ge=1, le=20,
        description="Number of results to return (1-20).")


class SecEdgarSearchInput(BaseModel):
    """SEC EDGAR full-text search input."""
    query: str = Field(...,
        description="Free-text query against filing body. Quote phrases "
                    "for exact match e.g. '\"share repurchase\"'.")
    forms: Optional[list[str]] = Field(None,
        description="Form types to filter e.g. ['10-K', '10-Q', '8-K'].")
    n_results: int = Field(10, ge=1, le=50,
        description="Number of filings to return (1-50).")


class FredQueryInput(BaseModel):
    """FRED macro time series input."""
    series_id: str = Field(...,
        description="FRED series identifier, case-sensitive. Common: "
                    "UNRATE / VIXCLS / DGS10 / T10Y2Y / CPIAUCSL / "
                    "FEDFUNDS / DTB3 / GDP / M2SL / WALCL.")
    start_date: Optional[str] = Field(None,
        description="YYYY-MM-DD lower bound (inclusive).")
    end_date: Optional[str] = Field(None,
        description="YYYY-MM-DD upper bound (inclusive).")


class QueryL4IterationsInput(BaseModel):
    """L4 discovery loop history — one entry per propose→critique→
    (pipeline)→ledger iteration."""
    limit: int = Field(
        20, ge=1, le=500,
        description="Number of newest-first entries to return.",
    )
    consensus: Optional[str] = Field(
        None,
        description="Filter by council consensus: APPROVE | NEEDS_REVISION | REJECT.",
    )
    alignment: Optional[str] = Field(
        None,
        description=(
            "Filter by verdict_alignment: agree | council_wrong | "
            "pipeline_resolved | not_runnable. Use to find calibration "
            "failures (council_wrong)."
        ),
    )


# ── Tool implementations (wrap existing modules) ───────────────────────


def query_intuition_rules(category=None, severity=None,
                           context_text=None) -> dict:
    from engine.research.intuition_rules import query_rules
    rules = query_rules(category=category, severity=severity,
                          context_text=context_text)
    return {
        "n_matched": len(rules),
        "rules": [
            {"id": r.id, "category": r.category, "severity": r.severity,
             "when": r.when, "then": r.then,
             "evidence_source": r.evidence_source}
            for r in rules
        ],
    }


def query_graveyard(family=None, candidate_title=None,
                     parent_family=None) -> dict:
    from engine.research.graveyard import (
        CandidateInfo, check_against_graveyard,
    )
    candidate = CandidateInfo(
        title=candidate_title or "",
        family=family,
        parent_family=parent_family,
    )
    match = check_against_graveyard(candidate)
    d = match.to_dict()
    return {
        "matched": d.get("matched", False),
        "recommendation": d.get("recommendation"),
        "cousin_count_in_family": d.get("cousin_count_in_family", 0),
        "signals_matched": d.get("signals_matched", []),
        "matched_entries": [
            {"name": (e.get("name") if isinstance(e, dict)
                       else getattr(e, "name", "?")),
             "verdict": (e.get("verdict") if isinstance(e, dict)
                          else getattr(e, "verdict", None))}
            for e in (d.get("matched_entries") or [])[:5]
        ],
        "explanation": (d.get("explanation") or "")[:400],
    }


def query_library(status=None, family=None) -> dict:
    import yaml as _y
    out = []
    for fp in sorted((REPO_ROOT / "data" / "research" /
                       "mechanism_library").glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = _y.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if status and d.get("status_in_our_book") != status:
            continue
        if family and d.get("family") != family:
            continue
        out.append({
            "id":     d.get("id", fp.stem),
            "family": d.get("family"),
            "status": d.get("status_in_our_book"),
            "purpose": d.get("purpose"),
            "proposed_role": (d.get("factor_exposure") or {}).get(
                "proposed_role"),
        })
    return {"n_matched": len(out), "entries": out}


def query_master_index(paper_id: str) -> dict:
    import yaml as _y
    p = REPO_ROOT / "data" / "research" / "mechanism_library" / \
        "_canonical_papers_tier1_2.yaml"
    if not p.exists():
        return {"found": False, "error": "master_index not found"}
    d = _y.safe_load(p.read_text(encoding="utf-8")) or {}
    papers = d.get("papers", {})
    entry = papers.get(paper_id)
    if not entry:
        return {"found": False, "paper_id": paper_id}
    return {
        "found": True,
        "paper_id": paper_id,
        "author": entry.get("author"),
        "year": entry.get("year"),
        "journal": entry.get("journal"),
        "title": entry.get("title"),
        "doi": entry.get("doi"),
        "verified": entry.get("verified", False),
        "tier": entry.get("tier"),
    }


def compute_cosine_with_book(candidate_returns_path: str) -> dict:
    """Compute cosine of candidate vs each deployed sleeve."""
    import pandas as _pd
    import numpy as _np
    from engine.portfolio.combined_book import (
        build_carry_book, build_equity_book, build_tsmom_book,
    )
    s = _pd.read_parquet(candidate_returns_path).iloc[:, 0]
    s.index = _pd.to_datetime(s.index)
    sleeves = {
        "equity": build_equity_book(),
        "carry":  build_carry_book(),
        "tsmom":  build_tsmom_book(),
    }
    out = {}
    for name, sleeve in sleeves.items():
        sleeve.index = _pd.to_datetime(sleeve.index)
        j = _pd.concat([s.rename("c"), sleeve.rename("s")], axis=1).dropna()
        if len(j) < 12:
            out[name] = None
            continue
        cv, sv = j["c"].values, j["s"].values
        nc, ns = _np.linalg.norm(cv), _np.linalg.norm(sv)
        out[name] = float(cv @ sv / (nc * ns)) if (nc * ns) > 0 else 0.0
    return {"cosines": out, "n_months": len(s)}


def estimate_sharpe_se(sharpe_ann: float, n_years: float) -> dict:
    """Bailey-LdP standard error of annualized Sharpe estimate."""
    se = math.sqrt((1 + 0.5 * sharpe_ann ** 2) / n_years)
    return {
        "sharpe_ann":    sharpe_ann,
        "n_years":       n_years,
        "se":            se,
        "interpretation": (
            f"observed Sharpe diff of < {se:.3f} is within 1 SE and "
            "may be sample noise, not real signal difference"
        ),
    }


def family_n_trials_lookup(family: str) -> dict:
    from engine.research.family_trial_counter import (
        count_trials_in_family, explain_count,
    )
    return explain_count(family)


def query_outcome_ledger(candidate_id=None) -> dict:
    p = REPO_ROOT / "data" / "research" / "override_ledger.jsonl"
    if not p.exists():
        return {"n": 0, "outcomes": []}
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") != "override_outcome":
            continue
        if candidate_id and r.get("candidate_id") != candidate_id:
            continue
        out.append(r)
    return {"n": len(out), "outcomes": out}


def get_candidate_suggestions(limit: int = 10) -> dict:
    from engine.research.suggestion_engine import (
        get_candidate_suggestions as _impl,
    )
    return _impl(limit=limit)


def query_l4_iterations(
    limit: int = 20,
    consensus: Optional[str] = None,
    alignment: Optional[str] = None,
) -> dict:
    from engine.research.outcome_ledger import (
        calibration_summary, read_l4_iterations,
    )
    rows = read_l4_iterations(
        limit=limit, consensus=consensus, alignment=alignment,
    )
    return {
        "n":           len(rows),
        "iterations":  rows,
        "calibration": calibration_summary(limit=200),
    }


def graveyard_summary(top_n_families: int = 10) -> dict:
    """Aggregate stats over the whole graveyard — for dashboard panels
    (Cockpit) where there's no specific candidate to query against.

    Counts dead entries, breaks down by family + failure_mode, and
    reports the recency of additions. Reads through the same
    build_graveyard() pipeline so cache + dedupe semantics match
    check_against_graveyard."""
    from collections import Counter
    from engine.research.graveyard import build_graveyard
    graveyard = build_graveyard()
    n_total = len(graveyard)
    fam_counts = Counter(e.family for e in graveyard if e.family)
    mode_counts = Counter(
        getattr(e, "failure_mode", None) for e in graveyard
        if getattr(e, "failure_mode", None)
    )
    # Recency: collect available date_killed attrs; sort newest first
    recent: list[dict] = []
    for e in graveyard:
        d = getattr(e, "date_killed", None) or getattr(e, "date", None)
        if d:
            recent.append({
                "name":         getattr(e, "name", "?"),
                "family":       e.family,
                "failure_mode": getattr(e, "failure_mode", None),
                "date":         str(d)[:10],
            })
    recent.sort(key=lambda r: r["date"], reverse=True)
    return {
        "n_total":         n_total,
        "n_families":      len(fam_counts),
        "top_families":    [
            {"family": f, "count": c}
            for f, c in fam_counts.most_common(int(top_n_families))
        ],
        "failure_modes":   [
            {"mode": (str(m) if m else "unknown"), "count": c}
            for m, c in mode_counts.most_common()
        ],
        "recent_deaths":   recent[:10],
    }


def query_override_ledger(candidate_id=None) -> dict:
    p = REPO_ROOT / "data" / "research" / "override_ledger.jsonl"
    if not p.exists():
        return {"n": 0, "overrides": []}
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") != "override_granted":
            continue
        if candidate_id and r.get("candidate_id") != candidate_id:
            continue
        out.append(r)
    return {"n": len(out), "overrides": out}


# ── Registry: name → (function, Pydantic schema, description, category, example_query) ──
#
# 5 categories — keeps the surface scannable as the registry grows:
#   knowledge      static doctrine / rule lookups
#   compute        deterministic numerical analysis
#   history        time-series of past system events
#   external_data  read-only external API calls (arxiv / SEC / FRED)
#   action         (reserved for future write-tools; none today)
#
# Senior discipline ([[feedback-no-emoji-icons-professional-ui-2026-06-01]]
# + tool-count > 10 hygiene): every entry carries 1 representative
# example_query so an LLM picking tools sees CONCRETE intent, not
# just abstract description.


def _import_arxiv():
    from engine.research.external_data_tools import arxiv_search
    return arxiv_search


def _import_sec():
    from engine.research.external_data_tools import sec_edgar_search
    return sec_edgar_search


def _import_fred():
    from engine.research.external_data_tools import fred_query
    return fred_query


# Lazy wrappers — defer the import of external_data_tools (heavyweight
# requests-using module) until the tool is actually invoked.
def _arxiv_search(*args, **kw):  return _import_arxiv()(*args, **kw)
def _sec_edgar_search(*args, **kw): return _import_sec()(*args, **kw)
def _fred_query(*args, **kw):    return _import_fred()(*args, **kw)


_TOOL_SPECS: list[tuple[str, Callable, type[BaseModel], str, str, str]] = [
    # (name, fn, schema, description, category, example_query)

    # ── knowledge ─────────────────────────────────────────────────────
    ("query_intuition_rules",
     query_intuition_rules, QueryIntuitionRulesInput,
     "Return codified senior-quant patterns matching filters. Use to "
     "check for known concerns (e.g. 'cosine', 'short sample') before "
     "proposing or critiquing a candidate.",
     "knowledge",
     "any FATAL_BLOCK rules about cross-sectional rank strategies"),
    ("query_graveyard",
     query_graveyard, QueryGraveyardInput,
     "Check if a candidate idea matches any RED entry in our graveyard. "
     "Returns recommendation ('block' / 'warn' / 'allow') + cousin "
     "count. Use BEFORE submitting any candidate to pipeline.",
     "knowledge",
     "is JP PEAD in graveyard"),
    ("query_library",
     query_library, QueryLibraryInput,
     "List mechanism library entries by status / family. Tells you "
     "what's currently DEPLOYED / PENDING_DEPLOY / RED in our book. "
     "Use to avoid proposing duplicates of deployed sleeves.",
     "knowledge",
     "all DEPLOYED sleeves"),
    ("query_master_index",
     query_master_index, QueryMasterIndexInput,
     "Look up a canonical paper by paper_id. Confirms paper exists in "
     "our verified master index. Use BEFORE citing any paper to avoid "
     "hallucinated citations.",
     "knowledge",
     "verify bernard_thomas_1989_jar is in our index"),
    ("graveyard_summary",
     graveyard_summary, GraveyardSummaryInput,
     "Aggregate stats over the whole graveyard — total dead, top "
     "families by death count, failure mode breakdown, recent "
     "additions. Use for dashboards / monitoring (Cockpit) where "
     "there's no specific candidate to check against.",
     "knowledge",
     "graveyard overview"),
    ("get_candidate_suggestions",
     get_candidate_suggestions, GetSuggestionsInput,
     "L1 candidate seed recommender — ranked list of ideas worth "
     "exploring next, blending library UNTESTED entries with a "
     "senior-curated seed pool. Scoring uses underexplored × no-"
     "cousin × role-gap heuristics. No LLM call — pure data scan.",
     "knowledge",
     "top 10 candidates ranked by score"),

    # ── compute ───────────────────────────────────────────────────────
    ("compute_cosine_with_book",
     compute_cosine_with_book, ComputeCosineInput,
     "Compute cosine of a candidate return series vs each deployed "
     "sleeve. Use to gauge orthogonality / diversification value.",
     "compute",
     "cosine of data/cache/_dpead_sn_pit_monthly.parquet with book"),
    ("estimate_sharpe_se",
     estimate_sharpe_se, EstimateSharpeSEInput,
     "Bailey-LdP SE of annualized Sharpe estimate. Use before claiming "
     "'Sharpe X > Sharpe Y' — observed diff may be within noise.",
     "compute",
     "SE of Sharpe 1.5 over 10 years"),
    ("family_n_trials_lookup",
     family_n_trials_lookup, FamilyNTrialsInput,
     "Returns within-family trial count for DeflSR. Use this n_trials "
     "(NOT codebase total) per Bailey-LdP §3.",
     "compute",
     "n_trials for earnings_underreaction family"),

    # ── history ───────────────────────────────────────────────────────
    ("query_outcome_ledger",
     query_outcome_ledger, QueryOutcomeLedgerInput,
     "Query past L4 proposal outcomes. Use to learn from prior "
     "REINFORCED / OVERTURNED / INCONCLUSIVE verdicts.",
     "history",
     "all override outcomes"),
    ("query_override_ledger",
     query_override_ledger, QueryOverrideLedgerInput,
     "Query past graveyard override requests + their outcomes. Use to "
     "understand prior override success rate (empirical Bayesian).",
     "history",
     "override events for jp_pead"),
    ("query_l4_iterations",
     query_l4_iterations, QueryL4IterationsInput,
     "L4 discovery loop history with calibration KPI (council vs "
     "pipeline agreement rate). Each entry is one propose-critique-"
     "(pipeline)-ledger iteration. Filter by consensus / alignment "
     "to find council mis-calibrations (alignment='council_wrong').",
     "history",
     "iterations where council was wrong"),

    # ── meta-discovery (helps LLMs scan the growing registry) ─────────
    ("list_tools_by_category",
     # Forward reference: defined below after _TOOL_SPECS init. Caller
     # only invokes through dispatch() which dereferences lazily.
     lambda *a, **kw: list_tools_by_category(*a, **kw),
     ListToolsByCategoryInput,
     "Meta-tool: surface the tool registry grouped by category, with "
     "1 example query per tool. Use when you don't know which tool fits "
     "the question — saves scanning a flat 15-entry list.",
     "knowledge",
     "tools for graveyard / library questions"),

    # ── external_data (NEW) ───────────────────────────────────────────
    ("arxiv_search",
     _arxiv_search, ArxivSearchInput,
     "Search arxiv.org for academic finance / quant papers (title, "
     "abstract, author). Free public API, no key. Use for paper "
     "discovery BEYOND our verified master_index. For verified-only "
     "citation, prefer query_master_index.",
     "external_data",
     "post-earnings drift cross-market evidence"),
    ("sec_edgar_search",
     _sec_edgar_search, SecEdgarSearchInput,
     "SEC EDGAR full-text search across all filings. Free public API. "
     "Useful for issuance / buyback / risk-factor / MD&A research. "
     "Quote phrases for exact match e.g. '\"share repurchase\"'. "
     "Optional forms=['10-K', '10-Q', '8-K'] filter.",
     "external_data",
     "share repurchase announcements 2024"),
    ("fred_query",
     _fred_query, FredQueryInput,
     "Fetch a FRED macro time series (unemployment, VIX history, 10y "
     "yield, term spread, fed funds, etc). Returns observations as "
     "(date, value) list, decimated to ~200 points for long series. "
     "Use for regime context and macro factor lookups.",
     "external_data",
     "10y Treasury yield since 2020 (DGS10)"),
]


# Build registry + category index
TOOLS: dict[str, tuple[Callable, type[BaseModel], str]] = {
    name: (fn, schema, desc)
    for name, fn, schema, desc, _cat, _ex in _TOOL_SPECS
}
TOOL_CATEGORIES: dict[str, str] = {
    name: cat for name, _fn, _s, _d, cat, _ex in _TOOL_SPECS
}
TOOL_EXAMPLES: dict[str, str] = {
    name: ex for name, _fn, _s, _d, _cat, ex in _TOOL_SPECS
}


def list_tools_by_category(category: Optional[str] = None) -> dict:
    """Meta-tool: surface the registry organized by category, with one
    example query per tool. Helps a tool-using LLM pick the right
    tool without scanning a flat 15-entry list.

    Args:
      category: optional filter — one of {knowledge, compute, history,
        external_data, action}. If None, returns the full grouped view.
    """
    out: dict[str, list[dict]] = {}
    for name, cat in TOOL_CATEGORIES.items():
        if category and cat != category:
            continue
        out.setdefault(cat, []).append({
            "name":        name,
            "description": TOOLS[name][2],
            "example":     TOOL_EXAMPLES[name],
        })
    return {
        "n_categories":   len(out),
        "n_tools_total":  sum(len(v) for v in out.values()),
        "by_category":    out,
    }


# ── Spec emitters for Claude SDK tool use ──────────────────────────────


def _pydantic_to_anthropic_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic v2 model JSON schema to Anthropic-friendly form.

    Anthropic wants: {type:object, properties:{...}, required:[...]}.
    Pydantic gives ~that out of the box; just unwrap.
    """
    schema = model.model_json_schema()
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


def tool_specs_for_anthropic() -> list[dict]:
    """Format usable in `anthropic.Anthropic().messages.create(tools=...)`.

    Each spec: {name, description, input_schema}.
    """
    return [
        {
            "name": name,
            "description": desc,
            "input_schema": _pydantic_to_anthropic_schema(schema),
        }
        for name, (_fn, schema, desc) in TOOLS.items()
    ]


def dispatch(tool_name: str, **kwargs) -> Any:
    """Execute tool by name with validated kwargs.

    Phase 4f: wrapped in a trace_log span so every tool invocation is
    visible in the Cockpit trace timeline. Span attrs carry tool_name,
    args (truncated), and result row count where applicable.

    Raises:
      KeyError if tool_name not registered.
      pydantic.ValidationError if kwargs don't match schema.
    """
    if tool_name not in TOOLS:
        raise KeyError(
            f"unknown tool {tool_name!r}; known: {list(TOOLS.keys())}"
        )
    fn, schema, _desc = TOOLS[tool_name]
    with _trace_span(
        f"tool.{tool_name}",
        kind_class="tool",
        tool=tool_name,
        # Args are typically tiny dicts of filter fields; safe to log
        args=kwargs,
    ):
        validated = schema(**kwargs)
        result = fn(**validated.model_dump(exclude_none=True))
        # Surface a meaningful size attribute for the timeline UI
        if isinstance(result, dict):
            for key in ("n", "n_matched", "n_total", "n_entries"):
                if isinstance(result.get(key), int):
                    _trace_add_attr(result_count=result[key])
                    break
        return result


def list_tool_names() -> list[str]:
    return list(TOOLS.keys())
