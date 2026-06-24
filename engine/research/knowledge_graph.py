"""engine/research/knowledge_graph.py — Mechanism Knowledge Graph v1.

Phase 1 Task #2 of project_agentic_ai_real_architecture_2026-05-29 memo.
The foundation for the L2 (Knowledge) layer of the 4-layer agentic AI
architecture. Pure engineering — NO LLM, NO scaffolding. Reads existing
research artifacts (gate_runs.jsonl, cross_review_ledger.jsonl, deployed
combined_book sleeves) and exposes a queryable graph.

Why a graph (not just jsonl scans)
----------------------------------
Today we have 22 gate runs across multiple mechanism families, sample
windows, asset classes, and outcomes. The flat jsonl format makes
non-trivial questions hard:
  - "Which (asset_class × mechanism_family) cells are blind spots?"
  - "What failure themes co-occur in RED candidates?"
  - "Which deployed sleeves share mechanism roots with my new candidate?"

A typed graph supports these in O(1) per edge traversal. Even at 22
candidates this saves significant time when the L3 (Reasoning) layer
needs context.

Node types
----------
- Candidate (the unit under test)
- MechanismFamily (canonical taxonomy — see _MECHANISM_TAXONOMY)
- AssetClass (equity | futures | etfs | mixed)
- SampleWindow (start, end, has_2008_gfc, has_2020_covid, has_2022_crash)
- Verdict (GREEN | YELLOW | RED | UNINTERPRETABLE)
- Theme (failure / success themes from cross-review ledger)
- Sleeve (currently-deployed sleeves)

Edge types
----------
- has_mechanism(Candidate → MechanismFamily)
- received(Candidate → Verdict)
- tested_on(Candidate → SampleWindow)
- in_asset_class(Candidate → AssetClass)
- has_theme(Candidate → Theme)
- deployed_as(Candidate → Sleeve)
- similar_to(Candidate → Candidate)     # auto-inferred from shared family
- uses_mechanism(Sleeve → MechanismFamily)

Design choices (kept tight, no LLM, no external deps beyond stdlib + pandas)
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_LEDGER = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
CROSS_REVIEW_LEDGER = REPO_ROOT / "data" / "research" / "cross_review_ledger.jsonl"


# ── Canonical mechanism family taxonomy ──────────────────────────────────────
# Each family is identified by KEYWORDS (case-insensitive substring match against
# the candidate's mechanism description or name). Order matters: more specific
# families come first.

_MECHANISM_TAXONOMY: list[tuple[str, list[str]]] = [
    ("earnings_underreaction", ["pead", "revision", "guidance", "drift", "earnings"]),
    ("vol_carry",               ["vix", "vrp", "variance risk premium",
                                  "karagozoglu", "volatility risk"]),
    ("carry",                   ["carry", "roll-yield", "roll yield", "kmpv",
                                  "term-structure", "term structure", "koijen"]),
    ("tsmom",                   ["tsmom", "time-series momentum",
                                  "time series momentum", "moskowitz-ooi", "mop"]),
    ("residual_momentum",       ["residual momentum", "blitz-huij-martens",
                                  "bhm 2011", "ff3-residual"]),
    ("momentum",                ["momentum"]),    # catch-all after residual
    ("quality",                 ["quality", "qmj", "novy-marx", "profitability",
                                  "gross profitability"]),
    ("lead_lag",                ["lead-lag", "lead lag", "dgnsde", "hong-lim-stein",
                                  "cohen-frazzini", "cross-corr", "information transmission"]),
    ("news_sentiment",          ["news", "sentiment", "attention", "rpna",
                                  "ess", "8-k", "edgar"]),
    ("regime_overlay",          ["regime", "msm", "macro overlay",
                                  "narrative", "multivariate msm"]),
    ("credit_risk",             ["credit", "hyg", "ief", "high-yield",
                                  "spread carry", "credit spread"]),
    ("crisis_alpha",            ["crisis alpha", "hurst-ooi-pedersen",
                                  "stress conditional", "crisis-hedge"]),
    ("breadth_expansion",       ["breadth", "g10 cross-country", "sovereign"]),
]

# Parent family rollup — for cousin-detection across same higher class.
# Many factors are sufficiently similar at the parent level that we should
# consider them potentially redundant (e.g. quality + residual momentum +
# momentum are all "equity cross-sectional factor").
_PARENT_FAMILIES: dict[str, list[str]] = {
    "equity_factor":      ["earnings_underreaction", "quality", "momentum",
                            "residual_momentum"],
    "cross_asset_carry":  ["carry", "vol_carry", "breadth_expansion"],
    "cross_asset_trend":  ["tsmom"],
    "network_effects":    ["lead_lag"],
    "alt_data":           ["news_sentiment"],
    "regime_management":  ["regime_overlay", "crisis_alpha"],
    "credit":             ["credit_risk"],
}

def parent_families_for(family: str) -> list[str]:
    """Return parent family/families containing this child family."""
    return [p for p, children in _PARENT_FAMILIES.items() if family in children]


def classify_mechanism(mechanism_str: str, candidate_name: str = "") -> list[str]:
    """Map a free-text mechanism description to canonical family/families.
    Returns the FIRST matching family (most specific) plus secondary if applicable."""
    s = (mechanism_str + " " + candidate_name).lower()
    matches = []
    for family, keywords in _MECHANISM_TAXONOMY:
        if any(kw in s for kw in keywords):
            matches.append(family)
    return matches if matches else ["unclassified"]


# ── Sample window stress-period coverage ─────────────────────────────────────

_CANONICAL_STRESS = [
    ("2008_gfc",       "2008-09-01", "2009-03-31"),
    ("2010_flash",     "2010-05-06", "2010-05-31"),
    ("2011_eu_debt",   "2011-08-01", "2011-09-30"),
    ("2015_china",     "2015-08-01", "2015-09-30"),
    ("2018_volm",      "2018-02-01", "2018-04-30"),
    ("2018_q4",        "2018-10-01", "2018-12-31"),
    ("2020_covid",     "2020-02-15", "2020-04-30"),
    ("2022_rate_crash","2022-03-01", "2022-10-31"),
    ("2023_svb",       "2023-03-01", "2023-04-30"),
]


@dataclasses.dataclass(frozen=True)
class SampleCoverage:
    start:           str
    end:             str
    n_months:        int
    stress_covered:  tuple[str, ...]
    stress_missed:   tuple[str, ...]

    @property
    def coverage_ratio(self) -> float:
        total = len(_CANONICAL_STRESS)
        return len(self.stress_covered) / total if total else 0.0


def _sample_coverage(start_str: str | None, end_str: str | None,
                     n_months: int | None) -> SampleCoverage:
    if not start_str or not end_str:
        return SampleCoverage("", "", n_months or 0, (), tuple(s for s, _, _ in _CANONICAL_STRESS))
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    covered, missed = [], []
    for label, ss, ee in _CANONICAL_STRESS:
        s = pd.Timestamp(ss)
        e = pd.Timestamp(ee)
        if not (end < s or start > e):
            covered.append(label)
        else:
            missed.append(label)
    return SampleCoverage(str(start.date()), str(end.date()), n_months or 0,
                           tuple(covered), tuple(missed))


# ── Asset-class classification ───────────────────────────────────────────────

_ASSET_CLASS_KEYWORDS = {
    "equity": ["pead", "revision", "novy-marx", "quality", "momentum",
                "residual momentum", "blitz-huij-martens", "qmj", "profitability"],
    "etfs":   ["spdr", "sector", "xlk", "xlf", "etf", "vxx", "vxz"],
    "futures": ["futures", "tsmom", "carry", "tr_ds_fut",
                 "moskowitz-ooi", "kmpv", "term-structure"],
    "vol":    ["vix", "vrp", "variance"],
    "credit": ["hyg", "ief", "high-yield"],
}


def classify_asset_class(mechanism_str: str, candidate_name: str = "") -> str:
    s = (mechanism_str + " " + candidate_name).lower()
    scores = {ac: sum(1 for kw in kws if kw in s)
              for ac, kws in _ASSET_CLASS_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "mixed"


# ── Canonical verdict normalization ──────────────────────────────────────────

def normalize_verdict(raw: str | None) -> str:
    if raw is None:
        return "UNKNOWN"
    s = raw.upper()
    if "GREEN" in s:
        return "GREEN"
    if "YELLOW" in s:
        return "YELLOW"
    if "RED" in s:
        return "RED"
    if "UNINTERPRETABLE" in s:
        return "UNINTERPRETABLE"
    return "UNKNOWN"


# ── Graph data structures ────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class Node:
    type:  str    # "Candidate" | "MechanismFamily" | "AssetClass" | ...
    id:    str
    attrs: tuple      # tuple of (key, value) pairs — frozen for hashing

    @classmethod
    def make(cls, type: str, id: str, **attrs) -> "Node":
        return cls(type, id, tuple(sorted(attrs.items())))

    def attr(self, key: str, default=None):
        for k, v in self.attrs:
            if k == key:
                return v
        return default


@dataclasses.dataclass(frozen=True)
class Edge:
    type:     str
    source:   Node
    target:   Node
    attrs:    tuple = ()


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], Node] = {}
        self.edges: list[Edge] = []
        self._out: dict[Node, list[Edge]] = defaultdict(list)
        self._in:  dict[Node, list[Edge]] = defaultdict(list)

    def add_node(self, type: str, id: str, **attrs) -> Node:
        key = (type, id)
        if key in self.nodes:
            return self.nodes[key]
        n = Node.make(type, id, **attrs)
        self.nodes[key] = n
        return n

    def add_edge(self, etype: str, source: Node, target: Node, **attrs) -> None:
        e = Edge(etype, source, target, tuple(sorted(attrs.items())))
        self.edges.append(e)
        self._out[source].append(e)
        self._in[target].append(e)

    def nodes_of_type(self, type: str) -> list[Node]:
        return [n for (t, _), n in self.nodes.items() if t == type]

    def out_edges(self, n: Node, etype: str | None = None) -> list[Edge]:
        es = self._out.get(n, [])
        return [e for e in es if etype is None or e.type == etype]

    def in_edges(self, n: Node, etype: str | None = None) -> list[Edge]:
        es = self._in.get(n, [])
        return [e for e in es if etype is None or e.type == etype]

    def neighbors(self, n: Node, etype: str | None = None,
                  direction: str = "out") -> list[Node]:
        if direction == "out":
            return [e.target for e in self.out_edges(n, etype)]
        elif direction == "in":
            return [e.source for e in self.in_edges(n, etype)]
        else:
            raise ValueError(f"unknown direction {direction!r}")


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_gate_runs(path: Path = GATE_LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_cross_review(path: Path = CROSS_REVIEW_LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _infer_sample_window(entry: dict) -> tuple[str | None, str | None, int | None]:
    """Get sample window from entry.

    Priority:
      1. explicit sample_start / sample_end fields (added 2026-05-29 to run_gate)
      2. heuristic regex parse of mechanism description (legacy entries)
      3. None
    """
    n_months = entry.get("n_months")
    # Priority 1: explicit fields
    if entry.get("sample_start") and entry.get("sample_end"):
        return entry["sample_start"], entry["sample_end"], n_months
    # Priority 2: regex parse
    desc = entry.get("mechanism", "") + " " + entry.get("name", "")
    m = re.search(r"(\d{4})\s*[-/]\s*(\d{4})", desc)
    if m:
        return f"{m.group(1)}-01-01", f"{m.group(2)}-12-31", n_months
    return None, None, n_months


# ── Deployed sleeve registry (manually anchored — small set) ─────────────────

DEPLOYED_SLEEVES = {
    "equity_book":  {"families": ["earnings_underreaction"], "weight": 0.70,
                      "mechanism_desc": "D-PEAD + analyst revision (earnings underreaction family)"},
    "carry_book":   {"families": ["carry", "breadth_expansion"], "weight": 0.25,
                      "mechanism_desc": "4-leg cross-asset carry (cmdty / FX / rates_us / rates_xc)"},
    "tsmom_book":   {"families": ["tsmom"], "weight": 0.05,
                      "mechanism_desc": "5-leg futures TSMOM (Moskowitz-Ooi-Pedersen 2012)"},
}


# ── Graph builder ────────────────────────────────────────────────────────────

def build_graph() -> KnowledgeGraph:
    g = KnowledgeGraph()

    # Pre-add canonical fixed nodes
    for v in ("GREEN", "YELLOW", "RED", "UNINTERPRETABLE", "UNKNOWN"):
        g.add_node("Verdict", v)
    for ac in ("equity", "etfs", "futures", "vol", "credit", "mixed"):
        g.add_node("AssetClass", ac)
    for family, _ in _MECHANISM_TAXONOMY:
        g.add_node("MechanismFamily", family)
    g.add_node("MechanismFamily", "unclassified")
    # Parent families + child→parent edges
    for parent, children in _PARENT_FAMILIES.items():
        parent_node = g.add_node("ParentFamily", parent)
        for child in children:
            child_node = g.add_node("MechanismFamily", child)
            g.add_edge("rolls_up_to", child_node, parent_node)

    # Load candidates from gate_runs.jsonl
    for entry in _load_gate_runs():
        name = entry.get("name") or "unnamed"
        mechanism = entry.get("mechanism") or ""
        verdict = normalize_verdict(entry.get("verdict"))
        sample_start, sample_end, n_months = _infer_sample_window(entry)
        sample_cov = _sample_coverage(sample_start, sample_end, n_months)

        # Candidate node
        cand = g.add_node(
            "Candidate", name,
            mechanism_desc=mechanism,
            standalone_sharpe=entry.get("standalone_sharpe"),
            alpha_t_ff5umd=entry.get("alpha_t_ff5umd"),
            deflated_sr=entry.get("deflated_sr"),
            oos_sharpe=entry.get("oos_sharpe"),
            corr_with_book=entry.get("corr_with_book"),
            ts=entry.get("ts"),
            n_months=n_months,
        )

        # Sample window node
        win_id = f"{sample_cov.start}_to_{sample_cov.end}_n{sample_cov.n_months}"
        win = g.add_node("SampleWindow", win_id,
                          start=sample_cov.start, end=sample_cov.end,
                          n_months=sample_cov.n_months,
                          stress_covered=sample_cov.stress_covered,
                          stress_missed=sample_cov.stress_missed,
                          coverage_ratio=round(sample_cov.coverage_ratio, 3))
        g.add_edge("tested_on", cand, win)

        # Mechanism family edges
        for family in classify_mechanism(mechanism, name):
            fam_node = g.add_node("MechanismFamily", family)
            g.add_edge("has_mechanism", cand, fam_node)

        # Asset class edge
        ac = classify_asset_class(mechanism, name)
        ac_node = g.add_node("AssetClass", ac)
        g.add_edge("in_asset_class", cand, ac_node)

        # Verdict edge
        ver_node = g.nodes[("Verdict", verdict)]
        g.add_edge("received", cand, ver_node)

    # Load themes from cross_review_ledger.jsonl
    for entry in _load_cross_review():
        cand_name = entry.get("candidate") or "unnamed"
        cand = g.add_node("Candidate", cand_name)
        for theme_name in entry.get("consensus", {}).get("themes", []):
            theme_node = g.add_node("Theme", theme_name)
            g.add_edge("has_theme", cand, theme_node)

    # Deployed sleeves
    for sname, sdata in DEPLOYED_SLEEVES.items():
        sleeve = g.add_node("Sleeve", sname,
                             weight=sdata["weight"],
                             mechanism_desc=sdata["mechanism_desc"])
        for fam in sdata["families"]:
            fam_node = g.add_node("MechanismFamily", fam)
            g.add_edge("uses_mechanism", sleeve, fam_node)

    # Auto-infer "similar_to" edges via shared mechanism family OR shared parent.
    # Two candidates are "similar_to" if they share at least one direct family
    # OR at least one parent family. The parent-level similarity catches
    # e.g. "Quality" and "Residual Momentum" both being equity_factor children
    # (the cousin-detection use case).
    candidates = g.nodes_of_type("Candidate")

    def _families_with_parents(c: Node) -> set[str]:
        direct = {n.id for n in g.neighbors(c, "has_mechanism")
                   if n.id != "unclassified"}
        parents = set()
        for f in direct:
            parents.update(parent_families_for(f))
        return direct | parents

    for c1 in candidates:
        keys1 = _families_with_parents(c1)
        if not keys1:
            continue
        for c2 in candidates:
            if c1 is c2:
                continue
            keys2 = _families_with_parents(c2)
            shared = keys1 & keys2
            if shared:
                # Tag whether similarity is direct family vs parent only
                direct1 = {n.id for n in g.neighbors(c1, "has_mechanism")}
                direct2 = {n.id for n in g.neighbors(c2, "has_mechanism")}
                level = "direct" if direct1 & direct2 else "parent"
                g.add_edge("similar_to", c1, c2,
                            shared=",".join(sorted(shared)),
                            level=level)

    return g


# ── Query API ────────────────────────────────────────────────────────────────

def candidates_by_mechanism_family(g: KnowledgeGraph, family: str) -> list[Node]:
    """All candidates that have been tested with the given mechanism family."""
    fam = g.nodes.get(("MechanismFamily", family))
    if fam is None:
        return []
    return g.neighbors(fam, "has_mechanism", direction="in")


def candidates_by_verdict(g: KnowledgeGraph, verdict: str) -> list[Node]:
    ver = g.nodes.get(("Verdict", verdict.upper()))
    if ver is None:
        return []
    return g.neighbors(ver, "received", direction="in")


def candidate_blind_spots(g: KnowledgeGraph) -> list[tuple[str, str]]:
    """Cells in (AssetClass × MechanismFamily) NEVER tested. Returns sorted
    list. The most interesting research opportunities."""
    asset_classes = [n.id for n in g.nodes_of_type("AssetClass")]
    families = [n.id for n in g.nodes_of_type("MechanismFamily")
                if n.id != "unclassified"]
    tested = set()
    for cand in g.nodes_of_type("Candidate"):
        ac_neighbors = g.neighbors(cand, "in_asset_class")
        fam_neighbors = g.neighbors(cand, "has_mechanism")
        for ac in ac_neighbors:
            for fam in fam_neighbors:
                if fam.id != "unclassified":
                    tested.add((ac.id, fam.id))
    untested = []
    for ac in asset_classes:
        for fam in families:
            if (ac, fam) not in tested:
                untested.append((ac, fam))
    return sorted(untested)


def similar_candidates(g: KnowledgeGraph, name: str) -> list[Node]:
    cand = g.nodes.get(("Candidate", name))
    if cand is None:
        return []
    return g.neighbors(cand, "similar_to")


def failure_theme_clusters(g: KnowledgeGraph) -> dict[str, list[str]]:
    """For RED candidates that have cross-review themes, group by theme."""
    red_cands = candidates_by_verdict(g, "RED")
    clusters: dict[str, list[str]] = defaultdict(list)
    for cand in red_cands:
        themes = g.neighbors(cand, "has_theme")
        for theme in themes:
            clusters[theme.id].append(cand.id)
    return dict(clusters)


def coverage_matrix(g: KnowledgeGraph) -> pd.DataFrame:
    """DataFrame: rows=mechanism family, cols=asset class, values=#candidates."""
    asset_classes = sorted([n.id for n in g.nodes_of_type("AssetClass")])
    families = sorted([n.id for n in g.nodes_of_type("MechanismFamily")
                        if n.id != "unclassified"])
    matrix = pd.DataFrame(0, index=families, columns=asset_classes, dtype=int)
    for cand in g.nodes_of_type("Candidate"):
        acs = [n.id for n in g.neighbors(cand, "in_asset_class")]
        fams = [n.id for n in g.neighbors(cand, "has_mechanism")
                 if n.id != "unclassified"]
        for ac in acs:
            for fam in fams:
                if ac in matrix.columns and fam in matrix.index:
                    matrix.loc[fam, ac] += 1
    return matrix


def deployed_overlap_check(g: KnowledgeGraph, candidate_name: str) -> dict:
    """For a new candidate, identify which deployed sleeves share mechanism
    at DIRECT family level OR PARENT family level.

    Parent-level overlap is the cousin-detection use case (e.g. residual_momentum
    cousin to deployed equity_book whose mechanism is earnings_underreaction —
    both belong to equity_factor parent)."""
    cand = g.nodes.get(("Candidate", candidate_name))
    if cand is None:
        return {"error": f"candidate {candidate_name!r} not in graph"}
    cand_direct = {n.id for n in g.neighbors(cand, "has_mechanism")
                    if n.id != "unclassified"}
    cand_parents = set()
    for f in cand_direct:
        cand_parents.update(parent_families_for(f))
    overlap = {}
    for sleeve in g.nodes_of_type("Sleeve"):
        sleeve_direct = {n.id for n in g.neighbors(sleeve, "uses_mechanism")}
        sleeve_parents = set()
        for f in sleeve_direct:
            sleeve_parents.update(parent_families_for(f))
        direct_shared = sleeve_direct & cand_direct
        parent_shared = (sleeve_parents | sleeve_direct) & (cand_parents | cand_direct)
        # Only count if there's any overlap
        if direct_shared or parent_shared - direct_shared:
            overlap[sleeve.id] = {
                "direct_shared_families": sorted(direct_shared),
                "parent_level_overlap":   sorted(parent_shared - direct_shared),
                "sleeve_weight":          sleeve.attr("weight"),
                "sleeve_mechanism":       sleeve.attr("mechanism_desc"),
                "overlap_strength":       "direct" if direct_shared else "parent_only",
            }
    return overlap


def stress_window_summary(g: KnowledgeGraph) -> pd.DataFrame:
    """For each candidate, summarize stress-window coverage of its sample."""
    rows = []
    for cand in g.nodes_of_type("Candidate"):
        wins = g.neighbors(cand, "tested_on")
        if not wins:
            continue
        w = wins[0]
        rows.append({
            "candidate":     cand.id,
            "verdict":       g.neighbors(cand, "received")[0].id
                              if g.neighbors(cand, "received") else "UNKNOWN",
            "sample_start":  w.attr("start"),
            "sample_end":    w.attr("end"),
            "n_months":      w.attr("n_months"),
            "stress_covered_n": len(w.attr("stress_covered", ())),
            "stress_missed_n":  len(w.attr("stress_missed", ())),
            "coverage_ratio":  w.attr("coverage_ratio"),
            "missed":          ",".join(w.attr("stress_missed", ())[:3]),
        })
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def summary(g: KnowledgeGraph) -> dict:
    """High-level summary of the graph for sanity-checking."""
    counts = defaultdict(int)
    for (t, _) in g.nodes:
        counts[t] += 1
    edge_counts = defaultdict(int)
    for e in g.edges:
        edge_counts[e.type] += 1
    return {
        "n_nodes_by_type": dict(counts),
        "n_edges_by_type": dict(edge_counts),
        "n_candidates":    counts["Candidate"],
        "n_total_edges":   len(g.edges),
    }
