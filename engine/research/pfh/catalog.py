"""engine/research/pfh/catalog.py — Labeled mechanism dataset.

Reads structured research history (mechanism_library + graveyard) and
produces a unified labeled dataset for Bayesian inference.

The atomic unit is a LabeledMechanism: a (family, role, market) triple
with a verdict in {GREEN, YELLOW, RED} and provenance metadata. The
mapping from raw YAML/JSON entries to LabeledMechanisms is hand-coded
here because the source files have heterogeneous schemas — a "purpose"
field in mechanism_library YAMLs and a free-text "why" in graveyard.json.

Caller contract: load_labeled_mechanisms() always returns the same
output for the same disk state (no LLM call, no randomness). PFH's
determinism depends on this.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
MECHANISM_LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
GRAVEYARD_PATH = REPO_ROOT / "data" / "research" / "graveyard.json"


@dataclass
class LabeledMechanism:
    """One labeled observation in the prior dataset.

    verdict ∈ {"GREEN", "YELLOW", "RED"}. YELLOW means "conditional
    GREEN" or "deployed but broken" — kept separate because for
    Bayesian inference we treat YELLOW as 0.5 weight on each direction
    (compromise between "use it as positive" and "ignore it").
    """
    name:               str
    family:             str          # raw family name as appears in source
    family_normalized:  str          # snake_case lowercase
    parent_family:      Optional[str]
    role:               Optional[str]   # alpha_seeker / risk_premium_harvester / etc.
    market:             Optional[str]   # inferred: us_equity / g10_futures / etc.
    verdict:            str             # GREEN / YELLOW / RED
    canonical_paper_id: Optional[str]
    publication_year:   Optional[int]
    failure_reason:     Optional[str]   # for RED only
    source:             str             # "library" / "graveyard"
    source_path:        str             # for audit trail

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Normalization helpers ──────────────────────────────────────────────


_SNAKE_RE = re.compile(r"[^\w]+")


# Hand-curated alias table — different source files use different
# family naming conventions, and naive snake-case normalization leaves
# semantically-identical families in separate cells. Each alias here
# is a SENIOR-LEVEL JUDGMENT that the source-family genuinely belongs
# to the canonical-family for Bayesian-prior aggregation purposes.
#
# Audit rule: aliases should only fire when the mechanism is the same;
# similar-but-different mechanisms (e.g. TSMOM vs naive risk-parity)
# stay separate even if they look related.
_FAMILY_ALIASES: dict[str, str] = {
    # PEAD family — graveyard uses "forward-earnings information" for
    # the same mechanism the library calls "earnings_underreaction".
    # Confirmed: China PEAD, guidance drift, restatement drift, D-PEAD-plus
    # overlay, pre-FOMC drift, labor-signal drift are all earnings-info
    # cousins of post_earnings_drift.
    "forward_earnings_information": "earnings_underreaction",

    # NOTE: The following graveyard families are INTENTIONALLY kept
    # separate from library equivalents because they bundle hetero-
    # geneous mechanisms:
    #   - "cross_sectional_equity_published" bundles 5 different
    #     equity anomalies (xs mom, 52-week-high, ST reversal, sector
    #     mom, idio vol). Aliasing them all to "momentum" would be
    #     overaggregation.
    #   - "macro_trend_risk_parity" bundles TSMOM + dual-momentum +
    #     risk-parity + cross-asset macro. Aliasing to "tsmom" would
    #     pollute the TSMOM cell with non-TSMOM evidence.
    #   - "text_machine_learning" + "fundamentals_accounting" are
    #     already at the right granularity for our purposes.
}


def _normalize_family(raw: str) -> str:
    """Lowercase + snake_case + alias-resolution.

    Aliases come from _FAMILY_ALIASES (hand-curated). Naive normalization
    alone would leave 'forward-earnings information' and 'earnings
    underreaction' in different cells even though they're the same
    mechanism family.
    """
    if not raw:
        return "unknown"
    s = _SNAKE_RE.sub("_", raw.strip().lower()).strip("_")
    if not s:
        return "unknown"
    return _FAMILY_ALIASES.get(s, s)


def _infer_market_from_name(name: str, family: str) -> Optional[str]:
    """Heuristic market label from name + family text. Returns one of
    {us_equity, intl_equity, em_equity, futures, fx, rates, credit,
     options, mixed} or None.
    """
    text = f"{name} {family}".lower()
    if "china" in text or "a-share" in text or "a share" in text:
        return "cn_equity"
    if "japan" in text or "jp " in text or "jpead" in text:
        return "jp_equity"
    if "eu " in text or "europe" in text:
        return "eu_equity"
    if "g10" in text or "futures" in text or "cmdty" in text:
        return "futures"
    if "fx" in text:
        return "fx"
    if "rates" in text or "bond" in text or "treasury" in text:
        return "rates"
    if "credit" in text or "corp bond" in text:
        return "credit"
    if "option" in text or "vol " in text or "vrp" in text:
        return "options"
    if "equity" in text or "stock" in text or "share" in text or "pead" in text:
        return "us_equity"
    return None


# ── Library YAML → LabeledMechanism ────────────────────────────────────


# Per discussion 2026-06-01: 6 GREEN, 2 YELLOW, 4 library-RED markers
# (cousin_anchor entries marked "tested → RED"), 2 UNTESTED candidates.
# This mapping table makes the verdict assignment explicit + auditable
# rather than scraping the noisy "purpose" comment string.
_LIBRARY_VERDICT_OVERRIDES = {
    # GREEN (deployed) — by purpose=deployed_sleeve / deploy_replacement /
    # hedge_replacement OR purpose=cousin_anchor with "DEPLOYED" in note
    "crisis_hedge_tlt_gld":        "GREEN",
    "post_earnings_drift_pit_sn":  "GREEN",
    "tail_hedge_put_spread":       "GREEN",
    "cross_asset_carry":           "GREEN",   # DEPLOYED 4-leg
    "post_earnings_drift":         "GREEN",   # DEPLOYED anchor (D_PEAD)
    "time_series_momentum":        "GREEN",   # DEPLOYED 5-leg Axis B

    # YELLOW — broken deployments, conditional alphas
    "mom_hedge_overlay":           "YELLOW",  # broken, decommissioning

    # RED (library cousin_anchor entries marked "tested → RED")
    "bond_xsmom":                  "RED",
    "quality_qmj":                 "RED",
    "residual_momentum":           "RED",
    "variance_risk_premium":       "RED",

    # UNTESTED (candidate, no verdict yet — excluded from labeled set)
    "equity_xsmom_jt":             None,
    "low_vol_bab":                 None,
}


def _load_library_mechanisms() -> list[LabeledMechanism]:
    """Read mechanism_library YAMLs → LabeledMechanism list."""
    out: list[LabeledMechanism] = []
    if not MECHANISM_LIBRARY_DIR.is_dir():
        return out
    for p in sorted(MECHANISM_LIBRARY_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue   # _canonical_papers etc.
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("library %s parse failed: %s", p.name, exc)
            continue

        mech_id = raw.get("id") or p.stem
        verdict = _LIBRARY_VERDICT_OVERRIDES.get(mech_id)
        if verdict is None:
            continue  # untested / unknown — excluded from labeled set

        family = str(raw.get("family") or "unknown")
        pub_year = None
        snoop = raw.get("was_known_before_our_data_cutoff") or {}
        if isinstance(snoop, dict):
            pub_date = snoop.get("publication_date")
            if isinstance(pub_date, str) and len(pub_date) >= 4:
                try:
                    pub_year = int(pub_date[:4])
                except ValueError:
                    pass

        out.append(LabeledMechanism(
            name=mech_id,
            family=family,
            family_normalized=_normalize_family(family),
            parent_family=raw.get("parent_family"),
            role=None,   # library schema doesn't pin a role; infer downstream
            market=_infer_market_from_name(mech_id, family),
            verdict=verdict,
            canonical_paper_id=raw.get("canonical_paper_id"),
            publication_year=pub_year,
            failure_reason=None,
            source="library",
            source_path=str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
        ))
    return out


# ── Graveyard → LabeledMechanism ──────────────────────────────────────


def _load_graveyard_mechanisms() -> list[LabeledMechanism]:
    if not GRAVEYARD_PATH.is_file():
        return []
    try:
        data = json.loads(GRAVEYARD_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("graveyard parse failed: %s", exc)
        return []

    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []

    out: list[LabeledMechanism] = []
    for e in entries:
        name = str(e.get("name") or "")
        family = str(e.get("family") or "unknown")
        verdict = str(e.get("verdict") or "RED").upper()
        if verdict not in ("RED", "YELLOW"):
            verdict = "RED"   # graveyard convention
        out.append(LabeledMechanism(
            name=name,
            family=family,
            family_normalized=_normalize_family(family),
            parent_family=None,
            role=None,
            market=_infer_market_from_name(name, family),
            verdict=verdict,
            canonical_paper_id=None,
            publication_year=None,
            failure_reason=str(e.get("why") or "")[:500],
            source="graveyard",
            source_path="data/research/graveyard.json",
        ))
    return out


# ── Public API ────────────────────────────────────────────────────────


def load_labeled_mechanisms() -> list[LabeledMechanism]:
    """Aggregate library + graveyard → unified LabeledMechanism list.

    Deterministic — same disk state ⇒ same output. PFH's reproducibility
    depends on this. Sort order is (source asc, name asc) so downstream
    aggregations are deterministic.
    """
    out = _load_library_mechanisms() + _load_graveyard_mechanisms()
    out.sort(key=lambda m: (m.source, m.name))
    return out


def overall_base_rate(labels: list[LabeledMechanism]) -> dict:
    """Headline counts + the overall P(GREEN) estimate to be used as
    weak prior centering for per-family Beta-Binomial inference.

    YELLOW counts as 0.5 GREEN and 0.5 RED in the success rate
    estimate (it's genuinely ambiguous — broken deployments and
    conditional alphas are not pure successes nor pure failures).
    """
    n_green  = sum(1 for m in labels if m.verdict == "GREEN")
    n_yellow = sum(1 for m in labels if m.verdict == "YELLOW")
    n_red    = sum(1 for m in labels if m.verdict == "RED")
    n_total  = n_green + n_yellow + n_red
    # Effective green: GREEN + 0.5 * YELLOW
    n_eff_green = n_green + 0.5 * n_yellow
    base_rate = (n_eff_green / n_total) if n_total else None
    return {
        "n_green":      n_green,
        "n_yellow":     n_yellow,
        "n_red":        n_red,
        "n_total":      n_total,
        "n_eff_green":  n_eff_green,
        "p_green":      round(base_rate, 4) if base_rate is not None else None,
    }


def per_family_counts(
    labels: list[LabeledMechanism],
) -> dict[str, dict]:
    """Per-family aggregated counts. Key is family_normalized."""
    by_fam: dict[str, dict] = {}
    for m in labels:
        f = m.family_normalized
        if f not in by_fam:
            by_fam[f] = {
                "family":  m.family,
                "n_green": 0,
                "n_yellow": 0,
                "n_red":   0,
                "members": [],
            }
        by_fam[f]["members"].append(m.name)
        if m.verdict == "GREEN":
            by_fam[f]["n_green"] += 1
        elif m.verdict == "YELLOW":
            by_fam[f]["n_yellow"] += 1
        elif m.verdict == "RED":
            by_fam[f]["n_red"] += 1
    return by_fam
