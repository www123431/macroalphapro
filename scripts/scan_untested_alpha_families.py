"""scripts/scan_untested_alpha_families.py — enumerate alpha-factor
families across (deployed / graveyard / untested) categories to
identify genuinely-orthogonal candidates worth exploring next.

Senior-quant doctrine: don't waste time on families that are:
  - Already deployed in our book (DEPLOYED)
  - Killed in graveyard (DEAD)
  - Covered by adjacent deployed sleeves (REDUNDANT)

Surface only: 'LIVE_UNTESTED' families with high orthogonality
potential vs current book.

Reference universe: Harvey-Liu-Zhu 2016 + Hou-Karolyi-Kho 2020
"Replicating Anomalies" — 452-anomaly catalog with survival rates.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBRARY = REPO_ROOT / "data" / "research" / "mechanism_library"
GRAVEYARD = REPO_ROOT / "data" / "research" / "graveyard.json"

# Senior reference list — major alpha factor families per HKK 2020
# "Replicating Anomalies" + standard academic categories.
# Each entry: (family_id, brief, replication_status_in_HKK_2020)
REFERENCE_FAMILIES = [
    # Behavioral / underreaction
    ("earnings_underreaction",  "PEAD / Bernard-Thomas 1989",         "SURVIVES (large)"),
    ("analyst_revision",        "Stickel 1991 revision drift",          "SURVIVES (partial)"),
    ("guidance_drift",          "management guidance drift",            "PARTIAL"),
    ("long_term_reversal",      "De Bondt-Thaler 1985 (3-5y)",         "SURVIVES (large)"),
    ("short_term_reversal",     "Jegadeesh 1990 1-month reversal",      "SURVIVES (large)"),
    ("post_earnings_drift_intl","cross-country PEAD (Hou-K-K 2011)",    "PARTIAL"),
    # Momentum
    ("momentum",                "Jegadeesh-Titman 1993 12-2 momentum",  "SURVIVES (large)"),
    ("industry_momentum",       "Moskowitz-Grinblatt 1999 industry-MOM","SURVIVES"),
    ("residual_momentum",       "Blitz-Huij-Martens 2011 FF3-residual", "SURVIVES"),
    ("tsmom",                   "Moskowitz-Ooi-Pedersen 2012 TSMOM",    "SURVIVES (large)"),
    # Quality / profitability
    ("quality_qmj",             "Asness-Frazzini-Pedersen 2019 QMJ",    "SURVIVES (large)"),
    ("profitability",           "Novy-Marx 2013 gross profitability",   "SURVIVES (large)"),
    ("accruals",                "Sloan 1996 accruals anomaly",          "SURVIVES (partial)"),
    # Investment / financing
    ("investment",              "Cooper-Gulen-Schill 2008 asset growth","SURVIVES"),
    ("issuance",                "Pontiff-Woodgate 2008 net issuance",   "SURVIVES (large)"),
    ("buybacks",                "Ikenberry-Lakonishok-Vermaelen 1995",  "SURVIVES"),
    # Risk premia (often deployed as carry / TSMOM)
    ("carry",                   "Koijen-Moskowitz cross-asset carry",   "SURVIVES (large)"),
    ("low_vol_bab",             "Frazzini-Pedersen 2014 BAB",           "SURVIVES (large)"),
    ("term_premium",            "rates term-structure carry",           "SURVIVES"),
    # Volatility / options
    ("variance_risk_premium",   "Carr-Wu 2009 VRP",                     "SURVIVES (large)"),
    ("vol_of_vol",              "Mueller-Hyder VVIX-related",            "PARTIAL"),
    ("skew_premium",            "Bali-Murray 2013 OTM skew premium",    "SURVIVES (small)"),
    # Macro / event-driven
    ("macro_lead_lag",          "Cooper-Priestley 2009 macro forecasts","PARTIAL"),
    ("industry_lead_lag",       "Hong-Lim-Stein 2000 slow info",        "SURVIVES (partial)"),
    ("liquidity_premium",       "Amihud 2002 illiquidity",              "SURVIVES (small)"),
    # Tail / hedge
    ("tail_hedge",              "Israelov 2017 put spread",             "SURVIVES (descriptive)"),
    ("cross_asset_hedge",       "TLT/GLD crisis hedge",                 "SURVIVES (descriptive)"),
    ("factor_hedge",            "MTUM-short style hedge",               "SURVIVES (descriptive)"),
    # Likely-dead per recent literature
    ("news_attention",          "Da-Engelberg-Gao 2011 attention",      "DECAYED (Hwang-Liu 2022)"),
    ("text_sentiment",          "Tetlock 2007 media sentiment",         "DECAYED"),
    ("insider_trading",         "Lakonishok-Lee 2001 insider",           "DECAYED"),
    ("13f_holdings",            "13F holdings-based signals",            "WEAK"),
    ("patent_alpha",            "Hirshleifer 2013 patent intensity",    "DECAYED"),
]


def load_library_families() -> dict[str, dict]:
    """Return {family_id: {status, ids, role, ...}} from library YAMLs."""
    out = defaultdict(lambda: {"ids": [], "status_set": set(), "roles": set()})
    for fp in sorted(LIBRARY.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        fam = d.get("family") or d.get("parent_family")
        if not fam:
            continue
        out[fam]["ids"].append(d.get("id", fp.stem))
        out[fam]["status_set"].add(d.get("status_in_our_book", "UNKNOWN"))
        if "factor_exposure" in d:
            r = d["factor_exposure"].get("proposed_role")
            if r:
                out[fam]["roles"].add(r)
    return dict(out)


def load_graveyard_families() -> set[str]:
    """Family names appearing in graveyard.json + library RED status."""
    families = set()
    # graveyard.json
    if GRAVEYARD.exists():
        try:
            g = json.loads(GRAVEYARD.read_text(encoding="utf-8"))
            for e in g.get("entries", []):
                if e.get("verdict") in ("RED", "AMBER"):
                    fam = e.get("family")
                    if fam:
                        families.add(fam)
        except Exception:
            pass
    # library YAML status_in_our_book = RED
    for fp in sorted(LIBRARY.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if d.get("status_in_our_book") == "RED":
            fam = d.get("family")
            if fam:
                families.add(fam)
    return families


def family_id_alias(fam: str) -> set[str]:
    """Map a family string to plausible alias keys for matching."""
    if not fam:
        return set()
    norm = fam.lower().strip().replace(" ", "_").replace("-", "_")
    # Use graveyard's normalization
    from engine.research.graveyard import _normalize_family
    return {norm, (_normalize_family(fam) or norm)}


def main():
    lib = load_library_families()
    deployed_families = set()
    pending_families = set()
    for fam, info in lib.items():
        if "DEPLOYED" in info["status_set"]:
            deployed_families.update(family_id_alias(fam))
        if "PENDING_DEPLOY" in info["status_set"]:
            pending_families.update(family_id_alias(fam))

    dead_families = set()
    for fam in load_graveyard_families():
        dead_families.update(family_id_alias(fam))

    print("=" * 95)
    print(" ALPHA FAMILY COVERAGE SCAN — find genuinely-orthogonal untested families")
    print("=" * 95)

    print(f"\n  [coverage state]")
    print(f"    library families:       {len(lib)}")
    print(f"    deployed (status DEPLOYED): {sorted([k for k,v in lib.items() if 'DEPLOYED' in v['status_set']])}")
    print(f"    pending (PENDING_DEPLOY):   {sorted([k for k,v in lib.items() if 'PENDING_DEPLOY' in v['status_set']])}")
    print(f"    graveyard distinct families: {len(load_graveyard_families())}")

    # Categorize REFERENCE_FAMILIES
    categories = {"DEPLOYED": [], "PENDING": [], "DEAD": [], "LIVE_UNTESTED": []}
    for (fam_id, brief, hkk_status) in REFERENCE_FAMILIES:
        aliases = family_id_alias(fam_id)
        if aliases & deployed_families:
            categories["DEPLOYED"].append((fam_id, brief, hkk_status))
        elif aliases & pending_families:
            categories["PENDING"].append((fam_id, brief, hkk_status))
        elif aliases & dead_families or "DECAYED" in hkk_status or "WEAK" in hkk_status:
            categories["DEAD"].append((fam_id, brief, hkk_status))
        else:
            categories["LIVE_UNTESTED"].append((fam_id, brief, hkk_status))

    for cat in ["DEPLOYED", "PENDING", "DEAD", "LIVE_UNTESTED"]:
        print(f"\n  [{cat}] ({len(categories[cat])} families)")
        for (fid, brief, status) in sorted(categories[cat], key=lambda x: x[0]):
            print(f"    {fid:<27} — {brief}  ({status})")


if __name__ == "__main__":
    main()
