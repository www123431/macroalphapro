"""engine/validation/pit_audit_book.py — BOOK-LEVEL point-in-time / look-ahead register.

Wraps up the quant-line PIT gate: every DEPLOYED mechanism gets a documented look-ahead
SURFACE + the as-of CONTROL that closes it + a verification status. D_PEAD is
DATA-VERIFIED by the deep audit (engine.validation.pit_audit_dpead); the rest are
CONSTRUCTION-VERIFIED — the signal is a pure function of as_of with the control cited at
a source anchor (institutional model-review = construction review + spot data checks).

The register is config-driven off get_registry(): a deployed strategy with NO documented
surface here is itself a FLAG (forces the documentation to exist). Deterministic, 0-LLM.

Output: data/validation/pit_audit_book_<date>.json + a verdict table.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "validation"


@dataclasses.dataclass
class LookAheadSurface:
    strategy: str
    surface: str            # the specific look-ahead risk for this mechanism
    control: str            # the as-of control that closes it
    anchor: str             # source-code anchor for the control
    verification: str       # "data" | "construction"
    status: str             # PASS | FLAG | INFO


# Per-deployed-mechanism look-ahead surface + as-of control (verified by reading the
# construction; D_PEAD additionally DATA-verified by the deep audit).
SURFACES: dict[str, LookAheadSurface] = {
    "K1_BAB": LookAheadSurface(
        "K1_BAB",
        "beta estimated from trailing returns; monthly rebalance",
        "compute_bab_signal(as_of=...) uses the bab_compat cache up to as_of only; "
        "rebalance = month-end (is_rebalance_day_k1). DQ Inspector Mode 2 gates cache "
        "staleness so a stale (future-blind) cache cannot silently feed the signal.",
        "paper_trade_combined.py:244 get_k1_bab_signal(as_of) + :148 is_rebalance_day_k1; "
        "DQ spec id=70 Mode 2",
        "construction", "PASS"),
    "D_PEAD": LookAheadSurface(
        "D_PEAD",
        "SUE expectation (seasonal-RW sigma) + announcement timing + entry day",
        "DATA-VERIFIED by the deep audit: sigma_8q excludes the current quarter, "
        "eps_adj_lag4 is the true prior-year quarter, rdq monotonic, entry rdq+1 skip-day, "
        "consensus anndats<=rdq-1. Two documented limitations (delisting, restatement).",
        "engine.validation.pit_audit_dpead (LOOK-AHEAD CLEAN + 2 documented limitations)",
        "data", "PASS"),
    "PATH_N": LookAheadSurface(
        "PATH_N",
        "S&P 500 index-inclusion drift; risk = trading before the add is public",
        "active = events where as_of in [entry_date, exit_date]; pending entries only from "
        "effective_date in (as_of, as_of+5d] — entry strictly after the public effective "
        "date, never on the announcement itself.",
        "paper_trade_combined.py:566 active-window + :472/:194 effective_date>(as_of)",
        "construction", "PASS"),
    "CTA_PQTIX": LookAheadSurface(
        "CTA_PQTIX",
        "managed-futures fund (PQTIX) NAV proxy; no cross-sectional estimated signal",
        "holds the fund return series; no firm-level signal to look ahead on; year-end "
        "rebalance (is_rebalance_day_cta). Trivial look-ahead surface.",
        "paper_trade_combined.py:627 get_cta_pqtix_signal + :201 is_rebalance_day_cta",
        "construction", "PASS"),
    "AC_TLT_GLD": LookAheadSurface(
        "AC_TLT_GLD",
        "fixed 50/50 TLT/GLD crisis hedge; no estimated signal",
        "static target weights, no parameter estimated from returns -> no look-ahead "
        "surface beyond the trade-timing lag.",
        "paper_trade_combined.py:662 get_ac_tlt_gld_signal (static weights)",
        "construction", "PASS"),
}

# Validated-but-not-deployed candidate (carry, spec id=77 DRAFT).
CANDIDATE_SURFACES: dict[str, LookAheadSurface] = {
    "cross_asset_carry": LookAheadSurface(
        "cross_asset_carry",
        "futures-curve carry (F_near/F_next) + front-contract roll returns",
        "carry signal uses settlement prices available at formation; front-contract "
        "monthly returns are forward (next-period) with the roll day NaN-masked to avoid "
        "the contract-switch jump. Cross-sectional rank at formation only.",
        "commodity_carry.py / crossasset_carry.py (roll-day NaN mask) + spec id=77 §2",
        "construction", "PASS"),
}


@dataclasses.dataclass
class BookPitReport:
    as_of: str
    surfaces: list
    undocumented: list      # deployed strategies with no surface entry (a FLAG)
    dpead_data_verified: bool
    book_clean: bool
    overall: str


def run_book_pit_audit() -> BookPitReport:
    """Aggregate the per-mechanism PIT surfaces across the LIVE book (config-driven from
    the registry) + the deep D_PEAD data audit + the carry candidate."""
    try:
        from engine.strategies import get_registry
        live = list(get_registry().names())
    except Exception as exc:
        logger.warning("registry unavailable (%s) — using SURFACES keys", exc)
        live = list(SURFACES)

    # Deep, data-verified D_PEAD audit drives that row's status.
    dpead_data_verified = False
    try:
        from engine.validation.pit_audit_dpead import run_pit_audit
        rep = run_pit_audit()
        dpead_data_verified = bool(rep.critical_pass)
        if "D_PEAD" in SURFACES:
            SURFACES["D_PEAD"].status = "PASS" if rep.critical_pass else "FLAG"
    except Exception as exc:
        logger.warning("D_PEAD deep audit unavailable (%s)", exc)

    surfaces, undocumented = [], []
    for name in live:
        s = SURFACES.get(name)
        if s is None:
            undocumented.append(name)                 # forces documentation to exist
        else:
            surfaces.append(s)
    surfaces += list(CANDIDATE_SURFACES.values())     # carry (candidate)

    book_clean = (not undocumented
                  and all(s.status in ("PASS", "INFO") for s in surfaces)
                  and dpead_data_verified)
    overall = ("BOOK LOOK-AHEAD CLEAN (every deployed mechanism keyed at as_of; "
               "D_PEAD data-verified, rest construction-verified)"
               if book_clean else "BOOK PIT INCOMPLETE — see undocumented / FLAG rows")
    return BookPitReport(as_of=datetime.date.today().isoformat(), surfaces=surfaces,
                         undocumented=undocumented, dpead_data_verified=dpead_data_verified,
                         book_clean=book_clean, overall=overall)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rep = run_book_pit_audit()
    print("\n" + "=" * 84)
    print(f"BOOK-LEVEL POINT-IN-TIME REGISTER — as_of {rep.as_of}")
    print(f"  OVERALL: {rep.overall}")
    print("=" * 84)
    for s in rep.surfaces:
        tag = "*" if s.strategy in CANDIDATE_SURFACES else " "
        print(f" {tag}[{s.status:4s}] {s.strategy:18s} ({s.verification}-verified)\n"
              f"      surface: {s.surface}\n      control: {s.control}\n      anchor:  {s.anchor}")
    if rep.undocumented:
        print(f"\n  UNDOCUMENTED deployed strategies (FLAG — add a surface entry): {rep.undocumented}")
    print("=" * 84)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"pit_audit_book_{rep.as_of}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"as_of": rep.as_of, "overall": rep.overall, "book_clean": rep.book_clean,
                   "dpead_data_verified": rep.dpead_data_verified,
                   "undocumented": rep.undocumented,
                   "surfaces": [dataclasses.asdict(s) for s in rep.surfaces]}, f,
                  indent=2, ensure_ascii=False)
    print(f"saved {out}")
    return 0 if rep.book_clean else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
