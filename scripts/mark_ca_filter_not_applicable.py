"""scripts/mark_ca_filter_not_applicable.py — operational cleanup.

Updates the 6 deployed/pending sleeves where CA filter (BTC paper §3)
does NOT apply with sleeve-specific reasons. Carry stays at the
PBB-validated k=3.0.

Per [[feedback-pre-implementation-fitness-check-2026-06-01]] +
[[project-multi-asset-ca-filter-gap-2026-06-01]]: borrowed concepts
must be assessed for fitness per deployed surface BEFORE building
infrastructure to apply them.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB = REPO_ROOT / "data" / "research" / "mechanism_library"

NA_PLAN: list[tuple[str, str, int, str, str]] = [
    # (sleeve_id, signal_type_for_audit, tcost_bps, reason_short, reason_long)
    (
        "post_earnings_drift", "cross_sect_rank", 30,
        "4-day event horizon enforces minimum hold",
        "PEAD is event-driven with a fixed 4-day post-announcement holding period. "
        "The event horizon itself enforces the minimum hold semantics the CA filter "
        "would provide for a free-trading strategy. Signal is a quintile rank "
        "assignment at the event date, not a per-period predicted return — so the "
        "paper's |ER| > k×tcost gate doesn't map: there's no per-period decision "
        "between rebalances. Cost rigor here comes from the Almgren-Chriss audit "
        "(already in place) + event-frequency analysis."
    ),
    (
        "post_earnings_drift_pit_sn", "cross_sect_rank", 30,
        "Same family as PEAD — 4-day event horizon",
        "PIT-clean Sales-Net SUE replacement candidate inherits the same 4-day "
        "event horizon as the deployed D_PEAD. CA filter does not apply for the "
        "same reason: event-driven holding period is the implicit gate."
    ),
    (
        "time_series_momentum", "vol_norm_zscore", 8,
        "Continuous per-asset z-score → vol-weighted, not L/S binary",
        "TSMOM uses a per-asset 12-1 momentum z-score that maps to continuous "
        "vol-weighted positions, not a binary {-1, 0, +1} long/short decision. "
        "The paper's CA filter formula assumes per-trade binary gating on point "
        "expected-return forecasts; tsmom's vol-target + signal-smoothing layer "
        "already encodes the equivalent dampening (small z → small position → "
        "small turnover) without needing a separate cost-gating filter. Adding "
        "CA on top would either double-count the smoothing or cause stale-position "
        "drift in regime transitions. Cost rigor: Almgren-Chriss audit (in place) "
        "+ vol-target's implicit signal damping is the established discipline."
    ),
    (
        "crisis_hedge_tlt_gld", "regime_indicator", 5,
        "Regime trigger IS the gate — no separate trade/hold decision",
        "Crisis hedge fires positions conditional on VIX 1y z-score regime "
        "(CALM/NORMAL/STRESS). The regime classification itself is the trade "
        "trigger — there is no separate 'predicted return > cost threshold' "
        "decision because the strategy doesn't predict returns; it allocates to "
        "TLT/GLD when STRESS regime is active. The CA filter abstraction (per-"
        "period predicted-return gate) doesn't map onto a regime trigger. Cost "
        "rigor for this sleeve: ETF-only universe (TLT/GLD, ~5bp RT) + Almgren-"
        "Chriss audit + regime-change frequency analysis."
    ),
    (
        "mom_hedge_overlay", "vol_norm_zscore", 10,
        "Single-asset continuous short overlay — no L/S decision per period",
        "Mom hedge overlay is a single-asset MTUM short with continuous "
        "beta-residual z weight. There is no per-period 'trade or hold' binary "
        "decision the CA filter could gate — the overlay weight is continuously "
        "adjusted as a hedge against the rest of the book's momentum exposure. "
        "Cost rigor: Almgren-Chriss audit (in place) + overlay weight is small "
        "enough (typically <5% gross) that filter overhead would exceed the cost "
        "it could save."
    ),
    (
        "tail_hedge_put_spread", "binary_trigger", 50,
        "Calendar-driven SPX put-spread roll — no per-period gate",
        "Tail hedge is a monthly calendar-driven SPX put-spread roll (delta -25 "
        "/ -10) at 5% notional. Roll dates are fixed by option expiration cycle, "
        "not by a predicted-return signal. There is no signal-versus-cost "
        "decision per period; the roll always happens on expiry. Cost rigor: "
        "Almgren-Chriss audit (in place) accounts for the ~50bp RT option spread "
        "cost as a known recurring expense, not a gateable event."
    ),
]

NA_BLOCK_TEMPLATE = """\
  # ── Phase 5.7 follow-up: CA filter NOT APPLICABLE for this sleeve ──
  # See [[project-multi-asset-ca-filter-gap-2026-06-01]] +
  #     [[feedback-pre-implementation-fitness-check-2026-06-01]].
  # Cost rigor for this sleeve comes from the cost_model Almgren-Chriss
  # audit above; CA filter (BTC paper §3) does not apply.
  ca_filter_k_method: not_applicable
  ca_filter_k_audit_date: "2026-06-01"
  ca_filter_k_not_applicable_reason: |
    {long_reason}
  ca_signal_type: {signal_type}
  tcost_round_trip_bps: {tcost_bps}
"""

OLD_BLOCK_RE = re.compile(
    r"\n  # ── Phase 5\.7 Cost-Aware Execution Filter \(CA\) ──.*?\n  tcost_round_trip_bps: \d+\n",
    re.DOTALL,
)


def update_yaml(yaml_path: Path, signal_type: str, tcost_bps: int,
                  long_reason: str) -> str:
    """Replace the existing paper_default CA block with not_applicable block."""
    text = yaml_path.read_text(encoding="utf-8")
    if "ca_filter_k_method: not_applicable" in text:
        return "already_na"
    # Indent the multi-line reason for YAML safety
    indented = long_reason.replace("\n", "\n    ")
    new_block = NA_BLOCK_TEMPLATE.format(
        long_reason=indented, signal_type=signal_type, tcost_bps=tcost_bps,
    )
    new_text, n = OLD_BLOCK_RE.subn("\n" + new_block, text)
    if n == 0:
        return "no_block_found"
    yaml_path.write_text(new_text, encoding="utf-8")
    return "updated"


def main() -> int:
    print("Marking CA filter NOT APPLICABLE for non-fitting sleeves:")
    for stem, sig_type, tcost_bps, short_reason, long_reason in NA_PLAN:
        yp = LIB / f"{stem}.yaml"
        if not yp.is_file():
            print(f"  {stem:30s} MISSING_YAML")
            continue
        status = update_yaml(yp, sig_type, tcost_bps, long_reason)
        print(f"  {stem:30s} {status:20s}  ({short_reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
