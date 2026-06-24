"""engine.research.signal_verification — Commit 2 of the flexibility chain.

Verification cards + redundancy gate + the proposed→dispatchable
approval ledger. Designed around the terminal bottleneck named in
the 2026-06-10 senior施工建议: human verification bandwidth. The
card's job is to make verifying ONE new signal a ≤5-minute read
with hand-checkable numbers, instead of a code review.

THE CARD
========
For a registry signal, generate_verification_card() produces:
  - formula source (the actual code, short)
  - fields used + their PIT rules (from FIELD_CATALOG)
  - SPOT CHECKS: concrete (permno, month) samples with raw field
    values AND the computed signal — hand-checkable against the
    underlying filing. No spot check = no verification, only trust.
  - coverage % + distribution tails (winsorization smell test)
  - REDUNDANCY: mean cross-sectional Spearman rank-corr vs every
    existing dispatchable signal. Chordia-Goyal-Saretto 2020: most
    "new" anomalies are correlated variants of old ones. |ρ| > 0.7
    forces family reassignment review — a variant self-declaring a
    fresh family would dodge Bailey-LdP family n_trials accounting
    (legalized p-hacking).

THE LEDGER (proposed → dispatchable)
====================================
Registry entries are CODE (the proposal). The human act of approval
is a LEDGER row (data/research/signal_approvals.jsonl) — mirrors the
ack-workflow pattern: append-only, reason required, auditable.
dispatchable status = status=="dispatchable" in code (grandfathered)
OR an approval row exists. Revocation = tombstone row.
"""
from __future__ import annotations

import datetime as _dt
import inspect
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
APPROVALS_LEDGER = (_REPO_ROOT / "data" / "research"
                      / "signal_approvals.jsonl")
CARDS_DIR = _REPO_ROOT / "docs" / "signal_cards"

# Redundancy bars (Chordia-Goyal-Saretto 2020 motivated)
REDUNDANCY_STRONG_BAR = 0.70   # forced family-review
REDUNDANCY_NOTE_BAR   = 0.40   # noted on card

# Card computation window — recent decade keeps the funda as-of
# merges fast while spanning a full cycle.
CARD_WINDOW_START = "2015-01-01"
CARD_WINDOW_END   = "2024-12-31"

ACK_REASON_MIN = 10


# ────────────────────────────────────────────────────────────────────
# Approval ledger
# ────────────────────────────────────────────────────────────────────
def load_approvals() -> dict[str, dict]:
    """Latest-row-wins per signal key; tombstones suppress."""
    if not APPROVALS_LEDGER.is_file():
        return {}
    latest: dict[str, dict] = {}
    for line in APPROVALS_LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        k = row.get("signal_key")
        if k:
            latest[k] = row
    return {k: r for k, r in latest.items()
            if r.get("kind") != "tombstone"}


def approve_signal(signal_key: str, *, actor: str, reason: str) -> dict:
    """Human approval: proposed → dispatchable. Appends a ledger row.
    Requires the signal to exist in the registry and a real reason
    (institutional ack standard)."""
    from engine.research.signal_registry import SIGNAL_REGISTRY
    if signal_key not in SIGNAL_REGISTRY:
        raise KeyError(f"unknown signal {signal_key!r}")
    reason = (reason or "").strip()
    if len(reason) < ACK_REASON_MIN:
        raise ValueError(f"reason must be >= {ACK_REASON_MIN} chars")
    row = {
        "ts":         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "signal_key": signal_key,
        "kind":       "approval",
        "actor":      actor[:60],
        "reason":     reason[:1000],
    }
    APPROVALS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with APPROVALS_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def revoke_signal(signal_key: str, *, actor: str, reason: str) -> dict:
    """Tombstone — pulls a signal back to proposed."""
    reason = (reason or "").strip()
    if len(reason) < ACK_REASON_MIN:
        raise ValueError(f"reason must be >= {ACK_REASON_MIN} chars")
    row = {
        "ts":         _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "signal_key": signal_key,
        "kind":       "tombstone",
        "actor":      actor[:60],
        "reason":     reason[:1000],
    }
    APPROVALS_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with APPROVALS_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


# ────────────────────────────────────────────────────────────────────
# Signal panel builder (card-window, RAW values, direction-free)
# ────────────────────────────────────────────────────────────────────
def _card_panel(signal_key: str) -> Optional[pd.DataFrame]:
    """Compute the signal's RAW wide panel over the card window using
    the template's own plumbing (so the card verifies EXACTLY what
    dispatch would compute)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _load_crsp_msf, _build_fundamental_signal,
    )
    from engine.research.signal_registry import get_signal
    from types import SimpleNamespace

    sdef = get_signal(signal_key)
    if sdef is None:
        return None
    msf = _load_crsp_msf()
    window = msf[
        (msf["month_end"] >= pd.Timestamp(CARD_WINDOW_START))
        & (msf["month_end"] <= pd.Timestamp(CARD_WINDOW_END))
    ]
    if sdef.kind == "crsp_panel":
        rets = window.pivot(index="month_end", columns="permno",
                               values="ret")
        mc   = window.pivot(index="month_end", columns="permno",
                               values="mktcap")
        return sdef.formula(SimpleNamespace(rets=rets, mktcap=mc))
    return _build_fundamental_signal(window, signal_key)


def _mean_cross_sectional_spearman(
    a: pd.DataFrame, b: pd.DataFrame, *, min_common: int = 100,
) -> Optional[float]:
    """Mean (over months) of cross-sectional Spearman rank corr.
    The standard 'are these the same signal' metric."""
    common_dates = a.index.intersection(b.index)
    rhos: list[float] = []
    for t in common_dates:
        xa = a.loc[t].dropna()
        xb = b.loc[t].dropna()
        common = xa.index.intersection(xb.index)
        if len(common) < min_common:
            continue
        ra = xa.loc[common].rank()
        rb = xb.loc[common].rank()
        rho = ra.corr(rb)
        if pd.notna(rho):
            rhos.append(float(rho))
    if not rhos:
        return None
    return float(np.mean(rhos))


# ────────────────────────────────────────────────────────────────────
# The card
# ────────────────────────────────────────────────────────────────────
def generate_verification_card(
    signal_key: str,
    *,
    n_spot_checks: int = 2,
    redundancy_against: Optional[tuple[str, ...]] = None,
    write_md: bool = True,
) -> Optional[dict]:
    """Build the verification card. Returns a JSON-safe dict; also
    writes docs/signal_cards/<key>_card.md unless write_md=False.

    redundancy_against: override the comparison set (default = all
    dispatchable signals except self). Tests use a narrow set."""
    from engine.research.signal_registry import (
        FIELD_CATALOG, SIGNAL_REGISTRY, dispatchable_signals, get_signal,
    )
    sdef = get_signal(signal_key)
    if sdef is None:
        return None

    panel = _card_panel(signal_key)
    if panel is None or panel.empty:
        return None

    # Coverage + distribution
    n_cells = panel.size
    n_valid = int(panel.notna().sum().sum())
    flat = panel.values[np.isfinite(panel.values.astype(float))]
    dist = {
        "p1":     float(np.percentile(flat, 1)),
        "p50":    float(np.percentile(flat, 50)),
        "p99":    float(np.percentile(flat, 99)),
        "mean":   float(np.mean(flat)),
    } if len(flat) else {}

    # Spot checks at the latest populated month. Pick the most-
    # COVERED permnos (longest non-null history ≈ stable large firms,
    # hand-verifiable against filings) — NOT abs-max values, which
    # selects denominator-pathology outliers (first real card run
    # 2026-06-10 picked op_profit=-18 micro-asset firms; useless for
    # hand verification).
    spot_checks: list[dict] = []
    try:
        last_t = panel.dropna(how="all").index.max()
        row = panel.loc[last_t].dropna()
        coverage = panel.notna().sum()
        picks = (coverage.loc[row.index]
                   .nlargest(n_spot_checks).index)
        for p in picks:
            spot_checks.append({
                "permno":    int(p),
                "month_end": str(pd.Timestamp(last_t).date()),
                "raw_signal_value": float(row[p]),
            })
    except Exception:
        logger.exception("card: spot checks failed for %s", signal_key)

    # Redundancy vs existing signals
    if redundancy_against is None:
        redundancy_against = tuple(
            k for k in dispatchable_signals() if k != signal_key)
    redundancy: dict[str, Optional[float]] = {}
    for other in redundancy_against:
        try:
            other_panel = _card_panel(other)
            redundancy[other] = (
                _mean_cross_sectional_spearman(panel, other_panel)
                if other_panel is not None else None)
        except Exception:
            logger.exception("card: redundancy vs %s failed", other)
            redundancy[other] = None

    strong = {k: v for k, v in redundancy.items()
                if v is not None and abs(v) >= REDUNDANCY_STRONG_BAR}
    family_review_required = bool(strong)
    suggested_family = None
    if strong:
        top = max(strong, key=lambda k: abs(strong[k]))
        suggested_family = SIGNAL_REGISTRY[top].family

    # Formula source for the card
    try:
        formula_src = inspect.getsource(sdef.formula).strip()
    except Exception:
        formula_src = "<source unavailable>"

    card = {
        "signal_key":       signal_key,
        "status":           sdef.status,
        "direction":        sdef.direction,
        "family_declared":  sdef.family,
        "paper_citation":   sdef.paper_citation,
        "pit_notes":        sdef.pit_notes,
        "formula_source":   formula_src,
        "fields": [
            {"key": f, "pit_rule": FIELD_CATALOG[f].pit_rule,
              "units": FIELD_CATALOG[f].units}
            for f in sdef.required_fields
        ],
        "window":           f"{CARD_WINDOW_START}:{CARD_WINDOW_END}",
        "coverage_pct":     round(100.0 * n_valid / n_cells, 1)
                              if n_cells else 0.0,
        "distribution":     dist,
        "spot_checks":      spot_checks,
        "redundancy":       {k: (round(v, 3) if v is not None else None)
                               for k, v in redundancy.items()},
        "redundancy_strong_bar": REDUNDANCY_STRONG_BAR,
        "family_review_required": family_review_required,
        "suggested_family": suggested_family,
        "generated_ts":     _dt.datetime.utcnow().isoformat(
                                timespec="seconds") + "Z",
    }

    if write_md:
        try:
            CARDS_DIR.mkdir(parents=True, exist_ok=True)
            (CARDS_DIR / f"{signal_key}_card.md").write_text(
                _render_card_md(card), encoding="utf-8")
        except Exception:
            logger.exception("card: md write failed for %s", signal_key)
    return card


def _render_card_md(card: dict) -> str:
    lines = [
        f"# SIGNAL VERIFICATION CARD: {card['signal_key']}",
        "",
        f"**status**: {card['status']}  ·  "
        f"**direction**: {card['direction']}  ·  "
        f"**family (declared)**: {card['family_declared']}",
        f"**paper**: {card['paper_citation']}",
        f"**PIT notes**: {card['pit_notes']}",
        f"**window**: {card['window']}  ·  "
        f"**coverage**: {card['coverage_pct']}%",
        "",
        "## Formula",
        "```python",
        card["formula_source"],
        "```",
        "",
        "## Fields (PIT rules)",
    ]
    for f in card["fields"]:
        lines.append(f"- `{f['key']}` — {f['pit_rule']} ({f['units']})")
    lines += ["", "## Spot checks (hand-verify against filings)"]
    for s in card["spot_checks"]:
        lines.append(f"- permno {s['permno']} @ {s['month_end']}: "
                       f"raw = {s['raw_signal_value']:+.6f}")
    d = card["distribution"]
    if d:
        lines += ["", "## Distribution",
                    f"p1={d['p1']:+.4f}  p50={d['p50']:+.4f}  "
                    f"p99={d['p99']:+.4f}  mean={d['mean']:+.4f}"]
    lines += ["", "## Redundancy (mean cross-sectional Spearman)"]
    for k, v in card["redundancy"].items():
        flag = ""
        if v is not None and abs(v) >= card["redundancy_strong_bar"]:
            flag = "  <-- STRONG (family review required)"
        lines.append(f"- vs `{k}`: "
                       f"{v if v is not None else 'n/a'}{flag}")
    if card["family_review_required"]:
        lines += ["",
                    f"**FAMILY REVIEW REQUIRED** — suggested family: "
                    f"`{card['suggested_family']}` (Bailey-LdP n_trials "
                    "must pool with the correlated incumbent unless a "
                    "written override reason is recorded)."]
    lines += ["", f"_generated {card['generated_ts']}_", ""]
    return "\n".join(lines)
