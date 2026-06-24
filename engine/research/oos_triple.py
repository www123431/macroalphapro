"""engine.research.oos_triple — bt-flex-1: auto in-paper / post-paper /
full-sample triple decay analysis.

Doctrine
========
Every dispatch whose FactorSpec carries `paper_original_window` (the
publication-era sample stated by the source paper) automatically
produces a three-window decay narrative:

  * in_paper      = intersection(our date_range, paper_original_window)
  * post_paper    = our date_range AFTER paper_original_window ends
  * full_sample   = the original strict-gate run

The triple is computed by SLICING the template's emitted PnL series
(`artifacts["pnl_series_df"]`) — NO template re-run, < 1s overhead
per dispatch.

Why
===
McLean-Pontiff 2016 documents post-publication Sharpe decay averaging
32-58% across 97 anomalies. The principal cannot assess
deploy-worthiness from a full-sample Sharpe alone; the in-paper /
post-paper split is the FIRST diagnostic question of any deployable
factor. Pre-bt-flex-1 the principal had to ask Claude to manually
re-slice and re-run; now every verdict carries this story
automatically.

Phase 1 (THIS COMMIT): slice + raw Sharpe / mean-t / NW-t on each
segment + decay severity classification + narrative string. Decay
narrative gets attached to template_result.metrics so verdict events
carry it forward.

Phase 2 (deferred): re-run lens stack on each segment for full lens
output triple (not just summary stats). Phase 2 doubles compute per
dispatch — only worth it once burndown is at scale and we want full
lens triple in /lab UI.

Severity thresholds (per McLean-Pontiff distribution):
  none          decay_pct ≥ -0.20   (Sharpe basically holds)
  mild          -0.40 ≤ decay_pct < -0.20
  severe        -0.70 ≤ decay_pct < -0.40   (within McLean-Pontiff band)
  broken        decay_pct < -0.70
  inconclusive  in_paper or post_paper has too few months

Minimum months for either segment to be evaluable: 24 (2 years —
below this the t-stat is too noisy to claim decay).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────


MIN_MONTHS_PER_SEGMENT = 24

# Severity bands — DOWNGRADE in decay direction (decay_pct is negative
# when post-paper Sharpe is lower than in-paper).
SEVERITY_BANDS: tuple[tuple[float, str], ...] = (
    (-0.20, "none"),       # decay_pct >= -0.20
    (-0.40, "mild"),       # -0.40 <= decay_pct < -0.20
    (-0.70, "severe"),     # -0.70 <= decay_pct < -0.40
    (float("-inf"), "broken"),
)


@_dc.dataclass(frozen=True)
class SegmentStats:
    """Raw stats on one date-window slice of the PnL series."""
    n_months:    int
    mean_bps:    float       # mean monthly return in bps
    std_bps:     float
    sharpe_ann:  float       # annualized
    t_stat:      float       # simple iid t-stat on mean
    nw_t_stat:   float        # Newey-West t (lag 6); falls back to t_stat if compute fails
    start_ym:    str         # "YYYY-MM"
    end_ym:      str

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


@_dc.dataclass(frozen=True)
class OOSTripleResult:
    """The three-window decay narrative produced for one dispatch."""
    paper_window:     str                # echo of spec.paper_original_window
    full_window:      str                # echo of spec.date_range
    in_paper:         Optional[SegmentStats]
    post_paper:       Optional[SegmentStats]
    full_sample:      SegmentStats       # always computed
    decay_pct:        Optional[float]    # (Sharpe_post - Sharpe_in) / |Sharpe_in|
    severity:         str                # see SEVERITY_BANDS or "inconclusive"
    narrative:        str
    pnl_column_used:  str                # which series we sliced

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_window":     self.paper_window,
            "full_window":      self.full_window,
            "in_paper":         self.in_paper.to_dict() if self.in_paper else None,
            "post_paper":       self.post_paper.to_dict() if self.post_paper else None,
            "full_sample":      self.full_sample.to_dict(),
            "decay_pct":        self.decay_pct,
            "severity":         self.severity,
            "narrative":        self.narrative,
            "pnl_column_used":  self.pnl_column_used,
        }


# ── Window parsing ──────────────────────────────────────────────────


def _parse_window(s: str) -> tuple[_dt.date, _dt.date]:
    """Parse 'YYYY-MM:YYYY-MM' OR 'YYYY-MM-DD:YYYY-MM-DD' inclusive into
    (start_date, end_date). Month-only strings interpreted as 1st / last day."""
    if ":" not in s:
        raise ValueError(f"window must contain ':': {s!r}")
    start_s, end_s = s.split(":", 1)
    start_s, end_s = start_s.strip(), end_s.strip()

    def _parse_one(t: str, end: bool) -> _dt.date:
        parts = t.split("-")
        if len(parts) == 3:
            return _dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            y, m = int(parts[0]), int(parts[1])
            if end:
                # Last day of month
                if m == 12:
                    return _dt.date(y, 12, 31)
                next_first = _dt.date(y, m + 1, 1)
                return next_first - _dt.timedelta(days=1)
            return _dt.date(y, m, 1)
        raise ValueError(f"unparseable window endpoint: {t!r}")

    return _parse_one(start_s, end=False), _parse_one(end_s, end=True)


def _intersect(
    a: tuple[_dt.date, _dt.date], b: tuple[_dt.date, _dt.date],
) -> Optional[tuple[_dt.date, _dt.date]]:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if s > e:
        return None
    return s, e


# ── Stats computation ───────────────────────────────────────────────


def _compute_segment_stats(pnl_segment) -> Optional[SegmentStats]:
    """Compute summary stats on a sliced PnL series (pandas Series indexed
    by month-end timestamps, values are monthly returns as decimals).

    Returns None if n < MIN_MONTHS_PER_SEGMENT or std is degenerate.
    """
    s = pnl_segment.dropna()
    n = len(s)
    if n < MIN_MONTHS_PER_SEGMENT:
        return None

    mean = float(s.mean())
    std = float(s.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return None

    mean_bps   = mean * 10000.0
    std_bps    = std * 10000.0
    sharpe_ann = (mean / std) * math.sqrt(12.0)
    t_stat     = (mean / (std / math.sqrt(n))) if std > 0 else 0.0

    # Newey-West HAC, lag=6 (12*0.5). If statsmodels unavailable or
    # computation fails, fall back to plain t.
    nw_t = t_stat
    try:
        import numpy as np
        import statsmodels.api as sm
        x = np.ones(n)
        ols = sm.OLS(s.values, x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        nw_t = float(ols.tvalues[0])
    except Exception as exc:
        logger.debug("oos_triple: Newey-West fell back to plain t: %s", exc)

    idx0 = s.index[0]
    idx1 = s.index[-1]
    return SegmentStats(
        n_months   = n,
        mean_bps   = mean_bps,
        std_bps    = std_bps,
        sharpe_ann = sharpe_ann,
        t_stat     = t_stat,
        nw_t_stat  = nw_t,
        start_ym   = idx0.strftime("%Y-%m") if hasattr(idx0, "strftime") else str(idx0)[:7],
        end_ym     = idx1.strftime("%Y-%m") if hasattr(idx1, "strftime") else str(idx1)[:7],
    )


def _classify_severity(decay_pct: float) -> str:
    """Bucket decay_pct into severity per SEVERITY_BANDS."""
    for floor, label in SEVERITY_BANDS:
        if decay_pct >= floor:
            return label
    return "broken"


# ── Public API ──────────────────────────────────────────────────────


def compute_oos_triple(
    pnl_series_df,
    *,
    full_window:      str,
    paper_window:     str,
    pnl_column:       str = "pnl_net_13bp",
) -> Optional[OOSTripleResult]:
    """Compute the in-paper / post-paper / full-sample triple from a
    PnL DataFrame.

    Returns None if no usable analysis can be produced (no overlap,
    DataFrame missing required column, etc.).
    """
    if pnl_series_df is None or pnl_column not in getattr(pnl_series_df, "columns", []):
        return None

    try:
        full_start, full_end = _parse_window(full_window)
        paper_start, paper_end = _parse_window(paper_window)
    except ValueError as exc:
        logger.warning("oos_triple: window parse failed: %s", exc)
        return None

    full_segment = pnl_series_df[pnl_column]
    full_stats = _compute_segment_stats(full_segment)
    if full_stats is None:
        return None

    # in_paper = intersection
    in_range = _intersect((full_start, full_end), (paper_start, paper_end))
    in_paper_stats: Optional[SegmentStats] = None
    if in_range is not None:
        in_seg = full_segment.loc[in_range[0]:in_range[1]]
        in_paper_stats = _compute_segment_stats(in_seg)

    # post_paper = full range AFTER paper_end
    post_paper_stats: Optional[SegmentStats] = None
    post_start = max(full_start, paper_end + _dt.timedelta(days=1))
    if post_start <= full_end:
        post_seg = full_segment.loc[post_start:full_end]
        post_paper_stats = _compute_segment_stats(post_seg)

    decay_pct: Optional[float] = None
    severity: str = "inconclusive"
    if in_paper_stats and post_paper_stats:
        in_sh = in_paper_stats.sharpe_ann
        post_sh = post_paper_stats.sharpe_ann
        if abs(in_sh) > 1e-6:
            decay_pct = (post_sh - in_sh) / abs(in_sh)
            severity = _classify_severity(decay_pct)
        elif post_sh > 0:
            decay_pct = float("inf")
            severity = "none"      # post Sharpe positive when in_paper was zero
        else:
            decay_pct = 0.0
            severity = "none"

    narrative = _build_narrative(
        in_paper_stats, post_paper_stats, full_stats,
        decay_pct, severity, paper_window, full_window,
    )

    return OOSTripleResult(
        paper_window     = paper_window,
        full_window      = full_window,
        in_paper         = in_paper_stats,
        post_paper       = post_paper_stats,
        full_sample      = full_stats,
        decay_pct        = decay_pct,
        severity         = severity,
        narrative        = narrative,
        pnl_column_used  = pnl_column,
    )


def _build_narrative(
    in_paper: Optional[SegmentStats],
    post_paper: Optional[SegmentStats],
    full_sample: SegmentStats,
    decay_pct: Optional[float],
    severity: str,
    paper_window: str,
    full_window: str,
) -> str:
    """1-3 sentence decay story per the McLean-Pontiff framing."""
    parts: list[str] = []
    if in_paper and post_paper:
        parts.append(
            f"In-paper ({in_paper.start_ym}~{in_paper.end_ym}, n={in_paper.n_months}mo): "
            f"Sharpe={in_paper.sharpe_ann:.2f} (NW-t={in_paper.nw_t_stat:.2f}); "
            f"Post-paper ({post_paper.start_ym}~{post_paper.end_ym}, "
            f"n={post_paper.n_months}mo): Sharpe={post_paper.sharpe_ann:.2f} "
            f"(NW-t={post_paper.nw_t_stat:.2f})."
        )
        if decay_pct is not None and math.isfinite(decay_pct):
            parts.append(
                f"Decay {decay_pct*100:+.0f}% ({severity}; McLean-Pontiff 2016 "
                f"avg post-pub Sharpe drop: 32-58% → 'severe' band)."
            )
        else:
            parts.append(f"Decay class: {severity}.")
    elif in_paper and not post_paper:
        parts.append(
            f"In-paper ({in_paper.start_ym}~{in_paper.end_ym}, n={in_paper.n_months}mo) "
            f"covered; post-paper window has <{MIN_MONTHS_PER_SEGMENT}mo so decay "
            f"cannot be tested. Add later data before deploy."
        )
    elif post_paper and not in_paper:
        parts.append(
            f"In-paper window {paper_window} has <{MIN_MONTHS_PER_SEGMENT}mo overlap "
            f"with our data ({full_window}); only post-paper "
            f"({post_paper.start_ym}~{post_paper.end_ym}, n={post_paper.n_months}mo) "
            f"is evaluable: Sharpe={post_paper.sharpe_ann:.2f} (NW-t="
            f"{post_paper.nw_t_stat:.2f}). Replication claim cannot be tested."
        )
    else:
        parts.append(
            f"Neither in-paper nor post-paper segment has >={MIN_MONTHS_PER_SEGMENT}mo; "
            f"decay analysis inconclusive."
        )
    return " ".join(parts)
