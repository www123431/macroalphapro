"""
engine/factor_lab/mining_runner.py — Tier 1 single-stock factor mining lab runner.

Status: NEW 2026-05-10 (F-LAB-E4).

Tier 1 mining lab parallel to existing Tier 2 ETF candidate gate
(`engine/factor_lab/runner.py`). Both share `engine/factor_lab/{types,power,
registry}` governance, but execution paths are separate:

  Tier 2 (existing, ETF):
    runner.py → b_plus_search.run_single_strategy_weekly → ETF universe_tier 1/2

  Tier 1 (this module, single-stock):
    mining_runner.py → own monthly walk-forward → vintage S&P 500 universe
                     → factor_library_singlename.FACTOR_REGISTRY_SINGLENAME

Boundary invariants (per spec_factor_lab.md §6 + memory
`feedback_factor_research_3_tier_framework`):
  - factor_library → factor_lab  (allowed)
  - factor_lab    →/ factor_library  (FORBIDDEN — Tier R rule enforces statically)
  - 0 LLM imports
  - Tier 1 mining specs are `factor_kind="infrastructure_spec"` → 0 trials
    contribution to EFFECTIVE_N_TRIALS

Tier 1 vs Tier 2 verdict semantics
----------------------------------
Tier 2 (BHY-corrected): PASS / MARGINAL / FAIL / FAIL_UNDERPOWERED (FactorState)
Tier 1 (raw, exploratory): one of:
  - "promotable_to_tier_2"    — strong directional (|NW t| ≥ 2.5 + sign match)
  - "directional_positive"    — moderate directional (|NW t| ≥ 1.65 + sign match)
  - "directional_against_prior" — reverse direction (sign flip from academic prior)
  - "noise"                   — |NW t| < 1.65 (no signal)

These thresholds are RAW (not BHY-corrected). Tier 1 → Tier 2 promotion is
NEVER automatic — even "promotable_to_tier_2" verdict requires manual review
+ re-registration as research_hypothesis spec + going through `runner.py`
BHY gate. This separation breaks HARKing R3.

Output artifacts (per spec_factor_lab capability_evidence convention)
--------------------------------------------------------------------
  data/factor_mining_lab/<factor_id>_<as_of>.json     — full session record
  docs/decisions/factor_mining_<factor_id>_<date>.md  — verdict markdown
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Locked Tier 1 verdict thresholds (raw, NOT BHY-corrected) ──────────────
T_PROMOTABLE_LOCKED:   float = 2.5    # |t| ≥ 2.5 (raw 1% two-sided) for promotion candidate
T_DIRECTIONAL_LOCKED:  float = 1.65   # |t| ≥ 1.65 (raw 10% two-sided) for directional positive
TC_BPS_LOCKED:         float = 12.0   # single-stock retail (matches Wave A)
GROSS_EXPOSURE_LOCKED: float = 1.0    # z-weighted long-short, gross = 1.0 (no leverage)

# Tier 1 verdict status string values (NOT FactorState — Tier 2 only)
VERDICT_PROMOTABLE              = "promotable_to_tier_2"
VERDICT_DIRECTIONAL_POSITIVE    = "directional_positive"
VERDICT_DIRECTIONAL_AGAINST     = "directional_against_prior"
VERDICT_NOISE                   = "noise"


# ── Output paths ────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR  = _REPO_ROOT / "data" / "factor_mining_lab"
_DECISIONS_DIR = _REPO_ROOT / "docs" / "decisions"


def _data_dir() -> Path:
    """Indirection so tests can monkeypatch."""
    return _DATA_DIR


def _decisions_dir() -> Path:
    return _DECISIONS_DIR


# ── Result dataclass ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class MiningResult:
    """Tier 1 mining session output (one factor × one universe × one window)."""
    factor_id:           str
    n_periods:            int
    monthly_returns_gross: pd.Series   # indexed by rebal_date
    monthly_returns_net:   pd.Series
    annualized_sharpe_net: float
    annualized_vol_net:    float
    cumulative_return_net: float
    nw_t_stat_net:         float       # raw NW t-stat (NOT BHY-corrected)
    sign_match:            bool        # mean_return aligned with expected_sign
    verdict:               str         # one of VERDICT_* constants above
    mean_n_active:         float       # mean # of active positions per period
    expected_sign:         int         # +1 or -1 (from FactorSpec)
    metadata:              dict        # window, universe size, caveats


# ── Public entry ────────────────────────────────────────────────────────────
def run_mining_session(
    factor_id:            str,
    universe_at_date_fn:  Callable[[datetime.date], list[str]],
    panel:                pd.DataFrame,
    start_date:           datetime.date,
    end_date:             datetime.date,
    *,
    persist_artifacts:    bool = True,
    rebalance_dates:      Optional[list[datetime.date]] = None,
) -> MiningResult:
    """Run Tier 1 mining walk-forward for one registered factor.

    Args:
        factor_id:           must be in FACTOR_REGISTRY_SINGLENAME
        universe_at_date_fn: callable returning universe at as_of (e.g. CRSP /
                             constituents_loader output)
        panel:               daily price panel (date index × ticker columns)
        start_date / end_date: walk-forward window (inclusive)
        persist_artifacts:   if True, writes JSON + verdict markdown to disk
        rebalance_dates:     optional explicit rebalance schedule; default = month-ends

    Returns:
        MiningResult with full session output + verdict.

    Raises:
        KeyError: factor_id not registered in FACTOR_REGISTRY_SINGLENAME
        ValueError: panel/universe issues
    """
    from engine.factor_library_singlename import get_factor

    spec = get_factor(factor_id)   # raises KeyError if not registered

    if not isinstance(start_date, datetime.date):
        raise TypeError(f"start_date must be datetime.date, got {type(start_date)}")
    if not isinstance(end_date, datetime.date):
        raise TypeError(f"end_date must be datetime.date, got {type(end_date)}")
    if start_date >= end_date:
        raise ValueError(f"start_date {start_date} must be < end_date {end_date}")

    if rebalance_dates is None:
        rebalance_dates = _generate_monthend_dates(start_date, end_date)
    if len(rebalance_dates) < 2:
        raise ValueError(
            f"need ≥ 2 rebalance dates for walk-forward; got {len(rebalance_dates)} "
            f"in window [{start_date}, {end_date}]. Widen window or check calendar."
        )

    if panel is None or panel.empty:
        raise ValueError("panel must be non-empty DataFrame")

    logger.info(
        "Tier 1 mining session: factor=%s, n_rebal=%d window=[%s, %s]",
        factor_id, len(rebalance_dates), start_date, end_date,
    )

    monthly_records: list[dict] = []
    n_active_per_period: list[int] = []

    for i, rebal_date in enumerate(rebalance_dates):
        # 1. Universe at as_of
        try:
            universe = universe_at_date_fn(rebal_date)
        except Exception as exc:
            logger.warning("universe lookup failed at %s: %s", rebal_date, exc)
            continue
        if not universe:
            continue

        # 2. Single-factor signal (cross-section z-score)
        try:
            z_signal = spec.signal_fn(
                as_of=rebal_date,
                universe=universe,
                panel=panel,
            )
        except Exception as exc:
            logger.warning("factor %s signal compute failed at %s: %s",
                           factor_id, rebal_date, exc)
            continue
        if z_signal is None or z_signal.empty:
            continue

        # 3. Build z-weighted long-short portfolio,
        #    apply expected_sign so mean_return > 0 == matches academic prior
        weights = _compute_portfolio_weights(z_signal, expected_sign=spec.expected_sign)
        if weights.empty:
            continue
        n_active_per_period.append(int((weights != 0).sum()))

        # 4. Realized return for next-period
        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            break
        try:
            gross_return = _compute_realized_return(
                weights=weights, panel=panel,
                period_start=rebal_date, period_end=next_rebal,
            )
        except Exception as exc:
            logger.warning("realized return failed @ %s: %s", rebal_date, exc)
            continue

        # 5. TC drag (turnover * bps; first period uses full position turnover)
        prev_weights = monthly_records[-1].get("weights") if monthly_records else None
        tc = _compute_tc_drag(weights, prev_weights, TC_BPS_LOCKED)

        monthly_records.append({
            "rebal_date":    rebal_date,
            "gross_return":  gross_return,
            "tc":            tc,
            "net_return":    gross_return - tc,
            "weights":       weights,
        })

    if not monthly_records:
        logger.error("Tier 1 mining: 0 successful periods for factor=%s", factor_id)
        return _empty_result(factor_id, spec.expected_sign,
                              start_date, end_date, len(rebalance_dates))

    # ── Aggregate ───────────────────────────────────────────────────────────
    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    monthly_gross = df["gross_return"].astype(float)
    monthly_net   = df["net_return"].astype(float)

    n_periods       = len(monthly_net)
    mean_net        = float(monthly_net.mean())
    std_net         = float(monthly_net.std(ddof=1))
    annualized_sharpe = (mean_net / std_net * np.sqrt(12.0)) if std_net > 1e-9 else 0.0
    annualized_vol  = std_net * np.sqrt(12.0)
    cum_ret         = float(np.prod(1.0 + monthly_net.values) - 1.0)

    nw_t = _compute_nw_t_stat(monthly_net.values)
    sign_match = (np.sign(mean_net) == 1)   # weights already incorporate expected_sign
                                             # → positive mean_net means matches prior

    verdict = _classify_verdict(
        nw_t_abs=abs(nw_t),
        sign_match=sign_match,
    )

    metadata = {
        "window_start":    start_date.isoformat(),
        "window_end":      end_date.isoformat(),
        "n_rebal_dates":   len(rebalance_dates),
        "n_successful":    n_periods,
        "factor_citation": spec.citation,
        "factor_formula":  spec.formula_summary,
        "tc_bps":          TC_BPS_LOCKED,
        "verdict_thresholds": {
            "promotable_t":    T_PROMOTABLE_LOCKED,
            "directional_t":   T_DIRECTIONAL_LOCKED,
        },
        "mean_n_active":   float(np.mean(n_active_per_period)) if n_active_per_period else 0.0,
        "tier_1_caveat":   (
            "Tier 1 mining verdict is RAW (NOT BHY-corrected). Promotion to Tier 2 "
            "requires manual review + re-registration as research_hypothesis spec + "
            "factor_lab.runner BHY gate. This is governance, not bug."
        ),
    }

    result = MiningResult(
        factor_id              = factor_id,
        n_periods              = n_periods,
        monthly_returns_gross  = monthly_gross,
        monthly_returns_net    = monthly_net,
        annualized_sharpe_net  = float(annualized_sharpe),
        annualized_vol_net     = float(annualized_vol),
        cumulative_return_net  = cum_ret,
        nw_t_stat_net          = float(nw_t),
        sign_match             = bool(sign_match),
        verdict                = verdict,
        mean_n_active          = float(metadata["mean_n_active"]),
        expected_sign          = int(spec.expected_sign),
        metadata               = metadata,
    )

    if persist_artifacts:
        try:
            _persist_artifacts(result, spec)
        except Exception as exc:
            logger.warning("Tier 1 mining: artifact persistence failed: %s", exc)

    return result


# ── Internals ───────────────────────────────────────────────────────────────
def _generate_monthend_dates(
    start: datetime.date, end: datetime.date,
) -> list[datetime.date]:
    """Month-end business dates between start and end (inclusive)."""
    dates: list[datetime.date] = []
    d = start
    while d <= end:
        next_month = d.replace(day=28) + datetime.timedelta(days=4)
        last_day = next_month - datetime.timedelta(days=next_month.day)
        if start <= last_day <= end:
            dates.append(last_day)
        d = last_day + datetime.timedelta(days=1)
    return sorted(set(dates))


def _compute_portfolio_weights(
    z_signal:      pd.Series,
    expected_sign: int,
) -> pd.Series:
    """Build z-weighted long-short portfolio, normalized to gross = 1.0.

    weights = expected_sign × z_signal_normalized
    → gross exposure = sum(|weights|) = 1.0
    → mean_return > 0 ⇔ matches academic prior (when expected_sign applied)

    NaN inputs are dropped; if all NaN or all zero, returns empty Series.
    """
    valid = z_signal.dropna()
    nonzero = valid[valid.abs() > 1e-12]
    if nonzero.empty:
        return pd.Series(dtype=float)
    gross = float(nonzero.abs().sum())
    if gross < 1e-12:
        return pd.Series(dtype=float)
    normalized = nonzero / gross
    return normalized * float(expected_sign)


def _compute_realized_return(
    weights:      pd.Series,
    panel:        pd.DataFrame,
    period_start: datetime.date,
    period_end:   datetime.date,
) -> float:
    """Realized return of weights over [period_start, period_end].

    For each ticker in weights, compute (price_end / price_start - 1),
    then weighted sum. NaN tickers contribute 0 (effectively skipped).
    """
    ret = 0.0
    valid_count = 0
    for ticker, w in weights.items():
        if ticker not in panel.columns:
            continue
        ts = panel[ticker].dropna()
        before_start = ts[ts.index <= pd.Timestamp(period_start)]
        before_end   = ts[ts.index <= pd.Timestamp(period_end)]
        if before_start.empty or before_end.empty:
            continue
        p_start = float(before_start.iloc[-1])
        p_end   = float(before_end.iloc[-1])
        if p_start <= 0 or p_end <= 0:
            continue
        r_i = (p_end / p_start) - 1.0
        if not np.isfinite(r_i):
            continue
        ret += float(w) * r_i
        valid_count += 1
    return ret if valid_count > 0 else 0.0


def _compute_tc_drag(
    weights_new:  pd.Series,
    weights_prev: Optional[pd.Series],
    bps_roundtrip: float,
) -> float:
    """TC drag = bps × turnover. Turnover = 0.5 × Σ|w_new - w_prev| (one-way)."""
    if weights_prev is None or weights_prev.empty:
        # First period: full turnover (positions established from cash)
        turnover = float(weights_new.abs().sum())
    else:
        diff = weights_new.subtract(weights_prev, fill_value=0.0)
        turnover = 0.5 * float(diff.abs().sum())
    return turnover * (bps_roundtrip / 10000.0)


def _compute_nw_t_stat(returns: np.ndarray) -> float:
    """NW-HAC t-stat with lag = floor(n^(1/3)) (Andrews 1991 rule of thumb).

    Standard NW formula:
        SE = √(γ_0 + 2 Σ_{j=1..L} (1 - j/(L+1)) γ_j) / √n
        γ_j = autocovariance at lag j

    Returns t = mean / SE; 0 if degenerate (n<2 or zero vol).
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return 0.0
    mean = float(arr.mean())
    if abs(mean) < 1e-15:
        return 0.0
    centered = arr - mean
    L = max(1, int(np.floor(n ** (1.0 / 3.0))))
    gamma_0 = float(np.dot(centered, centered) / n)
    nw_var = gamma_0
    for j in range(1, L + 1):
        gamma_j = float(np.dot(centered[j:], centered[:-j]) / n)
        weight = 1.0 - j / (L + 1.0)
        nw_var += 2.0 * weight * gamma_j
    if nw_var <= 1e-15:
        return 0.0
    se = np.sqrt(nw_var / n)
    return mean / se


def _classify_verdict(
    nw_t_abs:    float,
    sign_match:  bool,
) -> str:
    """Tier 1 verdict per locked thresholds (NOT BHY-corrected)."""
    if not sign_match:
        return VERDICT_DIRECTIONAL_AGAINST
    if nw_t_abs >= T_PROMOTABLE_LOCKED:
        return VERDICT_PROMOTABLE
    if nw_t_abs >= T_DIRECTIONAL_LOCKED:
        return VERDICT_DIRECTIONAL_POSITIVE
    return VERDICT_NOISE


def _empty_result(
    factor_id:     str,
    expected_sign: int,
    start_date:    datetime.date,
    end_date:      datetime.date,
    n_rebal:       int,
) -> MiningResult:
    """Empty-result placeholder for 0-successful-periods sessions."""
    empty = pd.Series(dtype=float)
    return MiningResult(
        factor_id              = factor_id,
        n_periods              = 0,
        monthly_returns_gross  = empty,
        monthly_returns_net    = empty,
        annualized_sharpe_net  = 0.0,
        annualized_vol_net     = 0.0,
        cumulative_return_net  = 0.0,
        nw_t_stat_net          = 0.0,
        sign_match             = False,
        verdict                = VERDICT_NOISE,
        mean_n_active          = 0.0,
        expected_sign          = int(expected_sign),
        metadata               = {
            "window_start":  start_date.isoformat(),
            "window_end":    end_date.isoformat(),
            "n_rebal_dates": n_rebal,
            "n_successful":  0,
            "error":         "0 successful periods — universe / panel coverage issue",
        },
    )


# ── Artifact persistence ────────────────────────────────────────────────────
def _persist_artifacts(result: MiningResult, spec) -> None:
    """Write JSON session record + verdict markdown to disk."""
    data_dir = _data_dir()
    decisions_dir = _decisions_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)

    today_iso = datetime.date.today().isoformat()
    json_path = data_dir / f"{result.factor_id}_{today_iso}.json"
    md_path   = decisions_dir / f"factor_mining_{result.factor_id}_{today_iso}.md"

    payload = {
        "factor_id":             result.factor_id,
        "verdict":               result.verdict,
        "n_periods":             result.n_periods,
        "annualized_sharpe_net": result.annualized_sharpe_net,
        "annualized_vol_net":    result.annualized_vol_net,
        "cumulative_return_net": result.cumulative_return_net,
        "nw_t_stat_net":         result.nw_t_stat_net,
        "sign_match":            result.sign_match,
        "mean_n_active":         result.mean_n_active,
        "expected_sign":         result.expected_sign,
        "monthly_returns_net":   {
            ts.isoformat() if hasattr(ts, "isoformat") else str(ts): float(v)
            for ts, v in result.monthly_returns_net.items()
        },
        "metadata":              result.metadata,
        "session_run_at":        datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = _render_verdict_markdown(result, spec)
    md_path.write_text(md, encoding="utf-8")

    logger.info("Tier 1 mining artifacts: %s + %s", json_path.name, md_path.name)


def _render_verdict_markdown(result: MiningResult, spec) -> str:
    """Generate Tier 1 mining verdict markdown.

    Style mirrors `docs/decisions/<spec>_VERDICT.md` Tier 2 format but with
    Tier 1 specific tags (raw thresholds, NOT_PRODUCTION_CLAIM, governance
    note about Tier 2 promotion gate).
    """
    today_iso = datetime.date.today().isoformat()
    return f"""# Tier 1 Mining Verdict — {result.factor_id} ({today_iso})

**Tier**: 1 (mining lab, P-LAB exempt, infrastructure_spec)
**Verdict**: `{result.verdict}`
**NOT a production claim** — Tier 2 promotion requires re-registration as
research_hypothesis spec + `factor_lab.runner` BHY gate. See
`feedback_factor_research_3_tier_framework.md`.

## Factor

- **Factor ID**: `{result.factor_id}`
- **Citation**: {spec.citation}
- **Formula**: {spec.formula_summary}
- **Expected sign** (academic prior): `{result.expected_sign:+d}` — {'high z → low future' if result.expected_sign == -1 else 'high z → high future'}

## Walk-forward result

- **Window**: {result.metadata.get("window_start")} → {result.metadata.get("window_end")}
- **Periods**: {result.n_periods} (of {result.metadata.get("n_rebal_dates")} rebalance dates)
- **Mean active positions**: {result.mean_n_active:.0f}
- **Annualized Sharpe (net)**: {result.annualized_sharpe_net:+.3f}
- **Annualized Vol (net)**: {result.annualized_vol_net:.3f}
- **Cumulative return (net)**: {result.cumulative_return_net:+.2%}
- **NW t-stat (net, raw)**: {result.nw_t_stat_net:+.3f}
- **Sign match (vs prior)**: {result.sign_match}

## Verdict thresholds (locked, raw)

| Verdict | Condition |
|---|---|
| `promotable_to_tier_2` | sign matches AND \\|t\\| ≥ {T_PROMOTABLE_LOCKED} |
| `directional_positive` | sign matches AND \\|t\\| ≥ {T_DIRECTIONAL_LOCKED} |
| `directional_against_prior` | sign opposite of expected_sign |
| `noise` | \\|t\\| < {T_DIRECTIONAL_LOCKED} |

NOT BHY-corrected. Tier 2 promotion gate applies BHY-FDR over EFFECTIVE_N_TRIALS.

## Caveats

{result.metadata.get("tier_1_caveat", "")}

## Reproducibility

- Run via: `scripts/run_factor_mining_lab.py --factor-id {result.factor_id}`
- JSON artifact: `data/factor_mining_lab/{result.factor_id}_{today_iso}.json`
- Module: `engine/factor_lab/mining_runner.py` (no LLM imports)
"""
