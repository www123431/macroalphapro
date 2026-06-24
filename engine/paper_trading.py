"""
engine/paper_trading.py — Forward-only three-arm paper trading runner (E-pivot).

Spec: docs/spec_paper_trading_three_arm_e.md (2026-05-03).

Three arms snapshot at every month-end:
  A baseline    : TSMOM + vol-targeting (no LLM debate)
  B production  : current sector_pipeline LLM debate (with confidence-scaled adjustment)
  C placebo     : random N(0, σ) placebo adjustment (same flip sectors as B)

Calling pattern (typically from orchestrator.run_monthly):

    from engine.paper_trading import snapshot_paper_trading_arms
    summary = snapshot_paper_trading_arms(
        t=datetime.date.today(),
        sig_df=current_signal_df,
        model=gemini_model,
        vix=current_vix,
        prev_signal_df=last_month_signal_df,
    )

Persistence: PaperTradingRun ORM (engine/memory.py).
Returns are backfilled at the next month-end via backfill_paper_trading_returns(t).

Per spec §1: forward-only design eliminates LLM lookahead bias from historical OOS.
Per spec §3.3: Arm C uses random N(0, σ) where σ defaults to DEFAULT_PLACEBO_SIGMA;
production should fit σ from observed sector_debate_output history when n ≥ 30.
"""
from __future__ import annotations

import datetime
import json
import logging

import numpy as np
import pandas as pd

from engine.memory import (
    PAPER_TRADING_ARMS,
    PaperTradingRun,
    SessionFactory,
)

logger = logging.getLogger(__name__)


# ── Defaults (spec §3, frozen) ────────────────────────────────────────────────

DEFAULT_PLACEBO_SIGMA = 0.02   # 2% per-sector adjustment std (placeholder; fit from history when n ≥ 30)
DEFAULT_PLACEBO_SEED  = 7
DEFAULT_TARGET_VOL    = 0.10
DEFAULT_MAX_WEIGHT    = 0.20   # clamp range for arm B/C adjustments


# ── Helper: build baseline weights ────────────────────────────────────────────

def _baseline_weights(
    sig_df:     pd.DataFrame,
    target_vol: float = DEFAULT_TARGET_VOL,
) -> pd.Series:
    """Baseline portfolio weights via construct_portfolio with no narrative context."""
    from engine.portfolio import construct_portfolio
    pw = construct_portfolio(
        sig_df, regime=None, target_vol=target_vol,
        narrative_context=None,
    )
    if pw.weights is None or pw.weights.empty:
        return pd.Series(dtype=float)
    return pw.weights.copy()


# ── Arm computation ───────────────────────────────────────────────────────────

def compute_arm_A_weights(sig_df: pd.DataFrame) -> pd.Series:
    """Arm A: baseline only, no LLM debate, no adjustment."""
    return _baseline_weights(sig_df)


def compute_arm_B_weights(
    sig_df:        pd.DataFrame,
    flip_sectors:  list[str],
    model,
    t:             datetime.date,
    vix:           float,
) -> tuple[pd.Series, dict]:
    """
    Arm B: baseline + sector_pipeline LLM debate adjustments for flip sectors.

    Returns (weights, debate_output_dict) where debate_output_dict captures
    LLM scaled_adj + confidence per sector for audit / persistence.
    """
    weights = _baseline_weights(sig_df)
    debate_output: dict = {}

    if not flip_sectors:
        return weights, debate_output
    if model is None:
        logger.warning("compute_arm_B_weights: model is None, skipping LLM debate (returning baseline)")
        return weights, debate_output

    from engine.sector_pipeline import run_sector_pipeline

    for sector in flip_sectors:
        try:
            result = run_sector_pipeline(
                model=model,
                sector_name=sector,
                t_day=t,
                vix=vix,
                decision_source="ai_drafted_paper_trading_B",
            )
        except Exception as exc:
            logger.warning("Arm B sector_pipeline failed for %s: %s", sector, exc)
            continue

        scaled_adj = float(result.get("scaled_adj", 0.0) or 0.0)
        ticker = (result.get("inputs") or {}).get("ticker_for_news")

        if ticker and ticker in weights.index and scaled_adj != 0.0:
            new_w = float(np.clip(
                weights[ticker] + scaled_adj,
                -DEFAULT_MAX_WEIGHT, DEFAULT_MAX_WEIGHT,
            ))
            weights[ticker] = new_w

        final_xai = (result.get("debate") or {}).get("final_xai") or {}
        debate_output[sector] = {
            "ticker": ticker,
            "scaled_adj": scaled_adj,
            "overall_confidence": final_xai.get("overall_confidence"),
            "qc_flags": result.get("qc_flags", []),
            "pending_approval_id": result.get("pending_approval_id"),
        }

    return weights, debate_output


def compute_arm_C_weights(
    sig_df:        pd.DataFrame,
    flip_sectors:  list[str],
    t:             datetime.date,
    seed:          int = DEFAULT_PLACEBO_SEED,
    sigma:         float = DEFAULT_PLACEBO_SIGMA,
) -> tuple[pd.Series, dict]:
    """
    Arm C: baseline + random N(0, σ) placebo adjustments for SAME flip sectors as B.

    Returns (weights, placebo_adj_dict) — Path 1 redesign (2026-05-03) adds per-sector
    placebo persistence so cluster-by-month T2 test is computable directly without RNG
    replay (RNG/seed state across numpy versions and flip-sector ordering is fragile).
    """
    weights = _baseline_weights(sig_df)
    placebo_adj_dict: dict = {}
    if not flip_sectors:
        return weights, placebo_adj_dict

    rng_seed = hash((seed, str(t))) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    for sector in flip_sectors:
        # Resolve ticker — same convention as Arm B
        if "ticker" in sig_df.columns and sector in sig_df.index:
            ticker = sig_df.loc[sector, "ticker"]
        else:
            ticker = sector  # fallback

        placebo_adj = float(rng.normal(0.0, sigma))
        if ticker in weights.index:
            new_w = float(np.clip(
                weights[ticker] + placebo_adj,
                -DEFAULT_MAX_WEIGHT, DEFAULT_MAX_WEIGHT,
            ))
            weights[ticker] = new_w
        placebo_adj_dict[sector] = {
            "ticker":     str(ticker),
            "placebo_adj": placebo_adj,
        }

    return weights, placebo_adj_dict


# ── Persistence ───────────────────────────────────────────────────────────────

def save_paper_trading_arm(
    arm:                  str,
    as_of:                datetime.date,
    weights:              pd.Series,
    sector_debate_output: dict | None = None,
    placebo_seed:         int | None = None,
    placebo_adjustments:  dict | None = None,
    notes:                str = "",
) -> int | None:
    """Insert one PaperTradingRun row (idempotent on (as_of, arm)). Returns row id."""
    if arm not in PAPER_TRADING_ARMS:
        raise ValueError(f"arm must be in {PAPER_TRADING_ARMS}, got {arm!r}")

    weights_dict = {str(k): float(v) for k, v in weights.items() if pd.notna(v)}

    with SessionFactory() as session:
        existing = session.query(PaperTradingRun).filter_by(
            as_of_date=as_of, arm=arm,
        ).first()
        if existing is not None:
            logger.info("save_paper_trading_arm: row exists for %s arm=%s, skip", as_of, arm)
            return existing.id

        # B-PLUS-PROD migration 2026-05-05: tag baseline signal so verdict
        # computation can filter pre/post migration runs separately.
        try:
            from engine.portfolio import PRODUCTION_SIGNAL as _prod_signal
        except Exception:
            _prod_signal = "ql01_bab"

        row = PaperTradingRun(
            as_of_date=as_of,
            arm=arm,
            weights_json=json.dumps(weights_dict, ensure_ascii=False),
            sector_debate_output=(
                json.dumps(sector_debate_output, ensure_ascii=False, default=str)
                if sector_debate_output else None
            ),
            placebo_seed=placebo_seed,
            placebo_adjustments=(
                json.dumps(placebo_adjustments, ensure_ascii=False, default=str)
                if placebo_adjustments else None
            ),
            notes=notes or None,
            signal_baseline=_prod_signal,
        )
        session.add(row)
        session.commit()
        logger.info("save_paper_trading_arm: persisted arm=%s as_of=%s id=%d",
                    arm, as_of, row.id)
        return row.id


def backfill_paper_trading_returns(t: datetime.date) -> int:
    """
    Compute next_month_return + cum_nav for any PaperTradingRun rows with
    as_of_date < t and next_month_return IS NULL.

    Returns count of rows updated.
    """
    from engine.backtest import _fetch_monthly_returns
    from engine.history import SECTOR_ETF

    with SessionFactory() as session:
        pending = session.query(PaperTradingRun).filter(
            PaperTradingRun.next_month_return.is_(None),
            PaperTradingRun.as_of_date < t,
        ).order_by(PaperTradingRun.as_of_date.asc()).all()

        if not pending:
            logger.info("backfill_paper_trading_returns: 0 pending rows")
            return 0

        min_date = min(r.as_of_date for r in pending)
        try:
            tickers = list(set(SECTOR_ETF.values()))
            returns = _fetch_monthly_returns(
                tickers,
                min_date - datetime.timedelta(days=10),
                t + datetime.timedelta(days=10),
            )
        except Exception as exc:
            logger.warning("backfill: monthly returns fetch failed: %s", exc)
            return 0

        if returns is None or returns.empty:
            return 0
        returns.index = pd.to_datetime(returns.index).normalize()

        # Initialize cum_nav per arm from latest existing
        latest_nav: dict[str, float] = {}
        for arm in PAPER_TRADING_ARMS:
            latest = session.query(PaperTradingRun).filter(
                PaperTradingRun.arm == arm,
                PaperTradingRun.cum_nav.isnot(None),
            ).order_by(PaperTradingRun.as_of_date.desc()).first()
            latest_nav[arm] = latest.cum_nav if latest else 1.0

        n_updated = 0
        for row in pending:
            row_ts = pd.Timestamp(row.as_of_date)
            future_returns = returns[returns.index > row_ts]
            if future_returns.empty:
                continue   # no t+1 data yet

            next_month_row = future_returns.iloc[0]
            try:
                weights = json.loads(row.weights_json or "{}")
            except Exception:
                weights = {}
            if not weights:
                continue

            port_ret = 0.0
            for tkr, w in weights.items():
                if tkr in next_month_row.index:
                    r_val = next_month_row[tkr]
                    if pd.notna(r_val):
                        port_ret += float(w) * float(r_val)

            row.next_month_return = port_ret
            latest_nav[row.arm] = latest_nav[row.arm] * (1.0 + port_ret)
            row.cum_nav = latest_nav[row.arm]
            n_updated += 1

        session.commit()
        logger.info("backfill_paper_trading_returns: updated %d rows", n_updated)

    # ── S2 Reflection backfill (spec §6) ─────────────────────────────────────
    # Co-trigger: when paper-trading returns arrive, sector_pipeline DecisionLog
    # rows for the same period typically have active_return filled by
    # verify_pending_decisions. Generate reflections for any that are still
    # pending. Failure here MUST NOT block paper-trading backfill.
    try:
        from engine.agents.reflection import generate_reflections_for_pending
        ref_summary = generate_reflections_for_pending(as_of=t)
        logger.info(
            "backfill_paper_trading_returns: reflection backfill summary=%s",
            ref_summary,
        )
    except Exception as exc:
        logger.warning(
            "backfill_paper_trading_returns: reflection backfill failed (%s); "
            "paper trading data unaffected", exc,
        )

    return n_updated


# ── Top-level snapshot ────────────────────────────────────────────────────────

def snapshot_paper_trading_arms(
    t:               datetime.date,
    sig_df:          pd.DataFrame,
    model,
    vix:             float = 20.0,
    prev_signal_df:  pd.DataFrame | None = None,
    placebo_seed:    int = DEFAULT_PLACEBO_SEED,
    placebo_sigma:   float = DEFAULT_PLACEBO_SIGMA,
) -> dict:
    """
    Snapshot all 3 arms at month-end t + backfill prev-month returns.

    Designed to be called from `engine.orchestrator.TradingCycleOrchestrator.run_monthly`.

    Returns summary dict with arms_persisted, flip_sectors, returns_backfilled, errors.
    """
    flip_sectors: list[str] = []
    if prev_signal_df is not None and not prev_signal_df.empty:
        for sector in sig_df.index:
            if sector in prev_signal_df.index:
                cur  = int(sig_df.loc[sector, "tsmom"])
                prev = int(prev_signal_df.loc[sector, "tsmom"])
                if cur != prev:
                    flip_sectors.append(sector)

    summary: dict = {
        "as_of_date":          str(t),
        "flip_sectors":        flip_sectors,
        "n_flip":              len(flip_sectors),
        "arms_persisted":      0,
        "returns_backfilled":  0,
        "errors":              [],
    }

    # Arm A
    try:
        weights_A = compute_arm_A_weights(sig_df)
        if not weights_A.empty:
            save_paper_trading_arm("A", t, weights_A, notes="baseline TSMOM + vol-target")
            summary["arms_persisted"] += 1
        else:
            summary["errors"].append("arm_A_empty_weights")
    except Exception as exc:
        logger.error("Arm A snapshot failed: %s", exc, exc_info=True)
        summary["errors"].append(f"arm_A: {exc}")

    # Arm B (production sector_pipeline LLM debate)
    try:
        weights_B, debate_output = compute_arm_B_weights(
            sig_df, flip_sectors, model, t, vix,
        )
        if not weights_B.empty:
            save_paper_trading_arm(
                "B", t, weights_B,
                sector_debate_output=debate_output,
                notes=f"production LLM debate, n_flip={len(flip_sectors)}",
            )
            summary["arms_persisted"] += 1
        else:
            summary["errors"].append("arm_B_empty_weights")
    except Exception as exc:
        logger.error("Arm B snapshot failed: %s", exc, exc_info=True)
        summary["errors"].append(f"arm_B: {exc}")

    # Arm C (placebo)
    try:
        weights_C, placebo_adj_dict = compute_arm_C_weights(
            sig_df, flip_sectors, t,
            seed=placebo_seed, sigma=placebo_sigma,
        )
        if not weights_C.empty:
            save_paper_trading_arm(
                "C", t, weights_C,
                placebo_seed=placebo_seed,
                placebo_adjustments=placebo_adj_dict,
                notes=f"random N(0,{placebo_sigma}) placebo, n_flip={len(flip_sectors)}",
            )
            summary["arms_persisted"] += 1
        else:
            summary["errors"].append("arm_C_empty_weights")
    except Exception as exc:
        logger.error("Arm C snapshot failed: %s", exc, exc_info=True)
        summary["errors"].append(f"arm_C: {exc}")

    # Backfill returns from previous month-ends
    try:
        summary["returns_backfilled"] = backfill_paper_trading_returns(t)
    except Exception as exc:
        logger.error("backfill returns failed: %s", exc, exc_info=True)
        summary["errors"].append(f"backfill: {exc}")

    # Paper Trading E v0.2 §11 mid-period checkpoints (n=12 / n=24).
    # Failure must NOT block the snapshot main flow.
    try:
        cp = check_paper_trading_e_checkpoints(t)
        summary["checkpoints"] = cp
    except Exception as exc:
        logger.warning("paper trading E checkpoint trigger failed: %s", exc)
        summary["checkpoints"] = {"error": str(exc)}

    return summary


# ── Paper Trading E v0.2 §11 mid-period checkpoints ──────────────────────────

CHECKPOINT_CONFIG_KEY = {12: "paper_trading_e_checkpoint_h12_fired_at",
                         24: "paper_trading_e_checkpoint_h24_fired_at"}


def check_paper_trading_e_checkpoints(
    as_of: datetime.date,
    *,
    force: bool = False,
) -> dict:
    """
    Per spec v0.2 §11, fire interim verdicts at h=12 (record only, no action)
    and h=24 (Lever 1 activation decision based on conviction-stratified hit
    rate analysis). Idempotent: a checkpoint fires at most once via
    SystemConfig stamp.

    Returns dict:
      {n_decisions, h12_fired, h24_fired, h12_just_fired, h24_just_fired,
       lever1_recommendation, notes}

    Wired into snapshot_paper_trading_arms; surfaced on pages/paper_trading.py.
    """
    from engine.memory import get_system_config, set_system_config

    out = {
        "as_of":               str(as_of),
        "n_decisions":         0,
        "h12_fired":           False,
        "h24_fired":           False,
        "h12_just_fired":      False,
        "h24_just_fired":      False,
        "lever1_recommendation": "wait",
        "notes":               "",
    }

    # Count Arm B decisions with realized returns (the population that drives
    # conviction analysis).
    with SessionFactory() as session:
        n_decisions = (
            session.query(PaperTradingRun)
            .filter_by(arm="B")
            .filter(PaperTradingRun.next_month_return.isnot(None))
            .count()
        )
    out["n_decisions"] = int(n_decisions)

    h12_stamp = get_system_config(CHECKPOINT_CONFIG_KEY[12], "")
    h24_stamp = get_system_config(CHECKPOINT_CONFIG_KEY[24], "")
    out["h12_fired"] = bool(h12_stamp)
    out["h24_fired"] = bool(h24_stamp)

    # h=12 — record-only checkpoint
    if (not out["h12_fired"] or force) and n_decisions >= 12:
        set_system_config(CHECKPOINT_CONFIG_KEY[12], str(as_of))
        out["h12_fired"] = True
        out["h12_just_fired"] = True
        out["notes"] += (
            f"h=12 checkpoint fired at as_of={as_of} (n_decisions={n_decisions}); "
            "record-only per spec §11. "
        )
        logger.info(
            "paper_trading_e checkpoint h=12 fired: n_decisions=%d, as_of=%s",
            n_decisions, as_of,
        )

    # h=24 — Lever 1 activation decision
    if (not out["h24_fired"] or force) and n_decisions >= 24:
        set_system_config(CHECKPOINT_CONFIG_KEY[24], str(as_of))
        out["h24_fired"] = True
        out["h24_just_fired"] = True
        try:
            conv = mid_checkpoint_conviction(horizon_target=24)
            out["lever1_recommendation"] = conv.get(
                "lever1_trigger_recommendation", "wait",
            )
            out["notes"] += (
                f"h=24 checkpoint fired at as_of={as_of}; "
                f"lever1={out['lever1_recommendation']}. "
            )
        except Exception as exc:
            logger.warning("mid_checkpoint_conviction failed: %s", exc)
            out["lever1_recommendation"] = "error"
            out["notes"] += f"h=24 fired but conviction analysis errored: {exc}. "
        logger.info(
            "paper_trading_e checkpoint h=24 fired: n_decisions=%d, lever1=%s",
            n_decisions, out["lever1_recommendation"],
        )

    return out


# ── Helpers for analysis ──────────────────────────────────────────────────────

def get_arm_history(arm: str) -> pd.DataFrame:
    """Return DataFrame with one row per month-end for arm."""
    if arm not in PAPER_TRADING_ARMS:
        raise ValueError(f"arm must be in {PAPER_TRADING_ARMS}")
    with SessionFactory() as session:
        rows = (
            session.query(PaperTradingRun)
            .filter_by(arm=arm)
            .order_by(PaperTradingRun.as_of_date.asc())
            .all()
        )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "as_of_date":          r.as_of_date,
            "next_month_return":   r.next_month_return,
            "cum_nav":             r.cum_nav,
            "weights_json":        r.weights_json,
            "n_holdings":          len(json.loads(r.weights_json or "{}")),
        } for r in rows])


def get_arm_returns_series(arm: str) -> pd.Series:
    """Return monthly returns time series for arm (NaN-dropped, indexed by date)."""
    df = get_arm_history(arm)
    if df.empty or "next_month_return" not in df.columns:
        return pd.Series(dtype=float)
    s = pd.Series(
        df["next_month_return"].values,
        index=pd.to_datetime(df["as_of_date"]),
    )
    return s.dropna()


# ── Path 1 redesign 2026-05-03: honest test statistics ────────────────────────
#
# Replaces the prior NW HAC Sharpe t-stat (type-I error 7-9% at small n) with:
#   - Primary stat: plain one-sided t-test on monthly (B - C) diff series
#   - Economic gates: ann_diff(B-C) >= 1.5%  AND  Sharpe(B) - Sharpe(C) >= 0.15
#   - Robustness:   stationary block bootstrap CI on Sharpe diff
#   - Verdict zones: insufficient_n / reject / accept / inconclusive
#   - Decision rule: h<36 advisory; h=36 first verdict; h=48 force-resolve
#
# Power analysis: docs/spec_paper_trading_three_arm_e.md §5 + memory
# project_paper_trading_e_power.md (2026-05-03).

# Constants — frozen once spec v0.2 ships, mutation breaks pre-registration.
STAT_T_THRESHOLD          = 1.645   # one-sided 5% alpha
ECON_ANN_DIFF_THRESHOLD   = 0.015   # 1.5% — Novy-Marx & Velikov (2016) TC-aware floor
ECON_SHARPE_DIFF_THRESHOLD = 0.15   # half López de Prado institutional standard
HORIZON_FIRST_VERDICT_MO  = 36
HORIZON_FORCE_RESOLVE_MO  = 48
N_BOOTSTRAP               = 2000


def compute_test_statistics(
    arm_b_returns: pd.Series | None = None,
    arm_c_returns: pd.Series | None = None,
) -> dict:
    """
    Honest verdict-zone test for paper trading E (Path 1 redesign 2026-05-03).

    Returns dict:
      n_obs              : int    months with paired B and C returns
      ann_diff           : float  annualised mean (B-C)
      t_stat             : float  one-sided t-stat on monthly (B-C) diff
      p_value            : float  one-sided p-value
      sharpe_b           : float  annualised Sharpe of arm B
      sharpe_c           : float  annualised Sharpe of arm C
      sharpe_diff        : float  sharpe_b - sharpe_c
      sharpe_diff_ci     : (lo, hi) 95% bootstrap CI of Sharpe diff
      stat_gate_passed   : bool   t_stat > 1.645
      econ_gate_passed   : bool   ann_diff >= 1.5%  AND  sharpe_diff >= 0.15
      verdict_zone       : str    {insufficient_n, reject, accept, inconclusive}
      horizon_action     : str    advisory / first_verdict / force_resolve

    Decision rule (pre-registered):
      n < 36 months                                         → insufficient_n  (advisory)
      t_stat < 0  OR  ann_diff < 0                          → reject
      stat_gate AND econ_gate                               → accept
      else (mixed/weak)                                     → inconclusive
      n >= 48 AND inconclusive                              → reject (force-resolve, default to skeptic)
    """
    import scipy.stats as _stats

    if arm_b_returns is None:
        arm_b_returns = get_arm_returns_series("B")
    if arm_c_returns is None:
        arm_c_returns = get_arm_returns_series("C")

    common = arm_b_returns.index.intersection(arm_c_returns.index)
    diff = (arm_b_returns.reindex(common) - arm_c_returns.reindex(common)).dropna()
    n = int(len(diff))

    out: dict = {
        "n_obs":              n,
        "ann_diff":           float("nan"),
        "t_stat":             float("nan"),
        "p_value":            float("nan"),
        "sharpe_b":           float("nan"),
        "sharpe_c":           float("nan"),
        "sharpe_diff":        float("nan"),
        "sharpe_diff_ci":     (float("nan"), float("nan")),
        "stat_gate_passed":   False,
        "econ_gate_passed":   False,
        "verdict_zone":       "insufficient_n",
        "horizon_action":     "advisory",
        "thresholds": {
            "stat_t":         STAT_T_THRESHOLD,
            "econ_ann_diff":  ECON_ANN_DIFF_THRESHOLD,
            "econ_sharpe":    ECON_SHARPE_DIFF_THRESHOLD,
            "h_first_verdict": HORIZON_FIRST_VERDICT_MO,
            "h_force_resolve": HORIZON_FORCE_RESOLVE_MO,
        },
    }
    if n < 12:
        return out

    mu = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd < 1e-12:
        return out
    t = mu / (sd / np.sqrt(n))
    p = float(1.0 - _stats.t.cdf(t, df=n - 1))

    out["ann_diff"] = float(mu * 12.0)
    out["t_stat"]   = float(t)
    out["p_value"]  = p

    def _ann_sharpe(r: pd.Series) -> float:
        r = r.dropna()
        if len(r) < 12:
            return float("nan")
        s = float(r.std(ddof=1))
        if s < 1e-12:
            return 0.0
        return float((r.mean() / s) * np.sqrt(12))

    sb = _ann_sharpe(arm_b_returns.reindex(common))
    sc = _ann_sharpe(arm_c_returns.reindex(common))
    out["sharpe_b"] = sb
    out["sharpe_c"] = sc
    out["sharpe_diff"] = (sb - sc) if not (np.isnan(sb) or np.isnan(sc)) else float("nan")

    out["sharpe_diff_ci"] = bootstrap_sharpe_diff_ci(
        arm_b_returns.reindex(common), arm_c_returns.reindex(common), n_boot=N_BOOTSTRAP,
    )

    stat_pass = (not np.isnan(t)) and (t > STAT_T_THRESHOLD)
    econ_pass = (
        (not np.isnan(out["ann_diff"])) and (out["ann_diff"] >= ECON_ANN_DIFF_THRESHOLD)
        and (not np.isnan(out["sharpe_diff"])) and (out["sharpe_diff"] >= ECON_SHARPE_DIFF_THRESHOLD)
    )
    out["stat_gate_passed"] = bool(stat_pass)
    out["econ_gate_passed"] = bool(econ_pass)

    if n < HORIZON_FIRST_VERDICT_MO:
        out["verdict_zone"]   = "insufficient_n"
        out["horizon_action"] = "advisory"
    else:
        if (t < 0) or (out["ann_diff"] < 0):
            zone = "reject"
        elif stat_pass and econ_pass:
            zone = "accept"
        else:
            zone = "inconclusive"
        if n >= HORIZON_FORCE_RESOLVE_MO and zone == "inconclusive":
            zone = "reject"  # force-resolve: default to skeptic per first-principle
            out["horizon_action"] = "force_resolve"
        else:
            out["horizon_action"] = (
                "force_resolve" if n >= HORIZON_FORCE_RESOLVE_MO else "first_verdict"
            )
        out["verdict_zone"] = zone

    return out


def bootstrap_sharpe_diff_ci(
    arm_b_returns: pd.Series,
    arm_c_returns: pd.Series,
    n_boot:        int = N_BOOTSTRAP,
    block_len:     int = 3,
    alpha:         float = 0.05,
    seed:          int = 11,
) -> tuple[float, float]:
    """
    Stationary block bootstrap (Politis-Romano 1994) of Sharpe(B)-Sharpe(C) diff.
    Block length 3 = quarterly persistence; standard for monthly returns.
    Returns 95% CI; nan,nan if insufficient data.
    """
    common = arm_b_returns.index.intersection(arm_c_returns.index)
    rb = arm_b_returns.reindex(common).dropna().values
    rc = arm_c_returns.reindex(common).dropna().values
    n = min(len(rb), len(rc))
    if n < 12:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    diffs_boot = np.empty(n_boot)
    for b in range(n_boot):
        # geometric block lengths around block_len
        idx = []
        i0 = int(rng.integers(0, n))
        while len(idx) < n:
            L = int(rng.geometric(1.0 / block_len))
            for j in range(L):
                idx.append((i0 + j) % n)
                if len(idx) >= n:
                    break
            i0 = int(rng.integers(0, n))
        idx = np.array(idx[:n])
        b_samp = rb[idx]
        c_samp = rc[idx]
        sb_samp = (b_samp.mean() / b_samp.std(ddof=1)) * np.sqrt(12) if b_samp.std(ddof=1) > 1e-12 else 0.0
        sc_samp = (c_samp.mean() / c_samp.std(ddof=1)) * np.sqrt(12) if c_samp.std(ddof=1) > 1e-12 else 0.0
        diffs_boot[b] = sb_samp - sc_samp

    lo = float(np.quantile(diffs_boot, alpha / 2))
    hi = float(np.quantile(diffs_boot, 1 - alpha / 2))
    return (lo, hi)


def mid_checkpoint_conviction(
    horizon_target: int = 12,
) -> dict:
    """
    Conviction-stratified hit-rate analysis for h=12 / h=24 mid-checkpoints.

    Pre-registered triggers (per spec v0.2 §11):
      h=12  : record only, no action
      h=24  : if hit_rate(high_conv) > 0.55 AND hit_rate(non_flip) ≈ hit_rate(flip),
              start Lever 1 (top-15 event expansion) for h=36 final.
              Otherwise, forfeit Lever 1 — keep flip-only design.

    Returns dict:
      n_decisions              : total Arm B sector-debate decisions observed
      hit_rate_overall         : sign(adj) == sign(realised next-month return)
      hit_rate_high_conv       : same, restricted to overall_confidence ≥ 70
      hit_rate_low_conv        : same, restricted to overall_confidence < 40
      hit_rate_by_conv_tier    : dict {"high": .., "mid": .., "low": ..}
      lever1_trigger_recommendation : "activate" | "forfeit" | "wait"
    """
    out = {
        "horizon_target":      horizon_target,
        "n_decisions":         0,
        "hit_rate_overall":    float("nan"),
        "hit_rate_high_conv":  float("nan"),
        "hit_rate_low_conv":   float("nan"),
        "hit_rate_by_conv_tier": {},
        "lever1_trigger_recommendation": "wait",
    }

    with SessionFactory() as session:
        rows = (
            session.query(PaperTradingRun)
            .filter_by(arm="B")
            .filter(PaperTradingRun.next_month_return.isnot(None))
            .order_by(PaperTradingRun.as_of_date.asc())
            .all()
        )
    if not rows:
        return out

    # Build per-decision panel: (sector, scaled_adj, confidence, realised_ret)
    # Realised next-month sector return is needed; we approximate it from arm-A
    # next_month_return at portfolio level is NOT per-sector — would require
    # joining backtest sector returns. For now, persist what we have and
    # leave per-sector return join as a dependency on backfill_paper_trading_returns
    # being upgraded (deferred — this function activates at h=24).
    panel: list[dict] = []
    for r in rows:
        if not r.sector_debate_output:
            continue
        try:
            debate = json.loads(r.sector_debate_output)
        except Exception:
            continue
        for sector, dat in debate.items():
            adj = dat.get("scaled_adj")
            conf = dat.get("overall_confidence")
            if adj is None:
                continue
            panel.append({
                "as_of":      r.as_of_date,
                "sector":     sector,
                "scaled_adj": float(adj),
                "confidence": float(conf) if conf is not None else float("nan"),
            })

    out["n_decisions"] = len(panel)
    if not panel:
        return out

    # NOTE: per-sector realised return join is not yet wired (requires per-sector
    # forward returns alongside portfolio next_month_return). Skeleton returns
    # decision count + confidence distribution; full hit-rate requires h≥12 data
    # AND a per-sector return join in backfill — deferred to first time this
    # checkpoint actually fires (n_decisions ≥ 30, projected month 12+).
    #
    # When that join lands:
    #   for d in panel:
    #       sec_ret_next_mo = sector_returns_lookup(d["as_of"], d["sector"])
    #       d["hit"] = float(np.sign(d["scaled_adj"]) == np.sign(sec_ret_next_mo))
    #   hit_rates by tier follow.
    out["lever1_trigger_recommendation"] = "wait"  # until h=24 + return join wired
    return out


# ── CLI: standalone monthly snapshot trigger ──────────────────────────────────

def _cli_snapshot(as_of_str: str | None = None, with_llm: bool = False) -> None:
    """
    CLI entry: snapshot one month-end.

    Usage:
        python -m engine.paper_trading                         # snapshot today's month-end, no LLM
        python -m engine.paper_trading --as-of 2026-05-31      # specific date
        python -m engine.paper_trading --with-llm              # use Gemini for Arm B (costs LLM tokens)

    Without --with-llm, Arm B falls back to baseline weights (graceful no-op).
    Use --with-llm only when you want Arm B to consume current production
    sector_pipeline LLM debate output.
    """
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if as_of_str:
        as_of = datetime.date.fromisoformat(as_of_str)
    else:
        as_of = datetime.date.today()

    print(f"=== Paper Trading Snapshot — as_of={as_of} ===")

    from engine.signal import get_signal_dataframe
    sig_df = get_signal_dataframe(as_of, 12, 1)
    prev_month_end = (datetime.date(as_of.year, as_of.month, 1) - datetime.timedelta(days=1))
    prev_sig_df = get_signal_dataframe(prev_month_end, 12, 1)

    if sig_df is None or sig_df.empty:
        print("[ERROR] signal_df empty for as_of=%s — abort" % as_of)
        return

    model = None
    if with_llm:
        try:
            from engine.key_pool import get_pool
            model = get_pool().get_model()
            print(f"[INFO] LLM model loaded: {type(model).__name__}")
        except Exception as exc:
            print(f"[WARN] failed to load LLM model: {exc} — Arm B will fallback to baseline")
            model = None
    else:
        print("[INFO] no --with-llm flag, Arm B will fallback to baseline (no LLM cost)")

    summary = snapshot_paper_trading_arms(
        t=as_of, sig_df=sig_df, model=model, vix=20.0,
        prev_signal_df=prev_sig_df,
    )
    print(f"\nSummary: {json.dumps(summary, indent=2, default=str, ensure_ascii=False)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Paper trading 3-arm forward snapshot.")
    p.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD month-end (default: today)")
    p.add_argument("--with-llm", action="store_true",
                   help="Run Arm B with real sector_pipeline LLM debate (costs Gemini tokens)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    _cli_snapshot(as_of_str=args.as_of, with_llm=args.with_llm)
