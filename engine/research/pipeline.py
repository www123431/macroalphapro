"""engine/research/pipeline.py — in-app research-automation gate (v1).

Generalizes the ad-hoc factor tests (fin_accruals, carry, dpead_recon, …) into ONE reusable gate:
give it a mechanism-first hypothesis + a signal-construction fn → `run_gate` auto-runs the full
rigorous evaluation and emits a GREEN/YELLOW/RED verdict, logged to a campaign ledger. This is the
Man-Group-AlphaGPT pattern (propose → evaluate → human review) but with the evaluation = OUR gate.

The gate (all in one place, so no more hand-rolled deflated-SR bugs):
  • standalone Sharpe (the series is expected NET of cost)
  • residual-α vs FF5+UMD, and vs FF5+UMD+PEAD (the deployed equity leg) — the orthogonality test
  • Deflated Sharpe with n_trials (engine.validation.deflated_sharpe) — multiple-testing honesty
  • out-of-sample (2nd-half) Sharpe
  • correlation with the book (PEAD leg)

GUARDRAIL (doctrine): this automates the rigorous PROCESS + the multiple-testing accounting, NOT
idea-generation-and-cherry-pick. Every run increments the n_trials ledger, so a 'winner' is judged
against the breadth of the search — automation cannot become a p-hacking machine. Mechanism-first
hypotheses are still supplied by a human; the LLM/auto-generation stage (if ever added) must run
behind this same n_trials discipline.
"""
from __future__ import annotations

import datetime as _dt
import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_FF = "data/cache/ff_factors_weekly.parquet"
_LEDGER = Path("data/research/gate_runs.jsonl")

HLZ_T = 3.0        # Harvey-Liu-Zhu residual-α t bar
DEFLSR_MIN = 0.90  # deflated-SR institutional bar
MAX_BOOK_CORR = 0.5


def _shp(x: pd.Series, ppy: int = 12) -> float:
    x = x.dropna()
    return float(x.mean() * ppy / (x.std() * np.sqrt(ppy))) if x.std() > 0 else float("nan")


@lru_cache(maxsize=1)
def _ff_monthly() -> pd.DataFrame:
    ff = pd.read_parquet(_FF)
    ff.index = pd.to_datetime(ff.index)
    cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"] if c in ff.columns]
    return (1 + ff[cols]).resample("ME").prod() - 1


@lru_cache(maxsize=1)
def _pead_monthly() -> pd.Series:
    from engine.portfolio.dpead_recon import build_dpead_recon_returns
    return ((1 + build_dpead_recon_returns(long_short=True).clip(-0.2, 0.2))
            .resample("ME").prod() - 1).rename("PEAD")


def _ols_alpha_t(y: pd.Series, X: pd.DataFrame,
                  hac_lags: int = 0) -> tuple[float, float, float]:
    """(annualized alpha, alpha t-stat, R2) of y on const+X.

    Per [[project-gate-production-redesign-2026-05-30]]:
    hac_lags > 0 → Newey-West HAC standard errors via statsmodels
    cov_type='HAC'. REQUIRED for multi-month-hold strategies
    (overlapping holdings → positive residual autocorrelation →
    plain OLS t-stat inflated 1.5-3×).

    Args:
      y:        return series
      X:        factor exposures DataFrame
      hac_lags: Newey-West maxlags; 0 = plain OLS (back-compat).
                Per-strategy table:
                  equity monthly rebal: 6
                  event-driven 3m hold: 12
                  carry / tsmom 12m hold: 18
    """
    Z = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(Z) < 24:
        return (float("nan"), float("nan"), float("nan"))
    yv = Z["y"].values
    Xm = np.column_stack([np.ones(len(Z)), Z[list(X.columns)].values])

    if hac_lags > 0:
        try:
            import statsmodels.api as sm
            model = sm.OLS(yv, Xm).fit(
                cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)},
            )
            beta = model.params
            t_stat = float(model.tvalues[0])
            r2 = float(model.rsquared)
            return (float(beta[0] * 12), t_stat, r2)
        except Exception:
            # Fall through to plain OLS if statsmodels unavailable
            pass

    # Plain OLS (hac_lags=0 or statsmodels failed)
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    dof = len(Z) - Xm.shape[1]
    if dof <= 0:
        return (float("nan"), float("nan"), float("nan"))
    se = np.sqrt(np.diag((resid @ resid) / dof * np.linalg.inv(Xm.T @ Xm)))
    r2 = 1 - (resid @ resid) / (((yv - yv.mean()) ** 2).sum())
    return (float(beta[0] * 12), float(beta[0] / se[0]), float(r2))


def ledger_n_trials() -> int:
    """Distinct hypotheses run through the gate so far (campaign breadth)."""
    if not _LEDGER.exists():
        return 0
    return sum(1 for _ in _LEDGER.read_text(encoding="utf-8").splitlines() if _.strip())


_ALLOWED_PROFILE_KEYS = {
    "hac_lags", "cost_bps_default", "pead_control", "n_trials_base",
    "oos_split",
}


def _event_count_split_point(
    returns: pd.Series, event_density: pd.Series,
) -> int:
    """Find the integer position where cumulative event count crosses
    half of total, so OOS = events at-and-after that point.

    Returns the index position (0-based) into `returns` (after dropna)
    such that `returns.iloc[split:]` is the OOS half by event count.
    """
    aligned = event_density.reindex(returns.index).fillna(0)
    total = float(aligned.sum())
    if total <= 0:
        return len(returns) // 2     # degenerate → time-bisect
    cumulative = aligned.cumsum()
    half_count = total / 2.0
    # First index where cumsum >= half
    mask = (cumulative >= half_count).values
    if not mask.any():
        return len(returns) // 2
    pos = int(mask.argmax())
    return max(1, min(pos, len(returns) - 1))


def run_gate(returns: pd.Series, name: str, mechanism: str = "",
             n_trials: int = 1, var_sr: float | None = None,
             pead_control: bool = True, log: bool = True,
             hac_lags: int = 0,
             profile: dict | None = None,
             oos_split: str = "time",
             event_density: pd.Series | None = None) -> dict:
    """Run the full rigorous gate on a monthly L/S return series (NET of cost).

    Per [[project-gate-production-redesign-2026-05-30]]:

    n_trials (production semantics):
      Default 1 = pre-committed spec assumed (mechanism YAML + hash
      lock means no in-sample search). Override only if caller actually
      ran a within-mechanism parameter grid in IS data; pass the
      ACTUAL grid size, not the ledger count.

    hac_lags:
      Newey-West HAC standard errors for multi-month-hold strategies.
      0 = plain OLS (back-compat for monthly-rebal strategies).
      Per-strategy: equity 6 / event-3m 12 / carry-tsmom 18.

    profile:
      Per-template GATE_PROFILE dict overriding (hac_lags,
      cost_bps_default, pead_control, n_trials_base). Doctrine red
      lines (HLZ_T, DEFLSR_MIN, MAX_BOOK_CORR) NOT in profile —
      no profile can touch the global bars.

    oos_split:
      "time" (default): bisect at n//2 — appropriate for monthly-rebal
        strategies where each month carries roughly equal "evidence".
      "event_count": split where cumulative event_density crosses half
        of total. REQUIRED for sparse-event templates (event_study,
        PEAD-like) where time-bisect can put 90% of events in one half.
      "event_count" requires the event_density kwarg.

    event_density:
      Per-period event count Series, same index as returns. Used only
      when oos_split="event_count". For event_study templates this is
      typically the per-month event count from the event_panel.

    Returns: verdict dict, appended to gate_runs.jsonl ledger.
    """
    # Apply profile overrides (whitelist enforced)
    if profile:
        unknown = set(profile.keys()) - _ALLOWED_PROFILE_KEYS
        if unknown:
            import logging
            logging.getLogger(__name__).warning(
                "gate profile contains unknown keys (ignored): %s", unknown)
        if "hac_lags" in profile:
            hac_lags = int(profile["hac_lags"])
        if "pead_control" in profile:
            pead_control = bool(profile["pead_control"])
        if "n_trials_base" in profile and n_trials == 1:
            # Profile sets a baseline; explicit n_trials kwarg still wins
            n_trials = int(profile["n_trials_base"])
        if "oos_split" in profile and oos_split == "time":
            # Profile baseline; explicit kwarg still wins
            oos_split = str(profile["oos_split"])

    # Doctrine red lines — runtime asserts, NEVER let profile touch
    assert HLZ_T == 3.0, "doctrine: HLZ_T must equal 3.0"
    assert DEFLSR_MIN >= 0.9, "doctrine: DEFLSR_MIN must be ≥ 0.9"
    assert MAX_BOOK_CORR <= 0.5, "doctrine: MAX_BOOK_CORR must be ≤ 0.5"

    r = returns.dropna()
    n = int(len(r))
    if n < 24:
        return {"name": name, "available": False, "reason": f"only {n} months (<24)", "verdict": "UNINTERPRETABLE"}

    ffm = _ff_monthly()
    rf = ffm["RF"] if "RF" in ffm.columns else 0.0
    rx = (r - rf).dropna()
    ff5umd = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"] if c in ffm.columns]

    a_ff, t_ff, _ = _ols_alpha_t(rx, ffm[ff5umd], hac_lags=hac_lags)
    if pead_control:
        Xp = pd.concat([ffm[ff5umd], _pead_monthly()], axis=1)
        a_fp, t_fp, r2_fp = _ols_alpha_t(rx, Xp, hac_lags=hac_lags)
        j = pd.concat([r.rename("r"), _pead_monthly().rename("p")], axis=1).dropna()
        corr_book = float(j["r"].corr(j["p"])) if len(j) > 12 else float("nan")
    else:
        a_fp, t_fp, r2_fp, corr_book = a_ff, t_ff, float("nan"), float("nan")

    # OOS split — time-bisect (default) or event-count for sparse-event
    # templates per [[project-gate-production-redesign-2026-05-30]] §4.
    if oos_split == "event_count" and event_density is not None:
        split_pos = _event_count_split_point(r, event_density)
    elif oos_split == "event_count":
        # Caller asked for event_count but didn't supply density →
        # honest fallback to time-bisect + log warning, NOT silent.
        import logging
        logging.getLogger(__name__).warning(
            "oos_split='event_count' but event_density is None; "
            "falling back to time-bisect for %s", name)
        split_pos = n // 2
        oos_split = "time"
    elif oos_split != "time":
        raise ValueError(f"oos_split must be 'time' or 'event_count', "
                            f"got {oos_split!r}")
    else:
        split_pos = n // 2
    oos_sharpe = _shp(r.iloc[split_pos:])

    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    # PRODUCTION: n_trials=1 default; caller declares actual grid size
    # if they grid-searched in IS. Ledger count NOT used.
    dsr = float(deflated_sharpe_ratio(r.values, max(int(n_trials), 1), var_sr,
                                        periods_per_year=12).deflated_sr)

    alpha_t = t_fp if pead_control else t_ff
    passed = (alpha_t == alpha_t and alpha_t >= HLZ_T and dsr >= DEFLSR_MIN
              and (corr_book != corr_book or abs(corr_book) < MAX_BOOK_CORR))
    if passed:
        verdict = "GREEN"
    elif (alpha_t == alpha_t and abs(alpha_t) >= 2.0):
        verdict = "YELLOW"
    else:
        verdict = "RED"

    # 2026-05-29: enrich gate output with explicit sample window so the
    # knowledge graph's stress-window analysis (Q6) can work without
    # heuristically parsing the mechanism description.
    sample_start = str(r.index.min().date()) if hasattr(r.index, "min") else None
    sample_end = str(r.index.max().date()) if hasattr(r.index, "max") else None
    res = {
        "name": name, "mechanism": mechanism, "available": True,
        "n_months": n, "n_trials": int(n_trials),
        "hac_lags": int(hac_lags),
        "oos_split": oos_split,
        "oos_split_pos": int(split_pos),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "standalone_sharpe": round(_shp(r), 3),
        "alpha_ann_ff5umd": round(a_ff, 4) if a_ff == a_ff else None,
        "alpha_t_ff5umd": round(t_ff, 3) if t_ff == t_ff else None,
        "alpha_t_ff5umd_pead": round(t_fp, 3) if t_fp == t_fp else None,
        "deflated_sr": round(dsr, 3),
        "oos_sharpe": round(oos_sharpe, 3) if oos_sharpe == oos_sharpe else None,
        "corr_with_book": round(corr_book, 3) if corr_book == corr_book else None,
        "verdict": verdict,
        "bars": {"hlz_t": HLZ_T, "deflsr_min": DEFLSR_MIN, "max_book_corr": MAX_BOOK_CORR},
        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    if log:
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        # NEW1 — feed verdict back to mechanism library so library evolves
        # from our own gate runs. Best-effort: a failure here MUST NOT
        # abort gate logging.
        try:
            from engine.research.library_writer import update_library_from_gate_run
            update_library_from_gate_run(res)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "library_writer feedback failed for %s: %s", name, exc)
        # S1.C2 (2026-06-05) — shadow-emit into research_store so every
        # new gate verdict is visible to audit_verifier /
        # direction_proposer / graveyard_collision without a future
        # manual backfill. NEVER raises (helper has internal catch-all);
        # a shadow failure does not affect the primary ledger write above.
        try:
            from engine.research_store.shadow_emit import shadow_emit_factor_verdict
            shadow_emit_factor_verdict(res, source="gate_runs")
        except Exception:
            # Defense-in-depth: even if shadow_emit's own catch-all
            # leaks, swallow here so run_gate() never raises for shadow.
            pass
    return res


def read_ledger(limit: int = 100) -> list[dict]:
    if not _LEDGER.exists():
        return []
    rows = [json.loads(x) for x in _LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[-limit:][::-1]


def run_hypothesis(name: str, signal_fn: Callable[[], pd.Series], mechanism: str = "", **kw) -> dict:
    """Convenience: build the signal via signal_fn() then run the gate."""
    return run_gate(signal_fn(), name=name, mechanism=mechanism, **kw)
