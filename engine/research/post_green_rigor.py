"""engine.research.post_green_rigor — Phase 4.1 (2026-06-13).

Two mechanical rigor checks that fire automatically on every new
cron-emitted GREEN / MARGINAL verdict:

  1. post-publication out-of-sample check
     Re-runs the template on the strict post-publication window
     (paper_pub_year + 1 onwards). Compares OOS verdict to
     full-sample verdict. SURVIVED → publication-bias-resistant;
     DEGRADED → verdict was driven by pre-pub period (suspect);
     DEAD → OOS verdict flipped to RED (publication killed it).

  2. risk-model spanning check
     Regresses the verdict's monthly PnL series on FF5+MOM with
     HAC SE lag 6. If |alpha-t| ≥ HLZ floor 3.0 → SPANNING_PASSED
     (verdict's alpha is orthogonal to standard equity risk model);
     MARGINAL band [1.65, 3.00) → INDETERMINATE; below → SUBSUMED
     (the GREEN may just be FF5+MOM exposure dressed up).

Why mechanical
==============
Both checks are determinate: no LLM, no taste, no design decision —
just re-run the template with different inputs and inspect output.
Per memory feedback_self_audit_blind_spots: structural mitigation is
mechanical replication tests, NOT prompt sophistication. These two
checks structurally enforce the two gaps the external audit
mitigation surfaced (VRP spanning + GP/A post-pub OOS).

Cost
====
~5-30 seconds per check per verdict. Run synchronously after dispatch
returns GREEN. At 15-20 cron verdicts/wk and ~10% GREEN rate (~2/wk),
adds <1min/wk total. Negligible.

Result destination
==================
Appends one RigorReport row per verdict to
data/research/post_green_rigor.jsonl. The cron-digest UI surface (B-2)
reads this file to surface "this GREEN survived OOS + spans FF5+MOM"
or "this GREEN failed OOS check — defer paper-trade" annotations.

Critical findings (DEAD post-pub OR SUBSUMED spanning) also bubble
up via WARNING logs for cron-tail visibility.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RIGOR_LEDGER_PATH = _REPO_ROOT / "data" / "research" / "post_green_rigor.jsonl"

_FF_WEEKLY_PATH = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_weekly.parquet"

# Spanning thresholds (mirrors verdict_thresholds for consistency)
_SPANNING_PASSED_T = 3.0   # HLZ floor — survives multi-test correction
_SPANNING_MARGIN_T = 1.65


# ── Dataclasses ──────────────────────────────────────────────────


@_dc.dataclass(frozen=True)
class PostPubOOSResult:
    """Status of post-publication out-of-sample re-run."""
    status:             str          # SURVIVED / DEGRADED / DEAD / SKIPPED
    note:               str
    paper_pub_year:     Optional[int]
    oos_window:         Optional[str]   # "YYYY-MM:YYYY-MM"
    oos_verdict:        Optional[str]
    oos_nw_t:           Optional[float]
    oos_sharpe:         Optional[float]
    full_sample_verdict: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


@_dc.dataclass(frozen=True)
class BorrowCostStressResult:
    """Phase 4.1.5 (2026-06-13): impact of realistic short-leg borrow
    cost on long-short verdict. Approximates ~50% × annual_borrow_cost_bp
    drag on combined PnL (since long-short ~50% short notional). Flags
    when verdict survives, weakens, or dies under realistic borrow cost.

    Academic anchors:
      - Cohen-Diether-Malloy 2007: HTB rate 100-300bp/yr for
        low-profitability / distressed stocks
      - D'Avolio 2002: median equity short fee ~20bp/yr for liquid
        names, tail of distribution >>1000bp/yr
      - AQR research: liquid long-short factor borrow cost typically
        25-75bp/yr; defaults to 50bp/yr for unknown universes
    """
    status:              str          # SURVIVED / MARGINAL / DEAD / SKIPPED
    note:                str
    annual_cost_bp:      float
    adjusted_sharpe:     Optional[float]
    adjusted_nw_t:       Optional[float]
    adjusted_mean_pnl:   Optional[float]
    delta_sharpe:        Optional[float]   # adjusted - gross
    n_obs_months:        Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


@_dc.dataclass(frozen=True)
class SpanningCheckResult:
    """Status of FF5+MOM spanning regression on monthly PnL."""
    status:        str          # SPANNING_PASSED / INDETERMINATE / SUBSUMED / SKIPPED
    note:          str
    model_name:    str          # "ff5_mom"
    alpha_monthly: Optional[float]
    alpha_t:       Optional[float]
    n_obs_months:  Optional[int]
    betas:         dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return _dc.asdict(self)


@_dc.dataclass(frozen=True)
class RigorReport:
    """Combined post-GREEN rigor output. Appended to ledger jsonl."""
    rigor_id:           str
    ts:                 str
    verdict_event_id:   Optional[str]
    hypothesis_id:      str
    family:             str
    template_name:      Optional[str]
    template_version:   Optional[str]
    original_verdict:   str
    post_pub_oos:       PostPubOOSResult
    spanning:           SpanningCheckResult
    borrow_cost:        Optional[BorrowCostStressResult]   # Check 3 (Phase 4.1.5)
    flags:              list[str]     # ["DEAD_POST_PUB", "SUBSUMED", ...]

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["post_pub_oos"] = self.post_pub_oos.to_dict()
        d["spanning"]     = self.spanning.to_dict()
        d["borrow_cost"]  = self.borrow_cost.to_dict() if self.borrow_cost else None
        return d


# ── Helpers ──────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_paper_pub_year(canonical_paper_window: Optional[str]) -> Optional[int]:
    """Parse '1972-01:2011-12' → 2011 (paper's data ends ~ publication year).
    Convention: papers typically publish 1-2 years after their data ends;
    we use the data end-year as a conservative pub-year proxy."""
    if not canonical_paper_window:
        return None
    try:
        end = canonical_paper_window.split(":", 1)[1]   # "2011-12"
        return int(end.split("-", 1)[0])
    except Exception:
        return None


def _today_iso_month() -> str:
    now = _dt.datetime.utcnow()
    return f"{now.year:04d}-{now.month:02d}"


def _classify_oos_status(
    full_verdict: str, oos_verdict: str,
) -> tuple[str, str]:
    """Compare full-sample vs OOS verdict.

      GREEN  → GREEN    = SURVIVED
      GREEN  → MARGINAL = DEGRADED
      GREEN  → RED      = DEAD
      MARGINAL → GREEN    = SURVIVED (signal got stronger OOS)
      MARGINAL → MARGINAL = SURVIVED
      MARGINAL → RED      = DEGRADED
      Any  → INSUFFICIENT_* = SKIPPED
    """
    if oos_verdict in {"INSUFFICIENT_DATA", "INSUFFICIENT_HISTORY",
                          "EXECUTION_ERROR", "PENDING_TEMPLATE_BUILD"}:
        return "SKIPPED", f"OOS rerun returned {oos_verdict}"
    if full_verdict == "GREEN":
        if oos_verdict == "GREEN":
            return "SURVIVED", "GREEN full → GREEN OOS"
        if oos_verdict == "MARGINAL":
            return "DEGRADED", "GREEN full → MARGINAL OOS (still alive but weakened)"
        return "DEAD", f"GREEN full → {oos_verdict} OOS (publication likely killed it)"
    if full_verdict == "MARGINAL":
        if oos_verdict in ("GREEN", "MARGINAL"):
            return "SURVIVED", f"MARGINAL full → {oos_verdict} OOS"
        return "DEGRADED", f"MARGINAL full → {oos_verdict} OOS"
    return "SKIPPED", f"unsupported full_verdict={full_verdict}"


# ── Check 1: post-pub OOS ────────────────────────────────────────


def check_post_pub_oos(
    spec, dispatch_fn: Callable,
    *, paper_pub_year: Optional[int] = None,
    today_iso_month: Optional[str] = None,
) -> PostPubOOSResult:
    """Re-run the template on strict post-publication window.

    If paper_pub_year not provided, attempts to resolve via
    TemplateContract.canonical_paper_window. If still missing,
    returns SKIPPED.
    """
    if paper_pub_year is None:
        try:
            from engine.agents.strengthener.templates._template_contract import (
                contract_for_scope,
            )
            contract = contract_for_scope(spec.signal_kind, spec.universe)
            if contract is not None:
                paper_pub_year = _parse_paper_pub_year(contract.canonical_paper_window)
        except Exception:
            pass

    if paper_pub_year is None:
        return PostPubOOSResult(
            status               = "SKIPPED",
            note                 = "no paper_pub_year known (TemplateContract missing canonical_paper_window)",
            paper_pub_year       = None,
            oos_window           = None,
            oos_verdict          = None,
            oos_nw_t             = None,
            oos_sharpe           = None,
            full_sample_verdict  = None,
        )

    oos_start = f"{paper_pub_year + 1}-01"
    oos_end   = today_iso_month or _today_iso_month()
    oos_window = f"{oos_start}:{oos_end}"

    # Reconstruct spec with new date_range (dataclasses.replace)
    try:
        oos_spec = _dc.replace(spec, date_range=oos_window)
    except Exception as exc:
        return PostPubOOSResult(
            status="SKIPPED", note=f"could not replace date_range: {exc}",
            paper_pub_year=paper_pub_year, oos_window=oos_window,
            oos_verdict=None, oos_nw_t=None, oos_sharpe=None,
            full_sample_verdict=None,
        )

    try:
        result = dispatch_fn(oos_spec)
    except Exception as exc:
        return PostPubOOSResult(
            status="SKIPPED", note=f"OOS dispatch raised: {exc}",
            paper_pub_year=paper_pub_year, oos_window=oos_window,
            oos_verdict=None, oos_nw_t=None, oos_sharpe=None,
            full_sample_verdict=None,
        )

    if result is None or getattr(result, "verdict", None) is None:
        return PostPubOOSResult(
            status="SKIPPED", note="OOS dispatch returned None",
            paper_pub_year=paper_pub_year, oos_window=oos_window,
            oos_verdict=None, oos_nw_t=None, oos_sharpe=None,
            full_sample_verdict=None,
        )

    m = result.metrics or {}
    return PostPubOOSResult(
        status               = "PENDING_COMPARE",   # filled by caller via _classify_oos_status
        note                 = "",
        paper_pub_year       = paper_pub_year,
        oos_window           = oos_window,
        oos_verdict          = result.verdict,
        oos_nw_t             = m.get("nw_t_gross") or m.get("nw_t_stat"),
        oos_sharpe           = m.get("sharpe_gross") or m.get("sharpe"),
        full_sample_verdict  = None,
    )


# ── Check 2: FF5+MOM spanning ────────────────────────────────────


def _load_ff5_mom_monthly() -> Optional[pd.DataFrame]:
    if not _FF_WEEKLY_PATH.is_file():
        return None
    df = pd.read_parquet(_FF_WEEKLY_PATH)
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def check_risk_model_spanning(
    pnl_series: pd.Series,
    *, model_factors: Optional[pd.DataFrame] = None,
) -> SpanningCheckResult:
    """Regress monthly PnL on FF5+MOM with HAC SE lag 6.
    SPANNING_PASSED if |alpha-t| ≥ 3.0, INDETERMINATE if [1.65, 3.00),
    SUBSUMED otherwise.

    pnl_series must be monthly-frequency (index = month-end timestamps,
    values = monthly returns or PnL).
    """
    if pnl_series is None or len(pnl_series) < 24:
        return SpanningCheckResult(
            status="SKIPPED",
            note=f"pnl_series too short (n={0 if pnl_series is None else len(pnl_series)} < 24mo)",
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=None, betas={},
        )

    if model_factors is None:
        model_factors = _load_ff5_mom_monthly()
    if model_factors is None or model_factors.empty:
        return SpanningCheckResult(
            status="SKIPPED",
            note="FF5+MOM monthly factor cache not loadable",
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=None, betas={},
        )

    # Identify RF column to EXCLUDE from factors (we never use it on the
    # LHS — every template's pnl_series is already long-short / spread /
    # variance-payoff = cash-neutral by construction. Subtracting RF
    # would double-deduct and silently flip alpha sign — caught
    # 2026-06-13 when VRP alpha came out -4.4 despite +0.49%/mo mean PnL).
    rf_col = None
    for c in ("RF", "rf", "Rf"):
        if c in model_factors.columns:
            rf_col = c
            break
    factor_cols = [c for c in model_factors.columns if c != rf_col]
    if not factor_cols:
        return SpanningCheckResult(
            status="SKIPPED", note="no factor columns in model",
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=None, betas={},
        )

    # Align indices via inner join on month-end
    y = pnl_series.copy()
    y.index = pd.to_datetime(y.index).to_period("M").to_timestamp("M")
    mf = model_factors.copy()
    mf.index = pd.to_datetime(mf.index).to_period("M").to_timestamp("M")

    df_parts = {"y": y}
    for c in factor_cols:
        df_parts[c] = mf[c]
    df = pd.concat(df_parts, axis=1).dropna()
    n = len(df)
    if n < 24:
        return SpanningCheckResult(
            status="SKIPPED",
            note=f"aligned series too short (n={n})",
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=n, betas={},
        )

    # LHS = raw PnL (cash-neutral by construction; NO RF subtraction)
    excess = df["y"]
    X = df[factor_cols].values

    try:
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        ols = sm.OLS(excess.values, X_const).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        alpha = float(ols.params[0])
        alpha_t = float(ols.tvalues[0])
        betas = {c: float(ols.params[1 + i]) for i, c in enumerate(factor_cols)}
    except Exception as exc:
        return SpanningCheckResult(
            status="SKIPPED",
            note=f"OLS failed: {exc}",
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=n, betas={},
        )

    abs_t = abs(alpha_t)
    if abs_t >= _SPANNING_PASSED_T:
        status, note = "SPANNING_PASSED", (
            f"|alpha-t|={abs_t:.2f} ≥ {_SPANNING_PASSED_T:.2f}; alpha is "
            f"orthogonal to FF5+MOM risk model"
        )
    elif abs_t >= _SPANNING_MARGIN_T:
        status, note = "INDETERMINATE", (
            f"|alpha-t|={abs_t:.2f} in [{_SPANNING_MARGIN_T:.2f}, "
            f"{_SPANNING_PASSED_T:.2f}); marginal vs equity risk model"
        )
    else:
        status, note = "SUBSUMED", (
            f"|alpha-t|={abs_t:.2f} < {_SPANNING_MARGIN_T:.2f}; verdict's "
            f"alpha likely just FF5+MOM exposure dressed up"
        )

    return SpanningCheckResult(
        status=status, note=note, model_name="ff5_mom",
        alpha_monthly=alpha, alpha_t=alpha_t, n_obs_months=n, betas=betas,
    )


# ── Check 3: borrow cost stress (Phase 4.1.5) ────────────────────


# Universe-specific borrow cost defaults (annual bp). Values derived
# from Cohen-Diether-Malloy 2007 + D'Avolio 2002 + AQR long-short
# factor reports. CONSERVATIVE: assume bottom decile of US equity
# cross-sec factors are partially HTB.
BORROW_COST_DEFAULTS_BP: dict[str, float] = {
    "us_equities_top_3000":   80.0,   # bottom decile of US factors
    "us_equities_sp500":      40.0,   # liquid large-cap, low borrow
    "us_equities_pead":       100.0,  # smallcap + earnings-driven, higher HTB
    "us_equities_sector_etf": 15.0,   # ETF basket, very low borrow
    "us_equities_spx_options":0.0,    # not long-short equity
    "us_balanced_60_40":      0.0,    # long-only base portfolio
    "ken_french_ff5_mom":     30.0,   # factor return series, frictionless
                                       # but realistic if implemented
    "fx_g10":                 20.0,   # FX swap funding spread
}


def check_borrow_cost_stress(
    pnl_series: pd.Series,
    *,
    universe: str,
    annual_borrow_cost_bp: Optional[float] = None,
) -> BorrowCostStressResult:
    """Stress test verdict under realistic short-leg borrow cost.

    Method: subtract estimated annual borrow cost / 12 (per month) /
    2 (half the position is short on average) from each monthly PnL.
    Re-compute Sharpe + NW-t. Compare to gross thresholds.

    Status:
      SURVIVED: adjusted_nw_t still >= 1.65 (verdict survives)
      MARGINAL: adjusted_nw_t in [1.0, 1.65) (weakened but not dead)
      DEAD:     adjusted_nw_t < 1.0 (borrow cost kills verdict)
      SKIPPED:  not long-short OR series too short
    """
    if annual_borrow_cost_bp is None:
        annual_borrow_cost_bp = BORROW_COST_DEFAULTS_BP.get(universe, 50.0)

    if annual_borrow_cost_bp <= 0.0:
        return BorrowCostStressResult(
            status="SKIPPED",
            note=f"universe='{universe}' is long-only OR cash-neutral non-equity (cost=0)",
            annual_cost_bp=annual_borrow_cost_bp,
            adjusted_sharpe=None, adjusted_nw_t=None,
            adjusted_mean_pnl=None, delta_sharpe=None, n_obs_months=None,
        )

    if pnl_series is None or len(pnl_series) < 24:
        return BorrowCostStressResult(
            status="SKIPPED",
            note=f"pnl_series too short (n={0 if pnl_series is None else len(pnl_series)})",
            annual_cost_bp=annual_borrow_cost_bp,
            adjusted_sharpe=None, adjusted_nw_t=None,
            adjusted_mean_pnl=None, delta_sharpe=None, n_obs_months=None,
        )

    # Half the position is short on average → effective drag = half of full cost
    monthly_drag = (annual_borrow_cost_bp / 10000.0) / 12.0 / 2.0
    adjusted = pnl_series - monthly_drag

    mean_pnl  = float(adjusted.mean())
    std_pnl   = float(adjusted.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        return BorrowCostStressResult(
            status="SKIPPED",
            note="adjusted series degenerate variance",
            annual_cost_bp=annual_borrow_cost_bp,
            adjusted_sharpe=None, adjusted_nw_t=None,
            adjusted_mean_pnl=None, delta_sharpe=None,
            n_obs_months=len(adjusted),
        )

    gross_mean = float(pnl_series.mean())
    gross_std  = float(pnl_series.std(ddof=1))
    gross_sharpe = gross_mean / gross_std * math.sqrt(12.0) if gross_std > 0 else float("nan")
    adjusted_sharpe = mean_pnl / std_pnl * math.sqrt(12.0)
    delta_sharpe = adjusted_sharpe - gross_sharpe

    try:
        import statsmodels.api as sm
        x = np.ones(len(adjusted))
        ols = sm.OLS(adjusted.values, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        adjusted_t = float(ols.tvalues[0])
    except Exception:
        adjusted_t = mean_pnl / (std_pnl / math.sqrt(len(adjusted)))

    if adjusted_t >= 1.65:
        status, note = "SURVIVED", (
            f"adjusted NW-t={adjusted_t:.2f} >= 1.65; verdict survives "
            f"{annual_borrow_cost_bp:.0f}bp/yr borrow cost"
        )
    elif adjusted_t >= 1.0:
        status, note = "MARGINAL", (
            f"adjusted NW-t={adjusted_t:.2f} in [1.0, 1.65); verdict "
            f"weakened but not dead under {annual_borrow_cost_bp:.0f}bp/yr"
        )
    else:
        status, note = "DEAD", (
            f"adjusted NW-t={adjusted_t:.2f} < 1.0; verdict DIES under "
            f"realistic {annual_borrow_cost_bp:.0f}bp/yr borrow cost"
        )

    return BorrowCostStressResult(
        status=status, note=note,
        annual_cost_bp=annual_borrow_cost_bp,
        adjusted_sharpe=adjusted_sharpe,
        adjusted_nw_t=adjusted_t,
        adjusted_mean_pnl=mean_pnl,
        delta_sharpe=delta_sharpe,
        n_obs_months=len(adjusted),
    )


# ── Combined runner ──────────────────────────────────────────────


def run_post_green_rigor(
    *,
    spec, dispatch_fn: Callable,
    verdict: str, hypothesis_id: str, family: str,
    template_result: Any,
    verdict_event_id: Optional[str] = None,
    ledger_path: Optional[Path] = None,
) -> RigorReport:
    """Run both rigor checks on a fresh verdict. Returns RigorReport;
    also appends one row to the ledger (jsonl).

    Only runs when verdict ∈ {GREEN, MARGINAL}. For RED / other,
    returns a SKIPPED-everywhere report with no ledger row written.
    """
    import uuid

    flags: list[str] = []

    # Short-circuit on non-eligible verdicts (no ledger write, no rerun cost)
    if verdict not in {"GREEN", "MARGINAL"}:
        return RigorReport(
            rigor_id          = str(uuid.uuid4()),
            ts                = _utc_iso(),
            verdict_event_id  = verdict_event_id,
            hypothesis_id     = hypothesis_id,
            family            = family,
            template_name     = getattr(template_result, "template_name", None),
            template_version  = getattr(template_result, "template_version", None),
            original_verdict  = verdict,
            post_pub_oos      = PostPubOOSResult(
                status="SKIPPED", note=f"verdict={verdict} not in {{GREEN, MARGINAL}}",
                paper_pub_year=None, oos_window=None, oos_verdict=None,
                oos_nw_t=None, oos_sharpe=None, full_sample_verdict=verdict,
            ),
            spanning          = SpanningCheckResult(
                status="SKIPPED", note=f"verdict={verdict} ineligible",
                model_name="ff5_mom", alpha_monthly=None, alpha_t=None,
                n_obs_months=None, betas={},
            ),
            borrow_cost       = None,
            flags             = [],
        )

    # OOS check
    oos = check_post_pub_oos(spec, dispatch_fn)
    if oos.status == "PENDING_COMPARE":
        status, note = _classify_oos_status(verdict, oos.oos_verdict or "")
        oos = _dc.replace(oos, status=status, note=note,
                            full_sample_verdict=verdict)
    if oos.status == "DEAD":
        flags.append("DEAD_POST_PUB")
    elif oos.status == "DEGRADED":
        flags.append("DEGRADED_POST_PUB")

    # Spanning check — SKIP when the verdict ITSELF is a spanning test
    # (the template already ran a spanning test; re-running FF5+MOM
    # spanning on a spanning_test's pnl_series is circular because the
    # series IS one of the FF5+MOM factors, e.g. MOM-on-FF5's pnl is
    # MOM itself which is in the rigor model → trivial SUBSUMED).
    is_spanning_verdict = (
        getattr(spec, "signal_kind", "") == "spanning_test"
        or str(family or "").upper().startswith("SPANNING_")
    )
    if is_spanning_verdict:
        spanning = SpanningCheckResult(
            status="SKIPPED",
            note=("verdict is itself a spanning test; secondary FF5+MOM "
                    "spanning is circular (test_asset is in the model)"),
            model_name="ff5_mom",
            alpha_monthly=None, alpha_t=None, n_obs_months=None, betas={},
        )
    else:
        artifacts = getattr(template_result, "artifacts", None) or {}
        pnl_df = artifacts.get("pnl_series_df")
        pnl_col = artifacts.get("pnl_gross_col") or artifacts.get("pnl_default_col")
        pnl_series = None
        if isinstance(pnl_df, pd.DataFrame) and pnl_col and pnl_col in pnl_df.columns:
            pnl_series = pnl_df[pnl_col].dropna()

        if pnl_series is None or pnl_series.empty:
            spanning = SpanningCheckResult(
                status="SKIPPED",
                note="template did not expose pnl_series_df + pnl_gross_col artifact",
                model_name="ff5_mom",
                alpha_monthly=None, alpha_t=None, n_obs_months=None, betas={},
            )
        else:
            spanning = check_risk_model_spanning(pnl_series)
        if spanning.status == "SUBSUMED":
            flags.append("SUBSUMED_BY_FF5_MOM")

    # Check 3: borrow cost stress (Phase 4.1.5)
    borrow_cost: Optional[BorrowCostStressResult] = None
    universe = getattr(spec, "universe", None) or ""
    artifacts = getattr(template_result, "artifacts", None) or {}
    pnl_df_b = artifacts.get("pnl_series_df")
    pnl_col_b = artifacts.get("pnl_gross_col") or artifacts.get("pnl_default_col")
    pnl_series_b = None
    if isinstance(pnl_df_b, pd.DataFrame) and pnl_col_b and pnl_col_b in pnl_df_b.columns:
        pnl_series_b = pnl_df_b[pnl_col_b].dropna()
    if pnl_series_b is not None and not pnl_series_b.empty:
        borrow_cost = check_borrow_cost_stress(pnl_series_b, universe=universe)
        if borrow_cost.status == "DEAD":
            flags.append("DEAD_UNDER_BORROW_COST")
        elif borrow_cost.status == "MARGINAL":
            flags.append("MARGINAL_UNDER_BORROW_COST")

    template_name = getattr(template_result, "template_name", None)
    template_version = getattr(template_result, "template_version", None)

    report = RigorReport(
        rigor_id          = str(uuid.uuid4()),
        ts                = _utc_iso(),
        verdict_event_id  = verdict_event_id,
        hypothesis_id     = hypothesis_id,
        family            = family,
        template_name     = template_name,
        template_version  = template_version,
        original_verdict  = verdict,
        post_pub_oos      = oos,
        spanning          = spanning,
        borrow_cost       = borrow_cost,
        flags             = flags,
    )

    # Append to ledger
    p = ledger_path or RIGOR_LEDGER_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(report.to_dict(), ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("post_green_rigor: ledger write failed: %s", exc)

    if flags:
        logger.warning(
            "post_green_rigor FLAGS hypothesis=%s verdict=%s flags=%s "
            "oos=%s spanning=%s",
            hypothesis_id, verdict, flags, oos.status, spanning.status,
        )

    # Phase 4.1 → research_store event (2026-06-13). Auto-resolve the
    # subject_id same way factor_verdict_emit does (tier_c_auto_<hash>_<sk>)
    # so the rigor event attaches to the same factor subject as its
    # parent verdict. NEVER raises — event emit failure is observability,
    # not pipeline-critical.
    try:
        signal_kind = getattr(spec, "signal_kind", "")
        short_hid = (hypothesis_id or "unknown")[:8]
        subject_id = f"tier_c_auto_{short_hid}_{signal_kind}"

        # Auto-register subject if missing (idempotent in registry)
        try:
            from engine.research_store import registry
            from engine.research_store.schema import SubjectType
            registry.register_subject(
                subject_id   = subject_id,
                subject_type = SubjectType.factor,
                family       = family or "OTHER",
                description  = f"Rigor pass on {hypothesis_id} ({signal_kind})",
                created_by   = "post_green_rigor.run_post_green_rigor",
            )
        except Exception:
            pass   # already-registered raises; that's fine

        from engine.research_store import emit
        emit.post_green_rigor_run(
            subject_id        = subject_id,
            rigor_id          = report.rigor_id,
            verdict_event_id  = verdict_event_id,
            original_verdict  = verdict,
            oos_status        = oos.status,
            oos_nw_t          = oos.oos_nw_t,
            spanning_status   = spanning.status,
            spanning_alpha_t  = spanning.alpha_t,
            borrow_status     = (borrow_cost.status if borrow_cost else None),
            borrow_adj_nw_t   = (borrow_cost.adjusted_nw_t if borrow_cost else None),
            flags             = flags,
            rigor_ledger_path = str(p),
            family            = family,
        )
    except Exception as exc:
        logger.warning(
            "post_green_rigor: emit to research_store failed (suppressed): %s",
            exc,
        )

    return report
