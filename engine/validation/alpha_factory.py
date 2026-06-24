"""engine/validation/alpha_factory.py — universe-AWARE go/no-go gate.

The standard pipeline every NEW factor candidate must pass before it can
be taken seriously. Wraps the Phase-1 validation battery (deflated
Sharpe / residual-alpha attribution / after-cost / decay / diversification)
into a single verdict.

CRITICAL design point (universe heterogeneity): you CANNOT run every
factor against the same fixed benchmark. FF5+UMD are US-single-stock
factors — regressing a cross-ETF strategy (K1 BAB) against them gave
R²=0.01 and a meaningless residual alpha. So the factory is NOT
universe-blind: each candidate DECLARES its universe context via a
CandidateSpec (benchmark, cost class, market proxy, trial count,
frequency), and the factory runs the universe-AGNOSTIC math with those
universe-SPECIFIC inputs. This forces the analyst to state the universe
up front and prevents benchmark mismatch at the source.

The MATH is universe-agnostic (deflated Sharpe, decay, effective bets);
the INPUTS are universe-specific (which factors, which cost class,
which market). The factory enforces that separation.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Every gate() run is appended here. The ledger is the anti-self-deception
# spine: you cannot quietly re-run a candidate (changing n_trials, benchmark,
# or cost_class) until it goes GREEN — every attempt is recorded, and gate()
# flags a re-screen of the SAME return series under CHANGED assumptions.
_LEDGER = Path("data/validation/factory_ledger.jsonl")

# Benchmark sets the residual-alpha regression uses. The factory picks the
# loader by name so a cross-ETF candidate is NOT regressed against US
# single-stock factors.
VALID_BENCHMARKS = ("ff5_umd", "aqr_bab", "market_only", "none")
VALID_FREQ = {"daily": 252, "weekly": 52, "monthly": 12}
# Cost classes map to round-trip bps (one-way base+half-spread x2),
# from engine.execution.cost_model instrument tiers.
COST_CLASS_ROUNDTRIP_BPS = {
    "etf_tier1":    10.0,
    "etf_tier2":    16.0,
    "ss_large":     26.0,
    "ss_mid":       50.0,
    "ss_small":     80.0,
    "mutual_fund":  0.0,
}


@dataclass(frozen=True)
class CandidateSpec:
    """A factor candidate + its UNIVERSE CONTEXT. The universe-specific
    fields are what make the factory's verdict correct rather than a
    benchmark-mismatched artifact.

    Fields:
      name:            label
      returns:         the candidate's return series (index = dates)
      frequency:       'daily' | 'weekly' | 'monthly' (sets annualization)
      n_trials:        multiple-testing breadth for THIS candidate's
                       research search (NOT a global constant)
      benchmark:       which factor set to regress for residual alpha —
                       MUST match the universe (US single stock → ff5_umd;
                       cross-ETF / BAB → aqr_bab; else market_only/none)
      cost_class:      instrument class for the after-cost drag
      annual_turnover: book-fraction turned over per year
      already_net:     True if `returns` is already net of cost (skip the
                       after-cost drag — e.g. D_PEAD walk-forward net)
      book_returns:    existing book (same frequency) for the
                       diversification / effective-bets check; optional
    """
    name:               str
    returns:            pd.Series
    frequency:          str = "weekly"
    n_trials:           int = 1
    benchmark:          str = "market_only"
    cost_class:         str = "ss_large"
    annual_turnover:    float = 4.0
    already_net:        bool = False
    book_returns:       Optional[pd.DataFrame] = None
    # Audit A2 F7 fields — regime-conditional gating
    # T2.5 (2026-06-05 audit C4 fix): default flipped False -> True.
    # Pre-T2.5 the gate was opt-in for "back-compat"; in practice no
    # caller set it explicitly, so a "regime-fragile" candidate that
    # collapses in 2008-09 or 2020-Q1 could pass the gate just by
    # having a strong full-sample DSR. That defeats the F7 fix.
    # Callers who genuinely don't want regime gating can pass False
    # explicitly; the regime_floor=-1.0 default is already permissive
    # ("negative Sharpe in a crisis OK, just don't collapse") so very
    # few previously-GREEN candidates will flip with this change.
    enable_regime_gate: bool = True
    # Minimum acceptable Sharpe in the WORST regime window. Floor of -1.0
    # means "we tolerate negative Sharpe in a crisis but not collapse".
    # Set to 0.0 for stricter "must be non-negative in every regime".
    regime_floor:       float = -1.0
    # Skip regimes with fewer obs than this — too noisy to gate on.
    regime_min_obs:     int   = 13


@dataclass(frozen=True)
class FactoryVerdict:
    name:                str
    n_obs:               int
    deflated_sr:         float
    residual_alpha_ann:  float
    residual_alpha_t:    float
    benchmark_used:      str
    net_deflated_sr:     float
    recent_alive:        str
    effective_bets_delta: Optional[float]   # change in book effective bets if added
    light:               str   # GREEN / YELLOW / RED
    reasons:             tuple
    # Audit A2 F7 fields. None if regime gate was disabled. Default-None
    # so old ledger entries deserialize without modification.
    regime_breakdown:    Optional[dict] = None      # {label → Sharpe (annualized)}
    worst_regime:        Optional[str]   = None
    worst_regime_sharpe: Optional[float] = None


def _ff_frame(freq: str) -> pd.DataFrame:
    """FF5+UMD+RF at the candidate frequency. Monthly candidates use Ken
    French MONTHLY factors (not the weekly loader) so the residual-alpha
    regression is frequency-correct, not an aliased approximation."""
    if freq == "monthly":
        from engine.validation.aqr_factors import load_ff_monthly
        return load_ff_monthly()
    from engine.validation.factor_data import load_factors_weekly
    if freq == "daily":
        logger.warning("ff5_umd has no daily loader; using weekly factors for "
                       "a daily candidate — attribution is approximate")
    return load_factors_weekly()


def _load_benchmark(benchmark: str, freq: str) -> Optional[pd.DataFrame]:
    """Return the factor frame for the requested benchmark at the candidate
    frequency, or None for market_only/none (handled separately)."""
    if benchmark == "ff5_umd":
        return _ff_frame(freq)
    if benchmark == "aqr_bab":
        from engine.validation.aqr_factors import load_bab_usa_monthly, load_ff_monthly
        bab = load_bab_usa_monthly().rename("BAB")
        ff = load_ff_monthly()
        return pd.concat([bab, ff[["Mkt-RF", "RF"]]], axis=1).dropna()
    return None


def _residual_alpha(spec: CandidateSpec) -> tuple[float, float, str]:
    """Universe-appropriate residual-alpha regression. Returns
    (alpha_annual, alpha_tstat, benchmark_label)."""
    import statsmodels.api as sm
    ppy = VALID_FREQ[spec.frequency]
    r = spec.returns.dropna().astype(float)

    if spec.benchmark in ("market_only", "none"):
        if spec.benchmark == "none":
            # alpha = mean return annualized, t from simple t-test
            t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else float("nan")
            return float(r.mean() * ppy), float(t), "none(raw mean)"
        # market_only: regress on Ken French market at the candidate freq
        fac = _ff_frame(spec.frequency)
        from engine.validation.factor_data import align_returns_to_factors
        rs, fs = align_returns_to_factors(r.to_frame("y"), fac)
        df = pd.concat([rs["y"], fs[["Mkt-RF", "RF"]]], axis=1).dropna()
        X = sm.add_constant(np.asarray(df["Mkt-RF"], dtype=float))
        m = sm.OLS(np.asarray(df["y"] - df["RF"], dtype=float), X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 8})
        return float(m.params[0] * ppy), float(m.tvalues[0]), "market_only"

    fac = _load_benchmark(spec.benchmark, spec.frequency)
    if fac is None:
        return float("nan"), float("nan"), f"{spec.benchmark}(unavailable)"
    # Align candidate to benchmark by nearest date
    from engine.validation.factor_data import align_returns_to_factors
    rs, fs = align_returns_to_factors(r.to_frame("y"), fac)
    rf = fs["RF"] if "RF" in fs else 0.0
    cols = [c for c in fs.columns if c != "RF"]
    df = pd.concat([rs["y"] - rf, fs[cols]], axis=1).dropna()
    if len(df) < 24:
        return float("nan"), float("nan"), f"{spec.benchmark}(insufficient overlap)"
    X = sm.add_constant(df[cols].values)
    m = sm.OLS(df.iloc[:, 0].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
    return float(m.params[0] * ppy), float(m.tvalues[0]), spec.benchmark


def _regime_breakdown(
    returns:       pd.Series,
    annualization: int,
    min_obs:       int,
) -> tuple[dict[str, float], Optional[str], Optional[float]]:
    """Per-regime annualized Sharpe via engine.factor_regression.regime.

    Returns:
      breakdown:           {regime_label → Sharpe} for regimes with ≥ min_obs
      worst_regime_label:  label of worst regime (or None if no usable regimes)
      worst_regime_sharpe: that regime's Sharpe

    Skips regimes with insufficient observations (too noisy to gate on).
    """
    from engine.factor_regression.regime import run_regime_decomposition

    rs = run_regime_decomposition(returns, annualization=annualization)
    breakdown: dict[str, float] = {}
    for r in rs:
        if r.n_obs >= min_obs:
            breakdown[r.label] = round(r.sharpe_annualized, 3)
    if not breakdown:
        return {}, None, None
    worst_label = min(breakdown, key=lambda k: breakdown[k])
    return breakdown, worst_label, breakdown[worst_label]


def _decide_verdict(
    *,
    net_dsr:       float,
    alpha_t:       float,
    alpha_ann:     float,
    bench_used:    str,
    decay_verdict: str,
    worst_regime_sharpe: Optional[float] = None,
    worst_regime_label:  Optional[str]   = None,
    regime_floor:        float           = -1.0,
) -> tuple[str, list[str]]:
    """Pure verdict-decision function. Returns (light, reasons).

    Doctrine (post audit A2 F1 2026-06-03):

      - GREEN requires ALL THREE: strong DSR AND real residual alpha
        AND alive recent window.
      - DEAD / WEAK recent window FORCES RED, never YELLOW. Decay
        failure means the alpha is no longer accruing; the strategy
        is not iterating-worthy, it's killed.
      - Stat failures (DSR collapses / no residual alpha) still RED.
      - YELLOW only for marginal stats with alive decay.

    Decay-string contract is currently "DEAD" / "WEAK" substrings (see
    decay_split). Audit F3 is the structural fix; this function uses
    the same substring contract for now and only fixes F1.
    """
    reasons: list[str] = []
    if np.isnan(net_dsr) or np.isnan(alpha_t):
        reasons.append("undefined inputs (check overlap/benchmark)")
        return "YELLOW", reasons

    strong_dsr   = net_dsr >= 0.90
    ok_dsr       = net_dsr >= 0.70
    real_alpha   = abs(alpha_t) >= 2.0 and alpha_ann > 0
    weak_alpha   = abs(alpha_t) >= 1.65 and alpha_ann > 0
    alive        = "DEAD" not in decay_verdict and "WEAK" not in decay_verdict

    # F1 fix: decay failure always blocks GREEN and downgrades to RED,
    # regardless of how good the other stats are. A dead strategy is not
    # iterating-worthy.
    if not alive:
        reasons.append(f"decay: {decay_verdict}")
        return "RED", reasons

    # F7 fix: regime-conditional gating. If a worst-regime Sharpe is
    # provided AND it falls below the floor, the strategy is regime-
    # fragile and cannot pass to GREEN. Same doctrine as F1 — a strategy
    # that collapses in any named crisis regime is not iterating-worthy.
    if worst_regime_sharpe is not None and worst_regime_sharpe < regime_floor:
        reasons.append(
            f"regime-fragile: {worst_regime_label} Sharpe "
            f"{worst_regime_sharpe:+.2f} < floor {regime_floor:+.2f}"
        )
        return "RED", reasons

    if strong_dsr and real_alpha:
        reasons.append("survives cost + multiple-testing, real residual alpha, alive")
        return "GREEN", reasons
    if (not ok_dsr) or (not weak_alpha):
        if not ok_dsr:
            reasons.append(f"net deflated SR {net_dsr:.2f} collapses under cost+trials")
        if not weak_alpha:
            reasons.append(
                f"no residual alpha vs {bench_used} (t={alpha_t:.2f}) — "
                f"likely just factor beta"
            )
        return "RED", reasons
    reasons.append("marginal — iterate, not deployable")
    return "YELLOW", reasons


def screen_candidate(spec: CandidateSpec) -> FactoryVerdict:
    """Run the full universe-aware battery and return a GREEN/YELLOW/RED
    verdict. Each sub-test uses the candidate's declared universe context."""
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    from engine.validation.rolling_sharpe import decay_split

    r = spec.returns.dropna().astype(float)
    ppy = VALID_FREQ[spec.frequency]
    reasons: list[str] = []

    # 1. Deflated Sharpe (multiple-testing corrected, candidate's own N)
    dsr = deflated_sharpe_ratio(r.values, n_trials=spec.n_trials,
                                periods_per_year=ppy).deflated_sr

    # 2. Residual alpha vs the UNIVERSE-APPROPRIATE benchmark
    alpha_ann, alpha_t, bench_used = _residual_alpha(spec)

    # 3. After-cost deflated Sharpe (skip if already net)
    if spec.already_net or spec.cost_class == "mutual_fund":
        net_dsr = dsr
    else:
        rt_bps = COST_CLASS_ROUNDTRIP_BPS.get(spec.cost_class, 26.0)
        annual_drag = spec.annual_turnover * rt_bps / 10000.0
        net = r - annual_drag / ppy
        net_dsr = deflated_sharpe_ratio(net.values, n_trials=spec.n_trials,
                                        periods_per_year=ppy).deflated_sr

    # 4. Decay (recent window still alive?)
    recent_weeks = {"daily": 756, "weekly": 156, "monthly": 36}[spec.frequency]
    decay = decay_split(r, recent_weeks=recent_weeks, ppy=ppy)

    # 5. Diversification delta (does adding it raise the book's effective bets?)
    eff_delta = None
    if spec.book_returns is not None and not spec.book_returns.empty:
        from engine.validation.diversification import effective_number_of_bets
        # Compare book WITHOUT the candidate vs WITH it. Drop the
        # candidate's own column from the book first (it may already be a
        # member, e.g. when re-screening an existing strategy).
        book_wo = spec.book_returns.drop(columns=[spec.name], errors="ignore").dropna()
        cand = r.rename(spec.name)
        common = book_wo.join(cand, how="inner").dropna()
        if len(common) > 30 and book_wo.shape[1] >= 2 and common.shape[1] > book_wo.shape[1]:
            base  = effective_number_of_bets(book_wo.loc[common.index].corr().values)
            withc = effective_number_of_bets(common.corr().values)
            eff_delta = float(withc - base)

    # ── 6. Regime-conditional gate (F7 fix; opt-in via CandidateSpec) ──
    regime_break: Optional[dict] = None
    worst_label:  Optional[str]  = None
    worst_sharpe: Optional[float] = None
    if spec.enable_regime_gate:
        regime_break, worst_label, worst_sharpe = _regime_breakdown(
            returns=r, annualization=ppy,
            min_obs=spec.regime_min_obs,
        )

    # ── Verdict logic (delegated to pure function for testability) ────────
    light, verdict_reasons = _decide_verdict(
        net_dsr=net_dsr, alpha_t=alpha_t, alpha_ann=alpha_ann,
        bench_used=bench_used, decay_verdict=decay.verdict,
        worst_regime_sharpe=worst_sharpe,
        worst_regime_label=worst_label,
        regime_floor=spec.regime_floor,
    )
    reasons.extend(verdict_reasons)

    return FactoryVerdict(
        name=spec.name, n_obs=len(r), deflated_sr=round(dsr, 3),
        residual_alpha_ann=round(alpha_ann, 4), residual_alpha_t=round(alpha_t, 2),
        benchmark_used=bench_used, net_deflated_sr=round(net_dsr, 3),
        recent_alive=decay.verdict, effective_bets_delta=eff_delta,
        light=light, reasons=tuple(reasons),
        regime_breakdown=regime_break,
        worst_regime=worst_label,
        worst_regime_sharpe=worst_sharpe,
    )


def _returns_hash(r: pd.Series) -> str:
    """Stable 16-char fingerprint of a return series (values + index). Lets the
    ledger detect that the SAME series was re-screened under changed
    assumptions."""
    rr = r.dropna().astype(float)
    b = pd.util.hash_pandas_object(rr, index=True).values.tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _ledger_prior(returns_hash: str, name: str) -> list[dict]:
    """Prior ledger entries matching this series fingerprint or name."""
    if not _LEDGER.exists():
        return []
    out = []
    for line in _LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("returns_hash") == returns_hash or rec.get("name") == name:
            out.append(rec)
    return out


def gate(spec: CandidateSpec, log: bool = True) -> FactoryVerdict:
    """Screen a candidate AND record it to the verdict ledger. This is the
    entry point a new factor must go through. Re-screening the same return
    series under a CHANGED assumption (n_trials / benchmark / cost_class)
    is flagged in the verdict reasons — making p-hacking-by-rerun visible
    rather than silent."""
    rhash = _returns_hash(spec.returns)
    prior = _ledger_prior(rhash, spec.name)
    v = screen_candidate(spec)

    # Anti-self-deception: same series, different assumptions => flag.
    changed = []
    for p in prior:
        if p.get("returns_hash") != rhash:
            continue
        for k in ("n_trials", "benchmark", "cost_class"):
            pv = p.get(k)
            cv = getattr(spec, k)
            if pv is not None and pv != cv:
                changed.append(f"{k}: {pv}->{cv}")
    if changed:
        flag = ("RE-SCREENED same series under changed assumptions ["
                + "; ".join(sorted(set(changed))) + f"] across {len(prior)} prior run(s)")
        v = FactoryVerdict(**{**v.__dict__, "reasons": v.reasons + (flag,)})

    if log:
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "name": spec.name, "light": v.light, "returns_hash": rhash,
            "frequency": spec.frequency, "benchmark": spec.benchmark,
            "cost_class": spec.cost_class, "n_trials": spec.n_trials,
            "annual_turnover": spec.annual_turnover, "already_net": spec.already_net,
            "n_obs": v.n_obs, "deflated_sr": v.deflated_sr,
            "net_deflated_sr": v.net_deflated_sr,
            "residual_alpha_ann": v.residual_alpha_ann,
            "residual_alpha_t": v.residual_alpha_t,
            "benchmark_used": v.benchmark_used, "recent_alive": v.recent_alive,
            "effective_bets_delta": v.effective_bets_delta,
            "n_prior_runs": len(prior), "reasons": list(v.reasons),
        }
        _LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with _LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # S1.C1 (2026-06-05) — shadow-emit into research_store so every
        # new verdict is visible to audit_verifier / direction_proposer /
        # graveyard_collision without a future manual backfill. NEVER
        # raises (helper has internal catch-all); a shadow failure does
        # not affect the primary ledger write above.
        try:
            from engine.research_store.shadow_emit import shadow_emit_factor_verdict
            shadow_emit_factor_verdict(rec, source="factory_ledger")
        except Exception:
            # Defense-in-depth: even if shadow_emit's own catch-all
            # somehow leaks, we explicitly swallow here so gate()
            # never raises for shadow reasons.
            pass
    return v


def render_table(verdicts: list[FactoryVerdict]) -> str:
    """One-table summary across candidates — the factory's headline output."""
    hdr = (f"{'CANDIDATE':<26}{'LIGHT':<8}{'defSR':>7}{'netSR':>7}"
           f"{'alpha%/yr':>11}{'t':>7}{'benchmark':>12}{'recent':>10}")
    rows = [hdr, "-" * len(hdr)]
    for v in verdicts:
        rec = "ALIVE" if ("DEAD" not in v.recent_alive and "WEAK" not in v.recent_alive) else "decayed"
        rows.append(
            f"{v.name[:25]:<26}{v.light:<8}{v.deflated_sr:>7.2f}{v.net_deflated_sr:>7.2f}"
            f"{v.residual_alpha_ann*100:>11.2f}{v.residual_alpha_t:>7.2f}"
            f"{v.benchmark_used[:11]:>12}{rec:>10}"
        )
    return "\n".join(rows)


def render_verdict(v: FactoryVerdict) -> str:
    lines = [
        f"[{v.light}] {v.name}  (n={v.n_obs})",
        f"  deflated SR        : {v.deflated_sr}  (net {v.net_deflated_sr})",
        f"  residual alpha     : {v.residual_alpha_ann*100:.2f}%/yr  "
        f"t={v.residual_alpha_t}  vs {v.benchmark_used}",
        f"  recent             : {v.recent_alive}",
    ]
    if v.effective_bets_delta is not None:
        lines.append(f"  eff-bets delta     : {v.effective_bets_delta:+.2f} (added to book)")
    lines.append(f"  reasons            : {'; '.join(v.reasons)}")
    return "\n".join(lines)
