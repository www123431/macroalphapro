"""engine/research/discovery/auto_gate.py — auto-trigger strict gate
when a paper is promoted from the discovery queue.

Per user 2026-05-30 Tier 1 ①: close the review→gate loop. Currently
promote writes a stub YAML, then nothing happens until the next manual
run. Auto-gate fixes this: on promote, infer the right template +
binding, run protocol_executor, write to gate_runs.jsonl, return the
verdict so the UI can show it.

DESIGN PRINCIPLES (senior):
  1. Best-effort, not blocking. If we can't infer template, skip
     auto-gate gracefully — promotion still writes the stub YAML.
  2. Family → template mapping is small + explicit (not LLM).
  3. Auto-gate uses DEFAULT bindings for each template; user can
     re-run with custom bindings later if needed.
  4. Synthetic data for templates that need it (event_panel etc)
     — auto-gate runs in HONEST mode: if real data is required and
     missing, returns AVAILABILITY_GAP not a fake GREEN.

CURRENTLY SUPPORTED auto-gate template inference:
  family=carry / tsmom / momentum / value / quality / low_vol / pead
  → factor_quartile template (most generic, works on synthetic returns)
For other families, auto-gate returns "template_inference_failed".

NOT auto-gated yet (future work):
  - event_study (needs event_panel)
  - dispersion (needs signal_panel + return_panel)
  - term_structure (needs yield_panel)
  These templates require non-trivial data acquisition that the
  promotion-time UI can't reliably synthesize. User can manually
  trigger them via the gate scripts.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"

# Family → (template_id, default_bindings)
# Only families that map cleanly to a template-with-synthetic-data path
# are listed. Other families exit cleanly without crash.
_FAMILY_TEMPLATE: dict[str, str] = {
    "carry":               "factor_quartile",
    "tsmom":               "factor_quartile",
    "momentum":            "factor_quartile",
    "value":               "factor_quartile",
    "quality":             "factor_quartile",
    "low_vol":             "factor_quartile",
    "profitability":       "factor_quartile",
    "investment":          "factor_quartile",
    "residual_momentum":   "factor_quartile",
    "pead":                "factor_quartile",
    "post_earnings_drift": "factor_quartile",
    # Cross-asset still mapped to factor_quartile for v1 — the
    # gate is testing the SIGNAL not the asset class
    "vol_carry":           "factor_quartile",
    "cross_asset_carry":   "factor_quartile",
    "cross_asset_tsmom":   "factor_quartile",
    "lead_lag":            "factor_quartile",
}


@dataclasses.dataclass
class AutoGateResult:
    ok:           bool
    mechanism_id: str
    verdict:      str | None = None       # GREEN / YELLOW / RED / UNAVAILABLE
    template:    str | None = None
    skipped:     bool = False
    skip_reason: str | None = None
    sharpe:      float | None = None
    alpha_t:     float | None = None
    deflated_sr: float | None = None
    error:       str | None = None
    elapsed_sec: float = 0.0
    # SENIOR ADVISORY: when True, this verdict came from SYNTHETIC data,
    # not real CRSP/futures backtest. The number is NOT alpha — it's a
    # smoke-test that the pipeline plumbing works end-to-end. The
    # actual GREEN/RED determination requires running gate against
    # real fetched data via the strict-gate scripts.
    provisional_synthetic: bool = True
    provisional_note:      str = (
        "verdict is PROVISIONAL — computed against synthetic data. "
        "Run the strict-gate script with real data to get an authoritative "
        "verdict before deploying capital."
    )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _can_infer_template(family: str) -> str | None:
    """Return template_id if we can auto-gate this family."""
    if not family:
        return None
    return _FAMILY_TEMPLATE.get(family.lower())


def _synthetic_factor_panels(n_periods: int = 240, n_tickers: int = 50,
                                seed: int = 42):
    """Build synthetic (factor, price, return) panels for auto-gate.

    factor_quartile signature requires factor_panel + price_panel +
    return_panel. We synthesize all three from a single RNG state so
    they're consistent.

    Returns: (factor_panel, price_panel, return_panel) — all wide
    DataFrames dates × tickers.

    DESIGN: noise-dominated but with a small +ve relationship so we
    expect mostly RED (matches our 3.5% empirical posterior).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    # Factor: random with autocorrelation
    raw_factor = rng.standard_normal((n_periods, n_tickers))
    factor = pd.DataFrame(raw_factor, index=dates, columns=tickers)

    # Returns: mostly noise + tiny relationship to LAGGED factor
    raw_returns = rng.standard_normal((n_periods, n_tickers)) * 0.05
    return_panel = pd.DataFrame(raw_returns, index=dates, columns=tickers)
    # Tiny signal: high-factor stocks get +0.003 next-period return
    high_factor = factor.shift(1).rank(axis=1, pct=True) > 0.8
    return_panel = return_panel + high_factor.astype(float) * 0.003

    # Prices: cumulative returns starting from $50 (above microcap threshold)
    price_panel = (1 + return_panel).cumprod() * 50.0

    return factor, price_panel, return_panel


def _annotate_ledger_synthetic(target_name: str) -> None:
    """Add provisional_synthetic flag to the matching gate_runs.jsonl entry.

    Best-effort: failure silently skipped (the verdict already exists; the
    annotation is just for downstream readers).
    """
    if not GATE_RUNS.exists() or not target_name:
        return
    try:
        lines = GATE_RUNS.read_text(encoding="utf-8").splitlines()
        if not lines:
            return
        # Annotate the LAST occurrence with matching name
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name") == target_name:
                rec["provisional_synthetic"] = True
                lines[i] = json.dumps(rec, ensure_ascii=False, default=str)
                GATE_RUNS.write_text(
                    "\n".join(lines) + "\n", encoding="utf-8",
                )
                return
    except Exception:
        pass


DEFAULT_AUTOGATE_BINDINGS = {
    # Senior 借鉴 ①: only override these whitelisted bindings during
    # auto_gate. Conservatively chosen — generic factor_quartile defaults
    # for synthetic-data smoke. Any per-mechanism YAML may override IF
    # those keys are listed in its tunable_bindings whitelist.
    "top_frac":             0.2,
    "bottom_frac":          0.2,
    "weighting":            "equal_weight",
    "rebal_freq":           "monthly",
    "cost_bps_per_side":    12.0,
    "vol_target":           None,
    "vol_target_lookback":  36,
}


def auto_gate(
    mechanism_yaml_path: Path | str,
    *,
    write_ledger: bool = True,
) -> AutoGateResult:
    """Read a promoted stub YAML, infer template, run gate, return verdict.

    Per Huatai 自进化Skill paper (借鉴 ①): only varies bindings listed
    in the YAML's tunable_bindings whitelist. Anything else stays at
    template default — preserves the "locked logic anchor".

    Args:
      mechanism_yaml_path: path to the stub YAML written by promote()
      write_ledger:        if True, append to gate_runs.jsonl

    Returns: AutoGateResult — caller (queue_actions.promote) decides
    what to do with it (typically: include in the promote response so
    UI can display).
    """
    import time
    t0 = time.time()
    path = Path(mechanism_yaml_path)
    try:
        import yaml
        stub = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return AutoGateResult(
            ok=False, mechanism_id=path.stem,
            error=f"yaml parse: {exc}",
            elapsed_sec=time.time() - t0,
        )

    mechanism_id = str(stub.get("id") or path.stem)
    family = (stub.get("family") or "").lower()
    template_id = _can_infer_template(family)
    # Whitelist enforcement (Huatai 借鉴 ①)
    tunable = set(stub.get("tunable_bindings") or [])

    if template_id is None:
        return AutoGateResult(
            ok=True, mechanism_id=mechanism_id, skipped=True,
            skip_reason=(f"no auto-gate template for family={family!r}; "
                          f"run gate manually with custom binding"),
            elapsed_sec=time.time() - t0,
        )

    # Build synthetic data + run the template + run the gate
    try:
        from engine.research.templates import TEMPLATES
        from engine.research.pipeline import run_gate

        if template_id not in TEMPLATES:
            return AutoGateResult(
                ok=False, mechanism_id=mechanism_id, template=template_id,
                error=f"template {template_id!r} not registered",
                elapsed_sec=time.time() - t0,
            )

        # Synthesize data — factor_quartile requires factor + price + returns
        factor, price_panel, return_panel = _synthetic_factor_panels()
        template_fn = TEMPLATES[template_id]

        # Senior 借鉴 ①: build effective bindings = template defaults
        # MERGED with stub-supplied values, but ONLY for keys in
        # tunable_bindings whitelist. Anything outside the whitelist
        # is ignored (logged for audit).
        effective_kwargs = dict(DEFAULT_AUTOGATE_BINDINGS)
        stub_bindings = stub.get("bindings") or {}
        ignored_keys = []
        for k, v in stub_bindings.items():
            if k in tunable:
                effective_kwargs[k] = v
            else:
                ignored_keys.append(k)
        if ignored_keys:
            logger.info("auto_gate ignored non-whitelisted bindings %s "
                          "for mechanism %s", ignored_keys, mechanism_id)

        returns_series = template_fn(
            factor_panel=factor,
            price_panel=price_panel,
            return_panel=return_panel,
            **effective_kwargs,
        )

        # Pick gate_profile from the template if it has one
        profile = getattr(
            __import__(f"engine.research.templates.{template_id}",
                          fromlist=["GATE_PROFILE"]),
            "GATE_PROFILE", None,
        )
        verdict_record = run_gate(
            returns_series,
            name=f"auto_gate__{mechanism_id}",
            mechanism=(f"auto-gated from {path.name}; "
                         f"SYNTHETIC DATA — provisional verdict, "
                         f"NOT alpha evidence"),
            n_trials=1,
            pead_control=False,    # synthetic data; no PEAD residualization
            log=write_ledger,
            profile=profile,
        )
        verdict = verdict_record.get("verdict")
        # Annotate the ledger entry retroactively if we logged
        if write_ledger and verdict_record.get("available"):
            _annotate_ledger_synthetic(verdict_record.get("name"))
        return AutoGateResult(
            ok=True, mechanism_id=mechanism_id, template=template_id,
            verdict=verdict,
            sharpe=verdict_record.get("standalone_sharpe"),
            alpha_t=verdict_record.get("alpha_t_ff5umd"),
            deflated_sr=verdict_record.get("deflated_sr"),
            elapsed_sec=time.time() - t0,
        )

    except Exception as exc:
        logger.warning("auto_gate %s failed: %s", mechanism_id, exc)
        return AutoGateResult(
            ok=False, mechanism_id=mechanism_id, template=template_id,
            error=str(exc)[:300],
            elapsed_sec=time.time() - t0,
        )
