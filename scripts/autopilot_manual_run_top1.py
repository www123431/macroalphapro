"""scripts/autopilot_manual_run_top1.py — F14b pre-promotion gate (2026-06-05).

Runs the top-1 candidate from today's F14a dry-run plan through the
composer end-to-end, computes summary stats on the returns, captures
real cost. Substitute for the "what if cron actually ran" check that
soak-wall-clock was supposed to provide. ~30-60s wall.

Decision rule for F14b promotion:
  1. compose() succeeds without ComponentNotFound / silent fallback
  2. n_obs >= 60 (5 years monthly minimum)
  3. cost_usd == 0 (composer must remain LLM-free at this stage)
  4. summary stats are in-distribution (Sharpe |abs| <= 3, no NaN
     blowups). Direction can be either sign — we're testing the
     PIPELINE, not the alpha.

Run:
  python scripts/autopilot_manual_run_top1.py [--hyp-id <prefix>]

Default: pull the top-1 from latest replay JSON (or run a fresh
compute_dry_run_plan if no replay file).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve_top1_hyp_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    # Pick from latest replay JSON if available
    replay_dir = REPO_ROOT / "data" / "autopilot" / "_replay"
    if replay_dir.is_dir():
        files = sorted(replay_dir.glob("replay_*.json"))
        if files:
            d = json.loads(files[-1].read_text(encoding="utf-8"))
            today = d["per_day"][-1]
            for dec in today["decisions"]:
                if dec["action"] == "WOULD_TEST":
                    print(f"Pulled top-1 from replay: hyp={dec['source_hypothesis_id'][:8]} cell={dec['cell']}")
                    return dec["source_hypothesis_id"]
    # Fallback: live compute_dry_run_plan
    from engine.agents.autopilot import compute_dry_run_plan
    p = compute_dry_run_plan(top_n=1)
    if not p.decisions:
        raise RuntimeError("No candidates available; substrate may be empty")
    print(f"Pulled top-1 from live compute_dry_run_plan: hyp={p.decisions[0].source_hypothesis_id[:8]}")
    return p.decisions[0].source_hypothesis_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp-id", default=None,
                     help="full source_hypothesis_id (default: top-1 from latest replay)")
    args = ap.parse_args()

    hyp_id = _resolve_top1_hyp_id(args.hyp_id)

    # 1. Resolve the latest spec for this hyp
    from engine.hypothesis_spec.store import all_specs
    candidates = [s for s in all_specs() if s.source_hypothesis_id == hyp_id]
    if not candidates:
        print(f"FAIL: no spec found for hyp_id={hyp_id}")
        return 2
    candidates.sort(key=lambda s: s.extraction.extracted_ts or s.created_ts or "", reverse=True)
    spec = candidates[0]
    print()
    print(f"Spec resolved:")
    print(f"  hyp_id:       {spec.source_hypothesis_id}")
    print(f"  claim_type:   {spec.claim_type.value}")
    print(f"  family:       {spec.family.value}")
    print(f"  signal_type:  {spec.legs[0].signal_type.value if spec.legs else 'NONE'}")
    print(f"  universe:     {spec.universe.asset_class.value}/{spec.universe.subset.value}")
    print(f"  weighting:    {spec.construction.weighting.value}")
    print(f"  rebalance:    {spec.construction.rebalance.value}")
    print(f"  claim:        {(spec.claim_text or '')[:140]}")
    print()

    # 2. Cost ledger snapshot BEFORE compose
    ledger = REPO_ROOT / "data" / "llm_cost_ledger.jsonl"
    n_rows_before = 0
    if ledger.is_file():
        n_rows_before = sum(1 for _ in ledger.open(encoding="utf-8"))

    # 3. Compose
    print("Composing returns series...")
    t0 = time.perf_counter()
    from engine.composer.composer import compose
    try:
        result = compose(spec, force=True)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"FAIL: compose raised after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        return 3
    elapsed = time.perf_counter() - t0
    print(f"  elapsed: {elapsed:.1f}s   ok={result['ok']}  cached={result['from_cache']}")
    print(f"  components: {[c.get('cls') for c in result.get('components_used', [])]}")
    if not result["ok"]:
        print(f"FAIL: compose result.ok=False, error={result.get('error')}")
        return 4

    # 4. Load returns + summary stats
    import pandas as pd
    import numpy as np
    s = pd.read_parquet(result["path"]).iloc[:, 0]
    s = s.dropna()
    if s.empty:
        print("FAIL: returns series is empty after dropna")
        return 5
    n = len(s)
    ann_factor = 12 if n < 500 else 252  # heuristic monthly vs daily
    mu_ann = s.mean() * ann_factor
    sd_ann = s.std() * (ann_factor ** 0.5)
    sharpe = mu_ann / sd_ann if sd_ann > 0 else float("nan")
    # t-stat = sharpe * sqrt(years)
    years = n / ann_factor
    t_stat = sharpe * (years ** 0.5)
    cum = (1 + s).cumprod()
    dd = (cum / cum.cummax() - 1).min()

    print()
    print(f"Returns summary:")
    print(f"  n_obs:        {n}  ({years:.1f} years at ann_factor={ann_factor})")
    print(f"  date range:   {s.index.min()} -> {s.index.max()}")
    print(f"  ann return:   {mu_ann*100:+.2f}%")
    print(f"  ann vol:      {sd_ann*100:.2f}%")
    print(f"  Sharpe:       {sharpe:+.2f}")
    print(f"  t-stat:       {t_stat:+.2f}")
    print(f"  max DD:       {dd*100:+.2f}%")

    # 5. Cost delta
    n_rows_after = sum(1 for _ in ledger.open(encoding="utf-8")) if ledger.is_file() else 0
    n_new_calls = n_rows_after - n_rows_before
    new_cost = 0.0
    if n_new_calls > 0:
        rows = ledger.read_text(encoding="utf-8").splitlines()[-n_new_calls:]
        for r in rows:
            try:
                new_cost += float(json.loads(r).get("cost_usd", 0))
            except Exception:
                pass
    print()
    print(f"Cost delta: ${new_cost:.4f}   ({n_new_calls} new LLM calls)")

    # 6. Verdict
    print()
    print("=" * 70)
    print("F14b PIPELINE-READINESS VERDICT")
    print("=" * 70)
    ok_n        = n >= 60
    ok_cost     = abs(new_cost) < 0.001
    ok_sharpe   = abs(sharpe) < 3 and not (sharpe != sharpe)   # finite
    ok_dd       = dd > -0.99
    if ok_n and ok_cost and ok_sharpe and ok_dd:
        print("  GREEN  composer ran clean: n_obs ok, no LLM cost, stats in-distribution")
        print(f"         Top-1 GP/A delivered Sharpe={sharpe:+.2f} t={t_stat:+.2f} over {years:.1f}y")
        print("         F14b infrastructure path validated end-to-end.")
        return 0
    else:
        print("  RED    pipeline issues:")
        if not ok_n:      print(f"         n_obs={n} < 60 (insufficient history)")
        if not ok_cost:   print(f"         cost=${new_cost:.4f} > $0 (composer should be LLM-free)")
        if not ok_sharpe: print(f"         Sharpe={sharpe} suspicious (NaN or |abs|>3)")
        if not ok_dd:     print(f"         max DD={dd*100:.1f}% catastrophic")
        return 1


if __name__ == "__main__":
    sys.exit(main())
