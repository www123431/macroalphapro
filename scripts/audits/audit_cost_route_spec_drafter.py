"""scripts/audit_cost_route_spec_drafter.py — R1 cost-route A/B audit.

Question: can we re-route spec_drafter workload from Anthropic Claude
(~$0.011/call) to Deepseek (~$0.002/call) without quality loss?
Re-route would save ~80% per backfill (~$2 → ~$0.40 for 200 specs).

Method: pick N random hypothesis claims from the corpus, run extract_spec
TWICE per claim (once via Claude, once via Deepseek), compare outputs
field-by-field, compute agreement rates + cost delta. Decision rule:

  agreement on claim_type AND family >= 0.90 → re-route OK, commit
  otherwise                                  → stay on Claude, log gap

Run:
  python scripts/audit_cost_route_spec_drafter.py [--n 20]

Output:
  - stdout: per-field agreement table + cost summary
  - data/cost_route_audit/<ts>.jsonl: per-claim raw side-by-side
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUT_DIR = REPO_ROOT / "data" / "cost_route_audit"


def _summarize_spec(spec) -> dict:
    """Pull the comparison-relevant fields out of a HypothesisSpec
    (or None on extractor failure)."""
    if spec is None:
        return {"ok": False}
    legs = spec.legs or ()
    primary = legs[0] if legs else None
    return {
        "ok":          True,
        "claim_type":  spec.claim_type.value,
        "family":      spec.family.value,
        "asset_class": spec.universe.asset_class.value,
        "subset":      spec.universe.subset.value,
        "weighting":   spec.construction.weighting.value,
        "rebalance":   spec.construction.rebalance.value,
        "direction":   spec.outcome.predicted_direction.value,
        "signal_type": (primary.signal_type.value if primary else "UNKNOWN"),
        "confidence":  spec.extraction.confidence,
    }


def _agreement(field: str, claude: list[dict], deepseek: list[dict]) -> tuple[int, int]:
    """Returns (n_match, n_compared) over the claims where both
    providers returned non-None specs."""
    n_match = 0
    n_compared = 0
    for c, d in zip(claude, deepseek):
        if not (c.get("ok") and d.get("ok")):
            continue
        n_compared += 1
        if c.get(field) == d.get(field):
            n_match += 1
    return n_match, n_compared


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="Number of hypothesis claims to sample (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from engine.research_store.hypothesis import load_hypotheses
    from engine.hypothesis_spec.extractor import extract_spec
    from engine.research_store.manifest import current_git_sha

    # Sample: stratify by mechanism_family so we don't accidentally
    # all-pick from CARRY (which dominates the corpus).
    all_hyps = load_hypotheses()
    latest_by_id = {}
    for h in all_hyps:
        prior = latest_by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            latest_by_id[h.hypothesis_id] = h

    by_family: dict[str, list] = {}
    for h in latest_by_id.values():
        by_family.setdefault(h.mechanism_family.value, []).append(h)

    rng = random.Random(args.seed)
    sample = []
    per_family_target = max(1, args.n // max(1, len(by_family)))
    for fam, lst in by_family.items():
        rng.shuffle(lst)
        sample.extend(lst[:per_family_target])
    rng.shuffle(sample)
    sample = sample[:args.n]
    print(f"Sample: {len(sample)} hypotheses across {len({h.mechanism_family.value for h in sample})} families")

    git_sha = current_git_sha() or ""

    # Run both providers per claim
    claude_results: list[dict] = []
    deepseek_results: list[dict] = []
    claude_total_cost = 0.0
    deepseek_total_cost = 0.0
    n_deepseek_hard_fail = 0

    for i, h in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}]  {h.hypothesis_id[:8]}  family={h.mechanism_family.value:<22}", end=" ", flush=True)
        # CLAUDE side
        spec_c = extract_spec(
            source_hypothesis_id = h.hypothesis_id,
            claim_text           = h.claim,
            mechanism_family     = h.mechanism_family.value,
            mechanism_subtype    = h.mechanism_subtype,
            git_sha              = git_sha,
            workload_override    = "spec_drafter",
        )
        c_summary = _summarize_spec(spec_c)
        # DEEPSEEK side
        try:
            spec_d = extract_spec(
                source_hypothesis_id = h.hypothesis_id,
                claim_text           = h.claim,
                mechanism_family     = h.mechanism_family.value,
                mechanism_subtype    = h.mechanism_subtype,
                git_sha              = git_sha,
                workload_override    = "spec_drafter_deepseek",
            )
        except Exception as exc:
            logger.warning("Deepseek call failed for %s: %s", h.hypothesis_id, exc)
            spec_d = None
            n_deepseek_hard_fail += 1
        d_summary = _summarize_spec(spec_d)
        c_summary["hypothesis_id"] = h.hypothesis_id
        d_summary["hypothesis_id"] = h.hypothesis_id
        claude_results.append(c_summary)
        deepseek_results.append(d_summary)
        print(f"C={c_summary.get('claim_type', 'FAIL'):<18} D={d_summary.get('claim_type', 'FAIL'):<18}")

    # Cost from ledger (read what we just spent)
    from pathlib import Path as _P
    ledger = _P("data/llm_cost_ledger.jsonl")
    if ledger.is_file():
        all_rows = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Take the last N*2 rows assuming they are this audit's
        recent = all_rows[-(len(sample) * 2):]
        for r in recent:
            if r.get("provider") == "anthropic":
                claude_total_cost += float(r.get("cost_usd", 0))
            elif r.get("provider") == "deepseek":
                deepseek_total_cost += float(r.get("cost_usd", 0))

    # Compute agreement
    print()
    print("=" * 70)
    print(f"AGREEMENT RATES (Claude vs Deepseek over {len(sample)} claims)")
    print("=" * 70)
    fields = ["claim_type", "family", "asset_class", "subset",
              "weighting", "rebalance", "direction", "signal_type"]
    for fld in fields:
        m, total = _agreement(fld, claude_results, deepseek_results)
        pct = (100 * m / total) if total else 0.0
        # ASCII only — Windows default GBK console codec barfs on ✓/✗
        marker = "OK " if pct >= 90 else ("~  " if pct >= 75 else "BAD")
        print(f"  {marker}  {fld:<14}  {m:>2}/{total:<2}  {pct:.0f}%")

    print()
    print("Output:")
    n_both_ok = sum(1 for c, d in zip(claude_results, deepseek_results)
                     if c.get("ok") and d.get("ok"))
    n_c_only = sum(1 for c, d in zip(claude_results, deepseek_results)
                     if c.get("ok") and not d.get("ok"))
    n_d_only = sum(1 for c, d in zip(claude_results, deepseek_results)
                     if not c.get("ok") and d.get("ok"))
    n_both_fail = sum(1 for c, d in zip(claude_results, deepseek_results)
                       if not c.get("ok") and not d.get("ok"))
    print(f"  both providers returned spec: {n_both_ok}")
    print(f"  Claude only / Deepseek only / both fail: {n_c_only} / {n_d_only} / {n_both_fail}")
    print(f"  Deepseek hard exception count: {n_deepseek_hard_fail}")
    print()
    print("Cost (this audit):")
    print(f"  Claude   total: ${claude_total_cost:.4f}")
    print(f"  Deepseek total: ${deepseek_total_cost:.4f}")
    savings = 0.0
    if claude_total_cost > 0:
        savings = 100 * (1 - deepseek_total_cost / claude_total_cost)
        print(f"  Deepseek vs Claude: {savings:.0f}% cost reduction")

    # Persist raw side-by-side
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = _OUT_DIR / f"audit_{ts}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c, d in zip(claude_results, deepseek_results):
            f.write(json.dumps({"hypothesis_id": c.get("hypothesis_id"),
                                  "claude": c, "deepseek": d},
                                 ensure_ascii=False) + "\n")
    print()
    print(f"Raw side-by-side: {out_path}")

    # Decision
    m_ct, t_ct = _agreement("claim_type", claude_results, deepseek_results)
    m_fa, t_fa = _agreement("family", claude_results, deepseek_results)
    ct_pct = 100 * m_ct / max(1, t_ct)
    fa_pct = 100 * m_fa / max(1, t_fa)
    print()
    print("=" * 70)
    print("DECISION")
    print("=" * 70)
    decision_pass = ct_pct >= 90 and fa_pct >= 90
    if decision_pass:
        print(f"  PASS: claim_type {ct_pct:.0f}% + family {fa_pct:.0f}% both >= 90%")
        print(f"  Recommendation: re-route spec_drafter -> deepseek-v4-pro")
        print(f"  Est savings: ~{savings:.0f}% per future backfill")
    else:
        print(f"  FAIL: claim_type {ct_pct:.0f}% / family {fa_pct:.0f}% < 90%")
        print(f"  Recommendation: KEEP spec_drafter on Claude")
        print(f"  Risk of re-route: classification drift in {100-min(ct_pct, fa_pct):.0f}% of corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
