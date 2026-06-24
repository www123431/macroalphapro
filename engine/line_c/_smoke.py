"""engine/line_c/_smoke.py — end-to-end PLUMBING smoke test on the partial text
snapshot (LM features only). NOT for conclusions: the partial snapshot is the
first-pulled transcript_ids (early-period biased). Validates that extraction ->
feature_panel (returns windowing, CAR, momentum, sector-neutralize) -> gate
(Fama-MacBeth + decile L/S + deflated Sharpe) all run clean before the full corpus.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import pandas as pd
from engine.line_c.text_features import extract_lm_from_parquet
from engine.line_c.feature_panel import build_panel
from engine.line_c.gate import run_gate

SNAP = "data/line_c/_text_snapshot.parquet"
lm = extract_lm_from_parquet(SNAP)   # streaming, memory-bounded
lm.to_parquet("data/line_c/_lm_features_partial.parquet")
print(f"LM features: {lm.shape}")

panel = build_panel(lm)
panel.to_parquet("data/line_c/_event_panel_partial.parquet")
rdq = pd.to_datetime(panel["rdq"])
print(f"\nPANEL: {panel.shape} | rdq {rdq.min().date()}->{rdq.max().date()} | permnos {panel['permno'].nunique()}")
for c in ["fwd_ret_21", "fwd_ret_63", "car_3d", "mom_12_1", "sue", "lm_net_tone", "lm_net_tone_sn"]:
    if c in panel.columns:
        print(f"  {c:16s} cov={panel[c].notna().mean()*100:5.1f}%  mean={panel[c].mean():+.5f}")

# sanity: SUE should predict fwd_ret (PEAD), CAR should correlate with SUE
d = panel[["sue", "car_3d", "fwd_ret_63"]].dropna()
print(f"\nsanity corr(SUE, CAR_3d)={d['sue'].corr(d['car_3d']):+.3f}  "
      f"corr(SUE, fwd63)={d['sue'].corr(d['fwd_ret_63']):+.3f}")

run_gate(panel, primary_feat="lm_net_tone_sn", target="fwd_ret_63", n_trials=24)
print("\nSMOKE OK — pipeline runs end to end.")
