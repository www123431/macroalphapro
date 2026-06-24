"""engine/line_c/run_full.py — full-corpus LM stage: extract -> panel -> gate.

Fast (~10 min) full-period (2011-2024, unbiased) verdict on whether the
Loughran-McDonald dictionary tone adds INCREMENTAL value over SUE. FinBERT tone
(slower, ~3.5h) is merged in a second pass via run_finbert.py + merge_and_gate.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import pandas as pd
from engine.line_c.text_features import extract_lm_from_parquet
from engine.line_c.feature_panel import build_panel
from engine.line_c.gate import run_gate

TEXT = "data/line_c/_transcripts_text_2011_2024.parquet"

lm = extract_lm_from_parquet(TEXT)
lm.to_parquet("data/line_c/_lm_features_full.parquet")
print(f"LM full features: {lm.shape}")

panel = build_panel(lm)
panel.to_parquet("data/line_c/_event_panel_lm.parquet")
rdq = pd.to_datetime(panel["rdq"])
print(f"\nPANEL: {panel.shape} | rdq {rdq.min().date()}->{rdq.max().date()} | permnos {panel['permno'].nunique()}")
for c in ["fwd_ret_21", "fwd_ret_63", "car_3d", "mom_12_1", "sue", "lm_net_tone"]:
    if c in panel.columns:
        print(f"  {c:14s} cov={panel[c].notna().mean()*100:5.1f}% mean={panel[c].mean():+.5f}")

run_gate(panel, primary_feat="lm_net_tone_sn", target="fwd_ret_63")
print("\nFULL LM-stage done.")
