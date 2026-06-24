"""engine/line_c/merge_gate.py — definitive verdict: LM + FinBERT -> panel -> gate.

Merges dictionary + FinBERT tone features, adds prior-call cosine similarity from
FinBERT embeddings (Lazy-Prices-style call-over-call CHANGE), builds the full
event panel, and runs both lenses + the audit battery on the full 2011-2024 set.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import numpy as np
import pandas as pd

from engine.line_c.feature_panel import build_panel, IDX_PATH
from engine.line_c.gate import run_gate

LM = "data/line_c/_lm_features_full.parquet"
FB = "data/line_c/_finbert_features_full.parquet"
EMB = "data/line_c/_finbert_emb_full.parquet"
PANEL_OUT = "data/line_c/_event_panel_full.parquet"


def prior_call_cosine() -> pd.DataFrame:
    """Cosine similarity of each call's FinBERT embedding to the SAME firm's prior
    call (low similarity = more new/different content; Lazy-Prices analog)."""
    import os
    if not os.path.exists(EMB):
        return pd.DataFrame(columns=["transcript_id", "prior_call_cosine"])
    emb = pd.read_parquet(EMB)
    idx = pd.read_parquet(IDX_PATH)[["transcript_id", "permno", "rdq"]]
    idx["rdq"] = pd.to_datetime(idx["rdq"])
    d = idx.merge(emb, on="transcript_id", how="inner").sort_values(["permno", "rdq"])
    emb_cols = [c for c in d.columns if c.startswith("emb_")]
    M = d[emb_cols].values.astype(np.float32)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    permno = d["permno"].values
    cos = np.full(len(d), np.nan)
    for i in range(1, len(d)):
        if permno[i] == permno[i - 1]:
            cos[i] = float(Mn[i] @ Mn[i - 1])
    d["prior_call_cosine"] = cos
    return d[["transcript_id", "prior_call_cosine"]]


if __name__ == "__main__":
    lm = pd.read_parquet(LM)
    fb = pd.read_parquet(FB)
    feats = lm.merge(fb, on="transcript_id", how="inner")
    pcc = prior_call_cosine()
    if not pcc.empty:
        feats = feats.merge(pcc, on="transcript_id", how="left")
    print(f"merged text features: {feats.shape}")

    panel = build_panel(feats)
    panel.to_parquet(PANEL_OUT)
    rdq = pd.to_datetime(panel["rdq"])
    print(f"\nFULL PANEL: {panel.shape} | {rdq.min().date()}->{rdq.max().date()} | permnos {panel['permno'].nunique()}")

    print("\n########## PRIMARY: FinBERT tone (sector-neutral) ##########")
    run_gate(panel, primary_feat="finbert_tone_sn", target="fwd_ret_63")
