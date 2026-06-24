"""engine/line_c/emb_content.py — Line C① completeness: FinBERT CONTENT embeddings.

Tests whether the FULL 768-d FinBERT content representation (not just tone) predicts
returns INCREMENTALLY over SUE. Disciplined against the 768-d/short-time-series
overfit trap:
  - PCA fit on TRAIN period only (<=2017), applied OOS (no look-ahead);
  - K=20 PCs -> small MLP (5-seed ensemble, dropout+L2+early-stop), train<=2017,
    predict OOS 2018-2023 (never tuned on OOS);
  - OOS NN prediction = 'content_score' -> SAME gate: Fama-MacBeth incremental t +
    residualized decile L/S + deflated Sharpe (honest n_trials).
A 768-d in-sample fit WOULD manufacture a fake signal; this design avoids it.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PANEL = "data/line_c/_event_panel_full.parquet"
EMB = "data/line_c/_finbert_emb_full.parquet"
K_PCS = 20
TARGET = "fwd_ret_63"
CONTROLS = ["sue", "car_3d", "mom_12_1", "log_size"]


def _train_nn(Xtr, ytr, Xva, yva, seed, device, hidden=(32, 16), max_epochs=60, patience=6):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]; d = h
    layers += [nn.Linear(d, 1)]
    m = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.HuberLoss(delta=0.01)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device).view(-1, 1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device).view(-1, 1)
    n, bs = len(Xtr_t), 4096
    best, best_state, wait = 1e9, None, 0
    for ep in range(max_epochs):
        m.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; opt.zero_grad()
            lf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = lf(m(Xva_t), yva_t).item()
        if v < best - 1e-7:
            best, best_state, wait = v, {k: t.detach().clone() for k, t in m.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m


def run():
    from sklearn.decomposition import PCA
    import torch
    from engine.line_c.gate import fama_macbeth, decile_ls, sharpe_block, add_residual

    panel = pd.read_parquet(PANEL)
    emb = pd.read_parquet(EMB)
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    df = panel.merge(emb, on="transcript_id", how="inner")
    df["rdq"] = pd.to_datetime(df["rdq"])
    df = df.dropna(subset=[TARGET] + CONTROLS).reset_index(drop=True)
    yr = df["rdq"].dt.year
    tr = df[yr <= 2017].copy()
    oos = df[yr >= 2018].copy()
    print(f"content test: train={len(tr)} (<=2017), OOS={len(oos)} (>=2018), emb_dim={len(emb_cols)}")

    # standardize emb on TRAIN, PCA on TRAIN, transform both (no look-ahead)
    mu, sd = tr[emb_cols].mean().values, tr[emb_cols].std().replace(0, 1).values
    Xtr_raw = (tr[emb_cols].values - mu) / sd
    Xoos_raw = (oos[emb_cols].values - mu) / sd
    pca = PCA(n_components=K_PCS, random_state=0).fit(Xtr_raw)
    Ptr, Poos = pca.transform(Xtr_raw), pca.transform(Xoos_raw)
    print(f"PCA {K_PCS} PCs explain {pca.explained_variance_ratio_.sum()*100:.1f}% of train emb variance")

    # train/val split within train (val = 2017 for early stopping)
    vmask = (tr["rdq"].dt.year == 2017).values
    ytr = np.clip(tr[TARGET].values, *np.percentile(tr[TARGET].values, [1, 99]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preds = np.zeros(len(oos))
    for s in range(5):
        m = _train_nn(Ptr[~vmask], ytr[~vmask], Ptr[vmask], ytr[vmask], seed=s, device=device)
        with torch.no_grad():
            preds += m(torch.tensor(Poos, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    oos["content_score"] = preds / 5

    n_trials = 24  # PCA K + NN arch/seed search, on top of the campaign breadth
    print("\n===== LENS A — Fama-MacBeth on OOS content_score (NN ensemble) =====")
    r = fama_macbeth(oos, "content_score", TARGET)
    for spec in ("uni", "ctrl"):
        rr = r.get(spec, {})
        print(f"  content_score {spec:5s}: n_q={rr.get('n_q',0)} coef_bps={rr.get('mean_bps',float('nan')):.2f} t={rr.get('t',float('nan')):.2f}")

    print("\n===== LENS B — decile L/S + Deflated Sharpe (OOS) =====")
    oos = add_residual(oos, "content_score")
    for f in ["content_score", "content_score_resid"]:
        for tgt in [TARGET, "fwd_ret_21"]:
            sb = sharpe_block(decile_ls(oos, f, tgt), n_trials=n_trials)
            print(f"  {f:22s} {tgt:11s} n={sb['n']:>3d} meanQ_bps={sb.get('mean_q_bps',float('nan')):>8.2f} "
                  f"SR={sb.get('sharpe_ann',float('nan')):>6.2f} deflSR={sb.get('deflated_sr',float('nan')):.3f}")

    # raw-PC incremental scan: how many single PCs have ctrl |t|>2 (multiple-testing context)
    print("\n===== PC-level incremental scan (ctrl |t|>2 of 20 PCs) =====")
    for k in range(K_PCS):
        oos[f"pc{k}"] = Poos[:, k]
    sig = []
    for k in range(K_PCS):
        t = fama_macbeth(oos, f"pc{k}", TARGET).get("ctrl", {}).get("t", float("nan"))
        if abs(t) > 2:
            sig.append((k, round(t, 2)))
    print(f"  PCs with incremental |t|>2: {sig if sig else 'NONE'}  (expected ~1 by chance at |t|>2 over 20)")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    run()
