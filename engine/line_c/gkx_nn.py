"""engine/line_c/gkx_nn.py — Line C ②: GKX-style NN cross-sectional return prediction.

Gu-Kelly-Xiu (2020, RFS) "Empirical Asset Pricing via Machine Learning": shallow
feed-forward NN ensembles on cross-sectionally-ranked firm features, strict OOS,
geometric-pyramid hidden layers, dropout + L2 + batchnorm + early stopping +
multi-seed ensembling (NN variance reduction).

DISCIPLINE (financial DL overfits notoriously; per campaign memory):
  - train 2014-2017, EARLY-STOP on val 2018, predict OOS 2019-2024 (never tuned on test)
  - architecture/seed search counted in deflated-Sharpe n_trials
  - audit: drop log_mcap (the ml_ensemble found size was ~half the HistGBM edge)
HONEST EXPECTATION (memory): features are arbitraged -> NN won't beat PEAD; this is
a capability demonstration of the full DL-for-asset-pricing stack held to the same
gate (deflated Sharpe + audit), NOT a hidden-alpha hunt.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engine.validation.deflated_sharpe import deflated_sharpe_ratio

logger = logging.getLogger(__name__)

PANEL = "data/cache/_ml_feature_panel.parquet"
FEATURES = ["mom_12_1", "rev_1m", "vol_6m", "sue", "log_mcap", "gp",
            "asset_growth", "bm", "iv_atm", "iv_skew", "news_ess"]
ARCHS = {"NN2": [16, 8], "NN3": [32, 16, 8], "NN4": [32, 16, 8, 4]}


def _load_splits(feats=FEATURES):
    df = pd.read_parquet(PANEL)
    df["month"] = pd.to_datetime(df["month"])
    df = df.dropna(subset=["y"]).copy()
    df[feats] = df[feats].fillna(0.0)             # centered-rank neutral
    tr = df[df["month"] <= "2017-12-31"]
    va = df[(df["month"] >= "2018-01-01") & (df["month"] <= "2018-12-31")]
    te = df[df["month"] >= "2019-01-01"]
    return tr, va, te


def _train_one(Xtr, ytr, Xva, yva, hidden, seed, device, max_epochs=60, patience=6):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    layers, d = [], Xtr.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        d = h
    layers += [nn.Linear(d, 1)]
    model = nn.Sequential(*layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.01)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device).view(-1, 1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device).view(-1, 1)
    n = len(Xtr_t); bs = 8192
    best_val, best_state, wait = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva_t), yva_t).item()
        if v < best_val - 1e-7:
            best_val, best_state, wait = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _ensemble_predict(tr, va, te, feats, hidden, k_seeds=5):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = tr[feats].values.astype(np.float32), tr["y"].values.astype(np.float32)
    Xva, yva = va[feats].values.astype(np.float32), va["y"].values.astype(np.float32)
    Xte = te[feats].values.astype(np.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    preds = np.zeros(len(te))
    for s in range(k_seeds):
        m = _train_one(Xtr, ytr, Xva, yva, hidden, seed=s, device=device)
        with torch.no_grad():
            preds += m(Xte_t).cpu().numpy().ravel()
    return preds / k_seeds


def _ls_series(te: pd.DataFrame, pred: np.ndarray, n_bins=10) -> pd.Series:
    d = te[["month", "y"]].copy(); d["pred"] = pred
    rows = []
    for m, g in d.groupby("month"):
        if len(g) < n_bins * 5:
            continue
        b = pd.qcut(g["pred"].rank(method="first"), n_bins, labels=False)
        rows.append((m, g.loc[b == n_bins - 1, "y"].mean() - g.loc[b == 0, "y"].mean()))
    return pd.Series(dict(rows)).sort_index()


def _report(name, ls, n_trials):
    ls = ls.dropna()
    sr = (ls.mean() / ls.std()) * np.sqrt(12) if ls.std() > 0 else np.nan
    dsr = deflated_sharpe_ratio(ls.values, n_trials=n_trials, periods_per_year=12)
    print(f"  {name:22s} n={len(ls):>3d}  meanM_bps={ls.mean()*1e4:>7.1f}  "
          f"SR_ann={sr:>6.2f}  deflSR={dsr.deflated_sr:>6.3f}")
    return {"n": int(len(ls)), "sharpe_ann": float(sr), "deflated_sr": float(dsr.deflated_sr)}


def run(n_trials=20):
    tr, va, te = _load_splits()
    print(f"GKX-NN splits: train={len(tr)} val={len(va)} test={len(te)} "
          f"({te['month'].dt.to_period('M').nunique()} OOS months)")
    out = {}

    # baseline: SUE-only cross-sectional sort
    out["baseline_sue"] = _report("baseline SUE-sort", _ls_series(te, te["sue"].values), n_trials)

    # architecture sensitivity (each = 5-seed ensemble)
    for name, hid in ARCHS.items():
        pred = _ensemble_predict(tr, va, te, FEATURES, hid)
        out[name] = _report(f"{name} {hid}", _ls_series(te, pred), n_trials)

    # AUDIT: NN3 without log_mcap (size-tilt check — ml_ensemble: HistGBM 1.34->0.76)
    no_size = [f for f in FEATURES if f != "log_mcap"]
    pred_ns = _ensemble_predict(tr, va, te, no_size, ARCHS["NN3"])
    out["NN3_no_size"] = _report("NN3 no-log_mcap", _ls_series(te.assign(), pred_ns), n_trials)

    # AUDIT: subperiod on NN3
    pred_n3 = _ensemble_predict(tr, va, te, FEATURES, ARCHS["NN3"])
    for lab, msk in [("NN3 2019-21", te["month"] <= "2021-12-31"), ("NN3 2022-24", te["month"] >= "2022-01-01")]:
        sub = te[msk]
        out[lab] = _report(lab, _ls_series(sub, pred_n3[msk.values]), n_trials)
    return out


if __name__ == "__main__":
    import warnings, json
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    res = run()
    print("\n=== JSON ===")
    print(json.dumps(res, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x))
