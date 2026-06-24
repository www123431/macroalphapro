"""engine/validation/factor_exposure.py — cross-asset factor exposure (5 macro betas).

A pure equity model (FF5) can't explain a multi-asset book (equity + bonds + carry + CTA + gold).
This regresses the book's daily return on five liquid ETF-proxy macro factors that SPAN the asset
classes the book trades, and decomposes book variance onto them:

  equity     SPY
  rates      TLT                 (long-duration Treasury)
  credit     HYG − LQD           (HY minus IG: strips the common rate move, isolates credit spread)
  commodity  DBC
  dollar     UUP

Output per factor: β (sensitivity) + % of book variance it explains (factor risk share, βᵀΣβ
decomposition); plus R² (total explained) and the idiosyncratic remainder (1 − R²). It surfaces the
book's TRUE factor footprint — e.g. low equity, meaningful rates/commodity/dollar from the carry/AC
sleeves — which FF5 alone hides.

Honest limits: ETF proxies (not academic factors); full-sample betas (~1y daily); factors are
mildly collinear (the credit = HY−IG construction reduces the rate overlap). In-sample / no
look-ahead. This is a Barra-LITE lens, not a full risk model (no constructed cross-asset
carry/momentum/trend style factors).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# factor → how to build its return series from panel columns. ("diff", a, b) = ret[a] - ret[b].
_FACTORS = [
    ("equity", "SPY"),
    ("rates", "TLT"),
    ("credit", ("diff", "HYG", "LQD")),
    ("commodity", "DBC"),
    ("dollar", "UUP"),
]


def compute_factor_exposure(weights: dict, panel: pd.DataFrame) -> dict:
    """Regress the combined book on the 5 macro-proxy factors; return betas + factor risk shares."""
    if panel is None or panel.empty:
        return {"available": False, "reason": "no returns panel"}
    cols = [t for t, w in weights.items() if t in panel.columns and abs(w) > 1e-9]
    if len(cols) < 2:
        return {"available": False, "reason": "fewer than 2 holdings have return history"}

    # book daily return series over the panel (current weights relived on history)
    R = panel[cols].fillna(0.0).tail(252 * 2)
    w = pd.Series({t: weights[t] for t in cols})
    book = R.mul(w, axis=1).sum(axis=1)

    # build the factor matrix on the same dates
    fac = {}
    missing = []
    for name, spec in _FACTORS:
        if isinstance(spec, tuple):                 # ("diff", a, b)
            _, a, b = spec
            if a in panel.columns and b in panel.columns:
                fac[name] = (panel[a] - panel[b]).reindex(book.index)
            else:
                missing.append(name)
        elif spec in panel.columns:
            fac[name] = panel[spec].reindex(book.index)
        else:
            missing.append(name)
    if len(fac) < 2:
        return {"available": False, "reason": f"factor proxies missing: {missing}"}

    F = pd.DataFrame(fac).reindex(book.index).fillna(0.0)
    names = list(F.columns)
    X = np.column_stack([np.ones(len(F))] + [F[n].values for n in names])   # intercept + factors
    y = book.values
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = float(beta_full[0])
    betas = {names[i]: float(beta_full[i + 1]) for i in range(len(names))}

    y_hat = X @ beta_full
    var_y = float(np.var(y, ddof=1))
    r2 = float(1.0 - np.var(y - y_hat, ddof=1) / var_y) if var_y > 0 else 0.0

    # factor risk shares: var(Xβ_factors) = bᵀ Σ_F b ; share_k = b_k (Σ_F b)_k / var(y) (sum = R²-ish)
    b = np.array([betas[n] for n in names])
    cov_f = np.cov(F[names].values, rowvar=False)
    cov_f = np.atleast_2d(cov_f)
    sig_b = cov_f @ b
    shares = (b * sig_b) / var_y if var_y > 0 else np.zeros_like(b)

    factors = [{
        "factor": names[i], "beta": round(betas[names[i]], 4),
        "risk_share": round(float(shares[i]), 4),
    } for i in range(len(names))]
    factors.sort(key=lambda f: -abs(f["risk_share"]))

    return {
        "available": True,
        "n_obs": int(len(book)),
        "period": [str(book.index[0])[:10], str(book.index[-1])[:10]],
        "r2": round(r2, 4),
        "idiosyncratic": round(max(0.0, 1.0 - r2), 4),
        "alpha_daily": round(alpha, 6),
        "factors": factors,
        "proxies": {"equity": "SPY", "rates": "TLT", "credit": "HYG−LQD", "commodity": "DBC", "dollar": "UUP"},
        "note": "ETF-proxy macro factors; β = sensitivity, risk_share = % of book variance; "
                "idiosyncratic = unexplained (1−R²). Barra-lite, not a full risk model.",
    }
