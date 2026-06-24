"""
engine/d_pead_plus/feature_combiner.py — OLS feature combination + ranking.

Spec id=74 §2.6 LOCK:
  - Cross-section z-score each of 6 features (SUE + 5 LLM) within quarter
  - OLS (no regularization) on dev window 2024-Q2 to 2024-Q4
  - Target: 60-day forward log return
  - Coefficients FROZEN after dev fit; applied unchanged to OOS

DOCTRINE: This is a DECISION-LAYER module. Must NEVER import LLM SDKs or
engine.d_pead_plus.llm_extractor. Pure deterministic Python + numpy/pandas/sklearn.

Verified by engine.d_pead_plus.doctrine.audit_decision_layer_imports().
"""
from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Spec id=74 §2.6 LOCKED
FEATURE_COLUMNS_LOCKED: tuple[str, ...] = (
    "sue",
    "tone_score",
    "forward_confidence",
    "macro_headwind_flag",   # cast bool→int for OLS
    "evasion_score",
    "linguistic_complexity",
)
TARGET_COLUMN_LOCKED:   str = "ret_60d_log"
DEV_QUARTERS_LOCKED:    tuple[str, ...] = ("2024Q2", "2024Q3", "2024Q4")

# Persistence
CACHE_DIR              = Path("data/d_pead_plus")
COEFFICIENTS_PATH      = CACHE_DIR / "v1_dev_coefficients.json"


@dataclass(frozen=True)
class FrozenCoefficients:
    """OLS coefficients learned on dev set; FROZEN for OOS application."""
    intercept:              float
    sue:                    float
    tone_score:             float
    forward_confidence:     float
    macro_headwind_flag:    float
    evasion_score:          float
    linguistic_complexity:  float
    r_squared:              float
    n_obs_dev:              int
    dev_window:             str
    feature_means:          dict   # for OOS standardization consistency
    feature_stds:           dict
    fit_at_utc:             str


def _quarter_label(d: datetime.date) -> str:
    """e.g. datetime.date(2024, 5, 15) -> '2024Q2'"""
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """Winsorize at 1%/99% to limit outlier influence."""
    lo = series.quantile(lower_pct)
    hi = series.quantile(upper_pct)
    return series.clip(lower=lo, upper=hi)


def _cross_section_zscore(df: pd.DataFrame, feature_col: str, group_col: str) -> pd.Series:
    """Within each quarter, z-score feature across firms."""
    return df.groupby(group_col)[feature_col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )


def prepare_panel(
    sue_df:        pd.DataFrame,   # cols: permno, rdq, sue, mcap
    llm_df:        pd.DataFrame,   # cols: permno, rdq, tone_score, ...
    forward_ret_df: pd.DataFrame,  # cols: permno, rdq, ret_60d_log
) -> pd.DataFrame:
    """Merge SUE + LLM features + forward returns into one panel.

    Returns DataFrame keyed by (permno, rdq) with columns:
      permno, rdq, quarter, sue, tone_score, forward_confidence,
      macro_headwind_flag, evasion_score, linguistic_complexity, ret_60d_log,
      + cross-section z-scored versions
    """
    # Normalize rdq dtype across all 3 frames to datetime64 BEFORE merge
    # (LLM extraction stored rdq as object/datetime.date; sue_df is datetime64).
    for d in (sue_df, llm_df, forward_ret_df):
        d["rdq"] = pd.to_datetime(d["rdq"])

    df = sue_df.merge(llm_df, on=["permno", "rdq"], how="inner")
    df = df.merge(forward_ret_df, on=["permno", "rdq"], how="inner")

    # Cast bool feature to int for OLS
    if df["macro_headwind_flag"].dtype == bool:
        df["macro_headwind_flag"] = df["macro_headwind_flag"].astype(float)

    # Winsorize SUE at 1%/99% (Path D convention; cap at ±10 already done in cache)
    df["sue"] = _winsorize(df["sue"])

    # Add quarter label
    df["rdq"]     = pd.to_datetime(df["rdq"]).dt.date
    df["quarter"] = df["rdq"].apply(_quarter_label)

    # Cross-section z-score each feature within quarter
    for col in FEATURE_COLUMNS_LOCKED:
        df[f"{col}_z"] = _cross_section_zscore(df, col, "quarter")

    # Drop NaNs
    needed_cols = list(FEATURE_COLUMNS_LOCKED) + [f"{c}_z" for c in FEATURE_COLUMNS_LOCKED] + [TARGET_COLUMN_LOCKED]
    df = df.dropna(subset=needed_cols)
    logger.info("prepare_panel: %d firm-quarter rows after merge + dropna", len(df))
    return df


def fit_dev_ols(dev_panel: pd.DataFrame) -> FrozenCoefficients:
    """Fit OLS on dev panel; return frozen coefficients.

    OLS spec: y_60d = β₀ + β₁·sue_z + β₂·tone_z + ... + β₆·complexity_z

    No regularization (per spec id=74 §2.6 LOCK).
    """
    if dev_panel.empty:
        raise ValueError("fit_dev_ols: dev_panel is empty")

    # Verify dev quarters only
    actual_quarters = set(dev_panel["quarter"].unique())
    if not actual_quarters.issubset(set(DEV_QUARTERS_LOCKED)):
        logger.warning("fit_dev_ols: dev_panel contains non-dev quarters: %s",
                       actual_quarters - set(DEV_QUARTERS_LOCKED))

    # Build design matrix
    X_cols = [f"{c}_z" for c in FEATURE_COLUMNS_LOCKED]
    X = dev_panel[X_cols].values
    y = dev_panel[TARGET_COLUMN_LOCKED].values

    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])

    # OLS via normal equations: β = (X'X)^{-1} X'y
    beta, residuals, rank, sv = np.linalg.lstsq(X_with_intercept, y, rcond=None)

    # R²
    y_pred = X_with_intercept @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Feature standardization stats (for OOS consistency check)
    feat_means = {c: float(dev_panel[c].mean()) for c in FEATURE_COLUMNS_LOCKED}
    feat_stds  = {c: float(dev_panel[c].std())  for c in FEATURE_COLUMNS_LOCKED}

    coeffs = FrozenCoefficients(
        intercept              = float(beta[0]),
        sue                    = float(beta[1]),
        tone_score             = float(beta[2]),
        forward_confidence     = float(beta[3]),
        macro_headwind_flag    = float(beta[4]),
        evasion_score          = float(beta[5]),
        linguistic_complexity  = float(beta[6]),
        r_squared              = r_squared,
        n_obs_dev              = int(len(dev_panel)),
        dev_window             = ",".join(DEV_QUARTERS_LOCKED),
        feature_means          = feat_means,
        feature_stds           = feat_stds,
        fit_at_utc             = datetime.datetime.utcnow().isoformat() + "Z",
    )
    logger.info("OLS fit: R²=%.4f, n=%d, intercept=%+.5f, beta_sue=%+.5f, beta_tone=%+.5f",
                r_squared, len(dev_panel), coeffs.intercept, coeffs.sue, coeffs.tone_score)
    return coeffs


def save_coefficients(coeffs: FrozenCoefficients) -> None:
    """Persist frozen coefficients to JSON. FREEZE after this point."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(COEFFICIENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(asdict(coeffs), f, indent=2, ensure_ascii=False)
    logger.info("Coefficients FROZEN at %s", COEFFICIENTS_PATH)


def load_coefficients() -> Optional[FrozenCoefficients]:
    """Load frozen coefficients from JSON if exists."""
    if not COEFFICIENTS_PATH.exists():
        return None
    with open(COEFFICIENTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return FrozenCoefficients(**data)


def apply_frozen_coefficients_oos(
    oos_panel: pd.DataFrame,
    coeffs:    FrozenCoefficients,
) -> pd.DataFrame:
    """Apply frozen dev coefficients to OOS panel; return panel with score column.

    score = β₀ + Σᵢ βᵢ · feature_zᵢ
    Higher score → predicted higher forward return → long.
    """
    df = oos_panel.copy()
    z_cols = [f"{c}_z" for c in FEATURE_COLUMNS_LOCKED]
    if not all(c in df.columns for c in z_cols):
        raise ValueError(f"OOS panel missing z-scored columns: {z_cols}")

    df["score"] = (
        coeffs.intercept
        + coeffs.sue                   * df["sue_z"]
        + coeffs.tone_score            * df["tone_score_z"]
        + coeffs.forward_confidence    * df["forward_confidence_z"]
        + coeffs.macro_headwind_flag   * df["macro_headwind_flag_z"]
        + coeffs.evasion_score         * df["evasion_score_z"]
        + coeffs.linguistic_complexity * df["linguistic_complexity_z"]
    )

    # Cross-section rank within quarter (decile 1-10)
    df["decile"] = df.groupby("quarter")["score"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop") + 1
    )
    df["long_flag"]  = (df["decile"] == 10).astype(int)
    df["short_flag"] = (df["decile"] == 1).astype(int)
    return df


def get_locked_constants() -> dict:
    return {
        "FEATURE_COLUMNS_LOCKED": FEATURE_COLUMNS_LOCKED,
        "TARGET_COLUMN_LOCKED":   TARGET_COLUMN_LOCKED,
        "DEV_QUARTERS_LOCKED":    DEV_QUARTERS_LOCKED,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== D-PEAD-Plus Feature Combiner — Locked Constants ===")
    for k, v in get_locked_constants().items():
        print(f"  {k}: {v}")
    coeffs = load_coefficients()
    if coeffs:
        print(f"\nLoaded frozen coefficients: R²={coeffs.r_squared:.4f}, n={coeffs.n_obs_dev}")
        print(f"  sue={coeffs.sue:+.5f}  tone={coeffs.tone_score:+.5f}  conf={coeffs.forward_confidence:+.5f}")
        print(f"  headwind={coeffs.macro_headwind_flag:+.5f}  evasion={coeffs.evasion_score:+.5f}  complexity={coeffs.linguistic_complexity:+.5f}")
    else:
        print("\nNo frozen coefficients yet. Run fit_dev_ols(dev_panel) after extraction.")
