"""engine.factor_regression — FF5+MOM factor regression for sleeves.

For each sleeve / combined-book return series, regress excess return
on Fama-French 5 + UMD (momentum) and report Newey-West HAC t-stats.

This is the standard "is your alpha real or just risk premium" test
that any senior PM / hedge-fund interviewer expects. Without this,
"Sharpe 1.0" doesn't distinguish alpha from beta-loading on known
factors.

Usage:
    from engine.factor_regression import (
        run_ff5_mom_regression, fetch_ff5_mom_weekly,
    )
    factors = fetch_ff5_mom_weekly()
    result = run_ff5_mom_regression(
        sleeve_returns=my_book["combined_return"],
        factor_returns=factors,
        sleeve_label="combined_5sleeve",
    )
    print(result.alpha_annualized, result.alpha_tstat_NW,
          result.alpha_t_clears_HLZ)
"""
from engine.factor_regression.ken_french import (
    fetch_ff5_mom_daily, fetch_ff5_mom_weekly,
)
from engine.factor_regression.regression import (
    FACTORS_FF5_MOM, FactorRegression, run_ff5_mom_regression,
)

__all__ = [
    "fetch_ff5_mom_daily", "fetch_ff5_mom_weekly",
    "FACTORS_FF5_MOM", "FactorRegression", "run_ff5_mom_regression",
]
