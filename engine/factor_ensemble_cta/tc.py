"""
engine/factor_ensemble_cta/tc.py — Path O TC constant (spec id=73 §2.3).

LOCKED at register time; modifying requires spec_amend with academic rationale.
PQTIX is a mutual fund (I-class institutional, $1M min, no front-load).
Roundtrip TC = mutual fund redemption fee + bid-ask spread on creation/redemption
unit. 25 bp roundtrip is the institutional standard for mutual fund SAA rebalance.
"""
TC_BPS_PER_EVENT_LOCKED: float = 25.0
