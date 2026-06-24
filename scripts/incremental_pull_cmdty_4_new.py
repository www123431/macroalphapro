"""scripts/incremental_pull_cmdty_4_new.py — surgical WRDS pull for the 4 new
commodity clscodes (Wheat / Coffee / Sugar / Cotton) added to COMMODITIES in
spec 77 §11 amendment 2026-05-29.

We do NOT re-pull the existing 20. Instead:
  1. Read existing cached _CONTR + _PX (the 20-commodity universe)
  2. Query WRDS for ONLY the 4 new clscodes' contract metadata
  3. Query WRDS for ONLY the new futcodes' settlements (in chunks)
  4. Concat with existing cache, dedup, write back

Idempotent: re-running is a no-op when the new clscodes are already in cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.validation.commodity_carry import (
    _pg_engine, _CONTR, _PX, _PXDIR, COMMODITIES,
)

NEW_CLSCODES = [2442, 3289, 3299, 1487]  # Wheat / Coffee / Sugar / Cotton


def main():
    assert os.path.exists(_CONTR), f"Master contracts cache missing: {_CONTR}"
    assert os.path.exists(_PX), f"Master prices cache missing: {_PX}"

    contracts_old = pd.read_parquet(_CONTR)
    prices_old = pd.read_parquet(_PX)

    print(f"existing cache: {len(contracts_old)} contracts / {len(prices_old)} settle rows")
    print(f"existing clscodes in cache: {sorted(contracts_old['clscode'].unique().tolist())[:10]}...")

    missing = [c for c in NEW_CLSCODES if c not in contracts_old["clscode"].values]
    if not missing:
        print("All 4 new clscodes already in cache — nothing to do.")
        return

    print(f"to pull: {missing}")

    from sqlalchemy import text
    eng = _pg_engine()
    try:
        cls_in = ",".join(str(c) for c in missing)
        contracts_new = pd.read_sql(text(
            "select futcode, clscode, lasttrddate, startdate, contrname, isocurrcode "
            f"from tr_ds_fut.wrds_contract_info where clscode in ({cls_in}) "
            "and isocurrcode='USD' and lasttrddate is not null"), eng)
        contracts_new["lasttrddate"] = pd.to_datetime(contracts_new["lasttrddate"])
        print(f"  pulled {len(contracts_new)} new contracts")

        new_futs = contracts_new["futcode"].dropna().astype(int).unique().tolist()
        print(f"  → {len(new_futs)} unique futcodes to pull settlements for")

        # Pull settlements in chunks of 1000, append-only (use high chunk indices to
        # avoid clobbering existing files which use 0..N).
        os.makedirs(_PXDIR, exist_ok=True)
        existing_chunks = sorted([f for f in os.listdir(_PXDIR) if f.startswith("chunk_")])
        next_idx = len(existing_chunks)
        print(f"  existing chunk files: {len(existing_chunks)} → new chunks start at {next_idx:03d}")

        new_prices_pieces = []
        CH = 1000
        for i in range(0, len(new_futs), CH):
            cpath = f"{_PXDIR}/chunk_{next_idx:03d}.parquet"
            next_idx += 1
            if os.path.exists(cpath):
                print(f"  skip existing chunk: {cpath}")
                continue
            chunk = ",".join(str(f) for f in new_futs[i:i + CH])
            part = pd.read_sql(text(
                "select distinct futcode, date_, settlement from tr_ds_fut.wrds_fut_contract "
                f"where futcode in ({chunk}) and date_ >= '2000-01-01' "
                "and settlement is not null"), eng)
            part.to_parquet(cpath, index=False)
            new_prices_pieces.append(part)
            print(f"  pulled chunk {cpath}: {len(part)} rows")

    finally:
        eng.dispose()

    # Merge contracts
    contracts_merged = pd.concat([contracts_old, contracts_new], ignore_index=True)
    contracts_merged = contracts_merged.drop_duplicates(subset=["futcode"])
    contracts_merged.to_parquet(_CONTR, index=False)
    print(f"merged contracts: {len(contracts_merged)} → {_CONTR}")

    # Merge prices
    new_prices = pd.concat(new_prices_pieces, ignore_index=True) if new_prices_pieces else pd.DataFrame()
    if not new_prices.empty:
        new_prices["date_"] = pd.to_datetime(new_prices["date_"])
        prices_merged = pd.concat([prices_old, new_prices], ignore_index=True)
        prices_merged = prices_merged.drop_duplicates(subset=["futcode", "date_"])
        prices_merged.to_parquet(_PX, index=False)
        print(f"merged settlements: {len(prices_merged)} → {_PX}")

    # Sanity: did all 4 new clscodes survive into the merged contracts?
    surviving = sorted(set(contracts_merged["clscode"].unique().tolist()) & set(NEW_CLSCODES))
    print(f"NEW clscodes now in cache: {surviving} (expected {NEW_CLSCODES})")

    print("\nDONE.")


if __name__ == "__main__":
    main()
