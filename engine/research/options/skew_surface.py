"""engine/research/options/skew_surface.py — adapter over OptionMetrics
IVOL skew surface caches.

DATA LIMITATION (2026-05-31, audited at build time):
  data/cache/_optionm_skew_surf.parquet:  ONLY delta=50 (ATM call)
                                          present in current cache
  data/cache/_optionm_put20_surf.parquet: put IV at delta=-20 only

  Full WRDS OptionMetrics surface has deltas {-90,-75,-50,-25,-10,
  10,25,50,75,90} × maturities {30,60,91,122,152,182,273,365,547,730}
  but only the 2 cached fields above are populated for this project.

  For Path C put-spread tail hedge we need OTM put IV at delta ~ -30
  (~5% OTM) and delta ~ -15 (~10-15% OTM). The cached delta=-20
  (put20) IS reasonable for ONE leg but the other leg requires
  interpolation OR re-fetch from WRDS.

  TODO (next session): scripts/fetch_optionm_full_skew_surface.py to
  pull the full surface for SPY (secid=8957) + other index ETFs.
  Until then, this adapter SURFACES the limitation explicitly via
  the AvailableDeltas enum + raises DataNotAvailableError for
  unsupported queries.

Adapter API (intended; some methods raise NotImplementedError until
full skew data is fetched):

  loader = SkewSurfaceLoader.from_cache()
  iv_atm = loader.get_iv(secid=8957, date=date(2024,1,2),
                          delta=50, cp_flag="C", maturity_days=30)
  iv_put20 = loader.get_iv(secid=8957, date=date(2024,1,2),
                            delta=-20, cp_flag="P", maturity_days=30)
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_SKEW_PATH = REPO_ROOT / "data" / "cache" / "_optionm_full_skew.parquet"
SKEW_PATH = REPO_ROOT / "data" / "cache" / "_optionm_skew_surf.parquet"
PUT20_PATH = REPO_ROOT / "data" / "cache" / "_optionm_put20_surf.parquet"
SECID_PERMNO_PATH = REPO_ROOT / "data" / "cache" / "_optionm_secid_permno.parquet"


class AvailableDeltas(IntEnum):
    """Deltas currently materialized in the local OptionMetrics cache.
    Extend this enum when wrds_skew_fetch ships."""

    ATM_CALL_DELTA50 = 50    # _optionm_skew_surf with cp_flag=C, delta=50
    PUT_DELTA_MINUS_20 = -20 # _optionm_put20_surf (encoded as positive in name)


class DataNotAvailableError(LookupError):
    """Raised when a (secid, date, delta, maturity) query is outside the
    cached data — caller should either re-fetch from WRDS or use a
    parametric skew model."""


@dataclass
class SkewSurfaceLoader:
    """Lazy loader over the cached IV surfaces.

    Caches DataFrames on first access; subsequent get_iv() calls are
    fast dict lookups via the pre-indexed pivot tables.
    """

    _full_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _skew_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _put20_df: Optional[pd.DataFrame] = field(default=None, init=False)
    _atm_pivot: Optional[pd.DataFrame] = field(default=None, init=False)
    _put20_pivot: Optional[pd.DataFrame] = field(default=None, init=False)
    _full_pivot: Optional[pd.DataFrame] = field(default=None, init=False)

    @classmethod
    def from_cache(cls) -> "SkewSurfaceLoader":
        return cls()

    # ── lazy loaders ────────────────────────────────────────────────

    def _load_full(self) -> pd.DataFrame:
        if self._full_df is None:
            if not FULL_SKEW_PATH.exists():
                raise FileNotFoundError(
                    f"full skew cache missing: {FULL_SKEW_PATH}. "
                    f"Run scripts/fetch_optionm_full_skew_surface.py"
                )
            df = pd.read_parquet(FULL_SKEW_PATH)
            df["date"] = pd.to_datetime(df["date"])
            df["secid"] = df["secid"].astype(int)
            df["delta"] = df["delta"].astype(int)
            df["days"] = df["days"].astype(int)
            self._full_df = df
        return self._full_df

    def _load_skew(self) -> pd.DataFrame:
        if self._skew_df is None:
            if not SKEW_PATH.exists():
                raise FileNotFoundError(f"cache missing: {SKEW_PATH}")
            df = pd.read_parquet(SKEW_PATH)
            df["date"] = pd.to_datetime(df["date"])
            self._skew_df = df
        return self._skew_df

    def _load_put20(self) -> pd.DataFrame:
        if self._put20_df is None:
            if not PUT20_PATH.exists():
                raise FileNotFoundError(f"cache missing: {PUT20_PATH}")
            df = pd.read_parquet(PUT20_PATH)
            df["date"] = pd.to_datetime(df["date"])
            self._put20_df = df
        return self._put20_df

    def _build_atm_pivot(self) -> pd.DataFrame:
        if self._atm_pivot is None:
            df = self._load_skew()
            atm = df[(df["cp_flag"] == "C") & (df["delta"] == 50.0)]
            self._atm_pivot = atm.pivot_table(
                index="date", columns="secid", values="impl_volatility",
            )
        return self._atm_pivot

    def _build_put20_pivot(self) -> pd.DataFrame:
        if self._put20_pivot is None:
            df = self._load_put20()
            self._put20_pivot = df.pivot_table(
                index="date", columns="secid", values="putiv",
            )
        return self._put20_pivot

    # ── public API ──────────────────────────────────────────────────

    def list_available_secids(self, source: Literal["atm", "put20"] = "atm") -> list[int]:
        pivot = self._build_atm_pivot() if source == "atm" else self._build_put20_pivot()
        return [int(s) for s in pivot.columns]

    def get_iv(
        self,
        *,
        secid: int,
        date: _dt.date | _dt.datetime | pd.Timestamp,
        delta: int,
        cp_flag: Literal["C", "P"],
        maturity_days: int = 30,
    ) -> float:
        """Return IV at (secid, date, delta, cp_flag, maturity_days).

        Currently supported queries (raises DataNotAvailableError
        otherwise):
          - cp_flag='C' AND delta=50 AND maturity_days=30 (ATM call)
          - cp_flag='P' AND delta=-20 AND maturity_days=30 (put20)

        maturity_days is accepted but the cached data is a single-tenor
        surface (likely 30-day standardized). Other maturities raise.
        """
        ts = pd.Timestamp(date)

        # ── Try FULL skew cache first (preferred, post-2026-05-31 fetch) ──
        if FULL_SKEW_PATH.exists():
            full = self._load_full()
            avail_deltas = sorted(full["delta"].unique())
            avail_days = sorted(full["days"].unique())
            avail_cp = set(full["cp_flag"].unique())
            if cp_flag not in avail_cp:
                raise DataNotAvailableError(
                    f"cp_flag={cp_flag!r} not in full skew cache (have {avail_cp})"
                )
            if delta not in avail_deltas:
                raise DataNotAvailableError(
                    f"delta={delta} not in full skew cache (have {avail_deltas}). "
                    f"For non-grid deltas use linear interpolation: see "
                    f"get_iv_interpolated()"
                )
            if maturity_days not in avail_days:
                raise DataNotAvailableError(
                    f"maturity_days={maturity_days} not in cache (have {avail_days})"
                )
            mask = ((full["secid"] == secid) & (full["cp_flag"] == cp_flag)
                    & (full["delta"] == delta) & (full["days"] == maturity_days)
                    & (full["date"] <= ts))
            sub = full[mask]
            if sub.empty:
                raise DataNotAvailableError(
                    f"no IV for secid={secid} cp={cp_flag} delta={delta} "
                    f"days={maturity_days} on or before {ts.date()}"
                )
            return float(sub.iloc[-1]["impl_volatility"])

        # ── Fallback to legacy thin caches if full not fetched yet ──
        if maturity_days != 30:
            raise DataNotAvailableError(
                f"only maturity_days=30 in legacy cache; got {maturity_days}. "
                f"Run scripts/fetch_optionm_full_skew_surface.py for full surface."
            )
        if cp_flag == "C" and delta == 50:
            pivot = self._build_atm_pivot()
            if secid not in pivot.columns:
                raise DataNotAvailableError(
                    f"secid {secid} not in ATM cache (have {len(pivot.columns)} secids)"
                )
            series = pivot[secid].dropna()
            available = series[series.index <= ts]
            if available.empty:
                raise DataNotAvailableError(
                    f"no ATM IV for secid {secid} on or before {ts.date()}"
                )
            return float(available.iloc[-1])

        if cp_flag == "P" and delta == -20:
            pivot = self._build_put20_pivot()
            if secid not in pivot.columns:
                raise DataNotAvailableError(
                    f"secid {secid} not in put20 cache"
                )
            series = pivot[secid].dropna()
            available = series[series.index <= ts]
            if available.empty:
                raise DataNotAvailableError(
                    f"no put20 IV for secid {secid} on or before {ts.date()}"
                )
            return float(available.iloc[-1])

        raise DataNotAvailableError(
            f"(cp_flag={cp_flag}, delta={delta}) not in current cache. "
            f"Run scripts/fetch_optionm_full_skew_surface.py for full surface."
        )

    def get_iv_interpolated(
        self,
        *,
        secid: int,
        date: _dt.date | _dt.datetime | pd.Timestamp,
        target_delta: float,
        cp_flag: Literal["C", "P"],
        maturity_days: int = 30,
    ) -> float:
        """Linear-interpolate IV across delta grid for non-standard deltas
        (e.g. delta=-30 for ~5% OTM put spread leg). Requires full skew
        cache; raises DataNotAvailableError otherwise.

        Interpolation is linear in delta within the same maturity tenor.
        For deltas outside the grid (e.g. delta=-95) returns NaN rather
        than extrapolating.
        """
        if not FULL_SKEW_PATH.exists():
            raise DataNotAvailableError(
                "full skew cache required for interpolation; "
                "run scripts/fetch_optionm_full_skew_surface.py"
            )
        full = self._load_full()
        ts = pd.Timestamp(date)
        # Filter to (secid, cp_flag, maturity, date <= ts) and pick the
        # latest date
        mask = ((full["secid"] == secid) & (full["cp_flag"] == cp_flag)
                & (full["days"] == maturity_days) & (full["date"] <= ts))
        sub = full[mask]
        if sub.empty:
            raise DataNotAvailableError(
                f"no surface for secid={secid} cp={cp_flag} days={maturity_days}"
                f" on or before {ts.date()}"
            )
        latest_date = sub["date"].max()
        latest = sub[sub["date"] == latest_date].sort_values("delta")
        deltas_avail = latest["delta"].values
        ivs = latest["impl_volatility"].values
        if target_delta < deltas_avail.min() or target_delta > deltas_avail.max():
            return float("nan")    # don't extrapolate
        import numpy as _np
        return float(_np.interp(target_delta, deltas_avail, ivs))

    def get_secid_for_permno(self, permno: int) -> Optional[int]:
        """Resolve a CRSP permno → OptionMetrics secid via cached map."""
        if not SECID_PERMNO_PATH.exists():
            return None
        df = pd.read_parquet(SECID_PERMNO_PATH)
        m = df[df["permno"] == permno]
        if m.empty:
            return None
        return int(m["secid"].iloc[0])
