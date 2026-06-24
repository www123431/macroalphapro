"""scripts/fetch_anchor_library.py — Tier C L2-4 Commit 1.

Download Ken French Data Library's published monthly risk-factor
series (FF-5 + Momentum) and standardize them as a single parquet
under `data/anchor_library/famafrench_monthly.parquet` for use as
ground-truth anchors in L2-4 residual-alpha regression.

Why Ken French data (not our own constructions):
  - Reproducibility: same numbers every quant in the world uses.
    AQR, HXZ, Two-Sigma, every reviewer cross-checks against this.
  - Decouples L2-4 anchor regression bugs from our backtest engine
    bugs — if our gp_at PnL has residual alpha vs French RMW = 0,
    the answer is unambiguous.
  - Free, small (~5MB total), CC-licensed for academic use.

Output schema (parquet columns):
  date         month-end DatetimeIndex (column, NOT index, for arrow)
  MKT_RF       market excess return (Mkt-RF)
  SMB          size factor (Small Minus Big)
  HML          value factor (High Minus Low book-to-market)
  RMW          profitability (Robust Minus Weak)  ← GP/A's parent axis
  CMA          investment (Conservative Minus Aggressive)
  RF           risk-free rate (1-mo T-bill)
  MOM          momentum (UMD / Carhart)

All values stored as DECIMAL (NOT percent). 0.0051 = 0.51% monthly.

Provenance:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
    Data_Library/f-f_5factors_2x3.html
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/
    Data_Library/f-f_momentum.html

Idempotent: --force re-downloads + overwrites; default skips if
the parquet exists.
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "data" / "anchor_library"
OUT_PATH  = OUT_DIR / "famafrench_monthly.parquet"
INDUSTRY_OUT_PATH = OUT_DIR / "industries_12_monthly.parquet"

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Ken French publishes everything via stable mirror URLs. These have
# been stable for 15+ years; if they break, manual download from
# the Data Library is the recovery path.
FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
            "ftp/F-F_Momentum_Factor_CSV.zip")

# Ken French CSVs ship as zips. Inside each zip is exactly one CSV
# whose filename matches the zip stem.
FF5_INNER_NAME = "F-F_Research_Data_5_Factors_2x3.csv"
MOM_INNER_NAME = "F-F_Momentum_Factor.csv"

# L2-6 lite (2026-06-09): Ken French 12-industry monthly portfolios
# for industry-attribution regression. 12 industries balance
# granularity and statistical power for monthly data.
INDUSTRY_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                  "ftp/12_Industry_Portfolios_CSV.zip")
INDUSTRY_INNER_NAME = "12_Industry_Portfolios.csv"
INDUSTRY_COLUMNS = (
    "NoDur",   # consumer non-durables
    "Durbl",   # consumer durables
    "Manuf",   # manufacturing
    "Enrgy",   # energy
    "Chems",   # chemicals
    "BusEq",   # business equipment / tech
    "Telcm",   # telecom
    "Utils",   # utilities
    "Shops",   # retail
    "Hlth",    # healthcare
    "Money",   # finance
    "Other",   # everything else
)


def _http_get_zip(url: str, *, timeout: float = 60.0) -> bytes:
    """Download a zip file with a polite User-Agent (Ken French's
    Dartmouth mirror sometimes 403s unidentified scrapers)."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": ("MacroAlphaPro/0.1 anchor-library "
                                  "fetcher; academic research")},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_csv_text(zip_bytes: bytes, inner_name: str) -> str:
    """Pull the inner CSV text out of a Ken French zip."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        # Be tolerant of small filename drift (case / spacing).
        match = None
        for n in names:
            if n.strip().lower() == inner_name.lower():
                match = n
                break
        if match is None:
            raise FileNotFoundError(
                f"expected {inner_name!r} in zip, got {names!r}"
            )
        with zf.open(match) as f:
            return f.read().decode("latin-1")  # FF uses latin-1


def _parse_french_monthly(
    csv_text: str,
    *,
    expected_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Parse a Ken French CSV into a clean monthly DataFrame.

    Ken French CSVs have a specific quirky shape:
      - N rows of header text + blank lines
      - one row of column names
      - the monthly data section (YYYYMM int dates)
      - blank line(s)
      - "Annual Factors" header + annual data section
      - sometimes a trailing copyright line

    We:
      1. Find the line whose tokens match expected_columns — that's
         the column-name row. Lines before it are header noise.
      2. Read everything after until we hit a blank line, a non-6-digit
         "date", or an "Annual Factors" string.
      3. Parse YYYYMM → month-end Timestamp.
      4. Divide all factor columns by 100 (FF publishes as percent).
    """
    lines = csv_text.splitlines()

    # Find header row.
    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        tokens = [t.strip() for t in line.split(",")]
        non_empty_tokens = [t for t in tokens if t]
        if set(expected_columns).issubset(set(non_empty_tokens)):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(
            f"could not find header row containing {expected_columns!r} "
            f"in first {len(lines)} lines"
        )

    # Collect data rows until end-of-monthly-section.
    data_rows: list[str] = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            # First blank line = end of monthly section (annual follows)
            if data_rows:
                break
            continue
        # "Annual Factors" / "Annual" header → end
        if any(stripped.lower().startswith(prefix) for prefix in
                ("annual", "yearly")):
            break
        # First token should be a YYYYMM int for monthly section.
        first = stripped.split(",", 1)[0].strip()
        if not (first.isdigit() and len(first) == 6):
            # Once we've started collecting and hit a non-date,
            # the monthly section is over.
            if data_rows:
                break
            continue
        data_rows.append(line)

    if not data_rows:
        raise ValueError("no monthly data rows parsed")

    # Re-assemble CSV text + read. skipinitialspace handles the
    # FF convention of padding numeric values with leading spaces
    # ("192701,   0.57" instead of "192701,0.57").
    csv_subset = ",".join(["date"] + list(expected_columns)) + "\n" \
        + "\n".join(data_rows)
    df = pd.read_csv(io.StringIO(csv_subset), skipinitialspace=True)

    # YYYYMM → month-end Timestamp
    df["date"] = pd.to_datetime(
        df["date"].astype(str), format="%Y%m"
    ) + pd.offsets.MonthEnd(0)

    # Percent → decimal (FF convention)
    for c in expected_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_famafrench_monthly() -> pd.DataFrame:
    """Pull FF5 + Momentum, merge on date, return a single
    monthly DataFrame with the canonical 7 columns."""
    logger.info("downloading FF-5 from %s", FF5_URL)
    ff5_zip = _http_get_zip(FF5_URL)
    ff5_csv = _extract_csv_text(ff5_zip, FF5_INNER_NAME)
    ff5_df  = _parse_french_monthly(
        ff5_csv,
        expected_columns=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    )
    ff5_df = ff5_df.rename(columns={"Mkt-RF": "MKT_RF"})
    logger.info("  FF5 parsed: %d months (%s → %s)",
                  len(ff5_df), ff5_df["date"].min().date(),
                  ff5_df["date"].max().date())

    logger.info("downloading Momentum from %s", MOM_URL)
    mom_zip = _http_get_zip(MOM_URL)
    mom_csv = _extract_csv_text(mom_zip, MOM_INNER_NAME)
    mom_df  = _parse_french_monthly(
        mom_csv,
        expected_columns=("Mom",),
    )
    mom_df = mom_df.rename(columns={"Mom": "MOM"})
    logger.info("  MOM parsed: %d months (%s → %s)",
                  len(mom_df), mom_df["date"].min().date(),
                  mom_df["date"].max().date())

    # Inner-join on date (MOM goes back further than FF5 in some years
    # but we want the intersection where ALL anchor factors are
    # defined — clean overlap window).
    merged = pd.merge(ff5_df, mom_df, on="date", how="inner")

    # Reorder columns for stable schema
    merged = merged[["date", "MKT_RF", "SMB", "HML", "RMW", "CMA",
                       "RF", "MOM"]]
    logger.info("merged: %d months (%s → %s), %d columns",
                  len(merged), merged["date"].min().date(),
                  merged["date"].max().date(),
                  len(merged.columns))
    return merged


def fetch_industries_12_monthly() -> pd.DataFrame:
    """Pull Ken French 12-Industry monthly value-weighted returns
    and return as month-end indexed DataFrame.

    The 12-Industry file contains 4 separate panels (Average VW
    Returns, Average EW Returns, Number of Firms, Avg Firm Size).
    We parse only the FIRST panel (VW returns) which is the
    institutional standard for industry attribution.
    """
    logger.info("downloading 12-Industry from %s", INDUSTRY_URL)
    zip_bytes = _http_get_zip(INDUSTRY_URL)
    csv_text  = _extract_csv_text(zip_bytes, INDUSTRY_INNER_NAME)
    df = _parse_french_monthly(
        csv_text, expected_columns=INDUSTRY_COLUMNS,
    )
    logger.info("  12-Industry parsed: %d months (%s → %s)",
                  len(df), df["date"].min().date(),
                  df["date"].max().date())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                          help="re-download even if parquet exists")
    parser.add_argument("--skip-ff5", action="store_true",
                          help="skip the FF5+MOM fetcher; only fetch industries")
    parser.add_argument("--skip-industries", action="store_true",
                          help="skip the 12-industry fetcher; only fetch FF5+MOM")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── FF5 + MOM ────────────────────────────────────────────────────
    if not args.skip_ff5:
        if OUT_PATH.exists() and not args.force:
            existing = pd.read_parquet(OUT_PATH)
            logger.info("FF5+MOM library already cached at %s "
                          "(%d months %s → %s); use --force to refresh",
                          OUT_PATH, len(existing),
                          existing["date"].min().date(),
                          existing["date"].max().date())
        else:
            df = fetch_famafrench_monthly()
            df.to_parquet(OUT_PATH, index=False)
            size_kb = OUT_PATH.stat().st_size / 1024
            logger.info("wrote %s (%.1f KB)", OUT_PATH, size_kb)
            print()
            print("=== Anchor library — FF5 + Momentum (monthly) ===")
            print(f"Path:    {OUT_PATH.relative_to(REPO_ROOT)}")
            print(f"Range:   {df['date'].min().date()} → "
                  f"{df['date'].max().date()}")
            print(f"Months:  {len(df)}")
            print(f"Columns: {list(df.columns)}")

    # ── 12-Industry (L2-6 lite, 2026-06-09) ──────────────────────────
    if not args.skip_industries:
        if INDUSTRY_OUT_PATH.exists() and not args.force:
            existing = pd.read_parquet(INDUSTRY_OUT_PATH)
            logger.info("12-Industry library already cached at %s "
                          "(%d months %s → %s); use --force to refresh",
                          INDUSTRY_OUT_PATH, len(existing),
                          existing["date"].min().date(),
                          existing["date"].max().date())
        else:
            df_ind = fetch_industries_12_monthly()
            df_ind.to_parquet(INDUSTRY_OUT_PATH, index=False)
            size_kb = INDUSTRY_OUT_PATH.stat().st_size / 1024
            logger.info("wrote %s (%.1f KB)", INDUSTRY_OUT_PATH, size_kb)
            print()
            print("=== Anchor library — Ken French 12-Industry (monthly) ===")
            print(f"Path:    {INDUSTRY_OUT_PATH.relative_to(REPO_ROOT)}")
            print(f"Range:   {df_ind['date'].min().date()} → "
                  f"{df_ind['date'].max().date()}")
            print(f"Months:  {len(df_ind)}")
            print(f"Industries: {list(df_ind.columns[1:])}")


if __name__ == "__main__":
    sys.exit(main())
