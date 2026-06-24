"""tests/test_fetch_anchor_library.py — Tier C L2-4 Commit 1.

Offline tests for the Ken French CSV parser in
scripts/fetch_anchor_library.py. Network downloads themselves are
NOT tested here (those are integration / manual). The parsing is
the fragile part — Ken French's CSV format has been quirky for
decades and we hit two surprises live during commit 1 build:
  - Header row format: `,Mom` not `Mom   ` (whitespace mismatch)
  - Numeric values padded with leading spaces ("   0.57")

These tests pin both of those + edge cases.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
import fetch_anchor_library as fal  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# _parse_french_monthly — happy paths
# ────────────────────────────────────────────────────────────────────
_FF5_FIXTURE = """\
This file was created from CRSP 202604.
Some preamble text that should be skipped.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,-0.39,-0.43,-0.94,0.61,-1.32,0.27
196308,5.07,-0.86,1.83,0.27,-0.40,0.25
196309,-1.57,-0.49,0.16,0.05,0.16,0.27

 Annual Factors: January-December

,Mkt-RF,SMB,HML,RMW,CMA,RF
1963,9.93,-7.21,8.96,4.43,-1.27,3.18

Copyright 2026 Kenneth R. French
"""


_MOM_FIXTURE = """\
This file was created using the 202604 CRSP database.
Momentum factor description...

,Mom
192701,   0.57
192702,   0.36
192703,   0.16

 Annual Factors:

,Mom
1927,21.43
"""


def test_parse_ff5_monthly_happy():
    df = fal._parse_french_monthly(
        _FF5_FIXTURE,
        expected_columns=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    )
    # Exactly 3 monthly rows (annual section excluded)
    assert len(df) == 3
    assert list(df.columns) == ["date", "Mkt-RF", "SMB", "HML",
                                  "RMW", "CMA", "RF"]
    # Date conversion to month-end
    assert df["date"].iloc[0] == pd.Timestamp("1963-07-31")
    assert df["date"].iloc[1] == pd.Timestamp("1963-08-31")
    assert df["date"].iloc[2] == pd.Timestamp("1963-09-30")
    # Percent → decimal (FF -0.39% → -0.0039)
    assert df["Mkt-RF"].iloc[0] == pytest.approx(-0.0039)
    assert df["Mkt-RF"].iloc[1] == pytest.approx(0.0507)
    assert df["RMW"].iloc[0]    == pytest.approx(0.0061)


def test_parse_momentum_handles_leading_whitespace():
    """Regression: live FF momentum file pads numerics with spaces
    (`   0.57` not `0.57`). Parser must handle via skipinitialspace."""
    df = fal._parse_french_monthly(
        _MOM_FIXTURE,
        expected_columns=("Mom",),
    )
    assert len(df) == 3
    assert df["date"].iloc[0] == pd.Timestamp("1927-01-31")
    assert df["Mom"].iloc[0] == pytest.approx(0.0057)
    assert df["Mom"].iloc[2] == pytest.approx(0.0016)


def test_parse_strips_annual_section():
    """The "Annual Factors" section must NOT bleed into the monthly
    DataFrame. Annual section has 4-digit dates (1963) which would
    parse incorrectly as YYYYMM if not stopped."""
    df = fal._parse_french_monthly(
        _FF5_FIXTURE,
        expected_columns=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"),
    )
    # If annual leaked in, we'd have 4 rows + Mkt-RF=9.93/100=0.0993
    # on the 4th. Assert exactly 3 + max date stays in 1963.
    assert len(df) == 3
    assert df["date"].max().year == 1963
    assert df["date"].max().month == 9


def test_parse_raises_when_header_missing():
    bad = "no header here\njust garbage\n"
    with pytest.raises(ValueError, match="header row"):
        fal._parse_french_monthly(
            bad, expected_columns=("Mkt-RF",),
        )


def test_parse_raises_when_no_data_rows():
    """Header but no monthly data (e.g. header followed by 'Annual'
    directly)."""
    bad = ",Mom\n\nAnnual Factors:\n,Mom\n1927,21.43\n"
    with pytest.raises(ValueError, match="no monthly data"):
        fal._parse_french_monthly(bad, expected_columns=("Mom",))


# ────────────────────────────────────────────────────────────────────
# _extract_csv_text
# ────────────────────────────────────────────────────────────────────
def test_extract_csv_text_finds_match_case_insensitive():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Foo.csv", "hello,world")
    out = fal._extract_csv_text(buf.getvalue(), "f-f_foo.csv")
    assert out == "hello,world"


def test_extract_csv_text_raises_when_member_absent():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("other.csv", "x")
    with pytest.raises(FileNotFoundError):
        fal._extract_csv_text(buf.getvalue(), "expected.csv")


# ────────────────────────────────────────────────────────────────────
# Integration shape test (skipped if anchor parquet missing)
# ────────────────────────────────────────────────────────────────────
def test_real_anchor_parquet_shape_if_present():
    """If anchor parquet exists (post fetcher run), check it has
    the documented schema."""
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "famafrench_monthly.parquet")
    if not p.exists():
        pytest.skip("anchor parquet not cached — run scripts/"
                      "fetch_anchor_library.py")
    df = pd.read_parquet(p)
    assert {"date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"} \
        == set(df.columns)
    assert len(df) > 600   # ~60+ years of months
    assert df["date"].iloc[0]  >= pd.Timestamp("1963-01-01")
    assert df["date"].iloc[-1] >= pd.Timestamp("2024-01-01")
    # Decimal not percent — equity premium should be ~0.6%/mo not 60
    assert -0.05 < df["MKT_RF"].mean() < 0.05
    assert  0.0  < df["MOM"].mean()    < 0.05
    # No NaN in factors (RF can have synthesized zeros at low rates)
    for c in ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
        assert df[c].notna().all(), f"{c} has NaN"


# ────────────────────────────────────────────────────────────────────
# L2-6 lite (2026-06-09): 12-Industry parser via synthetic fixture
# ────────────────────────────────────────────────────────────────────
_INDUSTRY_FIXTURE = """\
This file was created from CRSP 202604.
Average Value Weighted Returns -- Monthly

      ,  NoDur ,  Durbl ,  Manuf ,  Enrgy ,  Chems ,  BusEq ,  Telcm ,  Utils ,  Shops ,  Hlth  ,  Money ,  Other
192607,   1.45,   5.86,  -1.51,  -2.21,   1.81,  -1.31,   4.92,   1.45,   1.62,   3.47,   1.45,   2.18
192608,   3.23,   3.84,   2.42,   2.61,   3.22,   2.99,   2.40,   3.30,   2.31,   2.92,   2.10,   3.05
192609,   1.31,  -3.10,   0.31,  -1.40,   0.21,   0.14,   1.34,  -0.78,   0.85,   0.41,   0.31,   0.65

Average Equal Weighted Returns -- Monthly

      ,  NoDur ,  Durbl ,  Manuf ,  Enrgy ,  Chems ,  BusEq ,  Telcm ,  Utils ,  Shops ,  Hlth  ,  Money ,  Other
192607,   2.15,   4.86,  -2.51,   1.21,   1.21,  -1.81,   4.52,   2.45,   1.92,   3.97,   2.45,   2.78
"""


def test_parse_industries_only_picks_first_panel():
    """Ken French 12-Industry CSV has multiple panels (VW Returns,
    EW Returns, # Firms, Avg Firm Size). Parser MUST stop at first
    blank line so only VW returns end up in the parquet — that's
    the institutional standard."""
    df = fal._parse_french_monthly(
        _INDUSTRY_FIXTURE,
        expected_columns=fal.INDUSTRY_COLUMNS,
    )
    assert len(df) == 3   # only VW panel; EW panel rejected
    # First row values came from VW section, not EW
    assert df["NoDur"].iloc[0] == pytest.approx(0.0145)  # NOT 0.0215


def test_industry_columns_are_complete():
    """The 12 industry columns are a stable contract — any rename
    upstream would silently produce a worse downstream regression."""
    assert len(fal.INDUSTRY_COLUMNS) == 12
    assert "BusEq" in fal.INDUSTRY_COLUMNS    # tech canonical name
    assert "Money" in fal.INDUSTRY_COLUMNS    # finance canonical


def test_real_industry_parquet_shape_if_present():
    p = (Path(__file__).resolve().parents[1] / "data" / "anchor_library"
         / "industries_12_monthly.parquet")
    if not p.exists():
        pytest.skip("industry parquet not cached — run scripts/"
                      "fetch_anchor_library.py")
    df = pd.read_parquet(p)
    assert set(df.columns) == {"date"} | set(fal.INDUSTRY_COLUMNS)
    assert len(df) > 1000   # 1926+ → many decades
    # Decimal not percent
    for c in fal.INDUSTRY_COLUMNS:
        assert -0.10 < df[c].mean() < 0.10, (
            f"{c} mean {df[c].mean()} outside decimal range"
        )
        assert df[c].notna().all(), f"{c} has NaN"
    # Tech should have higher vol than utilities (textbook)
    sub = df[df["date"] >= pd.Timestamp("1965-01-01")]
    assert sub["BusEq"].std() > sub["Utils"].std()
