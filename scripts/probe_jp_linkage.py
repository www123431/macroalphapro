"""scripts/probe_jp_linkage.py — fresh probe of JP IBEStoCompustat
Global linking options. Try ISIN + oftic + cusip-9 variants."""
import os, sys
appdata = os.environ.get("APPDATA")
if appdata:
    os.environ["PGPASSFILE"] = os.path.join(appdata, "postgresql", "pgpass.conf")
import wrds
import pandas as pd

db = wrds.Connection(wrds_username="${WRDS_USER_2}")

print("=" * 80)
print(" JP IBEStoCompustat Global linkage probe")
print("=" * 80)

# 1. JP comp.g_company count via loc
print("\n[1] JP firms in comp.g_company (loc=JPN):")
r = db.raw_sql("SELECT COUNT(*) FROM comp.g_company WHERE loc='JPN'")
print(f"  {int(r.iloc[0,0]):,}")

# 2. Sample JP companies + their g_security records
print("\n[2] Sample JP firms with tic/cusip/isin:")
r = db.raw_sql("""SELECT s.gvkey, s.tic, s.cusip, s.isin, c.conm
                 FROM comp.g_company c JOIN comp.g_security s ON c.gvkey=s.gvkey
                 WHERE c.loc='JPN' AND s.tic IS NOT NULL LIMIT 8""")
print(r.to_string())

# 3. IBES JP tickers sample (oftic = TSE code?)
print("\n[3] Sample IBES JP-firms (curr_act=JPY EPS):")
r = db.raw_sql("""SELECT DISTINCT a.ticker, a.oftic, a.cname
                 FROM ibes.act_epsint a
                 WHERE a.curr_act='JPY' AND a.pdicity='QTR'
                 LIMIT 8""")
print(r.to_string())

# 4. ATTEMPT: link IBES oftic to comp.g_security.tic (both TSE codes for JP)
print("\n[4] Test linkage via IBES oftic to comp.g_security.tic:")
r = db.raw_sql("""SELECT COUNT(DISTINCT a.oftic) AS n_oftics
                 FROM ibes.act_epsint a
                 WHERE a.curr_act='JPY' AND a.pdicity='QTR'
                 AND a.oftic IS NOT NULL""")
print(f"  unique IBES oftics for JP: {int(r.iloc[0,0]):,}")

r = db.raw_sql("""SELECT COUNT(DISTINCT s.gvkey) AS n_matched
                 FROM ibes.act_epsint a
                 JOIN comp.g_company c ON c.loc='JPN'
                 JOIN comp.g_security s ON s.gvkey=c.gvkey AND s.tic=a.oftic
                 WHERE a.curr_act='JPY' AND a.pdicity='QTR'""")
print(f"  matched to gvkey via oftic=tic: {int(r.iloc[0,0]):,}")

# 5. Try ISIN linkage
print("\n[5] ISIN linkage attempt (ibes intl doesn't have ISIN; try via cusip-9):")
# IBES cusip is 8-char; CUSIP-9 is 8 + 1 check digit. comp.g_security might store CUSIP-9
# Try LEFT(comp.cusip,8) = ibes.cusip
r = db.raw_sql("""SELECT COUNT(DISTINCT s.gvkey) AS n_matched
                 FROM ibes.act_epsint a
                 JOIN ibes.id i ON i.ticker=a.ticker
                 JOIN comp.g_security s ON LEFT(s.cusip,8)=i.cusip
                 WHERE a.curr_act='JPY' AND a.pdicity='QTR'""")
print(f"  matched via LEFT(comp.cusip,8)=ibes.cusip: {int(r.iloc[0,0]):,}")

# 6. SOLD: which linkage works best, pull the gvkey list
print("\n[6] Final: build IBES_ticker → gvkey crosswalk via best linkage")
r = db.raw_sql("""SELECT DISTINCT a.ticker AS ibes_ticker, s.gvkey, a.cname
                 FROM ibes.act_epsint a
                 JOIN comp.g_company c ON c.loc='JPN'
                 JOIN comp.g_security s ON s.gvkey=c.gvkey AND s.tic=a.oftic
                 WHERE a.curr_act='JPY' AND a.pdicity='QTR'""")
print(f"  crosswalk rows: {len(r):,}")
r.to_parquet("data/cache/_jp_ibes_to_gvkey_crosswalk.parquet")
print(f"  saved: data/cache/_jp_ibes_to_gvkey_crosswalk.parquet")

print("\n[DONE]")
