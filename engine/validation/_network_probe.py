"""One-connection WRDS probe for NETWORK / linked-firm momentum data:
  (A) supply-chain: Compustat segment CUSTOMER file — does it carry a customer
      gvkey/identifier or only a name (name-match burden)?
  (B) shared-analyst co-coverage: IBES detail (analyst-level) coverage — size +
      the analyst id + ticker fields (clean link, no name matching).
  (C) any pre-linked supply-chain (FactSet Revere / wrdsapps).
Throwaway. Run: python -u -m engine.validation._network_probe
"""
import os
import socket
import time

import pandas as pd
from sqlalchemy import create_engine, text


def main():
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    h, p, d, u, pw = open(pg).read().strip().splitlines()[0].split(":")
    print("active account =", u)
    eng = create_engine(f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{d}",
                        connect_args={"sslmode": "require"}).execution_options(
        isolation_level="AUTOCOMMIT")

    def q(sql):
        try:
            return pd.read_sql(text(sql), eng)
        except Exception as e:
            return "ERR: " + str(e).splitlines()[0][:95]

    print("\n=== (A) Compustat segment / customer tables ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where (table_schema ilike 'comp%' or table_schema ilike '%seg%') "
          "and (table_name ilike '%seg%customer%' or table_name ilike '%customer%' "
          "or table_name ilike 'seg_%' or table_name ilike '%segment%') order by 1,2")
    print(t.to_string(index=False) if not isinstance(t, str) else t)
    for tbl in ("comp.seg_customer", "compseg.seg_customer", "comp_segments.seg_customer",
                "compsegd.seg_customer"):
        n = q(f"select count(*) n from {tbl}")
        if not isinstance(n, str):
            s, name = tbl.split(".")
            cols = q("select column_name from information_schema.columns "
                     f"where table_schema='{s}' and table_name='{name}' order by ordinal_position")
            print(f"  {tbl}: rows={int(n['n'][0])} cols={list(cols['column_name']) if not isinstance(cols,str) else cols}")
            samp = q(f"select * from {tbl} order by srcdate desc limit 3" if not isinstance(cols, str)
                     and "srcdate" in set(cols["column_name"]) else f"select * from {tbl} limit 3")
            if not isinstance(samp, str):
                print("    sample:\n", samp.to_string()[:800])
            break

    print("\n=== (B) IBES detail (analyst-level) for co-coverage ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where table_schema ilike 'ibes%' and (table_name ilike 'det%') order by 1,2 limit 20")
    print(t.to_string(index=False) if not isinstance(t, str) else t)
    for tbl in ("ibes.detu_epsus", "ibes.det_epsus", "ibes.detu_epsint"):
        n = q(f"select count(*) n from {tbl}")
        if not isinstance(n, str):
            s, name = tbl.split(".")
            cols = q("select column_name from information_schema.columns "
                     f"where table_schema='{s}' and table_name='{name}' order by ordinal_position")
            cl = list(cols["column_name"]) if not isinstance(cols, str) else cols
            print(f"  {tbl}: rows={int(n['n'][0])} cols={cl}")
            # analyst id field + distinct analysts/tickers
            if not isinstance(cols, str) and "analys" in set(cl):
                dd = q(f"select count(distinct analys) na, count(distinct ticker) nt from {tbl} "
                       f"where anndats >= '2011-01-01'")
                print("    distinct analysts/tickers (>=2011):", dd.to_dict('records') if not isinstance(dd, str) else dd)
            break

    print("\n=== (C) pre-linked supply chain (factset/revere/wrdsapps) ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where (table_schema ilike '%revere%' or table_schema ilike '%factset%' "
          "or table_schema ilike '%supply%') and table_name ilike '%rel%' order by 1,2 limit 15")
    print(t.to_string(index=False) if not isinstance(t, str) else t)
    eng.dispose()


if __name__ == "__main__":
    main()
