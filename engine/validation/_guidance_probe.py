"""One-connection WRDS probe of IBES Company-Issued-Guidance (CIG) for a
management-guidance-drift backtest. Finds the guidance table(s), columns, row
count, date range, US coverage, and the cusip<->permno link feasibility. Throwaway.
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
    eng = create_engine(f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{d}",
                        connect_args={"sslmode": "require"}).execution_options(
        isolation_level="AUTOCOMMIT")

    def q(sql):
        try:
            return pd.read_sql(text(sql), eng)
        except Exception as e:
            return "ERR: " + str(e).splitlines()[0][:90]

    # 1. find guidance tables across likely schemas
    print("=== guidance-like tables (ibes / tr_ibes) ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where (table_schema like 'ibes%' or table_schema like 'tr_ibes%') "
          "and table_name ilike '%guid%' order by 1,2")
    print(t.to_string(index=False) if not isinstance(t, str) else t)

    # 2. probe the most likely detail table
    for tbl in ("ibes.det_guidance", "ibes.detu_guidance", "ibes.guidance",
                "tr_ibes.det_guidance"):
        n = q(f"select count(*) n from {tbl}")
        print(f"\n--- {tbl}: rows = {n if isinstance(n,str) else int(n['n'][0])}")
        if isinstance(n, str):
            continue
        cols = q("select column_name, data_type from information_schema.columns "
                 f"where table_schema='{tbl.split('.')[0]}' and table_name='{tbl.split('.')[1]}' "
                 "order by ordinal_position")
        print("    columns:", list(cols["column_name"]) if not isinstance(cols, str) else cols)
        # date range + measure mix on the announce date
        rng = q(f"select min(anndats) mn, max(anndats) mx, count(distinct ticker) ntick "
                f"from {tbl}")
        print("    range/coverage:", rng.to_dict('records') if not isinstance(rng, str) else rng)
        meas = q(f"select measure, count(*) n from {tbl} group by measure order by n desc limit 8")
        print("    measure mix:", meas.to_dict('records') if not isinstance(meas, str) else meas)
        samp = q(f"select * from {tbl} order by anndats desc limit 4")
        if not isinstance(samp, str):
            print("    sample:\n", samp.to_string()[:1200])
        break

    # 3. cusip<->permno link feasibility (CRSP stocknames ncusip), like analyst_revision
    ov = q("select count(distinct g.cusip) n from ibes.det_guidance g "
           "where substr(g.cusip,1,8) in (select distinct ncusip from crsp.stocknames)")
    print("\n=== cusip overlap ibes.det_guidance <-> crsp.stocknames ncusip ===")
    print("   ", ov.to_dict('records') if not isinstance(ov, str) else ov)
    eng.dispose()


if __name__ == "__main__":
    main()
