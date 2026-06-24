"""One-connection WRDS probe of CRSP distributions for a dividend-change-drift
backtest (Michaely-Thaler-Womack). Confirms the table, the distcd codes that mark
REGULAR cash dividends, the declaration/announce date field, and date coverage.
Throwaway. Run: python -u -m engine.validation._dividend_probe
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
            return "ERR: " + str(e).splitlines()[0][:100]

    print("\n=== distribution tables (crsp / crsp_a_stock) ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where table_schema ilike 'crsp%' and (table_name ilike '%dist%' "
          "or table_name ilike '%dividend%') order by 1,2")
    print(t.to_string(index=False) if not isinstance(t, str) else t)

    for tbl in ("crsp.dsedist", "crsp.msedist", "crsp.stkdistributions", "crsp.dist"):
        n = q(f"select count(*) n from {tbl}")
        print(f"\n--- {tbl}: rows = {n if isinstance(n,str) else int(n['n'][0])}")
        if isinstance(n, str):
            continue
        s, name = tbl.split(".")
        cols = q("select column_name from information_schema.columns "
                 f"where table_schema='{s}' and table_name='{name}' order by ordinal_position")
        print("    columns:", list(cols["column_name"]) if not isinstance(cols, str) else cols)
        # distcd distribution + date range (column names vary by format)
        dcol = "distcd" if not isinstance(cols, str) and "distcd" in set(cols["column_name"]) else None
        ddt = next((c for c in ("dclrdt", "disexdt", "exdt", "rcrddt") if not isinstance(cols, str)
                    and c in set(cols["column_name"])), None)
        if dcol:
            vc = q(f"select {dcol}, count(*) n from {tbl} group by {dcol} order by n desc limit 12")
            print("    top distcd:", vc.to_dict("records") if not isinstance(vc, str) else vc)
        if ddt:
            rng = q(f"select min({ddt}) mn, max({ddt}) mx from {tbl} where {ddt} is not null")
            print(f"    {ddt} range:", rng.to_dict("records") if not isinstance(rng, str) else rng)
        samp = q(f"select * from {tbl} limit 3")
        if not isinstance(samp, str):
            print("    sample cols/vals:\n", samp.to_string()[:900])
        break
    eng.dispose()


if __name__ == "__main__":
    main()
