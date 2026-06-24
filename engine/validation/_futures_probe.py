"""One-connection WRDS probe for FUTURES data — specifically whether we have a
TERM-STRUCTURE / CURVE (multiple contracts/maturities per underlying per date),
which is the prerequisite for a genuine commodity/FX CARRY signal (vs the
momentum-proxy carry degenerates to on spot-only free data). Throwaway.
Run: python -u -m engine.validation._futures_probe
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

    print("\n=== schemas matching futures/commodity/cme/cftc/cmdty ===")
    sc = q("select schema_name from information_schema.schemata where "
           "schema_name ~* 'fut|commod|cme|cftc|cmdty|future' order by 1")
    print(list(sc["schema_name"]) if not isinstance(sc, str) else sc)

    print("\n=== tables matching futures/commod/contract/settle across all schemas ===")
    t = q("select table_schema, table_name from information_schema.tables where "
          "(table_name ~* 'fut|commod|contract|settle|cftc' ) "
          "and table_schema not in ('pg_catalog','information_schema') order by 1,2 limit 60")
    print(t.to_string(index=False) if not isinstance(t, str) else t)

    # probe likely curve tables for a maturity/expiration column + access
    print("\n=== access + curve-column probe on candidate futures tables ===")
    for tbl in ("tr_ds_fut.wrds_futures", "ds_fut.wrds_futures", "futures.contract",
                "cmdty.fut_price", "wrds_cmdty.fut", "tfn_fut.fut", "cme.futures"):
        n = q(f"select count(*) n from {tbl}")
        if not isinstance(n, str):
            s, name = tbl.split(".")
            cols = q("select column_name from information_schema.columns where "
                     f"table_schema='{s}' and table_name='{name}' order by ordinal_position")
            print(f"  {tbl}: rows={int(n['n'][0])} cols={list(cols['column_name']) if not isinstance(cols,str) else cols}")
    eng.dispose()


if __name__ == "__main__":
    main()
