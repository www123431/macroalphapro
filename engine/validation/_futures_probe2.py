"""Targeted probe of WRDS tr_ds_fut (Datastream Futures): SELECT access + whether it
carries the CURVE (per-contract maturity/expiry + settlement price + underlying map).
Throwaway."""
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
            return "ERR: " + str(e).splitlines()[0][:110]

    for tbl in ("tr_ds_fut.wrds_fut_contract", "tr_ds_fut.wrds_fut_series",
                "tr_ds_fut.wrds_contract_info", "tr_ds_fut.dsfutcontrval",
                "tr_ds_fut.dsfutcontrinfo", "optionm.futures_price", "optionm.futures"):
        s, name = tbl.split(".")
        cols = q("select column_name, data_type from information_schema.columns where "
                 f"table_schema='{s}' and table_name='{name}' order by ordinal_position")
        if isinstance(cols, str):
            print(f"\n{tbl}: cols ERR {cols}"); continue
        n = q(f"select count(*) n from {tbl}")
        print(f"\n=== {tbl}  rows={n if isinstance(n,str) else int(n['n'][0])} ===")
        print("  cols:", list(cols["column_name"]))
        if not isinstance(n, str) and int(n["n"][0]) > 0:
            samp = q(f"select * from {tbl} limit 2")
            if not isinstance(samp, str):
                print("  sample:\n", samp.to_string()[:700])
    eng.dispose()


if __name__ == "__main__":
    main()
