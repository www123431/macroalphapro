"""Map the tr_ds_fut futures universe: class master (dsfutclass) + contract-info
class distribution, to identify the COMMODITY (+ FX) classes for a genuine carry
build. Throwaway."""
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

    print("=== dsfutclass cols ===")
    c = q("select column_name from information_schema.columns where "
          "table_schema='tr_ds_fut' and table_name='dsfutclass' order by ordinal_position")
    print(list(c["column_name"]) if not isinstance(c, str) else c)
    print("\n=== dsfutclass sample (class master) ===")
    s = q("select * from tr_ds_fut.dsfutclass limit 25")
    print(s.to_string() if not isinstance(s, str) else s)

    # how many distinct classes, and contracts per class (liquidity proxy)
    print("\n=== contracts per class (top 40 by #contracts), with a name ===")
    r = q("select clscode, count(*) ncontr, min(startdate) mn, max(lasttrddate) mx "
          "from tr_ds_fut.wrds_contract_info group by clscode order by ncontr desc limit 40")
    print(r.to_string() if not isinstance(r, str) else r)
    eng.dispose()


if __name__ == "__main__":
    main()
