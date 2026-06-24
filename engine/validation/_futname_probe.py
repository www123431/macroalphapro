"""Identify COMMODITY + FX futures classes by name (wrds_contract_info.contrname),
with contract counts + date range, to fix the carry universe. Throwaway."""
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

    # commodity + FX contracts by name, grouped to class, with liquidity + span
    kw = ("crude|brent|gas oil|gasoil|natural gas|heating|gasoline|rbob|wti|"
          "gold|silver|copper|platinum|palladium|alumin|zinc|nickel|"
          "corn|wheat|soybean|soyabean|soya|sugar|coffee|cotton|cocoa|"
          "cattle|hog|lean|lumber|orange|rubber|"
          "euro fx|japanese yen|british pound|swiss franc|australian dollar|"
          "canadian dollar|dollar index")
    r = q("select clscode, max(contrname) contrname, max(exchtickersymb) sym, "
          "max(isocurrcode) ccy, count(*) ncontr, min(startdate) mn, max(lasttrddate) mx "
          "from tr_ds_fut.wrds_contract_info "
          f"where contrname ~* '{kw}' group by clscode having count(*) >= 30 "
          "order by ncontr desc limit 80")
    if isinstance(r, str):
        print(r)
    else:
        print("commodity/FX classes (>=30 contracts):", len(r))
        print(r.to_string())
    eng.dispose()


if __name__ == "__main__":
    main()
