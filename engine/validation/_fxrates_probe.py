"""Identify FX (currency) + RATES (bond/note) futures classes on tr_ds_fut by name,
for a cross-asset carry combination. Throwaway."""
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

    # FX: major currency futures vs USD (CME). RATES: US Treasury futures.
    fx = ("euro fx|japanese yen|british pound|swiss franc|australian dollar|"
          "canadian dollar|new zealand|mexican peso|dollar index|euro/us")
    rates = ("treasury|t-note|t-bond|10 year|10-year|5 year|2 year|30 year|"
             "ultra|fed fund|eurodollar|sofr|bund|bobl|schatz|gilt|jgb")
    for label, kw in (("FX", fx), ("RATES", rates)):
        r = q("select clscode, max(contrname) contrname, max(exchtickersymb) sym, "
              "max(isocurrcode) ccy, count(*) ncontr, min(startdate) mn, max(lasttrddate) mx "
              "from tr_ds_fut.wrds_contract_info "
              f"where contrname ~* '{kw}' and isocurrcode='USD' group by clscode "
              "having count(*) >= 40 order by ncontr desc limit 30")
        print(f"\n=== {label} classes (USD, >=40 contracts) ===")
        print(r.to_string() if not isinstance(r, str) else r)
    eng.dispose()


if __name__ == "__main__":
    main()
