"""Find clean USD-quoted CME currency futures for the G10 majors to widen the FX
carry leg (EUR/AUD/GBP/SEK/NOK + the 5 already used). Want USD-per-foreign quoting
(like 6J/6C/6S/6M/6N COMP) so the carry sign is consistent. Throwaway."""
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

    # CME COMP currency futures are named "<CCY> COMP." and USD-quoted (USD per unit)
    r = q("select clscode, max(contrname) contrname, max(exchtickersymb) sym, "
          "count(*) ncontr, min(startdate) mn, max(lasttrddate) mx "
          "from tr_ds_fut.wrds_contract_info "
          "where contrname ~* '(euro fx|australian dollar|british pound|swedish|norwegian|"
          "euro |brazilian real|south african rand) comp' and isocurrcode='USD' "
          "group by clscode having count(*) >= 60 order by ncontr desc limit 25")
    print("=== '<CCY> COMP.' USD-quoted currency futures (>=60 contracts) ===")
    print(r.to_string() if not isinstance(r, str) else r)
    eng.dispose()


if __name__ == "__main__":
    main()
