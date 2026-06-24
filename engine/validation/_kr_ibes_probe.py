"""Discovery: how to link I/B/E/S international (ibes.actu_epsint) to Compustat
Global prices (g_secd/g_security) for Korea, to get EXACT announce dates. Throwaway."""
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
                        connect_args={"sslmode": "require"}).execution_options(isolation_level="AUTOCOMMIT")

    def q(sql):
        try:
            return pd.read_sql(text(sql), eng)
        except Exception as e:
            return "ERR:" + str(e).splitlines()[0][:80]

    # 1. g_security columns + KOR sample (which IDs exist: cusip/isin/sedol?)
    gc = q("select column_name from information_schema.columns where table_schema='comp' "
           "and table_name='g_security' order by ordinal_position")
    print("g_security cols:", list(gc["column_name"]) if not isinstance(gc, str) else gc)
    idcols = [c for c in (gc["column_name"] if not isinstance(gc, str) else [])
              if c in ("gvkey", "iid", "isin", "sedol", "cusip", "excntry", "exchg", "tic")]
    samp = q(f"select {', '.join(idcols)} from comp.g_security where excntry='KOR' limit 6"
             if "excntry" in idcols else f"select {', '.join(idcols)} from comp.g_security limit 6")
    print("g_security KOR sample:\n", samp.to_string() if not isinstance(samp, str) else samp)
    print()

    # 2. I/B/E/S actuals for Korea via curr_act='KRW'
    n = q("select count(*) n, min(anndats) mn, max(anndats) mx from ibes.actu_epsint "
          "where curr_act='KRW' and pdicity='QTR' and measure='EPS'")
    print("ibes.actu_epsint KRW QTR EPS:\n", n.to_string() if not isinstance(n, str) else n)
    sa = q("select ticker, cusip, oftic, cname, pends, anndats, value from ibes.actu_epsint "
           "where curr_act='KRW' and pdicity='QTR' and measure='EPS' "
           "order by anndats desc limit 6")
    print("sample:\n", sa.to_string() if not isinstance(sa, str) else sa)
    print()

    # 3. cusip overlap test (if g_security has cusip)
    if "cusip" in idcols:
        ov = q("select count(distinct s.cusip) n_match from comp.g_security s "
               "where s.excntry='KOR' and substr(s.cusip,1,8) in "
               "(select distinct substr(cusip,1,8) from ibes.actu_epsint where curr_act='KRW')")
        print("KOR cusip overlap (g_security <-> ibes):\n", ov.to_string() if not isinstance(ov, str) else ov)
    eng.dispose()


if __name__ == "__main__":
    main()
