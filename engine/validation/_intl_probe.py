"""One-connection probe of WRDS international data (Japan/Korea) for value + PEAD:
Compustat Global / Worldscope prices+fundamentals + I/B/E/S international. Throwaway."""
import os
import socket
import time

from sqlalchemy import create_engine, text


def main():
    for _ in range(8):
        try:
            socket.gethostbyname("wrds-pgdata.wharton.upenn.edu"); break
        except Exception:
            time.sleep(4)
    pg = os.path.join(os.environ["APPDATA"], "postgresql", "pgpass.conf")
    h, p, d, u, pw = open(pg).read().strip().splitlines()[0].split(":")
    eng = create_engine(
        f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{d}",
        connect_args={"sslmode": "require"}).execution_options(isolation_level="AUTOCOMMIT")

    def q(c, sql):
        try:
            return [dict(r._mapping) for r in c.execute(text(sql))]
        except Exception as e:
            return "ERR:" + str(e).splitlines()[0][:70]

    with eng.connect() as c:
        # 1. discover global price/fundamental tables across schemas
        print("=== tables named like global price/fundamentals ===")
        r = q(c, "select table_schema, table_name from information_schema.tables "
                 "where table_name in ('g_secd','g_funda','g_security','g_names','secd','funda') "
                 "or table_name ~ '^g_' order by table_schema, table_name")
        if isinstance(r, str): print(r)
        else:
            for x in r[:40]: print(f"  {x['table_schema']}.{x['table_name']}")
        print()

        # 2. SELECT-probe candidate global + worldscope tables
        print("=== SELECT-probe (count) ===")
        for t in ["comp_global_daily.g_secd", "comp.g_secd", "comp.g_funda",
                  "comp_global_daily.g_security", "comp.g_security",
                  "tr_worldscope.wrds_ws_funda", "trws.wrds_ws_funda",
                  "ibes.actu_epsint", "ibes.statsumu_epsint", "ibes.id_int"]:
            r = q(c, f"select count(*) as n from {t}")
            print(f"  {t:38s} {r if isinstance(r, str) else r[0]['n']}")
        print()

        # 3. Japan/Korea coverage where accessible — try the most likely price table
        print("=== Japan/Korea coverage probe ===")
        for tbl, ccol in [("comp_global_daily.g_secd", "fic"), ("comp.g_secd", "fic")]:
            r = q(c, f"select {ccol}, count(*) n from {tbl} "
                     f"where {ccol} in ('JPN','KOR') group by {ccol}")
            print(f"  {tbl} by {ccol}: {r}")
        # ibes international country coverage
        for tbl in ["ibes.id_int", "ibes.actu_epsint"]:
            r = q(c, f"select count(*) n from {tbl} limit 1")
            if not isinstance(r, str):
                cols = q(c, "select column_name from information_schema.columns "
                            f"where table_schema='ibes' and table_name='{tbl.split('.')[1]}' order by ordinal_position")
                print(f"  {tbl} cols: {[x['column_name'] for x in cols][:18] if not isinstance(cols,str) else cols}")
    eng.dispose()


if __name__ == "__main__":
    main()
