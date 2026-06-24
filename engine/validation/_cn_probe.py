"""One-connection probe of WRDS China A-share data (wind_ashare / comp_global /
ibes intl) for a China PEAD backtest. Throwaway."""
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
        print("=== wind_ashare tables ===")
        r = q(c, "select table_name from information_schema.tables "
                 "where table_schema='wind_ashare' order by 1")
        print(r if isinstance(r, str) else [x["table_name"] for x in r])
        print()
        print("=== SELECT-probe candidate wind tables (count) ===")
        for t in ["wind_ashare.ashareeodprices", "wind_ashare.ashareeodderivativeindicator",
                  "wind_ashare.ashareincome", "wind_ashare.asharebalancesheet",
                  "wind_ashare.ashareprofitnotice", "wind_ashare.ashareprofitexpress",
                  "wind_ashare.ashareconsensusdata", "wind_ashare.ashareearningest",
                  "wind_ashare.asharedescription", "wind_ashare.asharecalendar"]:
            r = q(c, f"select count(*) as n from {t}")
            print(f"  {t:46s} {r if isinstance(r, str) else r[0]['n']}")
    eng.dispose()


if __name__ == "__main__":
    main()
