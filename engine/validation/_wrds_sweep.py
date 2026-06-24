"""Exhaustive WRDS access sweep: which remaining schemas are SELECT-able on
${WRDS_USER_2}, to 'finish off' WRDS before applying for external data. Throwaway."""
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
            return "ERR:" + str(e).splitlines()[0][:70]

    # candidate (schema, representative fact table) for alpha-relevant datasets
    cands = [
        ("audit", "auditnonreli"),          # restatements
        ("audit", "auditsox404"),            # internal control weakness
        ("audit", "auditopinion"),           # going concern
        ("tr_ibes_guidance", "det_guidance"),# management guidance
        ("zacks", "rec"),                    # Zacks recommendations
        ("zacks", "eps_est"),
        ("reprisk", "v2_risk_incidents"),    # ESG controversies
        ("trucost", "wrds_trucost_extract"),
        ("markit", "redgreen"),              # CDS
        ("fisd", "fisd_mergedissue"),        # bond reference
        ("dealscan", "facility"),            # syndicated loans
        ("tr_dealscan", "facility"),
        ("wrds_mutualfund", "holdings"),
        ("tfn", "s34"),                      # 13F institutional holdings
        ("tr_13f", "s34"),
        ("comp", "secm"),                    # any extra compustat
    ]
    seen = set()
    for sch, tbl in cands:
        # list a few tables in the schema (information_schema is open)
        if sch not in seen:
            t = q("select table_name from information_schema.tables where table_schema='%s' "
                  "order by 1 limit 8" % sch)
            tl = list(t["table_name"]) if not isinstance(t, str) else t
            print(f"\n## {sch}: {tl}")
            seen.add(sch)
        r = q("select count(*) n from %s.%s" % (sch, tbl))
        status = "OK rows=%s" % r["n"][0] if not isinstance(r, str) else r
        print(f"   SELECT {sch}.{tbl}: {status}")
    eng.dispose()


if __name__ == "__main__":
    main()
