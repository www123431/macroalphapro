"""One-connection WRDS probe of S&P Capital IQ earnings-call TRANSCRIPTS for the
Line C DL feature-extraction track. Checks: which ciq* schemas/tables are
SELECT-accessible on the active account, the transcript header + component-text
tables, the date range / row counts, and the companyid->ticker->gvkey/permno
linkage. Throwaway. Run: python -u -m engine.validation._ciq_probe
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
    print(f"active account = {u}")
    eng = create_engine(f"postgresql+psycopg2://{u}:{pw}@{h}:{p}/{d}",
                        connect_args={"sslmode": "require"}).execution_options(
        isolation_level="AUTOCOMMIT")

    def q(sql):
        try:
            return pd.read_sql(text(sql), eng)
        except Exception as e:
            return "ERR: " + str(e).splitlines()[0][:100]

    print("\n=== ciq* schemas visible ===")
    sc = q("select schema_name from information_schema.schemata where schema_name ilike 'ciq%' order by 1")
    print(list(sc["schema_name"]) if not isinstance(sc, str) else sc)

    print("\n=== transcript-related tables (ciq*) ===")
    t = q("select table_schema, table_name from information_schema.tables "
          "where table_schema ilike 'ciq%' and (table_name ilike '%transcript%' "
          "or table_name ilike '%event%') order by 1,2")
    print(t.to_string(index=False) if not isinstance(t, str) else t)

    # access + size test on the most likely tables
    print("\n=== SELECT-access + size probe ===")
    for tbl in ("ciq_transcripts.wrds_transcript_detail",
                "ciq_transcripts.ciqtranscript",
                "ciq_transcripts.ciqtranscriptcomponent",
                "ciq.ciqtranscript", "ciq.ciqtranscriptcomponent"):
        n = q(f"select count(*) n from {tbl}")
        print(f"  {tbl:46s} {n if isinstance(n,str) else int(n['n'][0])}")

    # find a date column on the transcript header + range
    print("\n=== transcript header columns (find date) ===")
    for tbl in ("ciq_transcripts.ciqtranscript", "ciq_transcripts.wrds_transcript_detail"):
        s, name = tbl.split("."); cols = q(
            "select column_name, data_type from information_schema.columns "
            f"where table_schema='{s}' and table_name='{name}' order by ordinal_position")
        if not isinstance(cols, str):
            print(f"  {tbl}:", list(cols["column_name"]))

    print("\n=== companyid->ticker/gvkey linkage ===")
    for tbl in ("ciq_common.wrds_ticker", "ciq_common.ciqsymbol", "ciq_common.wrds_gvkey"):
        n = q(f"select count(*) n from {tbl}")
        print(f"  {tbl:34s} {n if isinstance(n,str) else int(n['n'][0])}")
    eng.dispose()


if __name__ == "__main__":
    main()
