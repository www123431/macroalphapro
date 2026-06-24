"""engine/data/fetchers/wrds_catalog.py — probe WRDS PostgreSQL for
available schemas + tables, store as a versioned catalog.

User 2026-05-30: "WRDS尽量保证都可以连接上, 因为这里的数据比较好, 但前提是
真的需要这里面的数据, 所以你可以先拿一根探针把这里面的数据类目探全然后
存储下来".

Per [[feedback-wrds-care-and-probe-pattern-2026-05-30]] probe-first
pattern: don't burn WRDS quota on speculative fetches. Probe ONCE
(weekly cron), store catalog, then make wiring decisions based on
what's actually there.

CATALOG STRUCTURE (data/cache/wrds_catalog.json):
{
  "probed_at":  "2026-05-30T...",
  "account":    "${WRDS_USER_1}",
  "schemas":    {
    "crsp":    {
      "tables": ["dsf", "msf", "stocknames", "names", "dsedelist", ...],
      "row_counts_approx": {"dsf": 100000000, ...},
    },
    "comp":    {"tables": [...], ...},
    "ibes":    {"tables": [...], ...},
    "optionm": {"tables": [...], ...},
    ...
  },
  "errors":     [...]    # any schema we couldn't read
}

TOKEN → TABLE MAP (what our DATA_INVENTORY maps to in WRDS):
  crsp_dsf           → crsp.dsf
  crsp_msf           → crsp.msf
  compustat_quarterly → comp.fundq
  compustat_annual    → comp.funda
  ibes_summary       → ibes.statsum_epsus
  ibes_detail        → ibes.det_epsus
  ibes_guidance      → ibes.det_guidance
  optionm_iv_surface → optionm.vsurfd
  optionm_skew       → optionm.opprcd1
  ... etc.

This mapping is encoded in DATA_INVENTORY_TO_WRDS so future wiring
work knows exactly which tables to fetch.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "data" / "cache" / "wrds_catalog.json"

# Finance-relevant WRDS schemas to probe. WRDS has 100+ schemas; we
# limit to the ones our DATA_INVENTORY actually references or might.
TARGET_SCHEMAS = [
    "crsp",       # equity prices, indices, delisting
    "comp",       # Compustat fundamentals
    "ibes",       # I/B/E/S analyst forecasts
    "optionm",    # OptionMetrics
    "tr_ds_eq",   # Thomson Reuters Datastream equities
    "tr_ds_fut",  # Datastream futures
    "tr_ds_ind",  # Datastream indices
    "tr_13f",     # 13F holdings
    "trace",      # TRACE bond transactions
    "boardex",    # board / executive data
    "ravenpack",  # RPNA news sentiment
    "dera",       # SEC DERA insider transactions
    "sec_edgar",  # EDGAR filings
    "frb",        # Federal Reserve Board macro
]

# Mapping from our DATA_INVENTORY token namespace to actual WRDS tables.
# Used by data_resolver + future fetcher wiring to know which physical
# tables back each logical token.
# Verified against real catalog (account ${WRDS_USER_2}, 2026-05-30 probe).
# is_year_partitioned=True means the actual table name in WRDS is
# <table>YYYY (e.g. optionm.vsurfd2024); the fetcher must enumerate
# years and UNION ALL, not query a single table.
DATA_INVENTORY_TO_WRDS = {
    # ── Equity (all VERIFIED present in ${WRDS_USER_2} account) ───────────────
    "crsp_dsf":             {"schema": "crsp",       "table": "dsf"},
    "crsp_msf":             {"schema": "crsp",       "table": "msf"},
    "compustat_quarterly":  {"schema": "comp",       "table": "fundq"},
    "compustat_annual":     {"schema": "comp",       "table": "funda"},
    "ibes_summary":         {"schema": "ibes",       "table": "statsum_epsus"},
    "ibes_detail":          {"schema": "ibes",       "table": "det_epsus"},
    "ibes_guidance":        {"schema": "ibes",       "table": "det_guidance"},
    "tr13f_holdings":       {"schema": "tr_13f",     "table": "s34"},
    # ── OptionMetrics: year-partitioned (vsurfd1996...vsurfd2025) ─────
    "optionm_iv_surface":   {"schema": "optionm",    "table": "vsurfd2024",
                              "is_year_partitioned": True,
                              "table_prefix": "vsurfd"},
    "optionm_skew":         {"schema": "optionm",    "table": "opprcd2024",
                              "is_year_partitioned": True,
                              "table_prefix": "opprcd"},
    # ── Cross-asset futures (Datastream — dsfut* table family) ─────────
    "tr_ds_fut_settle":     {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    "cmdty_contracts":      {"schema": "tr_ds_fut",  "table": "dsfutcontr"},
    "cmdty_settle":         {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    "fx_contracts":         {"schema": "tr_ds_fut",  "table": "dsfutcontr"},
    "fx_settle":            {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    "rates_contracts":      {"schema": "tr_ds_fut",  "table": "dsfutcontr"},
    "rates_settle":         {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    "rates_xc_settle":      {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    "eqidx_contracts":      {"schema": "tr_ds_fut",  "table": "dsfutcontr"},
    "eqidx_settle":         {"schema": "tr_ds_fut",  "table": "dsfutcontrval"},
    # ── Probed but NOT subscribed in ${WRDS_USER_2} (kept for future) ─────────
    # edgar_8k_meta, dera_insider, trace_bond_monthly, ravenpack —
    # WRDS subscription doesn't include them; would need to upgrade
    # the WRDS plan or use alternative source.
    # ── Non-WRDS ───────────────────────────────────────────────────────
    "fred_macro":           {"schema": "frb",        "table": None},
}


# ── Probe ─────────────────────────────────────────────────────────────────

def probe_wrds(
    *,
    account: str = "${WRDS_USER_1}",
    target_schemas: list[str] | None = None,
    include_row_counts: bool = False,
    timeout_sec: float = 30.0,
) -> dict:
    """Connect to WRDS via wrds_direct, list tables per schema.

    Args:
      account: which WRDS account to use ("${WRDS_USER_1}" or "${WRDS_USER_2}")
      target_schemas: subset of TARGET_SCHEMAS to probe (default all)
      include_row_counts: if True, also run COUNT(*) for each table
        (SLOW + WRDS-resource-heavy; usually False)
      timeout_sec: per-query timeout

    Returns: catalog dict with shape documented in module docstring.
    """
    from engine.line_c.wrds_direct import connect as _connect_dsn

    target_schemas = target_schemas or TARGET_SCHEMAS
    result = {
        "probed_at":    datetime.datetime.utcnow().isoformat() + "Z",
        "account":      account,
        "schemas":      {},
        "errors":       [],
    }

    try:
        conn = _connect_dsn(account=account, connect_timeout=int(timeout_sec))
    except Exception as exc:
        result["errors"].append({
            "stage":  "connect",
            "error":  str(exc)[:300],
        })
        return result

    try:
        cur = conn.cursor()
        for schema in target_schemas:
            try:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s ORDER BY table_name",
                    (schema,),
                )
                tables = [r[0] for r in cur.fetchall()]
                entry: dict = {"tables": tables}
                if include_row_counts and tables:
                    counts = {}
                    for t in tables:
                        try:
                            cur.execute(
                                f"SELECT reltuples::bigint "
                                f"FROM pg_class WHERE oid = '{schema}.{t}'::regclass"
                            )
                            row = cur.fetchone()
                            counts[t] = int(row[0]) if row and row[0] else None
                        except Exception as exc:
                            counts[t] = None
                            logger.debug("count failed %s.%s: %s",
                                            schema, t, exc)
                    entry["row_counts_approx"] = counts
                result["schemas"][schema] = entry
            except Exception as exc:
                result["errors"].append({
                    "stage":  "probe_schema",
                    "schema": schema,
                    "error":  str(exc)[:300],
                })
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return result


# ── Save / load ───────────────────────────────────────────────────────────

def save_catalog(catalog: dict, path: Path | None = None) -> Path:
    out = path or CATALOG_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(catalog, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


def load_catalog(path: Path | None = None) -> dict | None:
    p = path or CATALOG_PATH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("catalog load failed: %s", exc)
        return None


# ── Lookup helpers ────────────────────────────────────────────────────────

def is_token_available_in_catalog(token: str,
                                       catalog: dict | None = None,
                                       ) -> tuple[bool, str | None]:
    """Check if a DATA_INVENTORY token's underlying WRDS table actually
    exists in the probed catalog.

    Returns: (available, reason_if_not). Reason is None when available.
    """
    if catalog is None:
        catalog = load_catalog()
    if catalog is None:
        return False, "no catalog probed yet — run probe_wrds + save_catalog"

    mapping = DATA_INVENTORY_TO_WRDS.get(token)
    if not mapping:
        return False, f"token {token!r} has no DATA_INVENTORY_TO_WRDS mapping"

    schema = mapping["schema"]
    table = mapping["table"]
    if not table:
        return False, f"token {token!r} doesn't come from WRDS (e.g. FRED)"

    schemas = catalog.get("schemas") or {}
    sch = schemas.get(schema)
    if not sch:
        return False, f"schema {schema!r} not probed yet"
    if table not in (sch.get("tables") or []):
        return False, f"table {schema}.{table} not present in WRDS account"
    return True, None


def summarize_catalog(catalog: dict | None = None) -> dict:
    """Compact summary for daily_summary / health checks."""
    if catalog is None:
        catalog = load_catalog()
    if catalog is None:
        return {"probed": False}
    schemas = catalog.get("schemas") or {}
    return {
        "probed":            True,
        "probed_at":         catalog.get("probed_at"),
        "n_schemas_probed":  len(schemas),
        "n_tables_total":    sum(len(s.get("tables") or [])
                                   for s in schemas.values()),
        "errors":            len(catalog.get("errors") or []),
    }
