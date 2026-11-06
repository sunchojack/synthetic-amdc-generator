"""Assemble generated synthetic Parquet tables into DuckDB."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from .config import DUCKDB_PATH, PARQUET, PRODUCTS
from src.crosswalk.dtypes import check_relation
from .pipeline import GENERATORS, SYNTHETIC_SIDE_TABLES
from src.amdc_service.ges_bridge import (
    PRODCOM_GES_BRIDGE_PROVIDER,
    create_firmen_id6_to_firmen_id2_provider_bridge,
    create_ges_firmen_id6_to_firmen_id2_bridge,
)


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        str(row[0])
        for row in con.execute(f'DESCRIBE "{table}"').fetchall()
    }


def _add_identifier_bridges(con: duckdb.DuckDBPyConnection) -> None:
    """Publish source-backed identifier bridges when both providers are usable."""
    tables = {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
    provider = PRODCOM_GES_BRIDGE_PROVIDER
    if provider.product not in tables:
        return
    required_provider_columns = {
        provider.firmen_id2_column,
        provider.firmen_id6_column,
        provider.year_column,
    }
    if not required_provider_columns <= _columns(con, provider.product):
        return

    create_firmen_id6_to_firmen_id2_provider_bridge(
        con,
        provider=provider,
    )
    if "ges" not in tables:
        return
    if not {"Firmen_ID6", "JAHR"} <= _columns(con, "ges"):
        return
    create_ges_firmen_id6_to_firmen_id2_bridge(
        con,
        ges_source_column="Firmen_ID6",
        provider=provider,
    )


def build(db_path: Path = DUCKDB_PATH, parquet_dir: Path = PARQUET) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_name(f".{db_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    con = duckdb.connect(str(tmp_path))
    try:
        for name in (*GENERATORS, *SYNTHETIC_SIDE_TABLES):
            src = parquet_dir / f"{name}.parquet"
            if not src.exists():
                continue
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM read_parquet(?)', [str(src)])
            check_relation(con, name)

        rows = [(p.table, p.oid, p.abbrev, min(p.years), max(p.years),
                 ", ".join(p.keys), p.calibration, p.note) for p in PRODUCTS]
        con.execute("""
            CREATE TABLE amdc_products (
                table_name VARCHAR, amdc_oid INTEGER, amdc_abbrev VARCHAR,
                first_year INTEGER, last_year INTEGER, firm_keys VARCHAR,
                calibration VARCHAR, note VARCHAR
            )
        """)
        con.executemany("INSERT INTO amdc_products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

        _add_identifier_bridges(con)
        con.execute("ANALYZE")
    finally:
        con.close()
    tmp_path.replace(db_path)
    return db_path
