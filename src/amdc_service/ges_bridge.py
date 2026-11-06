"""Publish and inspect the explicit GES Firmen_ID6-to-Firmen_ID2 bridge."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb


GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE = "ges_firmen_id6_to_firmen_id2"
GES_FIRMEN_ID6_TO_FIRMEN_ID2_COLUMNS = frozenset({
    "year",
    "Firmen_ID6",
    "Firmen_ID2",
    "source_rows",
    "n_firmen_id2_candidates",
    "bridge_status",
})
GES_FIRMEN_ID6_KIND = "Firmen_ID6"
IDENTIFIER_CONTRACT_COLUMNS = frozenset({"product", "identifier_kind"})


@dataclass(frozen=True)
class GesBridgeProvider:
    """One loaded product that observes Firmen_ID6 and Firmen_ID2 together."""

    product: str
    firmen_id2_column: str
    firmen_id6_column: str
    year_column: str

    @property
    def table_name(self) -> str:
        name = f"{self.product}_firmen_id6_to_firmen_id2"
        if name != GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE:
            return name
        return f"{self.product}_provider_firmen_id6_to_firmen_id2"


PRODCOM_GES_BRIDGE_PROVIDER = GesBridgeProvider(
    product="prodcom",
    firmen_id2_column="R_ID",
    firmen_id6_column="Firmen_ID6",
    year_column="PJA",
)
PRODCOM_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE = PRODCOM_GES_BRIDGE_PROVIDER.table_name


@dataclass(frozen=True)
class GesFirmenId6ToFirmenId2Contract:
    """The physical GES identifier column and all bridge contract errors."""

    source_column: str | None
    reasons: tuple[str, ...]


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _qualified_name(*parts: str | None) -> str:
    return ".".join(_quote_identifier(part) for part in parts if part is not None)


def _relation_columns(
    connection: duckdb.DuckDBPyConnection,
    *,
    catalog: str | None,
    schema: str,
    table: str,
) -> set[str]:
    catalog_filter = "" if catalog is None else "AND table_catalog = ?"
    parameters = [schema, table] if catalog is None else [schema, table, catalog]
    rows = connection.execute(
        f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? {catalog_filter}
        """,
        parameters,
    ).fetchall()
    return {str(row[0]) for row in rows}


def _relation_reason(
    connection: duckdb.DuckDBPyConnection,
    *,
    catalog: str | None,
    schema: str,
    table: str,
    required_columns: frozenset[str],
) -> str | None:
    columns = _relation_columns(
        connection,
        catalog=catalog,
        schema=schema,
        table=table,
    )
    if not columns:
        return f"{table} absent"
    missing = sorted(required_columns - columns)
    if missing:
        return f"{table} missing {', '.join(missing)}"
    return None


def resolve_ges_firmen_id6_source_column(
    columns: set[str],
    identifier_kind: str | None,
) -> tuple[str | None, str | None]:
    """Resolve an explicit Firmen_ID6 column or a declared R_ID alias."""
    if "Firmen_ID6" in columns:
        return "Firmen_ID6", None
    if "R_ID" not in columns:
        return None, "ges missing Firmen_ID6 or R_ID"
    if identifier_kind is None:
        return "R_ID", "ges identifier_kind absent"
    if identifier_kind != GES_FIRMEN_ID6_KIND:
        return (
            "R_ID",
            f"ges identifier_kind {identifier_kind} does not match "
            f"{GES_FIRMEN_ID6_KIND}",
        )
    return "R_ID", None


def inspect_ges_firmen_id6_to_firmen_id2_contract(
    connection: duckdb.DuckDBPyConnection,
    *,
    catalog: str | None,
) -> GesFirmenId6ToFirmenId2Contract:
    """Inspect the GES identifier and its published bridge without reading data rows."""
    ges_columns = _relation_columns(
        connection,
        catalog=catalog,
        schema="main",
        table="ges",
    )
    identifier_kind: str | None = None
    metadata_reason: str | None = None
    if "Firmen_ID6" not in ges_columns and "R_ID" in ges_columns:
        metadata_reason = _relation_reason(
            connection,
            catalog=catalog,
            schema="amdc_meta",
            table="identifier_contract",
            required_columns=IDENTIFIER_CONTRACT_COLUMNS,
        )
        if metadata_reason is None:
            row = connection.execute(
                f"SELECT identifier_kind FROM "
                f"{_qualified_name(catalog, 'amdc_meta', 'identifier_contract')} "
                "WHERE product = 'ges'"
            ).fetchone()
            if row is not None and row[0] is not None:
                identifier_kind = str(row[0])

    source_column, identifier_reason = resolve_ges_firmen_id6_source_column(
        ges_columns,
        identifier_kind,
    )
    reasons = [
        reason
        for reason in (
            metadata_reason,
            identifier_reason,
            _relation_reason(
                connection,
                catalog=catalog,
                schema="amdc_bridge",
                table=GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE,
                required_columns=GES_FIRMEN_ID6_TO_FIRMEN_ID2_COLUMNS,
            ),
        )
        if reason is not None
    ]
    return GesFirmenId6ToFirmenId2Contract(source_column, tuple(reasons))


def create_firmen_id6_to_firmen_id2_provider_bridge(
    connection: duckdb.DuckDBPyConnection,
    *,
    provider: GesBridgeProvider,
    provider_catalog: str | None = None,
) -> None:
    """Publish observed same-year Firmen_ID6-to-Firmen_ID2 pairs."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS amdc_bridge")
    year_column = _quote_identifier(provider.year_column)
    firmen_id6_column = _quote_identifier(provider.firmen_id6_column)
    firmen_id2_column = _quote_identifier(provider.firmen_id2_column)
    connection.execute(
        f"""
        CREATE VIEW {_qualified_name(None, 'amdc_bridge', provider.table_name)} AS
        WITH pairs AS (
            SELECT
                TRY_CAST({year_column} AS INTEGER) AS year,
                CAST({firmen_id6_column} AS VARCHAR) AS source_id,
                CAST({firmen_id2_column} AS VARCHAR) AS target_id,
                count(*) AS source_rows
            FROM {_qualified_name(provider_catalog, 'main', provider.product)}
            WHERE TRY_CAST({year_column} AS INTEGER) IS NOT NULL
              AND {firmen_id6_column} IS NOT NULL
              AND {firmen_id2_column} IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        classified AS (
            SELECT
                *,
                count(*) OVER (PARTITION BY year, source_id)
                    AS n_legal_unit_matches
            FROM pairs
        )
        SELECT
            {_quote_literal(provider.product)} AS source_product,
            year,
            source_id,
            target_id,
            source_rows,
            n_legal_unit_matches,
            CASE WHEN n_legal_unit_matches = 1
                 THEN 'unique' ELSE 'ambiguous' END AS bridge_status
        FROM classified
        """
    )


def create_ges_firmen_id6_to_firmen_id2_bridge(
    connection: duckdb.DuckDBPyConnection,
    *,
    ges_source_column: str,
    provider: GesBridgeProvider,
    source_catalog: str | None = None,
) -> None:
    """Publish one fail-closed bridge row for each GES year and Firmen_ID6."""
    connection.execute("CREATE SCHEMA IF NOT EXISTS amdc_bridge")
    ges_source = _quote_identifier(ges_source_column)
    connection.execute(
        f"""
        CREATE VIEW amdc_bridge.{GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE} AS
        WITH ges_ids AS (
            SELECT
                TRY_CAST(JAHR AS INTEGER) AS year,
                CAST({ges_source} AS VARCHAR) AS Firmen_ID6,
                count(*) AS source_rows
            FROM {_qualified_name(source_catalog, 'main', 'ges')}
            WHERE TRY_CAST(JAHR AS INTEGER) IS NOT NULL
              AND {ges_source} IS NOT NULL
            GROUP BY TRY_CAST(JAHR AS INTEGER), CAST({ges_source} AS VARCHAR)
        ),
        matches AS (
            SELECT
                ges_ids.year,
                ges_ids.Firmen_ID6,
                ges_ids.source_rows,
                count(DISTINCT provider.target_id) FILTER (
                    WHERE provider.target_id IS NOT NULL
                ) AS n_firmen_id2_candidates,
                min(provider.target_id) FILTER (
                    WHERE provider.target_id IS NOT NULL
                ) AS matched_firmen_id2
            FROM ges_ids
            LEFT JOIN {_qualified_name(source_catalog, 'amdc_bridge', provider.table_name)}
                AS provider
              ON provider.year = ges_ids.year
             AND provider.source_id = ges_ids.Firmen_ID6
            GROUP BY ges_ids.year, ges_ids.Firmen_ID6, ges_ids.source_rows
        )
        SELECT
            year,
            Firmen_ID6,
            CASE WHEN n_firmen_id2_candidates = 1
                 THEN matched_firmen_id2 ELSE NULL END AS Firmen_ID2,
            source_rows,
            n_firmen_id2_candidates,
            CASE
                WHEN n_firmen_id2_candidates = 1 THEN 'unique'
                WHEN n_firmen_id2_candidates > 1 THEN 'ambiguous'
                ELSE 'unmatched'
            END AS bridge_status
        FROM matches
        """
    )
