from __future__ import annotations

import unittest

import duckdb

from amdc_synth.assemble_synthetic_parquets_into_duckdb import _add_identifier_bridges
from src.amdc_service.ges_bridge import (
    GES_FIRMEN_ID6_TO_FIRMEN_ID2_COLUMNS,
    PRODCOM_GES_BRIDGE_PROVIDER,
    create_firmen_id6_to_firmen_id2_provider_bridge,
    create_ges_firmen_id6_to_firmen_id2_bridge,
)


class GesBridgeTests(unittest.TestCase):
    def test_bridge_is_unique_ambiguous_unmatched_and_same_year(self) -> None:
        connection = duckdb.connect()
        try:
            connection.execute(
                "CREATE TABLE prodcom (R_ID VARCHAR, Firmen_ID6 VARCHAR, PJA INTEGER)"
            )
            connection.execute(
                """
                INSERT INTO prodcom VALUES
                    ('legal-1', 'est-1', 2021),
                    ('legal-1', 'est-1', 2021),
                    ('legal-2', 'est-2', 2021),
                    ('legal-3', 'est-2', 2021),
                    ('legal-4', 'est-4', 2020)
                """
            )
            connection.execute("CREATE TABLE ges (Firmen_ID6 VARCHAR, JAHR INTEGER)")
            connection.execute(
                """
                INSERT INTO ges VALUES
                    ('est-1', 2021),
                    ('est-1', 2021),
                    ('est-2', 2021),
                    ('est-3', 2021),
                    ('est-4', 2021),
                    ('est-4', 2020)
                """
            )

            create_firmen_id6_to_firmen_id2_provider_bridge(
                connection,
                provider=PRODCOM_GES_BRIDGE_PROVIDER,
            )
            create_ges_firmen_id6_to_firmen_id2_bridge(
                connection,
                ges_source_column="Firmen_ID6",
                provider=PRODCOM_GES_BRIDGE_PROVIDER,
            )

            self.assertEqual(
                connection.execute(
                    "SELECT year, Firmen_ID6, Firmen_ID2, source_rows, "
                    "n_firmen_id2_candidates, bridge_status "
                    "FROM amdc_bridge.ges_firmen_id6_to_firmen_id2 "
                    "ORDER BY year, Firmen_ID6"
                ).fetchall(),
                [
                    (2020, "est-4", "legal-4", 1, 1, "unique"),
                    (2021, "est-1", "legal-1", 2, 1, "unique"),
                    (2021, "est-2", None, 1, 2, "ambiguous"),
                    (2021, "est-3", None, 1, 0, "unmatched"),
                    (2021, "est-4", None, 1, 0, "unmatched"),
                ],
            )
        finally:
            connection.close()

    def test_synthetic_publisher_uses_the_shared_contract(self) -> None:
        connection = duckdb.connect()
        try:
            connection.execute(
                "CREATE TABLE prodcom (R_ID VARCHAR, Firmen_ID6 VARCHAR, PJA INTEGER)"
            )
            connection.execute("INSERT INTO prodcom VALUES ('legal-1', 'est-1', 2021)")
            connection.execute("CREATE TABLE ges (Firmen_ID6 VARCHAR, JAHR INTEGER)")
            connection.execute("INSERT INTO ges VALUES ('est-1', 2021)")

            _add_identifier_bridges(connection)

            columns = {
                str(row[0])
                for row in connection.execute(
                    "DESCRIBE amdc_bridge.ges_firmen_id6_to_firmen_id2"
                ).fetchall()
            }
            self.assertEqual(columns, set(GES_FIRMEN_ID6_TO_FIRMEN_ID2_COLUMNS))
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
