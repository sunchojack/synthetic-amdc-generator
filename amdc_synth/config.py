"""Synthetic output settings and shared AMDC reference exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.crosswalk.config import (
    AMDC_REST,
    CATALOGUE,
    CODEBOOKS,
    DERA_PDF,
    JRC_PDF,
    KDB,
    KDB_GVGES,
    KDB_KN,
    KDB_OENACE,
    KDB_OEPRODCOM,
    REFERENCE,
    ROOT,
)
from src.crosswalk.schema import BY_TABLE, PRODUCTS, Product


OUT = ROOT / "out"
PARQUET = OUT / "parquet"
CACHE = OUT / "cache"
DUCKDB_PATH = OUT / "amdc_synthesized.duckdb"
REPORT_PATH = OUT / "calibration_report.md"


@dataclass
class Settings:
    """Runtime knobs. SCALE multiplies every published row count."""

    scale: float = 0.01
    seed: int = 20260727
    out_parquet: Path = field(default_factory=lambda: PARQUET)

    def rows(self, published: int) -> int:
        return max(1, round(published * self.scale))

    def firms(self, published: int) -> int:
        return max(1, round(published * self.scale))

    def distinct(self, stats, year: int) -> int:
        """Published distinct-value count for a variable and year, unscaled.

        Distinct code counts are not scaled: a pipeline has to cope with the real
        breadth of the CN and OEPRODCOM code space whatever the row count is.
        """
        try:
            return int(stats.unique_per_year[stats.years.index(year)])
        except (ValueError, IndexError):
            return 0
