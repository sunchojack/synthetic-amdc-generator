"""Build the year-stacked Combined Nomenclature reference table."""

from __future__ import annotations

import polars as pl

from .config import Settings
import src.crosswalk.codelists as cl

COLUMNS = ("jahr", "KN", "code", "kapitel", "hs4", "hs6", "masseinheit",
           "titel", "kurztitel")

# Reporting years the trade files cover; the nomenclature is stacked over the same span.
YEARS = tuple(range(2013, 2024))


def kn_combined(cfg: Settings, reg=None) -> pl.DataFrame:
    """One row per 8-digit CN code and reporting year, from the real KDB export."""
    frames = []
    for year in YEARS:
        goods = cl.cn8(year)
        with_unit = cl.cn8_special_unit(year)
        codes = [g.code for g in goods]
        frames.append(pl.DataFrame({
            "jahr": pl.Series([year] * len(codes), dtype=pl.Int32),
            "KN": pl.Series([f"{c:08d}" for c in codes], dtype=pl.Utf8),
            "code": pl.Series([f"{c:08d}" for c in codes], dtype=pl.Utf8),
            "kapitel": pl.Series([f"{c // 1_000_000:02d}" for c in codes], dtype=pl.Utf8),
            "hs4": pl.Series([f"{c // 10_000:04d}" for c in codes], dtype=pl.Utf8),
            "hs6": pl.Series([f"{c // 100:06d}" for c in codes], dtype=pl.Utf8),
            "masseinheit": pl.Series(
                [g.unit1 if g.unit1 else ("p/st" if g.code in with_unit else None)
                 for g in goods], dtype=pl.Utf8),
            "titel": pl.Series([g.title for g in goods], dtype=pl.Utf8),
            "kurztitel": pl.Series([g.short_title for g in goods], dtype=pl.Utf8),
        }))
    return pl.concat(frames).select(COLUMNS)
