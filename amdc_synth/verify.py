"""Machine-check generated data against the pushed AMDC metadata.

Checks fall into two tiers:

**Structural regression** — schema shape, row counts, file existence, identifier
ranges.  These pass by construction when the generator is bug-free, because the
generator reads the same catalogue the checks compare against.  They catch
programming errors in the generator (wrong column, dropped table, off-by-one in
a year range), not data-quality problems.

**Substantive verification** — value ranges, dispersion, cross-table identifier
overlap, weight-value relationships, CRM signal diversity.  These test properties
the downstream pipeline depends on and that a broken generator can silently
violate: degenerate distributions, non-positive economic values, vanished
cross-table linkages, or a planted signal too thin to recover.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from src.crosswalk.metadata import catalogue_variables, load_batch
from .config import (BY_TABLE, CATALOGUE, CODEBOOKS, DERA_PDF, JRC_PDF, KDB, OUT,
                     PRODUCTS, Settings)
from .gen_prodcom import _unit_code
from .pipeline import GENERATORS, SIDE_TABLES, SYNTHETIC_SIDE_TABLES
from .register import ID_BASE
from .regimes import GES_PUBLIC_UNITS_TARGET
import src.crosswalk.codelists as cl
from src.amdc_service.ges_bridge import (
    GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE,
    PRODCOM_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE,
)
from src.crosswalk.kdb_fetch import validate_archives


URS_COLUMNS = {
    "lse_re": ("R_ID", "jahr", "IS", "RF", "oenace", "FSW_NUTS3", "FSW_LKZ",
               "gcd_fsw", "USB", "SB", "UST", "RE_DB_DAT_NG", "RE_DB_DAT_SL"),
    "lse_loc": ("R_ID", "Firmen_ID5", "jahr", "oenace_s", "S_NUTS3", "S_LKZ",
                "gcd_s"),
    "lse_stat": ("R_ID", "Firmen_ID3", "JAHR", "UNT_IS", "UNT_OENACE"),
    "lse_m": ("R_ID", "jahr", "M_IS", "M_RF", "M_OENACE", "M_FSW_NUTS3",
              "M_FSW_LKZ", "M_FSW_GEMNR", "M_USB", "M_SB", "M_UST", "M_DB_DAT_NG",
              "M_DB_DAT_SL"),
    "lse_d": ("R_ID", "jahr", "T_IS", "T_RF", "T_OENACE", "T_FSW_NUTS3",
              "T_FSW_LKZ", "T_FSW_GEMNR", "T_USB", "T_SB", "T_UST", "T_DB_DAT_NG",
              "T_DB_DAT_SL"),
    "lse_g": ("R_ID", "Firmen_ID4", "JAHR", "GRP_ART", "GRP_LKZ",
              "GRP_DB_DAT_NG", "GRP_DB_DAT_SL"),
}

PHYSICAL_EXTRAS = {
    "intra": ("SV", "CO", "PRAEF", "VERKINL", "VERKGRE"),
    "extra": ("V1",),
}

INTRASTAT_EMPTY = ("SV", "CO", "PRAEF", "VERKINL", "VERKGRE")

YEAR_COLUMNS = ("BERJ", "PJA", "JAHR", "Jahr", "jahr")


@dataclass(frozen=True)
class Verification:
    checks: int
    tables: int


def _expected_columns(table: str, keys: tuple[str, ...]) -> set[str]:
    if table in URS_COLUMNS:
        return set(URS_COLUMNS[table])
    return set((*keys, *catalogue_variables(table), *PHYSICAL_EXTRAS.get(table, ())))


def _year_column(columns: list[str]) -> str | None:
    return next((name for name in YEAR_COLUMNS if name in columns), None)


def _count(lf: pl.LazyFrame, predicate: pl.Expr) -> int:
    return int(lf.select(predicate.fill_null(False).sum()).collect().item())


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _verify_sources(failures: list[str]) -> int:
    checks = 0
    for product in PRODUCTS:
        checks += 1
        _require((CATALOGUE / f"{product.oid}.json").is_file(),
                 f"missing catalogue record for {product.table}", failures)
        if product.metadata_batch:
            checks += 1
            _require((CODEBOOKS / product.metadata_batch).is_file(),
                     f"missing metadata batch for {product.table}", failures)
    checks += 3
    _require(KDB.is_dir(), f"missing classification export: {KDB}", failures)
    _require(JRC_PDF.is_file(), f"missing CRM code inventory: {JRC_PDF}", failures)
    _require(DERA_PDF.is_file(), f"missing DERA material inventory: {DERA_PDF}", failures)
    archive_failures = validate_archives(KDB) if KDB.is_dir() else ()
    checks += 1
    _require(not archive_failures,
             "invalid curated KDB inventory: " + "; ".join(archive_failures), failures)
    return checks


def verify_sources() -> int:
    failures: list[str] = []
    checks = _verify_sources(failures)
    if failures:
        raise RuntimeError("source verification failed:\n- " + "\n- ".join(failures))
    return checks


def _verify_product(product, parquet_dir: Path, cfg: Settings,
                    failures: list[str]) -> int:
    checks = 1
    path = parquet_dir / f"{product.table}.parquet"
    if not path.is_file():
        failures.append(f"missing Parquet table: {product.table}")
        return checks

    lf = pl.scan_parquet(path)
    columns = lf.collect_schema().names()
    expected = _expected_columns(product.table, product.keys)
    checks += 1
    _require(set(columns) == expected,
             f"{product.table} schema differs: missing={sorted(expected - set(columns))}, "
             f"extra={sorted(set(columns) - expected)}", failures)

    for key in product.keys:
        checks += 1
        if key in columns:
            _require(_count(lf, pl.col(key).is_null()) == 0,
                     f"{product.table}.{key} contains nulls", failures)

    checks += 1
    year_col = _year_column(columns)
    if year_col is None:
        failures.append(f"{product.table} has no reporting-year column")
        return checks
    observed_years = {
        int(value) for value in
        lf.select(pl.col(year_col).drop_nulls().unique()).collect().to_series().to_list()
    }
    _require(observed_years == set(product.years),
             f"{product.table} years differ: observed={sorted(observed_years)}, "
             f"expected={list(product.years)}", failures)

    batch = load_batch(product.table)
    if batch is not None:
        observed = lf.group_by(year_col).len().collect()
        observed_rows = {int(y): int(n) for y, n in observed.iter_rows()}
        for year, published in batch.rows_per_year().items():
            checks += 1
            _require(observed_rows.get(year) == cfg.rows(published),
                     f"{product.table} {year} rows={observed_rows.get(year)}, "
                     f"expected={cfg.rows(published)}", failures)
    return checks


def _verify_identifiers(parquet_dir: Path, failures: list[str]) -> int:
    path = parquet_dir / "_register.parquet"
    if not path.is_file():
        failures.append("missing Parquet table: _register")
        return 1
    lf = pl.scan_parquet(path)
    checks = 0
    for key, base in ID_BASE.items():
        checks += 1
        bounds = lf.select(pl.col(key).cast(pl.Int64).min().alias("lo"),
                           pl.col(key).cast(pl.Int64).max().alias("hi")).collect()
        lo, hi = bounds.row(0)
        _require(lo is not None and base <= lo <= hi < base + 1_000_000_000,
                 f"{key} leaves its identifier range", failures)
    return checks


def _verify_structure(parquet_dir: Path, cfg: Settings, failures: list[str]) -> int:
    checks = 0
    regimes = (
        pl.scan_parquet(parquet_dir / "_amdc_regimes.parquet")
        .filter(pl.col("kind") == "unit")
        .select("table_name", "first_year", "last_year", "columns", "multiplier_to_eur")
        .collect()
    )
    monetary_contract = {
        (str(table), int(first), int(last), str(columns), int(multiplier))
        for table, first, last, columns, multiplier in regimes.iter_rows()
        if multiplier is not None
    }
    for expected in {
        ("ges", 2008, 2023, "WE", 1_000),
    }:
        checks += 1
        _require(expected in monetary_contract,
                 f"AMDC monetary regime is missing or incorrect: {expected}", failures)

    intra = pl.scan_parquet(parquet_dir / "intra.parquet")
    invariants = {
        "Intrastat VL/EV": ((pl.col("EV") == 2) != pl.col("VL").is_null()),
        "Intrastat BL/EV": ((pl.col("EV") == 1) != pl.col("BL").is_null()),
        "Intrastat pre-2022 import UL": ((pl.col("EV") == 1) &
                                         (pl.col("BERJ") <= 2021) &
                                         pl.col("UL").is_null()),
        "Intrastat pre-2022 dispatch UL": ((pl.col("EV") == 2) &
                                           (pl.col("BERJ") <= 2021) &
                                           pl.col("UL").is_not_null()),
        "Intrastat pre-2022 GART": ((pl.col("BERJ") <= 2021) &
                                    (pl.col("GART").str.len_chars() >= 2)),
        "Intrastat post-2021 GART": ((pl.col("BERJ") >= 2022) &
                                     (pl.col("GART").str.len_chars() == 1)),
    }
    for name, predicate in invariants.items():
        checks += 1
        _require(_count(intra, predicate) == 0, f"{name} invariant failed", failures)
    for column in INTRASTAT_EMPTY:
        checks += 1
        _require(_count(intra, pl.col(column).is_not_null()) == 0,
                 f"Intrastat {column} is not all-null", failures)

    extra = pl.scan_parquet(parquet_dir / "extra.parquet")
    checks += 1
    _require(_count(extra, (pl.col("EV") == 2) & pl.col("PRAEF").is_not_null()) == 0,
             "Extrastat PRAEF is populated on exports", failures)
    checks += 1
    _require(_count(extra, ((pl.col("BERJ") == 2022) != pl.col("V1").is_not_null())) == 0,
             "Extrastat V1 is populated outside 2022 or missing within 2022", failures)
    v1 = extra.filter(pl.col("BERJ") == 2022).select(
        pl.col("V1").min().alias("lo"), pl.col("V1").max().alias("hi"),
        pl.col("V1").n_unique().alias("unique"), pl.len().alias("rows"),
    ).collect().row(0)
    checks += 1
    _require(v1[0] == 1 and v1[1] == v1[2] == v1[3],
             "Extrastat V1 is not a complete 2022 row counter", failures)

    prodcom = pl.scan_parquet(parquet_dir / "prodcom.parquet")
    checks += 1
    _require(_count(prodcom, (pl.col("PEH2").is_null() & (pl.col("PME2") != -1)) |
                    (pl.col("PEH2").is_not_null() & (pl.col("PME2") <= 0))) == 0,
             "PRODCOM second-unit fields disagree", failures)
    expected_second_units = pl.DataFrame(
        [
            {
                "PJA": year,
                "PCODE": f"{good.code:010d}",
                "expected_peh2": _unit_code(good.unit2) if good.unit2 else None,
            }
            for year in BY_TABLE["prodcom"].years
            for good in cl.oeprodcom(year)
        ],
        schema={"PJA": pl.Int32, "PCODE": pl.Utf8, "expected_peh2": pl.Int32},
    ).lazy()
    prodcom_second_units = prodcom.join(
        expected_second_units,
        on=["PJA", "PCODE"],
        how="left",
    )
    checks += 1
    _require(
        _count(
            prodcom_second_units,
            pl.col("PEH2").fill_null(-1) != pl.col("expected_peh2").fill_null(-1),
        ) == 0,
        "PRODCOM PEH2 disagrees with the selected PCODE declaration",
        failures,
    )
    checks += 1
    _require(_count(prodcom, ((pl.col("PEH1") == 0) & (pl.col("PME1") != -1)) |
                    ((pl.col("PEH1") != 0) & (pl.col("PME1") == -1))) == 0,
             "PRODCOM first-unit fields disagree", failures)

    ges_units = (pl.scan_parquet(parquet_dir / "ges.parquet")
                 .group_by("JAHR")
                 .agg(pl.col("Firmen_ID6").n_unique().alias("units"))
                 .collect())
    checks += 1
    _require(set(ges_units["units"].to_list()) == {cfg.firms(GES_PUBLIC_UNITS_TARGET)},
             "GES annual establishment count differs from the scaled AMDC target", failures)
    return checks


def _verify_classifications(parquet_dir: Path, failures: list[str]) -> int:
    checks = 0
    for table in ("intra", "extra"):
        values = (pl.scan_parquet(parquet_dir / f"{table}.parquet")
                  .select("BERJ", "KN").unique().collect())
        for year, part in values.partition_by("BERJ", as_dict=True).items():
            reporting_year = int(year[0])
            checks += 1
            _require(set(part["KN"].to_list()) <=
                     {f"{good.code:08d}" for good in cl.cn8(reporting_year)},
                     f"{table} {reporting_year} contains invalid CN8 codes", failures)

    values = (pl.scan_parquet(parquet_dir / "prodcom.parquet")
              .select("PJA", "PCODE").unique().collect())
    for year, part in values.partition_by("PJA", as_dict=True).items():
        reporting_year = int(year[0])
        checks += 1
        _require(set(part["PCODE"].to_list()) <=
                 {f"{good.code:010d}" for good in cl.oeprodcom(reporting_year)},
                 f"prodcom {reporting_year} contains invalid OEPRODCOM codes", failures)

    register = pl.scan_parquet(parquet_dir / "_register.parquet")
    observed = set(register.select("nace5").unique().collect()["nace5"].to_list())
    checks += 1
    _require(observed <= {int(item.code5) for item in cl.oenace2008()},
             "register contains invalid OENACE codes", failures)
    return checks


def _verify_database(out_dir: Path, failures: list[str]) -> int:
    db_path = out_dir / "amdc_synthesized.duckdb"
    if not db_path.is_file():
        failures.append("missing DuckDB database")
        return 1
    parquet_dir = out_dir / "parquet"
    expected = {
        *(("main", table) for table in GENERATORS),
        *(("main", table) for table in SYNTHETIC_SIDE_TABLES),
        ("main", "amdc_products"),
        ("amdc_bridge", GES_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE),
        ("amdc_bridge", PRODCOM_FIRMEN_ID6_TO_FIRMEN_ID2_TABLE),
    }
    checks = 1
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        relations = {
            (str(schema), str(table))
            for schema, table in con.execute(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema IN ('main', 'amdc_bridge')"
            ).fetchall()
        }
        _require(relations == expected,
                 f"DuckDB objects differ: missing={sorted(expected - relations)}, "
                 f"extra={sorted(relations - expected)}", failures)
        for table in (*GENERATORS, *SYNTHETIC_SIDE_TABLES):
            checks += 1
            db_rows = con.sql(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            parquet_rows = con.sql("SELECT count(*) FROM read_parquet(?)",
                                   params=[str(parquet_dir / f"{table}.parquet")]).fetchone()[0]
            _require(db_rows == parquet_rows,
                     f"DuckDB/Parquet row mismatch for {table}", failures)
        checks += 1
        _require(con.sql("SELECT count(*) FROM _crm_ground_truth").fetchone()[0] > 0,
                 "_crm_ground_truth is empty", failures)
        checks += 1
        trade_edges = con.sql("""
            SELECT count(*)
            FROM intra i JOIN _crm_ground_truth g USING (R_ID)
            WHERE list_contains(g.upstream_cn8, i.KN)
               OR list_contains(g.downstream_cn8, i.KN)
        """).fetchone()[0]
        _require(trade_edges > 0, "planted CRM trade signal is not recoverable", failures)
        checks += 1
        prodcom_edges = con.sql("""
            SELECT count(*)
            FROM prodcom p JOIN _crm_ground_truth g USING (R_ID)
            WHERE list_contains(g.prodcom8, substr(p.PCODE, 1, 8))
        """).fetchone()[0]
        _require(prodcom_edges > 0, "planted CRM PRODCOM signal is not recoverable", failures)
    finally:
        con.close()
    return checks


def _verify_separation(out_dir: Path, failures: list[str]) -> int:
    crosswalk_parquet = sorted((out_dir / "parquet").glob("_harm_*.parquet"))
    _require(not crosswalk_parquet, "synthetic output contains crosswalk Parquet", failures)
    _require(not (out_dir / "amdc_crosswalk.duckdb").exists(),
             "synthetic output contains crosswalk DuckDB", failures)
    return 2


def _verify_magnitudes(parquet_dir: Path, failures: list[str]) -> int:
    """Values are positive, finite, and span a realistic range."""
    checks = 0

    for table in ("intra", "extra"):
        lf = pl.scan_parquet(parquet_dir / f"{table}.parquet")
        for col in ("SW", "GW"):
            checks += 1
            bad = _count(lf, (pl.col(col) < 0) | pl.col(col).is_infinite()
                         | pl.col(col).is_nan())
            _require(bad == 0,
                     f"{table}.{col} has {bad} negative, infinite or NaN values",
                     failures)

        checks += 1
        log_std = (lf.select(pl.col("SW").filter(pl.col("SW") > 0)
                              .log().std())
                   .collect().item())
        _require(log_std is not None and log_std > 0.5,
                 f"{table}.SW log-dispersion too low ({log_std}); "
                 "values may be degenerate", failures)

        checks += 1
        directions = set(
            lf.select("EV").unique().collect()["EV"].to_list()
        )
        _require({1, 2} <= directions,
                 f"{table} missing direction(s): observed EV={sorted(directions)}",
                 failures)

        checks += 1
        n_vl = (lf.filter(pl.col("VL").is_not_null())
                .select(pl.col("VL").n_unique()).collect().item())
        _require(n_vl is not None and n_vl >= 5,
                 f"{table} has only {n_vl} distinct partner countries in VL",
                 failures)

    prodcom = pl.scan_parquet(parquet_dir / "prodcom.parquet")
    checks += 1
    bad_pw = _count(prodcom, (pl.col("PART") == 5) & (pl.col("PWERT") < 0))
    _require(bad_pw == 0,
             f"PRODCOM has {bad_pw} negative PWERT rows for PART=5", failures)

    checks += 1
    pw_std = (prodcom.select(pl.col("PWERT").filter(pl.col("PWERT") > 0)
                              .log().std())
              .collect().item())
    _require(pw_std is not None and pw_std > 0.5,
             f"PRODCOM PWERT log-dispersion too low ({pw_std})", failures)

    ges = pl.scan_parquet(parquet_dir / "ges.parquet")
    checks += 1
    bad_we = _count(ges, (pl.col("WE") < 0) | pl.col("WE").is_infinite()
                    | pl.col("WE").is_nan())
    _require(bad_we == 0,
             f"GES has {bad_we} negative, infinite or NaN WE values", failures)

    checks += 1
    we_std = (ges.select(pl.col("WE").filter(pl.col("WE") > 0).log().std())
              .collect().item())
    _require(we_std is not None and we_std > 0.5,
             f"GES WE log-dispersion too low ({we_std})", failures)

    return checks


def _verify_cross_table(parquet_dir: Path, failures: list[str]) -> int:
    """Identifier linkage across tables is not degenerate."""
    checks = 0
    reg_ids = set(
        pl.scan_parquet(parquet_dir / "_register.parquet")
        .select("R_ID").unique().collect()["R_ID"].to_list()
    )

    for table in ("intra", "extra"):
        table_ids = set(
            pl.scan_parquet(parquet_dir / f"{table}.parquet")
            .select("R_ID").unique().collect()["R_ID"].to_list()
        )
        checks += 1
        overlap = len(table_ids & reg_ids)
        rate = overlap / max(len(table_ids), 1)
        _require(rate > 0.50,
                 f"{table}: only {rate:.0%} of firms appear in the register "
                 f"({overlap}/{len(table_ids)})", failures)

    prodcom_ids = set(
        pl.scan_parquet(parquet_dir / "prodcom.parquet")
        .select("R_ID").unique().collect()["R_ID"].to_list()
    )
    checks += 1
    overlap = len(prodcom_ids & reg_ids)
    rate = overlap / max(len(prodcom_ids), 1)
    _require(rate > 0.80,
             f"PRODCOM: only {rate:.0%} of firms in the register "
             f"({overlap}/{len(prodcom_ids)})", failures)

    ges_ids = set(
        pl.scan_parquet(parquet_dir / "ges.parquet")
        .select("Firmen_ID6").unique().collect()["Firmen_ID6"].to_list()
    )
    reg_id6 = set(
        pl.scan_parquet(parquet_dir / "_register.parquet")
        .select("Firmen_ID6").unique().collect()["Firmen_ID6"].to_list()
    )
    checks += 1
    overlap = len(ges_ids & reg_id6)
    rate = overlap / max(len(ges_ids), 1)
    _require(rate > 0.80,
             f"GES: only {rate:.0%} of establishments in the register "
             f"({overlap}/{len(ges_ids)})", failures)

    gt_ids = set(
        pl.scan_parquet(parquet_dir / "_crm_ground_truth.parquet")
        .select("R_ID").unique().collect()["R_ID"].to_list()
    )
    trade_ids = set()
    for trade_table in ("intra", "extra"):
        trade_ids |= set(
            pl.scan_parquet(parquet_dir / f"{trade_table}.parquet")
            .select("R_ID").unique().collect()["R_ID"].to_list()
        )
    checks += 1
    overlap = len(gt_ids & trade_ids)
    rate = overlap / max(len(gt_ids), 1)
    _require(rate > 0.50,
             f"CRM ground truth: only {rate:.0%} of firms appear in trade "
             f"({overlap}/{len(gt_ids)})", failures)

    return checks


def _verify_signal_quality(parquet_dir: Path, failures: list[str]) -> int:
    """CRM signal has non-trivial structure and trade values are coupled."""
    checks = 0

    gt_path = parquet_dir / "_crm_ground_truth.parquet"
    if not gt_path.is_file():
        return checks
    gt = pl.read_parquet(gt_path)

    checks += 1
    n_materials = gt["material"].n_unique()
    _require(n_materials >= 3,
             f"CRM ground truth covers only {n_materials} materials", failures)

    checks += 1
    n_firms = gt["R_ID"].n_unique()
    _require(n_firms >= 3,
             f"CRM ground truth covers only {n_firms} firms", failures)

    checks += 1
    upstream = gt.select(pl.col("upstream_cn8").list.len().sum()).item()
    downstream = gt.select(pl.col("downstream_cn8").list.len().sum()).item()
    _require(upstream > 0 and downstream > 0,
             f"CRM ground truth lacks upstream ({upstream}) or downstream "
             f"({downstream}) codes", failures)

    intra = pl.scan_parquet(parquet_dir / "intra.parquet")
    positive = intra.filter((pl.col("SW") > 0) & (pl.col("GW") > 0))
    row_count = positive.select(pl.len()).collect().item()
    if row_count > 100:
        checks += 1
        corr = (positive
                .select(pl.corr(pl.col("SW").log(), pl.col("GW").log()))
                .collect().item())
        _require(corr is not None and corr > 0,
                 f"trade log(SW)-log(GW) correlation is non-positive ({corr:.3f}); "
                 "generated values lack commodity dependence", failures)

    return checks


def verify(out_dir: Path, cfg: Settings) -> Verification:
    failures: list[str] = []
    checks = verify_sources()
    parquet_dir = out_dir / "parquet"
    for product in PRODUCTS:
        checks += _verify_product(product, parquet_dir, cfg, failures)
    for table in SIDE_TABLES:
        checks += 1
        _require((parquet_dir / f"{table}.parquet").is_file(),
                 f"missing side table: {table}", failures)
    checks += _verify_identifiers(parquet_dir, failures)
    checks += _verify_structure(parquet_dir, cfg, failures)
    checks += _verify_classifications(parquet_dir, failures)
    checks += _verify_magnitudes(parquet_dir, failures)
    checks += _verify_cross_table(parquet_dir, failures)
    checks += _verify_signal_quality(parquet_dir, failures)
    checks += _verify_database(out_dir, failures)
    checks += _verify_separation(out_dir, failures)

    report_path = out_dir / "calibration_report.md"
    checks += 1
    _require(report_path.is_file() and f"SCALE = {cfg.scale:g}, seed = {cfg.seed}." in
             report_path.read_text(encoding="utf-8"),
             "calibration report is missing or belongs to another build", failures)
    if failures:
        raise RuntimeError("verification failed:\n- " + "\n- ".join(failures))

    result = Verification(checks=checks, tables=len(PRODUCTS) + len(SIDE_TABLES))
    (out_dir / "verification_report.json").write_text(json.dumps({
        "status": "passed", "scale": cfg.scale, "seed": cfg.seed,
        "checks": result.checks, "tables": result.tables,
    }, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    result = verify(out_dir, Settings(scale=args.scale, seed=args.seed,
                                      out_parquet=out_dir / "parquet"))
    print(f"verified {result.tables} tables with {result.checks} checks")


if __name__ == "__main__":
    main()
