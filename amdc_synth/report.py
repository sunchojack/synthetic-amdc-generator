"""Compare generated variables with published AMDC statistics."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.crosswalk.metadata import load_batch
from .config import PARQUET, PRODUCTS, REPORT_PATH, Settings
from .regimes import AMDC_REGIMES

# Generated column -> published variable, where the names differ.
_ALIAS = {
    "lse_re": {"IS": "is", "RF": "rf", "FSW_NUTS3": "fsw_nuts3", "FSW_LKZ": "fsw_lkz",
               "USB": "usb", "SB": "sb", "UST": "ust", "RE_DB_DAT_NG": "re_db_dat_ng",
               "RE_DB_DAT_SL": "re_db_dat_sl", "gcd_fsw": "gcd_fsw", "oenace": "oenace"},
    "lse_loc": {"S_NUTS3": "s_nuts3", "S_LKZ": "s_lkz", "gcd_s": "gcd_s",
                "oenace_s": "oenace_s"},
}

def _year_column(df: pl.DataFrame) -> str | None:
    for c in ("BERJ", "PJA", "JAHR", "Jahr", "jahr"):
        if c in df.columns:
            return c
    return None


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x == 0:
            return "0"
        if abs(x) >= 1000 or abs(x) < 0.01:
            return f"{x:,.4g}"
        return f"{x:,.3f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def _rows_section(cfg: Settings, table: str, df: pl.DataFrame) -> list[str]:
    b = load_batch(table)
    pub = b.rows_per_year()
    ycol = _year_column(df)
    out = ["| year | rows generated | rows published | published x SCALE | gap |",
           "|---|---|---|---|---|"]
    got = (df.group_by(ycol).len().sort(ycol) if ycol else pl.DataFrame())
    got_map = dict(zip(got[ycol].to_list(), got["len"].to_list())) if ycol else {}
    for year in sorted(set(pub) | set(got_map)):
        g = got_map.get(year)
        p = pub.get(year)
        tgt = cfg.rows(p) if p else None
        gap = f"{100 * (g - tgt) / tgt:+.1f}%" if (g and tgt) else "no published figure"
        out.append(f"| {year} | {_fmt(g)} | {_fmt(p)} | {_fmt(tgt)} | {gap} |")
    return out


def _variable_section(cfg: Settings, table: str, df: pl.DataFrame) -> list[str]:
    b = load_batch(table)
    alias = _ALIAS.get(table, {})
    out = ["| variable | missing rate obs / pub | distinct/yr obs / pub | mean obs / pub | "
           "p50 obs / pub | p90 obs / pub |",
           "|---|---|---|---|---|---|"]
    ycol = _year_column(df)
    for col in df.columns:
        key = alias.get(col, col)
        v = b.get(key)
        if v is None:
            continue
        s = df[col]
        miss_obs = s.null_count() / max(df.height, 1)
        miss_pub = v.total_missing / max(v.total_rows, 1)
        if ycol and col != ycol:
            per = (df.group_by(ycol)
                   .agg(pl.col(col).drop_nulls().n_unique().alias("_n"))
                   .sort(ycol)["_n"].to_list())
            d_obs = f"{min(per)}-{max(per)}" if per else "-"
        else:
            d_obs = str(s.drop_nulls().n_unique())
        d_pub = (f"{min(v.unique_per_year)}-{max(v.unique_per_year)}"
                 if v.unique_per_year else "-")
        if s.dtype.is_numeric() and v.numeric:
            mean = f"{_fmt(s.mean())} / {_fmt(v.numeric.get('mean'))}"
            p50 = f"{_fmt(s.quantile(0.50))} / {_fmt(v.numeric.get('m_p50'))}"
            p90 = f"{_fmt(s.quantile(0.90))} / {_fmt(v.numeric.get('m_p90'))}"
        else:
            mean = p50 = p90 = "-"
        out.append(f"| `{col}` | {miss_obs:.4f} / {miss_pub:.4f} | {d_obs} / {d_pub} | "
                   f"{mean} | {p50} | {p90} |")
    return out


CAVEATS = """
## Interpretation

- Published row counts are scaled; code-universe distinct counts are not.
- Structural missingness is reproduced; residual missingness varies with sampling.
- Published `m_*` quantiles are treated as disclosure-smoothed targets.
- GES, LSE and FATS magnitudes are modelled because no public summaries exist.
- PRODCOM 2023-2024 and unavailable classification vintages are extrapolated.
- Municipality identifiers are synthetic.

See `docs/data_methods/CORRESPONDENCE.md` for the source and fidelity boundary of each table.
"""


def write(cfg: Settings, tables: dict[str, int], seconds: dict[str, float],
          path: Path = REPORT_PATH, parquet_dir: Path = PARQUET) -> Path:
    lines = [
        "# Calibration report",
        "",
        f"SCALE = {cfg.scale:g}, seed = {cfg.seed}. Published figures are AMDC's own "
        "per-variable statistics, fetched from the public catalogue "
        "(`/amdc-data/rest/getFile/{oid}`) and stored under `amdc_reference/codebooks/`.",
        "",
        "## Tables built",
        "",
        "| table | AMDC oid | abbrev | years | keys | calibration | rows | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in PRODUCTS:
        n = tables.get(p.table)
        if n is None:
            continue
        lines.append(f"| `{p.table}` | {p.oid} | {p.abbrev} | {min(p.years)}-{max(p.years)} | "
                     f"{', '.join(p.keys)} | {p.calibration} | {n:,} | "
                     f"{seconds.get(p.table, 0):.1f} |")
    lines += [
        "",
        "Calibration grade: **A** = AMDC publishes per-variable statistics for this "
        "product and they are reproduced below. **B** = codebook gives the exact schema, "
        "units and value domains; magnitudes are modelled. **C** = codebook only.",
        CAVEATS,
        "",
        "## AMDC time regimes and unit conversions",
        "",
        "| table | years | kind | fields | source representation | conversion | comparison rule |",
        "|---|---|---|---|---|---|---|",
    ]
    for regime in AMDC_REGIMES:
        years = (str(regime.first_year) if regime.first_year == regime.last_year
                 else f"{regime.first_year}-{regime.last_year}")
        conversion = (f"x{regime.multiplier_to_eur:,} to EUR"
                      if regime.multiplier_to_eur is not None else "-")
        lines.append(
            f"| `{regime.table_name}` | {years} | {regime.kind} | {regime.columns} | "
            f"{regime.source_representation} | {conversion} | {regime.amdc_difference} |"
        )

    for p in PRODUCTS:
        if p.calibration != "A" or p.table not in tables:
            continue
        src = parquet_dir / f"{p.table}.parquet"
        if not src.exists():
            continue
        df = pl.read_parquet(src)
        lines += ["", f"## `{p.table}` - {p.abbrev}", "", "### Rows per year", ""]
        lines += _rows_section(cfg, p.table, df)
        lines += ["", "### Per variable", ""]
        lines += _variable_section(cfg, p.table, df)
        del df

    unpub = [p for p in PRODUCTS if p.calibration != "A" and p.table in tables]
    if unpub:
        lines += ["", "## Products with no published statistics", "",
                  "| table | rows | source of the schema |", "|---|---|---|"]
        for p in unpub:
            lines.append(f"| `{p.table}` | {tables[p.table]:,} | "
                         f"AMDC codebook, oid {p.oid} |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
