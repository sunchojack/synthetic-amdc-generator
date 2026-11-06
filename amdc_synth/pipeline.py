"""Build every table, plus the side tables that make the result usable."""

from __future__ import annotations

import time
from dataclasses import dataclass

import polars as pl

from . import gen_codes, gen_fats, gen_ges, gen_lse, gen_prodcom, gen_trade, gen_urs, register
from .config import Settings
from src.crosswalk.dtypes import check_frame
from .register import Register
from .regimes import regime_frame
from .crm import concentration, signal_materials

# Build order: reference data, register products, then the remaining products.
# The register comes first: it is the frame every other product is drawn from,
# and it is the documented route from Firmen_ID3 to R_ID that the data
# request flags as unresolved. AMDC publishes per-variable statistics for all six
# of its sub-files, so the synthetic frame is calibrated, not assumed.
GENERATORS = {
    "KN_combined": gen_codes.kn_combined,
    "lse_re": gen_urs.urs_re,
    "lse_loc": gen_urs.urs_loc,
    "lse_stat": gen_urs.urs_stat,
    "lse_m": gen_urs.urs_m,
    "lse_d": gen_urs.urs_d,
    "lse_g": gen_urs.urs_g,
    "intra": gen_trade.intra,
    "extra": gen_trade.extra,
    "prodcom": gen_prodcom.prodcom,
    "ges": gen_ges.ges,
    "lse_re_bis2020": gen_lse.lse_re_bis2020,
    "lse_re_ab2021": gen_lse.lse_re_ab2021,
    "ifats_re_bis2020": gen_fats.ifats_re_bis2020,
    "ifats_re_ab2021": gen_fats.ifats_re_ab2021,
    "ofats_re": gen_fats.ofats_re,
    "ofats_unt_ab2021": gen_fats.ofats_unt_ab2021,
}

SYNTHETIC_SIDE_TABLES = (
    "_register",
    "_register_panel",
    "_crm_ground_truth",
    "_lse_variable_bridge",
    "_amdc_regimes",
)
SIDE_TABLES = SYNTHETIC_SIDE_TABLES


@dataclass
class BuildResult:
    tables: dict[str, int]
    seconds: dict[str, float]


def _checked_write(frame: pl.DataFrame, parquet_dir, name: str) -> None:
    check_frame(frame, name)
    frame.write_parquet(parquet_dir / f"{name}.parquet")


def crm_ground_truth(reg: Register) -> pl.DataFrame:
    """Which firm sits in which value chain, and the codes that identify it.

    This is the answer sheet. A reconstruction run on the generated files can be
    scored against it directly: precision and recall over firm-material edges, or over
    the CN8 codes attributed to a material.
    """
    mats = {m.name: m for m in signal_materials()}
    firms = reg.firms.filter(pl.col("crm_material").is_not_null())
    rows = []
    for fid, id6, name, stage, nace, n3 in zip(
        firms["R_ID"], firms["Firmen_ID6"], firms["crm_material"],
        firms["crm_stage"], firms["nace5"], firms["nuts3"],
    ):
        m = mats.get(name)
        if m is None:
            continue
        conc = concentration(m)
        rows.append({
            "R_ID": fid,
            "Firmen_ID6": id6,
            "material": name,
            "stage": stage,
            "nace5": nace,
            "nuts3": n3,
            "critical": m.critical,
            "upstream_cn8": [f"{c:08d}" for c in m.upstream_cn],
            "downstream_cn8": [f"{c:08d}" for c in m.downstream_cn],
            "prodcom8": [f"{c:08d}" for c in m.prodcom("Processing/Refining", "Fabrication")],
            "dominant_suppliers": list(conc),
            "supplier_shares": [round(v, 3) for v in conc.values()],
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"R_ID": pl.Utf8})


def run(cfg: Settings, only: tuple[str, ...] = ()) -> BuildResult:
    parquet_dir = cfg.out_parquet
    parquet_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reg = register.build(cfg)
    seconds = {"_register": time.time() - t0}

    _checked_write(reg.firms, parquet_dir, "_register")
    _checked_write(reg.panel, parquet_dir, "_register_panel")
    _checked_write(crm_ground_truth(reg), parquet_dir, "_crm_ground_truth")
    _checked_write(gen_lse.variable_bridge(), parquet_dir, "_lse_variable_bridge")
    _checked_write(regime_frame(), parquet_dir, "_amdc_regimes")

    tables = {"_register": reg.firms.height, "_register_panel": reg.panel.height}
    for name, fn in GENERATORS.items():
        if only and name not in only:
            continue
        t = time.time()
        df = fn(cfg, reg)
        _checked_write(df, parquet_dir, name)
        tables[name] = df.height
        seconds[name] = time.time() - t
        del df
    return BuildResult(tables=tables, seconds=seconds)
