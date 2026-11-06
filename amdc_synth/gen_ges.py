"""Generate establishment-level material and energy inputs (GES)."""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import BY_TABLE, Settings
from .distributions import lognormal_with_mean
from .register import Register, year_slice
from .regimes import GES_PUBLIC_UNITS_TARGET
import src.crosswalk.codelists as cl
from src.crosswalk.schema import GES_COLUMNS as COLUMNS
from src.crosswalk.material_catalog import materials as crm_materials

# Input lines a reporting establishment declares per year.
LINES_PER_UNIT_MEAN = 11.0
# Share of a CRM firm's input lines that are its own material.
CRM_LINE_SHARE = 0.30
NOMINAL_GROWTH = 0.02

# Unit codes plausible for a material input line, and for an energy line.
_MATERIAL_UNITS = ("01", "02", "04", "05", "06", "11")
_ENERGY_UNITS = ("06", "07", "25", "37")
_ENERGY_CHAPTERS = {"05", "06", "19", "35"}


def _crm_gvges(year: int) -> dict[str, tuple[str, ...]]:
    """GV-GES codes per material, resolved through the CN8 correspondence."""
    xwalk = cl.gvges_from_cn8(year)
    valid = set(cl.gvges(year))
    out: dict[str, tuple[str, ...]] = {}
    for mat in crm_materials().values():
        codes: set[str] = set()
        for cn in mat.upstream_cn:
            codes.update(c for c in xwalk.get(cn, ()) if c in valid)
        if codes:
            out[mat.name] = tuple(sorted(codes))
    return out


def ges(cfg: Settings, reg: Register) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed + 404)
    frames = []
    baseline_annual_value: float | None = None
    for year in BY_TABLE["ges"].years:
        target_units = cfg.firms(GES_PUBLIC_UNITS_TARGET)
        annual_frame = year_slice(reg, year, nearest=True)
        if annual_frame.height == 0:
            annual_frame = reg.firms
        # The register's stable reporter flag supplies longitudinal continuity. At
        # very small SCALE values, frame churn can leave fewer flagged reporters than
        # the rounded annual target, so supplement from other eligible B-F units.
        pool = (
            annual_frame
            .filter(pl.col("is_manufacturing"))
            .sort(["ges_reporter", "activity_weight"], descending=[True, True])
            .head(target_units)
        )
        if pool.height < target_units:
            raise RuntimeError(
                f"GES {year}: only {pool.height} eligible establishments for "
                f"target {target_units}"
            )
        codes = np.array(cl.gvges(year))
        is_energy = np.array([code[:2] in _ENERGY_CHAPTERS for code in codes])
        crm_map = _crm_gvges(year)

        lines = 1 + rng.poisson(LINES_PER_UNIT_MEAN - 1, pool.height)
        idx = np.repeat(np.arange(pool.height), lines)
        n = idx.shape[0]

        # input mix leans on material goods, with an energy line or two per unit
        pick_energy = rng.random(n) < 0.18
        mat_pool = np.flatnonzero(~is_energy)
        en_pool = np.flatnonzero(is_energy)
        chosen = np.where(pick_energy,
                          en_pool[rng.integers(0, en_pool.size, n)] if en_pool.size else 0,
                          mat_pool[rng.integers(0, mat_pool.size, n)])

        # CRM users report the GV-GES code their material's CN8 lines map to
        names = pool["crm_material"].to_list()
        code_index = {c: i for i, c in enumerate(codes.tolist())}
        for u, name in enumerate(names):
            gv = crm_map.get(name) if name else None
            if not gv:
                continue
            rows = np.flatnonzero(idx == u)
            take = rows[rng.random(rows.size) < CRM_LINE_SHARE]
            if take.size:
                sel = np.asarray(gv)[rng.integers(0, len(gv), take.size)]
                chosen[take] = [code_index[c] for c in sel]
                pick_energy[take] = False

        CODE = codes[chosen]
        EG = np.where(is_energy[chosen], "1", "0")
        units = np.where(is_energy[chosen],
                         np.asarray(_ENERGY_UNITS)[rng.integers(0, len(_ENERGY_UNITS), n)],
                         np.asarray(_MATERIAL_UNITS)[rng.integers(0, len(_MATERIAL_UNITS), n)])
        # "00" means value only; those lines carry no quantity
        value_only = rng.random(n) < 0.06
        EH = np.where(value_only, "00", units)

        size = pool["employees"].to_numpy()[idx].astype(float) + 1.0
        value_factor = lognormal_with_mean(1.0, 1.8, n, rng)
        weighted_factor = value_factor * size
        # AMDC documents WE as EUR 1,000 throughout 2008-2023. Normalize the finite
        # synthetic tail to a shared baseline so neither a single draw nor scaled-down
        # frame churn can manufacture a currency-like break. Nominal drift is modelled,
        # not calibrated to an unpublished AMDC total.
        if baseline_annual_value is None:
            baseline_annual_value = float(size.sum()) * 4.5
        annual_target = baseline_annual_value * (1.0 + NOMINAL_GROWTH) ** (year - 2008)
        WE = np.round(weighted_factor * annual_target / weighted_factor.sum(), 1)
        ME = np.where(value_only, np.nan,
                      np.round(lognormal_with_mean(1.0, 3.0, n, rng) * WE * 12.0, 1))

        frames.append(pl.DataFrame({
            "Firmen_ID6": pool["Firmen_ID6"].to_numpy()[idx],
            "JAHR": pl.Series(np.full(n, f"{year}"), dtype=pl.Utf8),
            "NACE": pl.Series([f"{c:05d}" for c in pool["nace5"].to_numpy()[idx]], dtype=pl.Utf8),
            "CODE": pl.Series(CODE, dtype=pl.Utf8),
            "EG": pl.Series(EG, dtype=pl.Utf8),
            "EH": pl.Series(EH, dtype=pl.Utf8),
            "ME": ME,
            "WE": WE,
        }))
    return pl.concat(frames).select(COLUMNS)
