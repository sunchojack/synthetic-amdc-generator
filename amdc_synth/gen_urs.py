"""The six Unternehmensregister Final Frame files.

Reproducing the register matters for one specific reason. The data request needs
OFATS-UNT, which is keyed on Firmen_ID3, joined to everything else, which is keyed on
R_ID, and says the bridge "requires AMDC guidance". The register is that
bridge, and its published statistics say exactly how far it reaches:
URS-STAT exists only for 2018-2021, and inside those years it covers 99% of the
frame. So the gap is not a coverage failure, it is a date range - 834,511 of the
1,038,870 legal units ever appear in it, because the rest are only in the frame
before 2018 or after 2021. That is generated here as it is, so the linkage has to be
solved on this data the same way it will have to be solved at AMDC.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.crosswalk.metadata import load_batch
from src.crosswalk.schema import (
    URS_D_COLUMNS,
    URS_G_COLUMNS,
    URS_LOC_COLUMNS,
    URS_M_COLUMNS,
    URS_RE_COLUMNS,
    URS_STAT_COLUMNS,
)
from .config import Settings
from .distributions import (HeavyTail, allocate, ensure_all_levels,
                            lognormal_with_mean)
from .register import Register, year_slice
import src.crosswalk.codelists as cl


def _dates(rng, years: np.ndarray) -> pl.Series:
    """ISO date strings inside the given years. AMDC stores these as character;
    the exact layout is not documented in the codebook, so ISO is a choice."""
    month = rng.integers(1, 13, years.shape[0])
    day = rng.integers(1, 29, years.shape[0])
    return pl.Series([f"{y:04d}-{m:02d}-{d:02d}" for y, m, d in zip(years, month, day)],
                     dtype=pl.Utf8)


def _blank(series: pl.Series, rate: float, rng) -> pl.Series:
    """Null out a share of the values to match a published missing rate."""
    if rate <= 0:
        return series
    mask = rng.random(series.len()) < rate
    return series.scatter(np.flatnonzero(mask), None)


def urs_re(cfg: Settings, reg: Register) -> pl.DataFrame:
    b = load_batch("lse_re")
    rng = np.random.default_rng(cfg.seed + 11)
    frames = []
    for yi, year in enumerate(int(y) for y in b["kzr"].years):
        sub = year_slice(reg, year)
        n = sub.height
        if not n:
            continue
        emp = sub["employees"].to_numpy().astype(float)
        # VAT base scales with employment, heavy-tailed within size
        ust = np.round(lognormal_with_mean(1.0, 1.9, n, rng) * (1.0 + emp) * 62_000.0)
        df = pl.DataFrame({
            "R_ID": sub["R_ID"],
            "jahr": np.full(n, year, dtype=np.int32),
            "IS": _blank(sub["is_code"], b["is"].rate(year), rng),
            "RF": sub["rechtsform"],
            "oenace": sub["nace5"].cast(pl.Int32),
            "FSW_NUTS3": _blank(sub["nuts3"].cast(pl.Int32), b["fsw_nuts3"].rate(year), rng),
            "FSW_LKZ": pl.Series(np.where(rng.random(n) < 0.0001, "DE", "AT"), dtype=pl.Utf8),
            "gcd_fsw": _blank(sub["gem"].cast(pl.Int32), b["gcd_fsw"].rate(year), rng),
            "USB": _blank(pl.Series(np.round(emp * rng.uniform(0.75, 0.98, n), 1)),
                          b["usb"].rate(year), rng),
            "SB": _blank(pl.Series(np.round(rng.uniform(0.5, 2.0, n), 2)),
                         b["sb"].rate(year), rng),
            "UST": _blank(pl.Series(ust.astype(np.int64)), b["ust"].rate(year), rng),
            "RE_DB_DAT_NG": _blank(_dates(rng, sub["birth_year"].to_numpy()),
                                   b["re_db_dat_ng"].rate(year), rng),
            "RE_DB_DAT_SL": _blank(_dates(rng, np.where(sub["close_year"].to_numpy() > 0,
                                                        sub["close_year"].to_numpy(), year)),
                                   b["re_db_dat_sl"].rate(year), rng),
        })
        frames.append(df)
    return pl.concat(frames).select(URS_RE_COLUMNS)


def urs_loc(cfg: Settings, reg: Register) -> pl.DataFrame:
    b = load_batch("lse_loc")
    rng = np.random.default_rng(cfg.seed + 12)
    rows_by_year = b.rows_per_year()
    frames = []
    for year in (int(y) for y in b["kzs"].years):
        sub = year_slice(reg, year)
        if not sub.height:
            continue
        target = cfg.rows(rows_by_year[year])
        weights = sub["n_local_units"].to_numpy().astype(float)
        if target >= sub.height:
            reps = 1 + allocate(target - sub.height, weights / weights.sum())
        else:
            reps = allocate(target, weights / weights.sum())
        idx = np.repeat(np.arange(sub.height), reps)
        slot = np.concatenate([np.arange(count) for count in reps if count])
        n = idx.shape[0]
        # local units of one legal unit share its region most of the time
        moved = rng.random(n) < 0.12
        n3 = sub["nuts3"].to_numpy()[idx]
        other = np.array(sorted(cl.NUTS3))[rng.integers(0, len(cl.NUTS3), n)]
        n3 = np.where(moved, other, n3)
        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"].to_numpy()[idx],
            "Firmen_ID5": sub["Firmen_ID5"].to_numpy()[idx] + slot,
            "jahr": np.full(n, year, dtype=np.int32),
            "oenace_s": sub["nace5"].to_numpy()[idx].astype(np.int32),
            "S_NUTS3": _blank(pl.Series(n3.astype(np.int32)), b["s_nuts3"].rate(year), rng),
            "S_LKZ": pl.Series(np.where(rng.random(n) < 0.00006, "DE", "AT"), dtype=pl.Utf8),
            "gcd_s": _blank(pl.Series(sub["gem"].to_numpy()[idx].astype(np.int32)),
                            b["gcd_s"].rate(year), rng),
        }))
    return pl.concat(frames).select(URS_LOC_COLUMNS)


def urs_stat(cfg: Settings, reg: Register) -> pl.DataFrame:
    """Legal unit to statistical enterprise. Exists for 2018-2021 only."""
    b = load_batch("lse_stat")
    rng = np.random.default_rng(cfg.seed + 13)
    frames = []
    for yi, year in enumerate(int(y) for y in b["kzr"].years):
        sub = year_slice(reg, year)
        if not sub.height:
            continue
        # inside its year range the file covers virtually the whole frame
        target = min(cfg.firms(b["kzr"].unique_per_year[yi]), sub.height)
        keep = rng.permutation(sub.height)[:target]
        sub = sub[keep]
        n = sub.height
        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"],
            "Firmen_ID3": sub["Firmen_ID3"],
            "JAHR": np.full(n, year, dtype=np.int32),
            "UNT_IS": _blank(sub["is_code"], b["UNT_IS"].rate(year), rng),
            "UNT_OENACE": _blank(sub["nace5"].cast(pl.Int32), b["UNT_OENACE"].rate(year), rng),
        }))
    return pl.concat(frames).select(URS_STAT_COLUMNS)


def _related_unit(rng, sub: pl.DataFrame, prefix: str, year: int, b, country_mix,
                  rf_mix, emp_stats, country_column: str | None = None) -> pl.DataFrame:
    """One related-unit block (parent or subsidiary) with its own attributes.

    `country_column` names a register column holding the country already assigned to
    this relationship. Redrawing it here instead would leave URS-M and IFATS-RE
    disagreeing about which firms are foreign-controlled, and cross-checking those two
    is exactly what the ownership part of the data request does.
    """
    n = sub.height
    lkz = np.asarray(list(country_mix))[
        rng.choice(len(country_mix),
                   size=n,
                   p=np.asarray(list(country_mix.values()), dtype=float)
                   / sum(country_mix.values()))]
    if country_column is not None:
        assigned = sub[country_column].to_numpy()
        lkz = np.where(assigned == None, lkz, assigned)  # noqa: E711
    domestic = lkz == "AT"
    emp_fit = HeavyTail.fit(emp_stats.numeric)
    emp = np.round(emp_fit.sample(n, rng), 1)
    nace = sub["nace5"].to_numpy().astype(np.int32)
    n3 = sub["nuts3"].to_numpy().astype(np.int32)
    gem = sub["gem"].to_numpy().astype(np.int32)
    out = {
        f"{prefix}_IS": _blank(sub["is_code"], b[f"{prefix.lower()}_is"].rate(year), rng),
        f"{prefix}_RF": pl.Series(
            np.where(domestic,
                     np.asarray(list(rf_mix))[rng.choice(
                         len(rf_mix), size=n,
                         p=np.asarray(list(rf_mix.values()), dtype=float) / sum(rf_mix.values()))],
                     "AUS"),
            dtype=pl.Utf8),
        f"{prefix}_OENACE": pl.Series(np.where(domestic, nace, -1), dtype=pl.Int32).replace(-1, None),
        f"{prefix}_FSW_NUTS3": pl.Series(np.where(domestic, n3, -1), dtype=pl.Int32).replace(-1, None),
        f"{prefix}_FSW_LKZ": pl.Series(lkz, dtype=pl.Utf8),
        f"{prefix}_FSW_GEMNR": pl.Series(np.where(domestic, gem, -1), dtype=pl.Int32).replace(-1, None),
        f"{prefix}_USB": _blank(pl.Series(emp), b[f"{prefix.lower()}_usb"].rate(year), rng),
        f"{prefix}_SB": _blank(pl.Series(np.round(rng.uniform(0.3, 2.0, n), 2)),
                               b[f"{prefix.lower()}_sb"].rate(year), rng),
        f"{prefix}_UST": _blank(
            pl.Series(np.round(lognormal_with_mean(1.0, 2.2, n, rng)
                               * (1.0 + emp) * 70_000.0).astype(np.int64)),
            b[f"{prefix.lower()}_ust"].rate(year), rng),
        f"{prefix}_DB_DAT_NG": _blank(_dates(rng, sub["birth_year"].to_numpy()),
                                      b[f"{prefix.lower()}_db_dat_ng"].rate(year), rng),
        f"{prefix}_DB_DAT_SL": _blank(_dates(rng, np.full(n, year)),
                                      b[f"{prefix.lower()}_db_dat_sl"].rate(year), rng),
    }
    return pl.DataFrame(out)


def urs_m(cfg: Settings, reg: Register) -> pl.DataFrame:
    """Controlling parent per legal unit."""
    b = load_batch("lse_m")
    rng = np.random.default_rng(cfg.seed + 14)
    frames = []
    for yi, year in enumerate(int(y) for y in b["kzr"].years):
        sub = year_slice(reg, year)
        if not sub.height:
            continue
        target = min(cfg.firms(b["kzr"].unique_per_year[yi]), sub.height)
        priority = sub["has_parent"].cast(pl.Int8).to_numpy() + rng.random(sub.height)
        sub = sub[np.argpartition(-priority, target - 1)[:target]]
        n = sub.height
        block = _related_unit(rng, sub, "M", year, b, cl.PARENT_COUNTRY,
                              cl.RECHTSFORM, b["m_usb"], country_column="parent_country")
        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"],
            "jahr": np.full(n, year, dtype=np.int32),
        }).hstack(block))
    return pl.concat(frames).select(URS_M_COLUMNS)


def urs_d(cfg: Settings, reg: Register) -> pl.DataFrame:
    """Controlled subsidiaries per legal unit; several rows per parent."""
    b = load_batch("lse_d")
    rng = np.random.default_rng(cfg.seed + 15)
    rows_by_year = b.rows_per_year()
    frames = []
    for yi, year in enumerate(int(y) for y in b["kzr"].years):
        sub = year_slice(reg, year, "has_subsidiary")
        if not sub.height:
            continue
        target = min(cfg.firms(b["kzr"].unique_per_year[yi]), sub.height)
        sub = sub[rng.permutation(sub.height)[:target]]
        # one row per subsidiary and year: the published row count for the year is the
        # number of distinct subsidiaries, spread over the controlling units
        n_rows = cfg.rows(rows_by_year[year])
        w = sub["activity_weight"].to_numpy()
        idx = np.repeat(np.arange(sub.height), allocate(n_rows, w / w.sum()))
        sub = sub[idx]
        n = sub.height
        block = _related_unit(rng, sub, "T", year, b, cl.PARENT_COUNTRY,
                              cl.RECHTSFORM, b["t_usb"])
        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"],
            "jahr": np.full(n, year, dtype=np.int32),
        }).hstack(block))
    return pl.concat(frames).select(URS_D_COLUMNS)


def urs_g(cfg: Settings, reg: Register) -> pl.DataFrame:
    """Enterprise group per legal unit. 2018-2021 only."""
    b = load_batch("lse_g")
    rng = np.random.default_rng(cfg.seed + 16)
    art_keys = list(cl.GRP_ART)
    art_w = np.asarray(list(cl.GRP_ART.values()), dtype=float)
    art_w /= art_w.sum()
    lkz_keys = list(cl.GROUP_COUNTRY)
    lkz_w = np.asarray(list(cl.GROUP_COUNTRY.values()), dtype=float)
    lkz_w /= lkz_w.sum()
    frames = []
    for yi, year in enumerate(int(y) for y in b["kzr"].years):
        sub = year_slice(reg, year)
        if not sub.height:
            continue
        target = min(cfg.firms(b["kzr"].unique_per_year[yi]), sub.height)
        priority = sub["in_group"].cast(pl.Int8).to_numpy() + rng.random(sub.height)
        sub = sub[np.argpartition(-priority, target - 1)[:target]]
        n = sub.height
        lkz = ensure_all_levels(np.asarray(lkz_keys)[rng.choice(len(lkz_keys), n, p=lkz_w)],
                                lkz_keys, rng)
        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"],
            "Firmen_ID4": sub["Firmen_ID4"],
            "JAHR": np.full(n, year, dtype=np.int32),
            "GRP_ART": pl.Series(np.asarray(art_keys)[rng.choice(2, n, p=art_w)], dtype=pl.Utf8),
            "GRP_LKZ": _blank(pl.Series(lkz, dtype=pl.Utf8), b["GRP_LKZ"].rate(year), rng),
            "GRP_DB_DAT_NG": _blank(_dates(rng, sub["birth_year"].to_numpy()),
                                    b["GRP_DB_DAT_NG"].rate(year), rng),
            "GRP_DB_DAT_SL": _blank(_dates(rng, np.full(n, year)),
                                    b["GRP_DB_DAT_SL"].rate(year), rng),
        }))
    return pl.concat(frames).select(URS_G_COLUMNS)
