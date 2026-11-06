"""Generate inward and outward foreign-affiliate statistics."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.crosswalk.metadata import catalogue_variables
from .config import BY_TABLE, Settings
from .distributions import lognormal_with_mean
from .register import Register, year_slice
import src.crosswalk.codelists as cl

# Where Austrian subsidiaries abroad actually sit. Approximate, anchored on the known
# concentration in Germany and the CEE neighbourhood.
SUBSIDIARY_LOCATION: dict[str, float] = {
    "DE": 0.181, "CZ": 0.072, "HU": 0.063, "SK": 0.055, "RO": 0.052, "PL": 0.049,
    "SI": 0.041, "HR": 0.038, "IT": 0.035, "CH": 0.033, "RS": 0.028, "BG": 0.026,
    "US": 0.026, "RU": 0.022, "NL": 0.021, "GB": 0.019, "FR": 0.018, "TR": 0.016,
    "UA": 0.015, "BA": 0.014, "CN": 0.014, "ES": 0.012, "BE": 0.010, "SE": 0.009,
    "AT": 0.008, "BY": 0.007, "MK": 0.007, "AL": 0.006, "DK": 0.006, "IN": 0.006,
    "BR": 0.005, "MX": 0.005, "CA": 0.005, "SG": 0.004, "AE": 0.004, "JP": 0.004,
    "KR": 0.004, "AU": 0.003, "ZA": 0.003, "NO": 0.003, "FI": 0.003, "IE": 0.003,
    "PT": 0.003, "GR": 0.003, "LU": 0.003, "LT": 0.002, "LV": 0.002, "EE": 0.002,
    "CY": 0.002, "MT": 0.002, "VN": 0.002, "TH": 0.002, "ID": 0.002, "MY": 0.002,
}

_EU_MEMBERS_2021 = set(cl.EU27_2020)


def _pick(rng, mix: dict[str, float], n: int) -> np.ndarray:
    keys = list(mix)
    w = np.asarray(list(mix.values()), dtype=float)
    return np.asarray(keys)[rng.choice(len(keys), size=n, p=w / w.sum())]


def _nace_group(nace5: np.ndarray) -> pl.Series:
    """OENACE group, the 3-digit level FATS reports."""
    return pl.Series([f"{c // 100:03d}" for c in nace5], dtype=pl.Utf8)


def _ifats(cfg: Settings, reg: Register, table: str) -> pl.DataFrame:
    names = catalogue_variables(table)
    post = table.endswith("ab2021")
    rng = np.random.default_rng(cfg.seed + (708 if post else 707))
    year_key = "Jahr"
    frames = []
    for year in BY_TABLE[table].years:
        sub = year_slice(reg, year, "foreign_controlled", nearest=True)
        n = sub.height
        if not n:
            continue
        emp = sub["employees"].to_numpy().astype(float)
        turnover = lognormal_with_mean(1.0, 1.0, n, rng) * (25.0 + 190.0 * emp)
        personnel = np.where(emp > 0, emp * lognormal_with_mean(1.0, 0.3, n, rng) * 58.0, 0.0)
        prod_value = turnover * rng.uniform(0.6, 1.15, n)
        value_added = prod_value * rng.uniform(0.15, 0.55, n)
        invest = turnover * rng.uniform(0.0, 0.11, n)
        rd_flag = rng.random(n) < 0.17
        cols = {
            year_key: pl.Series(np.full(n, f"{year}"), dtype=pl.Utf8),
            "NACE": _nace_group(sub["nace5"].to_numpy()),
            "NACEAB": sub["nace_section"],
            "NUTS3": pl.Series([f"{v}" for v in sub["nuts3"].to_numpy()], dtype=pl.Utf8),
            "KONTRLAND": sub["parent_country"],
            "BESCH": pl.Series(np.round(emp, 1)),
            "UMSATZ": pl.Series(np.round(turnover, 1)),
            "PERSA": pl.Series(np.round(personnel, 1)),
            "PW": pl.Series(np.round(prod_value, 1)),
            "ISACH": pl.Series(np.round(invest, 1)),
            "FuE_AUSG": pl.Series(np.round(np.where(rd_flag, turnover * rng.uniform(0.002, 0.06, n),
                                                    np.nan), 1)),
            "FuE_BESCH": pl.Series(np.round(np.where(rd_flag, emp * rng.uniform(0.01, 0.15, n),
                                                     np.nan), 1)),
        }
        # BWSF before 2021, BWS from 2021: the same concept under two names
        cols["BWS" if post else "BWSF"] = pl.Series(np.round(value_added, 1))
        frames.append(pl.DataFrame({"R_ID": sub["R_ID"]}).hstack(
            pl.DataFrame({k: cols[k] for k in names})))
    return pl.concat(frames).select(["R_ID", *names])


def _ofats(cfg: Settings, reg: Register, table: str) -> pl.DataFrame:
    """Austrian-controlled units abroad, one row per foreign subsidiary and year."""
    names = catalogue_variables(table)
    on_id3 = table.startswith("ofats_unt")
    rng = np.random.default_rng(cfg.seed + (810 if on_id3 else 809))
    key = "Firmen_ID3" if on_id3 else "R_ID"
    frames = []
    for year in BY_TABLE[table].years:
        parents = year_slice(reg, year, "has_foreign_sub", nearest=True)
        if parents.height == 0:
            continue
        # a parent typically holds several foreign units
        reps = 1 + rng.poisson(1.7, parents.height)
        idx = np.repeat(np.arange(parents.height), reps)
        sub = parents[idx]
        n = sub.height
        seat = _pick(rng, SUBSIDIARY_LOCATION, n)
        # the foreign unit's activity is related to the parent's but not identical
        drift = rng.random(n) < 0.35
        nace_for = np.where(drift,
                            parents["nace5"].to_numpy()[
                                rng.integers(0, parents.height, n)],
                            sub["nace5"].to_numpy())
        emp = np.round(np.abs(rng.lognormal(3.0, 1.5, n)), 1)
        turnover = np.round(lognormal_with_mean(1.0, 1.2, n, rng) * (40.0 + 160.0 * emp), 1)
        # PERSA_AUE and ISACH_AUE are only collected from reporting year 2021
        collected = year >= 2021
        cols = {
            "Jahr": pl.Series(np.full(n, f"{year}"), dtype=pl.Utf8),
            "KONTRLAND": pl.Series(np.where(sub["parent_country"].to_numpy() == None,  # noqa: E711
                                            "AT", sub["parent_country"].to_numpy()),
                                   dtype=pl.Utf8),
            "NUTS2_INLMUTTER": sub["nuts2"],
            "NACE_INLMUTTER": _nace_group(sub["nace5"].to_numpy()),
            "NACEAB_INLMUTTER": sub["nace_section"],
            "SITZ_AUE": pl.Series(seat, dtype=pl.Utf8),
            "SITZ_IN_EU": pl.Series(np.where(np.isin(seat, list(_EU_MEMBERS_2021)), "J", "N"),
                                    dtype=pl.Utf8),
            "NACE_AUE": _nace_group(nace_for),
            "NACEAB_AUE": sub["nace_section"],
            "BESCH_AUE": pl.Series(emp),
            "UMSATZ_AUE": pl.Series(turnover),
            "PERSA_AUE": pl.Series(np.round(np.where(collected, emp * rng.uniform(30, 75, n),
                                                     np.nan), 1)),
            "ISACH_AUE": pl.Series(np.round(np.where(collected, turnover * rng.uniform(0, 0.1, n),
                                                     np.nan), 1)),
        }
        frames.append(pl.DataFrame({key: sub[key]}).hstack(
            pl.DataFrame({k: cols[k] for k in names})))
    return pl.concat(frames).select([key, *names])


def ifats_re_bis2020(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _ifats(cfg, reg, "ifats_re_bis2020")


def ifats_re_ab2021(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _ifats(cfg, reg, "ifats_re_ab2021")


def ofats_re(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _ofats(cfg, reg, "ofats_re")


def ofats_unt_ab2021(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _ofats(cfg, reg, "ofats_unt_ab2021")
