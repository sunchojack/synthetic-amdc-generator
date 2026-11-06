"""Generate both LSE-RE variable layouts around the 2021 schema break."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.crosswalk.metadata import catalogue_variables
from .config import BY_TABLE, Settings
from .distributions import lognormal_with_mean
from .register import Register, year_slice
from src.crosswalk.lse_bridge import frame as lse_bridge_frame

# Which questionnaire block a firm answers, by OENACE section.
BLOCK_BY_SECTION = {
    **{s: "P" for s in "BCDEF"},
    **{s: "D" for s in "GHIJLMNPQRS"},
    "K": "K",
}

# Variables in the pre-2021 file that are not part of a questionnaire block.
_CORE_PRE = {
    "Jahr", "NACE", "NUTS3", "GEM", "INSGES", "INSGWEI", "SBINS", "SBWEIB", "USBINS",
    "USBWEIB", "ANGINS", "ANGWEIB", "ARBINS", "ARBWEIB", "LEHRINS", "LEHRWEI", "TEILZ",
    "GERINGF", "GERWEIB", "HAINS", "HAW", "VZE", "ARBSTD", "EGES", "UMSATZ", "LUG",
    "PERSA", "AGES", "PW", "BWSF", "BGWSP", "BBUE", "BEZWD", "BEZWDVK",
}
_CORE_POST = {
    "JAHR", "OENACE", "NUTS3", "GEMNR", "RF", "HERK", "BESCHI", "BESCHW", "SBI", "SBW",
    "USBI", "USBW", "ANGI", "ANGW", "ARBI", "ARBW", "LEHRLI", "LEHRLW", "TEILZ",
    "GERINGI", "GERINGW", "VZE", "ARBSTD", "EGESAMT", "UMSATZ", "PLUG", "PPERSA",
    "AGESAMT", "AVORL", "PW", "BWS", "BGWSP", "BBUE", "BEZWD", "BEZWDWVK",
}

def variable_bridge() -> pl.DataFrame:
    """Official 2020/2021 variable key, preserved with merge recipes and drops."""
    return lse_bridge_frame()


def _economics(rng, sub: pl.DataFrame, year: int) -> dict[str, np.ndarray]:
    """A coherent set of annual accounts per firm, in 1,000 EUR."""
    n = sub.height
    emp = sub["employees"].to_numpy().astype(float)
    section = sub["nace_section"].to_numpy()
    manuf = np.isin(section, list("BCDEF"))
    trade = section == "G"

    # labour productivity per head, wider in services, with a year drift
    drift = 1.0 + 0.021 * (year - 2013)
    turnover = lognormal_with_mean(1.0, 0.9, n, rng) * (34.0 + 430.0 * emp) * drift
    wage_bill = np.where(emp > 0,
                         emp * lognormal_with_mean(1.0, 0.30, n, rng) * 52.0 * drift,
                         0.0)
    personnel = wage_bill * (1.0 + rng.uniform(0.26, 0.34, n))

    subsidies = np.where(rng.random(n) < 0.14, turnover * rng.uniform(0.001, 0.05, n), 0.0)
    other_income = turnover * rng.uniform(0.0, 0.09, n)
    total_income = turnover + other_income + subsidies

    resale_share = np.where(trade, rng.uniform(0.55, 0.95, n),
                            np.where(manuf, rng.uniform(0.0, 0.22, n), rng.uniform(0.0, 0.35, n)))
    resale_revenue = turnover * resale_share
    production_revenue = np.where(manuf, turnover - resale_revenue, 0.0)
    mining_revenue = np.where(section == "B", turnover * rng.uniform(0.5, 1.0, n), 0.0)

    resale_cost = resale_revenue * rng.uniform(0.72, 0.93, n)
    material = np.where(manuf, turnover * rng.uniform(0.42, 0.66, n),
                        turnover * rng.uniform(0.16, 0.46, n))
    energy = turnover * rng.uniform(0.004, 0.045, n)
    depreciation = turnover * rng.uniform(0.008, 0.075, n)
    interest = turnover * rng.uniform(0.0, 0.028, n)
    taxes = turnover * rng.uniform(0.001, 0.017, n)
    other_expense = turnover * rng.uniform(0.03, 0.19, n)
    total_expense = (personnel + resale_cost + material + energy + depreciation
                     + interest + taxes + other_expense)

    # The three national-accounts aggregates are built to satisfy their own identity,
    # PW = BWS + AVORL, because the data request computes intermediate consumption as
    # PW - BWSF for the pre-2021 series. If the identity does not hold in the fixture,
    # that residual is noise and the code path cannot be tested against it. Goods bought
    # for resale are netted out of the production value, so they stay out of AVORL.
    stock_change = turnover * rng.normal(0.0, 0.02, n)
    production_value = turnover - resale_cost + stock_change
    intermediate = np.minimum(material + energy + other_expense,
                              production_value * 0.94)
    value_added = production_value - intermediate
    operating_surplus = value_added - personnel
    trading_margin = resale_revenue - resale_cost

    stock_material = material * rng.uniform(0.03, 0.30, n)
    stock_fuel = energy * rng.uniform(0.02, 0.25, n)
    stock_resale = resale_cost * rng.uniform(0.03, 0.28, n)
    stock_wip = np.where(manuf, production_value * rng.uniform(0.0, 0.13, n), 0.0)
    stock_finished = np.where(manuf, production_value * rng.uniform(0.01, 0.16, n), 0.0)

    investment_tangible = turnover * rng.uniform(0.0, 0.11, n)
    investment_machinery = investment_tangible * rng.uniform(0.15, 0.85, n)
    investment_rights = turnover * rng.uniform(0.0, 0.012, n)

    return {
        "emp": emp, "manuf": manuf,
        "turnover": turnover, "total_income": total_income, "wage_bill": wage_bill,
        "personnel": personnel, "subsidies": subsidies, "resale_revenue": resale_revenue,
        "production_revenue": production_revenue, "mining_revenue": mining_revenue,
        "resale_cost": resale_cost, "material": material, "energy": energy,
        "depreciation": depreciation, "interest": interest, "taxes": taxes,
        "total_expense": total_expense, "intermediate": intermediate,
        "production_value": production_value, "value_added": value_added,
        "operating_surplus": operating_surplus, "trading_margin": trading_margin,
        "stock_material": stock_material, "stock_fuel": stock_fuel,
        "stock_resale": stock_resale, "stock_wip": stock_wip,
        "stock_finished": stock_finished,
        "investment_total": investment_tangible + investment_rights,
        "investment_tangible": investment_tangible,
        "investment_machinery": investment_machinery,
        "investment_rights": investment_rights,
    }


def _employment_block(rng, e, n) -> dict[str, np.ndarray]:
    emp = e["emp"]
    female = np.round(emp * rng.uniform(0.18, 0.62, n), 1)
    self_emp = np.round(np.minimum(emp, rng.uniform(0.0, 1.6, n)), 1)
    employed = np.round(np.maximum(emp - self_emp, 0.0), 1)
    white = np.round(employed * rng.uniform(0.2, 0.8, n), 1)
    return {
        "total": np.round(emp, 1), "female": female,
        "self_emp": self_emp, "self_emp_f": np.round(self_emp * 0.35, 1),
        "employed": employed, "employed_f": np.round(employed * 0.42, 1),
        "white": white, "white_f": np.round(white * 0.55, 1),
        "blue": np.round(employed - white, 1), "blue_f": np.round((employed - white) * 0.22, 1),
        "apprentice": np.round(employed * rng.uniform(0.0, 0.07, n), 1),
        "apprentice_f": np.round(employed * rng.uniform(0.0, 0.03, n), 1),
        "parttime": np.round(employed * rng.uniform(0.05, 0.55, n), 1),
        "marginal": np.round(employed * rng.uniform(0.0, 0.12, n), 1),
        "marginal_f": np.round(employed * rng.uniform(0.0, 0.07, n), 1),
        "home": np.round(employed * rng.uniform(0.0, 0.02, n), 1),
        "home_f": np.round(employed * rng.uniform(0.0, 0.01, n), 1),
        "fte": np.round(employed * rng.uniform(0.72, 0.99, n), 1),
        "hours": np.round(employed * rng.uniform(1300, 1750, n)).astype(np.int64),
    }


def _filler(rng, base: np.ndarray, n: int, null_rate: float) -> pl.Series:
    """A plausible detail line: a small share of the aggregate, mostly not applicable."""
    vals = np.round(base * rng.uniform(0.0, 0.14, n), 1)
    out = pl.Series(vals)
    mask = rng.random(n) < null_rate
    return out.scatter(np.flatnonzero(mask), None)


def _build(cfg: Settings, reg: Register, table: str) -> pl.DataFrame:
    post = table.endswith("ab2021")
    names = catalogue_variables(table)
    core = _CORE_POST if post else _CORE_PRE
    rng = np.random.default_rng(cfg.seed + (606 if post else 505))
    frames = []
    for year in BY_TABLE[table].years:
        sub = year_slice(reg, year, nearest=True)
        sub = sub.filter(pl.col("in_lse_scope"))
        n = sub.height
        if not n:
            continue
        e = _economics(rng, sub, year)
        emp = _employment_block(rng, e, n)
        block = np.array([BLOCK_BY_SECTION.get(s, "D") for s in sub["nace_section"].to_numpy()])
        # credit and insurance both sit in section K; split them
        insurer = (sub["nace_section"].to_numpy() == "K") & (rng.random(n) < 0.25)
        block = np.where(insurer, "V", block)

        cols: dict[str, pl.Series] = {}
        named = _named_columns(post, year, sub, e, emp, n, rng)
        for name in names:
            if name in named:
                series = named[name]
            elif name in core:
                series = _filler(rng, e["turnover"], n, 0.30)
            else:
                base = e["turnover"] if "ETAB" in name else e["total_expense"]
                series = _filler(rng, base, n, 0.72)
            # a firm answers one questionnaire block; the others are null for it, and
            # that is where most of the missingness in a 194-column file comes from
            prefix = name[0] if len(name) > 1 and name[1] == "_" else ""
            if prefix in ("P", "D", "K", "V") and name not in core:
                series = series.scatter(np.flatnonzero(block != prefix), None)
            cols[name] = series
        frames.append(pl.DataFrame({"R_ID": sub["R_ID"]}).hstack(
            pl.DataFrame({k: v for k, v in cols.items()})))
    return pl.concat(frames).select(["R_ID", *names])


def _named_columns(post, year, sub, e, emp, n, rng) -> dict[str, pl.Series]:
    """The variables the data request names, filled coherently."""
    r = lambda a: pl.Series(np.round(a, 1))  # noqa: E731
    common = {
        "NUTS3": pl.Series([f"{v}" for v in sub["nuts3"].to_numpy()], dtype=pl.Utf8),
        "VZE": r(emp["fte"]),
        "ARBSTD": pl.Series(emp["hours"], dtype=pl.Int64),
        "UMSATZ": r(e["turnover"]),
        "TEILZ": r(emp["parttime"]),
        "PW": r(e["production_value"]),
        "BGWSP": r(e["trading_margin"]),
        "BBUE": r(e["operating_surplus"]),
        "IGES": r(e["investment_total"]),
        "ISACH": r(e["investment_tangible"]),
    }
    if not post:
        return {
            **common,
            "Jahr": pl.Series(np.full(n, f"{year}"), dtype=pl.Utf8),
            "NACE": pl.Series([f"{c:05d}" for c in sub["nace5"].to_numpy()], dtype=pl.Utf8),
            "GEM": pl.Series([f"{g:05d}" for g in sub["gem"].to_numpy()], dtype=pl.Utf8),
            "INSGES": r(emp["total"]), "INSGWEI": r(emp["female"]),
            "SBINS": r(emp["self_emp"]), "SBWEIB": r(emp["self_emp_f"]),
            "USBINS": r(emp["employed"]), "USBWEIB": r(emp["employed_f"]),
            "ANGINS": r(emp["white"]), "ANGWEIB": r(emp["white_f"]),
            "ARBINS": r(emp["blue"]), "ARBWEIB": r(emp["blue_f"]),
            "LEHRINS": r(emp["apprentice"]), "LEHRWEI": r(emp["apprentice_f"]),
            "GERINGF": r(emp["marginal"]), "GERWEIB": r(emp["marginal_f"]),
            "HAINS": r(emp["home"]), "HAW": r(emp["home_f"]),
            "EGES": r(e["total_income"]), "LUG": r(e["wage_bill"]),
            "PERSA": r(e["personnel"]), "AGES": r(e["total_expense"]),
            "BWSF": r(e["value_added"]),
            "P_ETAB01": r(e["turnover"]),
            "P_ETAB03": r(e["production_revenue"]),
            "P_ETAB04": r(e["resale_revenue"]),
            "P_ETAB09": r(e["subsidies"]),
            "P_ETAB10": r(e["subsidies"] * rng.uniform(0.0, 0.5, n)),
            "P_ATAB01": r(e["energy"]),
            "P_ATAB02": r(e["material"]),
            "P_ATAB05": r(e["resale_cost"]),
            "P_ATAB14": r(e["interest"]),
            "P_ATAB15": r(e["depreciation"]),
            "P_ATAB16": r(e["taxes"]),
            "P_ATAB17": r(e["taxes"] * rng.uniform(0.0, 0.4, n)),
            "LAGERV1": r(e["stock_fuel"]), "LAGERV2": r(e["stock_material"]),
            "LAGERV3": r(e["stock_resale"]), "LAGERV4": r(e["stock_wip"]),
            "LAGERV5": r(e["stock_finished"]),
            "LAGERV6": r(e["stock_fuel"] + e["stock_material"] + e["stock_resale"]
                         + e["stock_wip"] + e["stock_finished"]),
            "ITAB04": r(e["investment_machinery"]),
            "ITAB11": r(e["investment_rights"]),
        }
    return {
        **common,
        "JAHR": pl.Series(np.full(n, f"{year}"), dtype=pl.Utf8),
        "OENACE": pl.Series([f"{c:05d}" for c in sub["nace5"].to_numpy()], dtype=pl.Utf8),
        "GEMNR": pl.Series([f"{g:05d}" for g in sub["gem"].to_numpy()], dtype=pl.Utf8),
        "RF": sub["rechtsform"],
        "HERK": pl.Series(np.where(sub["foreign_controlled"].to_numpy(), "A", "I"), dtype=pl.Utf8),
        "BESCHI": r(emp["total"]), "BESCHW": r(emp["female"]),
        "SBI": r(emp["self_emp"]), "SBW": r(emp["self_emp_f"]),
        "USBI": r(emp["employed"]), "USBW": r(emp["employed_f"]),
        "ANGI": r(emp["white"]), "ANGW": r(emp["white_f"]),
        "ARBI": r(emp["blue"]), "ARBW": r(emp["blue_f"]),
        "LEHRLI": r(emp["apprentice"]), "LEHRLW": r(emp["apprentice_f"]),
        "GERINGI": r(emp["marginal"]), "GERINGW": r(emp["marginal_f"]),
        "EGESAMT": r(e["total_income"]), "PLUG": r(e["wage_bill"]),
        "PPERSA": r(e["personnel"]), "AGESAMT": r(e["total_expense"]),
        "AVORL": r(e["intermediate"]),
        "BWS": r(e["value_added"]),
        "EPROD": r(e["production_revenue"]),
        "EGH": r(e["resale_revenue"]),
        "ESUBV": r(e["subsidies"]),
        "EBERG": pl.Series(np.round(e["mining_revenue"], 1)).scatter(
            np.flatnonzero(e["mining_revenue"] <= 0), None),
        "ABRENN": r(e["energy"]),
        "AMATEINS": r(e["material"]),
        "AHW": r(e["resale_cost"]),
        "AZINS": r(e["interest"]),
        "AABSCHR": r(e["depreciation"]),
        "ASTEU": r(e["taxes"]),
        "LVJBRENN": r(e["stock_fuel"]), "LVJROHST": r(e["stock_material"]),
        "LVJHW": r(e["stock_resale"]), "LVJUNFERT": r(e["stock_wip"]),
        "LVJFERTIG": r(e["stock_finished"]),
        "LVJGES": r(e["stock_fuel"] + e["stock_material"] + e["stock_resale"]
                    + e["stock_wip"] + e["stock_finished"]),
        "IMASCH": r(e["investment_machinery"]),
        "IKONZESS": r(e["investment_rights"]),
    }


def lse_re_bis2020(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _build(cfg, reg, "lse_re_bis2020")


def lse_re_ab2021(cfg: Settings, reg: Register) -> pl.DataFrame:
    return _build(cfg, reg, "lse_re_ab2021")
