"""The synthetic Unternehmensregister that every other dataset is drawn from.

AMDC hands out a different pseudonymised identifier per statistical unit type, and
which types a dataset carries decides what can be linked to what:

    R_ID  legal unit                   URS, LSE-RE, IFATS-RE, OFATS-RE, trade, PRODCOM
    Firmen_ID3  statistical enterprise       URS-STAT, LSE-STATU, IFATS-UNT, OFATS-UNT
    Firmen_ID4  enterprise group             URS-G
    Firmen_ID5  local unit (Arbeitsstaette)  URS-LOC, LSE-ARB
    Firmen_ID6  establishment (Betrieb)      GES, PRODCOM, LSE-BET
    Firmen_ID9  trader registration          Intrastat, Extrastat
    Firmen_ID1  PRODCOM reporting unit       PRODCOM

The six identifier spaces are numerically disjoint, so a join on the wrong key
returns nothing rather than silently returning garbage.

Sizing, composition and the identifier cardinalities come from AMDC's published
statistics for the Statistisches Unternehmensregister - Final Frame (oid 134674),
which covers 1,038,870 distinct legal units over 2013-2023. Regional and sectoral
composition are read back out of the published numeric quantiles of `fsw_nuts3` and
`oenace` rather than assumed, and legal form uses the published `rf` level counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.crosswalk.metadata import load_batch
from .config import Settings
from .distributions import (allocate, tilt_to_mean, weights_matching_numeric_quantiles,
                            zipf_weights)
import src.crosswalk.codelists as cl
from .crm import signal_materials

# Identifier spaces, one decade apart so a mis-keyed join cannot accidentally hit.
ID_BASE = {
    "Firmen_ID1": 1_000_000_000,
    "R_ID": 2_000_000_000,
    "Firmen_ID3": 3_000_000_000,
    "Firmen_ID4": 4_000_000_000,
    "Firmen_ID5": 5_000_000_000,
    "Firmen_ID6": 6_000_000_000,
    "Firmen_ID9": 9_000_000_000,
}

# Employee size classes, their share of the legal-unit population, and the within-class
# tail index. URS publishes `usb` only as a corrupted numeric summary, so the shares are
# an assumption. They are set over the whole 1.04 million legal units, most of which are
# sole traders or dormant entities with no employees, so that total employment lands near
# the ~4.5 million people Statistics Austria counts in employment.
SIZE_CLASSES = (
    ("0-9", 0.940, 0, 9, 4.0),
    ("10-49", 0.045, 10, 49, 2.2),
    ("50-249", 0.012, 50, 249, 2.6),
    ("250+", 0.003, 250, 4_000, 10.0),
)

# Distinct reporting units per dataset, from AMDC's published unique counts of the
# firm identifier column (kzr in the trade files, kzz in PRODCOM).
DISTINCT_UNITS = {"intra": 25_725, "extra": 93_976, "prodcom": 15_832}

FOREIGN_SUBSIDIARY_SHARE = 0.30

# GES publishes no unit count; assumed share of the manufacturing population.
GES_SAMPLE_SHARE = 0.055

LSE_SECTIONS = frozenset("BCDEFGHIJKLMN")

# Assumed section shares; AMDC's numeric OENACE summary does not identify this split.
SECTION_SHARE = {
    "A": 0.011, "B": 0.003, "C": 0.082, "D": 0.004, "E": 0.005, "F": 0.108,
    "G": 0.188, "H": 0.048, "I": 0.068, "J": 0.040, "K": 0.031, "L": 0.070,
    "M": 0.142, "N": 0.059, "O": 0.002, "P": 0.021, "Q": 0.058, "R": 0.021,
    "S": 0.068, "U": 0.001,
}

# Institutional-sector domain and weights from the published URS `is` counts.
IS_CODES = {
    "11": 0.60, "14": 0.24, "1311": 0.02, "1312": 0.02, "1313": 0.04,
    "122": 0.01, "125": 0.02, "126": 0.01, "127": 0.01, "128": 0.005,
    "129": 0.005, "15": 0.02, "123": 0.004, "124": 0.003, "UN": 0.002,
    "NR": 0.001,
}


@dataclass
class Register:
    firms: pl.DataFrame
    panel: pl.DataFrame          # one row per legal unit and year it is in the frame
    materials: list[str]


def _weighted_choice(rng, keys, weights, n):
    w = np.asarray([weights[k] for k in keys], dtype=float)
    w = w / w.sum()
    return np.asarray(keys)[rng.choice(len(keys), size=n, p=w)]


def _nullable(values, mask) -> pl.Series:
    """String column that is null wherever mask is False."""
    return pl.Series([v if m else None for v, m in zip(values, mask)], dtype=pl.Utf8)


def _select(rng, propensity: np.ndarray, k: int, eligible: np.ndarray | None = None,
            idio: float = 1.2) -> np.ndarray:
    """Pick exactly k units, preferring high propensity, with idiosyncratic noise.

    Exact counts matter: AMDC publishes the number of distinct reporting units per
    dataset, so participation is a selection of a known size, not a coin flip.
    """
    n = propensity.shape[0]
    score = propensity + idio * rng.normal(0.0, 1.0, n)
    if eligible is not None:
        score = np.where(eligible, score, -np.inf)
        k = min(k, int(eligible.sum()))
    k = max(0, min(k, n))
    out = np.zeros(n, dtype=bool)
    if k:
        out[np.argpartition(-score, k - 1)[:k]] = True
    return out


def _peak_frame_share(table: str, frame_per_year: dict[int, int]) -> float:
    """Largest share of the register frame that a satellite file covers in one year.

    The parent, subsidiary and group files each cover a subset of the frame. Sizing
    the flag on the union would undershoot the busiest year, so the flag is set on the
    peak per-year share instead and the union comes out slightly above the published
    one; that direction is reported.
    """
    b = load_batch(table)
    kzr = b["kzr"]
    best = 0.0
    for yi, y in enumerate(int(x) for x in kzr.years):
        frame = frame_per_year.get(y)
        if frame:
            best = max(best, kzr.unique_per_year[yi] / frame)
    return best


def _membership_at(rng, n: int, per_year: list[int], noise: float) -> np.ndarray:
    persistence = rng.normal(0.0, 1.0, n)
    member = np.zeros((n, len(per_year)), dtype=bool)
    for yi, k in enumerate(per_year):
        score = persistence + noise * rng.normal(0.0, 1.0, n)
        k = min(max(k, 0), n)
        if k:
            member[np.argpartition(-score, k - 1)[:k], yi] = True
    return member


def _membership(rng, n: int, per_year: list[int], seed: int) -> np.ndarray:
    """Year-by-year frame membership whose union covers every register unit.

    Per-year populations are exact by construction. The union is not: too little
    churn and the same units recur every year, too much and the frame turns over
    implausibly. The churn parameter is solved so the union equals n, which is the
    published distinct-unit count.
    """
    lo, hi = 0.05, 6.0
    best, best_gap = None, None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        m = _membership_at(np.random.default_rng(seed + 991), n, per_year, mid)
        union = int(m.any(axis=1).sum())
        gap = abs(union - n)
        if best_gap is None or gap < best_gap:
            best, best_gap = m, gap
        if union < n:
            lo = mid
        else:
            hi = mid
    return best


def build(cfg: Settings) -> Register:
    rng = np.random.default_rng(cfg.seed)
    urs = load_batch("lse_re")
    kzr = urs["kzr"]
    years = [int(y) for y in kzr.years]
    n = cfg.firms(kzr.total_unique)

    # --- activity: OENACE 2008. Section split from the documented prior, tilted so
    # the mean code number equals the published one.
    nace_pool = list(cl.oenace2008())
    nace_codes = np.array([int(x.code5) for x in nace_pool], dtype=np.int64)
    per_section: dict[str, int] = {}
    for x in nace_pool:
        per_section[x.section] = per_section.get(x.section, 0) + 1
    nace_w = np.array([SECTION_SHARE.get(x.section, 0.0) / per_section[x.section]
                       for x in nace_pool])
    nace_w = tilt_to_mean(nace_codes, nace_w / nace_w.sum(), urs["oenace"].numeric["mean"])
    pick = rng.choice(len(nace_pool), size=n, p=nace_w)
    nace5 = nace_codes[pick]
    sections = np.array([nace_pool[i].section for i in pick])

    # --- region: NUTS-3 from the published quantiles, then tilted onto the mean
    n3_codes = np.array(sorted(cl.NUTS3), dtype=np.int64)
    n3_w = weights_matching_numeric_quantiles(n3_codes, urs["fsw_nuts3"].numeric)
    n3_w = tilt_to_mean(n3_codes, n3_w, urs["fsw_nuts3"].numeric["mean"])
    nuts3 = n3_codes[rng.choice(len(n3_codes), size=n, p=n3_w)]
    nuts2 = np.array([cl.NUTS3[int(k)][0] for k in nuts3])
    bl_digit = np.array([cl.NUTS3[int(k)][2] for k in nuts3])
    # Municipality number: real Bundesland digit, synthetic district and slot. The
    # official GKZ list is not in the KDB export, so these are not real GKZ.
    gem = np.where(bl_digit == 9, 90_001,
                   bl_digit * 10_000 + rng.integers(1, 26, n) * 100 + rng.integers(1, 60, n))

    # --- size
    labels = [c[0] for c in SIZE_CLASSES]
    shares = np.array([c[1] for c in SIZE_CLASSES])
    cls_idx = rng.choice(len(SIZE_CLASSES), size=n, p=shares / shares.sum())
    size_class = np.asarray(labels)[cls_idx]
    emp = np.empty(n, dtype=np.int64)
    for i, (_, _, lo, hi, shape) in enumerate(SIZE_CLASSES):
        m = cls_idx == i
        k = int(m.sum())
        if k:
            emp[m] = np.floor(lo + (hi - lo) * rng.random(k) ** shape).astype(np.int64)
    size_rank = np.log1p(emp)

    # --- register membership per year. Per-year counts are exact; the churn is tuned
    # so that the union over years also matches the published distinct-unit count.
    per_year = [cfg.firms(u) for u in kzr.unique_per_year[:len(years)]]
    member = _membership(rng, n, per_year, cfg.seed)
    panel_rows = [(np.flatnonzero(member[:, yi]), year) for yi, year in enumerate(years)]

    first_seen = np.array([years[np.argmax(row)] if row.any() else 9999 for row in member])
    last_seen = np.array([years[len(years) - 1 - np.argmax(row[::-1])] if row.any() else 0
                          for row in member])

    # --- demography. Founding predates first appearance in the frame.
    born = np.minimum(first_seen - rng.geometric(0.10, n), first_seen)
    born = np.clip(born, 1900, 2023)
    closed = np.where(last_seen < max(years), last_seen + 1, 0)

    manuf = np.isin(sections, list("BCDEF"))
    in_lse = np.array([s in LSE_SECTIONS for s in sections])

    # --- dataset participation, correlated with size and manufacturing
    shift = 0.9 * (size_rank - size_rank.mean()) + 1.4 * manuf
    shared = rng.normal(0.0, 1.0, n)          # a firm that trades tends to do both
    base = shift + 0.8 * shared
    intra = _select(rng, base, cfg.firms(DISTINCT_UNITS["intra"]))
    extra = _select(rng, base, cfg.firms(DISTINCT_UNITS["extra"]))
    prod = _select(rng, base + 0.6 * shift, cfg.firms(DISTINCT_UNITS["prodcom"]), eligible=manuf)
    ges = _select(rng, base + 0.9 * shift, max(1, int(GES_SAMPLE_SHARE * manuf.sum())),
                  eligible=manuf)

    # --- ownership, sized from the URS parent, subsidiary and group files
    frame_per_year = {y: int(kzr.unique_per_year[yi]) for yi, y in enumerate(years)}
    has_parent = _select(rng, 0.7 * shift, int(_peak_frame_share("lse_m", frame_per_year) * n))
    parent_country = _weighted_choice(rng, list(cl.PARENT_COUNTRY), cl.PARENT_COUNTRY, n)
    foreign = has_parent & (parent_country != "AT")
    in_group = _select(rng, 0.5 * shift, int(_peak_frame_share("lse_g", frame_per_year) * n))
    group_country = _weighted_choice(rng, list(cl.GROUP_COUNTRY), cl.GROUP_COUNTRY, n)
    has_sub = _select(rng, 1.1 * shift, int(_peak_frame_share("lse_d", frame_per_year) * n))
    has_foreign_sub = has_sub & (rng.random(n) < FOREIGN_SUBSIDIARY_SHARE)

    # --- CRM roles: only a slice of manufacturing sits in a value chain
    mats = signal_materials()
    mat_names = [m.name for m in mats]
    in_chain = (manuf & prod) | (manuf & intra & (rng.random(n) < 0.25))
    in_chain &= rng.random(n) < 0.55
    crm_material = _nullable(np.asarray(mat_names)[rng.integers(0, len(mats), n)], in_chain)
    stage_pool = np.array(["Processing/Refining", "Fabrication", "Manufacture", "Recycling"])
    crm_stage = _nullable(stage_pool[rng.choice(4, size=n, p=[0.30, 0.34, 0.28, 0.08])], in_chain)

    # --- identifier family.
    # URS publishes 834,511 legal units against 796,683 statistical enterprises, so
    # roughly 95% of enterprises are a single legal unit; and 1,485,401 local units
    # against 6,063,466 legal-unit-years, so 1.27 local units per unit and year.
    seq = np.arange(n, dtype=np.int64)
    ent_of = np.maximum(np.cumsum(rng.random(n) < 0.955) - 1, 0)
    grp_of = np.maximum(np.cumsum(rng.random(n) < 0.385) - 1, 0)
    n_local = 1 + rng.poisson(0.27, n)

    firms = pl.DataFrame({
        "seq": seq,
        "R_ID": pl.Series([str(v) for v in (ID_BASE["R_ID"] + seq * 7 + 3)], dtype=pl.Utf8),
        "Firmen_ID3": ID_BASE["Firmen_ID3"] + ent_of * 11 + 5,
        "Firmen_ID4": ID_BASE["Firmen_ID4"] + grp_of * 3 + 1,
        "Firmen_ID5": ID_BASE["Firmen_ID5"] + seq * 13 + 1,
        "Firmen_ID6": pl.Series(
            [str(v) for v in (ID_BASE["Firmen_ID6"] + seq * 17 + 9)], dtype=pl.Utf8
        ),
        "Firmen_ID9": ID_BASE["Firmen_ID9"] + seq * 19 + 7,
        "Firmen_ID1": ID_BASE["Firmen_ID1"] + seq * 23 + 11,
        "nace5": nace5,
        "nace_section": pl.Series(sections, dtype=pl.Utf8),
        "nuts3": nuts3,
        "nuts2": pl.Series(nuts2, dtype=pl.Utf8),
        "gem": gem.astype(np.int64),
        "n_local_units": n_local.astype(np.int64),
        "size_class": pl.Series(size_class, dtype=pl.Utf8),
        "employees": emp,
        "rechtsform": pl.Series(_weighted_choice(rng, list(cl.RECHTSFORM), cl.RECHTSFORM, n),
                                dtype=pl.Utf8),
        "is_code": pl.Series(_weighted_choice(rng, list(IS_CODES), IS_CODES, n),
                             dtype=pl.Utf8),
        "birth_year": born.astype(np.int64),
        "close_year": closed.astype(np.int64),
        "first_seen": first_seen.astype(np.int64),
        "last_seen": last_seen.astype(np.int64),
        "in_lse_scope": in_lse,
        "is_manufacturing": manuf,
        "intra_trader": intra,
        "extra_trader": extra,
        "prodcom_reporter": prod,
        "ges_reporter": ges,
        "has_parent": has_parent,
        "parent_country": _nullable(parent_country, has_parent),
        "foreign_controlled": foreign,
        "in_group": in_group,
        "group_country": _nullable(group_country, in_group),
        "has_subsidiary": has_sub,
        "has_foreign_sub": has_foreign_sub,
        "crm_material": crm_material,
        "crm_stage": crm_stage,
        "activity_weight": np.exp(rng.normal(0.0, 1.25, n)) * (1.0 + emp) ** 0.85,
    })

    panel = pl.concat([
        pl.DataFrame({"seq": idx.astype(np.int64),
                      "jahr": np.full(idx.shape[0], year, dtype=np.int32)})
        for idx, year in panel_rows
    ])
    return Register(firms=firms, panel=panel, materials=mat_names)


def alive(firms: pl.DataFrame, year: int) -> pl.DataFrame:
    """Units in the frame in `year`, using their observed first and last appearance."""
    return firms.filter((pl.col("first_seen") <= year) & (pl.col("last_seen") >= year))


def year_slice(reg: Register, year: int, flag: str | None = None, *,
               nearest: bool = False) -> pl.DataFrame:
    if nearest:
        year = min(max(year, reg.panel["jahr"].min()), reg.panel["jahr"].max())
    sub = reg.panel.filter(pl.col("jahr") == year).join(reg.firms, on="seq", how="inner")
    return sub.filter(pl.col(flag)) if flag else sub


__all__ = ["Register", "build", "alive", "year_slice", "allocate", "zipf_weights", "ID_BASE"]
