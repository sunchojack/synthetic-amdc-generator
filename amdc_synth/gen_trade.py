"""Generate Intrastat and Extrastat from their published statistics."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.crosswalk.metadata import Batch, load_batch
from src.crosswalk.schema import EXTRA_COLUMNS, INTRA_COLUMNS, INTRA_EMPTY
from .config import Settings
from .distributions import (HeavyTail, allocate, ensure_all_levels,
                            ensure_distinct_count, firm_baskets, rank_couple,
                            rebalance_group_mass, tilt_to_mean, weights_from_levels,
                            zipf_weights)
from .register import Register, alive
import src.crosswalk.codelists as cl
from src.crosswalk.material_catalog import materials as crm_materials
from .crm import concentration

# Rank correlation imposed between net weight and statistical value. Real ITGS runs
# about 0.8 in logs; a lower figure here keeps unit values dispersed within a code.
VALUE_WEIGHT_RHO = 0.80

# Share of a CRM firm's import lines that are actually its raw-material lines.
CRM_LINE_SHARE = 0.35
# Share of an import line's origin drawn from the material's dominant suppliers.
CRM_ORIGIN_FIDELITY = 0.85


def _stable_rank(codes: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic pseudo-random ranking of a code universe.

    Popularity must not follow numeric order, but it must be stable across years so
    that a code heavily traded in 2016 is still heavily traded in 2017.
    """
    h = (codes.astype(np.int64) * np.int64(2654435761) + salt) & np.int64(0x7FFFFFFF)
    return np.argsort(np.argsort(h))


def _code_weights(codes: np.ndarray, salt: int, s: float) -> np.ndarray:
    rank = _stable_rank(codes, salt)
    w = zipf_weights(len(codes), s=s)
    return w[rank]


def _levels_for_year(stats, year: int, universe: list[str], levels: dict[str, int]) -> list[str]:
    """Trim the rarest levels so the year matches AMDC's distinct-value count."""
    try:
        target = stats.unique_per_year[stats.years.index(year)]
    except (ValueError, IndexError):
        return universe
    if target <= 0 or target >= len(universe):
        return universe
    ordered = sorted(universe, key=lambda k: -levels.get(k, 0))
    return ordered[:target]


def _draw_countries(rng, n: int, keys: list[str], weights: np.ndarray) -> np.ndarray:
    return np.asarray(keys)[rng.choice(len(keys), size=n, p=weights)]


def _third_country_weights(year: int) -> tuple[list[str], np.ndarray]:
    uni = list(cl.third_country_universe(year))
    anchors = cl.THIRD_COUNTRY_ANCHORS
    named = sum(anchors.values())
    rest = [c for c in uni if c not in anchors]
    tail = zipf_weights(len(rest), s=1.0) * max(1.0 - named, 0.05)
    w = np.array([anchors.get(c, 0.0) for c in uni], dtype=float)
    ti = 0
    for i, c in enumerate(uni):
        if c not in anchors:
            w[i] = tail[ti]
            ti += 1
    return uni, w / w.sum()


def _trader_rows(rng, cfg: Settings, firms: pl.DataFrame, flag: str, year: int,
                 n_rows: int, n_units: int) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Assign rows to the units reporting in this year, heavy-tailed within year.

    Returns the selected units, the unit index per row, and each unit's row count.
    The counts size the per-firm commodity basket.
    """
    pool = alive(firms.filter(pl.col(flag)), year)
    if pool.height == 0:
        pool = firms.filter(pl.col(flag))
    k = min(max(n_units, 1), pool.height)
    w = pool["activity_weight"].to_numpy()
    score = np.log(w) + rng.normal(0.0, 0.9, pool.height)
    sel = np.argpartition(-score, k - 1)[:k]
    sub = pool[sel]
    ww = sub["activity_weight"].to_numpy()
    counts = allocate(n_rows, ww / ww.sum())
    idx = np.repeat(np.arange(k), counts)
    return sub, idx, counts


def _crm_assignment(sub: pl.DataFrame, year: int):
    """Per-selected-unit CRM material, resolved to codes valid in `year`."""
    mats = crm_materials()
    valid = {g.code for g in cl.cn8(year)}
    names = sub["crm_material"].to_list()
    stages = sub["crm_stage"].to_list()
    out: list[tuple[np.ndarray, dict[str, float]] | None] = []
    for name, stage in zip(names, stages):
        mat = mats.get(name) if name else None
        if mat is None:
            out.append(None)
            continue
        pool = mat.upstream_cn if stage in (None, "Processing/Refining", "Fabrication") \
            else mat.upstream_cn + mat.downstream_cn
        if stage == "Recycling":
            pool = mat.scrap_cn or mat.upstream_cn
        codes = np.array([c for c in pool if c in valid], dtype=np.int64)
        out.append((codes, concentration(mat)) if codes.size else None)
    return out


def _apply_crm(rng, kn: np.ndarray, origin: np.ndarray, unit_idx: np.ndarray,
               is_import: np.ndarray, assignment, country_uni: list[str],
               country_w: np.ndarray) -> np.ndarray:
    """Overwrite a share of each CRM unit's import lines with its own codes.

    Returns the row indices that were turned into CRM lines, so the caller can keep
    the dispatch country consistent with the origin country on those rows.
    """
    touched: list[np.ndarray] = []
    for u, spec in enumerate(assignment):
        if spec is None:
            continue
        codes, conc = spec
        sel = np.flatnonzero((unit_idx == u) & is_import)
        if sel.size == 0:
            continue
        take = sel[rng.random(sel.size) < CRM_LINE_SHARE]
        if take.size == 0:
            continue
        kn[take] = codes[rng.integers(0, codes.size, take.size)]
        touched.append(take)
        if not conc:
            continue
        src = list(conc)
        p = np.array([conc[c] for c in src], dtype=float)
        p = p / p.sum()
        from_conc = rng.random(take.size) < CRM_ORIGIN_FIDELITY
        picked = np.asarray(src)[rng.choice(len(src), size=take.size, p=p)]
        fallback = _draw_countries(rng, take.size, country_uni, country_w)
        origin[take] = np.where(from_conc, picked, fallback)
    return np.concatenate(touched) if touched else np.empty(0, dtype=np.int64)


# ------------------------------------------------------------------ Intrastat

def intra(cfg: Settings, reg: Register) -> pl.DataFrame:
    b: Batch = load_batch("intra")
    rng = np.random.default_rng(cfg.seed + 101)
    rows_by_year = b.rows_per_year()
    berm, vl, ul, bl, gart, kn_s = (b["BERM"], b["VL"], b["UL"], b["BL"], b["GART"], b["KN"])
    kzr = b["kzr"]

    vl_keys_all = list(vl.levels)
    bl_keys_all = list(bl.levels)
    month_keys = [str(m) for m in range(1, 13)]
    month_w = weights_from_levels(berm.levels, month_keys)

    frames = []
    for yi, year in enumerate(sorted(rows_by_year)):
        n = cfg.rows(rows_by_year[year])
        exp_share = vl.rate(year)          # VL is missing exactly on dispatches
        n_units = cfg.firms(kzr.unique_per_year[yi]) if yi < len(kzr.unique_per_year) else 1
        sub, unit_idx, unit_counts = _trader_rows(rng, cfg, reg.firms, "intra_trader",
                                                  year, n, n_units)

        is_export = rng.random(n) < exp_share
        is_import = ~is_export

        # partner countries, conditional on direction
        vl_keys = _levels_for_year(vl, year, vl_keys_all, vl.levels)
        bl_keys = _levels_for_year(bl, year, bl_keys_all, bl.levels)
        vl_w = weights_from_levels(vl.levels, vl_keys)
        bl_w = weights_from_levels(bl.levels, bl_keys)

        VL = np.full(n, None, dtype=object)
        BL = np.full(n, None, dtype=object)
        VL[is_import] = _draw_countries(rng, int(is_import.sum()), vl_keys, vl_w)
        BL[is_export] = _draw_countries(rng, int(is_export.sum()), bl_keys, bl_w)

        # country of origin: always on arrivals, on dispatches only from 2022
        tc_uni, tc_w = _third_country_weights(year)
        UL = np.full(n, None, dtype=object)
        imp = np.flatnonzero(is_import)
        same = rng.random(imp.size) < 0.62          # origin equals dispatch country
        UL[imp] = np.where(same, VL[imp],
                           _draw_countries(rng, imp.size, tc_uni, tc_w))
        if year >= 2022:
            exp = np.flatnonzero(is_export)
            dom = rng.random(exp.size) < 0.70       # own production dominates dispatches
            UL[exp] = np.where(dom, "AT", _draw_countries(rng, exp.size, tc_uni, tc_w))
            # a residual 0.01-0.02% stays unreported even after it became mandatory
            UL[rng.random(n) < ul.rate(year)] = None

        # commodity code from that year's CN vintage
        goods = cl.cn8(year)
        codes = np.array([g.code for g in goods], dtype=np.int64)
        with_unit = cl.cn8_special_unit(year)
        has_unit = np.array([g.code in with_unit for g in goods])
        w = _code_weights(codes, salt=7, s=1.05)
        # the mean CN code number carries the chapter composition of Austrian trade
        w = tilt_to_mean(codes, w, kn_s.numeric["mean"])
        # SM is only reported where the CN code declares a special unit, so the row
        # mass on those codes is set to their share of the nomenclature
        w = rebalance_group_mass(w, has_unit, float(has_unit.mean()))
        KN = firm_baskets(rng, unit_counts, codes, w)
        KN = ensure_distinct_count(KN, codes, cfg.distinct(kn_s, year), rng)
        # VL stays an EU member state while UL becomes the true origin: this is how
        # a Chinese-origin CRM arrives in the intra-EU file, dispatched from Germany.
        _apply_crm(rng, KN, UL, unit_idx, is_import, _crm_assignment(sub, year), tc_uni, tc_w)

        # special unit is declared per CN code, so its zero pattern is structural
        code_to_unit = dict(zip(codes.tolist(), has_unit.tolist()))
        sm_flag = np.fromiter((code_to_unit.get(int(c), False) for c in KN), bool, n)
        sm_fit = HeavyTail.fit(b["SM"].numeric, zero_share=1.0 - sm_flag.mean(), n_rows=n)
        SM = np.zeros(n)
        SM[sm_flag] = sm_fit.sample_positive(int(sm_flag.sum()), rng)

        GW = HeavyTail.fit(b["GW"].numeric, n_rows=n).sample(n, rng)
        SW = HeavyTail.fit(b["SW"].numeric, n_rows=n).sample(n, rng)
        SW = rank_couple(GW, SW, VALUE_WEIGHT_RHO, rng)

        # transaction code: single digit to 2021, two digits from 2022
        keys = [k for k in gart.levels if (len(k) == 1) == (year <= 2021)]
        keys = _levels_for_year(gart, year, keys, gart.levels)
        GART = ensure_all_levels(_draw_countries(rng, n, keys,
                                                 weights_from_levels(gart.levels, keys)),
                                 keys, rng)

        frames.append(pl.DataFrame({
            "Firmen_ID9": sub["Firmen_ID9"].to_numpy()[unit_idx],
            "R_ID": sub["R_ID"].to_numpy()[unit_idx],
            "EV": np.where(is_import, 1, 2).astype(np.int32),
            "BERM": _draw_countries(rng, n, month_keys, month_w).astype(np.int32),
            "BERJ": np.full(n, year, dtype=np.int32),
            "VL": pl.Series(VL.tolist(), dtype=pl.Utf8),
            "UL": pl.Series(UL.tolist(), dtype=pl.Utf8),
            "BL": pl.Series(BL.tolist(), dtype=pl.Utf8),
            "KN": pl.Series([f"{c:08d}" for c in KN], dtype=pl.Utf8),
            "GW": np.round(GW, 3),
            "SM": np.round(SM, 6),
            "SW": np.round(SW, 2),
            "GART": pl.Series(GART.tolist(), dtype=pl.Utf8),
            **{c: pl.Series([None] * n, dtype=pl.Boolean) for c in INTRA_EMPTY},
        }))
    return pl.concat(frames).select(INTRA_COLUMNS)


# ------------------------------------------------------------------ Extrastat

def extra(cfg: Settings, reg: Register) -> pl.DataFrame:
    b: Batch = load_batch("extra")
    rng = np.random.default_rng(cfg.seed + 202)
    rows_by_year = b.rows_per_year()
    berm, ul, gart, kzr, kn_s = b["BERM"], b["UL"], b["GART"], b["kzr"], b["KN"]

    month_keys = [str(m) for m in range(1, 13)]
    month_w = weights_from_levels(berm.levels, month_keys)
    cat_specs = {name: (list(b[name].levels), weights_from_levels(b[name].levels, list(b[name].levels)))
                 for name in ("SV", "CO", "VERKINL", "VERKGRE", "PRAEF")}

    frames = []
    for yi, year in enumerate(sorted(rows_by_year)):
        n = cfg.rows(rows_by_year[year])
        exp_share = ul.rate(year)          # UL is missing exactly on exports
        n_units = cfg.firms(kzr.unique_per_year[yi]) if yi < len(kzr.unique_per_year) else 1
        sub, unit_idx, unit_counts = _trader_rows(rng, cfg, reg.firms, "extra_trader",
                                                  year, n, n_units)

        is_export = rng.random(n) < exp_share
        is_import = ~is_export
        tc_uni, tc_w = _third_country_weights(year)

        partner = _draw_countries(rng, n, tc_uni, tc_w)
        # arrivals: partner dispatches, Austria is the destination. exports: mirrored.
        VL = np.where(is_import, partner, "AT").astype(object)
        BL = np.where(is_import, "AT", partner).astype(object)
        # AMDC reports a small missing rate on VL and BL; reproduce it
        for arr, stats in ((VL, b["VL"]), (BL, b["BL"])):
            r = stats.rate(year)
            if r > 0:
                arr[rng.random(n) < r] = None

        UL = np.full(n, None, dtype=object)
        imp = np.flatnonzero(is_import)
        same = rng.random(imp.size) < 0.70
        UL[imp] = np.where(same, np.where(VL[imp] == None, partner[imp], VL[imp]),  # noqa: E711
                           _draw_countries(rng, imp.size, tc_uni, tc_w))

        goods = cl.cn8(year)
        codes = np.array([g.code for g in goods], dtype=np.int64)
        with_unit = cl.cn8_special_unit(year)
        has_unit = np.array([g.code in with_unit for g in goods])
        w = _code_weights(codes, salt=13, s=1.12)
        w = tilt_to_mean(codes, w, kn_s.numeric["mean"])
        w = rebalance_group_mass(w, has_unit, float(has_unit.mean()))
        KN = firm_baskets(rng, unit_counts, codes, w)
        KN = ensure_distinct_count(KN, codes, cfg.distinct(kn_s, year), rng)
        crm_rows = _apply_crm(rng, KN, UL, unit_idx, is_import,
                              _crm_assignment(sub, year), tc_uni, tc_w)
        # a CRM line is dispatched from the country it originates in
        VL[crm_rows] = UL[crm_rows]

        code_to_unit = dict(zip(codes.tolist(), has_unit.tolist()))
        sm_flag = np.fromiter((code_to_unit.get(int(c), False) for c in KN), bool, n)
        sm_fit = HeavyTail.fit(b["SM"].numeric, zero_share=1.0 - sm_flag.mean(), n_rows=n)
        SM = np.zeros(n)
        SM[sm_flag] = sm_fit.sample_positive(int(sm_flag.sum()), rng)
        GW = HeavyTail.fit(b["GW"].numeric, n_rows=n).sample(n, rng)
        SW = HeavyTail.fit(b["SW"].numeric, n_rows=n).sample(n, rng)
        SW = rank_couple(GW, SW, VALUE_WEIGHT_RHO, rng)

        def cat(name: str, mask: np.ndarray | None = None) -> pl.Series:
            keys, weights = cat_specs[name]
            keys = _levels_for_year(b[name], year, keys, b[name].levels)
            weights = weights_from_levels(b[name].levels, keys)
            vals = _draw_countries(rng, n, keys, weights).astype(object)
            if mask is not None:
                vals[~mask] = None
            r = b[name].rate(year)
            if mask is None and r > 0:
                vals[rng.random(n) < r] = None
            return pl.Series([None if v is None else int(v) for v in vals], dtype=pl.Int32)

        gkeys = _levels_for_year(gart, year, list(gart.levels), gart.levels)
        GART = ensure_all_levels(_draw_countries(rng, n, gkeys,
                                                 weights_from_levels(gart.levels, gkeys)),
                                 gkeys, rng)

        # V1 is populated only in 2022 and runs 1..N: an internal row counter
        V1 = (np.arange(1, n + 1, dtype=np.int64) if year == 2022 else None)

        frames.append(pl.DataFrame({
            "R_ID": sub["R_ID"].to_numpy()[unit_idx],
            "Firmen_ID9": sub["Firmen_ID9"].to_numpy()[unit_idx],
            "EV": np.where(is_import, 1, 2).astype(np.int32),
            "BERM": _draw_countries(rng, n, month_keys, month_w).astype(np.int32),
            "BERJ": np.full(n, year, dtype=np.int32),
            "SV": cat("SV"),
            "VL": pl.Series(VL.tolist(), dtype=pl.Utf8),
            "UL": pl.Series(UL.tolist(), dtype=pl.Utf8),
            "BL": pl.Series(BL.tolist(), dtype=pl.Utf8),
            "KN": pl.Series([f"{c:08d}" for c in KN], dtype=pl.Utf8),
            "GW": np.round(GW, 3),
            "SM": np.round(SM, 6),
            "SW": np.round(SW, 2),
            "CO": cat("CO"),
            "PRAEF": cat("PRAEF", mask=is_import),   # duty preference on arrivals only
            "VERKINL": cat("VERKINL"),
            "VERKGRE": cat("VERKGRE"),
            "GART": pl.Series(GART.tolist(), dtype=pl.Utf8),
            "V1": pl.Series(V1 if V1 is not None else [None] * n, dtype=pl.Int64),
        }))
    return pl.concat(frames).select(EXTRA_COLUMNS)
