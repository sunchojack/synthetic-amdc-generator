"""Classification, geography, and country code universes."""

from __future__ import annotations

import csv
import io
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, TextIO

from src.crosswalk.config import KDB_GVGES, KDB_KN, KDB_OENACE, KDB_OEPRODCOM

# ---------------------------------------------------------------- KDB readers

def _archive_location(path: Path) -> tuple[Path, str]:
    archive = path.parent.with_suffix(".zip")
    member = f"{path.parent.name}/{path.name}"
    return archive, member


@contextmanager
def _open_kdb_text(path: Path) -> Iterator[TextIO]:
    """Open a KDB text file from an extracted directory or its bundled ZIP."""
    if path.is_file():
        with path.open(encoding="latin-1", newline="") as fh:
            yield fh
        return

    archive, member = _archive_location(path)
    if not archive.is_file():
        raise FileNotFoundError(f"KDB file not found: {path} (or {archive})")
    with zipfile.ZipFile(archive) as zf, zf.open(member) as raw:
        with io.TextIOWrapper(raw, encoding="latin-1", newline="") as fh:
            yield fh


def kdb_file_exists(path: Path) -> bool:
    if path.is_file():
        return True
    archive, member = _archive_location(path)
    if not archive.is_file():
        return False
    with zipfile.ZipFile(archive) as zf:
        return member in zf.namelist()


def _read_cti(path: Path) -> list[dict[str, str]]:
    with _open_kdb_text(path) as fh:
        return list(csv.DictReader(fh, delimiter=";"))


def _clamp(year: int, lo: int, hi: int) -> int:
    return min(max(year, lo), hi)


@dataclass(frozen=True)
class Good:
    code: int
    unit1: str = ""
    unit2: str = ""
    title: str = ""
    short_title: str = ""


@lru_cache(maxsize=None)
def cn8(year: int) -> tuple[Good, ...]:
    """8-digit Combined Nomenclature for one year. unit1 is the CN special unit."""
    y = _clamp(year, 2002, 2026)
    rows = _read_cti(KDB_KN / f"KN{y}_DE_CTI.txt")
    out = []
    for r in rows:
        code = r["EDV-Code"]
        if len(code) == 8 and code.isdigit():
            out.append(Good(
                code=int(code),
                unit1=(r.get("Maßeinheit") or "").strip(),
                title=(r.get("Titel") or "").strip(),
                short_title=(r.get("Kurztitel") or "").strip(),
            ))
    return tuple(out)


# The KDB leaves the Maßeinheit column empty in the KN vintages before 2016, so the
# special-unit lookup falls back to the earliest vintage that populates it.
_CN_UNITS_FROM = 2016


@lru_cache(maxsize=None)
def cn8_special_unit(year: int) -> frozenset[int]:
    """CN8 codes that declare a special unit, with a fallback for pre-2016 vintages."""
    goods = cn8(year)
    codes = {g.code for g in goods if g.unit1}
    if codes:
        return frozenset(codes)
    fallback = {g.code for g in cn8(_CN_UNITS_FROM) if g.unit1}
    return frozenset(c for c in (g.code for g in goods) if c in fallback)


@lru_cache(maxsize=None)
def oeprodcom(year: int) -> tuple[Good, ...]:
    """10-digit OEPRODCOM for one year, with both declared quantity units."""
    y = _clamp(year, 2008, 2026)
    rows = _read_cti(KDB_OEPRODCOM / f"OEPRODCOM{y}_DE_CTI.txt")
    out = []
    for r in rows:
        code = r["EDV-Code"]
        if len(code) == 10 and code.isdigit():
            out.append(Good(
                code=int(code),
                unit1=(r.get("Maßeinheit") or "").strip(),
                unit2=(r.get("Maßeinheit2") or "").strip(),
            ))
    return tuple(out)


@lru_cache(maxsize=None)
def gvges(year: int) -> tuple[str, ...]:
    """7-digit Gueterverzeichnis fuer den Guetereinsatz. Only 2016-2020 on disk."""
    y = _clamp(year, 2016, 2020)
    rows = _read_cti(KDB_GVGES / f"GV-GES{y}_DE_CTI.txt")
    return tuple(r["EDV-Code"] for r in rows if len(r["EDV-Code"]) == 7 and r["EDV-Code"].isdigit())


@lru_cache(maxsize=None)
def gvges_from_cn8(year: int) -> dict[int, tuple[str, ...]]:
    """CN8 to GV-GES, from the official KDB correspondence table.

    This is what lets a CRM identified by its CN8 trade codes be found again on the
    input side of the Guetereinsatzstatistik, which reports a different classification
    entirely. Only 2016-2020 vintages are in the KDB export.
    """
    y = _clamp(year, 2016, 2020)
    rows = _read_cti(KDB_GVGES / f"GV-GES{y}_KN{y}_DE_12B.txt")
    out: dict[int, list[str]] = {}
    for r in rows:
        gv = r["EDV-Code"]
        cn = r.get("korrespondierender EDV-Code") or ""
        if len(gv) == 7 and gv.isdigit() and len(cn) == 8 and cn.isdigit():
            out.setdefault(int(cn), []).append(gv)
    return {k: tuple(sorted(set(v))) for k, v in out.items()}


@lru_cache(maxsize=None)
def oeprodcom_from_cn8(year: int) -> dict[int, tuple[int, ...]]:
    """CN8 to OEPRODCOM, from the official KDB correspondence table.

    The JRC inventory names a material's products in EU PRODCOM 2023. Only about a
    third of those codes survive as an 8-digit prefix of an OEPRODCOM code in any
    given year, because OEPRODCOM is the Austrian extension and both nomenclatures
    move. Routing through the CN8 codes the same inventory gives, using Statistik
    Austria's own concordance for the reporting year, recovers the rest.
    """
    y = _clamp(year, 2008, 2026)
    path = KDB_OEPRODCOM / f"OEPRODCOM{y}_KN{y}_DE_12B.txt"
    if not kdb_file_exists(path):
        return {}
    out: dict[int, set[int]] = {}
    for r in _read_cti(path):
        pc = r["EDV-Code"]
        cn = (r.get("korrespondierender EDV-Code") or "")
        if len(pc) == 10 and pc.isdigit() and len(cn) == 8 and cn.isdigit():
            out.setdefault(int(cn), set()).add(int(pc))
    return {k: tuple(sorted(v)) for k, v in out.items()}


@dataclass(frozen=True)
class Nace:
    code5: str      # 5 digits, as stored by AMDC
    section: str    # OENACE section letter, as stored in NACEAB


@lru_cache(maxsize=None)
def oenace2008() -> tuple[Nace, ...]:
    rows = _read_cti(KDB_OENACE / "OENACE2008_DE_CTI.txt")
    out = []
    for r in rows:
        edv = r["EDV-Code"]
        if len(edv) == 6 and edv[0].isalpha() and edv[1:].isdigit():
            out.append(Nace(edv[1:], edv[0]))
    return tuple(out)


# --------------------------------------------------------------- regions

# NUTS-3 stored by AMDC as an integer: the three digits after "AT". The register
# publishes 35 distinct values, which is the full official list.
NUTS3: dict[int, tuple[str, str, int]] = {
    # nuts3 -> (nuts2, name, Bundesland digit leading the municipality number)
    111: ("AT11", "Mittelburgenland", 1),
    112: ("AT11", "Nordburgenland", 1),
    113: ("AT11", "Suedburgenland", 1),
    121: ("AT12", "Mostviertel-Eisenwurzen", 3),
    122: ("AT12", "Niederoesterreich-Sued", 3),
    123: ("AT12", "Sankt Poelten", 3),
    124: ("AT12", "Waldviertel", 3),
    125: ("AT12", "Weinviertel", 3),
    126: ("AT12", "Wiener Umland/Nordteil", 3),
    127: ("AT12", "Wiener Umland/Suedteil", 3),
    130: ("AT13", "Wien", 9),
    211: ("AT21", "Klagenfurt-Villach", 2),
    212: ("AT21", "Oberkaernten", 2),
    213: ("AT21", "Unterkaernten", 2),
    221: ("AT22", "Graz", 6),
    222: ("AT22", "Liezen", 6),
    223: ("AT22", "Oestliche Obersteiermark", 6),
    224: ("AT22", "Oststeiermark", 6),
    225: ("AT22", "West- und Suedsteiermark", 6),
    226: ("AT22", "Westliche Obersteiermark", 6),
    311: ("AT31", "Innviertel", 4),
    312: ("AT31", "Linz-Wels", 4),
    313: ("AT31", "Muehlviertel", 4),
    314: ("AT31", "Steyr-Kirchdorf", 4),
    315: ("AT31", "Traunviertel", 4),
    321: ("AT32", "Lungau", 5),
    322: ("AT32", "Pinzgau-Pongau", 5),
    323: ("AT32", "Salzburg und Umgebung", 5),
    331: ("AT33", "Ausserfern", 7),
    332: ("AT33", "Innsbruck", 7),
    333: ("AT33", "Osttirol", 7),
    334: ("AT33", "Tiroler Oberland", 7),
    335: ("AT33", "Tiroler Unterland", 7),
    341: ("AT34", "Bludenz-Bregenzer Wald", 8),
    342: ("AT34", "Rheintal-Bodenseegebiet", 8),
}

# Legal form. Codes and shares are the published URS level counts for `rf`
# (6,063,466 legal-unit-years, 2013-2023), not an assumed distribution.
RECHTSFORM: dict[str, float] = {
    "EU": 3_841_446,   # Einzelunternehmen
    "GES": 1_342_684,  # Gesellschaft mit beschraenkter Haftung
    "KG": 293_858,
    "VRN": 147_328,    # Verein
    "OG": 135_386,
    "SON": 129_978,
    "GBR": 98_252,
    "AUS": 18_759,     # auslaendische Rechtsform
    "GEN": 17_433,
    "PST": 11_954,
    "AG": 9_940,
    "UN": 7_264,
    "KRK": 5_463,
    "NR": 1_888,
    "SE": 1_000,
    "SPA": 600,
    "STG": 500,
    "PRK": 400,
    "ERW": 300,
    "SPK": 250,
    "VAG": 200,
    "GBH": 150,
    "EWI": 100,
}

# Enterprise-group type, published URS `GRP_ART` level counts.
GRP_ART: dict[str, float] = {"DEG": 215_026, "GEG": 52_763}

# Country of the controlling parent, published URS `m_fsw_lkz` level counts.
PARENT_COUNTRY: dict[str, float] = {
    "AT": 271_133, "DE": 23_807, "CH": 8_638, "NL": 4_472, "GB": 3_669,
    "IT": 2_426, "LU": 1_800, "LI": 1_719, "US": 1_599, "FR": 1_568,
    "SK": 1_296, "CY": 1_116, "HU": 933, "SE": 842, "CZ": 700, "DK": 620,
    "ES": 540, "BE": 520, "JP": 430, "PL": 380, "SI": 340, "IE": 300,
    "NO": 260, "FI": 230, "RU": 210, "CN": 190, "TR": 150, "CA": 140,
    "KR": 130, "IN": 120, "VG": 110, "AE": 100, "MT": 90, "SG": 80,
}

# Country of the enterprise group, published URS `GRP_LKZ` level counts.
GROUP_COUNTRY: dict[str, float] = {
    "AT": 252_964, "DE": 6_117, "CH": 1_846, "NL": 1_019, "LU": 964,
    "GB": 641, "IT": 632, "LI": 522, "FR": 329, "SK": 262, "US": 216,
    "CY": 209, "SE": 160, "BE": 149, "CZ": 120, "DK": 100, "ES": 90,
    "HU": 85, "JP": 70, "PL": 60, "SI": 55, "IE": 50, "NO": 45, "FI": 40,
}


# --------------------------------------------------------------- countries

# Intrastat partner universes are taken verbatim from AMDC level_counts, so they
# are not defined here. The third-country universe below is the geonomenclature
# space Extrastat draws on.
EU27_2020 = (
    "BE", "BG", "CZ", "DK", "EE", "IE", "GR", "ES", "FR", "HR", "IT", "CY",
    "LV", "LT", "LU", "HU", "MT", "NL", "PL", "PT", "RO", "SI", "SK", "FI", "SE", "DE",
)

# Approximate shares of Austrian third-country trade, used where AMDC publishes no
# level counts (Extrastat VL/UL/BL, Intrastat UL). Tail is filled by Zipf.
THIRD_COUNTRY_ANCHORS: dict[str, float] = {
    "CN": 0.150, "US": 0.115, "CH": 0.105, "TR": 0.055, "GB": 0.052,
    "RU": 0.032, "JP": 0.030, "IN": 0.029, "KR": 0.024, "RS": 0.022,
    "BA": 0.020, "TW": 0.018, "UA": 0.017, "NO": 0.016, "MX": 0.014,
    "BR": 0.014, "CA": 0.013, "AE": 0.012, "TH": 0.012, "VN": 0.011,
    "ZA": 0.010, "IL": 0.010, "AU": 0.010, "MK": 0.009, "EG": 0.009,
    "MY": 0.008, "SA": 0.008, "ID": 0.008, "SG": 0.008, "HK": 0.007,
    "MA": 0.007, "AL": 0.006, "TN": 0.006, "CL": 0.006, "NZ": 0.005,
    "PK": 0.005, "AR": 0.005, "PH": 0.005, "KZ": 0.005, "BY": 0.004,
    "MD": 0.004, "GE": 0.003, "IR": 0.003, "NG": 0.003, "PE": 0.003,
    "CO": 0.003, "BD": 0.003, "LK": 0.002, "QA": 0.002, "KW": 0.002,
}

_TAIL_COUNTRIES = (
    "AD", "AF", "AG", "AI", "AM", "AO", "AQ", "AS", "AW", "AZ", "BB", "BF", "BH",
    "BI", "BJ", "BM", "BN", "BO", "BS", "BT", "BW", "BZ", "CD", "CF", "CG", "CI",
    "CK", "CM", "CR", "CU", "CV", "CW", "DJ", "DM", "DO", "DZ", "EC", "ER", "ET",
    "FJ", "FK", "FO", "GA", "GD", "GG", "GH", "GI", "GL", "GM", "GN", "GQ", "GT",
    "GU", "GW", "GY", "HN", "HT", "IQ", "IS", "JE", "JM", "JO", "KE", "KG", "KH",
    "KI", "KM", "KN", "KP", "KY", "LA", "LB", "LC", "LI", "LR", "LS", "LY", "MC",
    "MG", "MH", "ML", "MM", "MN", "MO", "MP", "MR", "MS", "MT", "MU", "MV", "MW",
    "MZ", "NA", "NC", "NE", "NI", "NP", "NR", "OM", "PA", "PF", "PG", "PM", "PN",
    "PS", "PW", "PY", "RW", "SB", "SC", "SD", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SY", "SZ", "TC", "TD", "TG", "TJ", "TL", "TM", "TO", "TT", "TV",
    "TZ", "UG", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", "VU", "WF", "WS", "XK",
    "YE", "ZM", "ZW",
)

# Geonomenclature pseudo-codes. QU/QV/QW/QX/QY/QZ are "not further specified"
# aggregates; XI/XU split the United Kingdom from 2021 (Northern Ireland Protocol).
SPECIAL_CODES_BY_YEAR: dict[int, tuple[str, ...]] = {
    **{y: ("QU", "QV", "QW", "QX", "QY", "QZ") for y in range(2013, 2021)},
    **{y: ("QU", "QV", "QW", "QX", "QY", "QZ", "XI", "XU") for y in range(2021, 2027)},
}


def third_country_universe(year: int) -> tuple[str, ...]:
    """Third-country geonomenclature for one year, ~227-230 codes as AMDC reports."""
    base = list(THIRD_COUNTRY_ANCHORS) + [c for c in _TAIL_COUNTRIES if c not in THIRD_COUNTRY_ANCHORS]
    base += [c for c in EU27_2020 if c not in base]
    if year <= 2020:
        base.append("GB")
    base += list(SPECIAL_CODES_BY_YEAR[year])
    seen: dict[str, None] = {}
    for c in base:
        seen.setdefault(c, None)
    return tuple(seen)
