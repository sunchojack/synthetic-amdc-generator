"""AMDC product inventory and canonical source-column contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    """One AMDC microdata product as advertised in the public catalogue."""

    oid: int
    abbrev: str
    table: str
    years: tuple[int, ...]
    keys: tuple[str, ...]
    calibration: str  # A = AMDC per-variable stats, B = codebook + aggregates, C = codebook only
    metadata_batch: str | None = None
    note: str = ""


PRODUCTS: tuple[Product, ...] = (
    Product(
        oid=134674, abbrev="URS-RE", table="lse_re",
        years=tuple(range(2013, 2024)), keys=("R_ID",), calibration="A",
        metadata_batch="154242__URS__metadata_batch_urs_re.json",
        note="Unternehmensregister Final Frame, legal units",
    ),
    Product(
        oid=134674, abbrev="URS-LOC", table="lse_loc",
        years=tuple(range(2013, 2024)), keys=("R_ID", "Firmen_ID5"), calibration="A",
        metadata_batch="154241__URS__metadata_batch_urs_loc.json",
        note="local units per legal unit",
    ),
    Product(
        oid=134674, abbrev="URS-STAT", table="lse_stat",
        years=tuple(range(2018, 2022)), keys=("R_ID", "Firmen_ID3"), calibration="A",
        metadata_batch="152517__URS__metadata_batch_urs_stat.json",
        note="statistical enterprise per legal unit; 2018-2021 only, which is the "
             "reason the ID3-to-ID2 bridge is incomplete",
    ),
    Product(
        oid=134674, abbrev="URS-M", table="lse_m",
        years=tuple(range(2013, 2022)), keys=("R_ID",), calibration="A",
        metadata_batch="152513__URS__metadata_batch_urs_m.json",
        note="controlling parent of the legal unit",
    ),
    Product(
        oid=134674, abbrev="URS-D", table="lse_d",
        years=tuple(range(2013, 2022)), keys=("R_ID",), calibration="A",
        metadata_batch="152515__URS__metadata_batch_urs_d.json",
        note="controlled subsidiaries of the legal unit",
    ),
    Product(
        oid=134674, abbrev="URS-G", table="lse_g",
        years=tuple(range(2018, 2022)), keys=("R_ID", "Firmen_ID4"), calibration="A",
        metadata_batch="152514__URS__metadata_batch_urs_g.json",
        note="enterprise group of the legal unit",
    ),
    Product(
        oid=61404, abbrev="Intrastat", table="intra",
        years=tuple(range(2013, 2024)), keys=("Firmen_ID9", "R_ID"), calibration="A",
        metadata_batch="152509__Intrastat__metadata_batch_intra.json",
        note="EU-internal trade in goods, monthly",
    ),
    Product(
        oid=61401, abbrev="Extrastat", table="extra",
        years=tuple(range(2013, 2024)), keys=("R_ID", "Firmen_ID9"), calibration="A",
        metadata_batch="152510__Extrastat__metadata_batch_extra.json",
        note="third-country trade in goods, monthly",
    ),
    Product(
        oid=92107, abbrev="KJEPB_P", table="prodcom",
        years=tuple(range(2014, 2025)), keys=("R_ID", "Firmen_ID1", "Firmen_ID6"),
        calibration="A", metadata_batch="152511__KJEPB_P__metadata_batch_prodcom.json",
        note="goods production PRODCOM, monthly; AMDC stats cover 2014-2022 only",
    ),
    Product(
        oid=61422, abbrev="GES", table="ges",
        years=tuple(range(2008, 2024)), keys=("Firmen_ID6",), calibration="B",
        note="Guetereinsatzstatistik, annual establishment sample. Linking to a legal "
             "unit requires the separate Firmen_ID6-to-Firmen_ID2 bridge.",
    ),
    Product(
        oid=108721, abbrev="LSE-RE", table="lse_re_bis2020",
        years=tuple(range(2008, 2021)), keys=("R_ID",), calibration="B",
        note="structural business statistics, legal units, pre-2021 variable set",
    ),
    Product(
        oid=124157, abbrev="LSE-RE-2021", table="lse_re_ab2021",
        years=tuple(range(2021, 2024)), keys=("R_ID",), calibration="B",
        note="structural business statistics, legal units, post-2020 variable set",
    ),
    Product(
        oid=118790, abbrev="IFATS-RE", table="ifats_re_bis2020",
        years=tuple(range(2008, 2021)), keys=("R_ID",), calibration="C",
        note="inward FATS, foreign-controlled units in Austria",
    ),
    Product(
        oid=153237, abbrev="IFATS-RE-2021", table="ifats_re_ab2021",
        years=tuple(range(2021, 2024)), keys=("R_ID",), calibration="C",
    ),
    Product(
        oid=119682, abbrev="OFATS-RE", table="ofats_re",
        years=tuple(range(2008, 2024)), keys=("R_ID",), calibration="C",
        note="outward FATS, Austrian subsidiaries abroad, keyed on domestic parent",
    ),
    Product(
        oid=125351, abbrev="OFATS-UNT", table="ofats_unt_ab2021",
        years=tuple(range(2021, 2024)), keys=("Firmen_ID3",), calibration="C",
        note="outward FATS on statistical units; keyed on Firmen_ID3, not ID2",
    ),
)

BY_TABLE = {p.table: p for p in PRODUCTS}

URS_RE_COLUMNS = ("R_ID", "jahr", "IS", "RF", "oenace", "FSW_NUTS3", "FSW_LKZ",
                  "gcd_fsw", "USB", "SB", "UST", "RE_DB_DAT_NG", "RE_DB_DAT_SL")
URS_LOC_COLUMNS = ("R_ID", "Firmen_ID5", "jahr", "oenace_s", "S_NUTS3", "S_LKZ", "gcd_s")
URS_STAT_COLUMNS = ("R_ID", "Firmen_ID3", "JAHR", "UNT_IS", "UNT_OENACE")
URS_M_COLUMNS = ("R_ID", "jahr", "M_IS", "M_RF", "M_OENACE", "M_FSW_NUTS3",
                 "M_FSW_LKZ", "M_FSW_GEMNR", "M_USB", "M_SB", "M_UST",
                 "M_DB_DAT_NG", "M_DB_DAT_SL")
URS_D_COLUMNS = ("R_ID", "jahr", "T_IS", "T_RF", "T_OENACE", "T_FSW_NUTS3",
                 "T_FSW_LKZ", "T_FSW_GEMNR", "T_USB", "T_SB", "T_UST",
                 "T_DB_DAT_NG", "T_DB_DAT_SL")
URS_G_COLUMNS = ("R_ID", "Firmen_ID4", "JAHR", "GRP_ART", "GRP_LKZ",
                 "GRP_DB_DAT_NG", "GRP_DB_DAT_SL")

# Columns Intrastat holds but never fills. AMDC types them logical.
INTRA_EMPTY = ("SV", "CO", "PRAEF", "VERKINL", "VERKGRE")
INTRA_COLUMNS = ("Firmen_ID9", "R_ID", "EV", "BERM", "BERJ", "VL", "UL", "BL",
                 "KN", "GW", "SM", "SW", "GART", *INTRA_EMPTY)
EXTRA_COLUMNS = ("R_ID", "Firmen_ID9", "EV", "BERM", "BERJ", "SV", "VL", "UL", "BL",
                 "KN", "GW", "SM", "SW", "CO", "PRAEF", "VERKINL", "VERKGRE", "GART", "V1")
PRODCOM_COLUMNS = ("R_ID", "Firmen_ID1", "Firmen_ID6", "PJA", "PMO", "PCODE", "PART",
                   "PEH1", "PME1", "PEH2", "PME2", "PWERT")
GES_COLUMNS = ("Firmen_ID6", "JAHR", "NACE", "CODE", "EG", "EH", "ME", "WE")

TABLE_COLUMNS = {
    "lse_re": URS_RE_COLUMNS,
    "lse_loc": URS_LOC_COLUMNS,
    "lse_stat": URS_STAT_COLUMNS,
    "lse_m": URS_M_COLUMNS,
    "lse_d": URS_D_COLUMNS,
    "lse_g": URS_G_COLUMNS,
    "intra": INTRA_COLUMNS,
    "extra": EXTRA_COLUMNS,
    "prodcom": PRODCOM_COLUMNS,
    "ges": GES_COLUMNS,
}
