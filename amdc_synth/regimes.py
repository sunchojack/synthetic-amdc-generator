"""Documented AMDC time regimes, unit conversions, and comparability breaks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl


GES_PUBLIC_UNITS_MIN = 2_500
GES_PUBLIC_UNITS_MAX = 2_700
GES_PUBLIC_UNITS_TARGET = 2_600
GES_VALUE_UNIT_EUR = 1_000

LSE_VALUE_UNIT_EUR = 1_000
FATS_VALUE_UNIT_EUR = 1_000


@dataclass(frozen=True)
class AmdcRegime:
    """One source regime that affects cross-year or cross-product comparison."""

    table_name: str
    first_year: int
    last_year: int
    kind: str
    columns: str
    source_representation: str
    multiplier_to_eur: int | None
    amdc_difference: str
    evidence: str


AMDC_REGIMES: tuple[AmdcRegime, ...] = (
    AmdcRegime(
        "intra", 2013, 2021, "availability", "UL",
        "Origin is populated on arrivals and absent on dispatches.", None,
        "From 2022, origin is also available on almost all dispatches.",
        "AMDC Intrastat metadata batch, oid 152509",
    ),
    AmdcRegime(
        "intra", 2022, 2023, "coding", "GART, UL",
        "GART uses two-digit transaction codes; UL expands to dispatches.", None,
        "Do not pool raw GART codes across the 2022 boundary.",
        "AMDC Intrastat metadata batch, oid 152509",
    ),
    AmdcRegime(
        "extra", 2022, 2022, "availability", "V1",
        "One-based row counter, present in 2022 only.", None,
        "V1 is not an economic measure and must not enter models.",
        "AMDC Extrastat metadata batch, oid 152510",
    ),
    AmdcRegime(
        "prodcom", 2023, 2024, "calibration", "all",
        "AMDC advertises these years but publishes no per-variable summary batch.", None,
        "Rows and marginals are extrapolated from 2022 and are not calibrated observations.",
        "AMDC KJEPB_P catalogue and metadata batch, oid 152511",
    ),
    AmdcRegime(
        "ges", 2008, 2023, "unit", "WE",
        "Input value stored in EUR 1,000 in every available year.", GES_VALUE_UNIT_EUR,
        "GES has no 2021/2022 currency-unit break; abrupt value changes are data/model changes.",
        "AMDC GES codebook, oid 125309",
    ),
    AmdcRegime(
        "ges", 2008, 2023, "population", "Firmen_ID6",
        "Concentration sample of approximately 2,500-2,700 establishments per year.", None,
        "The synthetic build targets the midpoint after applying SCALE.",
        "AMDC GES catalogue, oid 61422",
    ),
    AmdcRegime(
        "ges", 2008, 2015, "classification", "CODE",
        "Nearest available 2016 GV-GES vintage is reused.", None,
        "Real AMDC uses the reporting-year vintage; early synthetic code membership is approximate.",
        "Local KDB mirror contains GV-GES vintages 2016-2020",
    ),
    AmdcRegime(
        "ges", 2021, 2023, "classification", "CODE",
        "Nearest available 2020 GV-GES vintage is reused.", None,
        "Real AMDC uses the reporting-year vintage; late synthetic code membership is approximate.",
        "Local KDB mirror contains GV-GES vintages 2016-2020",
    ),
    AmdcRegime(
        "lse_re_bis2020", 2008, 2020, "schema", "monetary measures",
        "Pre-2021 LSE layout; monetary values are in EUR 1,000.", LSE_VALUE_UNIT_EUR,
        "BWSF is available; questionnaire blocks and population scope differ from 2021 onward.",
        "AMDC LSE codebook, oid 109741",
    ),
    AmdcRegime(
        "lse_re_ab2021", 2021, 2023, "schema", "monetary measures",
        "Post-2020 LSE layout; monetary values remain in EUR 1,000.", LSE_VALUE_UNIT_EUR,
        "BWS plus AVORL replace BWSF; names, scope, and population change at 2021.",
        "AMDC LSE codebook, oid 131453",
    ),
    AmdcRegime(
        "ifats_re_bis2020", 2008, 2020, "scope", "BWSF, monetary measures",
        "Legal-unit scope; monetary values are in EUR 1,000.", FATS_VALUE_UNIT_EUR,
        "The population expands and BWS replaces BWSF from 2021.",
        "AMDC IFATS codebook, oid 153271",
    ),
    AmdcRegime(
        "ifats_re_ab2021", 2021, 2023, "scope", "BWS, monetary measures",
        "Expanded legal-unit scope; monetary values remain in EUR 1,000.", FATS_VALUE_UNIT_EUR,
        "Do not interpret the 2021 level change as business dynamics alone.",
        "AMDC IFATS codebook, oid 153297",
    ),
    AmdcRegime(
        "ofats_re", 2008, 2020, "availability", "PERSA_AUE, ISACH_AUE",
        "Personnel cost and tangible investment are unavailable.", None,
        "Both fields become available from reporting year 2021.",
        "AMDC OFATS-RE codebook, oid 153230",
    ),
    AmdcRegime(
        "ofats_re", 2021, 2023, "unit", "monetary measures",
        "Monetary values are in EUR 1,000; PERSA_AUE and ISACH_AUE are populated.",
        FATS_VALUE_UNIT_EUR,
        "The new fields create a documented missingness break at 2021.",
        "AMDC OFATS-RE codebook, oid 153230",
    ),
    AmdcRegime(
        "ofats_unt_ab2021", 2021, 2023, "statistical unit", "Firmen_ID3",
        "Domestic controller is a statistical enterprise; monetary values are in EUR 1,000.",
        FATS_VALUE_UNIT_EUR,
        "Keep Firmen_ID3 opaque; identifier bridging is deferred.",
        "AMDC OFATS-UNT codebook, oid 153227",
    ),
)


def regime_frame() -> pl.DataFrame:
    """Return the regime contract as a synthetic-only metadata table."""
    return pl.DataFrame([asdict(regime) for regime in AMDC_REGIMES])
