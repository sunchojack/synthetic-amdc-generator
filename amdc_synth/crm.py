"""Synthetic supply assumptions layered over the source-backed material catalogue."""

from src.crosswalk.material_catalog import Material, materials


# Approximate global supply concentration by material, from public USGS/JRC figures.
# Drives the origin mix of CRM imports so that a country shock has a recoverable
# footprint. Residual mass is spread over the rest of the geonomenclature.
SUPPLY_CONCENTRATION: dict[str, dict[str, float]] = {
    "Gallium": {"CN": 0.80, "JP": 0.05, "KR": 0.04},
    "Germanium": {"CN": 0.60, "RU": 0.10, "US": 0.08},
    "Neodymium": {"CN": 0.85, "MY": 0.04},
    "Praseodymium": {"CN": 0.85, "MY": 0.04},
    "Dysprosium": {"CN": 0.88, "MY": 0.03},
    "Terbium": {"CN": 0.88},
    "Samarium": {"CN": 0.85},
    "Cerium": {"CN": 0.80},
    "Lanthanum": {"CN": 0.80},
    "Yttrium": {"CN": 0.85},
    "Scandium": {"CN": 0.60, "RU": 0.15},
    "Palladium": {"RU": 0.40, "ZA": 0.35, "CA": 0.08},
    "Platinum": {"ZA": 0.70, "RU": 0.11, "ZW": 0.08},
    "Rhodium": {"ZA": 0.80, "RU": 0.10},
    "Ruthenium": {"ZA": 0.80, "RU": 0.09},
    "Iridium": {"ZA": 0.85},
    "Cobalt": {"CD": 0.68, "CN": 0.12, "RU": 0.04},
    "Graphite": {"CN": 0.65, "MZ": 0.09, "MG": 0.06},
    "Magnesium": {"CN": 0.90, "US": 0.03},
    "Lithium": {"CL": 0.35, "AU": 0.30, "CN": 0.20},
    "Niobium": {"BR": 0.85, "CA": 0.08},
    "Tantalum": {"CD": 0.40, "RW": 0.30, "BR": 0.09},
    "Tungsten": {"CN": 0.80, "VN": 0.05},
    "Vanadium": {"CN": 0.40, "ZA": 0.25, "RU": 0.20},
    "Titanium": {"CN": 0.30, "RU": 0.20, "JP": 0.10},
    "Antimony": {"CN": 0.55, "TR": 0.15, "TJ": 0.12},
    "Bismuth": {"CN": 0.75, "LA": 0.06},
    "Boron": {"TR": 0.75, "US": 0.10},
    "Silicon": {"CN": 0.60, "NO": 0.15, "BR": 0.06},
    "Phosphate": {"MA": 0.35, "CN": 0.20, "RU": 0.12},
    "Phosphorus": {"CN": 0.70, "KZ": 0.10},
    "Beryllium": {"US": 0.65, "CN": 0.20},
    "Hafnium": {"FR": 0.50, "US": 0.25},
    "Manganese": {"ZA": 0.40, "GA": 0.20, "AU": 0.15},
    "Nickel": {"ID": 0.30, "RU": 0.18, "CA": 0.08},
    "Copper": {"CL": 0.30, "PE": 0.15, "CD": 0.10},
    "Bauxite": {"AU": 0.30, "GN": 0.25, "CN": 0.18},
    "Fluorspar": {"CN": 0.60, "MX": 0.15},
    "Baryte": {"CN": 0.35, "IN": 0.15, "MA": 0.10},
    "Strontium": {"ES": 0.35, "IR": 0.25, "CN": 0.15},
    "Zirconium": {"AU": 0.35, "ZA": 0.25, "CN": 0.10},
    "Helium": {"US": 0.50, "QA": 0.30, "DZ": 0.08},
}


def concentration(material: Material | str) -> dict[str, float]:
    name = material.name if isinstance(material, Material) else material
    return SUPPLY_CONCENTRATION.get(name, {})


def signal_materials(min_cn: int = 3) -> list[Material]:
    """Critical materials with enough codes and a known supply concentration."""
    out = [m for m in materials().values()
           if m.critical and len(m.upstream_cn) >= min_cn and concentration(m)]
    return sorted(out, key=lambda m: m.name)
