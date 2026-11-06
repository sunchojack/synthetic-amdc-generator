"""Repository inputs and output defaults for the reusable crosswalk."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = ROOT / "amdc_reference"
CATALOGUE = REFERENCE / "catalogue"
CODEBOOKS = REFERENCE / "codebooks"

# Curated Statistik Austria classification exports shipped as compact family ZIPs.
KDB = REFERENCE / "kdb"
KDB_KN = KDB / "gueter" / "kn"
KDB_OEPRODCOM = KDB / "gueter" / "oeprodcom"
KDB_GVGES = KDB / "gueter" / "gv-ges"
KDB_OENACE = KDB / "wirtschaftszweige" / "oenace"

JRC_PDF = CODEBOOKS / "JRC136041_01__crm_codes_CN8_AND_PRODCOM.pdf"
DERA_PDF = CODEBOOKS / "66_Rohstoffhandel und Wertschöpfung.pdf"
REFERENCE_CACHE = ROOT / ".cache" / "amdc_reference"
if not DERA_PDF.is_file():
    DERA_PDF = next(CODEBOOKS.glob("*Rohstoffhandel*.pdf"), DERA_PDF)

CROSSWALK = ROOT / "crosswalk"
PARQUET = CROSSWALK / "parquet"

# AMDC public catalogue, used to refresh amdc_reference/
AMDC_REST = "https://www.statistik.at/amdc-data/rest"
