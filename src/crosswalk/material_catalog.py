"""Parse the JRC material-to-CN and PRODCOM inventory."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from src.crosswalk.config import JRC_PDF, REFERENCE_CACHE


STAGES = ("Extraction/Mining", "Processing/Refining", "Fabrication", "Manufacture", "Recycling", "Use")

# EC 2023 criticality assessment: materials flagged Critical or Strategic that also
# appear in the JRC trade-code inventory.
CRITICAL = {
    "Antimony", "Arsenic", "Baryte", "Bauxite", "Beryllium", "Bismuth", "Boron",
    "Cerium", "Cobalt", "Coking coal", "Copper", "Dysprosium", "Erbium", "Europium",
    "Feldspar", "Fluorspar", "Gadolinium", "Gallium", "Germanium", "Graphite",
    "Hafnium", "Helium", "Holmium", "Iridium", "Lanthanum", "Lithium", "Lutetium",
    "Magnesium", "Manganese", "Neodymium", "Nickel", "Niobium", "Palladium",
    "Phosphate", "Phosphorus", "Platinum", "Praseodymium", "Rhodium", "Ruthenium",
    "Samarium", "Scandium", "Silicon", "Strontium", "Tantalum", "Terbium",
    "Thulium", "Titanium", "Tungsten", "Vanadium", "Ytterbium", "Yttrium", "Zirconium",
}


@dataclass
class Material:
    name: str
    critical: bool
    cn_by_stage: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    prodcom_by_stage: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    @property
    def upstream_cn(self) -> list[int]:
        """CN8 codes a downstream Austrian firm would buy in."""
        return sorted(set(self.cn_by_stage.get("Extraction/Mining", [])
                          + self.cn_by_stage.get("Processing/Refining", [])))

    @property
    def downstream_cn(self) -> list[int]:
        return sorted(set(self.cn_by_stage.get("Fabrication", [])
                          + self.cn_by_stage.get("Manufacture", [])))

    @property
    def scrap_cn(self) -> list[int]:
        return sorted(set(self.cn_by_stage.get("Recycling", [])))

    def prodcom(self, *stages: str) -> list[int]:
        out: set[int] = set()
        for s in stages:
            out.update(self.prodcom_by_stage.get(s, []))
        return sorted(out)


# ------------------------------------------------------------------ extraction

_HEADERS = ("Raw", "Commodity /", "Commodity /", "CN code", "Number", "Raw",
            "Supply Chain", "Supply Chain Stage", "PRODCOM")


def _column_bounds(header: str) -> list[int] | None:
    """Character offsets of each table column, read off the page header line."""
    pos, cur = [], 0
    for h in _HEADERS:
        i = header.find(h, cur)
        if i < 0:
            return None
        pos.append(i)
        cur = i + len(h)
    return pos


_STAGE_RE = re.compile("|".join(re.escape(s) for s in STAGES))
_CODE_RE = re.compile(r"\b(\d{8})\b")


def _parse_pdf() -> dict[str, Material]:
    import pypdf

    reader = pypdf.PdfReader(str(JRC_PDF))
    mats: dict[str, Material] = {}
    for page in reader.pages:
        text = page.extract_text(extraction_mode="layout") or ""
        if "CN code" not in text or "Supply Chain" not in text:
            continue
        lines = text.splitlines()
        header = next((l for l in lines if "CN code" in l and "PRODCOM" in l), None)
        bounds = _column_bounds(header) if header else None
        if not bounds:
            continue
        c_mat, _, _, c_cn, c_nr, _, c_stage, c_spec, c_pc = bounds
        last_mat = ""
        for line in lines:
            if len(line) < c_cn:
                continue
            cn_cell = line[c_cn:c_nr]
            m = _CODE_RE.search(cn_cell)
            if not m:
                continue
            stage_cell = line[c_stage:c_spec]
            sm = _STAGE_RE.search(stage_cell) or _STAGE_RE.search(line[c_stage:])
            if not sm:
                continue
            name = line[c_mat:bounds[1]].strip() or last_mat
            if not name or not name[0].isalpha():
                name = last_mat
            if not name:
                continue
            last_mat = name
            mat = mats.setdefault(name, Material(name=name, critical=name in CRITICAL))
            mat.cn_by_stage.setdefault(sm.group(0), []).append(int(m.group(1)))
            pm = _CODE_RE.search(line[c_pc:])
            if pm:
                mat.prodcom_by_stage.setdefault(sm.group(0), []).append(int(pm.group(1)))
    for mat in mats.values():
        for d in (mat.cn_by_stage, mat.prodcom_by_stage):
            for k, v in d.items():
                d[k] = sorted(set(v))
    return mats


_CACHE_FILE = REFERENCE_CACHE / "jrc_crm_codes.json"


def materials() -> dict[str, Material]:
    """CRM code sets from the JRC inventory, cached after first parse."""
    pdf_sha256 = hashlib.sha256(JRC_PDF.read_bytes()).hexdigest()
    if _CACHE_FILE.exists():
        raw = json.loads(_CACHE_FILE.read_text())
        if raw.get("jrc_pdf_sha256") == pdf_sha256:
            return {
                name: Material(
                    name=name,
                    critical=d["critical"],
                    cn_by_stage={k: list(v) for k, v in d["cn"].items()},
                    prodcom_by_stage={k: list(v) for k, v in d["prodcom"].items()},
                )
                for name, d in raw["materials"].items()
            }
    mats = _parse_pdf()
    REFERENCE_CACHE.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(
        {
            "jrc_pdf_sha256": pdf_sha256,
            "materials": {
                m.name: {"critical": m.critical, "cn": m.cn_by_stage, "prodcom": m.prodcom_by_stage}
                for m in mats.values()
            },
        },
        indent=1,
    ))
    return mats
