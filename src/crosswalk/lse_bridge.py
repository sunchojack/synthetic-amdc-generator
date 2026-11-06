"""Read the official LSE 2020/2021 variable key.

The workbook is the source of truth.  Keeping this tiny reader here avoids a second,
hand-written variable map in the generator and in the harmonization layer.
"""

from __future__ import annotations

import csv
import zipfile
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import re
from xml.etree import ElementTree as ET

import polars as pl

from src.crosswalk.config import CODEBOOKS


SOURCE = CODEBOOKS / "124514__LSE_-_RE__LSE_Schl_sselliste_Rechtliche_Einheiten_bis2020_ab2021.xlsx"
_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_DEFINITION_CHANGES = {("BWSF", "BWS")}


@dataclass(frozen=True)
class LseBridge:
    group: str
    old: str | None
    new: str | None
    target_expression: str | None
    target_tokens: tuple[str, ...]
    description: str
    relation: str


def _clean(value: str) -> str | None:
    value = value.strip()
    return None if not value or value.lower() == "na" else value


def _column(ref: str) -> str:
    return "".join(char for char in ref if char.isalpha())


def _tokens(expression: str | None) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Z][A-Z0-9_]*", expression or ""))


def _sheet_rows(path) -> list[dict[str, str]]:
    """Read the first XLSX sheet without adding a workbook dependency."""
    if not path.exists():
        with path.with_suffix(".csv").open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    with zipfile.ZipFile(path) as book:
        shared_root = ET.fromstring(book.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.iterfind(".//x:t", _NS))
                  for item in shared_root.findall("x:si", _NS)]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet.findall(".//x:sheetData/x:row", _NS):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", _NS):
            raw = cell.findtext("x:v", default="", namespaces=_NS)
            if cell.get("t") == "s" and raw:
                raw = shared[int(raw)]
            values[_column(cell.get("r", ""))] = raw
        rows.append(values)
    return rows


@lru_cache(maxsize=1)
def entries() -> tuple[LseBridge, ...]:
    """Return all documented old/new LSE variable links, including explicit drops."""
    rows = _sheet_rows(SOURCE)
    if not rows or rows[0] != {
        "A": "Merkmalsgruppe", "B": "Merkmalsname_bisBJ2020",
        "C": "Merkmalsname_abBJ2021", "D": "Merkmalsbeschreibung",
    }:
        raise RuntimeError(f"unexpected LSE key layout: {SOURCE}")

    raw = []
    for row in rows[1:]:
        old = _clean(row.get("B", ""))
        expression = _clean(row.get("C", ""))
        if not old and not expression:  # citation/footer row
            continue
        tokens = _tokens(expression)
        new = tokens[0] if len(tokens) == 1 else None
        raw.append((row.get("A", "").strip(), old, new, expression, tokens,
                    row.get("D", "").strip()))
    old_targets = Counter(old for _, old, new, expression, tokens, _ in raw
                          if old and new and len(tokens) == 1)
    new_sources = Counter(new for _, old, new, expression, tokens, _ in raw
                          if old and new and len(tokens) == 1)

    out = []
    for group, old, new, expression, tokens, description in raw:
        if old and len(tokens) > 1:
            relation = "derived_sum"
        elif old and new:
            # The key runs old -> new: source fields on the left, targets right.
            relation = f"{'1' if new_sources[new] == 1 else 'n'}:{'1' if old_targets[old] == 1 else 'n'}"
            if (old, new) in _DEFINITION_CHANGES:
                relation = "definition_change"
        elif old:
            relation = "dropped"
        else:
            relation = "new"
        out.append(LseBridge(group, old, new, expression, tokens, description, relation))
    return tuple(out)


def frame() -> pl.DataFrame:
    """Parquet-ready source key used by the synthetic fixture and crosswalk layer."""
    rows = entries()
    return pl.DataFrame({
        "group": [row.group for row in rows],
        "lse_bis2020": [row.old for row in rows],
        "lse_ab2021": [row.new for row in rows],
        "target_expression": [row.target_expression for row in rows],
        "target_tokens": [list(row.target_tokens) for row in rows],
        "description": [row.description for row in rows],
        "relation": [row.relation for row in rows],
        "source_file": [SOURCE.name] * len(rows),
    }, schema={
        "group": pl.Utf8, "lse_bis2020": pl.Utf8, "lse_ab2021": pl.Utf8,
        "target_expression": pl.Utf8, "target_tokens": pl.List(pl.Utf8),
        "description": pl.Utf8, "relation": pl.Utf8, "source_file": pl.Utf8,
    })
