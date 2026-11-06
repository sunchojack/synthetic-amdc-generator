"""Read AMDC's published per-variable statistics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from src.crosswalk.config import CATALOGUE, CODEBOOKS
from src.crosswalk.schema import BY_TABLE


@dataclass(frozen=True)
class VarStats:
    name: str
    cls: str
    semantic: str
    years: tuple[int, ...]
    rows_per_year: tuple[int, ...]
    missing_per_year: tuple[int, ...]
    missing_rate_per_year: tuple[float, ...]
    unique_per_year: tuple[int, ...]
    total_rows: int
    total_unique: int
    total_missing: int
    numeric: dict[str, float]
    levels: dict[str, int]

    @property
    def all_missing(self) -> bool:
        return self.total_rows > 0 and self.total_missing == self.total_rows

    def rate(self, year: int) -> float:
        """Missing rate for one year; 0.0 where the variable has no entry for it."""
        try:
            return self.missing_rate_per_year[self.years.index(year)]
        except (ValueError, IndexError):
            return 0.0

    def level_shares(self) -> dict[str, float]:
        tot = sum(self.levels.values())
        return {k: v / tot for k, v in self.levels.items()} if tot else {}


@dataclass(frozen=True)
class Batch:
    table: str
    vars: dict[str, VarStats]

    def __getitem__(self, k: str) -> VarStats:
        return self.vars[k]

    def get(self, k: str) -> VarStats | None:
        return self.vars.get(k)

    def rows_per_year(self) -> dict[int, int]:
        """Published row count per year, taken from the year-partition variable."""
        for cand in ("BERJ", "PJA", "JAHR", "Jahr", "jahr"):
            v = self.vars.get(cand)
            if v is not None and v.levels:
                return {int(y): n for y, n in v.levels.items()}
        v = next(iter(self.vars.values()))
        return dict(zip(v.years, v.rows_per_year))


def _years(raw: dict) -> tuple[int, ...]:
    ys = raw.get("years_available") or []
    return tuple(int(y) for y in ys)


def _seq(raw: dict, key: str) -> tuple:
    v = raw.get(key)
    return tuple(v) if isinstance(v, list) else ()


def _dict(raw: dict, key: str) -> dict:
    v = raw.get(key)
    return dict(v) if isinstance(v, dict) else {}


@lru_cache(maxsize=None)
def load_batch(table: str) -> Batch | None:
    """Load the metadata_batch JSON for a table, or None where AMDC ships none."""
    prod = BY_TABLE[table]
    if not prod.metadata_batch:
        return None
    path = CODEBOOKS / prod.metadata_batch
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, VarStats] = {}
    for name, v in raw.items():
        years = _years(v)
        rates = _seq(v, "missing_rate_per_year")
        miss = _seq(v, "missing_per_year")
        rows = _seq(v, "total_rows_per_year")
        # AMDC occasionally publishes one fewer rate than years (Intrastat SV et al.)
        if years and len(rates) < len(years):
            rates = rates + (rates[-1] if rates else 0.0,) * (len(years) - len(rates))
        out[name] = VarStats(
            name=name,
            cls=v.get("class") or "",
            semantic=v.get("semantic_type") or "",
            years=years,
            rows_per_year=rows,
            missing_per_year=miss,
            missing_rate_per_year=tuple(float(x) for x in rates),
            unique_per_year=_seq(v, "unique_values_per_year"),
            total_rows=int(v.get("total_rows") or 0),
            total_unique=int(v.get("total_unique_values") or 0),
            total_missing=int(v.get("total_missing") or 0),
            numeric=_dict(v, "numeric_summary"),
            levels=_dict(v, "level_counts"),
        )
    return Batch(table=table, vars=out)


@lru_cache(maxsize=None)
def catalogue_variables(table: str) -> list[str]:
    oid = BY_TABLE[table].oid
    path = CATALOGUE / f"{oid}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for grp in sorted(rec.get("featureGroup") or [], key=lambda g: g.get("order", 0)):
        for ft in sorted(grp.get("feature") or [], key=lambda f: f.get("order", 0)):
            names.append(ft["techName"])
    return names
