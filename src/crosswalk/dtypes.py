import re

import polars as pl

IDENTIFIER_DTYPES = {
    "R_ID": "VARCHAR",
    "Firmen_ID6": "VARCHAR",
    "KN": "VARCHAR",
    "hs4": "VARCHAR",
    "PCODE": "VARCHAR",
    "NACE": "VARCHAR",
    "OENACE": "VARCHAR",
    "NUTS3": "VARCHAR",
    "GART": "VARCHAR",
    "upstream_cn8": "VARCHAR[]",
    "downstream_cn8": "VARCHAR[]",
    "prodcom8": "VARCHAR[]",
}

_POLARS_DTYPES = {"VARCHAR": pl.Utf8, "VARCHAR[]": pl.List(pl.Utf8)}

_RELATION_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}")


def check_frame(df: pl.DataFrame, label: str) -> None:
    violations = []
    for name, expected in IDENTIFIER_DTYPES.items():
        if name not in df.schema:
            continue
        actual = df.schema[name]
        if actual != _POLARS_DTYPES[expected]:
            violations.append(f"{label}: column {name} expected {expected} got {actual}")
    if violations:
        raise TypeError("\n".join(violations))


def check_relation(con, relation: str) -> None:
    if not _RELATION_RE.fullmatch(relation):
        raise ValueError(f"invalid relation name: {relation}")
    rows = con.execute(f"DESCRIBE {relation}").fetchall()
    actual_by_name = {row[0]: row[1] for row in rows}
    violations = []
    for name, expected in IDENTIFIER_DTYPES.items():
        if name not in actual_by_name:
            continue
        actual = actual_by_name[name]
        if actual != expected:
            violations.append(f"{relation}: column {name} expected {expected} got {actual}")
    if violations:
        raise TypeError("\n".join(violations))
