"""Re-fetch the AMDC catalogue records, codebooks and published statistics.

The build never touches the network; it reads `amdc_reference/`. Run this when AMDC
updates a product - new reporting year, revised codebook, a variable added - and then
rebuild, so the generated data tracks the real catalogue.

    python -m amdc_synth.refresh

Only the public, unauthenticated catalogue is read. Nothing is sent.
"""

from __future__ import annotations

import json
import re
import urllib.request

from .config import AMDC_REST, CATALOGUE, CODEBOOKS, PRODUCTS

TIMEOUT = 60


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as fh:  # noqa: S310
        return fh.read()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def refresh() -> None:
    CATALOGUE.mkdir(parents=True, exist_ok=True)
    CODEBOOKS.mkdir(parents=True, exist_ok=True)

    summaries = _get(f"{AMDC_REST}/getProductSummaries")
    (CATALOGUE / "_all_product_summaries.json").write_bytes(summaries)
    print(f"catalogue: {len(json.loads(summaries))} products")

    for oid in sorted({p.oid for p in PRODUCTS}):
        raw = _get(f"{AMDC_REST}/getProduct/{oid}")
        (CATALOGUE / f"{oid}.json").write_bytes(raw)
        rec = json.loads(raw)
        abbrev = rec.get("abbreviation") or str(oid)
        for doc in rec.get("relevantDocs") or []:
            fn = doc.get("fileName")
            if not fn:
                continue
            out = CODEBOOKS / f"{doc['oid']}__{_safe(abbrev)}__{_safe(fn)}"
            out.write_bytes(_get(f"{AMDC_REST}/getFile/{doc['oid']}"))
            print(f"  {oid} {abbrev}: {out.name}")


if __name__ == "__main__":
    refresh()
