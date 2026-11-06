# Synthetic AMDC

This repository builds synthetic versions of 16 Austrian Micro Data Center products.
It provides a development fixture for the CRM study pipeline outside the AMDC environment.
Schemas and code domains follow public metadata. Confidential joint distributions are modelled.

> **Synthetic data warning:** generated values test adapters, joins, and pipeline behavior.
> They are not Austrian estimates and cannot validate economic magnitudes, recipes, or resilience.

## Quickstart

Create an environment, install the pinned dependencies, and build the synthetic source:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/build_synthetic_amdc.py
```

The default scale is `0.01`. Use a smaller isolated smoke build with:

```bash
.venv/bin/python src/build_synthetic_amdc.py --scale 0.001 --out-dir out/smoke
```

Scale changes row and firm counts, not classification breadth.

## Outputs

- `out/amdc_synthesized.duckdb` and synthetic Parquet tables: development source data.
- `out/calibration_report.md`: comparison of generated variables with published AMDC statistics.
- `out/parquet/`: generated product and synthetic support tables.

The generator verifies source inputs and the staged output before publishing it.

## External references

The JRC and DERA PDFs are not included in this repository. Obtain the exact files separately
and place them under `amdc_reference/codebooks/` using the filenames recorded in
`references/jrc/SOURCE_URL.txt` and `references/dera/SOURCE_URL.txt`.
