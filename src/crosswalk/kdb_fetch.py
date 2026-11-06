"""Fetch the Statistik Austria KDB inputs used by the synthetic corpus.

The repository ships compact ZIP archives containing only the classifications and
correspondence tables read by :mod:`src.crosswalk.codelists`. Rebuild those archives
from the public KDB endpoint or from an existing KDB export mirror with:

    python -m src.crosswalk.kdb_fetch
    python -m src.crosswalk.kdb_fetch --source-dir /path/to/exports
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.crosswalk.config import KDB

KDB_DOWNLOAD_BASE = "https://www.statistik.at/kdb/downloads/csv/prod"
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class KDBAsset:
    """One public KDB text export required at runtime."""

    family: str
    filename: str

    @property
    def relative_path(self) -> Path:
        return Path(self.family) / self.filename

    @property
    def url(self) -> str:
        return f"{KDB_DOWNLOAD_BASE}/{self.filename}"


def required_assets() -> tuple[KDBAsset, ...]:
    """Return the complete, deliberately narrow runtime KDB inventory."""
    assets = [KDBAsset("wirtschaftszweige/oenace", "OENACE2008_DE_CTI.txt")]
    assets.extend(
        KDBAsset("gueter/kn", f"KN{year}_DE_CTI.txt")
        for year in range(2013, 2027)
    )
    assets.extend(
        KDBAsset("gueter/kn", f"KN{year}_KN{year + 1}_DE_12B.txt")
        for year in range(2013, 2026)
    )
    for year in range(2013, 2024):
        assets.extend((
            KDBAsset("gueter/oeprodcom", f"OEPRODCOM{year}_DE_CTI.txt"),
            KDBAsset(
                "gueter/oeprodcom",
                f"OEPRODCOM{year}_KN{year}_DE_12B.txt",
            ),
            KDBAsset("gueter/prodcom", f"PRODCOM{year}_DE_CTI.txt"),
            KDBAsset(
                "gueter/prodcom",
                f"PRODCOM{year}_KN{year}_DE_12B.txt",
            ),
        ))
    # The crosswalk research window ends in 2023, but the existing synthetic
    # PRODCOM generator also emits 2024 and needs that year's OEPRODCOM code list.
    # This runtime input does not add a 2024 spoke or widen the KN family window.
    assets.append(KDBAsset("gueter/oeprodcom", "OEPRODCOM2024_DE_CTI.txt"))
    for year in range(2016, 2021):
        assets.extend((
            KDBAsset("gueter/gv-ges", f"GV-GES{year}_DE_CTI.txt"),
            KDBAsset("gueter/gv-ges", f"GV-GES{year}_KN{year}_DE_12B.txt"),
        ))
    return tuple(assets)


def _find_source(asset: KDBAsset, source_dir: Path) -> Path:
    candidates = (
        source_dir / asset.relative_path,
        source_dir / "kdb_txt" / asset.relative_path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = [path for path in source_dir.rglob(asset.filename) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"ambiguous local KDB source for {asset.filename}: {locations}")
    raise FileNotFoundError(
        f"missing {asset.relative_path} below local KDB source {source_dir}"
    )


def _read_asset(asset: KDBAsset, source_dir: Path | None) -> bytes:
    if source_dir is not None:
        return _find_source(asset, source_dir).read_bytes()
    try:
        with urllib.request.urlopen(asset.url, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
            return response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"failed to fetch {asset.url}: {error}") from error


def _archive_path(output_dir: Path, family: str) -> Path:
    family_path = Path(family)
    return output_dir / family_path.parent / f"{family_path.name}.zip"


def _zip_member(asset: KDBAsset) -> str:
    return f"{Path(asset.family).name}/{asset.filename}"


def _write_archive(
    archive_path: Path,
    assets: tuple[KDBAsset, ...],
    source_dir: Path | None,
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = archive_path.with_suffix(".zip.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for asset in assets:
                info = zipfile.ZipInfo(_zip_member(asset), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, _read_asset(asset, source_dir))
        temporary_path.replace(archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_archives(output_dir: Path, *, source_dir: Path | None = None) -> tuple[Path, ...]:
    """Build the curated family archives atomically and return their paths."""
    if source_dir is not None and not source_dir.is_dir():
        raise NotADirectoryError(f"local KDB source is not a directory: {source_dir}")

    grouped: defaultdict[str, list[KDBAsset]] = defaultdict(list)
    for asset in required_assets():
        grouped[asset.family].append(asset)

    archives = []
    for family in sorted(grouped):
        archive_path = _archive_path(output_dir, family)
        _write_archive(archive_path, tuple(grouped[family]), source_dir)
        archives.append(archive_path)
    return tuple(archives)


def validate_archives(kdb_dir: Path) -> tuple[str, ...]:
    """Return validation failures for the curated archive inventory."""
    failures = []
    grouped: defaultdict[Path, set[str]] = defaultdict(set)
    for asset in required_assets():
        grouped[_archive_path(kdb_dir, asset.family)].add(_zip_member(asset))

    for archive_path, expected_members in grouped.items():
        if not archive_path.is_file():
            failures.append(f"missing archive: {archive_path}")
            continue
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = set(archive.namelist())
                bad_member = archive.testzip()
        except (OSError, zipfile.BadZipFile) as error:
            failures.append(f"invalid archive {archive_path}: {error}")
            continue
        for member in sorted(expected_members - members):
            failures.append(f"missing {member} in {archive_path}")
        for member in sorted(members - expected_members):
            failures.append(f"unexpected {member} in {archive_path}")
        if bad_member is not None:
            failures.append(f"corrupt {bad_member} in {archive_path}")
    return tuple(failures)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KDB,
        help=f"curated KDB root (default: {KDB})",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="copy from an existing exports tree instead of downloading",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the curated archives without changing them",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    if args.check:
        failures = validate_archives(output_dir)
        if failures:
            for failure in failures:
                print(f"error: {failure}", file=sys.stderr)
            raise SystemExit(1)
        print(f"KDB archives valid: {len(required_assets())} files in {output_dir}")
        return

    source_dir = args.source_dir.resolve(strict=True) if args.source_dir is not None else None
    archives = build_archives(output_dir, source_dir=source_dir)
    for archive in archives:
        print(f"KDB archive: {archive}")
    print(f"KDB inputs ready: {len(required_assets())} files")


if __name__ == "__main__":
    main()
