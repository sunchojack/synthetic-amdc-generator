from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path

from amdc_synth.config import KDB
from src.crosswalk.kdb_fetch import build_archives, required_assets, validate_archives


def _write_source(source_dir: Path) -> None:
    for asset in required_assets():
        path = source_dir / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"EDV-Code;Titel\n{asset.filename};test\n", encoding="latin-1")


class KDBFetchTests(unittest.TestCase):
    def test_required_inventory_is_exact_and_unique(self) -> None:
        assets = required_assets()
        actual = {(asset.family, asset.filename) for asset in assets}
        expected = {("wirtschaftszweige/oenace", "OENACE2008_DE_CTI.txt")}
        expected.update(
            ("gueter/kn", f"KN{year}_DE_CTI.txt") for year in range(2013, 2027)
        )
        expected.update(
            ("gueter/kn", f"KN{year}_KN{year + 1}_DE_12B.txt")
            for year in range(2013, 2026)
        )
        for year in range(2013, 2024):
            expected.add(("gueter/oeprodcom", f"OEPRODCOM{year}_DE_CTI.txt"))
            expected.add((
                "gueter/oeprodcom", f"OEPRODCOM{year}_KN{year}_DE_12B.txt"
            ))
            expected.add(("gueter/prodcom", f"PRODCOM{year}_DE_CTI.txt"))
            expected.add((
                "gueter/prodcom", f"PRODCOM{year}_KN{year}_DE_12B.txt"
            ))
        expected.add(("gueter/oeprodcom", "OEPRODCOM2024_DE_CTI.txt"))
        for year in range(2016, 2021):
            expected.add(("gueter/gv-ges", f"GV-GES{year}_DE_CTI.txt"))
            expected.add(("gueter/gv-ges", f"GV-GES{year}_KN{year}_DE_12B.txt"))

        self.assertEqual(actual, expected)
        self.assertEqual(len(assets), 83)

    def test_build_archives_from_local_export_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "exports"
            _write_source(source_dir)
            first = build_archives(root / "first", source_dir=source_dir)
            second = build_archives(root / "second", source_dir=source_dir)

            self.assertEqual(len(first), 5)
            self.assertEqual(validate_archives(root / "first"), ())
            self.assertEqual(
                [hashlib.sha256(path.read_bytes()).digest() for path in first],
                [hashlib.sha256(path.read_bytes()).digest() for path in second],
            )
            for archive_path in first:
                with zipfile.ZipFile(archive_path) as archive:
                    self.assertTrue(all(
                        len(Path(member).parts) == 2 for member in archive.namelist()
                    ))
                    self.assertTrue(all(
                        info.date_time == (1980, 1, 1, 0, 0, 0)
                        for info in archive.infolist()
                    ))

    def test_build_archives_rejects_missing_file_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "exports"
            _write_source(source_dir)
            missing = source_dir / "gueter/kn/KN2025_KN2026_DE_12B.txt"
            missing.unlink()

            with self.assertRaisesRegex(FileNotFoundError, "KN2025_KN2026"):
                build_archives(root / "curated", source_dir=source_dir)

    def test_validation_rejects_unpinned_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "exports"
            _write_source(source_dir)
            build_archives(root / "curated", source_dir=source_dir)
            archive_path = root / "curated/gueter/kn.zip"
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("kn/KN2012_DE_CTI.txt", b"unexpected")

            self.assertTrue(any(
                "unexpected kn/KN2012_DE_CTI.txt" in failure
                for failure in validate_archives(root / "curated")
            ))

    def test_repository_archives_cover_exact_runtime_inventory(self) -> None:
        self.assertEqual(validate_archives(KDB), ())


if __name__ == "__main__":
    unittest.main()
