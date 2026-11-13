from __future__ import annotations

from pathlib import Path
import tempfile
import tomllib
import unittest

from scripts.validate_changed_ruff import select_changed_python_paths


ROOT = Path(__file__).resolve().parents[1]


class ChangedRuffSelectionTests(unittest.TestCase):
    def test_selects_python_source_and_skips_generated_or_missing_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "scripts").mkdir()
            (root / "scripts" / "validator.py").write_text("pass\n", encoding="utf-8")
            (root / "generated").mkdir()
            (root / "generated" / "model.py").write_text("pass\n", encoding="utf-8")
            vendor_paths = (
                "app/ui/dayu_widgets/widget.py",
                "ctd_vendor/module.py",
                "modules/ocr/pororo/pororo/vendor.py",
                "modules/ocr/pororo/engine.py",
            )
            for value in vendor_paths:
                vendor = root / value
                vendor.parent.mkdir(parents=True, exist_ok=True)
                vendor.write_text("pass\n", encoding="utf-8")
            (root / "owned.py").write_text("pass\n", encoding="utf-8")

            selected = select_changed_python_paths(
                [
                    "scripts/validator.py",
                    "generated/model.py",
                    *vendor_paths,
                    "owned.py",
                    "missing.py",
                    "README.md",
                ],
                root=root,
            )

        self.assertEqual(
            selected,
            ["modules/ocr/pororo/engine.py", "owned.py", "scripts/validator.py"],
        )

    def test_ruff_excludes_only_the_actual_vendored_roots(self) -> None:
        configuration = tomllib.loads((ROOT / "ruff.toml").read_text(encoding="utf-8"))
        excluded = configuration["extend-exclude"]

        self.assertTrue(
            {
                "app/ui/dayu_widgets/**",
                "ctd_vendor/**",
                "modules/ocr/pororo/pororo/**",
            }.issubset(excluded)
        )
        self.assertNotIn("**/dayu/**", excluded)
        self.assertNotIn("**/pororo/**", excluded)

    def test_external_symlink_is_not_linted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            outside = Path(temporary_directory) / "outside.py"
            root.mkdir()
            outside.write_text("pass\n", encoding="utf-8")
            try:
                (root / "linked.py").symlink_to(outside)
            except OSError:
                self.skipTest("symbolic links are unavailable in this environment")

            selected = select_changed_python_paths(["linked.py"], root=root)

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
