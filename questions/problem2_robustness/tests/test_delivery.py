"""交付边界测试不依赖官方数据或大型编码器，失败用例必须明确拒绝。"""

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.common import file_hash, write_json
from src.delivery_utils import check_manifest, extract_checked
from src.predict_attachment3 import check_values, export_rows, inside


class DeliveryTests(unittest.TestCase):
    def test_inside_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            for value in (
                "../outside",
                "/outside",
                "C:/outside",
                "a/../../b",
                "a\\b",
                "",
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    inside(d, value)
            self.assertEqual(
                inside(d, "models/a.pt"), Path(d).resolve() / "models/a.pt"
            )

    def test_probabilities_and_intensity(self):
        check_values([[0.2, 0.3, 0.5]], [0.4])
        for p, y in (
            ([[0.1, 0.1, 0.1]], [0]),
            ([[np.nan, 0, 1]], [0]),
            ([[-0.1, 0.1, 1]], [0]),
            ([[0, 0, 1]], [3.1]),
            ([[0, 0, 1]], [np.inf]),
        ):
            with self.assertRaises(ValueError):
                check_values(p, y)

    def test_export_keeps_unrounded_reference(self):
        row = {
            "source_file": "附件3_01.pkl",
            "row_index": 0,
            "predicted_polarity": "Positive",
            "predicted_intensity": 0.123456789,
            "p_negative": 0.1,
            "p_neutral": 0.2,
            "p_positive": 0.7,
        }
        with tempfile.TemporaryDirectory() as d:
            export_rows(d, [row])
            self.assertIn(
                "0.123457",
                (Path(d) / "附件3_情感预测结果.csv").read_text(encoding="utf-8-sig"),
            )
            self.assertIn(
                "0.123456789",
                (Path(d) / "predictions_unrounded.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(row["predicted_intensity"], 0.123456789)

    def test_manifest_detects_extra_and_modified(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "a.json", {"a": 1})
            write_json(
                root / "package_manifest.json",
                {"files": {"a.json": file_hash(root / "a.json")}},
            )
            check_manifest(root)
            write_json(root / "a.json", {"a": 2})
            with self.assertRaises(ValueError):
                check_manifest(root)
            write_json(root / "a.json", {"a": 1})
            write_json(root / "extra.json", {})
            with self.assertRaises(ValueError):
                check_manifest(root)

    def test_zip_traversal_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as d:
            archive, target = Path(d) / "bad.zip", Path(d) / "unpack"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.txt", "no")
            with self.assertRaises(ValueError):
                extract_checked(archive, target)
            self.assertFalse(target.exists())

    def test_safe_zip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root, target, archive = (
                Path(d) / "source",
                Path(d) / "unpack",
                Path(d) / "ok.zip",
            )
            write_json(root / "a.json", {"a": 1})
            write_json(
                root / "package_manifest.json",
                {"files": {"a.json": file_hash(root / "a.json")}},
            )
            with zipfile.ZipFile(archive, "w") as z:
                for path in root.iterdir():
                    z.write(path, path.name)
            extract_checked(archive, target)
            with self.assertRaises(ValueError):
                extract_checked(archive, target)


if __name__ == "__main__":
    unittest.main()
