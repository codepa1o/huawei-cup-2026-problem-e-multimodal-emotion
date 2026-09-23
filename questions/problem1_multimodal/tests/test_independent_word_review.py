"""独立标注工具测试；所有端点都是合成测试数据，不是实验结果。"""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import independent_word_review as review


class IndependentReviewTests(unittest.TestCase):
    def setUp(self):
        self.ref = [{"sample_id": "synthetic", "word_index": "0", "text": "word",
                     "duration_s": "2.0", "status": "pending", "human_start_s": "",
                     "human_end_s": "", "reviewer_code": "", "notes": "",
                     "auto_start_s": .2, "auto_end_s": .8, "alignment_status": "word_aligned"}]
        self.rows = [{k: v for k, v in self.ref[0].items() if k in review.FIELDS}]

    def marked(self):
        return [dict(self.rows[0], status="matched", human_start_s="0.1", human_end_s="0.9", reviewer_code="A")]

    def test_blank_is_pending(self):
        self.assertEqual(review.validate_rows(self.rows, self.ref)[("synthetic", "0")]["status"], "pending")

    def test_missing_and_duplicate_rejected(self):
        for rows in ([], self.rows * 2):
            with self.assertRaises(ValueError):
                review.validate_rows(rows, self.ref)

    def test_changed_word_rejected(self):
        with self.assertRaises(ValueError):
            review.validate_rows([dict(self.rows[0], text="another")], self.ref)

    def test_bad_boundaries_rejected(self):
        for start, end in (("nan", "1"), ("0", "inf"), ("-1", "1"), ("1", "1"), ("1", "3"), ("", "1")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                review.validate_rows([dict(self.marked()[0], human_start_s=start, human_end_s=end)], self.ref)

    def test_pending_cannot_contain_times(self):
        with self.assertRaises(ValueError):
            review.validate_rows([dict(self.rows[0], human_start_s="0")], self.ref)

    def test_exclusions_need_reason(self):
        with self.assertRaises(ValueError):
            review.validate_rows([dict(self.rows[0], status="text_mismatch", reviewer_code="A")], self.ref)

    def test_empty_metrics_are_null(self):
        self.assertIsNone(review.describe([]))

    def test_known_metrics(self):
        result = review.describe([.1, -.1, .2, -.2])
        self.assertAlmostEqual(result["mae_s"], .15)
        self.assertEqual(result["within_s"]["0.1"], .5)
        self.assertEqual(result["signed_bias_s"], 0)

    def test_selection_deterministic(self):
        items = [{"sample_id": f"{level}-{i}", "alignment_level": level, "video_duration_s": i}
                 for level in ("word", "partial_word", "utterance") for i in range(5)]
        self.assertEqual(review.select_samples(items), review.select_samples(items[::-1]))
        self.assertEqual(len(review.select_samples(items)), 9)

    def fixture(self, root, marked=False, fallback=False):
        project, bundle = root / "project", root / "bundle"
        (project / "outputs/stage0").mkdir(parents=True)
        review.write_csv(project / "outputs/stage0/manifest.csv", [], ["sample_id"])
        (bundle / "sealed_reference").mkdir(parents=True)
        ref = copy.deepcopy(self.ref)
        if fallback:
            ref[0].update(auto_start_s=None, auto_end_s=None, alignment_status="utterance_fallback")
        review.write_json(bundle / "sealed_reference/automatic.json", ref)
        review.write_json(bundle / "protocol.json", {
            "sources": {}, "manifest_sha256": review.digest(project / "outputs/stage0/manifest.csv"),
            "reference_sha256": review.digest(bundle / "sealed_reference/automatic.json")})
        for name, person in (("reviewer_A.csv", "A"), ("reviewer_B.csv", "B"), ("adjudicated.csv", "C")):
            rows = [dict(self.marked()[0], reviewer_code=person, notes="synthetic fixture")] if marked else self.rows
            review.write_csv(bundle / name, rows, review.FIELDS)
        return project, bundle

    def test_blank_evaluation_no_false_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, bundle = self.fixture(root)
            report = review.evaluate(project, bundle, root / "result")
            self.assertEqual(report["status"], "pending_human_review")
            self.assertIsNone(report["automatic_minus_adjudicated"])
            self.assertIsNone(report["scientific_accuracy_pass"])
            with self.assertRaises(FileExistsError):
                review.evaluate(project, bundle, root / "result")

    def test_human_consensus_known_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, bundle = self.fixture(root, marked=True)
            report = review.evaluate(project, bundle, root / "result")
            self.assertEqual(report["comparable_words"], 1)
            self.assertAlmostEqual(report["automatic_minus_adjudicated"]["mae_s"], .1)

    def test_fallback_not_word_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, bundle = self.fixture(root, marked=True, fallback=True)
            report = review.evaluate(project, bundle, root / "result")
            self.assertEqual(report["comparable_words"], 0)
            self.assertEqual(report["human_matched_but_no_automatic_word_boundary"], 1)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, bundle = self.fixture(root)
            with (project / "outputs/stage0/manifest.csv").open("a") as stream:
                stream.write("modified")
            with self.assertRaises(ValueError):
                review.evaluate(project, bundle, root / "result")

    def test_prepare_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                review.prepare(Path(temp), Path(temp))

    def test_same_person_and_premature_adjudication_rejected(self):
        """只在临时测试目录修改合成标注；不接触真实233词表。"""
        for problem in ("same_person", "pending_B", "missing_note"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                project, bundle = self.fixture(root, marked=True)
                name = "adjudicated.csv" if problem == "missing_note" else "reviewer_B.csv"
                rows = review.read_csv(bundle / name)
                if problem == "same_person":
                    rows[0]["reviewer_code"] = "A"
                elif problem == "pending_B":
                    rows = self.rows
                else:
                    rows[0]["notes"] = ""
                import csv
                with (bundle / name).open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=review.FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    review.evaluate(project, bundle, root / "result")


if __name__ == "__main__":
    unittest.main()
