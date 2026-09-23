"""阶段3的确定性单测：不用依赖真实100条视频也能检查关键边界。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import align_50


class AlignmentTests(unittest.TestCase):
    def test_overlap_pool_and_union_coverage(self):
        """重叠帧应按秒加权，而并集覆盖不能把重叠时长重复累加。"""
        windows = np.array([[0.0, 1.0], [1.0, 2.0]])
        intervals = np.array([[0.0, 0.8], [0.5, 1.5]])
        weights = align_50.overlap_weights(windows, intervals, np.array([True, True]))
        values, mask = align_50.pool(np.array([[2.0], [6.0]]), weights)
        self.assertTrue(mask.all())
        self.assertAlmostEqual(float(weights[0, 0]), 0.8)
        self.assertAlmostEqual(float(weights[0, 1]), 0.5)
        self.assertAlmostEqual(float(values[0, 0]), (0.8 * 2 + 0.5 * 6) / 1.3, delta=1e-6)
        np.testing.assert_allclose(
            align_50.union_coverage(windows, intervals, weights), [1.0, 0.5]
        )

    def test_visual_midpoints_have_no_stale_tail(self):
        """抽帧代表区到期后，不能把旧画面错误延伸到片尾。"""
        intervals = align_50.visual_intervals(
            np.array([[0.0, 0.03], [0.2, 0.23], [0.4, 0.43]]), decoded_end=0.55
        )
        np.testing.assert_allclose(intervals, [[0.0, 0.1], [0.1, 0.3], [0.3, 0.5]])
        weights = align_50.overlap_weights(
            np.array([[0.5, 0.55]]), intervals, np.ones(3, dtype=bool)
        )
        self.assertEqual(float(weights.sum()), 0.0)

    def test_utterance_fallback_and_scene_without_face(self):
        """整段文本回退需留标记；无人脸不影响全画面有效性。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timeline_dir = root / "stage1" / "timelines" / "v"
            feature_dir = root / "stage2" / "features_raw" / "v"
            timeline_dir.mkdir(parents=True)
            feature_dir.mkdir(parents=True)
            (timeline_dir / "0.json").write_text(json.dumps({
                "sample_id": "v$_$0", "status": "ok", "video_duration_s": 2.0,
                "alignment_level": "utterance",
                "video_frames": [{"index": 0, "start_s": 0.5, "end_s": 0.52}],
                "audio_frames": [],
                "words": [{"index": 0}],
            }), encoding="utf-8")
            np.savez_compressed(
                feature_dir / "0.npz", sample_id=np.array("v$_$0"),
                clip_text_time=np.array([0.0, 2.0]),
                clip_text=np.ones(768, dtype=np.float32),
                clip_text_observed=np.array(True),
                text=np.ones((1, 768), dtype=np.float32),
                text_time=np.array([[-1.0, -1.0]]),
                text_time_known_mask=np.array([False]),
                text_observed_mask=np.array([True]),
                audio=np.zeros((0, 18), dtype=np.float32),
                audio_time=np.zeros((0, 2), dtype=np.float32),
                audio_observed_mask=np.zeros(0, dtype=bool),
                audio_voiced_mask=np.zeros(0, dtype=bool),
                vision=np.concatenate((np.zeros((1, 52)), np.ones((1, 576))), axis=1),
                vision_time=np.array([[0.5, 0.52]]),
                vision_observed_mask=np.array([True]),
                vision_face_mask=np.array([False]),
                vision_source_frame_index=np.array([0]),
            )
            row = {"sample_id": "v$_$0", "video_id": "v", "clip_id": "0", "issue_codes": ""}
            feature_row = {
                "relative_feature_path": "features_raw/v/0.npz", "status": "needs_review",
                "issues": "vision:no_face_detected",
            }
            with patch.object(align_50, "OUTPUT", root):
                arrays, mapping, summary = align_50.align_sample(row, feature_row)
            self.assertTrue(arrays["text_clip_fallback_mask"].all())
            self.assertFalse(arrays["text_word_observed_mask"].any())
            self.assertTrue(np.all(arrays["text"] == 1))
            self.assertTrue(arrays["vision_scene_observed_mask"].any())
            self.assertFalse(arrays["vision_face_observed_mask"].any())
            self.assertFalse(arrays["audio_observed_mask"].any())
            self.assertEqual(mapping[0]["text_time_precision"], "clip")
            self.assertEqual(summary["text_clip_fallback_bins"], 50)


if __name__ == "__main__":
    unittest.main()
