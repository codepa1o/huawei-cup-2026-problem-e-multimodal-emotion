"""Small deterministic checks for the stage-2 feature definitions."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_features import AUDIO_DIM, audio_features, selected_video_frames


class FeatureContractTests(unittest.TestCase):
    def test_audio_sine_has_finite_features_and_expected_pitch(self):
        rate = 16000
        pcm = (0.4 * 32767 * np.sin(2 * np.pi * 200 * np.arange(rate) / rate)).astype(np.int16)
        frames = [{"sample_start": start} for start in range(0, rate - 400 + 1, 160)]
        values, voiced = audio_features(pcm, frames, rate)
        self.assertEqual(values.shape, (len(frames), AUDIO_DIM))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreater(voiced.mean(), 0.9)
        self.assertAlmostEqual(float(np.median(values[voiced, -2]) * 400), 200, delta=20)

    def test_visual_sampling_keeps_source_frame_indices(self):
        frames = [{"index": index, "start_s": index / 25} for index in range(25)]
        selected = selected_video_frames({"video_frames": frames, "video_duration_s": 1.0})
        self.assertEqual([frame["index"] for frame in selected], [0, 5, 10, 15, 20])


if __name__ == "__main__":
    unittest.main()
