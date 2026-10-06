"""Unit tests for FaceMeshTracker math, EMA smoothing, and calibration."""

import unittest
import numpy as np
from facepong.config import CameraConfig
from facepong.vision.tracker import FaceMeshTracker


class TestVisionTracker(unittest.TestCase):
    def setUp(self):
        self.config = CameraConfig(ema_alpha=0.25)
        self.tracker = FaceMeshTracker(self.config)

    def test_calibration_bounds(self):
        self.tracker.calibrate_baseline(neutral_y=0.50, range_spread=0.20)
        self.assertAlmostEqual(self.tracker._min_y_bound, 0.30, delta=0.01)
        self.assertAlmostEqual(self.tracker._max_y_bound, 0.70, delta=0.01)

    def test_process_black_frame_no_face(self):
        # Frame with no face
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detected, raw_y, smoothed_y, hud_preview = self.tracker.process_frame(frame)

        self.assertFalse(detected)
        self.assertEqual(hud_preview.shape, (self.config.preview_height, self.config.preview_width, 3))

    def test_camera_manual_selection(self):
        from facepong.vision.camera import find_preferred_camera_device
        idx, desc = find_preferred_camera_device(requested_index=2)
        self.assertEqual(idx, 2)
        self.assertIn("2", desc)

    def test_camera_auto_detection_returns_valid_device(self):
        from facepong.vision.camera import find_preferred_camera_device
        idx, desc = find_preferred_camera_device(requested_index=None)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        self.assertTrue(len(desc) > 0)

    def test_sensitivity_scales_calibration_range(self):
        high_sens_config = CameraConfig(head_sensitivity=3.0, default_range_spread=0.15)
        tracker = FaceMeshTracker(high_sens_config)
        tracker.calibrate_baseline(neutral_y=0.50)
        expected_spread = 0.15 / 3.0  # 0.05
        self.assertAlmostEqual(tracker._min_y_bound, 0.45, delta=0.01)
        self.assertAlmostEqual(tracker._max_y_bound, 0.55, delta=0.01)

    def test_threaded_camera_synthetic_generation(self):
        from facepong.vision.camera import ThreadedCamera
        cam = ThreadedCamera(device_index=999, width=160, height=120)
        synth = cam._generate_synthetic_frame()
        self.assertEqual(synth.shape, (120, 160, 3))

    def test_reset_for_new_camera_flags_recenter(self):
        tracker = FaceMeshTracker(self.config)
        tracker._recenter_needed = False
        tracker.reset_for_new_camera()
        self.assertTrue(tracker._recenter_needed)


if __name__ == "__main__":
    unittest.main()
