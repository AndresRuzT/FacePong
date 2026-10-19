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
        expected_spread = 0.15 / (3.0 ** 0.6)
        self.assertAlmostEqual(tracker._min_y_bound, 0.50 - expected_spread, delta=0.01)
        self.assertAlmostEqual(tracker._max_y_bound, 0.50 + expected_spread, delta=0.01)

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


    def test_cascade_fallback_when_facemesh_disabled(self):
        tracker = FaceMeshTracker(self.config)
        # Verify cascade loaded
        self.assertIsNotNone(tracker._cascade)
        self.assertFalse(tracker._cascade.empty())

        # Simulate mediapipe unavailable
        tracker._face_mesh = None
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detected, raw_y, smoothed_y, hud_preview = tracker.process_frame(frame)
        self.assertFalse(detected)
        self.assertEqual(hud_preview.shape, (self.config.preview_height, self.config.preview_width, 3))


    def test_device_profile_cycling_and_adaptation(self):
        from facepong.config import DeviceProfile
        tracker = FaceMeshTracker(self.config)
        self.assertEqual(tracker.device_profile, DeviceProfile.AUTO)

        # Profile cycling
        p1 = tracker.cycle_device_profile()
        self.assertEqual(p1, DeviceProfile.TV)
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.060)

        p2 = tracker.cycle_device_profile()
        self.assertEqual(p2, DeviceProfile.LAPTOP)
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.105)

        p3 = tracker.cycle_device_profile()
        self.assertEqual(p3, DeviceProfile.TABLET)
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.145)

        p4 = tracker.cycle_device_profile()
        self.assertEqual(p4, DeviceProfile.AUTO)

    def test_auto_profile_scales_with_screen_and_distance(self):
        from facepong.config import DeviceProfile
        tracker = FaceMeshTracker(self.config)
        tracker.device_profile = DeviceProfile.AUTO

        # Large screen (TV 1080p) should have tighter base spread than standard laptop
        tracker.set_screen_dimensions(1920, 1080)
        spread_tv = tracker.get_effective_range_spread()
        tracker.set_screen_dimensions(1280, 720)
        spread_laptop = tracker.get_effective_range_spread()
        self.assertLess(spread_tv, spread_laptop)

        # Distance normalization: face far away (couch, 0.12 fraction) should have tighter spread
        spread_far = tracker.get_effective_range_spread(face_height_fraction=0.12)
        spread_near = tracker.get_effective_range_spread(face_height_fraction=0.35)
        self.assertLess(spread_far, spread_near)

    def test_dual_player_calibration(self):
        tracker = FaceMeshTracker(self.config)
        tracker.calibrate_baseline(neutral_y=0.40, range_spread=0.15, player_idx=1)
        tracker.calibrate_baseline(neutral_y=0.60, range_spread=0.20, player_idx=2)

        self.assertAlmostEqual(tracker._min_y_bound, 0.25, delta=0.01)
        self.assertAlmostEqual(tracker._max_y_bound, 0.55, delta=0.01)
        self.assertAlmostEqual(tracker._p2_min_y_bound, 0.40, delta=0.01)
        self.assertAlmostEqual(tracker._p2_max_y_bound, 0.80, delta=0.01)

    def test_process_black_frame_extra_metadata(self):
        tracker = FaceMeshTracker(self.config)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detected, raw_y, smoothed_y, hud = tracker.process_frame(frame)
        self.assertFalse(detected)
        self.assertEqual(tracker.face_count, 0)
        self.assertFalse(tracker.p1_detected)
        self.assertFalse(tracker.p2_detected)

    def test_vision_pipeline_pvp_state(self):
        from facepong.vision.tracker import VisionPipeline
        pipeline = VisionPipeline(self.config)
        state = pipeline.get_state()
        self.assertEqual(state.face_count, 0)
        self.assertFalse(state.is_pvp)
        self.assertAlmostEqual(state.p1_smoothed_y, 0.5)
        self.assertAlmostEqual(state.p2_smoothed_y, 0.5)


if __name__ == "__main__":
    unittest.main()
