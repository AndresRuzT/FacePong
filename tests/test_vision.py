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
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.115)

        p2 = tracker.cycle_device_profile()
        self.assertEqual(p2, DeviceProfile.LAPTOP)
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.125)

        p3 = tracker.cycle_device_profile()
        self.assertEqual(p3, DeviceProfile.TABLET)
        self.assertAlmostEqual(tracker.get_effective_range_spread(), 0.135)

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

    def test_multi_face_detection_sets_pvp(self):
        tracker = FaceMeshTracker(self.config)
        # Mock _face_mesh or _yunet returning 2 faces
        sample_faces = [
            {"raw_y": 0.45, "center_x": 80.0, "fx": 60.0, "fy": 80.0, "fw": 40.0, "fh": 50.0, "nose_x": 80.0, "nose_y": 105.0},
            {"raw_y": 0.55, "center_x": 240.0, "fx": 220.0, "fy": 85.0, "fw": 40.0, "fh": 50.0, "nose_x": 240.0, "nose_y": 110.0},
        ]
        tracker._yunet_detect_multi = lambda frame, w, h: sample_faces
        tracker._face_mesh = None  # Force YuNet path
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        det, ry, sy, hud = tracker.process_frame(frame)

        self.assertTrue(det)
        self.assertEqual(tracker.face_count, 2)
        self.assertTrue(tracker.is_pvp)
        self.assertTrue(tracker.p1_detected)
        self.assertTrue(tracker.p2_detected)
        self.assertAlmostEqual(tracker.p1_raw_y, 0.45)
        self.assertAlmostEqual(tracker.p2_raw_y, 0.55)

    def test_pvp_spatial_isolation_prevents_face_swapping_on_drop(self):
        tracker = FaceMeshTracker(self.config)
        tracker.is_pvp = True

        # Frame 1: Both P1 and P2 detected
        faces_both = [
            {"raw_y": 0.30, "center_x": 80.0, "fx": 60.0, "fy": 60.0, "fw": 40.0, "fh": 50.0},
            {"raw_y": 0.60, "center_x": 240.0, "fx": 220.0, "fy": 120.0, "fw": 40.0, "fh": 50.0},
        ]
        tracker._yunet_detect_multi = lambda f, w, h: faces_both
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker.process_frame(frame)
        self.assertAlmostEqual(tracker.p1_raw_y, 0.30)
        self.assertAlmostEqual(tracker.p2_raw_y, 0.60)

        # Frame 2: Player 1 raises head and drops out; only Player 2 remains on right
        faces_only_p2 = [
            {"raw_y": 0.60, "center_x": 240.0, "fx": 220.0, "fy": 120.0, "fw": 40.0, "fh": 50.0},
        ]
        tracker._yunet_detect_multi = lambda f, w, h: faces_only_p2
        tracker.process_frame(frame)

        # Crucial: P1 raw_y must NOT jump to P2's 0.60 position!
        self.assertAlmostEqual(tracker.p1_raw_y, 0.30)
        self.assertAlmostEqual(tracker.p2_raw_y, 0.60)

    def test_pvp_nearest_neighbor_prevents_swap_when_p2_is_near_center(self):
        tracker = FaceMeshTracker(self.config)
        tracker.is_pvp = True

        # Frame 1: Both P1 (at 60px) and P2 (at 155px in 320px width -> slightly left of mid_x=160px)
        faces_both = [
            {"raw_y": 0.35, "center_x": 60.0, "fx": 40.0, "fy": 60.0, "fw": 40.0, "fh": 50.0},
            {"raw_y": 0.75, "center_x": 155.0, "fx": 135.0, "fy": 70.0, "fw": 40.0, "fh": 50.0},
        ]
        tracker._yunet_detect_multi = lambda f, w, h: faces_both
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker.process_frame(frame)
        self.assertAlmostEqual(tracker.p1_raw_y, 0.35)
        self.assertAlmostEqual(tracker.p2_raw_y, 0.75)

        # Frame 2: P1 temporarily drops out. Only P2 at 155px remains.
        faces_only_p2 = [
            {"raw_y": 0.75, "center_x": 155.0, "fx": 135.0, "fy": 70.0, "fw": 40.0, "fh": 50.0},
        ]
        tracker._yunet_detect_multi = lambda f, w, h: faces_only_p2
        tracker.process_frame(frame)

        # Even though 155 < 160 (mid_x), nearest-neighbor correctly identifies this as P2!
        self.assertTrue(tracker.p2_detected)
        self.assertAlmostEqual(tracker.p2_raw_y, 0.75)
        self.assertAlmostEqual(tracker.p1_raw_y, 0.35)

    def test_slew_rate_limiter_prevents_downward_spike_when_pitching_up(self):
        tracker = FaceMeshTracker(self.config)
        now = 100.0
        # Initialize at raised head position (raw_y = 0.25)
        tracker._update_player_state(0.25, 60.0, 240, now, player_idx=1)
        self.assertAlmostEqual(tracker._p1_last_valid_raw_y, 0.25)

        # Sudden anomalous downward spike to 0.70 after 30ms (simulating chest artifact)
        dt = 0.033
        smoothed = tracker._update_player_state(0.70, 60.0, 240, now + dt, player_idx=1)
        # Slew-rate limiter should prevent raw_y from jumping straight to 0.70
        self.assertLess(tracker._p1_last_valid_raw_y, 0.60)

    def test_stationary_noise_gate_eliminates_micro_jitter(self):
        tracker = FaceMeshTracker(self.config)
        now = 100.0
        # Initialize at center
        s0 = tracker._update_player_state(0.50, 60.0, 240, now, player_idx=1)
        self.assertAlmostEqual(s0, 0.50, delta=0.01)

        # Micro-noise fluctuation (+0.003, typical camera sensor noise)
        s1 = tracker._update_player_state(0.503, 60.0, 240, now + 0.033, player_idx=1)
        # Position should remain smoothly stabilized near 0.50 without discontinuous steps
        self.assertAlmostEqual(s1, 0.50, delta=0.015)

    def test_pitch_amplification_responds_to_head_nods(self):
        tracker = FaceMeshTracker(self.config)
        # Face dimensions in detect space (192x144)
        # Neutral face: eyes at y=55, nose at y=63, mouth at y=72
        face_neutral = np.array([
            [60, 45, 30, 35, 68, 55, 82, 55, 75, 63, 70, 72, 80, 72, 0.85]
        ], dtype=np.float32)
        # Nod UP: nose moves closer to eyes (y=59)
        face_nod_up = np.array([
            [60, 45, 30, 35, 68, 55, 82, 55, 75, 59, 70, 72, 80, 72, 0.85]
        ], dtype=np.float32)
        # Nod DOWN: nose moves closer to mouth (y=67)
        face_nod_down = np.array([
            [60, 45, 30, 35, 68, 55, 82, 55, 75, 67, 70, 72, 80, 72, 0.85]
        ], dtype=np.float32)

        class MockYuNet:
            def __init__(self, faces): self._faces = faces
            def setInputSize(self, size): pass
            def detect(self, img): return True, self._faces

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker._yunet = MockYuNet(face_neutral)
        r_neutral = tracker._yunet_detect_multi(frame, 320, 240)[0]["raw_y"]

        tracker._yunet = MockYuNet(face_nod_up)
        r_up = tracker._yunet_detect_multi(frame, 320, 240)[0]["raw_y"]

        tracker._yunet = MockYuNet(face_nod_down)
        r_down = tracker._yunet_detect_multi(frame, 320, 240)[0]["raw_y"]

        # Nod UP moves coordinate up (smaller raw_y), nod DOWN moves coordinate down (larger raw_y)
        self.assertLess(r_up, r_neutral)
        self.assertGreater(r_down, r_neutral)

    def test_no_baseline_drift_when_holding_clamped_angle(self):
        tracker = FaceMeshTracker(self.config)
        tracker.calibrate_baseline(neutral_y=0.58, range_spread=0.10)
        initial_min = tracker._min_y_bound
        initial_max = tracker._max_y_bound

        # Hold head at extreme low angle (raw_y = 0.80 > max_b = 0.68) for 30 consecutive frames
        t = 100.0
        for _ in range(30):
            t += 0.033
            tracker._update_player_state(0.80, 60.0, 240, t, player_idx=1)

        # Baseline bounds must remain solidly anchored without drifting
        self.assertAlmostEqual(tracker._min_y_bound, initial_min, delta=0.001)
        self.assertAlmostEqual(tracker._max_y_bound, initial_max, delta=0.001)

    def test_set_pvp_mode_clears_dual_player_state(self):
        tracker = FaceMeshTracker(self.config)
        tracker.is_pvp = True
        tracker.p2_detected = True
        tracker._p2_tracked_x = 240.0

        tracker.set_pvp_mode(False)
        self.assertFalse(tracker.is_pvp)
        self.assertFalse(tracker.p2_detected)
        self.assertIsNone(tracker._p2_tracked_x)

    def test_fluid_upward_nod_tracking_is_snappy_and_continuous(self):
        tracker = FaceMeshTracker(self.config)
        tracker.calibrate_baseline(neutral_y=0.50, range_spread=0.10)
        t = 100.0
        # Initialize at neutral center
        tracker._update_player_state(0.50, 60.0, 240, t, player_idx=1)
        self.assertAlmostEqual(tracker.p1_smoothed_y, 0.50, delta=0.02)

        # Move head upward to raw_y = 0.44 (a smooth upward nod)
        t += 0.033
        s1 = tracker._update_player_state(0.44, 60.0, 240, t, player_idx=1)
        # Should be moving upward immediately without lag or deadband pause
        self.assertLess(s1, 0.50)

        t += 0.033
        s2 = tracker._update_player_state(0.44, 60.0, 240, t, player_idx=1)
        # By frame 2, smoothed_y should have smoothly and closely tracked the target
        self.assertLess(s2, s1)
        self.assertLess(s2, 0.35)

    def test_one_euro_filter_stationary_noise_reduction(self):
        from facepong.vision.tracker import OneEuroFilter
        filt = OneEuroFilter(min_cutoff=1.0, beta=18.0)
        t = 10.0
        # Initialize
        filt.filter(0.50, t)

        # Feed sensor jitter around 0.50
        samples = [0.502, 0.498, 0.503, 0.497, 0.501, 0.499]
        filtered = []
        for s in samples:
            t += 0.033
            val, _ = filt.filter(s, t)
            filtered.append(val)

        # Output fluctuations must be significantly smaller than input noise
        inp_variance = max(samples) - min(samples)
        out_variance = max(filtered) - min(filtered)
        self.assertLess(out_variance, inp_variance * 0.45)

    def test_one_euro_filter_high_speed_zero_lag(self):
        from facepong.vision.tracker import OneEuroFilter
        filt = OneEuroFilter(min_cutoff=1.0, beta=18.0)
        t = 10.0
        filt.filter(0.50, t)

        # Fast head flick: jumps from 0.50 to 0.85 in 66ms (2 frames)
        t += 0.033
        v1, vel1 = filt.filter(0.85, t)
        # Filter cutoff should ramp up immediately and track closely without lag
        self.assertGreater(v1, 0.70)
        self.assertGreater(vel1, 1.5)


    def test_auto_centering_initializes_at_natural_sitting_position(self):
        """Verifies that an uncalibrated player sitting at raw_y=0.35 immediately centers at paddle=0.50."""
        tracker = FaceMeshTracker(self.config)
        t = 100.0
        # Player sits down with head in upper half of camera image (raw_y = 0.35)
        paddle_y = tracker._update_player_state(0.35, 60.0, 240, t, player_idx=1)
        # Paddle must immediately center at 0.50 (±0.02) without requiring manual calibration
        self.assertAlmostEqual(paddle_y, 0.50, delta=0.03)

    def test_anti_stuck_boundary_relaxation_unclumps_paddle(self):
        """Verifies that holding posture outside current bounds relaxes bounds and unclumps paddle."""
        tracker = FaceMeshTracker(self.config)
        t = 100.0
        # Initialize at 0.40
        tracker._update_player_state(0.40, 60.0, 240, t, player_idx=1)
        # User shifts posture significantly to 0.20 (above min_b) for 15 frames
        for _ in range(15):
            t += 0.033
            p_y = tracker._update_player_state(0.20, 60.0, 240, t, player_idx=1)
        # Bounds should have relaxed towards 0.20 instead of leaving user stuck forever
        self.assertLess(tracker._min_y_bound, 0.30)

    def test_rfb_320_detector_initialization_and_priors(self):
        """Verifies RFB-320 ONNX network and prior generation."""
        tracker = FaceMeshTracker(self.config)
        self.assertIsNotNone(tracker._rfb_net)
        self.assertIsNotNone(tracker._rfb_priors)
        self.assertEqual(tracker._rfb_priors.shape, (4420, 4))

    def test_velocity_derivative_direction_matches_paddle_motion(self):
        """Verifies that moving up produces negative velocity (towards 0) and moving down produces positive velocity."""
        tracker = FaceMeshTracker(self.config)
        t = 100.0
        # Initialize at baseline 0.50
        tracker._update_player_state(0.50, 60.0, 240, t, player_idx=1)

        # Move head UP (raw_y drops from 0.50 to 0.45)
        t += 0.033
        p_up = tracker._update_player_state(0.45, 60.0, 240, t, player_idx=1)
        self.assertLess(p_up, 0.50)
        self.assertLess(tracker._p1_velocity_y, 0.0)  # Velocity points UP (towards 0)

        # Move head DOWN (raw_y rises from 0.45 to 0.55)
        t += 0.033
        p_down = tracker._update_player_state(0.55, 60.0, 240, t, player_idx=1)
        self.assertGreater(p_down, p_up)
        self.assertGreater(tracker._p1_velocity_y, 0.0)  # Velocity points DOWN (towards 1)

    def test_yunet_roi_fast_path_detection(self):
        """Verifies fast 160x160 ROI detection crop and coordinate remapping."""
        tracker = FaceMeshTracker(self.config)

        class MockRoiYuNet:
            def setInputSize(self, size): pass
            def detect(self, img):
                # Returns face inside the 160x160 crop at center
                # [x, y, w, h, rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy, score]
                return True, np.array([[40, 40, 80, 80, 60, 65, 100, 65, 80, 85, 70, 105, 90, 105, 0.95]], dtype=np.float32)

        tracker._yunet = MockRoiYuNet()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        last_box = (100.0, 80.0, 80.0, 80.0)
        roi_res = tracker._detect_yunet_roi(frame, 320, 240, last_box)

        self.assertEqual(len(roi_res), 1)
        face = roi_res[0]
        self.assertIn("raw_y", face)
        self.assertGreater(face["score"], 0.8)
        self.assertGreater(face["fx"], 0)
        self.assertGreater(face["fy"], 0)


if __name__ == "__main__":
    unittest.main()
