"""MediaPipe Face Mesh tracker with exponential moving average filtering and adaptive baseline."""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from facepong.config import CameraConfig, DeviceProfile
from facepong.vision.camera import ThreadedCamera

logger = logging.getLogger(__name__)

# Key landmark indices in MediaPipe Face Mesh (468 points total)
NOSE_TIP_IDX = 1
GLABELLA_IDX = 168  # Mid-point between the eyes / root of nose for stable head tracking
FOREHEAD_IDX = 10
CHIN_IDX = 152
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291

# Subset of facial contour points for lightweight futuristic HUD wireframe
CONTOUR_KEYPOINTS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter (Casiez, Roussel, Vogel - CHI 2012).
    Eliminates camera sensor micro-jitter when stationary (low cutoff frequency)
    while providing instantaneous, 1:1 zero-lag tracking on active movement (high cutoff frequency).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 18.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: float = 0.0

    def reset(self, x0: Optional[float] = None, t0: float = 0.0) -> None:
        """Resets filter memory to baseline position."""
        self.x_prev = float(x0) if x0 is not None else None
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    @staticmethod
    def _alpha(dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(0.01, cutoff))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> Tuple[float, float]:
        """
        Filters raw value x at timestamp t.
        Returns tuple of (filtered_position, filtered_velocity).
        """
        if self.x_prev is None or self.t_prev <= 0.0:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = float(t)
            return float(x), 0.0

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = float(t)

        # 1. Filter velocity derivative to eliminate high-frequency pixel quantization noise
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # 2. Dynamic cutoff frequency scales proportionally with instantaneous velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return float(x_hat), float(dx_hat)


@dataclass
class TrackingState:
    """Thread-safe snapshot of face tracking data."""
    face_detected: bool = False
    raw_y: float = 0.5
    normalized_y: float = 0.5
    smoothed_y: float = 0.5
    velocity_y: float = 0.0
    sample_time: float = 0.0
    last_detected_time: float = 0.0
    preview_surface_buffer: Optional[np.ndarray] = None
    calibration_progress: float = 0.0
    is_calibrated: bool = False
    device_profile_label: str = "AUTO"
    # Multi-player PvP fields
    face_count: int = 0
    is_pvp: bool = False
    p1_detected: bool = False
    p1_raw_y: float = 0.5
    p1_smoothed_y: float = 0.5
    p1_velocity_y: float = 0.0
    p2_detected: bool = False
    p2_raw_y: float = 0.5
    p2_smoothed_y: float = 0.5
    p2_velocity_y: float = 0.0


class FaceMeshTracker:
    """Processes camera frames to track facial position, produce HUD previews, and auto-anchor baseline."""

    def __init__(self, config: CameraConfig):
        self.config = config

        # Multi-player PvP state
        self.face_count: int = 0
        self.is_pvp: bool = False

        # Player 1 (Left paddle / Solo) state
        self.p1_detected: bool = False
        self.p1_raw_y: float = 0.5
        self.p1_smoothed_y: float = 0.5
        self._p1_last_valid_raw_y: float = 0.5
        self._p1_last_valid_detection_time: float = 0.0
        self._p1_is_first_sample: bool = True
        self._p1_clamped_at_min_count: int = 0
        self._p1_clamped_at_max_count: int = 0
        self._p1_velocity_y: float = 0.0
        self._p1_last_raw_y_sample_time: float = 0.0

        # Player 2 (Right paddle in PvP) state
        self.p2_detected: bool = False
        self.p2_raw_y: float = 0.5
        self.p2_smoothed_y: float = 0.5
        self._p2_last_valid_raw_y: float = 0.5
        self._p2_last_valid_detection_time: float = 0.0
        self._p2_is_first_sample: bool = True
        self._p2_clamped_at_min_count: int = 0
        self._p2_clamped_at_max_count: int = 0
        self._p2_velocity_y: float = 0.0
        self._p2_last_raw_y_sample_time: float = 0.0

        # Player horizontal reference tracking
        self._p1_tracked_x: Optional[float] = None
        self._p2_tracked_x: Optional[float] = None

        # Backwards compatible alias for single player
        self._smoothed_y: float = 0.5
        self._is_first_sample: bool = True

        # Sensitivity-aware calibration bounds
        sensitivity = max(0.5, getattr(config, "head_sensitivity", 2.8))
        base_spread = getattr(config, "default_range_spread", 0.14)
        initial_spread = base_spread / (sensitivity ** 0.6)
        self._min_y_bound: float = max(0.02, 0.5 - initial_spread)
        self._max_y_bound: float = min(0.98, 0.5 + initial_spread)
        self._p2_min_y_bound: float = max(0.02, 0.5 - initial_spread)
        self._p2_max_y_bound: float = min(0.98, 0.5 + initial_spread)
        self._calibration_samples: List[float] = []

        # Grace period & adaptive recentering state
        self._recenter_needed: bool = False
        self._is_calibrated: bool = False
        self._clamped_at_min_count: int = 0
        self._clamped_at_max_count: int = 0
        self._last_valid_raw_y: float = 0.5
        self._last_valid_detection_time: float = 0.0
        self._velocity_y: float = 0.0
        self._last_raw_y_sample_time: float = 0.0

        # Haar Cascade bounding box smoothing and ROI tracking state
        self._cascade_box: Optional[Tuple[float, float, float, float]] = None
        self._cascade_miss_count: int = 0

        # Detectors
        self._face_mesh = None
        self._yunet = None
        self._cascade = None

        # YuNet ROI tracking state (2-phase detection: ROI fast path + full scan fallback)
        self._yunet_last_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in detect-space
        self._yunet_roi_miss: int = 0

        # Device profile & screen adaptive sensitivity
        self.device_profile: DeviceProfile = getattr(config, "device_profile", DeviceProfile.AUTO)
        self._screen_height: int = 720

        # One Euro Adaptive Low-Pass Filters (CHI 2012)
        min_cutoff, beta = self.get_filter_params()
        self._p1_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)
        self._p2_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=1.0)

        self._init_detector()

    def get_filter_params(self) -> Tuple[float, float]:
        """Returns (min_cutoff, beta) tuned for device profile and sensitivity."""
        sens = max(0.5, getattr(self.config, "head_sensitivity", 2.4)) / 2.4
        if self.device_profile == DeviceProfile.TV:
            return 1.1, 24.0 * sens
        elif self.device_profile == DeviceProfile.LAPTOP:
            return 0.95, 18.0 * sens
        elif self.device_profile == DeviceProfile.TABLET:
            return 0.85, 14.0 * sens
        return 0.95, 18.0 * sens

    def set_screen_dimensions(self, width: int, height: int) -> None:
        """Notifies tracker of the active display resolution to adjust sensitivity scaling."""
        self._screen_height = height

    def get_effective_range_spread(self, face_height_fraction: Optional[float] = None) -> float:
        """
        Calculates optimal head motion spread standardized across device classes:
          - TV: player is 1.5-3m away on a large display -> ultra-responsive spread (~0.042)
          - LAPTOP: player is ~50cm away at desk -> balanced effortless spread (~0.055)
          - TABLET: player is ~35cm away, close-up -> gentle spread (~0.075)
          - AUTO: detects screen resolution and scales inversely with physical distance (face size)
        """
        if self.device_profile == DeviceProfile.TV:
            base_spread = 0.042
        elif self.device_profile == DeviceProfile.LAPTOP:
            base_spread = 0.055
        elif self.device_profile == DeviceProfile.TABLET:
            base_spread = 0.075
        else:
            # AUTO profile:
            # TVs / large displays are >= 1000px height
            if self._screen_height >= 1000:
                base_spread = 0.046
            else:
                sensitivity = max(0.5, getattr(self.config, "head_sensitivity", 3.0))
                config_spread = getattr(self.config, "default_range_spread", 0.080)
                base_spread = config_spread / (sensitivity ** 0.5)

        # Distance normalization: face size is inversely proportional to player distance
        if face_height_fraction is not None and face_height_fraction > 0.04:
            ref_scale = 0.20  # reference face height fraction
            dist_multiplier = float(np.clip(face_height_fraction / ref_scale, 0.75, 1.25))
            return float(base_spread * dist_multiplier)

        return float(base_spread)

    def get_effective_ema_alpha(self) -> float:
        """Returns baseline EMA smoothing factor tuned for device ergonomics."""
        if self.device_profile == DeviceProfile.TV:
            return 0.32
        elif self.device_profile == DeviceProfile.LAPTOP:
            return 0.28
        elif self.device_profile == DeviceProfile.TABLET:
            return 0.24
        return getattr(self.config, "ema_alpha", 0.28)

    def cycle_device_profile(self) -> DeviceProfile:
        """Cycles between AUTO -> TV -> LAPTOP -> TABLET -> AUTO."""
        order = [DeviceProfile.AUTO, DeviceProfile.TV, DeviceProfile.LAPTOP, DeviceProfile.TABLET]
        try:
            idx = order.index(self.device_profile)
            next_idx = (idx + 1) % len(order)
        except ValueError:
            next_idx = 0
        self.device_profile = order[next_idx]
        min_c, b_val = self.get_filter_params()
        self._p1_filter.min_cutoff, self._p1_filter.beta = min_c, b_val
        self._p2_filter.min_cutoff, self._p2_filter.beta = min_c, b_val
        self.calibrate_baseline(self._last_valid_raw_y)
        logger.info("Switched device profile to: %s", self.device_profile.value)
        return self.device_profile

    def _init_detector(self) -> None:
        """Initializes OpenCV YuNet DNN as primary detector (edge-optimized for Raspberry Pi 4)."""
        # 1. OpenCV YuNet Deep Neural Network Face & Landmark Detector (Fast, lightweight edge architecture)
        try:
            # Use 2 threads so 2 CPU cores remain completely dedicated to 60 FPS Pygame render loop on Raspberry Pi
            cv2.setNumThreads(2)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
            if os.path.exists(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
                self._yunet = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=(192, 144),
                    score_threshold=0.20,
                    nms_threshold=0.3,
                    top_k=8,
                )
                logger.info("OpenCV YuNet DNN tracker initialized (192x144, 2 threads): %s", yunet_path)
        except Exception as exc:
            logger.warning("OpenCV YuNet initialization failed: %s", exc)

        # 2. MediaPipe Solutions FaceMesh (fallback ONLY if YuNet is unavailable)
        if self._yunet is None:
            try:
                if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                    self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=2,
                        refine_landmarks=False,
                        min_detection_confidence=0.35,
                        min_tracking_confidence=0.35,
                    )
                    logger.info("MediaPipe Face Mesh tracker initialized as fallback (max=2)")
            except Exception as exc:
                logger.debug("MediaPipe Solutions initialization failed (%s).", exc)


        # 3. OpenCV Haar Cascade as emergency fallback
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            bundled_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            system_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")

            for c_path in (bundled_path, system_path):
                if c_path and os.path.exists(c_path):
                    cascade = cv2.CascadeClassifier(c_path)
                    if not cascade.empty():
                        self._cascade = cascade
                        logger.info("OpenCV Haar Cascade initialized from: %s", c_path)
                        break
        except Exception as exc:
            logger.warning("OpenCV Cascade initialization error: %s", exc)

        if self._face_mesh is None and self._yunet is None and self._cascade is None:
            logger.error("No face tracking backend available! Neither YuNet, MediaPipe, nor OpenCV Haar Cascade could be loaded.")

    def set_pvp_mode(self, is_pvp: bool) -> None:
        """Explicitly sets or clears 2-player competitive PvP mode."""
        self.is_pvp = is_pvp
        if not is_pvp:
            self.p2_detected = False
            self._p2_tracked_x = None

    def reset_for_new_camera(self) -> None:
        """Flags that a camera switch occurred and baseline must re-anchor to new camera frame."""
        self._recenter_needed = True
        self._is_first_sample = True
        self._is_calibrated = False
        self._yunet_last_box = None
        self._yunet_roi_miss = 0
        self._clamped_at_min_count = 0
        self._clamped_at_max_count = 0
        self._p1_tracked_x = None
        self._p2_tracked_x = None
        self._p1_filter.reset()
        self._p2_filter.reset()
        logger.info("FaceMeshTracker flagged for baseline re-anchor on new camera feed")

    def calibrate_baseline(self, neutral_y: float, range_spread: Optional[float] = None, player_idx: int = 1) -> None:
        """Sets comfortable dynamic range around player's neutral head position based on sensitivity."""
        if range_spread is None:
            range_spread = self.get_effective_range_spread()
        min_b = max(0.02, neutral_y - range_spread)
        max_b = min(0.98, neutral_y + range_spread)
        self._is_calibrated = True
        if player_idx == 1:
            self._min_y_bound = min_b
            self._max_y_bound = max_b
            self._p1_is_first_sample = False
            self._is_first_sample = False
            self._p1_last_valid_raw_y = neutral_y
            self._last_valid_raw_y = neutral_y
            self._p1_filter.reset(0.5)
            logger.info(
                "Calibrated P1 head range: [%.2f, %.2f] around neutral %.2f (spread: ±%.3f, profile: %s)",
                self._min_y_bound, self._max_y_bound, neutral_y, range_spread, self.device_profile.value,
            )
        else:
            self._p2_min_y_bound = min_b
            self._p2_max_y_bound = max_b
            self._p2_is_first_sample = False
            self._p2_last_valid_raw_y = neutral_y
            self._p2_filter.reset(0.5)
            logger.info(
                "Calibrated P2 head range: [%.2f, %.2f] around neutral %.2f (spread: ±%.3f, profile: %s)",
                self._p2_min_y_bound, self._p2_max_y_bound, neutral_y, range_spread, self.device_profile.value,
            )

    def _update_player_state(
        self, raw_y: float, face_height: float, frame_height: int, now: float, player_idx: int = 1
    ) -> float:
        """Computes velocity, auto-baseline anchoring, distance scaling, non-linear curve, and EMA for P1 or P2."""
        if player_idx == 1:
            last_raw = self._p1_last_valid_raw_y
            last_time = self._p1_last_raw_y_sample_time
            is_first = self._p1_is_first_sample
            recenter = self._recenter_needed
            min_b = self._min_y_bound
            max_b = self._max_y_bound
            smoothed = self.p1_smoothed_y
            c_max = self._p1_clamped_at_max_count
            c_min = self._p1_clamped_at_min_count
            vel = self._p1_velocity_y
        else:
            last_raw = self._p2_last_valid_raw_y
            last_time = self._p2_last_raw_y_sample_time
            is_first = self._p2_is_first_sample
            recenter = self._recenter_needed
            min_b = self._p2_min_y_bound
            max_b = self._p2_max_y_bound
            smoothed = self.p2_smoothed_y
            c_max = self._p2_clamped_at_max_count
            c_min = self._p2_clamped_at_min_count
            vel = self._p2_velocity_y

        if last_time > 0:
            dt_s = max(0.001, now - last_time)

            # Smooth soft deadband: cancels camera sensor micro-quantization when head is stationary (<0.0035)
            # with smooth Hermite transition to eliminate any discrete threshold step
            if not is_first and last_raw > 0:
                diff_raw = abs(raw_y - last_raw)
                if diff_raw < 0.0035:
                    raw_y = last_raw
                elif diff_raw < 0.0085:
                    t_blend = (diff_raw - 0.0035) / 0.0050
                    blend = t_blend * t_blend * (3.0 - 2.0 * t_blend)
                    raw_y = last_raw + (raw_y - last_raw) * blend

            # Slew-rate limiter: prevents anomalous single-frame video glints while allowing instant fast flicks
            if not is_first and last_raw > 0:
                max_step = max(0.25, 9.0 * dt_s)
                if (raw_y - last_raw) > max_step:
                    raw_y = last_raw + max_step
                elif (last_raw - raw_y) > max_step:
                    raw_y = last_raw - max_step

        # Re-anchor baseline if needed
        if recenter or is_first:
            face_frac = (face_height / float(frame_height)) if (face_height > 0 and frame_height > 0) else None
            init_spread = self.get_effective_range_spread(face_frac)
            min_b = max(0.02, raw_y - init_spread)
            max_b = min(0.98, raw_y + init_spread)
            smoothed = 0.5
            is_first = False

        # In AUTO mode, adapt sensitivity based on physical distance with deadband hysteresis
        # only before calibration is fixed, preventing continuous micro-drift when stationary
        if not self._is_calibrated and self.device_profile == DeviceProfile.AUTO and face_height > 0 and frame_height > 0:
            target_spread = self.get_effective_range_spread(face_height / float(frame_height))
            curr_spread = 0.5 * (max_b - min_b)
            if abs(target_spread - curr_spread) > 0.012:
                new_spread = 0.95 * curr_spread + 0.05 * target_spread
                mid_y = 0.5 * (min_b + max_b)
                min_b = max(0.02, mid_y - new_spread)
                max_b = min(0.98, mid_y + new_spread)

        span = max(0.01, max_b - min_b)
        mid_b = 0.5 * (min_b + max_b)

        # Track clamped duration for telemetry only (no dynamic baseline drift during play
        # to guarantee the paddle NEVER drifts or moves on its own while holding an angle)
        if raw_y > max_b:
            c_max += 1
        else:
            c_max = 0

        if raw_y < min_b:
            c_min += 1
        else:
            c_min = 0

        # Linear ergonomic pitch mapping:
        # Looking down is slightly stiffer on the neck than looking up.
        # An asymmetric downward span (0.85x) makes looking down reach the bottom effortlessly.
        if raw_y <= mid_b:
            # Looking UP
            half_span_up = max(0.01, (mid_b - min_b) * 1.05)
            norm = min(1.0, max(0.0, (mid_b - raw_y) / half_span_up))
            linear_y = 0.5 - 0.5 * norm
        else:
            # Looking DOWN: responsive span (0.85x) for easy, subtle downward nods
            half_span_down = max(0.01, (max_b - mid_b) * 0.85)
            norm = min(1.0, max(0.0, (raw_y - mid_b) / half_span_down))
            linear_y = 0.5 + 0.5 * norm

        mapped_y = float(np.clip(linear_y, 0.0, 1.0))

        # One Euro Filter: adaptive low-pass filter providing 1:1 instantaneous response
        # on active head nods while keeping the paddle completely stationary when holding still
        filt = self._p1_filter if player_idx == 1 else self._p2_filter
        if is_first:
            filt.reset(mapped_y, now)
            smoothed = mapped_y
            vel = 0.0
        else:
            smoothed, vel = filt.filter(mapped_y, now)

        # Save back into state
        if player_idx == 1:
            self._p1_last_valid_raw_y = raw_y
            self._p1_last_raw_y_sample_time = now
            self._p1_last_valid_detection_time = now
            self._p1_is_first_sample = is_first
            self._min_y_bound = min_b
            self._max_y_bound = max_b
            self.p1_smoothed_y = smoothed
            self._p1_clamped_at_max_count = c_max
            self._p1_clamped_at_min_count = c_min
            self._p1_velocity_y = vel
            self._smoothed_y = smoothed
            self._last_valid_raw_y = raw_y
            self._last_valid_detection_time = now
            self._is_first_sample = is_first
        else:
            self._p2_last_valid_raw_y = raw_y
            self._p2_last_raw_y_sample_time = now
            self._p2_last_valid_detection_time = now
            self._p2_is_first_sample = is_first
            self._p2_min_y_bound = min_b
            self._p2_max_y_bound = max_b
            self.p2_smoothed_y = smoothed
            self._p2_clamped_at_max_count = c_max
            self._p2_clamped_at_min_count = c_min
            self._p2_velocity_y = vel

        return smoothed

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, float, np.ndarray]:
        """
        Executes face detection supporting 1 or 2 players simultaneously.
        Partitions candidates horizontally: Leftmost = Player 1, Rightmost = Player 2.
        
        Returns:
            Tuple of (effective_p1_detected, p1_raw_y, p1_smoothed_y, hud_rgb_preview)
        """
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        now = time.perf_counter()

        faces: List[dict] = []

        # ─── 1. OpenCV YuNet DNN (Ultra-fast, zero-drop single-pass multi-face detector) ───
        if self._yunet is not None:
            try:
                faces = self._yunet_detect_multi(frame, w, h)
            except Exception as exc:
                logger.debug("OpenCV YuNet multi-face process exception: %s", exc)

        # ─── 2. MediaPipe Face Mesh fallback (Runs ONLY if YuNet is absent) ───
        elif self._face_mesh is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self._face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                if results and results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        landmarks = fl.landmark
                        # Anatomical facial landmarks: nose tip + glabella
                        nose_point = landmarks[NOSE_TIP_IDX]
                        glabella_point = landmarks[GLABELLA_IDX]
                        ry = float(0.55 * nose_point.y + 0.45 * glabella_point.y)
                        nose_x = nose_point.x * w
                        nose_y = nose_point.y * h

                        det_fh = abs(landmarks[152].y - landmarks[10].y) * h
                        det_fw = abs(landmarks[454].x - landmarks[234].x) * w
                        faces.append({
                            "raw_y": ry,
                            "center_x": nose_x,
                            "fx": nose_x - det_fw * 0.5,
                            "fy": nose_y - det_fh * 0.5,
                            "fw": det_fw, "fh": det_fh,
                            "nose_x": nose_x, "nose_y": nose_y,
                        })
            except Exception as exc:
                logger.debug("MediaPipe multi-face exception: %s", exc)

        # ─── 3. Haar Cascade fallback (Runs ONLY if YuNet and MediaPipe are absent) ───
        elif self._cascade is not None and not self._cascade.empty():
            det, ry, fx, fy, fw, fh = self._cascade_detect(frame, w, h)
            if det:
                faces.append({
                    "raw_y": ry,
                    "center_x": fx + fw * 0.5,
                    "fx": fx, "fy": fy, "fw": fw, "fh": fh,
                    "nose_x": fx + fw * 0.5, "nose_y": fy + fh * 0.5,
                })

        # Filter out tiny noise (faces < 14px)
        faces = [f for f in faces if f.get("fh", 20) >= 14 and f.get("fw", 20) >= 14]

        # Prioritize the 2 largest faces in foreground to ignore background spectators/passersby
        if len(faces) > 2:
            faces.sort(key=lambda f: f["fw"] * f["fh"], reverse=True)
            faces = faces[:2]

        # Horizontal ordering: Leftmost is Player 1, Rightmost is Player 2
        faces.sort(key=lambda f: f["center_x"])

        # Running horizontal spatial tracking for P1 and P2
        p1_ref_x = self._p1_tracked_x if self._p1_tracked_x is not None else (0.35 * w)
        p2_ref_x = self._p2_tracked_x if self._p2_tracked_x is not None else (0.65 * w)

        if len(faces) >= 2:
            p1_cand = faces[0]
            p2_cand = faces[-1]
            if (p2_cand["center_x"] - p1_cand["center_x"]) >= (0.08 * w):
                p1_face = p1_cand
                p2_face = p2_cand
                self._p1_tracked_x = 0.85 * p1_ref_x + 0.15 * p1_face["center_x"]
                self._p2_tracked_x = 0.85 * p2_ref_x + 0.15 * p2_face["center_x"]
            else:
                p1_face = max(faces, key=lambda f: f["fw"] * f["fh"])
                p2_face = None
        elif len(faces) == 1:
            cand = faces[0]
            # If in PvP and Player 2 was seen recently (<1.5s), perform nearest-neighbor association
            if self.is_pvp and (now - self._p2_last_valid_detection_time < 1.5):
                # Nearest-neighbor matching to previous player position:
                # Guarantees Player 2 NEVER gets swapped into Player 1 when standing near center
                d1 = abs(cand["center_x"] - p1_ref_x)
                d2 = abs(cand["center_x"] - p2_ref_x)
                if d1 <= d2:
                    p1_face = cand
                    p2_face = None
                    self._p1_tracked_x = 0.85 * p1_ref_x + 0.15 * p1_face["center_x"]
                else:
                    p1_face = None
                    p2_face = cand
                    self._p2_tracked_x = 0.85 * p2_ref_x + 0.15 * p2_face["center_x"]
            else:
                p1_face = cand
                p2_face = None
                self._p1_tracked_x = 0.85 * p1_ref_x + 0.15 * p1_face["center_x"]
        else:
            p1_face = None
            p2_face = None

        active_count = (1 if p1_face is not None else 0) + (1 if p2_face is not None else 0)
        self.face_count = active_count
        if not self.is_pvp and active_count >= 2:
            self.is_pvp = True

        # Update Player 1 state
        if p1_face is not None:
            effective_p1_detected = True
            self.p1_detected = True
            self.p1_raw_y = p1_face["raw_y"]
            self.p1_smoothed_y = self._update_player_state(p1_face["raw_y"], p1_face["fh"], h, now, player_idx=1)
        else:
            if (now - self._p1_last_valid_detection_time) < 0.4 and not self._p1_is_first_sample:
                effective_p1_detected = True
                self.p1_detected = True
            else:
                effective_p1_detected = False
                self.p1_detected = False

        # Update Player 2 state
        if p2_face is not None:
            raw_y_p2 = p2_face["raw_y"]
            self.p2_detected = True
            self.p2_raw_y = raw_y_p2
            self.p2_smoothed_y = self._update_player_state(raw_y_p2, p2_face["fh"], h, now, player_idx=2)
        else:
            if (now - self._p2_last_valid_detection_time) < 0.4 and not self._p2_is_first_sample:
                self.p2_detected = True
            else:
                self.p2_detected = False

        if self._recenter_needed:
            self._recenter_needed = False

        # Build dual-player cyber HUD preview
        hud_frame = self._build_hud_preview_multi(frame, w, h, p1_face, p2_face)

        return effective_p1_detected, self.p1_raw_y, self.p1_smoothed_y, hud_frame

    def _yunet_detect_multi(self, frame: np.ndarray, w: int, h: int) -> List[dict]:
        """
        Detects multiple faces using OpenCV YuNet DNN with downscaled single-pass scan.
        Optimized for 60 FPS edge computing on Raspberry Pi 4.
        """
        # Edge-optimized scan size (192x144) guarantees steady 30-40+ FPS inference on Raspberry Pi 4 CPU
        net_w, net_h = 192, 144
        detect_frame = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
        detect_scale_x = w / float(net_w)
        detect_scale_y = h / float(net_h)

        self._yunet.setInputSize((net_w, net_h))
        _, faces = self._yunet.detect(detect_frame)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            score = float(face[-1])
            if score < 0.20:
                continue
            fx = float(face[0] * detect_scale_x)
            fy = float(face[1] * detect_scale_y)
            fw = float(face[2] * detect_scale_x)
            fh = float(face[3] * detect_scale_y)

            # Minimum size threshold: face must be at least 8px in scan space (detects distant players)
            if fh < 8 or fw < 8:
                continue

            # Key landmarks
            re_x = float(face[4] * detect_scale_x)
            re_y = float(face[5] * detect_scale_y)
            le_x = float(face[6] * detect_scale_x)
            le_y = float(face[7] * detect_scale_y)
            nose_x = float(face[8] * detect_scale_x)
            nose_y = float(face[9] * detect_scale_y)
            rm_x = float(face[10] * detect_scale_x)
            rm_y = float(face[11] * detect_scale_y)
            lm_x = float(face[12] * detect_scale_x)
            lm_y = float(face[13] * detect_scale_y)

            # Anatomical mid-face anchor: combines eyes + face center + nose tip for strong vertical responsiveness
            glabella_y = 0.5 * (re_y + le_y)
            face_center_y = fy + fh * 0.5
            anchor_y = 0.40 * glabella_y + 0.35 * face_center_y + 0.25 * nose_y
            raw_y = float(anchor_y / float(h))

            center_x = fx + fw * 0.5

            results.append({
                "raw_y": raw_y,
                "center_x": center_x,
                "fx": fx, "fy": fy, "fw": fw, "fh": fh,
                "nose_x": nose_x, "nose_y": nose_y,
                "re_x": re_x, "re_y": re_y, "le_x": le_x, "le_y": le_y,
                "rm_x": rm_x, "rm_y": rm_y, "lm_x": lm_x, "lm_y": lm_y,
                "score": score,
            })

        if not results:
            return []

        # Deduplicate vertical artifacts (e.g. neck/chest detection below true face)
        # If two candidates share almost the same X coordinate (|x1 - x2| < 0.16 * w):
        # The topmost one in the camera image (smaller fy) is the head; discard the bottom one (chest artifact).
        deduped = []
        results.sort(key=lambda f: f["fy"])
        for cand in results:
            is_chest_artifact = False
            for kept in deduped:
                if abs(cand["center_x"] - kept["center_x"]) < (0.16 * w):
                    is_chest_artifact = True
                    break
            if not is_chest_artifact:
                deduped.append(cand)

        # Sort horizontally: leftmost is first, rightmost is last
        deduped.sort(key=lambda f: f["center_x"])
        return deduped

    def _yunet_detect_2phase(
        self, frame: np.ndarray, w: int, h: int,
    ) -> Tuple[bool, float, float, float, float, float, float, float,
               float, float, float, float, float, float, float, float]:
        """Backward compatible 2-phase YuNet detection returning primary face."""
        faces = self._yunet_detect_multi(frame, w, h)
        if not faces:
            return (False, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        best = faces[0]
        return (
            True, best["raw_y"], best["nose_x"], best["nose_y"],
            best["fx"], best["fy"], best["fw"], best["fh"],
            best["re_x"], best["re_y"], best["le_x"], best["le_y"],
            best["rm_x"], best["rm_y"], best["lm_x"], best["lm_y"],
        )

    def _cascade_detect(
        self, frame: np.ndarray, w: int, h: int,
    ) -> Tuple[bool, float, float, float, float, float]:
        """Haar Cascade fallback detection. Returns (detected, raw_y, fx, fy, fw, fh)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if w > 360:
            scale_factor = 320.0 / float(w)
            detect_img = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
            inv_scale = 1.0 / scale_factor
        else:
            detect_img = gray
            inv_scale = 1.0

        detect_eq = cv2.equalizeHist(detect_img)
        faces_arr = self._cascade.detectMultiScale(
            detect_eq, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36),
        )
        if len(faces_arr) == 0:
            return (False, 0.5, 0, 0, 0, 0)

        r_fx, r_fy, r_fw, r_fh = max(faces_arr, key=lambda b: b[2] * b[3])
        det_fx = r_fx * inv_scale
        det_fy = r_fy * inv_scale
        det_fw = r_fw * inv_scale
        det_fh = r_fh * inv_scale

        # Temporal bounding box smoothing
        if self._cascade_box is None:
            self._cascade_box = (det_fx, det_fy, det_fw, det_fh)
        else:
            s_x, s_y, s_w, s_h = self._cascade_box
            if abs(det_fy - s_y) < s_h * 0.8:
                ba = 0.35
                self._cascade_box = (
                    s_x * (1.0 - ba) + det_fx * ba, s_y * (1.0 - ba) + det_fy * ba,
                    s_w * (1.0 - ba) + det_fw * ba, s_h * (1.0 - ba) + det_fh * ba,
                )
            else:
                self._cascade_box = (det_fx, det_fy, det_fw, det_fh)

        orig_fx, orig_fy, orig_fw, orig_fh = self._cascade_box
        center_y = orig_fy + orig_fh * 0.52
        raw_y = float(center_y / h)
        return (True, raw_y, orig_fx, orig_fy, orig_fw, orig_fh)

    def _build_hud_preview_multi(
        self, frame: np.ndarray, w: int, h: int,
        p1_face: Optional[dict], p2_face: Optional[dict],
    ) -> np.ndarray:
        """Renders dual-player HUD preview overlay with independent P1 (cyan) and P2 (crimson) bounding reticles."""
        preview_h = self.config.preview_height
        preview_w = self.config.preview_width
        hud_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        hud_frame = cv2.convertScaleAbs(hud_frame, alpha=0.45, beta=0)

        scale_x = preview_w / float(w)
        scale_y = preview_h / float(h)

        # Center demarcation divider in PvP mode
        if self.is_pvp or p2_face is not None:
            mid_x = preview_w // 2
            for dy in range(4, preview_h - 16, 8):
                cv2.line(hud_frame, (mid_x, dy), (mid_x, min(preview_h - 16, dy + 4)), (55, 65, 88), 1)

        # Render Player 1 (Cyan)
        if p1_face is not None:
            fx, fy, fw, fh = p1_face["fx"], p1_face["fy"], p1_face["fw"], p1_face["fh"]
            rx1 = int(fx * scale_x)
            ry1 = int(fy * scale_y)
            rx2 = int((fx + fw) * scale_x)
            ry2 = int((fy + fh) * scale_y)
            cv2.rectangle(hud_frame, (rx1, ry1), (rx2, ry2), (0, 240, 255), 1)
            cv2.putText(hud_frame, "P1", (rx1 + 2, max(11, ry1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 240, 255), 1, cv2.LINE_AA)
            nx = int(p1_face.get("nose_x", fx + fw * 0.5) * scale_x)
            ny = int(p1_face.get("nose_y", fy + fh * 0.5) * scale_y)
            cv2.circle(hud_frame, (nx, ny), 3, (0, 255, 200), -1)

        # Render Player 2 (Crimson/Red)
        if p2_face is not None:
            fx, fy, fw, fh = p2_face["fx"], p2_face["fy"], p2_face["fw"], p2_face["fh"]
            rx1 = int(fx * scale_x)
            ry1 = int(fy * scale_y)
            rx2 = int((fx + fw) * scale_x)
            ry2 = int((fy + fh) * scale_y)
            cv2.rectangle(hud_frame, (rx1, ry1), (rx2, ry2), (255, 60, 90), 1)
            cv2.putText(hud_frame, "P2", (rx1 + 2, max(11, ry1 - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 60, 90), 1, cv2.LINE_AA)
            nx = int(p2_face.get("nose_x", fx + fw * 0.5) * scale_x)
            ny = int(p2_face.get("nose_y", fy + fh * 0.5) * scale_y)
            cv2.circle(hud_frame, (nx, ny), 3, (255, 60, 90), -1)

        # Status text at bottom of PIP
        if self.is_pvp:
            p1_tag = "LOCK" if self.p1_detected else "SEEK"
            p2_tag = "LOCK" if self.p2_detected else "SEEK"
            stat_text = f"P1:{p1_tag}  P2:{p2_tag}"
            stat_col = (0, 255, 180) if (self.p1_detected and self.p2_detected) else (255, 190, 0)
        elif self.p1_detected:
            stat_text = f"LOCKED • {self.device_profile.value}"
            stat_col = (0, 255, 180)
        else:
            stat_text = "SEARCHING FOR FACE"
            stat_col = (0, 80, 255)

        cv2.putText(hud_frame, stat_text, (6, preview_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, stat_col, 1, cv2.LINE_AA)

        return cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)

    def _build_hud_preview(self, *args, **kwargs) -> np.ndarray:
        """Backward compatibility stub for legacy calls."""
        frame = args[0] if len(args) > 0 else np.zeros((240, 320, 3), dtype=np.uint8)
        w = args[1] if len(args) > 1 else frame.shape[1]
        h = args[2] if len(args) > 2 else frame.shape[0]
        return self._build_hud_preview_multi(frame, w, h, None, None)

    def close(self) -> None:
        """Releases detector resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        self._yunet = None


class VisionPipeline:
    """Threaded manager that coordinates video capture and face tracking."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = ThreadedCamera(
            device_index=config.device_index,
            width=config.capture_width,
            height=config.capture_height,
            target_fps=config.target_fps,
        )
        self.tracker = FaceMeshTracker(config)

        self._state = TrackingState()
        self._lock = threading.Lock()
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Watchdog for camera hardware switches
        self._last_camera_device_counter: int = 0

        # Calibration tracking
        self._calibrating = False
        self._calibration_start_time = 0.0
        self._calibration_samples_p1: List[float] = []
        self._calibration_samples_p2: List[float] = []

    def start(self) -> None:
        """Starts camera and the dedicated computer vision pipeline."""
        self.camera.start()
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._run_pipeline, daemon=True, name="VisionPipelineWorker")
        self._worker_thread.start()

    def begin_calibration(self) -> None:
        """Initiates head position calibration routine for active players."""
        with self._lock:
            self._calibrating = True
            self._calibration_start_time = time.perf_counter()
            self._calibration_samples_p1.clear()
            self._calibration_samples_p2.clear()
            self._state.is_calibrated = False
            self._state.calibration_progress = 0.0

    def finish_calibration(self) -> None:
        """Computes calibrated baseline and finalizes calibration state."""
        with self._lock:
            self._finish_calibration()

    def set_screen_dimensions(self, width: int, height: int) -> None:
        """Notifies tracker of screen resolution to calibrate adaptive sensitivity."""
        self.tracker.set_screen_dimensions(width, height)

    def cycle_device_profile(self) -> DeviceProfile:
        """Cycles between AUTO -> TV -> LAPTOP -> TABLET -> AUTO."""
        new_prof = self.tracker.cycle_device_profile()
        with self._lock:
            self._state.device_profile_label = new_prof.value
        return new_prof

    def set_device_profile(self, profile: DeviceProfile) -> None:
        """Sets a specific device profile."""
        self.tracker.device_profile = profile
        self.tracker.calibrate_baseline(self.tracker._last_valid_raw_y)
        with self._lock:
            self._state.device_profile_label = profile.value

    def recenter_baseline(self) -> None:
        """Forces tracker to re-anchor neutral position on next frame."""
        self.tracker.reset_for_new_camera()

    def set_pvp_mode(self, is_pvp: bool) -> None:
        """Sets or resets PvP mode in tracker and thread-safe tracking state."""
        with self._lock:
            self._state.is_pvp = is_pvp
        self.tracker.set_pvp_mode(is_pvp)

    def _run_pipeline(self) -> None:
        """Continuous pipeline loop running at camera FPS with instant event-driven wakeup."""
        while self._is_running:
            # Wait for camera thread to deliver a fresh frame (zero polling latency)
            self.camera.frame_ready_event.wait(timeout=0.035)
            self.camera.frame_ready_event.clear()

            # Detect hardware camera switch (e.g. unplug USB -> fallback to integrated)
            dev_counter = getattr(self.camera, "device_change_counter", 0)
            if dev_counter != self._last_camera_device_counter:
                self._last_camera_device_counter = dev_counter
                self.tracker.reset_for_new_camera()
                logger.info("VisionPipeline detected camera device change (counter %d). Resetting tracker.", dev_counter)

            has_frame, frame = self.camera.read(copy=False)
            if has_frame and frame is not None:
                detected, raw_y, smoothed_y, hud_preview = self.tracker.process_frame(frame)
                now = time.perf_counter()

                with self._lock:
                    self._state.face_detected = detected
                    self._state.raw_y = raw_y
                    self._state.smoothed_y = smoothed_y
                    self._state.velocity_y = self.tracker._p1_velocity_y
                    self._state.sample_time = now
                    self._state.preview_surface_buffer = hud_preview
                    self._state.device_profile_label = self.tracker.device_profile.value
                    self._state.face_count = self.tracker.face_count
                    self._state.is_pvp = self.tracker.is_pvp
                    self._state.p1_detected = self.tracker.p1_detected
                    self._state.p1_raw_y = self.tracker.p1_raw_y
                    self._state.p1_smoothed_y = self.tracker.p1_smoothed_y
                    self._state.p1_velocity_y = self.tracker._p1_velocity_y
                    self._state.p2_detected = self.tracker.p2_detected
                    self._state.p2_raw_y = self.tracker.p2_raw_y
                    self._state.p2_smoothed_y = self.tracker.p2_smoothed_y
                    self._state.p2_velocity_y = self.tracker._p2_velocity_y

                    if detected:
                        self._state.last_detected_time = now

                    # Handle calibration gathering
                    if self._calibrating:
                        if self.tracker.p1_detected:
                            self._calibration_samples_p1.append(self.tracker.p1_raw_y)
                        if self.tracker.p2_detected:
                            self._calibration_samples_p2.append(self.tracker.p2_raw_y)

                        elapsed = now - self._calibration_start_time
                        progress = min(1.0, elapsed / max(0.1, self.config.calibration_duration_sec))
                        self._state.calibration_progress = progress

                        if elapsed >= self.config.calibration_duration_sec:
                            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Computes calibrated baseline from accumulated samples for all active players."""
        self._calibrating = False
        if self._calibration_samples_p1:
            median_y_p1 = float(np.median(self._calibration_samples_p1))
            self.tracker.calibrate_baseline(median_y_p1, player_idx=1)
        if self._calibration_samples_p2:
            median_y_p2 = float(np.median(self._calibration_samples_p2))
            self.tracker.calibrate_baseline(median_y_p2, player_idx=2)
        elif self._calibration_samples_p1:
            median_y_p1 = float(np.median(self._calibration_samples_p1))
            self.tracker.calibrate_baseline(median_y_p1, player_idx=2)

        self._state.is_calibrated = True
        logger.info(
            "Calibration finished: P1 samples=%d, P2 samples=%d",
            len(self._calibration_samples_p1), len(self._calibration_samples_p2)
        )

    def get_state(self) -> TrackingState:
        """Thread-safe snapshot copy of current tracking data."""
        with self._lock:
            return TrackingState(
                face_detected=self._state.face_detected,
                raw_y=self._state.raw_y,
                normalized_y=self._state.normalized_y,
                smoothed_y=self._state.smoothed_y,
                velocity_y=self._state.velocity_y,
                sample_time=self._state.sample_time,
                last_detected_time=self._state.last_detected_time,
                preview_surface_buffer=self._state.preview_surface_buffer,
                calibration_progress=self._state.calibration_progress,
                is_calibrated=self._state.is_calibrated,
                device_profile_label=self.tracker.device_profile.value,
                face_count=self._state.face_count,
                is_pvp=self._state.is_pvp,
                p1_detected=self._state.p1_detected,
                p1_raw_y=self._state.p1_raw_y,
                p1_smoothed_y=self._state.p1_smoothed_y,
                p1_velocity_y=self._state.p1_velocity_y,
                p2_detected=self._state.p2_detected,
                p2_raw_y=self._state.p2_raw_y,
                p2_smoothed_y=self._state.p2_smoothed_y,
                p2_velocity_y=self._state.p2_velocity_y,
            )

    def stop(self) -> None:
        """Stops the pipeline and cleans up camera resources."""
        self._is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self.camera.stop()
        self.tracker.close()
