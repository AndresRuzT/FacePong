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

        # Micro-tremor deadband anchor state
        self._p1_last_deadband_y: float = 0.5
        self._p2_last_deadband_y: float = 0.5

        # Sub-Pixel Lucas-Kanade Optical Flow (KLT) tracking state
        self._p1_klt_pts: Optional[np.ndarray] = None
        self._p1_prev_gray: Optional[np.ndarray] = None
        self._p1_klt_y: Optional[float] = None
        self._p2_klt_pts: Optional[np.ndarray] = None
        self._p2_prev_gray: Optional[np.ndarray] = None
        self._p2_klt_y: Optional[float] = None

        # Haar Cascade bounding box smoothing and ROI tracking state
        self._cascade_box: Optional[Tuple[float, float, float, float]] = None
        self._cascade_miss_count: int = 0

        # Detectors
        self._detector_net: Optional[cv2.dnn.Net] = None
        self._slim_net: Optional[cv2.dnn.Net] = None
        self._priors: Optional[np.ndarray] = None
        self._face_mesh = None
        self._yunet = None
        self._rfb_net: Optional[cv2.dnn.Net] = None
        self._rfb_priors: Optional[np.ndarray] = None
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
        """Returns (min_cutoff, beta) tuned for physical camera tracking space and sensitivity."""
        sens_ratio = max(0.5, getattr(self.config, "head_sensitivity", 1.8)) / 1.8
        if self.device_profile == DeviceProfile.TV:
            return 1.0, 1.8 * sens_ratio
        elif self.device_profile == DeviceProfile.LAPTOP:
            return 1.0, 1.5 * sens_ratio
        elif self.device_profile == DeviceProfile.TABLET:
            return 1.0, 1.2 * sens_ratio
        return 1.0, 1.5 * sens_ratio

    def set_screen_dimensions(self, width: int, height: int) -> None:
        """Notifies tracker of the active display resolution to adjust sensitivity scaling."""
        self._screen_height = height

    def get_effective_range_spread(self, face_height_fraction: Optional[float] = None) -> float:
        """
        Calculates optimal head motion spread standardized across device classes:
          - TV: player is 1.5-3m away on a large display -> balanced ergonomic spread (~0.115)
          - LAPTOP: player is ~50cm away at desk -> effortless spread (~0.125)
          - TABLET: player is ~35cm away, close-up -> comfortable spread (~0.135)
          - AUTO: scales with screen resolution and player ergonomics
        """
        if self.device_profile == DeviceProfile.TV:
            base_spread = 0.115
        elif self.device_profile == DeviceProfile.LAPTOP:
            base_spread = 0.125
        elif self.device_profile == DeviceProfile.TABLET:
            base_spread = 0.135
        else:
            sensitivity = max(0.5, getattr(self.config, "head_sensitivity", 2.4))
            config_spread = getattr(self.config, "default_range_spread", 0.170)
            std_spread = config_spread / (sensitivity ** 0.6)
            if self._screen_height >= 1000:
                base_spread = std_spread * 0.90
            else:
                base_spread = std_spread

        # Distance normalization: subtle compensation for player distance to maintain comfortable angular travel
        if face_height_fraction is not None and face_height_fraction > 0.04:
            ref_scale = 0.20  # reference face height fraction
            dist_multiplier = float(np.clip(face_height_fraction / ref_scale, 0.88, 1.12))
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
        """Initializes Ultra-Light-Fast-Generic-Face-Detector (Slim-320 ONNX) as primary edge engine."""
        # 1. Primary Engine: Ultra-Light-Fast-Generic-Face-Detector (Slim-320 ONNX / RFB-320 ONNX)
        # Dedicated edge neural network optimized for ARM Cortex-A72 / Raspberry Pi 4 CPU (~12ms per frame)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        slim_path = os.path.join(current_dir, "version-slim-320.onnx")
        rfb_path = os.path.join(current_dir, "version-RFB-320.onnx")
        model_path = slim_path if os.path.exists(slim_path) else rfb_path

        cv2.setNumThreads(2)
        if os.path.exists(model_path):
            try:
                net = cv2.dnn.readNetFromONNX(model_path)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self._detector_net = net
                self._slim_net = net
                self._rfb_net = net
                self._priors = self._generate_rfb_priors((320, 240))
                self._rfb_priors = self._priors
                logger.info("Ultra-Light-Fast ONNX face detector initialized as primary engine: %s", model_path)
            except Exception as exc:
                logger.warning("Ultra-Light-Fast ONNX initialization failed: %s", exc)

        # 2. YuNet DNN secondary edge fallback
        try:
            yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
            if os.path.exists(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
                self._yunet = cv2.FaceDetectorYN.create(
                    model=yunet_path,
                    config="",
                    input_size=(320, 240),
                    score_threshold=0.15,
                    nms_threshold=0.3,
                    top_k=8,
                )
        except Exception as exc:
            logger.debug("YuNet initialization failed: %s", exc)

        # 3. OpenCV Haar Cascade emergency fallback
        try:
            bundled_path = os.path.join(current_dir, "haarcascade_frontalface_default.xml")
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            system_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            for c_path in (bundled_path, system_path):
                if c_path and os.path.exists(c_path):
                    cascade = cv2.CascadeClassifier(c_path)
                    if not cascade.empty():
                        self._cascade = cascade
                        break
        except Exception as exc:
            logger.debug("OpenCV Cascade initialization error: %s", exc)

        if self._detector_net is None and self._yunet is None and self._cascade is None:
            logger.error("No face tracking backend available! Neither Ultra-Light-Fast, YuNet, nor OpenCV Haar Cascade could be loaded.")

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
        self._p1_klt_pts = None
        self._p1_prev_gray = None
        self._p1_klt_y = None
        self._p2_klt_pts = None
        self._p2_prev_gray = None
        self._p2_klt_y = None
        self._p1_filter.reset()
        self._p2_filter.reset()
        self._p1_last_deadband_y = 0.5
        self._p2_last_deadband_y = 0.5
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
            self._p1_last_deadband_y = neutral_y
            self._p1_klt_y = neutral_y
            self._p1_filter.reset(neutral_y)
            logger.info(
                "Calibrated P1 head range: [%.2f, %.2f] around neutral %.2f (spread: ±%.3f, profile: %s)",
                self._min_y_bound, self._max_y_bound, neutral_y, range_spread, self.device_profile.value,
            )
        else:
            self._p2_min_y_bound = min_b
            self._p2_max_y_bound = max_b
            self._p2_is_first_sample = False
            self._p2_last_valid_raw_y = neutral_y
            self._p2_last_deadband_y = neutral_y
            self._p2_klt_y = neutral_y
            self._p2_filter.reset(neutral_y)
            logger.info(
                "Calibrated P2 head range: [%.2f, %.2f] around neutral %.2f (spread: ±%.3f, profile: %s)",
                self._p2_min_y_bound, self._p2_max_y_bound, neutral_y, range_spread, self.device_profile.value,
            )

    def _update_player_state(
        self, raw_y: float, face_height: float, frame_height: int, now: float, player_idx: int = 1
    ) -> float:
        """Computes velocity, auto-baseline anchoring, physical One-Euro filtering, and ergonomic pitch mapping."""
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

            # Slew-rate limiter: prevents anomalous single-frame video glints while allowing instant fast flicks
            if not is_first and last_raw > 0:
                max_step = max(0.25, 9.0 * dt_s)
                if (raw_y - last_raw) > max_step:
                    raw_y = last_raw + max_step
                elif (last_raw - raw_y) > max_step:
                    raw_y = last_raw - max_step

        # One Euro Filter applied directly to raw physical camera coordinate:
        # Filtering in physical sensor space wipes out 100% of sub-pixel noise and camera quantization
        filt = self._p1_filter if player_idx == 1 else self._p2_filter
        if is_first:
            filt.reset(raw_y, now)
            clean_raw_y = raw_y
            raw_vel = 0.0
            if player_idx == 1:
                self._p1_last_deadband_y = raw_y
            else:
                self._p2_last_deadband_y = raw_y
        else:
            clean_raw_y, raw_vel = filt.filter(raw_y, now)

        # Continuous soft-knee noise gate: suppresses resting sensor micro-tremor without staircase quantization
        # Smooth quadratic attenuation eliminates jumps and keeps 100% continuous fluid tracking
        noise_threshold = 0.0030
        last_deadband = self._p1_last_deadband_y if player_idx == 1 else self._p2_last_deadband_y
        if is_first:
            if player_idx == 1:
                self._p1_last_deadband_y = clean_raw_y
            else:
                self._p2_last_deadband_y = clean_raw_y
        else:
            diff = clean_raw_y - last_deadband
            abs_diff = abs(diff)
            if abs_diff < noise_threshold:
                gain = (abs_diff / noise_threshold) ** 2
                clean_raw_y = last_deadband + diff * gain
            else:
                if player_idx == 1:
                    self._p1_last_deadband_y = clean_raw_y
                else:
                    self._p2_last_deadband_y = clean_raw_y

        # Distance-adaptive dynamic range spread based on face size
        face_frac = (face_height / float(frame_height)) if (face_height > 0 and frame_height > 0) else None
        target_spread = self.get_effective_range_spread(face_frac)

        # Baseline anchoring and continuous dynamic auto-centering
        time_since_last = (now - last_time) if last_time > 0 else 999.0
        if recenter or is_first or (not self._is_calibrated and time_since_last > 1.0):
            # Immediate lock: resting posture becomes immediate arena center (y = 0.50)
            mid_b = clean_raw_y
            min_b = max(0.02, mid_b - target_spread)
            max_b = min(0.98, mid_b + target_spread)
            is_first = False
            c_min = 0
            c_max = 0
        else:
            mid_b = 0.5 * (min_b + max_b)

            if not self._is_calibrated:
                # 1. Gentle continuous drift towards resting center when stationary
                if abs(raw_vel) < 0.05 and (min_b + 0.01) < clean_raw_y < (max_b - 0.01):
                    drift_step = np.clip(0.03 * dt_s * (clean_raw_y - mid_b), -0.002, 0.002)
                    mid_b += float(drift_step)

                # 2. Anti-stuck boundary relaxation: prevents paddle from locking against ceiling or floor
                if clean_raw_y < min_b:
                    c_min += 1
                    if c_min > 8:  # ~0.25s clamped at top
                        mid_b += (clean_raw_y - min_b) * 0.15
                else:
                    c_min = 0

                if clean_raw_y > max_b:
                    c_max += 1
                    if c_max > 8:  # ~0.25s clamped at bottom
                        mid_b += (clean_raw_y - max_b) * 0.15
                else:
                    c_max = 0

                # Spread adaptation for distance changes
                curr_spread = 0.5 * (max_b - min_b)
                if abs(target_spread - curr_spread) > 0.008:
                    curr_spread = 0.95 * curr_spread + 0.05 * target_spread
                min_b = max(0.02, mid_b - curr_spread)
                max_b = min(0.98, mid_b + curr_spread)
                mid_b = 0.5 * (min_b + max_b)
            else:
                # In calibrated mode: baseline bounds remain strictly anchored as calibrated
                if clean_raw_y < min_b:
                    c_min += 1
                else:
                    c_min = 0

                if clean_raw_y > max_b:
                    c_max += 1
                else:
                    c_max = 0

        # Ergonomic pitch mapping with cubic Hermite smoothstep:
        # Looking up involves cervical extension (0.70x span allows effortless ceiling reach without neck strain)
        # Looking down involves cervical flexion (0.85x span allows natural floor reach)
        if clean_raw_y <= mid_b:
            half_span_up = max(0.01, (mid_b - min_b) * 0.70)
            norm = min(1.0, max(0.0, (mid_b - clean_raw_y) / half_span_up))
            smooth_norm = norm * norm * (3.0 - 2.0 * norm)
            linear_y = 0.5 - 0.5 * smooth_norm
            norm_vel = 0.5 * (raw_vel / half_span_up)
        else:
            half_span_down = max(0.01, (max_b - mid_b) * 0.85)
            norm = min(1.0, max(0.0, (clean_raw_y - mid_b) / half_span_down))
            smooth_norm = norm * norm * (3.0 - 2.0 * norm)
            linear_y = 0.5 + 0.5 * smooth_norm
            norm_vel = 0.5 * (raw_vel / half_span_down)

        smoothed = float(np.clip(linear_y, 0.0, 1.0))
        vel = float(norm_vel)

        # Save back into state
        if player_idx == 1:
            self._p1_last_valid_raw_y = clean_raw_y
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
            self._last_valid_raw_y = clean_raw_y
            self._last_valid_detection_time = now
            self._is_first_sample = is_first
        else:
            self._p2_last_valid_raw_y = clean_raw_y
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

    def _apply_klt_optical_flow(
        self, gray: np.ndarray, face: Optional[dict], h: int, w: int, player_idx: int = 1
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Sub-pixel Lucas-Kanade Optical Flow (KLT) tracker with complementary neural anchoring.
        Measures exact physical head displacement with sub-pixel precision, eliminating discrete quantization steps.
        """
        pts_attr = "_p1_klt_pts" if player_idx == 1 else "_p2_klt_pts"
        prev_attr = "_p1_prev_gray" if player_idx == 1 else "_p2_prev_gray"
        klt_y_attr = "_p1_klt_y" if player_idx == 1 else "_p2_klt_y"

        old_pts = getattr(self, pts_attr)
        prev_gray = getattr(self, prev_attr)
        last_klt_y = getattr(self, klt_y_attr)

        det_anchor_y = face["raw_y"] if face is not None else None
        tracked_pts = None

        if prev_gray is not None and old_pts is not None and len(old_pts) >= 4:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, old_pts, None, winSize=(15, 15), maxLevel=2
            )
            valid = (status.ravel() == 1)
            if np.sum(valid) >= 4:
                valid_old = old_pts[valid, 0]
                valid_new = new_pts[valid, 0]
                dy = float(np.median(valid_new[:, 1] - valid_old[:, 1]))
                norm_dy = dy / float(h)

                tracked_pts = valid_new.reshape(-1, 1, 2)
                last_ref = last_klt_y if last_klt_y is not None else (det_anchor_y if det_anchor_y is not None else 0.5)

                if det_anchor_y is not None:
                    # High-frequency differential fusion: merges sub-pixel KLT delta with absolute anatomical anchor
                    flow_predicted = last_ref + norm_dy
                    fused_y = 0.50 * det_anchor_y + 0.50 * flow_predicted
                else:
                    fused_y = last_ref + norm_dy

                setattr(self, klt_y_attr, fused_y)
                setattr(self, pts_attr, tracked_pts)
                setattr(self, prev_attr, gray.copy())
                return fused_y, tracked_pts

        # Seed or refresh feature points on rigid facial triangle (eyes to nose bridge)
        if det_anchor_y is not None:
            setattr(self, klt_y_attr, det_anchor_y)
            fx, fy, fw, fh = face["fx"], face["fy"], face["fw"], face["fh"]
            mask = np.zeros_like(gray)
            y1 = max(0, int(fy + 0.15 * fh))
            y2 = min(h, int(fy + 0.70 * fh))
            x1 = max(0, int(fx + 0.15 * fw))
            x2 = min(w, int(fx + 0.85 * fw))
            mask[y1:y2, x1:x2] = 255
            new_features = cv2.goodFeaturesToTrack(
                gray, maxCorners=25, qualityLevel=0.015, minDistance=5, mask=mask
            )
            setattr(self, pts_attr, new_features)
            setattr(self, prev_attr, gray.copy())
            return det_anchor_y, new_features

        setattr(self, prev_attr, gray.copy())
        return (last_klt_y if last_klt_y is not None else 0.5), None

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

        # ─── 1. Ultra-Light-Fast-Generic-Face-Detector (Primary ONNX Engine) ───
        # Native 320x240 single-pass inference (~12ms on Raspberry Pi 4 CPU, zero queue buildup)
        if (self._detector_net is not None or self._rfb_net is not None) and (self._priors is not None or self._rfb_priors is not None):
            try:
                faces = self._onnx_detect_multi(frame, w, h)
            except Exception as exc:
                logger.debug("Ultra-Light-Fast ONNX detect exception: %s", exc)

        # ─── 2. Fallbacks (YuNet / Haar Cascade) ───
        if not faces and self._yunet is not None:
            try:
                faces = self._yunet_detect_multi(frame, w, h)
            except Exception as exc:
                logger.debug("YuNet detect exception: %s", exc)

        if not faces and self._cascade is not None and not self._cascade.empty():
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
            raw_y_p1 = p1_face["raw_y"]
            self.p1_raw_y = raw_y_p1
            self.p1_smoothed_y = self._update_player_state(raw_y_p1, p1_face["fh"], h, now, player_idx=1)
        else:
            if (now - self._p1_last_valid_detection_time) < 0.35 and not self._p1_is_first_sample:
                effective_p1_detected = True
                self.p1_detected = True
                raw_y_p1 = self._p1_last_valid_raw_y
                self.p1_raw_y = raw_y_p1
                self.p1_smoothed_y = self._update_player_state(raw_y_p1, 60.0, h, now, player_idx=1)
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
            if (now - self._p2_last_valid_detection_time) < 0.35 and not self._p2_is_first_sample:
                self.p2_detected = True
                raw_y_p2 = self._p2_last_valid_raw_y
                self.p2_raw_y = raw_y_p2
                self.p2_smoothed_y = self._update_player_state(raw_y_p2, 60.0, h, now, player_idx=2)
            else:
                self.p2_detected = False

        if self._recenter_needed:
            self._recenter_needed = False

        # Build dual-player cyber HUD preview
        hud_frame = self._build_hud_preview_multi(frame, w, h, p1_face, p2_face)

        return effective_p1_detected, self.p1_raw_y, self.p1_smoothed_y, hud_frame

    def _detect_yunet_roi(self, frame: np.ndarray, w: int, h: int, last_box: Tuple[float, float, float, float]) -> List[dict]:
        """Runs fast 160x160 ROI detection around previous face position (11ms on Raspberry Pi 4)."""
        fx, fy, fw, fh = last_box
        cx = fx + fw * 0.5
        cy = fy + fh * 0.5
        roi_size = max(144, int(max(fw, fh) * 2.2))
        rx1 = max(0, int(cx - roi_size * 0.5))
        ry1 = max(0, int(cy - roi_size * 0.5))
        if rx1 + roi_size > w:
            rx1 = max(0, w - roi_size)
        if ry1 + roi_size > h:
            ry1 = max(0, h - roi_size)
        rx2 = min(w, rx1 + roi_size)
        ry2 = min(h, ry1 + roi_size)
        rw = rx2 - rx1
        rh = ry2 - ry1
        if rw < 32 or rh < 32:
            return []

        roi = frame[ry1:ry2, rx1:rx2]
        roi_w, roi_h = 160, 160
        roi_resized = cv2.resize(roi, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
        self._yunet.setInputSize((roi_w, roi_h))
        _, faces = self._yunet.detect(roi_resized)
        if faces is None or len(faces) == 0:
            return []

        face = faces[0]
        score = float(face[-1])
        if score < 0.15:
            return []

        scale_x = rw / float(roi_w)
        scale_y = rh / float(roi_h)
        det_fx = rx1 + float(face[0] * scale_x)
        det_fy = ry1 + float(face[1] * scale_y)
        det_fw = float(face[2] * scale_x)
        det_fh = float(face[3] * scale_y)
        re_x = rx1 + float(face[4] * scale_x)
        re_y = ry1 + float(face[5] * scale_y)
        le_x = rx1 + float(face[6] * scale_x)
        le_y = ry1 + float(face[7] * scale_y)
        nose_x = rx1 + float(face[8] * scale_x)
        nose_y = ry1 + float(face[9] * scale_y)
        rm_x = rx1 + float(face[10] * scale_x)
        rm_y = ry1 + float(face[11] * scale_y)
        lm_x = rx1 + float(face[12] * scale_x)
        lm_y = ry1 + float(face[13] * scale_y)

        glabella_y = 0.5 * (re_y + le_y)
        box_center_y = det_fy + 0.5 * det_fh
        anchor_y = 0.35 * glabella_y + 0.45 * nose_y + 0.20 * box_center_y
        raw_y = float(anchor_y / float(h))

        return [{
            "raw_y": raw_y,
            "center_x": det_fx + det_fw * 0.5,
            "fx": det_fx, "fy": det_fy, "fw": det_fw, "fh": det_fh,
            "nose_x": nose_x, "nose_y": nose_y,
            "re_x": re_x, "re_y": re_y, "le_x": le_x, "le_y": le_y,
            "rm_x": rm_x, "rm_y": rm_y, "lm_x": lm_x, "lm_y": lm_y,
            "score": score,
        }]

    def _yunet_detect_multi(self, frame: np.ndarray, w: int, h: int) -> List[dict]:
        """
        Detects multiple faces using OpenCV YuNet DNN with native 256x192 scan.
        Ensures robust face and landmark detection (21ms on Raspberry Pi 4).
        """
        detect_w, detect_h = min(256, w), min(192, h)
        if w != detect_w or h != detect_h:
            detect_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
            detect_scale_x = w / float(detect_w)
            detect_scale_y = h / float(detect_h)
        else:
            detect_frame = frame
            detect_scale_x = 1.0
            detect_scale_y = 1.0

        self._yunet.setInputSize((detect_w, detect_h))
        _, faces = self._yunet.detect(detect_frame)
        if faces is None or len(faces) == 0:
            return []

        results = []
        for face in faces:
            score = float(face[-1])
            if score < 0.15:
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

            # Anatomical mid-face anchor: rigid skull landmarks (eyes midpoint + nose tip + bounding box center)
            # Highly responsive to natural head pitch rotation while eliminating single-landmark flutter
            glabella_y = 0.5 * (re_y + le_y)
            box_center_y = fy + 0.5 * fh
            anchor_y = 0.35 * glabella_y + 0.45 * nose_y + 0.20 * box_center_y
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
        if deduped:
            self._yunet_last_box = (deduped[0]["fx"], deduped[0]["fy"], deduped[0]["fw"], deduped[0]["fh"])
            self._yunet_roi_miss = 0
        return deduped

    @staticmethod
    def _generate_rfb_priors(image_size: Tuple[int, int] = (320, 240)) -> np.ndarray:
        """Precomputes anchor priors for Ultra-Light-Fast-Face RFB-320 ONNX network."""
        w, h = image_size
        feature_maps = [[math.ceil(h / s), math.ceil(w / s)] for s in [8, 16, 32, 64]]
        min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        strides = [8, 16, 32, 64]
        priors = []
        for k, (f_h, f_w) in enumerate(feature_maps):
            scale_w = w / strides[k]
            scale_h = h / strides[k]
            for i in range(f_h):
                for j in range(f_w):
                    cx = (j + 0.5) / scale_w
                    cy = (i + 0.5) / scale_h
                    for min_box in min_boxes[k]:
                        priors.append([cx, cy, min_box / w, min_box / h])
        return np.array(priors, dtype=np.float32)

    def _onnx_detect_multi(self, frame: np.ndarray, w: int, h: int) -> List[dict]:
        """
        Detects faces using Ultra-Light-Fast-Face (Slim-320 / RFB-320 ONNX) network.
        Specialized for edge inference (~12ms on Raspberry Pi 4 CPU) with zero queue delay.
        """
        net = self._detector_net or self._slim_net or self._rfb_net
        priors = self._priors if self._priors is not None else self._rfb_priors
        if net is None or priors is None:
            return []

        detect_w, detect_h = min(320, w), min(240, h)
        if w != detect_w or h != detect_h:
            detect_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
            detect_scale_x = w / float(detect_w)
            detect_scale_y = h / float(detect_h)
        else:
            detect_frame = frame
            detect_scale_x = 1.0
            detect_scale_y = 1.0

        blob = cv2.dnn.blobFromImage(
            detect_frame, 1.0 / 128.0, (detect_w, detect_h), (127.0, 127.0, 127.0), swapRB=True
        )
        net.setInput(blob)
        scores, boxes = net.forward(["scores", "boxes"])

        conf = scores[0, :, 1]
        score_thresh = 0.50
        mask = conf > score_thresh
        if not np.any(mask):
            return []

        sub_conf = conf[mask]
        sub_boxes = boxes[0, mask, :]
        sub_priors = priors[mask, :]

        cx = (sub_priors[:, 0] + sub_boxes[:, 0] * 0.1 * sub_priors[:, 2]) * detect_w * detect_scale_x
        cy = (sub_priors[:, 1] + sub_boxes[:, 1] * 0.1 * sub_priors[:, 3]) * detect_h * detect_scale_y
        bw = (sub_priors[:, 2] * np.exp(sub_boxes[:, 2] * 0.2)) * detect_w * detect_scale_x
        bh = (sub_priors[:, 3] * np.exp(sub_boxes[:, 3] * 0.2)) * detect_h * detect_scale_y

        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5

        cv_boxes = [[int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])] for i in range(len(x1))]
        cv_scores = [float(sub_conf[i]) for i in range(len(sub_conf))]

        indices = cv2.dnn.NMSBoxes(cv_boxes, cv_scores, score_thresh, 0.3)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw_i, bh_i = cv_boxes[idx]
                if bh_i < 8 or bw_i < 8:
                    continue
                raw_y = (by + bh_i * 0.42) / float(h)
                results.append({
                    "raw_y": float(raw_y),
                    "center_x": float(bx + bw_i * 0.5),
                    "fx": float(bx), "fy": float(by), "fw": float(bw_i), "fh": float(bh_i),
                    "nose_x": float(bx + bw_i * 0.5), "nose_y": float(by + bh_i * 0.42),
                    "score": float(cv_scores[idx]),
                })

        if not results:
            return []

        # Deduplicate vertical artifacts
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

        deduped.sort(key=lambda f: f["center_x"])
        return deduped

    def _rfb_detect_multi(self, frame: np.ndarray, w: int, h: int) -> List[dict]:
        return self._onnx_detect_multi(frame, w, h)

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
        hud_frame = hud_frame >> 1

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
            ny = int(p2_face.get("nose_y", fy + fh * 0.42) * scale_y)
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
            # Wait for camera thread to deliver a fresh frame (zero polling latency, responsive fallback timeout)
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
