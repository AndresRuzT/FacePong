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

from facepong.config import CameraConfig
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


@dataclass
class TrackingState:
    """Thread-safe snapshot of face tracking data."""
    face_detected: bool = False
    raw_y: float = 0.5
    normalized_y: float = 0.5
    smoothed_y: float = 0.5
    last_detected_time: float = 0.0
    preview_surface_buffer: Optional[np.ndarray] = None
    calibration_progress: float = 0.0
    is_calibrated: bool = False


class FaceMeshTracker:
    """Processes camera frames to track facial position, produce HUD previews, and auto-anchor baseline."""

    def __init__(self, config: CameraConfig):
        self.config = config

        # Exponential Moving Average state
        self._smoothed_y: float = 0.5
        self._is_first_sample: bool = True

        # Sensitivity-aware calibration bounds
        sensitivity = max(0.5, getattr(config, "head_sensitivity", 2.4))
        base_spread = getattr(config, "default_range_spread", 0.18)
        initial_spread = base_spread / (sensitivity ** 0.6)
        self._min_y_bound: float = max(0.02, 0.5 - initial_spread)
        self._max_y_bound: float = min(0.98, 0.5 + initial_spread)
        self._calibration_samples: List[float] = []

        # Grace period & adaptive recentering state
        self._recenter_needed: bool = False
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
        self._yunet_raw_y_smooth: Optional[float] = None
        self._cascade = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes MediaPipe Face Mesh OR OpenCV YuNet Deep Neural Network as primary detectors."""
        # 1. MediaPipe Solutions FaceMesh (laptop / environments with solutions API)
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,
                    min_detection_confidence=0.35,
                    min_tracking_confidence=0.35,
                )
                logger.info("MediaPipe Face Mesh tracker initialized (Solutions API)")
        except Exception as exc:
            logger.debug("MediaPipe Solutions initialization failed (%s).", exc)

        # 2. OpenCV YuNet Deep Neural Network Face & Landmark Detector (Raspberry Pi & edge devices)
        if self._face_mesh is None:
            try:
                cv2.setNumThreads(4)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                yunet_path = os.path.join(current_dir, "face_detection_yunet_2023mar.onnx")
                if os.path.exists(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
                    self._yunet = cv2.FaceDetectorYN.create(
                        model=yunet_path,
                        config="",
                        input_size=(320, 240),
                        score_threshold=0.35,
                        nms_threshold=0.3,
                        top_k=10,
                    )
                    logger.info("OpenCV YuNet Deep Neural Network tracker initialized from %s (320x240, 4 threads)", yunet_path)
            except Exception as exc:
                logger.warning("OpenCV YuNet initialization failed: %s", exc)

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
            logger.error("No face tracking backend available! Neither MediaPipe, YuNet, nor OpenCV Haar Cascade could be loaded.")

    def reset_for_new_camera(self) -> None:
        """Flags that a camera switch occurred and baseline must re-anchor to new camera frame."""
        self._recenter_needed = True
        self._is_first_sample = True
        self._yunet_raw_y_smooth = None
        self._clamped_at_min_count = 0
        self._clamped_at_max_count = 0
        logger.info("FaceMeshTracker flagged for baseline re-anchor on new camera feed")

    def calibrate_baseline(self, neutral_y: float, range_spread: Optional[float] = None) -> None:
        """Sets comfortable dynamic range around player's neutral head position based on sensitivity."""
        if range_spread is None:
            sensitivity = max(0.5, getattr(self.config, "head_sensitivity", 2.4))
            base_spread = getattr(self.config, "default_range_spread", 0.18)
            range_spread = base_spread / (sensitivity ** 0.6)
        self._min_y_bound = max(0.02, neutral_y - range_spread)
        self._max_y_bound = min(0.98, neutral_y + range_spread)
        logger.info(
            "Calibrated head range: [%.2f, %.2f] around neutral %.2f (effective spread: ±%.3f)",
            self._min_y_bound, self._max_y_bound, neutral_y, range_spread,
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, float, np.ndarray]:
        """
        Executes face detection, extracts smoothed Y, adapts baseline if clamped,
        and renders cyber HUD preview.

        Returns:
            Tuple of (effective_face_detected, raw_y, smoothed_y, hud_rgb_preview)
        """
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_face_detected = False
        raw_y = self._last_valid_raw_y
        now = time.perf_counter()

        # Render preview HUD frame: SIMD-accelerated dimming without extra zeros allocation
        preview_h = self.config.preview_height
        preview_w = self.config.preview_width
        hud_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        hud_frame = cv2.convertScaleAbs(hud_frame, alpha=0.45, beta=0)

        scale_x = preview_w / float(w)
        scale_y = preview_h / float(h)

        if self._face_mesh is not None:
            try:
                # Set writeable=False to allow MediaPipe to pass frame by reference without memory copy
                rgb_frame.flags.writeable = False
                results = self._face_mesh.process(rgb_frame)
                rgb_frame.flags.writeable = True
                if results and results.multi_face_landmarks:
                    raw_face_detected = True
                    landmarks = results.multi_face_landmarks[0].landmark
                    nose_point = landmarks[NOSE_TIP_IDX]
                    glabella_point = landmarks[GLABELLA_IDX]
                    # Combined anatomical mid-face anchor: stable against mouth movement, yaw, and nodding
                    raw_y = float(0.55 * nose_point.y + 0.45 * glabella_point.y)

                    # Render futuristic landmarks onto HUD preview
                    nose_px = int(nose_point.x * preview_w)
                    nose_py = int(nose_point.y * preview_h)

                    for idx in CONTOUR_KEYPOINTS:
                        pt = landmarks[idx]
                        cv2.circle(hud_frame, (int(pt.x * preview_w), int(pt.y * preview_h)), 1, (0, 180, 255), -1)

                    for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX, CHIN_IDX, FOREHEAD_IDX):
                        pt = landmarks[idx]
                        cv2.circle(hud_frame, (int(pt.x * preview_w), int(pt.y * preview_h)), 2, (255, 0, 180), -1)

                    cv2.circle(hud_frame, (nose_px, nose_py), 6, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.line(hud_frame, (nose_px - 10, nose_py), (nose_px + 10, nose_py), (0, 255, 200), 1)
                    cv2.line(hud_frame, (nose_px, nose_py - 10), (nose_px, nose_py + 10), (0, 255, 200), 1)

                    cv2.putText(
                        hud_frame,
                        "TRACKING: LOCKED",
                        (8, preview_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 180),
                        1,
                        cv2.LINE_AA,
                    )
            except Exception as exc:
                logger.debug("MediaPipe Face Mesh process exception: %s", exc)

        elif self._yunet is not None:
            try:
                # YuNet at native 320x240 capture resolution with 4-thread acceleration
                net_w, net_h = 320, 240
                if w != net_w or h != net_h:
                    detect_frame = cv2.resize(frame, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
                    scale_to_orig_x = w / float(net_w)
                    scale_to_orig_y = h / float(net_h)
                else:
                    detect_frame = frame
                    scale_to_orig_x = 1.0
                    scale_to_orig_y = 1.0

                if getattr(self, "_yunet_cur_size", None) != (net_w, net_h):
                    self._yunet.setInputSize((net_w, net_h))
                    self._yunet_cur_size = (net_w, net_h)

                _, faces = self._yunet.detect(detect_frame)
                if faces is not None and len(faces) > 0:
                    raw_face_detected = True
                    # Pick largest detected face
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    fx = float(best_face[0] * scale_to_orig_x)
                    fy = float(best_face[1] * scale_to_orig_y)
                    fw = float(best_face[2] * scale_to_orig_x)
                    fh = float(best_face[3] * scale_to_orig_y)
                    re_x = float(best_face[4] * scale_to_orig_x)
                    re_y = float(best_face[5] * scale_to_orig_y)
                    le_x = float(best_face[6] * scale_to_orig_x)
                    le_y = float(best_face[7] * scale_to_orig_y)
                    nose_x = float(best_face[8] * scale_to_orig_x)
                    nose_y = float(best_face[9] * scale_to_orig_y)
                    rm_x = float(best_face[10] * scale_to_orig_x)
                    rm_y = float(best_face[11] * scale_to_orig_y)
                    lm_x = float(best_face[12] * scale_to_orig_x)
                    lm_y = float(best_face[13] * scale_to_orig_y)

                    # Anatomical mid-face anchor matching MediaPipe: glabella (eye midpoint) + nose tip
                    glabella_y = 0.5 * (re_y + le_y)
                    instant_raw_y = float((0.55 * nose_y + 0.45 * glabella_y) / h)

                    # Smooth out single-shot detector sub-pixel jitter
                    if self._yunet_raw_y_smooth is None or self._is_first_sample:
                        self._yunet_raw_y_smooth = instant_raw_y
                    else:
                        noise_diff = abs(instant_raw_y - self._yunet_raw_y_smooth)
                        # Adaptive filter: heavy smoothing for micro-jitter, responsive for deliberate movement
                        filter_alpha = 0.28 if noise_diff < 0.015 else min(0.70, 0.28 + noise_diff * 3.5)
                        self._yunet_raw_y_smooth = filter_alpha * instant_raw_y + (1.0 - filter_alpha) * self._yunet_raw_y_smooth
                    raw_y = self._yunet_raw_y_smooth

                    # 1. Render futuristic facial contour wireframe (oval of cyan dots)
                    cx_p = int((fx + fw * 0.5) * scale_x)
                    cy_p = int((fy + fh * 0.52) * scale_y)
                    rx = fw * 0.46 * scale_x
                    ry = fh * 0.52 * scale_y
                    for deg in range(0, 360, 24):
                        rad = math.radians(deg)
                        px = int(cx_p + rx * math.cos(rad))
                        py = int(cy_p + ry * math.sin(rad))
                        cv2.circle(hud_frame, (px, py), 1, (0, 180, 255), -1)

                    # 2. Render eyes and mouth keypoints in magenta
                    for kx, ky in [(re_x, re_y), (le_x, le_y), (rm_x, rm_y), (lm_x, lm_y)]:
                        cv2.circle(hud_frame, (int(kx * scale_x), int(ky * scale_y)), 2, (255, 0, 180), -1)

                    # 3. Render glowing nose reticle / crosshair
                    nose_px = int(nose_x * scale_x)
                    nose_py = int(nose_y * scale_y)
                    cv2.circle(hud_frame, (nose_px, nose_py), 6, (0, 255, 200), 1, cv2.LINE_AA)
                    cv2.line(hud_frame, (nose_px - 10, nose_py), (nose_px + 10, nose_py), (0, 255, 200), 1)
                    cv2.line(hud_frame, (nose_px, nose_py - 10), (nose_px, nose_py + 10), (0, 255, 200), 1)

                    cv2.putText(
                        hud_frame,
                        "TRACKING: LOCKED",
                        (8, preview_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 180),
                        1,
                        cv2.LINE_AA,
                    )
            except Exception as exc:
                logger.debug("OpenCV YuNet process exception: %s", exc)

        # Fallback to OpenCV Haar Cascade ONLY if neither MediaPipe nor YuNet is available
        if not raw_face_detected and self._face_mesh is None and self._yunet is None and self._cascade is not None and not self._cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downscale only if frame width > 360 to keep speed high while preserving face detail
            if w > 360:
                scale_factor = 320.0 / float(w)
                detect_img = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
                inv_scale = 1.0 / scale_factor
            else:
                detect_img = gray
                inv_scale = 1.0

            det_h, det_w = detect_img.shape[:2]
            faces = []
            roi_offset_x = 0
            roi_offset_y = 0

            # 1. ROI Fast Path: If face was tracked recently, search local ROI first (3x-5x faster on Raspberry Pi)
            if self._cascade_box is not None and self._cascade_miss_count < 4:
                cb_x, cb_y, cb_w, cb_h = self._cascade_box
                d_x = cb_x / inv_scale
                d_y = cb_y / inv_scale
                d_w = cb_w / inv_scale
                d_h = cb_h / inv_scale

                pad_w = int(d_w * 0.45)
                pad_h = int(d_h * 0.45)
                rx1 = max(0, int(d_x - pad_w))
                ry1 = max(0, int(d_y - pad_h))
                rx2 = min(det_w, int(d_x + d_w + pad_w))
                ry2 = min(det_h, int(d_y + d_h + pad_h))

                roi_img = detect_img[ry1:ry2, rx1:rx2]
                if roi_img.shape[0] >= 32 and roi_img.shape[1] >= 32:
                    roi_eq = cv2.equalizeHist(roi_img)
                    roi_faces = self._cascade.detectMultiScale(
                        roi_eq,
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(28, 28),
                    )
                    if len(roi_faces) > 0:
                        faces = roi_faces
                        roi_offset_x = rx1
                        roi_offset_y = ry1

            # 2. Full Frame Scan: if ROI did not find a face, scan full frame to acquire / re-acquire
            if len(faces) == 0:
                detect_eq = cv2.equalizeHist(detect_img)
                faces = self._cascade.detectMultiScale(
                    detect_eq,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(36, 36),
                )
                roi_offset_x = 0
                roi_offset_y = 0

            if len(faces) > 0:
                raw_face_detected = True
                self._cascade_miss_count = 0
                r_fx, r_fy, r_fw, r_fh = max(faces, key=lambda b: b[2] * b[3])
                # Convert back to full frame coordinates
                det_fx = (r_fx + roi_offset_x) * inv_scale
                det_fy = (r_fy + roi_offset_y) * inv_scale
                det_fw = r_fw * inv_scale
                det_fh = r_fh * inv_scale

                # Temporal smoothing on bounding box: removes discrete pixel quantization jitter
                if self._cascade_box is None:
                    self._cascade_box = (det_fx, det_fy, det_fw, det_fh)
                else:
                    s_x, s_y, s_w, s_h = self._cascade_box
                    if abs(det_fy - s_y) < s_h * 0.8:
                        box_alpha = 0.35  # Smooth filter prevents jitter while maintaining fast response
                        self._cascade_box = (
                            s_x * (1.0 - box_alpha) + det_fx * box_alpha,
                            s_y * (1.0 - box_alpha) + det_fy * box_alpha,
                            s_w * (1.0 - box_alpha) + det_fw * box_alpha,
                            s_h * (1.0 - box_alpha) + det_fh * box_alpha,
                        )
                    else:
                        self._cascade_box = (det_fx, det_fy, det_fw, det_fh)

                orig_fx, orig_fy, orig_fw, orig_fh = self._cascade_box

                # Calibrated 0.52 offset matches mid-face landmark vertical level precisely
                center_y = orig_fy + orig_fh * 0.52
                raw_y = float(center_y / h)

                p_fx = int(orig_fx * scale_x)
                p_fy = int(orig_fy * scale_y)
                p_fw = int(orig_fw * scale_x)
                p_fh = int(orig_fh * scale_y)
                cv2.rectangle(hud_frame, (p_fx, p_fy), (p_fx + p_fw, p_fy + p_fh), (0, 240, 255), 1)
                cx = p_fx + p_fw // 2
                cy = p_fy + p_fh // 2
                cv2.circle(hud_frame, (cx, cy), 4, (0, 255, 200), -1)

                cv2.putText(
                    hud_frame,
                    "TRACKING: LOCKED",
                    (8, preview_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
            else:
                self._cascade_miss_count += 1

        if raw_face_detected:
            # Kinematic velocity estimation
            if self._last_raw_y_sample_time > 0:
                dt_sample = max(0.001, now - self._last_raw_y_sample_time)
                instant_vel = (raw_y - self._last_valid_raw_y) / dt_sample
                self._velocity_y = 0.5 * self._velocity_y + 0.5 * instant_vel
            self._last_raw_y_sample_time = now
            self._last_valid_raw_y = raw_y
            self._last_valid_detection_time = now
            effective_detected = True

            # Re-anchor baseline immediately if camera was switched or requested
            if self._recenter_needed:
                self.calibrate_baseline(raw_y)
                self._smoothed_y = 0.5
                self._is_first_sample = True
                self._recenter_needed = False
                logger.info("Auto-recalibrated neutral baseline to %.2f on new camera frame", raw_y)

            # Stable bounded dynamic window tracking:
            # Only drift baseline if player is consistently clamped at an edge for prolonged period (>20 frames),
            # preventing momentary head movements or noise spikes from displacing the player's zero position.
            span = max(0.01, (self._max_y_bound - self._min_y_bound))
            if raw_y > self._max_y_bound:
                self._clamped_at_max_count += 1
                if self._clamped_at_max_count > 20:
                    drift = (raw_y - self._max_y_bound) * 0.08
                    self._max_y_bound += drift
                    self._min_y_bound += drift
            else:
                self._clamped_at_max_count = 0

            if raw_y < self._min_y_bound:
                self._clamped_at_min_count += 1
                if self._clamped_at_min_count > 20:
                    drift = (raw_y - self._min_y_bound) * 0.08
                    self._min_y_bound += drift
                    self._max_y_bound += drift
            else:
                self._clamped_at_min_count = 0

            # Map raw_y into normalized linear [0.0, 1.0]
            linear_y = (raw_y - self._min_y_bound) / span
            linear_y = float(np.clip(linear_y, 0.0, 1.0))

            # Ergonomic progressive sensitivity curve:
            # Softens micro-movements around the neutral center for intuitive fine-control,
            # while scaling up smoothly for effortless reach towards the boundaries.
            u = 2.0 * (linear_y - 0.5)
            abs_u = abs(u)
            sign_u = 1.0 if u >= 0 else -1.0
            curved_u = sign_u * (0.65 * abs_u + 0.35 * (abs_u ** 1.35))
            mapped_y = float(np.clip(0.5 + 0.5 * curved_u, 0.0, 1.0))

            # Dynamic Adaptive Alpha:
            # Use base_alpha for subtle movements to eliminate micro-jitter,
            # and smoothly scale up to 0.65 during sudden head flicks without jerky over-correction.
            base_alpha = getattr(self.config, "ema_alpha", 0.28)
            if self._is_first_sample:
                self._smoothed_y = mapped_y
                self._is_first_sample = False
            else:
                diff = abs(mapped_y - self._smoothed_y)
                alpha_boost = min(1.0, (diff / 0.15) ** 1.3)
                effective_alpha = base_alpha + (0.45 - base_alpha) * alpha_boost
                self._smoothed_y = (effective_alpha * mapped_y) + ((1.0 - effective_alpha) * self._smoothed_y)

        else:
            # Grace period (0.45s): maintains smooth tracking during momentary blinks / motion blur
            if (now - self._last_valid_detection_time) < 0.45 and not self._is_first_sample:
                effective_detected = True
                # Hold last valid smoothed position steadily (avoids phantom velocity drift/jerks)
                cv2.putText(
                    hud_frame,
                    "TRACKING: HOLD",
                    (8, preview_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                effective_detected = False
                self._velocity_y = 0.0
                self._yunet_raw_y_smooth = None
                cv2.putText(
                    hud_frame,
                    "NO FACE DETECTED",
                    (8, preview_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 70, 255),
                    1,
                    cv2.LINE_AA,
                )

        hud_rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        return effective_detected, raw_y, self._smoothed_y, hud_rgb

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
        self._calibration_samples: List[float] = []

    def start(self) -> None:
        """Starts camera and the dedicated computer vision pipeline."""
        self.camera.start()
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._run_pipeline, daemon=True, name="VisionPipelineWorker")
        self._worker_thread.start()

    def begin_calibration(self) -> None:
        """Initiates head position calibration routine."""
        with self._lock:
            self._calibrating = True
            self._calibration_start_time = time.perf_counter()
            self._calibration_samples.clear()
            self._state.is_calibrated = False
            self._state.calibration_progress = 0.0

    def finish_calibration(self) -> None:
        """Computes calibrated baseline and finalizes calibration state."""
        with self._lock:
            self._finish_calibration()

    def recenter_baseline(self) -> None:
        """Forces tracker to re-anchor neutral position on next frame."""
        self.tracker.reset_for_new_camera()

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
                    self._state.preview_surface_buffer = hud_preview
                    if detected:
                        self._state.last_detected_time = now

                    # Handle calibration gathering
                    if self._calibrating:
                        if detected:
                            self._calibration_samples.append(raw_y)

                        elapsed = now - self._calibration_start_time
                        progress = min(1.0, elapsed / max(0.1, self.config.calibration_duration_sec))
                        self._state.calibration_progress = progress

                        if elapsed >= self.config.calibration_duration_sec:
                            self._finish_calibration()

    def _finish_calibration(self) -> None:
        """Computes calibrated baseline from accumulated samples."""
        self._calibrating = False
        if self._calibration_samples:
            median_y = float(np.median(self._calibration_samples))
            self.tracker.calibrate_baseline(median_y)
        self._state.is_calibrated = True
        logger.info("Calibration finished with %d samples", len(self._calibration_samples))

    def get_state(self) -> TrackingState:
        """Thread-safe snapshot copy of current tracking data."""
        with self._lock:
            return TrackingState(
                face_detected=self._state.face_detected,
                raw_y=self._state.raw_y,
                normalized_y=self._state.normalized_y,
                smoothed_y=self._state.smoothed_y,
                last_detected_time=self._state.last_detected_time,
                preview_surface_buffer=self._state.preview_surface_buffer,
                calibration_progress=self._state.calibration_progress,
                is_calibrated=self._state.is_calibrated,
            )

    def stop(self) -> None:
        """Stops the pipeline and cleans up camera resources."""
        self._is_running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self.camera.stop()
        self.tracker.close()
