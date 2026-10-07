"""MediaPipe Face Mesh tracker with exponential moving average filtering and adaptive baseline."""

import logging
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
        base_spread = getattr(config, "default_range_spread", 0.14)
        initial_spread = base_spread / sensitivity
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

        # Detectors
        self._face_mesh = None
        self._cascade = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes MediaPipe Face Mesh AND OpenCV Haar Cascade as complementary detectors."""
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                # Balanced confidence thresholds prevent drifting ROI and eliminate sudden snaps
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,  # Higher FPS on Raspberry Pi
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                logger.info("MediaPipe Face Mesh tracker initialized (fast-motion resilient)")
        except Exception as exc:
            logger.warning("MediaPipe initialization failed (%s).", exc)

        # Initialize OpenCV Cascade as active motion-blur backup if XML model is available on system
        try:
            cascade_dir = getattr(cv2.data, "haarcascades", "")
            cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._cascade = cascade
                    logger.info("OpenCV Haar Cascade initialized as motion-blur backup detector")
        except Exception as exc:
            logger.debug("OpenCV Cascade backup not available: %s", exc)

    def reset_for_new_camera(self) -> None:
        """Flags that a camera switch occurred and baseline must re-anchor to new camera frame."""
        self._recenter_needed = True
        self._is_first_sample = True
        self._clamped_at_min_count = 0
        self._clamped_at_max_count = 0
        logger.info("FaceMeshTracker flagged for baseline re-anchor on new camera feed")

    def calibrate_baseline(self, neutral_y: float, range_spread: Optional[float] = None) -> None:
        """Sets comfortable dynamic range around player's neutral head position based on sensitivity."""
        if range_spread is None:
            sensitivity = max(0.5, getattr(self.config, "head_sensitivity", 2.4))
            base_spread = getattr(self.config, "default_range_spread", 0.14)
            range_spread = base_spread / sensitivity
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
            # Set writeable=False to allow MediaPipe to pass frame by reference without memory copy
            rgb_frame.flags.writeable = False
            results = self._face_mesh.process(rgb_frame)
            rgb_frame.flags.writeable = True
            if results.multi_face_landmarks:
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

        # Fallback to OpenCV Haar Cascade if MediaPipe missed this frame (e.g. motion blur during sudden head flick)
        if not raw_face_detected and self._cascade is not None and not self._cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downscale 2x for fast execution without frame drops
            small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            faces = self._cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))
            if len(faces) > 0:
                raw_face_detected = True
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                # Calibrated 0.52 offset matches mid-face landmark vertical level precisely
                center_y = (fy * 2.0) + (fh * 2.0) * 0.52
                raw_y = float(center_y / h)

                p_fx = int(fx * 2.0 * scale_x)
                p_fy = int(fy * 2.0 * scale_y)
                p_fw = int(fw * 2.0 * scale_x)
                p_fh = int(fh * 2.0 * scale_y)
                cv2.rectangle(hud_frame, (p_fx, p_fy), (p_fx + p_fw, p_fy + p_fh), (0, 240, 255), 1)
                cx = p_fx + p_fw // 2
                cy = p_fy + p_fh // 2
                cv2.circle(hud_frame, (cx, cy), 4, (0, 255, 200), -1)

                cv2.putText(
                    hud_frame,
                    "CASCADE: LOCKED",
                    (8, preview_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 180),
                    1,
                    cv2.LINE_AA,
                )

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

            # Anti-deadband dynamic window tracking:
            # If the user moves beyond calibrated bounds during a sudden flick,
            # shift the active window so there is zero dead zone when reversing direction!
            span = max(0.01, (self._max_y_bound - self._min_y_bound))
            if raw_y > self._max_y_bound:
                self._max_y_bound = raw_y
                self._min_y_bound = self._max_y_bound - span
            elif raw_y < self._min_y_bound:
                self._min_y_bound = raw_y
                self._max_y_bound = self._min_y_bound + span

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
            # Use base_alpha for stillness/subtle movements to eliminate micro-jitter,
            # and smoothly scale up to 0.85 during sudden head flicks for instant response.
            base_alpha = getattr(self.config, "ema_alpha", 0.25)
            if self._is_first_sample:
                self._smoothed_y = mapped_y
                self._is_first_sample = False
            else:
                diff = abs(mapped_y - self._smoothed_y)
                alpha_boost = min(1.0, (diff / 0.12) ** 1.3)
                effective_alpha = base_alpha + (0.85 - base_alpha) * alpha_boost
                self._smoothed_y = (effective_alpha * mapped_y) + ((1.0 - effective_alpha) * self._smoothed_y)

        else:
            # Grace period (0.35s): maintains smooth tracking during momentary blinks / motion blur
            if (now - self._last_valid_detection_time) < 0.35 and not self._is_first_sample:
                effective_detected = True
                # Inertial coasting during momentary detector drop prevents freezing/stuck paddle
                if abs(self._velocity_y) > 0.05:
                    span = max(0.01, (self._max_y_bound - self._min_y_bound))
                    dt_grace = max(0.001, now - self._last_raw_y_sample_time)
                    delta_norm = (self._velocity_y * dt_grace) / span
                    self._smoothed_y = float(np.clip(self._smoothed_y + delta_norm * 0.35, 0.0, 1.0))
                    self._velocity_y *= 0.80

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
