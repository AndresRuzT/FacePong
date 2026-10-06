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

        # Detectors
        self._face_mesh = None
        self._cascade = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initializes MediaPipe Face Mesh or falls back to OpenCV Haar Cascade."""
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                # Use 0.45 detection confidence for superior distance tracking
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,  # Higher FPS on Raspberry Pi
                    min_detection_confidence=0.45,
                    min_tracking_confidence=0.45,
                )
                logger.info("MediaPipe Face Mesh tracker initialized (confidence 0.45)")
                return
        except Exception as exc:
            logger.warning("MediaPipe initialization failed (%s). Falling back to OpenCV Cascade.", exc)

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV Haar Cascade face tracker initialized as fallback")
        except Exception as exc:
            logger.error("Failed to initialize OpenCV Cascade fallback: %s", exc)

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

        # Render preview HUD frame
        preview_h = self.config.preview_height
        preview_w = self.config.preview_width
        hud_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        hud_frame = cv2.addWeighted(hud_frame, 0.45, np.zeros_like(hud_frame), 0.55, 0)

        scale_x = preview_w / float(w)
        scale_y = preview_h / float(h)

        if self._face_mesh is not None:
            results = self._face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                raw_face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark
                nose_point = landmarks[NOSE_TIP_IDX]
                raw_y = float(nose_point.y)

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

        elif self._cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30))
            if len(faces) > 0:
                raw_face_detected = True
                fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
                center_y = fy + fh / 2.0
                raw_y = float(center_y / h)

                p_fx = int(fx * scale_x)
                p_fy = int(fy * scale_y)
                p_fw = int(fw * scale_x)
                p_fh = int(fh * scale_y)
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

            # Map raw_y into [0.0, 1.0]
            span = max(0.01, (self._max_y_bound - self._min_y_bound))
            mapped_y = (raw_y - self._min_y_bound) / span
            mapped_y = float(np.clip(mapped_y, 0.0, 1.0))

            # Auto-drift adaptation: if player shifted posture and stays pinned at boundary for 15 frames (~0.5s)
            if mapped_y <= 0.01:
                self._clamped_at_min_count += 1
                self._clamped_at_max_count = 0
                if self._clamped_at_min_count >= 15:
                    current_center = (self._min_y_bound + self._max_y_bound) / 2.0
                    self.calibrate_baseline(current_center - 0.02)
                    self._clamped_at_min_count = 0
            elif mapped_y >= 0.99:
                self._clamped_at_max_count += 1
                self._clamped_at_min_count = 0
                if self._clamped_at_max_count >= 15:
                    current_center = (self._min_y_bound + self._max_y_bound) / 2.0
                    self.calibrate_baseline(current_center + 0.02)
                    self._clamped_at_max_count = 0
            else:
                self._clamped_at_min_count = 0
                self._clamped_at_max_count = 0

            # Exponential Moving Average (EMA)
            if self._is_first_sample:
                self._smoothed_y = mapped_y
                self._is_first_sample = False
            else:
                alpha = self.config.ema_alpha
                self._smoothed_y = (alpha * mapped_y) + ((1.0 - alpha) * self._smoothed_y)

        else:
            # Grace period (0.35s): prevents paddle stutter during blinks or quick head tilts at distance
            if (now - self._last_valid_detection_time) < 0.35 and not self._is_first_sample:
                effective_detected = True
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
        """Continuous pipeline loop running at camera FPS."""
        frame_interval = 1.0 / max(1, self.config.target_fps)
        while self._is_running:
            loop_start = time.perf_counter()

            # Detect hardware camera switch (e.g. unplug USB -> fallback to integrated)
            dev_counter = getattr(self.camera, "device_change_counter", 0)
            if dev_counter != self._last_camera_device_counter:
                self._last_camera_device_counter = dev_counter
                self.tracker.reset_for_new_camera()
                logger.info("VisionPipeline detected camera device change (counter %d). Resetting tracker.", dev_counter)

            has_frame, frame = self.camera.read()
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

            elapsed_loop = time.perf_counter() - loop_start
            sleep_duration = max(0.001, frame_interval - elapsed_loop)
            time.sleep(sleep_duration)

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
