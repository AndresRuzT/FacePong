"""MediaPipe Face Mesh tracker with exponential moving average filtering."""

import logging
import threading
import time
from dataclasses import dataclass, field
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
    """Processes camera frames to track facial position and produce HUD previews."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Set to False for higher FPS on Raspberry Pi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Exponential Moving Average state
        self._smoothed_y: float = 0.5
        self._is_first_sample: bool = True

        # Calibration bounds
        self._min_y_bound: float = config.default_min_y
        self._max_y_bound: float = config.default_max_y
        self._calibration_samples: List[float] = []

    def calibrate_baseline(self, neutral_y: float, range_spread: float = 0.22) -> None:
        """Sets comfortable dynamic range around player's neutral head position."""
        self._min_y_bound = max(0.05, neutral_y - range_spread)
        self._max_y_bound = min(0.95, neutral_y + range_spread)
        logger.info("Calibrated head range: [%.2f, %.2f] around neutral %.2f",
                    self._min_y_bound, self._max_y_bound, neutral_y)

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, float, float, np.ndarray]:
        """
        Executes face mesh detection, extracts smoothed Y, and renders a cyber HUD frame.
        
        Returns:
            Tuple of (face_detected, raw_y, smoothed_y, hud_rgb_preview)
        """
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        face_detected = False
        raw_y = 0.5

        # Render preview HUD frame
        preview_h = self.config.preview_height
        preview_w = self.config.preview_width
        hud_frame = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
        # Apply dark cyberpunk tint
        hud_frame = cv2.addWeighted(hud_frame, 0.45, np.zeros_like(hud_frame), 0.55, 0)

        scale_x = preview_w / float(w)
        scale_y = preview_h / float(h)

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            # Extract nose tip
            nose_point = landmarks[NOSE_TIP_IDX]
            raw_y = float(nose_point.y)

            # Map raw_y according to calibrated bounds into [0.0, 1.0]
            mapped_y = (raw_y - self._min_y_bound) / max(0.01, (self._max_y_bound - self._min_y_bound))
            mapped_y = float(np.clip(mapped_y, 0.0, 1.0))

            # Apply Exponential Moving Average (EMA)
            if self._is_first_sample:
                self._smoothed_y = mapped_y
                self._is_first_sample = False
            else:
                alpha = self.config.ema_alpha
                self._smoothed_y = (alpha * mapped_y) + ((1.0 - alpha) * self._smoothed_y)

            # Render futuristic landmarks onto HUD preview
            nose_px = int(nose_point.x * preview_w)
            nose_py = int(nose_point.y * preview_h)

            # Draw outer contour points
            for idx in CONTOUR_KEYPOINTS:
                pt = landmarks[idx]
                px = int(pt.x * preview_w)
                py = int(pt.y * preview_h)
                cv2.circle(hud_frame, (px, py), 1, (0, 180, 255), -1)

            # Draw eyes and chin markers
            for idx in (LEFT_EYE_IDX, RIGHT_EYE_IDX, CHIN_IDX, FOREHEAD_IDX):
                pt = landmarks[idx]
                px = int(pt.x * preview_w)
                py = int(pt.y * preview_h)
                cv2.circle(hud_frame, (px, py), 2, (255, 0, 180), -1)

            # Draw animated cyber reticle on nose tip
            cv2.circle(hud_frame, (nose_px, nose_py), 6, (0, 255, 200), 1, cv2.LINE_AA)
            cv2.line(hud_frame, (nose_px - 10, nose_py), (nose_px + 10, nose_py), (0, 255, 200), 1)
            cv2.line(hud_frame, (nose_px, nose_py - 10), (nose_px, nose_py + 10), (0, 255, 200), 1)

            # Status label
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
            # No face detected
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

        # Convert HUD frame to RGB for Pygame
        hud_rgb = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
        return face_detected, raw_y, self._smoothed_y, hud_rgb

    def close(self) -> None:
        """Releases MediaPipe resources."""
        self._face_mesh.close()


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

    def _run_pipeline(self) -> None:
        """Continuous pipeline loop running at camera FPS."""
        frame_interval = 1.0 / max(1, self.config.target_fps)
        while self._is_running:
            loop_start = time.perf_counter()

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
