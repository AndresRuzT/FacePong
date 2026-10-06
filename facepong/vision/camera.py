"""Threaded camera capture to minimize video buffer latency on embedded systems."""

import logging
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThreadedCamera:
    """Captures camera frames in a dedicated thread to ensure zero buffer lag."""

    def __init__(self, device_index: int = 0, width: int = 320, height: int = 240, target_fps: int = 30):
        self.device_index = device_index
        self.target_width = width
        self.target_height = height
        self.target_fps = target_fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._is_opened = False
        self._use_synthetic = False

    def start(self) -> bool:
        """Initializes the camera capture device and starts the worker thread."""
        try:
            # Attempt to open video capture
            self._capture = cv2.VideoCapture(self.device_index)
            if self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._is_opened = True
                logger.info("Camera %d opened successfully at %dx%d", 
                            self.device_index, self.target_width, self.target_height)
            else:
                logger.warning("Camera index %d could not be opened. Using synthetic fallback.", self.device_index)
                self._use_synthetic = True
                self._is_opened = False
        except Exception as exc:
            logger.warning("Error initializing camera: %s. Using synthetic fallback.", exc)
            self._use_synthetic = True
            self._is_opened = False

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        return self._is_opened

    def _capture_loop(self) -> None:
        """Continuous frame polling loop running in the background thread."""
        frame_interval = 1.0 / max(1, self.target_fps)
        while self._is_running:
            loop_start = time.perf_counter()

            if self._is_opened and self._capture is not None:
                success, frame = self._capture.read()
                if success and frame is not None:
                    with self._lock:
                        self._latest_frame = frame
                else:
                    time.sleep(0.01)
            elif self._use_synthetic:
                # Generate a subtle synthetic placeholder frame if camera is unavailable
                frame = self._generate_synthetic_frame()
                with self._lock:
                    self._latest_frame = frame

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates a mock video frame when no physical camera is connected."""
        frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        # Subtle dark grid
        frame[:, :] = (15, 18, 28)
        cv2.putText(
            frame,
            "CAMERA OFFLINE",
            (20, self.target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 100, 150),
            1,
            cv2.LINE_AA,
        )
        return frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the freshest captured frame in a thread-safe manner."""
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()
        return False, None

    @property
    def is_available(self) -> bool:
        """Indicates whether a physical camera is successfully providing frames."""
        return self._is_opened

    def stop(self) -> None:
        """Stops the worker thread and releases video resources."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._capture and self._capture.isOpened():
            self._capture.release()
            logger.info("Camera device %d released", self.device_index)
        self._is_opened = False
