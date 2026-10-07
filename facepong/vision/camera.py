"""Threaded video capture module with dynamic hot-plugging and auto-recovery."""

import glob
import logging
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def open_video_capture(device_index: int) -> Optional[cv2.VideoCapture]:
    """
    Opens a VideoCapture device using V4L2 on Linux to avoid FFMPEG index limits,
    falling back to default backends if needed.
    """
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sys.platform.startswith("linux") else [cv2.CAP_ANY]
    for backend in backends:
        try:
            cap = cv2.VideoCapture(device_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        except Exception:
            pass
    return None


def probe_available_cameras(exclude_index: Optional[int] = None) -> List[Tuple[int, int, str, bool]]:
    """
    Probes system video capture devices and evaluates suitability.
    
    Returns:
        List of (score, device_index, description, is_external_usb)
    """
    # Mute OpenCV internal V4L2 probe warnings
    if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    candidates: List[Tuple[int, int, str, bool]] = []

    # 1. Linux / Raspberry Pi: Query sysfs video4linux
    v4l_devices = sorted(glob.glob("/sys/class/video4linux/video*"))
    if v4l_devices:
        for dev_path in v4l_devices:
            try:
                base = os.path.basename(dev_path)
                idx = int(base.replace("video", ""))

                if exclude_index is not None and idx == exclude_index:
                    continue

                name_file = os.path.join(dev_path, "name")
                name = "Video Device"
                if os.path.exists(name_file):
                    with open(name_file, "r", errors="ignore") as f:
                        name = f.read().strip()

                device_link = ""
                try:
                    device_link = os.path.realpath(os.path.join(dev_path, "device"))
                except Exception:
                    pass

                is_internal = any(
                    k in name.lower()
                    for k in ("integrated", "internal", "built-in", "builtin", "front camera")
                )
                is_usb = ("usb" in device_link.lower()) and not is_internal

                # Verify actual frame acquisition capability
                cap = open_video_capture(idx)
                can_read = False
                if cap is not None:
                    ret, frame = cap.read()
                    can_read = bool(ret and frame is not None)
                    cap.release()

                if not can_read:
                    continue

                if is_usb:
                    score = 100
                    desc = f"External USB Camera: {name}"
                    candidates.append((score, idx, desc, True))
                elif is_internal:
                    score = 50
                    desc = f"Integrated Webcam: {name}"
                    candidates.append((score, idx, desc, False))
                else:
                    score = 60
                    desc = f"Capture Device: {name}"
                    candidates.append((score, idx, desc, False))
            except Exception as exc:
                logger.debug("Failed examining %s: %s", dev_path, exc)
                continue

    # 2. Fallback probe across indices 0..4 if no sysfs candidates found
    if not candidates:
        for idx in range(4):
            if exclude_index is not None and idx == exclude_index:
                continue
            cap = open_video_capture(idx)
            if cap is not None:
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    candidates.append((40, idx, f"Camera index {idx}", False))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates


def find_preferred_camera_device(requested_index: Optional[int] = None) -> Tuple[int, str]:
    """
    Identifies and selects the optimal video capture device.
    Prioritizes external USB webcams over integrated internal cameras.

    Returns:
        Tuple of (camera_index, device_description)
    """
    if requested_index is not None and requested_index >= 0:
        return requested_index, f"User-specified camera (index {requested_index})"

    candidates = probe_available_cameras()
    if candidates:
        best_score, best_idx, best_desc, _ = candidates[0]
        logger.info(
            "Auto-detected cameras: %s. Selected: %s (index %d)",
            [f"{c[2]} [score {c[0]}]" for c in candidates],
            best_desc,
            best_idx,
        )
        return best_idx, best_desc

    logger.warning("No functional video devices detected. Defaulting to index 0.")
    return 0, "Default video device (index 0)"


class ThreadedCamera:
    """
    Captures camera frames in a dedicated thread with zero buffer lag,
    fault tolerance, disconnection failover, and hot-plug recovery.
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        width: int = 320,
        height: int = 240,
        target_fps: int = 30,
    ):
        self.device_index, self.device_description = find_preferred_camera_device(device_index)
        self.is_external_usb = (
            "external" in self.device_description.lower()
            or ("usb" in self.device_description.lower() and "integrated" not in self.device_description.lower())
        )
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

        # Watchdog, hotplug and switch detection state
        self.device_change_counter = 0
        self._consecutive_failures = 0
        self._last_reconnect_attempt = 0.0
        self._last_usb_check = 0.0
        # Instant frame notification event for zero-latency pipeline wakeups
        self.frame_ready_event = threading.Event()

    def _open_device(self, idx: int) -> bool:
        """Attempts to open and configure a specific video device index."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        for attempt in range(2):
            cap = open_video_capture(idx)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._capture = cap
                self._is_opened = True
                self._use_synthetic = False
                self._consecutive_failures = 0
                self.device_change_counter += 1
                return True
            time.sleep(0.15)
        return False

    def start(self) -> bool:
        """Initializes the camera capture device and starts the worker thread."""
        logger.info("Initializing camera capture: %s (index %d)", self.device_description, self.device_index)

        opened = self._open_device(self.device_index)
        if not opened:
            logger.warning("Preferred camera %d failed to open. Probing available fallbacks...", self.device_index)
            candidates = probe_available_cameras(exclude_index=self.device_index)
            for _, alt_idx, alt_desc, alt_is_usb in candidates:
                if self._open_device(alt_idx):
                    self.device_index = alt_idx
                    self.device_description = alt_desc
                    self.is_external_usb = alt_is_usb
                    opened = True
                    logger.info("Switched to fallback camera: %s (index %d)", alt_desc, alt_idx)
                    break

        if not opened:
            logger.warning("No physical camera could be opened. Initializing synthetic video stream.")
            self._use_synthetic = True
            self._is_opened = False

        self._is_running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="CameraWorker")
        self._thread.start()
        return self._is_opened

    def _handle_disconnect(self) -> None:
        """Handles physical camera disconnect by releasing handle and probing for alternatives."""
        logger.warning(
            "Camera %d disconnected or read failure threshold exceeded. Releasing handle...",
            self.device_index,
        )
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

        self._is_opened = False
        self._use_synthetic = True
        self._consecutive_failures = 0

        # Attempt immediate failover to another available camera (e.g. integrated webcam)
        self._attempt_reconnect()

    def _attempt_reconnect(self) -> bool:
        """Searches for any active camera and attaches to it."""
        idx, desc = find_preferred_camera_device()
        if self._open_device(idx):
            self.device_index = idx
            self.device_description = desc
            self.is_external_usb = (
                "external" in desc.lower()
                or ("usb" in desc.lower() and "integrated" not in desc.lower())
            )
            logger.info("Camera successfully restored / switched to: %s (index %d)", desc, idx)
            return True
        return False

    def _check_for_usb_hotplug(self) -> None:
        """Checks if a preferred external USB camera was connected while using fallback."""
        if self.is_external_usb:
            return

        candidates = probe_available_cameras(exclude_index=self.device_index)
        usb_candidates = [c for c in candidates if c[0] >= 100]
        if usb_candidates:
            best_score, best_idx, best_desc, _ = usb_candidates[0]
            logger.info(
                "Hot-plugged external USB camera detected: %s (index %d). Upgrading from fallback %s...",
                best_desc,
                best_idx,
                self.device_description,
            )
            if self._open_device(best_idx):
                self.device_index = best_idx
                self.device_description = best_desc
                self.is_external_usb = True
                logger.info("Successfully switched to hot-plugged external USB camera: %s", best_desc)
            else:
                # Reopen previous fallback camera if new device failed to initialize
                self._open_device(self.device_index)

    def _capture_loop(self) -> None:
        """Continuous frame polling loop running in the background thread."""
        frame_interval = 1.0 / max(1, self.target_fps)
        while self._is_running:
            loop_start = time.perf_counter()
            now = loop_start

            if self._is_opened and self._capture is not None:
                success, frame = self._capture.read()
                if success and frame is not None:
                    self._consecutive_failures = 0
                    with self._lock:
                        self._latest_frame = frame
                    self.frame_ready_event.set()

                    # Periodically check if preferred USB camera was plugged in
                    if not self.is_external_usb and (now - self._last_usb_check > 2.0):
                        self._last_usb_check = now
                        self._check_for_usb_hotplug()
                else:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 6:
                        self._handle_disconnect()
            else:
                # Offline: generate synthetic placeholder
                synth = self._generate_synthetic_frame()
                with self._lock:
                    self._latest_frame = synth
                self.frame_ready_event.set()

                # Attempt reconnect every 1.5 seconds
                if now - self._last_reconnect_attempt > 1.5:
                    self._last_reconnect_attempt = now
                    self._attempt_reconnect()

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates a mock video frame when physical camera is unavailable."""
        frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        frame[:, :] = (15, 18, 28)
        cv2.putText(
            frame,
            "CAMERA RECONNECTING...",
            (16, self.target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 100, 150),
            1,
            cv2.LINE_AA,
        )
        return frame

    def read(self, copy: bool = False) -> Tuple[bool, Optional[np.ndarray]]:
        """Returns the freshest captured frame in a thread-safe manner."""
        with self._lock:
            if self._latest_frame is not None:
                return True, (self._latest_frame.copy() if copy else self._latest_frame)
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
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None
            logger.info("Camera device %d released", self.device_index)
        self._is_opened = False
