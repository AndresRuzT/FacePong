#!/usr/bin/env python3
"""FacePong - Interactive Computer Vision Pong for Raspberry Pi.

Research and Exhibition Project developed for AudacIA - Centro de Investigación
en Robótica e Inteligencia Artificial, Universidad Simón Bolívar.
"""

import argparse
import logging
import sys

from facepong.config import (
    AudioConfig,
    CameraConfig,
    DisplayConfig,
    GameConfig,
    PhysicsConfig,
)
from facepong.engine import GameEngine


def parse_arguments() -> argparse.Namespace:
    """Parses command-line configuration arguments."""
    parser = argparse.ArgumentParser(
        description="FacePong: AI-powered Pong controlled by facial gestures on Raspberry Pi."
    )
    parser.add_argument("--width", type=int, default=1280, help="Display resolution width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Display resolution height (default: 720)")
    parser.add_argument("--fullscreen", action="store_true", help="Launch in fullscreen mode for kiosks")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--cam-width", type=int, default=320, help="Camera capture width (default: 320)")
    parser.add_argument("--cam-height", type=int, default=240, help="Camera capture height (default: 240)")
    parser.add_argument("--alpha", type=float, default=0.22, help="EMA smoothing factor for head tracking (default: 0.22)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Player absence timeout in seconds (default: 5.0)")
    parser.add_argument("--win-score", type=int, default=5, help="Points needed to win match (default: 5)")
    parser.add_argument("--no-sound", action="store_true", help="Disable procedural audio synthesis")
    parser.add_argument("--debug", action="store_true", help="Enable debug telemetry overlay")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def main() -> int:
    """Application entry point."""
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = GameConfig(
        display=DisplayConfig(
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
        ),
        camera=CameraConfig(
            device_index=args.camera,
            capture_width=args.cam_width,
            capture_height=args.cam_height,
            ema_alpha=args.alpha,
            inactivity_timeout_sec=args.timeout,
        ),
        physics=PhysicsConfig(
            winning_score=args.win_score,
        ),
        audio=AudioConfig(
            enabled=not args.no_sound,
        ),
        debug_mode=args.debug,
    )

    try:
        engine = GameEngine(config)
        engine.start()
        return 0
    except Exception as exc:
        logging.critical("Fatal error executing FacePong: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
