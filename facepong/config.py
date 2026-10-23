"""Configuration parameters for FacePong."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class DeviceProfile(str, Enum):
    AUTO = "AUTO"
    TV = "TV"
    LAPTOP = "LAPTOP"
    TABLET = "TABLET"


@dataclass(frozen=True)
class DisplayConfig:
    width: int = 1280
    height: int = 720
    target_fps: int = 60
    title: str = "FacePong // AudacIA Cyber-Arena"
    fullscreen: bool = False
    windowed_maximized: bool = True


@dataclass(frozen=True)
class CameraConfig:
    device_index: Optional[int] = None
    capture_width: int = 320
    capture_height: int = 240
    target_fps: int = 30
    preview_width: int = 132
    preview_height: int = 70
    flip_horizontal: bool = True
    device_profile: DeviceProfile = DeviceProfile.AUTO
    # EMA smoothing alpha: base smoothing for idle/micro-jitter elimination
    ema_alpha: float = 0.28
    # Sensitivity multiplier: balanced for natural, elegant head nods
    head_sensitivity: float = 2.4
    # Calibration range spread around neutral head position
    default_range_spread: float = 0.170
    # Inactivity timeout (seconds) before resetting match to attract screen
    inactivity_timeout_sec: float = 5.0
    # Head position normalization defaults around neutral ~0.50
    default_min_y: float = 0.39
    default_max_y: float = 0.61
    # Calibration countdown in seconds
    calibration_duration_sec: float = 3.0
    # Time window (seconds) allowed in attract mode for a 2nd player to join before launching 1P vs AI
    solo_start_delay_sec: float = 15.0


@dataclass(frozen=True)
class PhysicsConfig:
    paddle_width: int = 14
    paddle_height: int = 96
    paddle_margin: int = 28
    paddle_speed_keyboard: float = 650.0  # px/sec
    paddle_max_speed: float = 3800.0       # px/sec fast, snappy response on abrupt head movements
    ball_radius: int = 8
    ball_serve_speed: float = 340.0       # px/sec readable, gentle launch speed on serve
    ball_initial_speed: float = 520.0     # px/sec full gameplay rally speed
    ball_max_speed: float = 1100.0        # px/sec
    ball_speed_step: float = 30.0         # px/sec added per paddle bounce
    max_bounce_angle_deg: float = 60.0
    winning_score: int = 5
    goal_pause_sec: float = 1.2


@dataclass(frozen=True)
class AIConfig:
    # Base movement speed of the AI paddle (responsive arcade speed)
    base_speed: float = 400.0
    # Dynamic speed boundaries based on score difference
    min_speed: float = 280.0
    max_speed: float = 540.0
    # Error margin in pixels introduced to the predicted target (scales with shot difficulty)
    max_error_offset: float = 95.0
    min_error_offset: float = 12.0
    # Prediction update interval (simulates human visual reaction latency)
    reaction_interval_sec: float = 0.16


@dataclass(frozen=True)
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 44100
    master_volume: float = 0.5


@dataclass(frozen=True)
class ColorPalette:
    background: Tuple[int, int, int] = (10, 12, 22)
    grid_lines: Tuple[int, int, int] = (22, 27, 44)
    divider_line: Tuple[int, int, int] = (40, 52, 80)
    
    # Neon Player (Cyan)
    player_primary: Tuple[int, int, int] = (0, 240, 255)
    player_glow: Tuple[int, int, int] = (0, 130, 180)
    
    # Neon AI (Hot Pink / Magenta)
    ai_primary: Tuple[int, int, int] = (255, 45, 120)
    ai_glow: Tuple[int, int, int] = (180, 20, 80)
    
    # Ball & Accents (Fluorescent Green / Electric Lime)
    ball_core: Tuple[int, int, int] = (225, 255, 120)
    ball_glow: Tuple[int, int, int] = (50, 255, 20)
    amber_accent: Tuple[int, int, int] = (255, 190, 40)
    
    # UI Elements
    text_primary: Tuple[int, int, int] = (235, 245, 255)
    text_muted: Tuple[int, int, int] = (110, 130, 160)
    danger: Tuple[int, int, int] = (255, 60, 60)
    success: Tuple[int, int, int] = (40, 255, 140)


@dataclass
class GameConfig:
    display: DisplayConfig = field(default_factory=DisplayConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    colors: ColorPalette = field(default_factory=ColorPalette)
    debug_mode: bool = False
