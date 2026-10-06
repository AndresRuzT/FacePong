"""Configuration parameters for FacePong."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


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
    capture_width: int = 640
    capture_height: int = 480
    target_fps: int = 30
    preview_width: int = 192
    preview_height: int = 144
    flip_horizontal: bool = True
    # EMA smoothing alpha: base for steady posture, dynamically scaled on fast motion
    ema_alpha: float = 0.25
    fast_alpha: float = 0.65
    # Sensitivity multiplier: higher = less head motion needed to move paddle across court
    head_sensitivity: float = 2.4
    # Calibration range spread around neutral head Y (smaller = higher sensitivity)
    default_range_spread: float = 0.14
    # Inactivity timeout (seconds) before resetting match to attract screen
    inactivity_timeout_sec: float = 5.0
    # Head range normalization defaults (fraction of frame height)
    default_min_y: float = 0.43
    default_max_y: float = 0.57
    # Calibration countdown in seconds
    calibration_duration_sec: float = 3.0


@dataclass(frozen=True)
class PhysicsConfig:
    paddle_width: int = 16
    paddle_height: int = 110
    paddle_margin: int = 35
    paddle_speed_keyboard: float = 650.0  # px/sec
    paddle_max_speed: float = 1250.0       # px/sec max speed to eliminate abrupt jumps
    ball_radius: int = 8
    ball_initial_speed: float = 520.0     # px/sec
    ball_max_speed: float = 1100.0        # px/sec
    ball_speed_step: float = 30.0         # px/sec added per paddle bounce
    max_bounce_angle_deg: float = 60.0
    winning_score: int = 5
    goal_pause_sec: float = 1.2


@dataclass(frozen=True)
class AIConfig:
    # Base movement speed of the AI paddle
    base_speed: float = 480.0
    # Dynamic speed boundaries based on score difference
    min_speed: float = 320.0
    max_speed: float = 780.0
    # Error margin in pixels introduced to the predicted target
    max_error_offset: float = 90.0
    min_error_offset: float = 10.0
    # Prediction update interval (simulates visual reaction latency)
    reaction_interval_sec: float = 0.12


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
    
    # Ball & Accents
    ball_core: Tuple[int, int, int] = (255, 255, 255)
    ball_glow: Tuple[int, int, int] = (0, 255, 180)
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
