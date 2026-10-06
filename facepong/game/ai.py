"""Adaptive AI controller featuring trajectory raycasting and dynamic difficulty adjustment."""

import math
import random
from typing import Tuple
from facepong.config import AIConfig, PhysicsConfig
from facepong.game.entities import Ball, Paddle


class AdaptiveAIController:
    """Controls the opponent paddle with trajectory prediction and dynamic difficulty adjustment."""

    def __init__(self, ai_config: AIConfig, physics_config: PhysicsConfig, screen_width: int, screen_height: int):
        self.config = ai_config
        self.physics_config = physics_config
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Dynamic difficulty state [0.0 = easiest, 1.0 = unbeatable master]
        self.difficulty_level: float = 0.5
        self.current_speed: float = ai_config.base_speed

        # Prediction and reaction delay state
        self._target_y: float = screen_height / 2.0
        self._time_since_last_reaction: float = 0.0
        self._intentional_error: float = 0.0
        self._last_trajectory_predicted: float = screen_height / 2.0

    def adjust_difficulty(self, player_score: int, ai_score: int, rally_count: int = 0) -> None:
        """
        Dynamically adjusts difficulty based on score differential.
        Ensures matches remain competitive and rubber-banded.
        """
        score_diff = player_score - ai_score  # Positive: player winning; Negative: AI winning

        # Map score differential (-3 to +3) into difficulty range [0.15, 0.95]
        # Base balanced difficulty is 0.50
        target_diff = 0.50 + (score_diff * 0.12) + (min(rally_count, 10) * 0.01)
        target_diff = max(0.15, min(0.95, target_diff))

        # Smooth difficulty transition
        self.difficulty_level = 0.8 * self.difficulty_level + 0.2 * target_diff

        # Scale speed according to difficulty
        speed_span = self.config.max_speed - self.config.min_speed
        self.current_speed = self.config.min_speed + (speed_span * self.difficulty_level)

    def predict_ball_trajectory(self, ball: Ball, ai_paddle_x: float) -> float:
        """
        Calculates geometric trajectory intersection at the AI paddle plane,
        simulating elastic bounces on top and bottom boundaries.
        """
        if ball.vx <= 0:
            # Ball moving away: return to center position
            return self.screen_height / 2.0

        time_to_reach = (ai_paddle_x - ball.x) / max(1.0, ball.vx)
        if time_to_reach <= 0:
            return ball.y

        # Bound range for bouncing
        top_boundary = 10.0 + ball.radius
        bottom_boundary = self.screen_height - 10.0 - ball.radius
        field_height = bottom_boundary - top_boundary

        if field_height <= 0:
            return ball.y

        total_vertical_distance = ball.y + (ball.vy * time_to_reach)
        relative_y = total_vertical_distance - top_boundary

        # Bounce cycle folding
        cycles = int(relative_y // field_height)
        remainder = relative_y % field_height

        if cycles % 2 == 0:
            # Traveling downwards on even cycle
            predicted_y = top_boundary + remainder
        else:
            # Traveling upwards on odd bounce cycle
            predicted_y = bottom_boundary - remainder

        return max(top_boundary, min(bottom_boundary, predicted_y))

    def update(self, dt: float, ball: Ball, ai_paddle: Paddle, player_score: int, ai_score: int, rally_count: int = 0) -> None:
        """Updates AI reaction calculation and moves the AI paddle."""
        self.adjust_difficulty(player_score, ai_score, rally_count)

        self._time_since_last_reaction += dt
        reaction_threshold = self.config.reaction_interval_sec * (1.6 - self.difficulty_level * 0.8)

        # Update prediction periodically to simulate human latency
        if self._time_since_last_reaction >= reaction_threshold:
            self._time_since_last_reaction = 0.0

            ai_paddle_target_x = ai_paddle.rect.left - ball.radius
            raw_target_y = self.predict_ball_trajectory(ball, ai_paddle_target_x)
            self._last_trajectory_predicted = raw_target_y

            # Introduce intentional human-like error inversely proportional to difficulty
            error_magnitude = (1.0 - self.difficulty_level) * self.config.max_error_offset
            # If difficulty is high, error is negligible
            if self.difficulty_level > 0.8:
                error_magnitude = self.config.min_error_offset

            self._intentional_error = random.uniform(-error_magnitude, error_magnitude)
            self._target_y = raw_target_y + self._intentional_error

        # Steer paddle towards target coordinate with clamped speed
        diff = self._target_y - ai_paddle.y
        max_step = self.current_speed * dt

        if abs(diff) <= max_step:
            ai_paddle.target_y = self._target_y
        else:
            ai_paddle.target_y += math.copysign(max_step, diff)

        ai_paddle.clamp_target()
        ai_paddle.update(dt)

    @property
    def predicted_target_y(self) -> float:
        """Returns the raw predicted intersection coordinate (useful for debug HUD)."""
        return self._last_trajectory_predicted
