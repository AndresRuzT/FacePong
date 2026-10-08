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
        # Welcoming initial baseline (0.35) makes the game fair and beatable from the start
        self.difficulty_level: float = 0.35
        self.current_speed: float = ai_config.base_speed

        # Prediction and reaction delay state
        self._target_y: float = screen_height / 2.0
        self._time_since_last_reaction: float = 0.0
        self._intentional_error: float = 0.0
        self._last_trajectory_predicted: float = screen_height / 2.0
        self._last_ball_vx: float = 0.0

    def adjust_difficulty(self, player_score: int, ai_score: int, rally_count: int = 0) -> None:
        """
        Dynamically adjusts difficulty based on score differential.
        Ensures matches remain accessible, fun, and winnable for the player.
        """
        score_diff = player_score - ai_score  # Positive: player winning; Negative: AI winning

        # Balanced baseline difficulty (0.42):
        # When player is behind, difficulty drops gracefully down to ~0.20,
        # still defending easy serves but vulnerable to angled/fast shots.
        # When player is ahead, AI steps up progressively.
        target_diff = 0.42 + (score_diff * 0.10) + (min(rally_count, 6) * 0.015)
        target_diff = max(0.20, min(0.75, target_diff))

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
        arena = getattr(ball, "arena_rect", None)
        top_boundary = (arena.top + ball.radius + 2) if arena else (10.0 + ball.radius)
        bottom_boundary = (arena.bottom - ball.radius - 2) if arena else (self.screen_height - 10.0 - ball.radius)
        center_y = arena.centery if arena else (self.screen_height / 2.0)

        if ball.vx <= 0:
            # Ball moving away: return to center position
            return center_y

        time_to_reach = (ai_paddle_x - ball.x) / max(1.0, ball.vx)
        if time_to_reach <= 0:
            return ball.y

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

        # Bank shot human uncertainty:
        # Each wall bounce introduces realistic estimation variance, rewarding player for trick angles
        if cycles >= 1:
            uncertainty = min(2, cycles) * 35.0 * (1.0 - self.difficulty_level)
            predicted_y += random.uniform(-uncertainty, uncertainty)

        return max(top_boundary, min(bottom_boundary, predicted_y))

    def update(self, dt: float, ball: Ball, ai_paddle: Paddle, player_score: int, ai_score: int, rally_count: int = 0) -> None:
        """Updates AI reaction calculation and moves the AI paddle."""
        self.adjust_difficulty(player_score, ai_score, rally_count)

        # If ball just reversed direction towards AI (player hit ball), reset reaction timer
        # to guarantee genuine human visual reaction latency to the player's shot
        if ball.vx > 0 and self._last_ball_vx <= 0:
            self._time_since_last_reaction = 0.0
        self._last_ball_vx = ball.vx

        self._time_since_last_reaction += dt
        reaction_threshold = self.config.reaction_interval_sec * (1.5 - self.difficulty_level * 0.7)

        # Update prediction periodically to simulate human latency
        if self._time_since_last_reaction >= reaction_threshold:
            self._time_since_last_reaction = 0.0

            ai_paddle_target_x = ai_paddle.rect.left - ball.radius
            raw_target_y = self.predict_ball_trajectory(ball, ai_paddle_target_x)
            self._last_trajectory_predicted = raw_target_y

            # Error margin scales dynamically with shot difficulty:
            # - Gentle/center serves (travel_dist < 60px, low speed, 0 bounces): error is minimal (<=14px),
            #   preventing the AI from clumsily giving away easy serve points.
            # - Hard drives, fast smashes, or trick bank shots: error increases up to max_error_offset (>=65px),
            #   which exceeds the paddle half-height (55px), allowing skilled player shots to score!
            dx_to_ai = max(1.0, ai_paddle_target_x - ball.x)
            time_to_reach = dx_to_ai / max(1.0, abs(ball.vx)) if ball.vx > 0 else 1.0
            total_y_dist = abs(ball.vy * time_to_reach)
            
            # Difficulty factor [0.0 = trivial straight serve, 1.0 = sharp difficult shot]
            shot_difficulty = min(1.0, (total_y_dist / 320.0) * 0.7 + (abs(ball.vy) / 380.0) * 0.3)
            
            # Base error bound based on current dynamic difficulty
            max_possible_error = max(self.config.min_error_offset, (1.0 - self.difficulty_level) * self.config.max_error_offset)
            
            # On easy straight serves, AI never deliberately mispredicts by more than 14px
            effective_error_magnitude = 14.0 + (max_possible_error - 14.0) * shot_difficulty
            effective_error_magnitude = max(self.config.min_error_offset, effective_error_magnitude)
            
            self._intentional_error = random.uniform(-effective_error_magnitude, effective_error_magnitude)
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
