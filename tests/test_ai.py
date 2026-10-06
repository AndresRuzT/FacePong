"""Unit tests for adaptive AI trajectory prediction and difficulty adjustment."""

import unittest
from facepong.config import AIConfig, PhysicsConfig
from facepong.game.ai import AdaptiveAIController
from facepong.game.entities import Ball, Paddle


class TestAdaptiveAI(unittest.TestCase):
    def setUp(self):
        self.ai_config = AIConfig()
        self.phys_config = PhysicsConfig()
        self.screen_w = 1280
        self.screen_h = 720
        self.ai = AdaptiveAIController(
            self.ai_config, self.phys_config, self.screen_w, self.screen_h
        )

    def test_difficulty_scales_with_player_lead(self):
        initial_diff = self.ai.difficulty_level

        # Player leading significantly: 4 - 0
        self.ai.adjust_difficulty(player_score=4, ai_score=0)
        self.assertGreater(self.ai.difficulty_level, initial_diff)

    def test_difficulty_scales_down_with_ai_lead(self):
        initial_diff = self.ai.difficulty_level

        # AI leading significantly: 0 - 4
        self.ai.adjust_difficulty(player_score=0, ai_score=4)
        self.assertLess(self.ai.difficulty_level, initial_diff)

    def test_trajectory_prediction_straight(self):
        ball = Ball(self.screen_w, self.screen_h, self.phys_config)
        ball.x = 200
        ball.y = 360
        ball.vx = 500  # Straight horizontal travel
        ball.vy = 0

        predicted_y = self.ai.predict_ball_trajectory(ball, ai_paddle_x=1200)
        self.assertAlmostEqual(predicted_y, 360, delta=2.0)

    def test_trajectory_prediction_with_bounces(self):
        ball = Ball(self.screen_w, self.screen_h, self.phys_config)
        ball.x = 200
        ball.y = 100
        ball.vx = 400
        ball.vy = 800  # Rapid vertical velocity requiring bounces

        predicted_y = self.ai.predict_ball_trajectory(ball, ai_paddle_x=1200)
        # Prediction must stay within screen bounds
        self.assertGreaterEqual(predicted_y, 10)
        self.assertLessEqual(predicted_y, self.screen_h - 10)


if __name__ == "__main__":
    unittest.main()
