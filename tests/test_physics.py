"""Unit tests for FacePong game physics, entities, and particle system."""

import unittest
import pygame
from facepong.config import PhysicsConfig
from facepong.game.entities import Ball, Paddle, ParticleSystem


class TestPhysicsAndEntities(unittest.TestCase):
    def setUp(self):
        pygame.init()
        self.config = PhysicsConfig()
        self.screen_w = 1280
        self.screen_h = 720

    def test_paddle_clamping(self):
        paddle = Paddle(x=50, y=360, width=16, height=100, screen_height=self.screen_h)
        # Attempt to move far past top edge
        paddle.target_y = -500
        paddle.clamp_target()
        half_h = 100 / 2.0
        self.assertGreaterEqual(paddle.target_y, half_h + 10)

        # Attempt to move far past bottom edge
        paddle.target_y = 2000
        paddle.clamp_target()
        self.assertLessEqual(paddle.target_y, self.screen_h - half_h - 10)

    def test_paddle_reset_to_center(self):
        paddle = Paddle(x=50, y=100, width=16, height=100, screen_height=self.screen_h)
        paddle.reset_to_center(360.0)
        self.assertAlmostEqual(paddle.y, 360.0)
        self.assertAlmostEqual(paddle.target_y, 360.0)
        self.assertEqual(paddle.rect.centery, 360)

    def test_paddle_smooth_damp_interpolation(self):
        paddle = Paddle(x=50, y=100, width=16, height=100, screen_height=self.screen_h)
        paddle.target_y = 500.0
        # Run 10 frames of 60 FPS
        for _ in range(10):
            paddle.update(0.016)
        # Position should smoothly advance towards 500 without overshooting
        self.assertGreater(paddle.y, 100.0)
        self.assertLessEqual(paddle.y, 500.0)
        self.assertGreater(paddle.current_velocity, 0.0)

    def test_ball_wall_bounce(self):
        ball = Ball(self.screen_w, self.screen_h, self.config)
        ball.x = 640
        ball.y = 8  # Above top boundary
        ball.vy = -300

        wall_bounce, _ = ball.update(0.01)
        self.assertTrue(wall_bounce)
        self.assertGreater(ball.vy, 0)  # Rebounds downwards

    def test_ball_paddle_collision(self):
        paddle = Paddle(x=50, y=360, width=20, height=100, screen_height=self.screen_h)
        ball = Ball(self.screen_w, self.screen_h, self.config)
        ball.x = paddle.rect.right + 2
        ball.y = 360
        ball.vx = -400  # Moving left towards player paddle
        ball.vy = 0

        hit = ball.handle_paddle_collision(paddle, is_player=True)
        self.assertTrue(hit)
        self.assertGreater(ball.vx, 0)  # Rebounds to the right

    def test_ball_serve_characteristics(self):
        ball = Ball(self.screen_w, self.screen_h, self.config)
        ball.serve(direction_to_player=True)
        # Serve speed should match ball_serve_speed
        self.assertEqual(ball.speed, self.config.ball_serve_speed)
        self.assertTrue(ball.is_serve_in_flight)
        # Trajectory should be heading towards player (vx < 0) and almost horizontal
        self.assertLess(ball.vx, 0)
        self.assertLess(abs(ball.vy), abs(ball.vx) * 0.15)  # within ±8.5 degrees

        # When hitting paddle on serve, speed should ramp up to at least ball_initial_speed
        paddle = Paddle(x=50, y=360, width=20, height=100, screen_height=self.screen_h)
        ball.x = paddle.rect.right + 2
        ball.y = 360
        hit = ball.handle_paddle_collision(paddle, is_player=True)
        self.assertTrue(hit)
        self.assertFalse(ball.is_serve_in_flight)
        self.assertGreaterEqual(ball.speed, self.config.ball_initial_speed)

    def test_particle_system_lifecycle(self):
        ps = ParticleSystem(max_particles=50)
        ps.emit(100, 100, (255, 255, 255), count=20)
        self.assertEqual(len(ps.particles), 20)

        # Advance time significantly so particles expire
        ps.update(1.0)
        self.assertEqual(len(ps.particles), 0)

    def test_neon_ball_fluorescent_colors(self):
        from facepong.config import ColorPalette
        colors = ColorPalette()
        # Ensure ball glow is dominant fluorescent green (G significantly higher than B)
        self.assertGreater(colors.ball_glow[1], colors.ball_glow[2] * 2)
        # Ensure ball glow is distinct from player cyan
        self.assertNotEqual(colors.ball_glow, colors.player_primary)

    def test_screen_effects_state_and_lifecycle(self):
        from facepong.config import GameConfig
        from facepong.ui.renderer import NeonRenderer
        screen = pygame.Surface((1280, 720))
        cfg = GameConfig()
        renderer = NeonRenderer(screen, cfg)

        # Test player goal trigger
        renderer.trigger_goal_effect("PLAYER")
        self.assertEqual(renderer.active_screen_effect, "player_goal")
        self.assertGreater(renderer.effect_timer, 0.0)

        # Test AI goal trigger (activates shake)
        renderer.trigger_goal_effect("AI")
        self.assertEqual(renderer.active_screen_effect, "ai_goal")
        self.assertGreater(renderer.shake_timer, 0.0)

        # Run render_screen_effects step
        renderer.render_screen_effects(time_sec=1.0, dt=0.016)
        self.assertIsNotNone(renderer.shake_offset)

        # Test match end trigger
        renderer.trigger_match_end("PLAYER")
        self.assertEqual(renderer.match_result, "player_win")

        # Test reset
        renderer.reset_match_effects()
        self.assertIsNone(renderer.active_screen_effect)
        self.assertIsNone(renderer.match_result)
        self.assertEqual(renderer.shake_offset, (0, 0))


if __name__ == "__main__":
    unittest.main()

