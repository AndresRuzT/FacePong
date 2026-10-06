"""Main game engine coordinating the vision pipeline, game loop, and exhibition states."""

import logging
import math
import sys
import time
import pygame

from facepong.config import GameConfig
from facepong.game.ai import AdaptiveAIController
from facepong.game.audio import SoundManager
from facepong.game.entities import Ball, Paddle, ParticleSystem
from facepong.game.state import ExhibitionState, GameStateManager
from facepong.ui.renderer import NeonRenderer
from facepong.ui.screens import ScreenManager
from facepong.vision.tracker import TrackingState, VisionPipeline

logger = logging.getLogger(__name__)


class GameEngine:
    """Master controller managing game lifecycle, event loops, and subsystems."""

    def __init__(self, config: GameConfig):
        self.config = config
        self._is_running = False

        # Initialize Pygame display
        pygame.init()
        display_flags = pygame.DOUBLEBUF
        if config.display.fullscreen:
            display_flags |= pygame.FULLSCREEN

        self.screen = pygame.display.set_mode(
            (config.display.width, config.display.height),
            display_flags,
        )
        pygame.display.set_caption(config.display.title)
        self.clock = pygame.time.Clock()

        # Core Subsystems
        self.vision = VisionPipeline(config.camera)
        self.state_mgr = GameStateManager(
            winning_score=config.physics.winning_score,
            inactivity_timeout_sec=config.camera.inactivity_timeout_sec,
        )
        self.ai = AdaptiveAIController(
            config.ai,
            config.physics,
            config.display.width,
            config.display.height,
        )
        self.sound = SoundManager(config.audio)
        self.renderer = NeonRenderer(self.screen, config)
        self.screens = ScreenManager(self.renderer, config)
        self.particles = ParticleSystem(max_particles=150)

        # Game Entities
        screen_w = config.display.width
        screen_h = config.display.height
        paddle_w = config.physics.paddle_width
        paddle_h = config.physics.paddle_height
        margin = config.physics.paddle_margin

        self.player_paddle = Paddle(
            x=margin + paddle_w / 2.0,
            y=screen_h / 2.0,
            width=paddle_w,
            height=paddle_h,
            screen_height=screen_h,
        )
        self.ai_paddle = Paddle(
            x=screen_w - margin - paddle_w / 2.0,
            y=screen_h / 2.0,
            width=paddle_w,
            height=paddle_h,
            screen_height=screen_h,
        )
        self.ball = Ball(screen_w, screen_h, config.physics)

        # Keyboard override tracker
        self._keyboard_active = False

    def start(self) -> None:
        """Starts background threads and executes the main loop."""
        self.vision.start()
        self._is_running = True
        logger.info("FacePong Engine initialized and running")

        last_time = time.perf_counter()

        try:
            while self._is_running:
                current_time = time.perf_counter()
                dt = min(0.05, current_time - last_time)  # Clamp dt to prevent physics tunneling
                last_time = current_time

                self._handle_events(dt)
                self._update(dt)
                self._render()

                self.clock.tick(self.config.display.target_fps)
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Exiting cleanly.")
        finally:
            self._shutdown()

    def _handle_events(self, dt: float) -> None:
        """Processes OS events, window controls, and keyboard inputs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._is_running = False
                    return
                elif event.key == pygame.K_d:
                    self.config.debug_mode = not self.config.debug_mode
                elif event.key == pygame.K_SPACE:
                    if self.state_mgr.current_state == ExhibitionState.ATTRACT:
                        self.state_mgr.change_state(ExhibitionState.CALIBRATING)
                        self.vision.begin_calibration()
                    elif self.state_mgr.current_state == ExhibitionState.GAME_OVER:
                        self.state_mgr.reset_to_attract()

        # Keyboard paddle fallback (W/S or UP/DOWN)
        keys = pygame.key.get_pressed()
        move_dir = 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_dir -= 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_dir += 1.0

        if move_dir != 0.0:
            self._keyboard_active = True
            self.player_paddle.move_keyboard(move_dir, dt)

    def _update(self, dt: float) -> None:
        """Updates game state machine, physics, and entities."""
        tracking_state = self.vision.get_state()
        state = self.state_mgr.current_state

        # Update presence watchdog
        timed_out = self.state_mgr.update_watchdog(
            tracking_state.face_detected,
            tracking_state.last_detected_time,
            dt,
        )
        if timed_out:
            logger.info("Match canceled due to player inactivity. Returning to attract mode.")

        # Update player paddle from face tracking if not actively using keyboard
        if tracking_state.face_detected and not self._keyboard_active:
            self.player_paddle.set_target_normalized_y(tracking_state.smoothed_y)

        # Update particle effects
        self.particles.update(dt)

        # State Machine Transitions
        if state == ExhibitionState.ATTRACT:
            self._update_attract_mode(dt)

        elif state == ExhibitionState.CALIBRATING:
            self._update_calibration_mode(dt, tracking_state)

        elif state == ExhibitionState.PLAYING:
            self._update_playing_mode(dt)

        elif state == ExhibitionState.POINT_SCORED:
            self._update_point_scored_mode(dt)

        elif state == ExhibitionState.GAME_OVER:
            # Subtle slow drift of paddles during game over
            self.player_paddle.update(dt)
            self.ai_paddle.update(dt)

    def _update_attract_mode(self, dt: float) -> None:
        """Simulates autonomous gameplay demo during idle kiosk state."""
        # AI vs AI mini demo for visual attraction
        self.ball.update(dt)

        # Bounce off screen left and right in demo mode
        if self.ball.x <= self.config.physics.ball_radius + 15:
            self.ball.vx = abs(self.ball.vx)
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.player_primary, count=8)
        elif self.ball.x >= self.config.display.width - self.config.physics.ball_radius - 15:
            self.ball.vx = -abs(self.ball.vx)
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.ai_primary, count=8)

        # Paddles track ball center loosely
        self.player_paddle.target_y = self.ball.y
        self.player_paddle.update(dt, smooth_factor=8.0)
        self.ai_paddle.target_y = self.ball.y
        self.ai_paddle.update(dt, smooth_factor=8.0)

    def _update_calibration_mode(self, dt: float, tracking_state: TrackingState) -> None:
        """Executes calibration countdown before match start."""
        self.player_paddle.update(dt)
        self.ai_paddle.update(dt)

        if tracking_state.is_calibrated or tracking_state.calibration_progress >= 1.0:
            self.state_mgr.start_new_match()
            self.ball.serve(direction_to_player=True)
            self.sound.play("start")
            self.state_mgr.change_state(ExhibitionState.PLAYING)
            self._keyboard_active = False

    def _update_playing_mode(self, dt: float) -> None:
        """Advances active competitive gameplay physics and collisions."""
        self.player_paddle.update(dt)
        self.ai.update(
            dt,
            self.ball,
            self.ai_paddle,
            self.state_mgr.player_score,
            self.state_mgr.ai_score,
            self.state_mgr.rally_count,
        )

        bounced_wall, _ = self.ball.update(dt)
        if bounced_wall:
            self.sound.play("wall_hit")
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.ball_core, count=6)

        # Player Paddle Collision
        if self.ball.handle_paddle_collision(self.player_paddle, is_player=True):
            self.sound.play("paddle_hit")
            self.state_mgr.increment_rally()
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.player_primary, count=18)

        # AI Paddle Collision
        if self.ball.handle_paddle_collision(self.ai_paddle, is_player=False):
            self.sound.play("paddle_hit")
            self.state_mgr.increment_rally()
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.ai_primary, count=18)

        # Goal Detection
        if self.ball.x < 0:
            # AI Scored
            self.particles.emit(10, self.ball.y, self.config.colors.ai_primary, count=35, speed_range=(150.0, 450.0))
            is_game_over = self.state_mgr.record_goal("AI")
            self.sound.play("opponent_goal")
            if is_game_over:
                self.state_mgr.change_state(ExhibitionState.GAME_OVER)
            else:
                self.state_mgr.change_state(ExhibitionState.POINT_SCORED)
                self.state_mgr.pause_timer = self.config.physics.goal_pause_sec

        elif self.ball.x > self.config.display.width:
            # Player Scored
            self.particles.emit(
                self.config.display.width - 10,
                self.ball.y,
                self.config.colors.player_primary,
                count=35,
                speed_range=(150.0, 450.0),
            )
            is_game_over = self.state_mgr.record_goal("PLAYER")
            self.sound.play("goal")
            if is_game_over:
                self.state_mgr.change_state(ExhibitionState.GAME_OVER)
            else:
                self.state_mgr.change_state(ExhibitionState.POINT_SCORED)
                self.state_mgr.pause_timer = self.config.physics.goal_pause_sec

    def _update_point_scored_mode(self, dt: float) -> None:
        """Brief interlude before launching next ball serve."""
        self.player_paddle.update(dt)
        self.ai_paddle.update(dt)

        self.state_mgr.pause_timer -= dt
        if self.state_mgr.pause_timer <= 0:
            # Serve towards whichever side conceded
            serve_to_player = (self.ball.x < self.config.display.width / 2.0)
            self.ball.serve(direction_to_player=serve_to_player)
            self.sound.play("beep")
            self.state_mgr.change_state(ExhibitionState.PLAYING)

    def _render(self) -> None:
        """Renders the appropriate scene depending on exhibition state."""
        self.renderer.render_background()
        self.renderer.render_field()

        state = self.state_mgr.current_state
        tracking_state = self.vision.get_state()

        if state == ExhibitionState.ATTRACT:
            # Render demo paddles and ball
            self.renderer.render_paddle(self.player_paddle, self.config.colors.player_primary, self.config.colors.player_glow)
            self.renderer.render_paddle(self.ai_paddle, self.config.colors.ai_primary, self.config.colors.ai_glow)
            self.renderer.render_ball(self.ball)
            self.renderer.render_particles(self.particles)
            self.screens.draw_attract_screen(tracking_state)

        elif state == ExhibitionState.CALIBRATING:
            self.screens.draw_calibration_screen(tracking_state, tracking_state.calibration_progress)

        elif state in (ExhibitionState.PLAYING, ExhibitionState.POINT_SCORED):
            self.renderer.render_paddle(self.player_paddle, self.config.colors.player_primary, self.config.colors.player_glow)
            self.renderer.render_paddle(self.ai_paddle, self.config.colors.ai_primary, self.config.colors.ai_glow)
            self.renderer.render_ball(self.ball)
            self.renderer.render_particles(self.particles)
            self.renderer.render_scores(self.state_mgr.player_score, self.state_mgr.ai_score)

            # Camera PIP & AI Difficulty meter
            self.renderer.render_camera_pip(tracking_state, position=(25, 25))
            self.renderer.render_ai_difficulty_meter(self.ai.difficulty_level, position=(25, self.config.display.height - 40))

            # Inactivity alert if player temporarily vanishes
            if not tracking_state.face_detected and tracking_state.last_detected_time > 0:
                elapsed = time.perf_counter() - tracking_state.last_detected_time
                remaining = self.config.camera.inactivity_timeout_sec - elapsed
                if remaining < 3.5:
                    self.screens.draw_inactivity_warning(remaining)

            # Debug Overlay
            if self.config.debug_mode:
                self._render_debug_overlay()

        elif state == ExhibitionState.GAME_OVER:
            self.renderer.render_particles(self.particles)
            self.screens.draw_game_over_screen(self.state_mgr, tracking_state)

        pygame.display.flip()

    def _render_debug_overlay(self) -> None:
        """Renders diagnostic overlay with telemetry metrics."""
        fps = self.clock.get_fps()
        db_lines = [
            f"FPS: {fps:.1f}",
            f"AI Speed: {self.ai.current_speed:.0f} px/s | Diff: {self.ai.difficulty_level:.2f}",
            f"Ball Speed: {self.ball.speed:.0f} px/s",
            f"Rally Count: {self.state_mgr.rally_count}",
        ]
        y_offset = 200
        for line in db_lines:
            s = self.renderer.font_hud.render(line, True, self.config.colors.amber_accent)
            self.screen.blit(s, (25, y_offset))
            y_offset += 20

    def _shutdown(self) -> None:
        """Performs graceful resource termination."""
        logger.info("Shutting down FacePong subsystems...")
        self.vision.stop()
        pygame.quit()
