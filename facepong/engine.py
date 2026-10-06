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
        display_flags = pygame.DOUBLEBUF | pygame.RESIZABLE
        if config.display.fullscreen:
            display_flags |= pygame.FULLSCREEN

        display_w = config.display.width
        display_h = config.display.height

        # Fit desktop resolution in windowed mode
        if not config.display.fullscreen and getattr(config.display, "windowed_maximized", True):
            desktop_info = pygame.display.Info()
            if desktop_info.current_w > 640 and desktop_info.current_h > 480:
                display_w = desktop_info.current_w
                display_h = max(600, desktop_info.current_h - 45)

        self.screen = pygame.display.set_mode((display_w, display_h), display_flags)
        pygame.display.set_caption(config.display.title)
        self.clock = pygame.time.Clock()

        # Calculate arena playfield geometry
        self._calculate_arena_geometry(display_w, display_h)

        # Core Subsystems
        self.vision = VisionPipeline(config.camera)
        self.state_mgr = GameStateManager(
            winning_score=config.physics.winning_score,
            inactivity_timeout_sec=config.camera.inactivity_timeout_sec,
        )
        self.ai = AdaptiveAIController(
            config.ai,
            config.physics,
            display_w,
            display_h,
        )
        self.sound = SoundManager(config.audio)
        self.renderer = NeonRenderer(self.screen, config, arena_rect=self.arena_rect)
        self.screens = ScreenManager(self.renderer, config)
        self.particles = ParticleSystem(max_particles=150)

        # Game Entities (Positioned strictly inside arena bounds)
        paddle_w = config.physics.paddle_width
        paddle_h = config.physics.paddle_height
        margin = config.physics.paddle_margin

        self.player_paddle = Paddle(
            x=self.arena_rect.left + margin + paddle_w / 2.0,
            y=self.arena_rect.centery,
            width=paddle_w,
            height=paddle_h,
            screen_height=display_h,
            min_y=self.arena_rect.top,
            max_y=self.arena_rect.bottom,
        )
        self.ai_paddle = Paddle(
            x=self.arena_rect.right - margin - paddle_w / 2.0,
            y=self.arena_rect.centery,
            width=paddle_w,
            height=paddle_h,
            screen_height=display_h,
            min_y=self.arena_rect.top,
            max_y=self.arena_rect.bottom,
        )
        self.ball = Ball(display_w, display_h, config.physics, arena_rect=self.arena_rect)

        # Keyboard override tracker
        self._keyboard_active = False

        # State transition and calibration tracking
        self._previous_state = ExhibitionState.ATTRACT
        self._last_calibration_second = -1
        self._calibration_progress = 0.0

    def _calculate_arena_geometry(self, w: int, h: int) -> None:
        """Computes rectangular playfield arena leaving top header for HUD/PIP and shortening width by 15%."""
        header_h = 118
        margin_x = 24
        margin_b = 18
        available_w = max(300, w - 2 * margin_x)
        # Shorten horizontal distance by 15% for faster, more dynamic rallies
        arena_w = int(available_w * 0.85)
        arena_left = (w - arena_w) // 2
        self.arena_rect = pygame.Rect(
            arena_left,
            header_h,
            arena_w,
            max(200, h - header_h - margin_b),
        )

    def _on_window_resize(self, new_w: int, new_h: int) -> None:
        """Adapts arena, entities, and UI when window size changes."""
        w = max(640, new_w)
        h = max(480, new_h)
        self.screen = pygame.display.set_mode((w, h), pygame.DOUBLEBUF | pygame.RESIZABLE)
        self._calculate_arena_geometry(w, h)
        self.renderer.update_geometry(self.screen, self.arena_rect)
        self.screens.update_geometry(self.renderer)

        margin = self.config.physics.paddle_margin
        paddle_w = self.config.physics.paddle_width
        self.player_paddle.x = self.arena_rect.left + margin + paddle_w / 2.0
        self.player_paddle.set_bounds(self.arena_rect.top, self.arena_rect.bottom)
        self.ai_paddle.x = self.arena_rect.right - margin - paddle_w / 2.0
        self.ai_paddle.set_bounds(self.arena_rect.top, self.arena_rect.bottom)
        self.ball.set_arena(self.arena_rect)

    def start(self) -> None:
        """Starts background threads and executes the main loop."""
        self.vision.start()
        self._is_running = True
        logger.info("FacePong Engine initialized and running")

        last_time = time.perf_counter()

        try:
            while self._is_running:
                current_time = time.perf_counter()
                dt = min(0.05, current_time - last_time)
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

            if event.type == pygame.VIDEORESIZE:
                self._on_window_resize(event.w, event.h)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._is_running = False
                    return
                elif event.key == pygame.K_d:
                    self.config.debug_mode = not self.config.debug_mode
                elif event.key == pygame.K_SPACE:
                    if self.state_mgr.current_state == ExhibitionState.ATTRACT:
                        self.state_mgr.change_state(ExhibitionState.CALIBRATING)
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

        # Re-fetch state in case watchdog triggered a transition
        state = self.state_mgr.current_state

        # Detect transition into CALIBRATING
        if state == ExhibitionState.CALIBRATING and self._previous_state != ExhibitionState.CALIBRATING:
            self.vision.begin_calibration()
            self._last_calibration_second = -1
            self._calibration_progress = 0.0
        self._previous_state = state

        # Update player paddle from face tracking during active gameplay
        if state == ExhibitionState.PLAYING:
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
            self.player_paddle.update(dt)
            self.ai_paddle.update(dt)

    def _update_attract_mode(self, dt: float) -> None:
        """Simulates autonomous gameplay demo during idle kiosk state."""
        self.ball.update(dt)

        # Bounce off arena boundaries in demo mode
        left_bound = self.arena_rect.left + self.config.physics.ball_radius + 4
        right_bound = self.arena_rect.right - self.config.physics.ball_radius - 4

        if self.ball.x <= left_bound:
            self.ball.x = left_bound
            self.ball.vx = abs(self.ball.vx)
            self.particles.emit(self.ball.x, self.ball.y, self.config.colors.player_primary, count=8)
        elif self.ball.x >= right_bound:
            self.ball.x = right_bound
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

        now = time.perf_counter()
        elapsed = now - self.state_mgr.state_enter_time
        duration = max(0.1, self.config.camera.calibration_duration_sec)
        self._calibration_progress = min(1.0, elapsed / duration)

        # Trigger countdown beeps (3, 2, 1)
        remaining_seconds = max(1, math.ceil(duration - elapsed))
        if remaining_seconds != self._last_calibration_second:
            self._last_calibration_second = remaining_seconds
            self.sound.play("beep")

        if self._calibration_progress >= 1.0 or tracking_state.is_calibrated:
            self.vision.finish_calibration()
            self.state_mgr.start_new_match()
            center_y = float(self.arena_rect.centery)
            self.player_paddle.reset_to_center(center_y)
            self.ai_paddle.reset_to_center(center_y)
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

        # Goal Detection relative to arena frame
        if self.ball.x < self.arena_rect.left:
            # AI Scored
            self.particles.emit(self.arena_rect.left + 5, self.ball.y, self.config.colors.ai_primary, count=35, speed_range=(150.0, 450.0))
            is_game_over = self.state_mgr.record_goal("AI")
            self.sound.play("opponent_goal")
            if is_game_over:
                self.state_mgr.change_state(ExhibitionState.GAME_OVER)
            else:
                self.state_mgr.change_state(ExhibitionState.POINT_SCORED)
                self.state_mgr.pause_timer = self.config.physics.goal_pause_sec
                # Smoothly target arena center for both paddles
                center_y = float(self.arena_rect.centery)
                self.player_paddle.target_y = center_y
                self.ai_paddle.target_y = center_y

        elif self.ball.x > self.arena_rect.right:
            # Player Scored
            self.particles.emit(
                self.arena_rect.right - 5,
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
                # Smoothly target arena center for both paddles
                center_y = float(self.arena_rect.centery)
                self.player_paddle.target_y = center_y
                self.ai_paddle.target_y = center_y

    def _update_point_scored_mode(self, dt: float) -> None:
        """Brief interlude before launching next ball serve; smoothly centers both paddles."""
        center_y = float(self.arena_rect.centery)
        self.player_paddle.target_y = center_y
        self.ai_paddle.target_y = center_y
        self.player_paddle.update(dt, smooth_factor=16.0)
        self.ai_paddle.update(dt, smooth_factor=16.0)

        self.state_mgr.pause_timer -= dt
        if self.state_mgr.pause_timer <= 0:
            # Snap cleanly to center baseline right before serve
            self.player_paddle.reset_to_center(center_y)
            self.ai_paddle.reset_to_center(center_y)
            serve_to_player = (self.ball.x < self.arena_rect.centerx)
            self.ball.serve(direction_to_player=serve_to_player)
            self.sound.play("beep")
            self.state_mgr.change_state(ExhibitionState.PLAYING)

    def _render(self) -> None:
        """Renders scene with outer bezel HUD and unoccluded arena."""
        now = time.perf_counter()
        dt = 1.0 / max(1, self.config.display.target_fps)

        # 1. Background with animated perspective synthwave grid inside arena
        self.renderer.render_background(time_sec=now, dt=dt)

        # 2. Outer glowing arena bezel & field dividers
        self.renderer.render_arena_frame(time_sec=now)

        # 3. Exterior Top Header Bar (HUD & Camera PIP completely outside the arena)
        tracking_state = self.vision.get_state()
        state = self.state_mgr.current_state

        match_dur = self.state_mgr.match_duration_sec or (
            now - self.state_mgr.match_start_time if self.state_mgr.match_start_time > 0 else 0.0
        )
        self.renderer.render_exterior_header(
            tracking_state=tracking_state,
            player_score=self.state_mgr.player_score,
            ai_score=self.state_mgr.ai_score,
            ai_difficulty=self.ai.difficulty_level,
            match_duration_sec=match_dur,
            rally_count=self.state_mgr.rally_count,
        )

        # 4. Arena Entities
        if state == ExhibitionState.ATTRACT:
            self.renderer.render_paddle(self.player_paddle, self.config.colors.player_primary, self.config.colors.player_glow)
            self.renderer.render_paddle(self.ai_paddle, self.config.colors.ai_primary, self.config.colors.ai_glow)
            self.renderer.render_ball(self.ball)
            self.renderer.render_particles(self.particles)
            self.screens.draw_attract_screen(tracking_state)

        elif state == ExhibitionState.CALIBRATING:
            self.screens.draw_calibration_screen(tracking_state, self._calibration_progress)

        elif state in (ExhibitionState.PLAYING, ExhibitionState.POINT_SCORED):
            self.renderer.render_paddle(self.player_paddle, self.config.colors.player_primary, self.config.colors.player_glow)
            self.renderer.render_paddle(self.ai_paddle, self.config.colors.ai_primary, self.config.colors.ai_glow)
            self.renderer.render_ball(self.ball)
            self.renderer.render_particles(self.particles)

            if not tracking_state.face_detected and tracking_state.last_detected_time > 0:
                elapsed = time.perf_counter() - tracking_state.last_detected_time
                remaining = self.config.camera.inactivity_timeout_sec - elapsed
                if remaining < 3.5:
                    self.screens.draw_inactivity_warning(remaining)

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
            f"Camera: {self.vision.camera.device_description}",
        ]
        y_offset = self.arena_rect.top + 20
        for line in db_lines:
            s = self.renderer.font_hud.render(line, True, self.config.colors.amber_accent)
            self.screen.blit(s, (self.arena_rect.left + 20, y_offset))
            y_offset += 20

    def _shutdown(self) -> None:
        """Performs graceful resource termination."""
        logger.info("Shutting down FacePong subsystems...")
        self.vision.stop()
        pygame.quit()
