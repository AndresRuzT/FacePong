"""Cyberpunk retro-futuristic arcade rendering engine."""

import math
import random
from typing import List, Optional, Tuple
import pygame
import numpy as np

from facepong.config import ColorPalette, GameConfig
from facepong.game.entities import Ball, Paddle, ParticleSystem
from facepong.vision.tracker import TrackingState


class AmbientParticle:
    """Subtle floating cyber mote for arcade background atmosphere."""

    def __init__(self, bounds: pygame.Rect):
        self.bounds = bounds
        self.x = random.uniform(bounds.left + 10, bounds.right - 10)
        self.y = random.uniform(bounds.top + 10, bounds.bottom - 10)
        self.speed_y = random.uniform(12.0, 26.0)
        self.phase = random.uniform(0.0, math.pi * 2)
        self.amplitude = random.uniform(8.0, 20.0)
        self.size = int(max(1, round(random.uniform(1.5, 3.0))))
        self.color = random.choice([
            (0, 240, 255),    # Cyan
            (255, 45, 120),   # Pink
            (0, 255, 180),    # Emerald
            (140, 100, 255),  # Violet
        ])
        # Pre-rendered surface to avoid dynamic Surface allocations every frame on Raspberry Pi
        self._surf = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(self._surf, (*self.color, 140), (self.size, self.size), self.size)

    def update_and_draw(self, surface: pygame.Surface, dt: float, time_sec: float) -> None:
        self.y -= self.speed_y * dt
        if self.y < self.bounds.top:
            self.y = self.bounds.bottom - 5
            self.x = random.uniform(self.bounds.left + 10, self.bounds.right - 10)

        sway_x = self.x + math.sin(time_sec * 1.5 + self.phase) * self.amplitude
        surface.blit(self._surf, (int(sway_x - self.size), int(self.y - self.size)))


class NeonRenderer:
    """Arcade renderer creating neon glows, outer bezel HUD, and camera PIP."""

    def __init__(self, screen: pygame.Surface, config: GameConfig, arena_rect: Optional[pygame.Rect] = None):
        self.screen = screen
        self.config = config
        self.colors = config.colors
        self.width = screen.get_width()
        self.height = screen.get_height()

        # Outer bezel geometry
        header_h = 118
        margin_x = 24
        margin_b = 18
        self.arena_rect = arena_rect or pygame.Rect(
            margin_x,
            header_h,
            max(200, self.width - 2 * margin_x),
            max(200, self.height - header_h - margin_b),
        )

        # Initialize fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("monospace", 56, bold=True)
        self.font_subtitle = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 54, bold=True)
        self.font_hud = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_banner = pygame.font.SysFont("monospace", 26, bold=True)

        # Ambient particles for deep arcade atmosphere
        self.ambient_motes = [AmbientParticle(self.arena_rect) for _ in range(25)]

        # Pre-rendered surface cache to eliminate dynamic allocations at 60 FPS on Raspberry Pi
        self._init_surface_cache()

        # Lightweight screen effects state for Raspberry Pi
        self.active_screen_effect: Optional[str] = None
        self.effect_timer: float = 0.0
        self.effect_duration: float = 0.0
        self.shake_offset: Tuple[int, int] = (0, 0)
        self.shake_timer: float = 0.0
        self.match_result: Optional[str] = None

    def _init_surface_cache(self) -> None:
        """Precomputes neon bloom surfaces to maintain rock-solid 60 FPS on Raspberry Pi."""
        pw = self.config.physics.paddle_width + 8
        ph = self.config.physics.paddle_height + 8
        self._player_paddle_glow = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(self._player_paddle_glow, (*self.colors.player_glow, 75), self._player_paddle_glow.get_rect(), border_radius=6)

        self._ai_paddle_glow = pygame.Surface((pw, ph), pygame.SRCALPHA)
        pygame.draw.rect(self._ai_paddle_glow, (*self.colors.ai_glow, 75), self._ai_paddle_glow.get_rect(), border_radius=6)

        glow_radius = self.config.physics.ball_radius + 6
        self._ball_glow_radius = glow_radius
        self._ball_glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(self._ball_glow_surf, (*self.colors.ball_glow, 90), (glow_radius, glow_radius), glow_radius)

        self._trail_surfs = []
        for i in range(10):
            fraction = (i + 1) / 10.0
            r = max(1, int(self.config.physics.ball_radius * fraction * 0.8))
            alpha = int(140 * fraction)
            t_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(t_surf, (*self.colors.ball_glow, alpha), (r, r), r)
            self._trail_surfs.append((r, t_surf))

        self._particle_cache = {}

    def update_geometry(self, screen: pygame.Surface, arena_rect: pygame.Rect) -> None:
        """Adapts renderer to new window dimensions."""
        self.screen = screen
        self.width = screen.get_width()
        self.height = screen.get_height()
        self.arena_rect = arena_rect
        self.ambient_motes = [AmbientParticle(self.arena_rect) for _ in range(25)]

    def render_background(self, time_sec: float = 0.0, dt: float = 0.016) -> None:
        """Blits dynamic cyber background with animated synthwave grid inside arena."""
        self.screen.fill(self.colors.background)

        # Clip grid and ambient effects to inside the arena
        prev_clip = self.screen.get_clip()
        self.screen.set_clip(self.arena_rect)

        # Animated synthwave grid lines moving downwards
        spacing = 42
        y_offset = (time_sec * 32.0) % spacing
        y = self.arena_rect.top + y_offset
        while y < self.arena_rect.bottom:
            pygame.draw.line(self.screen, self.colors.grid_lines, (self.arena_rect.left, int(y)), (self.arena_rect.right, int(y)), 1)
            y += spacing

        # Vertical grid lines
        for x in range(self.arena_rect.left, self.arena_rect.right, spacing):
            pygame.draw.line(self.screen, self.colors.grid_lines, (x, self.arena_rect.top), (x, self.arena_rect.bottom), 1)

        # Ambient cyber motes drifting smoothly
        for mote in self.ambient_motes:
            mote.update_and_draw(self.screen, dt, time_sec)

        self.screen.set_clip(prev_clip)

    def render_arena_frame(self, time_sec: float = 0.0) -> None:
        """Renders glowing outer cybernetic bezel around the playfield."""
        pulse = 0.82 + 0.18 * math.sin(time_sec * 3.5)
        border_glow = (
            int(self.colors.player_glow[0] * pulse),
            int(self.colors.player_glow[1] * pulse),
            int(self.colors.player_glow[2] * pulse),
        )

        # Outer bloom ring
        glow_rect = self.arena_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, border_glow, glow_rect, width=1, border_radius=10)

        # Main solid arena border
        pygame.draw.rect(self.screen, self.colors.player_primary, self.arena_rect, width=2, border_radius=8)

        # Cybernetic corner accents (angled brackets)
        bracket_len = 16
        for cx, cy in [
            (self.arena_rect.left, self.arena_rect.top),
            (self.arena_rect.right, self.arena_rect.top),
            (self.arena_rect.left, self.arena_rect.bottom),
            (self.arena_rect.right, self.arena_rect.bottom),
        ]:
            dx = bracket_len if cx == self.arena_rect.left else -bracket_len
            dy = bracket_len if cy == self.arena_rect.top else -bracket_len
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (cx + dx, cy), 3)
            pygame.draw.line(self.screen, (255, 255, 255), (cx, cy), (cx, cy + dy), 3)

        # Center court divider line (strictly inside arena)
        dash_len = 14
        gap_len = 12
        center_x = self.arena_rect.centerx
        y = self.arena_rect.top + 8
        while y < self.arena_rect.bottom - 8:
            pygame.draw.line(
                self.screen,
                self.colors.divider_line,
                (center_x, int(y)),
                (center_x, int(min(self.arena_rect.bottom - 8, y + dash_len))),
                2,
            )
            y += dash_len + gap_len

    def trigger_goal_effect(self, scorer: str) -> None:
        """Triggers dynamic goal feedback (cyan rush for player, crimson alert for AI)."""
        if scorer == "PLAYER":
            self.active_screen_effect = "player_goal"
            self.effect_timer = 0.45
            self.effect_duration = 0.45
        else:
            self.active_screen_effect = "ai_goal"
            self.effect_timer = 0.35
            self.effect_duration = 0.35
            self.shake_timer = 0.25

    def trigger_match_end(self, winner: str) -> None:
        """Activates match conclusion visuals (gold victory or red cyber-glitch defeat)."""
        self.match_result = "player_win" if winner == "PLAYER" else "ai_win"

    def reset_match_effects(self) -> None:
        """Resets visual event states for a new match."""
        self.active_screen_effect = None
        self.effect_timer = 0.0
        self.shake_offset = (0, 0)
        self.shake_timer = 0.0
        self.match_result = None

    def render_screen_effects(self, time_sec: float = 0.0, dt: float = 0.016) -> None:
        """Draws lightweight screen flash/shake effects without alpha surface allocations."""
        # 1. Screen shake physics
        if self.shake_timer > 0.0:
            self.shake_timer = max(0.0, self.shake_timer - dt)
            ratio = self.shake_timer / 0.25
            mag = max(1, int(4.0 * ratio))
            self.shake_offset = (random.randint(-mag, mag), random.randint(-mag, mag))
        else:
            self.shake_offset = (0, 0)

        # 2. Goal scoring transient visual effects
        if self.active_screen_effect is not None and self.effect_timer > 0.0:
            self.effect_timer = max(0.0, self.effect_timer - dt)
            prog = self.effect_timer / max(0.01, self.effect_duration)

            if self.active_screen_effect == "player_goal":
                # Neon Cyan expanding pulse ring around arena
                expand = int((1.0 - prog) * 20.0)
                pulse_rect = self.arena_rect.inflate(expand * 2, expand * 2)
                glow_col = (0, int(240 * prog), int(255 * prog))
                pygame.draw.rect(self.screen, glow_col, pulse_rect, width=max(1, int(3 * prog) + 1), border_radius=10)

                # Cyber shockwave streak across the arena towards AI
                sweep_x = self.arena_rect.left + int((1.0 - prog) * self.arena_rect.width)
                if sweep_x < self.arena_rect.right:
                    pygame.draw.line(self.screen, (0, 255, 200), (sweep_x, self.arena_rect.top), (sweep_x, self.arena_rect.bottom), 2)

            elif self.active_screen_effect == "ai_goal":
                # Danger Red glitch strobe on the arena border
                if int(time_sec * 30.0) % 2 == 0:
                    danger_col = (int(255 * prog), int(40 * prog), int(40 * prog))
                    pygame.draw.rect(self.screen, danger_col, self.arena_rect, width=3, border_radius=8)
                    # Hazard indicator on player's side
                    pl_x = self.arena_rect.left + 8
                    pygame.draw.line(self.screen, danger_col, (pl_x, self.arena_rect.top + 8), (pl_x, self.arena_rect.bottom - 8), 3)

        # 3. Match outcome sustained visual effects (Victory vs Defeat)
        if self.match_result == "player_win":
            # Victory: Pulsing Golden Aurora Bezel
            pulse = 0.5 + 0.5 * math.sin(time_sec * 6.0)
            gold_col = (int(255 * (0.8 + 0.2 * pulse)), int(215 * (0.7 + 0.3 * pulse)), int(30 + 70 * pulse))
            glow_rect = self.arena_rect.inflate(int(8 + 6 * pulse), int(8 + 6 * pulse))
            pygame.draw.rect(self.screen, gold_col, glow_rect, width=2, border_radius=10)
            pygame.draw.rect(self.screen, (255, 240, 140), self.arena_rect, width=2, border_radius=8)

            # Golden corner champion brackets
            for cx, cy in [
                (self.arena_rect.left, self.arena_rect.top),
                (self.arena_rect.right, self.arena_rect.top),
                (self.arena_rect.left, self.arena_rect.bottom),
                (self.arena_rect.right, self.arena_rect.bottom),
            ]:
                dx = 24 if cx == self.arena_rect.left else -24
                dy = 24 if cy == self.arena_rect.top else -24
                pygame.draw.line(self.screen, gold_col, (cx, cy), (cx + dx, cy), 3)
                pygame.draw.line(self.screen, gold_col, (cx, cy), (cx, cy + dy), 3)

        elif self.match_result == "ai_win":
            # Defeat: Red Cyber-Glitch Bezel and subtle scanlines
            pulse = 0.5 + 0.5 * math.sin(time_sec * 7.5)
            red_col = (int(180 + 75 * pulse), 30, int(50 + 40 * pulse))
            pygame.draw.rect(self.screen, red_col, self.arena_rect, width=2, border_radius=8)

            # Intermittent scanline glitch bar
            if int(time_sec * 10.0) % 3 == 0:
                glitch_y = self.arena_rect.top + int((time_sec * 160.0) % max(1, self.arena_rect.height))
                pygame.draw.line(self.screen, (255, 45, 60), (self.arena_rect.left + 10, glitch_y), (self.arena_rect.right - 10, glitch_y), 1)

    def render_exterior_header(
        self,
        tracking_state: TrackingState,
        player_score: int,
        ai_score: int,
        ai_difficulty: float,
        match_duration_sec: float = 0.0,
        rally_count: int = 0,
    ) -> None:
        """
        Renders the outer scoreboard, AI difficulty telemetry, and camera PIP
        completely outside the playing field (in the top header bar).
        """
        header_h = self.arena_rect.top
        # Top banner background
        header_rect = pygame.Rect(0, 0, self.width, header_h)
        pygame.draw.rect(self.screen, (10, 12, 20), header_rect)
        pygame.draw.line(self.screen, self.colors.divider_line, (0, header_h - 1), (self.width, header_h - 1), 1)

        # 1. Left: Camera PIP (Completely outside arena)
        pip_w = 136
        pip_h = 92
        pip_x = self.arena_rect.left
        pip_y = 12
        self.render_camera_pip_box(tracking_state, (pip_x, pip_y), (pip_w, pip_h))

        # 2. Center: Scoreboard
        center_x = self.width // 2
        score_text = f"{player_score:02d}  :  {ai_score:02d}"
        surf_score = self.font_score.render(score_text, True, self.colors.text_primary)
        score_rect = surf_score.get_rect(center=(center_x, 38))
        self.screen.blit(surf_score, score_rect)

        # Team names
        lbl_p = self.font_hud.render("PLAYER [FACE]", True, self.colors.player_primary)
        lbl_ai = self.font_hud.render("AUDACIA AI", True, self.colors.ai_primary)
        self.screen.blit(lbl_p, (score_rect.left - 130, 28))
        self.screen.blit(lbl_ai, (score_rect.right + 18, 28))

        # Match telemetry beneath scores
        mins = int(match_duration_sec // 60)
        secs = int(match_duration_sec % 60)
        telemetry = f"RALLY: {rally_count:02d}   •   TIME: {mins:02d}:{secs:02d}   •   FIRST TO 5"
        surf_telem = self.font_hud.render(telemetry, True, self.colors.amber_accent)
        self.screen.blit(surf_telem, surf_telem.get_rect(center=(center_x, 82)))

        # 3. Right: AI Neural Core Telemetry Meter
        meter_w = 175
        meter_h = 10
        meter_x = self.arena_rect.right - meter_w
        meter_y = 48

        core_title = self.font_hud.render("AUDACIA NEURAL CORE", True, self.colors.text_primary)
        self.screen.blit(core_title, (meter_x, 22))

        # Meter background
        pygame.draw.rect(self.screen, (24, 28, 42), (meter_x, meter_y, meter_w, meter_h), border_radius=3)
        fill_w = int(meter_w * max(0.0, min(1.0, ai_difficulty)))
        fill_col = self.colors.player_primary if ai_difficulty < 0.55 else self.colors.ai_primary
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_col, (meter_x, meter_y, fill_w, meter_h), border_radius=3)
        pygame.draw.rect(self.screen, self.colors.divider_line, (meter_x, meter_y, meter_w, meter_h), width=1, border_radius=3)

        diff_pct = self.font_hud.render(f"DDA {int(ai_difficulty * 100)}%", True, fill_col)
        self.screen.blit(diff_pct, (meter_x, meter_y + 16))

        dda_tag = self.font_hud.render("ADAPTIVE", True, self.colors.text_muted)
        self.screen.blit(dda_tag, (meter_x + meter_w - dda_tag.get_width(), meter_y + 16))

    def render_camera_pip_box(self, tracking_state: TrackingState, position: Tuple[int, int], size: Tuple[int, int]) -> None:
        """Draws framed camera preview in designated rectangular area."""
        px, py = position
        pw, ph = size

        border_col = self.colors.player_primary if tracking_state.face_detected else self.colors.danger

        # Outer border
        pygame.draw.rect(self.screen, (16, 20, 32), (px, py, pw, ph), border_radius=4)
        pygame.draw.rect(self.screen, border_col, (px, py, pw, ph), width=1, border_radius=4)

        if tracking_state.preview_surface_buffer is not None:
            raw_rgb = tracking_state.preview_surface_buffer
            cam_surface = pygame.image.frombuffer(raw_rgb.tobytes(), (self.config.camera.preview_width, self.config.camera.preview_height), "RGB")
            # Fast nearest-neighbor hardware scale for high FPS on Raspberry Pi
            scaled_cam = pygame.transform.scale(cam_surface, (pw - 4, ph - 22))
            self.screen.blit(scaled_cam, (px + 2, py + 2))
        else:
            placeholder = pygame.Surface((pw - 4, ph - 22))
            placeholder.fill((20, 24, 38))
            self.screen.blit(placeholder, (px + 2, py + 2))

        # Status tag
        stat_text = "FACE LOCKED" if tracking_state.face_detected else "SEARCHING"
        stat_surf = self.font_hud.render(stat_text, True, border_col)
        self.screen.blit(stat_surf, (px + 4, py + ph - 16))

    def render_paddle(self, paddle: Paddle, primary_color: Tuple[int, int, int], glow_color: Tuple[int, int, int]) -> None:
        """Renders paddle with pre-cached neon layered bloom."""
        rect = paddle.rect

        # Outer bloom glow from pre-rendered cache
        glow_surf = self._player_paddle_glow if primary_color == self.colors.player_primary else self._ai_paddle_glow
        glow_rect = rect.inflate(8, 8)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Solid inner paddle
        pygame.draw.rect(self.screen, primary_color, rect, border_radius=4)
        # Core highlight
        inner_rect = rect.inflate(-6, -6)
        if inner_rect.width > 0 and inner_rect.height > 0:
            pygame.draw.rect(self.screen, (255, 255, 255), inner_rect, border_radius=2)

    def render_ball(self, ball: Ball) -> None:
        """Renders ball with speed trail and radial neon bloom using cached surfaces."""
        # Draw fading comet trail from pre-rendered cache
        trail_pts = list(ball.trail)
        n = len(trail_pts)
        for i, pt in enumerate(trail_pts):
            idx = min(9, int((i + 1) / max(1, n) * 9.99))
            r, t_surf = self._trail_surfs[idx]
            self.screen.blit(t_surf, (int(pt[0] - r), int(pt[1] - r)))

        bx, by = int(ball.x), int(ball.y)

        # Outer glow ring from pre-rendered cache
        gr = self._ball_glow_radius
        self.screen.blit(self._ball_glow_surf, (bx - gr, by - gr))

        # Core ball
        pygame.draw.circle(self.screen, self.colors.ball_core, (bx, by), ball.radius)

    def render_particles(self, particle_system: ParticleSystem) -> None:
        """Draws living collision spark particles using discrete cached surfaces."""
        for p in particle_system.particles:
            alpha_ratio = max(0.0, min(1.0, p.life / max(0.01, p.max_life)))
            alpha_bucket = int(min(255, round(255 * alpha_ratio / 32.0) * 32))
            if alpha_bucket <= 0:
                continue
            size = max(1, int(round(p.size * alpha_ratio)))
            key = (p.color, size, alpha_bucket)
            p_surf = self._particle_cache.get(key)
            if p_surf is None:
                p_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(p_surf, (*p.color, alpha_bucket), (size, size), size)
                self._particle_cache[key] = p_surf
            self.screen.blit(p_surf, (int(p.x - size), int(p.y - size)))
