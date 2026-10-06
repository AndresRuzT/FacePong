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
        self.size = random.uniform(1.5, 3.0)
        self.color = random.choice([
            (0, 240, 255),    # Cyan
            (255, 45, 120),   # Pink
            (0, 255, 180),    # Emerald
            (140, 100, 255),  # Violet
        ])

    def update_and_draw(self, surface: pygame.Surface, dt: float, time_sec: float) -> None:
        self.y -= self.speed_y * dt
        if self.y < self.bounds.top:
            self.y = self.bounds.bottom - 5
            self.x = random.uniform(self.bounds.left + 10, self.bounds.right - 10)

        sway_x = self.x + math.sin(time_sec * 1.5 + self.phase) * self.amplitude
        alpha = int(90 + 50 * math.sin(time_sec * 2.0 + self.phase))

        p_surf = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        pygame.draw.circle(p_surf, (*self.color, max(20, min(200, alpha))), (int(self.size), int(self.size)), int(self.size))
        surface.blit(p_surf, (int(sway_x - self.size), int(self.y - self.size)))


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
            scaled_cam = pygame.transform.smoothscale(cam_surface, (pw - 4, ph - 22))
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
        """Renders paddle with neon layered bloom."""
        rect = paddle.rect

        # Outer bloom glow
        glow_rect = rect.inflate(8, 8)
        glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*glow_color, 75), glow_surf.get_rect(), border_radius=6)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Solid inner paddle
        pygame.draw.rect(self.screen, primary_color, rect, border_radius=4)
        # Core highlight
        inner_rect = rect.inflate(-6, -6)
        if inner_rect.width > 0 and inner_rect.height > 0:
            pygame.draw.rect(self.screen, (255, 255, 255), inner_rect, border_radius=2)

    def render_ball(self, ball: Ball) -> None:
        """Renders ball with speed trail and radial neon bloom."""
        # Draw fading comet trail
        trail_pts = list(ball.trail)
        n = len(trail_pts)
        for i, pt in enumerate(trail_pts):
            fraction = (i + 1) / max(1, n)
            radius = int(ball.radius * fraction * 0.8)
            if radius > 1:
                alpha = int(140 * fraction)
                trail_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (*self.colors.ball_glow, alpha), (radius, radius), radius)
                self.screen.blit(trail_surf, (int(pt[0] - radius), int(pt[1] - radius)))

        bx, by = int(ball.x), int(ball.y)

        # Outer glow ring
        glow_radius = ball.radius + 6
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.colors.ball_glow, 90), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (bx - glow_radius, by - glow_radius))

        # Core ball
        pygame.draw.circle(self.screen, self.colors.ball_core, (bx, by), ball.radius)

    def render_particles(self, particle_system: ParticleSystem) -> None:
        """Draws living collision spark particles."""
        for p in particle_system.particles:
            alpha_ratio = max(0.0, min(1.0, p.life / max(0.01, p.max_life)))
            alpha = int(255 * alpha_ratio)
            size = max(1, int(p.size * alpha_ratio))

            p_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, (*p.color, alpha), (size, size), size)
            self.screen.blit(p_surf, (int(p.x - size), int(p.y - size)))
