"""Cyberpunk retro-futuristic arcade rendering engine."""

import math
from typing import List, Optional, Tuple
import pygame
import numpy as np

from facepong.config import ColorPalette, GameConfig
from facepong.game.entities import Ball, Paddle, ParticleSystem
from facepong.vision.tracker import TrackingState


class NeonRenderer:
    """Specialized arcade renderer creating neon glows, HUD elements, and camera PIP."""

    def __init__(self, screen: pygame.Surface, config: GameConfig):
        self.screen = screen
        self.config = config
        self.colors = config.colors
        self.width = config.display.width
        self.height = config.display.height

        # Initialize fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("monospace", 56, bold=True)
        self.font_subtitle = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 72, bold=True)
        self.font_hud = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_banner = pygame.font.SysFont("monospace", 28, bold=True)

        # Pre-calculated grid surface for performance
        self._grid_surface = self._build_retro_grid()

    def _build_retro_grid(self) -> pygame.Surface:
        """Pre-renders an arcade grid background to save CPU cycles on Raspberry Pi."""
        grid = pygame.Surface((self.width, self.height))
        grid.fill(self.colors.background)

        # Subtle horizontal grid lines
        grid_color = self.colors.grid_lines
        spacing = 40
        for y in range(0, self.height, spacing):
            pygame.draw.line(grid, grid_color, (0, y), (self.width, y), 1)

        for x in range(0, self.width, spacing):
            pygame.draw.line(grid, grid_color, (x, 0), (x, self.height), 1)

        return grid

    def render_background(self) -> None:
        """Blits the static grid background."""
        self.screen.blit(self._grid_surface, (0, 0))

    def render_field(self) -> None:
        """Draws center court dividing line and boundary neon glow."""
        # Top and bottom border lines
        pygame.draw.line(self.screen, self.colors.divider_line, (0, 10), (self.width, 10), 2)
        pygame.draw.line(self.screen, self.colors.divider_line, (0, self.height - 10), (self.width, self.height - 10), 2)

        # Dashed center court divider
        dash_len = 16
        gap_len = 14
        center_x = self.width // 2
        y = 15
        while y < self.height - 15:
            pygame.draw.line(
                self.screen,
                self.colors.divider_line,
                (center_x, y),
                (center_x, min(self.height - 15, y + dash_len)),
                2,
            )
            y += dash_len + gap_len

    def render_paddle(self, paddle: Paddle, primary_color: Tuple[int, int, int], glow_color: Tuple[int, int, int]) -> None:
        """Renders paddle with neon layered bloom."""
        rect = paddle.rect

        # Outer bloom glow (simulated with slightly expanded alpha rect)
        glow_rect = rect.inflate(8, 8)
        glow_surf = pygame.Surface((glow_rect.width, glow_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, (*glow_color, 70), glow_surf.get_rect(), border_radius=6)
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
        """Draws living particles."""
        for p in particle_system.particles:
            alpha_ratio = max(0.0, min(1.0, p.life / max(0.01, p.max_life)))
            alpha = int(255 * alpha_ratio)
            size = max(1, int(p.size * alpha_ratio))

            p_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, (*p.color, alpha), (size, size), size)
            self.screen.blit(p_surf, (int(p.x - size), int(p.y - size)))

    def render_scores(self, player_score: int, ai_score: int) -> None:
        """Draws glowing arcade digital scoreboard."""
        surf_p = self.font_score.render(f"{player_score:02d}", True, self.colors.player_primary)
        surf_ai = self.font_score.render(f"{ai_score:02d}", True, self.colors.ai_primary)

        center_x = self.width // 2
        self.screen.blit(surf_p, (center_x - 140, 25))
        self.screen.blit(surf_ai, (center_x + 55, 25))

        # Labels
        lbl_p = self.font_hud.render("PLAYER [FACE]", True, self.colors.player_glow)
        lbl_ai = self.font_hud.render("AUDACIA AI", True, self.colors.ai_glow)
        self.screen.blit(lbl_p, (center_x - 140, 95))
        self.screen.blit(lbl_ai, (center_x + 55, 95))

    def render_camera_pip(self, tracking_state: TrackingState, position: Tuple[int, int] = (25, 25)) -> None:
        """
        Renders the Picture-In-Picture camera monitor with cybernetic HUD frame.
        """
        px, py = position
        pw = self.config.camera.preview_width
        ph = self.config.camera.preview_height

        # Outer cyber frame border
        frame_rect = pygame.Rect(px - 4, py - 4, pw + 8, ph + 8)
        border_color = self.colors.player_primary if tracking_state.face_detected else self.colors.danger
        pygame.draw.rect(self.screen, (15, 18, 30), frame_rect, border_radius=6)
        pygame.draw.rect(self.screen, border_color, frame_rect, width=2, border_radius=6)

        # Blit camera preview if available
        if tracking_state.preview_surface_buffer is not None:
            raw_rgb = tracking_state.preview_surface_buffer
            cam_surface = pygame.image.frombuffer(
                raw_rgb.tobytes(), (pw, ph), "RGB"
            )
            self.screen.blit(cam_surface, (px, py))
        else:
            # Placeholder surface
            placeholder = pygame.Surface((pw, ph))
            placeholder.fill((20, 24, 38))
            self.screen.blit(placeholder, (px, py))

        # Cybernetic corner brackets
        bracket_len = 10
        for cx, cy in [(px, py), (px + pw, py), (px, py + ph), (px + pw, py + ph)]:
            dx = bracket_len if cx == px else -bracket_len
            dy = bracket_len if cy == py else -bracket_len
            pygame.draw.line(self.screen, border_color, (cx, cy), (cx + dx, cy), 2)
            pygame.draw.line(self.screen, border_color, (cx, cy), (cx, cy + dy), 2)

        # Status text below PIP
        status_text = "FACE LOCKED" if tracking_state.face_detected else "NO FACE FOUND"
        status_surf = self.font_hud.render(status_text, True, border_color)
        self.screen.blit(status_surf, (px, py + ph + 8))

    def render_ai_difficulty_meter(self, difficulty: float, position: Tuple[int, int] = (25, 660)) -> None:
        """Renders dynamic difficulty indicator gauge."""
        x, y = position
        width = 192
        height = 12

        lbl = self.font_hud.render("AI DIFFICULTY ADAPTATION", True, self.colors.text_muted)
        self.screen.blit(lbl, (x, y - 18))

        # Background bar
        pygame.draw.rect(self.screen, (30, 36, 52), (x, y, width, height), border_radius=3)

        # Filled meter with color gradient based on difficulty
        fill_w = int(width * max(0.0, min(1.0, difficulty)))
        fill_color = self.colors.player_primary if difficulty < 0.5 else self.colors.ai_primary
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, (x, y, fill_w, height), border_radius=3)
        pygame.draw.rect(self.screen, self.colors.divider_line, (x, y, width, height), width=1, border_radius=3)

        # Percentage label
        pct_lbl = self.font_hud.render(f"{int(difficulty * 100)}%", True, fill_color)
        self.screen.blit(pct_lbl, (x + width + 8, y - 2))
