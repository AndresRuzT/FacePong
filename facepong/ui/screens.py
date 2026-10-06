"""Exhibition flow screen renderers: Attract, Calibration, and Game Over."""

import math
import time
from typing import Optional
import pygame

from facepong.config import ColorPalette, GameConfig
from facepong.game.state import GameStateManager
from facepong.ui.renderer import NeonRenderer
from facepong.vision.tracker import TrackingState


class ScreenManager:
    """Renders the distinct state screens of the exhibition kiosk."""

    def __init__(self, renderer: NeonRenderer, config: GameConfig):
        self.renderer = renderer
        self.config = config
        self.colors = config.colors
        self.screen = renderer.screen

    def update_geometry(self, renderer: NeonRenderer) -> None:
        """Synchronizes renderer changes."""
        self.renderer = renderer
        self.screen = renderer.screen

    def draw_attract_screen(self, tracking_state: TrackingState) -> None:
        """Renders the idle kiosk attract display with pulsing invitations."""
        now = time.perf_counter()
        arena = self.renderer.arena_rect
        center_x = arena.centerx

        # Glowing Header
        title_y = arena.top + 55
        title_surf = self.renderer.font_title.render("FACEPONG", True, self.colors.player_primary)
        title_rect = title_surf.get_rect(center=(center_x, title_y))

        # Title shadow / bloom
        bloom_surf = self.renderer.font_title.render("FACEPONG", True, self.colors.player_glow)
        self.screen.blit(bloom_surf, (title_rect.x + 2, title_rect.y + 2))
        self.screen.blit(title_surf, title_rect)

        # Lab branding
        sub_surf = self.renderer.font_subtitle.render(
            "AUDACIA  •  CENTRO DE INVESTIGACIÓN EN ROBÓTICA E IA", True, self.colors.amber_accent
        )
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(center_x, title_y + 50)))

        tech_surf = self.renderer.font_hud.render(
            "MediaPipe Face Mesh  •  Exponential Moving Average  •  Dynamic AI Adaptation",
            True,
            self.colors.text_muted,
        )
        self.screen.blit(tech_surf, tech_surf.get_rect(center=(center_x, title_y + 80)))

        # Pulsing Call-to-Action
        pulse = 0.5 + 0.5 * math.sin(now * 4.0)
        alpha = int(140 + 115 * pulse)
        callout_color = (
            int(self.colors.player_primary[0] * (alpha / 255.0)),
            int(self.colors.player_primary[1] * (alpha / 255.0)),
            int(self.colors.player_primary[2] * (alpha / 255.0)),
        )

        prompt_str = "►  STAND IN FRONT OF THE CAMERA TO BEGIN  ◄"
        prompt_surf = self.renderer.font_banner.render(prompt_str, True, callout_color)
        self.screen.blit(prompt_surf, prompt_surf.get_rect(center=(center_x, title_y + 160)))

        # Keyboard fallback hint
        kb_surf = self.renderer.font_hud.render(
            "[ Press SPACE to Start with Keyboard  |  W / S or UP / DOWN to Move ]",
            True,
            self.colors.text_muted,
        )
        self.screen.blit(kb_surf, kb_surf.get_rect(center=(center_x, title_y + 195)))

        # Features box
        box_y = title_y + 230
        box_rect = pygame.Rect(center_x - 310, box_y, 620, 105)
        pygame.draw.rect(self.screen, (16, 20, 32), box_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.colors.divider_line, box_rect, width=1, border_radius=8)

        feat1 = self.renderer.font_hud.render("• MOVE YOUR HEAD UP AND DOWN TO CONTROL YOUR PADDLE", True, self.colors.text_primary)
        feat2 = self.renderer.font_hud.render("• ADAPTIVE NEURAL OPPONENT TUNES DIFFICULTY TO YOUR SKILL", True, self.colors.text_primary)
        feat3 = self.renderer.font_hud.render("• AUTONOMOUS PRESENCE WATCHDOG AUTO-RESETS WHEN IDLE", True, self.colors.text_primary)

        self.screen.blit(feat1, (center_x - 290, box_y + 16))
        self.screen.blit(feat2, (center_x - 290, box_y + 44))
        self.screen.blit(feat3, (center_x - 290, box_y + 72))

    def draw_calibration_screen(self, tracking_state: TrackingState, progress: float) -> None:
        """Renders the 3-second face calibration sequence inside the arena."""
        arena = self.renderer.arena_rect
        center_x = arena.centerx
        center_y = arena.centery

        # Header
        hdr = self.renderer.font_banner.render("CALIBRATING FACIAL TRACKING", True, self.colors.player_primary)
        self.screen.blit(hdr, hdr.get_rect(center=(center_x, arena.top + 50)))

        sub = self.renderer.font_hud.render("Please face the camera in your neutral position", True, self.colors.text_muted)
        self.screen.blit(sub, sub.get_rect(center=(center_x, arena.top + 85)))

        # Circular Cyber Reticle in center
        radius = 75
        pygame.draw.circle(self.screen, (25, 32, 50), (center_x, center_y), radius, 2)
        pygame.draw.circle(self.screen, self.colors.player_glow, (center_x, center_y), radius + 8, 1)

        # Progress Arc
        sweep_angle = int(progress * 360)
        if sweep_angle > 0:
            rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
            pygame.draw.arc(self.screen, self.colors.player_primary, rect, 0, math.radians(sweep_angle), 4)

        # Countdown number
        remaining = max(1, math.ceil(self.config.camera.calibration_duration_sec * (1.0 - progress)))
        cd_surf = self.renderer.font_title.render(str(remaining), True, self.colors.player_primary)
        self.screen.blit(cd_surf, cd_surf.get_rect(center=(center_x, center_y)))

        # Status text below reticle
        lock_status = "POSITION LOCKED • HOLD STILL" if tracking_state.face_detected else "SEARCHING FOR FACE..."
        col = self.colors.player_primary if tracking_state.face_detected else self.colors.danger
        lbl = self.renderer.font_hud.render(lock_status, True, col)
        self.screen.blit(lbl, lbl.get_rect(center=(center_x, center_y + radius + 35)))

    def draw_game_over_screen(self, state_mgr: GameStateManager, tracking_state: TrackingState) -> None:
        """Renders match results and statistical telemetry inside arena."""
        arena = self.renderer.arena_rect
        center_x = arena.centerx

        winner_player = (state_mgr.winner == "PLAYER")
        winner_color = self.colors.player_primary if winner_player else self.colors.ai_primary
        title_text = "PLAYER VICTORIOUS!" if winner_player else "AUDACIA AI DOMINATION!"

        # Banner
        banner_surf = self.renderer.font_title.render(title_text, True, winner_color)
        self.screen.blit(banner_surf, banner_surf.get_rect(center=(center_x, arena.top + 60)))

        # Final Scoreboard
        score_surf = self.renderer.font_score.render(
            f"{state_mgr.player_score:02d}  -  {state_mgr.ai_score:02d}", True, self.colors.text_primary
        )
        self.screen.blit(score_surf, score_surf.get_rect(center=(center_x, arena.top + 130)))

        # Match Stats Card
        card_rect = pygame.Rect(center_x - 240, arena.top + 185, 480, 150)
        pygame.draw.rect(self.screen, (16, 20, 32), card_rect, border_radius=8)
        pygame.draw.rect(self.screen, winner_color, card_rect, width=1, border_radius=8)

        mins = int(state_mgr.match_duration_sec // 60)
        secs = int(state_mgr.match_duration_sec % 60)
        dur_str = f"{mins:02d}:{secs:02d}"

        stats = [
            f"MATCH DURATION:        {dur_str}",
            f"LONGEST RALLY:         {state_mgr.max_rally_match} VOLLEYS",
            f"FINAL SCORE:           {state_mgr.player_score} - {state_mgr.ai_score}",
            f"AUTONOMOUS RESET:      RETURNING TO TITLE SHORTLY",
        ]

        for i, text in enumerate(stats):
            t_surf = self.renderer.font_hud.render(text, True, self.colors.text_primary)
            self.screen.blit(t_surf, (center_x - 210, arena.top + 205 + i * 30))

        # Bottom Prompt
        prompt = self.renderer.font_banner.render("STEP BACK TO RETURN TO ATTRACT MODE", True, self.colors.amber_accent)
        self.screen.blit(prompt, prompt.get_rect(center=(center_x, arena.bottom - 45)))

    def draw_inactivity_warning(self, seconds_left: float) -> None:
        """Draws overlay banner if player presence has lapsed during match."""
        arena = self.renderer.arena_rect
        center_x = arena.centerx
        banner_rect = pygame.Rect(center_x - 280, arena.top + 20, 560, 44)
        pygame.draw.rect(self.screen, (40, 15, 20), banner_rect, border_radius=6)
        pygame.draw.rect(self.screen, self.colors.danger, banner_rect, width=2, border_radius=6)

        msg = f"FACE LOST - RESETTING IN {max(0.0, seconds_left):.1f}s"
        surf = self.renderer.font_banner.render(msg, True, self.colors.danger)
        self.screen.blit(surf, surf.get_rect(center=(center_x, arena.top + 42)))
