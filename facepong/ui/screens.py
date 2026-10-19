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

    def draw_attract_screen(self, tracking_state: TrackingState, countdown_sec: Optional[float] = None) -> None:
        """Renders a clean, retro-arcade attract display with minimal text and smooth pulse effects."""
        now = time.perf_counter()
        arena = self.renderer.arena_rect
        center_x = arena.centerx
        center_y = arena.centery

        # 1. Iconic Arcade Logo with retro multi-pass glow
        title_y = center_y - 65
        title_text = "FACEPONG"

        # Retro chromatic depth shadow (Deep Violet / Magenta)
        shadow_surf = self.renderer.font_title.render(title_text, True, (130, 20, 80))
        self.screen.blit(shadow_surf, shadow_surf.get_rect(center=(center_x + 3, title_y + 3)))

        # Neon Cyan Glow layer
        glow_surf = self.renderer.font_title.render(title_text, True, self.colors.player_glow)
        self.screen.blit(glow_surf, glow_surf.get_rect(center=(center_x, title_y)))

        # Bright Cyan Core
        title_surf = self.renderer.font_title.render(title_text, True, self.colors.player_primary)
        self.screen.blit(title_surf, title_surf.get_rect(center=(center_x, title_y)))

        # 2. Main Arcade Call-To-Action (Pulsing bilingual prompt)
        # Smoothly alternates between Spanish and English every 3.5 seconds
        period = 3.5
        cycle = (now / period) % 2.0
        is_spanish = cycle < 1.0

        pulse = 0.5 + 0.5 * math.sin(now * 4.5)
        pulse_alpha = int(140 + 115 * pulse)
        callout_color = (
            int(self.colors.amber_accent[0] * (pulse_alpha / 255.0)),
            int(self.colors.amber_accent[1] * (pulse_alpha / 255.0)),
            int(self.colors.amber_accent[2] * (pulse_alpha / 255.0)),
        )

        fc = getattr(tracking_state, "face_count", 1 if tracking_state.face_detected else 0)
        if fc >= 2:
            prompt_str = "► ¡2 JUGADORES DETECTADOS! • INICIANDO PvP... ◄" if is_spanish else "► 2 PLAYERS DETECTED • STARTING PvP... ◄"
            kb_str = "[ ESPACIO: INICIAR PvP   •   P1 (IZQ) vs P2 (DER) ]" if is_spanish else "[ SPACE: START PvP   •   P1 (LEFT) vs P2 (RIGHT) ]"
        elif fc == 1:
            if countdown_sec is not None and countdown_sec > 0:
                rem = max(1, math.ceil(countdown_sec))
                prompt_str = f"► 1 JUGADOR • ESPERANDO 2DO JUGADOR ({rem}s) ◄" if is_spanish else f"► 1 PLAYER • WAITING FOR 2ND PLAYER ({rem}s) ◄"
                kb_str = "[ ESPACIO: JUGAR YA VS IA   •   P1 (IZQ) vs P2 (DER) ]" if is_spanish else "[ SPACE: PLAY NOW VS AI   •   P1 (LEFT) vs P2 (RIGHT) ]"
            else:
                prompt_str = "► 1 JUGADOR DETECTADO • MIRA AL FRENTE PARA JUGAR ◄" if is_spanish else "► 1 PLAYER DETECTED • LOOK FORWARD TO PLAY ◄"
                kb_str = "[ ESPACIO: INICIAR   •   T: PERFIL DISPOSITIVO ]" if is_spanish else "[ SPACE: START   •   T: DEVICE PROFILE ]"
        else:
            if countdown_sec is not None and countdown_sec > 0:
                rem = max(1, math.ceil(countdown_sec))
                prompt_str = f"► BUSCANDO JUGADORES ({rem}s) • MIRA A LA CÁMARA ◄" if is_spanish else f"► SEARCHING FOR PLAYERS ({rem}s) • LOOK AT CAMERA ◄"
            else:
                prompt_str = "►  MIRA A LA CÁMARA PARA JUGAR  ◄" if is_spanish else "►  LOOK AT THE CAMERA TO PLAY  ◄"
            kb_str = "[ ESPACIO: INICIAR   •   T: PERFIL DISPOSITIVO ]" if is_spanish else "[ SPACE: START   •   T: DEVICE PROFILE ]"

        prompt_surf = self.renderer.font_banner.render(prompt_str, True, callout_color)
        self.screen.blit(prompt_surf, prompt_surf.get_rect(center=(center_x, center_y + 25)))

        # 3. Minimal Subtle Secondary Prompt (Keyboard alternative + Profile hint)
        kb_surf = self.renderer.font_hud.render(kb_str, True, self.colors.text_muted)
        self.screen.blit(kb_surf, kb_surf.get_rect(center=(center_x, center_y + 68)))

    def draw_calibration_screen(self, tracking_state: TrackingState, progress: float, is_pvp: bool = False) -> None:
        """Renders the face calibration sequence inside the arena (single or dual reticle)."""
        arena = self.renderer.arena_rect
        center_x = arena.centerx
        center_y = arena.centery

        title = "CALIBRATING DUAL FACIAL TRACKING" if is_pvp else "CALIBRATING FACIAL TRACKING"
        hdr = self.renderer.font_banner.render(title, True, self.colors.player_primary)
        self.screen.blit(hdr, hdr.get_rect(center=(center_x, arena.top + 45)))

        sub_text = (
            "Player 1 (Left) & Player 2 (Right): Look at the camera"
            if is_pvp
            else "Please face the camera in your neutral position"
        )
        sub = self.renderer.font_hud.render(sub_text, True, self.colors.text_muted)
        self.screen.blit(sub, sub.get_rect(center=(center_x, arena.top + 75)))

        remaining = max(1, math.ceil(self.config.camera.calibration_duration_sec * (1.0 - progress)))
        sweep_angle = int(progress * 360)

        if is_pvp:
            # Dual Reticles: Left = Player 1 (Cyan), Right = Player 2 (Crimson)
            spacing = min(220, arena.width // 4)
            p1_cx = center_x - spacing
            p2_cx = center_x + spacing
            radius = 65

            for cx, label, detected, col in [
                (p1_cx, "PLAYER 1", tracking_state.p1_detected, self.colors.player_primary),
                (p2_cx, "PLAYER 2", tracking_state.p2_detected, self.colors.ai_primary),
            ]:
                pygame.draw.circle(self.screen, (25, 32, 50), (cx, center_y), radius, 2)
                glow_col = self.colors.player_glow if col == self.colors.player_primary else self.colors.ai_glow
                pygame.draw.circle(self.screen, glow_col, (cx, center_y), radius + 6, 1)

                if sweep_angle > 0:
                    rect = pygame.Rect(cx - radius, center_y - radius, radius * 2, radius * 2)
                    pygame.draw.arc(self.screen, col, rect, 0, math.radians(sweep_angle), 4)

                cd_surf = self.renderer.font_title.render(str(remaining), True, col)
                self.screen.blit(cd_surf, cd_surf.get_rect(center=(cx, center_y)))

                name_surf = self.renderer.font_hud.render(label, True, col)
                self.screen.blit(name_surf, name_surf.get_rect(center=(cx, center_y - radius - 18)))

                lock_str = "LOCKED" if detected else "SEARCHING..."
                lock_col = col if detected else self.colors.danger
                lbl = self.renderer.font_hud.render(lock_str, True, lock_col)
                self.screen.blit(lbl, lbl.get_rect(center=(cx, center_y + radius + 25)))
        else:
            # Circular Cyber Reticle in center
            radius = 75
            pygame.draw.circle(self.screen, (25, 32, 50), (center_x, center_y), radius, 2)
            pygame.draw.circle(self.screen, self.colors.player_glow, (center_x, center_y), radius + 8, 1)

            if sweep_angle > 0:
                rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
                pygame.draw.arc(self.screen, self.colors.player_primary, rect, 0, math.radians(sweep_angle), 4)

            cd_surf = self.renderer.font_title.render(str(remaining), True, self.colors.player_primary)
            self.screen.blit(cd_surf, cd_surf.get_rect(center=(center_x, center_y)))

            lock_status = "POSITION LOCKED • HOLD STILL" if tracking_state.face_detected else "SEARCHING FOR FACE..."
            col = self.colors.player_primary if tracking_state.face_detected else self.colors.danger
            lbl = self.renderer.font_hud.render(lock_status, True, col)
            self.screen.blit(lbl, lbl.get_rect(center=(center_x, center_y + radius + 35)))

    def draw_game_over_screen(self, state_mgr: GameStateManager, tracking_state: TrackingState) -> None:
        """Renders match results and statistical telemetry inside arena."""
        arena = self.renderer.arena_rect
        center_x = arena.centerx

        if state_mgr.is_pvp:
            winner_p1 = (state_mgr.winner == "PLAYER 1")
            winner_color = self.colors.player_primary if winner_p1 else self.colors.ai_primary
            title_text = "PLAYER 1 VICTORIOUS!" if winner_p1 else "PLAYER 2 VICTORIOUS!"
        else:
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
        mode_str = "2-PLAYER HUMAN PvP" if state_mgr.is_pvp else "SOLO VS AUDACIA AI"

        stats = [
            f"MATCH MODE:            {mode_str}",
            f"MATCH DURATION:        {dur_str}",
            f"LONGEST RALLY:         {state_mgr.max_rally_match} VOLLEYS",
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
