"""Procedural sound synthesizer for arcade audio without external asset dependencies."""

import logging
from typing import Dict, Optional
import numpy as np
import pygame

from facepong.config import AudioConfig

logger = logging.getLogger(__name__)


class SoundManager:
    """Generates and plays retro arcade sound effects procedurally."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self._enabled = config.enabled
        self._sounds: Dict[str, pygame.mixer.Sound] = {}

        if self._enabled:
            self._init_mixer()

    def _init_mixer(self) -> None:
        """Safely initializes pygame mixer with fallback if audio driver is absent."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.config.sample_rate, size=-16, channels=1, buffer=512)
            self._generate_arcade_sounds()
            logger.info("Audio synthesizer initialized successfully")
        except Exception as exc:
            logger.warning("Audio device unavailable (%s). Running in silent mode.", exc)
            self._enabled = False

    def _generate_tone(self, frequency: float, duration_sec: float, wave_type: str = "square") -> pygame.mixer.Sound:
        """Synthesizes a short waveform with an amplitude envelope to prevent clipping."""
        sample_rate = self.config.sample_rate
        total_samples = int(sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)

        if wave_type == "square":
            wave = np.sign(np.sin(2.0 * np.pi * frequency * t))
        elif wave_type == "sine":
            wave = np.sin(2.0 * np.pi * frequency * t)
        elif wave_type == "saw":
            wave = 2.0 * (t * frequency - np.floor(0.5 + t * frequency))
        else:
            wave = np.sin(2.0 * np.pi * frequency * t)

        # Apply exponential decay envelope
        decay = np.exp(-t * (4.0 / max(0.05, duration_sec)))
        audio_data = wave * decay * self.config.master_volume

        # Convert to 16-bit signed PCM
        int16_samples = (audio_data * 32767).astype(np.int16)

        mixer_init = pygame.mixer.get_init()
        if mixer_init and mixer_init[2] == 2:
            # Stereo: duplicate mono samples to both channels
            int16_samples = np.column_stack((int16_samples, int16_samples))

        sound = pygame.sndarray.make_sound(int16_samples)
        return sound

    def _generate_arcade_sounds(self) -> None:
        """Precomputes retro sound effects."""
        try:
            # High-pitch crisp pop for paddle collision
            self._sounds["paddle_hit"] = self._generate_tone(520.0, 0.07, wave_type="square")
            # Lower pitch blip for boundary bounce
            self._sounds["wall_hit"] = self._generate_tone(260.0, 0.05, wave_type="sine")
            # Positive high chime for player goal
            self._sounds["goal"] = self._generate_tone(780.0, 0.22, wave_type="square")
            # Low tone for opponent score
            self._sounds["opponent_goal"] = self._generate_tone(180.0, 0.25, wave_type="saw")
            # Calibration / countdown beep
            self._sounds["beep"] = self._generate_tone(880.0, 0.08, wave_type="sine")
            # Start game high chime
            self._sounds["start"] = self._generate_tone(1046.0, 0.18, wave_type="square")
        except Exception as exc:
            logger.warning("Failed to synthesize audio effects: %s", exc)
            self._enabled = False

    def play(self, sound_name: str) -> None:
        """Plays the designated sound effect if audio is active."""
        if not self._enabled:
            return
        sound = self._sounds.get(sound_name)
        if sound:
            try:
                sound.play()
            except Exception:
                pass
