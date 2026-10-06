"""Game state management and exhibition presence watchdog."""

from enum import Enum, auto
import time
from typing import Optional


class ExhibitionState(Enum):
    ATTRACT = auto()      # Idle attractor mode, displays promotional demo
    CALIBRATING = auto()  # 3-second face detection calibration
    PLAYING = auto()      # Active match
    POINT_SCORED = auto() # Brief pause after a goal before re-serving
    GAME_OVER = auto()    # Match summary screen


class GameStateManager:
    """Oversees exhibition state transitions, match scores, and presence timeouts."""

    def __init__(self, winning_score: int = 5, inactivity_timeout_sec: float = 5.0):
        self.winning_score = winning_score
        self.inactivity_timeout_sec = inactivity_timeout_sec

        self.current_state = ExhibitionState.ATTRACT
        self.player_score = 0
        self.ai_score = 0
        self.rally_count = 0
        self.max_rally_match = 0
        self.winner: Optional[str] = None  # "PLAYER" or "AI"

        # Timers
        self.state_enter_time = time.perf_counter()
        self.pause_timer = 0.0
        self.inactivity_warning_timer = 0.0
        self.match_start_time = 0.0
        self.match_duration_sec = 0.0

        # Face detection stability in attract mode
        self._face_seen_start_time: Optional[float] = None

    def change_state(self, new_state: ExhibitionState) -> None:
        """Transitions into a new exhibition state."""
        self.current_state = new_state
        self.state_enter_time = time.perf_counter()

        if new_state == ExhibitionState.PLAYING and self.match_start_time == 0.0:
            self.match_start_time = time.perf_counter()

    def start_new_match(self) -> None:
        """Resets scores and statistics for a fresh contest."""
        self.player_score = 0
        self.ai_score = 0
        self.rally_count = 0
        self.max_rally_match = 0
        self.winner = None
        self.match_start_time = time.perf_counter()
        self.match_duration_sec = 0.0

    def record_goal(self, scorer: str) -> bool:
        """
        Registers a point scored.
        
        Returns:
            True if match has reached winning score.
        """
        if scorer == "PLAYER":
            self.player_score += 1
        else:
            self.ai_score += 1

        self.rally_count = 0

        if self.player_score >= self.winning_score:
            self.winner = "PLAYER"
            self.match_duration_sec = time.perf_counter() - self.match_start_time
            return True
        elif self.ai_score >= self.winning_score:
            self.winner = "AI"
            self.match_duration_sec = time.perf_counter() - self.match_start_time
            return True

        return False

    def increment_rally(self) -> None:
        """Tracks consecutive paddle exchanges."""
        self.rally_count += 1
        if self.rally_count > self.max_rally_match:
            self.max_rally_match = self.rally_count

    def update_watchdog(self, face_detected: bool, last_detected_time: float, dt: float) -> bool:
        """
        Monitors player presence.
        
        Returns:
            True if the match timed out due to player abandonment and was reset.
        """
        now = time.perf_counter()

        # In attract mode: trigger calibration if a face is continuously detected for >0.8s
        if self.current_state == ExhibitionState.ATTRACT:
            if face_detected:
                if self._face_seen_start_time is None:
                    self._face_seen_start_time = now
                elif now - self._face_seen_start_time >= 0.8:
                    self._face_seen_start_time = None
                    self.change_state(ExhibitionState.CALIBRATING)
            else:
                self._face_seen_start_time = None
            return False

        # In active gameplay or calibration: check for abandoned exhibition
        if self.current_state in (ExhibitionState.PLAYING, ExhibitionState.POINT_SCORED, ExhibitionState.CALIBRATING):
            time_since_seen = now - last_detected_time
            if not face_detected and last_detected_time > 0 and time_since_seen > self.inactivity_timeout_sec:
                # Player left the station: abort match and return to attract mode
                self.reset_to_attract()
                return True

        # In Game Over: return to attract mode after 7 seconds or after 3.5s if player leaves
        if self.current_state == ExhibitionState.GAME_OVER:
            state_time = now - self.state_enter_time
            if state_time >= 7.0:
                self.reset_to_attract()
            elif not face_detected and state_time >= 3.5:
                self.reset_to_attract()

        return False

    def reset_to_attract(self) -> None:
        """Aborts current game and returns system to idle attract screen."""
        self.player_score = 0
        self.ai_score = 0
        self.rally_count = 0
        self.winner = None
        self._face_seen_start_time = None
        self.change_state(ExhibitionState.ATTRACT)
