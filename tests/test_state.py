"""Unit tests for exhibition flow state machine and presence watchdog."""

import unittest
import time
from facepong.game.state import ExhibitionState, GameStateManager


class TestGameStateManager(unittest.TestCase):
    def setUp(self):
        self.mgr = GameStateManager(winning_score=3, inactivity_timeout_sec=2.0)

    def test_initial_state_is_attract(self):
        self.assertEqual(self.mgr.current_state, ExhibitionState.ATTRACT)

    def test_scoring_and_win_condition(self):
        self.mgr.start_new_match()
        self.mgr.change_state(ExhibitionState.PLAYING)

        over = self.mgr.record_goal("PLAYER")
        self.assertFalse(over)
        self.assertEqual(self.mgr.player_score, 1)

        self.mgr.record_goal("PLAYER")
        over = self.mgr.record_goal("PLAYER")  # 3 goals reached
        self.assertTrue(over)
        self.assertEqual(self.mgr.winner, "PLAYER")

    def test_watchdog_resets_on_prolonged_inactivity(self):
        self.mgr.start_new_match()
        self.mgr.change_state(ExhibitionState.PLAYING)

        # Simulate player disappearance 3 seconds ago (timeout is 2.0s)
        now = time.perf_counter()
        stale_time = now - 3.5

        timed_out = self.mgr.update_watchdog(
            face_detected=False,
            last_detected_time=stale_time,
            dt=0.016,
        )
        self.assertTrue(timed_out)
        self.assertEqual(self.mgr.current_state, ExhibitionState.ATTRACT)


if __name__ == "__main__":
    unittest.main()
