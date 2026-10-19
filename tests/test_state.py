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

    def test_single_face_waits_for_detection_window(self):
        # Single face detected does NOT immediately trigger calibration at 0.9s
        now = time.perf_counter()
        self.mgr.update_watchdog(face_detected=True, last_detected_time=now, dt=0.016, face_count=1)
        self.mgr._face_seen_start_time = now - 0.9
        self.mgr.boot_time = now - 0.9
        self.mgr.update_watchdog(face_detected=True, last_detected_time=now, dt=0.016, face_count=1)
        self.assertEqual(self.mgr.current_state, ExhibitionState.ATTRACT)

        # After 15s window has elapsed, single face starts 1P vs AI match
        self.mgr._face_seen_start_time = now - 15.1
        self.mgr.boot_time = now - 15.1
        self.mgr.update_watchdog(face_detected=True, last_detected_time=now, dt=0.016, face_count=1)
        self.assertEqual(self.mgr.current_state, ExhibitionState.CALIBRATING)
        self.assertFalse(self.mgr.is_pvp)

    def test_watchdog_auto_detects_pvp_when_two_faces_present(self):
        # Two faces detected triggers PvP after 0.8s stability without waiting 15s
        now = time.perf_counter()
        self.mgr.update_watchdog(face_detected=True, last_detected_time=now, dt=0.016, face_count=2)
        self.mgr._two_faces_seen_start_time = now - 0.9
        self.mgr.update_watchdog(face_detected=True, last_detected_time=now, dt=0.016, face_count=2)
        self.assertEqual(self.mgr.current_state, ExhibitionState.CALIBRATING)
        self.assertTrue(self.mgr.is_pvp)

    def test_attract_countdown_remaining_reports_active_timer(self):
        now = time.perf_counter()
        self.mgr.boot_time = now - 5.0
        self.mgr.solo_start_delay_sec = 15.0
        rem = self.mgr.attract_countdown_remaining
        self.assertIsNotNone(rem)
        self.assertAlmostEqual(rem, 10.0, delta=0.5)

    def test_pvp_goal_attribution_and_winner(self):
        self.mgr.is_pvp = True
        self.mgr.start_new_match()
        self.mgr.change_state(ExhibitionState.PLAYING)

        over = self.mgr.record_goal("PLAYER 1")
        self.assertFalse(over)
        self.assertEqual(self.mgr.player_score, 1)

        over = self.mgr.record_goal("PLAYER 2")
        self.assertFalse(over)
        self.assertEqual(self.mgr.ai_score, 1)

        self.mgr.record_goal("PLAYER 1")
        over = self.mgr.record_goal("PLAYER 1")
        self.assertTrue(over)
        self.assertEqual(self.mgr.winner, "PLAYER 1")

    def test_reset_to_attract_clears_pvp(self):
        self.mgr.is_pvp = True
        self.mgr.reset_to_attract()
        self.assertFalse(self.mgr.is_pvp)
        self.assertEqual(self.mgr.current_state, ExhibitionState.ATTRACT)


if __name__ == "__main__":
    unittest.main()
