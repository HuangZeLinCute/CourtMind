"""Behavioral regression tests; these do not establish real-video accuracy."""
import unittest
from unittest.mock import Mock

import numpy as np

from badminton_analysis.shuttlecock.base import ShuttleDetection as D
from badminton_analysis.shuttlecock.temporal_ensemble import TemporalEnsemble
from badminton_analysis.detection.shuttlecock import ShuttlecockTracker
from badminton_analysis.system import BadmintonAnalysisSystem


def consensus(x, y):
    return [("tracknet", D(x, y, True, .5)),
            ("yolo", D(x + 2, y + 1, True, .7))]


class FusionTests(unittest.TestCase):
    def test_outlier_not_averaged_into_pair(self):
        t = TemporalEnsemble((1920, 1080))
        result = t.update(consensus(500, 400) + [("tracknet_v4", D(900, 600, True, .99))], 1)
        self.assertTrue(result.visible)
        self.assertLess(abs(result.x - 501), 3)
        self.assertEqual(t.diagnostics['sources'], ['tracknet', 'yolo'])

    def test_single_frame_false_positive_is_not_emitted(self):
        t = TemporalEnsemble((1920, 1080))
        self.assertFalse(t.update([('yolo', D(500, 400, True, .99))], 1).visible)
        self.assertFalse(t.update([], 2).visible)

    def test_same_source_duplicates_cannot_vote(self):
        t = TemporalEnsemble((1920, 1080))
        result = t.update([('yolo', D(500 + i, 400, True, .9)) for i in range(20)], 1)
        self.assertFalse(result.visible)
        self.assertLessEqual(t.diagnostics['candidates'], 5)

    def test_lower_confidence_yolo_candidate_can_win_by_agreement(self):
        t = TemporalEnsemble((1920, 1080))
        result = t.update(consensus(500, 400) + [('yolo', D(900, 900, True, .99))], 1)
        self.assertTrue(result.visible)
        self.assertLess(result.x, 510)

    def test_ambiguous_distinct_pairs_abstain(self):
        t = TemporalEnsemble((1920, 1080))
        result = t.update(consensus(400, 400) + consensus(1000, 600), 1)
        self.assertFalse(result.visible)
        self.assertEqual(t.diagnostics['reason'], 'ambiguous')

    def test_close_chain_does_not_count_as_three_way_consensus(self):
        t = TemporalEnsemble((1920, 1080))
        measurements = t._measurements([('tracknet', D(500, 400, True, .5)),
            ('yolo', D(530, 400, True, .7)), ('tracknet_v4', D(560, 400, True, .8))])
        self.assertFalse(any(len(m[3]) == 3 for m in measurements))

    def test_missing_frames_not_fabricated_and_reacquired(self):
        t = TemporalEnsemble((1920, 1080))
        for frame in range(1, 6):
            self.assertTrue(t.update(consensus(300 + frame * 20, 400), frame).visible)
        self.assertFalse(t.update([], 6).visible)
        self.assertFalse(t.update([], 7).visible)
        result = t.update(consensus(460, 400), 8)
        self.assertTrue(result.visible)
        self.assertLess(abs(result.x - 461), 3)

    def test_sharp_turn_reacquires_without_smoothing_lag(self):
        t = TemporalEnsemble((1920, 1080))
        for frame in range(1, 8):
            t.update(consensus(300 + frame * 30, 500), frame)
        results = [t.update(consensus(510 - step * 30, 500 - step * 10), 7 + step)
                   for step in range(1, 5)]
        self.assertTrue(results[-1].visible)
        self.assertLess(abs(results[-1].x - 391), 3)

    def test_resolution_invariance(self):
        small, large = TemporalEnsemble((960, 540)), TemporalEnsemble((1920, 1080))
        for frame in range(1, 8):
            ds = consensus(150 + frame * 10, 200)
            doubled = [(s, D(d.x * 2, d.y * 2, d.visible, d.confidence)) for s, d in ds]
            a, b = small.update(ds, frame), large.update(doubled, frame)
            self.assertEqual(a.visible, b.visible)
            self.assertAlmostEqual(a.x * 2, b.x)

    def test_invalid_values_and_long_gap_reset(self):
        t = TemporalEnsemble((1920, 1080))
        self.assertFalse(t.update([('yolo', D(float('nan'), 400, True, .8)),
            ('tracknet', D(-3, 400, True, .5))], 1).visible)
        for i in range(2, 5):
            t.update(consensus(500, 400), i)
        r = t.update([('yolo', D(500, 400, True, .9))], 40)
        self.assertFalse(r.visible)
        self.assertEqual(t.diagnostics['reason'], 'unconfirmed_single_source')

    def test_motion_gate_not_applied_twice(self):
        tracker = ShuttlecockTracker(None)
        tracker.set_detection(200, 400)
        tracker.update_trajectory([200, 400])
        tracker.set_detection(600, 400)
        self.assertEqual(tracker.update_trajectory([600, 400], temporal_validated=True), [600, 400])

    def test_system_uses_all_candidates_and_one_based_alignment(self):
        system = BadmintonAnalysisSystem.__new__(BadmintonAnalysisSystem)
        system.frame_width, system.frame_height, system.fps = 1920, 1080, 30
        system.ensemble_tracker = None
        system.tracknet_trajectories = {'tracknet': [D(500, 400, True, .5)],
                                      'tracknet_v4': [D(1000, 600, True, .9)]}
        system.shuttlecock_tracker = ShuttlecockTracker(None)
        system.shuttlecock_tracker.detect_candidates = Mock(return_value=[
            {'point': (100, 800), 'confidence': .99},
            {'point': (501, 401), 'confidence': .7}])
        result = system._ensemble_detection(np.zeros((1080, 1920, 3), dtype=np.uint8), None, 1)
        self.assertLess(abs(result[0] - 501), 3)
        system.shuttlecock_tracker.detect_candidates.assert_called_once()


if __name__ == '__main__':
    unittest.main()
