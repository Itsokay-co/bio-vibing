"""Tests for OpenWearablesProvider._build_hypnogram — pure static method."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from providers.open_wearables import OpenWearablesProvider


class TestBuildHypnogram(unittest.TestCase):

    build = staticmethod(OpenWearablesProvider._build_hypnogram)

    def test_basic_intervals(self):
        intervals = [
            {"stage": "deep", "start_time": "2025-01-01T23:00:00Z", "end_time": "2025-01-01T23:30:00Z"},
            {"stage": "light", "start_time": "2025-01-01T23:30:00Z", "end_time": "2025-01-02T00:00:00Z"},
            {"stage": "rem", "start_time": "2025-01-02T00:00:00Z", "end_time": "2025-01-02T00:30:00Z"},
        ]
        h = self.build(intervals, "2025-01-01T23:00:00Z", "2025-01-02T00:30:00Z")
        self.assertIsNotNone(h)
        self.assertEqual(len(h), 18)  # 90min / 5min = 18 slots
        # First 6 should be deep (1), next 6 light (2), last 6 REM (3)
        self.assertEqual(h[:6], "111111")
        self.assertEqual(h[6:12], "222222")
        self.assertEqual(h[12:], "333333")

    def test_empty_intervals(self):
        h = self.build([], "2025-01-01T23:00:00Z", "2025-01-02T07:00:00Z")
        self.assertIsNone(h)

    def test_none_intervals(self):
        h = self.build(None, "2025-01-01T23:00:00Z", "2025-01-02T07:00:00Z")
        self.assertIsNone(h)

    def test_missing_timestamps(self):
        h = self.build([{"stage": "deep"}], None, None)
        self.assertIsNone(h)

    def test_gap_defaults_awake(self):
        intervals = [
            {"stage": "deep", "start_time": "2025-01-01T23:00:00Z", "end_time": "2025-01-01T23:15:00Z"},
            # 15min gap here
            {"stage": "light", "start_time": "2025-01-01T23:30:00Z", "end_time": "2025-01-01T23:45:00Z"},
        ]
        h = self.build(intervals, "2025-01-01T23:00:00Z", "2025-01-01T23:45:00Z")
        self.assertIsNotNone(h)
        # slots: 0-15min=deep(111), 15-30min=gap=awake(444), 30-45min=light(222)
        self.assertEqual(h, "111444222")

    def test_unknown_stage_maps_awake(self):
        intervals = [
            {"stage": "unknown", "start_time": "2025-01-01T23:00:00Z", "end_time": "2025-01-01T23:10:00Z"},
        ]
        h = self.build(intervals, "2025-01-01T23:00:00Z", "2025-01-01T23:10:00Z")
        self.assertEqual(h, "44")


if __name__ == '__main__':
    unittest.main()
