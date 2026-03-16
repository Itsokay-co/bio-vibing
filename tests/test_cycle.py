"""Tests for lib/cycle.py — menstrual cycle detection."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from cycle import detect_cycle_phases
from conftest import make_readiness_record, make_sleep_record, date_range


class TestCycleDetection(unittest.TestCase):

    def test_insufficient_data(self):
        dates = date_range("2025-01-01", 10)
        readiness = [make_readiness_record(d) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        result = detect_cycle_phases(readiness, sleep)
        self.assertEqual(result.get("current_phase"), "unknown")

    def test_phases_from_tags(self):
        dates = date_range("2025-01-01", 60)
        readiness = [make_readiness_record(d) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        tags = [
            {"day": "2025-01-01", "tag_type": "period", "comment": "period"},
            {"day": "2025-01-02", "tag_type": "period", "comment": "period"},
            {"day": "2025-01-29", "tag_type": "period", "comment": "period"},
            {"day": "2025-01-30", "tag_type": "period", "comment": "period"},
        ]
        result = detect_cycle_phases(readiness, sleep, period_tags=tags)
        self.assertIn(result.get("confidence"), ("high", "medium", "low"))

    def test_consecutive_period_days_grouped(self):
        dates = date_range("2025-01-01", 60)
        readiness = [make_readiness_record(d) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        # 5 consecutive days should be one period start
        tags = [{"day": f"2025-01-0{i}", "tag_type": "period", "comment": "period"}
                for i in range(1, 6)]
        result = detect_cycle_phases(readiness, sleep, period_tags=tags)
        # Should not crash, should detect something
        self.assertIn("current_phase", result)

    def test_biometric_detection_30d(self):
        dates = date_range("2025-01-01", 35)
        # Simulate temp shift at day 15 (follicular → luteal)
        readiness = []
        for i, d in enumerate(dates):
            temp = -0.2 if i < 15 else 0.3  # clear shift
            readiness.append(make_readiness_record(d, temp_deviation_c=temp))
        sleep = [make_sleep_record(d) for d in dates]
        result = detect_cycle_phases(readiness, sleep)
        self.assertIn("current_phase", result)

    def test_empty_inputs(self):
        result = detect_cycle_phases([], [])
        self.assertEqual(result.get("current_phase"), "unknown")


if __name__ == '__main__':
    unittest.main()
