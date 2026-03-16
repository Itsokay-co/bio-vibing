"""Tests for lib/validation.py — dedup, outliers, gaps."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from validation import validate_data
from conftest import make_sleep_record, make_biometric_data


class TestValidation(unittest.TestCase):

    def test_dedup_same_day_provider(self):
        data = make_biometric_data(days=3)
        # Duplicate sleep record
        data["sleep"].append(data["sleep"][0].copy())
        result = validate_data(data)
        v = result.get("_validation", {})
        self.assertGreaterEqual(v.get("duplicates_removed", 0), 1)

    def test_outlier_flagging(self):
        data = make_biometric_data(days=3)
        data["sleep"][0]["avg_hrv_ms"] = 600  # > 500 limit
        result = validate_data(data)
        v = result.get("_validation", {})
        self.assertGreater(v.get("outliers_flagged", 0), 0)

    def test_gap_detection(self):
        data = make_biometric_data(days=2)
        data["sleep"] = [
            make_sleep_record("2025-01-01"),
            make_sleep_record("2025-01-10"),
        ]
        result = validate_data(data)
        v = result.get("_validation", {})
        warnings = v.get("warnings", [])
        has_gap = any("gap" in w.lower() for w in warnings)
        self.assertTrue(has_gap, f"Expected gap warning, got: {warnings}")

    def test_empty_passthrough(self):
        data = {
            "provider": "test", "period_start": "2025-01-01",
            "period_end": "2025-01-14", "sleep": [], "readiness": [],
            "activity": [], "heartrate": [], "workouts": [],
            "stress": [], "spo2": [], "resilience": [], "tags": [],
            "body_composition": [], "respiration": [], "meals": [],
            "user": None, "optimal_bedtime": None, "warnings": [],
        }
        result = validate_data(data)
        v = result.get("_validation", {})
        self.assertEqual(v.get("duplicates_removed", 0), 0)

    def test_sorted_by_date(self):
        data = make_biometric_data(days=5)
        data["sleep"].reverse()
        result = validate_data(data)
        days = [r["day"] for r in result["sleep"]]
        self.assertEqual(days, sorted(days))


if __name__ == '__main__':
    unittest.main()
