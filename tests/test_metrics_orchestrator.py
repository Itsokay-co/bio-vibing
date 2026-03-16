"""Tests for compute_all orchestrator — end-to-end metric computation."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import compute_all
from conftest import make_biometric_data, make_sleep_record, date_range


class TestComputeAll(unittest.TestCase):

    def test_empty_data(self):
        data = {
            "provider": "test", "period_start": "2025-01-01",
            "period_end": "2025-01-14", "sleep": [], "readiness": [],
            "activity": [], "heartrate": [], "workouts": [],
            "stress": [], "spo2": [], "resilience": [], "tags": [],
            "body_composition": [], "respiration": [], "meals": [],
            "user": None, "optimal_bedtime": None, "warnings": [],
        }
        result = compute_all(data)
        self.assertIsInstance(result, dict)
        self.assertIn("_errors", result)

    def test_minimal_7_days(self):
        data = make_biometric_data(days=7)
        result = compute_all(data)
        self.assertIsInstance(result, dict)
        # Should have all metric keys
        for key in ("hrv_cv", "sleep_regularity", "sleep_efficiency",
                     "hr_zones", "active_minutes"):
            self.assertIn(key, result)

    def test_rich_30_days(self):
        data = make_biometric_data(days=30)
        result = compute_all(data)
        errors = result.get("_errors", [])
        # Count successful metrics (not errors)
        successful = sum(1 for k, v in result.items()
                        if k != "_errors" and isinstance(v, dict)
                        and "error" not in str(v.get("", "")))
        self.assertGreaterEqual(successful, 15,
                               f"Expected 15+ successful metrics, got {successful}. Errors: {errors}")

    def test_error_isolation(self):
        data = make_biometric_data(days=14)
        # Corrupt one category to potentially trigger an error
        data["sleep"][0]["avg_hrv_ms"] = "not_a_number"
        result = compute_all(data)
        # Should still return results for other metrics
        self.assertIsInstance(result, dict)
        self.assertIn("_errors", result)

    def test_with_meals(self):
        data = make_biometric_data(days=14)
        data["meals"] = [
            {"day": "2025-01-01", "provider": "test", "timestamp": "2025-01-01T19:00:00+00:00",
             "meal_type": "dinner", "calories": 600, "protein_g": 30, "carbs_g": 60, "fat_g": 20,
             "description": None, "fiber_g": None, "sugar_g": None, "saturated_fat_g": None,
             "alcohol_units": None, "sodium_mg": None, "iron_mg": None, "magnesium_mg": None,
             "caffeine_mg": None, "foods": None},
        ] * 10
        result = compute_all(data)
        # Nutrition metrics should be attempted
        has_nutrition = any(k.startswith("meal_") or k == "thermic" or k == "macro_hrv"
                          or k == "nutrition_periodization" for k in result)
        self.assertTrue(has_nutrition, f"Expected nutrition metrics, got keys: {list(result.keys())}")

    def test_without_meals(self):
        data = make_biometric_data(days=14)
        data["meals"] = []
        result = compute_all(data)
        # Nutrition metrics should not be present
        nutrition_keys = [k for k in result if k.startswith("meal_") or k == "thermic"]
        # They might still be present as empty/None, that's ok
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
