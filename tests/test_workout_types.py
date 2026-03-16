"""Tests for lib/workout_types.py — normalize_workout_type."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from workout_types import normalize_workout_type


class TestWorkoutTypes(unittest.TestCase):

    def test_direct_match(self):
        self.assertEqual(normalize_workout_type("running", "oura"), "running")

    def test_apple_health_prefix(self):
        result = normalize_workout_type("HKWorkoutActivityTypeRunning", "apple_health")
        self.assertEqual(result, "running")

    def test_case_insensitive(self):
        result = normalize_workout_type("YOGA", "whoop")
        self.assertEqual(result, "yoga")

    def test_unknown_fallback(self):
        result = normalize_workout_type("zorbing_extreme", "oura")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_empty_string(self):
        result = normalize_workout_type("", "oura")
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()
