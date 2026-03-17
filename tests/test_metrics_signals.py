"""Tests for signal metrics: baselines, forward signals, stress/inflammation proxies, discovery."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_personal_baselines, compute_forward_signals,
                     compute_stress_proxy, compute_inflammation_proxy,
                     compute_correlation_discovery)
from conftest import (make_sleep_record, make_readiness_record, make_spo2_record,
                      make_stress_record, make_workout_record, make_respiration_record,
                      make_biometric_data, date_range, _deterministic_float)


class TestPersonalBaselines(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 3)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_personal_baselines(sleep, [], [], [], [])
        self.assertIsInstance(result, dict)

    def test_valid_30d(self):
        dates = date_range("2025-01-01", 30)
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + _deterministic_float(f"bl-{d}", 0, 20))
                 for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        spo2 = [make_spo2_record(d) for d in dates]
        stress = [make_stress_record(d) for d in dates]
        resp = [make_respiration_record(d) for d in dates]
        result = compute_personal_baselines(sleep, readiness, spo2, stress, resp)
        self.assertIsInstance(result, dict)


class TestForwardSignals(unittest.TestCase):

    def test_insufficient(self):
        result = compute_forward_signals([], [], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        workouts = [make_workout_record(dates[i]) for i in range(0, 14, 3)]
        result = compute_forward_signals(sleep, readiness, workouts)
        self.assertIsInstance(result, dict)


class TestStressProxy(unittest.TestCase):

    def test_empty(self):
        result = compute_stress_proxy([], [], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        meals = [{"day": d, "provider": "test", "timestamp": f"{d}T19:00:00+00:00",
                  "meal_type": "dinner", "calories": 600, "protein_g": 30,
                  "carbs_g": 60, "fat_g": 20, "description": None,
                  "fiber_g": None, "sugar_g": None, "saturated_fat_g": None,
                  "alcohol_units": None, "sodium_mg": None, "iron_mg": None,
                  "magnesium_mg": None, "caffeine_mg": None, "foods": None}
                 for d in dates]
        result = compute_stress_proxy(sleep, readiness, meals)
        self.assertIsInstance(result, dict)


class TestInflammationProxy(unittest.TestCase):

    def test_empty(self):
        result = compute_inflammation_proxy([], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        result = compute_inflammation_proxy(sleep, readiness)
        self.assertIsInstance(result, dict)


class TestCorrelationDiscovery(unittest.TestCase):

    def test_insufficient(self):
        data = {"sleep": [], "readiness": [], "activity": [], "heartrate": [],
                "workouts": [], "meals": [], "stress": [], "spo2": [],
                "gut_scores": [], "glucose": []}
        result = compute_correlation_discovery(data)
        self.assertIsInstance(result, dict)

    def test_valid_rich_data(self):
        data = make_biometric_data(days=30)
        result = compute_correlation_discovery(data)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
