"""Tests for Suna-dependent metrics: postmeal HR, caffeine, food effects, gut correlations."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_postmeal_hr_response, compute_caffeine_sleep_coupling,
                     compute_food_item_effects, compute_bdi_meal_coupling,
                     compute_gut_score_correlations, compute_digestive_state_biometrics)
from conftest import (make_sleep_record, make_heartrate_record, make_spo2_record,
                      generate_hr_timeseries, date_range)


def _make_meal(day, hour=12, meal_type="lunch", caffeine=0, **kw):
    rec = {"day": day, "provider": "test",
           "timestamp": f"{day}T{hour:02d}:00:00+00:00",
           "meal_type": meal_type, "calories": 500,
           "protein_g": 30, "carbs_g": 50, "fat_g": 20,
           "description": "test meal", "fiber_g": 5, "sugar_g": 10,
           "saturated_fat_g": 5, "alcohol_units": None,
           "sodium_mg": None, "iron_mg": None, "magnesium_mg": None,
           "caffeine_mg": caffeine, "foods": [{"name": "chicken", "calories": 300}]}
    rec.update(kw)
    return rec


def _make_gut_score(day, score=75):
    return {"day": day, "provider": "suna", "score": score,
            "level": "good" if score >= 75 else "normal",
            "components": {"overnight": {"value": 0.8, "contribution": "positive"}}}


def _make_digestive_state(day, duration=120):
    return {"day": day, "provider": "suna",
            "meal_time": f"{day}T12:00:00+00:00", "meal_type": "lunch",
            "duration_min": duration, "phases": {"phase_1": 30, "phase_2": 90},
            "confidence": 0.8}


class TestPostmealHr(unittest.TestCase):

    def test_empty(self):
        result = compute_postmeal_hr_response([], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 7)
        meals = [_make_meal(d) for d in dates]
        hr = []
        for d in dates:
            hr.extend(generate_hr_timeseries(d, count=288))
        result = compute_postmeal_hr_response(meals, hr)
        self.assertIsInstance(result, dict)

    def test_no_meals(self):
        hr = generate_hr_timeseries("2025-01-01", count=288)
        result = compute_postmeal_hr_response([], hr)
        self.assertIsInstance(result, dict)


class TestCaffeineSleep(unittest.TestCase):

    def test_empty(self):
        result = compute_caffeine_sleep_coupling([], [])
        self.assertIsInstance(result, dict)

    def test_with_caffeine(self):
        dates = date_range("2025-01-01", 14)
        meals = [_make_meal(d, caffeine=200) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_caffeine_sleep_coupling(meals, sleep)
        self.assertIsInstance(result, dict)

    def test_no_caffeine(self):
        dates = date_range("2025-01-01", 14)
        meals = [_make_meal(d, caffeine=0) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_caffeine_sleep_coupling(meals, sleep)
        self.assertIsInstance(result, dict)


class TestFoodEffects(unittest.TestCase):

    def test_empty(self):
        result = compute_food_item_effects([], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        meals = [_make_meal(d) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_food_item_effects(meals, sleep)
        self.assertIsInstance(result, dict)


class TestBdiMeal(unittest.TestCase):

    def test_empty(self):
        result = compute_bdi_meal_coupling([], [], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        spo2 = [make_spo2_record(d, breathing_disturbance_index=5.0) for d in dates]
        meals = [_make_meal(d) for d in dates]
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_bdi_meal_coupling(spo2, meals, sleep)
        self.assertIsInstance(result, dict)


class TestGutCorrelations(unittest.TestCase):

    def test_empty(self):
        result = compute_gut_score_correlations([], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        gut = [_make_gut_score(d, score=70 + i) for i, d in enumerate(dates)]
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_gut_score_correlations(gut, sleep)
        self.assertIsInstance(result, dict)


class TestDigestiveStateBiometrics(unittest.TestCase):

    def test_empty(self):
        result = compute_digestive_state_biometrics([], [], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        states = [_make_digestive_state(d, duration=100 + i * 10) for i, d in enumerate(dates)]
        hr = []
        for d in dates:
            hr.extend(generate_hr_timeseries(d, count=288))
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_digestive_state_biometrics(states, hr, sleep)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
