"""Tests for glucose metrics: variability, postmeal response, patterns."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_glucose_variability, compute_postmeal_glucose,
                     compute_glucose_patterns)
from conftest import date_range


def _make_glucose(timestamp, value, trend="flat"):
    return {"timestamp": timestamp, "provider": "test",
            "value_mgdl": value, "trend": trend, "trend_rate": None}


def _make_meal(day, hour=12, meal_type="lunch"):
    return {"day": day, "provider": "test",
            "timestamp": f"{day}T{hour:02d}:00:00+00:00",
            "meal_type": meal_type, "calories": 500,
            "protein_g": 30, "carbs_g": 50, "fat_g": 20,
            "description": None, "fiber_g": None, "sugar_g": None,
            "saturated_fat_g": None, "alcohol_units": None,
            "sodium_mg": None, "iron_mg": None, "magnesium_mg": None,
            "caffeine_mg": None, "foods": None}


class TestGlucoseVariability(unittest.TestCase):

    def test_insufficient(self):
        glucose = [_make_glucose("2025-01-01T12:00:00Z", 100)]
        result = compute_glucose_variability(glucose)
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_stable(self):
        # 100 readings around 100 mg/dL with minimal variation
        glucose = [_make_glucose(f"2025-01-01T{i//12:02d}:{(i%12)*5:02d}:00Z",
                                 98 + (i % 5)) for i in range(100)]
        result = compute_glucose_variability(glucose)
        self.assertIsNotNone(result['mean'])
        self.assertIsNotNone(result['cv'])
        self.assertEqual(result['interpretation'], 'stable')

    def test_time_in_range(self):
        # Mix of in-range and out-of-range
        glucose = ([_make_glucose(f"2025-01-01T{i:02d}:00:00Z", 110) for i in range(20)]
                   + [_make_glucose(f"2025-01-01T{20+i}:00:00Z", 200) for i in range(4)])
        result = compute_glucose_variability(glucose)
        self.assertIsNotNone(result['time_in_range_pct'])
        self.assertGreater(result['time_in_range_pct'], 50)
        self.assertGreater(result['time_above_pct'], 0)

    def test_empty(self):
        result = compute_glucose_variability([])
        self.assertEqual(result['interpretation'], 'insufficient_data')


class TestPostmealGlucose(unittest.TestCase):

    def test_insufficient(self):
        result = compute_postmeal_glucose([], [])
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_valid_response(self):
        # Baseline ~100, post-meal spike to 150 at 45min, recovery to 110 at 2h
        meal = _make_meal("2025-01-01", hour=12)
        glucose = []
        # Pre-meal baseline (30 min before)
        for m in range(6):
            glucose.append(_make_glucose(f"2025-01-01T11:{30+m*5:02d}:00+00:00", 100))
        # Post-meal rise and fall over 3h
        for m in range(36):
            mins = m * 5
            if mins < 45:
                val = 100 + (50 * mins / 45)  # rise to 150
            else:
                val = 150 - (40 * (mins - 45) / 135)  # fall to ~110
            glucose.append(_make_glucose(f"2025-01-01T12:{mins % 60:02d}:00+00:00"
                                        if mins < 60 else
                                        f"2025-01-01T{12 + mins//60}:{mins % 60:02d}:00+00:00",
                                        round(val, 1)))
        result = compute_postmeal_glucose(glucose, [meal])
        self.assertGreater(len(result['responses']), 0)
        self.assertIsNotNone(result['avg_peak_delta'])
        self.assertGreater(result['avg_peak_delta'], 20)

    def test_no_meals(self):
        glucose = [_make_glucose(f"2025-01-01T12:{i*5:02d}:00Z", 100) for i in range(12)]
        result = compute_postmeal_glucose(glucose, [])
        self.assertEqual(result['interpretation'], 'insufficient_data')


class TestGlucosePatterns(unittest.TestCase):

    def test_empty(self):
        result = compute_glucose_patterns([])
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_dawn_effect(self):
        glucose = []
        # Nocturnal (00-06): low glucose ~85
        for h in range(6):
            for m in range(0, 60, 5):
                glucose.append(_make_glucose(f"2025-01-01T{h:02d}:{m:02d}:00+00:00", 85))
        # Dawn (04-08): rising to 105
        for h in range(6, 8):
            for m in range(0, 60, 5):
                glucose.append(_make_glucose(f"2025-01-01T{h:02d}:{m:02d}:00+00:00", 105))
        # Daytime (08-22): ~110
        for h in range(8, 22):
            for m in range(0, 60, 5):
                glucose.append(_make_glucose(f"2025-01-01T{h:02d}:{m:02d}:00+00:00", 110))
        result = compute_glucose_patterns(glucose)
        self.assertIsNotNone(result['nocturnal_avg'])
        self.assertIsNotNone(result['dawn_avg'])
        self.assertIsNotNone(result['daytime_avg'])

    def test_no_dawn_with_few_readings(self):
        glucose = [_make_glucose(f"2025-01-01T12:{i*5:02d}:00Z", 100) for i in range(3)]
        result = compute_glucose_patterns(glucose)
        self.assertIsNone(result['nocturnal_avg'])


if __name__ == '__main__':
    unittest.main()
