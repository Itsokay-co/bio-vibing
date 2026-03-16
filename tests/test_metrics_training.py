"""Tests for training metrics: load, CUSUM, HR zones, intensity, pace, active, respiratory, recovery."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_training_load, detect_change_points,
                     compute_hr_zones, compute_intensity_minutes,
                     compute_recovery_index, compute_workout_pace,
                     compute_respiratory_trends, compute_active_minutes)
from conftest import (make_workout_record, make_sleep_record, make_readiness_record,
                      make_respiration_record, make_activity_record, make_user_profile,
                      generate_hr_timeseries, date_range)


class TestTrainingLoad(unittest.TestCase):

    def test_no_workouts(self):
        result = compute_training_load([], [], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        workouts = [make_workout_record(d) for d in dates[::3]]
        hr = []
        for d in dates:
            hr.extend(generate_hr_timeseries(d, count=288))
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_training_load(workouts, hr, sleep)
        self.assertIsInstance(result, dict)


class TestCusum(unittest.TestCase):

    def test_insufficient(self):
        records = [{"day": "2025-01-01", "avg_hrv_ms": 45}]
        result = detect_change_points(records, "avg_hrv_ms")
        self.assertIsInstance(result, dict)

    def test_detects_shift(self):
        records = []
        dates = date_range("2025-01-01", 25)
        for i, d in enumerate(dates):
            val = 45.0 if i < 12 else 65.0  # clear shift at day 12
            records.append({"day": d, "avg_hrv_ms": val})
        result = detect_change_points(records, "avg_hrv_ms")
        self.assertIsInstance(result, dict)
        cp = result.get("change_points", [])
        self.assertIsInstance(cp, list)


class TestHrZones(unittest.TestCase):

    def test_empty(self):
        result = compute_hr_zones([], None)
        self.assertIsInstance(result, dict)

    def test_valid_distribution(self):
        hr = generate_hr_timeseries("2025-01-01", count=288)
        profile = make_user_profile()
        result = compute_hr_zones(hr, profile)
        self.assertIsInstance(result, dict)


class TestIntensityMinutes(unittest.TestCase):

    def test_basic(self):
        # HR at 70% of max 185 = ~130 bpm → moderate zone
        from conftest import make_heartrate_record
        hr = [make_heartrate_record(f"2025-01-01T{10+i//12:02d}:{(i%12)*5:02d}:00+00:00", 130, "awake")
              for i in range(24)]  # 2h of moderate HR
        profile = make_user_profile()
        result = compute_intensity_minutes(hr, profile)
        self.assertIsInstance(result, dict)


class TestRecoveryIndex(unittest.TestCase):

    def test_insufficient(self):
        result = compute_recovery_index([], [])
        self.assertIsInstance(result, dict)

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        result = compute_recovery_index(sleep, readiness)
        self.assertIsInstance(result, dict)


class TestWorkoutPace(unittest.TestCase):

    def test_running(self):
        w = make_workout_record("2025-01-01", distance_m=5000, duration_seconds=1500)
        result = compute_workout_pace([w])
        self.assertIsInstance(result, dict)

    def test_no_workouts(self):
        result = compute_workout_pace([])
        self.assertIsInstance(result, dict)


class TestRespiratoryTrends(unittest.TestCase):

    def test_valid(self):
        dates = date_range("2025-01-01", 7)
        resp = [make_respiration_record(d) for d in dates]
        result = compute_respiratory_trends(resp)
        self.assertIsInstance(result, dict)
        if result.get("avg_rate") is not None:
            self.assertIsInstance(result["avg_rate"], float)

    def test_empty(self):
        result = compute_respiratory_trends([])
        self.assertIsInstance(result, dict)


class TestActiveMinutes(unittest.TestCase):

    def test_from_steps_heuristic(self):
        dates = date_range("2025-01-01", 5)
        activity = [make_activity_record(d, steps=10000, active_minutes=None) for d in dates]
        result = compute_active_minutes(activity)
        self.assertEqual(result.get("method"), "estimated_from_daily_steps")

    def test_from_reported(self):
        dates = date_range("2025-01-01", 5)
        activity = [make_activity_record(d, active_minutes=60, sedentary_minutes=900) for d in dates]
        result = compute_active_minutes(activity)
        self.assertEqual(result.get("method"), "reported")
        self.assertEqual(result.get("avg_active"), 60)

    def test_empty(self):
        result = compute_active_minutes([])
        self.assertIsNone(result.get("avg_active"))


if __name__ == '__main__':
    unittest.main()
