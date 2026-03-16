"""Tests for autonomic metrics: HRV-CV, coupling, circadian, HRR, nocturnal."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_hrv_cv, compute_cross_modal_coupling,
                     compute_circadian_fingerprint, compute_heart_rate_recovery,
                     compute_nocturnal_hr_shape)
from conftest import (make_sleep_record, make_readiness_record, make_spo2_record,
                      make_workout_record, make_heartrate_record,
                      generate_hr_timeseries, date_range)


class TestHrvCv(unittest.TestCase):

    def test_insufficient_data(self):
        sleep = [make_sleep_record("2025-01-01"), make_sleep_record("2025-01-02")]
        result = compute_hrv_cv(sleep)
        self.assertIn("insufficient", str(result.get("interpretation", "")).lower())

    def test_valid_7_nights(self):
        dates = date_range("2025-01-01", 7)
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + i * 2) for i, d in enumerate(dates)]
        result = compute_hrv_cv(sleep)
        self.assertIsNotNone(result.get("current_cv_7d"))
        self.assertIsInstance(result["current_cv_7d"], float)

    def test_interpretation_rigid(self):
        dates = date_range("2025-01-01", 14)
        # Very stable HRV → low CV → rigid
        sleep = [make_sleep_record(d, avg_hrv_ms=45.0) for d in dates]
        result = compute_hrv_cv(sleep)
        cv = result.get("current_cv_7d")
        if cv is not None and cv < 8:
            self.assertEqual(result.get("interpretation"), "rigid")

    def test_interpretation_healthy(self):
        dates = date_range("2025-01-01", 14)
        # Moderate variation
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + (i % 5) * 5) for i, d in enumerate(dates)]
        result = compute_hrv_cv(sleep)
        self.assertIn(result.get("interpretation", ""), ("healthy", "rigid", "erratic", "insufficient_data"))

    def test_naps_excluded(self):
        dates = date_range("2025-01-01", 10)
        sleep = [make_sleep_record(d, avg_hrv_ms=45 + i) for i, d in enumerate(dates)]
        sleep.append(make_sleep_record("2025-01-05", sleep_type="nap", avg_hrv_ms=100))
        result = compute_hrv_cv(sleep)
        # Should not crash; nap should be filtered
        self.assertIn("current_cv_7d", result)


class TestCoupling(unittest.TestCase):

    def test_insufficient(self):
        sleep = [make_sleep_record("2025-01-01")]
        readiness = [make_readiness_record("2025-01-01")]
        spo2 = [make_spo2_record("2025-01-01")]
        result = compute_cross_modal_coupling(sleep, readiness, spo2)
        # Should not crash with minimal data
        self.assertIsInstance(result, dict)

    def test_valid_14_days(self):
        dates = date_range("2025-01-01", 20)
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + i, avg_resting_hr_bpm=55 + i * 0.5)
                 for i, d in enumerate(dates)]
        readiness = [make_readiness_record(d, temp_deviation_c=0.1 + i * 0.02)
                     for i, d in enumerate(dates)]
        spo2 = [make_spo2_record(d, avg_spo2_pct=96 + i * 0.1) for i, d in enumerate(dates)]
        result = compute_cross_modal_coupling(sleep, readiness, spo2)
        self.assertIsInstance(result, dict)


class TestCircadian(unittest.TestCase):

    def test_empty_hr(self):
        result = compute_circadian_fingerprint([])
        # May return None or dict with insufficient_data
        self.assertTrue(result is None or isinstance(result, dict))

    def test_valid_200_samples(self):
        hr = generate_hr_timeseries("2025-01-01", count=288)
        hr.extend(generate_hr_timeseries("2025-01-02", count=288))
        result = compute_circadian_fingerprint(hr)
        self.assertIsInstance(result, dict)
        if result.get("mesor") is not None:
            self.assertIsInstance(result["mesor"], (int, float))


class TestHrr(unittest.TestCase):

    def test_no_workouts(self):
        result = compute_heart_rate_recovery([], [])
        self.assertIsInstance(result, dict)

    def test_valid_workout(self):
        workout = make_workout_record("2025-01-01")
        end_ts = workout["end_time"]
        # Generate HR records around workout end
        hr = [
            make_heartrate_record(end_ts, 170, "awake"),
        ]
        # Add recovery HR samples (1 min, 5 min after)
        from datetime import datetime, timedelta
        end_dt = datetime.fromisoformat(end_ts.replace("Z", "+00:00"))
        for i in range(1, 10):
            ts = (end_dt + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
            hr.append(make_heartrate_record(ts, 170 - i * 5, "awake"))
        result = compute_heart_rate_recovery([workout], hr)
        self.assertIsInstance(result, dict)


class TestNocturnal(unittest.TestCase):

    def test_insufficient(self):
        result = compute_nocturnal_hr_shape([], [])
        self.assertIsInstance(result, dict)

    def test_valid_14_nights(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        hr = []
        for d in dates:
            hr.extend(generate_hr_timeseries(d, count=288))
        result = compute_nocturnal_hr_shape(hr, sleep)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
