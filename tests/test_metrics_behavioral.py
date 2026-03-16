"""Tests for behavioral metrics: alcohol, allostatic, early warning, entropy, temp amplitude."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_alcohol_detection, compute_allostatic_load,
                     compute_early_warning_signals, compute_daily_entropy,
                     compute_temp_amplitude_trend)
from conftest import (make_sleep_record, make_readiness_record, make_spo2_record,
                      make_stress_record, date_range, _deterministic_float)


class TestAlcohol(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 10)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_alcohol_detection(sleep)
        self.assertIsInstance(result, dict)

    def test_valid_with_anomaly(self):
        dates = date_range("2025-01-01", 20)
        sleep = [make_sleep_record(d, avg_hrv_ms=45, avg_resting_hr_bpm=58, efficiency=88)
                 for d in dates]
        # Inject anomalous night (alcohol-like: low HRV, high RHR, low eff)
        sleep[15] = make_sleep_record(dates[15], avg_hrv_ms=25, avg_resting_hr_bpm=72, efficiency=65)
        result = compute_alcohol_detection(sleep)
        self.assertIsInstance(result, dict)
        self.assertIn("flagged_nights", result)


class TestAllostatic(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 5)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_allostatic_load(sleep, [], [], [])
        self.assertIsInstance(result, dict)

    def test_low_load(self):
        dates = date_range("2025-01-01", 40)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        spo2 = [make_spo2_record(d) for d in dates]
        stress = [make_stress_record(d) for d in dates]
        result = compute_allostatic_load(sleep, readiness, spo2, stress)
        self.assertIsInstance(result, dict)


class TestEarlyWarning(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 5)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_early_warning_signals(sleep)
        self.assertIsInstance(result, dict)

    def test_stable(self):
        dates = date_range("2025-01-01", 30)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_early_warning_signals(sleep)
        self.assertIsInstance(result, dict)


class TestEntropy(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 5)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_daily_entropy(sleep, [])
        self.assertIsInstance(result, dict)

    def test_valid_30_days(self):
        dates = date_range("2025-01-01", 35)
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + _deterministic_float(f"e-{d}", 0, 20))
                 for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        result = compute_daily_entropy(sleep, readiness)
        self.assertIsInstance(result, dict)
        if result.get("hrv_entropy") is not None:
            self.assertIsInstance(result["hrv_entropy"], float)
            self.assertIn(result.get("hrv_interpretation"), ("rigid", "healthy", "chaotic", "insufficient_data"))


class TestTempAmplitude(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 10)
        readiness = [make_readiness_record(d) for d in dates]
        result = compute_temp_amplitude_trend(readiness)
        self.assertIsInstance(result, dict)

    def test_valid_35_days(self):
        dates = date_range("2025-01-01", 35)
        readiness = [make_readiness_record(d, temp_deviation_c=_deterministic_float(f"ta-{d}", -0.3, 0.5))
                     for d in dates]
        result = compute_temp_amplitude_trend(readiness)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
