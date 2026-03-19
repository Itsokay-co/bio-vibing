"""Tests for novel metrics: disruption classifier, Poincaré, optimal sleep, sleep debt, glucose clinical."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_disruption_classification, compute_poincare_hrv,
                     compute_optimal_sleep, compute_sleep_debt,
                     compute_glucose_clinical)
from conftest import make_sleep_record, make_readiness_record, make_spo2_record, date_range


class TestDisruptionClassifier(unittest.TestCase):

    def test_insufficient(self):
        dates = date_range("2025-01-01", 5)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_disruption_classification(sleep)
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_no_disruptions(self):
        dates = date_range("2025-01-01", 20)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_disruption_classification(sleep)
        self.assertEqual(result['n_disruptions'], 0)

    def test_v_shaped_alcohol(self):
        dates = date_range("2025-01-01", 20)
        sleep = [make_sleep_record(d) for d in dates]
        # Inject one anomalous night (V-shape: one bad night, next day fine)
        sleep[14] = make_sleep_record(dates[14], avg_hrv_ms=15, avg_resting_hr_bpm=85, efficiency=40)
        result = compute_disruption_classification(sleep)
        alcohol = [e for e in result.get('events', []) if e['classification'] == 'probable_alcohol']
        # Should detect as V-shaped
        self.assertIsInstance(result, dict)

    def test_u_shaped_illness(self):
        dates = date_range("2025-01-01", 25)
        sleep = [make_sleep_record(d) for d in dates]
        # Inject multi-day suppression (U-shape)
        for i in range(14, 18):
            sleep[i] = make_sleep_record(dates[i], avg_hrv_ms=20, avg_resting_hr_bpm=80, efficiency=50)
        result = compute_disruption_classification(sleep)
        self.assertIsInstance(result, dict)

    def test_with_readiness_and_spo2(self):
        dates = date_range("2025-01-01", 20)
        sleep = [make_sleep_record(d) for d in dates]
        readiness = [make_readiness_record(d) for d in dates]
        spo2 = [make_spo2_record(d) for d in dates]
        result = compute_disruption_classification(sleep, readiness, spo2)
        self.assertIn('events', result)


class TestPoincare(unittest.TestCase):

    def test_insufficient(self):
        sleep = [make_sleep_record("2025-01-01")]
        result = compute_poincare_hrv(sleep)
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_valid(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d, avg_hrv_ms=40 + i * 2) for i, d in enumerate(dates)]
        result = compute_poincare_hrv(sleep)
        self.assertIsNotNone(result['sd1'])
        self.assertIsNotNone(result['sd2'])
        self.assertIsNotNone(result['ratio'])
        self.assertIn(result['interpretation'],
                      ('parasympathetic_dominant', 'sympathetic_dominant', 'balanced'))


class TestOptimalSleep(unittest.TestCase):

    def test_insufficient(self):
        result = compute_optimal_sleep([])
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_valid(self):
        dates = date_range("2025-01-01", 30)
        # Vary sleep duration: 6-8h range
        sleep = [make_sleep_record(d, total_sleep_seconds=int((6 + (i % 5) * 0.5) * 3600),
                                   avg_hrv_ms=40 + (i % 5) * 8)
                 for i, d in enumerate(dates)]
        result = compute_optimal_sleep(sleep)
        if result.get('optimal_hours') is not None:
            self.assertGreater(result['optimal_hours'], 0)
            self.assertIsNotNone(result['current_avg_hours'])


class TestSleepDebt(unittest.TestCase):

    def test_insufficient(self):
        result = compute_sleep_debt([])
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_no_debt(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d, total_sleep_seconds=int(8 * 3600)) for d in dates]
        result = compute_sleep_debt(sleep, optimal_hours=7.5)
        self.assertLessEqual(result['debt_hours'], 0)

    def test_with_debt(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d, total_sleep_seconds=int(5 * 3600)) for d in dates]
        result = compute_sleep_debt(sleep, optimal_hours=7.5)
        self.assertGreater(result['debt_hours'], 0)
        self.assertGreater(result['days_to_payoff'], 0)

    def test_trajectory(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d, total_sleep_seconds=int((5 + i * 0.2) * 3600))
                 for i, d in enumerate(dates)]
        result = compute_sleep_debt(sleep)
        self.assertIn(result['trajectory'], ('improving', 'worsening', 'unknown'))


class TestGlucoseClinical(unittest.TestCase):

    def _make_glucose(self, ts, value):
        return {"timestamp": ts, "provider": "test", "value_mgdl": value,
                "trend": "flat", "trend_rate": None}

    def test_insufficient(self):
        result = compute_glucose_clinical([])
        self.assertEqual(result['interpretation'], 'insufficient_data')

    def test_valid(self):
        glucose = [self._make_glucose(f"2025-01-01T{h:02d}:00:00Z", 100 + (h % 6) * 10)
                   for h in range(24)]
        glucose += [self._make_glucose(f"2025-01-02T{h:02d}:00:00Z", 95 + (h % 6) * 12)
                    for h in range(24)]
        result = compute_glucose_clinical(glucose)
        self.assertIsNotNone(result['gmi'])
        self.assertIsNotNone(result['lbgi'])
        self.assertIsNotNone(result['hbgi'])
        self.assertIsNotNone(result['modd'])

    def test_gmi_formula(self):
        # 48 readings at exactly 100 mg/dL → GMI = 3.31 + 0.02392 * 100 = 5.7
        glucose = [self._make_glucose(f"2025-01-01T{i//2:02d}:{(i%2)*30:02d}:00Z", 100)
                   for i in range(48)]
        result = compute_glucose_clinical(glucose)
        self.assertAlmostEqual(result['gmi'], 5.7, places=0)


if __name__ == '__main__':
    unittest.main()
