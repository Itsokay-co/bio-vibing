"""Tests for sleep metrics: SRI, transitions, deep distribution, chronotype, efficiency."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from metrics import (compute_sleep_regularity, compute_sleep_transitions,
                     compute_deep_sleep_distribution, compute_chronotype,
                     compute_sleep_efficiency)
from conftest import make_sleep_record, make_hypnogram, date_range


class TestSleepRegularity(unittest.TestCase):

    def test_insufficient(self):
        sleep = [make_sleep_record("2025-01-01")]
        result = compute_sleep_regularity(sleep)
        self.assertIn("insufficient", str(result).lower())

    def test_regular_14_nights(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_sleep_regularity(sleep)
        self.assertIsInstance(result, dict)
        if result.get("sri_score") is not None:
            self.assertGreater(result["sri_score"], 0)


class TestTransitions(unittest.TestCase):

    def test_no_hypnogram(self):
        dates = date_range("2025-01-01", 10)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_sleep_transitions(sleep)
        self.assertIsInstance(result, dict)

    def test_valid_hypnogram(self):
        dates = date_range("2025-01-01", 10)
        h = make_hypnogram("normal")
        sleep = [make_sleep_record(d, hypnogram_5min=h) for d in dates]
        result = compute_sleep_transitions(sleep)
        self.assertIsInstance(result, dict)
        if result.get("fragmentation_index") is not None:
            self.assertIsInstance(result["fragmentation_index"], (int, float))

    def test_fragmented_hypnogram(self):
        dates = date_range("2025-01-01", 10)
        h = make_hypnogram("fragmented")
        sleep = [make_sleep_record(d, hypnogram_5min=h) for d in dates]
        result = compute_sleep_transitions(sleep)
        self.assertIsInstance(result, dict)


class TestDeepDistribution(unittest.TestCase):

    def test_no_data(self):
        result = compute_deep_sleep_distribution([])
        self.assertIn("insufficient", str(result).lower())

    def test_front_loaded(self):
        dates = date_range("2025-01-01", 21)
        h = make_hypnogram("deep_heavy")
        sleep = [make_sleep_record(d, hypnogram_5min=h) for d in dates]
        result = compute_deep_sleep_distribution(sleep)
        self.assertIsInstance(result, dict)


class TestChronotype(unittest.TestCase):

    def test_insufficient(self):
        result = compute_chronotype([])
        self.assertIn("insufficient", str(result).lower())

    def test_valid_14_nights(self):
        dates = date_range("2025-01-01", 14)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_chronotype(sleep)
        self.assertIsInstance(result, dict)


class TestSleepEfficiency(unittest.TestCase):

    def test_valid(self):
        dates = date_range("2025-01-01", 7)
        sleep = [make_sleep_record(d) for d in dates]
        result = compute_sleep_efficiency(sleep)
        self.assertIsInstance(result, dict)
        if result.get("avg_efficiency") is not None:
            self.assertGreater(result["avg_efficiency"], 0)
            self.assertLessEqual(result["avg_efficiency"], 100)

    def test_zero_total(self):
        sleep = [make_sleep_record("2025-01-01", total_sleep_seconds=0)]
        result = compute_sleep_efficiency(sleep)
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
