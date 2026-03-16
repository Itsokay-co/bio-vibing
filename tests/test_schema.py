"""Tests for lib/schema.py — dataclass instantiation and serialization."""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from schema import BiometricData, SleepRecord, UserProfile, asdict


class TestSchema(unittest.TestCase):

    def test_biometric_data_defaults(self):
        bd = BiometricData("test", "2025-01-01", "2025-01-14")
        self.assertEqual(bd.provider, "test")
        self.assertEqual(bd.sleep, [])
        self.assertIsNone(bd.user)
        self.assertEqual(bd.warnings, [])

    def test_to_dict_keys(self):
        bd = BiometricData("test", "2025-01-01", "2025-01-14")
        d = bd.to_dict()
        for key in ("provider", "period_start", "period_end", "user",
                     "sleep", "readiness", "activity", "heartrate",
                     "workouts", "body_composition", "respiration", "meals"):
            self.assertIn(key, d)

    def test_sleep_record_optional_fields(self):
        sr = SleepRecord(day="2025-01-01", provider="test")
        self.assertIsNone(sr.avg_hrv_ms)
        self.assertIsNone(sr.hypnogram_5min)
        self.assertIsNone(sr.efficiency)

    def test_asdict_roundtrip(self):
        bd = BiometricData("test", "2025-01-01", "2025-01-14")
        d1 = asdict(bd)
        d2 = bd.to_dict()
        self.assertEqual(d1, d2)

    def test_nested_asdict(self):
        sr = SleepRecord(day="2025-01-01", provider="test", avg_hrv_ms=42.0)
        bd = BiometricData("test", "2025-01-01", "2025-01-01", sleep=[sr])
        d = bd.to_dict()
        self.assertIsInstance(d["sleep"], list)
        self.assertIsInstance(d["sleep"][0], dict)
        self.assertEqual(d["sleep"][0]["avg_hrv_ms"], 42.0)


if __name__ == '__main__':
    unittest.main()
