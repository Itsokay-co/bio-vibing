"""Live integration tests against running Open Wearables instance.

Requires: docker compose up -d && make seed in the OW repo.
Tests are skipped if OW is not reachable at localhost:8000.
"""
import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

# Check if OW is reachable
OW_AVAILABLE = False
try:
    import urllib.request
    # Use docs endpoint (no auth required) to check if OW is running
    urllib.request.urlopen("http://localhost:8000/docs", timeout=3)
    OW_AVAILABLE = True
except Exception:
    pass

# Set env vars for provider
os.environ.setdefault("OPEN_WEARABLES_API_KEY", "sk-dd76eaa87ed4663a9904e2dee8f58d7a")
os.environ.setdefault("OPEN_WEARABLES_USER_ID", "aff1a473-7d45-4796-a95b-06cb90bb5427")
os.environ.setdefault("OPEN_WEARABLES_URL", "http://localhost:8000")

SD, ED = "2025-09-01", "2026-03-16"


def skip_if_unavailable(func):
    def wrapper(self):
        if not OW_AVAILABLE:
            self.skipTest("OW instance not reachable at localhost:8000")
        return func(self)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class TestOWIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if OW_AVAILABLE:
            from providers.open_wearables import OpenWearablesProvider
            cls.provider = OpenWearablesProvider()
        else:
            cls.provider = None

    @skip_if_unavailable
    def test_connection(self):
        result = self.provider.test_connection()
        self.assertTrue(result["connected"], f"Connection failed: {result.get('info')}")

    @skip_if_unavailable
    def test_fetch_sleep(self):
        records = self.provider.fetch_sleep(SD, ED)
        self.assertGreater(len(records), 0, "No sleep records returned")
        self.assertIsNotNone(records[0].day)

    @skip_if_unavailable
    def test_fetch_activity(self):
        records = self.provider.fetch_activity(SD, ED)
        self.assertGreater(len(records), 0, "No activity records returned")
        self.assertIsNotNone(records[0].steps)

    @skip_if_unavailable
    def test_fetch_activity_enriched(self):
        records = self.provider.fetch_activity(SD, ED)
        r = records[0]
        # Should have enriched fields from OW
        self.assertIsNotNone(r.active_minutes, "active_minutes not populated")

    @skip_if_unavailable
    def test_fetch_workouts(self):
        records = self.provider.fetch_workouts(SD, ED)
        self.assertGreater(len(records), 0, "No workout records returned")

    @skip_if_unavailable
    def test_fetch_heartrate(self):
        self.provider.fetch_sleep(SD, ED)  # populate sleep times cache
        records = self.provider.fetch_heartrate(SD, ED)
        self.assertGreater(len(records), 0, "No HR records returned")
        self.assertIsNotNone(records[0].bpm)
        # Check that some records are tagged as sleep
        sources = {r.source for r in records}
        self.assertIn("awake", sources)

    @skip_if_unavailable
    def test_fetch_body_composition(self):
        records = self.provider.fetch_body_composition(SD, ED)
        self.assertGreater(len(records), 0, "No body composition returned")
        self.assertIsNotNone(records[0].weight_kg)

    @skip_if_unavailable
    def test_fetch_respiration(self):
        self.provider.fetch_sleep(SD, ED)  # populate cache
        records = self.provider.fetch_respiration(SD, ED)
        self.assertGreater(len(records), 0, "No respiration records returned")

    @skip_if_unavailable
    def test_fetch_user_profile(self):
        profile = self.provider.fetch_user_profile()
        self.assertIsNotNone(profile)
        self.assertIsNotNone(profile.age)

    @skip_if_unavailable
    def test_full_pipeline(self):
        """End-to-end: fetch all data → compute_all → verify metrics."""
        from schema import BiometricData
        from dataclasses import asdict
        from metrics import compute_all

        p = self.provider
        data = BiometricData(
            provider="open_wearables",
            period_start=SD, period_end=ED,
            user=p.fetch_user_profile(),
            sleep=p.fetch_sleep(SD, ED),
            readiness=p.fetch_readiness(SD, ED),
            activity=p.fetch_activity(SD, ED),
            stress=p.fetch_stress(SD, ED),
            spo2=p.fetch_spo2(SD, ED),
            heartrate=p.fetch_heartrate(SD, ED),
            workouts=p.fetch_workouts(SD, ED),
            body_composition=p.fetch_body_composition(SD, ED),
            respiration=p.fetch_respiration(SD, ED),
        )

        results = compute_all(asdict(data))
        self.assertIsInstance(results, dict)
        self.assertIn("_errors", results)

        # Count successful metrics
        successful = [k for k in results if k != "_errors" and isinstance(results[k], dict)]
        self.assertGreaterEqual(len(successful), 15,
                               f"Expected 15+ metrics, got {len(successful)}: {successful}")

        errors = results.get("_errors", [])
        self.assertLess(len(errors), 5,
                       f"Too many errors ({len(errors)}): {errors}")


if __name__ == '__main__':
    unittest.main()
