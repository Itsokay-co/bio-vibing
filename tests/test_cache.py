"""Tests for lib/cache.py — set/get/clear/versioning."""
import sys, os, unittest, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import cache


class TestCache(unittest.TestCase):

    def setUp(self):
        self._orig_dir = cache.CACHE_DIR
        self._orig_file = cache.CACHE_FILE
        self._orig_version = cache.CACHE_VERSION
        self.tmpdir = tempfile.mkdtemp()
        cache.CACHE_DIR = self.tmpdir
        cache.CACHE_FILE = os.path.join(self.tmpdir, "cache.json")

    def tearDown(self):
        cache.CACHE_DIR = self._orig_dir
        cache.CACHE_FILE = self._orig_file
        cache.CACHE_VERSION = self._orig_version
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_set_get_roundtrip(self):
        cache.set_cached("test", "2025-01-01", "2025-01-14", "sleep", {"records": 10})
        result = cache.get_cached("test", "2025-01-01", "2025-01-14", "sleep")
        self.assertEqual(result, {"records": 10})

    def test_ttl_expiration(self):
        cache.set_cached("test", "2025-01-01", "2025-01-14", "sleep", {"x": 1})
        # Read with very short TTL
        result = cache.get_cached("test", "2025-01-01", "2025-01-14", "sleep", ttl=0)
        self.assertIsNone(result)

    def test_clear_cache(self):
        cache.set_cached("test", "2025-01-01", "2025-01-14", "sleep", {"x": 1})
        cache.clear_cache()
        result = cache.get_cached("test", "2025-01-01", "2025-01-14", "sleep")
        self.assertIsNone(result)

    def test_version_invalidation(self):
        cache.set_cached("test", "2025-01-01", "2025-01-14", "sleep", {"x": 1})
        cache.CACHE_VERSION += 1
        result = cache.get_cached("test", "2025-01-01", "2025-01-14", "sleep")
        self.assertIsNone(result)

    def test_make_key_deterministic(self):
        k1 = cache._make_key("oura", "2025-01-01", "2025-01-14", "sleep")
        k2 = cache._make_key("oura", "2025-01-01", "2025-01-14", "sleep")
        self.assertEqual(k1, k2)
        k3 = cache._make_key("whoop", "2025-01-01", "2025-01-14", "sleep")
        self.assertNotEqual(k1, k3)


if __name__ == '__main__':
    unittest.main()
