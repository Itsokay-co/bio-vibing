#!/usr/bin/env python3
"""Run all bio-vibing tests. No external dependencies required.

Usage:
    python3 tests/run_all.py          # run all tests
    python3 tests/run_all.py -v       # verbose
    python3 tests/test_validation.py  # single file
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.dirname(os.path.abspath(__file__)),
        pattern='test_*.py',
    )
    verbosity = 2 if '-v' in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
