"""Nightscout CGM provider — open-source continuous glucose monitoring.

Reads sensor glucose values from a self-hosted Nightscout instance.
5-min resolution, unlimited history.

Env vars: NIGHTSCOUT_URL, NIGHTSCOUT_API_SECRET
"""

import hashlib
import json
import os
import urllib.error
import urllib.request
import urllib.parse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import GlucoseRecord, UserProfile
from providers.base import BaseProvider


class NightscoutProvider(BaseProvider):
    name = "nightscout"

    def __init__(self):
        self.base_url = os.environ.get("NIGHTSCOUT_URL", "").rstrip("/")
        api_secret = os.environ.get("NIGHTSCOUT_API_SECRET", "")

        if not self.base_url:
            raise ValueError(
                "NIGHTSCOUT_URL required. Set to your Nightscout instance URL."
            )

        # Nightscout auth: SHA1 hash of API_SECRET
        self.api_secret_hash = (
            hashlib.sha1(api_secret.encode()).hexdigest() if api_secret else ""
        )

    def _request(self, path, params=None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params, doseq=True)
        headers = {"Content-Type": "application/json"}
        if self.api_secret_hash:
            headers["API-SECRET"] = self.api_secret_hash
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def test_connection(self):
        try:
            data = self._request("/api/v1/status.json")
            return {
                "connected": True,
                "info": f"Nightscout {data.get('version', 'unknown')} — {data.get('name', '')}",
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self):
        return UserProfile(provider=self.name)

    def fetch_glucose(self, start_date, end_date):
        """Fetch sensor glucose values (SGV) from Nightscout."""
        records = []
        try:
            params = {
                "find[dateString][$gte]": f"{start_date}T00:00:00Z",
                "find[dateString][$lte]": f"{end_date}T23:59:59Z",
                "count": 10000,
            }
            data = self._request("/api/v1/entries/sgv.json", params)
            for entry in data:
                ts = entry.get("dateString", "")
                sgv = entry.get("sgv")
                if not ts or sgv is None:
                    continue
                # Map Nightscout direction to trend string
                direction = entry.get("direction", "")
                trend = _map_nightscout_trend(direction)
                records.append(GlucoseRecord(
                    timestamp=ts,
                    provider=self.name,
                    value_mgdl=float(sgv),
                    trend=trend,
                    trend_rate=entry.get("delta"),
                ))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"nightscout: fetch_glucose failed: {e}", file=sys.stderr)
        return records

    # --- Stubs for base class interface ---
    def fetch_sleep(self, start_date, end_date):
        return []

    def fetch_readiness(self, start_date, end_date):
        return []

    def fetch_activity(self, start_date, end_date):
        return []

    def fetch_stress(self, start_date, end_date):
        return []


def _map_nightscout_trend(direction):
    """Map Nightscout direction string to standard trend."""
    mapping = {
        "DoubleUp": "rising_fast",
        "SingleUp": "rising",
        "FortyFiveUp": "rising_slightly",
        "Flat": "flat",
        "FortyFiveDown": "falling_slightly",
        "SingleDown": "falling",
        "DoubleDown": "falling_fast",
        "NOT COMPUTABLE": "not_computable",
        "RATE OUT OF RANGE": "not_computable",
    }
    return mapping.get(direction, "not_computable")
