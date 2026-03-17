"""Dexcom CGM provider — continuous glucose monitoring via official API.

Reads estimated glucose values (EGVs) at 5-min resolution.
Supports both production and sandbox environments.

Env vars: DEXCOM_ACCESS_TOKEN, DEXCOM_BASE_URL (optional, defaults to production)
"""

import json
import os
import urllib.error
import urllib.request
import urllib.parse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import GlucoseRecord, UserProfile
from providers.base import BaseProvider


class DexcomProvider(BaseProvider):
    name = "dexcom"

    def __init__(self):
        self.access_token = os.environ.get("DEXCOM_ACCESS_TOKEN", "")
        self.base_url = os.environ.get(
            "DEXCOM_BASE_URL", "https://api.dexcom.com"
        ).rstrip("/")

        if not self.access_token:
            raise ValueError(
                "DEXCOM_ACCESS_TOKEN required. Get one from developer.dexcom.com."
            )

    def _request(self, path, params=None):
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())

    def test_connection(self):
        try:
            # Fetch a single EGV to verify connection
            data = self._request(
                "/v3/users/self/dataRange"
            )
            return {
                "connected": True,
                "info": f"Dexcom — data available",
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self):
        return UserProfile(provider=self.name)

    def fetch_glucose(self, start_date, end_date):
        """Fetch estimated glucose values (EGVs) from Dexcom API v3."""
        records = []
        try:
            data = self._request(
                "/v3/users/self/egvs",
                {
                    "startDate": f"{start_date}T00:00:00",
                    "endDate": f"{end_date}T23:59:59",
                },
            )
            for rec in data.get("records", []):
                ts = rec.get("displayTime") or rec.get("systemTime", "")
                value = rec.get("value")
                if not ts or value is None:
                    continue
                trend_name = rec.get("trend", "")
                trend = _map_dexcom_trend(trend_name)
                trend_rate = rec.get("trendRate")
                records.append(GlucoseRecord(
                    timestamp=ts,
                    provider=self.name,
                    value_mgdl=float(value),
                    trend=trend,
                    trend_rate=float(trend_rate) if trend_rate is not None else None,
                ))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"dexcom: fetch_glucose failed: {e}", file=sys.stderr)
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


def _map_dexcom_trend(trend_name):
    """Map Dexcom v3 trend string to standard trend."""
    mapping = {
        "doubleUp": "rising_fast",
        "singleUp": "rising",
        "fortyFiveUp": "rising_slightly",
        "flat": "flat",
        "fortyFiveDown": "falling_slightly",
        "singleDown": "falling",
        "doubleDown": "falling_fast",
        "none": "not_computable",
        "notComputable": "not_computable",
        "rateOutOfRange": "not_computable",
    }
    return mapping.get(trend_name, "not_computable")
