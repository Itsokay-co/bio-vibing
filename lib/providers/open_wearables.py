"""Open Wearables API provider — meta-provider for multi-device access.

Consumes the open-wearables REST API to access data from any wearable
connected through the platform (Garmin, Whoop, Polar, Suunto, Strava, etc).

Requires a self-hosted or cloud open-wearables instance.
Env vars: OPEN_WEARABLES_API_KEY, OPEN_WEARABLES_USER_ID, OPEN_WEARABLES_URL
"""

import json
import os
import urllib.request
import urllib.parse
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, WorkoutRecord,
    SpO2Record, HeartRateRecord,
)
from providers.base import BaseProvider
from workout_types import normalize_workout_type


class OpenWearablesProvider(BaseProvider):
    name = "open_wearables"

    def __init__(self):
        self.api_key = os.environ.get("OPEN_WEARABLES_API_KEY", "")
        self.user_id = os.environ.get("OPEN_WEARABLES_USER_ID", "")
        self.base_url = os.environ.get(
            "OPEN_WEARABLES_URL", "http://localhost:8000"
        ).rstrip("/")

        if not self.api_key or not self.user_id:
            raise ValueError(
                "OPEN_WEARABLES_API_KEY and OPEN_WEARABLES_USER_ID required. "
                "Get these from your open-wearables instance."
            )

    def _request(self, path, params=None):
        url = f"{self.base_url}/api/v1{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def test_connection(self):
        try:
            data = self._request(f"/users/{self.user_id}")
            name = f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            return {
                "connected": True,
                "info": f"User: {name or data.get('id', 'N/A')}",
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self):
        try:
            data = self._request(f"/users/{self.user_id}")
            return UserProfile(
                provider=self.name,
                age=data.get("age"),
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date, end_date):
        records = []
        try:
            data = self._request(
                f"/users/{self.user_id}/summaries/sleep",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            for r in data.get("data", []):
                stages = r.get("stages") or {}
                records.append(SleepRecord(
                    day=r.get("date", ""),
                    provider=self.name,
                    total_sleep_seconds=int(r.get("duration_minutes", 0) * 60) or None,
                    deep_sleep_seconds=int(stages.get("deep_minutes", 0) * 60) or None,
                    rem_sleep_seconds=int(stages.get("rem_minutes", 0) * 60) or None,
                    light_sleep_seconds=int(stages.get("light_minutes", 0) * 60) or None,
                    awake_seconds=int(stages.get("awake_minutes", 0) * 60) or None,
                    efficiency=r.get("efficiency_percent"),
                    avg_resting_hr_bpm=r.get("avg_heart_rate_bpm"),
                    avg_hrv_ms=r.get("avg_hrv_sdnn_ms"),
                    bedtime_start=r.get("start_time"),
                    bedtime_end=r.get("end_time"),
                    sleep_type="long_sleep",
                ))
        except Exception:
            pass
        return records

    def fetch_readiness(self, start_date, end_date):
        return []

    def fetch_activity(self, start_date, end_date):
        records = []
        try:
            data = self._request(
                f"/users/{self.user_id}/summaries/activity",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            for r in data.get("data", []):
                hr = r.get("heart_rate") or {}
                records.append(ActivityRecord(
                    day=r.get("date", ""),
                    provider=self.name,
                    steps=r.get("steps"),
                    total_calories=r.get("total_calories_kcal"),
                ))
        except Exception:
            pass
        return records

    def fetch_workouts(self, start_date, end_date):
        records = []
        try:
            data = self._request(
                f"/users/{self.user_id}/events/workouts",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            for r in data.get("data", []):
                start = r.get("start_time", "")
                records.append(WorkoutRecord(
                    day=start[:10] if start else r.get("date", ""),
                    provider=self.name,
                    activity=normalize_workout_type(r.get("type", ""), "open_wearables"),
                    calories=r.get("calories_kcal"),
                    distance_m=r.get("distance_meters"),
                    duration_seconds=r.get("duration_seconds"),
                    start_time=start or None,
                    end_time=r.get("end_time"),
                    avg_hr_bpm=r.get("avg_heart_rate_bpm"),
                    max_hr_bpm=r.get("max_heart_rate_bpm"),
                    elevation_gain_m=r.get("elevation_gain_meters"),
                ))
        except Exception:
            pass
        return records

    def fetch_stress(self, start_date, end_date):
        return []
