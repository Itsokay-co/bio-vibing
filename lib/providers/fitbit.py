"""Fitbit Web API provider."""

import json
import os
import urllib.request
from typing import Optional
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord,
)
from providers.base import BaseProvider


class FitbitProvider(BaseProvider):
    name = "fitbit"
    BASE_URL = "https://api.fitbit.com"

    def __init__(self):
        self.token = os.environ.get("FITBIT_ACCESS_TOKEN", "")
        if not self.token:
            raise ValueError(
                "FITBIT_ACCESS_TOKEN not set. "
                "Get yours at https://dev.fitbit.com"
            )

    def _request(self, path: str) -> dict:
        url = f"{self.BASE_URL}{path}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def test_connection(self) -> dict:
        try:
            data = self._request("/1/user/-/profile.json")
            user = data.get("user", {})
            return {
                "connected": True,
                "info": (
                    f"Name: {user.get('displayName', 'N/A')}, "
                    f"Age: {user.get('age', 'N/A')}, "
                    f"Weight: {user.get('weight', 'N/A')} kg"
                ),
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self) -> Optional[UserProfile]:
        try:
            data = self._request("/1/user/-/profile.json")
            user = data.get("user", {})
            return UserProfile(
                provider=self.name,
                age=user.get("age"),
                weight_kg=user.get("weight"),
                height_m=user.get("height") / 100 if user.get("height") else None,  # cm to m
                biological_sex=user.get("gender"),
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []
        data = self._request(
            f"/1.2/user/-/sleep/date/{start_date}/{end_date}.json"
        )
        for d in data.get("sleep", []):
            levels = d.get("levels", {}).get("summary", {})
            records.append(SleepRecord(
                day=d.get("dateOfSleep", ""),
                provider=self.name,
                score=None,  # Fitbit doesn't have a composite sleep score in API
                deep_sleep_seconds=levels.get("deep", {}).get("minutes", 0) * 60 or None,
                rem_sleep_seconds=levels.get("rem", {}).get("minutes", 0) * 60 or None,
                light_sleep_seconds=levels.get("light", {}).get("minutes", 0) * 60 or None,
                total_sleep_seconds=d.get("duration", 0) // 1000 or None,  # ms to seconds
                efficiency=d.get("efficiency"),
                avg_hrv_ms=None,  # HRV is a separate endpoint
                avg_resting_hr_bpm=None,  # HR is a separate endpoint
                sleep_type="long_sleep" if d.get("isMainSleep") else "nap",
            ))

        # Fetch HRV data separately — supplementary enrichment, don't fail the whole fetch
        try:
            hrv_data = self._request(
                f"/1/user/-/hrv/date/{start_date}/{end_date}.json"
            )
            hrv_by_day = {}
            for entry in hrv_data.get("hrv", []):
                day = entry.get("dateTime", "")
                val = entry.get("value", {}).get("dailyRmssd")
                if day and val:
                    hrv_by_day[day] = val

            for r in records:
                if r.day in hrv_by_day:
                    r.avg_hrv_ms = hrv_by_day[r.day]
        except Exception:
            pass  # HRV is optional enrichment on top of already-fetched sleep data

        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        # Fitbit doesn't have a readiness/recovery score in the public API
        return []

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        records = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current <= end:
            day_str = current.strftime("%Y-%m-%d")
            data = self._request(f"/1/user/-/activities/date/{day_str}.json")
            summary = data.get("summary", {})
            records.append(ActivityRecord(
                day=day_str,
                provider=self.name,
                score=None,  # No composite score
                steps=summary.get("steps"),
                total_calories=summary.get("caloriesOut"),
                met_average=None,
            ))
            current += timedelta(days=1)
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        # Fitbit doesn't expose stress data through the public API
        return []
