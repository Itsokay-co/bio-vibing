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
    ActivityRecord, StressRecord, HeartRateRecord, WorkoutRecord,
)
from providers.base import BaseProvider
from workout_types import normalize_workout_type


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
                height_m=user.get("height") / 100 if user.get("height") else None,
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
                score=None,
                deep_sleep_seconds=levels.get("deep", {}).get("minutes", 0) * 60 or None,
                rem_sleep_seconds=levels.get("rem", {}).get("minutes", 0) * 60 or None,
                light_sleep_seconds=levels.get("light", {}).get("minutes", 0) * 60 or None,
                total_sleep_seconds=d.get("duration", 0) // 1000 or None,
                efficiency=d.get("efficiency"),
                avg_hrv_ms=None,
                avg_resting_hr_bpm=None,
                sleep_type="long_sleep" if d.get("isMainSleep") else "nap",
                bedtime_start=d.get("startTime"),
                bedtime_end=d.get("endTime"),
                awake_seconds=levels.get("wake", {}).get("minutes", 0) * 60 or None,
            ))

        # Enrich with HRV
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
            pass

        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        return []

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        records = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current <= end:
            day_str = current.strftime("%Y-%m-%d")
            try:
                data = self._request(f"/1/user/-/activities/date/{day_str}.json")
                summary = data.get("summary", {})
                records.append(ActivityRecord(
                    day=day_str,
                    provider=self.name,
                    score=None,
                    steps=summary.get("steps"),
                    total_calories=summary.get("caloriesOut"),
                    met_average=None,
                ))
            except Exception:
                pass
            current += timedelta(days=1)
        return records

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        """Fetch individual workout records from Fitbit activities log."""
        records = []
        try:
            # Fitbit activities list uses afterDate + sort + limit pagination
            data = self._request(
                f"/1/user/-/activities/list.json"
                f"?afterDate={start_date}&sort=asc&offset=0&limit=100"
            )
            for d in data.get("activities", []):
                start_time = d.get("startTime", "")
                day = d.get("startDate", start_time[:10] if start_time else "")
                # Filter to date range
                if day and day > end_date:
                    break
                duration_ms = d.get("activeDuration", d.get("duration", 0))
                records.append(WorkoutRecord(
                    day=day,
                    provider=self.name,
                    activity=normalize_workout_type(d.get("activityName", ""), "fitbit"),
                    calories=d.get("calories"),
                    distance_m=d.get("distance", 0) * 1000 if d.get("distanceUnit") == "Kilometer" else d.get("distance"),
                    duration_seconds=duration_ms // 1000 if duration_ms else None,
                    start_time=start_time or None,
                    avg_hr_bpm=d.get("averageHeartRate"),
                    elevation_gain_m=d.get("elevationGain"),
                ))
        except Exception:
            pass
        return records

    def fetch_heartrate(self, start_date: str, end_date: str) -> list:
        """Fetch intraday heart rate data (requires Fitbit personal app or partner access)."""
        records = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        while current <= end:
            day_str = current.strftime("%Y-%m-%d")
            try:
                data = self._request(
                    f"/1/user/-/activities/heart/date/{day_str}/1d/5min.json"
                )
                dataset = (data.get("activities-heart-intraday", {})
                           .get("dataset", []))
                for point in dataset:
                    time_str = point.get("time", "")
                    records.append(HeartRateRecord(
                        timestamp=f"{day_str}T{time_str}",
                        provider=self.name,
                        bpm=point.get("value"),
                        source="awake",
                    ))
            except Exception:
                pass
            current += timedelta(days=1)
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        return []
