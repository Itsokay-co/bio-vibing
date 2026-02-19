"""Whoop API v2 provider."""

import json
import os
import urllib.request
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord,
)
from providers.base import BaseProvider


class WhoopProvider(BaseProvider):
    name = "whoop"
    BASE_URL = "https://api.prod.whoop.com/developer/v1"

    def __init__(self):
        self.token = os.environ.get("WHOOP_ACCESS_TOKEN", "")
        if not self.token:
            raise ValueError(
                "WHOOP_ACCESS_TOKEN not set. "
                "Get yours at https://developer.whoop.com"
            )

    def _request(self, endpoint: str, params: str = "") -> dict:
        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            url += f"?{params}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {self.token}"}
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    def _paginate(self, endpoint: str, params: str = "") -> list:
        """Whoop uses cursor-based pagination."""
        all_records = []
        next_token = None
        while True:
            p = params
            if next_token:
                p += f"&nextToken={next_token}" if p else f"nextToken={next_token}"
            data = self._request(endpoint, p)
            all_records.extend(data.get("records", []))
            next_token = data.get("next_token")
            if not next_token:
                break
        return all_records

    def test_connection(self) -> dict:
        try:
            data = self._request("user/profile/basic")
            return {
                "connected": True,
                "info": (
                    f"User ID: {data.get('user_id', 'N/A')}, "
                    f"First: {data.get('first_name', 'N/A')}, "
                    f"Last: {data.get('last_name', 'N/A')}"
                ),
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self) -> Optional[UserProfile]:
        try:
            data = self._request("user/profile/basic")
            body = self._request("user/measurement/body")
            return UserProfile(
                provider=self.name,
                height_m=body.get("height_meter"),
                weight_kg=body.get("weight_kilogram"),
                biological_sex=None,  # Not exposed in Whoop API
                age=None,  # Not directly exposed
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []
        data = self._paginate(
            "activity/sleep",
            f"start={start_date}T00:00:00.000Z&end={end_date}T23:59:59.999Z"
        )
        for d in data:
            score_obj = d.get("score", {})
            day = d.get("start", "")[:10]  # Extract date from ISO timestamp
            records.append(SleepRecord(
                day=day,
                provider=self.name,
                score=None,  # Whoop doesn't have a sleep score
                deep_sleep_seconds=score_obj.get("stage_summary", {}).get("total_slow_wave_sleep_time_milli", 0) // 1000 or None,
                rem_sleep_seconds=score_obj.get("stage_summary", {}).get("total_rem_sleep_time_milli", 0) // 1000 or None,
                light_sleep_seconds=score_obj.get("stage_summary", {}).get("total_light_sleep_time_milli", 0) // 1000 or None,
                total_sleep_seconds=score_obj.get("stage_summary", {}).get("total_in_bed_time_milli", 0) // 1000 or None,
                efficiency=score_obj.get("sleep_efficiency_percentage"),
                avg_hrv_ms=score_obj.get("respiratory_rate"),  # Whoop exposes HRV via recovery, not sleep
                avg_resting_hr_bpm=None,  # HR from recovery endpoint
                sleep_type="long_sleep" if not score_obj.get("nap") else "nap",
            ))
        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        """Whoop recovery ≈ readiness."""
        records = []
        data = self._paginate(
            "recovery",
            f"start={start_date}T00:00:00.000Z&end={end_date}T23:59:59.999Z"
        )
        for d in data:
            score_obj = d.get("score", {})
            day = d.get("created_at", d.get("cycle", {}).get("start", ""))[:10]
            records.append(ReadinessRecord(
                day=day,
                provider=self.name,
                score=int(score_obj.get("recovery_score", 0)) or None,
                temp_deviation_c=score_obj.get("skin_temp_celsius"),
                resting_hr_score=None,
                hrv_balance_score=None,
                recovery_index_score=None,
                sleep_balance_score=None,
                activity_balance_score=None,
                temp_body_score=None,
            ))
        return records

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        """Whoop workouts → activity."""
        records = []
        data = self._paginate(
            "activity/workout",
            f"start={start_date}T00:00:00.000Z&end={end_date}T23:59:59.999Z"
        )
        # Group by day
        by_day = {}
        for d in data:
            day = d.get("start", "")[:10]
            score_obj = d.get("score", {})
            if day not in by_day:
                by_day[day] = {"strain": 0, "cals": 0}
            by_day[day]["strain"] = max(by_day[day]["strain"], score_obj.get("strain", 0))
            by_day[day]["cals"] += score_obj.get("kilojoule", 0) / 4.184  # kJ to kcal

        for day, vals in sorted(by_day.items()):
            # Normalize strain (0-21) to 0-100
            normalized_score = min(100, int(vals["strain"] / 21 * 100))
            records.append(ActivityRecord(
                day=day,
                provider=self.name,
                score=normalized_score or None,
                steps=None,  # Whoop doesn't track steps
                total_calories=int(vals["cals"]) or None,
                met_average=None,
            ))
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        # Whoop doesn't have a dedicated stress endpoint
        return []
