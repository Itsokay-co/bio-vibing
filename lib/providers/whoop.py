"""Whoop API v2 provider."""

import json
import os
import urllib.request
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, SpO2Record, WorkoutRecord,
)
from providers.base import BaseProvider


# Whoop sport_name → normalized activity type
WHOOP_WORKOUT_TYPES = {
    "running": "running", "walking": "walking", "hiking/rucking": "hiking",
    "track & field": "running", "stroller walking": "walking",
    "cycling": "cycling", "mountain biking": "mountain_biking",
    "spin": "indoor_cycling", "assault bike": "indoor_cycling",
    "swimming": "swimming", "water polo": "water_polo",
    "rowing": "rowing", "kayaking": "kayaking",
    "paddleboarding": "stand_up_paddleboarding", "surfing": "surfing",
    "weightlifting": "strength_training", "powerlifting": "strength_training",
    "strength trainer": "strength_training", "functional fitness": "cardio_training",
    "elliptical": "elliptical", "stairmaster": "stair_climbing",
    "hiit": "cardio_training", "jumping rope": "cardio_training",
    "yoga": "yoga", "hot yoga": "yoga", "pilates": "pilates",
    "stretching": "stretching", "meditation": "meditation",
    "skiing": "alpine_skiing", "cross country skiing": "cross_country_skiing",
    "snowboarding": "snowboarding", "ice skating": "ice_skating",
    "soccer": "soccer", "basketball": "basketball", "football": "american_football",
    "baseball": "baseball", "volleyball": "volleyball", "rugby": "rugby",
    "lacrosse": "lacrosse", "cricket": "cricket",
    "ice hockey": "hockey", "field hockey": "hockey",
    "tennis": "tennis", "squash": "squash", "badminton": "badminton",
    "table tennis": "table_tennis", "pickleball": "pickleball",
    "boxing": "boxing", "kickboxing": "boxing", "martial arts": "martial_arts",
    "jiu jitsu": "martial_arts", "wrestling": "wrestling",
    "rock climbing": "rock_climbing", "golf": "golf",
    "dance": "dance", "gymnastics": "gymnastics",
    "crossfit": "cardio_training", "obstacle course racing": "cardio_training",
}


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
        self._recovery_cache = None

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

    def _date_params(self, start_date: str, end_date: str) -> str:
        return f"start={start_date}T00:00:00.000Z&end={end_date}T23:59:59.999Z"

    def _get_recovery_data(self, start_date: str, end_date: str) -> list:
        """Fetch and cache recovery data (used by readiness, sleep enrichment, and SpO2)."""
        if self._recovery_cache is None:
            self._recovery_cache = self._paginate(
                "recovery", self._date_params(start_date, end_date)
            )
        return self._recovery_cache

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
                biological_sex=None,
                age=None,
                max_hr_bpm=body.get("max_heart_rate"),
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []
        data = self._paginate(
            "activity/sleep", self._date_params(start_date, end_date)
        )
        for d in data:
            score_obj = d.get("score", {}) or {}
            stage = score_obj.get("stage_summary", {}) or {}
            day = d.get("start", "")[:10]
            records.append(SleepRecord(
                day=day,
                provider=self.name,
                score=None,
                deep_sleep_seconds=stage.get("total_slow_wave_sleep_time_milli", 0) // 1000 or None,
                rem_sleep_seconds=stage.get("total_rem_sleep_time_milli", 0) // 1000 or None,
                light_sleep_seconds=stage.get("total_light_sleep_time_milli", 0) // 1000 or None,
                total_sleep_seconds=stage.get("total_in_bed_time_milli", 0) // 1000 or None,
                awake_seconds=stage.get("total_awake_time_milli", 0) // 1000 or None,
                efficiency=score_obj.get("sleep_efficiency_percentage"),
                avg_hrv_ms=None,
                avg_resting_hr_bpm=None,
                sleep_type="long_sleep" if not d.get("nap") else "nap",
                bedtime_start=d.get("start"),
                bedtime_end=d.get("end"),
            ))

        # Enrich sleep records with HRV and RHR from recovery data
        self._enrich_sleep_from_recovery(records, start_date, end_date)
        return records

    def _enrich_sleep_from_recovery(self, sleep_records: list, start_date: str, end_date: str):
        """Merge recovery HRV and RHR into sleep records by day."""
        try:
            recovery_data = self._get_recovery_data(start_date, end_date)
        except Exception:
            return

        recovery_by_day = {}
        for d in recovery_data:
            if d.get("score_state") != "SCORED":
                continue
            score = d.get("score", {}) or {}
            day = d.get("created_at", d.get("cycle", {}).get("start", ""))[:10]
            if day:
                recovery_by_day[day] = score

        for rec in sleep_records:
            score = recovery_by_day.get(rec.day)
            if score:
                hrv = score.get("hrv_rmssd_milli")
                if hrv is not None:
                    rec.avg_hrv_ms = float(hrv)
                rhr = score.get("resting_heart_rate")
                if rhr is not None:
                    rec.avg_resting_hr_bpm = float(rhr)

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        """Whoop recovery = readiness. Filters out unscored records."""
        records = []
        data = self._get_recovery_data(start_date, end_date)
        for d in data:
            if d.get("score_state") != "SCORED":
                continue
            score_obj = d.get("score", {}) or {}
            day = d.get("created_at", d.get("cycle", {}).get("start", ""))[:10]
            records.append(ReadinessRecord(
                day=day,
                provider=self.name,
                score=int(score_obj.get("recovery_score", 0)) or None,
                temp_deviation_c=score_obj.get("skin_temp_celsius"),
            ))
        return records

    def fetch_spo2(self, start_date: str, end_date: str) -> list:
        """Extract SpO2 from recovery data."""
        records = []
        try:
            data = self._get_recovery_data(start_date, end_date)
        except Exception:
            return records
        for d in data:
            if d.get("score_state") != "SCORED":
                continue
            score_obj = d.get("score", {}) or {}
            spo2 = score_obj.get("spo2_percentage")
            if spo2 is None:
                continue
            day = d.get("created_at", d.get("cycle", {}).get("start", ""))[:10]
            records.append(SpO2Record(
                day=day,
                provider=self.name,
                avg_spo2_pct=float(spo2),
            ))
        return records

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        """Whoop workouts aggregated into daily activity."""
        records = []
        data = self._paginate(
            "activity/workout", self._date_params(start_date, end_date)
        )
        by_day = {}
        for d in data:
            day = d.get("start", "")[:10]
            score_obj = d.get("score", {}) or {}
            if day not in by_day:
                by_day[day] = {"strain": 0, "cals": 0}
            by_day[day]["strain"] = max(by_day[day]["strain"], score_obj.get("strain", 0))
            by_day[day]["cals"] += score_obj.get("kilojoule", 0) / 4.184

        for day, vals in sorted(by_day.items()):
            normalized_score = min(100, int(vals["strain"] / 21 * 100))
            records.append(ActivityRecord(
                day=day,
                provider=self.name,
                score=normalized_score or None,
                steps=None,
                total_calories=int(vals["cals"]) or None,
                met_average=None,
            ))
        return records

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        """Fetch individual workout records with HR data."""
        records = []
        data = self._paginate(
            "activity/workout", self._date_params(start_date, end_date)
        )
        for d in data:
            score_obj = d.get("score", {}) or {}
            if d.get("score_state") == "PENDING_STRAIN":
                continue
            sport = d.get("sport_name", d.get("sport_id", ""))
            activity = WHOOP_WORKOUT_TYPES.get(
                str(sport).lower(), str(sport).lower().replace(" ", "_")
            )
            start = d.get("start", "")
            end = d.get("end", "")
            duration = None
            if start and end:
                try:
                    from datetime import datetime
                    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    e = datetime.fromisoformat(end.replace("Z", "+00:00"))
                    duration = int((e - s).total_seconds())
                except (ValueError, AttributeError):
                    pass

            records.append(WorkoutRecord(
                day=start[:10] if start else "",
                provider=self.name,
                activity=activity,
                calories=score_obj.get("kilojoule", 0) / 4.184 if score_obj.get("kilojoule") else None,
                distance_m=score_obj.get("distance_meter"),
                duration_seconds=duration,
                start_time=start or None,
                end_time=end or None,
                avg_hr_bpm=score_obj.get("average_heart_rate"),
                max_hr_bpm=score_obj.get("max_heart_rate"),
                elevation_gain_m=score_obj.get("altitude_gain_meter"),
            ))
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        return []
