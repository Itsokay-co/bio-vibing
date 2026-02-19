"""Oura Ring API v2 provider."""

import json
import os
import urllib.request
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, SpO2Record,
    ResilienceRecord, TagRecord, HeartRateRecord,
    WorkoutRecord,
)
from providers.base import BaseProvider


class OuraProvider(BaseProvider):
    name = "oura"
    BASE_URL = "https://api.ouraring.com/v2/usercollection"

    def __init__(self):
        self.token = os.environ.get("OURA_ACCESS_TOKEN", "")
        if not self.token:
            raise ValueError(
                "OURA_ACCESS_TOKEN not set. "
                "Get yours at https://cloud.ouraring.com/v2/docs"
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

    def _fetch_endpoint(self, endpoint: str, start: str, end: str) -> list:
        data = self._request(endpoint, f"start_date={start}&end_date={end}")
        return data.get("data", [])

    def test_connection(self) -> dict:
        try:
            data = self._request("personal_info")
            if "detail" in data:
                return {"connected": False, "info": data["detail"]}
            return {
                "connected": True,
                "info": (
                    f"User ID: {data.get('id', 'N/A')}, "
                    f"Age: {data.get('age', 'N/A')}, "
                    f"Weight: {data.get('weight', 'N/A')} kg, "
                    f"Height: {data.get('height', 'N/A')} m, "
                    f"Sex: {data.get('biological_sex', 'N/A')}"
                ),
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self) -> Optional[UserProfile]:
        try:
            data = self._request("personal_info")
            if "detail" in data:
                return None
            return UserProfile(
                provider=self.name,
                age=data.get("age"),
                weight_kg=data.get("weight"),
                height_m=data.get("height"),
                biological_sex=data.get("biological_sex"),
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []

        # Daily sleep scores
        daily = self._fetch_endpoint("daily_sleep", start_date, end_date)
        score_by_day = {d["day"]: d.get("score") for d in daily}

        # Detailed sleep stages
        detailed = self._fetch_endpoint("sleep", start_date, end_date)
        for d in detailed:
            day = d.get("day", "")
            records.append(SleepRecord(
                day=day,
                provider=self.name,
                score=score_by_day.get(day),
                deep_sleep_seconds=d.get("deep_sleep_duration"),
                rem_sleep_seconds=d.get("rem_sleep_duration"),
                light_sleep_seconds=d.get("light_sleep_duration"),
                total_sleep_seconds=d.get("total_sleep_duration"),
                efficiency=d.get("efficiency"),
                avg_hrv_ms=d.get("average_hrv"),
                avg_resting_hr_bpm=d.get("average_heart_rate"),
                sleep_type=d.get("type"),
                bedtime_start=d.get("bedtime_start"),
                bedtime_end=d.get("bedtime_end"),
                onset_latency_seconds=d.get("latency"),
                awake_seconds=d.get("awake_time"),
                hypnogram_5min=d.get("sleep_phase_5_min"),
            ))

        # If no detailed data, fall back to daily scores only
        if not detailed and daily:
            for d in daily:
                records.append(SleepRecord(
                    day=d["day"],
                    provider=self.name,
                    score=d.get("score"),
                ))

        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("daily_readiness", start_date, end_date)
        records = []
        for d in data:
            c = d.get("contributors", {})
            records.append(ReadinessRecord(
                day=d["day"],
                provider=self.name,
                score=d.get("score"),
                temp_deviation_c=d.get("temperature_deviation"),
                temp_body_score=c.get("body_temperature"),
                resting_hr_score=c.get("resting_heart_rate"),
                hrv_balance_score=c.get("hrv_balance"),
                recovery_index_score=c.get("recovery_index"),
                sleep_balance_score=c.get("sleep_balance"),
                activity_balance_score=c.get("activity_balance"),
            ))
        return records

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("daily_activity", start_date, end_date)
        records = []
        for d in data:
            met = d.get("met", {})
            records.append(ActivityRecord(
                day=d["day"],
                provider=self.name,
                score=d.get("score"),
                steps=d.get("steps"),
                total_calories=d.get("total_calories"),
                met_average=met.get("average") if isinstance(met, dict) else None,
            ))
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("daily_stress", start_date, end_date)
        records = []
        for d in data:
            records.append(StressRecord(
                day=d["day"],
                provider=self.name,
                stress_high_minutes=d.get("stress_high"),
                recovery_high_minutes=d.get("recovery_high"),
            ))
        return records

    def fetch_spo2(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("daily_spo2", start_date, end_date)
        records = []
        for d in data:
            spo2_pct = d.get("spo2_percentage", {})
            records.append(SpO2Record(
                day=d["day"],
                provider=self.name,
                avg_spo2_pct=spo2_pct.get("average") if isinstance(spo2_pct, dict) else None,
                breathing_disturbance_index=d.get("breathing_disturbance_index"),
            ))
        return records

    def fetch_resilience(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("daily_resilience", start_date, end_date)
        records = []
        for d in data:
            contributors = d.get("contributors", {})
            records.append(ResilienceRecord(
                day=d["day"],
                provider=self.name,
                level=d.get("level"),
                sleep_recovery=contributors.get("sleep_recovery"),
                daytime_recovery=contributors.get("daytime_recovery"),
            ))
        return records

    def fetch_tags(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("enhanced_tag", start_date, end_date)
        records = []
        for d in data:
            records.append(TagRecord(
                day=d.get("day", d.get("start_day", "")),
                provider=self.name,
                timestamp=d.get("timestamp", d.get("start_timestamp")),
                tag_type=d.get("tag_type_code"),
                comment=d.get("comment"),
            ))
        return records

    def fetch_heartrate(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("heartrate", start_date, end_date)
        records = []
        for d in data:
            records.append(HeartRateRecord(
                timestamp=d.get("timestamp", ""),
                provider=self.name,
                bpm=d.get("bpm"),
                source=d.get("source"),
            ))
        return records

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        data = self._fetch_endpoint("workout", start_date, end_date)
        records = []
        for d in data:
            # Calculate duration from start/end datetime
            duration = None
            start_dt = d.get("start_datetime")
            end_dt = d.get("end_datetime")
            if start_dt and end_dt:
                from datetime import datetime
                try:
                    fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
                    s = datetime.strptime(start_dt, fmt)
                    e = datetime.strptime(end_dt, fmt)
                    duration = int((e - s).total_seconds())
                except (ValueError, TypeError):
                    pass

            records.append(WorkoutRecord(
                day=d.get("day", ""),
                provider=self.name,
                activity=d.get("activity"),
                calories=d.get("calories"),
                distance_m=d.get("distance"),
                intensity=d.get("intensity"),
                duration_seconds=duration,
                start_time=start_dt,
                end_time=end_dt,
            ))
        return records

    def fetch_sleep_time(self, start_date: str, end_date: str) -> Optional[str]:
        data = self._fetch_endpoint("sleep_time", start_date, end_date)
        if data:
            latest = data[-1]
            optimal = latest.get("optimal_bedtime", {})
            if isinstance(optimal, dict):
                start_offset = optimal.get("start_offset")  # seconds before midnight
                end_offset = optimal.get("end_offset")
                if start_offset is not None and end_offset is not None:
                    # Convert negative offsets from midnight to clock times
                    # e.g. -13500 = 13500s before midnight = 20:15
                    start_h = 24 - abs(start_offset) // 3600
                    start_m = (abs(start_offset) % 3600) // 60
                    end_h = 24 - abs(end_offset) // 3600
                    end_m = (abs(end_offset) % 3600) // 60
                    return f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"
        return None
