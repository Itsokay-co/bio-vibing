"""Garmin Connect export provider.

Parses CSV/JSON files exported from Garmin Connect web.
Set GARMIN_EXPORT_DIR env var to the directory containing exported files.

Supported export files:
- Sleep data (JSON from Garmin Connect API or wellness export)
- Activities (CSV from Garmin Connect > Activities > Export CSV)
- Body composition (CSV from Garmin Connect > Health Stats)
"""

import csv
import json
import os
from datetime import datetime
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, WorkoutRecord,
    BodyCompositionRecord,
)
from providers.base import BaseProvider
from workout_types import normalize_workout_type


class GarminProvider(BaseProvider):
    name = "garmin"

    def __init__(self):
        self.export_dir = os.environ.get("GARMIN_EXPORT_DIR", "")
        if not self.export_dir:
            raise ValueError(
                "GARMIN_EXPORT_DIR not set. "
                "Export your data from Garmin Connect web and point "
                "this to the directory containing the exported files."
            )
        if not os.path.isdir(self.export_dir):
            raise ValueError(f"Garmin export directory not found: {self.export_dir}")

    def _find_file(self, *patterns):
        """Find a file matching any of the given name patterns."""
        for f in os.listdir(self.export_dir):
            fl = f.lower()
            for p in patterns:
                if p.lower() in fl:
                    return os.path.join(self.export_dir, f)
        return None

    def _read_csv(self, filepath):
        """Read CSV file and return list of dicts."""
        if not filepath or not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _read_json(self, filepath):
        """Read JSON file and return parsed data."""
        if not filepath or not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _parse_date(self, val):
        """Parse various date formats to YYYY-MM-DD."""
        if not val:
            return ""
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(val.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return val[:10]

    def _safe_float(self, val):
        if val is None or val == "":
            return None
        try:
            # Handle comma as decimal separator
            return float(str(val).replace(",", "."))
        except (ValueError, TypeError):
            return None

    def _safe_int(self, val):
        f = self._safe_float(val)
        return int(f) if f is not None else None

    def test_connection(self) -> dict:
        try:
            files = os.listdir(self.export_dir)
            csv_files = [f for f in files if f.endswith((".csv", ".json"))]
            return {
                "connected": True,
                "info": f"Export directory: {len(csv_files)} data files found",
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self) -> Optional[UserProfile]:
        # Garmin exports don't typically include profile in a standard file
        return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []

        # Try JSON sleep data first (from Garmin API exports)
        sleep_file = self._find_file("sleep", "sleepData")
        if sleep_file and sleep_file.endswith(".json"):
            data = self._read_json(sleep_file)
            for d in data:
                day = self._parse_date(d.get("calendarDate", ""))
                if not day or day < start_date or day > end_date:
                    continue
                records.append(SleepRecord(
                    day=day,
                    provider=self.name,
                    deep_sleep_seconds=d.get("deepSleepDurationInSeconds"),
                    light_sleep_seconds=d.get("lightSleepDurationInSeconds"),
                    rem_sleep_seconds=d.get("remSleepInSeconds"),
                    awake_seconds=d.get("awakeDurationInSeconds"),
                    total_sleep_seconds=d.get("durationInSeconds"),
                    avg_resting_hr_bpm=self._safe_float(d.get("averageHeartRate")),
                    sleep_type="long_sleep",
                ))
            return records

        # Fallback: CSV sleep data
        sleep_csv = self._find_file("sleep")
        if sleep_csv and sleep_csv.endswith(".csv"):
            rows = self._read_csv(sleep_csv)
            for r in rows:
                day = self._parse_date(
                    r.get("Calendar Date", r.get("calendarDate", r.get("Date", "")))
                )
                if not day or day < start_date or day > end_date:
                    continue
                records.append(SleepRecord(
                    day=day,
                    provider=self.name,
                    deep_sleep_seconds=self._safe_int(r.get("Deep Sleep (seconds)", r.get("deepSleepDurationInSeconds"))),
                    light_sleep_seconds=self._safe_int(r.get("Light Sleep (seconds)", r.get("lightSleepDurationInSeconds"))),
                    rem_sleep_seconds=self._safe_int(r.get("REM Sleep (seconds)", r.get("remSleepInSeconds"))),
                    awake_seconds=self._safe_int(r.get("Awake (seconds)", r.get("awakeDurationInSeconds"))),
                    total_sleep_seconds=self._safe_int(r.get("Duration (seconds)", r.get("durationInSeconds"))),
                    sleep_type="long_sleep",
                ))

        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        return []

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        records = []
        # Garmin dailies JSON
        daily_file = self._find_file("daily", "dailies", "summary")
        if daily_file and daily_file.endswith(".json"):
            data = self._read_json(daily_file)
            for d in data:
                day = self._parse_date(d.get("calendarDate", ""))
                if not day or day < start_date or day > end_date:
                    continue
                records.append(ActivityRecord(
                    day=day,
                    provider=self.name,
                    steps=self._safe_int(d.get("steps")),
                    total_calories=self._safe_int(d.get("activeKilocalories")),
                ))
        return records

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        records = []
        # Activities CSV (standard Garmin Connect export)
        act_file = self._find_file("activities", "Activities")
        if not act_file:
            return records

        rows = self._read_csv(act_file) if act_file.endswith(".csv") else self._read_json(act_file)
        for r in rows:
            day = self._parse_date(
                r.get("Date", r.get("startTimeLocal", r.get("date", "")))
            )
            if not day or day < start_date or day > end_date:
                continue

            activity_type = r.get("Activity Type", r.get("activityType", ""))
            duration_str = r.get("Time", r.get("duration", ""))
            duration = None
            if duration_str:
                # Handle HH:MM:SS format
                parts = str(duration_str).split(":")
                if len(parts) == 3:
                    try:
                        duration = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    except ValueError:
                        pass
                else:
                    duration = self._safe_int(duration_str)

            records.append(WorkoutRecord(
                day=day,
                provider=self.name,
                activity=normalize_workout_type(activity_type, "garmin"),
                calories=self._safe_float(r.get("Calories", r.get("activeKilocalories"))),
                distance_m=self._safe_float(r.get("Distance", r.get("distanceInMeters"))),
                duration_seconds=duration,
                avg_hr_bpm=self._safe_float(r.get("Avg HR", r.get("averageHeartRateInBeatsPerMinute"))),
                max_hr_bpm=self._safe_float(r.get("Max HR", r.get("maxHeartRateInBeatsPerMinute"))),
                elevation_gain_m=self._safe_float(r.get("Elev Gain", r.get("elevationGainInMeters"))),
                avg_speed_mps=self._safe_float(r.get("Avg Speed", r.get("averageSpeedInMetersPerSecond"))),
            ))

        return records

    def fetch_body_composition(self, start_date: str, end_date: str) -> list:
        records = []
        body_file = self._find_file("body", "weight", "bodyComp")
        if not body_file:
            return records

        rows = self._read_csv(body_file) if body_file.endswith(".csv") else self._read_json(body_file)
        for r in rows:
            day = self._parse_date(
                r.get("Date", r.get("calendarDate", r.get("measurementTimeInSeconds", "")))
            )
            if not day or day < start_date or day > end_date:
                continue

            weight_g = self._safe_float(r.get("weightInGrams"))
            weight_kg = weight_g / 1000 if weight_g else self._safe_float(r.get("Weight", r.get("weight")))

            records.append(BodyCompositionRecord(
                day=day,
                provider=self.name,
                weight_kg=weight_kg,
                body_fat_pct=self._safe_float(r.get("Body Fat %", r.get("bodyFatInPercent"))),
                bmi=self._safe_float(r.get("BMI", r.get("bodyMassIndex"))),
            ))

        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        return []
