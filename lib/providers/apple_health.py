"""Apple Health XML export provider.

Parses the export.xml file from iOS Health app export.
Set APPLE_HEALTH_EXPORT env var to the path of the exported XML file.
"""

import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, HeartRateRecord,
    SpO2Record, WorkoutRecord, BodyCompositionRecord,
    RespirationRecord,
)
from providers.base import BaseProvider
from workout_types import normalize_workout_type


# Apple Health type identifiers
HK_HRV = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
HK_HR = "HKQuantityTypeIdentifierHeartRate"
HK_RHR = "HKQuantityTypeIdentifierRestingHeartRate"
HK_STEPS = "HKQuantityTypeIdentifierStepCount"
HK_CALORIES = "HKQuantityTypeIdentifierActiveEnergyBurned"
HK_BODY_TEMP = "HKQuantityTypeIdentifierBodyTemperature"
HK_SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
HK_WEIGHT = "HKQuantityTypeIdentifierBodyMass"
HK_HEIGHT = "HKQuantityTypeIdentifierHeight"
HK_SPO2 = "HKQuantityTypeIdentifierOxygenSaturation"
HK_RESP_RATE = "HKQuantityTypeIdentifierRespiratoryRate"
HK_VO2MAX = "HKQuantityTypeIdentifierVO2Max"
HK_BODY_FAT = "HKQuantityTypeIdentifierBodyFatPercentage"
HK_BMI = "HKQuantityTypeIdentifierBodyMassIndex"
HK_LEAN_MASS = "HKQuantityTypeIdentifierLeanBodyMass"


class AppleHealthProvider(BaseProvider):
    name = "apple_health"

    def __init__(self):
        self.export_path = os.environ.get("APPLE_HEALTH_EXPORT", "")
        if not self.export_path:
            raise ValueError(
                "APPLE_HEALTH_EXPORT not set. "
                "Export your health data from iPhone: "
                "Settings > Health > Export All Health Data. "
                "Then set APPLE_HEALTH_EXPORT to the path of export.xml"
            )
        if not os.path.exists(self.export_path):
            raise ValueError(f"Apple Health export not found at: {self.export_path}")

        self._records_cache = None
        self._workouts_cache = None

    def _get_records(self):
        """Lazy-load and cache the XML records."""
        if self._records_cache is not None:
            return self._records_cache

        self._records_cache = []
        self._workouts_cache = []
        for event, elem in ET.iterparse(self.export_path, events=("end",)):
            if elem.tag == "Record":
                self._records_cache.append(elem.attrib)
                elem.clear()
            elif elem.tag == "Workout":
                self._workouts_cache.append(elem.attrib)
                elem.clear()
        return self._records_cache

    def _get_workouts(self):
        """Get workout elements (parsed alongside records)."""
        if self._workouts_cache is None:
            self._get_records()
        return self._workouts_cache

    def _filter_records(self, type_id: str, start_date: str, end_date: str) -> list:
        records = self._get_records()
        filtered = []
        for r in records:
            if r.get("type") != type_id:
                continue
            date = r.get("startDate", "")[:10]
            if start_date <= date <= end_date:
                filtered.append(r)
        return filtered

    def _daily_average(self, records: list, value_key: str = "value") -> dict:
        by_day = defaultdict(list)
        for r in records:
            day = r.get("startDate", "")[:10]
            try:
                val = float(r.get(value_key, 0))
                by_day[day].append(val)
            except (ValueError, TypeError):
                continue
        return {day: sum(vals) / len(vals) for day, vals in by_day.items()}

    def _daily_sum(self, records: list, value_key: str = "value") -> dict:
        by_day = defaultdict(float)
        for r in records:
            day = r.get("startDate", "")[:10]
            try:
                by_day[day] += float(r.get(value_key, 0))
            except (ValueError, TypeError):
                continue
        return dict(by_day)

    def test_connection(self) -> dict:
        try:
            records = self._get_records()
            if not records:
                return {"connected": False, "info": "Export file found but contains no records"}

            dates = sorted(set(r.get("startDate", "")[:10] for r in records if r.get("startDate")))
            types = set(r.get("type", "") for r in records)
            workouts = self._get_workouts()

            return {
                "connected": True,
                "info": (
                    f"Records: {len(records)}, Workouts: {len(workouts)}, "
                    f"Date range: {dates[0]} to {dates[-1]}, "
                    f"Data types: {len(types)}"
                ),
            }
        except Exception as e:
            return {"connected": False, "info": str(e)}

    def fetch_user_profile(self) -> Optional[UserProfile]:
        try:
            records = self._get_records()
            weight = None
            height = None
            for r in reversed(records):
                if r.get("type") == HK_WEIGHT and weight is None:
                    weight = float(r.get("value", 0))
                elif r.get("type") == HK_HEIGHT and height is None:
                    height = float(r.get("value", 0))
                if weight and height:
                    break
            return UserProfile(
                provider=self.name,
                weight_kg=weight,
                height_m=height,
                age=None,
                biological_sex=None,
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []
        sleep_records = self._filter_records(HK_SLEEP, start_date, end_date)

        # Deduplicate by (startDate, endDate, value) to handle overlapping sources
        seen = set()
        deduped = []
        for r in sleep_records:
            key = (r.get("startDate", ""), r.get("endDate", ""), r.get("value", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        by_day = defaultdict(lambda: {"asleep": 0, "deep": 0, "rem": 0, "core": 0, "awake": 0})
        for r in deduped:
            day = r.get("startDate", "")[:10]
            value = r.get("value", "")
            try:
                start = datetime.strptime(r["startDate"][:19], "%Y-%m-%d %H:%M:%S")
                end_t = datetime.strptime(r["endDate"][:19], "%Y-%m-%d %H:%M:%S")
                duration = int((end_t - start).total_seconds())
            except (ValueError, KeyError):
                continue

            if "AsleepDeep" in value:
                by_day[day]["deep"] += duration
            elif "AsleepREM" in value:
                by_day[day]["rem"] += duration
            elif "AsleepCore" in value:
                by_day[day]["core"] += duration
            elif "Asleep" in value:
                by_day[day]["asleep"] += duration
            elif "Awake" in value or "InBed" in value:
                by_day[day]["awake"] += duration

        hrv_by_day = self._daily_average(self._filter_records(HK_HRV, start_date, end_date))
        hr_by_day = self._daily_average(self._filter_records(HK_HR, start_date, end_date))
        rhr_by_day = self._daily_average(self._filter_records(HK_RHR, start_date, end_date))

        for day, vals in sorted(by_day.items()):
            total = vals["deep"] + vals["rem"] + vals["core"] + vals["asleep"]
            if total < 1800:
                continue
            total_in_bed = total + vals["awake"]
            efficiency = (total / total_in_bed * 100) if total_in_bed > 0 else None

            records.append(SleepRecord(
                day=day,
                provider=self.name,
                score=None,
                deep_sleep_seconds=vals["deep"] or None,
                rem_sleep_seconds=vals["rem"] or None,
                light_sleep_seconds=vals["core"] or None,
                total_sleep_seconds=total or None,
                awake_seconds=vals["awake"] or None,
                efficiency=round(efficiency, 1) if efficiency else None,
                avg_hrv_ms=hrv_by_day.get(day),
                avg_resting_hr_bpm=rhr_by_day.get(day) or hr_by_day.get(day),
                sleep_type="long_sleep",
            ))
        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        records = []
        temp_by_day = self._daily_average(
            self._filter_records(HK_BODY_TEMP, start_date, end_date)
        )
        if temp_by_day:
            baseline = sum(temp_by_day.values()) / len(temp_by_day)
            for day, temp in sorted(temp_by_day.items()):
                records.append(ReadinessRecord(
                    day=day,
                    provider=self.name,
                    score=None,
                    temp_deviation_c=round(temp - baseline, 2),
                ))
        return records

    def fetch_activity(self, start_date: str, end_date: str) -> list:
        records = []
        steps_by_day = self._daily_sum(self._filter_records(HK_STEPS, start_date, end_date))
        cals_by_day = self._daily_sum(self._filter_records(HK_CALORIES, start_date, end_date))
        all_days = sorted(set(list(steps_by_day.keys()) + list(cals_by_day.keys())))
        for day in all_days:
            records.append(ActivityRecord(
                day=day,
                provider=self.name,
                score=None,
                steps=int(steps_by_day.get(day, 0)) or None,
                total_calories=int(cals_by_day.get(day, 0)) or None,
                met_average=None,
            ))
        return records

    def fetch_heartrate(self, start_date: str, end_date: str) -> list:
        records = []
        hr_records = self._filter_records(HK_HR, start_date, end_date)
        for r in hr_records:
            try:
                records.append(HeartRateRecord(
                    timestamp=r.get("startDate", ""),
                    provider=self.name,
                    bpm=int(float(r.get("value", 0))),
                    source="awake",
                ))
            except (ValueError, TypeError):
                continue
        return records

    def fetch_spo2(self, start_date: str, end_date: str) -> list:
        records = []
        spo2_by_day = self._daily_average(self._filter_records(HK_SPO2, start_date, end_date))
        for day, val in sorted(spo2_by_day.items()):
            # Apple Health stores SpO2 as fraction (0-1), convert to percentage
            pct = val * 100 if val <= 1.0 else val
            records.append(SpO2Record(
                day=day,
                provider=self.name,
                avg_spo2_pct=round(pct, 1),
            ))
        return records

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        """Parse Workout XML elements from the export."""
        records = []
        workouts = self._get_workouts()
        for w in workouts:
            day = w.get("startDate", "")[:10]
            if not day or day < start_date or day > end_date:
                continue

            # Duration from attribute or start/end times
            duration = None
            dur_str = w.get("duration")
            if dur_str:
                try:
                    duration = int(float(dur_str) * 60)  # minutes to seconds
                except (ValueError, TypeError):
                    pass

            # Map Apple workout type to normalized name
            workout_type = w.get("workoutActivityType", "")
            activity = normalize_workout_type(workout_type, "apple_health")

            calories = None
            cal_str = w.get("totalEnergyBurned")
            if cal_str:
                try:
                    calories = float(cal_str)
                except (ValueError, TypeError):
                    pass

            distance = None
            dist_str = w.get("totalDistance")
            if dist_str:
                try:
                    distance = float(dist_str) * 1000  # km to meters
                except (ValueError, TypeError):
                    pass

            records.append(WorkoutRecord(
                day=day,
                provider=self.name,
                activity=activity or None,
                calories=calories,
                distance_m=distance,
                duration_seconds=duration,
                start_time=w.get("startDate"),
                end_time=w.get("endDate"),
            ))
        return records

    def fetch_body_composition(self, start_date: str, end_date: str) -> list:
        records = []
        weight_by_day = self._daily_average(self._filter_records(HK_WEIGHT, start_date, end_date))
        fat_by_day = self._daily_average(self._filter_records(HK_BODY_FAT, start_date, end_date))
        bmi_by_day = self._daily_average(self._filter_records(HK_BMI, start_date, end_date))
        lean_by_day = self._daily_average(self._filter_records(HK_LEAN_MASS, start_date, end_date))

        all_days = sorted(set(
            list(weight_by_day.keys()) + list(fat_by_day.keys()) +
            list(bmi_by_day.keys()) + list(lean_by_day.keys())
        ))
        for day in all_days:
            records.append(BodyCompositionRecord(
                day=day,
                provider=self.name,
                weight_kg=weight_by_day.get(day),
                body_fat_pct=fat_by_day.get(day),
                bmi=bmi_by_day.get(day),
                lean_mass_kg=lean_by_day.get(day),
            ))
        return records

    def fetch_respiration(self, start_date: str, end_date: str) -> list:
        records = []
        resp_by_day = self._daily_average(self._filter_records(HK_RESP_RATE, start_date, end_date))
        for day, val in sorted(resp_by_day.items()):
            records.append(RespirationRecord(
                day=day,
                provider=self.name,
                avg_respiratory_rate=round(val, 1),
            ))
        return records

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        return []
