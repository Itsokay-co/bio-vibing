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
    ActivityRecord, StressRecord,
)
from providers.base import BaseProvider


# Apple Health type identifiers
HK_HRV = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
HK_HR = "HKQuantityTypeIdentifierHeartRate"
HK_STEPS = "HKQuantityTypeIdentifierStepCount"
HK_CALORIES = "HKQuantityTypeIdentifierActiveEnergyBurned"
HK_BODY_TEMP = "HKQuantityTypeIdentifierBodyTemperature"
HK_SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
HK_WEIGHT = "HKQuantityTypeIdentifierBodyMass"
HK_HEIGHT = "HKQuantityTypeIdentifierHeight"


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

        self._tree = None
        self._records_cache = None

    def _get_records(self):
        """Lazy-load and cache the XML records."""
        if self._records_cache is not None:
            return self._records_cache

        self._records_cache = []
        # Stream parse to handle large files
        for event, elem in ET.iterparse(self.export_path, events=("end",)):
            if elem.tag == "Record":
                self._records_cache.append(elem.attrib)
                elem.clear()  # Free memory
        return self._records_cache

    def _filter_records(self, type_id: str, start_date: str, end_date: str) -> list:
        records = self._get_records()
        filtered = []
        for r in records:
            if r.get("type") != type_id:
                continue
            date = r.get("startDate", "")[:10]  # YYYY-MM-DD
            if start_date <= date <= end_date:
                filtered.append(r)
        return filtered

    def _daily_average(self, records: list, value_key: str = "value") -> dict:
        """Group records by day and compute daily averages."""
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

            dates = [r.get("startDate", "")[:10] for r in records if r.get("startDate")]
            dates = sorted(set(dates))
            types = set(r.get("type", "") for r in records)

            return {
                "connected": True,
                "info": (
                    f"Records: {len(records)}, "
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
            for r in reversed(records):  # Most recent first
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
                age=None,  # Not in export
                biological_sex=None,  # In Me > Health Details but not easily parsed
            )
        except Exception:
            return None

    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        records = []
        sleep_records = self._filter_records(HK_SLEEP, start_date, end_date)

        # Group sleep records by night (date of start)
        by_day = defaultdict(lambda: {"asleep": 0, "deep": 0, "rem": 0, "core": 0, "awake": 0})
        for r in sleep_records:
            day = r.get("startDate", "")[:10]
            value = r.get("value", "")
            start = datetime.strptime(r["startDate"][:19], "%Y-%m-%d %H:%M:%S")
            end_t = datetime.strptime(r["endDate"][:19], "%Y-%m-%d %H:%M:%S")
            duration = int((end_t - start).total_seconds())

            if "AsleepDeep" in value:
                by_day[day]["deep"] += duration
            elif "AsleepREM" in value:
                by_day[day]["rem"] += duration
            elif "AsleepCore" in value:
                by_day[day]["core"] += duration  # Core ≈ light sleep
            elif "Asleep" in value:
                by_day[day]["asleep"] += duration
            elif "Awake" in value or "InBed" in value:
                by_day[day]["awake"] += duration

        # HRV by day (SDNN, not RMSSD — flagged in provider name)
        hrv_by_day = self._daily_average(
            self._filter_records(HK_HRV, start_date, end_date)
        )
        # Resting HR by day
        hr_by_day = self._daily_average(
            self._filter_records(HK_HR, start_date, end_date)
        )

        for day, vals in sorted(by_day.items()):
            total = vals["deep"] + vals["rem"] + vals["core"] + vals["asleep"]
            if total < 1800:  # Less than 30 min, skip
                continue
            total_in_bed = total + vals["awake"]
            efficiency = (total / total_in_bed * 100) if total_in_bed > 0 else None

            records.append(SleepRecord(
                day=day,
                provider=self.name,
                score=None,  # Apple Health doesn't compute scores
                deep_sleep_seconds=vals["deep"] or None,
                rem_sleep_seconds=vals["rem"] or None,
                light_sleep_seconds=vals["core"] or None,
                total_sleep_seconds=total or None,
                efficiency=round(efficiency, 1) if efficiency else None,
                avg_hrv_ms=hrv_by_day.get(day),  # Note: SDNN, not RMSSD
                avg_resting_hr_bpm=hr_by_day.get(day),
                sleep_type="long_sleep",
            ))
        return records

    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        # Apple Health doesn't have readiness scores
        # We can provide temperature deviation if available
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
        steps_by_day = self._daily_sum(
            self._filter_records(HK_STEPS, start_date, end_date)
        )
        cals_by_day = self._daily_sum(
            self._filter_records(HK_CALORIES, start_date, end_date)
        )
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

    def fetch_stress(self, start_date: str, end_date: str) -> list:
        # Apple Health doesn't expose stress metrics
        return []
