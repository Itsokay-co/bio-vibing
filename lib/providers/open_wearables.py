"""Open Wearables API provider — meta-provider for multi-device access.

Consumes the open-wearables REST API to access data from any wearable
connected through the platform (Garmin, Whoop, Polar, Suunto, Strava, etc).

Requires a self-hosted or cloud open-wearables instance.
Env vars: OPEN_WEARABLES_API_KEY, OPEN_WEARABLES_USER_ID, OPEN_WEARABLES_URL
"""

import json
import os
import urllib.error
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord,
    ActivityRecord, StressRecord, WorkoutRecord,
    SpO2Record, HeartRateRecord, BodyCompositionRecord,
    RespirationRecord,
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
        # Side-channel caches populated during fetch calls
        self._sleep_spo2 = {}   # day -> avg_spo2_pct
        self._sleep_resp = {}   # day -> avg_respiratory_rate
        self._sleep_times = {}  # day -> (bedtime_start, bedtime_end)
        self._body_temp_c = None  # latest body/skin temp from body summary

    def _request(self, path, params=None):
        url = f"{self.base_url}/api/v1{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "X-Open-Wearables-API-Key": self.api_key,
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
            age = data.get("age")
            profile = UserProfile(provider=self.name, age=age)
            # Enrich from body summary
            try:
                body = self._request(f"/users/{self.user_id}/summaries/body")
                slow = body.get("slow_changing") or {}
                profile.weight_kg = slow.get("weight_kg")
                h_cm = slow.get("height_cm")
                if h_cm:
                    profile.height_m = round(h_cm / 100.0, 2)
                profile.age = slow.get("age") or age
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError):
                pass  # body summary optional
            if profile.age and not profile.max_hr_bpm:
                profile.max_hr_bpm = 220 - profile.age
            return profile
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_user_profile failed: {e}", file=sys.stderr)
            return None

    def fetch_readiness(self, start_date, end_date):
        """Build readiness records from recovery_score + RHR + HRV timeseries.

        The /summaries/recovery endpoint returns 501 in the current OW codebase,
        so we synthesize ReadinessRecords from timeseries data instead.
        """
        types = ["recovery_score", "resting_heart_rate", "heart_rate_variability_sdnn"]
        samples = self._fetch_timeseries(types, start_date, end_date, resolution="1hour")
        # Aggregate by day
        daily = {}  # day -> {score, rhr, hrv}
        for s in samples:
            ts = s.get("timestamp", "")
            day = ts[:10]
            if not day:
                continue
            if day not in daily:
                daily[day] = {"scores": [], "rhrs": [], "hrvs": []}
            t = s.get("type", "")
            v = s.get("value")
            if v is None:
                continue
            if t == "recovery_score":
                daily[day]["scores"].append(v)
            elif t == "resting_heart_rate":
                daily[day]["rhrs"].append(v)
            elif t == "heart_rate_variability_sdnn":
                daily[day]["hrvs"].append(v)

        records = []
        for day, vals in sorted(daily.items()):
            scores = vals["scores"]
            score = round(sum(scores) / len(scores)) if scores else None
            records.append(ReadinessRecord(
                day=day,
                provider=self.name,
                score=score,
                temp_deviation_c=None,  # populated below if body temp available
            ))

        # Inject temperature deviation if body temp was cached
        if self._body_temp_c is not None and len(records) > 1:
            baseline = self._body_temp_c  # single snapshot, use as baseline
            for rec in records:
                rec.temp_deviation_c = 0.0  # relative to snapshot baseline
        return records

    def fetch_activity(self, start_date, end_date):
        records = []
        try:
            data = self._request(
                f"/users/{self.user_id}/summaries/activity",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            for r in data.get("data", []):
                hr = r.get("heart_rate") or {}
                intensity = r.get("intensity_minutes") or {}
                records.append(ActivityRecord(
                    day=r.get("date", ""),
                    provider=self.name,
                    steps=r.get("steps"),
                    total_calories=r.get("total_calories_kcal"),
                    distance_m=r.get("distance_meters"),
                    floors_climbed=r.get("floors_climbed"),
                    elevation_m=r.get("elevation_meters"),
                    active_minutes=r.get("active_minutes"),
                    sedentary_minutes=r.get("sedentary_minutes"),
                    moderate_minutes=intensity.get("moderate"),
                    vigorous_minutes=intensity.get("vigorous"),
                    avg_hr_bpm=hr.get("avg_bpm"),
                    max_hr_bpm=hr.get("max_bpm"),
                ))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_activity failed: {e}", file=sys.stderr)
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
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_workouts failed: {e}", file=sys.stderr)
        return records

    def fetch_spo2(self, start_date, end_date):
        """Return SpO2 records from data cached during fetch_sleep."""
        return [
            SpO2Record(day=day, provider=self.name, avg_spo2_pct=val)
            for day, val in sorted(self._sleep_spo2.items())
        ]

    def fetch_body_composition(self, start_date, end_date):
        """Fetch body composition from body summary endpoint."""
        try:
            body = self._request(f"/users/{self.user_id}/summaries/body")
            slow = body.get("slow_changing") or {}
            latest = body.get("latest") or {}
            # Cache temperature for fetch_readiness
            self._body_temp_c = (
                latest.get("body_temperature_celsius")
                or latest.get("skin_temperature_celsius")
            )
            # Body summary is a single snapshot, not per-day
            today = end_date or start_date
            if not slow.get("weight_kg") and not slow.get("body_fat_percent"):
                return []
            return [BodyCompositionRecord(
                day=today,
                provider=self.name,
                weight_kg=slow.get("weight_kg"),
                body_fat_pct=slow.get("body_fat_percent"),
                lean_mass_kg=slow.get("lean_body_mass"),
                muscle_mass_kg=slow.get("muscle_mass_kg"),
                bmi=slow.get("bmi"),
            )]
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_body_composition failed: {e}", file=sys.stderr)
            return []

    # --- Sleep sessions with stage intervals (hypnogram) ---

    @staticmethod
    def _build_hypnogram(stage_intervals, start_time_str, end_time_str):
        """Convert sleep stage intervals to 5-min hypnogram string.

        OW stages: deep, light, rem, awake, in_bed, sleeping, unknown
        Oura encoding: 1=deep, 2=light, 3=REM, 4=awake
        """
        stage_map = {
            "deep": "1", "light": "2", "rem": "3", "awake": "4",
            "in_bed": "4", "sleeping": "2", "unknown": "4",
        }
        if not stage_intervals or not start_time_str or not end_time_str:
            return None
        try:
            start = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

        # Parse intervals into (start_dt, end_dt, code) tuples
        parsed = []
        for iv in stage_intervals:
            stage_code = stage_map.get(iv.get("stage", ""), "4")
            try:
                iv_start = datetime.fromisoformat(iv["start_time"].replace("Z", "+00:00"))
                iv_end = datetime.fromisoformat(iv["end_time"].replace("Z", "+00:00"))
                parsed.append((iv_start, iv_end, stage_code))
            except (ValueError, TypeError, KeyError):
                continue
        if not parsed:
            return None

        # Sample at 5-min marks
        hypnogram = []
        t = start
        while t < end:
            code = "4"  # default awake
            for iv_start, iv_end, iv_code in parsed:
                if iv_start <= t < iv_end:
                    code = iv_code
                    break
            hypnogram.append(code)
            t += timedelta(minutes=5)

        return "".join(hypnogram) if hypnogram else None

    def _fetch_sleep_sessions(self, start_date, end_date):
        """Fetch sleep sessions and enrich existing sleep records with hypnogram."""
        try:
            data = self._request(
                f"/users/{self.user_id}/events/sleep",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            sessions = {}
            for s in data.get("data", []):
                start = s.get("start_time", "")
                end = s.get("end_time", "")
                day = end[:10] if end else start[:10]
                if not day:
                    continue
                intervals = s.get("sleep_stage_intervals") or []
                hypnogram = self._build_hypnogram(intervals, start, end)
                is_nap = s.get("is_nap", False)
                if not is_nap and hypnogram:
                    sessions[day] = hypnogram
            return sessions
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_sleep_sessions failed: {e}", file=sys.stderr)
            return {}

    def fetch_sleep(self, start_date, end_date):
        records = []
        self._sleep_spo2 = {}
        self._sleep_resp = {}
        self._sleep_times = {}
        try:
            data = self._request(
                f"/users/{self.user_id}/summaries/sleep",
                {"start_date": start_date, "end_date": end_date, "limit": 100},
            )
            for r in data.get("data", []):
                day = r.get("date", "")
                stages = r.get("stages") or {}
                records.append(SleepRecord(
                    day=day,
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
                # Cache sleep times for HR source tagging
                bt_start, bt_end = r.get("start_time"), r.get("end_time")
                if bt_start and bt_end:
                    self._sleep_times[day] = (bt_start, bt_end)
                # Cache side-channel data for fetch_spo2 / fetch_respiration
                spo2 = r.get("avg_spo2_percent")
                if spo2 is not None:
                    self._sleep_spo2[day] = spo2
                resp = r.get("avg_respiratory_rate")
                if resp is not None:
                    self._sleep_resp[day] = resp
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as e:
            print(f"open_wearables: fetch_sleep failed: {e}", file=sys.stderr)

        # Enrich with hypnogram from sleep sessions
        hypnograms = self._fetch_sleep_sessions(start_date, end_date)
        for rec in records:
            h = hypnograms.get(rec.day)
            if h:
                rec.hypnogram_5min = h
        return records

    # --- Timeseries API ---

    def _fetch_timeseries(self, types, start_date, end_date, resolution="5min"):
        """Fetch timeseries data with cursor pagination.

        Args:
            types: list of SeriesType enum strings (e.g. ["heart_rate"])
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            resolution: raw, 1min, 5min, 15min, 1hour
        Returns:
            list of {timestamp, type, value, unit} dicts
        """
        all_samples = []
        cursor = None
        params = {
            "start_time": f"{start_date}T00:00:00",
            "end_time": f"{end_date}T23:59:59",
            "resolution": resolution,
            "limit": 100,
        }
        # urlencode doesn't handle repeated params, build manually
        type_params = "&".join(f"types={t}" for t in types)
        try:
            while True:
                p = dict(params)
                if cursor:
                    p["cursor"] = cursor
                base_qs = urllib.parse.urlencode(p)
                full_qs = f"{base_qs}&{type_params}" if type_params else base_qs
                url = f"{self.base_url}/api/v1/users/{self.user_id}/timeseries?{full_qs}"
                req = urllib.request.Request(
                    url,
                    headers={
                        "X-Open-Wearables-API-Key": self.api_key,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read())
                for s in data.get("data", []):
                    all_samples.append(s)
                pag = data.get("pagination") or {}
                if pag.get("has_more") and pag.get("next_cursor"):
                    cursor = pag["next_cursor"]
                else:
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            print(f"open_wearables: timeseries {types} failed: {e}", file=sys.stderr)
        return all_samples

    def fetch_heartrate(self, start_date, end_date):
        """Fetch 5-min heart rate timeseries."""
        samples = self._fetch_timeseries(["heart_rate"], start_date, end_date, resolution="5min")
        # Build sorted list of sleep windows for efficient lookup
        sleep_windows = sorted(self._sleep_times.values())
        records = []
        for s in samples:
            ts = s.get("timestamp", "")
            bpm = s.get("value")
            if not ts or bpm is None:
                continue
            # Check if timestamp falls within any sleep window
            source = "awake"
            for bt_start, bt_end in sleep_windows:
                if bt_start <= ts <= bt_end:
                    source = "sleep"
                    break
                if ts < bt_start:
                    break  # past all possible windows (sorted)
            records.append(HeartRateRecord(
                timestamp=ts,
                provider=self.name,
                bpm=int(round(bpm)),
                source=source,
            ))
        return records

    def fetch_respiration(self, start_date, end_date):
        """Return respiration records from timeseries or sleep summary cache."""
        # Try timeseries first for full-day data
        samples = self._fetch_timeseries(["respiratory_rate"], start_date, end_date, resolution="1hour")
        if samples:
            daily = {}
            for s in samples:
                day = s.get("timestamp", "")[:10]
                v = s.get("value")
                if day and v is not None:
                    daily.setdefault(day, []).append(v)
            return [
                RespirationRecord(
                    day=day, provider=self.name,
                    avg_respiratory_rate=round(sum(vals) / len(vals), 1),
                    min_respiratory_rate=round(min(vals), 1),
                    max_respiratory_rate=round(max(vals), 1),
                )
                for day, vals in sorted(daily.items())
            ]
        # Fallback: sleep summary cache
        return [
            RespirationRecord(day=day, provider=self.name, avg_respiratory_rate=val)
            for day, val in sorted(self._sleep_resp.items())
        ]

    def fetch_stress(self, start_date, end_date):
        """Fetch stress from garmin_stress_level timeseries."""
        samples = self._fetch_timeseries(
            ["garmin_stress_level"], start_date, end_date, resolution="5min"
        )
        if not samples:
            return []
        # Aggregate daily: stress > 50 = stress minutes, < 30 = recovery minutes
        # Each 5-min sample represents 5 minutes
        daily = {}
        for s in samples:
            day = s.get("timestamp", "")[:10]
            v = s.get("value")
            if not day or v is None:
                continue
            if day not in daily:
                daily[day] = {"stress": 0, "recovery": 0}
            if v > 50:
                daily[day]["stress"] += 5
            elif v < 30:
                daily[day]["recovery"] += 5
        return [
            StressRecord(
                day=day, provider=self.name,
                stress_high_minutes=vals["stress"],
                recovery_high_minutes=vals["recovery"],
            )
            for day, vals in sorted(daily.items())
        ]
