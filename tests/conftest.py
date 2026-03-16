"""Synthetic data builders for bio-vibing tests.

All builders return plain dicts matching asdict(BiometricData) shape.
No external dependencies — stdlib only.
"""
import sys
import os
from datetime import datetime, timedelta
import math
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))


def date_range(start="2025-01-01", days=14):
    """Return list of YYYY-MM-DD strings."""
    base = datetime.strptime(start, "%Y-%m-%d")
    return [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


def _next_day(day):
    d = datetime.strptime(day, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def _prev_day(day):
    d = datetime.strptime(day, "%Y-%m-%d")
    return (d - timedelta(days=1)).strftime("%Y-%m-%d")


def _deterministic_float(seed_str, lo, hi):
    """Deterministic pseudo-random float from a string seed."""
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return lo + (h % 10000) / 10000 * (hi - lo)


def make_sleep_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "score": None,
        "deep_sleep_seconds": 5400,
        "rem_sleep_seconds": 7200,
        "light_sleep_seconds": 14400,
        "total_sleep_seconds": 28800,
        "efficiency": 88.0,
        "avg_hrv_ms": 45.0,
        "avg_resting_hr_bpm": 58.0,
        "sleep_type": "long_sleep",
        "bedtime_start": f"{_prev_day(day)}T23:00:00+00:00",
        "bedtime_end": f"{day}T07:00:00+00:00",
        "onset_latency_seconds": None,
        "awake_seconds": 1800,
        "hypnogram_5min": None,
    }
    rec.update(overrides)
    return rec


def make_readiness_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "score": 75,
        "temp_deviation_c": 0.1,
        "temp_body_score": None,
        "resting_hr_score": None,
        "hrv_balance_score": None,
        "recovery_index_score": None,
        "sleep_balance_score": None,
        "activity_balance_score": None,
    }
    rec.update(overrides)
    return rec


def make_activity_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "score": None,
        "steps": 8000,
        "total_calories": 2200,
        "met_average": None,
        "distance_m": None,
        "floors_climbed": None,
        "elevation_m": None,
        "active_minutes": None,
        "sedentary_minutes": None,
        "moderate_minutes": None,
        "vigorous_minutes": None,
        "avg_hr_bpm": None,
        "max_hr_bpm": None,
    }
    rec.update(overrides)
    return rec


def make_heartrate_record(timestamp, bpm, source="awake"):
    return {
        "timestamp": timestamp,
        "provider": "test",
        "bpm": bpm,
        "source": source,
    }


def make_workout_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "activity": "running",
        "calories": 350.0,
        "distance_m": 5000.0,
        "intensity": "moderate",
        "duration_seconds": 1800,
        "start_time": f"{day}T07:30:00+00:00",
        "end_time": f"{day}T08:00:00+00:00",
        "avg_hr_bpm": 155.0,
        "max_hr_bpm": 175.0,
        "elevation_gain_m": None,
        "avg_speed_mps": None,
        "avg_power_watts": None,
    }
    rec.update(overrides)
    return rec


def make_stress_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "stress_high_minutes": 45,
        "recovery_high_minutes": 30,
    }
    rec.update(overrides)
    return rec


def make_spo2_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "avg_spo2_pct": 97.5,
        "breathing_disturbance_index": None,
    }
    rec.update(overrides)
    return rec


def make_respiration_record(day, **overrides):
    rec = {
        "day": day,
        "provider": "test",
        "avg_respiratory_rate": 15.2,
        "min_respiratory_rate": 12.0,
        "max_respiratory_rate": 18.5,
    }
    rec.update(overrides)
    return rec


def make_tag_record(day, tag_type="exercise", comment="morning run"):
    return {
        "day": day,
        "provider": "test",
        "timestamp": f"{day}T08:00:00+00:00",
        "tag_type": tag_type,
        "comment": comment,
    }


def make_user_profile(**overrides):
    rec = {
        "provider": "test",
        "age": 35,
        "weight_kg": 75.0,
        "height_m": 1.78,
        "biological_sex": "male",
        "max_hr_bpm": 185,
    }
    rec.update(overrides)
    return rec


def make_hypnogram(pattern="normal"):
    """Generate 5-min hypnogram string. 1=deep,2=light,3=REM,4=awake."""
    if pattern == "normal":
        # ~8h = 96 five-min slots, realistic cycling
        cycle = "44" + "11111122223333" + "22221111" + "33332222" + "11112222333322"
        h = (cycle * 3)[:96]
    elif pattern == "fragmented":
        # Many awakenings
        h = ("224411" + "4422" * 3 + "11" + "44" + "3322") * 3
        h = h[:96]
    elif pattern == "deep_heavy":
        # Lots of deep in first half
        h = "1111111111111111" + "22223333" * 5 + "22221111" * 2
        h = h[:96]
    else:
        h = "2" * 96
    return h


def generate_hr_timeseries(day, count=288, base_bpm=65):
    """Generate HR samples for one day at 5-min resolution.

    First 96 samples (8h) tagged as 'sleep' with lower HR.
    Remaining tagged as 'awake' with higher HR.
    """
    records = []
    base_dt = datetime.strptime(f"{_prev_day(day)}T23:00:00", "%Y-%m-%dT%H:%M:%S")
    for i in range(count):
        ts = (base_dt + timedelta(minutes=i * 5)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        # Deterministic variation
        v = _deterministic_float(f"{day}-{i}", -5, 5)
        if i < 96:  # sleep window
            bpm = int(base_bpm + v)
            source = "sleep"
        else:
            bpm = int(base_bpm + 15 + v)
            source = "awake"
        records.append(make_heartrate_record(ts, max(40, bpm), source))
    return records


def make_biometric_data(days=14, start="2025-01-01", **overrides):
    """Generate a complete asdict(BiometricData) with all categories."""
    dates = date_range(start, days)
    end_date = dates[-1] if dates else start

    sleep = [make_sleep_record(d, avg_hrv_ms=40 + _deterministic_float(f"hrv-{d}", 0, 20),
                                avg_resting_hr_bpm=55 + _deterministic_float(f"rhr-{d}", 0, 10))
             for d in dates]
    readiness = [make_readiness_record(d, temp_deviation_c=_deterministic_float(f"temp-{d}", -0.3, 0.5))
                 for d in dates]
    activity = [make_activity_record(d, steps=int(6000 + _deterministic_float(f"steps-{d}", 0, 8000)))
                for d in dates]
    stress = [make_stress_record(d) for d in dates]
    spo2 = [make_spo2_record(d, avg_spo2_pct=96 + _deterministic_float(f"spo2-{d}", 0, 3))
            for d in dates]
    respiration = [make_respiration_record(d) for d in dates]

    heartrate = []
    for d in dates:
        heartrate.extend(generate_hr_timeseries(d, count=288))

    # Workouts every 3rd day
    workouts = [make_workout_record(dates[i]) for i in range(0, len(dates), 3)]

    data = {
        "provider": "test",
        "period_start": start,
        "period_end": end_date,
        "user": make_user_profile(),
        "sleep": sleep,
        "readiness": readiness,
        "activity": activity,
        "stress": stress,
        "spo2": spo2,
        "resilience": [],
        "tags": [],
        "heartrate": heartrate,
        "workouts": workouts,
        "body_composition": [],
        "respiration": respiration,
        "meals": [],
        "optimal_bedtime": None,
        "warnings": [],
    }
    data.update(overrides)
    return data
