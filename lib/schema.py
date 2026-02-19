"""Normalized biometric data schema.

All providers map their API responses to these dataclasses.
Every field is Optional — providers return None for metrics they don't support.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SleepRecord:
    day: str  # YYYY-MM-DD
    provider: str  # "oura", "whoop", "fitbit", "apple_health"
    score: Optional[int] = None  # 0-100 composite
    deep_sleep_seconds: Optional[int] = None
    rem_sleep_seconds: Optional[int] = None
    light_sleep_seconds: Optional[int] = None
    total_sleep_seconds: Optional[int] = None
    efficiency: Optional[float] = None  # 0-100%
    avg_hrv_ms: Optional[float] = None  # RMSSD in ms (SDNN for Apple Health)
    avg_resting_hr_bpm: Optional[float] = None
    sleep_type: Optional[str] = None  # "long_sleep", "nap", etc.
    bedtime_start: Optional[str] = None  # ISO 8601 timestamp
    bedtime_end: Optional[str] = None  # ISO 8601 timestamp
    onset_latency_seconds: Optional[int] = None  # time to fall asleep
    awake_seconds: Optional[int] = None  # total awake time during sleep
    hypnogram_5min: Optional[str] = None  # sleep stage string: 1=deep, 2=light, 3=REM, 4=awake


@dataclass
class ReadinessRecord:
    day: str
    provider: str
    score: Optional[int] = None  # 0-100
    temp_deviation_c: Optional[float] = None
    temp_body_score: Optional[int] = None
    resting_hr_score: Optional[int] = None
    hrv_balance_score: Optional[int] = None
    recovery_index_score: Optional[int] = None
    sleep_balance_score: Optional[int] = None
    activity_balance_score: Optional[int] = None


@dataclass
class ActivityRecord:
    day: str
    provider: str
    score: Optional[int] = None
    steps: Optional[int] = None
    total_calories: Optional[int] = None
    met_average: Optional[float] = None


@dataclass
class StressRecord:
    day: str
    provider: str
    stress_high_minutes: Optional[int] = None
    recovery_high_minutes: Optional[int] = None


@dataclass
class SpO2Record:
    day: str
    provider: str
    avg_spo2_pct: Optional[float] = None
    breathing_disturbance_index: Optional[float] = None


@dataclass
class ResilienceRecord:
    day: str
    provider: str
    level: Optional[str] = None  # "excellent", "good", "fair", "limited"
    sleep_recovery: Optional[float] = None
    daytime_recovery: Optional[float] = None


@dataclass
class TagRecord:
    day: str
    provider: str
    timestamp: Optional[str] = None
    tag_type: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class HeartRateRecord:
    timestamp: str  # ISO 8601
    provider: str
    bpm: Optional[int] = None
    source: Optional[str] = None  # "rest", "awake", "sleep"


@dataclass
class WorkoutRecord:
    day: str
    provider: str
    activity: Optional[str] = None  # "walking", "running", etc.
    calories: Optional[float] = None
    distance_m: Optional[float] = None
    intensity: Optional[str] = None  # "low", "moderate", "high"
    duration_seconds: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class MealRecord:
    day: str  # YYYY-MM-DD
    provider: str  # "suna", "passio", "manual"
    timestamp: Optional[str] = None  # ISO 8601 when meal was eaten
    meal_type: Optional[str] = None  # "breakfast", "lunch", "dinner", "snack"
    description: Optional[str] = None  # "chicken breast with rice"
    # Macros
    calories: Optional[int] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sugar_g: Optional[float] = None
    saturated_fat_g: Optional[float] = None
    alcohol_units: Optional[float] = None
    # Micros (clinically relevant subset)
    sodium_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    magnesium_mg: Optional[float] = None
    caffeine_mg: Optional[float] = None  # not from Passio — separate tracking
    # Food-level detail
    foods: Optional[list] = None  # [{"name": "chicken breast", "calories": 165, "protein_g": 31, ...}]


@dataclass
class UserProfile:
    provider: str
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_m: Optional[float] = None
    biological_sex: Optional[str] = None


@dataclass
class BiometricData:
    provider: str
    period_start: str
    period_end: str
    user: Optional[UserProfile] = None
    sleep: list = field(default_factory=list)  # list[SleepRecord]
    readiness: list = field(default_factory=list)  # list[ReadinessRecord]
    activity: list = field(default_factory=list)  # list[ActivityRecord]
    stress: list = field(default_factory=list)  # list[StressRecord]
    spo2: list = field(default_factory=list)  # list[SpO2Record]
    resilience: list = field(default_factory=list)  # list[ResilienceRecord]
    tags: list = field(default_factory=list)  # list[TagRecord]
    heartrate: list = field(default_factory=list)  # list[HeartRateRecord]
    workouts: list = field(default_factory=list)  # list[WorkoutRecord]
    meals: list = field(default_factory=list)  # list[MealRecord]
    optimal_bedtime: Optional[str] = None
    warnings: list = field(default_factory=list)  # list[str] — fetch errors surfaced to user

    def to_dict(self):
        return asdict(self)
