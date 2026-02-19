"""Abstract base class for wearable data providers."""

from abc import ABC, abstractmethod
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (
    UserProfile, SleepRecord, ReadinessRecord, ActivityRecord,
    StressRecord, SpO2Record, ResilienceRecord, TagRecord,
    HeartRateRecord, WorkoutRecord,
)


class BaseProvider(ABC):
    """All wearable providers implement this interface."""

    name: str = "base"

    @abstractmethod
    def test_connection(self) -> dict:
        """Test API connection. Returns dict with 'connected' bool and 'info' str."""
        ...

    @abstractmethod
    def fetch_user_profile(self) -> Optional[UserProfile]:
        """Fetch user profile info."""
        ...

    @abstractmethod
    def fetch_sleep(self, start_date: str, end_date: str) -> list:
        """Fetch sleep records for date range. Returns list[SleepRecord]."""
        ...

    @abstractmethod
    def fetch_readiness(self, start_date: str, end_date: str) -> list:
        """Fetch readiness/recovery records. Returns list[ReadinessRecord]."""
        ...

    @abstractmethod
    def fetch_activity(self, start_date: str, end_date: str) -> list:
        """Fetch activity records. Returns list[ActivityRecord]."""
        ...

    @abstractmethod
    def fetch_stress(self, start_date: str, end_date: str) -> list:
        """Fetch stress records. Returns list[StressRecord]."""
        ...

    def fetch_spo2(self, start_date: str, end_date: str) -> list:
        """Fetch SpO2/blood oxygen records. Returns list[SpO2Record]."""
        return []

    def fetch_resilience(self, start_date: str, end_date: str) -> list:
        """Fetch resilience/recovery records. Returns list[ResilienceRecord]."""
        return []

    def fetch_tags(self, start_date: str, end_date: str) -> list:
        """Fetch user-entered tags. Returns list[TagRecord]."""
        return []

    def fetch_heartrate(self, start_date: str, end_date: str) -> list:
        """Fetch 5-min interval heart rate data. Returns list[HeartRateRecord]."""
        return []

    def fetch_workouts(self, start_date: str, end_date: str) -> list:
        """Fetch workout/exercise sessions. Returns list[WorkoutRecord]."""
        return []

    def fetch_sleep_time(self, start_date: str, end_date: str) -> Optional[str]:
        """Fetch optimal bedtime recommendation. Returns time string or None."""
        return None

    def fetch_available_data(self, start_date: str, end_date: str) -> dict:
        """Check what data is available. Returns dict of endpoint -> record count."""
        counts = {}
        for name, method in [
            ("sleep", self.fetch_sleep),
            ("readiness", self.fetch_readiness),
            ("activity", self.fetch_activity),
            ("stress", self.fetch_stress),
        ]:
            try:
                records = method(start_date, end_date)
                counts[name] = len(records)
            except Exception:
                counts[name] = 0
        return counts
