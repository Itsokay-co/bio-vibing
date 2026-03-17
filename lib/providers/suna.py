"""Suna provider — fetches meals and scores from Suna Health API.

Consumer only — does not implement any scoring logic.

Env vars:
    SUNA_API_URL — base URL
    SUNA_API_KEY — Bearer token
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import (MealRecord, GutScore, OvernightScore,
                    DigestiveState, DailyWindows)


class SunaProvider:
    """Fetches data from Suna Health REST API."""

    name = "suna"

    def __init__(self):
        self.api_url = os.environ.get("SUNA_API_URL", "").rstrip("/")
        self.api_key = os.environ.get("SUNA_API_KEY", "")

        if not self.api_url or not self.api_key:
            raise ValueError(
                "Suna provider requires SUNA_API_URL and SUNA_API_KEY env vars"
            )

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None,
             retries: int = 2) -> list | dict:
        """Authenticated GET to Suna API. Returns unwrapped data from envelope."""
        url = f"{self.api_url}/v1{path}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {k: v for k, v in params.items() if v is not None})

        for attempt in range(retries + 1):
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self.api_key}")
            req.add_header("Accept", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    body = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries:
                    try:
                        retry_after = int(e.headers.get("Retry-After", "5"))
                    except (ValueError, TypeError):
                        retry_after = 5
                    time.sleep(min(retry_after, 30))
                    continue
                err_body = e.read().decode() if e.fp else ""
                raise RuntimeError(
                    f"Suna API error {e.code}: {err_body}") from e

            # Unwrap Stripe-style envelope
            if isinstance(body, dict):
                if body.get("status") == "error":
                    err = body.get("error", {})
                    raise RuntimeError(
                        f"Suna API: {err.get('code', 'unknown')} — "
                        f"{err.get('message', 'no message')}")
                return body.get("data", body)
            return body

        raise RuntimeError(f"Suna API: retries exhausted for {path}")

    # ------------------------------------------------------------------
    # Connection test
    # ------------------------------------------------------------------

    def test_connection(self) -> dict:
        """GET /v1/user/devices — verify API key and show data counts."""
        try:
            data = self._get("/user/me")
            return {
                "connected": True,
                "info": f"Suna API ({self.api_url})",
                "available_data": data if isinstance(data, dict) else {},
            }
        except Exception as e:
            return {"connected": False, "info": str(e), "available_data": {}}

    # ------------------------------------------------------------------
    # Meals
    # ------------------------------------------------------------------

    def fetch_meals(self, start_date: str, end_date: str) -> list:
        """GET /v1/nutrition/meals → list[MealRecord]"""
        data = self._get("/nutrition/meals", {
            "start_date": start_date,
            "end_date": end_date,
            "limit": 200,
        })
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        records = []
        for row in data:
            macros = row.get("macros", {})
            micros = row.get("micros", {})
            records.append(MealRecord(
                day=row.get("day", ""),
                provider="suna",
                timestamp=row.get("timestamp"),
                meal_type=row.get("meal_type"),
                description=row.get("description"),
                calories=_to_int(macros.get("calories", row.get("calories"))),
                protein_g=macros.get("protein_g", row.get("protein_g")),
                carbs_g=macros.get("carbs_g", row.get("carbs_g")),
                fat_g=macros.get("fat_g", row.get("fat_g")),
                fiber_g=macros.get("fiber_g", row.get("fiber_g")),
                sugar_g=macros.get("sugar_g", row.get("sugar_g")),
                saturated_fat_g=macros.get("saturated_fat_g",
                                           row.get("saturated_fat_g")),
                alcohol_units=macros.get("alcohol_units",
                                         row.get("alcohol_units")),
                sodium_mg=micros.get("sodium_mg", row.get("sodium_mg")),
                iron_mg=micros.get("iron_mg", row.get("iron_mg")),
                magnesium_mg=micros.get("magnesium_mg",
                                        row.get("magnesium_mg")),
                caffeine_mg=micros.get("caffeine_mg",
                                       row.get("caffeine_mg")),
                foods=_parse_foods(row.get("foods", row.get("food_items"))),
            ))
        return records

    # ------------------------------------------------------------------
    # Proprietary scores (consumed, not computed here)
    # ------------------------------------------------------------------

    def fetch_gut_scores(self, start_date: str, end_date: str) -> list:
        """GET /v1/digestion/scores/daily → list[GutScore]"""
        data = self._get("/digestion/scores/daily", {
            "start_date": start_date, "end_date": end_date,
        })
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        records = []
        for row in data:
            records.append(GutScore(
                day=row.get("date", row.get("day", "")),
                score=row.get("score"),
                level=row.get("level"),
                components=row.get("components"),
            ))
        return records

    def fetch_overnight_scores(self, start_date: str, end_date: str) -> list:
        """GET /v1/digestion/scores/overnight → list[OvernightScore]"""
        data = self._get("/digestion/scores/overnight", {
            "start_date": start_date, "end_date": end_date,
        })
        if not isinstance(data, list):
            data = data.get("data", []) if isinstance(data, dict) else []

        return [OvernightScore(
            day=row.get("date", row.get("day", "")),
            score=row.get("score"),
            level=row.get("level"),
            deviation=row.get("deviation", row.get("deviation_14d",
                              row.get("deviation_from_baseline"))),
            details={k: v for k, v in row.items()
                     if k not in ("date", "day", "score", "level",
                                  "deviation", "deviation_14d",
                                  "deviation_from_baseline")} or None,
        ) for row in data]

    def fetch_digestive_states(self, start_date: str, end_date: str) -> list:
        """GET /v1/digestion/timeline → list[DigestiveState]"""
        data = self._get("/digestion/timeline", {
            "start_date": start_date, "end_date": end_date,
        })
        if not isinstance(data, list):
            # Timeline endpoint may return {date, timelines: [...]}
            if isinstance(data, dict) and "timelines" in data:
                data = data["timelines"]
            else:
                data = data.get("data", []) if isinstance(data, dict) else []

        return [DigestiveState(
            day=row.get("date", row.get("day", "")),
            meal_time=row.get("meal_time"),
            meal_type=row.get("meal_type"),
            duration_min=row.get("total_processing_min",
                                 row.get("duration_min")),
            phases=row.get("phases"),
            confidence=row.get("confidence"),
        ) for row in data]

    def fetch_windows(self, start_date: str, end_date: str) -> list:
        """GET /v1/digestion/windows → list[DailyWindows]"""
        data = self._get("/digestion/windows", {
            "start_date": start_date, "end_date": end_date,
        })
        if not isinstance(data, list):
            data = [data] if isinstance(data, dict) and "day" in data else []

        records = []
        for row in data:
            eat = row.get("eat", {})
            train = row.get("train", {})
            recovery = row.get("recovery", row.get("recovery_context", {}))
            records.append(DailyWindows(
                day=row.get("date", row.get("day", "")),
                eat_start=eat.get("start") if isinstance(eat, dict)
                          else row.get("eat_start"),
                eat_end=eat.get("end") if isinstance(eat, dict)
                        else row.get("eat_end"),
                train_start=train.get("start") if isinstance(train, dict)
                            else row.get("train_start"),
                train_end=train.get("end") if isinstance(train, dict)
                          else row.get("train_end"),
                sleep_start=row.get("sleep", {}).get("start")
                            if isinstance(row.get("sleep"), dict)
                            else row.get("sleep_start"),
                recovery_level=recovery.get("level")
                               if isinstance(recovery, dict) else None,
                adjustments=recovery if isinstance(recovery, dict) else None,
            ))
        return records

    def fetch_insights(self, start_date: str, end_date: str) -> list:
        """GET /v1/digestion/insights → list[dict]"""
        data = self._get("/digestion/insights", {
            "start_date": start_date, "end_date": end_date,
        })
        return data if isinstance(data, list) else []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _to_int(val) -> Optional[int]:
    """Convert to int if not None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_foods(val) -> Optional[list]:
    """Parse foods field — handles JSONB string or list."""
    if val is None:
        return None
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            return None
    return val if isinstance(val, list) else None
