"""Suna provider — fetches meal records from Supabase for crossover metrics."""

import json
import os
import sys
import urllib.request
import urllib.error
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import MealRecord


class SunaProvider:
    """Fetches meal data from Suna's Supabase meal_records table."""

    name = "suna"

    def __init__(self):
        self.supabase_url = os.environ.get("SUNA_SUPABASE_URL", "")
        self.supabase_key = os.environ.get("SUNA_SUPABASE_KEY", "")
        self.user_id = os.environ.get("SUNA_USER_ID", "")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Suna provider requires SUNA_SUPABASE_URL and SUNA_SUPABASE_KEY env vars"
            )

    def test_connection(self) -> dict:
        try:
            meals = self.fetch_meals("2000-01-01", "2099-12-31")
            return {
                "connected": True,
                "info": f"Suna Supabase ({len(meals)} meal records)",
                "available_data": {"meals": len(meals)},
            }
        except Exception as e:
            return {"connected": False, "info": str(e), "available_data": {}}

    def fetch_meals(self, start_date: str, end_date: str) -> list:
        """Fetch meal records from Supabase REST API."""
        url = (
            f"{self.supabase_url}/rest/v1/meal_records"
            f"?day=gte.{start_date}&day=lte.{end_date}"
            f"&order=day.desc,timestamp.desc"
        )
        if self.user_id:
            url += f"&user_id=eq.{self.user_id}"

        req = urllib.request.Request(url)
        req.add_header("apikey", self.supabase_key)
        req.add_header("Authorization", f"Bearer {self.supabase_key}")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                rows = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Supabase API error {e.code}: {body}") from e

        records = []
        for row in rows:
            # Parse food_items from JSONB
            food_items = None
            if row.get("food_items"):
                fi = row["food_items"]
                if isinstance(fi, str):
                    try:
                        fi = json.loads(fi)
                    except json.JSONDecodeError:
                        fi = None
                if isinstance(fi, list):
                    food_items = fi

            records.append(MealRecord(
                day=row.get("day", ""),
                provider="suna",
                timestamp=row.get("timestamp"),
                meal_type=row.get("meal_type"),
                description=row.get("description"),
                calories=_to_int(row.get("calories")),
                protein_g=row.get("protein_g"),
                carbs_g=row.get("carbs_g"),
                fat_g=row.get("fat_g"),
                fiber_g=row.get("fiber_g"),
                sugar_g=row.get("sugar_g"),
                saturated_fat_g=row.get("saturated_fat_g"),
                alcohol_units=row.get("alcohol_units"),
                sodium_mg=row.get("sodium_mg"),
                iron_mg=row.get("iron_mg"),
                magnesium_mg=row.get("magnesium_mg"),
                caffeine_mg=row.get("caffeine_mg"),
                foods=food_items,
            ))

        return records


def _to_int(val) -> Optional[int]:
    """Convert to int if not None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
