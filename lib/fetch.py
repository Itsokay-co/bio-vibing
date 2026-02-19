"""Entry point for fetching normalized biometric data.

Usage from skills:
    import sys, os
    sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
    from fetch import fetch_biometrics, test_connection

    data = fetch_biometrics(days=14)
"""

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import BiometricData
from providers import detect_provider, get_provider, list_configured_providers
import cache


def test_connection(provider_name: Optional[str] = None) -> dict:
    """Test connection to the configured wearable.

    Returns dict with keys: provider, connected, info, available_data
    """
    provider = get_provider(provider_name)
    result = provider.test_connection()
    return {
        "provider": provider.name,
        **result,
    }


def fetch_biometrics(
    days: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider_name: Optional[str] = None,
    use_cache: bool = True,
) -> BiometricData:
    """Fetch normalized biometric data from the configured wearable.

    Args:
        days: Number of days to fetch (ignored if start_date/end_date provided)
        start_date: Start date YYYY-MM-DD (optional)
        end_date: End date YYYY-MM-DD (optional, defaults to today)
        provider_name: Force a specific provider (auto-detects if None)
        use_cache: Whether to use the local cache

    Returns:
        BiometricData with normalized records from the provider
    """
    provider = get_provider(provider_name)

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    data = BiometricData(
        provider=provider.name,
        period_start=start_date,
        period_end=end_date,
    )

    # Fetch each category with caching
    for category, method, attr in [
        ("sleep", provider.fetch_sleep, "sleep"),
        ("readiness", provider.fetch_readiness, "readiness"),
        ("activity", provider.fetch_activity, "activity"),
        ("stress", provider.fetch_stress, "stress"),
        ("spo2", provider.fetch_spo2, "spo2"),
        ("resilience", provider.fetch_resilience, "resilience"),
        ("tags", provider.fetch_tags, "tags"),
        ("heartrate", provider.fetch_heartrate, "heartrate"),
        ("workouts", provider.fetch_workouts, "workouts"),
    ]:
        cached = None
        if use_cache:
            cached = cache.get_cached(provider.name, start_date, end_date, category)

        if cached is not None:
            setattr(data, attr, cached)
        else:
            try:
                records = method(start_date, end_date)
                serialized = [asdict(r) for r in records]
                setattr(data, attr, serialized)
                if use_cache:
                    cache.set_cached(provider.name, start_date, end_date, category, serialized)
            except Exception as e:
                setattr(data, attr, [])
                data.warnings.append(f"Failed to fetch {category}: {e}")

    # Meals from Suna provider (separate from wearable provider)
    try:
        from providers.suna import SunaProvider
        suna = SunaProvider()
        cached_meals = None
        if use_cache:
            cached_meals = cache.get_cached("suna", start_date, end_date, "meals")
        if cached_meals is not None:
            data.meals = cached_meals
        else:
            meals = suna.fetch_meals(start_date, end_date)
            serialized_meals = [asdict(m) for m in meals]
            data.meals = serialized_meals
            if use_cache:
                cache.set_cached("suna", start_date, end_date, "meals", serialized_meals)
    except (ValueError, ImportError):
        pass  # Suna not configured — meals stay empty
    except Exception as e:
        data.warnings.append(f"Failed to fetch meals from Suna: {e}")

    # User profile (not cached by date range)
    try:
        profile = provider.fetch_user_profile()
        data.user = asdict(profile) if profile else None
    except Exception:
        data.user = None

    # Optimal bedtime (not cached)
    try:
        data.optimal_bedtime = provider.fetch_sleep_time(start_date, end_date)
    except Exception:
        data.optimal_bedtime = None

    # Surface warnings so users know data may be incomplete
    if data.warnings:
        print(f"\n--- WARNINGS ({len(data.warnings)}) ---", file=sys.stderr)
        for w in data.warnings:
            print(f"  ! {w}", file=sys.stderr)
        print("", file=sys.stderr)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch biometric data")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test connection only")
    args = parser.parse_args()

    if args.test:
        result = test_connection(args.provider)
        print(json.dumps(result, indent=2))
    else:
        data = fetch_biometrics(
            days=args.days,
            start_date=args.start,
            end_date=args.end,
            provider_name=args.provider,
            use_cache=not args.no_cache,
        )
        print(json.dumps(asdict(data), indent=2))
