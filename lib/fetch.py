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
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema import BiometricData
from providers import detect_provider, get_provider, list_configured_providers
import cache


def _fetch_with_retry(method, *args, max_retries=2):
    """Call a provider method with retry on transient HTTP errors."""
    for attempt in range(max_retries + 1):
        try:
            return method(*args)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise


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
        ("body_composition", provider.fetch_body_composition, "body_composition"),
        ("respiration", provider.fetch_respiration, "respiration"),
        ("glucose", provider.fetch_glucose, "glucose"),
    ]:
        cached = None
        if use_cache:
            cached = cache.get_cached(provider.name, start_date, end_date, category)

        if cached is not None:
            setattr(data, attr, cached)
        else:
            try:
                records = _fetch_with_retry(method, start_date, end_date)
                serialized = [asdict(r) for r in records]
                setattr(data, attr, serialized)
                if use_cache:
                    cache.set_cached(provider.name, start_date, end_date, category, serialized)
            except Exception as e:
                setattr(data, attr, [])
                data.warnings.append(f"Failed to fetch {category}: {e}")

    # Suna API — meals + scores (single integration point)
    try:
        from providers.suna import SunaProvider
        suna = SunaProvider()

        # Meals
        cached_meals = cache.get_cached("suna", start_date, end_date, "meals") if use_cache else None
        if cached_meals is not None:
            data.meals = cached_meals
        else:
            meals = suna.fetch_meals(start_date, end_date)
            serialized_meals = [asdict(m) for m in meals]
            data.meals = serialized_meals
            if use_cache:
                cache.set_cached("suna", start_date, end_date, "meals", serialized_meals)

        # Proprietary scores (gut scores, overnight, states, windows, insights)
        for category, fetch_fn, attr in [
            ("gut_scores", suna.fetch_gut_scores, "gut_scores"),
            ("overnight_scores", suna.fetch_overnight_scores, "overnight_scores"),
            ("digestive_states", suna.fetch_digestive_states, "digestive_states"),
            ("daily_windows", suna.fetch_windows, "daily_windows"),
            ("suna_insights", suna.fetch_insights, "suna_insights"),
        ]:
            try:
                cached = cache.get_cached("suna", start_date, end_date, category) if use_cache else None
                if cached is not None:
                    setattr(data, attr, cached)
                else:
                    result = fetch_fn(start_date, end_date)
                    serialized = [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in result]
                    setattr(data, attr, serialized)
                    if use_cache:
                        cache.set_cached("suna", start_date, end_date, category, serialized)
            except Exception as e:
                data.warnings.append(f"Failed to fetch {category} from Suna: {e}")

    except (ValueError, ImportError):
        pass  # Suna not configured — meals and scores stay empty
    except Exception as e:
        data.warnings.append(f"Failed to fetch from Suna API: {e}")

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


def fetch_biometrics_multi(
    days: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> BiometricData:
    """Fetch from ALL configured providers and merge by priority.

    Day-keyed records: keep highest-priority per day per category.
    Timestamp-keyed records (heartrate): merge all.
    Priority order from BIOMETRIC_PROVIDER_PRIORITY env var or detection order.
    """
    configured = list_configured_providers()
    if not configured:
        raise ValueError("No wearable providers configured")

    priority_str = os.environ.get("BIOMETRIC_PROVIDER_PRIORITY", "")
    if priority_str:
        priority = [p.strip() for p in priority_str.split(",") if p.strip()]
    else:
        priority = configured

    # Fetch from each provider
    datasets = []
    for pname in configured:
        try:
            d = fetch_biometrics(
                days=days, start_date=start_date, end_date=end_date,
                provider_name=pname, use_cache=use_cache,
            )
            datasets.append((pname, d))
        except Exception:
            continue

    if not datasets:
        raise ValueError("All providers failed to return data")

    # Use highest-priority dataset as base
    def _priority_rank(name):
        try:
            return priority.index(name)
        except ValueError:
            return 999

    datasets.sort(key=lambda x: _priority_rank(x[0]))
    base_name, base = datasets[0]

    # Merge day-keyed records from lower-priority providers (fill gaps only)
    day_keyed = ['sleep', 'readiness', 'activity', 'stress', 'spo2',
                 'resilience', 'body_composition', 'respiration', 'workouts']
    for pname, d in datasets[1:]:
        dd = asdict(d)
        for key in day_keyed:
            existing_days = {r.get('day') for r in getattr(base, key, [])}
            for r in dd.get(key, []):
                if r.get('day') not in existing_days:
                    getattr(base, key).append(r)

        # Heartrate: merge all (timestamp-keyed, no conflicts)
        for hr in dd.get('heartrate', []):
            base.heartrate.append(hr)

        # Meals: merge all
        for m in dd.get('meals', []):
            base.meals.append(m)

        base.warnings.extend(d.warnings)

    base.provider = "multi"
    return base


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
