"""Data validation layer — run before metrics computation.

Deduplicates, sorts, flags outliers, and warns about gaps.
All functions are pure: dict in, dict out.
"""


# Outlier thresholds
OUTLIER_LIMITS = {
    'avg_hrv_ms': (0, 500),
    'avg_resting_hr_bpm': (20, 250),
    'efficiency': (0, 100),
    'score': (0, 100),
    'deep_sleep_seconds': (0, 36000),   # max 10h
    'rem_sleep_seconds': (0, 36000),
    'total_sleep_seconds': (0, 72000),  # max 20h
    'avg_spo2_pct': (50, 100),
    'bpm': (20, 250),
}


def validate_data(data: dict) -> dict:
    """Validate and clean biometric data dict before metrics.

    Args:
        data: Result of asdict(BiometricData)

    Returns:
        Cleaned copy of data with '_validation' key added containing:
        - duplicates_removed: int
        - outliers_flagged: int
        - warnings: list[str]
    """
    warnings = []
    dupes_removed = 0
    outliers_flagged = 0

    # Process each day-keyed record list
    for key in ['sleep', 'readiness', 'activity', 'stress', 'spo2',
                'resilience', 'body_composition', 'respiration']:
        records = data.get(key, [])
        if not records:
            continue

        # Deduplicate by (day, provider)
        seen = set()
        deduped = []
        for r in records:
            ident = (r.get('day', ''), r.get('provider', ''))
            if ident in seen:
                dupes_removed += 1
                continue
            seen.add(ident)
            deduped.append(r)

        # Sort by date
        deduped.sort(key=lambda r: r.get('day', ''))

        # Flag outliers
        for r in deduped:
            for field, (lo, hi) in OUTLIER_LIMITS.items():
                val = r.get(field)
                if val is not None and (val < lo or val > hi):
                    outliers_flagged += 1
                    warnings.append(
                        f"{key}/{r.get('day', '?')}: {field}={val} outside [{lo}, {hi}]"
                    )

        data[key] = deduped

    # Check for date gaps in sleep data
    sleep = data.get('sleep', [])
    if len(sleep) >= 2:
        from datetime import datetime, timedelta
        days_with_data = sorted(set(s.get('day', '') for s in sleep if s.get('day')))
        if len(days_with_data) >= 2:
            gaps = []
            for i in range(1, len(days_with_data)):
                try:
                    d1 = datetime.strptime(days_with_data[i - 1], "%Y-%m-%d")
                    d2 = datetime.strptime(days_with_data[i], "%Y-%m-%d")
                    gap = (d2 - d1).days
                    if gap > 2:
                        gaps.append(f"{days_with_data[i-1]} to {days_with_data[i]} ({gap} days)")
                except ValueError:
                    pass
            if gaps:
                warnings.append(f"Sleep data gaps: {', '.join(gaps[:5])}")

    data['_validation'] = {
        'duplicates_removed': dupes_removed,
        'outliers_flagged': outliers_flagged,
        'warnings': warnings,
    }
    return data
