"""Menstrual cycle detection from wearable biometrics.

Detects cycle phases from temperature deviation, HRV, and resting heart rate —
no user tagging required. Uses the thermal shift method (same principle as
fertility awareness) enhanced with autonomic nervous system signals.

Usage:
    from cycle import detect_cycle_phases

    phases = detect_cycle_phases(readiness_records, sleep_records)
    print(phases['current_phase'])       # "follicular", "luteal", "menstrual"
    print(phases['estimated_cycle_day']) # 18
    print(phases['detected_periods'])    # ["2025-12-04", "2026-01-03", ...]
"""

from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Optional


def detect_cycle_phases(
    readiness: list,
    sleep: list,
    period_tags: Optional[list] = None,
) -> dict:
    """Detect menstrual cycle phases from biometric data.

    Uses three signals (temperature, HRV, RHR) with temperature as primary.
    If period tags exist, uses them as ground truth and fills in phase estimates
    between them. If not, detects phases purely from biometric patterns.

    Args:
        readiness: List of readiness record dicts with 'day' and 'temp_deviation_c'
        sleep: List of sleep record dicts with 'day', 'avg_hrv_ms', 'avg_resting_hr_bpm'
        period_tags: Optional list of tag dicts with 'day' and 'tag_type' containing 'period'

    Returns:
        Dict with:
            current_phase: str — "follicular", "luteal", "menstrual", or "unknown"
            estimated_cycle_day: int or None
            detected_periods: list of YYYY-MM-DD period start dates
            cycle_length: estimated cycle length in days
            confidence: "high" (tags), "medium" (strong thermal shift), "low" (weak signal)
            phases: list of {start, end, phase} dicts for the full timeline
            next_period_estimate: YYYY-MM-DD or None
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # --- If we have period tags, use them as ground truth ---
    known_period_starts = _extract_period_starts(period_tags) if period_tags else []

    if known_period_starts:
        return _phases_from_tags(known_period_starts, today)

    # --- No tags: detect from biometrics ---
    temp_by_day = {
        r['day']: r['temp_deviation_c']
        for r in readiness
        if r.get('temp_deviation_c') is not None
    }
    hrv_by_day = {
        s['day']: s['avg_hrv_ms']
        for s in sleep
        if s.get('avg_hrv_ms')
    }
    rhr_by_day = {
        s['day']: s['avg_resting_hr_bpm']
        for s in sleep
        if s.get('avg_resting_hr_bpm')
    }

    if len(temp_by_day) < 21:
        return _unknown_result("Insufficient data — need 21+ days of temperature readings")

    return _detect_from_biometrics(temp_by_day, hrv_by_day, rhr_by_day, today)


def _extract_period_starts(tags: list) -> list:
    """Extract period start dates from tag records.

    Groups consecutive period-tagged days into periods,
    returns the first day of each group.
    """
    period_days = sorted(set(
        t['day'] for t in tags
        if 'period' in (t.get('tag_type') or '').lower()
    ))
    if not period_days:
        return []

    # Group consecutive days (within 3 days of each other = same period)
    groups = []
    current_group = [period_days[0]]
    for day in period_days[1:]:
        prev = datetime.strptime(current_group[-1], "%Y-%m-%d")
        curr = datetime.strptime(day, "%Y-%m-%d")
        if (curr - prev).days <= 3:
            current_group.append(day)
        else:
            groups.append(current_group)
            current_group = [day]
    groups.append(current_group)

    return [g[0] for g in groups]


def _phases_from_tags(period_starts: list, today: str) -> dict:
    """Build cycle phase timeline from known period start dates."""
    # Estimate cycle length from gaps between periods
    if len(period_starts) >= 2:
        gaps = []
        for i in range(1, len(period_starts)):
            d1 = datetime.strptime(period_starts[i - 1], "%Y-%m-%d")
            d2 = datetime.strptime(period_starts[i], "%Y-%m-%d")
            gap = (d2 - d1).days
            if 21 <= gap <= 40:
                gaps.append(gap)
        cycle_length = round(mean(gaps)) if gaps else 28
    else:
        cycle_length = 28

    # Build phase timeline
    phases = []
    for i, ps in enumerate(period_starts):
        start_dt = datetime.strptime(ps, "%Y-%m-%d")
        phases.append({
            "start": ps,
            "end": (start_dt + timedelta(days=6)).strftime("%Y-%m-%d"),
            "phase": "menstrual",
        })
        phases.append({
            "start": (start_dt + timedelta(days=7)).strftime("%Y-%m-%d"),
            "end": (start_dt + timedelta(days=14)).strftime("%Y-%m-%d"),
            "phase": "follicular",
        })
        next_period = period_starts[i + 1] if i + 1 < len(period_starts) else None
        luteal_end = next_period if next_period else (start_dt + timedelta(days=cycle_length - 1)).strftime("%Y-%m-%d")
        phases.append({
            "start": (start_dt + timedelta(days=15)).strftime("%Y-%m-%d"),
            "end": luteal_end,
            "phase": "luteal",
        })

    # Current phase
    last_period = datetime.strptime(period_starts[-1], "%Y-%m-%d")
    today_dt = datetime.strptime(today, "%Y-%m-%d")
    days_since = (today_dt - last_period).days

    if days_since < 0:
        current_phase = "unknown"
        cycle_day = None
    elif days_since <= 7:
        current_phase = "menstrual"
        cycle_day = days_since + 1
    elif days_since <= 14:
        current_phase = "follicular"
        cycle_day = days_since + 1
    elif days_since <= cycle_length:
        current_phase = "luteal"
        cycle_day = days_since + 1
    else:
        # Past expected cycle length — period may be late
        current_phase = "luteal (extended)"
        cycle_day = days_since + 1

    # Estimate next period
    next_period_est = (last_period + timedelta(days=cycle_length)).strftime("%Y-%m-%d")

    return {
        "current_phase": current_phase,
        "estimated_cycle_day": cycle_day,
        "detected_periods": period_starts,
        "cycle_length": cycle_length,
        "confidence": "high",
        "source": "period_tags",
        "phases": phases,
        "next_period_estimate": next_period_est,
        "note": None,
    }


def _detect_from_biometrics(
    temp_by_day: dict,
    hrv_by_day: dict,
    rhr_by_day: dict,
    today: str,
) -> dict:
    """Detect cycle phases purely from temperature + HRV + RHR patterns.

    Two-pass approach:
    1. Temperature shift detection — compare trailing 7-day mean against
       a lagged baseline to find follicular→luteal transitions (ovulation)
       and luteal→follicular transitions (period onset).
    2. Corroborate with HRV/RHR — luteal phase should show lower HRV
       and higher RHR.
    """
    all_days = sorted(temp_by_day.keys())
    if not all_days:
        return _unknown_result("No temperature data available")

    # --- Step 1: Fill calendar gaps and smooth ---
    # Build a continuous calendar from first to last day
    first_dt = datetime.strptime(all_days[0], "%Y-%m-%d")
    last_dt = datetime.strptime(all_days[-1], "%Y-%m-%d")
    calendar = []
    dt = first_dt
    while dt <= last_dt:
        calendar.append(dt.strftime("%Y-%m-%d"))
        dt += timedelta(days=1)

    # Interpolate missing days (carry forward last known value)
    filled_temp = {}
    last_val = 0
    for day in calendar:
        if day in temp_by_day:
            filled_temp[day] = temp_by_day[day]
            last_val = temp_by_day[day]
        else:
            filled_temp[day] = last_val

    # 7-day rolling mean for stable signal
    smoothed = _rolling_mean(calendar, filled_temp, window=7)

    # --- Step 2: Detect thermal shifts via rolling comparison ---
    # Compare recent 5-day mean against prior 10-day mean
    # A rise of >0.15C = possible ovulation (follicular→luteal)
    # A drop of >0.15C = possible period onset (luteal→follicular)
    shifts = {}
    for i, day in enumerate(calendar):
        if i < 12:
            continue
        recent = [smoothed[calendar[j]] for j in range(i - 4, i + 1) if calendar[j] in smoothed]
        baseline = [smoothed[calendar[j]] for j in range(i - 12, i - 4) if calendar[j] in smoothed]
        if recent and baseline:
            shifts[day] = mean(recent) - mean(baseline)

    # --- Step 3: Find luteal phases as sustained warm periods ---
    # A luteal phase = sustained positive shift lasting 8-16 days
    warm_phases = []
    phase_start = None
    phase_days = []
    below_count = 0

    for day in calendar:
        shift = shifts.get(day, 0)
        temp_val = smoothed.get(day, 0)

        # Consider "warm" if either the shift is positive or absolute temp is elevated
        is_warm = shift > 0.08 or temp_val > 0.15

        if is_warm:
            if phase_start is None:
                phase_start = day
            phase_days.append(day)
            below_count = 0
        else:
            below_count += 1
            if below_count <= 2 and phase_start:
                # Tolerate up to 2 consecutive cold days within a warm phase
                phase_days.append(day)
            elif phase_start:
                # Warm phase ended
                if len(phase_days) >= 7:
                    warm_phases.append({
                        "start": phase_start,
                        "end": phase_days[-below_count] if below_count > 0 else phase_days[-1],
                    })
                phase_start = None
                phase_days = []
                below_count = 0

    # Handle ongoing warm phase
    if phase_start and len(phase_days) >= 7:
        end = phase_days[-below_count] if below_count > 0 else phase_days[-1]
        warm_phases.append({"start": phase_start, "end": end})

    # --- Step 4: Merge overlapping or close warm phases ---
    merged = []
    for wp in warm_phases:
        if merged:
            prev_end = datetime.strptime(merged[-1]['end'], "%Y-%m-%d")
            curr_start = datetime.strptime(wp['start'], "%Y-%m-%d")
            if (curr_start - prev_end).days <= 3:
                merged[-1]['end'] = wp['end']
                continue
        merged.append(wp)
    warm_phases = merged

    # --- Step 5: Filter by physiological plausibility ---
    # Luteal phases are typically 10-16 days. Allow 7-20 for noisy data.
    plausible = []
    for wp in warm_phases:
        duration = (datetime.strptime(wp['end'], "%Y-%m-%d") -
                    datetime.strptime(wp['start'], "%Y-%m-%d")).days + 1
        if 7 <= duration <= 22:
            plausible.append(wp)
        elif duration > 22:
            # Too long — might be two merged cycles. Take the last 14 days.
            new_start = (datetime.strptime(wp['end'], "%Y-%m-%d") -
                         timedelta(days=13)).strftime("%Y-%m-%d")
            plausible.append({"start": new_start, "end": wp['end']})
    warm_phases = plausible

    # --- Step 6: Corroborate with HRV/RHR ---
    # For each warm phase, check if HRV is lower and RHR is higher than
    # the surrounding cold period. This increases confidence.
    corroborated = 0
    for wp in warm_phases:
        wp_start = datetime.strptime(wp['start'], "%Y-%m-%d")
        wp_end = datetime.strptime(wp['end'], "%Y-%m-%d")

        warm_hrv = [hrv_by_day[d] for d in hrv_by_day
                     if wp['start'] <= d <= wp['end']]
        cold_hrv = [hrv_by_day[d] for d in hrv_by_day
                     if d < wp['start'] or d > wp['end']]
        warm_rhr = [rhr_by_day[d] for d in rhr_by_day
                     if wp['start'] <= d <= wp['end']]
        cold_rhr = [rhr_by_day[d] for d in rhr_by_day
                     if d < wp['start'] or d > wp['end']]

        if warm_hrv and cold_hrv and warm_rhr and cold_rhr:
            # Luteal: lower HRV, higher RHR
            hrv_confirms = mean(warm_hrv) < mean(cold_hrv)
            rhr_confirms = mean(warm_rhr) > mean(cold_rhr)
            if hrv_confirms or rhr_confirms:
                corroborated += 1

    # --- Step 7: Detect period starts ---
    # The period starts when temperature begins falling from the luteal peak,
    # not when it finishes falling. Find the peak within each warm phase and
    # look for the first decline after it.
    detected_periods = []
    for wp in warm_phases:
        wp_start_dt = datetime.strptime(wp['start'], "%Y-%m-%d")
        wp_end_dt = datetime.strptime(wp['end'], "%Y-%m-%d")

        # Find peak temperature day within the warm phase
        wp_days = [d for d in calendar if wp['start'] <= d <= wp['end']]
        if not wp_days:
            continue

        peak_day = max(wp_days, key=lambda d: smoothed.get(d, 0))
        peak_dt = datetime.strptime(peak_day, "%Y-%m-%d")

        # Period starts ~1-2 days after the peak (when temp begins declining)
        # Use the midpoint between peak and end of warm phase as the estimate
        days_after_peak = (wp_end_dt - peak_dt).days
        offset = max(1, days_after_peak // 2)
        period_est_dt = peak_dt + timedelta(days=offset)
        period_est = period_est_dt.strftime("%Y-%m-%d")

        if period_est <= today:
            detected_periods.append(period_est)

    # --- Step 8: Estimate cycle length ---
    if len(detected_periods) >= 2:
        gaps = []
        for i in range(1, len(detected_periods)):
            d1 = datetime.strptime(detected_periods[i - 1], "%Y-%m-%d")
            d2 = datetime.strptime(detected_periods[i], "%Y-%m-%d")
            gap = (d2 - d1).days
            if 21 <= gap <= 40:
                gaps.append(gap)
        cycle_length = round(mean(gaps)) if gaps else 28
    else:
        cycle_length = 28

    # --- Step 9: Current phase ---
    if detected_periods:
        last_period = datetime.strptime(detected_periods[-1], "%Y-%m-%d")
        days_since = (datetime.strptime(today, "%Y-%m-%d") - last_period).days

        if days_since <= 7:
            current_phase = "menstrual"
        elif days_since <= 14:
            current_phase = "follicular"
        else:
            current_phase = "luteal"
        cycle_day = days_since + 1
    else:
        # No detected periods — use recent temperature trend
        recent_temps = [smoothed[d] for d in calendar[-7:] if d in smoothed]
        if recent_temps and mean(recent_temps) > 0.1:
            current_phase = "luteal"
        elif recent_temps and mean(recent_temps) < -0.1:
            current_phase = "follicular"
        else:
            current_phase = "unknown"
        cycle_day = None

    # --- Step 10: Confidence ---
    if len(warm_phases) >= 2 and corroborated >= 1:
        confidence = "medium"
    elif len(warm_phases) >= 2:
        confidence = "medium"
    elif len(warm_phases) >= 1 and corroborated >= 1:
        confidence = "low-medium"
    else:
        confidence = "low"

    # Build phase timeline
    phases = []
    for wp in warm_phases:
        phases.append({"start": wp['start'], "end": wp['end'], "phase": "luteal"})

    next_period_est = None
    if detected_periods:
        last_p = datetime.strptime(detected_periods[-1], "%Y-%m-%d")
        next_period_est = (last_p + timedelta(days=cycle_length)).strftime("%Y-%m-%d")

    return {
        "current_phase": current_phase,
        "estimated_cycle_day": cycle_day,
        "detected_periods": detected_periods,
        "cycle_length": cycle_length,
        "confidence": confidence,
        "source": "biometric_detection",
        "phases": phases,
        "next_period_estimate": next_period_est,
        "note": "Detected from temperature + HRV + RHR patterns (no period tags logged)",
    }


def _rolling_mean(days: list, values: dict, window: int = 5) -> dict:
    """Compute rolling mean over a sorted list of days."""
    result = {}
    for i, day in enumerate(days):
        start = max(0, i - window + 1)
        window_days = days[start:i + 1]
        window_vals = [values[d] for d in window_days if d in values]
        if window_vals:
            result[day] = mean(window_vals)
    return result


def _detect_sustained_phases(
    days: list, scores: dict, threshold: float, min_duration: int = 5
) -> list:
    """Find sustained periods where scores stay above threshold."""
    phases = []
    current_start = None
    current_days = []

    for day in days:
        score = scores.get(day, 0)
        if score > threshold:
            if current_start is None:
                current_start = day
            current_days.append(day)
        else:
            if current_start and len(current_days) >= min_duration:
                phases.append({"start": current_start, "end": current_days[-1]})
            current_start = None
            current_days = []

    # Handle ongoing warm phase
    if current_start and len(current_days) >= min_duration:
        phases.append({"start": current_start, "end": current_days[-1]})

    return phases


def _unknown_result(note: str) -> dict:
    return {
        "current_phase": "unknown",
        "estimated_cycle_day": None,
        "detected_periods": [],
        "cycle_length": None,
        "confidence": "none",
        "source": "insufficient_data",
        "phases": [],
        "next_period_estimate": None,
        "note": note,
    }
