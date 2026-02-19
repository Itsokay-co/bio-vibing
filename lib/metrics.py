"""Novel biometric metrics computations.

Reusable across skills. All functions are pure — they take record dicts
(as returned by asdict(BiometricData)) and return structured result dicts.

No external dependencies beyond Python stdlib.

Usage:
    from metrics import compute_hrv_cv, compute_cross_modal_coupling, ...
"""

from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pearson_r(x: list, y: list) -> float:
    """Pearson correlation coefficient, stdlib only."""
    n = len(x)
    if n < 3:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _det3(m):
    """Determinant of a 3x3 matrix (list of 3 lists of 3)."""
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
          - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
          + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _cosinor_fit(hours: list, values: list) -> Optional[dict]:
    """Fit cosinor model HR(t) = M + A*cos(wt) + B*sin(wt) via normal equations.

    Args:
        hours: fractional hours (0-24) for each data point
        values: HR values at each time point

    Returns:
        dict with mesor, amplitude, acrophase_hour, r_squared, or None if insufficient data
    """
    n = len(hours)
    if n < 10:
        return None

    omega = 2 * math.pi / 24.0

    # Build sums for the normal equation X^T X beta = X^T y
    # X columns: [1, cos(wt), sin(wt)]
    s1 = float(n)
    sc = sum(math.cos(omega * t) for t in hours)
    ss = sum(math.sin(omega * t) for t in hours)
    scc = sum(math.cos(omega * t) ** 2 for t in hours)
    sss = sum(math.sin(omega * t) ** 2 for t in hours)
    scs = sum(math.cos(omega * t) * math.sin(omega * t) for t in hours)
    sy = sum(values)
    syc = sum(v * math.cos(omega * t) for t, v in zip(hours, values))
    sys_ = sum(v * math.sin(omega * t) for t, v in zip(hours, values))

    # Solve via Cramer's rule
    mat = [[s1, sc, ss], [sc, scc, scs], [ss, scs, sss]]
    det_main = _det3(mat)
    if abs(det_main) < 1e-10:
        return None

    M = _det3([[sy, sc, ss], [syc, scc, scs], [sys_, scs, sss]]) / det_main
    A = _det3([[s1, sy, ss], [sc, syc, scs], [ss, sys_, sss]]) / det_main
    B = _det3([[s1, sc, sy], [sc, scc, syc], [ss, scs, sys_]]) / det_main

    amplitude = math.sqrt(A ** 2 + B ** 2)
    acrophase_rad = math.atan2(-B, A)
    acrophase_hour = (acrophase_rad * 24 / (2 * math.pi)) % 24

    # R-squared
    y_mean = sum(values) / n
    ss_tot = sum((v - y_mean) ** 2 for v in values)
    ss_res = sum((v - (M + A * math.cos(omega * t) + B * math.sin(omega * t))) ** 2
                 for t, v in zip(hours, values))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "mesor": M,
        "amplitude": amplitude,
        "acrophase_hour": acrophase_hour,
        "r_squared": max(0, r_squared),
    }


def _parse_iso_dt(ts: str) -> Optional[datetime]:
    """Parse ISO 8601 timestamp, handling common formats."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _safe_stdev(vals):
    """Standard deviation that returns 0 for fewer than 2 values."""
    return stdev(vals) if len(vals) > 1 else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_hrv_cv(sleep: list, windows: Optional[list] = None) -> dict:
    """Compute rolling coefficient of variation of nightly HRV.

    CV = (stdev / mean) * 100. Measures day-to-day HRV variability.
    A healthy autonomic system shows moderate variability (CV 8-25%).
    Very low = rigid/overregulated. Very high = chaotic/stressed.

    Args:
        sleep: Sleep record dicts with 'day' and 'avg_hrv_ms'
        windows: Rolling window sizes (default [7, 14])

    Returns:
        dict with current_cv_7d, current_cv_14d, interpretation, trend
    """
    if windows is None:
        windows = [7, 14]

    # Extract nightly HRV, one per day (prefer long_sleep)
    hrv_by_day = {}
    for s in sorted(sleep, key=lambda x: x.get('day', '')):
        if s.get('avg_hrv_ms') and s.get('sleep_type', 'long_sleep') in ('long_sleep', None):
            hrv_by_day[s['day']] = s['avg_hrv_ms']

    days = sorted(hrv_by_day.keys())

    if len(days) < 3:
        return {
            "cv_7d": [], "cv_14d": [],
            "current_cv_7d": None, "current_cv_14d": None,
            "interpretation": "insufficient_data", "trend": "unknown",
        }

    result = {}
    for w in windows:
        key = f"cv_{w}d"
        cv_series = []
        for i in range(w - 1, len(days)):
            window_vals = [hrv_by_day[days[j]] for j in range(i - w + 1, i + 1)]
            if len(window_vals) >= 3:
                m = mean(window_vals)
                s = _safe_stdev(window_vals)
                cv = (s / m * 100) if m > 0 else 0
                cv_series.append({"day": days[i], "cv_pct": round(cv, 1)})
        result[key] = cv_series

    # Current values
    current_7 = result.get("cv_7d", [])[-1]["cv_pct"] if result.get("cv_7d") else None
    current_14 = result.get("cv_14d", [])[-1]["cv_pct"] if result.get("cv_14d") else None

    # Interpretation based on current 7-day CV
    cv = current_7
    if cv is None:
        interpretation = "insufficient_data"
    elif cv < 8:
        interpretation = "rigid"
    elif cv <= 25:
        interpretation = "healthy"
    else:
        interpretation = "chaotic"

    # Trend: compare first half vs second half
    cv_7d = result.get("cv_7d", [])
    if len(cv_7d) >= 6:
        mid = len(cv_7d) // 2
        first_half = mean([c["cv_pct"] for c in cv_7d[:mid]])
        second_half = mean([c["cv_pct"] for c in cv_7d[mid:]])
        diff = second_half - first_half
        if diff > 3:
            trend = "increasing"
        elif diff < -3:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    return {
        "cv_7d": result.get("cv_7d", []),
        "cv_14d": result.get("cv_14d", []),
        "current_cv_7d": current_7,
        "current_cv_14d": current_14,
        "interpretation": interpretation,
        "trend": trend,
    }


def compute_cross_modal_coupling(
    sleep: list,
    readiness: list,
    spo2: list,
    window: int = 14,
) -> dict:
    """Compute correlation between temp deviation, HRV, RHR, and SpO2.

    Healthy systems show tight coupling: when temp rises, HRV drops and
    RHR rises. Decoupling may signal unusual physiological states.

    Args:
        sleep: Sleep record dicts with 'day', 'avg_hrv_ms', 'avg_resting_hr_bpm'
        readiness: Readiness record dicts with 'day' and 'temp_deviation_c'
        spo2: SpO2 record dicts with 'day' and 'avg_spo2_pct'
        window: Number of days for the rolling window

    Returns:
        dict with correlations, coupling_score (0-100), decoupling_events
    """
    # Build day-aligned data
    day_data = {}
    for r in readiness:
        if r.get('temp_deviation_c') is not None:
            day_data.setdefault(r['day'], {})['temp'] = r['temp_deviation_c']
    for s in sleep:
        if s.get('sleep_type', 'long_sleep') not in ('long_sleep', None):
            continue
        d = day_data.setdefault(s.get('day', ''), {})
        if s.get('avg_hrv_ms'):
            d['hrv'] = s['avg_hrv_ms']
        if s.get('avg_resting_hr_bpm'):
            d['rhr'] = s['avg_resting_hr_bpm']
    for sp in spo2:
        if sp.get('avg_spo2_pct') and sp['avg_spo2_pct'] > 0:
            day_data.setdefault(sp['day'], {})['spo2'] = sp['avg_spo2_pct']

    # Get the latest window of days with at least temp + one other metric
    all_days = sorted(d for d in day_data if len(day_data[d]) >= 2)
    if len(all_days) < 5:
        return {
            "correlations": {},
            "coupling_score": None,
            "decoupling_events": [],
            "latest_window": None,
        }

    window_days = all_days[-window:]

    # Expected coupling directions
    pairs = [
        ("temp", "hrv", -1, "temp\u2194HRV"),
        ("temp", "rhr", +1, "temp\u2194RHR"),
        ("hrv", "rhr", -1, "HRV\u2194RHR"),
        ("temp", "spo2", -1, "temp\u2194SpO2"),
    ]

    correlations = {}
    decoupling_events = []
    valid_r_values = []

    for metric_a, metric_b, expected_sign, label in pairs:
        x_vals = []
        y_vals = []
        for d in window_days:
            dd = day_data.get(d, {})
            if metric_a in dd and metric_b in dd:
                x_vals.append(dd[metric_a])
                y_vals.append(dd[metric_b])

        if len(x_vals) < 5:
            continue

        r = _pearson_r(x_vals, y_vals)
        sign_matches = (r * expected_sign) > 0
        strong = abs(r) > 0.2
        coupled = sign_matches and strong

        correlations[label] = {
            "r": round(r, 3),
            "n": len(x_vals),
            "expected_sign": "negative" if expected_sign < 0 else "positive",
            "coupled": coupled,
        }
        valid_r_values.append(abs(r) if coupled else 0)

        if not coupled and strong:
            decoupling_events.append({
                "pair": label,
                "r": round(r, 3),
                "description": f"{label}: r={r:.2f} (expected {'negative' if expected_sign < 0 else 'positive'}, got {'positive' if r > 0 else 'negative'})",
            })

    # Coupling score: average |r| where coupling is expected, scaled to 100
    coupling_score = round(mean(valid_r_values) * 100) if valid_r_values else None

    return {
        "correlations": correlations,
        "coupling_score": coupling_score,
        "decoupling_events": decoupling_events,
        "latest_window": {
            "start": window_days[0],
            "end": window_days[-1],
        } if window_days else None,
    }


def compute_circadian_fingerprint(heartrate: list) -> Optional[dict]:
    """Fit cosinor model to 5-min HR data to extract circadian rhythm parameters.

    Extracts: mesor (24h mean HR), amplitude (daily HR swing), acrophase
    (time of peak HR), and R-squared (rhythm strength). Compares weekday
    vs weekend patterns to quantify social jetlag.

    Args:
        heartrate: HeartRateRecord dicts with 'timestamp', 'bpm', 'source'

    Returns:
        dict with mesor, amplitude, acrophase_hour, rhythm_strength,
        social_jetlag_hours, weekday/weekend splits, or None
    """
    if not heartrate:
        return None

    # Parse timestamps and extract hour-of-day + day-of-week
    all_hours = []
    all_bpm = []
    weekday_hours = []
    weekday_bpm = []
    weekend_hours = []
    weekend_bpm = []

    for hr in heartrate:
        if not hr.get('bpm') or not hr.get('timestamp'):
            continue
        dt = _parse_iso_dt(hr['timestamp'])
        if dt is None:
            continue

        fractional_hour = dt.hour + dt.minute / 60.0
        bpm = hr['bpm']

        all_hours.append(fractional_hour)
        all_bpm.append(bpm)

        if dt.weekday() < 5:
            weekday_hours.append(fractional_hour)
            weekday_bpm.append(bpm)
        else:
            weekend_hours.append(fractional_hour)
            weekend_bpm.append(bpm)

    # Overall fit
    overall = _cosinor_fit(all_hours, all_bpm)
    if overall is None:
        return None

    # Rhythm strength classification
    r2 = overall["r_squared"]
    if r2 > 0.3:
        rhythm_strength = "strong"
    elif r2 > 0.15:
        rhythm_strength = "moderate"
    else:
        rhythm_strength = "weak"

    # Weekday vs weekend fits
    weekday_fit = _cosinor_fit(weekday_hours, weekday_bpm)
    weekend_fit = _cosinor_fit(weekend_hours, weekend_bpm)

    # Social jetlag: acrophase difference
    social_jetlag = None
    if weekday_fit and weekend_fit:
        diff = abs(weekday_fit["acrophase_hour"] - weekend_fit["acrophase_hour"])
        if diff > 12:
            diff = 24 - diff
        social_jetlag = round(diff, 1)

    return {
        "mesor": round(overall["mesor"], 1),
        "amplitude": round(overall["amplitude"], 1),
        "acrophase_hour": round(overall["acrophase_hour"], 1),
        "goodness_of_fit": round(overall["r_squared"], 3),
        "rhythm_strength": rhythm_strength,
        "weekday": {
            "mesor": round(weekday_fit["mesor"], 1),
            "amplitude": round(weekday_fit["amplitude"], 1),
            "acrophase_hour": round(weekday_fit["acrophase_hour"], 1),
            "r_squared": round(weekday_fit["r_squared"], 3),
        } if weekday_fit else None,
        "weekend": {
            "mesor": round(weekend_fit["mesor"], 1),
            "amplitude": round(weekend_fit["amplitude"], 1),
            "acrophase_hour": round(weekend_fit["acrophase_hour"], 1),
            "r_squared": round(weekend_fit["r_squared"], 3),
        } if weekend_fit else None,
        "social_jetlag_hours": social_jetlag,
        "n_samples": len(all_hours),
    }


def compute_heart_rate_recovery(
    workouts: list,
    heartrate: list,
) -> dict:
    """Compute slow-phase heart rate recovery for each workout.

    Matches workout end times to nearest HR readings to compute how fast
    HR drops 5 and 10 minutes post-exercise. Limited to slow-phase recovery
    due to 5-min HR resolution.

    Args:
        workouts: WorkoutRecord dicts with 'end_time', 'day', 'activity'
        heartrate: HeartRateRecord dicts with 'timestamp', 'bpm'

    Returns:
        dict with per-workout HRR, averages, trend, and fitness indicator
    """
    if not workouts or not heartrate:
        return {
            "workouts": [], "avg_hrr5": None, "avg_hrr10": None,
            "trend": "unknown", "fitness_indicator": "insufficient_data",
        }

    # Build HR index: {rounded_5min_datetime: bpm}
    hr_index = {}
    for hr in heartrate:
        if not hr.get('bpm') or not hr.get('timestamp'):
            continue
        dt = _parse_iso_dt(hr['timestamp'])
        if dt:
            # Remove timezone info for comparison
            dt = dt.replace(tzinfo=None)
            hr_index[dt] = hr['bpm']

    if not hr_index:
        return {
            "workouts": [], "avg_hrr5": None, "avg_hrr10": None,
            "trend": "unknown", "fitness_indicator": "insufficient_data",
        }

    hr_times = sorted(hr_index.keys())

    def find_nearest_hr(target_dt, max_delta_min=5):
        """Find the HR reading closest to target_dt within max_delta_min."""
        best = None
        best_delta = timedelta(minutes=max_delta_min + 1)
        for t in hr_times:
            delta = abs(t - target_dt)
            if delta < best_delta:
                best_delta = delta
                best = t
            elif t > target_dt + timedelta(minutes=max_delta_min):
                break
        if best and best_delta <= timedelta(minutes=max_delta_min):
            return hr_index[best]
        return None

    results = []
    for w in workouts:
        if not w.get('end_time'):
            continue
        end_dt = _parse_iso_dt(w['end_time'])
        if not end_dt:
            continue
        end_dt = end_dt.replace(tzinfo=None)

        # Find HR at end, +5min, +10min
        hr_end = find_nearest_hr(end_dt, max_delta_min=3)
        hr_5 = find_nearest_hr(end_dt + timedelta(minutes=5), max_delta_min=3)
        hr_10 = find_nearest_hr(end_dt + timedelta(minutes=10), max_delta_min=3)

        if hr_end is None:
            continue

        hrr5 = (hr_end - hr_5) if hr_5 is not None else None
        hrr10 = (hr_end - hr_10) if hr_10 is not None else None

        results.append({
            "day": w.get('day', ''),
            "activity": w.get('activity', 'unknown'),
            "hr_end": hr_end,
            "hr_5min": hr_5,
            "hr_10min": hr_10,
            "hrr5": hrr5,
            "hrr10": hrr10,
        })

    if not results:
        return {
            "workouts": [], "avg_hrr5": None, "avg_hrr10": None,
            "trend": "unknown", "fitness_indicator": "insufficient_data",
        }

    # Averages
    hrr5_vals = [r["hrr5"] for r in results if r["hrr5"] is not None]
    hrr10_vals = [r["hrr10"] for r in results if r["hrr10"] is not None]
    avg_hrr5 = round(mean(hrr5_vals), 1) if hrr5_vals else None
    avg_hrr10 = round(mean(hrr10_vals), 1) if hrr10_vals else None

    # Fitness indicator (slow-phase norms, lower than standard 1-min fast-phase)
    if avg_hrr5 is None:
        fitness = "insufficient_data"
    elif avg_hrr5 > 15:
        fitness = "excellent"
    elif avg_hrr5 > 10:
        fitness = "good"
    elif avg_hrr5 > 5:
        fitness = "fair"
    else:
        fitness = "poor"

    # Trend: compare first half vs second half chronologically
    if len(hrr5_vals) >= 4:
        mid = len(hrr5_vals) // 2
        first_avg = mean(hrr5_vals[:mid])
        second_avg = mean(hrr5_vals[mid:])
        diff = second_avg - first_avg
        if diff > 3:
            trend = "improving"
        elif diff < -3:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    return {
        "workouts": results,
        "avg_hrr5": avg_hrr5,
        "avg_hrr10": avg_hrr10,
        "trend": trend,
        "fitness_indicator": fitness,
    }








# ---------------------------------------------------------------------------
# D. Alcohol Night Detector
# ---------------------------------------------------------------------------

def compute_alcohol_detection(sleep: list, baseline_days: int = 14) -> dict:
    """Detect probable alcohol nights from HRV/RHR/efficiency z-scores."""
    records = []
    for s in sleep:
        if (s.get('day') and s.get('avg_hrv_ms') is not None
                and s.get('avg_resting_hr_bpm') is not None
                and s.get('efficiency') is not None):
            records.append(s)
    records.sort(key=lambda x: x['day'])

    if len(records) < baseline_days + 1:
        return {"flagged_nights": [], "per_night": [], "frequency": None}

    per_night = []
    flagged = []
    for i in range(baseline_days, len(records)):
        window = records[max(0, i - baseline_days):i]
        hrvs = [w['avg_hrv_ms'] for w in window]
        rhrs = [w['avg_resting_hr_bpm'] for w in window]
        effs = [w['efficiency'] for w in window]

        if len(hrvs) < 3:
            continue

        hrv_mean, hrv_sd = mean(hrvs), stdev(hrvs) if len(hrvs) > 1 else 1
        rhr_mean, rhr_sd = mean(rhrs), stdev(rhrs) if len(rhrs) > 1 else 1
        eff_mean, eff_sd = mean(effs), stdev(effs) if len(effs) > 1 else 1

        cur = records[i]
        z_hrv = (cur['avg_hrv_ms'] - hrv_mean) / hrv_sd if hrv_sd > 0 else 0
        z_rhr = (cur['avg_resting_hr_bpm'] - rhr_mean) / rhr_sd if rhr_sd > 0 else 0
        z_eff = (cur['efficiency'] - eff_mean) / eff_sd if eff_sd > 0 else 0

        # Alcohol: HRV drops, RHR rises, efficiency drops
        alcohol_score = round((-z_hrv + z_rhr - z_eff) / 3, 2)
        probable = alcohol_score > 1.5

        entry = {"day": cur['day'], "score": alcohol_score, "probable_alcohol": probable}
        per_night.append(entry)
        if probable:
            flagged.append(cur['day'])

    freq = len(flagged) / len(per_night) if per_night else None

    return {
        "flagged_nights": flagged,
        "per_night": per_night[-30:],  # last 30 for brevity
        "frequency": round(freq, 3) if freq is not None else None,
    }


# ---------------------------------------------------------------------------
# E. CUSUM Change-Point Detection
# ---------------------------------------------------------------------------

def detect_change_points(daily_records: list, metric_key: str,
                         day_key: str = "day",
                         threshold: float = 5.0,
                         drift: float = 0.5) -> dict:
    """CUSUM algorithm for detecting regime shifts in a daily metric.

    Args:
        daily_records: List of dicts with day_key and metric_key fields.
        metric_key: Which numeric field to analyze.
        threshold: CUSUM threshold for triggering a change point.
        drift: Allowance for natural variation.
    """
    points = [(r[day_key], r[metric_key]) for r in daily_records
              if r.get(day_key) and r.get(metric_key) is not None]
    points.sort(key=lambda x: x[0])

    if len(points) < 7:
        return {"change_points": [], "current_regime_start": None}

    # Baseline from first 7 days
    target = mean([p[1] for p in points[:7]])
    s_pos, s_neg = 0.0, 0.0
    change_points = []
    regime_start = points[0][0]

    for day, val in points[7:]:
        s_pos = max(0, s_pos + (val - target) - drift)
        s_neg = max(0, s_neg - (val - target) - drift)

        if s_pos > threshold:
            change_points.append({"day": day, "direction": "up",
                                  "magnitude": round(val - target, 2)})
            target = val
            s_pos, s_neg = 0.0, 0.0
            regime_start = day
        elif s_neg > threshold:
            change_points.append({"day": day, "direction": "down",
                                  "magnitude": round(val - target, 2)})
            target = val
            s_pos, s_neg = 0.0, 0.0
            regime_start = day

    return {
        "change_points": change_points,
        "current_regime_start": regime_start,
    }


# ---------------------------------------------------------------------------
# F. Allostatic Load Index
# ---------------------------------------------------------------------------

def compute_allostatic_load(sleep: list, readiness: list, spo2: list,
                            stress: list, baseline_days: int = 30) -> dict:
    """Multi-metric stress burden score (0-6).

    Counts how many metrics deviate >1 SD from personal baseline
    in the unfavorable direction.
    """
    # Build day-aligned metrics
    by_day = {}
    for s in sleep:
        d = s.get('day')
        if not d:
            continue
        by_day.setdefault(d, {})
        if s.get('avg_hrv_ms') is not None:
            by_day[d]['hrv'] = s['avg_hrv_ms']
        if s.get('avg_resting_hr_bpm') is not None:
            by_day[d]['rhr'] = s['avg_resting_hr_bpm']
        if s.get('efficiency') is not None:
            by_day[d]['eff'] = s['efficiency']
    for r in readiness:
        d = r.get('day')
        if d and r.get('temp_deviation_c') is not None:
            by_day.setdefault(d, {})['temp'] = r['temp_deviation_c']
    for s in spo2:
        d = s.get('day')
        if d and s.get('avg_spo2_pct') is not None:
            by_day.setdefault(d, {})['spo2'] = s['avg_spo2_pct']
    for s in stress:
        d = s.get('day')
        if not d:
            continue
        sh = s.get('stress_high_minutes') or 0
        rh = s.get('recovery_high_minutes') or 0
        ratio = sh / rh if rh > 0 else sh
        by_day.setdefault(d, {})['stress_ratio'] = ratio

    sorted_days = sorted(by_day.keys())
    if len(sorted_days) < baseline_days + 7:
        return {"load_score": None, "classification": "insufficient_data",
                "per_metric": {}, "trend": "unknown"}

    # Baselines
    bl = {}
    # unfavorable direction: hrv=low, rhr=high, temp=high, eff=low, spo2=low, stress_ratio=high
    directions = {'hrv': -1, 'rhr': 1, 'temp': 1, 'eff': -1, 'spo2': -1, 'stress_ratio': 1}
    for k in directions:
        vals = [by_day[d][k] for d in sorted_days[:baseline_days] if k in by_day[d]]
        if len(vals) >= 7:
            bl[k] = (mean(vals), stdev(vals) if len(vals) > 1 else 0)

    # Current 7d means
    recent_7d = sorted_days[-7:]
    current = {}
    for k in bl:
        vals = [by_day[d][k] for d in recent_7d if k in by_day[d]]
        if vals:
            current[k] = mean(vals)

    # Prior 7d (for trend)
    prior_7d = sorted_days[-14:-7]
    prior = {}
    for k in bl:
        vals = [by_day[d][k] for d in prior_7d if k in by_day[d]]
        if vals:
            prior[k] = mean(vals)

    # Compute z-scores and count
    per_metric = {}
    load = 0
    for k in bl:
        if k not in current or bl[k][1] == 0:
            continue
        z = (current[k] - bl[k][0]) / bl[k][1]
        unfavorable = (directions[k] * z) > 1.0
        per_metric[k] = {"z_score": round(z, 2), "unfavorable": unfavorable}
        if unfavorable:
            load += 1

    # Trend
    prev_load = 0
    for k in bl:
        if k not in prior or bl[k][1] == 0:
            continue
        z = (prior[k] - bl[k][0]) / bl[k][1]
        if (directions[k] * z) > 1.0:
            prev_load += 1
    trend = "increasing" if load > prev_load else "decreasing" if load < prev_load else "stable"

    classification = "low" if load <= 1 else "moderate" if load <= 3 else "high"

    return {
        "load_score": load,
        "classification": classification,
        "per_metric": per_metric,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# H. Training Load (TRIMP + ACWR)
# ---------------------------------------------------------------------------

def compute_training_load(workouts: list, heartrate: list, sleep: list) -> dict:
    """Training Impulse and Acute:Chronic Workload Ratio.

    TRIMP = duration_min * HR_reserve_fraction.
    ACWR = 7d_load / (28d_load / 4).
    """
    # Get resting HR from sleep (lowest avg)
    rhr_vals = [s['avg_resting_hr_bpm'] for s in sleep
                if s.get('avg_resting_hr_bpm') is not None]
    if not rhr_vals:
        return {"acwr": None, "acwr_zone": "insufficient_data",
                "weekly_trimp": None, "per_workout": []}
    resting_hr = min(mean(rhr_vals[:7]) if len(rhr_vals) >= 7 else mean(rhr_vals),
                     min(rhr_vals))

    # Get max HR from workouts/heartrate
    max_hr_candidates = [hr['bpm'] for hr in heartrate
                         if hr.get('bpm') and hr.get('source') == 'workout']
    max_hr = max(max_hr_candidates) if max_hr_candidates else 190  # fallback

    # Build HR index by timestamp for workout matching
    hr_index = {}
    for hr in heartrate:
        if hr.get('timestamp') and hr.get('bpm'):
            hr_index[hr['timestamp'][:16]] = hr['bpm']  # minute resolution

    per_workout = []
    for w in workouts:
        dur = w.get('duration_seconds')
        if not dur or dur < 300:  # skip < 5min
            continue

        # Try to get avg HR during workout from HR data
        avg_hr = None
        if w.get('start_time') and w.get('end_time'):
            try:
                start = _parse_iso_dt(w['start_time'])
                end = _parse_iso_dt(w['end_time'])
                workout_hrs = []
                for ts, bpm in hr_index.items():
                    try:
                        t = datetime.strptime(ts, "%Y-%m-%dT%H:%M")
                        if start <= t <= end:
                            workout_hrs.append(bpm)
                    except ValueError:
                        continue
                if workout_hrs:
                    avg_hr = mean(workout_hrs)
            except (ValueError, TypeError):
                pass

        if avg_hr is None:
            continue  # can't compute TRIMP without HR

        dur_min = dur / 60
        hr_reserve = (avg_hr - resting_hr) / (max_hr - resting_hr) if max_hr > resting_hr else 0
        hr_reserve = max(0, min(1, hr_reserve))
        trimp = round(dur_min * hr_reserve, 1)

        per_workout.append({
            "day": w.get('day', ''),
            "activity": w.get('activity', 'unknown'),
            "trimp": trimp,
            "avg_hr": round(avg_hr, 1),
            "duration_min": round(dur_min, 1),
        })

    if not per_workout:
        return {"acwr": None, "acwr_zone": "insufficient_data",
                "weekly_trimp": None, "per_workout": []}

    # Compute ACWR
    per_workout.sort(key=lambda x: x['day'])
    today = datetime.strptime(per_workout[-1]['day'], "%Y-%m-%d")
    d7 = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    d28 = (today - timedelta(days=28)).strftime("%Y-%m-%d")

    acute = sum(w['trimp'] for w in per_workout if w['day'] >= d7)
    chronic_pool = [w for w in per_workout if w['day'] >= d28]
    chronic_avg = sum(w['trimp'] for w in chronic_pool) / 4 if chronic_pool else 0

    acwr = round(acute / chronic_avg, 2) if chronic_avg > 0 else None
    zone = "insufficient_data"
    if acwr is not None:
        if acwr < 0.8:
            zone = "undertraining"
        elif acwr <= 1.3:
            zone = "sweet_spot"
        elif acwr <= 1.5:
            zone = "caution"
        else:
            zone = "danger"

    return {
        "acwr": acwr,
        "acwr_zone": zone,
        "weekly_trimp": round(acute, 1),
        "per_workout": per_workout,
    }


# ---------------------------------------------------------------------------
# I. Nocturnal HR Curve Shape
# ---------------------------------------------------------------------------

def compute_nocturnal_hr_shape(heartrate: list, sleep: list) -> dict:
    """Analyze shape of the nocturnal HR curve.

    Extracts nadir timing, dipping ratio, and morning slope.
    """
    # Get nighttime and daytime HR
    night_hrs = []
    day_hrs = []
    for hr in heartrate:
        if not hr.get('bpm') or not hr.get('timestamp'):
            continue
        src = hr.get('source', '')
        if src in ('sleep', 'rest'):
            try:
                t = _parse_iso_dt(hr['timestamp'])
                night_hrs.append((t, hr['bpm']))
            except (ValueError, TypeError):
                continue
        elif src in ('awake',):
            day_hrs.append(hr['bpm'])

    if len(night_hrs) < 10 or not day_hrs:
        return {"nadir_hour": None, "nadir_bpm": None, "dipping_pct": None,
                "morning_slope": None, "classification": "insufficient_data"}

    night_hrs.sort(key=lambda x: x[0])

    # Find nadir
    nadir_bpm = min(h[1] for h in night_hrs)
    nadir_entry = next(h for h in night_hrs if h[1] == nadir_bpm)
    nadir_hour = nadir_entry[0].hour + nadir_entry[0].minute / 60

    # Dipping ratio
    day_mean = mean(day_hrs)
    night_mean = mean([h[1] for h in night_hrs])
    dipping_pct = round((day_mean - night_mean) / day_mean * 100, 1) if day_mean > 0 else 0

    # Morning slope (last 6 readings ~30min)
    morning_slope = None
    if len(night_hrs) >= 6:
        last_6 = night_hrs[-6:]
        # Simple slope: (last - first) / time_diff_hours
        t_diff = (last_6[-1][0] - last_6[0][0]).total_seconds() / 3600
        if t_diff > 0:
            morning_slope = round((last_6[-1][1] - last_6[0][1]) / t_diff, 1)

    classification = "dipper" if dipping_pct >= 10 else "non-dipper"

    return {
        "nadir_hour": round(nadir_hour, 1),
        "nadir_bpm": nadir_bpm,
        "dipping_pct": dipping_pct,
        "morning_slope": morning_slope,
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# J. Critical Slowing Down (Early Warning Signals)
# ---------------------------------------------------------------------------

def compute_early_warning_signals(sleep: list, window: int = 14) -> dict:
    """Detect rising autocorrelation + variance as pre-transition warning.

    When both increase simultaneously over 2+ weeks, a regime shift
    may be approaching (critical slowing down phenomenon).
    """
    # Extract daily HRV and RHR series
    records = [(s['day'], s.get('avg_hrv_ms'), s.get('avg_resting_hr_bpm'))
               for s in sleep if s.get('day')]
    records.sort(key=lambda x: x[0])
    records = [(d, h, r) for d, h, r in records if h is not None and r is not None]

    if len(records) < window * 2:
        return {"warning_level": "insufficient_data",
                "hrv_autocorr_trend": None, "hrv_variance_trend": None,
                "rhr_autocorr_trend": None, "rhr_variance_trend": None}

    def _rolling_stats(values, win):
        autocorrs = []
        variances = []
        for i in range(win, len(values)):
            chunk = values[i - win:i]
            if len(chunk) < win:
                continue
            variances.append(stdev(chunk) ** 2 if len(chunk) > 1 else 0)
            if len(chunk) >= 3:
                autocorrs.append(_pearson_r(chunk[:-1], chunk[1:]))
            else:
                autocorrs.append(0)
        return autocorrs, variances

    hrvs = [r[1] for r in records]
    rhrs = [r[2] for r in records]

    hrv_ac, hrv_var = _rolling_stats(hrvs, window)
    rhr_ac, rhr_var = _rolling_stats(rhrs, window)

    def _trend(series):
        if len(series) < 4:
            return "unknown"
        first_half = mean(series[:len(series) // 2])
        second_half = mean(series[len(series) // 2:])
        diff = second_half - first_half
        if diff > 0.05:
            return "rising"
        elif diff < -0.05:
            return "falling"
        return "stable"

    hrv_ac_trend = _trend(hrv_ac)
    hrv_var_trend = _trend(hrv_var)
    rhr_ac_trend = _trend(rhr_ac)
    rhr_var_trend = _trend(rhr_var)

    # Warning if both autocorrelation AND variance rising for either metric
    warning = "none"
    if (hrv_ac_trend == "rising" and hrv_var_trend == "rising"):
        warning = "approaching_transition"
    if (rhr_ac_trend == "rising" and rhr_var_trend == "rising"):
        warning = "approaching_transition"

    return {
        "warning_level": warning,
        "hrv_autocorr_trend": hrv_ac_trend,
        "hrv_variance_trend": hrv_var_trend,
        "rhr_autocorr_trend": rhr_ac_trend,
        "rhr_variance_trend": rhr_var_trend,
    }


# ---------------------------------------------------------------------------
# M. Sample Entropy
# ---------------------------------------------------------------------------

def _sample_entropy(values: list, m: int = 2, r_factor: float = 0.2) -> Optional[float]:
    """Compute sample entropy of a time series.

    Args:
        values: Numeric time series (needs 10+ points).
        m: Template length.
        r_factor: Tolerance as fraction of stdev.
    """
    n = len(values)
    if n < 10:
        return None
    sd = stdev(values) if n > 1 else 0
    if sd == 0:
        return 0.0
    r = r_factor * sd

    def _count_matches(template_len):
        count = 0
        templates = [values[i:i + template_len] for i in range(n - template_len)]
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if all(abs(templates[i][k] - templates[j][k]) <= r
                       for k in range(template_len)):
                    count += 1
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)

    if b == 0:
        return None
    return round(-math.log(a / b) if a > 0 else math.log(n), 3)


def compute_daily_entropy(sleep: list, readiness: list,
                          window: int = 60) -> dict:
    """Sample entropy of daily HRV, RHR, and temperature series.

    Uses r_factor=0.3 for daily biometric data (wider tolerance than
    the standard 0.2 to account for day-to-day noise in short series).
    """
    # Build sorted daily series
    by_day = {}
    for s in sleep:
        d = s.get('day')
        if not d:
            continue
        by_day.setdefault(d, {})
        if s.get('avg_hrv_ms') is not None:
            by_day[d]['hrv'] = s['avg_hrv_ms']
        if s.get('avg_resting_hr_bpm') is not None:
            by_day[d]['rhr'] = s['avg_resting_hr_bpm']
    for r in readiness:
        d = r.get('day')
        if d and r.get('temp_deviation_c') is not None:
            by_day.setdefault(d, {})['temp'] = r['temp_deviation_c']

    sorted_days = sorted(by_day.keys())[-window:]

    hrvs = [by_day[d]['hrv'] for d in sorted_days if 'hrv' in by_day[d]]
    rhrs = [by_day[d]['rhr'] for d in sorted_days if 'rhr' in by_day[d]]
    temps = [by_day[d]['temp'] for d in sorted_days if 'temp' in by_day[d]]

    # Use r_factor=0.3 for daily data (more template matches with short series)
    hrv_ent = _sample_entropy(hrvs, r_factor=0.3)
    rhr_ent = _sample_entropy(rhrs, r_factor=0.3)
    temp_ent = _sample_entropy(temps, r_factor=0.3)

    # Thresholds calibrated for daily biometric data with r=0.3*SD
    def _classify(e):
        if e is None:
            return "insufficient_data"
        if e < 0.5:
            return "rigid"
        if e < 2.0:
            return "healthy"
        return "chaotic"

    return {
        "hrv_entropy": hrv_ent,
        "rhr_entropy": rhr_ent,
        "temp_entropy": temp_ent,
        "hrv_interpretation": _classify(hrv_ent),
        "rhr_interpretation": _classify(rhr_ent),
        "temp_interpretation": _classify(temp_ent),
    }


# ---------------------------------------------------------------------------
# O. Temperature Amplitude Trend
# ---------------------------------------------------------------------------

def compute_temp_amplitude_trend(readiness: list, window: int = 30) -> dict:
    """Track whether nightly temperature amplitude is dampening over time."""
    temps = [(r['day'], r['temp_deviation_c']) for r in readiness
             if r.get('day') and r.get('temp_deviation_c') is not None]
    temps.sort(key=lambda x: x[0])

    if len(temps) < window:
        return {"current_amplitude": None, "amplitude_trend": "insufficient_data",
                "slope": None}

    # Rolling SD as amplitude proxy
    amplitudes = []
    for i in range(window, len(temps) + 1):
        chunk = [t[1] for t in temps[i - window:i]]
        if len(chunk) > 1:
            amplitudes.append(stdev(chunk))

    if len(amplitudes) < 2:
        return {"current_amplitude": round(amplitudes[0], 3) if amplitudes else None,
                "amplitude_trend": "insufficient_data", "slope": None}

    current = amplitudes[-1]

    # Slope via simple linear regression
    n = len(amplitudes)
    x_mean = (n - 1) / 2
    y_mean = mean(amplitudes)
    num = sum((i - x_mean) * (amplitudes[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0

    trend = "dampening" if slope < -0.001 else "increasing" if slope > 0.001 else "stable"

    return {
        "current_amplitude": round(current, 3),
        "amplitude_trend": trend,
        "slope": round(slope, 5),
    }


# ---------------------------------------------------------------------------
# A. Sleep Regularity Index (SRI)
# ---------------------------------------------------------------------------

def compute_sleep_regularity(sleep: list) -> dict:
    """Sleep Regularity Index — probability of same sleep/wake state at t and t+24h.

    Uses bedtime_start/bedtime_end to construct sleep windows.
    SRI of 100 = perfectly regular, 0 = random.
    """
    # Build sleep windows per day
    windows = []
    for s in sleep:
        if not s.get('bedtime_start') or not s.get('bedtime_end'):
            continue
        if s.get('sleep_type') not in ('long_sleep', None):
            continue
        try:
            start = _parse_iso_dt(s['bedtime_start'])
            end = _parse_iso_dt(s['bedtime_end'])
            windows.append((start, end))
        except (ValueError, TypeError):
            continue

    windows.sort(key=lambda x: x[0])
    if len(windows) < 7:
        return {"sri_score": None, "classification": "insufficient_data", "trend": "unknown"}

    # For each pair of consecutive sleep windows, check minute-by-minute overlap
    # Simplified: compare in 15-min bins across 24h
    total_matches = 0
    total_bins = 0

    for i in range(1, len(windows)):
        prev_start, prev_end = windows[i - 1]
        curr_start, curr_end = windows[i]

        # Check that these are ~24h apart
        gap = (curr_start - prev_start).total_seconds() / 3600
        if gap < 18 or gap > 30:
            continue

        # Compare 15-min bins over the shared 24h period
        for offset_min in range(0, 1440, 15):
            prev_time = prev_start.replace(hour=0, minute=0, second=0) + timedelta(minutes=offset_min)
            curr_time = prev_time + timedelta(hours=24)

            prev_asleep = prev_start <= prev_time <= prev_end
            curr_asleep = curr_start <= curr_time <= curr_end

            if prev_asleep == curr_asleep:
                total_matches += 1
            total_bins += 1

    if total_bins == 0:
        return {"sri_score": None, "classification": "insufficient_data", "trend": "unknown"}

    sri = round(total_matches / total_bins * 100, 1)

    # Classification
    if sri >= 85:
        classification = "regular"
    elif sri >= 70:
        classification = "moderate"
    else:
        classification = "irregular"

    # Trend: first half vs second half
    mid = len(windows) // 2
    # Simplified trend from consistency of sleep onset times
    first_onsets = [(w[0].hour * 60 + w[0].minute) for w in windows[:mid]]
    second_onsets = [(w[0].hour * 60 + w[0].minute) for w in windows[mid:]]
    first_sd = stdev(first_onsets) if len(first_onsets) > 1 else 999
    second_sd = stdev(second_onsets) if len(second_onsets) > 1 else 999
    trend = "improving" if second_sd < first_sd * 0.8 else "declining" if second_sd > first_sd * 1.2 else "stable"

    return {
        "sri_score": sri,
        "classification": classification,
        "trend": trend,
    }




# ---------------------------------------------------------------------------
# K. Sleep Stage Transition Matrix
# ---------------------------------------------------------------------------

def compute_sleep_transitions(sleep: list) -> dict:
    """Build a Markov transition matrix from hypnogram data.

    Stage encoding: 1=deep, 2=light, 3=REM, 4=awake.
    """
    stage_names = {'1': 'deep', '2': 'light', '3': 'REM', '4': 'awake'}
    transitions = {}
    for a in stage_names:
        for b in stage_names:
            transitions[f"{stage_names[a]}→{stage_names[b]}"] = 0

    total_transitions = 0
    awakenings = 0
    stage_shifts = 0
    total_sleep_hours = 0
    cycle_counts = []

    for s in sleep:
        hypno = s.get('hypnogram_5min')
        if not hypno or s.get('sleep_type') not in ('long_sleep', None):
            continue

        total_sleep_hours += len(hypno) * 5 / 60

        # Count transitions
        cycles_this_night = 0
        in_cycle = False
        for i in range(1, len(hypno)):
            prev, curr = hypno[i - 1], hypno[i]
            if prev not in stage_names or curr not in stage_names:
                continue
            key = f"{stage_names[prev]}→{stage_names[curr]}"
            transitions[key] = transitions.get(key, 0) + 1
            total_transitions += 1

            if prev != curr:
                stage_shifts += 1
                if curr == '4' and prev != '4':
                    awakenings += 1
                # Count REM→light transitions as cycle boundaries
                if prev == '3' and curr == '2':
                    cycles_this_night += 1
                    in_cycle = True

        if cycles_this_night > 0:
            cycle_counts.append(cycles_this_night + 1)  # N transitions = N+1 cycles

    if total_transitions == 0:
        return {"transition_matrix": {}, "fragmentation_index": None,
                "avg_cycle_count": None, "avg_cycle_duration_min": None,
                "n_nights": 0, "awakenings_per_night": None}

    # Normalize to probabilities
    # Group by source state
    prob_matrix = {}
    for key, count in transitions.items():
        src = key.split("→")[0]
        src_total = sum(v for k, v in transitions.items() if k.startswith(src + "→"))
        prob = round(count / src_total, 3) if src_total > 0 else 0
        if prob > 0:
            prob_matrix[key] = prob

    n_nights = len([s for s in sleep if s.get('hypnogram_5min')
                    and s.get('sleep_type') in ('long_sleep', None)])
    frag = round((awakenings + stage_shifts) / total_sleep_hours, 1) if total_sleep_hours > 0 else None
    avg_cycles = round(mean(cycle_counts), 1) if cycle_counts else None
    avg_cycle_dur = round(total_sleep_hours * 60 / sum(cycle_counts), 0) if cycle_counts and sum(cycle_counts) > 0 else None

    return {
        "transition_matrix": prob_matrix,
        "fragmentation_index": frag,
        "avg_cycle_count": avg_cycles,
        "avg_cycle_duration_min": avg_cycle_dur,
        "n_nights": n_nights,
        "awakenings_per_night": round(awakenings / n_nights, 1) if n_nights > 0 else None,
    }


# ---------------------------------------------------------------------------
# L. Deep Sleep Front-Loading Ratio
# ---------------------------------------------------------------------------

def compute_deep_sleep_distribution(sleep: list) -> dict:
    """Measure how much deep sleep occurs in the first vs second half of the night.

    Healthy ratio >0.60 (front-loaded). Disrupted <0.50.
    """
    ratios = []
    first_deep_epochs = []

    for s in sleep:
        hypno = s.get('hypnogram_5min')
        if not hypno or s.get('sleep_type') not in ('long_sleep', None):
            continue

        mid = len(hypno) // 2
        first_half_deep = sum(1 for ch in hypno[:mid] if ch == '1')
        second_half_deep = sum(1 for ch in hypno[mid:] if ch == '1')
        total_deep = first_half_deep + second_half_deep

        if total_deep > 0:
            ratios.append(first_half_deep / total_deep)

        # Time to first deep epoch
        for i, ch in enumerate(hypno):
            if ch == '1':
                first_deep_epochs.append(i * 5)  # minutes
                break

    if not ratios:
        return {"front_loading_ratio": None, "avg_first_deep_min": None,
                "classification": "insufficient_data"}

    avg_ratio = round(mean(ratios), 3)
    avg_first = round(mean(first_deep_epochs), 0) if first_deep_epochs else None

    classification = "healthy" if avg_ratio >= 0.60 else "moderate" if avg_ratio >= 0.50 else "disrupted"

    return {
        "front_loading_ratio": avg_ratio,
        "avg_first_deep_min": avg_first,
        "classification": classification,
        "n_nights": len(ratios),
    }


# ---------------------------------------------------------------------------
# Q. MSFsc Chronotype
# ---------------------------------------------------------------------------

def compute_chronotype(sleep: list) -> dict:
    """Compute chronotype from mid-sleep on free days (sleep-corrected).

    MSFsc = MSF - 0.5 * (avg_sleep_free - avg_sleep_all).
    Social jetlag = |MSFsc - MSW|.
    """
    workday_mids = []
    free_mids = []
    workday_durs = []
    free_durs = []
    all_durs = []

    for s in sleep:
        if not s.get('bedtime_start') or not s.get('total_sleep_seconds'):
            continue
        if s.get('sleep_type') not in ('long_sleep', None):
            continue

        try:
            start = _parse_iso_dt(s['bedtime_start'])
            dur_h = s['total_sleep_seconds'] / 3600
            # Mid-sleep in hours from midnight
            mid = start.hour + start.minute / 60 + dur_h / 2
            if mid > 24:
                mid -= 24
            # Handle pre-midnight starts
            if start.hour >= 18:
                mid = (start.hour - 24) + start.minute / 60 + dur_h / 2

            day_dt = datetime.strptime(s['day'], "%Y-%m-%d")
            is_free = day_dt.weekday() >= 5  # Sat=5, Sun=6

            all_durs.append(dur_h)
            if is_free:
                free_mids.append(mid)
                free_durs.append(dur_h)
            else:
                workday_mids.append(mid)
                workday_durs.append(dur_h)
        except (ValueError, TypeError):
            continue

    if not free_mids or not workday_mids:
        return {"chronotype_hour": None, "social_jetlag_hours": None,
                "classification": "insufficient_data"}

    msw = mean(workday_mids)
    msf = mean(free_mids)
    avg_sleep_free = mean(free_durs)
    avg_sleep_all = mean(all_durs)

    # Sleep-corrected
    msf_sc = msf - 0.5 * (avg_sleep_free - avg_sleep_all)
    social_jetlag = round(abs(msf_sc - msw), 1)

    # Classify chronotype
    if msf_sc < 2.5:
        chrono = "early"
    elif msf_sc < 4.5:
        chrono = "intermediate"
    else:
        chrono = "late"

    return {
        "chronotype_hour": round(msf_sc, 1),
        "social_jetlag_hours": social_jetlag,
        "classification": chrono,
        "workday_mid_sleep": round(msw, 1),
        "free_mid_sleep": round(msf, 1),
    }


# ---------------------------------------------------------------------------
# G. Tag-Biometric Correlation Engine
# ---------------------------------------------------------------------------

def compute_tag_effects(tags: list, sleep: list,
                        min_occurrences: int = 3) -> dict:
    """Bayesian n-of-1: does a tagged event affect biometrics?

    For each tag type, compares tagged-night vs untagged-night metrics
    using conjugate Normal-Normal model.
    """
    # Build sleep lookup by day
    sleep_by_day = {}
    for s in sleep:
        if s.get('day') and s.get('sleep_type') in ('long_sleep', None):
            sleep_by_day[s['day']] = s

    # Group tag days by tag_type
    tag_days = {}
    for t in tags:
        tt = t.get('tag_type', '') or ''
        tt = tt.replace('tag_generic_', '').replace('_', ' ').strip()
        if not tt:
            continue
        tag_days.setdefault(tt, set()).add(t.get('day', ''))

    metrics_to_test = [
        ('avg_hrv_ms', 'HRV', True),    # higher is better
        ('avg_resting_hr_bpm', 'RHR', False),  # lower is better
        ('efficiency', 'Efficiency', True),
        ('deep_sleep_seconds', 'Deep Sleep', True),
        ('score', 'Sleep Score', True),
    ]

    results = {}
    for tag_type, days in tag_days.items():
        tagged_days = days & set(sleep_by_day.keys())
        if len(tagged_days) < min_occurrences:
            continue

        untagged_days = set(sleep_by_day.keys()) - days

        tag_result = {"n_tagged": len(tagged_days), "n_untagged": len(untagged_days), "metrics": {}}

        for field, label, higher_better in metrics_to_test:
            tagged_vals = [sleep_by_day[d][field] for d in tagged_days
                          if sleep_by_day[d].get(field) is not None]
            untagged_vals = [sleep_by_day[d][field] for d in untagged_days
                            if sleep_by_day[d].get(field) is not None]

            if len(tagged_vals) < min_occurrences or len(untagged_vals) < 3:
                continue

            t_mean = mean(tagged_vals)
            u_mean = mean(untagged_vals)
            t_var = stdev(tagged_vals) ** 2 if len(tagged_vals) > 1 else 1
            u_var = stdev(untagged_vals) ** 2 if len(untagged_vals) > 1 else 1

            diff = t_mean - u_mean

            # Cohen's d
            pooled_sd = ((t_var + u_var) / 2) ** 0.5
            cohens_d = round(diff / pooled_sd, 2) if pooled_sd > 0 else 0

            # Bayesian posterior probability
            # P(tagged > untagged) using normal approximation
            diff_var = t_var / len(tagged_vals) + u_var / len(untagged_vals)
            z = diff / (diff_var ** 0.5) if diff_var > 0 else 0
            prob = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)

            tag_result["metrics"][label] = {
                "mean_diff": round(diff, 1),
                "cohens_d": cohens_d,
                "prob_effect": prob if higher_better else 1 - prob,
                "direction": "better" if (diff > 0) == higher_better else "worse",
            }

        if tag_result["metrics"]:
            results[tag_type] = tag_result

    return {"tag_effects": results, "n_tag_types_analyzed": len(results)}


# ---------------------------------------------------------------------------
# P. Cycle-Phase Performance Matrix
# ---------------------------------------------------------------------------

def compute_phase_performance(sleep: list, workouts: list,
                              cycle_result: dict) -> dict:
    """Aggregate workout recovery metrics by menstrual cycle phase.

    Requires output from detect_cycle_phases() to map days to phases.
    """
    # We need detected_periods or phase assignments
    # Use a simplified approach: map each day to a phase based on cycle_result
    detected = cycle_result.get('detected_periods', [])
    cycle_len = cycle_result.get('estimated_cycle_length', 28) or 28

    if not detected:
        return {"phases": {}, "best_phase_for_intensity": None,
                "recommendation": "Need cycle data — log periods in Oura app"}

    # Map each day to a phase
    def _get_phase(day_str):
        day_dt = datetime.strptime(day_str, "%Y-%m-%d")
        for i, ps in enumerate(detected):
            ps_dt = datetime.strptime(ps, "%Y-%m-%d")
            next_ps_dt = (datetime.strptime(detected[i + 1], "%Y-%m-%d")
                         if i + 1 < len(detected)
                         else ps_dt + timedelta(days=cycle_len))
            if ps_dt <= day_dt < next_ps_dt:
                day_in_cycle = (day_dt - ps_dt).days
                if day_in_cycle < 5:
                    return "menstrual"
                elif day_in_cycle < 14:
                    return "follicular"
                else:
                    return "luteal"
        return None

    # Build sleep by day
    sleep_by_day = {}
    for s in sleep:
        if s.get('day') and s.get('sleep_type') in ('long_sleep', None):
            sleep_by_day[s['day']] = s

    # Aggregate by phase
    phases = {"menstrual": [], "follicular": [], "luteal": []}
    for s in sleep:
        d = s.get('day')
        if not d or s.get('sleep_type') not in ('long_sleep', None):
            continue
        phase = _get_phase(d)
        if phase:
            phases[phase].append(s)

    result = {}
    for phase, records in phases.items():
        if not records:
            continue
        hrvs = [r['avg_hrv_ms'] for r in records if r.get('avg_hrv_ms')]
        rhrs = [r['avg_resting_hr_bpm'] for r in records if r.get('avg_resting_hr_bpm')]
        scores = [r['score'] for r in records if r.get('score')]
        effs = [r['efficiency'] for r in records if r.get('efficiency')]

        result[phase] = {
            "avg_hrv": round(mean(hrvs), 1) if hrvs else None,
            "avg_rhr": round(mean(rhrs), 1) if rhrs else None,
            "avg_sleep_score": round(mean(scores), 0) if scores else None,
            "avg_efficiency": round(mean(effs), 0) if effs else None,
            "n_nights": len(records),
        }

    # Best phase for intensity (highest HRV)
    best = None
    best_hrv = 0
    for phase, data in result.items():
        if data.get('avg_hrv') and data['avg_hrv'] > best_hrv:
            best_hrv = data['avg_hrv']
            best = phase

    rec = None
    if best:
        rec = f"Your HRV is highest during {best} phase — best window for intense training"

    return {
        "phases": result,
        "best_phase_for_intensity": best,
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Q. Nutrition × Biometric Crossover Metrics
# ---------------------------------------------------------------------------

def _get_previous_evening_meals(meals: list, sleep_day: str) -> list:
    """Get meals from the evening before a given sleep day.

    'Evening' = meals after 17:00 on the day before, or with timestamp
    within 5 hours before bed. If no timestamps, fall back to all meals
    from the previous calendar day.
    """
    try:
        sleep_dt = datetime.strptime(sleep_day, "%Y-%m-%d")
        prev_day = (sleep_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return []

    evening = []
    fallback = []
    for m in meals:
        if m.get('day') == prev_day:
            fallback.append(m)
            ts = m.get('timestamp')
            if ts:
                mdt = _parse_iso_dt(ts)
                if mdt and mdt.hour >= 17:
                    evening.append(m)
    return evening if evening else fallback


def _classify_meal_profile(meals: list) -> Optional[str]:
    """Classify a set of meals as high_protein, high_carb, high_fat, or balanced."""
    total_cal = sum(m.get('calories') or 0 for m in meals)
    total_pro = sum(m.get('protein_g') or 0 for m in meals)
    total_carb = sum(m.get('carbs_g') or 0 for m in meals)
    total_fat = sum(m.get('fat_g') or 0 for m in meals)

    if total_cal == 0 and total_pro == 0 and total_carb == 0 and total_fat == 0:
        return None

    # Use calories from macros if no calorie field
    if total_cal == 0:
        total_cal = total_pro * 4 + total_carb * 4 + total_fat * 9

    if total_cal <= 0:
        return None

    pro_pct = (total_pro * 4 / total_cal) * 100 if total_cal > 0 else 0
    carb_pct = (total_carb * 4 / total_cal) * 100 if total_cal > 0 else 0
    fat_pct = (total_fat * 9 / total_cal) * 100 if total_cal > 0 else 0

    if pro_pct >= 35:
        return "high_protein"
    elif carb_pct >= 55:
        return "high_carb"
    elif fat_pct >= 45:
        return "high_fat"
    return "balanced"


def _daily_macro_totals(meals: list) -> dict:
    """Aggregate meals by day → {day: {calories, protein_g, carbs_g, fat_g, ...}}."""
    by_day = {}
    for m in meals:
        d = m.get('day')
        if not d:
            continue
        if d not in by_day:
            by_day[d] = {
                'calories': 0, 'protein_g': 0.0, 'carbs_g': 0.0,
                'fat_g': 0.0, 'fiber_g': 0.0, 'sugar_g': 0.0,
                'saturated_fat_g': 0.0, 'sodium_mg': 0.0,
                'iron_mg': 0.0, 'magnesium_mg': 0.0,
                'alcohol_units': 0.0, 'meal_count': 0,
            }
        entry = by_day[d]
        entry['calories'] += m.get('calories') or 0
        entry['protein_g'] += m.get('protein_g') or 0
        entry['carbs_g'] += m.get('carbs_g') or 0
        entry['fat_g'] += m.get('fat_g') or 0
        entry['fiber_g'] += m.get('fiber_g') or 0
        entry['sugar_g'] += m.get('sugar_g') or 0
        entry['saturated_fat_g'] += m.get('saturated_fat_g') or 0
        entry['sodium_mg'] += m.get('sodium_mg') or 0
        entry['iron_mg'] += m.get('iron_mg') or 0
        entry['magnesium_mg'] += m.get('magnesium_mg') or 0
        entry['alcohol_units'] += m.get('alcohol_units') or 0
        entry['meal_count'] += 1
    return by_day


def compute_meal_sleep_effects(meals: list, sleep: list,
                               min_occurrences: int = 3) -> dict:
    """Compare sleep quality by previous-evening meal macro profile.

    Groups nights by dinner profile (high_protein, high_carb, high_fat,
    balanced) and compares HRV, deep sleep, REM, efficiency, onset latency
    using Bayesian n-of-1 comparison.
    """
    if not meals or not sleep:
        return {"profiles": {}, "best_dinner_profile": None, "n_nights": 0}

    # Build sleep lookup (long_sleep only)
    sleep_by_day = {}
    for s in sleep:
        if s.get('day') and s.get('sleep_type') in ('long_sleep', None):
            sleep_by_day[s['day']] = s

    # Classify each night by previous evening's meal profile
    profile_nights = {}  # profile → [sleep_record]
    for day in sleep_by_day:
        evening = _get_previous_evening_meals(meals, day)
        if not evening:
            continue
        profile = _classify_meal_profile(evening)
        if profile:
            profile_nights.setdefault(profile, []).append(sleep_by_day[day])

    # Metrics to compare
    metrics = [
        ('avg_hrv_ms', 'HRV (ms)', True),
        ('avg_resting_hr_bpm', 'RHR (bpm)', False),
        ('deep_sleep_seconds', 'Deep Sleep (s)', True),
        ('rem_sleep_seconds', 'REM (s)', True),
        ('efficiency', 'Efficiency (%)', True),
        ('onset_latency_seconds', 'Onset Latency (s)', False),
    ]

    profiles = {}
    all_nights = [s for nights in profile_nights.values() for s in nights]

    for profile, nights in profile_nights.items():
        if len(nights) < min_occurrences:
            continue

        other_nights = [s for p, ns in profile_nights.items()
                        if p != profile for s in ns]
        if len(other_nights) < 3:
            continue

        profile_result = {"n_nights": len(nights), "metrics": {}}

        for field, label, higher_better in metrics:
            p_vals = [n[field] for n in nights if n.get(field) is not None]
            o_vals = [n[field] for n in other_nights if n.get(field) is not None]

            if len(p_vals) < min_occurrences or len(o_vals) < 3:
                continue

            p_mean = mean(p_vals)
            o_mean = mean(o_vals)
            p_var = stdev(p_vals) ** 2 if len(p_vals) > 1 else 1
            o_var = stdev(o_vals) ** 2 if len(o_vals) > 1 else 1
            diff = p_mean - o_mean

            pooled_sd = ((p_var + o_var) / 2) ** 0.5
            cohens_d = round(diff / pooled_sd, 2) if pooled_sd > 0 else 0

            diff_var = p_var / len(p_vals) + o_var / len(o_vals)
            z = diff / (diff_var ** 0.5) if diff_var > 0 else 0
            prob = round(0.5 * (1 + math.erf(z / math.sqrt(2))), 3)

            profile_result["metrics"][label] = {
                "mean": round(p_mean, 1),
                "vs_other_mean": round(o_mean, 1),
                "mean_diff": round(diff, 1),
                "cohens_d": cohens_d,
                "prob_effect": prob if higher_better else 1 - prob,
                "direction": "better" if (diff > 0) == higher_better else "worse",
            }

        if profile_result["metrics"]:
            profiles[profile] = profile_result

    # Best profile = highest average HRV
    best = None
    best_hrv = 0
    for p, data in profiles.items():
        hrv_metric = data['metrics'].get('HRV (ms)')
        if hrv_metric and hrv_metric['mean'] > best_hrv:
            best_hrv = hrv_metric['mean']
            best = p

    return {
        "profiles": profiles,
        "best_dinner_profile": best,
        "n_nights": sum(d['n_nights'] for d in profiles.values()),
    }


def compute_meal_circadian_alignment(meals: list, sleep: list) -> dict:
    """Score meal timing relative to sleep onset.

    Measures last-meal-to-bedtime gap and meal timing regularity.
    """
    if not meals or not sleep:
        return {"avg_gap_hours": None, "regularity_score": None,
                "late_meal_pct": None, "alignment_score": None,
                "n_days": 0}

    # Build bedtime lookup
    bed_by_day = {}
    for s in sleep:
        bt = s.get('bedtime_start')
        if bt and s.get('day') and s.get('sleep_type') in ('long_sleep', None):
            bed_dt = _parse_iso_dt(bt)
            if bed_dt:
                bed_by_day[s['day']] = bed_dt

    # Get last meal timestamp per day
    last_meal_by_day = {}
    for m in meals:
        d = m.get('day')
        ts = m.get('timestamp')
        if not d or not ts:
            continue
        mdt = _parse_iso_dt(ts)
        if not mdt:
            continue
        if d not in last_meal_by_day or mdt > last_meal_by_day[d]:
            last_meal_by_day[d] = mdt

    # Compute gap for each day
    gaps = []
    meal_hours = []  # hour-of-day for last meal
    for day, bed_dt in bed_by_day.items():
        prev_day = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        # Check both same-day (if bedtime is late) and prev-day
        meal_dt = last_meal_by_day.get(prev_day) or last_meal_by_day.get(day)
        if meal_dt:
            # Strip tzinfo for comparison (mixed tz-aware/naive)
            b = bed_dt.replace(tzinfo=None)
            m = meal_dt.replace(tzinfo=None)
            gap_h = (b - m).total_seconds() / 3600
            if -2 < gap_h < 12:  # sanity: ignore nonsensical gaps
                gaps.append(gap_h)
                meal_hours.append(meal_dt.hour + meal_dt.minute / 60)

    if not gaps:
        return {"avg_gap_hours": None, "regularity_score": None,
                "late_meal_pct": None, "alignment_score": None,
                "n_days": 0}

    avg_gap = round(mean(gaps), 1)

    # Late meals = < 2h before bed
    late_count = sum(1 for g in gaps if g < 2)
    late_pct = round(100 * late_count / len(gaps), 0)

    # Regularity = inverse of meal timing variability (0-100)
    if len(meal_hours) > 1:
        sd = stdev(meal_hours)
        regularity = round(max(0, 100 - sd * 20), 0)  # 5h SD = 0, 0 SD = 100
    else:
        regularity = None

    # Alignment score (0-100)
    # Optimal gap: 2.5-4h. Each 0.5h deviation = -5 points
    gap_score = max(0, 100 - abs(avg_gap - 3.0) * 15)
    late_penalty = late_pct * 0.3
    reg_bonus = (regularity / 100 * 15) if regularity else 0
    alignment = round(min(100, max(0, gap_score - late_penalty + reg_bonus)), 0)

    return {
        "avg_gap_hours": avg_gap,
        "regularity_score": regularity,
        "late_meal_pct": late_pct,
        "alignment_score": alignment,
        "n_days": len(gaps),
    }




def compute_thermic_effect(meals: list, readiness: list) -> dict:
    """Correlate meal macros with next-night temperature deviation."""
    if not meals or not readiness:
        return {"protein_temp_r": None, "carb_temp_r": None,
                "fat_temp_r": None, "late_meal_temp_impact": None,
                "optimal_last_meal_time": None, "n_days": 0}

    readiness_by_day = {r['day']: r for r in readiness
                        if r.get('day') and r.get('temp_deviation_c') is not None}

    daily_meals = _daily_macro_totals(meals)

    # Correlate each macro with next-day temp deviation
    protein_vals, carb_vals, fat_vals, temp_vals = [], [], [], []
    for day, totals in daily_meals.items():
        next_day = (datetime.strptime(day, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day in readiness_by_day:
            temp = readiness_by_day[next_day]['temp_deviation_c']
            if totals['protein_g'] > 0:
                protein_vals.append(totals['protein_g'])
                carb_vals.append(totals['carbs_g'])
                fat_vals.append(totals['fat_g'])
                temp_vals.append(temp)

    n = len(temp_vals)
    protein_r = round(_pearson_r(protein_vals, temp_vals), 2) if n >= 5 else None
    carb_r = round(_pearson_r(carb_vals, temp_vals), 2) if n >= 5 else None
    fat_r = round(_pearson_r(fat_vals, temp_vals), 2) if n >= 5 else None

    # Late meal impact: compare temp deviation on late-dinner vs early-dinner nights
    late_temps = []
    early_temps = []
    for m in meals:
        ts = m.get('timestamp')
        day = m.get('day')
        if not ts or not day:
            continue
        mdt = _parse_iso_dt(ts)
        if not mdt or mdt.hour < 17:
            continue
        next_day = (datetime.strptime(day, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day in readiness_by_day:
            temp = readiness_by_day[next_day]['temp_deviation_c']
            if mdt.hour >= 21:
                late_temps.append(temp)
            elif mdt.hour < 20:
                early_temps.append(temp)

    late_impact = None
    if len(late_temps) >= 3 and len(early_temps) >= 3:
        late_impact = round(mean(late_temps) - mean(early_temps), 3)

    # Optimal last meal time: find the meal hour that correlates with lowest
    # next-night temp deviation (least thermic interference with sleep)
    optimal_time = None
    meal_hour_temps = {}
    for m in meals:
        ts = m.get('timestamp')
        day = m.get('day')
        if not ts or not day:
            continue
        mdt = _parse_iso_dt(ts)
        if not mdt or mdt.hour < 17:
            continue
        next_day = (datetime.strptime(day, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day in readiness_by_day:
            h = mdt.hour
            meal_hour_temps.setdefault(h, []).append(
                abs(readiness_by_day[next_day]['temp_deviation_c']))

    if meal_hour_temps:
        avg_by_hour = {h: mean(ts) for h, ts in meal_hour_temps.items()
                       if len(ts) >= 2}
        if avg_by_hour:
            optimal_time = f"{min(avg_by_hour, key=avg_by_hour.get)}:00"

    return {
        "protein_temp_r": protein_r,
        "carb_temp_r": carb_r,
        "fat_temp_r": fat_r,
        "late_meal_temp_impact": late_impact,
        "optimal_last_meal_time": optimal_time,
        "n_days": n,
    }


def compute_macro_hrv_coupling(meals: list, sleep: list,
                               cycle_result: dict = None) -> dict:
    """Correlate daily macro ratios with next-night HRV.

    Finds the individual-optimal macro split. With magnesium data,
    also correlates Mg intake with HRV.
    """
    if not meals or not sleep:
        return {"protein_hrv_r": None, "carb_hrv_r": None,
                "fat_hrv_r": None, "magnesium_hrv_r": None,
                "optimal_split": None, "cycle_adjusted": None,
                "n_days": 0}

    sleep_by_day = {}
    for s in sleep:
        if s.get('day') and s.get('avg_hrv_ms') is not None and \
                s.get('sleep_type') in ('long_sleep', None):
            sleep_by_day[s['day']] = s

    daily = _daily_macro_totals(meals)

    # Pair: day's macros → next-night HRV
    pro_pcts, carb_pcts, fat_pcts, mg_vals, hrv_vals = [], [], [], [], []
    for day, totals in daily.items():
        next_day = (datetime.strptime(day, "%Y-%m-%d")
                    + timedelta(days=1)).strftime("%Y-%m-%d")
        if next_day not in sleep_by_day:
            continue
        cal = totals['calories']
        if cal <= 0:
            continue

        hrv = sleep_by_day[next_day]['avg_hrv_ms']
        pro_pct = (totals['protein_g'] * 4 / cal) * 100
        carb_pct = (totals['carbs_g'] * 4 / cal) * 100
        fat_pct = (totals['fat_g'] * 9 / cal) * 100

        pro_pcts.append(pro_pct)
        carb_pcts.append(carb_pct)
        fat_pcts.append(fat_pct)
        hrv_vals.append(hrv)

        if totals['magnesium_mg'] > 0:
            mg_vals.append(totals['magnesium_mg'])

    n = len(hrv_vals)
    pro_r = round(_pearson_r(pro_pcts, hrv_vals), 2) if n >= 5 else None
    carb_r = round(_pearson_r(carb_pcts, hrv_vals), 2) if n >= 5 else None
    fat_r = round(_pearson_r(fat_pcts, hrv_vals), 2) if n >= 5 else None

    mg_r = None
    if len(mg_vals) >= 5 and len(mg_vals) == n:
        mg_r = round(_pearson_r(mg_vals, hrv_vals), 2)

    # Find optimal split: sort by HRV, take top quartile, average their macros
    optimal = None
    if n >= 8:
        paired = sorted(zip(hrv_vals, pro_pcts, carb_pcts, fat_pcts), reverse=True)
        top_q = paired[:max(2, n // 4)]
        optimal = {
            "protein_pct": round(mean([t[1] for t in top_q]), 0),
            "carb_pct": round(mean([t[2] for t in top_q]), 0),
            "fat_pct": round(mean([t[3] for t in top_q]), 0),
        }

    # Cycle-adjusted: split correlations by cycle phase
    cycle_adj = None
    if cycle_result and cycle_result.get('detected_periods'):
        detected = cycle_result['detected_periods']
        cycle_len = cycle_result.get('estimated_cycle_length', 28) or 28

        def _in_luteal(day_str):
            day_dt = datetime.strptime(day_str, "%Y-%m-%d")
            for i, ps in enumerate(detected):
                ps_dt = datetime.strptime(ps, "%Y-%m-%d")
                next_ps = (datetime.strptime(detected[i + 1], "%Y-%m-%d")
                           if i + 1 < len(detected)
                           else ps_dt + timedelta(days=cycle_len))
                if ps_dt <= day_dt < next_ps:
                    return (day_dt - ps_dt).days >= 14
            return False

        fol_hrv, fol_pro = [], []
        lut_hrv, lut_pro = [], []
        for day, totals in daily.items():
            next_day = (datetime.strptime(day, "%Y-%m-%d")
                        + timedelta(days=1)).strftime("%Y-%m-%d")
            if next_day not in sleep_by_day or totals['calories'] <= 0:
                continue
            hrv = sleep_by_day[next_day]['avg_hrv_ms']
            pro_pct = (totals['protein_g'] * 4 / totals['calories']) * 100
            if _in_luteal(day):
                lut_hrv.append(hrv)
                lut_pro.append(pro_pct)
            else:
                fol_hrv.append(hrv)
                fol_pro.append(pro_pct)

        cycle_adj = {}
        if len(fol_hrv) >= 5:
            cycle_adj["follicular_protein_hrv_r"] = round(
                _pearson_r(fol_pro, fol_hrv), 2)
        if len(lut_hrv) >= 5:
            cycle_adj["luteal_protein_hrv_r"] = round(
                _pearson_r(lut_pro, lut_hrv), 2)

    return {
        "protein_hrv_r": pro_r,
        "carb_hrv_r": carb_r,
        "fat_hrv_r": fat_r,
        "magnesium_hrv_r": mg_r,
        "optimal_split": optimal,
        "cycle_adjusted": cycle_adj,
        "n_days": n,
    }


def compute_nutrition_periodization(meals: list, workouts: list,
                                    sleep: list,
                                    cycle_result: dict = None) -> dict:
    """Score nutritional timing relative to training and cycle phases.

    Evaluates:
    - Training-day protein adequacy
    - Rest-day vs training-day caloric adjustment
    - Cycle-phase macro distribution
    - Overall periodization score
    """
    if not meals:
        return {"score": None, "training_day_adequacy": None,
                "rest_day_comparison": None, "cycle_nutrition": None,
                "gaps": [], "n_days": 0}

    daily = _daily_macro_totals(meals)
    n_days = len(daily)
    gaps = []

    # Training vs rest day comparison
    workout_days = set()
    for w in (workouts or []):
        d = w.get('day')
        if d:
            workout_days.add(d)

    train_cals, rest_cals = [], []
    train_pro, rest_pro = [], []
    for day, totals in daily.items():
        if day in workout_days:
            train_cals.append(totals['calories'])
            train_pro.append(totals['protein_g'])
        else:
            rest_cals.append(totals['calories'])
            rest_pro.append(totals['protein_g'])

    training_adequacy = None
    if train_pro:
        avg_train_pro = mean(train_pro)
        training_adequacy = {
            "avg_training_day_protein_g": round(avg_train_pro, 1),
            "avg_training_day_calories": round(mean(train_cals), 0) if train_cals else None,
            "protein_adequate": avg_train_pro >= 90,
        }
        if avg_train_pro < 90:
            gaps.append(f"Training-day protein avg {avg_train_pro:.0f}g — target 90g+")

    rest_comparison = None
    if train_cals and rest_cals:
        train_avg = mean(train_cals)
        rest_avg = mean(rest_cals)
        diff_pct = round(100 * (train_avg - rest_avg) / rest_avg, 0) if rest_avg > 0 else 0
        rest_comparison = {
            "training_day_avg_cal": round(train_avg, 0),
            "rest_day_avg_cal": round(rest_avg, 0),
            "difference_pct": diff_pct,
        }
        if diff_pct < 5:
            gaps.append("No caloric periodization — consider +10-20% on training days")

    # Cycle-phase nutrition
    cycle_nutrition = None
    if cycle_result and cycle_result.get('detected_periods'):
        detected = cycle_result['detected_periods']
        cycle_len = cycle_result.get('estimated_cycle_length', 28) or 28

        phase_nutrition = {"menstrual": [], "follicular": [], "luteal": []}
        for day, totals in daily.items():
            day_dt = datetime.strptime(day, "%Y-%m-%d")
            for i, ps in enumerate(detected):
                ps_dt = datetime.strptime(ps, "%Y-%m-%d")
                next_ps = (datetime.strptime(detected[i + 1], "%Y-%m-%d")
                           if i + 1 < len(detected)
                           else ps_dt + timedelta(days=cycle_len))
                if ps_dt <= day_dt < next_ps:
                    dic = (day_dt - ps_dt).days
                    if dic < 5:
                        phase_nutrition["menstrual"].append(totals)
                    elif dic < 14:
                        phase_nutrition["follicular"].append(totals)
                    else:
                        phase_nutrition["luteal"].append(totals)
                    break

        cycle_nutrition = {}
        for phase, totals_list in phase_nutrition.items():
            if totals_list:
                cycle_nutrition[phase] = {
                    "avg_calories": round(mean(t['calories'] for t in totals_list), 0),
                    "avg_protein_g": round(mean(t['protein_g'] for t in totals_list), 1),
                    "avg_iron_mg": round(mean(t['iron_mg'] for t in totals_list), 1),
                    "avg_carbs_g": round(mean(t['carbs_g'] for t in totals_list), 1),
                    "n_days": len(totals_list),
                }

        # Check cycle-specific gaps
        mens = cycle_nutrition.get("menstrual", {})
        if mens and mens.get('avg_iron_mg', 0) < 10:
            gaps.append(f"Menstrual phase iron avg {mens.get('avg_iron_mg', 0):.1f}mg — below 10mg")

    # Overall score (0-100)
    score_components = []

    # Protein adequacy (40% weight)
    all_protein = [d['protein_g'] for d in daily.values() if d['protein_g'] > 0]
    if all_protein:
        pro_score = min(100, (mean(all_protein) / 90) * 100)
        score_components.append(('protein', pro_score, 0.4))

    # Caloric periodization (20% weight)
    if train_cals and rest_cals and mean(rest_cals) > 0:
        diff = mean(train_cals) - mean(rest_cals)
        period_score = min(100, max(0, 50 + diff / mean(rest_cals) * 250))
        score_components.append(('periodization', period_score, 0.2))

    # Fiber adequacy (20% weight, target 25g)
    all_fiber = [d['fiber_g'] for d in daily.values() if d['fiber_g'] > 0]
    if all_fiber:
        fiber_score = min(100, (mean(all_fiber) / 25) * 100)
        score_components.append(('fiber', fiber_score, 0.2))
        if mean(all_fiber) < 15:
            gaps.append(f"Fiber avg {mean(all_fiber):.0f}g/day — target 25g+ for gut health")

    # Meal regularity (20% weight)
    meal_counts = [d['meal_count'] for d in daily.values()]
    if meal_counts:
        sd = _safe_stdev(meal_counts)
        reg_score = max(0, 100 - sd * 30)
        score_components.append(('regularity', reg_score, 0.2))

    overall_score = None
    if score_components:
        total_weight = sum(w for _, _, w in score_components)
        overall_score = round(
            sum(s * w for _, s, w in score_components) / total_weight, 0)

    return {
        "score": overall_score,
        "training_day_adequacy": training_adequacy,
        "rest_day_comparison": rest_comparison,
        "cycle_nutrition": cycle_nutrition,
        "gaps": gaps,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=== metrics.py smoke test ===\n")

    # Generate 60 days of synthetic data
    base_date = datetime(2026, 1, 1)
    days = [(base_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(60)]

    # Helper: generate a random hypnogram (~7h sleep)
    def _make_hypnogram():
        stages = []
        # ~84 epochs (7h * 12 per hour). Pattern: 4422211133322211...
        for cycle in range(4):
            stages.extend(['4'] * random.randint(0, 2))  # brief wake
            stages.extend(['2'] * random.randint(4, 8))   # light
            stages.extend(['1'] * random.randint(2, 6))   # deep (more early)
            stages.extend(['2'] * random.randint(2, 4))   # light
            stages.extend(['3'] * random.randint(3, 6))   # REM
        return ''.join(stages[:84])

    # Synthetic sleep records (with bedtime + hypnogram)
    sleep = []
    for i, d in enumerate(days):
        dt = datetime.strptime(d, "%Y-%m-%d")
        bed_h = 22 + random.gauss(0, 0.5)
        bed_start = dt + timedelta(hours=bed_h)
        total_sec = int(25200 + random.gauss(0, 3600))  # ~7h
        bed_end = bed_start + timedelta(seconds=total_sec)
        sleep.append({
            "day": d,
            "avg_hrv_ms": 30 + random.gauss(0, 5),
            "avg_resting_hr_bpm": 60 + random.gauss(0, 3),
            "score": int(70 + random.gauss(0, 8)),
            "deep_sleep_seconds": int(3600 + random.gauss(0, 600)),
            "rem_sleep_seconds": int(5400 + random.gauss(0, 600)),
            "efficiency": 85 + random.gauss(0, 5),
            "total_sleep_seconds": total_sec,
            "sleep_type": "long_sleep",
            "bedtime_start": bed_start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "bedtime_end": bed_end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "onset_latency_seconds": int(300 + random.gauss(0, 120)),
            "awake_seconds": int(1200 + random.gauss(0, 300)),
            "hypnogram_5min": _make_hypnogram(),
        })

    # Synthetic readiness records
    readiness = [{"day": d, "temp_deviation_c": 0.1 * math.sin(2 * math.pi * i / 28) + random.gauss(0, 0.1),
                  "score": int(75 + random.gauss(0, 5))} for i, d in enumerate(days)]

    # Synthetic SpO2
    spo2 = [{"day": d, "avg_spo2_pct": 96 + random.gauss(0, 0.5)} for d in days]

    # Synthetic stress
    stress = [{"day": d, "stress_high_minutes": int(30 + random.gauss(0, 10)),
               "recovery_high_minutes": int(40 + random.gauss(0, 10))} for d in days]

    # Generate 5-min HR data (7 days)
    hr_data = []
    for d in days[:7]:
        dt = datetime.strptime(d, "%Y-%m-%d")
        for h in range(24):
            for m in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                t = dt + timedelta(hours=h, minutes=m)
                circadian_val = 65 + 10 * math.sin(2 * math.pi * (h - 6) / 24) + random.gauss(0, 3)
                src = "sleep" if 0 <= h < 7 else "awake" if 7 <= h < 22 else "rest"
                hr_data.append({
                    "timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"),
                    "bpm": int(circadian_val),
                    "source": src,
                })

    # Workouts
    workouts = [{"day": days[i], "end_time": f"{days[i]}T18:00:00",
                 "start_time": f"{days[i]}T17:00:00",
                 "activity": "walking", "duration_seconds": 3600}
                for i in range(0, 60, 3)]
    # HR around workout times
    workout_hr = list(hr_data)
    for w in workouts:
        dt = _parse_iso_dt(w['end_time'])
        for offset in [-5, 0, 5, 10, 15]:
            t = dt + timedelta(minutes=offset)
            bpm = 120 - offset * 2 + random.gauss(0, 3) if offset >= 0 else 100 + random.gauss(0, 3)
            workout_hr.append({"timestamp": t.strftime("%Y-%m-%dT%H:%M:%S"),
                               "bpm": int(bpm), "source": "workout" if offset <= 0 else "awake"})

    # Tags
    tags = []
    for i in range(0, 60, 4):
        tags.append({"day": days[i], "tag_type": "tag_generic_alcohol"})

    test_num = 0
    def test(name):
        global test_num
        test_num += 1
        print(f"\n{test_num}. {name}:")

    # --- Phase 1 metrics ---
    test("HRV-CV")
    cv = compute_hrv_cv(sleep)
    print(f"   7d CV: {cv['current_cv_7d']}%, Interp: {cv['interpretation']}, Trend: {cv['trend']}")

    test("Cross-Modal Coupling")
    coupling = compute_cross_modal_coupling(sleep, readiness, spo2)
    print(f"   Score: {coupling['coupling_score']}")

    test("Circadian Fingerprint")
    circ = compute_circadian_fingerprint(hr_data)
    if circ:
        print(f"   Mesor: {circ['mesor']} bpm, Amplitude: {circ['amplitude']}, R²={circ['goodness_of_fit']}")

    test("Heart Rate Recovery")
    hrr = compute_heart_rate_recovery(workouts, workout_hr)
    print(f"   HRR-5min: {hrr['avg_hrr5']}, Fitness: {hrr['fitness_indicator']}")

    # --- Phase 2 metrics ---
    test("Alcohol Detection")
    alc = compute_alcohol_detection(sleep)
    print(f"   Flagged nights: {len(alc['flagged_nights'])}, Frequency: {alc['frequency']}")

    test("CUSUM Change-Points")
    cp = detect_change_points(sleep, "avg_hrv_ms")
    print(f"   Change points: {len(cp['change_points'])}, Regime start: {cp['current_regime_start']}")

    test("Allostatic Load")
    al = compute_allostatic_load(sleep, readiness, spo2, stress)
    print(f"   Load: {al['load_score']}/6, Class: {al['classification']}, Trend: {al['trend']}")

    test("Training Load (TRIMP/ACWR)")
    tl = compute_training_load(workouts, workout_hr, sleep)
    print(f"   ACWR: {tl['acwr']}, Zone: {tl['acwr_zone']}, Weekly TRIMP: {tl['weekly_trimp']}")

    test("Nocturnal HR Shape")
    nhr = compute_nocturnal_hr_shape(hr_data, sleep)
    print(f"   Nadir: {nhr['nadir_bpm']} bpm at {nhr['nadir_hour']}h, Dipping: {nhr['dipping_pct']}%, Class: {nhr['classification']}")

    test("Early Warning Signals")
    ews = compute_early_warning_signals(sleep)
    print(f"   Warning: {ews['warning_level']}, HRV AC trend: {ews['hrv_autocorr_trend']}, Var trend: {ews['hrv_variance_trend']}")

    test("Sample Entropy (Daily)")
    ent = compute_daily_entropy(sleep, readiness)
    print(f"   HRV: {ent['hrv_entropy']} ({ent['hrv_interpretation']}), RHR: {ent['rhr_entropy']} ({ent['rhr_interpretation']})")

    test("Temperature Amplitude Trend")
    tat = compute_temp_amplitude_trend(readiness)
    print(f"   Amplitude: {tat['current_amplitude']}, Trend: {tat['amplitude_trend']}")

    test("Sleep Regularity Index")
    sri = compute_sleep_regularity(sleep)
    print(f"   SRI: {sri['sri_score']}, Class: {sri['classification']}, Trend: {sri['trend']}")

    test("Sleep Transitions")
    st = compute_sleep_transitions(sleep)
    print(f"   Frag: {st['fragmentation_index']}, Cycles: {st['avg_cycle_count']}, Nights: {st['n_nights']}")
    top_trans = sorted(st.get('transition_matrix', {}).items(), key=lambda x: -x[1])[:3]
    print(f"   Top transitions: {', '.join(f'{k}: {v}' for k,v in top_trans)}")

    test("Deep Sleep Front-Loading")
    dfl = compute_deep_sleep_distribution(sleep)
    print(f"   Ratio: {dfl['front_loading_ratio']}, First deep at: {dfl['avg_first_deep_min']}min, Class: {dfl['classification']}")

    test("Chronotype (MSFsc)")
    ch = compute_chronotype(sleep)
    print(f"   MSFsc: {ch['chronotype_hour']}h, Social jetlag: {ch['social_jetlag_hours']}h, Type: {ch['classification']}")

    test("Tag-Biometric Effects")
    te = compute_tag_effects(tags, sleep)
    print(f"   Tag types analyzed: {te['n_tag_types_analyzed']}")
    for tag, info in te['tag_effects'].items():
        metrics_str = ', '.join(f"{m}: d={v['cohens_d']}" for m, v in info['metrics'].items())
        print(f"   {tag} (n={info['n_tagged']}): {metrics_str}")

    test("Cycle-Phase Performance")
    # Simulate cycle result
    cycle_result = {"detected_periods": ["2026-01-03", "2026-01-31"], "estimated_cycle_length": 28}
    pp = compute_phase_performance(sleep, workouts, cycle_result)
    for phase, data in pp['phases'].items():
        print(f"   {phase}: HRV={data['avg_hrv']}, RHR={data['avg_rhr']}, n={data['n_nights']}")
    print(f"   Best for intensity: {pp['best_phase_for_intensity']}")

    # --- Phase 3: Crossover metrics ---

    # Synthetic meal data
    meal_foods = [
        ("chicken breast", 165, 31, 0, 3.6),
        ("rice", 206, 4.3, 45, 0.4),
        ("broccoli", 55, 3.7, 11, 0.6),
        ("salmon", 208, 20, 0, 13),
        ("pasta", 220, 8, 43, 1.3),
        ("cheese", 113, 7, 0.4, 9),
        ("eggs", 155, 13, 1.1, 11),
        ("bread", 79, 2.7, 15, 1),
        ("steak", 271, 26, 0, 18),
        ("salad", 20, 1.5, 3.5, 0.2),
    ]
    syn_meals = []
    for i, d in enumerate(days):
        # 2-3 meals per day
        for meal_idx in range(random.randint(2, 3)):
            food = meal_foods[(i * 3 + meal_idx) % len(meal_foods)]
            hour = [8, 13, 19][meal_idx] + random.randint(-1, 1)
            syn_meals.append({
                "day": d,
                "provider": "suna",
                "timestamp": f"{d}T{hour:02d}:00:00",
                "meal_type": ["breakfast", "lunch", "dinner"][meal_idx],
                "calories": food[1] + random.randint(-20, 20),
                "protein_g": food[2] + random.gauss(0, 2),
                "carbs_g": food[3] + random.gauss(0, 3),
                "fat_g": food[4] + random.gauss(0, 1),
                "fiber_g": random.uniform(2, 8),
                "sugar_g": random.uniform(3, 15),
                "saturated_fat_g": random.uniform(1, 8),
                "sodium_mg": random.uniform(200, 800),
                "iron_mg": random.uniform(1, 5),
                "magnesium_mg": random.uniform(20, 80),
                "alcohol_units": 0.5 if random.random() < 0.1 else 0,
                "foods": [{"name": food[0], "calories": food[1],
                           "protein_g": food[2], "fat_g": food[4]}],
            })

    test("Meal-Sleep Effects")
    mse = compute_meal_sleep_effects(syn_meals, sleep)
    print(f"   Profiles: {list(mse['profiles'].keys())}, Best: {mse['best_dinner_profile']}, Nights: {mse['n_nights']}")

    test("Meal Circadian Alignment")
    mca = compute_meal_circadian_alignment(syn_meals, sleep)
    print(f"   Gap: {mca['avg_gap_hours']}h, Regularity: {mca['regularity_score']}, Late: {mca['late_meal_pct']}%, Score: {mca['alignment_score']}")

    test("Thermic Effect")
    te2 = compute_thermic_effect(syn_meals, readiness)
    print(f"   Protein-temp r: {te2['protein_temp_r']}, Carb r: {te2['carb_temp_r']}, Fat r: {te2['fat_temp_r']}")
    print(f"   Late meal impact: {te2['late_meal_temp_impact']}, Optimal time: {te2['optimal_last_meal_time']}")

    test("Macro-HRV Coupling")
    mhc = compute_macro_hrv_coupling(syn_meals, sleep, cycle_result)
    print(f"   Protein-HRV r: {mhc['protein_hrv_r']}, Carb r: {mhc['carb_hrv_r']}, Fat r: {mhc['fat_hrv_r']}")
    print(f"   Mg-HRV r: {mhc['magnesium_hrv_r']}, Optimal: {mhc['optimal_split']}")
    if mhc['cycle_adjusted']:
        print(f"   Cycle-adj: {mhc['cycle_adjusted']}")

    test("Nutrition Periodization")
    np_ = compute_nutrition_periodization(syn_meals, workouts, sleep, cycle_result)
    print(f"   Score: {np_['score']}, Gaps: {len(np_['gaps'])}")
    if np_['training_day_adequacy']:
        print(f"   Training protein: {np_['training_day_adequacy']['avg_training_day_protein_g']}g")
    for g in np_['gaps']:
        print(f"   ! {g}")

    print(f"\n=== All {test_num} tests passed ===")


# ---------------------------------------------------------------------------
# Orchestrator — single entry point for all metrics
# ---------------------------------------------------------------------------

def compute_all(data: dict) -> dict:
    """Run all metrics on a BiometricData dict. Single entry point.

    Args:
        data: Result of ``asdict(fetch_biometrics(days=N))``.

    Returns:
        Dict keyed by metric name, each value is the metric's result dict.
    """
    sleep = data.get('sleep', [])
    readiness = data.get('readiness', [])
    spo2 = data.get('spo2', [])
    stress = data.get('stress', [])
    heartrate = data.get('heartrate', [])
    workouts = data.get('workouts', [])
    tags = data.get('tags', [])
    meals = data.get('meals', [])

    from cycle import detect_cycle_phases
    cycle = detect_cycle_phases(tags, sleep, readiness)

    results = {
        # Autonomic & Circadian
        'hrv_cv': compute_hrv_cv(sleep),
        'coupling': compute_cross_modal_coupling(sleep, readiness, spo2),
        'circadian': compute_circadian_fingerprint(heartrate),
        'hrr': compute_heart_rate_recovery(workouts, heartrate),

        # Sleep Architecture
        'sleep_regularity': compute_sleep_regularity(sleep),
        'transitions': compute_sleep_transitions(sleep),
        'deep_distribution': compute_deep_sleep_distribution(sleep),
        'chronotype': compute_chronotype(sleep),
        'nocturnal': compute_nocturnal_hr_shape(heartrate, sleep),

        # Behavioral Patterns
        'alcohol': compute_alcohol_detection(sleep),
        'allostatic': compute_allostatic_load(sleep, readiness, spo2, stress),
        'early_warning': compute_early_warning_signals(sleep),
        'entropy': compute_daily_entropy(sleep, readiness),
        'temp_amplitude': compute_temp_amplitude_trend(readiness),

        # Training
        'training': compute_training_load(workouts, heartrate, sleep),

        # Change Detection
        'cusum_hrv': detect_change_points(sleep, 'avg_hrv_ms'),
        'cusum_rhr': detect_change_points(sleep, 'avg_resting_hr_bpm'),

        # Tags & Cycle
        'tag_effects': compute_tag_effects(tags, sleep),
        'cycle': cycle,
        'phase_performance': compute_phase_performance(sleep, workouts, cycle),
    }

    # Nutrition crossover (only if meals exist)
    if meals:
        cycle_or_none = cycle if cycle.get('current_phase') != 'unknown' else None
        results.update({
            'meal_sleep': compute_meal_sleep_effects(meals, sleep),
            'meal_circadian': compute_meal_circadian_alignment(meals, sleep),
            'thermic': compute_thermic_effect(meals, readiness),
            'macro_hrv': compute_macro_hrv_coupling(meals, sleep, cycle_or_none),
            'nutrition_periodization': compute_nutrition_periodization(
                meals, workouts, sleep, cycle_or_none),
        })

    return results
