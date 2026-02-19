---
name: smart-mode
description: "Deep biometric interpretation with cycle-phase awareness and cross-modal analysis. Separates signal from noise. Use when the user wants to understand what their data actually means."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "4.0"
allowed-tools: Bash(python3:*)
---

# Smart Mode — Deep Biometric Analysis

Generate a deep biometric interpretation with cycle awareness and cross-modal analysis.

## Steps

Pull 60 days of data with all extended metrics:

```bash
python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from cycle import detect_cycle_phases
from metrics import (compute_hrv_cv, compute_cross_modal_coupling,
                     compute_circadian_fingerprint, compute_heart_rate_recovery,
                     compute_alcohol_detection,
                     detect_change_points, compute_allostatic_load,
                     compute_training_load, compute_nocturnal_hr_shape,
                     compute_early_warning_signals, compute_daily_entropy,
                     compute_temp_amplitude_trend, compute_sleep_regularity,
                     compute_sleep_transitions,
                     compute_deep_sleep_distribution, compute_chronotype,
                     compute_tag_effects, compute_phase_performance,
                     compute_meal_sleep_effects, compute_meal_circadian_alignment,
                     compute_thermic_effect, compute_macro_hrv_coupling,
                     compute_nutrition_periodization)
from dataclasses import asdict
from datetime import datetime, timedelta
from statistics import mean, stdev
from collections import defaultdict

data = fetch_biometrics(days=60)
d = asdict(data)

sleep = [s for s in d['sleep'] if s.get('sleep_type') in ('long_sleep', None)] or d['sleep']

def stats(vals, label, unit=""):
    if not vals: return
    avg = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 0
    print(f"  {label}: {avg:.1f}{unit} (SD {sd:.1f}, range {min(vals):.1f}-{max(vals):.1f}, n={len(vals)})")

# --- User profile ---
user = d.get('user') or {}
print(f"=== BIOMETRIC DEEP DIVE — {d['provider']} ===")
print(f"Period: {d['period_start']} to {d['period_end']}")
if user:
    bmi = user.get('weight_kg', 0) / (user.get('height_m', 1) ** 2) if user.get('height_m') else None
    parts = []
    if user.get('age'): parts.append(f"{user['age']}y")
    if user.get('biological_sex'): parts.append(user['biological_sex'])
    if user.get('weight_kg'): parts.append(f"{user['weight_kg']}kg")
    if bmi: parts.append(f"BMI {bmi:.1f}")
    if parts: print(f"Subject: {', '.join(parts)}")

# --- Core metrics ---
print(f"\n--- SLEEP ARCHITECTURE ---")
if sleep:
    stats([s['total_sleep_seconds']/3600 for s in sleep if s.get('total_sleep_seconds')], "Total sleep", "h")
    stats([s['deep_sleep_seconds']/60 for s in sleep if s.get('deep_sleep_seconds')], "Deep sleep", " min")
    stats([s['rem_sleep_seconds']/60 for s in sleep if s.get('rem_sleep_seconds')], "REM sleep", " min")
    stats([s['efficiency'] for s in sleep if s.get('efficiency')], "Efficiency", "%")
    stats([s['score'] for s in sleep if s.get('score')], "Sleep score")

    # Deep/REM as % of total
    total_vals = [s['total_sleep_seconds'] for s in sleep if s.get('total_sleep_seconds')]
    deep_vals = [s['deep_sleep_seconds'] for s in sleep if s.get('deep_sleep_seconds')]
    rem_vals = [s['rem_sleep_seconds'] for s in sleep if s.get('rem_sleep_seconds')]
    if total_vals and deep_vals and rem_vals:
        print(f"  Architecture: Deep {mean(deep_vals)/mean(total_vals)*100:.0f}%, REM {mean(rem_vals)/mean(total_vals)*100:.0f}%")

print(f"\n--- AUTONOMIC NERVOUS SYSTEM ---")
if sleep:
    stats([s['avg_hrv_ms'] for s in sleep if s.get('avg_hrv_ms')], "HRV (RMSSD)", " ms")
    stats([s['avg_resting_hr_bpm'] for s in sleep if s.get('avg_resting_hr_bpm')], "Resting HR", " bpm")
    paired = [(s['avg_hrv_ms'], s['avg_resting_hr_bpm']) for s in sleep if s.get('avg_hrv_ms') and s.get('avg_resting_hr_bpm')]
    if paired:
        ratios = [hrv/hr for hrv, hr in paired]
        print(f"  HRV:HR ratio: {mean(ratios):.2f}")

# --- AUTONOMIC FLEXIBILITY ---
print(f"\n--- AUTONOMIC FLEXIBILITY ---")
hrv_cv = compute_hrv_cv(sleep)
if hrv_cv['current_cv_7d'] is not None:
    print(f"  HRV-CV (7d): {hrv_cv['current_cv_7d']:.1f}% ({hrv_cv['interpretation']})")
    if hrv_cv['current_cv_14d'] is not None:
        print(f"  HRV-CV (14d): {hrv_cv['current_cv_14d']:.1f}%")
    print(f"  Trend: {hrv_cv['trend']}")
else:
    print("  Insufficient HRV data for flexibility analysis")

print(f"\n--- RECOVERY & READINESS ---")
readiness = d['readiness']
if readiness:
    stats([r['score'] for r in readiness if r.get('score')], "Readiness score")
    stats([r['temp_deviation_c'] for r in readiness if r.get('temp_deviation_c') is not None], "Temp deviation", "C")

# --- CROSS-MODAL COUPLING ---
print(f"\n--- CROSS-MODAL COUPLING ---")
spo2 = d.get('spo2', [])
coupling = compute_cross_modal_coupling(sleep, readiness, spo2)
if coupling['coupling_score'] is not None:
    print(f"  Coupling score: {coupling['coupling_score']}/100")
    for pair, info in coupling['correlations'].items():
        status = "coupled" if info['coupled'] else "DECOUPLED"
        print(f"  {pair}: r={info['r']:.2f} ({status}, expected {info['expected_sign']}, n={info['n']})")
    if coupling['decoupling_events']:
        for event in coupling['decoupling_events']:
            print(f"  Warning: {event['description']}")
else:
    print("  Insufficient multi-signal data for coupling analysis")

print(f"\n--- STRESS & RESILIENCE ---")
stress = d.get('stress', [])
if stress:
    stats([s['stress_high_minutes'] for s in stress if s.get('stress_high_minutes') is not None], "Stress high", " min")
    stats([s['recovery_high_minutes'] for s in stress if s.get('recovery_high_minutes') is not None], "Recovery high", " min")
resilience = d.get('resilience', [])
if resilience:
    levels = [r['level'] for r in resilience if r.get('level')]
    if levels:
        from collections import Counter
        level_counts = Counter(levels)
        print(f"  Resilience distribution: {dict(level_counts)}")

# --- SpO2 / BREATHING ---
print(f"\n--- BLOOD OXYGEN & BREATHING ---")
spo2 = d.get('spo2', [])
if spo2:
    valid_spo2 = [s['avg_spo2_pct'] for s in spo2 if s.get('avg_spo2_pct') and s['avg_spo2_pct'] > 0]
    bdi_vals = [s['breathing_disturbance_index'] for s in spo2 if s.get('breathing_disturbance_index') is not None]
    if valid_spo2:
        stats(valid_spo2, "Average SpO2", "%")
    if bdi_vals:
        stats(bdi_vals, "Breathing Disturbance Index")
        elevated_bdi = sum(1 for b in bdi_vals if b > 1.5)
        if elevated_bdi:
            print(f"  Elevated BDI on {elevated_bdi}/{len(bdi_vals)} nights")
else:
    print("  No SpO2 data available")

# --- HEART RATE BREAKDOWN ---
print(f"\n--- HEART RATE (5-min intervals) ---")
heartrate = d.get('heartrate', [])
if heartrate:
    by_source = {}
    for hr in heartrate:
        src = hr.get('source', 'unknown')
        if hr.get('bpm'):
            by_source.setdefault(src, []).append(hr['bpm'])
    for src in ['rest', 'sleep', 'awake', 'workout']:
        if src in by_source:
            vals = by_source[src]
            stats(vals, f"HR ({src})", " bpm")
    print(f"  Total samples: {len(heartrate)} ({len(heartrate)//max(1,len(set(hr.get('timestamp','')[:10] for hr in heartrate)))} avg/day)")
else:
    print("  No intraday heart rate data")

# --- CIRCADIAN RHYTHM ---
print(f"\n--- CIRCADIAN RHYTHM ---")
circadian = compute_circadian_fingerprint(heartrate)
if circadian:
    acro_h = int(circadian['acrophase_hour'])
    acro_m = int((circadian['acrophase_hour'] % 1) * 60)
    print(f"  Mesor (24h mean HR): {circadian['mesor']} bpm")
    print(f"  Amplitude (daily swing): {circadian['amplitude']} bpm")
    print(f"  Peak HR time: {acro_h:02d}:{acro_m:02d}")
    print(f"  Rhythm strength: {circadian['rhythm_strength']} (R2={circadian['goodness_of_fit']:.3f})")
    if circadian.get('weekday') and circadian.get('weekend'):
        wd_h = int(circadian['weekday']['acrophase_hour'])
        wd_m = int((circadian['weekday']['acrophase_hour'] % 1) * 60)
        we_h = int(circadian['weekend']['acrophase_hour'])
        we_m = int((circadian['weekend']['acrophase_hour'] % 1) * 60)
        print(f"  Weekday peak: {wd_h:02d}:{wd_m:02d}, Weekend peak: {we_h:02d}:{we_m:02d}")
        if circadian['social_jetlag_hours'] is not None:
            print(f"  Social jetlag: {circadian['social_jetlag_hours']:.1f}h")
else:
    print("  Insufficient heart rate data for circadian analysis")

# --- EXERCISE ---
print(f"\n--- EXERCISE ---")
workouts = d.get('workouts', [])
if workouts:
    activities = {}
    total_cal = 0
    total_dur = 0
    for w in workouts:
        a = w.get('activity', 'unknown')
        activities[a] = activities.get(a, 0) + 1
        total_cal += w.get('calories', 0)
        total_dur += w.get('duration_seconds', 0)
    print(f"  Sessions: {len(workouts)} over {len(set(w['day'] for w in workouts))} days")
    for act, count in sorted(activities.items(), key=lambda x: -x[1]):
        print(f"    {act}: {count} sessions")
    print(f"  Total: {total_dur/60:.0f} min, {total_cal:.0f} cal")
    durations = [w['duration_seconds']/60 for w in workouts if w.get('duration_seconds')]
    if durations:
        stats(durations, "Session length", " min")
    if heartrate:
        hrr = compute_heart_rate_recovery(workouts, heartrate)
        if hrr['workouts']:
            print(f"\n  Heart Rate Recovery (slow phase, 5-min resolution):")
            print(f"    Avg HRR-5min: {hrr['avg_hrr5']} bpm ({hrr['fitness_indicator']})")
            if hrr['avg_hrr10'] is not None:
                print(f"    Avg HRR-10min: {hrr['avg_hrr10']} bpm")
            print(f"    Trend: {hrr['trend']}")
            print(f"    Based on {len(hrr['workouts'])} workouts with post-exercise HR data")
else:
    print("  No workout data")

# --- CYCLE PHASE DETECTION ---
print(f"\n--- MENSTRUAL CYCLE CONTEXT ---")
tags = d.get('tags', [])
cycle = detect_cycle_phases(readiness, sleep, period_tags=tags or None)

if cycle['current_phase'] != 'unknown':
    print(f"  Source: {cycle['source']}")
    print(f"  Detected periods: {', '.join(cycle['detected_periods']) if cycle['detected_periods'] else 'none'}")
    print(f"  Cycle length: ~{cycle['cycle_length']} days")
    print(f"  Current phase: {cycle['current_phase']} (day {cycle['estimated_cycle_day']})")
    print(f"  Confidence: {cycle['confidence']}")
    if cycle['next_period_estimate']:
        print(f"  Next period estimate: {cycle['next_period_estimate']}")
    if cycle.get('note'):
        print(f"  {cycle['note']}")
else:
    print(f"  {cycle.get('note', 'Could not detect cycle phase')}")
    if user.get('biological_sex') in ('female', 'Female', 'F'):
        print(f"  TIP: Log period in Oura app for higher-confidence cycle detection")

# --- SLEEP REGULARITY ---
print(f"\n--- SLEEP REGULARITY ---")
sri = compute_sleep_regularity(sleep)
if sri['sri_score'] is not None:
    print(f"  SRI: {sri['sri_score']}/100 ({sri['classification']})")
    print(f"  Trend: {sri['trend']}")
else:
    print(f"  Insufficient bedtime data for regularity analysis")

# --- SLEEP TRANSITIONS ---
print(f"\n--- SLEEP STAGE TRANSITIONS ---")
st = compute_sleep_transitions(sleep)
if st['fragmentation_index'] is not None:
    print(f"  Fragmentation index: {st['fragmentation_index']} transitions/hour")
    print(f"  Avg sleep cycles: {st['avg_cycle_count']}")
    if st['avg_cycle_duration_min']:
        print(f"  Avg cycle duration: {st['avg_cycle_duration_min']} min")
    print(f"  Awakenings/night: {st['awakenings_per_night']}")
    matrix = st.get('transition_matrix', {})
    if 'deep->awake' in matrix:
        print(f"  P(deep->awake): {matrix['deep->awake']:.3f}")
    if 'REM->awake' in matrix:
        print(f"  P(REM->awake): {matrix['REM->awake']:.3f}")
else:
    print(f"  No hypnogram data for transition analysis")

# --- DEEP SLEEP DISTRIBUTION ---
print(f"\n--- DEEP SLEEP DISTRIBUTION ---")
dfl = compute_deep_sleep_distribution(sleep)
if dfl['front_loading_ratio'] is not None:
    print(f"  Front-loading ratio: {dfl['front_loading_ratio']:.2f} ({dfl['classification']})")
    if dfl['avg_first_deep_min'] is not None:
        print(f"  First deep sleep epoch: {dfl['avg_first_deep_min']:.0f} min into sleep")
else:
    print(f"  No hypnogram data for distribution analysis")

# --- ALCOHOL DETECTION ---
print(f"\n--- ALCOHOL NIGHT DETECTION ---")
alc = compute_alcohol_detection(sleep)
if alc['per_night']:
    if alc['flagged_nights']:
        print(f"  Probable alcohol nights: {', '.join(alc['flagged_nights'][-5:])}")
        print(f"  Frequency: {alc['frequency']*100:.1f}% of nights")
    else:
        print(f"  No probable alcohol nights detected")
else:
    print(f"  Insufficient data for detection")

# --- ALLOSTATIC LOAD ---
print(f"\n--- ALLOSTATIC LOAD INDEX ---")
al = compute_allostatic_load(sleep, readiness, spo2, stress)
if al['load_score'] is not None:
    print(f"  Load: {al['load_score']}/6 ({al['classification']})")
    print(f"  Trend: {al['trend']}")
    for metric, info in al['per_metric'].items():
        flag = " *" if info['unfavorable'] else ""
        print(f"    {metric}: z={info['z_score']:+.2f}{flag}")
else:
    print(f"  Insufficient data")

# --- EARLY WARNING SIGNALS ---
print(f"\n--- EARLY WARNING SIGNALS ---")
ews = compute_early_warning_signals(sleep)
if ews['warning_level'] != 'insufficient_data':
    print(f"  Warning level: {ews['warning_level']}")
    print(f"  HRV autocorrelation: {ews['hrv_autocorr_trend']}, variance: {ews['hrv_variance_trend']}")
    print(f"  RHR autocorrelation: {ews['rhr_autocorr_trend']}, variance: {ews['rhr_variance_trend']}")
else:
    print(f"  Insufficient data for early warning analysis")

# --- SAMPLE ENTROPY ---
print(f"\n--- COMPLEXITY ANALYSIS (SAMPLE ENTROPY) ---")
ent = compute_daily_entropy(sleep, readiness)
for metric in ['hrv', 'rhr', 'temp']:
    val = ent.get(f'{metric}_entropy')
    interp = ent.get(f'{metric}_interpretation', 'unknown')
    if val is not None:
        print(f"  {metric.upper()}: SampEn={val:.3f} ({interp})")

# --- TEMPERATURE AMPLITUDE ---
print(f"\n--- TEMPERATURE AMPLITUDE ---")
tat = compute_temp_amplitude_trend(readiness)
if tat['current_amplitude'] is not None:
    print(f"  Current amplitude (30d SD): {tat['current_amplitude']:.3f}C")
    print(f"  Trend: {tat['amplitude_trend']}")
else:
    print(f"  Insufficient data")

# --- NOCTURNAL HR SHAPE ---
print(f"\n--- NOCTURNAL HR SHAPE ---")
nhr = compute_nocturnal_hr_shape(heartrate, sleep)
if nhr['nadir_bpm'] is not None:
    nadir_h = int(nhr['nadir_hour'])
    nadir_m = int((nhr['nadir_hour'] % 1) * 60)
    print(f"  Nadir: {nhr['nadir_bpm']} bpm at {nadir_h:02d}:{nadir_m:02d}")
    print(f"  Dipping ratio: {nhr['dipping_pct']}% ({nhr['classification']})")
    if nhr['morning_slope'] is not None:
        print(f"  Morning HR slope: {nhr['morning_slope']} bpm/hour")
else:
    print(f"  Insufficient data")

# --- TRAINING LOAD ---
if workouts:
    print(f"\n--- TRAINING LOAD ---")
    tl = compute_training_load(workouts, heartrate, sleep)
    if tl['acwr'] is not None:
        print(f"  Weekly TRIMP: {tl['weekly_trimp']}")
        print(f"  ACWR: {tl['acwr']} ({tl['acwr_zone']})")
    elif tl['per_workout']:
        print(f"  {len(tl['per_workout'])} workouts tracked, but insufficient data for ACWR")
    else:
        print(f"  No workout HR data available for TRIMP calculation")

# --- CHRONOTYPE ---
print(f"\n--- CHRONOTYPE ---")
chrono = compute_chronotype(sleep)
if chrono['chronotype_hour'] is not None:
    h = int(chrono['chronotype_hour'])
    m = int((chrono['chronotype_hour'] % 1) * 60)
    print(f"  Chronotype (MSFsc): {h:02d}:{m:02d} ({chrono['classification']})")
    print(f"  Workday mid-sleep: {chrono['workday_mid_sleep']}h, Free-day: {chrono['free_mid_sleep']}h")
    print(f"  Social jetlag: {chrono['social_jetlag_hours']}h")
else:
    print(f"  Insufficient bedtime data for chronotype analysis")

# --- CUSUM CHANGE POINTS ---
print(f"\n--- CHANGE-POINT DETECTION (CUSUM) ---")
cp_hrv = detect_change_points(sleep, "avg_hrv_ms")
cp_rhr = detect_change_points(sleep, "avg_resting_hr_bpm")
if cp_hrv['change_points']:
    print(f"  HRV regime shifts: {len(cp_hrv['change_points'])}")
    for cp in cp_hrv['change_points'][-3:]:
        print(f"    {cp['day']}: {cp['direction']} ({cp['magnitude']:+.1f} ms)")
if cp_rhr['change_points']:
    print(f"  RHR regime shifts: {len(cp_rhr['change_points'])}")
    for cp in cp_rhr['change_points'][-3:]:
        print(f"    {cp['day']}: {cp['direction']} ({cp['magnitude']:+.1f} bpm)")
if not cp_hrv['change_points'] and not cp_rhr['change_points']:
    print(f"  No significant regime shifts detected")

# --- TAG EFFECTS ---
if tags:
    print(f"\n--- TAG-BIOMETRIC CORRELATIONS ---")
    te = compute_tag_effects(tags, sleep)
    if te['tag_effects']:
        for tag, info in te['tag_effects'].items():
            print(f"  '{tag}' (n={info['n_tagged']} nights):")
            for metric, v in info['metrics'].items():
                prob_str = f"P={v['prob_effect']:.0%}"
                print(f"    {metric}: {v['direction']} (d={v['cohens_d']}, {prob_str})")
    else:
        print(f"  No tags with enough occurrences for analysis (need {3}+)")

# --- CYCLE PHASE PERFORMANCE ---
if cycle['current_phase'] != 'unknown':
    print(f"\n--- CYCLE-PHASE PERFORMANCE ---")
    pp = compute_phase_performance(sleep, workouts, cycle)
    if pp['phases']:
        for phase, data in pp['phases'].items():
            parts = []
            if data.get('avg_hrv'): parts.append(f"HRV={data['avg_hrv']}")
            if data.get('avg_rhr'): parts.append(f"RHR={data['avg_rhr']}")
            if data.get('avg_sleep_score'): parts.append(f"Sleep={data['avg_sleep_score']:.0f}")
            print(f"  {phase}: {', '.join(parts)} (n={data['n_nights']})")
        if pp['recommendation']:
            print(f"  {pp['recommendation']}")

# --- Optimal bedtime ---
if d.get('optimal_bedtime'):
    print(f"\n--- BEDTIME RECOMMENDATION ---")
    print(f"  Oura optimal window: {d['optimal_bedtime']}")

# === NUTRITION x BIOMETRIC CROSSOVER ===
meals = d.get('meals', [])
if meals:
    print(f"\n{'='*50}")
    print(f"=== NUTRITION x BIOMETRIC CROSSOVER ===")
    print(f"{'='*50}")

    # --- Meal-Sleep Effects ---
    print(f"\n--- MEAL -> SLEEP EFFECTS ---")
    mse = compute_meal_sleep_effects(meals, sleep)
    if mse['profiles']:
        for profile, info in mse['profiles'].items():
            print(f"  {profile} dinners (n={info['n_nights']} nights):")
            for metric, v in info['metrics'].items():
                prob_str = f"P={v['prob_effect']:.0%}"
                print(f"    {metric}: {v['mean']:.1f} vs {v['vs_other_mean']:.1f} ({v['direction']}, d={v['cohens_d']}, {prob_str})")
        if mse['best_dinner_profile']:
            print(f"  Best dinner profile for recovery: {mse['best_dinner_profile']}")
    else:
        print(f"  Insufficient data for meal-sleep analysis")

    # --- Meal Circadian Alignment ---
    print(f"\n--- MEAL TIMING ---")
    mca = compute_meal_circadian_alignment(meals, sleep)
    if mca['avg_gap_hours'] is not None:
        print(f"  Last meal -> bed gap: {mca['avg_gap_hours']}h")
        if mca['regularity_score'] is not None:
            print(f"  Meal timing regularity: {mca['regularity_score']}/100")
        print(f"  Late meals (<2h before bed): {mca['late_meal_pct']}%")
        print(f"  Alignment score: {mca['alignment_score']}/100")
    else:
        print(f"  No meal timestamps for timing analysis")

    # --- Thermic Effect ---
    print(f"\n--- THERMIC EFFECT ---")
    te2 = compute_thermic_effect(meals, readiness)
    if te2['n_days'] >= 5:
        if te2['protein_temp_r'] is not None:
            print(f"  Protein -> temp: r={te2['protein_temp_r']}")
            print(f"  Carbs -> temp: r={te2['carb_temp_r']}")
            print(f"  Fat -> temp: r={te2['fat_temp_r']}")
        if te2['late_meal_temp_impact'] is not None:
            print(f"  Late meal temp impact: {te2['late_meal_temp_impact']:+.3f}C vs early dinner")
        if te2['optimal_last_meal_time']:
            print(f"  Optimal last meal time: {te2['optimal_last_meal_time']}")
    else:
        print(f"  Insufficient data (need 5+ days)")

    # --- Macro-HRV Coupling ---
    print(f"\n--- MACRO -> HRV COUPLING ---")
    mhc = compute_macro_hrv_coupling(meals, sleep, cycle if cycle['current_phase'] != 'unknown' else None)
    if mhc['n_days'] >= 5:
        if mhc['protein_hrv_r'] is not None:
            print(f"  Protein% -> HRV: r={mhc['protein_hrv_r']}")
            print(f"  Carb% -> HRV: r={mhc['carb_hrv_r']}")
            print(f"  Fat% -> HRV: r={mhc['fat_hrv_r']}")
        if mhc['magnesium_hrv_r'] is not None:
            print(f"  Magnesium -> HRV: r={mhc['magnesium_hrv_r']}")
        if mhc['optimal_split']:
            s = mhc['optimal_split']
            print(f"  Optimal split: {s['protein_pct']:.0f}P / {s['carb_pct']:.0f}C / {s['fat_pct']:.0f}F")
        if mhc['cycle_adjusted']:
            for k, v in mhc['cycle_adjusted'].items():
                print(f"  {k}: r={v}")
    else:
        print(f"  Insufficient data (need 5+ days)")

    # --- Nutrition Periodization ---
    print(f"\n--- NUTRITION PERIODIZATION ---")
    np_ = compute_nutrition_periodization(meals, workouts, sleep,
             cycle if cycle['current_phase'] != 'unknown' else None)
    if np_['score'] is not None:
        print(f"  Periodization score: {np_['score']}/100")
        if np_['training_day_adequacy']:
            ta = np_['training_day_adequacy']
            print(f"  Training-day protein: {ta['avg_training_day_protein_g']}g")
        if np_['rest_day_comparison']:
            rc = np_['rest_day_comparison']
            print(f"  Training vs rest-day calories: {rc['difference_pct']:+.0f}%")
        if np_['cycle_nutrition']:
            for phase, data in np_['cycle_nutrition'].items():
                print(f"  {phase}: {data['avg_calories']:.0f}cal, {data['avg_protein_g']:.0f}g protein (n={data['n_days']})")
        for g in np_['gaps']:
            print(f"  * {g}")
    else:
        print(f"  Insufficient data")

else:
    print(f"\n--- NUTRITION x BIOMETRIC CROSSOVER ---")
    print(f"  No meal data available. Connect Suna for nutrition-biometric insights.")

PYEOF
```

## Interpretation Framework

Present the numbers and let the data speak. Focus on:

1. **Patterns** — What changed, what's stable, what's trending
2. **Cross-modal signals** — Where metrics agree or disagree
3. **Cycle context** — If detected, note how phase affects interpretation
4. **Actionable observations** — Concrete data points the user can act on

Keep the tone direct. Present data, not conclusions.
