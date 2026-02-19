---
name: weekly-report
description: "Actionable weekly health report with sleep debt tracking, personal best comparison, bedtime coaching, and resilience trends. Not just stats — tells you what to DO. Currently supports Oura Ring, Whoop, Fitbit, and Apple Health."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "3.0"
allowed-tools: Bash(python3:*)
---

# Weekly Report — What to do this week

Compare this week to last week AND your personal best. Track sleep debt. Surface actionable recommendations.

## Steps

Pull 90 days of data (this week + last week + 3-month baseline for personal best detection):

```bash
python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from cycle import detect_cycle_phases
from metrics import (compute_hrv_cv, compute_sleep_regularity,
                     compute_allostatic_load,
                     compute_training_load, compute_chronotype,
                     compute_alcohol_detection, compute_early_warning_signals)
from dataclasses import asdict
from datetime import datetime, timedelta
from statistics import mean, stdev
from collections import defaultdict

data = fetch_biometrics(days=90)
d = asdict(data)

today = datetime.now()
this_week_start = (today - timedelta(days=6)).strftime("%Y-%m-%d")
last_week_start = (today - timedelta(days=13)).strftime("%Y-%m-%d")
end = today.strftime("%Y-%m-%d")

# --- Helpers ---
def split_weeks(records, date_key="day"):
    this_w = [r for r in records if r.get(date_key, "") >= this_week_start]
    last_w = [r for r in records if last_week_start <= r.get(date_key, "") < this_week_start]
    return last_w, this_w

def safe_mean(vals):
    return mean(vals) if vals else 0

def safe_stdev(vals):
    return stdev(vals) if len(vals) > 1 else 0

# --- Main sleep (exclude naps) ---
sleep = [s for s in d['sleep'] if s.get('sleep_type') in ('long_sleep', None)] or d['sleep']

# --- Weekly aggregation for personal best ---
def week_key(day_str):
    dt = datetime.strptime(day_str, "%Y-%m-%d")
    return (dt - timedelta(days=dt.weekday())).strftime("%Y-%m-%d")

weeks = defaultdict(list)
for s in sleep:
    if s.get('day'):
        weeks[week_key(s['day'])].append(s)

week_composites = {}
for wk, records in weeks.items():
    scores = [s['score'] for s in records if s.get('score')]
    effs = [s['efficiency'] for s in records if s.get('efficiency')]
    totals = [s['total_sleep_seconds']/3600 for s in records if s.get('total_sleep_seconds')]
    if scores and effs and totals:
        week_composites[wk] = {
            'score': safe_mean(scores),
            'efficiency': safe_mean(effs),
            'total_h': safe_mean(totals),
            'composite': safe_mean(scores) * 0.4 + safe_mean(effs) * 0.3 + min(safe_mean(totals)/8*100, 100) * 0.3,
        }

# --- Output ---
print(f"\n{'='*65}")
print(f"  WEEKLY HEALTH REPORT — {d['provider']}")
print(f"  {this_week_start} to {end}")
print(f"{'='*65}")

# Optimal bedtime
if d.get('optimal_bedtime'):
    print(f"\n  Optimal bedtime window: {d['optimal_bedtime']}")

# --- Sleep debt tracker ---
target_h = 7.5
this_week_sleep = [s for s in sleep if s.get('day', '') >= this_week_start]
total_hours = [s['total_sleep_seconds']/3600 for s in this_week_sleep if s.get('total_sleep_seconds')]
if total_hours:
    nightly_debt = [target_h - h for h in total_hours]
    cumulative_debt = sum(nightly_debt)
    print(f"\n  SLEEP DEBT (vs {target_h}h target)")
    print(f"  {'Night':<12} {'Slept':>6} {'Debt':>7}")
    print(f"  {'-'*28}")
    running = 0
    for s in sorted(this_week_sleep, key=lambda x: x.get('day', '')):
        if s.get('total_sleep_seconds'):
            h = s['total_sleep_seconds'] / 3600
            debt = target_h - h
            running += debt
            flag = " !!" if debt > 2 else ""
            print(f"  {s['day']:<12} {h:>5.1f}h {debt:>+6.1f}h{flag}")
    print(f"  {'':12} {'TOTAL':>6} {cumulative_debt:>+6.1f}h {'← critical' if cumulative_debt > 7 else '← concerning' if cumulative_debt > 3 else ''}")

# --- Night-to-night consistency ---
if len(total_hours) > 1:
    consistency_sd = safe_stdev(total_hours)
    print(f"\n  Consistency: σ = {consistency_sd:.1f}h {'(erratic — aim for <1h variation)' if consistency_sd > 1.5 else '(moderate)' if consistency_sd > 0.8 else '(good)'}")

# --- Week vs week comparison ---
last_w, this_w = split_weeks(sleep)
all_baseline = [s for s in sleep if s.get('day', '') < this_week_start]

def compare(last_vals, this_vals, baseline_vals, label, unit="", higher_is_better=True):
    if not last_vals or not this_vals:
        return None
    last_avg = safe_mean(last_vals)
    this_avg = safe_mean(this_vals)
    if last_avg == 0: return None
    pct = ((this_avg - last_avg) / abs(last_avg)) * 100
    baseline_sd = safe_stdev(baseline_vals) if baseline_vals else 0
    anomaly = abs(this_avg - safe_mean(baseline_vals)) > baseline_sd if baseline_sd > 0 else False
    direction = "up" if pct > 2 else "down" if pct < -2 else "flat"
    good = (direction == "up" and higher_is_better) or (direction == "down" and not higher_is_better)
    return {"label": label, "last": f"{last_avg:.1f}{unit}", "this": f"{this_avg:.1f}{unit}",
            "change": f"{pct:+.1f}%", "good": good, "direction": direction, "anomaly": anomaly}

results = []
if sleep:
    results.append(compare([s['score'] for s in last_w if s.get('score')], [s['score'] for s in this_w if s.get('score')], [s['score'] for s in all_baseline if s.get('score')], "Sleep Score", "", True))
    for field, label, unit, hib, div in [
        ("total_sleep_seconds", "Total Sleep", " hr", True, 3600),
        ("deep_sleep_seconds", "Deep Sleep", " min", True, 60),
        ("rem_sleep_seconds", "REM Sleep", " min", True, 60),
        ("efficiency", "Efficiency", "%", True, 1),
        ("avg_hrv_ms", "HRV", " ms", True, 1),
        ("avg_resting_hr_bpm", "Resting HR", " bpm", False, 1),
    ]:
        results.append(compare([s[field]/div for s in last_w if s.get(field)], [s[field]/div for s in this_w if s.get(field)], [s[field]/div for s in all_baseline if s.get(field)], label, unit, hib))

# Readiness
readiness = d['readiness']
if readiness:
    last_r, this_r = split_weeks(readiness)
    results.append(compare([r['score'] for r in last_r if r.get('score')], [r['score'] for r in this_r if r.get('score')], [r['score'] for r in readiness if r.get('score')], "Readiness", "", True))

# SpO2
spo2 = d.get('spo2', [])
if spo2:
    last_s, this_s = split_weeks(spo2)
    results.append(compare([s['avg_spo2_pct'] for s in last_s if s.get('avg_spo2_pct')], [s['avg_spo2_pct'] for s in this_s if s.get('avg_spo2_pct')], [s['avg_spo2_pct'] for s in spo2 if s.get('avg_spo2_pct')], "SpO2", "%", True))

results = [r for r in results if r is not None]

print(f"\n  {'Metric':<16} {'Last Wk':>10} {'This Wk':>10} {'Change':>8}")
print(f"  {'-'*48}")
for r in results:
    marker = " *" if r["anomaly"] else ""
    print(f"  {r['label']:<16} {r['last']:>10} {r['this']:>10} {r['change']:>8}{marker}")

# Wins/Watch
wins = [r for r in results if r["good"] and r["direction"] != "flat"]
flags = [r for r in results if not r["good"] and r["direction"] != "flat"]
if wins:
    print(f"\n  WINS: {', '.join(w['label'] + ' ' + w['change'] for w in wins)}")
if flags:
    print(f"  WATCH: {', '.join(f['label'] + ' ' + f['change'] for f in flags)}")

# --- Personal best comparison ---
if week_composites:
    best_week = max(week_composites, key=lambda w: week_composites[w]['composite'])
    best = week_composites[best_week]
    current_week = week_key(this_week_start)
    if current_week in week_composites:
        curr = week_composites[current_week]
        print(f"\n  PERSONAL BEST WEEK: {best_week}")
        print(f"    Score: {best['score']:.0f} (you: {curr['score']:.0f})")
        print(f"    Efficiency: {best['efficiency']:.0f}% (you: {curr['efficiency']:.0f}%)")
        print(f"    Sleep: {best['total_h']:.1f}h (you: {curr['total_h']:.1f}h)")

# --- Resilience trend ---
resilience = d.get('resilience', [])
if resilience:
    last_res, this_res = split_weeks(resilience)
    this_levels = [r['level'] for r in this_res if r.get('level')]
    if this_levels:
        print(f"\n  RESILIENCE: {', '.join(this_levels)}")

# --- Autonomic flexibility ---
hrv_cv_all = compute_hrv_cv(sleep)
if hrv_cv_all['current_cv_7d'] is not None:
    # Compute for this week vs last week
    this_w_sleep = [s for s in sleep if s.get('day', '') >= this_week_start]
    last_w_sleep = [s for s in sleep if last_week_start <= s.get('day', '') < this_week_start]
    this_cv = compute_hrv_cv(this_w_sleep, windows=[7])
    last_cv = compute_hrv_cv(last_w_sleep, windows=[7])
    print(f"\n  AUTONOMIC FLEXIBILITY:")
    print(f"    HRV-CV (60d): {hrv_cv_all['current_cv_7d']:.1f}% ({hrv_cv_all['interpretation']})")
    if this_cv['current_cv_7d'] is not None and last_cv['current_cv_7d'] is not None:
        diff = this_cv['current_cv_7d'] - last_cv['current_cv_7d']
        print(f"    This week: {this_cv['current_cv_7d']:.1f}% vs Last week: {last_cv['current_cv_7d']:.1f}% ({diff:+.1f}%)")
    print(f"    Trend: {hrv_cv_all['trend']}")

# --- Sleep regularity ---
sri = compute_sleep_regularity(sleep)
if sri['sri_score'] is not None:
    print(f"\n  SLEEP REGULARITY: {sri['sri_score']}/100 ({sri['classification']})")
    if sri['classification'] == 'irregular':
        print(f"    Irregular schedule is likely hurting more than any single bad night")

# --- Chronotype ---
chrono = compute_chronotype(sleep)
if chrono['chronotype_hour'] is not None:
    h = int(chrono['chronotype_hour'])
    m = int((chrono['chronotype_hour'] % 1) * 60)
    print(f"\n  CHRONOTYPE: {chrono['classification']} (mid-sleep {h:02d}:{m:02d})")
    if chrono['social_jetlag_hours'] > 1:
        print(f"    Social jetlag: {chrono['social_jetlag_hours']}h — aim for <1h")

# --- Allostatic load ---
stress = d.get('stress', [])
al = compute_allostatic_load(sleep, readiness, spo2, stress)
if al['load_score'] is not None and al['load_score'] >= 2:
    print(f"\n  STRESS BURDEN: {al['load_score']}/6 ({al['classification']}, {al['trend']})")
    flagged = [k for k, v in al['per_metric'].items() if v['unfavorable']]
    if flagged:
        print(f"    Overloaded: {', '.join(flagged)}")

# --- Training load ---
workouts = d.get('workouts', [])
heartrate = d.get('heartrate', [])
if workouts:
    tl = compute_training_load(workouts, heartrate, sleep)
    if tl['acwr'] is not None:
        print(f"\n  TRAINING LOAD: ACWR {tl['acwr']} ({tl['acwr_zone']}), Weekly TRIMP {tl['weekly_trimp']}")
        if tl['acwr_zone'] == 'danger':
            print(f"    ⚠ Injury risk — back off intensity")
        elif tl['acwr_zone'] == 'undertraining':
            print(f"    Consider increasing training volume")

# --- Alcohol detection ---
alc = compute_alcohol_detection(sleep)
if alc['flagged_nights']:
    recent_flags = [n for n in alc['flagged_nights'] if n >= this_week_start]
    if recent_flags:
        print(f"\n  ALCOHOL: Probable alcohol nights this week: {', '.join(recent_flags)}")

# --- Early warning ---
ews = compute_early_warning_signals(sleep)
if ews['warning_level'] == 'approaching_transition':
    print(f"\n  ⚠ EARLY WARNING: Rising autocorrelation + variance — body approaching a transition")

# --- Cycle context ---
tags = d.get('tags', [])
readiness = d['readiness']
cycle = detect_cycle_phases(readiness, sleep, period_tags=tags or None)
if cycle['current_phase'] != 'unknown':
    phase = cycle['current_phase']
    day = cycle['estimated_cycle_day']
    print(f"\n  CYCLE: {phase} phase (day {day}), confidence: {cycle['confidence']}")
    if phase in ('luteal', 'luteal (extended)'):
        print(f"    Expect: temp elevated, HRV lower, RHR higher — don't over-interpret dips")
    if cycle['next_period_estimate']:
        print(f"    Next period estimate: {cycle['next_period_estimate']}")

# --- Tags this week ---
this_week_tags = [t for t in tags if t.get('day', '') >= this_week_start]
if this_week_tags:
    print(f"\n  LOGGED EVENTS:")
    for t in this_week_tags:
        label = (t.get('tag_type') or '').replace('tag_generic_', '').replace('_', ' ')
        comment = f" — {t['comment']}" if t.get('comment') else ""
        print(f"    {t['day']}: {label}{comment}")

# --- Workout summary ---
workouts = d.get('workouts', [])
this_week_workouts = [w for w in workouts if w.get('day', '') >= this_week_start]
if this_week_workouts:
    total_cal = sum(w.get('calories', 0) for w in this_week_workouts)
    total_dur = sum(w.get('duration_seconds', 0) for w in this_week_workouts) / 60
    activities = {}
    for w in this_week_workouts:
        a = w.get('activity', 'unknown')
        activities[a] = activities.get(a, 0) + 1
    activity_str = ', '.join(f"{v}x {k}" for k, v in activities.items())
    print(f"\n  EXERCISE: {len(this_week_workouts)} sessions ({activity_str})")
    print(f"    Total: {total_dur:.0f} min, {total_cal:.0f} cal")

    # Exercise-sleep correlation
    workout_days = set(w['day'] for w in this_week_workouts)
    workout_sleep = [s for s in this_week_sleep if s.get('day') in workout_days and s.get('score')]
    rest_sleep = [s for s in this_week_sleep if s.get('day') not in workout_days and s.get('score')]
    if workout_sleep and rest_sleep:
        wo_avg = safe_mean([s['score'] for s in workout_sleep])
        rest_avg = safe_mean([s['score'] for s in rest_sleep])
        diff = wo_avg - rest_avg
        print(f"    Sleep on workout days: {wo_avg:.0f} vs rest days: {rest_avg:.0f} ({diff:+.0f})")

# --- Best/worst night ---
scored = [s for s in this_week_sleep if s.get('score')]
if scored:
    best_night = max(scored, key=lambda s: s['score'])
    worst_night = min(scored, key=lambda s: s['score'])
    print(f"\n  Best night:  {best_night['day']} (score {best_night['score']}, {(best_night.get('total_sleep_seconds') or 0)/3600:.1f}h)")
    print(f"  Worst night: {worst_night['day']} (score {worst_night['score']}, {(worst_night.get('total_sleep_seconds') or 0)/3600:.1f}h)")

print()
PYEOF
```

## Interpretation

After running the data pull, present the report conversationally. Focus on:

1. **Sleep debt** — If cumulative debt is >3h, this is the #1 priority. Reference the optimal bedtime window.
2. **Consistency** — If σ >1.5h, irregular schedule is likely hurting more than any single bad night.
3. **Personal best gap** — What made her best week work? Was she going to bed earlier? More consistent?
4. **One concrete recommendation** — Not "improve your sleep hygiene." Something specific like "Your best sleep nights this week both started before 10:30pm. Your worst started after midnight. Try matching your optimal bedtime window of [X]."
5. **Resilience** — If stuck at "limited" or "adequate," connect it to the sleep debt.

Keep the tone direct and actionable. This is a coach, not a doctor.
