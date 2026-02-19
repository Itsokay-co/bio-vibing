---
name: pull
description: "Pull wearable biometric data for analysis. Use when the user wants to see their sleep, HRV, readiness, heart rate, temperature, activity, or stress data. Arguments: number of days (default 14). Currently supports Oura Ring, Whoop, Fitbit, and Apple Health."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "2.0"
allowed-tools: Bash(python3:*)
---

# Pull

Fetch wearable biometric data and present human-readable insights.

## Arguments

- First argument: number of days to pull (default: 14)

## Steps

1. Fetch and analyze biometric data:

```bash
DAYS=${1:-14}
python3 << PYEOF
import sys, os
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from statistics import mean

DAYS = int("$DAYS")
data = fetch_biometrics(days=DAYS)

print(f"Pulling {DAYS} days from {data['provider']}: {data['period_start']} to {data['period_end']}")

# --- SLEEP ---
sleep = [s for s in data['sleep'] if s.get('sleep_type') == 'long_sleep'] or data['sleep']
if sleep:
    scores = [s['score'] for s in sleep if s.get('score')]
    deep = [s['deep_sleep_seconds']/60 for s in sleep if s.get('deep_sleep_seconds')]
    rem = [s['rem_sleep_seconds']/60 for s in sleep if s.get('rem_sleep_seconds')]
    light = [s['light_sleep_seconds']/60 for s in sleep if s.get('light_sleep_seconds')]
    total = [s['total_sleep_seconds']/60 for s in sleep if s.get('total_sleep_seconds')]
    efficiency = [s['efficiency'] for s in sleep if s.get('efficiency')]
    hrv = [s['avg_hrv_ms'] for s in sleep if s.get('avg_hrv_ms')]
    hr = [s['avg_resting_hr_bpm'] for s in sleep if s.get('avg_resting_hr_bpm')]

    def fmt(vals, unit=''):
        if not vals: return 'N/A'
        avg = mean(vals)
        return f'{avg:.1f}{unit} avg (range {min(vals):.1f}-{max(vals):.1f})'

    print(f'\nSLEEP ({len(sleep)} nights)')
    if scores: print(f'  Score: {mean(scores):.0f} avg (range {min(scores)}-{max(scores)})')
    if total: print(f'  Total sleep: {fmt(total, " min")}')
    if deep: print(f'  Deep: {fmt(deep, " min")}')
    if rem: print(f'  REM: {fmt(rem, " min")}')
    if light: print(f'  Light: {fmt(light, " min")}')
    if efficiency: print(f'  Efficiency: {fmt(efficiency, "%")}')
    if hrv: print(f'  Avg HRV: {fmt(hrv, " ms")}')
    if hr: print(f'  Avg resting HR: {fmt(hr, " bpm")}')

    scored = [s for s in sleep if s.get('score')]
    if scored:
        best = max(scored, key=lambda s: s['score'])
        worst = min(scored, key=lambda s: s['score'])
        print(f'  Best night: {best["day"]} (score {best["score"]})')
        print(f'  Worst night: {worst["day"]} (score {worst["score"]})')
else:
    print('\nNo sleep data found.')

# --- READINESS ---
readiness = data['readiness']
if readiness:
    scores = [r['score'] for r in readiness if r.get('score')]
    if scores:
        print(f'\nREADINESS ({len(readiness)} days)')
        print(f'  Score: {mean(scores):.0f} avg (range {min(scores)}-{max(scores)})')
        best = max(readiness, key=lambda r: r.get('score', 0))
        worst = min(readiness, key=lambda r: r.get('score', 0))
        print(f'  Best day: {best["day"]} (score {best.get("score", "N/A")})')
        print(f'  Worst day: {worst["day"]} (score {worst.get("score", "N/A")})')

        # Contributors
        for key, label in [
            ('temp_deviation_c', 'Temp Deviation'),
            ('hrv_balance_score', 'HRV Balance'),
            ('recovery_index_score', 'Recovery Index'),
            ('sleep_balance_score', 'Sleep Balance'),
            ('activity_balance_score', 'Activity Balance'),
        ]:
            vals = [r[key] for r in readiness if r.get(key) is not None]
            if vals:
                print(f'  {label}: {mean(vals):.0f} avg')

# --- ACTIVITY ---
activity = data['activity']
if activity:
    scores = [a['score'] for a in activity if a.get('score')]
    steps = [a['steps'] for a in activity if a.get('steps')]
    cals = [a['total_calories'] for a in activity if a.get('total_calories')]

    print(f'\nACTIVITY ({len(activity)} days)')
    if scores: print(f'  Score: {mean(scores):.0f} avg (range {min(scores)}-{max(scores)})')
    if steps: print(f'  Steps: {mean(steps):.0f} avg/day (range {min(steps)}-{max(steps)})')
    if cals: print(f'  Calories: {mean(cals):.0f} avg/day')

# --- STRESS ---
stress = data['stress']
if stress:
    stress_high = [s['stress_high_minutes'] for s in stress if s.get('stress_high_minutes') is not None]
    recovery = [s['recovery_high_minutes'] for s in stress if s.get('recovery_high_minutes') is not None]

    print(f'\nSTRESS ({len(stress)} days)')
    if stress_high: print(f'  High stress minutes: {mean(stress_high):.0f} avg/day')
    if recovery: print(f'  Recovery minutes: {mean(recovery):.0f} avg/day')

PYEOF
```

## Output Format

Present results grouped by category with trend arrows:
- Up arrow (trending up from first half to second half of period)
- Down arrow (trending down)
- Flat (within 5% change)

End with a one-line summary: "Your [best metric] is strong. [Worst metric] needs attention."
