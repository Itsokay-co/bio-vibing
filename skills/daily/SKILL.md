---
name: daily
description: "Quick morning briefing — last night's sleep with personal baseline context, today's readiness, optimal windows, and watch items. Scannable, under 30 lines. Use when the user wants a fast daily check-in."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "1.0"
allowed-tools: Bash(python3:*)
---

# Daily Briefing — What matters today

Quick morning check-in with personal baseline context. Under 30 lines.

## Steps

Pull 30 days of data (enough for baselines, fast):

```bash
python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from dataclasses import asdict
from metrics import (compute_personal_baselines, compute_forward_signals,
                     compute_training_load, compute_chronotype,
                     compute_allostatic_load, compute_alcohol_detection,
                     compute_early_warning_signals, compute_stress_proxy)

data = fetch_biometrics(days=30)
d = asdict(data)
sleep = d.get('sleep', [])
readiness = d.get('readiness', [])
meals = d.get('meals', [])
spo2 = d.get('spo2', [])
stress = d.get('stress', [])
heartrate = d.get('heartrate', [])
workouts = d.get('workouts', [])
respiration = d.get('respiration', [])

# --- LAST NIGHT ---
long_sleep = sorted([s for s in sleep if s.get('sleep_type') in ('long_sleep', None) and s.get('score')],
                    key=lambda x: x['day'])
if long_sleep:
    last = long_sleep[-1]
    total_h = round(last.get('total_sleep_seconds', 0) / 3600, 1)
    deep_m = round(last.get('deep_sleep_seconds', 0) / 60)
    rem_m = round(last.get('rem_sleep_seconds', 0) / 60)
    total_m = round(last.get('total_sleep_seconds', 0) / 60)
    deep_pct = round(deep_m / total_m * 100) if total_m else 0
    onset_m = round(last.get('onset_latency_seconds', 0) / 60) if last.get('onset_latency_seconds') else None

    print(f"LAST NIGHT")
    print(f"  Sleep: {total_h}h (score {last.get('score', '?')})", end="")

    # Baseline context
    bl = compute_personal_baselines(sleep, readiness, spo2, stress, respiration)
    if bl.get('status') == 'ok' and 'sleep_score' in bl.get('metrics', {}):
        z = bl['metrics']['sleep_score'].get('current', {}).get('1d', {}).get('z_score')
        if z is not None:
            direction = "above" if z > 0 else "below"
            print(f" — {'+' if z > 0 else ''}{z} SD {direction} your baseline", end="")
    print()

    print(f"  Deep: {deep_m} min ({deep_pct}%) | HRV: {last.get('avg_hrv_ms', '?')} ms | RHR: {last.get('avg_resting_hr_bpm', '?')} bpm", end="")
    if onset_m is not None:
        print(f" | Onset: {onset_m} min", end="")
    print()

    # Alcohol detection
    alc = compute_alcohol_detection(sleep)
    if alc.get('probable_alcohol_nights'):
        if last['day'] in [n['day'] for n in alc['probable_alcohol_nights']]:
            print(f"  Probable alcohol night detected")
else:
    print("  No sleep data")

# --- SUNA GUT SCORES (if connected) ---
gut_scores = d.get('gut_scores', [])
overnight_scores = d.get('overnight_scores', [])
if gut_scores:
    latest_gs = sorted(gut_scores, key=lambda x: x.get('day', ''))[-1]
    print(f"  Gut Score: {latest_gs.get('score', '?')} ({latest_gs.get('level', '?')})")
if overnight_scores:
    latest_on = sorted(overnight_scores, key=lambda x: x.get('day', ''))[-1]
    print(f"  Overnight gut: {latest_on.get('score', '?')} ({latest_on.get('level', '?')})")

# --- TODAY'S SIGNALS ---
print()
print("TODAY'S SIGNALS")

# Recovery / readiness
if readiness:
    latest_r = sorted([r for r in readiness if r.get('score')], key=lambda x: x['day'])[-1:]
    if latest_r:
        print(f"  Readiness: {latest_r[0]['score']}/100")

# Training load
tl = compute_training_load(workouts, heartrate, sleep)
acwr = tl.get('acwr')
if acwr is not None:
    zone = tl.get('zone', 'unknown')
    print(f"  ACWR: {acwr} ({zone})")

# Stress proxy
sp = compute_stress_proxy(sleep, readiness, meals)
if sp.get('stress_level') is not None:
    print(f"  Stress: {sp['stress_level']}/100 ({sp['level']})")

# --- WINDOWS ---
windows = d.get('daily_windows', [])
if windows:
    latest_w = sorted(windows, key=lambda x: x.get('day', ''))[-1]
    print()
    print("WINDOWS")
    if latest_w.get('eat_start') and latest_w.get('eat_end'):
        print(f"  Eat: {latest_w['eat_start']} – {latest_w['eat_end']}")
    if latest_w.get('train_start'):
        te = latest_w.get('train_end', '')
        print(f"  Train: {latest_w['train_start']}" + (f" – {te}" if te else ""))
    if latest_w.get('sleep_start'):
        print(f"  Sleep: {latest_w['sleep_start']}")
    rl = latest_w.get('recovery_level')
    if rl:
        print(f"  Recovery: {rl}")
else:
    # Derive from chronotype if no Suna windows
    chrono = compute_chronotype(sleep)
    if chrono.get('classification'):
        print()
        print("TIMING")
        print(f"  Chronotype: {chrono['classification']}")
        if chrono.get('social_jetlag_hours') and chrono['social_jetlag_hours'] > 0.5:
            print(f"  Social jetlag: {chrono['social_jetlag_hours']}h")

# --- WATCH ---
watch_items = []

# Sleep debt
fs = compute_forward_signals(sleep, readiness, workouts)
debt = fs.get('sleep_debt', {})
if debt.get('weekly_debt_hours') and debt['weekly_debt_hours'] > 3:
    ntc = debt.get('nights_to_clear')
    msg = f"Sleep debt: {debt['weekly_debt_hours']}h"
    if ntc:
        msg += f" — clears in ~{ntc} nights"
    watch_items.append(msg)

# HRV declining
hrv_proj = fs.get('hrv_projection', {})
if hrv_proj.get('direction') == 'declining' and abs(hrv_proj.get('slope_per_day', 0)) > 0.5:
    watch_items.append(f"HRV declining ({hrv_proj['slope_per_day']}/day over 7 days)")

# Early warning
ew = compute_early_warning_signals(sleep)
if ew.get('warning_level') == 'elevated':
    watch_items.append("Early warning: rising variance + autocorrelation")

# Allostatic load
al = compute_allostatic_load(sleep, readiness, spo2, stress)
if al.get('classification') in ('high', 'very_high'):
    watch_items.append(f"Allostatic load: {al['classification']} ({al.get('load_score', '?')}/6)")

if watch_items:
    print()
    print("WATCH")
    for item in watch_items:
        print(f"  {item}")

PYEOF
```

## Interpretation

Present the output directly — it's already formatted for scanning. Add brief context only if the user asks follow-up questions. Key framing:

- **Baseline z-scores**: positive = above your personal average, negative = below. ±1 SD is normal variation.
- **Gut Score** (if Suna connected): 0-100 composite health score.
- **Windows** (if Suna connected): optimal EAT/TRAIN/SLEEP timing.
- **WATCH section**: only appears if something is off. No watch = all good.
