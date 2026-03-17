---
name: gut
description: "Digestive-focused analysis — Suna gut scores, post-meal HR response, caffeine-sleep coupling, food effects, eating windows, and gut-wearable correlations. Works with or without Suna API connection."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "1.0"
allowed-tools: Bash(python3:*)
---

# Gut Analysis — Digestive health meets biometrics

Combines Suna scores (if connected) with open analytics from wearable + meal data.

## Steps

Pull 30 days and run digestive analysis:

```bash
python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from dataclasses import asdict
from statistics import mean
from metrics import (compute_postmeal_hr_response, compute_caffeine_sleep_coupling,
                     compute_food_item_effects, compute_bdi_meal_coupling,
                     compute_meal_sleep_effects, compute_meal_circadian_alignment,
                     compute_gut_score_correlations, compute_digestive_state_biometrics)

data = fetch_biometrics(days=30)
d = asdict(data)
sleep = d.get('sleep', [])
readiness = d.get('readiness', [])
meals = d.get('meals', [])
heartrate = d.get('heartrate', [])
spo2 = d.get('spo2', [])
gut_scores = d.get('gut_scores', [])
digestive_states = d.get('digestive_states', [])
daily_windows = d.get('daily_windows', [])
suna_insights = d.get('suna_insights', [])

print("GUT ANALYSIS — 30 days")
print()

# --- SUNA GUT SCORES (if connected) ---
if gut_scores:
    scores = [g.get('score', 0) for g in gut_scores if g.get('score') is not None]
    if scores:
        print("GUT SCORE TREND")
        print(f"  Avg: {round(mean(scores))} | Best: {max(scores)} | Worst: {min(scores)}")

        # Trend: first half vs second half
        if len(scores) >= 6:
            mid = len(scores) // 2
            first = mean(scores[:mid])
            second = mean(scores[mid:])
            diff = second - first
            trend = "improving" if diff > 2 else "declining" if diff < -2 else "stable"
            print(f"  Trend: {trend} ({'+' if diff > 0 else ''}{round(diff, 1)}/period)")

        # Components (latest)
        latest = sorted(gut_scores, key=lambda x: x.get('day', ''))[-1]
        components = []
        for k, v in (latest.get('components') or {}).items():
            if v is not None:
                components.append(f"{k} {round(v*100) if isinstance(v, (int, float)) else v}")
        if components:
            print(f"  Components: {' | '.join(components)}")
        print()

# --- DIGESTIVE STATES (if connected) ---
if digestive_states:
    proc_times = [ds.get('duration_min') for ds in digestive_states
                  if ds.get('duration_min') is not None]
    if proc_times:
        print("PROCESSING TIMES")
        print(f"  Avg: {round(mean(proc_times)/60, 1)}h | Range: {round(min(proc_times)/60, 1)}h - {round(max(proc_times)/60, 1)}h")

        # By meal type
        by_type = {}
        for ds in digestive_states:
            mt = ds.get('meal_type', 'unknown')
            pt = ds.get('duration_min')
            if pt is not None:
                by_type.setdefault(mt, []).append(pt)
        for mt, pts in sorted(by_type.items()):
            if len(pts) >= 2:
                print(f"  {mt}: {round(mean(pts)/60, 1)}h avg (n={len(pts)})")
        print()

# --- SUNA INSIGHTS (if connected) ---
if suna_insights:
    print("SUNA INSIGHTS")
    for ins in suna_insights[:5]:
        print(f"  {ins.get('headline', ins.get('type', '?'))}")
    print()

# --- POST-MEAL HR RESPONSE (open analytics) ---
if meals and heartrate:
    pmhr = compute_postmeal_hr_response(meals, heartrate)
    if pmhr.get('n_meals_analyzed', 0) > 0:
        print(f"POST-MEAL HR RESPONSE ({pmhr['n_meals_analyzed']} meals)")
        for mt, summary in pmhr.get('by_meal_type', {}).items():
            print(f"  {mt}: +{summary['avg_peak_delta']} bpm peak at {summary['avg_time_to_peak']}min (n={summary['n']})")

        by_prof = pmhr.get('by_macro_profile', {})
        if by_prof:
            best_prof = min(by_prof.items(), key=lambda x: x[1]['avg_peak_delta'])
            print(f"  Best response: {best_prof[0]} meals (+{best_prof[1]['avg_peak_delta']} bpm)")
        if pmhr.get('trend') != 'insufficient_data':
            print(f"  Trend: {pmhr['trend']}")
        print()

# --- CAFFEINE → SLEEP (open analytics) ---
if meals:
    caf = compute_caffeine_sleep_coupling(meals, sleep)
    if caf.get('n_days_with_caffeine', 0) >= 3:
        print(f"CAFFEINE → SLEEP ({caf['n_days_with_caffeine']} caffeine days)")
        print(f"  Avg daily: {caf.get('daily_avg_mg', 0)}mg")
        corr = caf.get('correlations', {})
        for label, r in corr.items():
            if abs(r) >= 0.15:
                print(f"  Caffeine × {label}: r={r}")
        print()

# --- FOOD EFFECTS (open analytics) ---
if meals:
    fe = compute_food_item_effects(meals, sleep)
    if fe.get('n_foods_analyzed', 0) > 0:
        print(f"FOOD EFFECTS ({fe['n_foods_analyzed']} foods)")
        if fe.get('best_foods'):
            print(f"  Best for sleep: {', '.join(fe['best_foods'][:3])}")
        if fe.get('worst_foods'):
            print(f"  Worst for sleep: {', '.join(fe['worst_foods'][:3])}")
        print()

# --- MEAL TIMING (existing metrics) ---
if meals and sleep:
    mca = compute_meal_circadian_alignment(meals, sleep)
    if mca and mca.get('avg_gap_hours') is not None:
        print("MEAL TIMING")
        print(f"  Last meal → bed gap: {mca['avg_gap_hours']}h avg")
        if mca.get('late_meal_pct') is not None:
            print(f"  Late meals (<2h before bed): {mca['late_meal_pct']}%")
        if mca.get('alignment_score') is not None:
            print(f"  Alignment score: {mca['alignment_score']}/100")
        print()

# --- BDI × DINNER (open analytics) ---
if spo2 and meals and sleep:
    bdi = compute_bdi_meal_coupling(spo2, meals, sleep)
    if bdi.get('n_nights', 0) >= 5:
        corr = bdi.get('correlations', {})
        significant = {k: v for k, v in corr.items() if abs(v) >= 0.2}
        if significant:
            print("DINNER × BREATHING DISTURBANCE")
            for k, v in significant.items():
                print(f"  {k} × BDI: r={v}")
            print()

# --- GUT SCORE × WEARABLE (if Suna connected) ---
if gut_scores and sleep:
    gc = compute_gut_score_correlations(gut_scores, sleep)
    corr = gc.get('correlations', {})
    significant = {k: v for k, v in corr.items() if abs(v) >= 0.15}
    if significant:
        print("GUT SCORE × WEARABLE PATTERNS")
        for k, v in significant.items():
            print(f"  {k}: r={v}")
        print()

# --- WINDOWS (if Suna connected) ---
if daily_windows:
    latest_w = sorted(daily_windows, key=lambda x: x.get('day', ''))[-1]
    print("TODAY'S WINDOWS")
    if latest_w.get('eat_start') and latest_w.get('eat_end'):
        print(f"  Eat: {latest_w['eat_start']} – {latest_w['eat_end']}")
    if latest_w.get('train_start'):
        print(f"  Train: {latest_w['train_start']}" + (f" – {latest_w.get('train_end', '')}" if latest_w.get('train_end') else ""))
    if latest_w.get('sleep_start'):
        print(f"  Sleep: {latest_w['sleep_start']}")
    if latest_w.get('recovery_level'):
        print(f"  Recovery: {latest_w['recovery_level']}")

if not meals:
    print("No meal data available. Connect Suna for nutrition-biometric insights.")

PYEOF
```

## Interpretation

Present the output directly. Key framing:

- **Gut Score** (Suna API): 0-100 composite. Higher = better.
- **Processing times** (Suna API): How long meals take to process. Normal adult range 3-5h. Evening meals typically 30-50% longer.
- **Post-meal HR response** (open analytics): Heart rate rises after eating (thermic effect). Normal: 5-20 bpm peak at 30-60min, returns to baseline within 2h.
- **Food effects** (open analytics): Cohen's d effect sizes from Bayesian comparison. |d| > 0.5 = meaningful. Minimum 3 occurrences.
- **Windows** (Suna API): Optimal EAT/TRAIN/SLEEP timing.
- Without Suna API, shows wearable-only analytics (post-meal HR, caffeine, food effects, meal timing).
