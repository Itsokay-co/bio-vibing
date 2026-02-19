---
name: analyze
description: "Track biometric impact of any life event — with dose-escalation support, SpO2 monitoring, and cycle-phase awareness. Use when the user started/stopped something and wants to see the data. Arguments: event name and event date(s)."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "3.0"
allowed-tools: Bash(python3:*)
---

# Analyze — Event Impact & Dose Tracking

Compare biometric data before and after a life event. Supports multiple event dates for dose escalation tracking.

## Arguments

- First argument: event name (e.g., "Quit alcohol", "Started creatine", "New job")
- Second argument: event date(s) — single date (YYYY-MM-DD) or comma-separated for dose changes (e.g., "2026-02-02,2026-03-02,2026-03-30")

## Steps

```bash
EVENT_NAME="${1:-Event}"
EVENT_DATES="${2:-$(date +%Y-%m-%d)}"
python3 << PYEOF
import sys, os
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from cycle import detect_cycle_phases
from dataclasses import asdict
from datetime import datetime, timedelta
from statistics import mean, stdev
from collections import defaultdict

event_name = "$EVENT_NAME"
event_dates_str = "$EVENT_DATES"
event_dates = [d.strip() for d in event_dates_str.split(",")]

first_event = datetime.strptime(event_dates[0], "%Y-%m-%d")
pre_start = (first_event - timedelta(days=28)).strftime("%Y-%m-%d")
post_end = datetime.now().strftime("%Y-%m-%d")

print(f"Analyzing: {event_name}")
print(f"Event date(s): {', '.join(event_dates)}")
print(f"Pre-period: {pre_start} to {event_dates[0]}")
print(f"Post-period: {event_dates[0]} to {post_end}")

data = fetch_biometrics(start_date=pre_start, end_date=post_end)
d = asdict(data)

sleep = [s for s in d['sleep'] if s.get('sleep_type') in ('long_sleep', None)] or d['sleep']

def safe_mean(vals): return mean(vals) if vals else 0
def safe_stdev(vals): return stdev(vals) if len(vals) > 1 else 0

# --- PRE/POST COMPARISON ---
def split_pre_post(records, event_date, date_key="day"):
    pre = [r for r in records if r.get(date_key, "") < event_date]
    post = [r for r in records if r.get(date_key, "") >= event_date]
    return pre, post

def compare(pre_vals, post_vals, label, unit="", higher_is_better=True):
    if not pre_vals or not post_vals: return None
    pre_mean = safe_mean(pre_vals)
    post_mean = safe_mean(post_vals)
    if pre_mean == 0: return None
    pct = ((post_mean - pre_mean) / abs(pre_mean)) * 100
    pre_sd = safe_stdev(pre_vals)
    significant = abs(post_mean - pre_mean) > pre_sd if pre_sd > 0 else abs(pct) > 10
    direction = "up" if pct > 0 else "down"
    good = (direction == "up" and higher_is_better) or (direction == "down" and not higher_is_better)
    return {"label": label, "pre": f"{pre_mean:.1f}{unit}", "post": f"{post_mean:.1f}{unit}",
            "change": f"{pct:+.1f}%", "significant": significant,
            "flag": "GOOD" if good and significant else ("FLAG" if not good and significant else ""),
            "pre_mean": pre_mean, "post_mean": post_mean}

# Main comparison against first event date
pre_sleep, post_sleep = split_pre_post(sleep, event_dates[0])
pre_ready, post_ready = split_pre_post(d['readiness'], event_dates[0])

results = []
if sleep:
    results.append(compare([s['score'] for s in pre_sleep if s.get('score')], [s['score'] for s in post_sleep if s.get('score')], "Sleep Score", "", True))
    for field, label, unit, hib in [
        ("deep_sleep_seconds", "Deep Sleep", " min", True),
        ("rem_sleep_seconds", "REM Sleep", " min", True),
        ("total_sleep_seconds", "Total Sleep", " min", True),
        ("efficiency", "Efficiency", "%", True),
        ("avg_hrv_ms", "HRV", " ms", True),
        ("avg_resting_hr_bpm", "Resting HR", " bpm", False),
    ]:
        div = 60 if "seconds" in field else 1
        results.append(compare([s[field]/div for s in pre_sleep if s.get(field)], [s[field]/div for s in post_sleep if s.get(field)], label, unit, hib))

if d['readiness']:
    results.append(compare([r['score'] for r in pre_ready if r.get('score')], [r['score'] for r in post_ready if r.get('score')], "Readiness", "", True))
    results.append(compare([r['temp_deviation_c'] for r in pre_ready if r.get('temp_deviation_c') is not None], [r['temp_deviation_c'] for r in post_ready if r.get('temp_deviation_c') is not None], "Temp Deviation", "°C", False))

results = [r for r in results if r is not None]

print(f"\n{'='*70}")
print(f"BEFORE / AFTER: {event_name}")
print(f"{'='*70}\n")
print(f"{'Metric':<20} {'Pre':>10} {'Post':>10} {'Change':>10} {'Signal':>8}")
print("-" * 62)
for r in results:
    sig = " ***" if r["significant"] else ""
    flag = f"  {r['flag']}" if r["flag"] else ""
    print(f"{r['label']:<20} {r['pre']:>10} {r['post']:>10} {r['change']:>10}{flag}{sig}")

significant = [r for r in results if r["significant"]]
if significant:
    print(f"\nKEY FINDINGS:")
    for f in significant:
        tag = f["flag"] or "NOTE"
        print(f"  [{tag}] {f['label']}: {f['pre']} → {f['post']} ({f['change']})")

# --- DOSE ESCALATION TRACKING ---
if len(event_dates) > 1:
    print(f"\n{'='*70}")
    print(f"DOSE ESCALATION TIMELINE")
    print(f"{'='*70}")
    for i, date in enumerate(event_dates):
        label = f"Dose {i+1}" if i > 0 else "Start"
        next_date = event_dates[i+1] if i+1 < len(event_dates) else post_end
        period_sleep = [s for s in sleep if date <= s.get('day', '') < next_date]

        hrv_vals = [s['avg_hrv_ms'] for s in period_sleep if s.get('avg_hrv_ms')]
        rhr_vals = [s['avg_resting_hr_bpm'] for s in period_sleep if s.get('avg_resting_hr_bpm')]
        eff_vals = [s['efficiency'] for s in period_sleep if s.get('efficiency')]

        days_in_period = (datetime.strptime(next_date, "%Y-%m-%d") - datetime.strptime(date, "%Y-%m-%d")).days
        print(f"\n  {label} ({date}, {days_in_period} days):")
        if hrv_vals: print(f"    HRV: {safe_mean(hrv_vals):.0f}ms")
        if rhr_vals: print(f"    RHR: {safe_mean(rhr_vals):.1f}bpm")
        if eff_vals: print(f"    Efficiency: {safe_mean(eff_vals):.0f}%")

# --- CYCLE PHASE CONTEXT ---
tags = d.get('tags', [])
cycle = detect_cycle_phases(d['readiness'], sleep, period_tags=tags or None)
if cycle['current_phase'] != 'unknown':
    print(f"\n  Cycle detection ({cycle['source']}, confidence: {cycle['confidence']}):")
    print(f"    Current phase: {cycle['current_phase']} (day {cycle['estimated_cycle_day']})")
    print(f"    Cycle length: ~{cycle['cycle_length']} days")
    if cycle['detected_periods']:
        print(f"    Detected periods: {', '.join(cycle['detected_periods'])}")
    # Check if any event dates fall in luteal phase
    for ed in event_dates:
        for period_start in cycle['detected_periods']:
            gap = (datetime.strptime(ed, "%Y-%m-%d") - datetime.strptime(period_start, "%Y-%m-%d")).days
            if 14 <= gap <= cycle.get('cycle_length', 28):
                print(f"    Note: {ed} falls in estimated luteal phase — biometric shifts may overlap")

print()
PYEOF
```

## Output Format

Present as:
1. **Before/after table** with significance markers
2. **Key findings** — only statistically significant changes
3. **Dose escalation** (if multiple dates) — per-period adaptation metrics
4. **Cycle context** — flag if event coincides with luteal phase
