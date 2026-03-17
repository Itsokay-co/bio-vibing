---
name: discover
description: "Automated pattern discovery — finds what correlates with your best and worst sleep nights. Scans workouts, meals, caffeine, tags, gut scores, and timing patterns. Use when the user wants to know what actually affects their sleep."
compatibility: Requires a wearable token environment variable
metadata:
  author: suna-health
  version: "1.0"
allowed-tools: Bash(python3:*)
---

# Discover — What drives your best and worst nights

Automated correlation discovery across 60 days. Finds patterns you didn't know to look for.

## Steps

Pull 60 days and run correlation discovery:

```bash
python3 << 'PYEOF'
import sys, os, json
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from fetch import fetch_biometrics
from dataclasses import asdict
from metrics import (compute_correlation_discovery, compute_gut_score_correlations,
                     compute_caffeine_sleep_coupling, compute_food_item_effects)

data = fetch_biometrics(days=60)
d = asdict(data)

# --- CORRELATION DISCOVERY ---
disc = compute_correlation_discovery(d)
print(f"DISCOVERY REPORT — {disc.get('n_days_analyzed', 0)} days analyzed")
print()

# Best/worst night profiles
best = disc.get('best_nights')
worst = disc.get('worst_nights')
if best and worst:
    print(f"YOUR BEST SLEEP NIGHTS (avg score {best['avg_score']}):")
    if best.get('common_factors'):
        print(f"  Common factors: {', '.join(best['common_factors'])}")
    print(f"YOUR WORST SLEEP NIGHTS (avg score {worst['avg_score']}):")
    if worst.get('common_factors'):
        print(f"  Common factors: {', '.join(worst['common_factors'])}")
    print()

# Top correlations
correlations = disc.get('correlations', [])
if correlations:
    print("TOP CORRELATIONS:")
    for c in correlations[:10]:
        sign = "+" if c['direction'] == 'positive' else "-"
        print(f"  {sign}  {c['feature']} → {c['outcome']} r={c['r']} (n={c['n']})")
    print()

# --- CAFFEINE COUPLING ---
meals = d.get('meals', [])
sleep = d.get('sleep', [])
if meals:
    caf = compute_caffeine_sleep_coupling(meals, sleep)
    if caf.get('n_days_with_caffeine', 0) >= 5:
        print("CAFFEINE → SLEEP:")
        print(f"  Avg daily: {caf.get('daily_avg_mg', 0)}mg ({caf['n_days_with_caffeine']} days with caffeine)")
        corr = caf.get('correlations', {})
        if corr.get('onset_latency'):
            print(f"  Caffeine × onset latency: r={corr['onset_latency']}")
        if corr.get('deep_sleep'):
            print(f"  Caffeine × deep sleep: r={corr['deep_sleep']}")
        hvl = caf.get('high_vs_low', {})
        if hvl.get('onset_latency'):
            h = hvl['onset_latency']
            print(f"  High caffeine days: onset {round(h['high_caffeine_avg']/60, 1)}min vs low: {round(h['low_caffeine_avg']/60, 1)}min")
        print()

# --- FOOD EFFECTS ---
    fe = compute_food_item_effects(meals, sleep)
    if fe.get('n_foods_analyzed', 0) > 0:
        print(f"FOOD EFFECTS ({fe['n_foods_analyzed']} foods analyzed):")
        if fe.get('best_foods'):
            print(f"  Best for sleep: {', '.join(fe['best_foods'][:3])}")
        if fe.get('worst_foods'):
            print(f"  Worst for sleep: {', '.join(fe['worst_foods'][:3])}")

        for name, effect in list(fe.get('food_effects', {}).items())[:5]:
            metrics = effect.get('metrics', {})
            n = effect.get('n_nights', 0)
            effects_str = ', '.join(f"{m}: d={v['cohens_d']} ({v['direction']})"
                                    for m, v in metrics.items())
            print(f"  {name} (n={n}): {effects_str}")
        print()

# --- GUT SCORE PATTERNS (if Suna connected) ---
gut_scores = d.get('gut_scores', [])
if gut_scores:
    gc = compute_gut_score_correlations(gut_scores, sleep)
    print(f"GUT SCORE PATTERNS ({gc.get('n_days', 0)} days):")
    print(f"  Avg score: {gc.get('avg_score', 0)}")

    corr = gc.get('correlations', {})
    for k, v in corr.items():
        print(f"  {k}: r={v}")

    if gc.get('best_day') and gc.get('worst_day'):
        dow = gc.get('by_day_of_week', {})
        print(f"  Best day: {gc['best_day']} ({dow.get(gc['best_day'], '?')})")
        print(f"  Worst day: {gc['worst_day']} ({dow.get(gc['worst_day'], '?')})")

PYEOF
```

## Interpretation

Present the output directly. Key framing:

- **Correlations**: Pearson r values. |r| > 0.3 = moderate, |r| > 0.5 = strong. Minimum n=7 for inclusion.
- **Food effects**: Cohen's d effect sizes. |d| > 0.5 = medium, |d| > 0.8 = large. Minimum 3 occurrences.
- **Caffeine**: Positive r with onset latency = more caffeine → longer time to fall asleep.
- **Gut score patterns**: Only if Suna API is connected. Shows how gut scores correlate with next-night sleep.
- Correlation ≠ causation. These are patterns worth investigating, not proven causes.
