---
name: connect
description: Test wearable API connection and show available data. Use when the user wants to verify their setup or see what data is accessible. Currently supports Oura Ring, Whoop, Fitbit, and Apple Health.
compatibility: Requires a wearable token environment variable (e.g., OURA_ACCESS_TOKEN, WHOOP_ACCESS_TOKEN, FITBIT_ACCESS_TOKEN) or APPLE_HEALTH_EXPORT path.
metadata:
  author: suna-health
  version: "2.0"
allowed-tools: Bash(python3:*)
---

# Connect

Test the wearable API connection and report what data is available.

## Steps

1. Detect which wearable is configured and test the connection:

```bash
python3 << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ.get('CLAUDE_PLUGIN_ROOT', '.'), 'lib'))
from providers import detect_provider, get_provider, list_configured_providers, PROVIDERS

# Show detection result
configured = list_configured_providers()
if not configured:
    env_vars = [f"  {v['env']} ({k})" for k, v in PROVIDERS.items()]
    print("ERROR: No wearable configured.")
    print("Set one of these environment variables:")
    print("\n".join(env_vars))
    sys.exit(1)

active = detect_provider()
print(f"Detected provider: {active}")
if len(configured) > 1:
    print(f"All configured: {', '.join(configured)}")
    print(f"(Override with BIOMETRIC_PROVIDER env var)")

# Test connection
provider = get_provider(active)
result = provider.test_connection()

if not result["connected"]:
    print(f"\nERROR: Connection failed — {result['info']}")
    sys.exit(1)

print(f"\nConnected to {active}")
print(f"  {result['info']}")

# Check available data
from datetime import datetime, timedelta
end = datetime.now().strftime("%Y-%m-%d")
start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

print(f"\nAvailable data:")
counts = provider.fetch_available_data(start, end)
for endpoint, count in counts.items():
    print(f"  {endpoint.replace('_', ' ').title()}: {count} records")

# Fetch user profile
profile = provider.fetch_user_profile()
if profile:
    parts = []
    if profile.age: parts.append(f"{profile.age}y")
    if profile.biological_sex: parts.append(profile.biological_sex)
    if profile.weight_kg: parts.append(f"{profile.weight_kg}kg")
    if profile.height_m: parts.append(f"{profile.height_m}m")
    if parts:
        print(f"\nUser: {', '.join(parts)}")

print(f"\nReady to biohack.")
PYEOF
```

## Output Format

Present the results as:

```
Detected provider: [provider]

Connected to [provider]
  [user info]

Available data:
  Sleep:     [N] records
  Readiness: [N] records
  Activity:  [N] records
  Stress:    [N] records

User: [age]y, [sex], [weight]kg

Ready to biohack.
```
