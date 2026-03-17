# bio-vibing

Biohacker's toolbox — 45+ biometric analytics via Claude Code.

Connect wearables, CGM, and nutrition. Pull biometrics. Discover patterns. Track interventions. All from your terminal.

Supports: **Oura** | **Whoop** | **Fitbit** | **Apple Health** | **Garmin** | **Open Wearables** | **Dexcom** | **Nightscout**

## Setup

### 1. Get your access token

| Provider | How to connect |
|----------|---------------|
| **Oura Ring** | [cloud.ouraring.com/personal-access-tokens](https://cloud.ouraring.com/personal-access-tokens) → Create token |
| **Whoop** | [developer.whoop.com](https://developer.whoop.com) → OAuth2 flow → access token |
| **Fitbit** | [dev.fitbit.com](https://dev.fitbit.com) → OAuth2 flow → access token |
| **Apple Health** | iPhone → Settings → Health → Export All Health Data → unzip → point to `export.xml` |
| **Garmin** | Garmin Connect → Settings → Export Data → point to export directory |
| **Open Wearables** | Self-host [openwearables.io](https://openwearables.io) → create API key |
| **Dexcom CGM** | [developer.dexcom.com](https://developer.dexcom.com) → OAuth2 → access token |
| **Nightscout CGM** | Your Nightscout URL + API secret |

### 2. Set the token

```bash
cp .env.example .env
# Edit .env and paste your token
```

### 3. Launch Claude Code with the plugin

```bash
claude --plugin-dir /path/to/bio-vibing
```

## Skills

| Skill | What it does | Example |
|-------|-------------|---------|
| `/bio-vibing:connect` | Test connection, show available data | "Am I connected?" |
| `/bio-vibing:pull` | Fetch N days of biometric data | `/bio-vibing:pull 30` |
| `/bio-vibing:daily` | Morning briefing with baseline context | "How did I sleep?" |
| `/bio-vibing:analyze` | Before/after event comparison | `/bio-vibing:analyze "Quit caffeine" 2026-02-01` |
| `/bio-vibing:discover` | Automated pattern discovery | "What correlates with my best sleep?" |
| `/bio-vibing:gut` | Nutrition-biometric crossover analysis | "How do meals affect my sleep?" |
| `/bio-vibing:smart-mode` | Deep 60-day analysis (45+ metrics) | "What does my data mean?" |
| `/bio-vibing:weekly-report` | This week vs last week + coaching | "How was my week?" |

## How it works

```
Wearable APIs → Provider Adapter → Normalized Schema → 45+ Metrics → Skills
                                        ↓
                                  CGM (Dexcom/Nightscout)
                                  Nutrition (Suna API)
```

Auto-detects your wearable from env vars. All data normalizes to a common schema, so skills work the same regardless of device. Multiple wearables? Set `BIOMETRIC_PROVIDER=oura` to pick one.

## Provider coverage

| Metric | Oura | Whoop | Fitbit | Apple | Garmin | OW | Dexcom | Nightscout |
|--------|:----:|:-----:|:------:|:-----:|:------:|:--:|:------:|:----------:|
| Sleep stages | Y | Y | Y | Y | Y | Y | - | - |
| HRV | Y | Y | Y | Y* | - | Y | - | - |
| Resting HR | Y | Y | Y | Y | Y | Y | - | - |
| Readiness | Y | Y | - | Y | - | Y | - | - |
| Temperature | Y | Y | - | Y | - | Y | - | - |
| Steps | Y | - | Y | Y | Y | Y | - | - |
| Workouts | Y | Y | Y | Y | Y | Y | - | - |
| Body composition | - | - | - | Y | Y | Y | - | - |
| Glucose (5-min) | - | - | - | - | - | - | Y | Y |
| Stress | Y | - | - | - | - | Y | - | - |

\* Apple Health uses SDNN; others use RMSSD.

## Analytics

45+ metrics including:
- **Autonomic**: HRV coefficient of variation, cross-modal coupling, circadian fingerprint
- **Sleep**: regularity index, stage transitions, deep sleep distribution, chronotype
- **Training**: TRIMP, acute/chronic workload ratio, HR zones, recovery index
- **Behavioral**: alcohol detection, allostatic load, early warning signals, entropy
- **Glucose**: variability (CV, time-in-range), post-meal response curves, dawn phenomenon
- **Nutrition**: meal-sleep effects, caffeine-sleep coupling, food item effects, macro-HRV coupling
- **Signals**: personal baselines (z-scores), forward signals, stress/inflammation proxies
- **Discovery**: automated correlation discovery across all data

## Built by

[Chad](https://github.com/itsokay-co) — physician, builder, co-founder of [SUNA Health](https://suna.health).

## License

Apache-2.0
