# bio-vibing

Biohacking skills for wearables — advanced biometric analysis via Claude Code.

Connect your wearable. Pull your biometrics. Analyze life events. Get deep reports. All from your terminal.

Supports: **Oura Ring** | **Whoop** | **Fitbit** | **Apple Health**

## Setup

### 1. Get your access token

| Wearable | How to get a token |
|----------|-------------------|
| **Oura Ring** | [cloud.ouraring.com/personal-access-tokens](https://cloud.ouraring.com/personal-access-tokens) → Create a new token |
| **Whoop** | [developer.whoop.com](https://developer.whoop.com) → Create an app → Complete OAuth2 flow to get access token |
| **Fitbit** | [dev.fitbit.com](https://dev.fitbit.com) → Register app → Complete OAuth2 flow to get access token |
| **Apple Health** | iPhone → Settings → Health → Export All Health Data → unzip → point to `export.xml` |

### 2. Set the token

```bash
cp .env.example .env
# Edit .env and paste your token
```

Or export directly:

```bash
# Pick one:
export OURA_ACCESS_TOKEN=your_token_here
export WHOOP_ACCESS_TOKEN=your_token_here
export FITBIT_ACCESS_TOKEN=your_token_here
export APPLE_HEALTH_EXPORT=/path/to/export.xml
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
| `/bio-vibing:analyze` | Before/after event comparison | `/bio-vibing:analyze "Quit caffeine" 2026-02-01` |
| `/bio-vibing:smart-mode` | Deep analysis of your data | "What does my data mean?" |
| `/bio-vibing:weekly-report` | This week vs last week summary | "How was my week?" |

## How it works

The plugin auto-detects your wearable based on which environment variable is set. All data gets normalized to a common format, so the analysis skills work the same regardless of your device.

```
Wearable APIs → Provider Adapter → Normalized Schema → Analysis Skills
```

If you have multiple wearables configured, set `BIOMETRIC_PROVIDER=oura` (or whoop, fitbit, apple_health) to pick one.

## Demo

```
/bio-vibing:connect
/bio-vibing:pull 45
/bio-vibing:analyze "Quit caffeine" 2026-02-01
/bio-vibing:smart-mode
/bio-vibing:weekly-report
```

## Provider coverage

| Metric | Oura | Whoop | Fitbit | Apple Health |
|--------|:----:|:-----:|:------:|:------------:|
| Sleep stages | Y | Y | Y | Y |
| HRV | Y | Y | Y | Y* |
| Resting HR | Y | Y | Y | Y |
| Readiness/Recovery | Y | Y | - | - |
| Temperature | Y | Y | - | Y |
| Steps | Y | - | Y | Y |
| Stress | Y | - | - | - |
| Calories | Y | Y | Y | Y |

\* Apple Health uses SDNN methodology; others use RMSSD.

## Built by

[Chad](https://github.com/itsokay-co) — physician, builder, co-founder of [SUNA Health](https://suna.health) (building the Fitbit for your gut).

## License

Apache-2.0
