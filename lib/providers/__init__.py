"""Provider registry and auto-detection."""

import os
from typing import Optional

from providers.base import BaseProvider

# Registry: env var -> (module, class name)
PROVIDERS = {
    "oura": {"env": "OURA_ACCESS_TOKEN", "module": "providers.oura", "class": "OuraProvider"},
    "whoop": {"env": "WHOOP_ACCESS_TOKEN", "module": "providers.whoop", "class": "WhoopProvider"},
    "fitbit": {"env": "FITBIT_ACCESS_TOKEN", "module": "providers.fitbit", "class": "FitbitProvider"},
    "apple_health": {"env": "APPLE_HEALTH_EXPORT", "module": "providers.apple_health", "class": "AppleHealthProvider"},
}

# Detection priority order
DETECTION_ORDER = ["oura", "whoop", "fitbit", "apple_health"]


def detect_provider() -> Optional[str]:
    """Auto-detect which provider is configured based on env vars.

    Returns provider name or None if nothing is configured.
    BIOMETRIC_PROVIDER env var overrides auto-detection.
    """
    override = os.environ.get("BIOMETRIC_PROVIDER", "").lower()
    if override and override in PROVIDERS:
        return override

    for name in DETECTION_ORDER:
        env_var = PROVIDERS[name]["env"]
        if os.environ.get(env_var):
            return name

    return None


def get_provider(name: Optional[str] = None) -> BaseProvider:
    """Get a provider instance by name, or auto-detect.

    Raises ValueError if no provider is configured.
    """
    if name is None:
        name = detect_provider()

    if name is None:
        env_vars = [f"  {v['env']} ({k})" for k, v in PROVIDERS.items()]
        raise ValueError(
            "No wearable configured. Set one of these environment variables:\n"
            + "\n".join(env_vars)
        )

    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")

    info = PROVIDERS[name]
    import importlib
    mod = importlib.import_module(info["module"])
    cls = getattr(mod, info["class"])
    return cls()


def list_configured_providers() -> list:
    """Return list of provider names that have env vars set."""
    configured = []
    for name in DETECTION_ORDER:
        env_var = PROVIDERS[name]["env"]
        if os.environ.get(env_var):
            configured.append(name)
    return configured
