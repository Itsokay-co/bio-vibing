"""Unified workout type normalization across providers.

Maps provider-specific activity names/IDs to normalized types.
Ported from open-wearables constants/workout_types/.
"""

# Normalized category -> all known provider strings
NORMALIZED_TYPES = {
    "running": ["running", "run", "trail_running", "trail run", "treadmill",
                "track & field", "stroller jogging", "jogging"],
    "walking": ["walking", "walk", "stroller walking", "dog walking",
                "caddying", "toddlerwearing", "babywearing", "hiking/rucking",
                "hiking", "rucking", "mountaineering"],
    "cycling": ["cycling", "ride", "road cycling", "gravel cycling",
                "mountain biking", "mountain_biking", "mountainbikeride",
                "gravelride", "ebikeride", "velomobile"],
    "indoor_cycling": ["spin", "indoor_cycling", "indoor cycling",
                       "assault bike", "stationary cycling", "virtualride"],
    "swimming": ["swimming", "swim", "pool swimming", "pool_swimming",
                 "open_water_swimming", "open water swimming", "lap swimming"],
    "strength_training": ["weightlifting", "powerlifting", "strength trainer",
                          "strength_training", "weight training", "weighttraining",
                          "traditionalstrengthtraining"],
    "cardio_training": ["functional fitness", "hiit", "cardio_training",
                        "jumping rope", "jump rope", "obstacle course racing",
                        "parkour", "crossfit", "mixed_metabolic_cardio_training",
                        "high_intensity_interval_training"],
    "yoga": ["yoga", "hot yoga"],
    "pilates": ["pilates"],
    "stretching": ["stretching", "flexibility", "cooldown"],
    "meditation": ["meditation", "mind_and_body", "mindandbody"],
    "elliptical": ["elliptical"],
    "stair_climbing": ["stairmaster", "stair_climbing", "climber",
                       "stadium steps", "stairstepper"],
    "rowing": ["rowing", "indoor rowing", "kayaking", "canoeing", "paddling"],
    "alpine_skiing": ["skiing", "alpine_skiing", "downhill skiing",
                      "alpineski", "backcountryski"],
    "cross_country_skiing": ["cross country skiing", "cross_country_skiing",
                             "nordicski"],
    "snowboarding": ["snowboarding", "snowboard"],
    "ice_skating": ["ice skating", "ice_skating", "iceskate"],
    "soccer": ["soccer", "football (soccer)"],
    "basketball": ["basketball"],
    "american_football": ["football", "american football",
                          "american_football", "australian football"],
    "baseball": ["baseball", "softball"],
    "volleyball": ["volleyball", "beach volleyball"],
    "rugby": ["rugby"],
    "lacrosse": ["lacrosse"],
    "cricket": ["cricket"],
    "hockey": ["ice hockey", "field hockey", "hockey"],
    "tennis": ["tennis"],
    "squash": ["squash"],
    "badminton": ["badminton"],
    "table_tennis": ["table tennis", "table_tennis", "tabletennis"],
    "pickleball": ["pickleball"],
    "boxing": ["boxing", "kickboxing", "box fitness"],
    "martial_arts": ["martial arts", "martial_arts", "jiu jitsu",
                     "fencing", "taekwondo", "karate", "judo"],
    "wrestling": ["wrestling"],
    "rock_climbing": ["rock climbing", "rock_climbing", "climbing",
                      "bouldering"],
    "golf": ["golf"],
    "dance": ["dance", "dancing", "barre", "barre3", "socialdance"],
    "gymnastics": ["gymnastics"],
    "surfing": ["surfing", "water skiing", "wakeboarding", "kite boarding",
                "kitesurfing", "kitesurf", "windsurf"],
    "stand_up_paddleboarding": ["paddleboarding", "stand up paddling",
                                 "standup_paddleboarding",
                                 "standuppaddleboarding"],
    "sailing": ["sailing", "sail"],
    "diving": ["diving"],
    "water_polo": ["water polo", "water_polo", "waterpolo"],
    "triathlon": ["triathlon"],
    "skateboarding": ["skateboard", "skateboarding", "inline_skating",
                      "inline skating", "roller skating", "rollerski"],
}

# Build reverse lookup: lowercase provider string -> normalized type
_REVERSE_MAP = {}
for norm_type, variants in NORMALIZED_TYPES.items():
    for variant in variants:
        _REVERSE_MAP[variant.lower()] = norm_type


def normalize_workout_type(raw, provider=""):
    """Normalize a provider-specific workout type string.

    Args:
        raw: Raw activity string from provider
        provider: Provider name (for provider-specific handling)

    Returns:
        Normalized type string (e.g. "running", "indoor_cycling", "yoga")
    """
    if not raw:
        return "other"

    # Strip Apple Health prefix
    cleaned = raw
    if cleaned.startswith("HKWorkoutActivityType"):
        cleaned = cleaned[len("HKWorkoutActivityType"):]

    # Normalize
    key = cleaned.lower().strip().replace("-", " ").replace("_", " ")

    # Direct lookup
    if key in _REVERSE_MAP:
        return _REVERSE_MAP[key]

    # Try with underscores
    key_under = key.replace(" ", "_")
    if key_under in _REVERSE_MAP:
        return _REVERSE_MAP[key_under]

    # Fallback: return cleaned lowercase with underscores
    return key.replace(" ", "_") or "other"
