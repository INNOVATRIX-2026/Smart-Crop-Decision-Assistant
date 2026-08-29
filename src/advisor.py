"""Rule-based **crop management advisor**.

Given a target crop and a field's current readings, this compares each reading to
the crop's learned comfort zone (from :mod:`src.model` stats) and emits prioritized,
human-readable management actions: fertiliser dosing, pH correction, irrigation,
and heat/humidity risk alerts.  It also produces a 0–100 *suitability score* so the
user can see, at a glance, how well the field matches the crop.

Design note: thresholds are derived from the dataset's own per-crop quartiles
rather than hard-coded, so advice adapts to whatever data is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config

# Severity ordering for sorting (higher = more urgent).
SEVERITY_RANK = {"critical": 3, "warning": 2, "good": 1, "info": 0}
SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "good": "🟢", "info": "🔵"}


@dataclass
class Action:
    feature: str
    severity: str          # critical | warning | good | info
    title: str
    detail: str
    current: float | None = None
    target: str | None = None


@dataclass
class Advice:
    crop: str
    suitability: float                       # 0-100
    actions: list[Action] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.suitability >= 80:
            return "Excellent match"
        if self.suitability >= 60:
            return "Workable with management"
        if self.suitability >= 40:
            return "Marginal — significant inputs needed"
        return "Poor match — consider another crop"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _position(value: float, lo: float, hi: float) -> float:
    """Fractional position of ``value`` within ``[lo, hi]`` (can be <0 or >1)."""
    if hi <= lo:
        return 0.5
    return (value - lo) / (hi - lo)


def _band(value: float, s: dict) -> tuple[str, float]:
    """Classify a reading against a crop's per-feature stats.

    Returns ``(band, penalty)`` where band is low/ok/high and penalty in [0,1]
    measures how far outside the comfort zone the reading sits.
    """
    q25, q75 = s["q25"], s["q75"]
    lo, hi = s["min"], s["max"]
    if q25 <= value <= q75:
        return "ok", 0.0
    if value < q25:
        span = max(q25 - lo, 1e-6)
        penalty = min(1.0, (q25 - value) / span)
        return "low", penalty
    span = max(hi - q75, 1e-6)
    penalty = min(1.0, (value - q75) / span)
    return "high", penalty


# Human-friendly nutrient dosing guidance.
_NUTRIENT_NAMES = {"N": "nitrogen", "P": "phosphorus", "K": "potassium"}
_NUTRIENT_SOURCES = {
    "N": "urea / ammonium-based fertiliser or well-rotted manure",
    "P": "single super phosphate (SSP) or bone meal",
    "K": "muriate of potash (MOP) or wood ash",
}


def _nutrient_action(feat: str, value: float, s: dict, band: str, penalty: float) -> Action:
    name = _NUTRIENT_NAMES[feat]
    target = f"{s['q25']:.0f}–{s['q75']:.0f} kg/ha (ideal ≈ {s['median']:.0f})"
    if band == "low":
        deficit = s["median"] - value
        sev = "critical" if penalty > 0.6 else "warning"
        return Action(
            feat, sev,
            f"Apply {name} — soil is deficient",
            f"{name.title()} reads {value:.0f} kg/ha, below the {s['q25']:.0f} kg/ha "
            f"this crop prefers. Add roughly {deficit:.0f} kg/ha via {_NUTRIENT_SOURCES[feat]}, "
            "split across the season for efficiency.",
            current=value, target=target,
        )
    if band == "high":
        sev = "warning" if penalty > 0.6 else "info"
        return Action(
            feat, sev,
            f"Ease off {name} — soil is already rich",
            f"{name.title()} reads {value:.0f} kg/ha, above the typical "
            f"{s['q75']:.0f} kg/ha. Skip or reduce {name} fertiliser this cycle to "
            "avoid runoff, salt buildup, and wasted input cost.",
            current=value, target=target,
        )
    return Action(
        feat, "good", f"{name.title()} is in the ideal range",
        f"{name.title()} of {value:.0f} kg/ha sits comfortably in this crop's "
        "preferred band — maintain current practice.",
        current=value, target=target,
    )


def _ph_action(value: float, s: dict, band: str) -> Action:
    target = f"{s['q25']:.1f}–{s['q75']:.1f} (ideal ≈ {s['median']:.1f})"
    if band == "low":  # acidic
        return Action(
            "ph", "warning", "Raise soil pH — too acidic",
            f"pH {value:.1f} is more acidic than this crop's preferred "
            f"{s['q25']:.1f}. Apply agricultural lime (dolomite) and re-test in "
            "4–6 weeks to nudge pH upward.",
            current=value, target=target,
        )
    if band == "high":  # alkaline
        return Action(
            "ph", "warning", "Lower soil pH — too alkaline",
            f"pH {value:.1f} is more alkaline than the preferred {s['q75']:.1f}. "
            "Incorporate elemental sulphur, gypsum, or organic compost to gradually "
            "acidify the soil.",
            current=value, target=target,
        )
    return Action(
        "ph", "good", "Soil pH is well matched",
        f"pH {value:.1f} is within the ideal band — nutrients will stay available "
        "to the crop.",
        current=value, target=target,
    )


def _rainfall_action(value: float, s: dict, band: str) -> Action:
    target = f"{s['q25']:.0f}–{s['q75']:.0f} mm (ideal ≈ {s['median']:.0f})"
    if band == "low":
        deficit = s["median"] - value
        return Action(
            "rainfall", "warning", "Irrigation likely needed — rainfall is low",
            f"Recent rainfall (~{value:.0f} mm) is below the {s['q25']:.0f} mm this "
            f"crop expects. Plan supplemental irrigation to cover the ~{deficit:.0f} mm "
            "shortfall, ideally via drip to conserve water.",
            current=value, target=target,
        )
    if band == "high":
        return Action(
            "rainfall", "warning", "Ensure drainage — rainfall is high",
            f"Recent rainfall (~{value:.0f} mm) exceeds the typical {s['q75']:.0f} mm. "
            "Ensure fields drain well to prevent waterlogging and root rot; consider "
            "raised beds or drainage channels.",
            current=value, target=target,
        )
    return Action(
        "rainfall", "good", "Water supply looks balanced",
        f"Rainfall of ~{value:.0f} mm matches this crop's needs — monitor and "
        "irrigate only if a dry spell sets in.",
        current=value, target=target,
    )


def _temperature_action(value: float, s: dict, band: str) -> Action:
    target = f"{s['q25']:.1f}–{s['q75']:.1f} °C (ideal ≈ {s['median']:.1f})"
    if band == "low":
        return Action(
            "temperature", "warning", "Cold stress risk — temperature is low",
            f"At {value:.1f} °C conditions are cooler than this crop's preferred "
            f"{s['q25']:.1f} °C. Delay sowing to a warmer window, or use mulching / "
            "row covers to retain heat.",
            current=value, target=target,
        )
    if band == "high":
        return Action(
            "temperature", "warning", "Heat stress risk — temperature is high",
            f"At {value:.1f} °C conditions are hotter than the preferred "
            f"{s['q75']:.1f} °C. Increase irrigation frequency, apply mulch to cool "
            "roots, and consider shade or a cooler planting date.",
            current=value, target=target,
        )
    return Action(
        "temperature", "good", "Temperature is favourable",
        f"{value:.1f} °C is within this crop's comfortable range.",
        current=value, target=target,
    )


def _humidity_action(value: float, s: dict, band: str) -> Action:
    target = f"{s['q25']:.0f}–{s['q75']:.0f}% (ideal ≈ {s['median']:.0f})"
    if band == "high":
        return Action(
            "humidity", "warning", "Disease watch — humidity is high",
            f"Humidity of {value:.0f}% is above the typical {s['q75']:.0f}% for this "
            "crop, which favours fungal disease. Scout regularly, improve airflow with "
            "wider spacing, and have preventive fungicide ready.",
            current=value, target=target,
        )
    if band == "low":
        return Action(
            "humidity", "info", "Dry air — monitor for moisture stress",
            f"Humidity of {value:.0f}% is below the usual {s['q25']:.0f}%. Watch for "
            "faster soil drying and increased water demand.",
            current=value, target=target,
        )
    return Action(
        "humidity", "good", "Humidity is in a healthy range",
        f"{value:.0f}% humidity is well suited to this crop.",
        current=value, target=target,
    )


_BUILDERS = {
    "N": _nutrient_action, "P": _nutrient_action, "K": _nutrient_action,
    "ph": _ph_action, "rainfall": _rainfall_action,
    "temperature": _temperature_action, "humidity": _humidity_action,
}


def advise(crop: str, features: dict, crop_stats: dict) -> Advice:
    """Produce prioritized management actions and a suitability score for ``crop``."""
    crop = crop.lower()
    if crop not in crop_stats:
        raise KeyError(f"No statistics available for crop '{crop}'.")
    stats = crop_stats[crop]

    actions: list[Action] = []
    penalties: list[float] = []

    for feat in config.FEATURES:
        value = float(features[feat])
        s = stats[feat]
        band, penalty = _band(value, s)
        penalties.append(penalty)

        builder = _BUILDERS[feat]
        if feat in ("N", "P", "K"):
            actions.append(builder(feat, value, s, band, penalty))
        else:
            actions.append(builder(value, s, band))

    # Suitability: average comfort across features, mapped to 0-100.
    suitability = round(100 * (1 - sum(penalties) / len(penalties)), 1)

    # Sort: most urgent first, then by feature order for stability.
    actions.sort(key=lambda a: (-SEVERITY_RANK[a.severity], config.FEATURES.index(a.feature)))

    return Advice(crop=crop, suitability=suitability, actions=actions)
