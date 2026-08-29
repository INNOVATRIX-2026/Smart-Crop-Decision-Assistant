"""Soil properties from SoilGrids, plus pedotransfer to water-holding limits.

Two jobs:
  1. Fetch texture, bulk density, pH, organic carbon and total N for a lat/lon.
  2. Convert texture + organic matter into field capacity and wilting point via the
     Saxton & Rawls (2006) pedotransfer functions — the numbers the FAO-56 water
     balance actually needs, and which no API supplies directly.

**Verified SoilGrids behaviour (checked live, Aug 2026).** Scaling factors are real
and must be applied: pH is returned ×10, sand/clay/silt in g/kg (÷10 for %), bulk
density in cg/cm³ (÷100), organic carbon in dg/kg (÷10), nitrogen in cg/kg (÷100).

**It also has coverage gaps over India.** Queries over Punjab — both Ludhiana city
and surrounding farmland — return HTTP 200 with every value ``null``, while Haryana,
Madhya Pradesh and Tamil Nadu return data. Punjab is a major wheat region, so a
null response is a normal case to design for, not an error. The resolution order is:

    exact point -> nearby offsets -> texture preset chosen by the user

and the provenance of every value is tracked so the UI can say which is which.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import requests

from . import config
from .engine.nutrients import SoilNutrients
from .engine.water import SoilWater

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# Depth layers to average, with their thickness in cm for weighting. The top 30 cm
# carries most root activity and most of the nutrient pool.
DEPTHS = (("0-5cm", 5.0), ("5-15cm", 10.0), ("15-30cm", 15.0))

# Scaling divisors confirmed against the live API's `unit_measure.d_factor`.
D_FACTOR = {
    "sand": 10.0,      # g/kg  -> %
    "clay": 10.0,      # g/kg  -> %
    "silt": 10.0,      # g/kg  -> %
    "bdod": 100.0,     # cg/cm3 -> kg/dm3
    "phh2o": 10.0,     # pH*10 -> pH
    "soc": 10.0,       # dg/kg -> g/kg
    "nitrogen": 100.0,  # cg/kg -> g/kg
}
PROPERTIES = tuple(D_FACTOR)

# Offsets (degrees) tried when the exact pixel is null. ~0.05 deg is roughly 5 km.
FALLBACK_OFFSETS = (0.05, -0.05, 0.10, -0.10)

# Van Bemmelen factor: organic matter % from organic carbon %.
OM_FROM_OC = 1.724

# Plant-available (KMnO4-oxidizable) nitrogen as a fraction of TOTAL soil nitrogen.
# SoilGrids reports total N; Soil Health Card fertility bands describe available N.
# Literature puts the available fraction at roughly 5-10 % of total. 0.08 taken.
# This single number decides the N fertility class and hence the nitrogen dose, so
# it is among the most important values to verify. [REVIEW]
AVAILABLE_N_FRACTION = 0.08

CACHE_DIR = config.DATA_DIR / "cache"

# Representative sand/clay percentages per USDA texture class, for the last-resort
# fallback when SoilGrids has no data. Values are class midpoints. [REVIEW]
TEXTURE_PRESETS: dict[str, dict[str, float]] = {
    "Sandy":      {"sand": 85.0, "clay": 5.0,  "soc": 4.0},
    "Sandy loam": {"sand": 65.0, "clay": 10.0, "soc": 6.0},
    "Loam":       {"sand": 40.0, "clay": 20.0, "soc": 8.0},
    "Clay loam":  {"sand": 30.0, "clay": 33.0, "soc": 10.0},
    "Clay":       {"sand": 20.0, "clay": 50.0, "soc": 12.0},
}
DEFAULT_TEXTURE = "Loam"

PROV_MEASURED = "measured"
PROV_ESTIMATED = "estimated"
PROV_ASSUMED = "assumed"


# --------------------------------------------------------------------------
# Pedotransfer — Saxton & Rawls (2006)
# --------------------------------------------------------------------------
def saxton_rawls(sand_pct: float, clay_pct: float, om_pct: float) -> tuple[float, float]:
    """Field capacity and wilting point (m³/m³) from texture and organic matter.

    Saxton & Rawls (2006), *Soil Water Characteristic Estimates by Texture and
    Organic Matter for Hydrologic Solutions*, Table 1. ``S`` and ``C`` enter as
    decimal fractions; ``OM`` as a percentage.

    Sanity check for a loam (40 % sand, 20 % clay, 1.5 % OM): FC ≈ 0.269,
    WP ≈ 0.131, so ~0.14 m³/m³ of available water — squarely in the published
    0.13-0.17 range for loams.
    """
    s = max(0.0, min(1.0, sand_pct / 100.0))
    c = max(0.0, min(1.0, clay_pct / 100.0))
    om = max(0.0, min(8.0, om_pct))

    # Wilting point (1500 kPa)
    wp_t = (
        -0.024 * s + 0.487 * c + 0.006 * om
        + 0.005 * (s * om) - 0.013 * (c * om) + 0.068 * (s * c) + 0.031
    )
    wp = wp_t + (0.14 * wp_t - 0.02)

    # Field capacity (33 kPa)
    fc_t = (
        -0.251 * s + 0.195 * c + 0.011 * om
        + 0.006 * (s * om) - 0.027 * (c * om) + 0.452 * (s * c) + 0.299
    )
    fc = fc_t + (1.283 * fc_t * fc_t - 0.374 * fc_t - 0.015)

    # Guard against the equations straying outside physical bounds at extreme
    # inputs — a negative or inverted pair would crash the water balance.
    wp = max(0.02, min(0.45, wp))
    fc = max(wp + 0.02, min(0.55, fc))
    return fc, wp


# --------------------------------------------------------------------------
# Soil profile
# --------------------------------------------------------------------------
@dataclass
class SoilProfile:
    """Soil properties for one location, with per-field provenance."""

    latitude: float
    longitude: float
    sand_pct: float
    clay_pct: float
    silt_pct: float
    bulk_density: float          # kg/dm3
    ph: float
    soc_g_kg: float              # soil organic carbon
    total_n_g_kg: float
    source: str                  # human-readable provenance summary
    provenance: dict[str, str] = field(default_factory=dict)
    fetched_at: str | None = None

    @property
    def om_pct(self) -> float:
        """Organic matter %, from organic carbon."""
        return (self.soc_g_kg / 10.0) * OM_FROM_OC

    @property
    def texture_class(self) -> str:
        """Coarse USDA-style label, for display."""
        if self.clay_pct >= 40:
            return "Clay"
        if self.clay_pct >= 27:
            return "Clay loam"
        if self.sand_pct >= 70:
            return "Sandy"
        if self.sand_pct >= 52:
            return "Sandy loam"
        return "Loam"

    def water(self) -> SoilWater:
        """Water-holding limits for the FAO-56 balance."""
        fc, wp = saxton_rawls(self.sand_pct, self.clay_pct, self.om_pct)
        return SoilWater(theta_fc=fc, theta_wp=wp)

    def nutrients(self, root_depth_m: float = 0.30) -> SoilNutrients:
        """Plant-available nutrients, kg/ha.

        Nitrogen is *estimated*. SoilGrids reports **total** soil N (typically
        2000-8000 kg/ha over 30 cm), whereas the Soil Health Card fertility bands
        (280-560 kg/ha for 'medium') describe **plant-available** N as measured by
        KMnO4 oxidation. Comparing the two directly would misclassify almost every
        soil as 'high' and cut the nitrogen dose accordingly — the same units
        confusion this rewrite exists to remove. Available N is therefore taken as a
        documented fraction of total N.

        Phosphorus and potassium are *assumed* at the medium class: SoilGrids carries
        neither, and inventing a number would be worse than admitting the gap.
        """
        # Soil mass over the sampled depth, kg/ha.
        soil_mass_kg_ha = 10_000.0 * root_depth_m * self.bulk_density * 1000.0
        total_n_kg_ha = self.total_n_g_kg * soil_mass_kg_ha / 1000.0
        available_n_kg_ha = total_n_kg_ha * AVAILABLE_N_FRACTION

        return SoilNutrients(
            available={
                "N": available_n_kg_ha,
                "P2O5": 15.0,   # midpoint of the SHC 'medium' band (10-25)
                "K2O": 150.0,   # midpoint of the SHC 'medium' band (108-280)
            },
            provenance={
                "N": PROV_ESTIMATED,
                "P2O5": PROV_ASSUMED,
                "K2O": PROV_ASSUMED,
            },
        )


def from_texture(
    latitude: float, longitude: float, texture: str = DEFAULT_TEXTURE,
    ph: float = 7.0, note: str = "texture preset",
) -> SoilProfile:
    """Build a profile from a named texture class, for when SoilGrids has no data."""
    preset = TEXTURE_PRESETS.get(texture, TEXTURE_PRESETS[DEFAULT_TEXTURE])
    sand, clay = preset["sand"], preset["clay"]
    return SoilProfile(
        latitude=latitude, longitude=longitude,
        sand_pct=sand, clay_pct=clay, silt_pct=max(0.0, 100.0 - sand - clay),
        bulk_density=1.40, ph=ph, soc_g_kg=preset["soc"], total_n_g_kg=0.7,
        source=f"{note} ({texture})",
        provenance={k: PROV_ASSUMED for k in
                    ("sand", "clay", "silt", "bdod", "ph", "soc", "nitrogen")},
    )


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def _cache_path(latitude: float, longitude: float) -> Path:
    return CACHE_DIR / f"soil_{latitude:.2f}_{longitude:.2f}.json"


def _query(latitude: float, longitude: float, timeout: int = 60) -> dict[str, float]:
    """One SoilGrids call. Returns scaled values, omitting any that came back null."""
    params: list[tuple[str, object]] = [("lon", longitude), ("lat", latitude)]
    params += [("property", p) for p in PROPERTIES]
    params += [("depth", d) for d, _ in DEPTHS]
    params.append(("value", "mean"))

    resp = requests.get(SOILGRIDS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    layers = resp.json().get("properties", {}).get("layers", [])

    out: dict[str, float] = {}
    for layer in layers:
        name = layer.get("name")
        if name not in D_FACTOR:
            continue
        total = weight = 0.0
        for depth in layer.get("depths", []):
            label = depth.get("label")
            thickness = dict(DEPTHS).get(label)
            raw = (depth.get("values") or {}).get("mean")
            if thickness is None or raw is None:
                continue
            total += float(raw) * thickness
            weight += thickness
        if weight > 0:
            out[name] = (total / weight) / D_FACTOR[name]
    return out


def fetch_soil(
    latitude: float, longitude: float,
    texture_fallback: str = DEFAULT_TEXTURE,
    use_cache: bool = True,
) -> SoilProfile:
    """Resolve soil properties for a location, degrading gracefully.

    Order: cache -> exact point -> nearby offsets -> texture preset. Never raises on
    a network or coverage failure; the returned profile's ``source`` and
    ``provenance`` record which path was taken.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(latitude, longitude)

    if use_cache and cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            age = (date.today() - date.fromisoformat(raw["fetched_at"][:10])).days
            raw["source"] = f"{raw['source']} · cached {age}d"
            return SoilProfile(**raw)
        except Exception:
            pass  # corrupt cache is not worth failing over

    attempts: list[tuple[float, float, str]] = [(latitude, longitude, "SoilGrids")]
    for off in FALLBACK_OFFSETS:
        attempts.append(
            (latitude + off, longitude + off, f"SoilGrids ~{abs(off) * 111:.0f} km offset")
        )

    for lat, lon, label in attempts:
        try:
            values = _query(lat, lon)
        except Exception:
            continue
        # Texture is the minimum viable result — without it there is no pedotransfer.
        if "sand" not in values or "clay" not in values:
            continue

        sand = values["sand"]
        clay = values["clay"]
        silt = values.get("silt", max(0.0, 100.0 - sand - clay))
        prov = {k: PROV_MEASURED for k in values}

        profile = SoilProfile(
            latitude=latitude, longitude=longitude,
            sand_pct=sand, clay_pct=clay, silt_pct=silt,
            bulk_density=values.get("bdod", 1.40),
            ph=values.get("phh2o", 7.0),
            soc_g_kg=values.get("soc", 8.0),
            total_n_g_kg=values.get("nitrogen", 0.7),
            source=label,
            provenance=prov,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
        )
        try:
            cache_file.write_text(json.dumps(profile.__dict__, indent=2), encoding="utf-8")
        except Exception:
            pass
        return profile

    # Nothing available anywhere near this point.
    return from_texture(
        latitude, longitude, texture_fallback,
        note="no SoilGrids coverage here — texture preset",
    )
