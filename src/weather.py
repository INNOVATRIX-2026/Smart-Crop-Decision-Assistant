"""Live weather via the free Open-Meteo API (no API key required).

Two calls are used:
    * geocoding  — turn a place name into latitude/longitude
    * forecast   — current temperature & humidity, plus recent daily rainfall
"""

from __future__ import annotations

import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def geocode(city: str) -> dict | None:
    """Resolve a place name to coordinates, or ``None`` if not found."""
    resp = requests.get(
        GEO_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    top = results[0]
    return {
        "name": top["name"],
        "country": top.get("country", ""),
        "admin1": top.get("admin1", ""),
        "latitude": top["latitude"],
        "longitude": top["longitude"],
    }


def get_weather(latitude: float, longitude: float) -> dict:
    """Return ``{temperature (°C), humidity (%), rainfall (mm)}`` for a location.

    ``rainfall`` approximates the dataset's growing-period rainfall as the total
    precipitation over the last 30 days, which is a better proxy for growing
    conditions than a single day's reading.
    """
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m",
            "daily": "precipitation_sum",
            "past_days": 31,
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})
    precip = [p for p in daily.get("precipitation_sum", []) if p is not None]
    monthly_rain = round(sum(precip[-30:]), 1) if precip else 0.0

    return {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "rainfall": monthly_rain,
    }
