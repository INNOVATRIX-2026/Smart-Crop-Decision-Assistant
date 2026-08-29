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


# ==========================================================================
# Daily series for the water balance
# ==========================================================================
# The engine needs a dated series of reference evapotranspiration and rainfall
# spanning sowing date -> today -> forecast horizon, not the single scalar above.
#
# **Verified live (Aug 2026):** `et0_fao_evapotranspiration` is available as a daily
# field on BOTH the forecast and archive endpoints, in mm. That means FAO-56
# reference ET comes straight from the API and we never implement Penman-Monteith.
#
# The forecast endpoint serves up to 92 `past_days`, so one call usually covers the
# whole season to date plus 16 days ahead. Only sowing dates older than that need
# the separate archive endpoint, which lags real time by a few days.

from datetime import date, datetime, timedelta  # noqa: E402  (grouped with the section)
from pathlib import Path  # noqa: E402
import json  # noqa: E402

from . import config  # noqa: E402
from .engine.water import WeatherDay  # noqa: E402

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_FIELDS = "et0_fao_evapotranspiration,precipitation_sum,temperature_2m_max,temperature_2m_min"
MAX_PAST_DAYS = 92          # forecast endpoint limit
FORECAST_DAYS = 16
CACHE_DIR = config.DATA_DIR / "cache"

SOURCE_LIVE = "LIVE"
SOURCE_CACHED = "CACHED"
SOURCE_NONE = "UNAVAILABLE"


def _rows_to_days(daily: dict, today: date) -> list[WeatherDay]:
    """Turn an Open-Meteo `daily` block into WeatherDay objects, skipping nulls."""
    out: list[WeatherDay] = []
    times = daily.get("time") or []
    eto = daily.get("et0_fao_evapotranspiration") or []
    rain = daily.get("precipitation_sum") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []

    for i, stamp in enumerate(times):
        day = date.fromisoformat(stamp)
        e = eto[i] if i < len(eto) else None
        r = rain[i] if i < len(rain) else None
        if e is None:
            # ETo drives the whole balance; a null day cannot be silently treated
            # as zero demand, so it is skipped and the gap is visible in coverage.
            continue
        out.append(
            WeatherDay(
                day=day,
                eto_mm=float(e),
                rain_mm=float(r or 0.0),
                tmax_c=tmax[i] if i < len(tmax) and tmax[i] is not None else None,
                tmin_c=tmin[i] if i < len(tmin) and tmin[i] is not None else None,
                is_forecast=day > today,
            )
        )
    return out


def _fetch_forecast(latitude: float, longitude: float, past_days: int) -> dict:
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": latitude, "longitude": longitude,
            "daily": DAILY_FIELDS,
            "past_days": max(0, min(past_days, MAX_PAST_DAYS)),
            "forecast_days": FORECAST_DAYS,
            "timezone": "auto",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("daily", {})


def _fetch_archive(latitude: float, longitude: float, start: date, end: date) -> dict:
    resp = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": DAILY_FIELDS, "timezone": "auto",
        },
        timeout=45,
    )
    resp.raise_for_status()
    return resp.json().get("daily", {})


def _cache_path(latitude: float, longitude: float, sowing: date) -> Path:
    return CACHE_DIR / f"wx_{latitude:.2f}_{longitude:.2f}_{sowing.isoformat()}.json"


def get_daily_series(
    latitude: float,
    longitude: float,
    sowing_date: date,
    today: date | None = None,
    use_cache: bool = True,
) -> tuple[list[WeatherDay], str]:
    """Daily ETo and rainfall from sowing date through the forecast horizon.

    Returns ``(days, provenance)`` where provenance is ``LIVE``, ``CACHED · Nd``, or
    ``UNAVAILABLE``. Never raises on a network failure — a dead connection falls back
    to the most recent cached snapshot so a demo degrades instead of collapsing.
    """
    today = today or date.today()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(latitude, longitude, sowing_date)
    days_back = (today - sowing_date).days

    try:
        daily = _fetch_forecast(latitude, longitude, days_back)
        days = _rows_to_days(daily, today)

        # Sowing older than the forecast endpoint's window: top up from the archive.
        if days_back > MAX_PAST_DAYS:
            archive_end = today - timedelta(days=MAX_PAST_DAYS + 1)
            try:
                older = _rows_to_days(
                    _fetch_archive(latitude, longitude, sowing_date, archive_end), today
                )
                days = older + days
            except Exception:
                pass  # partial coverage is still usable; the balance starts later

        days = [d for d in days if d.day >= sowing_date]
        if days:
            try:
                cache_file.write_text(
                    json.dumps({
                        "fetched_at": datetime.now().isoformat(timespec="seconds"),
                        "days": [
                            {"day": d.day.isoformat(), "eto_mm": d.eto_mm,
                             "rain_mm": d.rain_mm, "tmax_c": d.tmax_c,
                             "tmin_c": d.tmin_c, "is_forecast": d.is_forecast}
                            for d in days
                        ],
                    }, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return days, SOURCE_LIVE
    except Exception:
        pass

    if use_cache and cache_file.exists():
        try:
            raw = json.loads(cache_file.read_text(encoding="utf-8"))
            age = (today - date.fromisoformat(raw["fetched_at"][:10])).days
            days = [
                WeatherDay(
                    day=date.fromisoformat(r["day"]), eto_mm=r["eto_mm"],
                    rain_mm=r["rain_mm"], tmax_c=r.get("tmax_c"),
                    tmin_c=r.get("tmin_c"), is_forecast=r.get("is_forecast", False),
                )
                for r in raw["days"]
            ]
            return days, f"{SOURCE_CACHED} · {age}d old"
        except Exception:
            pass

    return [], SOURCE_NONE


def rain_forecast_map(days: list[WeatherDay]) -> dict[date, float]:
    """Date -> forecast rainfall, for the fertiliser leaching check."""
    return {d.day: d.rain_mm for d in days}
