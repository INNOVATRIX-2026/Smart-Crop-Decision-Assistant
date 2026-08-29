"""Shared fixtures. Nothing here touches the network or the clock."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.crops import get_crop
from src.engine.nutrients import SoilNutrients
from src.engine.water import SoilWater, WeatherDay

SOWING = date(2025, 11, 15)          # typical rabi wheat sowing, North India
TODAY = SOWING + timedelta(days=40)  # mid-tillering


@pytest.fixture
def wheat():
    return get_crop("wheat")


@pytest.fixture
def loam():
    """A medium loam: 0.15 m³/m³ of available water per metre of depth."""
    return SoilWater(theta_fc=0.30, theta_wp=0.15)


def make_weather(
    start: date,
    days: int,
    eto_mm: float = 4.0,
    rain_mm: float = 0.0,
    rain_on: dict[int, float] | None = None,
    forecast_from: int | None = None,
) -> list[WeatherDay]:
    """Build a synthetic weather series.

    ``rain_on`` maps a day offset to a rainfall total, overriding ``rain_mm``.
    Days at or after ``forecast_from`` are flagged as forecast rather than archive.
    """
    rain_on = rain_on or {}
    out = []
    for i in range(days):
        out.append(
            WeatherDay(
                day=start + timedelta(days=i),
                eto_mm=eto_mm,
                rain_mm=rain_on.get(i, rain_mm),
                is_forecast=forecast_from is not None and i >= forecast_from,
            )
        )
    return out


@pytest.fixture
def dry_weather():
    """60 days from sowing, no rain at all — forces a depletion trigger."""
    return make_weather(SOWING, 60, eto_mm=4.0, rain_mm=0.0, forecast_from=40)


@pytest.fixture
def medium_soil_nutrients():
    """Soil testing 'medium' for all three nutrients per Soil Health Card bands."""
    return SoilNutrients(
        available={"N": 350.0, "P2O5": 15.0, "K2O": 150.0},
        provenance={"N": "estimated", "P2O5": "measured", "K2O": "measured"},
    )
