"""FAO-56 water balance and irrigation prescription.

These are the physical invariants from the plan. They exist to catch the class of
bug that motivated this rewrite: a number that looks plausible on screen but is
not physically meaningful.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from src.engine.water import (
    ACTION_DRAINAGE,
    ACTION_IRRIGATE_NOW,
    ACTION_IRRIGATE_SCHEDULED,
    ACTION_NO_ACTION,
    ACTION_RAIN_EXPECTED,
    HEAVY_RAIN_MM,
    INTERCEPTION_MM,
    PRACTICAL_MAX_APPLICATION_MM,
    RUNOFF_FRACTION_HEAVY,
    SoilWater,
    effective_rainfall,
    recommend_irrigation,
    run_balance,
)

from .conftest import SOWING, TODAY, make_weather

IRRIGATE = (ACTION_IRRIGATE_NOW, ACTION_IRRIGATE_SCHEDULED)


# --------------------------------------------------------------------------
# Soil water limits
# --------------------------------------------------------------------------
def test_taw_formula():
    """TAW = (θ_FC − θ_WP) × 1000 × Zr — FAO-56 Eq. 82."""
    soil = SoilWater(theta_fc=0.30, theta_wp=0.15)
    assert soil.taw_mm(1.0) == pytest.approx(150.0)
    assert soil.taw_mm(0.5) == pytest.approx(75.0)
    assert soil.taw_mm(0.0) == pytest.approx(0.0)


def test_inverted_soil_limits_are_rejected():
    """Wilting point above field capacity is physically impossible."""
    with pytest.raises(ValueError, match="implausible"):
        SoilWater(theta_fc=0.15, theta_wp=0.30)


def test_out_of_range_soil_limits_are_rejected():
    with pytest.raises(ValueError):
        SoilWater(theta_fc=1.4, theta_wp=0.2)


# --------------------------------------------------------------------------
# Effective rainfall
# --------------------------------------------------------------------------
def test_light_rain_is_fully_intercepted():
    assert effective_rainfall(0.0) == 0.0
    assert effective_rainfall(INTERCEPTION_MM) == 0.0


def test_moderate_rain_loses_only_interception():
    assert effective_rainfall(10.0) == pytest.approx(10.0 - INTERCEPTION_MM)


def test_heavy_rain_also_loses_runoff():
    gross = HEAVY_RAIN_MM + 20.0
    expected = (gross - INTERCEPTION_MM) * (1.0 - RUNOFF_FRACTION_HEAVY)
    assert effective_rainfall(gross) == pytest.approx(expected)


def test_effective_rainfall_never_exceeds_gross():
    for mm in (0, 1, 5, 25, 50, 200):
        assert effective_rainfall(float(mm)) <= mm


# --------------------------------------------------------------------------
# The balance
# --------------------------------------------------------------------------
def test_depletion_stays_within_physical_bounds(wheat, loam, dry_weather):
    """0 ≤ Dr ≤ TAW always — you cannot remove water that is not there."""
    for b in run_balance(wheat, loam, dry_weather, SOWING):
        assert 0.0 <= b.depletion_mm <= b.taw_mm + 1e-9


def test_depletion_rises_monotonically_without_rain(wheat, loam, dry_weather):
    balance = run_balance(wheat, loam, dry_weather, SOWING)
    depletions = [b.depletion_mm for b in balance]
    assert all(b >= a - 1e-9 for a, b in zip(depletions, depletions[1:]))
    assert depletions[-1] > depletions[0]


def test_rain_reduces_depletion(wheat, loam):
    dry = run_balance(wheat, loam, make_weather(SOWING, 30, rain_mm=0.0), SOWING)
    wet = run_balance(wheat, loam, make_weather(SOWING, 30, rain_mm=5.0), SOWING)
    assert wet[-1].depletion_mm < dry[-1].depletion_mm


def test_recorded_irrigation_reduces_depletion(wheat, loam, dry_weather):
    day = SOWING + timedelta(days=20)
    without = run_balance(wheat, loam, dry_weather, SOWING)
    with_irrig = run_balance(wheat, loam, dry_weather, SOWING, irrigations={day: 40.0})
    assert with_irrig[-1].depletion_mm < without[-1].depletion_mm


def test_etc_is_kc_times_eto(wheat, loam):
    balance = run_balance(wheat, loam, make_weather(SOWING, 30, eto_mm=5.0), SOWING)
    for b in balance:
        assert b.etc_mm == pytest.approx(b.kc * b.eto_mm)


def test_pre_sowing_days_are_skipped(wheat, loam):
    weather = make_weather(SOWING - timedelta(days=10), 30)
    balance = run_balance(wheat, loam, weather, SOWING)
    assert all(b.das >= 0 for b in balance)
    assert balance[0].day == SOWING


def test_balance_is_order_independent(wheat, loam, dry_weather):
    """Input order must not matter — the balance sorts by date itself."""
    forward = run_balance(wheat, loam, dry_weather, SOWING)
    shuffled = run_balance(wheat, loam, list(reversed(dry_weather)), SOWING)
    assert [b.depletion_mm for b in forward] == pytest.approx(
        [b.depletion_mm for b in shuffled]
    )


def test_available_fraction_complements_depletion(wheat, loam, dry_weather):
    for b in run_balance(wheat, loam, dry_weather, SOWING):
        assert b.available_fraction == pytest.approx(1.0 - b.depletion_fraction)
        assert 0.0 <= b.available_fraction <= 1.0


# --------------------------------------------------------------------------
# The prescription
# --------------------------------------------------------------------------
def test_dry_season_triggers_irrigation(wheat, loam, dry_weather):
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    assert plan.action in IRRIGATE
    assert plan.net_depth_mm > 0
    assert plan.day is not None


def test_irrigation_depth_never_exceeds_available_water(wheat, loam, dry_weather):
    """The invariant named in the plan: you cannot refill more than TAW."""
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    trigger = next(b for b in plan.balance if b.day == plan.day)
    assert plan.net_depth_mm <= trigger.taw_mm + 1e-9


def test_gross_depth_exceeds_net_by_application_efficiency(wheat, loam, dry_weather):
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY, method="surface")
    eff = wheat.water.application_efficiency["surface"]
    assert plan.gross_depth_mm == pytest.approx(plan.net_depth_mm / eff)
    assert plan.gross_depth_mm > plan.net_depth_mm


def test_drip_needs_less_water_than_flood(wheat, loam, dry_weather):
    flood = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY, method="surface")
    drip = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY, method="drip")
    assert drip.gross_depth_mm < flood.gross_depth_mm
    assert drip.net_depth_mm == pytest.approx(flood.net_depth_mm)


def test_unknown_irrigation_method_is_rejected(wheat, loam, dry_weather):
    with pytest.raises(ValueError, match="unknown irrigation method"):
        recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY, method="telepathy")


def test_volume_conversions(wheat, loam, dry_weather):
    """1 mm over 1 ha = 10 m³ = 10 000 L."""
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    assert plan.litres_per_ha == pytest.approx(plan.gross_depth_mm * 10_000)
    assert plan.cubic_metres_per_ha == pytest.approx(plan.gross_depth_mm * 10)


def test_ample_rain_means_no_irrigation(wheat, loam):
    """Daily rain above peak ETc should keep the crop out of deficit entirely."""
    weather = make_weather(SOWING, 60, eto_mm=4.0, rain_mm=8.0, forecast_from=40)
    plan = recommend_irrigation(wheat, loam, weather, SOWING, TODAY)
    assert plan.action == ACTION_NO_ACTION
    assert plan.net_depth_mm == 0.0


def test_forecast_rain_suppresses_a_triggered_irrigation(wheat, loam, dry_weather):
    """The invariant from the plan, and the advice farmers most want.

    Compare two runs differing only by a large rain event on the trigger day. The
    invariant is simply that no water is prescribed — which of the non-irrigating
    actions results depends on how much rain and when, so asserting a specific one
    would over-specify.
    """
    dry_plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    assert dry_plan.action in IRRIGATE  # precondition for this test to mean anything

    trigger_offset = (dry_plan.day - SOWING).days
    # Enough gross rain to clear the deficit after interception and runoff losses.
    needed = dry_plan.net_depth_mm / (1.0 - RUNOFF_FRACTION_HEAVY) + INTERCEPTION_MM + 10
    wet = make_weather(
        SOWING, 60, eto_mm=4.0, rain_on={trigger_offset: needed}, forecast_from=40
    )
    wet_plan = recommend_irrigation(wheat, loam, wet, SOWING, TODAY)

    assert wet_plan.action not in IRRIGATE
    assert wet_plan.net_depth_mm == 0.0
    assert wet_plan.gross_depth_mm == 0.0


def test_rain_shortly_after_the_trigger_defers_rather_than_irrigates(wheat, loam, dry_weather):
    """Rain landing a couple of days *after* the deficit hits the trigger.

    This is the realistic case that exercises the rain-expected path: depletion
    does cross RAW, but rain arrives inside the lookahead window and covers it.
    """
    dry_plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    assert dry_plan.action in IRRIGATE

    trigger_offset = (dry_plan.day - SOWING).days
    needed = dry_plan.net_depth_mm / (1.0 - RUNOFF_FRACTION_HEAVY) + INTERCEPTION_MM + 10
    wet = make_weather(
        SOWING, 60, eto_mm=4.0,
        rain_on={trigger_offset + 2: needed}, forecast_from=40,
    )
    plan = recommend_irrigation(wheat, loam, wet, SOWING, TODAY)

    assert plan.action == ACTION_RAIN_EXPECTED
    assert plan.net_depth_mm == 0.0
    assert "rain" in plan.explanation.conclusion.lower()


def test_saturation_is_a_warning_not_a_competing_action(wheat, loam, dry_weather):
    """Waterlogging must not suppress the more useful 'rain is coming' headline.

    Guards the ordering bug where the drainage check ran first and stole every
    case where heavy rain relieved a deficit.
    """
    dry_plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    trigger_offset = (dry_plan.day - SOWING).days
    needed = dry_plan.net_depth_mm / (1.0 - RUNOFF_FRACTION_HEAVY) + INTERCEPTION_MM + 10
    wet = make_weather(
        SOWING, 60, eto_mm=4.0,
        rain_on={trigger_offset + 2: needed}, forecast_from=40,
    )
    plan = recommend_irrigation(wheat, loam, wet, SOWING, TODAY)

    assert plan.action == ACTION_RAIN_EXPECTED
    assert any("aterlogging" in w for w in plan.warnings), (
        "saturation risk should still be reported, as a warning"
    )


def test_drainage_becomes_primary_when_there_is_no_deficit(wheat, loam):
    """With no water deficit to report, saturation risk is the headline."""
    weather = make_weather(
        SOWING, 60, eto_mm=4.0, rain_mm=6.0, rain_on={45: 120.0}, forecast_from=40
    )
    plan = recommend_irrigation(wheat, loam, weather, SOWING, TODAY)
    assert plan.action == ACTION_DRAINAGE
    assert plan.net_depth_mm == 0.0


def test_excessive_single_application_is_flagged(wheat, loam, dry_weather):
    """A 130 mm flood usually means unrecorded prior irrigation — say so."""
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    if plan.net_depth_mm > PRACTICAL_MAX_APPLICATION_MM:
        assert any("exceeds normal practice" in w for w in plan.warnings)


def test_recording_past_irrigation_delays_the_next_one(wheat, loam, dry_weather):
    """Logging prior irrigation pushes the next application later.

    Note it does *not* make the next application smaller: by the later date the
    root zone is deeper, so TAW and RAW are both larger and more water is needed
    to refill. Delay is the invariant; depth is not.
    """
    naive = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    informed = recommend_irrigation(
        wheat, loam, dry_weather, SOWING, TODAY,
        irrigations={SOWING + timedelta(days=21): 50.0},
    )
    assert informed.day > naive.day


def test_empty_weather_degrades_gracefully(wheat, loam):
    """No data must produce a clear no-action, not a crash or a fake number."""
    plan = recommend_irrigation(wheat, loam, [], SOWING, TODAY)
    assert plan.action == ACTION_NO_ACTION
    assert plan.net_depth_mm == 0.0
    assert plan.current is None
    assert "cannot be reconstructed" in plan.explanation.conclusion


def test_recommendation_is_deterministic(wheat, loam, dry_weather):
    """Same inputs, identical output — proves no hidden clock or I/O."""
    a = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    b = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    assert (a.action, a.day, a.net_depth_mm) == (b.action, b.day, b.net_depth_mm)
    assert a.explanation.conclusion == b.explanation.conclusion


# --------------------------------------------------------------------------
# Explanation trace
# --------------------------------------------------------------------------
def test_explanation_shows_its_working(wheat, loam, dry_weather):
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    exp = plan.explanation
    assert exp.inputs, "no inputs recorded"
    assert exp.steps, "no computation steps recorded"
    assert exp.conclusion
    assert exp.sources, "no source citation"

    labels = " ".join(s.label for s in exp.steps)
    for expected in ("Total available water", "Readily available water", "depletion"):
        assert expected.lower() in labels.lower()


def test_explanation_reports_depth_matching_the_plan(wheat, loam, dry_weather):
    """The trace must describe the number actually prescribed."""
    plan = recommend_irrigation(wheat, loam, dry_weather, SOWING, TODAY)
    net_steps = [s for s in plan.explanation.steps if "Net depth" in s.label]
    assert net_steps
    assert net_steps[0].value == pytest.approx(plan.net_depth_mm)
