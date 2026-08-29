"""FAO-56 soil water balance and irrigation prescription.

This module answers *when to irrigate and how much*, replacing the previous
approach in ``src/advisor.py`` which derived an "irrigation shortfall" from the
gap between a field's rainfall reading and a dataset quartile. That number had no
physical meaning: it knew nothing of evapotranspiration, rooting depth, or how
much water the soil can actually hold.

Method — FAO-56 Ch. 8, root-zone depletion tracking::

    TAW   = 1000 * (θ_FC - θ_WP) * Zr        total available water        [mm]
    RAW   = p * TAW                          readily available water      [mm]
    ETc   = Kc(das) * ETo                    crop evapotranspiration      [mm/day]
    Dr[i] = Dr[i-1] - P_eff[i] - I[i] + ETc[i]   root-zone depletion      [mm]

Irrigate when ``Dr >= RAW``; net depth refills the root zone to field capacity,
i.e. ``Dr`` itself. Gross depth divides by application efficiency.

**The key idea: no soil-moisture sensor is required.** ``Dr`` is not measured, it
is *reconstructed* by replaying rainfall and evapotranspiration from the sowing
date to today using the weather archive, then projected into the forecast. That is
what makes plot-specific irrigation advice possible with nothing but a location, a
crop, and a sowing date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from ..crops import CropSpec
from .explain import Explanation

# --------------------------------------------------------------------------
# Effective rainfall
# --------------------------------------------------------------------------
# FAO-56 does not prescribe a daily effective-rainfall method (CROPWAT's USDA-SCS
# formula is monthly). The daily approximation below is deliberately simple and
# conservative: a fixed interception/evaporation loss on any rain day, plus a
# runoff fraction once daily rain is heavy enough to exceed infiltration.
# Marked [REVIEW] — worth replacing with a curve-number method if plot slope and
# soil hydrologic group ever become available.
INTERCEPTION_MM = 2.0
HEAVY_RAIN_MM = 25.0
RUNOFF_FRACTION_HEAVY = 0.15

# How far ahead to look for rain that would make an irrigation unnecessary.
RAIN_LOOKAHEAD_DAYS = 3

# Depletion fraction above which we call the crop stressed (Dr / TAW).
SEVERE_STRESS_FRACTION = 0.90

# Net depth beyond which a single application stops being practical: water runs
# off or drains below the root zone rather than being stored. Typical wheat
# irrigations are 50-75 mm net. Used only to raise a warning, never to silently
# cap the prescription — under-reporting a real deficit would be worse. [REVIEW]
PRACTICAL_MAX_APPLICATION_MM = 75.0


def effective_rainfall(rain_mm: float) -> float:
    """Fraction of gross rainfall that reaches and stays in the root zone."""
    if rain_mm <= INTERCEPTION_MM:
        return 0.0
    net = rain_mm - INTERCEPTION_MM
    if rain_mm > HEAVY_RAIN_MM:
        net *= 1.0 - RUNOFF_FRACTION_HEAVY
    return max(0.0, net)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SoilWater:
    """Soil water-holding properties for the balance.

    ``theta_fc`` and ``theta_wp`` are volumetric water contents (m³/m³) at field
    capacity and permanent wilting point. Both come from pedotransfer functions
    over SoilGrids texture — see ``src.soil``.
    """

    theta_fc: float
    theta_wp: float

    def __post_init__(self) -> None:
        if not 0.0 < self.theta_wp < self.theta_fc < 1.0:
            raise ValueError(
                f"implausible soil water limits: theta_wp={self.theta_wp}, "
                f"theta_fc={self.theta_fc} (need 0 < wp < fc < 1)"
            )

    def taw_mm(self, root_depth_m: float) -> float:
        """Total available water in the root zone, mm."""
        return 1000.0 * (self.theta_fc - self.theta_wp) * root_depth_m


@dataclass(frozen=True)
class WeatherDay:
    """One day of weather, from either the archive or the forecast."""

    day: date
    eto_mm: float          # FAO-56 reference evapotranspiration
    rain_mm: float
    tmax_c: float | None = None
    tmin_c: float | None = None
    is_forecast: bool = False


@dataclass(frozen=True)
class BalanceDay:
    """Computed water-balance state for one day."""

    day: date
    das: int               # days after sowing
    stage: str
    kc: float
    root_depth_m: float
    eto_mm: float
    etc_mm: float
    rain_mm: float
    rain_eff_mm: float
    irrigation_mm: float
    taw_mm: float
    raw_mm: float
    depletion_mm: float
    is_forecast: bool

    @property
    def depletion_fraction(self) -> float:
        return self.depletion_mm / self.taw_mm if self.taw_mm > 0 else 0.0

    @property
    def available_fraction(self) -> float:
        """Fraction of available water remaining — what a farmer intuits as 'moisture'."""
        return max(0.0, 1.0 - self.depletion_fraction)

    @property
    def needs_water(self) -> bool:
        return self.depletion_mm >= self.raw_mm

    @property
    def severely_stressed(self) -> bool:
        return self.depletion_fraction >= SEVERE_STRESS_FRACTION


# --------------------------------------------------------------------------
# The balance
# --------------------------------------------------------------------------
def run_balance(
    crop: CropSpec,
    soil: SoilWater,
    weather: list[WeatherDay],
    sowing_date: date,
    irrigations: dict[date, float] | None = None,
    initial_depletion_mm: float = 0.0,
) -> list[BalanceDay]:
    """Replay the root-zone water balance day by day.

    ``weather`` should run from the sowing date through the end of the forecast,
    sorted ascending. ``irrigations`` maps a date to the *net* depth applied (mm),
    letting the farmer record what they have already done.

    ``initial_depletion_mm`` defaults to 0 — the root zone at field capacity at
    sowing, which is the usual state after pre-sowing irrigation or monsoon onset.
    Expose it as an override rather than an assumption where it matters.
    """
    irrigations = irrigations or {}
    out: list[BalanceDay] = []
    depletion = float(initial_depletion_mm)

    for wd in sorted(weather, key=lambda d: d.day):
        das = (wd.day - sowing_date).days
        if das < 0:
            continue  # pre-sowing days carry no crop water demand

        stage = crop.water.stage_at(das)
        kc = crop.water.kc_at(das)
        root_depth = crop.water.root_depth_at(das)
        taw = soil.taw_mm(root_depth)
        raw = crop.water.depletion_fraction_p * taw

        etc = kc * wd.eto_mm
        rain_eff = effective_rainfall(wd.rain_mm)
        applied = float(irrigations.get(wd.day, 0.0))

        depletion = depletion - rain_eff - applied + etc
        # Depletion cannot go below zero (excess drains) nor exceed TAW (no water
        # left to remove). Clamping here keeps the state physically meaningful.
        depletion = min(max(depletion, 0.0), taw)

        out.append(
            BalanceDay(
                day=wd.day,
                das=das,
                stage=stage,
                kc=kc,
                root_depth_m=root_depth,
                eto_mm=wd.eto_mm,
                etc_mm=etc,
                rain_mm=wd.rain_mm,
                rain_eff_mm=rain_eff,
                irrigation_mm=applied,
                taw_mm=taw,
                raw_mm=raw,
                depletion_mm=depletion,
                is_forecast=wd.is_forecast,
            )
        )

    return out


# --------------------------------------------------------------------------
# The prescription
# --------------------------------------------------------------------------
ACTION_IRRIGATE_NOW = "irrigate_now"
ACTION_IRRIGATE_SCHEDULED = "irrigate_scheduled"
ACTION_RAIN_EXPECTED = "rain_expected"
ACTION_NO_ACTION = "no_action"
ACTION_DRAINAGE = "drainage_needed"


@dataclass(frozen=True)
class IrrigationPlan:
    """A concrete irrigation prescription: what, how much, and when."""

    action: str
    urgency: str                    # critical | warning | good | info
    day: date | None
    days_from_today: int | None
    net_depth_mm: float
    gross_depth_mm: float
    method: str
    explanation: Explanation
    current: BalanceDay | None
    balance: tuple[BalanceDay, ...]
    warnings: tuple[str, ...] = ()

    @property
    def litres_per_ha(self) -> float:
        """1 mm over 1 ha = 10 000 L."""
        return self.gross_depth_mm * 10_000.0

    @property
    def cubic_metres_per_ha(self) -> float:
        return self.gross_depth_mm * 10.0


def recommend_irrigation(
    crop: CropSpec,
    soil: SoilWater,
    weather: list[WeatherDay],
    sowing_date: date,
    today: date,
    method: str = "surface",
    irrigations: dict[date, float] | None = None,
    initial_depletion_mm: float = 0.0,
) -> IrrigationPlan:
    """Decide whether, when, and how much to irrigate.

    ``today`` is passed in rather than read from the clock so the engine stays
    deterministic and testable.
    """
    balance = run_balance(
        crop, soil, weather, sowing_date,
        irrigations=irrigations, initial_depletion_mm=initial_depletion_mm,
    )

    efficiency = crop.water.application_efficiency.get(method)
    if efficiency is None:
        raise ValueError(
            f"unknown irrigation method {method!r}; "
            f"available: {sorted(crop.water.application_efficiency)}"
        )

    exp = Explanation().cite("FAO-56 Ch. 8 (root-zone depletion)")

    if not balance:
        return IrrigationPlan(
            action=ACTION_NO_ACTION, urgency="info", day=None, days_from_today=None,
            net_depth_mm=0.0, gross_depth_mm=0.0, method=method,
            explanation=exp.conclude(
                "No weather data covering the period since sowing, so soil moisture "
                "cannot be reconstructed."
            ),
            current=None, balance=(),
        )

    # State as of today (or the closest earlier day we have).
    past = [b for b in balance if b.day <= today]
    current = past[-1] if past else balance[0]

    exp.add_input("Crop", crop.label()) \
       .add_input("Growth stage", current.stage) \
       .add_input("Days after sowing", float(current.das), "days") \
       .add_input("Rooting depth", current.root_depth_m, "m") \
       .add_input("Field capacity", soil.theta_fc, "m³/m³") \
       .add_input("Wilting point", soil.theta_wp, "m³/m³")

    exp.add_step(
        "Total available water (TAW)",
        f"({soil.theta_fc:.3f} − {soil.theta_wp:.3f}) × 1000 × {current.root_depth_m:.2f} m",
        current.taw_mm, "mm",
    ).add_step(
        "Readily available water (RAW)",
        f"p {crop.water.depletion_fraction_p:.2f} × TAW {current.taw_mm:.0f} mm",
        current.raw_mm, "mm",
    ).add_step(
        "Current root-zone depletion",
        f"replayed from sowing ({current.das} days of rain and ETc)",
        current.depletion_mm, "mm",
    ).add_step(
        "Soil water remaining",
        f"1 − {current.depletion_mm:.0f} / {current.taw_mm:.0f}",
        current.available_fraction * 100.0, "% of available",
    ).set_threshold(
        f"Irrigate when depletion reaches RAW = {crop.water.depletion_fraction_p:.0%} "
        f"of TAW — {current.raw_mm:.0f} mm today, and rising as roots deepen"
    )

    # --- waterlogging risk -------------------------------------------------
    # Assessed as data, not as an early return. Saturation risk and an irrigation
    # deficit are not competing answers: when rain is about to cover the deficit,
    # "don't irrigate, rain is coming" is the headline a farmer needs and drainage
    # is a caveat on it. Drainage only becomes the primary action when there is no
    # water deficit to report.
    tolerance = int(crop.limits.get("waterlogging_tolerance_days", 0) or 0)
    saturated_ahead = [
        b for b in balance
        if b.day >= today and b.depletion_mm <= 0.0 and b.rain_eff_mm > HEAVY_RAIN_MM
    ]
    drainage_warning: str | None = None
    if saturated_ahead:
        first_sat = saturated_ahead[0]
        drainage_warning = (
            f"Waterlogging risk: {first_sat.rain_mm:.0f} mm forecast on "
            f"{first_sat.day:%d %b} onto an already-full root zone. "
            f"{crop.label()} tolerates roughly {tolerance or 'a few'} day(s) of "
            f"saturation — check that the field drains."
        )

    # --- find the first day the crop needs water --------------------------
    ahead = [b for b in balance if b.day >= today]
    trigger = next((b for b in ahead if b.needs_water), None)

    if trigger is None:
        # No deficit. Drainage, if any, is now the only thing worth saying.
        if saturated_ahead:
            first_sat = saturated_ahead[0]
            exp.add_step(
                "Heavy rain onto an already-full root zone",
                f"{first_sat.rain_mm:.0f} mm forecast on {first_sat.day:%d %b} "
                f"with depletion at 0 mm",
                first_sat.rain_mm, "mm",
            ).conclude(
                f"Do not irrigate. Ensure the field drains before "
                f"{first_sat.day:%d %b} — {crop.label()} tolerates roughly "
                f"{tolerance or 'a few'} day(s) of saturation."
            )
            return IrrigationPlan(
                action=ACTION_DRAINAGE, urgency="warning", day=first_sat.day,
                days_from_today=(first_sat.day - today).days,
                net_depth_mm=0.0, gross_depth_mm=0.0, method=method,
                explanation=exp, current=current, balance=tuple(balance),
            )

        last = ahead[-1] if ahead else current
        exp.conclude(
            f"No irrigation needed within the forecast window. Depletion peaks near "
            f"{max((b.depletion_mm for b in ahead), default=current.depletion_mm):.0f} mm, "
            f"below the {current.raw_mm:.0f} mm trigger. Re-check after {last.day:%d %b}."
        )
        return IrrigationPlan(
            action=ACTION_NO_ACTION, urgency="good", day=None, days_from_today=None,
            net_depth_mm=0.0, gross_depth_mm=0.0, method=method,
            explanation=exp, current=current, balance=tuple(balance),
        )

    # --- would forecast rain make it unnecessary? -------------------------
    window_end = trigger.day + timedelta(days=RAIN_LOOKAHEAD_DAYS)
    window = [b for b in balance if trigger.day <= b.day <= window_end]
    rain_ahead = sum(b.rain_eff_mm for b in window)
    if rain_ahead >= trigger.depletion_mm and rain_ahead > 0:
        exp.add_step(
            "Effective rain forecast within "
            f"{RAIN_LOOKAHEAD_DAYS} days of the trigger",
            " + ".join(f"{b.rain_eff_mm:.0f}" for b in window if b.rain_eff_mm > 0),
            rain_ahead, "mm",
        ).conclude(
            f"Hold off. Depletion reaches the {trigger.raw_mm:.0f} mm trigger on "
            f"{trigger.day:%d %b}, but ~{rain_ahead:.0f} mm of effective rain is "
            f"forecast within {RAIN_LOOKAHEAD_DAYS} days — enough to cover the "
            f"{trigger.depletion_mm:.0f} mm deficit. Re-check if the rain misses."
        )
        return IrrigationPlan(
            action=ACTION_RAIN_EXPECTED, urgency="info", day=trigger.day,
            days_from_today=(trigger.day - today).days,
            net_depth_mm=0.0, gross_depth_mm=0.0, method=method,
            explanation=exp, current=current, balance=tuple(balance),
            warnings=(drainage_warning,) if drainage_warning else (),
        )

    # --- prescribe -------------------------------------------------------
    # Net depth refills the root zone to field capacity. It can never exceed TAW,
    # since depletion is clamped there.
    net = min(trigger.depletion_mm, trigger.taw_mm)
    gross = net / efficiency
    days_out = (trigger.day - today).days
    immediate = days_out <= 1

    critical = trigger.stage in crop.water.critical_stages or trigger.severely_stressed
    urgency = "critical" if (immediate and critical) else ("warning" if immediate else "info")

    exp.add_step(
        f"Trigger reached on {trigger.day:%d %b}",
        f"depletion {trigger.depletion_mm:.0f} mm ≥ RAW {trigger.raw_mm:.0f} mm "
        f"(root zone {trigger.root_depth_m:.2f} m by then)",
        trigger.depletion_mm, "mm",
    ).add_step(
        "Net depth to refill to field capacity",
        f"equals the depletion on {trigger.day:%d %b}",
        net, "mm",
    ).add_step(
        f"Gross depth at {method} efficiency {efficiency:.0%}",
        f"{net:.0f} mm ÷ {efficiency:.2f}",
        gross, "mm",
    ).add_step(
        "Volume per hectare",
        f"{gross:.0f} mm × 10 m³ per mm",
        gross * 10.0, "m³/ha",
    )

    # A single application far above normal practice usually means the depletion
    # history is wrong — most often because earlier irrigations were never
    # recorded, so the balance has been accumulating deficit since sowing. Surface
    # that rather than quietly prescribing a physically awkward flood.
    warnings: list[str] = []
    if drainage_warning:
        warnings.append(drainage_warning)
    if net > PRACTICAL_MAX_APPLICATION_MM:
        warnings.append(
            f"{net:.0f} mm in one application exceeds normal practice "
            f"(~{PRACTICAL_MAX_APPLICATION_MM:.0f} mm net). Much of it would run off or "
            f"drain past the roots. If irrigations have already been applied since "
            f"sowing, record them — otherwise split this across two applications "
            f"2-3 days apart."
        )

    when = "today" if days_out <= 0 else (
        "tomorrow" if days_out == 1 else f"in {days_out} days ({trigger.day:%d %b})"
    )
    note = ""
    if trigger.stage in crop.water.critical_stages:
        note = (
            f" This falls in the {trigger.stage} stage, which is one of "
            f"{crop.label()}'s moisture-critical periods — delaying costs yield."
        )
    exp.conclude(
        f"Apply {gross:.0f} mm ({gross * 10:.0f} m³/ha) {when}. "
        f"Root-zone depletion reaches {trigger.depletion_mm:.0f} mm against a "
        f"{trigger.raw_mm:.0f} mm trigger, and forecast rain does not cover it.{note}"
    )

    return IrrigationPlan(
        action=ACTION_IRRIGATE_NOW if immediate else ACTION_IRRIGATE_SCHEDULED,
        urgency=urgency,
        day=trigger.day,
        days_from_today=days_out,
        net_depth_mm=net,
        gross_depth_mm=gross,
        method=method,
        explanation=exp,
        current=current,
        balance=tuple(balance),
        warnings=tuple(warnings),
    )
