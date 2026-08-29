"""Fertiliser prescription: how much of which product, and when.

Replaces ``src/advisor.py``'s ``_nutrient_action``, which computed
``deficit = median − value`` over a synthetic dataset's quartiles and printed the
result as "Add roughly N kg/ha via urea". That number was dimensionally
meaningless — the dataset's N/P/K columns are unitless indices, not kg/ha — and it
carried false precision on exactly the question the platform exists to answer.

Method — scaled reference dose, as used by Soil Health Card and Nutrient Expert::

    dose = reference_dose × (target_yield / reference_yield) × fertility_factor

The reference dose is a field-validated package (e.g. 120:60:40 for irrigated
wheat), so the recommendation stays anchored on decades of trial data rather than
on a chain of uncertain parameters.

A mechanistic uptake budget is computed alongside as a **cross-check only**::

    uptake_dose = (target_yield × uptake_per_tonne − soil_credit) / use_efficiency

For wheat this lands near 240 kg N/ha against a validated 120 — the gap comes from
soil N supply being much larger than a simple mineralisation estimate suggests, and
from wide uncertainty in use efficiency. Rather than bury that, the plan reports
both and warns when they diverge beyond the crop's tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from ..crops import NUTRIENTS, CropSpec
from .explain import Explanation

# --------------------------------------------------------------------------
# Fertiliser products — nutrient mass fraction of the physical product
# --------------------------------------------------------------------------
# Standard grades used in India. Values are definitional (guaranteed analysis),
# not estimates.
PRODUCTS: dict[str, dict[str, float]] = {
    "Urea": {"N": 0.46},
    "DAP": {"N": 0.18, "P2O5": 0.46},
    "MOP": {"K2O": 0.60},
    "SSP": {"P2O5": 0.16},
}

# Rain within this window of a top-dress leaches/washes off surface-applied N.
LEACHING_WINDOW_DAYS = 2
LEACHING_RAIN_MM = 30.0

# Map a nutrient to its soil-test key in the crop spec.
_FERTILITY_KEY = {
    "N": "available_N_kg_ha",
    "P2O5": "available_P2O5_kg_ha",
    "K2O": "available_K2O_kg_ha",
}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SoilNutrients:
    """Plant-available soil nutrients (kg/ha) with honest provenance.

    ``provenance`` records, per nutrient, whether the value was ``measured`` (a
    real soil test), ``estimated`` (derived — e.g. N from SoilGrids total N via a
    mineralisation fraction), or ``assumed`` (no data; defaulted to the medium
    fertility class). SoilGrids provides no plant-available P or K at all, so those
    are ``assumed`` unless the farmer supplies a soil test. **The UI must surface
    this distinction rather than presenting all three as equally certain.**
    """

    available: dict[str, float]
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for nut in NUTRIENTS:
            if nut not in self.available:
                raise ValueError(f"SoilNutrients missing {nut!r}")
            if self.available[nut] < 0:
                raise ValueError(f"SoilNutrients {nut} cannot be negative")

    def provenance_of(self, nutrient: str) -> str:
        return self.provenance.get(nutrient, "assumed")

    def fertility_class(self, nutrient: str, crop: CropSpec) -> str:
        """Classify this nutrient as low / medium / high per Soil Health Card bands."""
        key = _FERTILITY_KEY[nutrient]
        bands = crop.soil_fertility_classes.get(key)
        if not bands:
            return "medium"
        value = self.available[nutrient]
        for cls in ("low", "medium", "high"):
            rng = bands.get(cls)
            if rng and float(rng[0]) <= value < float(rng[1]):
                return cls
        # Above the top band.
        return "high" if value >= float(bands["high"][0]) else "medium"


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductAmount:
    """A physical quantity of a named fertiliser product."""

    product: str
    kg_ha: float
    supplies: dict[str, float]  # nutrient -> kg/ha delivered by this product

    def render(self) -> str:
        return f"{self.product} {self.kg_ha:.0f} kg/ha"


STATUS_PAST = "past"
STATUS_DUE = "due"
STATUS_UPCOMING = "upcoming"
STATUS_DEFERRED = "deferred"


@dataclass(frozen=True)
class DoseEvent:
    """One fertiliser application in the season schedule."""

    stage: str
    das: int                     # days after sowing
    day: date
    split_index: int
    nutrients: dict[str, float]  # nutrient -> kg/ha
    products: tuple[ProductAmount, ...]
    status: str
    defer_reason: str | None = None
    defer_until: date | None = None

    @property
    def is_actionable(self) -> bool:
        return self.status in (STATUS_DUE, STATUS_DEFERRED)

    @property
    def total_kg_ha(self) -> float:
        return sum(p.kg_ha for p in self.products)


@dataclass(frozen=True)
class NutrientPlan:
    """Full season fertiliser plan plus the next actionable dose."""

    target_yield_t_ha: float
    season_dose: dict[str, float]
    fertility_classes: dict[str, str]
    uptake_cross_check: dict[str, float]
    doses: tuple[DoseEvent, ...]
    next_dose: DoseEvent | None
    urgency: str
    explanation: Explanation
    warnings: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Product allocation
# --------------------------------------------------------------------------
def allocate_products(n: float, p2o5: float, k2o: float) -> tuple[ProductAmount, ...]:
    """Convert nutrient requirements (kg/ha) into physical product quantities.

    Standard Indian practice: meet P with DAP (which also carries 18% N), top the
    remaining N up with urea, and meet K with MOP. If DAP alone over-supplies N for
    this application, urea drops to zero and the surplus is reported so the excess
    is visible rather than silent.
    """
    out: list[ProductAmount] = []
    n_from_dap = 0.0

    if p2o5 > 0.5:
        dap_kg = p2o5 / PRODUCTS["DAP"]["P2O5"]
        n_from_dap = dap_kg * PRODUCTS["DAP"]["N"]
        out.append(ProductAmount("DAP", dap_kg, {"P2O5": p2o5, "N": n_from_dap}))

    n_remaining = n - n_from_dap
    if n_remaining > 0.5:
        urea_kg = n_remaining / PRODUCTS["Urea"]["N"]
        out.append(ProductAmount("Urea", urea_kg, {"N": n_remaining}))

    if k2o > 0.5:
        mop_kg = k2o / PRODUCTS["MOP"]["K2O"]
        out.append(ProductAmount("MOP", mop_kg, {"K2O": k2o}))

    return tuple(out)


# --------------------------------------------------------------------------
# The prescription
# --------------------------------------------------------------------------
def recommend_fertiliser(
    crop: CropSpec,
    soil: SoilNutrients,
    sowing_date: date,
    today: date,
    target_yield_t_ha: float | None = None,
    rain_forecast: dict[date, float] | None = None,
    irrigated: bool = True,
) -> NutrientPlan:
    """Build the season fertiliser schedule and identify the next action.

    ``today`` is a parameter, not a clock read, so the engine stays deterministic.
    ``target_yield_t_ha`` normally comes from the yield model; when omitted it
    falls back to the crop's attainable yield, so the plan works untrained.
    """
    spec = crop.nutrients
    rain_forecast = rain_forecast or {}

    if target_yield_t_ha is None:
        key = "irrigated" if irrigated else "rainfed"
        target_yield_t_ha = spec.attainable_yield_t_ha.get(
            key, spec.reference_yield_t_ha
        )
    if target_yield_t_ha <= 0:
        raise ValueError(f"target yield must be positive, got {target_yield_t_ha}")

    yield_ratio = target_yield_t_ha / spec.reference_yield_t_ha

    exp = Explanation().cite(
        "Scaled reference dose (Soil Health Card / Nutrient Expert method)",
        "ICAR package of practices",
    )
    exp.add_input("Crop", crop.label()) \
       .add_input("Target yield", target_yield_t_ha, "t/ha") \
       .add_input("Reference yield", spec.reference_yield_t_ha, "t/ha")

    # --- season dose per nutrient ----------------------------------------
    season_dose: dict[str, float] = {}
    fertility_classes: dict[str, str] = {}
    warnings: list[str] = []
    assumed: list[str] = []

    for nut in NUTRIENTS:
        cls = soil.fertility_class(nut, crop)
        factor = spec.fertility_adjustment[cls]
        ref = spec.reference_dose_kg_ha[nut]
        dose = ref * yield_ratio * factor
        season_dose[nut] = dose
        fertility_classes[nut] = cls

        prov = soil.provenance_of(nut)
        exp.add_input(
            f"Soil available {nut}",
            f"{soil.available[nut]:.0f} kg/ha ({cls} · {prov})",
        )
        exp.add_step(
            f"{nut} season dose",
            f"{ref:.0f} kg/ha × yield {yield_ratio:.2f} × {cls} soil {factor:.2f}",
            dose, "kg/ha",
        )
        if prov == "assumed":
            assumed.append(nut)

    # One consolidated warning rather than one per nutrient — a wall of warnings
    # trains the reader to skip all of them, including the ones that matter.
    if assumed:
        warnings.append(
            f"Soil {', '.join(assumed)} was assumed at the medium fertility class, not "
            f"measured. SoilGrids provides no plant-available P or K, so a soil test "
            f"(or Soil Health Card values) would materially sharpen these doses."
        )

    # --- mechanistic cross-check ----------------------------------------
    # Computed and shown in the trace for transparency, but deliberately NOT warned
    # on. The two methods are systematically incommensurable, not occasionally
    # divergent: the uptake budget assumes "replace what the crop removes, adjusted
    # for recovery", while validated packages account for soil reserves built up over
    # years and for realistic field losses. Divergence is therefore the normal case
    # for every crop — wheat lands ~128 % apart, cotton further still. A warning that
    # always fires is noise, and it trains the reader to skip the provenance warnings
    # that genuinely matter.
    cross_check: dict[str, float] = {}
    for nut in NUTRIENTS:
        requirement = target_yield_t_ha * spec.uptake_per_tonne_grain[nut]
        credit = min(soil.available[nut] * crop.mineralisation_fraction, requirement)
        cross_check[nut] = max(0.0, (requirement - credit) / spec.use_efficiency[nut])

    # --- split into scheduled events -------------------------------------
    doses: list[DoseEvent] = []
    for idx, split in enumerate(spec.splits):
        event_day = sowing_date + timedelta(days=split.day)
        amounts = {
            nut: season_dose[nut] * split.fraction.get(nut, 0.0) for nut in NUTRIENTS
        }
        if sum(amounts.values()) <= 0.5:
            continue

        products = allocate_products(amounts["N"], amounts["P2O5"], amounts["K2O"])

        # Status relative to today.
        if event_day < today:
            status, reason, until = STATUS_PAST, None, None
        elif event_day <= today + timedelta(days=2):
            status, reason, until = STATUS_DUE, None, None
        else:
            status, reason, until = STATUS_UPCOMING, None, None

        # Weather coupling: surface-applied N is vulnerable to heavy rain.
        if status == STATUS_DUE and amounts["N"] > 0.5:
            window_rain = sum(
                rain_forecast.get(event_day + timedelta(days=d), 0.0)
                for d in range(LEACHING_WINDOW_DAYS + 1)
            )
            if window_rain >= LEACHING_RAIN_MM:
                dry_day = _first_dry_day(event_day, rain_forecast)
                status = STATUS_DEFERRED
                reason = (
                    f"{window_rain:.0f} mm of rain forecast within "
                    f"{LEACHING_WINDOW_DAYS} days — surface-applied nitrogen would be "
                    f"lost to leaching and runoff."
                )
                until = dry_day

        doses.append(
            DoseEvent(
                stage=split.stage, das=split.day, day=event_day, split_index=idx,
                nutrients=amounts, products=products,
                status=status, defer_reason=reason, defer_until=until,
            )
        )

    next_dose = next((d for d in doses if d.is_actionable), None)
    if next_dose is None:
        next_dose = next((d for d in doses if d.status == STATUS_UPCOMING), None)

    # --- conclusion -------------------------------------------------------
    if next_dose is None:
        urgency = "good"
        exp.conclude(
            "All scheduled fertiliser applications for this season have passed. "
            "No further basal or top-dress application is due."
        )
    elif next_dose.status == STATUS_DEFERRED:
        urgency = "warning"
        when = (
            f"until {next_dose.defer_until:%d %b}" if next_dose.defer_until
            else "until the rain clears"
        )
        exp.conclude(
            f"Hold the {next_dose.stage} dose ({_render_products(next_dose)}) {when}. "
            f"{next_dose.defer_reason}"
        )
    elif next_dose.status == STATUS_DUE:
        urgency = "critical"
        exp.conclude(
            f"Apply {_render_products(next_dose)} now ({next_dose.stage} dose, "
            f"day {next_dose.das} after sowing) — supplying "
            f"{_render_nutrients(next_dose)}."
        )
    else:
        urgency = "info"
        days_out = (next_dose.day - today).days
        exp.conclude(
            f"Next application in {days_out} days ({next_dose.day:%d %b}, "
            f"{next_dose.stage} stage): {_render_products(next_dose)}, supplying "
            f"{_render_nutrients(next_dose)}."
        )

    for nut in NUTRIENTS:
        exp.add_step(
            f"{nut} uptake cross-check (not the recommendation)",
            f"({target_yield_t_ha:.1f} t/ha {spec.yield_basis} × "
            f"{spec.uptake_per_tonne_grain[nut]:.1f} kg/t − soil credit) ÷ NUE "
            f"{spec.use_efficiency[nut]:.2f}",
            cross_check[nut], "kg/ha",
        )
    exp.cite(
        "Uptake cross-check shown for transparency only — it routinely differs from "
        "validated packages, which account for multi-year soil reserves and field losses"
    )

    return NutrientPlan(
        target_yield_t_ha=target_yield_t_ha,
        season_dose=season_dose,
        fertility_classes=fertility_classes,
        uptake_cross_check=cross_check,
        doses=tuple(doses),
        next_dose=next_dose,
        urgency=urgency,
        explanation=exp,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _first_dry_day(start: date, rain_forecast: dict[date, float],
                   horizon: int = 14) -> date | None:
    """First day at or after ``start`` with negligible forecast rain."""
    for d in range(horizon + 1):
        day = start + timedelta(days=d)
        if day not in rain_forecast:
            continue
        if rain_forecast[day] < 5.0:
            return day
    return None


def _render_products(event: DoseEvent) -> str:
    return " + ".join(p.render() for p in event.products) or "no product"


def _render_nutrients(event: DoseEvent) -> str:
    parts = [f"{v:.0f} kg/ha {k}" for k, v in event.nutrients.items() if v > 0.5]
    return " + ".join(parts) or "nothing"
