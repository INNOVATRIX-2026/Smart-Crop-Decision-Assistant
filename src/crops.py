"""Load and validate hand-authored agronomic crop reference data.

This module replaces the per-crop statistics previously derived from the training
dataset (``src.model.compute_crop_stats``). That approach read its "ideal ranges"
out of a synthetic Kaggle dataset, so the thresholds it produced — and every
fertiliser dose computed from them — were statistical artifacts rather than
agronomic facts.

Here, reference values come from ``data/crops/*.yaml`` where each number carries a
source tag, and the loader *validates* them: a malformed spec raises at import
time rather than silently producing a wrong dose in the field.

The FAO-56 crop coefficient curve is reconstructed as a piecewise-linear function
of days-after-sowing, which is exactly the construction in FAO-56 Ch. 6::

    Kc │      ┌──────────┐
       │     ╱            ╲
       │ ───┘              ╲───
       └──────────────────────── days
         ini   dev    mid   late
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config

CROPS_DIR = config.ROOT / "data" / "crops"


class CropSpecError(ValueError):
    """Raised when a crop YAML file is missing or internally inconsistent."""


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One FAO-56 growth stage."""

    name: str
    days: int


@dataclass(frozen=True)
class _Curve:
    """Piecewise-linear curve over days-after-sowing."""

    points: tuple[tuple[float, float], ...]  # ((day, value), ...) ascending by day

    def at(self, day: float) -> float:
        pts = self.points
        if day <= pts[0][0]:
            return pts[0][1]
        if day >= pts[-1][0]:
            return pts[-1][1]
        for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
            if d0 <= day <= d1:
                if d1 == d0:
                    return v1
                frac = (day - d0) / (d1 - d0)
                return v0 + frac * (v1 - v0)
        return pts[-1][1]  # pragma: no cover - unreachable given the guards above


@dataclass(frozen=True)
class WaterSpec:
    """FAO-56 water parameters for one crop."""

    kc_initial: float
    kc_mid: float
    kc_end: float
    stages: tuple[Stage, ...]
    root_depth_initial_m: float
    root_depth_max_m: float
    depletion_fraction_p: float
    yield_response_ky: float
    application_efficiency: dict[str, float]
    critical_stages: tuple[str, ...]
    _kc_curve: _Curve
    _root_curve: _Curve

    @property
    def total_days(self) -> int:
        return sum(s.days for s in self.stages)

    def kc_at(self, das: float) -> float:
        """Crop coefficient at ``das`` days after sowing (FAO-56 Ch. 6 curve)."""
        return self._kc_curve.at(das)

    def root_depth_at(self, das: float) -> float:
        """Effective rooting depth (m) at ``das`` days after sowing."""
        return self._root_curve.at(das)

    def stage_at(self, das: float) -> str:
        """Name of the growth stage containing ``das``.

        Days before sowing return the first stage; days past harvest return the
        last, so callers never have to special-case out-of-season dates.
        """
        if das < 0:
            return self.stages[0].name
        elapsed = 0
        for stage in self.stages:
            elapsed += stage.days
            if das < elapsed:
                return stage.name
        return self.stages[-1].name

    def stage_start_day(self, name: str) -> int:
        """Days-after-sowing at which the named stage begins."""
        elapsed = 0
        for stage in self.stages:
            if stage.name == name:
                return elapsed
            elapsed += stage.days
        raise CropSpecError(f"unknown stage {name!r}")


@dataclass(frozen=True)
class NutrientSplit:
    """One fertiliser application event."""

    stage: str
    day: int
    fraction: dict[str, float]  # nutrient -> fraction of season dose


@dataclass(frozen=True)
class NutrientSpec:
    """Nutrient budget parameters for one crop.

    ``reference_dose_kg_ha`` is the PRIMARY basis for the recommended dose: a
    field-validated package scaled by yield target and soil fertility class, which
    is how Soil Health Card and Nutrient Expert work. ``uptake_per_tonne_grain``
    and ``use_efficiency`` drive a mechanistic cross-check only — that budget
    carries too many uncertain parameters to size a dose by itself.
    """

    uptake_per_tonne_grain: dict[str, float]
    use_efficiency: dict[str, float]
    splits: tuple[NutrientSplit, ...]
    attainable_yield_t_ha: dict[str, float]
    reference_dose_kg_ha: dict[str, float]
    reference_yield_t_ha: float
    fertility_adjustment: dict[str, float]
    cross_check_tolerance: float
    yield_basis: str = "grain"
    """What a tonne of yield refers to — 'grain', 'seed cotton', 'millable cane'.

    Not cosmetic: uptake figures are *per tonne of this*, and sugarcane yields
    ~80 t/ha of cane against wheat's ~5 t/ha of grain. Labelling both as "grain"
    would invite exactly the units confusion this rewrite exists to remove.
    """

    @property
    def nutrients(self) -> tuple[str, ...]:
        return tuple(self.uptake_per_tonne_grain)


@dataclass(frozen=True)
class CropSpec:
    """Complete agronomic reference for one crop."""

    crop: str
    display_name: dict[str, str]
    category: str
    season: str
    water: WaterSpec
    nutrients: NutrientSpec
    soil_fertility_classes: dict[str, dict[str, list[float]]]
    mineralisation_fraction: float
    limits: dict

    def label(self, lang: str = "en") -> str:
        return self.display_name.get(lang) or self.display_name.get("en") or self.crop.title()


# --------------------------------------------------------------------------
# Parsing / validation
# --------------------------------------------------------------------------
NUTRIENTS = ("N", "P2O5", "K2O")


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise CropSpecError(f"{where}: missing required key {key!r}")
    return mapping[key]


def _build_kc_curve(kc_ini: float, kc_mid: float, kc_end: float,
                    stages: tuple[Stage, ...]) -> _Curve:
    """Reconstruct the FAO-56 Kc curve as control points.

    Flat at ``kc_ini`` through the initial stage, ramping to ``kc_mid`` across
    development, flat through mid-season, then ramping to ``kc_end`` across the
    late stage.
    """
    bounds: dict[str, tuple[int, int]] = {}
    elapsed = 0
    for stage in stages:
        bounds[stage.name] = (elapsed, elapsed + stage.days)
        elapsed += stage.days

    for name in ("initial", "development", "mid", "late"):
        if name not in bounds:
            raise CropSpecError(
                f"stages must include {name!r}; got {[s.name for s in stages]}"
            )

    return _Curve(
        (
            (0.0, kc_ini),
            (float(bounds["initial"][1]), kc_ini),
            (float(bounds["development"][1]), kc_mid),
            (float(bounds["mid"][1]), kc_mid),
            (float(bounds["late"][1]), kc_end),
        )
    )


def _build_root_curve(initial_m: float, max_m: float,
                      stages: tuple[Stage, ...]) -> _Curve:
    """Root depth grows from sowing to its maximum by the end of development."""
    elapsed = 0
    dev_end = None
    for stage in stages:
        elapsed += stage.days
        if stage.name == "development":
            dev_end = elapsed
            break
    if dev_end is None:  # pragma: no cover - _build_kc_curve already enforces this
        raise CropSpecError("stages must include 'development'")
    total = sum(s.days for s in stages)
    return _Curve(((0.0, initial_m), (float(dev_end), max_m), (float(total), max_m)))


def _parse(raw: dict, path: Path) -> CropSpec:
    where = path.name
    name = str(_require(raw, "crop", where)).strip().lower()

    # --- water -----------------------------------------------------------
    w = _require(raw, "water", where)
    kc = _require(w, "kc", f"{where}:water")
    stages = tuple(
        Stage(name=str(_require(s, "name", f"{where}:water.stages")),
              days=int(_require(s, "days", f"{where}:water.stages")))
        for s in _require(w, "stages", f"{where}:water")
    )
    if not stages:
        raise CropSpecError(f"{where}: water.stages is empty")
    for s in stages:
        if s.days <= 0:
            raise CropSpecError(f"{where}: stage {s.name!r} has non-positive days ({s.days})")

    kc_ini = float(_require(kc, "initial", f"{where}:water.kc"))
    kc_mid = float(_require(kc, "mid", f"{where}:water.kc"))
    kc_end = float(_require(kc, "end", f"{where}:water.kc"))
    for label, val in (("initial", kc_ini), ("mid", kc_mid), ("end", kc_end)):
        if not 0.0 < val < 2.0:
            raise CropSpecError(
                f"{where}: water.kc.{label} = {val} is outside the plausible 0-2 range"
            )

    rd = _require(w, "root_depth_m", f"{where}:water")
    rd_ini = float(_require(rd, "initial", f"{where}:water.root_depth_m"))
    rd_max = float(_require(rd, "max", f"{where}:water.root_depth_m"))
    if not 0.0 < rd_ini <= rd_max:
        raise CropSpecError(
            f"{where}: root depth initial ({rd_ini}) must be >0 and <= max ({rd_max})"
        )

    p = float(_require(w, "depletion_fraction_p", f"{where}:water"))
    if not 0.0 < p < 1.0:
        raise CropSpecError(f"{where}: depletion_fraction_p = {p} must be in (0, 1)")

    eff = {str(k): float(v) for k, v in _require(
        w, "application_efficiency", f"{where}:water").items()}
    for method, val in eff.items():
        if not 0.0 < val <= 1.0:
            raise CropSpecError(
                f"{where}: application_efficiency.{method} = {val} must be in (0, 1]"
            )

    water = WaterSpec(
        kc_initial=kc_ini,
        kc_mid=kc_mid,
        kc_end=kc_end,
        stages=stages,
        root_depth_initial_m=rd_ini,
        root_depth_max_m=rd_max,
        depletion_fraction_p=p,
        yield_response_ky=float(w.get("yield_response_ky", 1.0)),
        application_efficiency=eff,
        critical_stages=tuple(str(s) for s in w.get("critical_stages", ())),
        _kc_curve=_build_kc_curve(kc_ini, kc_mid, kc_end, stages),
        _root_curve=_build_root_curve(rd_ini, rd_max, stages),
    )

    for cs in water.critical_stages:
        if cs not in {s.name for s in stages}:
            raise CropSpecError(f"{where}: critical_stages names unknown stage {cs!r}")

    # --- nutrients -------------------------------------------------------
    n = _require(raw, "nutrients", where)
    uptake = {str(k): float(v) for k, v in _require(
        n, "uptake_per_tonne_grain", f"{where}:nutrients").items()}
    use_eff = {str(k): float(v) for k, v in _require(
        n, "use_efficiency", f"{where}:nutrients").items()}

    for nut in NUTRIENTS:
        if nut not in uptake:
            raise CropSpecError(f"{where}: nutrients.uptake_per_tonne_grain missing {nut!r}")
        if nut not in use_eff:
            raise CropSpecError(f"{where}: nutrients.use_efficiency missing {nut!r}")
        if uptake[nut] <= 0:
            raise CropSpecError(f"{where}: uptake of {nut} must be positive")
        if not 0.0 < use_eff[nut] <= 1.0:
            raise CropSpecError(
                f"{where}: use_efficiency.{nut} = {use_eff[nut]} must be in (0, 1]"
            )

    splits = tuple(
        NutrientSplit(
            stage=str(_require(s, "stage", f"{where}:nutrients.splits")),
            day=int(_require(s, "day", f"{where}:nutrients.splits")),
            fraction={str(k): float(v) for k, v in
                      _require(s, "fraction", f"{where}:nutrients.splits").items()},
        )
        for s in _require(n, "splits", f"{where}:nutrients")
    )
    if not splits:
        raise CropSpecError(f"{where}: nutrients.splits is empty")

    # Every nutrient's splits must sum to exactly one season dose. A drift here
    # would silently under- or over-apply, so it is a hard error, not a warning.
    for nut in NUTRIENTS:
        total = sum(s.fraction.get(nut, 0.0) for s in splits)
        if abs(total - 1.0) > 1e-6:
            raise CropSpecError(
                f"{where}: nutrients.splits for {nut} sum to {total:.4f}, expected 1.0"
            )

    for s in splits:
        if s.day < 0:
            raise CropSpecError(f"{where}: split {s.stage!r} has negative day {s.day}")
        if s.day > water.total_days:
            raise CropSpecError(
                f"{where}: split {s.stage!r} at day {s.day} falls after harvest "
                f"(season is {water.total_days} days)"
            )

    ref_dose_raw = _require(n, "reference_dose_kg_ha", f"{where}:nutrients")
    ref_yield = float(_require(ref_dose_raw, "at_yield_t_ha", f"{where}:reference_dose_kg_ha"))
    if ref_yield <= 0:
        raise CropSpecError(f"{where}: reference_dose_kg_ha.at_yield_t_ha must be positive")
    ref_dose = {k: float(v) for k, v in ref_dose_raw.items() if k != "at_yield_t_ha"}
    for nut in NUTRIENTS:
        if nut not in ref_dose:
            raise CropSpecError(f"{where}: reference_dose_kg_ha missing {nut!r}")

    fertility = {str(k): float(v) for k, v in _require(
        n, "fertility_adjustment", f"{where}:nutrients").items()}
    for cls in ("low", "medium", "high"):
        if cls not in fertility:
            raise CropSpecError(f"{where}: nutrients.fertility_adjustment missing {cls!r}")
    if not fertility["low"] >= fertility["medium"] >= fertility["high"]:
        raise CropSpecError(
            f"{where}: fertility_adjustment must not increase with fertility "
            f"(got low={fertility['low']}, medium={fertility['medium']}, "
            f"high={fertility['high']}) — a richer soil cannot need more fertiliser"
        )

    nutrients = NutrientSpec(
        uptake_per_tonne_grain=uptake,
        use_efficiency=use_eff,
        splits=splits,
        attainable_yield_t_ha={str(k): float(v) for k, v in _require(
            n, "attainable_yield_t_ha", f"{where}:nutrients").items()},
        reference_dose_kg_ha=ref_dose,
        reference_yield_t_ha=ref_yield,
        fertility_adjustment=fertility,
        cross_check_tolerance=float(n.get("cross_check_tolerance", 0.6)),
        yield_basis=str(n.get("yield_basis", "grain")),
    )

    # --- misc ------------------------------------------------------------
    mineralisation = float(_require(raw, "mineralisation_fraction", where))
    if not 0.0 < mineralisation < 0.5:
        raise CropSpecError(
            f"{where}: mineralisation_fraction = {mineralisation} is implausible "
            "(expected a small fraction such as 0.02)"
        )

    return CropSpec(
        crop=name,
        display_name={str(k): str(v) for k, v in raw.get("display_name", {}).items()}
        or {"en": name.title()},
        category=str(raw.get("category", "crop")),
        season=str(raw.get("season", "")),
        water=water,
        nutrients=nutrients,
        soil_fertility_classes=_require(raw, "soil_fertility_classes", where),
        mineralisation_fraction=mineralisation,
        limits=raw.get("limits", {}),
    )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_crops() -> dict[str, CropSpec]:
    """Load and validate every crop spec in ``data/crops/``.

    Cached: the YAML files are static reference data, so they are parsed once per
    process. Raises :class:`CropSpecError` if the directory is empty, so a broken
    deployment fails loudly instead of silently offering no crops.
    """
    if not CROPS_DIR.is_dir():
        raise CropSpecError(f"crop reference directory not found: {CROPS_DIR}")

    specs: dict[str, CropSpec] = {}
    for path in sorted(CROPS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise CropSpecError(f"{path.name}: invalid YAML — {exc}") from exc
        if not isinstance(raw, dict):
            raise CropSpecError(f"{path.name}: expected a YAML mapping at the top level")
        spec = _parse(raw, path)
        if spec.crop in specs:
            raise CropSpecError(f"duplicate crop {spec.crop!r} in {path.name}")
        specs[spec.crop] = spec

    if not specs:
        raise CropSpecError(f"no crop specs found in {CROPS_DIR}")
    return specs


def get_crop(name: str) -> CropSpec:
    """Look up one crop spec by name (case-insensitive)."""
    crops = load_crops()
    key = name.strip().lower()
    if key not in crops:
        raise CropSpecError(
            f"no reference data for crop {name!r}; available: {sorted(crops)}"
        )
    return crops[key]


def available_crops() -> list[str]:
    """Sorted list of crops with reference data."""
    return sorted(load_crops())
