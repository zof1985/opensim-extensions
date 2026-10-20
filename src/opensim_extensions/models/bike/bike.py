from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ...bodies import (
    Box,
    Cylinder,
    Ellipsoid,
    add_pinned_body,
    add_welded_body,
    cylinder_between,
)
from ...opensim import import_opensim

__all__ = ["Bike", "IRON_DENSITY_KG_M3"]

opensim = import_opensim()

IRON_DENSITY_KG_M3 = 7870.0
_MM = 1e-3
_SCREEN_THICKNESS_MM = 6.0
_SCREEN_MOUNT_RADIUS_MM = 6.0
_SCREEN_COLOR = (0.25, 0.65, 0.8)
_SCREEN_MOUNT_COLOR = (0.2, 0.2, 0.2)
_SADDLE_COLOR = (0.12, 0.12, 0.12)
COMPONENTS = (
    "frame",
    "seat_tube",
    "seat",
    "stem",
    "handlebar_slide",
    "handlebar",
    "cranks",
    "pedals",
)
_CRANK_BAR = (0.03, 0.015)  # crank bar (height along y, thickness along z), m
_AXLE_RADIUS = 0.012


def _horn_direction(pitch_deg: float, roll_deg: float, side: str) -> np.ndarray:
    """Return a horn's unit direction; positive pitch raises it and roll toes it in."""
    if side not in {"r", "l"}:
        raise ValueError("side must be 'r' or 'l'")
    side_sign = 1.0 if side == "r" else -1.0
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    return np.array(
        [
            math.cos(pitch) * math.cos(roll),
            math.sin(pitch),
            -side_sign * math.cos(pitch) * math.sin(roll),
        ]
    )


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite value")
    return value


def _range(name: str, value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must be (min, max)")
    low, high = (_finite(name, v) for v in value)
    if low > high:
        raise ValueError(f"{name} minimum must not exceed its maximum")
    return low, high


def _steps(name: str, value: int) -> int:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _vec3(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must have three components (x, y, z)")
    return tuple(_finite(name, v) for v in value)  # type: ignore[return-value]


def _pedal_size(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must be (length, width, height)")
    return tuple(_positive(name, v) for v in value)  # type: ignore[return-value]


def _force(name: str, value: float) -> float:
    value = _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _check_step(name: str, step: int, steps: int) -> int:
    if int(step) != step or not 0 <= step <= steps:
        raise ValueError(f"{name} must be an integer between 0 and {steps}")
    return int(step)


def _density_property(component: str):
    def getter(self) -> float:
        return self._densities[component]

    def setter(self, value: float) -> None:
        self.set_density(component, value)

    return property(
        getter, setter, doc=f"Density of the {component} material (kg/m^3)."
    )


def _parameter_property(name: str, doc: str):
    def getter(self):
        return getattr(self, f"_{name}")

    def setter(self, value) -> None:
        getattr(self, f"set_{name}")(value)

    return property(getter, setter, doc=doc)


class Bike(opensim.Model):
    """A fixed exercise-bike OpenSim model; only cranks and pedals move.

    Conventions (world frame, ground at ``y = 0``): ``x`` points forward,
    ``y`` up, ``z`` to the rider's right. Lengths are given in millimetres,
    angles in degrees, densities in kg/m^3 and forces in newtons. Everything
    is welded to ground except the crank (coordinate ``crank_angle``; the two
    cranks are 180 degrees apart) and the two pedals (coordinates
    ``pedal_right_angle`` and ``pedal_left_angle``, rotating about the axis at
    the crank end). Directed angles in the XY plane are measured from +X
    counterclockwise towards +Y (right-hand rotation about +Z); an upward
    direction pointing behind the pedals therefore has an angle greater than
    90 degrees.

    Each adjustable element (seat tube, seat slide, stem, handlebar slide) has
    a range split in ``steps`` intervals (``steps + 1`` positions); the
    ``*_step`` index selects the fixed position (default: the middle one).
    Public coordinate and position properties return millimetres; geometry
    construction and OpenSim calculations use metres internally.
    The default stem origin is calibrated for ``fit_user()``'s 170 cm, 70 kg
    male reference rider in a 45-degree trunk, 75-degree shoulder and
    15-degree elbow pose.

    Parameters
    ----------
    seat_tube_angle_deg
        Directed angle of the seat tube from the pedal axis, measured from +X
        counterclockwise about +Z. A seat above and behind the pedal axis has
        an angle greater than 90 degrees.
    seat_tube_range_mm, seat_tube_steps
        (min, max) distance of the seat tube top from the pedal axis along
        the tube, and number of intervals.
    crank_axis_position_mm
        (x, y, z) of the pedal axis centre in world coordinates.
    seat_height_mm
        Seat thickness.
    seat_horizontal_range_mm, seat_horizontal_steps, seat_slide_angle_deg
        Seat slide range relative to the seat tube top centre, intervals,
        and direction from +X, counterclockwise about +Z (0 deg points forward).
    stem_origin_mm
        (x, y, z) of the handlebar stem base relative to the pedal axis.
    stem_excursion_mm, stem_steps, stem_angle_deg
        Stem extension range ``[0, excursion]``, intervals, and stem angle
        measured from +X, counterclockwise about +Z.
    handlebar_horizontal_range_mm, handlebar_horizontal_steps, handlebar_slide_angle_deg
        Same as for the seat, relative to the stem top; 0 deg points along +X.
    crank_length_mm, pedal_size_mm
        Crank length and pedal (length along x, width along z, height).
    pedal_spacing_mm
        Internal distance between the pedal faces in millimetres. Defaults to
        185 mm, preserving the former pedal geometry.
    crank_spacing_mm
        Distance between the crank centre planes; controls the crank axle.
    screen_width_mm, screen_height_mm
        Screen dimensions (width across the handlebar, height along the panel).
    screen_inclination_deg
        Directed angle of the screen's upward direction from +X,
        counterclockwise about +Z. A top leaning back towards the rider has an
        angle greater than 90 degrees.
    screen_vertical_offset_mm, screen_horizontal_offset_mm
        Positive offsets of the screen base from the handlebar centre, in the
        upward and forward directions respectively.
    handlebar_diameter_mm, handlebar_width_mm, horn_length_mm
        Cross bar and horns; the horns share the bar diameter and extend
        forward from the bar ends.
    horn_pitch_deg, horn_roll_deg
        Pitch and medial roll of the horns. Positive pitch raises the tips;
        positive roll turns each horn towards the bike centreline.
    densities
        One density for all components, or a mapping per component
        (``frame``, ``seat_tube``, ``seat``, ``stem``, ``handlebar_slide``,
        ``handlebar``,
        ``cranks``, ``pedals``). Defaults to iron.
    brake_force_n
        Resistive force at each pedal, opposing the crank rotation.
    tube_diameter_mm, seat_length_mm, seat_width_mm, frame_width_mm, frame_margin_mm
        Remaining dimensions (defaults provided).
    """

    def __init__(
        self,
        seat_tube_angle_deg: float = 104.0,
        seat_tube_range_mm: Sequence[float] = (250.0, 1200.0),
        seat_tube_steps: int = 20,
        crank_axis_position_mm: Sequence[float] = (0.0, 250.0, 0.0),
        seat_height_mm: float = 50.0,
        seat_horizontal_range_mm: Sequence[float] = (-100.0, 100.0),
        seat_horizontal_steps: int = 25,
        seat_slide_angle_deg: float = 0.0,
        stem_origin_mm: Sequence[float] = (500, 500, 0.0),
        stem_excursion_mm: float = 250.0,
        stem_steps: int = 25,
        stem_angle_deg: float = 102.0,
        handlebar_horizontal_range_mm: Sequence[float] = (-100.0, 100.0),
        handlebar_horizontal_steps: int = 10,
        handlebar_slide_angle_deg: float = 10.0,
        crank_length_mm: float = 172.5,
        pedal_size_mm: Sequence[float] = (110.0, 70.0, 25.0),
        handlebar_diameter_mm: float = 25.0,
        handlebar_width_mm: float = 500.0,
        horn_pitch_deg: float = 35.0,
        horn_roll_deg: float = 10.0,
        horn_length_mm: float = 150.0,
        densities: float | Mapping[str, float] = IRON_DENSITY_KG_M3,
        brake_force_n: float = 0.0,
        *,
        seat_tube_step: int | None = None,
        seat_horizontal_step: int | None = None,
        stem_step: int | None = None,
        handlebar_horizontal_step: int | None = None,
        tube_diameter_mm: float = 40.0,
        seat_length_mm: float = 260.0,
        seat_width_mm: float = 140.0,
        crank_spacing_mm: float = 170.0,
        pedal_spacing_mm: float = 185.0,
        frame_width_mm: float = 500.0,
        frame_margin_mm: float = 300.0,
        screen_width_mm: float = 150.0,
        screen_height_mm: float = 100.0,
        screen_inclination_deg: float = 65.0,
        screen_vertical_offset_mm: float = 100.0,
        screen_horizontal_offset_mm: float = 50.0,
    ):
        self.opensim = opensim
        self._building = True
        self._seat_tube_angle_deg = _finite("seat_tube_angle_deg", seat_tube_angle_deg)
        self._seat_tube_range_mm = _range("seat_tube_range_mm", seat_tube_range_mm)
        self._seat_tube_steps = _steps("seat_tube_steps", seat_tube_steps)
        self._crank_axis_position_mm = _vec3(
            "crank_axis_position_mm", crank_axis_position_mm
        )
        self._seat_height_mm = _positive("seat_height_mm", seat_height_mm)
        self._seat_horizontal_range_mm = _range(
            "seat_horizontal_range_mm", seat_horizontal_range_mm
        )
        self._seat_horizontal_steps = _steps(
            "seat_horizontal_steps", seat_horizontal_steps
        )
        self._seat_slide_angle_deg = _finite(
            "seat_slide_angle_deg", seat_slide_angle_deg
        )
        self._stem_origin_mm = _vec3("stem_origin_mm", stem_origin_mm)
        self._stem_excursion_mm = _positive("stem_excursion_mm", stem_excursion_mm)
        self._stem_steps = _steps("stem_steps", stem_steps)
        self._stem_angle_deg = _finite("stem_angle_deg", stem_angle_deg)
        self._handlebar_horizontal_range_mm = _range(
            "handlebar_horizontal_range_mm", handlebar_horizontal_range_mm
        )
        self._handlebar_horizontal_steps = _steps(
            "handlebar_horizontal_steps", handlebar_horizontal_steps
        )
        self._handlebar_slide_angle_deg = _finite(
            "handlebar_slide_angle_deg", handlebar_slide_angle_deg
        )
        self._crank_length_mm = _positive("crank_length_mm", crank_length_mm)
        self._pedal_size_mm = _pedal_size("pedal_size_mm", pedal_size_mm)
        self._handlebar_diameter_mm = _positive(
            "handlebar_diameter_mm", handlebar_diameter_mm
        )
        self._handlebar_width_mm = _positive("handlebar_width_mm", handlebar_width_mm)
        self._horn_pitch_deg = _finite("horn_pitch_deg", horn_pitch_deg)
        self._horn_roll_deg = _finite("horn_roll_deg", horn_roll_deg)
        self._horn_length_mm = _positive("horn_length_mm", horn_length_mm)
        self._brake_force_n = _force("brake_force_n", brake_force_n)
        self._tube_diameter_mm = _positive("tube_diameter_mm", tube_diameter_mm)
        self._seat_length_mm = _positive("seat_length_mm", seat_length_mm)
        self._seat_width_mm = _positive("seat_width_mm", seat_width_mm)
        self._crank_spacing_mm = _positive("crank_spacing_mm", crank_spacing_mm)
        self._pedal_spacing_mm = _positive("pedal_spacing_mm", pedal_spacing_mm)
        self._frame_width_mm = _positive("frame_width_mm", frame_width_mm)
        self._frame_margin_mm = _positive("frame_margin_mm", frame_margin_mm)
        self._screen_width_mm = _positive("screen_width_mm", screen_width_mm)
        self._screen_height_mm = _positive("screen_height_mm", screen_height_mm)
        self._screen_inclination_deg = _finite(
            "screen_inclination_deg", screen_inclination_deg
        )
        self._screen_vertical_offset_mm = _positive(
            "screen_vertical_offset_mm", screen_vertical_offset_mm
        )
        self._screen_horizontal_offset_mm = _positive(
            "screen_horizontal_offset_mm", screen_horizontal_offset_mm
        )

        self._densities = {name: IRON_DENSITY_KG_M3 for name in COMPONENTS}
        self._store_densities(densities)

        self._seat_tube_step = _check_step(
            "seat_tube_step",
            self._seat_tube_steps // 2 if seat_tube_step is None else seat_tube_step,
            self._seat_tube_steps,
        )
        self._seat_horizontal_step = _check_step(
            "seat_horizontal_step",
            (
                self._seat_horizontal_steps // 2
                if seat_horizontal_step is None
                else seat_horizontal_step
            ),
            self._seat_horizontal_steps,
        )
        self._stem_step = _check_step(
            "stem_step",
            self._stem_steps // 2 if stem_step is None else stem_step,
            self._stem_steps,
        )
        self._handlebar_horizontal_step = _check_step(
            "handlebar_horizontal_step",
            (
                self._handlebar_horizontal_steps // 2
                if handlebar_horizontal_step is None
                else handlebar_horizontal_step
            ),
            self._handlebar_horizontal_steps,
        )

        self._angles_deg = {
            "crank_angle": 0.0,
            "pedal_right_angle": 0.0,
            "pedal_left_angle": 0.0,
        }
        self._build()
        self._building = False

    # ---------------------------------------------------------------- internals
    def _store_densities(self, densities: float | Mapping[str, float]) -> None:
        items = (
            densities.items()
            if isinstance(densities, Mapping)
            else [(c, densities) for c in COMPONENTS]
        )
        staged = {}
        for name, value in items:
            if name not in COMPONENTS:
                raise ValueError(
                    f"Unknown component '{name}'; expected one of {COMPONENTS}"
                )
            staged[name] = _positive(f"{name} density", value)
        self._densities.update(staged)

    def _apply(self) -> None:
        if not self._building:
            self._build()

    @staticmethod
    def _positions(low: float, high: float, steps: int) -> np.ndarray:
        return np.linspace(low, high, steps + 1)

    # --------------------------------------------------------- derived values
    @property
    def seat_tube_positions_mm(self) -> np.ndarray:
        """All selectable seat tube lengths (mm)."""
        return self._positions(*self._seat_tube_range_mm, self._seat_tube_steps)

    @property
    def seat_horizontal_positions_mm(self) -> np.ndarray:
        """All selectable seat slide offsets (mm)."""
        return self._positions(
            *self._seat_horizontal_range_mm, self._seat_horizontal_steps
        )

    @property
    def stem_positions_mm(self) -> np.ndarray:
        """All selectable stem extensions (mm)."""
        return self._positions(0.0, self._stem_excursion_mm, self._stem_steps)

    @property
    def handlebar_horizontal_positions_mm(self) -> np.ndarray:
        """All selectable handlebar slide offsets (mm)."""
        return self._positions(
            *self._handlebar_horizontal_range_mm, self._handlebar_horizontal_steps
        )

    @property
    def crank_axis_position(self) -> np.ndarray:
        """Pedal axis centre in world coordinates (mm)."""
        return np.asarray(self._crank_axis_position_mm, dtype=float)

    @property
    def _crank_axis_position_m(self) -> np.ndarray:
        """Pedal axis centre in world coordinates (m), for OpenSim calculations."""
        return np.asarray(self._crank_axis_position_mm) * _MM

    @property
    def pedal_lateral_offset(self) -> float:
        """Distance (mm) of each pedal's outer face from the bike's sagittal plane."""
        return self._pedal_lateral_offset_m / _MM

    @property
    def _pedal_lateral_offset_m(self) -> float:
        return self.pedal_spacing_mm * _MM / 2 + self._pedal_size_mm[1] * _MM

    @property
    def pedal_spacing_mm(self) -> float:
        """Distance between the inner faces of the pedals, in millimetres."""
        return self._pedal_spacing_mm

    @pedal_spacing_mm.setter
    def pedal_spacing_mm(self, value: float) -> None:
        self.set_pedal_spacing_mm(value)

    def _seat_tube_top_at(self, tube_step: int) -> np.ndarray:
        angle = math.radians(self._seat_tube_angle_deg)
        length = self.seat_tube_positions_mm[tube_step] * _MM
        return self._crank_axis_position_m + length * np.array(
            [math.cos(angle), math.sin(angle), 0.0]
        )

    def _seat_position_at(self, tube_step: int, slide_step: int) -> np.ndarray:
        slide = math.radians(self._seat_slide_angle_deg)
        offset = self.seat_horizontal_positions_mm[slide_step] * _MM
        lift = np.array([0.0, self._seat_height_mm * _MM / 2, 0.0])
        return (
            self._seat_tube_top_at(tube_step)
            + offset * np.array([math.cos(slide), math.sin(slide), 0.0])
            + lift
        )

    def _stem_top_at(self, stem_step: int) -> np.ndarray:
        angle = math.radians(self._stem_angle_deg)
        length = self.stem_positions_mm[stem_step] * _MM
        return self._stem_base_m + length * np.array(
            [math.cos(angle), math.sin(angle), 0.0]
        )

    def _handlebar_position_at(self, stem_step: int, slide_step: int) -> np.ndarray:
        slide = math.radians(self._handlebar_slide_angle_deg)
        offset = self.handlebar_horizontal_positions_mm[slide_step] * _MM
        return self._stem_top_at(stem_step) + offset * np.array(
            [math.cos(slide), math.sin(slide), 0.0]
        )

    @property
    def seat_tube_top(self) -> np.ndarray:
        """Seat tube top centre in world coordinates (mm)."""
        return self._seat_tube_top_at(self._seat_tube_step) / _MM

    @property
    def seat_position(self) -> np.ndarray:
        """Seat centre in world coordinates (mm); the seat rests on the slide point."""
        return (
            self._seat_position_at(self._seat_tube_step, self._seat_horizontal_step)
            / _MM
        )

    @property
    def stem_base(self) -> np.ndarray:
        """Stem base in world coordinates (mm)."""
        return self._stem_base_m / _MM

    @property
    def _stem_base_m(self) -> np.ndarray:
        return self._crank_axis_position_m + np.asarray(self._stem_origin_mm) * _MM

    @property
    def stem_top(self) -> np.ndarray:
        """Stem top centre in world coordinates (mm)."""
        return self._stem_top_at(self._stem_step) / _MM

    @property
    def handlebar_position(self) -> np.ndarray:
        """Handlebar centre point in world coordinates (mm)."""
        return self._handlebar_position_m / _MM

    @property
    def _handlebar_position_m(self) -> np.ndarray:
        return self._handlebar_position_at(
            self._stem_step, self._handlebar_horizontal_step
        )

    @property
    def _screen_base_m(self) -> np.ndarray:
        return self._handlebar_position_m + _MM * np.array(
            [self._screen_horizontal_offset_mm, self._screen_vertical_offset_mm, 0.0]
        )

    @property
    def screen_base(self) -> np.ndarray:
        """Screen base centre in world coordinates (mm)."""
        return self._screen_base_m / _MM

    @property
    def _screen_center_m(self) -> np.ndarray:
        angle = math.radians(self._screen_inclination_deg)
        up_the_panel = np.array([math.cos(angle), math.sin(angle), 0.0])
        return self._screen_base_m + (self._screen_height_mm * _MM / 2) * up_the_panel

    @property
    def screen_center(self) -> np.ndarray:
        """Screen centre in world coordinates (mm)."""
        return self._screen_center_m / _MM

    @property
    def component_masses_kg(self) -> dict[str, float]:
        """Mass of each component group (kg)."""
        return dict(self._component_masses)

    @property
    def total_mass_kg(self) -> float:
        """Total mass of the bike (kg)."""
        return float(sum(self._component_masses.values()))

    # ------------------------------------------------------------------ setters
    def set_density(self, component: str, value: float) -> None:
        """Set the material density (kg/m^3) of one component."""
        self._store_densities({component: value})
        self._apply()

    def set_densities(self, densities: float | Mapping[str, float]) -> None:
        """Set one density for all components, or a mapping per component."""
        self._store_densities(densities)
        self._apply()

    def _set_angle(self, name: str, value: float) -> None:
        self._angles_deg[name] = _finite(name, value)
        self._reset_state()

    def set_crank_angle_deg(self, value: float) -> None:
        """Rotate the cranks to ``value`` degrees (0 = right crank pointing forward)."""
        self._set_angle("crank_angle", value)

    def set_pedal_right_angle_deg(self, value: float) -> None:
        self._set_angle("pedal_right_angle", value)

    def set_pedal_left_angle_deg(self, value: float) -> None:
        self._set_angle("pedal_left_angle", value)

    crank_angle_deg = property(
        lambda self: self._angles_deg["crank_angle"], set_crank_angle_deg
    )
    pedal_right_angle_deg = property(
        lambda self: self._angles_deg["pedal_right_angle"], set_pedal_right_angle_deg
    )
    pedal_left_angle_deg = property(
        lambda self: self._angles_deg["pedal_left_angle"], set_pedal_left_angle_deg
    )

    def set_brake_force_n(self, value: float) -> None:
        self._brake_force_n = _force("brake_force_n", value)
        self._apply()

    def set_pedal_spacing_mm(self, value: float) -> None:
        self._pedal_spacing_mm = _positive("pedal_spacing_mm", value)
        self._apply()

    def _make_setter(name: str, check):  # noqa: N805 - class-body factory
        def setter(self, value) -> None:
            setattr(self, f"_{name}", check(name, value))
            self._apply()

        setter.__name__ = f"set_{name}"
        setter.__doc__ = f"Set ``{name}`` and rebuild the model."
        return setter

    def _make_steps_setter(name: str, step_name: str):  # noqa: N805
        def setter(self, value) -> None:
            setattr(self, f"_{name}", _steps(name, value))
            current = getattr(self, f"_{step_name}")
            setattr(self, f"_{step_name}", min(current, getattr(self, f"_{name}")))
            self._apply()

        setter.__name__ = f"set_{name}"
        setter.__doc__ = (
            f"Set ``{name}`` (the selected step is clamped) and rebuild the model."
        )
        return setter

    def _make_step_setter(name: str, steps_name: str):  # noqa: N805
        def setter(self, value) -> None:
            setattr(
                self,
                f"_{name}",
                _check_step(name, value, getattr(self, f"_{steps_name}")),
            )
            self._apply()

        setter.__name__ = f"set_{name}"
        setter.__doc__ = f"Select the ``{name}`` position index and rebuild the model."
        return setter

    set_seat_tube_angle_deg = _make_setter("seat_tube_angle_deg", _finite)
    set_seat_tube_range_mm = _make_setter("seat_tube_range_mm", _range)
    set_seat_tube_steps = _make_steps_setter("seat_tube_steps", "seat_tube_step")
    set_seat_tube_step = _make_step_setter("seat_tube_step", "seat_tube_steps")
    set_crank_axis_position_mm = _make_setter("crank_axis_position_mm", _vec3)
    set_seat_height_mm = _make_setter("seat_height_mm", _positive)
    set_seat_horizontal_range_mm = _make_setter("seat_horizontal_range_mm", _range)
    set_seat_horizontal_steps = _make_steps_setter(
        "seat_horizontal_steps", "seat_horizontal_step"
    )
    set_seat_horizontal_step = _make_step_setter(
        "seat_horizontal_step", "seat_horizontal_steps"
    )
    set_seat_slide_angle_deg = _make_setter("seat_slide_angle_deg", _finite)
    set_stem_origin_mm = _make_setter("stem_origin_mm", _vec3)
    set_stem_excursion_mm = _make_setter("stem_excursion_mm", _positive)
    set_stem_steps = _make_steps_setter("stem_steps", "stem_step")
    set_stem_step = _make_step_setter("stem_step", "stem_steps")
    set_stem_angle_deg = _make_setter("stem_angle_deg", _finite)
    set_handlebar_horizontal_range_mm = _make_setter(
        "handlebar_horizontal_range_mm", _range
    )
    set_handlebar_horizontal_steps = _make_steps_setter(
        "handlebar_horizontal_steps", "handlebar_horizontal_step"
    )
    set_handlebar_horizontal_step = _make_step_setter(
        "handlebar_horizontal_step", "handlebar_horizontal_steps"
    )
    set_handlebar_slide_angle_deg = _make_setter("handlebar_slide_angle_deg", _finite)
    set_crank_length_mm = _make_setter("crank_length_mm", _positive)
    set_pedal_size_mm = _make_setter("pedal_size_mm", _pedal_size)
    set_handlebar_diameter_mm = _make_setter("handlebar_diameter_mm", _positive)
    set_handlebar_width_mm = _make_setter("handlebar_width_mm", _positive)
    set_horn_pitch_deg = _make_setter("horn_pitch_deg", _finite)
    set_horn_roll_deg = _make_setter("horn_roll_deg", _finite)
    set_horn_length_mm = _make_setter("horn_length_mm", _positive)
    set_tube_diameter_mm = _make_setter("tube_diameter_mm", _positive)
    set_seat_length_mm = _make_setter("seat_length_mm", _positive)
    set_seat_width_mm = _make_setter("seat_width_mm", _positive)
    set_crank_spacing_mm = _make_setter("crank_spacing_mm", _positive)
    set_frame_width_mm = _make_setter("frame_width_mm", _positive)
    set_frame_margin_mm = _make_setter("frame_margin_mm", _positive)
    set_screen_width_mm = _make_setter("screen_width_mm", _positive)
    set_screen_height_mm = _make_setter("screen_height_mm", _positive)
    set_screen_inclination_deg = _make_setter("screen_inclination_deg", _finite)
    set_screen_vertical_offset_mm = _make_setter("screen_vertical_offset_mm", _positive)
    set_screen_horizontal_offset_mm = _make_setter(
        "screen_horizontal_offset_mm", _positive
    )
    del _make_setter, _make_steps_setter, _make_step_setter

    # --------------------------------------------------------------- the model
    def _add_static(
        self,
        component: str,
        name: str,
        shapes,
        location=(0, 0, 0),
        orientation=(0, 0, 0),
        parent=None,
    ):
        mass = add_welded_body(
            self,
            name,
            shapes,
            self._densities[component],
            parent=parent,
            location=location,
            orientation=orientation,
        )
        self._component_masses[component] = (
            self._component_masses.get(component, 0.0) + mass
        )

    def _seat_shapes(self) -> list[Ellipsoid]:
        length = self._seat_length_mm * _MM
        width = self._seat_width_mm * _MM
        height = self._seat_height_mm * _MM
        return [
            Ellipsoid(
                center=(-0.16 * length, 0.0, 0.0),
                radii=(0.34 * length, 0.5 * height, 0.5 * width),
                color=_SADDLE_COLOR,
            ),
            Ellipsoid(
                center=(0.16 * length, -0.06 * height, 0.0),
                radii=(0.34 * length, 0.38 * height, 0.20 * width),
                color=_SADDLE_COLOR,
            ),
        ]

    def _clear_model(self) -> None:
        self.updForceSet().clearAndDestroy()
        self.updJointSet().clearAndDestroy()
        self.updBodySet().clearAndDestroy()
        self.finalizeFromProperties()

    def _build(self) -> None:
        if hasattr(self, "state"):
            self._clear_model()
        else:
            opensim.Model.__init__(self)
            self.setName("Bike")
        self._component_masses: dict[str, float] = {}
        tube_radius = self._tube_diameter_mm * _MM / 2
        axis = self._crank_axis_position_m
        stem_base, stem_top = self._stem_base_m, self._stem_top_at(self._stem_step)
        seat_tube_top = self._seat_tube_top_at(self._seat_tube_step)
        margin = self._frame_margin_mm * _MM
        half_width = self._frame_width_mm * _MM / 2

        # Frame (world coordinates): rail along x under the pedal axis, rear and
        # front feet, a vertical support to the pedal axis and a strut from the
        # pedal axis to the stem base.
        x_min = min(axis[0], stem_base[0], seat_tube_top[0]) - margin
        x_max = max(axis[0], stem_base[0]) + margin
        rail_y = tube_radius
        rail_z = axis[2]
        frame = [
            cylinder_between(
                (x_min, rail_y, rail_z), (x_max, rail_y, rail_z), tube_radius
            ),
            cylinder_between(
                (x_min, rail_y, rail_z - half_width),
                (x_min, rail_y, rail_z + half_width),
                tube_radius,
            ),
            cylinder_between(
                (x_max, rail_y, rail_z - half_width),
                (x_max, rail_y, rail_z + half_width),
                tube_radius,
            ),
            cylinder_between((axis[0], rail_y, rail_z), tuple(axis), tube_radius),
        ]
        if np.linalg.norm(stem_base - axis) > 1e-9:
            frame.append(cylinder_between(tuple(axis), tuple(stem_base), tube_radius))
        self._add_static("frame", "frame", frame)
        self._add_static(
            "seat_tube",
            "seat_tube",
            [cylinder_between(axis, seat_tube_top, tube_radius)],
        )
        if np.linalg.norm(stem_top - stem_base) < 1e-3:
            # Zero excursion: keep a minimal stub so the tube is not degenerate.
            angle = math.radians(self._stem_angle_deg)
            stem_base = stem_top - 1e-3 * np.array(
                [math.cos(angle), math.sin(angle), 0.0]
            )
        self._add_static(
            "stem", "stem", [cylinder_between(stem_base, stem_top, tube_radius)]
        )

        # Surface on which the handlebar slides, from the lowest to the highest
        # offset of the configured range, along the slide direction.
        offsets = self.handlebar_horizontal_positions_mm * _MM
        slide = math.radians(self._handlebar_slide_angle_deg)
        low, high = float(offsets.min()), float(offsets.max())
        self._add_static(
            "handlebar_slide",
            "handlebar_slide",
            [Box(size=(max(high - low, 1e-3), tube_radius, 2 * tube_radius))],
            location=stem_top
            + (low + high) / 2 * np.array([math.cos(slide), math.sin(slide), 0.0]),
            orientation=(0.0, 0.0, slide),
        )
        self._add_static(
            "seat",
            "seat",
            self._seat_shapes(),
            location=self._seat_position_at(
                self._seat_tube_step, self._seat_horizontal_step
            ),
        )

        radius = self._handlebar_diameter_mm * _MM / 2
        width = self._handlebar_width_mm * _MM
        horn = self._horn_length_mm * _MM
        handlebar_shapes = [Cylinder(radius=radius, length=width, axis="z")]
        for side in ("r", "l"):
            side_sign = 1.0 if side == "r" else -1.0
            start = np.array([0.0, 0.0, side_sign * width / 2])
            end = start + horn * _horn_direction(
                self._horn_pitch_deg, self._horn_roll_deg, side
            )
            handlebar_shapes.append(cylinder_between(start, end, radius))
        self._add_static(
            "handlebar",
            "handlebar",
            handlebar_shapes,
            location=self._handlebar_position_m,
        )
        inclination = math.radians(self._screen_inclination_deg)
        panel_rotation = inclination - math.pi / 2
        mount = cylinder_between(
            (0.0, 0.0, 0.0),
            (
                self._screen_horizontal_offset_mm * _MM,
                self._screen_vertical_offset_mm * _MM,
                0.0,
            ),
            _SCREEN_MOUNT_RADIUS_MM * _MM,
        )
        mount.color = _SCREEN_MOUNT_COLOR
        self._add_static(
            "handlebar",
            "screen_mount",
            [mount],
            location=self._handlebar_position_m,
        )
        screen_mount = self.getBodySet().get("screen_mount")
        self._add_static(
            "handlebar",
            "screen",
            [
                Box(
                    center=(0.0, self._screen_height_mm * _MM / 2, 0.0),
                    size=(
                        _SCREEN_THICKNESS_MM * _MM,
                        self._screen_height_mm * _MM,
                        self._screen_width_mm * _MM,
                    ),
                    color=_SCREEN_COLOR,
                )
            ],
            location=self._screen_base_m - self._handlebar_position_m,
            orientation=(0.0, 0.0, panel_rotation),
            parent=screen_mount,
        )

        self._build_drivetrain(axis)
        self._add_brake()
        self.finalizeConnections()
        self._reset_state()

    def _reset_state(self) -> None:
        """Re-initialize the system and restore the stored crank and pedal angles."""
        self.state = self.initSystem()
        for name, value in self._angles_deg.items():
            self.getCoordinateSet().get(name).setValue(self.state, math.radians(value))
        self.realizePosition(self.state)

    def fit_user(
        self,
        user: Any | None = None,
        shoe_height_cm: float = 3.0,
        trunk_angle_deg: float | None = 45.0,
        shoulder_flexion_deg: float = 75.0,
        elbow_flexion_deg: float = 15.0,
        allow_ik_adjustment: bool = True,
    ):
        """Fit ``user`` on the bike and return a :class:`~opensim_extensions.Framework`.

        The bike is adjusted in place (seat tube, seat slide, stem and handlebar
        slide steps, crank and pedal angles) and the user is posed on it; the
        returned framework contains ``user`` and ``bike`` with both feet welded
        to their pedals. If ``user`` is omitted, the 170 cm, 70 kg male
        reference user is fitted. See
        :func:`opensim_extensions.models.bike.fitting.fit_user` for the full
        procedure.

        Parameters
        ----------
        user : User or None, default=None
            The user to place on the bike; omitted uses the reference rider.
        shoe_height_cm
            Shoe sole height, added to the user's crotch height.
        trunk_angle_deg
            Forward trunk inclination from the vertical in degrees, positive
            towards +X. ``None`` uses the user's current trunk posture as the
            starting reference, which IK may adjust to reach the handlebar.
        shoulder_flexion_deg
            Desired shoulder flexion in degrees.
        elbow_flexion_deg
            Desired elbow flexion in degrees (0 = straight arm).
        allow_ik_adjustment
            If true, inverse kinematics may adjust the requested flexion
            angles; otherwise they remain fixed while other degrees of freedom
            can still be fitted.
        """
        from .fitting import fit_user

        return fit_user(
            self,
            user,
            shoe_height_cm,
            trunk_angle_deg,
            shoulder_flexion_deg,
            elbow_flexion_deg,
            allow_ik_adjustment,
        )

    def _build_drivetrain(self, axis) -> None:
        length = self._crank_length_mm * _MM
        spacing = self._crank_spacing_mm * _MM
        crank_height, crank_thickness = _CRANK_BAR
        density = self._densities["cranks"]

        def crank_bar(side: float) -> Box:
            return Box(
                center=(length / 2, 0, side * spacing / 2),
                size=(length, crank_height, crank_thickness),
            )

        # The right crank carries the axle; the left crank is welded to it, 180 degrees opposite.
        right, mass = add_pinned_body(
            self,
            "crank_right",
            [crank_bar(1), Cylinder(radius=_AXLE_RADIUS, length=spacing, axis="z")],
            density,
            "crank_angle",
            location=axis,
        )
        total = mass
        left_shapes = [crank_bar(-1)]
        left_mass = add_welded_body(
            self,
            "crank_left",
            left_shapes,
            density,
            parent=right,
            orientation=(0, 0, math.pi),
        )
        self._component_masses["cranks"] = total + left_mass

        pedal_length, pedal_width, pedal_height = (v * _MM for v in self._pedal_size_mm)
        outward = self.pedal_spacing_mm * _MM / 2 + pedal_width / 2
        pedal_masses = 0.0
        # The left crank frame is rotated by pi; counter-rotating the pedal joint
        # frame makes pedal angle 0 mean "pedal parallel to the ground" on both sides.
        for side, crank_name, sign in (
            ("right", "crank_right", 1.0),
            ("left", "crank_left", -1.0),
        ):
            crank = self.getBodySet().get(crank_name)
            _, mass = add_pinned_body(
                self,
                f"pedal_{side}",
                [Box(size=(pedal_length, pedal_height, pedal_width))],
                self._densities["pedals"],
                f"pedal_{side}_angle",
                parent=crank,
                location=(length, 0, sign * outward),
                orientation=(0, 0, 0 if sign > 0 else -math.pi),
            )
            pedal_masses += mass
        self._component_masses["pedals"] = pedal_masses

    def _add_brake(self) -> None:
        # Both pedals apply the brake force at the crank length; tanh keeps the
        # resistance smooth around zero speed.
        torque = 2.0 * self._brake_force_n * self._crank_length_mm * _MM
        brake = opensim.ExpressionBasedCoordinateForce(
            "crank_angle", f"-{torque:.10g}*tanh(qdot/0.1)"
        )
        brake.setName("pedal_brake")
        self.addForce(brake)


_PARAMETER_DOCS = {
    "seat_tube_angle_deg": "Directed angle from +X to the seat tube, about +Z (deg).",
    "seat_tube_range_mm": "(min, max) seat tube length from the pedal axis (mm).",
    "seat_tube_steps": "Number of seat-tube intervals; N intervals give N+1 positions.",
    "seat_tube_step": "Selected seat-tube position index in 0..seat_tube_steps.",
    "crank_axis_position_mm": "Pedal axis centre (x, y, z) in world coordinates (mm).",
    "seat_height_mm": "Seat thickness (mm).",
    "seat_horizontal_range_mm": "Seat slide range relative to the seat tube top (mm).",
    "seat_horizontal_steps": "Intervals of the seat slide range.",
    "seat_horizontal_step": "Selected seat slide index.",
    "seat_slide_angle_deg": "Seat slide direction from +X, counterclockwise about +Z (deg).",
    "stem_origin_mm": "Stem base (x, y, z) relative to the pedal axis (mm).",
    "stem_excursion_mm": "Stem extension range [0, excursion] (mm).",
    "stem_steps": "Intervals of the stem extension.",
    "stem_step": "Selected stem extension index.",
    "stem_angle_deg": "Directed angle from +X to the stem, counterclockwise about +Z (deg).",
    "handlebar_horizontal_range_mm": "Handlebar slide range relative to the stem top (mm).",
    "handlebar_horizontal_steps": "Intervals of the handlebar slide range.",
    "handlebar_horizontal_step": "Selected handlebar slide index.",
    "handlebar_slide_angle_deg": "Handlebar slide direction from +X, counterclockwise about +Z (deg).",
    "crank_length_mm": "Crank length (mm).",
    "pedal_size_mm": "Pedal (length, width, height) (mm).",
    "handlebar_diameter_mm": "Diameter of the cross bar and horns (mm).",
    "handlebar_width_mm": "Cross bar width (mm).",
    "horn_pitch_deg": "Horn pitch; positive angles raise both horn tips (deg).",
    "horn_roll_deg": "Horn roll; positive angles turn both horns medially (deg).",
    "horn_length_mm": "Length of each horn (mm).",
    "brake_force_n": "Brake force at each pedal (N).",
    "tube_diameter_mm": "Diameter of the frame and seat/stem tubes (mm).",
    "seat_length_mm": "Seat length along x (mm).",
    "seat_width_mm": "Seat width along z (mm).",
    "crank_spacing_mm": "Distance between the two crank centres (mm).",
    "frame_width_mm": "Width of the ground frame feet (mm).",
    "frame_margin_mm": "Frame overhang beyond the pedal axis and stem (mm).",
    "screen_width_mm": "Screen width across the handlebar (mm).",
    "screen_height_mm": "Screen height along the panel (mm).",
    "screen_inclination_deg": "Directed angle from +X to the screen's upward direction, about +Z (deg).",
    "screen_vertical_offset_mm": "Positive screen-base offset above the handlebar centre (mm).",
    "screen_horizontal_offset_mm": "Positive screen-base offset forward from the handlebar centre (mm).",
}
for _name, _doc in _PARAMETER_DOCS.items():
    setattr(Bike, _name, _parameter_property(_name, _doc))

for _component in COMPONENTS:
    setattr(Bike, f"{_component}_density", _density_property(_component))
