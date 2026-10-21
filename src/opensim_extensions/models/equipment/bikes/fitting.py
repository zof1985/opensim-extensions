"""Place an anthropometric :class:`~opensim_extensions.User` on a :class:`Bike`."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from ....bodies.shapes import matrix_to_euler_xyz
from ....framework import Framework
from ....opensim import import_opensim
from .bike import _horn_direction

__all__ = ["fit_user"]

opensim = import_opensim()

_TOLERANCE_M = 5e-3
_BOTTOM_ANGLES = np.radians(np.arange(0.0, 360.0, 1.0))
_FOOT_AXIS_ALIGNMENT_WEIGHT = 0.038
_FOOT_PLANE_ALIGNMENT_WEIGHT = 0.9
_PRESERVED_POSTURE_COORDINATES = (
    "neck_base_flexion",
    "neck_base_lateral_flexion",
    "neck_flexion",
    "neck_lateral_flexion",
    "neck_rotation",
    "wrist_flex_r",
    "wrist_dev_r",
    "wrist_flex_l",
    "wrist_dev_l",
)


def _pose(frame: Any, state: Any) -> tuple[np.ndarray, np.ndarray]:
    transform = frame.getTransformInGround(state)
    rotation = np.array([[transform.R().get(i, j) for j in range(3)] for i in range(3)])
    position = np.array([transform.p().get(i) for i in range(3)])
    return rotation, position


def _vec(value: Any) -> np.ndarray:
    return np.array([value.get(i) for i in range(3)])


def _pitch(rotation: np.ndarray, neutral: np.ndarray) -> float:
    """Rotation about z of ``rotation`` relative to the ``neutral`` (flat foot) orientation."""
    relative = rotation @ neutral.T
    return math.atan2(relative[1, 0], relative[0, 0])


class _Rider:
    """Kinematic access to the user's coordinates and landmarks (radians, metres)."""

    def __init__(self, user: Any):
        self.user = user
        self.coordinates = user.getCoordinateSet()
        self.initial_lumbar = self.coordinates.get("lumbar_extension").getValue(
            user.state
        )
        self.initial_preserved_posture = {
            name: self.coordinates.get(name).getValue(user.state)
            for name in _PRESERVED_POSTURE_COORDINATES
        }
        self.neutral()
        self.hip_offset = self._mid_hip() - self._pelvis_translation()
        self.foot = {}
        for side in ("r", "l"):
            self.foot[side] = {
                "neutral": self.body_rotation(f"calcn_{side}"),
            }
        self.palm = {
            "r": float(user.right_palm_length),
            "l": float(user.left_palm_length),
        }

    @property
    def state(self) -> Any:
        return self.user.state

    def value(self, name: str) -> float:
        return self.coordinates.get(name).getValue(self.state)

    def bounds(self, name: str) -> tuple[float, float]:
        coordinate = self.coordinates.get(name)
        return coordinate.getRangeMin(), coordinate.getRangeMax()

    def set(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.coordinates.get(name).setValue(self.state, float(value), False)
        self.user.realizePosition(self.state)

    def neutral(self) -> None:
        """Straight standing posture; only the pelvis translation is kept."""
        keep = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}
        values = {}
        for index in range(self.coordinates.getSize()):
            name = self.coordinates.get(index).getName()
            if name not in keep:
                values[name] = 0.0
        values.update(self.initial_preserved_posture)
        self.set(values)

    def marker(self, name: str) -> np.ndarray:
        return _vec(self.user.getMarkerSet().get(name).getLocationInGround(self.state))

    def body_rotation(self, name: str) -> np.ndarray:
        return _pose(self.user.getBodySet().get(name), self.state)[0]

    def station(self, body: str, local: np.ndarray) -> np.ndarray:
        frame = self.user.getBodySet().get(body)
        return _vec(frame.findStationLocationInGround(self.state, opensim.Vec3(*local)))

    def _pelvis_translation(self) -> np.ndarray:
        return np.array(
            [self.value(n) for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")]
        )

    def _mid_hip(self) -> np.ndarray:
        return (np.asarray(self.user.right_hip) + np.asarray(self.user.left_hip)) / 2

    def place_pelvis(self, hip_target: np.ndarray) -> None:
        translation = hip_target - self.hip_offset
        self.set(dict(zip(("pelvis_tx", "pelvis_ty", "pelvis_tz"), translation)))

    def trunk_angle(self, lumbar: float) -> float:
        """Forward inclination (rad) from the vertical of the mid-hip to mid-shoulder line."""
        self.set({"lumbar_extension": lumbar})
        centers = self.user.joint_centers
        hips = (np.asarray(centers["right_hip"]) + np.asarray(centers["left_hip"])) / 2
        shoulders = (
            np.asarray(centers["right_shoulder"]) + np.asarray(centers["left_shoulder"])
        ) / 2
        line = shoulders - hips
        return math.atan2(line[0], line[1])

    def lumbar_for_trunk(self, angle: float) -> float:
        """Lumbar extension giving a trunk inclination of ``angle`` (rad), pelvis upright."""
        low, high = self.bounds("lumbar_extension")
        grid = np.linspace(low, high, 41)
        errors = [abs(self.trunk_angle(q) - angle) for q in grid]
        centre = grid[int(np.argmin(errors))]
        step = grid[1] - grid[0]
        result = minimize_scalar(
            lambda q: abs(self.trunk_angle(q) - angle),
            bounds=(max(low, centre - step), min(high, centre + step)),
            method="bounded",
            options={"xatol": 1e-8},
        )
        return float(result.x)

    def sync_couplers(self) -> None:
        """Set the dependent (patella) coordinates from their coupling functions."""
        constraints = self.user.getConstraintSet()
        for index in range(constraints.getSize()):
            coupler = opensim.CoordinateCouplerConstraint.safeDownCast(
                constraints.get(index)
            )
            if coupler is None:
                continue
            names = coupler.getIndependentCoordinateNames()
            values = opensim.Vector(names.getSize(), 0.0)
            for k in range(names.getSize()):
                values.set(k, self.value(names.get(k)))
            self.coordinates.get(coupler.getDependentCoordinateName()).setValue(
                self.state, coupler.getFunction().calcValue(values), False
            )
        self.user.realizePosition(self.state)


def _multistart(residual, starts, lower, upper):
    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            xtol=1e-12,
            ftol=1e-12,
        )
        if best is None or result.cost < best.cost:
            best = result
    return best


class _LegFit:
    """Three-dimensional leg placement for one seat position."""

    def __init__(self, rider: _Rider, bike: Any, shoe: float):
        self.rider = rider
        self.shoe = shoe
        self.half_pedal = bike._pedal_size_mm[2] * 1e-3 / 2
        self.axis = bike._crank_axis_position_m
        self.crank = bike._crank_length_mm * 1e-3
        self.lateral = bike._pedal_lateral_offset_m
        self.half_pedal_width = bike._pedal_size_mm[1] * 1e-3 / 2

    def solve(self, hip_target: np.ndarray) -> dict[str, Any]:
        rider = self.rider
        rider.neutral()
        rider.place_pelvis(hip_target)

        def leg(side: str):
            """Residual and coordinate names of one leg.

            The toe marker represents the medial/first-metatarsal landmark and
            goes on the inner pedal edge. The toe/MT5 midpoint projects to the
            pedal centre in its local plane. The sole plane is aligned with the
            pedal's top plane, and the heel-to-forefoot axis with its
            longitudinal axis. The subtalar coordinate can correct sole roll.
            The left ankle is neutral: the left pedal takes the foot pitch.
            """
            sign = 1.0 if side == "r" else -1.0
            names = [
                f"hip_flexion_{side}",
                f"hip_adduction_{side}",
                f"hip_rotation_{side}",
                f"knee_angle_{side}",
            ]
            names += [f"ankle_angle_{side}"] if side == "r" else []
            names += [f"subtalar_angle_{side}"]
            pedal = (self.axis + sign * np.array([self.crank, 0.0, 0.0]))[:2]
            pedal_center_z = self.axis[2] + sign * (
                self.lateral - self.half_pedal_width
            )
            inner_edge_z = pedal_center_z - sign * self.half_pedal_width

            def residual_parts(q):
                values = dict(zip(names, q))
                values.setdefault(f"ankle_angle_{side}", 0.0)
                rider.set(values)
                heel = rider.marker(f"{side.upper()}CAL")
                mt5 = rider.marker(f"{side.upper()}MT5")
                toe = rider.marker(f"{side.upper()}TOE")
                midpoint = (mt5 + toe) / 2
                rotation = rider.body_rotation(f"calcn_{side}")
                neutral = rider.foot[side]["neutral"]
                pitch = _pitch(rotation, neutral)
                pedal_pitch = pitch if side == "l" else 0.0
                normal = np.array([-math.sin(pedal_pitch), math.cos(pedal_pitch)])
                tangent = np.array([math.cos(pedal_pitch), math.sin(pedal_pitch)])
                forefoot_offset = midpoint[:2] - pedal
                forefoot_clearance = float(np.dot(forefoot_offset, normal))
                forefoot_longitudinal_offset = float(np.dot(forefoot_offset, tangent))
                foot_axis = midpoint - heel
                foot_axis /= np.linalg.norm(foot_axis)
                pedal_axis = np.array([tangent[0], tangent[1], 0.0])
                pedal_normal = np.array([normal[0], normal[1], 0.0])
                sole_normal = np.cross(toe - heel, mt5 - heel)
                sole_normal /= np.linalg.norm(sole_normal)
                if np.dot(sole_normal, pedal_normal) < 0:
                    sole_normal = -sole_normal
                plane_alignment = sole_normal - pedal_normal
                axis_alignment = foot_axis - pedal_axis
                contact = np.array(
                    [
                        forefoot_clearance - self.half_pedal - self.shoe,
                        toe[2] - inner_edge_z,
                        forefoot_longitudinal_offset,
                        midpoint[2] - pedal_center_z,
                    ]
                )
                orientation = np.concatenate(
                    (
                        _FOOT_PLANE_ALIGNMENT_WEIGHT * plane_alignment,
                        _FOOT_AXIS_ALIGNMENT_WEIGHT * axis_alignment,
                    )
                )
                return contact, orientation

            def residual(q):
                contact, orientation = residual_parts(q)
                return np.concatenate((contact, orientation))

            def contact_residual(q):
                contact, _ = residual_parts(q)
                return contact

            return names, residual, contact_residual

        solutions = {}
        for side in ("r", "l"):
            names, residual, contact_residual = leg(side)
            bounds = np.array([rider.bounds(n) for n in names])
            starts = [
                [h, 0.0, 0.0, k] + ([0.0] if side == "r" else []) + [0.0]
                for h in (0.6, 1.1, 1.6)
                for k in (0.8, 1.5, 2.1)
            ]
            solutions[side] = (
                names,
                residual,
                contact_residual,
                _multistart(residual, starts, bounds[:, 0], bounds[:, 1]),
            )
        rider.neutral()
        rider.place_pelvis(hip_target)
        values = {"ankle_angle_l": 0.0}
        for names, _, _, solution in solutions.values():
            values.update(zip(names, solution.x))
        rider.set(values)
        pitch_l = _pitch(rider.body_rotation("calcn_l"), rider.foot["l"]["neutral"])
        return {
            "values": values,
            "knee_x": float(rider.user.right_knee[0]),
            "residual": float(
                max(
                    np.abs(contact_residual(sol.x)).max()
                    for _, _, contact_residual, sol in solutions.values()
                )
            ),
            "left_pedal_pitch": pitch_l,
        }


def _choose_tube_step(bike: Any, target: float) -> int:
    nominal = int(np.argmin(np.abs(bike.seat_slide_positions_mm)))
    lift = np.array([0.0, bike._seat_height_mm * 1e-3 / 2, 0.0])
    axis = bike._crank_axis_position_m
    crank = bike._crank_length_mm * 1e-3
    pedal_tops = axis[None, :2] + crank * np.stack(
        [np.cos(_BOTTOM_ANGLES), np.sin(_BOTTOM_ANGLES)], axis=1
    )
    pedal_tops[:, 1] += bike._pedal_size_mm[2] * 1e-3 / 2
    errors = []
    for step in range(bike._seat_tube_steps + 1):
        seat_top = (bike._seat_position_at(step, nominal) + lift)[:2]
        errors.append(abs(np.linalg.norm(pedal_tops - seat_top, axis=1).max() - target))
    return int(np.argmin(errors))


def _handlebar_pose(
    bike: Any, tube_step: int, slide_step: int
) -> tuple[np.ndarray, np.ndarray]:
    return bike._handlebar_position_at(tube_step, slide_step), np.eye(3)


def _horn_local(bike: Any, side: str) -> np.ndarray:
    sign = 1.0 if side == "r" else -1.0
    start = np.array([0.0, 0.0, sign * bike._handlebar_width_mm * 1e-3 / 2])
    direction = _horn_direction(bike._horn_pitch_deg, bike._horn_roll_deg, side)
    return start + bike._horn_length_mm * 1e-3 / 2 * direction


class _ArmFit:
    """Fit both hands while sharing shoulder and elbow flexion bilaterally."""

    NAMES = (
        "arm_flex_r",
        "arm_add_r",
        "elbow_flex_r",
        "arm_flex_l",
        "arm_add_l",
        "elbow_flex_l",
    )
    # Forearms in the neutral position (thumbs up): 90 degrees from full supination.
    NEUTRAL_FOREARM = {"pro_sup_r": math.pi / 2, "pro_sup_l": math.pi / 2}

    def __init__(self, rider: _Rider, lumbar: float, shoulder: float, elbow: float):
        self.rider = rider
        self.lumbar = lumbar
        self.shoulder = shoulder
        self.elbow = elbow
        self.lower = np.array([rider.bounds(n)[0] for n in self.NAMES])
        self.upper = np.array([rider.bounds(n)[1] for n in self.NAMES])
        self.reference = np.array([shoulder, 0.0, elbow, shoulder, 0.0, elbow])

    def hands(self, x: np.ndarray) -> np.ndarray:
        rider = self.rider
        lumbar = float(x[len(self.NAMES)]) if len(x) > len(self.NAMES) else self.lumbar
        rider.set(
            dict(zip(self.NAMES, x[: len(self.NAMES)]))
            | {"lumbar_extension": lumbar}
            | self.NEUTRAL_FOREARM
        )
        return np.concatenate(
            [
                rider.station(f"hand_{s}", np.array([0.0, -rider.palm[s] / 2, 0.0]))
                for s in ("r", "l")
            ]
        )

    def reference_hands(self) -> np.ndarray:
        """Hand positions with the requested shoulder and elbow flexions and no adduction."""
        return self.hands(np.clip(self.reference, self.lower, self.upper))

    def solve(
        self,
        grips: np.ndarray,
        *,
        adjust_trunk: bool = False,
    ) -> dict[str, Any]:
        reference = np.clip(self.reference, self.lower, self.upper)
        variable_indices = np.array([0, 1, 2, 4])
        lower = self.lower[variable_indices]
        upper = self.upper[variable_indices]
        parameter_reference = reference[variable_indices]
        weights = np.array([1.0, 1e-2, 1.0, 1.0, 1e-2, 1.0])
        if adjust_trunk:
            lumbar_bounds = np.array([self.rider.bounds("lumbar_extension")])
            reference = np.concatenate((reference, [self.lumbar]))
            parameter_reference = np.concatenate((parameter_reference, [self.lumbar]))
            lower = np.concatenate((lower, lumbar_bounds[:, 0]))
            upper = np.concatenate((upper, lumbar_bounds[:, 1]))
            weights = np.concatenate((weights, [1.0]))

        def expand(parameters: np.ndarray) -> np.ndarray:
            arms = np.array(
                [
                    parameters[0],
                    parameters[1],
                    parameters[2],
                    parameters[0],
                    parameters[3],
                    parameters[2],
                ]
            )
            return np.concatenate((arms, [parameters[4]])) if adjust_trunk else arms

        def residual(parameters: np.ndarray) -> np.ndarray:
            values = expand(parameters)
            hand_error_mm = (self.hands(values) - grips) * 1000.0
            posture_error = 0.1 * np.sqrt(weights) * (values - reference)
            return np.concatenate((hand_error_mm, posture_error))

        starts = [
            parameter_reference,
            np.array([1.0, 0.0, 0.3, 0.0]),
            np.array([0.4, 0.0, 1.2, 0.0]),
        ]
        if adjust_trunk:
            starts = [
                (
                    np.concatenate((start, [self.lumbar]))
                    if len(start) == len(variable_indices)
                    else start
                )
                for start in starts
            ]
        result = _multistart(residual, starts, lower, upper)
        x = expand(result.x)
        residual_mm = float(np.abs(self.hands(x) - grips).max() * 1000.0)
        lumbar = float(x[len(self.NAMES)]) if adjust_trunk else self.lumbar
        return {
            "values": dict(zip(self.NAMES, x))
            | {"lumbar_extension": lumbar}
            | self.NEUTRAL_FOREARM,
            "residual": residual_mm / 1000.0,
            "flexion_deg": math.degrees(x[0]),
            "elbow_deg": math.degrees(x[2]),
        }

    def solve_with_fixed_flexions(self, grips: np.ndarray) -> dict[str, Any]:
        """Keep requested flexion coordinates fixed while fitting shoulder adduction."""
        reference = np.clip(self.reference, self.lower, self.upper)
        free_indices = np.array([1, 4])
        lower = self.lower[free_indices]
        upper = self.upper[free_indices]

        def residual(adduction: np.ndarray) -> np.ndarray:
            values = reference.copy()
            values[free_indices] = adduction
            hand_error = self.hands(values) - grips
            regularization = 0.1 * (adduction - reference[free_indices])
            return np.concatenate((hand_error, regularization))

        starts = (
            reference[free_indices],
            np.array([lower[0], upper[1]]),
            np.array([upper[0], lower[1]]),
        )
        result = _multistart(residual, starts, lower, upper)
        values = reference.copy()
        values[free_indices] = result.x
        hand_error = float(np.abs(self.hands(values) - grips).max())
        return {
            "values": dict(zip(self.NAMES, values))
            | {"lumbar_extension": self.lumbar}
            | self.NEUTRAL_FOREARM,
            "residual": hand_error,
            "flexion_deg": math.degrees((values[0] + values[3]) / 2),
            "elbow_deg": math.degrees((values[2] + values[5]) / 2),
        }


def _mount(
    parent_body: Any, name: str, rotation: np.ndarray, position: np.ndarray
) -> None:
    """Add (or update) a physical offset frame on ``parent_body``."""
    frame = opensim.PhysicalOffsetFrame.safeDownCast(
        next((c for c in parent_body.getComponentsList() if c.getName() == name), None)
    )
    if frame is None:
        frame = opensim.PhysicalOffsetFrame()
        frame.setName(name)
        frame.setParentFrame(parent_body)
        parent_body.addComponent(frame)
    frame.set_translation(opensim.Vec3(*position))
    frame.set_orientation(opensim.Vec3(*matrix_to_euler_xyz(rotation)))


def fit_user(
    bike: Any,
    user: Any | None = None,
    shoe_height_mm: float = 30.0,
    trunk_angle_deg: float | None = 45.0,
    shoulder_flexion_deg: float = 75.0,
    elbow_flexion_deg: float = 15.0,
    allow_ik_adjustment: bool = True,
) -> Framework:
    """Fit ``user`` on ``bike`` and return a framework of the two models.

    Procedure (the bike and the user are modified in place):

    1. The right crank is at maximum advancement (``crank_angle = 0``). The
       toe marker (used as the medial forefoot landmark) is aligned with the
       inner edge of each pedal, and the midpoint between the toe and MT5
       markers is over the pedal centre. Its distance from the pedal top,
       measured along the pedal normal, is ``shoe_height_mm``. The right foot
       (and pedal) is parallel to the ground; the left pedal follows the foot
       pitch while the left ankle stays neutral (flexion 0). Both feet are
       welded to their pedals, which stay free to rotate; the forefoot
       midpoint therefore retains its radial offset from each pedal axis.
    2. The seat tube step is the one for which the maximum distance, over a
       crank revolution, between the seat top (at the slide step closest to 0)
       and the pedal top surface is closest to the user's crotch height plus
       the shoe height.
    3. With the tube fixed, the seat slide step is the one for which the right
       knee joint centre is closest to the vertical of the right pedal axis.
       The user's hip centres sit on the seat top surface, pelvis upright.
    4. The trunk starts at the user's current ``lumbar_extension`` when
       ``trunk_angle_deg`` is ``None``, otherwise at the requested inclination
       (forward from the vertical, of the mid-hip to mid-shoulder line;
       0 = upright, pelvis upright). A specified trunk angle is kept fixed;
       when it is ``None``, inverse kinematics may adapt it to place the hands
       on the horns while the handlebar tube is selected to bring the handlebar centre
       near seat-top height. Shoulder flexion and elbow flexion are shared
       between sides; shoulder adduction remains independently adjustable.

    The solution is the best available among the discrete steps; a
    ``UserWarning`` is issued when a constraint cannot be met within 5 mm.
    The achieved values are stored in ``bike.fit_report``. Modifying the bike
    afterwards rebuilds it and invalidates the foot couplings, so call
    ``fit_user`` again.

    Parameters
    ----------
    user : User or None, default=None
        Anthropometric user (facing +x, right side at +z). If omitted, uses a
        170 cm, 70 kg male reference user.
    shoe_height_mm : float, default=30.0
        Sole height of the shoes in millimetres.
    trunk_angle_deg : float or None, default=45.0
        Fixed trunk inclination in degrees. ``None`` starts from the user's
        current lumbar posture and lets inverse kinematics adjust the trunk.
    shoulder_flexion_deg : float, default=75.0
        Desired shoulder flexion (``arm_flex``) in degrees.
    elbow_flexion_deg : float, default=15.0
        Desired elbow flexion (``elbow_flex``) in degrees; 0 is a straight arm.
    allow_ik_adjustment : bool, default=True
        Allow inverse kinematics to adjust shoulder and elbow flexions and,
        when ``trunk_angle_deg`` is ``None``, the trunk. When false, these
        angles remain fixed while shoulder adduction can still be fitted.

    Returns
    -------
    Framework
        Framework with members ``user`` and ``bike``, and the couplings
        ``foot_right`` and ``foot_left``.
    """
    from ...user import User

    if user is None:
        user = User("M", height_cm=170.0, mass_kg=70.0)
    elif not isinstance(user, User):
        raise TypeError("user must be a User instance")
    if not isinstance(allow_ik_adjustment, bool):
        raise TypeError("allow_ik_adjustment must be a bool")
    shoe = float(shoe_height_mm)
    if not math.isfinite(shoe) or shoe < 0:
        raise ValueError("shoe_height_mm must be a non-negative finite value")
    shoe *= 1e-3
    for label, value in (
        ("shoulder_flexion_deg", shoulder_flexion_deg),
        ("elbow_flexion_deg", elbow_flexion_deg),
        ("trunk_angle_deg", 0.0 if trunk_angle_deg is None else trunk_angle_deg),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")

    rider = _Rider(user)
    crotch_height = user._reference.values["crotchheight"] * 1e-3
    seat_lift = np.array([0.0, bike._seat_height_mm * 1e-3 / 2, 0.0])

    tube_step = _choose_tube_step(bike, crotch_height + shoe)
    bike.set_seat_tube_step(tube_step)

    legs = _LegFit(rider, bike, shoe)
    axis = bike._crank_axis_position_m
    pedal_x = axis[0] + bike._crank_length_mm * 1e-3
    candidates = []
    for slide in range(bike._seat_slide_steps + 1):
        hip = bike._seat_position_at(tube_step, slide) + seat_lift
        solution = legs.solve(hip)
        error = abs(solution["knee_x"] - pedal_x)
        candidates.append((error + 10 * solution["residual"], error, slide, solution))
    _, knee_error, slide_step, legs_solution = min(candidates, key=lambda item: item[0])
    bike.set_seat_slide_step(slide_step)
    hip = bike._seat_position_at(tube_step, slide_step) + seat_lift
    legs_solution = legs.solve(hip)
    if legs_solution["residual"] > _TOLERANCE_M:
        warnings.warn(
            "the feet cannot be placed on the pedals within 5 mm with this bike and user",
            UserWarning,
            stacklevel=3,
        )

    seat_top_y = hip[1]
    if trunk_angle_deg is None:
        lumbar = rider.initial_lumbar
    else:
        lumbar = rider.lumbar_for_trunk(math.radians(trunk_angle_deg))
    arms = _ArmFit(
        rider,
        lumbar,
        math.radians(shoulder_flexion_deg),
        math.radians(elbow_flexion_deg),
    )
    if not allow_ik_adjustment:
        for coordinate_name, angle in (
            ("arm_flex_r", shoulder_flexion_deg),
            ("arm_flex_l", shoulder_flexion_deg),
            ("elbow_flex_r", elbow_flexion_deg),
            ("elbow_flex_l", elbow_flexion_deg),
        ):
            low, high = rider.bounds(coordinate_name)
            value = math.radians(angle)
            if not low <= value <= high:
                raise ValueError(
                    f"{coordinate_name} must be within [{math.degrees(low):g}, "
                    f"{math.degrees(high):g}] degrees when IK adjustment is disabled"
                )

    def fit_arms(grips: np.ndarray) -> dict[str, Any]:
        if allow_ik_adjustment:
            return arms.solve(grips, adjust_trunk=trunk_angle_deg is None)
        return arms.solve_with_fixed_flexions(grips)

    def grips_at(tube: int, slide: int) -> np.ndarray:
        centre, rotation = _handlebar_pose(bike, tube, slide)
        return np.concatenate(
            [centre + rotation @ _horn_local(bike, s) for s in ("r", "l")]
        )

    steps = [
        (tube, slide)
        for tube in range(bike._handlebar_tube_steps + 1)
        for slide in range(bike._handlebar_slide_steps + 1)
    ]
    if trunk_angle_deg is None:
        # The starting trunk posture is a reference for the IK; the handlebar tube
        # keeps the handlebar near seat height while the slide and trunk fit the hands.
        options = []
        for slide in range(bike._handlebar_slide_steps + 1):
            tube = int(
                np.argmin(
                    [
                        abs(bike._handlebar_position_at(step, slide)[1] - seat_top_y)
                        for step in range(bike._handlebar_tube_steps + 1)
                    ]
                )
            )
            arm = fit_arms(grips_at(tube, slide))
            if allow_ik_adjustment:
                score = (
                    abs(arm["flexion_deg"] - shoulder_flexion_deg)
                    + abs(arm["elbow_deg"] - elbow_flexion_deg)
                    + 1e3 * max(arm["residual"] - _TOLERANCE_M, 0.0)
                )
            else:
                score = arm["residual"]
            options.append((score, slide, tube, arm))
        _, handlebar_slide, tube_step, arm_solution = min(
            options, key=lambda item: item[0]
        )
    else:
        # The trunk and arm angles define where the hands are: take the closest handlebar step.
        wanted = arms.reference_hands().reshape(2, 3)[:, :2]
        errors = [
            np.linalg.norm(grips_at(*step).reshape(2, 3)[:, :2] - wanted, axis=1).mean()
            for step in steps
        ]
        tube_step, handlebar_slide = steps[int(np.argmin(errors))]
        arm_solution = fit_arms(grips_at(tube_step, handlebar_slide))
    bike.set_handlebar_tube_step(tube_step)
    bike.set_handlebar_slide_step(handlebar_slide)
    height_error = bike._handlebar_position_m[1] - seat_top_y
    if arm_solution["residual"] > _TOLERANCE_M or (
        trunk_angle_deg is None and abs(height_error) > 0.02
    ):
        warnings.warn(
            "the handlebar cannot be set as requested with the hands on the horns "
            f"(height error {height_error * 1e3:.0f} mm, hand error {arm_solution['residual'] * 1e3:.0f} mm)",
            UserWarning,
            stacklevel=3,
        )
    # Final posture.
    rider.neutral()
    rider.place_pelvis(hip)
    rider.set(legs_solution["values"])
    rider.set(arm_solution["values"])
    trunk_deg = math.degrees(rider.trunk_angle(rider.value("lumbar_extension")))
    rider.sync_couplers()
    user.update_state()

    bike._angles_deg.update(
        crank_angle=0.0,
        pedal_right_angle=0.0,
        pedal_left_angle=math.degrees(legs_solution["left_pedal_pitch"]),
    )
    bike._reset_state()

    for side, name in (("r", "right"), ("l", "left")):
        calcn = user.getBodySet().get(f"calcn_{side}")
        pedal = bike.getBodySet().get(f"pedal_{name}")
        foot_rotation, foot_position = _pose(calcn, user.state)
        pedal_rotation, pedal_position = _pose(pedal, bike.state)
        _mount(calcn, f"pedal_mount_{side}", np.eye(3), np.zeros(3))
        _mount(
            pedal,
            f"foot_mount_{name}",
            pedal_rotation.T @ foot_rotation,
            pedal_rotation.T @ (foot_position - pedal_position),
        )
    user.reinitialize()
    bike.finalizeConnections()
    bike._reset_state()

    framework = Framework(user=user, bike=bike)
    for side, name in (("r", "right"), ("l", "left")):
        framework.couple_weld(
            ("user", f"/bodyset/calcn_{side}/pedal_mount_{side}"),
            ("bike", f"/bodyset/pedal_{name}/foot_mount_{name}"),
            name=f"foot_{name}",
        )
    framework.initSystem()

    bike.fit_report = {
        "seat_tube_step": tube_step,
        "seat_slide_step": slide_step,
        "handlebar_tube_step": tube_step,
        "handlebar_slide_step": handlebar_slide,
        "knee_to_pedal_axis_mm": (legs_solution["knee_x"] - pedal_x) * 1e3,
        "handlebar_height_error_mm": float(height_error) * 1e3,
        "shoulder_flexion_deg": arm_solution["flexion_deg"],
        "trunk_angle_deg": trunk_deg,
        "elbow_flexion_deg": arm_solution["elbow_deg"],
        "hand_error_mm": arm_solution["residual"] * 1e3,
        "left_pedal_angle_deg": bike.pedal_left_angle_deg,
    }
    return framework
