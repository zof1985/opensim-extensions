"""Place an anthropometric :class:`~opensim_extensions.User` on a :class:`Bike`."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize, minimize_scalar

from ...bodies.shapes import euler_xyz_to_matrix, matrix_to_euler_xyz
from ...framework import Framework
from ...opensim import import_opensim

__all__ = ["fit_user"]

opensim = import_opensim()

_TOLERANCE_M = 5e-3
_BOTTOM_ANGLES = np.radians(np.arange(0.0, 360.0, 1.0))
_FLEXION_TARGET_DEG = 90.0


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
        self.initial_lumbar = self.coordinates.get("lumbar_extension").getValue(user.state)
        self.neutral()
        self.hip_offset = self._mid_hip() - self._pelvis_translation()
        self.foot = {}
        for side in ("r", "l"):
            heel, toe, mt5 = (self.marker(f"{side.upper()}{name}") for name in ("CAL", "TOE", "MT5"))
            self.foot[side] = {
                "neutral": self.body_rotation(f"calcn_{side}"),
                # Height of the 5th metatarsal marker above the sole plane,
                # tangent to the lowest foot marker in the neutral posture.
                "mt5_height": float(mt5[1] - min(heel[1], toe[1], mt5[1])),
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
        self.set(values)

    def marker(self, name: str) -> np.ndarray:
        return _vec(self.user.getMarkerSet().get(name).getLocationInGround(self.state))

    def body_rotation(self, name: str) -> np.ndarray:
        return _pose(self.user.getBodySet().get(name), self.state)[0]

    def station(self, body: str, local: np.ndarray) -> np.ndarray:
        frame = self.user.getBodySet().get(body)
        return _vec(frame.findStationLocationInGround(self.state, opensim.Vec3(*local)))

    def _pelvis_translation(self) -> np.ndarray:
        return np.array([self.value(n) for n in ("pelvis_tx", "pelvis_ty", "pelvis_tz")])

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
        shoulders = (np.asarray(centers["right_shoulder"]) + np.asarray(centers["left_shoulder"])) / 2
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
            coupler = opensim.CoordinateCouplerConstraint.safeDownCast(constraints.get(index))
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
        result = least_squares(residual, np.clip(start, lower, upper), bounds=(lower, upper), xtol=1e-12, ftol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    return best


class _LegFit:
    """Sagittal-plane leg placement for one seat position."""

    def __init__(self, rider: _Rider, bike: Any, shoe: float):
        self.rider = rider
        self.shoe = shoe
        self.half_pedal = bike._pedal_size_mm[2] * 1e-3 / 2
        self.axis = bike.crank_axis_position
        self.crank = bike._crank_length_mm * 1e-3

    def _pedal_offset(self, side: str, pitch: float) -> np.ndarray:
        height = self.half_pedal + self.shoe + self.rider.foot[side]["mt5_height"]
        return np.array([-math.sin(pitch) * height, math.cos(pitch) * height])

    def solve(self, hip_target: np.ndarray) -> dict[str, Any]:
        rider = self.rider
        rider.neutral()
        rider.place_pelvis(hip_target)
        pedal_r = (self.axis + np.array([self.crank, 0.0, 0.0]))[:2]
        pedal_l = (self.axis - np.array([self.crank, 0.0, 0.0]))[:2]

        def right(q):
            rider.set({"hip_flexion_r": q[0], "knee_angle_r": q[1], "ankle_angle_r": q[2]})
            mt5 = rider.marker("RMT5")
            pitch = _pitch(rider.body_rotation("calcn_r"), rider.foot["r"]["neutral"])
            target = pedal_r + self._pedal_offset("r", 0.0)
            return np.array([mt5[0] - target[0], mt5[1] - target[1], pitch])

        def left(q):
            rider.set({"hip_flexion_l": q[0], "knee_angle_l": q[1], "ankle_angle_l": 0.0})
            mt5 = rider.marker("LMT5")
            pitch = _pitch(rider.body_rotation("calcn_l"), rider.foot["l"]["neutral"])
            target = pedal_l + self._pedal_offset("l", pitch)
            return np.array([mt5[0] - target[0], mt5[1] - target[1]])

        hip_r, knee_r, ankle_r = (rider.bounds(n) for n in ("hip_flexion_r", "knee_angle_r", "ankle_angle_r"))
        hip_l, knee_l = (rider.bounds(n) for n in ("hip_flexion_l", "knee_angle_l"))
        starts = [(h, k) for h in (0.6, 1.1, 1.6) for k in (0.8, 1.5, 2.1)]
        sol_r = _multistart(
            right,
            [(h, k, 0.0) for h, k in starts],
            np.array([hip_r[0], knee_r[0], ankle_r[0]]),
            np.array([hip_r[1], knee_r[1], ankle_r[1]]),
        )
        sol_l = _multistart(
            left,
            starts,
            np.array([hip_l[0], knee_l[0]]),
            np.array([hip_l[1], knee_l[1]]),
        )
        rider.neutral()
        rider.place_pelvis(hip_target)
        values = {
            "hip_flexion_r": sol_r.x[0],
            "knee_angle_r": sol_r.x[1],
            "ankle_angle_r": sol_r.x[2],
            "hip_flexion_l": sol_l.x[0],
            "knee_angle_l": sol_l.x[1],
            "ankle_angle_l": 0.0,
        }
        rider.set(values)
        pitch_l = _pitch(rider.body_rotation("calcn_l"), rider.foot["l"]["neutral"])
        return {
            "values": values,
            "knee_x": float(rider.user.right_knee[0]),
            "residual": float(max(np.abs(right(sol_r.x)).max(), np.abs(left(sol_l.x)).max())),
            "left_pedal_pitch": pitch_l,
        }


def _choose_tube_step(bike: Any, target: float) -> int:
    nominal = int(np.argmin(np.abs(bike.seat_horizontal_positions_mm)))
    lift = np.array([0.0, bike._seat_height_mm * 1e-3 / 2, 0.0])
    axis = bike.crank_axis_position
    crank = bike._crank_length_mm * 1e-3
    pedal_tops = axis[None, :2] + crank * np.stack([np.cos(_BOTTOM_ANGLES), np.sin(_BOTTOM_ANGLES)], axis=1)
    pedal_tops[:, 1] += bike._pedal_size_mm[2] * 1e-3 / 2
    errors = []
    for step in range(bike._seat_tube_steps + 1):
        seat_top = (bike._seat_position_at(step, nominal) + lift)[:2]
        errors.append(abs(np.linalg.norm(pedal_tops - seat_top, axis=1).max() - target))
    return int(np.argmin(errors))


def _handlebar_pose(bike: Any, stem_step: int, slide_step: int) -> tuple[np.ndarray, np.ndarray]:
    rotation = euler_xyz_to_matrix(
        (
            math.radians(bike._handlebar_roll_deg),
            math.radians(bike._handlebar_tilt_deg),
            math.radians(bike._handlebar_pitch_deg),
        )
    )
    return bike._handlebar_position_at(stem_step, slide_step), rotation


def _horn_local(bike: Any, side: str) -> np.ndarray:
    sign = 1.0 if side == "r" else -1.0
    return np.array([bike._horn_length_mm * 1e-3 / 2, 0.0, sign * bike._handlebar_width_mm * 1e-3 / 2])


class _ArmFit:
    """Arm placement with the trunk fixed and the hands on the horns.

    Per arm the unknowns are shoulder flexion, shoulder adduction and elbow
    flexion: three unknowns for the three hand coordinates. The small
    regularization towards the requested flexions only picks the solution
    among near-singular ones.
    """

    NAMES = ("arm_flex_r", "arm_add_r", "elbow_flex_r", "arm_flex_l", "arm_add_l", "elbow_flex_l")

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
        rider.set(dict(zip(self.NAMES, x)) | {"lumbar_extension": self.lumbar})
        return np.concatenate(
            [rider.station(f"hand_{s}", np.array([0.0, -rider.palm[s] / 2, 0.0])) for s in ("r", "l")]
        )

    def reference_hands(self) -> np.ndarray:
        """Hand positions with the requested shoulder and elbow flexions and no adduction."""
        return self.hands(np.clip(self.reference, self.lower, self.upper))

    def solve(self, grips: np.ndarray) -> dict[str, Any]:
        weights = np.array([1.0, 1e-2, 1.0, 1.0, 1e-2, 1.0])
        starts = (self.reference, np.array([1.0, 0.0, 0.3, 1.0, 0.0, 0.3]), np.array([0.4, 0.0, 1.2, 0.4, 0.0, 1.2]))
        best = None
        for start in starts:
            result = minimize(
                lambda x: float(np.sum(weights * (x - self.reference) ** 2)),
                np.clip(start, self.lower, self.upper),
                method="SLSQP",
                bounds=list(zip(self.lower, self.upper)),
                constraints={"type": "eq", "fun": lambda x: self.hands(x) - grips},
                options={"maxiter": 300, "ftol": 1e-10},
            )
            residual = float(np.abs(self.hands(result.x) - grips).max())
            key = (max(residual - _TOLERANCE_M, 0.0), result.fun)
            if best is None or key < best[0]:
                best = (key, residual, result)
        _, residual, result = best
        x = result.x
        return {
            "values": dict(zip(self.NAMES, x)) | {"lumbar_extension": self.lumbar},
            "residual": residual,
            "flexion_deg": math.degrees((x[0] + x[3]) / 2),
            "elbow_deg": math.degrees((x[2] + x[5]) / 2),
        }

def _mount(parent_body: Any, name: str, rotation: np.ndarray, position: np.ndarray) -> None:
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
    user: Any,
    shoe_height_cm: float = 3.0,
    trunk_angle_deg: float | None = None,
    shoulder_flexion_deg: float = 90.0,
    elbow_flexion_deg: float = 5.0,
) -> Framework:
    """Fit ``user`` on ``bike`` and return a framework of the two models.

    Procedure (the bike and the user are modified in place):

    1. The right crank is at maximum advancement (``crank_angle = 0``). Each
       foot rests on its pedal with the 5th-metatarsal marker above the pedal
       axis and the sole ``shoe_height_cm`` above the pedal top surface. The
       right foot (and pedal) is parallel to the ground; the left pedal is
       rotated so that the left ankle is neutral (flexion 0). Both feet are
       welded to their pedals, which stay free to rotate.
    2. The seat tube step is the one for which the maximum distance, over a
       crank revolution, between the seat top (at the slide step closest to 0)
       and the pedal top surface is closest to the user's crotch height plus
       the shoe height.
    3. With the tube fixed, the seat slide step is the one for which the right
       knee joint centre is closest to the vertical of the right pedal axis.
       The user's hip centres sit on the seat top surface, pelvis upright.
    4. The trunk is fixed: at the user's current ``lumbar_extension`` when
       ``trunk_angle_deg`` is ``None``, otherwise at the requested inclination
       (forward from the vertical, of the mid-hip to mid-shoulder line;
       0 = upright, pelvis upright). With ``trunk_angle_deg=None`` the stem
       step brings the handlebar centre as close as possible to the seat top
       height and the handlebar slide step gets the shoulder and elbow
       flexions closest to the requested ones. With ``trunk_angle_deg`` given
       the seat-height rule no longer applies: trunk, shoulder flexion and
       elbow flexion define where the hands are, and the stem and slide steps
       closest to that position are used.

    The solution is the best available among the discrete steps; a
    ``UserWarning`` is issued when a constraint cannot be met within 5 mm.
    The achieved values are stored in ``bike.fit_report``. Modifying the bike
    afterwards rebuilds it and invalidates the foot couplings, so call
    ``fit_user`` again.

    Parameters
    ----------
    user : User
        Anthropometric user (facing +x, right side at +z).
    shoe_height_cm : float, default=3.0
        Sole height of the shoes in centimetres.
    trunk_angle_deg : float or None, default=None
        Trunk inclination in degrees; ``None`` keeps the user's current lumbar posture.
    shoulder_flexion_deg : float, default=90.0
        Desired shoulder flexion (``arm_flex``) in degrees.
    elbow_flexion_deg : float, default=5.0
        Desired elbow flexion (``elbow_flex``) in degrees; 0 is a straight arm.

    Returns
    -------
    Framework
        Framework with members ``user`` and ``bike``, and the couplings
        ``foot_right`` and ``foot_left``.
    """
    from ..user import User

    if not isinstance(user, User):
        raise TypeError("user must be a User instance")
    shoe = float(shoe_height_cm)
    if not math.isfinite(shoe) or shoe < 0:
        raise ValueError("shoe_height_cm must be a non-negative finite value")
    shoe *= 1e-2
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
    report: dict[str, Any] = {}

    tube_step = _choose_tube_step(bike, crotch_height + shoe)
    bike.set_seat_tube_step(tube_step)

    legs = _LegFit(rider, bike, shoe)
    axis = bike.crank_axis_position
    pedal_x = axis[0] + bike._crank_length_mm * 1e-3
    candidates = []
    for slide in range(bike._seat_horizontal_steps + 1):
        hip = bike._seat_position_at(tube_step, slide) + seat_lift
        solution = legs.solve(hip)
        error = abs(solution["knee_x"] - pedal_x)
        candidates.append((error + 10 * solution["residual"], error, slide, solution))
    _, knee_error, slide_step, legs_solution = min(candidates, key=lambda item: item[0])
    bike.set_seat_horizontal_step(slide_step)
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
    trunk_deg = math.degrees(rider.trunk_angle(lumbar))
    arms = _ArmFit(rider, lumbar, math.radians(shoulder_flexion_deg), math.radians(elbow_flexion_deg))

    def grips_at(stem: int, slide: int) -> np.ndarray:
        centre, rotation = _handlebar_pose(bike, stem, slide)
        return np.concatenate([centre + rotation @ _horn_local(bike, s) for s in ("r", "l")])

    steps = [(stem, slide) for stem in range(bike._stem_steps + 1) for slide in range(bike._handlebar_horizontal_steps + 1)]
    if trunk_angle_deg is None:
        # The trunk stays as it is: the stem sets the handlebar at the seat height and
        # the slide gets the arm angles as close as possible to the requested ones.
        options = []
        for slide in range(bike._handlebar_horizontal_steps + 1):
            stem = int(
                np.argmin(
                    [
                        abs(bike._handlebar_position_at(step, slide)[1] - seat_top_y)
                        for step in range(bike._stem_steps + 1)
                    ]
                )
            )
            arm = arms.solve(grips_at(stem, slide))
            score = (
                abs(arm["flexion_deg"] - shoulder_flexion_deg)
                + abs(arm["elbow_deg"] - elbow_flexion_deg)
                + 1e3 * max(arm["residual"] - _TOLERANCE_M, 0.0)
            )
            options.append((score, slide, stem, arm))
        _, handlebar_slide, stem_step, arm_solution = min(options, key=lambda item: item[0])
    else:
        # The trunk and arm angles define where the hands are: take the closest handlebar step.
        wanted = arms.reference_hands().reshape(2, 3)[:, :2]
        errors = [np.linalg.norm(grips_at(*step).reshape(2, 3)[:, :2] - wanted, axis=1).mean() for step in steps]
        stem_step, handlebar_slide = steps[int(np.argmin(errors))]
        arm_solution = arms.solve(grips_at(stem_step, handlebar_slide))
    bike.set_stem_step(stem_step)
    bike.set_handlebar_horizontal_step(handlebar_slide)
    height_error = bike.handlebar_position[1] - seat_top_y
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
        "seat_horizontal_step": slide_step,
        "stem_step": stem_step,
        "handlebar_horizontal_step": handlebar_slide,
        "knee_to_pedal_axis_mm": (legs_solution["knee_x"] - pedal_x) * 1e3,
        "handlebar_height_error_mm": float(height_error) * 1e3,
        "shoulder_flexion_deg": arm_solution["flexion_deg"],
        "trunk_angle_deg": trunk_deg,
        "elbow_flexion_deg": arm_solution["elbow_deg"],
        "hand_error_mm": arm_solution["residual"] * 1e3,
        "left_pedal_angle_deg": bike.pedal_left_angle_deg,
    }
    return framework
